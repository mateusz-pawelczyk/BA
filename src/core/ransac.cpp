#include "core/ransac.hpp"

#include <stdexcept>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <queue>
#include <tuple>
#include <cassert>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

double r2_metric(Eigen::MatrixXd D, FlatModel *model)
{
    Eigen::VectorXd mean = D.colwise().mean();
    double ss_res = model->quadratic_loss(D).sum();
    double ss_tot = (D.rowwise() - mean.transpose()).squaredNorm();

    return -(1.0 - ss_res / ss_tot);
}

double rss_metric(Eigen::MatrixXd D, FlatModel *model)
{
    return model->quadratic_loss(D).sum();
}

double mse_metric(Eigen::MatrixXd D, FlatModel *model)
{
    return model->quadratic_loss(D).mean();
}

double rmse_metric(Eigen::MatrixXd D, FlatModel *model)
{
    return std::sqrt(mse_metric(D, model));
}

RANSAC::RANSAC(int max_iterations, double threshold, double train_data_percenatge, int min_inliners, MetricType metric)
    : max_iterations(max_iterations), threshold(threshold), train_data_percentage(train_data_percenatge), min_inliners(min_inliners)
{
    // Validate the metric
    if (metric != MetricType::R2 && metric != MetricType::RSS && metric != MetricType::MSE && metric != MetricType::RMSE)
    {
        throw std::runtime_error("Invalid metric type.");
    }

    // Set the metric
    switch (metric)
    {
    case MetricType::R2:
        metric_fn2 = &r2_metric;
        break;
    case MetricType::RSS:
        metric_fn2 = &rss_metric;
        break;
    case MetricType::MSE:
        metric_fn2 = &mse_metric;
        break;
    case MetricType::RMSE:
        metric_fn2 = &rmse_metric;
        break;
    }
}

// Careful: Can return nullptr
std::unique_ptr<Model> RANSAC::run(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y, Model *model, std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> loss_fn, std::function<double(Eigen::VectorXd, Eigen::VectorXd)> metric_fn)
{
    if (model == nullptr)
    {
        throw std::runtime_error("Model can't be `nullptr`.");
    }

    int N = X.rows();
    int d = X.cols();
    int subset_size = static_cast<int>(std::ceil(train_data_percentage * N));

    std::unique_ptr<Model> bestModel;
    double bestModelError = INFINITY;

    while (bestModel == nullptr)
    {
#pragma omp parallel for
        for (int iter = 0; iter < max_iterations; ++iter)
        {
            // Thread-local resources
            std::vector<int> local_indices(N);
            std::iota(local_indices.begin(), local_indices.end(), 0);
            std::random_device rd;
            std::mt19937 local_g(rd());
            Eigen::MatrixXd X_subset(subset_size, d);
            Eigen::VectorXd Y_subset(subset_size);
            auto local_model = model->clone(); // Clone model for this thread

            // 1. Sample subset using local resources
            sampleRandomSubset(X, Y, X_subset, Y_subset, local_indices, local_g);

            // 2. Fit the local model
            local_model->fit(X_subset, Y_subset);

            // 3. Compute loss for all data
            Eigen::VectorXd Y_pred = local_model->predict(X);
            Eigen::VectorXd loss = loss_fn(Y, Y_pred);

            // 4. Extract inliers
            std::vector<int> inliers = findInliers(loss, threshold);
            if (inliers.size() < min_inliners)
                continue;

            // 5. Refit on inliers
            Eigen::MatrixXd X_inliers(inliers.size(), d);
            Eigen::VectorXd Y_inliers(inliers.size());
            for (size_t i = 0; i < inliers.size(); ++i)
            {
                X_inliers.row(i) = X.row(inliers[i]);
                Y_inliers[i] = Y[inliers[i]];
            }
            local_model->fit(X_inliers, Y_inliers);

            // 6. Evaluate the model
            Eigen::VectorXd Y_pred_inliers = local_model->predict(X_inliers);
            double error = metric_fn(Y_inliers, Y_pred_inliers);

#pragma omp critical
            {
                if (error < bestModelError)
                {
                    bestModel = local_model->clone();
                    bestModelError = error;
                }
            }
        }

        if (!bestModel)
        {
            threshold *= 1.25;
            std::cout << "Increased threshold to " << threshold << std::endl;
        }
    }

    return bestModel;
}

std::unique_ptr<FlatModel> RANSAC::castToModel(std::unique_ptr<Model> basePtr) const
{
    // Use dynamic_cast to ensure safety at runtime
    FlatModel *derivedRawPtr = dynamic_cast<FlatModel *>(basePtr.get());
    if (!derivedRawPtr)
    {
        throw std::runtime_error("Failed to cast Base to Derived");
    }

    // Transfer ownership using a custom deleter
    return std::unique_ptr<FlatModel>(
        static_cast<FlatModel *>(basePtr.release())); // Transfer ownership
}

/**
 * @brief Randomly samples a subset from X, Y using the given indices and random engine.
 */
void RANSAC::sampleRandomSubset(const Eigen::MatrixXd &X,
                                const Eigen::VectorXd &Y,
                                Eigen::MatrixXd &X_subset,
                                Eigen::VectorXd &Y_subset,
                                std::vector<int> &indices,
                                std::mt19937 &g) const
{
    // Shuffle the indices
    std::shuffle(indices.begin(), indices.end(), g);

    // Copy the first subset_size rows
    const int subset_size = static_cast<int>(X_subset.rows());
    for (int i = 0; i < subset_size; ++i)
    {
        X_subset.row(i) = X.row(indices[i]);
        Y_subset[i] = Y[indices[i]];
    }
}

/**
 * @brief Randomly samples a subset from D using the given indices and random engine.
 */
void RANSAC::sampleRandomSubset(const Eigen::MatrixXd &D,
                                Eigen::MatrixXd &D_subset,
                                std::vector<int> &indices,
                                std::mt19937 &g) const
{
    // Shuffle the indices
    std::shuffle(indices.begin(), indices.end(), g);

    // Copy the first subset_size rows
    const int subset_size = static_cast<int>(D_subset.rows());
    for (int i = 0; i < subset_size; ++i)
    {
        D_subset.row(i) = D.row(indices[i]);
    }
}

/**
 * @brief Finds inliers based on a loss threshold.
 */
std::vector<int> RANSAC::findInliers(const Eigen::VectorXd &loss_values, double threshold) const
{
    std::vector<int> inliers;
    inliers.reserve(loss_values.size());
    for (int i = 0; i < loss_values.size(); ++i)
    {
        if (loss_values(i) < threshold)
        {
            inliers.push_back(i);
        }
    }
    return inliers;
}

std::unique_ptr<FlatModel> RANSAC::run(const Eigen::MatrixXd &D, FlatModel *model, int best_model_count, std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> loss_fn, std::function<double(Eigen::VectorXd, Eigen::VectorXd)> metric_fn, MedianSDF *averager) const
{
    if (model == nullptr)
    {
        throw std::runtime_error("Model can't be `nullptr`.");
    }
    int N = D.rows();
    int d = D.cols() - 1;

    Eigen::MatrixXd X = D.leftCols(d);
    Eigen::VectorXd Y = D.col(d);

    int subset_size = static_cast<int>(std::ceil(train_data_percentage * N));

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0,1,2,...

    // Minheap based on error
    // using ModelEntry = std::pair<double, std::unique_ptr<FlatModel>>;
    auto compare = [](const FlatModelEntry &a, const FlatModelEntry &b)
    {
        return a.first < b.first;
    };

    std::priority_queue<FlatModelEntry, std::vector<FlatModelEntry>, decltype(compare)> heap(compare);

    Eigen::MatrixXd X_subset(subset_size, d);
    Eigen::VectorXd Y_subset(subset_size);

    double threshold = this->threshold;

    while (heap.empty())
    {
#pragma omp parallel
        {
            // Each thread has its own random engine:
            std::random_device rd_thread;
            std::mt19937 g(rd_thread());

            // Local heap for thread safety because multiple threads shouldn't access the same heap
            decltype(heap) local_heap(compare);

#pragma omp for
            for (int iter = 0; iter < max_iterations; ++iter)
            {
                // 1. Randomly sample a subset
                sampleRandomSubset(X, Y, X_subset, Y_subset, indices, g);

                // 2. Fit the model to the random subset
                model->reset();
                model->fit(X_subset, Y_subset);

                // 3. Compute loss for ALL data
                Eigen::VectorXd Y_pred = model->predict(X);
                Eigen::VectorXd loss = loss_fn(Y, Y_pred);

                // 4. Extract inliers based on the loss threshold
                std::vector<int> inliers = findInliers(loss, threshold);
                if (inliers.size() < min_inliners)
                    continue; // Skip if inliers are less than the threshold

                // 5. Refit the model using inliers
                Eigen::MatrixXd X_inliers(inliers.size(), d);
                Eigen::VectorXd Y_inliers(inliers.size());
                for (size_t i = 0; i < inliers.size(); ++i)
                {
                    X_inliers.row(i) = X.row(inliers[i]);
                    Y_inliers[i] = Y[inliers[i]];
                }
                model->fit(X_inliers, Y_inliers);

                // 6. Evaluate the model
                Eigen::VectorXd Y_pred_inliers = model->predict(X_inliers);
                double error = metric_fn(Y_inliers, Y_pred_inliers);

                // 7. Store into local (thread safe) heap
                local_heap.emplace(error, castToModel(model->clone()));
                if (local_heap.size() > best_model_count)
                {
                    // Pop the worst model (and maintain the `best_model_count` models)
                    local_heap.pop();
                }
            }

// Merge local heaps into global heap (Done once per thread)
#pragma omp critical
            {
                while (!local_heap.empty())
                {
                    // FIX: Pop from local_heap into a local variable,
                    //      then emplace into the global heap
                    auto topVal = std::move(const_cast<FlatModelEntry &>(local_heap.top()));
                    local_heap.pop();

                    heap.emplace(std::move(topVal));
                    if (heap.size() > best_model_count)
                    {
                        heap.pop();
                    }
                }
            }
        } // end parallel region
        if (heap.empty())
        {
            // run again with 25% higher threshold
            threshold *= 1.25;
            std::cout << "No good model found. Running again with higher threshold: " << threshold << std::endl;
        }
    }

    // Get the top models and their errors from the heap
    std::vector<std::unique_ptr<FlatModel>> models;
    std::vector<double> errors;
    models.reserve(best_model_count);
    errors.reserve(best_model_count);

    // Gather the top models from the heap and sort them by error
    gatherTopModels(heap, models, errors);
    averager->fit(models, errors);
    std::unique_ptr<FlatModel> fm = castToModel(averager->clone());
    return fm;
}

std::unique_ptr<FlatModel> RANSAC::run2(const Eigen::MatrixXd &D,
                                        FlatModel *model,
                                        int best_model_count,
                                        MedianSDF *averager) const
{
    if (model == nullptr)
    {
        throw std::runtime_error("Model can't be `nullptr`.");
    }

    int N = D.rows();
    int d = model->get_dimension();
    int n = model->get_ambient_dimension();

    if (n != D.cols())
    {
        throw std::runtime_error("Dimension mismatch between model and data.");
    }

    int subset_size = static_cast<int>(std::ceil(train_data_percentage * N));

    // We still keep a reference copy of the original indices:
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0); // 0,1,2,... N-1

    // Define a comparator for our max-heap (we store the "best" models).
    auto compare = [](const FlatModelEntry &a, const FlatModelEntry &b)
    {
        return a.first < b.first;
    };

    // Global heap: we will merge thread-local heaps into it via critical sections.
    std::priority_queue<FlatModelEntry, std::vector<FlatModelEntry>, decltype(compare)> heap(compare);

    Eigen::MatrixXd D_subset(subset_size, n);
    double threshold = this->threshold;

    // Because of the "while (heap.empty())" loop, we may have to repeat
    // until we find enough inliers. Each pass can raise the threshold.
    while (heap.empty())
    {

#pragma omp parallel
        {
            // Each thread has its own random engine:
            std::random_device rd_thread;
            std::mt19937 g_local(rd_thread());

            // We also use a local heap for each thread
            // to avoid concurrent accesses to 'heap'.
            decltype(heap) local_heap(compare);

#pragma omp for
            for (int iter = 0; iter < max_iterations; ++iter)
            {
                // 1. Make a private copy of 'indices' to shuffle
                std::vector<int> local_indices = indices;
                std::shuffle(local_indices.begin(), local_indices.end(), g_local);

                // 2. Create the D_subset from those shuffled indices
                for (int i = 0; i < subset_size; ++i)
                {
                    D_subset.row(i) = D.row(local_indices[i]);
                }

                // 3. Fit the model to the random subset
                model->reset();
                model->fit(D_subset);

                // 4. Compute loss for ALL data
                Eigen::VectorXd loss = model->quadratic_loss(D);

                // 5. Extract inliers
                std::vector<int> inliers = findInliers(loss, threshold);
                if (inliers.size() < min_inliners)
                {
                    continue; // not enough inliers, skip
                }

                // 6. Refit using inliers
                Eigen::MatrixXd D_inliers(inliers.size(), n);
                for (size_t i = 0; i < inliers.size(); ++i)
                {
                    D_inliers.row(i) = D.row(inliers[i]);
                }
                // Print top 2 inliers for debugging
                model->fit(D_inliers);

                // 7. Evaluate the model and push onto local heap
                double error = metric_fn2(D_inliers, model);

                // check if error is nan or infinity
                if (std::isnan(error) || std::isinf(error))
                {
                    // Print some debugging information to identify the exact parameter causing the issue
                    std::cout << "=====================" << std::endl;
                    std::cout << "Error: " << error << std::endl;
                    // Print the data matrix
                    std::cout << "Data matrix:\n"
                              << std::endl;
                    std::cout << D_inliers << std::endl;
                    throw std::runtime_error("Error is nan or inf.");
                }
                local_heap.emplace(error, castToModel(model->clone()));

                // Keep only the best X in the local heap
                if (local_heap.size() > best_model_count)
                {
                    local_heap.pop();
                }
            }

// Now merge the local heap into the global heap safely
#pragma omp critical
            {
                while (!local_heap.empty())
                {
                    auto topVal = std::move(const_cast<FlatModelEntry &>(local_heap.top()));
                    local_heap.pop();

                    heap.emplace(std::move(topVal));
                    if (heap.size() > best_model_count)
                    {
                        heap.pop();
                    }
                }
            }
        } // end parallel region

        if (heap.empty())
        {
            threshold *= 1.25;
            std::cout << "[FAST/AVERAGED] No good d-flat found. Running again with higher threshold: " << threshold << std::endl;
        }
    }

    // Collect top models from the heap
    std::vector<std::unique_ptr<FlatModel>> models;
    std::vector<double> errors;
    models.reserve(best_model_count);
    errors.reserve(best_model_count);
    gatherTopModels(heap, models, errors);
    averager->fit(models, errors);
    std::unique_ptr<FlatModel> fm = castToModel(averager->clone());
    return fm;
}

std::unique_ptr<FlatModel> RANSAC::run_slow(const Eigen::MatrixXd &D,
                                            FlatModel *model,
                                            int best_model_count,
                                            MedianSDF *averager) const
{
    if (model == nullptr)
    {
        throw std::runtime_error("Model can't be `nullptr`.");
    }

    int N = D.rows();
    int d = model->get_dimension();
    int n = model->get_ambient_dimension();

    if (n != D.cols())
    {
        throw std::runtime_error("Dimension mismatch between model and data.");
    }

    int subset_size = static_cast<int>(std::ceil(train_data_percentage * N));

    // We still keep a reference copy of the original indices:
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0); // 0,1,2,... N-1

    // Define a comparator for our max-heap (we store the "best" models).
    auto compare = [](const FlatModelEntry &a, const FlatModelEntry &b)
    {
        return a.first < b.first;
    };

    std::random_device rd;

    // Global heap: we will merge thread-local heaps into it via critical sections.
    std::priority_queue<FlatModelEntry, std::vector<FlatModelEntry>, decltype(compare)> heap(compare);

    Eigen::MatrixXd D_subset(subset_size, n);
    double threshold = this->threshold;

    // Because of the "while (heap.empty())" loop, we may have to repeat
    // until we find enough inliers. Each pass can raise the threshold.
    while (heap.empty())
    {
        for (int iter = 0; iter < max_iterations; ++iter)
        {
            std::mt19937 g(rd() + iter);

            // 1. Shuffle the data
            // Inside the for-loop:
            std::vector<int> local_indices(N);
            std::iota(local_indices.begin(), local_indices.end(), 0);
            std::shuffle(local_indices.begin(), local_indices.end(), g);

            // 2. Create the D_subset from those shuffled indices
            for (int i = 0; i < subset_size; ++i)
            {
                D_subset.row(i) = D.row(local_indices[i]);
            }

            // 3. Fit the model to the random subset
            model->reset();
            model->fit(D_subset);

            // 4. Compute loss for ALL data
            Eigen::VectorXd loss = model->quadratic_loss(D);

            // 5. Extract inliers
            std::vector<int> inliers = findInliers(loss, threshold);
            if (inliers.size() < min_inliners)
            {
                continue; // not enough inliers, skip
            }

            // 6. Refit using inliers
            Eigen::MatrixXd D_inliers(inliers.size(), n);
            for (size_t i = 0; i < inliers.size(); ++i)
            {
                D_inliers.row(i) = D.row(inliers[i]);
            }
            model->fit(D_inliers);

            // 7. Evaluate the model and push onto local heap
            double error = metric_fn2(D_inliers, model);

            // check if error is nan or infinity
            if (std::isnan(error) || std::isinf(error))
            {
                // Print some debugging information to identify the exact parameter causing the issue
                std::cout << "=====================" << std::endl;
                std::cout << "Error: " << error << std::endl;
                // Print the data matrix
                std::cout << "Data matrix:\n"
                          << std::endl;
                std::cout << D_inliers << std::endl;
                throw std::runtime_error("Error is nan or inf.");
            }

            heap.emplace(error, castToModel(model->clone()));

            // Keep only the best X in the local heap
            if (heap.size() > best_model_count)
            {
                heap.pop();
            }
        }

        if (heap.empty())
        {
            threshold *= 1.25;
            std::cout << "[SLOW] No good d-flat found. Running again with higher threshold: " << threshold << std::endl;
        }
    }

    // Collect top models from the heap
    std::vector<std::unique_ptr<FlatModel>> models;
    std::vector<double> errors;
    models.reserve(best_model_count);
    errors.reserve(best_model_count);
    gatherTopModels(heap, models, errors);
    averager->fit(models, errors);
    std::unique_ptr<FlatModel> fm = castToModel(averager->clone());
    return fm;
}
