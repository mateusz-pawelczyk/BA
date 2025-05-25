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

double r2_orthogonal_metric(const Eigen::MatrixXd &D, FlatModel *model)
{
    double R2 = model->R2(D);

    return -R2;
}

double r2_regression_metric(const Eigen::MatrixXd &D, FlatModel *model)
{
    int d = model->get_dimension();
    int n = model->get_ambient_dimension();

    Eigen::MatrixXd X = D.leftCols(d);
    Eigen::MatrixXd Y = D.rightCols(n - d);

    double R2 = model->R2(X, Y);

    return -R2;
}

double mse_orthogonal_metric(const Eigen::MatrixXd &D, FlatModel *model)
{
    double MSE = model->MSE(D);
    return MSE;
}

double mse_regression_metric(const Eigen::MatrixXd &D, FlatModel *model)
{
    int d = model->get_dimension();
    int n = model->get_ambient_dimension();

    Eigen::MatrixXd X = D.leftCols(d);
    Eigen::MatrixXd Y = D.rightCols(n - d);

    double MSE = model->MSE(X, Y);
    return MSE;
}

Eigen::VectorXd regression_loss(const Eigen::MatrixXd &D, FlatModel *model)
{
    int d = model->get_dimension();
    int n = model->get_ambient_dimension();
    Eigen::MatrixXd X = D.leftCols(d);
    Eigen::MatrixXd Y = D.rightCols(n - d);

    Eigen::MatrixXd Y_true = model->predict(X);

    return (Y_true - Y).rowwise().squaredNorm();
}

Eigen::VectorXd orthogonal_loss(const Eigen::MatrixXd &D, FlatModel *model)
{
    return model->quadratic_loss(D);
}

RANSAC::RANSAC(int max_iterations, double threshold, double train_data_percenatge, int min_inliners, MetricType metric, DistanceType distance)
    : max_iterations(max_iterations), threshold(threshold), train_data_percentage(train_data_percenatge), min_inliners(min_inliners)
{
    // Validate the metric
    if (metric != MetricType::R2 && metric != MetricType::MSE)
    {
        throw std::runtime_error("Invalid metric type.");
    }

    // Set the metric
    switch (metric)
    {
    case MetricType::R2:
        if (distance == DistanceType::Orthogonal)
        {
            metric_fn2 = &r2_orthogonal_metric;
            loss_fn = &orthogonal_loss;
        }
        else if (distance == DistanceType::Regression)
        {
            metric_fn2 = &r2_regression_metric;
            loss_fn = &regression_loss;
        }
        else
        {
            throw std::runtime_error("Invalid distance type.");
        }
        break;
    case MetricType::MSE:
        if (distance == DistanceType::Orthogonal)
        {
            metric_fn2 = &mse_orthogonal_metric;
            loss_fn = &orthogonal_loss;
        }
        else if (distance == DistanceType::Regression)
        {
            metric_fn2 = &mse_regression_metric;
            loss_fn = &regression_loss;
        }
        else
        {
            throw std::runtime_error("Invalid distance type.");
        }
        break;
    default:
        throw std::runtime_error("Invalid metric type.");

    }
}

// // Careful: Can return nullptr
// std::unique_ptr<Model> RANSAC::run(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y, Model *model, std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> loss_fn, std::function<double(Eigen::VectorXd, Eigen::VectorXd)> metric_fn)
// {
//     if (model == nullptr)
//     {
//         throw std::runtime_error("Model can't be `nullptr`.");
//     }

//     int N = X.rows();
//     int d = X.cols();
//     int subset_size = static_cast<int>(std::ceil(train_data_percentage * N));

//     std::unique_ptr<Model> bestModel;
//     double bestModelError = INFINITY;

//     while (bestModel == nullptr)
//     {
// #pragma omp parallel for
//         for (int iter = 0; iter < max_iterations; ++iter)
//         {
//             // Thread-local resources
//             std::vector<int> local_indices(N);
//             std::iota(local_indices.begin(), local_indices.end(), 0);
//             std::random_device rd;
//             std::mt19937 local_g(rd());
//             Eigen::MatrixXd X_subset(subset_size, d);
//             Eigen::VectorXd Y_subset(subset_size);
//             auto local_model = model->clone(); // Clone model for this thread

//             // 1. Sample subset using local resources
//             sampleRandomSubset(X, Y, X_subset, Y_subset, local_indices, local_g);

//             // 2. Fit the local model
//             local_model->fit(X_subset, Y_subset);

//             // 3. Compute loss for all data
//             Eigen::VectorXd Y_pred = local_model->predict(X);
//             Eigen::VectorXd loss = loss_fn(Y, Y_pred);

//             // 4. Extract inliers
//             std::vector<int> inliers = findInliers(loss, threshold);
//             if (inliers.size() < min_inliners)
//                 continue;

//             // 5. Refit on inliers
//             Eigen::MatrixXd X_inliers(inliers.size(), d);
//             Eigen::VectorXd Y_inliers(inliers.size());
//             for (size_t i = 0; i < inliers.size(); ++i)
//             {
//                 X_inliers.row(i) = X.row(inliers[i]);
//                 Y_inliers[i] = Y[inliers[i]];
//             }
//             local_model->fit(X_inliers, Y_inliers);

//             // 6. Evaluate the model
//             Eigen::VectorXd Y_pred_inliers = local_model->predict(X_inliers);
//             double error = metric_fn(Y_inliers, Y_pred_inliers);

// #pragma omp critical
//             {
//                 if (error < bestModelError)
//                 {
//                     bestModel = local_model->clone();
//                     bestModelError = error;
//                 }
//             }
//         }

//         if (!bestModel)
//         {
//             threshold *= 1.25;
//             std::cout << "Increased threshold to " << threshold << std::endl;
//         }
//     }

//     return bestModel;
// }

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

// std::unique_ptr<FlatModel> RANSAC::run(const Eigen::MatrixXd &D, FlatModel *model, int best_model_count, std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> loss_fn, std::function<double(Eigen::VectorXd, Eigen::VectorXd)> metric_fn, FlatAverager *averager) const
// {
//     if (model == nullptr)
//     {
//         throw std::runtime_error("Model can't be `nullptr`.");
//     }
//     int N = D.rows();
//     int d = D.cols() - 1;

//     Eigen::MatrixXd X = D.leftCols(d);
//     Eigen::VectorXd Y = D.col(d);

//     int subset_size = static_cast<int>(std::ceil(train_data_percentage * N));

//     std::vector<int> indices(N);
//     std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0,1,2,...

//     // Minheap based on error
//     // using ModelEntry = std::pair<double, std::unique_ptr<FlatModel>>;
//     auto compare = [](const FlatModelEntry &a, const FlatModelEntry &b)
//     {
//         return a.first < b.first;
//     };

//     std::priority_queue<FlatModelEntry, std::vector<FlatModelEntry>, decltype(compare)> heap(compare);

//     Eigen::MatrixXd X_subset(subset_size, d);
//     Eigen::VectorXd Y_subset(subset_size);

//     double threshold = this->threshold;

//     while (heap.empty())
//     {
// #pragma omp parallel
//         {
//             // Each thread has its own random engine:
//             std::random_device rd_thread;
//             std::mt19937 g(rd_thread());

//             // Local heap for thread safety because multiple threads shouldn't access the same heap
//             decltype(heap) local_heap(compare);

// #pragma omp for
//             for (int iter = 0; iter < max_iterations; ++iter)
//             {
//                 // 1. Randomly sample a subset
//                 sampleRandomSubset(X, Y, X_subset, Y_subset, indices, g);

//                 // 2. Fit the model to the random subset
//                 model->reset();
//                 model->fit(X_subset, Y_subset);

//                 // 3. Compute loss for ALL data
//                 Eigen::VectorXd Y_pred = model->predict(X);
//                 Eigen::VectorXd loss = loss_fn(Y, Y_pred);

//                 // 4. Extract inliers based on the loss threshold
//                 std::vector<int> inliers = findInliers(loss, threshold);
//                 if (inliers.size() < min_inliners)
//                     continue; // Skip if inliers are less than the threshold

//                 // 5. Refit the model using inliers
//                 Eigen::MatrixXd X_inliers(inliers.size(), d);
//                 Eigen::VectorXd Y_inliers(inliers.size());
//                 for (size_t i = 0; i < inliers.size(); ++i)
//                 {
//                     X_inliers.row(i) = X.row(inliers[i]);
//                     Y_inliers[i] = Y[inliers[i]];
//                 }
//                 model->fit(X_inliers, Y_inliers);

//                 // 6. Evaluate the model
//                 Eigen::VectorXd Y_pred_inliers = model->predict(X_inliers);
//                 double error = metric_fn(Y_inliers, Y_pred_inliers);

//                 // 7. Store into local (thread safe) heap
//                 local_heap.emplace(error, castToModel(model->clone()));
//                 if (local_heap.size() > best_model_count)
//                 {
//                     // Pop the worst model (and maintain the `best_model_count` models)
//                     local_heap.pop();
//                 }
//             }

// // Merge local heaps into global heap (Done once per thread)
// #pragma omp critical
//             {
//                 while (!local_heap.empty())
//                 {
//                     // FIX: Pop from local_heap into a local variable,
//                     //      then emplace into the global heap
//                     auto topVal = std::move(const_cast<FlatModelEntry &>(local_heap.top()));
//                     local_heap.pop();

//                     heap.emplace(std::move(topVal));
//                     if (heap.size() > best_model_count)
//                     {
//                         heap.pop();
//                     }
//                 }
//             }
//         } // end parallel region
//         if (heap.empty())
//         {
//             // run again with 25% higher threshold
//             threshold *= 1.25;
//             std::cout << "No good model found. Running again with higher threshold: " << threshold << std::endl;
//         }
//     }

//     // Get the top models and their errors from the heap
//     std::vector<std::unique_ptr<FlatModel>> models;
//     std::vector<double> errors;
//     models.reserve(best_model_count);
//     errors.reserve(best_model_count);

//     // Gather the top models from the heap and sort them by error
//     gatherTopModels(heap, models, errors);
//     averager->fit(models, errors);
//     std::unique_ptr<FlatModel> fm = castToModel(averager->clone());
//     return fm;
// }

std::unique_ptr<FlatModel> RANSAC::run2(const Eigen::MatrixXd &D,
                                        FlatModel *model,
                                        int best_model_count,
                                        FlatAverager *averager,
                                        bool weighted_average) const
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
                if (inliers.size() < std::max(min_inliners, d + 1))
                {
                    continue; // not enough inliers, skip
                }

                // 6. Refit using inliers
                Eigen::MatrixXd D_inliers(inliers.size(), n);
                for (size_t i = 0; i < inliers.size(); ++i)
                {
                    D_inliers.row(i) = D.row(inliers[i]);
                }

                // model->fit(D_inliers);

                // 7. Evaluate the model and push onto local heap
                double error = metric_fn2(D_inliers, model); // orthognal vs regression

                // check if error is nan or infinity
                if (std::isnan(error) || std::isinf(error))
                {
                    std::cout << "=====================" << std::endl;
                    std::cout << "Error: " << error << std::endl;

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
    if (models.size() > 1)
    {
        averager->fit(models, weighted_average ? errors : std::vector<double>());
        std::unique_ptr<FlatModel> fm = castToModel(averager->clone());
        return fm;
    }
    else
    {
        return std::move(models[0]);
    }
}

std::unique_ptr<FlatModel> RANSAC::run_slow(const Eigen::MatrixXd &D,
                                            FlatModel *model,
                                            int best_model_count,
                                            FlatAverager *averager,
                                            bool weighted_average) const
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
            Eigen::VectorXd loss = loss_fn(D, model);

            // 5. Extract inliers
            std::vector<int> inliers = findInliers(loss, threshold);
            if (inliers.size() < std::max(min_inliners, d + 1))
            {
                continue; // not enough inliers, skip
            }

            // 6. Refit using inliers
            Eigen::MatrixXd D_inliers(inliers.size(), n);
            for (size_t i = 0; i < inliers.size(); ++i)
            {
                D_inliers.row(i) = D.row(inliers[i]);
            }
            // model->fit(D_inliers);

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
            // std::cout << "[SLOW] No good d-flat found. Running again with higher threshold: " << threshold << std::endl;
        }
    }

    // Collect top models from the heap
    std::vector<std::unique_ptr<FlatModel>> models;
    std::vector<double> errors;
    models.reserve(best_model_count);
    errors.reserve(best_model_count);
    gatherTopModels(heap, models, errors);
    averager->fit(models, weighted_average ? errors : std::vector<double>());
    std::unique_ptr<FlatModel> fm = castToModel(averager->clone());
    return fm;
}


// Add necessary includes at the top of your .cpp file
#include <omp.h>     // For OpenMP
#include <atomic>    // For std::atomic (safer error handling in parallel)
#include <iostream>  // Ensure std::cerr is available

// (Your other includes: numeric, algorithm, stdexcept, cmath, etc.)

std::unique_ptr<FlatModel> RANSAC::run_fast(const Eigen::MatrixXd &D,
                                            FlatModel *prototype_model, // Renamed for clarity
                                            int best_model_count,
                                            FlatAverager *averager,
                                            bool weighted_average) const
{
    if (prototype_model == nullptr)
    {
        throw std::runtime_error("Prototype model can't be `nullptr`.");
    }

    int N = D.rows();
    int d_model_dim = prototype_model->get_dimension();
    int n_ambient = prototype_model->get_ambient_dimension();

    if (n_ambient != D.cols())
    {
        throw std::runtime_error("Dimension mismatch between model and data.");
    }
    if (N == 0) {
        throw std::runtime_error("Input data D is empty.");
    }
    if (this->train_data_percentage <= 0.0 || this->train_data_percentage > 1.0) {
        throw std::runtime_error("train_data_percentage must be between 0 (exclusive) and 1 (inclusive).");
    }

    int subset_size = static_cast<int>(std::ceil(this->train_data_percentage * N));
    if (subset_size == 0 && N > 0) {
        subset_size = 1;
    }
    if (subset_size > N) {
        subset_size = N;
    }
    if (subset_size < d_model_dim + 1 && N >= d_model_dim +1) { // Basic check, model->fit might have more specific needs
        // This condition depends on the specific requirements of `prototype_model->fit()`
        // For generic RANSAC, subset_size should be at least the minimal number of points to define a model.
        // For this project, it's d+1 for AffineFit.
        // If train_data_percentage is too low, subset_size might be < d+1.
        // We might want to ensure subset_size >= d_model_dim + 1 if N allows.
        // For now, let the model's fit() handle insufficient points in D_subset if it occurs.
    }


    auto compare_models = [](const FlatModelEntry &a, const FlatModelEntry &b) {
        return a.first < b.first; // Max-heap behavior on error (top() is largest error)
    };

    std::priority_queue<FlatModelEntry, std::vector<FlatModelEntry>, decltype(compare_models)> global_heap(compare_models);

    std::random_device master_rd_device; // One master random_device
    double current_threshold = this->threshold;
    unsigned int base_seed_for_pass = master_rd_device(); // Initial base seed

    std::atomic<bool> nan_inf_error_detected(false); // For safer error handling from parallel regions
    while (global_heap.empty())
    {
        // Reset for the new pass
        nan_inf_error_detected.store(false);

        // Determine number of threads that will be used
        int num_threads_to_use = 1;
        #pragma omp parallel
        {
            #pragma omp single
            num_threads_to_use = omp_get_num_threads();
        }
        
        std::vector<std::priority_queue<FlatModelEntry, std::vector<FlatModelEntry>, decltype(compare_models)>>
            local_heaps;
        local_heaps.reserve(num_threads_to_use); // Optional pre-allocation
        for (int i = 0; i < num_threads_to_use; ++i) {
            local_heaps.emplace_back(compare_models); // Construct each PQ with the comparator
        }

        #pragma omp parallel
        {
            // Thread-local resources
            std::unique_ptr<FlatModel> thread_local_model = castToModel(prototype_model->clone());
            Eigen::MatrixXd D_subset_local(subset_size, n_ambient); // Pre-sized
            std::vector<int> indices_local(N); // Pre-sized
            Eigen::MatrixXd D_inliers_local;   // Resized as needed

            int thread_id = omp_get_thread_num();
            auto& current_thread_local_heap = local_heaps[thread_id];
            // std::mt19937 is created and seeded per iteration for determinism

            #pragma omp for schedule(dynamic) // Dynamic schedule for potentially uneven iteration times
            for (int iter = 0; iter < this->max_iterations; ++iter)
            {
                if (nan_inf_error_detected.load()) {
                    continue; 
                }// Stop processing if error elsewhere

                // Deterministic seeding for this iteration
                std::mt19937 g_local(base_seed_for_pass + iter);

                std::iota(indices_local.begin(), indices_local.end(), 0);
                std::shuffle(indices_local.begin(), indices_local.end(), g_local);

                if (subset_size > 0) {
                    for (int i = 0; i < subset_size; ++i) {
                        D_subset_local.row(i) = D.row(indices_local[i]);
                    }
                } else if (N > 0) { // subset_size is 0 but N > 0
                    continue; 
                } else { // N == 0
                    continue;
                }


                thread_local_model->reset();
                thread_local_model->fit(D_subset_local);

                Eigen::VectorXd loss = this->loss_fn(D, thread_local_model.get());

                std::vector<int> inliers_indices = findInliers(loss, current_threshold);

                if (inliers_indices.size() < static_cast<size_t>(std::max(this->min_inliners, d_model_dim + 1)))
                {
                    continue;
                }

                D_inliers_local.resize(inliers_indices.size(), n_ambient);
                for (size_t i = 0; i < inliers_indices.size(); ++i)
                {
                    D_inliers_local.row(i) = D.row(inliers_indices[i]);
                }

                double error = this->metric_fn2(D_inliers_local, thread_local_model.get());


                if (std::isnan(error) || std::isinf(error))
                {
                    // Safely report error and signal to stop
                    if (!nan_inf_error_detected.exchange(true)) { // Ensure only first thread reports fully
                        // Use std::cerr for errors, less prone to cout buffering issues from threads
                        std::cerr << "=====================" << std::endl;
                        std::cerr << "CRITICAL ERROR in RANSAC (parallel section)" << std::endl;
                        std::cerr << "Thread ID: " << omp_get_thread_num() << ", Iteration: " << iter << std::endl;
                        std::cerr << "Current threshold: " << current_threshold << std::endl;
                        std::cerr << "Error: " << error << std::endl;
                        // Optionally print D_inliers_local if small and useful
                        if (D_inliers_local.size() < 200) {
                             std::cerr << "D_inliers_local (" << D_inliers_local.rows() << "x" << D_inliers_local.cols() << "):\n" << D_inliers_local << std::endl;
                        } else {
                             std::cerr << "D_inliers_local too large to print (" << D_inliers_local.rows() << "x" << D_inliers_local.cols() << ")" << std::endl;
                        }
                    }
                    // Note: OpenMP 'for' loops can be cancelled with '#pragma omp cancel for' in OpenMP 4.0+
                    // if cancellation is enabled. Otherwise, threads will finish their current iteration.
                    // The 'if (nan_inf_error_detected.load()) continue;' helps stop further work.
                    continue; // Skip adding to heap
                }


                std::unique_ptr<FlatModel> thread_local_model = castToModel(prototype_model->clone()); // CORRECTED

                // Clone *the fitted* model, and emplace it:
                auto model_clone = castToModel(thread_local_model->clone());
                current_thread_local_heap.emplace(error, std::move(model_clone));

                // Now pop off the worst if we exceed best_model_count
                if (current_thread_local_heap.size() > static_cast<size_t>(best_model_count)) {
                    current_thread_local_heap.pop();
                }
            } // End of #pragma omp for
        } // End of #pragma omp parallel

        // Check for error flag after parallel region before proceeding
        if (nan_inf_error_detected.load()) {
            throw std::runtime_error("Error (NaN or Inf) detected during parallel RANSAC execution.");
        }

        // Merge local heaps into the global heap (serially)
        // This part needs to correctly handle moving std::unique_ptr
        for (auto& local_h : local_heaps) {
            std::vector<FlatModelEntry> temp_drain; // Temporary vector to hold entries from local_h
            temp_drain.reserve(local_h.size());
            while(!local_h.empty()) {
                // Move from priority_queue's top. This is tricky.
                // const_cast is one way but generally unsafe if not careful.
                // A safer pattern for unique_ptr in PQ:
                // 1. Get error (copy).
                // 2. Get a non-const ref to unique_ptr in top() (requires const_cast or helper).
                // 3. Move unique_ptr.
                // 4. Pop.
                // For simplicity and because we are draining:
                FlatModelEntry& top_entry_ref = const_cast<FlatModelEntry&>(local_h.top());
                temp_drain.push_back(std::move(top_entry_ref)); // Moves the pair, including unique_ptr
                local_h.pop(); // Pop the now moved-from element
            }

            for(auto& entry_to_add : temp_drain) {
                global_heap.push(std::move(entry_to_add));
                if (global_heap.size() > static_cast<size_t>(best_model_count)) {
                    global_heap.pop();
                }
            }
        }
        
        if (global_heap.empty())
        {
            current_threshold *= 1.25;
            base_seed_for_pass = master_rd_device(); // Get a new base seed for the next pass
            // std::cout << "[RANSAC Parallel] No good d-flat. Higher threshold: " << current_threshold << std::endl;
        }
    } // End of while(global_heap.empty())

    std::vector<std::unique_ptr<FlatModel>> top_models;
    std::vector<double> top_errors;
    top_models.reserve(global_heap.size());
    top_errors.reserve(global_heap.size());

    gatherTopModels(global_heap, top_models, top_errors); // Assumes this can drain the global_heap

    averager->fit(top_models, weighted_average ? top_errors : std::vector<double>());
    std::unique_ptr<FlatModel> final_model = castToModel(averager->clone());
    
    return final_model;
}