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


// Careful: Can return nullptr
std::unique_ptr<Model> RANSAC::run(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y, Model* model) {
    if (model == nullptr) {
        throw std::runtime_error("Model can't be `nullptr`.");
    }

    int N = X.rows();
    int d = X.cols();
    int subset_size = static_cast<int>(std::ceil(train_data_percentage * N));

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0); // fill indices with 0, 1, 2, 3, ...

    std::random_device rd;
    std::mt19937 g(rd());

    std::unique_ptr<Model> bestModel;
    double bestModelError = INFINITY;

    Eigen::MatrixXd X_subset(subset_size, d);
    Eigen::VectorXd Y_subset(subset_size);

    while (bestModel == nullptr) {
        #pragma omp parallel for
        for (int iter = 0; iter < max_iterations; ++iter) {
            // 1. Randomly sample a subset
            sampleRandomSubset(X, Y, X_subset, Y_subset, indices, g);

            // 2. Fit the model to the random subset
            model->fit(X_subset, Y_subset);

            // 3. Compute loss for ALL data
            Eigen::VectorXd Y_pred = model->predict(X);
            Eigen::VectorXd loss   = loss_fn(Y, Y_pred);

            // 4. Extract inliers based on the loss threshold
            std::vector<int> inliers = findInliers(loss, threshold);
            if (inliers.size() < min_inliners)
                continue; // Skip if inliers are less than the threshold

            // 5. Refit the model using inliers
            Eigen::MatrixXd X_inliers(inliers.size(), d);
            Eigen::VectorXd Y_inliers(inliers.size());
            for (size_t i = 0; i < inliers.size(); ++i) {
                X_inliers.row(i) = X.row(inliers[i]);
                Y_inliers[i]     = Y[inliers[i]];
            }
            model->fit(X_inliers, Y_inliers);

            // 6. Evaluate the model
            Eigen::VectorXd Y_pred_inliers = model->predict(X_inliers);
            double error = metric_fn(Y_inliers, Y_pred_inliers);

            #pragma omp critical
            // Update the best model if current model is better
            if (error < bestModelError) {
                auto clonedModel = model->clone();
                bestModel = std::move(clonedModel);
                bestModelError = error;
            }
        }

        if (bestModel == nullptr) {
            // run again with 25% higher threshold
            threshold *= 1.25;
            std::cout << "No good single model found. Running again with higher threshold: " << threshold << std::endl;
        }
    }
    

    return bestModel;
}



std::unique_ptr<FlatModel> RANSAC::castToModel(std::unique_ptr<Model> basePtr) const {
    // Use dynamic_cast to ensure safety at runtime
    FlatModel* derivedRawPtr = dynamic_cast<FlatModel*>(basePtr.get());
    if (!derivedRawPtr) {
        throw std::runtime_error("Failed to cast Base to Derived");
    }

    // Transfer ownership using a custom deleter
    return std::unique_ptr<FlatModel>(
        static_cast<FlatModel*>(basePtr.release())); // Transfer ownership
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
    for (int i = 0; i < subset_size; ++i) {
        X_subset.row(i) = X.row(indices[i]);
        Y_subset[i]     = Y[indices[i]];
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
    for (int i = 0; i < subset_size; ++i) {
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
    for (int i = 0; i < loss_values.size(); ++i) {
        if (loss_values(i) < threshold) {
            inliers.push_back(i);
        }
    }
    return inliers;
}



std::unique_ptr<FlatModel> RANSAC::run(const Eigen::MatrixXd& D, FlatModel *model, int best_model_count) const
{
    if (model == nullptr) {
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
    auto compare = [](const FlatModelEntry& a, const FlatModelEntry& b) {
        return a.first < b.first; 
    };

    std::priority_queue<FlatModelEntry, std::vector<FlatModelEntry>, decltype(compare)> heap(compare);

    Eigen::MatrixXd X_subset(subset_size, d);
    Eigen::VectorXd Y_subset(subset_size);

    double threshold = this->threshold;

    while (heap.empty()) {
        #pragma omp parallel
        {
            // Each thread has its own random engine:
            std::random_device rd_thread;
            std::mt19937 g(rd_thread());

            // Local heap for thread safety because multiple threads shouldn't access the same heap
            decltype(heap) local_heap(compare);

            #pragma omp for
            for (int iter = 0; iter < max_iterations; ++iter) {
                // 1. Randomly sample a subset
                sampleRandomSubset(X, Y, X_subset, Y_subset, indices, g);

                // 2. Fit the model to the random subset
                model->reset();
                model->fit(X_subset, Y_subset);

                // 3. Compute loss for ALL data
                Eigen::VectorXd Y_pred = model->predict(X);
                Eigen::VectorXd loss   = loss_fn(Y, Y_pred);

                // 4. Extract inliers based on the loss threshold
                std::vector<int> inliers = findInliers(loss, threshold);
                if (inliers.size() < min_inliners)
                    continue; // Skip if inliers are less than the threshold

                // 5. Refit the model using inliers
                Eigen::MatrixXd X_inliers(inliers.size(), d);
                Eigen::VectorXd Y_inliers(inliers.size());
                for (size_t i = 0; i < inliers.size(); ++i) {
                    X_inliers.row(i) = X.row(inliers[i]);
                    Y_inliers[i]     = Y[inliers[i]];
                }
                model->fit(X_inliers, Y_inliers);

                // 6. Evaluate the model
                Eigen::VectorXd Y_pred_inliers = model->predict(X_inliers);
                double error = metric_fn(Y_inliers, Y_pred_inliers);

                // 7. Store into local (thread safe) heap
                local_heap.emplace(error, castToModel(model->clone()));
                if (local_heap.size() > best_model_count) {
                    // Pop the worst model (and maintain the `best_model_count` models)
                    local_heap.pop();
                }
            }

            // Merge local heaps into global heap (Done once per thread)
            #pragma omp critical
            {
                while (!local_heap.empty()) {
                    // FIX: Pop from local_heap into a local variable,
                    //      then emplace into the global heap
                    auto topVal = std::move(const_cast<FlatModelEntry&>(local_heap.top()));
                    local_heap.pop();

                    heap.emplace(std::move(topVal));
                    if (heap.size() > best_model_count) {
                        heap.pop();
                    }
                }
            }
        } // end parallel region
        if (heap.empty()) { 
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

    if (models.size() > 1) {
        std::unique_ptr<FlatModel> fm = medianSDF(models, d, &errors);
        return fm;
    } else {
        return std::move(models[0]);
    }
}

double r2_metric(Eigen::MatrixXd D, FlatModel *model) {
    Eigen::VectorXd mean = D.colwise().mean();
    double ss_res = model->quadratic_loss(D).sum();
    double ss_tot = (D.rowwise() - mean.transpose()).squaredNorm();

    return 1.0 - ss_res / ss_tot;
}

std::unique_ptr<FlatModel> RANSAC::run2(const Eigen::MatrixXd &D, 
                                        FlatModel *model, 
                                        int best_model_count) const
{
    if (model == nullptr) {
        throw std::runtime_error("Model can't be `nullptr`.");
    }

    int N = D.rows();
    int d = model->get_dimension();
    int n = model->get_ambient_dimension();

    if (n != D.cols()) {
        throw std::runtime_error("Dimension mismatch between model and data.");
    }

    int subset_size = static_cast<int>(std::ceil(train_data_percentage * N));

    // We still keep a reference copy of the original indices:
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0); // 0,1,2,... N-1

    // Define a comparator for our max-heap (we store the "best" models).
    auto compare = [](const FlatModelEntry& a, const FlatModelEntry& b) {
        return a.first < b.first; 
    };

    // Global heap: we will merge thread-local heaps into it via critical sections.
    std::priority_queue<FlatModelEntry, std::vector<FlatModelEntry>, decltype(compare)> heap(compare);

    Eigen::MatrixXd D_subset(subset_size, n);
    double threshold = this->threshold;

    // Because of the "while (heap.empty())" loop, we may have to repeat
    // until we find enough inliers. Each pass can raise the threshold.
    while (heap.empty()) {

        #pragma omp parallel
        {
            // Each thread has its own random engine:
            std::random_device rd_thread;
            std::mt19937 g_local(rd_thread());

            // We also use a local heap for each thread
            // to avoid concurrent accesses to 'heap'.
            decltype(heap) local_heap(compare);

            #pragma omp for
            for (int iter = 0; iter < max_iterations; ++iter) {
                // 1. Make a private copy of 'indices' to shuffle
                std::vector<int> local_indices = indices;
                std::shuffle(local_indices.begin(), local_indices.end(), g_local);

                
                

                // 2. Create the D_subset from those shuffled indices
                for (int i = 0; i < subset_size; ++i) {
                    D_subset.row(i) = D.row(local_indices[i]);
                }

                // 3. Fit the model to the random subset
                model->reset();
                model->fit(D_subset);

                // 4. Compute loss for ALL data
                Eigen::VectorXd loss = model->quadratic_loss(D);

                // 5. Extract inliers
                std::vector<int> inliers = findInliers(loss, threshold);
                if (inliers.size() < min_inliners) {
                    continue; // not enough inliers, skip
                }

                // 6. Refit using inliers
                Eigen::MatrixXd D_inliers(inliers.size(), n);
                for (size_t i = 0; i < inliers.size(); ++i) {
                    D_inliers.row(i) = D.row(inliers[i]);
                }
                // Print top 2 inliers for debugging
                model->fit(D_inliers);


                // 7. Evaluate the model and push onto local heap
                double error = -r2_metric(D_inliers, model); 
                local_heap.emplace(error, castToModel(model->clone()));

                // Keep only the best X in the local heap
                if (local_heap.size() > best_model_count) {
                    local_heap.pop();
                }
            }

            // Now merge the local heap into the global heap safely
            #pragma omp critical
            {
                while (!local_heap.empty()) {
                    auto topVal = std::move(const_cast<FlatModelEntry&>(local_heap.top()));
                    local_heap.pop();

                    heap.emplace(std::move(topVal));
                    if (heap.size() > best_model_count) {
                        heap.pop();
                    }
                }
            }
        } // end parallel region

        if (heap.empty()) {
            threshold *= 1.25;
            std::cout << "No good d-flat found. Running again with higher threshold: " 
                      << threshold << std::endl;
        }
    }

    // Collect top models from the heap
    std::vector<std::unique_ptr<FlatModel>> models;
    std::vector<double> errors;
    models.reserve(best_model_count);
    errors.reserve(best_model_count);
    gatherTopModels(heap, models, errors);

    if (models.size() > 1) {
        std::unique_ptr<FlatModel> fm = medianSDF(models, n - 1, &errors); 
        return fm;
    } else {
        std::cout << "Only one model found for the d-flat." << std::endl;
        return medianSDF(models, n - 1, &errors);
    }
}

//// ---------- HELPER ----------

Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& A, double tolerance = 1e-6) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    const Eigen::MatrixXd& U = svd.matrixU();
    const Eigen::MatrixXd& V = svd.matrixV();
    const Eigen::VectorXd& S = svd.singularValues();

    Eigen::MatrixXd S_inv = Eigen::MatrixXd::Zero(S.size(), S.size());
    
    // Invert non-zero singular values
    for (int i = 0; i < S.size(); ++i) {
        if (S(i) > tolerance) { // Threshold to handle numerical precision issues
            S_inv(i, i) = 1.0 / S(i);
        }
    }

    // pseudo-inverse: V * S_inv * U^T
    return V * S_inv * U.transpose();
}

//// ----------------------------


std::vector<double> RANSAC::getWeights(std::vector<double>* errors) const {
    int model_count = errors->size();
    std::vector<double> weights(model_count);

    if (errors != nullptr) {
        for (int i = 0; i < model_count; ++i) {
            errors->at(i) += 1.0;
        }
        double epsilon = 1e-15;
        double error_sum = std::accumulate(errors->begin(), errors->end(), 0.0) + 1.0; 

        for (int i = 0; i < errors->size(); ++i) {
            weights[i] = 1.0 - ((errors->at(i) + epsilon) / error_sum); 
        }
        // Compute the sum of weights for normalization
        double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);

        // Normalize the weights to ensure they sum to 1
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] /= weight_sum;
        }
    } else {
        std::cout << "Errors were not provided. Assuming uniform distribution of errors." << std::endl;
        std::fill(weights.begin(), weights.end(), 1.0 / model_count);
    }


    return weights;

}

std::pair<int, int> RANSAC::validateAndGetFlatDimensions(const std::vector<std::unique_ptr<FlatModel>>& models) const {
    // Assume all A_i have the same dimensions
    int n = models[0]->get_ambient_dimension(); // Space dimension
    int d = models[0]->get_dimension(); // Dimension of each flat (should be consistent)

    // Validate the dimension of the flats
    for (size_t i = 1; i < models.size(); ++i) {
        // Here, the parametric representation is also computed for every model. No need to do it later
        if (models[i]->get_ambient_dimension() != n || models[i]->get_dimension() != d) {
            throw std::runtime_error("All A matrices must have the same dimensions.");
        }
    }

    return {n, d};
}

std::unique_ptr<FlatModel> RANSAC::medianSDF(std::vector<std::unique_ptr<FlatModel>>& models, int k, std::vector<double>* errors) const {
    int model_count = models.size();

    // Validate input sizes
    if (model_count == 0) {
        throw std::runtime_error("At least one flat must be provided.");
    }
    std::vector<double> weights = getWeights(errors);

    // Throw an error if the dimensions are inconsistent
    auto [n, d] = validateAndGetFlatDimensions(models);

    // Validate the parameter k
    if (k < 0 || k > n - 1) {
        throw std::invalid_argument("k must be in the range [0, n-1], where n is the ambient dimension.");
    }

    // Initialize Q_star and r_star
    Eigen::MatrixXd Q_star = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd r_star = Eigen::VectorXd::Zero(n);
    std::cout << "======================" << std::endl;
    // Accumulate Q_i and r_i from all input flats
    for (size_t i = 0; i < model_count; ++i) {
        models[i]->orthonormalize();
        auto [A, b] = models[i]->get_parametric_repr();
        auto [Q, r] = models[i]->get_QR();

        Q_star += Q * weights[i];
        r_star -= r * weights[i]; // Careful: In the paper it's the POSITIVE sum, but that is wrong!
        std::cout << "Q_star:\n" << Q << std::endl;
        std::cout << "r_star:\n" << r << std::endl;
        std::cout << "weights:\n" << weights[i] << std::endl;
    }
        std::cout << "======================" << std::endl;
    
    // eigen-decomposition on Q_star
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Q_star);
    if (eigensolver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition of Q_star failed.");
    }

    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues(); // Sorted in ascending order
    Eigen::MatrixXd U = eigensolver.eigenvectors(); // Columns are eigenvectors

    // Select the first k eigenvectors corresponding to the smallest k eigenvalues
    Eigen::MatrixXd A;
    if (k > 0) {
        A = U.leftCols(k); // n x k matrix
    } else {
        A = Eigen::MatrixXd::Zero(n, 0); // Empty matrix for k=0
    }


    Eigen::MatrixXd Q_star_pseudo_inv = pseudoInverse(Q_star);

    // I - A A^T (From the formular in the paper)
    Eigen::MatrixXd projection = Eigen::MatrixXd::Identity(n, n) - A * A.transpose();

    // b = (I - A A^T) * Q_star^+ * r_star
    Eigen::VectorXd b = projection * Q_star_pseudo_inv * r_star;

    // Override the parametric representation of the first model (also overrides all other representations)
    models[0]->override_parametric(A, b);

    // Verify that b is orthogonal to the subspace spanned by A
    Eigen::VectorXd check = A.transpose() * b;
    if (check.norm() > 1e-6) {
        throw std::runtime_error("b is not orthogonal to the subspace spanned by A.");
    }

    std::unique_ptr<FlatModel> result = std::move(models.front());
    return result;
}