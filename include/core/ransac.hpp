#pragma once

#include <Eigen/Core>
#include <memory>
#include <random>
#include <queue>

#include "core/model.hpp"
#include "core/flat_model.hpp"

using FlatModelEntry = std::pair<double, std::unique_ptr<FlatModel>>;

class RANSAC {
public:
    RANSAC(int max_iterations, double threshold, double train_data_percenatge, int min_inliners, Eigen::VectorXd (*loss_fn)(Eigen::VectorXd, Eigen::VectorXd), double (*metric_fn)(Eigen::VectorXd, Eigen::VectorXd)) 
        : max_iterations(max_iterations), threshold(threshold), train_data_percentage(train_data_percenatge), min_inliners(min_inliners), loss_fn(loss_fn), metric_fn(metric_fn) {}
    std::unique_ptr<Model> run(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y, Model* model);

    std::unique_ptr<FlatModel> run(const Eigen::MatrixXd& D, FlatModel* model, int best_model_count=1) const;
    std::unique_ptr<FlatModel> run2(const Eigen::MatrixXd& D, FlatModel* model, int best_model_count=1) const;

private:
    Eigen::VectorXd getInliner(const Eigen::VectorXd& Y, const Eigen::VectorXd& Y_pred) const;
    int max_iterations;
    double threshold;
    double train_data_percentage;
    int min_inliners;

    Eigen::VectorXd (*loss_fn)(Eigen::VectorXd, Eigen::VectorXd);
    double (*metric_fn)(Eigen::VectorXd, Eigen::VectorXd);

    // MedianSDF
    std::unique_ptr<FlatModel> medianSDF(std::vector<std::unique_ptr<FlatModel>>& models, int k, std::vector<double>* errors = nullptr) const;



    // === helper methods for RANSAC ===

    // Helper function to cast Model -> FlatModel in a safe way
    std::unique_ptr<FlatModel> castToModel(std::unique_ptr<Model> basePtr) const;

   
    void sampleRandomSubset(const Eigen::MatrixXd &X, 
                            const Eigen::VectorXd &Y,
                            Eigen::MatrixXd &X_subset, 
                            Eigen::VectorXd &Y_subset,
                            std::vector<int> &indices,
                            std::mt19937 &g) const;

    void sampleRandomSubset(const Eigen::MatrixXd &D,
                                Eigen::MatrixXd &D_subset,
                                std::vector<int> &indices, 
                                std::mt19937 &g) const;

    std::vector<int> findInliers(const Eigen::VectorXd &loss_values, double threshold) const;

    // Gathers top models from the global heap into sorted vectors
    template <typename Comparator>
    void gatherTopModels(
        std::priority_queue<
            FlatModelEntry, 
            std::vector<FlatModelEntry>,
            Comparator
        > &heap,
        std::vector<std::unique_ptr<FlatModel>> &models, 
        std::vector<double> &errors
    ) const;

    // Helper function to merge local heaps into the global heap
    template <typename Comparator>
    void mergeLocalHeap(
        std::priority_queue<
            FlatModelEntry, 
            std::vector<FlatModelEntry>,
            Comparator
        > &local_heap,
        std::priority_queue<
            FlatModelEntry, 
            std::vector<FlatModelEntry>,
            Comparator
        > &heap
    ) const;


    // === Helper methods for MedianSDF ===
    std::vector<double> getWeights(std::vector<double>* errors) const;

    std::pair<int, int> validateAndGetFlatDimensions(const std::vector<std::unique_ptr<FlatModel>>& models) const;

};

    #include "core/ransac.tpp"
