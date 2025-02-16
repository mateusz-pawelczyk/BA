#pragma once

#include <Eigen/Core>
#include <memory>
#include <random>
#include <queue>

#include "core/model.hpp"
#include "core/flat_model.hpp"
#include "models/mean_sdf.hpp"

// different metrics (aws ENUM or something)
enum class MetricType
{
    R2_Orthogonal,
    R2_Regression,
    MSE_Orthogonal,
    MSE_Regression
};

using FlatModelEntry = std::pair<double, std::unique_ptr<FlatModel>>;

class RANSAC
{
public:
    RANSAC(int max_iterations, double threshold, double train_data_percenatge, int min_inliners, MetricType metric = MetricType::R2_Orthogonal);
    std::unique_ptr<Model> run(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y, Model *model, std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> loss_fn, std::function<double(Eigen::VectorXd, Eigen::VectorXd)> metric_fn);

    std::unique_ptr<FlatModel> run(const Eigen::MatrixXd &D, FlatModel *model, int best_model_count, std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> loss_fn, std::function<double(Eigen::VectorXd, Eigen::VectorXd)> metric_fn, FlatAverager *averager) const;
    std::unique_ptr<FlatModel> run2(const Eigen::MatrixXd &D,
                                    FlatModel *model,
                                    int best_model_count,
                                    FlatAverager *averager,
                                    bool weighted_average = false) const;

    std::unique_ptr<FlatModel> run_slow(const Eigen::MatrixXd &D,
                                        FlatModel *model,
                                        int best_model_count,
                                        FlatAverager *averager,
                                        bool weighted_average = false) const;

private:
    Eigen::VectorXd getInliner(const Eigen::VectorXd &Y, const Eigen::VectorXd &Y_pred) const;
    int max_iterations;
    double threshold;
    double train_data_percentage;
    int min_inliners;

    // Eigen::VectorXd (*loss_fn)(Eigen::VectorXd, Eigen::VectorXd);
    // double (*metric_fn)(Eigen::VectorXd, Eigen::VectorXd);
    std::function<double(Eigen::MatrixXd, FlatModel *)> metric_fn2;
    std::function<Eigen::VectorXd(Eigen::MatrixXd, FlatModel *)> loss_fn;

    // FlatAverager
    // std::unique_ptr<FlatModel> FlatAverager(std::vector<std::unique_ptr<FlatModel>> &models, int k, std::vector<double> *errors = nullptr) const;

    // === Evaluation Metrics ===
    // double r2_metric(Eigen::MatrixXd D, FlatModel *model) const;
    // double rss_metric(Eigen::MatrixXd D, FlatModel *model) const;
    // double mse_metric(Eigen::MatrixXd D, FlatModel *model) const;
    // double rmse_metric(Eigen::MatrixXd D, FlatModel *model) const;

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
            Comparator> &heap,
        std::vector<std::unique_ptr<FlatModel>> &models,
        std::vector<double> &errors) const;

    // Helper function to merge local heaps into the global heap
    template <typename Comparator>
    void mergeLocalHeap(
        std::priority_queue<
            FlatModelEntry,
            std::vector<FlatModelEntry>,
            Comparator> &local_heap,
        std::priority_queue<
            FlatModelEntry,
            std::vector<FlatModelEntry>,
            Comparator> &heap) const;
};

#include "core/ransac.tpp"
