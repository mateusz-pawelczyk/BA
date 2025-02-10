#include "models/mean_sdf.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <numeric>
#include <iostream>

std::unique_ptr<Model> MeanSDF::clone() const
{
    return std::make_unique<MeanSDF>(*this);
}

void MeanSDF::computeAggregatedQr(const std::vector<Eigen::MatrixXd> &Q_list, const std::vector<Eigen::VectorXd> &r_list, Eigen::MatrixXd &Q_star, Eigen::VectorXd &r_star, const std::vector<double> &weights) const
{
    int model_count = Q_list.size();

    // Initialize Q_star and r_star
    Q_star = Eigen::MatrixXd::Zero(n, n);
    r_star = Eigen::VectorXd::Zero(n);

    std::vector<double> new_weights = weights;
    if (weights.empty())
    {
        double model_count_inv = 1.0 / static_cast<double>(model_count);

        new_weights = std::vector<double>(model_count, model_count_inv);
    }

    for (size_t i = 0; i < model_count; ++i)
    {
        Q_star += Q_list[i] * new_weights[i];
        r_star += r_list[i] * new_weights[i];
    }
}
