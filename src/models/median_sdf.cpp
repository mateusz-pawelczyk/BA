#include "models/median_sdf.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <numeric>
#include <iostream>

std::unique_ptr<Model> MedianSDF::clone() const
{
    return std::make_unique<MedianSDF>(*this);
}

void MedianSDF::computeAggregatedQr(const std::vector<Eigen::MatrixXd> &Q_list, const std::vector<Eigen::VectorXd> &r_list, Eigen::MatrixXd &Q_star, Eigen::VectorXd &r_star, const std::vector<double> &weights) const
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

    computeMedianQ(Q_star, Q_list, new_weights);
    computeMedianR(r_star, r_list, new_weights);

    // If Q_star has nan values, print Q_list and r_list
    if (Q_star.hasNaN())
    {
        std::cout << "Q_star has NaN values. Printing Q_list and r_list:\n";
        for (int i = 0; i < model_count; i++)
        {
            std::cout << "Q_list[" << i << "]:\n"
                      << Q_list[i] << std::endl;
            std::cout << "r_list[" << i << "]:\n"
                      << r_list[i] << std::endl;
        }
    }
}

void MedianSDF::computeMedianQ(Eigen::MatrixXd &Q_star, std::vector<Eigen::MatrixXd> Q_list, const std::vector<double> &weights) const
{
    int n = Q_list.size();
    int r = Q_list[0].rows();
    int c = Q_list[0].cols();

    Eigen::MatrixXd prev(r, c);
    Eigen::MatrixXd curr(r, c);

    curr.setOnes();
    prev.setZero();

    // Initialize coefficients with normalized weights
    Eigen::ArrayXd coefs(n);
    double sum_weights = 0.0;
    for (double w : weights)
        sum_weights += w;
    for (int i = 0; i < n; ++i)
    {
        coefs(i) = weights[i] / sum_weights;
    }

    const double epsilon = 1e-10; // prevent division by zero
    int it = 0;

    Eigen::Map<const Eigen::ArrayXd> weights_array(weights.data(), n);

    while (it < max_it && (curr - prev).norm() > err_tol)
    {
        prev = curr;
        curr.setZero();
        for (int i = 0; i < n; i++)
        {
            curr += coefs(i) * Q_list[i];
        }

        for (int i = 0; i < n; i++)
        {
            coefs(i) = (curr - Q_list[i]).norm() + epsilon;
        }
        coefs = weights_array / coefs;
        coefs /= coefs.sum();

        it++;
    }

    Q_star = curr;
}

void MedianSDF::computeMedianR(Eigen::VectorXd &r_star, std::vector<Eigen::VectorXd> r_list, const std::vector<double> &weights) const
{
    int n = r_list.size();
    int r = r_list[0].rows();
    int c = r_list[0].cols();

    Eigen::MatrixXd prev(r, c);
    Eigen::MatrixXd curr(r, c);

    curr.setOnes();
    prev.setZero();

    // Initialize coefficients with normalized weights
    Eigen::ArrayXd coefs(n);
    double sum_weights = 0.0;
    for (double w : weights)
        sum_weights += w;
    for (int i = 0; i < n; ++i)
    {
        coefs(i) = weights[i] / sum_weights;
    }

    const double epsilon = 1e-10; // prevent division by zero
    int it = 0;

    Eigen::Map<const Eigen::ArrayXd> weights_array(weights.data(), n);

    while (it < max_it && (curr - prev).norm() > err_tol)
    {
        prev = curr;
        curr.setZero();
        for (int i = 0; i < n; i++)
        {
            curr += coefs(i) * r_list[i];
        }

        for (int i = 0; i < n; i++)
        {
            coefs(i) = (curr - r_list[i]).norm() + epsilon;
        }
        coefs = weights_array / coefs;
        coefs /= coefs.sum();

        it++;
    }

    r_star = curr;
}
