#pragma once

#include "core/flat_averager.hpp"

class MedianSDF : public FlatAverager
{
public:
    MedianSDF(int d, int n, double err_tol, double max_it) : FlatAverager(d, n), err_tol(err_tol), max_it(max_it) {}

    std::unique_ptr<Model> clone() const override;

protected:
    void computeAggregatedQr(const std::vector<Eigen::MatrixXd> &Q_list,
                             const std::vector<Eigen::VectorXd> &r_list,
                             Eigen::MatrixXd &Q_star,
                             Eigen::VectorXd &r_star,
                             const std::vector<double> &weights = {}) const override;

private:
    double err_tol;
    double max_it;

    void computeMedianQ(Eigen::MatrixXd &Q_star, std::vector<Eigen::MatrixXd> Q_list, const std::vector<double> &weights) const;
    void computeMedianR(Eigen::VectorXd &r_star, std::vector<Eigen::VectorXd> r_list, const std::vector<double> &weights) const;
};