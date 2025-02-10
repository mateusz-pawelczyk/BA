#pragma once

#include "core/flat_averager.hpp"

class MeanSDF : public FlatAverager
{
public:
    MeanSDF(int d, int n) : FlatAverager(d, n) {}

    std::unique_ptr<Model> clone() const override;

protected:
    void computeAggregatedQr(const std::vector<Eigen::MatrixXd> &Q_list,
                             const std::vector<Eigen::VectorXd> &r_list,
                             Eigen::MatrixXd &Q_star,
                             Eigen::VectorXd &r_star,
                             const std::vector<double> &weights = {}) const override;
};