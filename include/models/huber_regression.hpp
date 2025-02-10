#pragma once

#include "core/flat_model.hpp" // Base class
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <memory>
#include <vector>
#include <stdexcept>

// Cost functor for Huber regression.
struct HuberCostFunctor
{
    HuberCostFunctor(const Eigen::VectorXd &x, double y);

    template <typename T>
    bool operator()(const T *const parameters, T *residual) const;

    const Eigen::VectorXd x_;
    const double y_;
};

// HuberRegression class, derived from FlatModel.
class HuberRegression : public FlatModel
{
public:
    HuberRegression(int d, int n);
    virtual ~HuberRegression() noexcept override = default;

    void fit(const Eigen::MatrixXd &D) override;
    double predict(const Eigen::VectorXd &point) const override;
    Eigen::VectorXd predict(const Eigen::MatrixXd &data) const override;
    std::unique_ptr<Model> clone() const override;
};
