#pragma once

#include "core/flat_model.hpp" // Base class
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <memory>
#include <vector>
#include <stdexcept>

struct HuberCostFunctor
{
    // Constructor: stores feature vector and target.
    HuberCostFunctor(const Eigen::VectorXd &x, double y) : x_(x), y_(y) {}

    // New operator() signature for DynamicAutoDiffCostFunction.
    template <typename T>
    bool operator()(T const *const *parameters, T *residuals) const
    {
        // parameters[0] is our single parameter block.
        const T *param = parameters[0];
        T prediction = T(0);
        // Compute dot(w, x)
        for (int j = 0; j < x_.size(); ++j)
        {
            prediction += param[j] * T(x_(j));
        }
        // Add the bias term (last parameter)
        prediction += param[x_.size()];
        residuals[0] = T(y_) - prediction;
        return true;
    }

    const Eigen::VectorXd x_;
    const double y_;
};

// HuberRegression class derived from FlatModel.
// It implements a robust regression using a Huber loss function via Ceres.
class HuberRegression : public FlatModel
{
public:
    // Constructor: d is the flat's dimension, n the ambient space dimension.
    HuberRegression(int d, int n);
    virtual ~HuberRegression() noexcept override = default;

    // Fit the model using a data matrix D.
    // Assumes each row of D is: [feature1, feature2, ..., feature_k, target]
    void fit(const Eigen::MatrixXd &D) override;

    // Create a clone of this model.
    std::unique_ptr<Model> clone() const override;
};