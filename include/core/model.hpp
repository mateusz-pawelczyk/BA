#pragma once

#include <Eigen/Core>
#include <memory>

class Model
{
public:
    virtual ~Model() noexcept = default;

    virtual std::unique_ptr<Model> clone() const = 0;

    virtual void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y) = 0; // No const at the end, because fit will override the model parameter
    virtual void fit(const Eigen::MatrixXd &D) = 0;

    // Predicting possibly multivariate outputs, which is why one prediction is a vector and multiple predictions are a matrix
    virtual Eigen::VectorXd predict(const Eigen::VectorXd &point) = 0;
    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd &data) = 0;

    virtual double MSE(const Eigen::MatrixXd &X_true, const Eigen::VectorXd &Y_true);

    // Visualize
    virtual void visualize(const std::string &name, double sideLen, double lineRadius, float flatAlpha) = 0;
};
