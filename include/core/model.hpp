#pragma once

#include <Eigen/Core>
#include <memory>

class Model {
public:
    virtual ~Model() = default;  

    virtual std::unique_ptr<Model> clone() const = 0;

    virtual void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y) = 0;  // No const at the end, because fit will override the model parameter
    virtual void fit(const Eigen::MatrixXd& D) = 0;
    
    virtual double predict(const Eigen::VectorXd& point) const = 0;
    virtual Eigen::VectorXd predict(const Eigen::MatrixXd& data) const = 0; // const at the end to signal that no member variable will be overriden

    virtual double MSE(const Eigen::MatrixXd& X_true, const Eigen::VectorXd& Y_true);

    // Visualize
    virtual void visualize(const std::string& name, double sideLen, double lineRadius, float flatAlpha) = 0;
};

