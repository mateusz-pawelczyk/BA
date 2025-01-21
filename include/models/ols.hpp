#pragma once

#include <Eigen/Core>
#include <iostream>

#include "core/flat_model.hpp"

class OLS : public FlatModel {
public:
    OLS(int d, int n) : FlatModel(d, n) {}

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y) override;
    void fit(const Eigen::MatrixXd& D) override;

    double predict(const Eigen::VectorXd& point) const override;
    Eigen::VectorXd predict(const Eigen::MatrixXd& data) const override;
    
    // Cloning
    std::unique_ptr<Model> clone() const override;

    // Visualize
    void visualize(const std::string& name, double sideLen, double lineRadius, float flatAlpha) override;

};