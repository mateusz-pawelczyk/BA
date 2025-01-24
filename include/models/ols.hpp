#pragma once

#include <Eigen/Core>
#include <iostream>

#include "core/flat_model.hpp"

class OLS : public FlatModel
{
public:
    OLS(int d, int n) : FlatModel(d, n) {}

    void fit(const Eigen::MatrixXd &D) override;

    // Cloning
    std::unique_ptr<Model> clone() const override;
};