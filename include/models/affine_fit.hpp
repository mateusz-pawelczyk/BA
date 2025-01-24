#pragma once

#include "core/flat_model.hpp"

class AffineFit : public FlatModel
{
public:
    AffineFit(int d, int n) : FlatModel(d, n) {}

    void fit(const Eigen::MatrixXd &D) override;

    // Cloning
    std::unique_ptr<Model> clone() const override;

private:
};