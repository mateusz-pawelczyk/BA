#pragma once

#include "core/flat_model.hpp"

class MedianSDF : public FlatModel
{
public:
    MedianSDF(int d, int n) : FlatModel(d, n) {}

    void fit(const Eigen::MatrixXd &D) override;
    void fit(const std::vector<std::unique_ptr<FlatModel>> &models, const std::vector<double> &errors = std::vector<double>());

    std::unique_ptr<Model> clone() const override;

private:
    Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd &A, double tolerance = 1e-6);
    std::vector<double> getWeights(const std::vector<double> &errors, int model_count) const;
    void validateFlatDimensions(const std::vector<std::unique_ptr<FlatModel>> &models) const;
};