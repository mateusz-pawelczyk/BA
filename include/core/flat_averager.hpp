#pragma once

#include "core/flat_model.hpp"

class FlatAverager : public FlatModel
{

public:
    virtual ~FlatAverager() noexcept = default;
    FlatAverager(int d, int n) : FlatModel(d, n) {}

    virtual void fit(const Eigen::MatrixXd &D) override;
    // virtual void fit(const std::vector<std::unique_ptr<FlatModel>> &models);
    virtual void fit(const std::vector<std::unique_ptr<FlatModel>> &models, const std::vector<double> &errors = {});

    virtual std::unique_ptr<Model> clone() const override = 0;

protected:
    virtual void computeAggregatedQr(const std::vector<Eigen::MatrixXd> &Q_list,
                                     const std::vector<Eigen::VectorXd> &r_list,
                                     Eigen::MatrixXd &Q_star,
                                     Eigen::VectorXd &r_star,
                                     const std::vector<double> &weights = {}) const = 0;

private:
    Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd &A, double tolerance = 1e-6);
    std::vector<double> getWeights(const std::vector<double> &errors, int model_count) const;
    void validateFlatDimensions(const std::vector<std::unique_ptr<FlatModel>> &models) const;
};