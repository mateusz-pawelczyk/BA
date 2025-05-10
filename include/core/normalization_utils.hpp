// normalization_utils.hpp
#pragma once
#include <Eigen/Core>
#include <memory>
#include <stdexcept>
#include "core/flat_model.hpp"

/// Holds per-column mean & std-dev.
struct NormParams {
  Eigen::VectorXd mu;     // length = p+1
  Eigen::VectorXd sigma;  // length = p+1
};

/// Normalize a dataset D (N x (p+1)), writing result into D_norm.
/// Returns the NormParams so you can later de-normalize.
inline NormParams normalizeDataset(
    const Eigen::MatrixXd &D,
    Eigen::MatrixXd       &D_norm)
{
    const int N = D.rows();
    const int M = D.cols();  // = p+1
    D_norm.resize(N, M);

    NormParams params;
    params.mu    .resize(M);
    params.sigma .resize(M);

    for (int j = 0; j < M; ++j) {
        // 1) Compute mean
        double μ = D.col(j).mean();
        // 2) Compute population std-dev
        Eigen::ArrayXd centered = D.col(j).array() - μ;
        double σ = std::sqrt((centered.square().sum()) / N);
        if (σ == 0.0) σ = 1.0;

        // Store
        params.mu(j)    = μ;
        params.sigma(j) = σ;

        // 3) Normalize column j
        D_norm.col(j) = (D.col(j).array() - μ) / σ;
    }

    return params;
}

/// Given a FlatModel trained on *normalized* data, undo the normalization
/// and return a *new* FlatModel in the original scale.
inline std::unique_ptr<FlatModel> denormalizeFlatModel(
    FlatModel    &modelNorm,
    const NormParams   &params)
{
    // 1) Extract normalized weights & bias
    auto [Wnorm_mat, Bnorm_vec] = modelNorm.get_explicit_repr();
    const int p = Wnorm_mat.rows();       // number of features
    if ((int)params.mu.size() != p+1)
        throw std::invalid_argument("NormParams size mismatch.");

    Eigen::VectorXd Wnorm = Wnorm_mat.col(0); 
    double           Bnorm = Bnorm_vec(0);

    // 2) Compute original weights & bias
    Eigen::VectorXd Worig(p);
    for (int j = 0; j < p; ++j) {
        Worig(j) = (params.sigma(p) / params.sigma(j)) * Wnorm(j);
    }
    double Borig = params.mu(p)
                 + params.sigma(p)*Bnorm
                 - Worig.dot(params.mu.head(p));

    // 3) Clone the model and override its explicit representation
    std::unique_ptr<Model> baseClone = modelNorm.clone();  
    Model *        rawPtr     = baseClone.release();
    auto *flatPtr = dynamic_cast<FlatModel*>(rawPtr);
    if (!flatPtr) {
      delete rawPtr;
      throw std::runtime_error("denormalize: clone() not a FlatModel");
    }
    std::unique_ptr<FlatModel> modelOrig(flatPtr);

    // 4) Apply de-normalized coefficients
    Eigen::MatrixXd Wmat = Worig;            // (p x 1)
    Eigen::VectorXd Bvec(1); 
    Bvec(0) = Borig;
    modelOrig->override_explicit(Wmat, Bvec);

    return modelOrig;
}
