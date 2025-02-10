#pragma once

#include "core/flat_model.hpp"
#include <Eigen/Core>

namespace FlatSampler
{
    // FlatMode -> Points
    Eigen::MatrixXd sampleFlat(FlatModel &model, int N, double noise = 0.0, double outlierRatio = 0.0, double outlierStrength = 1.0, bool saltAndPepper = false);
    std::vector<std::unique_ptr<FlatModel>> sampleFlat2(FlatModel &model, int N, int k, double noise = 0.0, double outlierRatio = 0.0, double outlierStrength = 1.0, bool saltAndPepper = false);

    // === Helper Functions ===
    Eigen::MatrixXd sampleGaussianPoints(int N, int d);
    Eigen::MatrixXd sampleCanonicalPoints(int num_points, int dim);
    Eigen::MatrixXd sampleCanonicalGaussianPoints(int N, int d);

    void addNoise(Eigen::MatrixXd &points, const Eigen::MatrixXd &N, double noise);
    void addOutlier(Eigen::MatrixXd &points, const Eigen::MatrixXd &N, double strength, double ratio, bool saltAndPepper);
}