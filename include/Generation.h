// Generation.h
#ifndef GENERATION_H  // Include guard to prevent multiple inclusions
#define GENERATION_H

#include <Eigen/Core>
#include <vector>

namespace Generation {
    Eigen::MatrixXd generateHyperPlane3d(const Eigen::Vector2d& w, double b, int N, double noise = 1.0, double outlierRatio = 0.0, double outlierStrength = 5.0);
    Eigen::MatrixXd generateHyperPlane(const Eigen::VectorXd& w, double b, int N, double noise = 1.0, double outlierRatio = 0.0, double outlierStrength = 5.0);
}
// #include "Generation.tpp"
#endif  // GENERATION_H
