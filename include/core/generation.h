// Generation.h
#ifndef GENERATION_H  // Include guard to prevent multiple inclusions
#define GENERATION_H

#include <Eigen/Core>
#include <vector>

namespace Generation {
    Eigen::MatrixXd generateHyperPlane(const Eigen::VectorXd& w, double b, int N, double noise = 1.0, double outlierRatio = 0.0, double outlierStrength = 5.0);
    void flatParameterGUI(Eigen::VectorXf& w_ui, float& b_ui);
}
// #include "Generation.tpp"
#endif  // GENERATION_H
