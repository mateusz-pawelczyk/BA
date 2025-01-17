#include "core/model.hpp"
#include <iostream>

double Model::MSE(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y_true) {
    Eigen::VectorXd Y_pred = predict(X);

    Eigen::VectorXd residuals = Y_true - Y_pred;

    return residuals.squaredNorm() / Y_true.size();
}
