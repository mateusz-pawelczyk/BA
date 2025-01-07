#ifndef REGRESSION_CPP
#define REGRESSION_CPP

#include "Regression.h"
#include <Eigen/Cholesky>
#include <iostream>
    
namespace Regression {
    std::tuple<Eigen::Vector2d, double> OLS3d(Eigen::MatrixXd X) {
        int d = X.cols() - 1;

        Eigen::VectorXd Y = X.col(d); // Y = last column of X
        Eigen::MatrixXd tmp = X.leftCols(d); // temporarily save the left columns so we can shift them to the right
        X.rightCols(d) = tmp; // shift left columns to the right
        X.col(0).setConstant(1.0); // first column contains just 1s for b

        Eigen::VectorXd w_prime = (X.transpose() * X).ldlt().solve(X.transpose() * Y);

        double b = w_prime.coeff(0);  // Extract bias
        Eigen::Vector2d w = w_prime.segment(1, 2);  // Extract 2D weights

        return std::make_tuple(w, b);
    }

    std::tuple<Eigen::VectorXd, double> OLS(Eigen::MatrixXd X) {
        int d = X.cols() - 1;

        Eigen::VectorXd Y = X.col(d); // Y = last column of X
        Eigen::MatrixXd tmp = X.leftCols(d); // temporarily save the left columns so we can shift them to the right
        X.rightCols(d) = tmp; // shift left columns to the right
        X.col(0).setConstant(1.0); // first column contains just 1s for b

        Eigen::VectorXd w_prime = (X.transpose() * X).ldlt().solve(X.transpose() * Y);

        double b = w_prime.coeff(0);  // Extract bias
        Eigen::VectorXd w = w_prime.segment(1, d);  // Extract weights

        return std::make_tuple(w, b);
    }
}

#endif