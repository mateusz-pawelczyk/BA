#ifndef REGRESSION_H
#define REGRESSION_H

#include <vector>
#include <tuple>
#include <Eigen/Core>



namespace Regression {
    std::tuple<Eigen::Vector2d, double> OLS3d(Eigen::MatrixXd points);
    std::tuple<Eigen::VectorXd, double> OLS(Eigen::MatrixXd points);

}

#endif