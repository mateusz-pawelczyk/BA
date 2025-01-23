#include <gtest/gtest.h>
#include "core/flat_model.hpp"
#include "models/affine_fit.hpp"
#include "core/ransac.hpp"
#include <random>

// TEST(FlatOrientationInvariance, MSE) {
//     for (int n = 2; n < 10; n++) {
//         for (int d = 1; d < n; d++) {
//             Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, d);
//             Eigen::VectorXd b_vec = Eigen::VectorXd::Random(n);

//             AffineFit model(d, n);
//             RANSAC ransac(1000, 0.0001, 0.2, 0.3 * 50, MetricType::MSE);
//         }
//     }

// }