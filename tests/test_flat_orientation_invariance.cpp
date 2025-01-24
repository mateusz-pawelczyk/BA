#include <gtest/gtest.h>
#include "core/flat_model.hpp"
#include "models/affine_fit.hpp"
#include "core/ransac.hpp"
#include "core/flat_sampler.hpp"
#include <random>

// TEST(FlatOrientationInvariance, MSE)
// {
//     for (int n = 2; n < 10; n++)
//     {
//         for (int d = 1; d < n; d++)
//         {
//             int N = 1000;

//             Eigen::VectorXd mse = Eigen::VectorXd::Zero(100);
//             for (int i = 0; i < 100; ++i)
//             {
//                 Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, d);
//                 Eigen::VectorXd b_vec = Eigen::VectorXd::Random(n);

//                 AffineFit model(d, n);
//                 RANSAC ransac(1000, 0.0001, 0.2, 0.3 * 0.2 * 1000, MetricType::MSE);

//                 AffineFit *m = new AffineFit(n - 1, n);
//                 m->override_parametric(A, b_vec);

//                 Eigen::MatrixXd points = FlatSampler::sampleFlat(*m, N, 0.25, 0.25, 10.0, 1.0, true);

//                 std::unique_ptr<FlatModel> averagedBestModel = ransac.run2(points, m, 10);
//                 mse(i) = averagedBestModel->MSE(points.leftCols(d), points.col(d));
//             }

//             // Calculate the mean and standard deviation of the MSE
//             double mean = mse.mean();
//             double std = mse.array().sqrt().mean();

//             // Assert if the standard deviation is less than 0.1
//             ASSERT_LT(std, 0.1);
//         }
//     }
// }