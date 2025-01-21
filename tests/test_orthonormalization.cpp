#include <gtest/gtest.h>
#include "core/flat_model.hpp"
#include "models/affine_fit.hpp"
#include <random>
#include <Eigen/QR>
#include <Eigen/Dense>

void arePointsOnOrthonormalPlane(const Eigen::MatrixXd& A, const Eigen::MatrixXd& X, const Eigen::VectorXd& b) {
    int N = X.rows();
    int n = A.rows();
    int d = A.cols();

    Eigen::MatrixXd P = A * A.transpose(); // Projection Matrix P = A (A^T A)^-1 A^T = A * I * A^T since A is orthonormal
    Eigen::MatrixXd P_perp = Eigen::MatrixXd::Identity(n, n) - P; // P_perp = I - P
    Eigen::MatrixXd B = b.transpose().replicate(N, 1); // (N x n) matrix with b in each row

    Eigen::MatrixXd X_perp = (X - B) * P_perp; // X_perp^T = P_perp * (X - b)

    // Now, the projections of the points onto the orthogonal complement of the column-space of A are int the rows of X_perp and should be zero

    for (int i = 0; i < X_perp.rows(); ++i) {
        ASSERT_NEAR(X_perp.row(i).norm(), 0.0, 1e-6);
    }
}

void test_parametric_orthonormal_equality(const Eigen::MatrixXd& A1, const Eigen::VectorXd& b_vec1, const Eigen::MatrixXd& A2, const Eigen::VectorXd& b_vec2, int d, int n, int num_points=100) {
    Eigen::MatrixXd points_subspace = Eigen::MatrixXd::Random(num_points, d);
    Eigen::MatrixXd points = points_subspace * A1.transpose() + b_vec1.transpose().replicate(num_points, 1);

    arePointsOnOrthonormalPlane(A2, points, b_vec2);
}


TEST(OrthonormalizationTest, Parametric_is_orthonormal) {
    for (int n = 2; n < 10; n++) {
        for (int d = 1; d < n; d++) {
            Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, d);
            Eigen::VectorXd b_vec = Eigen::VectorXd::Random(n);

            AffineFit model(d, n);
            model.override_parametric(A, b_vec);

            model.orthonormalize();

            auto [A_orthonormal, b_vec_orthonormal] = model.get_parametric_repr();
            Eigen::MatrixXd ATA = A_orthonormal.transpose() * A_orthonormal;

            Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(d, d);
            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    ASSERT_NEAR(ATA(i, j), identity(i, j), 1e-6);
                }
            }

            Eigen::VectorXd ATb = A_orthonormal.transpose() * b_vec_orthonormal;
            ASSERT_NEAR(ATb.norm(), 0.0, 1e-6);
        }
    }
    
}


TEST(OrthonormalizationTest, Parametric_is_same_subspace) {
    for (int n = 2; n < 10; n++) {
        for (int d = 1; d < n; d++) {
            Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, d);
            Eigen::VectorXd b_vec = Eigen::VectorXd::Random(n);

            AffineFit model(d, n);
            model.override_parametric(A, b_vec);

            model.orthonormalize();

            auto [A_orthonormal, b_vec_orthonormal] = model.get_parametric_repr();
            test_parametric_orthonormal_equality(A, b_vec, A_orthonormal, b_vec_orthonormal, d, n);
        }
    }
}


TEST(OrthonormalizationTest, Implicit_is_orthonormal) {
    for (int n = 2; n < 10; n++) {
        for (int d = 1; d < n; d++) {
            Eigen::MatrixXd N = Eigen::MatrixXd::Random(n - d, n);
            Eigen::VectorXd c = Eigen::VectorXd::Random(n - d);

            AffineFit model(d, n);
            model.override_implicit(N, c);

            model.orthonormalize();

            auto [N_orthonormal, c_orthonormal] = model.get_implicit_repr();
            Eigen::MatrixXd NNT = N_orthonormal * N_orthonormal.transpose();

            Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n - d, n - d);
            for (int i = 0; i < n - d; ++i) {
                for (int j = 0; j < n - d; ++j) {
                    ASSERT_NEAR(NNT(i, j), identity(i, j), 1e-6);
                }
            }
        }
    }
}

TEST(OrthonormalizationTest, Implicit_is_orthonormal_to_A) {
    for (int n = 2; n < 10; n++) {
        for (int d = 1; d < n; d++) {
            Eigen::MatrixXd N = Eigen::MatrixXd::Random(n - d, n);
            Eigen::VectorXd c = Eigen::VectorXd::Random(n - d);

            AffineFit model(d, n);
            model.override_implicit(N, c);
            auto [A, b_vec] = model.get_parametric_repr();

            model.orthonormalize();

            auto [N_orthonormal, c_orthonormal] = model.get_implicit_repr();

            Eigen::MatrixXd N_A = N_orthonormal * A;

            Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n - d, n - d);
            for (int i = 0; i < n - d; ++i) {
                for (int j = 0; j < d; ++j) {
                    ASSERT_NEAR(N_A(i, j), 0.0, 1e-6);
                }
            }
        }
    }
}

TEST(OrthonormalizationTest, Implicit_is_same_subspace) {
    for (int n = 2; n < 10; n++) {
        for (int d = 1; d < n; d++) {
            // Randomly generate the initial N and c
            Eigen::MatrixXd N = Eigen::MatrixXd::Random(n - d, n);
            Eigen::VectorXd c = Eigen::VectorXd::Random(n - d);

            // Initialize the model with the affine flat
            AffineFit model(d, n);
            model.override_implicit(N, c);
            auto [A, b_vec] = model.get_parametric_repr();

            // Ensure that the flat satisfies the parametric definition
            Eigen::VectorXd N_b = N * b_vec + c;
            ASSERT_NEAR(N_b.norm(), 0.0, 1e-6);

            // Orthonormalize the flat
            model.orthonormalize();

            auto [N_orthonormal, c_orthonormal] = model.get_implicit_repr();

            // Generate random points in the parametric subspace
            int num_points = 100;
            Eigen::MatrixXd subspace_points = Eigen::MatrixXd::Random(num_points, d);
            Eigen::MatrixXd points = subspace_points * A.transpose() + b_vec.transpose().replicate(num_points, 1);

            // Check that these points satisfy the orthonormalized implicit representation
            Eigen::MatrixXd complement_points = points * N_orthonormal.transpose() +
                                                c_orthonormal.transpose().replicate(num_points, 1);

            for (int i = 0; i < num_points; ++i) {
                ASSERT_NEAR(complement_points.row(i).norm(), 0.0, 1e-6);
            }
        }
    }
}
