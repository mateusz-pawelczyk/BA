#include <gtest/gtest.h>
#include "core/flat_model.hpp"
#include "models/affine_fit.hpp"
#include <random>


// Helper function to generate random doubles in a range
double random_double(double min, double max) {
    return min + static_cast<double>(rand()) / RAND_MAX * (max - min);
}

void test_parametric_explicit_equality(const Eigen::MatrixXd& A, const Eigen::VectorXd& b_vec, const Eigen::VectorXd& w, double b, int d, int num_points=100) {
    Eigen::MatrixXd y_points = Eigen::MatrixXd::Random(num_points, d);
    Eigen::MatrixXd x_points = y_points * A.transpose() + b_vec.transpose().replicate(num_points, 1);

    Eigen::MatrixXd z_points = w.transpose() * x_points.leftCols(d).transpose() + Eigen::RowVectorXd::Ones(num_points) * b;

    // Check if z_points == x_points.col(d)
    for (int i = 0; i < num_points; ++i) {
        ASSERT_NEAR(z_points(i), x_points(i, d), 1e-6);
    }
}

void test_explicit_implicit_equality(const Eigen::MatrixXd& N, const Eigen::VectorXd& c, const Eigen::VectorXd& w, double b, int d, int num_points=100) {
    Eigen::MatrixXd points = Eigen::MatrixXd::Random(num_points, d + 1);
    Eigen::VectorXd points_last = w.transpose() * points.leftCols(d).transpose() + Eigen::RowVectorXd::Ones(num_points) * b;
    points.col(d) = points_last;

    Eigen::MatrixXd complement_points = points * N.transpose() + c.transpose().replicate(num_points, 1);

    // Check if complement_points == 0
    for (int i = 0; i < num_points; ++i) {
        ASSERT_NEAR(complement_points(i, 0), 0.0, 1e-6);
    }
}

void test_parametric_implicit_equality(const Eigen::MatrixXd& A, const Eigen::VectorXd& b_vec, const Eigen::MatrixXd& N, const Eigen::VectorXd& c, int d, int n, int num_points=100) {
    Eigen::MatrixXd points_subspace = Eigen::MatrixXd::Random(num_points, d);
    Eigen::MatrixXd points = points_subspace * A.transpose() + b_vec.transpose().replicate(num_points, 1);

    Eigen::MatrixXd complement_points = points * N.transpose() + c.transpose().replicate(num_points, 1);

    // Check if complement_points == 0
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < n - d; ++j) {
            ASSERT_NEAR(complement_points(i, j), 0.0, 1e-6);
        }
    }
}

TEST(ConversionTest, ExplicitToParametric_PointSatisfyability) {
    int num_parameter_changes = 10; // Number of random test cases
    int max_n = 10; // Maximum number of dimensions

    for (int d = 1; d < 10; d++) {
        for (int t = 0; t < num_parameter_changes; ++t) {
            // Randomly generate plane parameters
            Eigen::VectorXd w = Eigen::VectorXd::Random(d);
            double b = random_double(-100.0, 100.0);

            // Create and initialize the model
            AffineFit model(d, d + 1);
            std::cout << "Overriding explicit" << std::endl;
            model.override_explicit(w, b);

            // Convert to parametric representation
            auto [A, b_vec] = model.get_parametric_repr();

            test_parametric_explicit_equality(A, b_vec, w, b, d);
        }
    }
}


TEST(ConversionTest, ParametricToExplicit_PointSatisfyability) {
    int num_parameter_changes = 10; // Number of random test cases
    int max_n = 10; // Maximum number of dimensions

    for (int d = 1; d < 10; d++) {
        for (int t = 0; t < num_parameter_changes; ++t) {
            // Randomly generate plane parameters
            Eigen::MatrixXd A = Eigen::MatrixXd::Random(d + 1, d);
            Eigen::VectorXd b_vec = Eigen::VectorXd::Random(d + 1);

            // Create and initialize the model
            AffineFit model(d, d + 1);
            model.override_parametric(A, b_vec);

            // Convert to parametric representation
            auto [w, b] = model.get_explicit_repr();

            // Sample `y` points from the equation x = A*y + b_vec
            test_parametric_explicit_equality(A, b_vec, w, b, d);
        }
    }
}

TEST(ConversionTest, ExplicitToImplicit_PointSatisfyability) {
    int num_parameter_changes = 10; // Number of random test cases
    int max_n = 10; // Maximum number of dimensions

    for (int d = 1; d < 10; d++) {
        for (int t = 0; t < num_parameter_changes; ++t) {
            // Randomly generate plane parameters
            Eigen::VectorXd w = Eigen::VectorXd::Random(d);
            double b = random_double(-100.0, 100.0);

            // Create and initialize the model
            AffineFit model(d, d + 1);
            std::cout << "Overriding explicit" << std::endl;

            model.override_explicit(w, b);

            // Convert to parametric representation
            auto [N, c] = model.get_implicit_repr();

            test_explicit_implicit_equality(N, c, w, b, d);
        }
    }
}


TEST(ConversionTest, ImplicitToExplicit_PointSatisfyability) {
    int num_parameter_changes = 10; // Number of random test cases
    int max_n = 10; // Maximum number of dimensions

    for (int d = 1; d < 10; d++) {
        for (int t = 0; t < num_parameter_changes; ++t) {
            // Randomly generate plane parameters
            Eigen::MatrixXd N = Eigen::MatrixXd::Random(1, d + 1);
            Eigen::VectorXd c = Eigen::VectorXd::Random(1);

            // Create and initialize the model
            AffineFit model(d, d + 1);
            model.override_implicit(N, c);

            // Convert to parametric representation
            auto [w, b] = model.get_explicit_repr();

            test_explicit_implicit_equality(N, c, w, b, d);
        }
    }
}

TEST(ConversionTest, ParametricToImplicit_PointSatisfyability) {
    int num_parameter_changes = 10; // Number of random test cases
    int max_n = 10; // Maximum number of dimensions

    for (int n = 2; n < max_n; n++) {
        for (int d = 1; d < n; d++) {
            for (int t = 0; t < num_parameter_changes; ++t) {
                // Randomly generate plane parameters
                Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, d);
                Eigen::VectorXd b_vec = Eigen::VectorXd::Random(n);

                // Create and initialize the model
                AffineFit model(d, n);
                model.override_parametric(A, b_vec);

                // Convert to parametric representation
                auto [N, c] = model.get_implicit_repr();

                // Check for orthogonality between A and N
                Eigen::MatrixXd N_A = N * A;
                
                // Check if A_N == 0
                for (int i = 0; i < n - d; ++i) {
                    for (int j = 0; j < d; ++j) {
                        ASSERT_NEAR(N_A(i, j), 0.0, 1e-6);
                    }
                }

                // Sample `y` points from the equation x = A*y + b_vec
                test_parametric_implicit_equality(A, b_vec, N, c, d, n);
            }
        }
    }
}

TEST(ConversionTest, ImplicitToParametric_PointSatisfyability) {
    int num_parameter_changes = 10; // Number of random test cases
    int max_n = 10; // Maximum number of dimensions

    for (int n = 2; n < max_n; n++) {
        for (int d = 1; d < n; d++) {
            for (int t = 0; t < num_parameter_changes; ++t) {
                // Randomly generate plane parameters
                Eigen::MatrixXd N = Eigen::MatrixXd::Random(n - d, n);
                Eigen::VectorXd c = Eigen::VectorXd::Random(n - d);

                // Create and initialize the model
                AffineFit model(d, n);
                model.override_implicit(N, c);

                // Convert to parametric representation
                auto [A, b_vec] = model.get_parametric_repr();

                // Check for orthogonality between A and N
                Eigen::MatrixXd N_A = N * A;
                
                // Check if A_N == 0
                for (int i = 0; i < n - d; ++i) {
                    for (int j = 0; j < d; ++j) {
                        ASSERT_NEAR(N_A(i, j), 0.0, 1e-6);
                    }
                }

                // Sample `y` points from the equation x = A*y + b_vec
                test_parametric_implicit_equality(A, b_vec, N, c, d, n);
            }
        }
    }
}

TEST(ConversionTest, OverrideParametric_ImplicitConversion) {
    int num_parameter_changes = 10; // Number of random test cases
    int max_n = 10; // Maximum number of dimensions

    for (int n = 2; n < max_n; n++) {
        for (int d = 1; d < n; d++) {
            for (int t = 0; t < num_parameter_changes; ++t) {
                // Randomly generate plane parameters
                Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, d);
                Eigen::VectorXd b_vec = Eigen::VectorXd::Random(n);

                Eigen::MatrixXd N = Eigen::MatrixXd::Random(n - d, n);
                Eigen::VectorXd c = Eigen::VectorXd::Random(n - d);

                // Create and initialize the model
                AffineFit model(d, n);
                model.override_implicit(N, c);
                model.override_parametric(A, b_vec);

                auto [N_new, c_new] = model.get_implicit_repr();

                // Check for orthogonality between A and N
                Eigen::MatrixXd N_A = N_new * A;
                
                // Check if A_N == 0
                for (int i = 0; i < n - d; ++i) {
                    for (int j = 0; j < d; ++j) {
                        ASSERT_NEAR(N_A(i, j), 0.0, 1e-6);
                    }
                }

                // Sample `y` points from the equation x = A*y + b_vec
                test_parametric_implicit_equality(A, b_vec, N_new, c_new, d, n);

            }
        }
    }
}               


