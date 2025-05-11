#include "models/huber_regression.hpp" // Corresponding header
#include <Eigen/Dense>                 // For Eigen matrix operations
#include <Eigen/Eigenvalues>           // For SelfAdjointEigenSolver
#include <ceres/ceres.h>               // Ceres Solver main header
#include <vector>                      // For std::vector
#include <iostream>                    // For cerr/cout (optional, for debugging)

// Using the detail namespace for functors
using namespace huber_regression_detail;

HuberRegression::HuberRegression(int d_features,
                                 int n_ambient,
                                 DistanceType dist_type,
                                 double delta)
  // Initialize FlatModel:
  // For vertical regression (y = w^T x + b):
  //   The flat is the graph of the function, dim d_features, in ambient space (d_features+1).
  //   So, FlatModel(d_features, d_features + 1).
  // For orthogonal regression (hyperplane in n_ambient space):
  //   The flat is an (n_ambient-1) dimensional hyperplane in n_ambient space.
  //   So, FlatModel(n_ambient - 1, n_ambient).
  : FlatModel(dist_type == DistanceType::Regression ? d_features : (n_ambient - 1),
              dist_type == DistanceType::Regression ? (d_features + 1) : n_ambient),
    distance_type_(dist_type),
    huber_delta_(delta)
{
    // Constructor body (if needed)
}

void HuberRegression::fit(const Eigen::MatrixXd &D) {
  const int num_samples = D.rows();
  if (num_samples == 0) {
    // Or throw an error, or set a default model state
    std::cerr << "HuberRegression::fit: Input data matrix D is empty." << std::endl;
    // Initialize to a default flat if possible, or ensure model is marked as not fitted.
    // For now, just return. The base FlatModel's members (A, b_vec, etc.) will remain uninitialized.
    return;
  }
  const int total_cols = D.cols();
  ceres::Problem problem;

  if (distance_type_ == DistanceType::Regression) {
    // --- Vertical Huber Regression ---
    // D has columns [x_1, ..., x_d, y]. So total_cols = d_features + 1.
    // d_features_ is FlatModel::d
    // n_ambient_ for vertical is FlatModel::n (which is d_features_ + 1)
    const int num_features = this->d; // From FlatModel, intrinsic dim of flat = num actual features

    if (total_cols != num_features + 1) {
        throw std::runtime_error("HuberRegression::fit (Vertical): Data matrix D has incorrect number of columns. Expected "
                                 + std::to_string(num_features + 1) + ", got " + std::to_string(total_cols));
    }

    // Parameters: [w_1, ..., w_d, b] (num_features + 1 parameters)
    // Initial guess using OLS: (X_aug^T X_aug)^-1 X_aug^T y
    Eigen::MatrixXd X_aug(num_samples, num_features + 1);
    X_aug.leftCols(num_features) = D.leftCols(num_features); // Features x
    X_aug.col(num_features).setOnes();                       // Augment with 1s for bias
    Eigen::VectorXd y_vec = D.col(num_features);             // Target y

    // Solve OLS for initial guess
    Eigen::VectorXd initial_params_eigen = X_aug.colPivHouseholderQr().solve(y_vec);
    std::vector<double> params_vec(initial_params_eigen.data(),
                                   initial_params_eigen.data() + initial_params_eigen.size());

    // Add residual blocks to Ceres problem
    for (int i = 0; i < num_samples; ++i) {
      Eigen::VectorXd xi = D.row(i).head(num_features).transpose(); // Features for sample i
      double yi = D(i, num_features);                               // Target for sample i

      // Create cost function for this sample
      auto* cost_functor = new HuberVerticalFunctor(xi, yi);
      auto* cost_function =
          new ceres::DynamicAutoDiffCostFunction<HuberVerticalFunctor>(cost_functor);
      cost_function->AddParameterBlock(num_features + 1); // w and b
      cost_function->SetNumResiduals(1);

      // Use Huber loss
      auto* loss_function = new ceres::HuberLoss(huber_delta_);
      problem.AddResidualBlock(cost_function, loss_function, params_vec.data());
    }

    // Configure and run the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false; // Set to true for debug
    // options.max_num_iterations = 100; // Example
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // if (options.minimizer_progress_to_stdout) {
    //    std::cout << summary.BriefReport() << std::endl;
    // }

    // Extract optimized parameters w and b
    // W is d x 1 (for y = W^T x + B, W is d x 1, B is 1x1)
    // FlatModel explicit form: y_vec = X * W_matrix + B_vector (W_matrix is d_features x 1, B_vector is 1 x 1)
    Eigen::MatrixXd W_fit(num_features, 1);
    for (int j = 0; j < num_features; ++j) {
      W_fit(j, 0) = params_vec[j];
    }
    Eigen::VectorXd B_fit(1);
    B_fit(0) = params_vec[num_features];

    // Store in FlatModel (explicit representation: y = Wx + B)
    // Note: FlatModel's W is d x (n-d). Here d=num_features, n=num_features+1. So n-d=1. W is num_features x 1.
    // FlatModel's B is (n-d) x 1. Here B is 1x1.
    override_explicit(W_fit, B_fit);

  } else {
    // --- Orthogonal Huber Regression ---
    // D has columns [p_1, ..., p_n_ambient]. total_cols = n_ambient.
    // n_ambient is FlatModel::n (ambient dimension)
    // The hyperplane has dimension n_ambient - 1, which is FlatModel::d
    const int ambient_dim = this->n; // From FlatModel, ambient dimension

    if (total_cols != ambient_dim) {
        throw std::runtime_error("HuberRegression::fit (Orthogonal): Data matrix D has incorrect number of columns. Expected "
                                 + std::to_string(ambient_dim) + ", got " + std::to_string(total_cols));
    }


    // Parameters: normal vector n (ambient_dim doubles) and offset c (1 double)
    // Total ambient_dim + 1 parameters. Stored contiguously: [n_1,...,n_N, c]
    std::vector<double> params_vec(ambient_dim + 1);

    // Initial guess for n and c using PCA
    Eigen::VectorXd mu = D.colwise().mean(); // Centroid of points
    Eigen::MatrixXd M_centered = D.rowwise() - mu.transpose(); // Centered data
    Eigen::MatrixXd C = M_centered.transpose() * M_centered;   // Covariance matrix (scaled)

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(C);
    Eigen::VectorXd initial_normal = eigen_solver.eigenvectors().col(0); // Smallest eigenvector for n
    if (initial_normal.norm() < 1e-9) { // Handle case of zero normal (e.g. all points identical)
        // Corrected: Use Eigen::VectorXd::Unit(size, index) for dynamic vectors
        initial_normal = Eigen::VectorXd::Unit(ambient_dim, 0); // Default to first axis normal
    } else {
        initial_normal.normalize();
    }
    
    double initial_c = -initial_normal.dot(mu); // c = -n^T * mu

    // Populate initial parameters vector [n_1, ..., n_N, c]
    for (int j = 0; j < ambient_dim; ++j) {
      params_vec[j] = initial_normal(j);
    }
    params_vec[ambient_dim] = initial_c;

    // Add residual blocks for each data point
    for (int i = 0; i < num_samples; ++i) {
      Eigen::VectorXd pi = D.row(i).transpose(); // Data point p_i

      auto* cost_functor = new HuberOrthogonalFunctor(pi);
      auto* cost_function =
          new ceres::DynamicAutoDiffCostFunction<HuberOrthogonalFunctor>(cost_functor);
      // Parameter blocks for this cost function:
      cost_function->AddParameterBlock(ambient_dim); // Normal vector n
      cost_function->AddParameterBlock(1);           // Offset c
      cost_function->SetNumResiduals(1);

      auto* loss_function = new ceres::HuberLoss(huber_delta_);
      problem.AddResidualBlock(cost_function,
                               loss_function,
                               params_vec.data(),             // Pointer to start of n
                               params_vec.data() + ambient_dim); // Pointer to c
    }

    // Add constraint residual block: ||n||^2 - 1 = 0
    auto* norm_constraint_functor = new UnitNormConstraintFunctor(ambient_dim);
    auto* norm_cost_function =
        new ceres::DynamicAutoDiffCostFunction<UnitNormConstraintFunctor>(norm_constraint_functor);
    norm_cost_function->AddParameterBlock(ambient_dim); // The normal vector n
    norm_cost_function->SetNumResiduals(1);
    // Add this constraint block to the problem. No loss function means squared error,
    // so Ceres will try to minimize (||n||^2 - 1)^2.
    // This residual is not scaled by HuberLoss.
    // The parameters argument to AddResidualBlock for a DynamicAutoDiffCostFunction should be a pointer to an array of pointers.
    // Since norm_cost_function only depends on params_vec.data() (the normal part), we pass it like this:
    std::vector<double*> norm_param_blocks;
    norm_param_blocks.push_back(params_vec.data());
    // Corrected: Pass the std::vector<double*> directly to use the appropriate overload.
    problem.AddResidualBlock(norm_cost_function, nullptr, norm_param_blocks);


    // Configure and run the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false; // Set to true for debug
    // options.max_num_iterations = 200; // Example
    // options.function_tolerance = 1e-8; // Example
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // if (options.minimizer_progress_to_stdout) {
    //    std::cout << summary.BriefReport() << std::endl;
    // }

    // Extract optimized normal n and offset c
    Eigen::Map<Eigen::VectorXd> n_opt_map(params_vec.data(), ambient_dim);
    // Explicitly re-normalize n, as the constraint is numerical.
    if (n_opt_map.norm() > 1e-9) { // Avoid normalizing a zero vector
        n_opt_map.normalize();
    } else {
        // Handle degenerate case, e.g., set to a default normal
        std::cerr << "Warning: Optimized normal vector has near-zero norm. Setting to default." << std::endl;
        // Corrected: Use Eigen::VectorXd::Unit(size, index) for dynamic vectors
        n_opt_map = Eigen::VectorXd::Unit(ambient_dim, 0);
    }


    Eigen::RowVectorXd N_fit(ambient_dim); // FlatModel implicit N is 1 x ambient_dim (for single hyperplane)
    for (int j = 0; j < ambient_dim; ++j) {
      N_fit(j) = n_opt_map(j);
    }
    Eigen::VectorXd c_fit(1);
    c_fit(0) = params_vec[ambient_dim];

    // Store in FlatModel (implicit representation: Np + c = 0)
    // FlatModel's N is (n-d) x n. Here n=ambient_dim, d=ambient_dim-1. So n-d=1. N is 1 x ambient_dim.
    // FlatModel's c is (n-d) x 1. Here c is 1x1.
    override_implicit(N_fit, c_fit);
  }
}

std::unique_ptr<Model> HuberRegression::clone() const {
  return std::make_unique<HuberRegression>(*this);
}