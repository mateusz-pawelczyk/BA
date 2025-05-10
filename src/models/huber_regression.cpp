#include "models/huber_regression.hpp"

HuberRegression::HuberRegression(int d, int n, DistanceType dist) : FlatModel(d, n), distance_(dist) {}

void HuberRegression::fit(const Eigen::MatrixXd &D) {
    const int m = D.rows();
    const int total_cols = D.cols();
    
    std::vector<double> parameters;
    if (distance_ == DistanceType::Regression) {
      // num_features + 1: [w₀…w_{d−1}, b]
      int num_features = total_cols - 1;
      parameters.assign(num_features + 1, 0.0);
    } else {
      // ambient_dim + 1: [n₀…n_{n−1}, c]
      parameters.assign(total_cols + 1, 0.0);
    }
  
    ceres::Problem problem;
    for (int i = 0; i < m; ++i) {
      Eigen::VectorXd x = D.row(i).head(total_cols - (distance_ == DistanceType::Regression ? 1 : 0)).transpose();
  
      if (distance_ == DistanceType::Regression) {
        double y = D(i, total_cols - 1);
        auto* f = new HuberVerticalFunctor(x, y);
        auto* cost_function = new ceres::DynamicAutoDiffCostFunction<HuberVerticalFunctor>(f);
        cost_function->SetNumResiduals(1);
        cost_function->AddParameterBlock(x.size() + 1);

        ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(cost_function, loss, parameters.data());
      } else {
        auto* f = new HuberOrthogonalFunctor(D.row(i).transpose());
        auto* cost_function = new ceres::DynamicAutoDiffCostFunction<HuberOrthogonalFunctor>(f);
        cost_function->SetNumResiduals(1);
        cost_function->AddParameterBlock(D.cols() + 1);

        ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(cost_function, loss, parameters.data());
      }
  
      
    }

    // Solver options.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    // Solve the optimization problem.
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Extract and store
  if (distance_ == DistanceType::Regression) {
    int d = D.cols() - 1;
    Eigen::VectorXd w(d);
    for (int j = 0; j < d; ++j) w[j] = parameters[j];
    Eigen::VectorXd b(1); b[0] = parameters[d];
    override_explicit(w, b);
  }  else if (distance_ == DistanceType::Orthogonal) {
    const int ambient_dim = D.cols();
    parameters.assign(ambient_dim + 1, 0.0);
    parameters[0] = 1.0;  // set at least one component of n nonzero
        
    ceres::Problem problem;
    for (int i = 0; i < m; ++i) {
        Eigen::VectorXd x = D.row(i).transpose();
        auto* f = new HuberOrthogonalFunctor(x);
        auto* cost_function = new ceres::DynamicAutoDiffCostFunction<HuberOrthogonalFunctor>(f);
        cost_function->SetNumResiduals(1);
        cost_function->AddParameterBlock(ambient_dim + 1);

        ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(cost_function, loss, parameters.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::RowVectorXd normal(ambient_dim);
    for (int j = 0; j < ambient_dim; ++j)
        normal(j) = parameters[j];
    double c = parameters[ambient_dim];

    this->override_implicit(normal, Eigen::VectorXd::Constant(1, c));
}
}

// Clone the model.
std::unique_ptr<Model> HuberRegression::clone() const
{
    return std::make_unique<HuberRegression>(*this);
}
