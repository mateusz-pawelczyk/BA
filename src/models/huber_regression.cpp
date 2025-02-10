#include "models/huber_regression.hpp"

HuberRegression::HuberRegression(int d, int n) : FlatModel(d, n) {}

void HuberRegression::fit(const Eigen::MatrixXd &D)
{
    const int m = D.rows();
    const int total_cols = D.cols();
    const int num_features = total_cols - 1; // Last column is the target.

    // Parameter vector: first num_features for weights, then bias.
    std::vector<double> parameters(num_features + 1, 0.0);

    ceres::Problem problem;
    for (int i = 0; i < m; ++i)
    {
        // Extract features and target.
        Eigen::VectorXd x = D.row(i).head(num_features).transpose();
        double y = D(i, total_cols - 1);

        // Create a cost functor.
        auto *functor = new HuberCostFunctor(x, y);

        // Use a dynamic autodiff cost function.
        ceres::DynamicAutoDiffCostFunction<HuberCostFunctor> *cost_function =
            new ceres::DynamicAutoDiffCostFunction<HuberCostFunctor>(functor);
        cost_function->SetNumResiduals(1);
        cost_function->AddParameterBlock(num_features + 1);

        // Use a Huber loss (delta = 1.0) to down–weight outliers.
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Add the residual block.
        problem.AddResidualBlock(cost_function, loss_function, parameters.data());
    }

    // Solver options.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    // Solve the optimization problem.
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Extract the optimized parameters.
    Eigen::VectorXd w_new(num_features);
    for (int j = 0; j < num_features; ++j)
    {
        w_new(j) = parameters[j];
    }
    double b_new = parameters[num_features];

    // Update the model’s explicit representation.
    override_explicit(w_new, b_new);
}

// Clone the model.
std::unique_ptr<Model> HuberRegression::clone() const
{
    return std::make_unique<HuberRegression>(*this);
}
