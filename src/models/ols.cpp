#include "models/ols.hpp"

#include <Eigen/Core>
#include <Eigen/Cholesky>

std::unique_ptr<Model> OLS::clone() const
{
    return std::make_unique<OLS>(*this);
}

void OLS::fit(const Eigen::MatrixXd &D)
{
    int N = D.rows();

    if (N <= d)
    {
        throw std::runtime_error("OLS::fit(D): You must provide at least d+1 points.");
    }

    if (D.cols() != n)
    {
        throw std::runtime_error("OLS::fit(D): Dimension mismatch between D and n.");
    }

    Eigen::MatrixXd X = D.leftCols(d);
    Eigen::MatrixXd Y = D.rightCols(n - d);

    Eigen::MatrixXd X_prime = Eigen::MatrixXd(N, d + 1); // Augmented column for b.value()
    X_prime.leftCols(d) = X;
    X_prime.col(d).setConstant(1.0); // last column contains just 1s for b.value()

    Eigen::MatrixXd W_prime = (X_prime.transpose() * X_prime).ldlt().solve(X_prime.transpose() * Y);

    // Extract A and b
    Eigen::MatrixXd W = W_prime.topRows(d);
    Eigen::VectorXd b_subspace = W.bottomRows(1).transpose();
    //// TODO: Store the explicit representation as W \in R^{d x (n-d)} and b_subspace \in R^{n-d}

    // Store the parametric representation
    A = Eigen::MatrixXd(n, d);
    A->topLeftCorner(d, d) = Eigen::MatrixXd::Identity(d, d);
    A->bottomRows(n - d) = W.transpose();

    b_vec = Eigen::VectorXd(n);
    b_vec->head(d).setZero();
    b_vec->tail(n - d) = b_subspace;
}
