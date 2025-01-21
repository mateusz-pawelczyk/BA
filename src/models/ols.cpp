#include "models/ols.hpp"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>

std::unique_ptr<Model> OLS::clone() const {
    return std::make_unique<OLS>(*this);
} 

void OLS::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y) {
    if (X.cols() != d) {
        throw std::runtime_error("OLS: Dimension mismatch between X and d. If this model is not supposed to be a hyperplane, please use OLS::fit(D) instead.");
    }

    int N = X.rows();

    Eigen::MatrixXd X_prime = Eigen::MatrixXd(N, d + 1); // Augmented column for b.value()
    X_prime.rightCols(d) = X;
    X_prime.col(0).setConstant(1.0); // first column contains just 1s for b.value()

    Eigen::VectorXd w_prime = (X_prime.transpose() * X_prime).ldlt().solve(X_prime.transpose() * Y);

    b = w_prime.coeff(0);  // Extract bias
    w = w_prime.segment(1, d);  // Extract weights
}

void OLS::fit(const Eigen::MatrixXd& D) {
    int N = D.rows();

    if (N <= d) {
        throw std::runtime_error("OLS::fit(D): You must provide at least d+1 points.");
    }

    if (D.cols() != n) {
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


double OLS::predict(const Eigen::VectorXd& x) const {
    return w.value().dot(x) + b.value();
}

Eigen::VectorXd OLS::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd one = Eigen::VectorXd::Ones(X.rows());
    return X * w.value() + b.value() * one;
}

void OLS::visualize(const std::string& name, double sideLen, double lineRadius, float flatAlpha) {
    if (d < 1 || d > 2) return;

    constexpr int numPoints = 2;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    double offset = 0.1;

    if (d == 1) {
        Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(numPoints, 0.0 - offset, sideLen + offset);
        vertices = Eigen::MatrixXd::Zero(numPoints, 3);
        vertices.col(0) = x;
        vertices.col(1) = x * w.value()(0) + Eigen::VectorXd::Ones(numPoints) * b.value();

        faces.resize(numPoints - 1, 2);
        faces.col(0) = Eigen::VectorXi::LinSpaced(numPoints - 1, 0, numPoints - 2);
        faces.col(1) = faces.col(0).array() + 1;

        polyscope::registerCurveNetwork(name, vertices, faces)->setRadius(lineRadius);
    } else {
        Eigen::Matrix<double, 4, 2> xy;
        xy << 0.0 - offset, 0.0 - offset,
              sideLen + offset, 0.0 - offset,
              0.0 - offset, sideLen + offset,
              sideLen + offset, sideLen + offset;

        vertices = Eigen::MatrixXd::Zero(4, 3);
        vertices.leftCols(2) = xy;
        vertices.col(2) = xy * w.value() + Eigen::Vector4d::Ones() * b.value();

        faces.resize(1, 4);
        faces << 0, 1, 3, 2;

        polyscope::registerSurfaceMesh(name, vertices, faces)->setTransparency(flatAlpha);
    }
}