#include "models/affine_fit.hpp"

#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>

#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>


std::unique_ptr<Model> AffineFit::clone() const  {
    return std::make_unique<AffineFit>(*this);
} 


void AffineFit::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y) {
    int point_count = X.rows();

    if (point_count > d.value() + 1) {
        point_count = d.value() + 1;
        //std::cout << "X cannot have more than d+1 points, where d is the dimension of the flat. First d+1 points are taken." << std::endl;
    }
    
    Eigen::MatrixXd X_prime(point_count, d.value() + 1);
    X_prime.leftCols(d.value()) = X.topRows(point_count);
    X_prime.col(d.value()).setOnes();
    Eigen::VectorXd Y_prime = Y.head(point_count);
    Eigen::VectorXd wb = X_prime.colPivHouseholderQr().solve(Y_prime);

    w = wb.head(d.value());
    b = wb(d.value());

}

void AffineFit::fit(const Eigen::MatrixXd& D) {
    //// TODO: Store the explicit representation as W \in R^{d x (n-d)} and b_subspace \in R^{n-d}
    if (D.rows() <= d.value()) {
        throw std::runtime_error("AffineFit::fit(D): You must provide d+1 points.");
    }


    int N = d.value() + 1;
    int n = D.cols();

    Eigen::MatrixXd X = D.topRows(d.value() + 1).leftCols(d.value());
    Eigen::MatrixXd Y = D.topRows(d.value() + 1).rightCols(n - d.value());

    Eigen::MatrixXd X_prime(N, d.value() + 1);
    X_prime.leftCols(d.value()) = X;
    X_prime.col(d.value()).setOnes();

    Eigen::MatrixXd W_prime = X_prime.colPivHouseholderQr().solve(Y);

    // Extract A and b
    Eigen::MatrixXd W = W_prime.topRows(d.value());
    Eigen::VectorXd b_subspace = W_prime.bottomRows(1).transpose();

    // Store the parametric representation
    A = Eigen::MatrixXd(n, d.value());
    A->topLeftCorner(d.value(), d.value()) = Eigen::MatrixXd::Identity(d.value(), d.value());
    A->bottomRows(n - d.value()) = W.transpose();

    b_vec = Eigen::VectorXd(n);
    b_vec->head(d.value()).setZero();
    b_vec->tail(n - d.value()) = b_subspace;
}


double AffineFit::predict(const Eigen::VectorXd& x) const {
    return w->dot(x) + (*b);
}

Eigen::VectorXd AffineFit::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd one = Eigen::VectorXd::Ones(X.rows());
    return X * (*w) + (*b) * one;
}

Eigen::VectorXd AffineFit::get_w() const {
    return *w;
}

double AffineFit::get_b() const {
    return *b;
}

void AffineFit::visualize(const std::string& name, double sideLen, double lineRadius, float flatAlpha) {
    if (d.value() < 1 || d.value() > 2) return;
    constexpr int numPoints = 2;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
        double halfSideLen = sideLen / 2.0;

    if (d.value() == 1) {
        get_explicit_repr();
        Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(numPoints, -halfSideLen, halfSideLen);
        vertices = Eigen::MatrixXd::Zero(numPoints, 3);
        vertices.col(0) = x;
        vertices.col(1) = x * (*w)(0) + Eigen::VectorXd::Ones(numPoints) * (*b);

        faces.resize(numPoints - 1, 2);
        faces.col(0) = Eigen::VectorXi::LinSpaced(numPoints - 1, 0, numPoints - 2);
        faces.col(1) = faces.col(0).array() + 1;

        polyscope::registerCurveNetwork(name, vertices, faces)->setRadius(lineRadius);
    } else {
        orthonormalize();
        get_parametric_repr();
        
        Eigen::Matrix<double, 4, 2> xy;
        xy << -halfSideLen, -halfSideLen,
              halfSideLen, -halfSideLen,
              -halfSideLen, halfSideLen,
              halfSideLen, halfSideLen;

        vertices = Eigen::MatrixXd::Zero(4, 3);
        vertices = xy * A->transpose() + b_vec->transpose().replicate(4, 1);

        faces.resize(1, 4);
        faces << 0, 1, 3, 2;

        polyscope::registerSurfaceMesh(name, vertices, faces)->setTransparency(flatAlpha);
    }
}