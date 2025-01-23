#include "models/affine_fit.hpp"

#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>

#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>

std::unique_ptr<Model> AffineFit::clone() const
{
    return std::make_unique<AffineFit>(*this);
}

void AffineFit::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y)
{
    int point_count = X.rows();

    if (point_count > d + 1)
    {
        point_count = d + 1;
        // std::cout << "X cannot have more than d+1 points, where d is the dimension of the flat. First d+1 points are taken." << std::endl;
    }

    Eigen::MatrixXd X_prime(point_count, d + 1);
    X_prime.leftCols(d) = X.topRows(point_count);
    X_prime.col(d).setOnes();
    Eigen::VectorXd Y_prime = Y.head(point_count);
    Eigen::VectorXd wb = X_prime.colPivHouseholderQr().solve(Y_prime);

    w = wb.head(d);
    b = wb(d);
}

void AffineFit::fit(const Eigen::MatrixXd &D)
{
    //// TODO: Store the explicit representation as W \in R^{d x (n-d)} and b_subspace \in R^{n-d}
    if (D.rows() <= d)
    {
        throw std::runtime_error("AffineFit::fit(D): You must provide d+1 points.");
    }

    if (D.cols() != n)
    {
        throw std::runtime_error("AffineFit::fit(D): Dimension mismatch between D and n.");
    }

    int N = d + 1;

    Eigen::VectorXd b_vecTmp = D.row(0).transpose();
    Eigen::MatrixXd ATmp = Eigen::MatrixXd(n, d);

    int offset = 1;
    for (int i = 1; i < N && i < D.rows(); ++i)
    {
        ATmp.col(i - offset) = D.row(i).transpose() - b_vecTmp;
        for (int j = 0; j < i - offset; ++j)
        {
            if (ATmp.col(i - offset).dot(ATmp.col(j)) < 1e-6)
            {
                offset++;
                N++;
                break;
            }
        }
    }

    A = ATmp;
    b_vec = b_vecTmp;
    orthonormalize();
    // std::cout << "A (Manual):\n"
    //           << A.value() << std::endl;
    // std::cout << "b_vec (Manual):\n"
    //           << b_vec.value() << std::endl;
    // reset();
    // N = d + 1;

    // Eigen::MatrixXd X = D.topRows(d + 1).leftCols(d);
    // Eigen::MatrixXd Y = D.topRows(d + 1).rightCols(n - d);

    // Eigen::MatrixXd X_prime(N, d + 1);
    // X_prime.leftCols(d) = X;
    // X_prime.col(d).setOnes();

    // Eigen::MatrixXd W_prime = X_prime.colPivHouseholderQr().solve(Y);

    // // Extract A and b
    // Eigen::MatrixXd W = W_prime.topRows(d);
    // Eigen::VectorXd b_subspace = W_prime.bottomRows(1).transpose();

    // // Store the parametric representation
    // A = Eigen::MatrixXd(n, d);
    // A->topLeftCorner(d, d) = Eigen::MatrixXd::Identity(d, d);
    // A->bottomRows(n - d) = W.transpose();

    // b_vec = Eigen::VectorXd(n);
    // b_vec->head(d).setZero();
    // b_vec->tail(n - d) = b_subspace;

    // std::cout << "A (QR):\n"
    //           << *A << std::endl;
    // std::cout << "b_vec (QR):\n"
    //           << *b_vec << std::endl;
}

double AffineFit::predict(const Eigen::VectorXd &x) const
{
    return w->dot(x) + (*b);
}

Eigen::VectorXd AffineFit::predict(const Eigen::MatrixXd &X) const
{
    Eigen::VectorXd one = Eigen::VectorXd::Ones(X.rows());
    return X * (*w) + (*b) * one;
}

void AffineFit::visualize(const std::string &name, double sideLen, double lineRadius, float flatAlpha)
{
    if (d < 1 || d > 2)
        return;
    constexpr int numPoints = 2;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    double halfSideLen = sideLen / 2.0;

    if (d == 1)
    {
        orthonormalize();
        get_parametric_repr();

        Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(numPoints, -halfSideLen, halfSideLen);
        vertices = Eigen::MatrixXd::Zero(4, 3);

        vertices = x * A->transpose() + b_vec->transpose().replicate(2, 1);

        faces.resize(numPoints - 1, 2);
        faces.col(0) = Eigen::VectorXi::LinSpaced(numPoints - 1, 0, numPoints - 2);
        faces.col(1) = faces.col(0).array() + 1;

        polyscope::registerCurveNetwork(name, vertices, faces)->setRadius(lineRadius);
    }
    else
    {
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