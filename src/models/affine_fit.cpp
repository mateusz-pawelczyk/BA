#include "models/affine_fit.hpp"

#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>

std::unique_ptr<Model> AffineFit::clone() const
{
    return std::make_unique<AffineFit>(*this);
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
}
