#include "core/flat_model.hpp"

#include <stdexcept>
#include <Eigen/Dense>
#include <iostream>
#include <optional>

#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>

// Attention: X and Y will merge into a single data matrix `D` and the model will be fit to `D`!
void FlatModel::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y)
{
    if (n != d + 1)
    {
        throw std::runtime_error("FlatModel::fit(X, Y): This method is only valid for hyperplanes (n = d + 1).");
    }
    if (X.cols() != d)
    {
        throw std::runtime_error("FlatModel::fit(X, Y): Dimension mismatch between X.cols() = " + std::to_string(X.cols()) + " and d = " + std::to_string(d));
    }
    if (Y.size() != X.rows())
    {
        throw std::runtime_error("FlatModel::fit(X, Y): Dimension mismatch between Y.size() = " + std::to_string(Y.size()) + " and X.rows() = " + std::to_string(X.rows()));
    }

    Eigen::MatrixXd D(X.rows(), X.cols() + 1);
    D << X, Y;
    fit(D);
};

Eigen::VectorXd FlatModel::predict(const Eigen::VectorXd &x)
{
    if (x.size() != d)
    {
        throw std::runtime_error("AffineFit::predict(x): Dimension mismatch between x (" + std::to_string(x.size()) + ") and d (" + std::to_string(d) + ")");
    }
    get_explicit_repr();
    auto [W_ref, B_ref] = get_explicit_repr();

    return W_ref.transpose() * x + B_ref;
}

Eigen::MatrixXd FlatModel::predict(const Eigen::MatrixXd &X)
{
    if (X.cols() != d)
    {
        throw std::runtime_error("AffineFit::predict(X): Dimension mismatch between X (" + std::to_string(X.cols()) + ") and d (" + std::to_string(d) + ")");
    }
    auto [W_ref, B_ref] = get_explicit_repr();
    int N = X.rows();
    Eigen::MatrixXd B_expanded = B_ref.replicate(N, 1);

    return X * W_ref + B_ref.transpose().replicate(N, 1);
}

void FlatModel::visualize(const std::string &name, double sideLen, double lineRadius, float flatAlpha)
{
    if (d < 0 || d > 2 || n <= 0 || n > 3)
        return;
    constexpr int numPoints = 2;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;

    double halfSideLen = sideLen / 2.0;
    if (d == 0)
    {
        get_parametric_repr();
        Eigen::MatrixXd points = Eigen::MatrixXd(1, 3);
        points << b_vec->transpose();

        std::vector<glm::vec3> glmPoints;
        for (const auto &v : points.rowwise())
        {
            glmPoints.push_back({v.coeff(0), v.coeff(1), v.size() >= 3 ? v.coeff(2) : 0.0});
        }

        polyscope::PointRenderMode renderMode = polyscope::PointRenderMode::Sphere;

        polyscope::registerPointCloud(name, glmPoints)->setPointRenderMode(renderMode)->setPointRadius(lineRadius);
    }
    else if (d == 1)
    {
        orthonormalize();
        get_parametric_repr();

        Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(numPoints, -halfSideLen, halfSideLen);
        vertices = Eigen::MatrixXd::Zero(numPoints, n);

        vertices = x * A->transpose() + b_vec->transpose().replicate(numPoints, 1);

        faces.resize(numPoints - 1, 2);
        faces.col(0) = Eigen::VectorXi::LinSpaced(numPoints - 1, 0, numPoints - 2);
        faces.col(1) = faces.col(0).array() + 1;

        Eigen::MatrixXd vertices3D = Eigen::MatrixXd::Zero(numPoints, 3);
        vertices3D.block(0, 0, numPoints, n) = vertices;

        polyscope::registerCurveNetwork(name, vertices3D, faces)->setRadius(lineRadius);
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

// ============================================================================
// SVD-based helper for computing the orthogonal complement of the column-space.
// ============================================================================
Eigen::MatrixXd orthogonalComplement(const Eigen::MatrixXd &U)
{
    // Perform the Full Singular Value Decomposition (SVD) of U^T
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(U.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Extract the matrix V from the SVD
    Eigen::MatrixXd V = svd.matrixV();

    // Identify the orthogonal complement
    // Columns of V corresponding to zero singular values span the orthogonal complement
    int rank = svd.rank();               // Rank of the input matrix
    return V.rightCols(V.cols() - rank); // Orthogonal complement
}

// ============================================================================
// Parametric -> Implicit
//    x = A*y + b_vec   ==>   N*x + c = 0
// ============================================================================
void FlatModel::parametric_to_implicit()
{
    // Basic checks
    if (!A.has_value() || !b_vec.has_value())
    {
        throw std::runtime_error("parametric_to_implicit: A and b_vec must be set!");
    }
    if (A->rows() != b_vec->rows())
    {
        throw std::runtime_error("parametric_to_implicit: Dimension mismatch between A and b_vec!");
    }

    // Optionally check A for full column rank (if you want a robust parametric representation)
    Eigen::FullPivLU<Eigen::MatrixXd> lu(*A);
    if (lu.rank() < A->cols())
    {
        std::cout << "A:\n"
                  << *A << std::endl;
        throw std::invalid_argument("parametric_to_implicit: Matrix A does not have full column rank.");
    }

    // 1) N = orthogonal complement of A
    //    (the rows of N span the left-orthogonal subspace of A)
    N = orthogonalComplement(*A).transpose();

    // 2) c = -N * b_vec <==> N*x + c = 0
    c = -(*N) * (*b_vec);

    if (orthonormalized)
    {
        orthonormalize_implicit();
    }
}

// ============================================================================
// Implicit -> Parametric
//    N*x + c = 0   ==>   x = A*y + b_vec
// ============================================================================
void FlatModel::implicit_to_parametric()
{
    // Basic checks
    if (!N.has_value() || !c.has_value())
    {
        throw std::runtime_error("implicit_to_parametric: N and c must be set!");
    }
    // 1) A = Null Space of N
    //    i.e. columns of A span all solutions to N*x = 0
    A = orthogonalComplement(N->transpose());

    // 2) Find particular solution b_vec to N*b_vec = -c  (or N*b_vec + c = 0)
    //    We'll solve:  N*b_vec = -c  =>  b_vec = ?
    //    but currently the user had N*x + c = 0 => N*x = -c.
    //    So we want b_vec s.t. N*b_vec = -c
    //    We can do b_vec = solve(N, -c).
    Eigen::VectorXd rhs = -(*c);

    // We can attempt direct solution if rank is okay:
    Eigen::FullPivLU<Eigen::MatrixXd> lu(*N);
    if (lu.rank() == N->rows())
    {
        // This system is consistent
        b_vec = lu.solve(rhs);
    }
    else
    {
        // Use least-squares solution
        b_vec = N->jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
    }

    if (orthonormalized)
    {
        orthonormalize_parametric();
    }
}

void FlatModel::parametric_to_explicit()
{
    if (!A.has_value() || !b_vec.has_value())
    {
        throw std::runtime_error("parametric_to_explicit: A and b_vec must be set!");
    }
    const Eigen::MatrixXd &Aref = *A;
    const Eigen::VectorXd &bref = *b_vec;

    const int n = Aref.rows();
    const int d_local = Aref.cols();

    Eigen::MatrixXd A_top = Aref.topRows(d);        // d x d
    Eigen::MatrixXd A_bot = Aref.bottomRows(n - d); // (n - d) x d

    Eigen::MatrixXd W_local; // will hold the solution (d x (n-d))
    bool solved = false;

    // Attempt a QR decomposition first (fast and robust if A_top is full rank).
    {
        // Note: we work with A_top.transpose() since our system is
        //   A_top^T * W = A_bot^T.
        auto qr = A_top.transpose().colPivHouseholderQr();
        if (qr.rank() == A_top.cols()) // full rank check
        {
            // Solve for W, note that the right-hand side is A_bot^T.
            // Then the solution W satisfies A_top^T * W = A_bot^T.
            W_local = qr.solve(A_bot.transpose());
            solved = true;
        }
    }

    // Fallback: if QR did not work (e.g. A_top is nearly singular),
    // use a FullPivLU decomposition.
    if (!solved)
    {
        auto lu = A_top.transpose().fullPivLu();
        if (!lu.isInvertible())
            throw std::runtime_error("A_top^T is singular, cannot solve for W.");
        W_local = lu.solve(A_bot.transpose());
    }

    Eigen::VectorXd B_top = bref.head(d);
    Eigen::VectorXd B_bot = bref.tail(n - d);
    Eigen::VectorXd B_local = B_bot - W_local.transpose() * B_top;

    W = W_local;
    B = B_local;

    if (orthonormalized)
    {
        orthonormalize_explicit();
    }
}

void FlatModel::explicit_to_parametric()
{

    const Eigen::MatrixXd &W_ref = *W;
    const Eigen::VectorXd &B_ref = *B;

    int m = W_ref.cols(); // Number of dependent variables (m = n - d)
    if (B_ref.size() != m || n != d + m)
    {
        throw std::runtime_error("explicit_to_parametric: B has incorrect dimension.");
    }

    // Construct A matrix: [I; W^T]
    Eigen::MatrixXd A_new = Eigen::MatrixXd::Zero(n, d);
    A_new.topLeftCorner(d, d).setIdentity();
    A_new.bottomRows(m) = W_ref.transpose();

    // Construct B vector: [0; B]
    Eigen::VectorXd b_new(n);
    b_new.head(d).setZero();
    b_new.tail(m) = B_ref;

    A = A_new;
    b_vec = b_new;

    if (orthonormalized)
    {
        orthonormalize_parametric();
    }
}

void FlatModel::implicit_to_explicit()
{
    if (!N.has_value() || !c.has_value())
    {
        throw std::runtime_error("implicit_to_explicit: N and c must be set!");
    }
    // n: ambient dimension, d: free dimension, so m = n-d.
    const int m = n - d;
    if (N->cols() != n)
    {
        throw std::runtime_error("implicit_to_explicit: N->cols() must equal n (the ambient dimension).");
    }
    if (N->rows() != m)
    {
        throw std::runtime_error("implicit_to_explicit: N->rows() must equal n-d.");
    }

    // Partition N into N_left (first d columns) and N_right (last m columns).
    const Eigen::MatrixXd &Nmat = *N;
    Eigen::MatrixXd N_left = Nmat.block(0, 0, m, d);
    Eigen::MatrixXd N_right = Nmat.block(0, d, m, m);

    // Check that N_right is invertible.
    Eigen::FullPivLU<Eigen::MatrixXd> luN_right(N_right);
    if (luN_right.rank() < m)
    {
        throw std::runtime_error("implicit_to_explicit: The dependent block of N is singular.");
    }

    // Compute W^T = -N_right^{-1} * N_left, then transpose.
    Eigen::MatrixXd W_transpose = -luN_right.solve(N_left);
    Eigen::MatrixXd W_local = W_transpose.transpose(); // Now W_local is d x (n-d)

    // Compute B = -N_right^{-1} * c.
    Eigen::VectorXd B_local = -luN_right.solve(*c);

    // Save the explicit representation.
    W = W_local;
    B = B_local;

    if (orthonormalized)
    {
        orthonormalize_explicit();
    }
}

void FlatModel::explicit_to_implicit()
{
    if (!W.has_value() || !B.has_value())
    {
        throw std::runtime_error("explicit_to_implicit: W and B must be set!");
    }
    const Eigen::MatrixXd &Wmat = *W; // Expected size: d x m, where m = n-d.
    const Eigen::VectorXd &Bvec = *B; // Expected size: m.
    const int m = Wmat.cols();
    if (Bvec.size() != m || n != d + m)
    {
        throw std::runtime_error("explicit_to_implicit: Dimension mismatch between W, B, and ambient dimension.");
    }

    // Build N = [ -W^T, I ] which has size (n-d) x n.
    Eigen::MatrixXd N_local(m, n);
    N_local.block(0, 0, m, d) = -Wmat.transpose();               // -W^T (m x d)
    N_local.block(0, d, m, m) = Eigen::MatrixXd::Identity(m, m); // I (m x m)

    // Compute c = -B (since B in the explicit form is [0;B]).
    Eigen::VectorXd c_local = -Bvec;

    // Save the implicit representation.
    N = N_local;
    c = c_local;

    if (orthonormalized)
    {
        orthonormalize_implicit();
    }
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> FlatModel::get_implicit_repr()
{
    if (N.has_value() && c.has_value())
    {
    }
    else if (A.has_value() && b_vec.has_value())
    {
        parametric_to_implicit();
    }
    else if (W.has_value() && B.has_value())
    {
        explicit_to_implicit();
    }
    else
    {
        throw std::runtime_error("Model wasn't fitted yet. Cannot get implicit representation.");
    }

    return std::pair<Eigen::MatrixXd, Eigen::VectorXd>{*N, *c};
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> FlatModel::get_explicit_repr()
{
    if (W.has_value() && B.has_value())
    {
    }
    else if (A.has_value() && b_vec.has_value())
    {
        parametric_to_explicit();
    }
    else if (N.has_value() && c.has_value())
    {
        implicit_to_explicit();
    }
    else
    {
        throw std::runtime_error("Model wasn't fitted yet. Cannot get explicit representation.");
    }

    return std::pair<Eigen::MatrixXd, Eigen::VectorXd>{*W, *B};
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> FlatModel::get_parametric_repr()
{
    if (A.has_value() && b_vec.has_value())
    {
    }
    else if (N.has_value() && c.has_value())
    {
        implicit_to_parametric();
    }
    else if (W.has_value() && B.has_value())
    {
        explicit_to_parametric();
    }
    else
    {
        throw std::runtime_error("Model wasn't fitted yet. Cannot get parametric representation.");
    }

    return std::pair<Eigen::MatrixXd, Eigen::VectorXd>{A.value(), b_vec.value()};
}

int FlatModel::get_dimension()
{
    return d;
}

int FlatModel::get_ambient_dimension()
{
    return n;
}

void FlatModel::reset()
{
    A.reset();
    b_vec.reset();
    N.reset();
    c.reset();
    W.reset();
    B.reset();
    Q.reset();
    r.reset();
    orthonormalized = false;
}

double FlatModel::R2(const Eigen::MatrixXd &D)
{
    Eigen::VectorXd mean = D.colwise().mean();
    double ss_res = quadratic_loss(D).sum();
    double ss_tot = (D.rowwise() - mean.transpose()).squaredNorm();

    return 1.0 - ss_res / ss_tot;
}

double FlatModel::R2(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y)
{
    Eigen::MatrixXd Y_hat = predict(X);
    Eigen::MatrixXd residuals = Y - Y_hat;
    Eigen::MatrixXd mean_Y = Y.colwise().mean().replicate(Y.rows(), 1);
    Eigen::MatrixXd total_variance = Y - mean_Y;

    double SS_res = (residuals.array().square().sum());
    double SS_tot = (total_variance.array().square().sum());

    return 1.0 - (SS_res / SS_tot);
}

double FlatModel::MSE(const Eigen::MatrixXd &D)
{
    return quadratic_loss(D).mean();
}

double FlatModel::MSE(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y)
{
    Eigen::MatrixXd Y_hat = predict(X);
    Eigen::MatrixXd residuals = Y - Y_hat;

    double mse = (residuals.array().square().sum()) / Y.rows();
    return mse;
}

void FlatModel::override_parametric(const Eigen::MatrixXd &Anew, const Eigen::VectorXd &bnew)
{
    reset();

    A = Anew;
    b_vec = bnew;

    n = Anew.rows();
    d = Anew.cols();

    parametric_to_explicit();
    parametric_to_implicit();

    if (Q.has_value() || r.has_value())
    {
        compute_QR();
    }
}

void FlatModel::override_implicit(const Eigen::MatrixXd &Nnew, const Eigen::VectorXd &cnew)
{
    reset();

    N = Nnew;
    c = cnew;

    n = Nnew.cols();
    d = n - Nnew.rows();

    implicit_to_explicit();
    implicit_to_parametric();

    if (Q.has_value() || r.has_value())
    {
        compute_QR();
    }
}

void FlatModel::override_explicit(const Eigen::MatrixXd &Wnew, const Eigen::VectorXd &Bnew)
{
    reset();

    W = Wnew;
    B = Bnew;

    d = Wnew.rows();
    n = d + Bnew.size();

    explicit_to_implicit();
    explicit_to_parametric();

    if (Q.has_value() || r.has_value())
    {
        compute_QR();
    }
}

void FlatModel::orthonormalize()
{
    if (A.has_value())
    {
        orthonormalize_parametric();
    }
    if (N.has_value())
    {
        orthonormalize_implicit();
    }
    if (W.has_value())
    {
        orthonormalize_explicit();
    }
    if (!A.has_value() && !N.has_value() && !W.has_value())
    {
        throw std::runtime_error("Model wasn't fitted yet. Cannot orthogonalize.");
    }

    orthonormalized = true;
}

void FlatModel::orthonormalize_parametric()
{
    if (!A.has_value() || !b_vec.has_value())
    {
        throw std::runtime_error("orthogonolize_parametric: A and b_vec must be set!");
    }

    // QR decomposition of A (Q is orthogonal)
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A.value());

    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(A->rows(), A->cols());

    A = Q;

    Eigen::VectorXd b_proj = A.value() * (A.value().transpose() * b_vec.value());
    b_vec = b_vec.value() - b_proj;
}

void FlatModel::orthonormalize_implicit()
{
    // Ensure N and c are set
    if (!N.has_value() || !c.has_value())
    {
        throw std::runtime_error("N and c must be set before orthonormalizing.");
    }
    int rows = n - d;

    // Transpose N
    Eigen::MatrixXd N_t = N->transpose(); // N_t is n x (n-d) => example: 5 x 2

    // Perform the (reduced) QR factorization: N^T = Q * R
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(N_t);

    // Extract Q_hat (n x (n-d)) and R_hat ((n-d) x (n-d))
    Eigen::MatrixXd Q_full = qr.householderQ();
    Eigen::MatrixXd Q_hat = Q_full.leftCols(rows); // Q_hat is n x (n-d)

    // Extract R_hat from the upper triangular part of matrixQR()
    Eigen::MatrixXd R_full = qr.matrixQR().template triangularView<Eigen::Upper>();
    Eigen::MatrixXd R_hat = R_full.topLeftCorner(rows, rows); // (n-d) x (n-d)

    Eigen::MatrixXd M = Q_hat.transpose();

    // Solve R_hat^T * d = c
    Eigen::VectorXd d_vec = R_hat.transpose().colPivHouseholderQr().solve(c.value());

    // Update the internal representation
    N = M;
    c = d_vec;
}

void FlatModel::orthonormalize_explicit()
{
    // Clean up later, this method is not used anymore
}

void FlatModel::compute_QR()
{
    orthonormalize();
    if (A.has_value() && b_vec.has_value())
    {
        Q = Eigen::MatrixXd::Identity(A->rows(), A->rows()) - A.value() * A.value().transpose();
        r = -b_vec.value();
    }
    else if (N.has_value() && c.has_value())
    {
        Q = N.value().transpose() * N.value();
        r = N.value().transpose() * c.value();
    }
    else
    {
        get_parametric_repr();
        compute_QR();
    }
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> FlatModel::get_QR()
{
    if (!Q.has_value() || !r.has_value())
    {
        compute_QR();
    }
    return std::pair<Eigen::MatrixXd, Eigen::VectorXd>{Q.value(), r.value()};
}

double FlatModel::quadratic_loss(const Eigen::VectorXd &point)
{
    if (!Q.has_value() || !r.has_value())
    {
        compute_QR();
    }
    return (point.transpose() * Q.value() * point)(0) + 2 * r->dot(point) + r->dot(r.value());
}

Eigen::VectorXd FlatModel::quadratic_loss(const Eigen::MatrixXd &points)
{
    if (!Q.has_value() || !r.has_value())
    {
        compute_QR();
    }

    int N = points.rows();

    Eigen::VectorXd result(N);

    for (int i = 0; i < N; ++i)
    {
        result(i) = quadratic_loss(static_cast<Eigen::VectorXd>(points.row(i)));
    }

    return result;
}
