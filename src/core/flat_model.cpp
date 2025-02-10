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

double FlatModel::predict(const Eigen::VectorXd &x) const
{
    if (n != d + 1)
    {
        throw std::runtime_error("AffineFit::predict(x): This method is only valid for hyperplanes (n = d + 1).");
    }
    if (x.size() != d)
    {
        throw std::runtime_error("AffineFit::predict(x): Dimension mismatch between x (" + std::to_string(x.size()) + ") and d (" + std::to_string(d) + ")");
    }
    // TODO: If w or b is not set, try to compute them

    return w->dot(x) + (*b);
}

Eigen::VectorXd FlatModel::predict(const Eigen::MatrixXd &X) const
{
    if (n != d + 1)
    {
        throw std::runtime_error("AffineFit::predict(X): This method is only valid for hyperplanes (n = d + 1).");
    }
    if (X.cols() != d)
    {
        throw std::runtime_error("AffineFit::predict(X): Dimension mismatch between X (" + std::to_string(X.cols()) + ") and d (" + std::to_string(d) + ")");
    }
    // TODO: If w or b is not set, try to compute them

    Eigen::VectorXd one = Eigen::VectorXd::Ones(X.rows());
    return X * (*w) + (*b) * one;
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

// ============================================================================
// Parametric -> Explicit  (Hyperplane case: dimension is n-1)
//    x = A*y + b_vec   ==>   x_{n-1} = w^T x_{0..n-2} + b
// ============================================================================
void FlatModel::parametric_to_explicit()
{
    if (!A.has_value() || !b_vec.has_value())
    {
        throw std::runtime_error("parametric_to_explicit: A and b_vec must be set!");
    }
    const Eigen::MatrixXd &Aref = *A;
    const Eigen::VectorXd &bref = *b_vec;

    // Must be a hyperplane => d = n-1
    const int n = Aref.rows();
    const int d_local = Aref.cols();
    if (d_local != n - 1)
    {
        throw std::runtime_error("parametric_to_explicit: Only valid if subspace dimension = n-1 (hyperplane). You have d = " + std::to_string(d_local) + ", n = " + std::to_string(n));
    }

    //// TODO: Explain better what to do
    // We want:
    //    For i = 0..n-2:  x_i = b_vec(i) + A(i,:) * y
    //    For i = n-1:    x_{n-1} = b_vec(n-1) + A(n-1,:) * y
    //
    //  And in "explicit" form we want:
    //    x_{n-1} = w^T x_{0..n-2} + b
    //
    // Matching for all y:
    //    b_vec(n-1) + A(n-1,:) * y  ==  w^T [ b_vec(0..n-2) + A(0..n-2,:) * y ] + b
    //
    // => A(n-1,:) = w^T * A(0..n-2,:)
    // => b_vec(n-1) = w^T * b_vec(0..n-2) + b

    Eigen::MatrixXd A_top = Aref.topRows(n - 1); // (n-1) x (n-1)
    // bottomRow = A(n-1, :), dimension: 1 x (n-1)
    Eigen::RowVectorXd A_bot = Aref.row(n - 1);

    // Solve bottomRow^T = topBlock^T * w
    // => w = (topBlock^T)^{-1} * bottomRow^T, if invertible
    // More robustly, we use a pseudo-inverse or direct solve:
    Eigen::VectorXd wLocal;
    {
        Eigen::FullPivLU<Eigen::MatrixXd> lu2(A_top.transpose());
        if (lu2.isInvertible())
        {
            wLocal = lu2.solve(A_bot.transpose());
        }
        else
        {
            std::cout << "Warning: A^T is not invertible. Using SVD-based pseudo-inverse." << std::endl;
            // fallback to SVD-based pseudo-inverse
            wLocal = A_top.transpose()
                         .jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                         .solve(A_bot.transpose());
        }
    }

    // b = b_vec(n-1) - w^T * b_vec(0..n-2)
    Eigen::VectorXd b_top = bref.head(n - 1); // b_vec(0..n-2)
    double bLocal = bref(n - 1) - wLocal.dot(b_top);

    // Store in the optional fields: w, b
    w = wLocal; // an (n-1)-vector
    this->b = bLocal;

    if (orthonormalized)
    {
        orthonormalize_explicit();
    }
}

// ============================================================================
// Explicit -> Parametric  (Hyperplane case: x_{n-1} = w^T x_{0..n-2} + b)
//    => x = A*y + b_vec
//    We pick y = x_{0..n-2}, so dimension d = n-1
// ============================================================================
void FlatModel::explicit_to_parametric()
{
    if (!w.has_value() || !b.has_value())
    {
        throw std::runtime_error("explicit_to_parametric: w and b must be set!");
    }

    const Eigen::VectorXd &wref = *w; // (n-1)-vector
    double b_scal = *this->b;

    // Construct A as n x (n-1):
    //   The top (n-1) rows = Identity_(n-1)
    //   The last row = w^T
    //
    // Construct b_vec:
    //   The top (n-1) entries = 0
    //   The last entry = b

    int n = static_cast<int>(wref.size()) + 1; // since w is size (n-1)
    Eigen::MatrixXd Anew = Eigen::MatrixXd::Zero(n, n - 1);
    // top block = Identity
    Anew.topLeftCorner(n - 1, n - 1).setIdentity();
    // last row = w^T
    Anew.row(n - 1) = wref.transpose();

    // b_vec:
    Eigen::VectorXd bvecNew(n);
    bvecNew.setZero();
    bvecNew(n - 1) = b_scal;

    // Store in .A and .b_vec
    A = Anew;
    b_vec = bvecNew;

    if (orthonormalized)
    {
        orthonormalize_parametric();
    }
}

// ============================================================================
// Implicit -> Explicit  (Single linear equation => 1 x n => a hyperplane)
//    N*x + c = 0, with N in R^{1 x n}, c in R
//    => x_{n-1} = w^T x_{0..n-2} + b
// ============================================================================
void FlatModel::implicit_to_explicit()
{
    if (!N.has_value() || !c.has_value())
    {
        throw std::runtime_error("implicit_to_explicit: N and c must be set!");
    }
    // Must have exactly 1 row => hyperplane
    if (N->rows() != 1)
    {
        throw std::runtime_error("implicit_to_explicit: Only valid if we have a single linear equation (N is 1 x n).");
    }

    const Eigen::RowVectorXd &Nrow = N->row(0); // dimension n
    double cval = (*c)(0);                      // single scalar c

    int n = Nrow.size();
    // We want x_{n-1} = w^T x_{0..n-2} + b.
    // Original eq: Nrow * x + cval = 0 => sum_{i=0..n-1} (Nrow_i * x_i) + cval = 0.
    // Typically we isolate x_{n-1}. So we require Nrow(n-1) != 0, or we do some check.
    double alpha = Nrow(n - 1);
    if (std::abs(alpha) < 1e-15)
    {
        throw std::runtime_error("implicit_to_explicit: Can't solve for x_{n-1}, coefficient is 0. "
                                 "Try reordering or a different explicit coordinate.");
    }

    // => Nrow(n-1)*x_{n-1} = -cval - sum_{i=0..n-2}(Nrow_i * x_i)
    // => x_{n-1} = -1/alpha * cval - (1/alpha)*sum_{i=0..n-2}( Nrow_i * x_i )
    // => x_{n-1} = b + w^T x_{0..n-2},
    //    where w_i = -(Nrow_i / Nrow(n-1)),  b = -(cval / Nrow(n-1)).

    double bLocal = -cval / alpha;
    Eigen::VectorXd wLocal(n - 1); // for the first (n-1) coords
    for (int i = 0; i < n - 1; ++i)
    {
        wLocal(i) = -(Nrow(i) / alpha);
    }

    w = wLocal;
    this->b = bLocal;

    if (orthonormalized)
    {
        orthonormalize_explicit();
    }
}

// ============================================================================
// Explicit -> Implicit
//    x_{n-1} = w^T x_{0..n-2} + b
//    => x_{n-1} - w^T x_{0..n-2} - b = 0
//    => [ -w^T, 1 ] * x + (-b) = 0
// ============================================================================
void FlatModel::explicit_to_implicit()
{
    if (!w.has_value() || !b.has_value())
    {
        throw std::runtime_error("explicit_to_implicit: w and b must be set!");
    }
    const Eigen::VectorXd &wref = *w;
    double bval = *this->b;

    // Let's define:
    //   N(0, 0..n-2) = -w^T
    //   N(0, n-1)    = 1
    //   c(0)         = -b
    // So we get N * x + c = 0
    int n = static_cast<int>(wref.size()) + 1;
    Eigen::MatrixXd Nnew(1, n);
    Nnew.setZero();

    // Fill: N = [ -w^T, 1 ]
    for (int i = 0; i < n - 1; ++i)
    {
        Nnew(0, i) = -wref(i);
    }
    Nnew(0, n - 1) = 1.0;

    // c = [ -b ]
    Eigen::VectorXd cnew(1);
    cnew(0) = -bval;

    // Store in .N and .c
    N = Nnew;
    c = cnew;

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
    else if (w.has_value() && b.has_value())
    {
        explicit_to_implicit();
    }
    else
    {
        throw std::runtime_error("Model wasn't fitted yet. Cannot get implicit representation.");
    }

    return std::pair<Eigen::MatrixXd, Eigen::VectorXd>{*N, *c};
}

std::pair<Eigen::VectorXd, double> FlatModel::get_explicit_repr()
{
    if (w.has_value() && b.has_value())
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

    return std::pair<Eigen::VectorXd, double>{*w, *b};
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
    else if (w.has_value() && b.has_value())
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
    w.reset();
    b.reset();
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

void FlatModel::override_parametric(const Eigen::MatrixXd &Anew, const Eigen::VectorXd &bnew)
{
    reset();

    A = Anew;
    b_vec = bnew;

    n = Anew.rows();
    d = Anew.cols();

    if (n == d + 1)
    {
        parametric_to_explicit();
    }

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

    if (n == d + 1)
    {
        implicit_to_explicit();
    }
    implicit_to_parametric();

    if (Q.has_value() || r.has_value())
    {
        compute_QR();
    }
}

void FlatModel::override_explicit(const Eigen::VectorXd &wnew, double bnew)
{
    reset();

    w = wnew;
    b = bnew;

    d = wnew.size();
    n = d + 1;

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
    if (w.has_value())
    {
        orthonormalize_explicit();
    }
    if (!A.has_value() && !N.has_value() && !w.has_value())
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

double FlatModel::quadratic_loss(const Eigen::VectorXd point)
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