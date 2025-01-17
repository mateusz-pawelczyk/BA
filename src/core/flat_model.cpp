#include "core/flat_model.hpp"

#include <stdexcept>
#include <Eigen/Dense>
#include <iostream>
#include <optional>

// ============================================================================
// SVD-based helper for computing the orthogonal complement of the column-space.
// ============================================================================
Eigen::MatrixXd orthogonalComplement(const Eigen::MatrixXd& U) {
    // Perform the Full Singular Value Decomposition (SVD) of U^T
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(U.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    // Extract the matrix V from the SVD
    Eigen::MatrixXd V = svd.matrixV();
    
    // Identify the orthogonal complement
    // Columns of V corresponding to zero singular values span the orthogonal complement
    int rank = svd.rank(); // Rank of the input matrix
    return V.rightCols(V.cols() - rank); // Orthogonal complement
}


// ============================================================================
// Parametric -> Implicit
//    x = A*y + b_vec   ==>   N*x + c = 0
// ============================================================================
void FlatModel::parametric_to_implicit() {
    // Basic checks
    if (!A.has_value() || !b_vec.has_value()) {
        throw std::runtime_error("parametric_to_implicit: A and b_vec must be set!");
    }
    if (A->rows() != b_vec->rows()) {
        throw std::runtime_error("parametric_to_implicit: Dimension mismatch between A and b_vec!");
    }

    // Optionally check A for full column rank (if you want a robust parametric representation)
    Eigen::FullPivLU<Eigen::MatrixXd> lu(*A);
    if (lu.rank() < A->cols()) {
        throw std::invalid_argument("parametric_to_implicit: Matrix A does not have full column rank.");
    }

    // 1) N = orthogonal complement of A
    //    (the rows of N span the left-orthogonal subspace of A)
    N = orthogonalComplement(*A).transpose();

    // 2) c = -N * b_vec
    c = -(*N) * (*b_vec);
    // Print the normal form

    if (orthonormalized) {
        orthonormalize_implicit();
    }
}

// ============================================================================
// Implicit -> Parametric
//    N*x + c = 0   ==>   x = A*y + b_vec
// ============================================================================
void FlatModel::implicit_to_parametric() {
    // Basic checks
    if (!N.has_value() || !c.has_value()) {
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
    if (lu.rank() == N->rows()) {
        // This system is consistent
        b_vec = lu.solve(rhs);
    } else {
        // Use least-squares solution
        b_vec = N->jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
    }

    if (orthonormalized) {
        orthonormalize_parametric();
    }
}

// ============================================================================
// Parametric -> Explicit  (Hyperplane case: dimension is n-1)
//    x = A*y + b_vec   ==>   x_{n-1} = w^T x_{0..n-2} + b
// ============================================================================
void FlatModel::parametric_to_explicit() {
    if (!A.has_value() || !b_vec.has_value()) {
        throw std::runtime_error("parametric_to_explicit: A and b_vec must be set!");
    }
    const Eigen::MatrixXd& Aref = *A;
    const Eigen::VectorXd& bref = *b_vec;

    // Must be a hyperplane => d = n-1
    const int n = Aref.rows();
    const int d_local = Aref.cols();
    if (d_local != n - 1) {
        throw std::runtime_error("parametric_to_explicit: Only valid if subspace dimension = n-1 (hyperplane).");
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


    Eigen::MatrixXd A_top = Aref.topRows(n - 1);   // (n-1) x (n-1)
    // bottomRow = A(n-1, :), dimension: 1 x (n-1)
    Eigen::RowVectorXd A_bot = Aref.row(n - 1);

    // Solve bottomRow^T = topBlock^T * w
    // => w = (topBlock^T)^{-1} * bottomRow^T, if invertible
    // More robustly, we use a pseudo-inverse or direct solve:
    Eigen::VectorXd wLocal;
    {
        Eigen::FullPivLU<Eigen::MatrixXd> lu2(A_top.transpose());
        if (lu2.isInvertible()) {
            wLocal = lu2.solve(A_bot.transpose());
        } else {
            std::cout << "Warning: A^T is not invertible. Using SVD-based pseudo-inverse." << std::endl;
            // fallback to SVD-based pseudo-inverse
            wLocal = A_top.transpose()
                     .jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                     .solve(A_bot.transpose());
        }
    }

    // b = b_vec(n-1) - w^T * b_vec(0..n-2)
    Eigen::VectorXd b_top = bref.head(n - 1);  // b_vec(0..n-2)
    double bLocal = bref(n - 1) - wLocal.dot(b_top);

    // Store in the optional fields: w, b
    w = wLocal;  // an (n-1)-vector
    this->b = bLocal;

    if (orthonormalized) {
        orthonormalize_explicit();
    }
}

// ============================================================================
// Explicit -> Parametric  (Hyperplane case: x_{n-1} = w^T x_{0..n-2} + b)
//    => x = A*y + b_vec
//    We pick y = x_{0..n-2}, so dimension d = n-1
// ============================================================================
void FlatModel::explicit_to_parametric() {
    if (!w.has_value() || !b.has_value()) {
        throw std::runtime_error("explicit_to_parametric: w and b must be set!");
    }

    const Eigen::VectorXd& wref = *w;  // (n-1)-vector
    double b_scal = *this->b;

    // Construct A as n x (n-1):
    //   The top (n-1) rows = Identity_(n-1)
    //   The last row = w^T
    //
    // Construct b_vec:
    //   The top (n-1) entries = 0
    //   The last entry = b

    int n = static_cast<int>(wref.size()) + 1;  // since w is size (n-1)
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

    if (orthonormalized) {
        orthonormalize_parametric();
    }
}

// ============================================================================
// Implicit -> Explicit  (Single linear equation => 1 x n => a hyperplane)
//    N*x + c = 0, with N in R^{1 x n}, c in R
//    => x_{n-1} = w^T x_{0..n-2} + b
// ============================================================================
void FlatModel::implicit_to_explicit() {
    if (!N.has_value() || !c.has_value()) {
        throw std::runtime_error("implicit_to_explicit: N and c must be set!");
    }
    // Must have exactly 1 row => hyperplane
    if (N->rows() != 1) {
        throw std::runtime_error("implicit_to_explicit: Only valid if we have a single linear equation (N is 1 x n).");
    }

    const Eigen::RowVectorXd& Nrow = N->row(0);  // dimension n
    double cval = (*c)(0);   // single scalar c

    int n = Nrow.size();
    // We want x_{n-1} = w^T x_{0..n-2} + b.
    // Original eq: Nrow * x + cval = 0 => sum_{i=0..n-1} (Nrow_i * x_i) + cval = 0.
    // Typically we isolate x_{n-1}. So we require Nrow(n-1) != 0, or we do some check.
    double alpha = Nrow(n - 1);
    if (std::abs(alpha) < 1e-15) {
        throw std::runtime_error("implicit_to_explicit: Can't solve for x_{n-1}, coefficient is 0. "
                                 "Try reordering or a different explicit coordinate.");
    }

    // => Nrow(n-1)*x_{n-1} = -cval - sum_{i=0..n-2}(Nrow_i * x_i)
    // => x_{n-1} = -1/alpha * cval - (1/alpha)*sum_{i=0..n-2}( Nrow_i * x_i )
    // => x_{n-1} = b + w^T x_{0..n-2}, 
    //    where w_i = -(Nrow_i / Nrow(n-1)),  b = -(cval / Nrow(n-1)).

    double bLocal = -cval / alpha;  
    Eigen::VectorXd wLocal(n - 1);  // for the first (n-1) coords
    for (int i = 0; i < n - 1; ++i) {
        wLocal(i) = -(Nrow(i) / alpha);
    }

    w = wLocal;
    this->b = bLocal;

    if (orthonormalized) {
        orthonormalize_explicit();
    }
}

// ============================================================================
// Explicit -> Implicit
//    x_{n-1} = w^T x_{0..n-2} + b
//    => x_{n-1} - w^T x_{0..n-2} - b = 0
//    => [ -w^T, 1 ] * x + (-b) = 0
// ============================================================================
void FlatModel::explicit_to_implicit() {
    if (!w.has_value() || !b.has_value()) {
        throw std::runtime_error("explicit_to_implicit: w and b must be set!");
    }
    const Eigen::VectorXd& wref = *w;
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
    for (int i = 0; i < n - 1; ++i) {
        Nnew(0, i) = -wref(i);
    }
    Nnew(0, n - 1) = 1.0;

    // c = [ -b ]
    Eigen::VectorXd cnew(1);
    cnew(0) = -bval;

    // Store in .N and .c
    N = Nnew;
    c = cnew;

    if (orthonormalized) {
        orthonormalize_implicit();
    }
}


std::pair<Eigen::MatrixXd, Eigen::VectorXd> FlatModel::get_implicit_repr() {
    if (N.has_value() && c.has_value()) {}
    else if (A.has_value() && b_vec.has_value()) {
        parametric_to_implicit();
    } else if (w.has_value() && b.has_value()) {
        explicit_to_implicit();
    } else {
        throw std::runtime_error("Model wasn't fitted yet. Cannot get implicit representation.");
    }

    return std::pair<Eigen::MatrixXd, Eigen::VectorXd> {*N, *c};
}

std::pair<Eigen::VectorXd, double> FlatModel::get_explicit_repr() {
    if (w.has_value() && b.has_value()) {}
    else if (A.has_value() && b_vec.has_value()) {
        parametric_to_explicit();
    } else if (N.has_value() && c.has_value()) {
        implicit_to_explicit();
    } else {
        throw std::runtime_error("Model wasn't fitted yet. Cannot get explicit representation.");
    }

    return std::pair<Eigen::VectorXd, double> {*w, *b};
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> FlatModel::get_parametric_repr() {
    if (A.has_value() && b_vec.has_value()) {}
    else if (N.has_value() && c.has_value()) {
        implicit_to_parametric();
    } else if (w.has_value() && b.has_value()) {
        explicit_to_parametric();
    } else {
        throw std::runtime_error("Model wasn't fitted yet. Cannot get parametric representation.");
    }

    return std::pair<Eigen::MatrixXd, Eigen::VectorXd> {A.value(), b_vec.value()};
}


int FlatModel::get_dimension() {
    if (d.has_value()) {} 
    else if (A.has_value()) {
        d = A->cols();
    } else if (N.has_value()) {
        d = N->cols() - N->rows();
    } else if (w.has_value()) {
        d = w->size();
    } else {
        throw std::runtime_error("Model wasn't fitted yet. Cannot get dimension.");
    }

    return d.value();
}

int FlatModel::get_ambient_dimension() {
    if (n.has_value()) {} 
    else if (A.has_value()) {
        n = A->rows();
    } else if (N.has_value()) {
        n = N->cols();
    } else if (w.has_value()) {
        n = w->size() + 1;
    } else {
        throw std::runtime_error("Model wasn't fitted yet. Cannot get ambient dimension.");
    }

    return n.value();
}


void FlatModel::override_parametric(const Eigen::MatrixXd& Anew, const Eigen::VectorXd& bnew) {
    A = Anew;
    b_vec = bnew;

    orthonormalized = false;

    if (n.value() == d.value() + 1) {
        parametric_to_explicit();
    }
    
    parametric_to_implicit();

    if (Q.has_value() || r.has_value()) {
        compute_QR();
    }

}

void FlatModel::override_implicit(const Eigen::MatrixXd& Nnew, const Eigen::VectorXd& cnew) {
    N = Nnew;
    c = cnew;

    orthonormalized = false;


    if (n.value() == d.value() + 1) {
        implicit_to_explicit();
    }
    implicit_to_parametric();

    if (Q.has_value() || r.has_value()) {
        compute_QR();
    }
}

void FlatModel::override_explicit(const Eigen::VectorXd& wnew, double bnew) {
    w = wnew;
    b = bnew;

    orthonormalized = false;

    explicit_to_implicit();
    explicit_to_parametric();

    if (Q.has_value() || r.has_value()) {
        compute_QR();
    }
}

void FlatModel::orthonormalize() {
    if (A.has_value()) {
        orthonormalize_parametric();
    } 
    if (N.has_value()) {
        orthonormalize_implicit();
    }
    if (w.has_value()) {
        orthonormalize_explicit();
    } 
    if (!A.has_value() && !N.has_value() && !w.has_value()) {
        throw std::runtime_error("Model wasn't fitted yet. Cannot orthogonalize.");
    }

    orthonormalized = true;
}

void FlatModel::orthonormalize_parametric() {
    if (!A.has_value() || !b_vec.has_value()) {
        throw std::runtime_error("orthogonolize_parametric: A and b_vec must be set!");
    }

    // QR decomposition of A (Q is orthogonal)
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A.value());
    
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(A->rows(), A->cols());
    
    A = Q;

    Eigen::VectorXd b_proj = A.value() * (A.value().transpose() * b_vec.value());
    b_vec = b_vec.value() - b_proj;

}

void FlatModel::orthonormalize_implicit() {
    // QR decomposition of N^T
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(N.value().transpose());
    Eigen::MatrixXd Q = qr.householderQ();
    Eigen::MatrixXd R = qr.matrixR().topLeftCorner(N->rows(), N->rows());

    // Orthonormalize N
    N = Q.leftCols(N->rows()).transpose();

    // Adjust c
    c = R.transpose().inverse() * c.value();
}


void FlatModel::orthonormalize_explicit() {
    // Clean up later, this method is not used anymore
}

void FlatModel::compute_QR() {
    orthonormalize();
    if (A.has_value() && b_vec.has_value()) {
        Q = Eigen::MatrixXd::Identity(A->rows(), A->rows()) - A.value() * A.value().transpose();
        r = -b_vec.value();
    } else if (N.has_value() && c.has_value()) {
        Q = N.value().transpose() * N.value();
        r = N.value().transpose() * c.value();
    } else {
        get_parametric_repr();
        compute_QR();
    }    
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> FlatModel::get_QR() {
    if (!Q.has_value() || !r.has_value()) {
        compute_QR();
    }
    return std::pair<Eigen::MatrixXd, Eigen::VectorXd> {Q.value(), r.value()};
}

double FlatModel::quadratic_loss(const Eigen::VectorXd point) {
    if (!Q.has_value() || !r.has_value()) {
        compute_QR();
    }
    return (point.transpose() * Q.value() * point)(0) + 2 * r->dot(point) + r->dot(r.value());
}