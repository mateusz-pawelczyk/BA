#pragma once

#include "core/flat_model.hpp" // Base class
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <memory>
#include <vector>
#include "core/types.hpp"
#include <Eigen/Eigenvalues>
// #include <ceres/local_parameterization.h> // Removed as per request

namespace huber_regression_detail {

// Vertical-error Huber residual: r = y – (wᵀx + b)
// This functor calculates the residual for vertical regression.
// The parameters are [w_1, ..., w_d, b].
struct HuberVerticalFunctor {
  HuberVerticalFunctor(const Eigen::VectorXd &x_in, double y_in)
    : x_(x_in), y_(y_in), dim_(x_in.size()) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residual) const {
    const T* w_and_b = parameters[0]; // Combined w and b parameters
    T prediction = T(0.0);

    // Dot product w^T * x
    for (int j = 0; j < dim_; ++j) {
      prediction += w_and_b[j] * T(x_(j));
    }
    // Add bias term
    prediction += w_and_b[dim_];

    residual[0] = T(y_) - prediction;
    return true;
  }

private:
  const Eigen::VectorXd x_; // Feature vector (d dimensions)
  const double y_;          // Target value
  const int dim_;           // Dimension of x_ (d)
};

// Orthogonal-error Huber residual: signed distance to hyperplane (nᵀx + c) / ‖n‖
// This functor calculates the residual for orthogonal regression.
// Parameters are [n_1, ..., n_N] and [c].
struct HuberOrthogonalFunctor {
  HuberOrthogonalFunctor(const Eigen::VectorXd &point_in) : point_(point_in), dim_(point_in.size()) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residual) const {
    const T* n_params = parameters[0]; // Normal vector n (N dimensions)
    const T* c_param = parameters[1];  // Offset c (1 dimension)

    T numerator = T(0.0);
    // Dot product n^T * point
    for (int i = 0; i < dim_; ++i) {
      numerator += n_params[i] * T(point_(i));
    }
    numerator += c_param[0]; // Add offset c

    T norm_n_sq = T(0.0);
    // Squared norm of n: ||n||^2
    for (int i = 0; i < dim_; ++i) {
      norm_n_sq += n_params[i] * n_params[i];
    }

    // Avoid division by zero or very small norm if n is not yet constrained.
    // The UnitNormConstraintFunctor should drive norm_n towards 1.
    T norm_n = ceres::sqrt(norm_n_sq + T(1e-12)); // Add epsilon for stability

    residual[0] = numerator / norm_n;
    return true;
  }

private:
  const Eigen::VectorXd point_; // Data point (N dimensions)
  const int dim_;               // Dimension of point_ and normal vector n (N)
};

// Constraint functor to enforce ‖n‖² - 1 = 0
// This functor helps enforce that the normal vector n has unit norm.
// The residual is n^T*n - 1.
struct UnitNormConstraintFunctor {
    UnitNormConstraintFunctor(int dimension) : dim_(dimension) {}

    template <typename T>
    // Corrected signature: Takes an array of pointers to parameter blocks
    bool operator()(T const* const* parameters, T* residual) const {
        const T* n_params = parameters[0]; // Extract the first parameter block (n)
        T norm_sq = T(0.0);
        for (int i = 0; i < dim_; ++i) {
            norm_sq += n_params[i] * n_params[i];
        }
        residual[0] = norm_sq - T(1.0);
        return true;
    }

private:
    const int dim_; // Dimension of the normal vector n
};

} // namespace huber_regression_detail


class HuberRegression : public FlatModel {
public:
  // d = # features (for vertical), n_ambient = ambient dim (for orthogonal)
  HuberRegression(int d_features,
                  int n_ambient,
                  DistanceType dist = DistanceType::Regression,
                  double delta = 1.0);

  virtual ~HuberRegression() noexcept override = default;

  // D: each row = [x₁ … x_d, y] for vertical regression,
  //            = [p₁ … p_n] for orthogonal regression
  void fit(const Eigen::MatrixXd &D) override;

  std::unique_ptr<Model> clone() const override;

private:
  DistanceType distance_type_;
  double huber_delta_;
  // d_features_ and n_ambient_ are available from FlatModel base class as 'd' and 'n' respectively.
  // 'd' in FlatModel is the intrinsic dimension of the flat.
  // 'n' in FlatModel is the ambient dimension.
  // For vertical regression: y = w^T x + b. x is d-dim. Ambient space for (x,y) is (d+1)-dim.
  //   FlatModel's d = d_features, FlatModel's n = d_features + 1.
  // For orthogonal regression: n^T p + c = 0. p is n_ambient-dim.
  //   FlatModel's d = n_ambient - 1 (dimension of hyperplane), FlatModel's n = n_ambient.
};