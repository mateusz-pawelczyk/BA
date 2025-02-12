#pragma once

#include "core/model.hpp"
#include <Eigen/Core>
#include <optional>
#include <tuple>

class FlatModel : public Model
{
protected:
    std::optional<Eigen::MatrixXd> A;     // (n x d) Matrix with direction-vectors in the columns
    std::optional<Eigen::VectorXd> b_vec; // Bias- (or offset-) n-vector

    std::optional<Eigen::VectorXd> w; // (d - 1) weight vector
    std::optional<double> b;          // bias

    std::optional<Eigen::MatrixXd> N; // ((n - d) x n) Matrix with the normal vectors of the flat in the rows
    std::optional<Eigen::VectorXd> c; // offset n-vector of normal form

    std::optional<Eigen::MatrixXd> Q; // (n x n) Matrix with the orthogonal complement of A
    std::optional<Eigen::VectorXd> r; // offset n-vector of the orthogonal complement of A

    int d; // Dimension of the flat
    int n; // Ambient space dimension
    bool orthonormalized = false;

    void parametric_to_implicit();
    void implicit_to_parametric();
    void parametric_to_explicit();
    void explicit_to_parametric();
    void implicit_to_explicit();
    void explicit_to_implicit();

    void orthonormalize_parametric();
    void orthonormalize_implicit();
    void orthonormalize_explicit();

    void compute_QR();

public:
    virtual ~FlatModel() noexcept = default;

    FlatModel(int d, int n) : d(d), n(n) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y) override;
    virtual void fit(const Eigen::MatrixXd &D) override = 0;

    virtual double predict(const Eigen::VectorXd &point) override;
    virtual Eigen::VectorXd predict(const Eigen::MatrixXd &data) override;

    // Cloning
    virtual std::unique_ptr<Model> clone() const override = 0;

    // Visualize
    virtual void visualize(const std::string &name, double sideLen, double lineRadius, float flatAlpha) override;

    std::pair<Eigen::MatrixXd, Eigen::VectorXd> get_implicit_repr();
    std::pair<Eigen::VectorXd, double> get_explicit_repr();
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> get_parametric_repr();

    int get_dimension();
    int get_ambient_dimension();

    void override_parametric(const Eigen::MatrixXd &Anew, const Eigen::VectorXd &bnew);
    void override_implicit(const Eigen::MatrixXd &Nnew, const Eigen::VectorXd &cnew);
    void override_explicit(const Eigen::VectorXd &wnew, double bnew);

    void orthonormalize();
    bool is_orthonormalized() { return orthonormalized; }

    std::pair<Eigen::MatrixXd, Eigen::VectorXd> get_QR();

    double quadratic_loss(const Eigen::VectorXd point);
    Eigen::VectorXd quadratic_loss(const Eigen::MatrixXd &points);

    void reset();

    double R2(const Eigen::MatrixXd &D);
    double R2(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y);
};
