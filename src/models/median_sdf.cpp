#include "models/median_sdf.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <numeric>
#include <iostream>

std::unique_ptr<Model> MedianSDF::clone() const
{
    return std::make_unique<MedianSDF>(*this);
}

void MedianSDF::fit(const Eigen::MatrixXd &D)
{
}
void MedianSDF::fit(const std::vector<std::unique_ptr<FlatModel>> &models, const std::vector<double> &errors)
{
    int model_count = models.size();

    // Validate input sizes
    if (model_count == 0)
    {
        throw std::runtime_error("At least one flat must be provided.");
    }
    std::vector<double> weights = getWeights(errors, model_count);

    // Throw an error if the dimensions are inconsistent
    validateFlatDimensions(models);

    // Initialize Q_star and r_star
    Eigen::MatrixXd Q_star = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd r_star = Eigen::VectorXd::Zero(n);
    // Accumulate Q_i and r_i from all input flats
    for (size_t i = 0; i < model_count; ++i)
    {
        models[i]->orthonormalize();
        auto [A, b] = models[i]->get_parametric_repr();
        auto [Q, r] = models[i]->get_QR();

        Q_star += Q * weights[i];
        r_star -= r * weights[i]; // Careful: In the paper it's the POSITIVE sum, but that is wrong!
    }

    // eigen-decomposition on Q_star
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Q_star);
    if (eigensolver.info() != Eigen::Success)
    {
        throw std::runtime_error("Eigen decomposition of Q_star failed.");
    }

    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues(); // Sorted in ascending order
    Eigen::MatrixXd U = eigensolver.eigenvectors();          // Columns are eigenvectors
    // Select the first k eigenvectors corresponding to the smallest k eigenvalues
    A = Eigen::MatrixXd(n, d);
    if (d > 0)
    {
        A = U.leftCols(d); // n x k matrix
    }
    else
    {
        A = Eigen::MatrixXd::Zero(n, 0); // Empty matrix for k=0
    }

    Eigen::MatrixXd Q_star_pseudo_inv = pseudoInverse(Q_star);

    // I - A A^T (From the formular in the paper)
    Eigen::MatrixXd projection = Eigen::MatrixXd::Identity(n, n) - A.value() * A->transpose();

    // b = (I - A A^T) * Q_star^+ * r_star
    b_vec = projection * Q_star_pseudo_inv * r_star;

    // Verify that b is orthogonal to the subspace spanned by A
    Eigen::VectorXd check = A->transpose() * b_vec.value();
    if (check.norm() > 1e-6)
    {
        throw std::runtime_error("b is not orthogonal to the subspace spanned by A.");
    }
}

Eigen::MatrixXd MedianSDF::pseudoInverse(const Eigen::MatrixXd &A, double tolerance)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    const Eigen::MatrixXd &U = svd.matrixU();
    const Eigen::MatrixXd &V = svd.matrixV();
    const Eigen::VectorXd &S = svd.singularValues();

    Eigen::MatrixXd S_inv = Eigen::MatrixXd::Zero(S.size(), S.size());

    // Invert non-zero singular values
    for (int i = 0; i < S.size(); ++i)
    {
        if (S(i) > tolerance)
        { // Threshold to handle numerical precision issues
            S_inv(i, i) = 1.0 / S(i);
        }
    }

    // pseudo-inverse: V * S_inv * U^T
    return V * S_inv * U.transpose();
}

std::vector<double> MedianSDF::getWeights(const std::vector<double> &errors, int model_count) const
{
    // If no models are provided, no weights can be computed
    if (model_count <= 0)
    {
        return std::vector<double>();
    }

    std::vector<double> weights(model_count);

    if (!errors.empty() && errors.size() == static_cast<size_t>(model_count))
    {
        std::vector<double> modified_errors = errors;

        // Add 1.0 to each error in the copied vector (for R^2 metric, to ensure starting with 0)
        for (size_t i = 0; i < modified_errors.size(); ++i)
        {
            modified_errors[i] += 1.0;
        }

        const double epsilon = 1e-15;

        const double error_sum = std::accumulate(modified_errors.begin(), modified_errors.end(), 0.0) + 1.0 + epsilon;

        // initial weights
        for (size_t i = 0; i < modified_errors.size(); ++i)
        {
            const double numerator = modified_errors[i] + epsilon;
            weights[i] = 1.0 - (numerator / error_sum);
        }

        double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);

        // fall back to uniform distribution when errors are invalid
        if (weight_sum <= 0.0)
        {
            std::fill(weights.begin(), weights.end(), 1.0 / model_count);
        }
        else
        {
            // Normalize weights
            for (auto &w : weights)
            {
                w /= weight_sum;
            }
        }
    }
    else
    {
        // Fallback to uniform weights if errors are invalid
        std::cout << "Errors were not provided or size mismatch. Assuming uniform distribution." << std::endl;
        std::fill(weights.begin(), weights.end(), 1.0 / model_count);
    }

    return weights;
}

void MedianSDF::validateFlatDimensions(const std::vector<std::unique_ptr<FlatModel>> &models) const
{
    int n = models[0]->get_ambient_dimension(); // Space dimension

    for (size_t i = 1; i < models.size(); ++i)
    {
        // Here, the parametric representation is also computed for every model. No need to do it later
        if (models[i]->get_ambient_dimension() != n)
        {
            throw std::runtime_error("MedianSDF: All models must live in the same ambient space.");
        }
    }
}