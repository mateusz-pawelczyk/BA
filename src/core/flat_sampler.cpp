#include "core/flat_sampler.hpp"
#include "models/affine_fit.hpp"
#include <Eigen/Dense>
#include <random>
#include <iostream>

namespace FlatSampler
{
    std::random_device rd;
    std::mt19937 generator(rd());

    Eigen::MatrixXd sampleFlat(FlatModel &model, int N, double noise, double outlierRatio, double outlierStrength, bool saltAndPepper)
    {
        int d = model.get_dimension();
        int n = model.get_ambient_dimension();

        noise *= std::sqrt(d);
        outlierStrength *= std::sqrt(d);

        model.orthonormalize();

        auto [A, b] = model.get_parametric_repr();

        // Generate points on the flat
        Eigen::MatrixXd points = sampleGaussianPoints(N, d);

        points = points * A.transpose() + b.transpose().replicate(N, 1);

        // Add noise
        auto [N_model, c] = model.get_implicit_repr();
        addNoise(points, N_model, noise);

        // Add outliers (salt-and-pepper noise)
        std::uniform_real_distribution<double> dist;
        addOutlier(points, N_model, outlierStrength, outlierRatio, saltAndPepper);

        return points;
    }

    /// @brief Samples `k`-dimensional affine subspaces (flats) contained within a given `d`-dimensional flat.
    /// @param model The `d`-dimensional flat model in `n`-dimensional space from which to sample.
    /// @param N Number of `k`-flats to generate.
    /// @param k Dimension of the sampled flats (must satisfy `k` < `d`).
    /// @param noise strength of the noise
    /// @param outlierRatio how much of the sampled flats should be outliers
    /// @param outlierStrength how strong the outliers should be
    /// @param saltAndPepper whether to use salt-and-pepper noise
    /// @return Vector of sampled flats
    std::vector<std::unique_ptr<FlatModel>> sampleFlat2(FlatModel &model, int N, int k, double noise, double outlierRatio, double outlierStrength, bool saltAndPepper)
    {
        int d = model.get_dimension();
        int n = model.get_ambient_dimension();

        if (k >= d)
        {
            throw std::runtime_error("Sampled flats must have dimension (k) less than the flat (d) they are sampled from.");
        }

        model.orthonormalize();
        auto [A, b_vec] = model.get_parametric_repr();

        std::vector<std::unique_ptr<FlatModel>> models;
        models.reserve(N);
        for (int i = 0; i < N; ++i)
        {
            // minimum number of points to fit the flat
            int min_points = k + 1;

            // we sample 2 * min_points to account for random colinearity
            // AffineFit::fit will skip colinear points, so we need some buffer to be sure
            Eigen::MatrixXd subSamplePoints = sampleCanonicalGaussianPoints(2 * min_points, d);
            Eigen::MatrixXd ambient_points = subSamplePoints * A.transpose() + b_vec.transpose().replicate(2 * min_points, 1);

            auto new_flat = std::make_unique<AffineFit>(k, n);
            new_flat->fit(ambient_points);
            models.push_back(std::move(new_flat));
        }

        return models;
    }

    Eigen::MatrixXd sampleCanonicalPoints(int num_points, int dim)
    {
        Eigen::MatrixXd points = Eigen::MatrixXd::Zero(num_points, dim);
        for (int i = 0; i < std::min(num_points, dim); ++i)
        {
            points(i, i) = 1.0; // First 'dim' points form a simplex
        }
        return points;
    }

    void addNoise(Eigen::MatrixXd &points, const Eigen::MatrixXd &N, double noise)
    {
        std::normal_distribution<double> gaussianDist(0.0, noise);
        for (auto normal : N.rowwise())
        {
            for (int i = 0; i < points.rows(); ++i)
            {
                points.row(i) += normal * gaussianDist(generator);
            }
        }
    }

    void addOutlier(Eigen::MatrixXd &points, const Eigen::MatrixXd &N, double strength, double ratio, bool saltAndPepper)
    {
        std::uniform_real_distribution<double> dist;
        double outlierPosNegRatio = dist(generator);
        for (auto normal : N.rowwise())
        {
            for (int i = 0; i < points.rows(); ++i)
            {
                if (dist(generator) < ratio)
                {
                    Eigen::VectorXd outlierOffset = normal * strength;

                    if (!saltAndPepper)
                    {
                        outlierOffset *= dist(generator);
                    }

                    if (dist(generator) < outlierPosNegRatio)
                    {
                        outlierOffset *= -1;
                    }

                    points.row(i) += outlierOffset;
                }
            }
        }
    }

    Eigen::MatrixXd sampleGaussianPoints(int N, int d)
    {
        std::normal_distribution<double> gaussianDist(0.0, 1.0);
        Eigen::MatrixXd points = Eigen::MatrixXd::Zero(N, d);
        std::generate(points.data(), points.data() + points.size(), [&]()
                      { return gaussianDist(generator); });
        // Normalize so that total variance ~ 1
        // points /= std::sqrt(d);
        return points;
    }

    Eigen::MatrixXd sampleCanonicalGaussianPoints(int N, int d)
    {
        Eigen::MatrixXd points = sampleGaussianPoints(N, d) + sampleCanonicalPoints(N, d);

        return points;
    }

}