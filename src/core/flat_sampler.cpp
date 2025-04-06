#include "core/flat_sampler.hpp"
#include "models/affine_fit.hpp"
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <vector>
#include <algorithm>

namespace FlatSampler
{
    // Global random generator.
    std::random_device rd;
    std::mt19937 generator(rd());

    /*------------------------------------------------------------
     *                   HELPER FUNCTIONS
     *------------------------------------------------------------*/
    
    // 1) Sample standard Gaussian points.
    Eigen::MatrixXd sampleGaussianPoints(int N, int d)
    {
        std::normal_distribution<double> gaussianDist(0.0, 1.0);
        Eigen::MatrixXd points(N, d);
        for (int i = 0; i < N * d; ++i)
        {
            points.data()[i] = gaussianDist(generator);
        }
        return points;
    }

    // 2) Sample canonical points (row i has 1 at (i,i), for i < d).
    Eigen::MatrixXd sampleCanonicalPoints(int num_points, int dim)
    {
        Eigen::MatrixXd points = Eigen::MatrixXd::Zero(num_points, dim);
        for (int i = 0; i < std::min(num_points, dim); ++i)
        {
            points(i, i) = 1.0;
        }
        return points;
    }

    // 3) Combined "canonical + Gaussian" points.
    Eigen::MatrixXd sampleCanonicalGaussianPoints(int N, int d)
    {
        Eigen::MatrixXd gauss = sampleGaussianPoints(N, d);
        Eigen::MatrixXd canon = sampleCanonicalPoints(N, d);
        return gauss + canon;
    }

    // 4) Add noise: for each normal row in `normals`, add a small Gaussian offset to each point.
    void addNoiseToPoints(Eigen::MatrixXd &points, const Eigen::MatrixXd &normals, double noise, std::mt19937 &gen)
    {
        if (noise <= 0.0) return;
        
        std::normal_distribution<double> gaussianDist(0.0, noise);
        // Each row of `normals` is a normal direction. We add a random offset in that direction.
        // If normals is (m x d), we loop over each row:
        for (int row = 0; row < normals.rows(); ++row)
        {
            Eigen::VectorXd normal = normals.row(row);
            for (int i = 0; i < points.rows(); ++i)
            {
                double offsetFactor = gaussianDist(gen);
                points.row(i) += offsetFactor * normal.transpose();
            }
        }
    }

    /*------------------------------------------------------------
     *    KEY FIX: DECIDE "OUTLIER OR NOT" EXACTLY ONCE PER POINT
     *------------------------------------------------------------*/
    void addOutlierToPoints(Eigen::MatrixXd &points,
                            const Eigen::MatrixXd &normals,
                            double strength, 
                            double ratio,
                            bool saltAndPepper,
                            std::vector<bool> &outlierMask,
                            std::mt19937 &gen)
    {
        if (ratio <= 0.0 || strength <= 0.0 || normals.rows() == 0)
        {
            // Nothing to do
            return;
        }

        // This distribution decides if a point is outlier or not (once per point).
        std::bernoulli_distribution outlierDist(ratio);

        // For each point, we decide if it’s outlier or not, then we apply outlier offsets.
        // That ensures that "ratio" is indeed the fraction of outliers on average.
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int i = 0; i < points.rows(); ++i)
        {
            // Decide if the i-th point is outlier or not.
            bool isOutlier = outlierDist(gen);
            if (isOutlier)
            {
                outlierMask[i] = true;

                // Add outlier offset in each normal direction. 
                // Optionally randomizing sign or magnitude, depending on saltAndPepper.
                for (int row = 0; row < normals.rows(); ++row)
                {
                    Eigen::VectorXd normal = normals.row(row);
                    Eigen::VectorXd offset = strength * normal;

                    // If not salt&pepper, randomize the magnitude further.
                    if (!saltAndPepper)
                    {
                        // scale offset by random factor in [0,1].
                        offset *= dist(gen);
                    }

                    // Random sign flip for variety if you want (similar to old code).
                    // (You could also do a separate sign flip per direction or do none at all.)
                    if (dist(gen) < 0.5)
                    {
                        offset *= -1.0;
                    }

                    points.row(i) += offset.transpose();
                }
            }
        }
    }

    /*------------------------------------------------------------
     *                   sampleFlat
     *------------------------------------------------------------*/
    Eigen::MatrixXd sampleFlat(FlatModel &model,
                               int N,
                               double noise,
                               double outlierRatio,
                               double outlierStrength,
                               bool saltAndPepper)
    {
        // Prelims
        int d = model.get_dimension();
        // int n = model.get_ambient_dimension(); // if needed

        // Scale noise and outlier by sqrt(d) if that is the desired effect
        noise *= std::sqrt(d);
        outlierStrength *= std::sqrt(d);

        // Orthonormalize the model so parametric & implicit forms are consistent.
        model.orthonormalize();

        // Parametric representation: points on the flat = X * A^T + b^T
        auto [A, b] = model.get_parametric_repr();

        // Generate base points from a d-dimensional Gaussian.
        Eigen::MatrixXd points = sampleGaussianPoints(N, d);
        // Map them to the flat in n-dimensional space.
        points = points * A.transpose() + b.transpose().replicate(N, 1);

        // Get the normal directions for noise/outlier addition
        auto [N_model, c] = model.get_implicit_repr(); // N_model is (n-d) x n if the flat is dimension d in n-D

        // Add noise
        addNoiseToPoints(points, N_model, noise, generator);

        // Add outliers
        std::vector<bool> outlierMask(points.rows(), false);
        addOutlierToPoints(points, N_model, outlierStrength, outlierRatio, saltAndPepper, outlierMask, generator);

        return points;
    }

    /*------------------------------------------------------------
     *                  sampleFlat2
     *------------------------------------------------------------*/
    /// @brief Samples `k`-dimensional affine subspaces (flats) contained within a given `d`-dimensional flat.
    /// @param model The `d`-dimensional flat model in `n`-dimensional space from which to sample.
    /// @param N Number of `k`-flats to generate.
    /// @param k Dimension of the sampled flats (must satisfy `k` < `d`).
    /// @param noise Strength of the noise.
    /// @param outlierRatio Fraction of the sampled flats that should be outliers.
    /// @param outlierStrength Strength of the outliers.
    /// @param saltAndPepper Whether to use salt-and-pepper noise.
    /// @return Vector of sampled flats.
    std::vector<std::unique_ptr<FlatModel>>
    sampleFlat2(FlatModel &model, int N, int k, double noise, double outlierRatio, double outlierStrength, bool saltAndPepper)
    {
        int d = model.get_dimension();
        int n = model.get_ambient_dimension();

        if (k >= d)
        {
            throw std::runtime_error("Sampled flats must have dimension (k) less than the flat (d) they are sampled from.");
        }

        // Orthonormalize for consistent parametric & implicit forms
        model.orthonormalize();
        auto [A, b_vec] = model.get_parametric_repr();

        std::vector<std::unique_ptr<FlatModel>> flats;
        flats.reserve(N);

        for (int i = 0; i < N; ++i)
        {
            // We only need enough points to fit a k-flat, but let's do extra in case of colinearity:
            int minPoints = k + 1;
            int sampleCount = 2 * minPoints;

            // Sample from a "canonical + Gaussian" distribution in the d-dimensional space of the original flat
            Eigen::MatrixXd subPoints = sampleCanonicalGaussianPoints(sampleCount, d);

            // Map them into the nD ambient space
            Eigen::MatrixXd ambientPoints = subPoints * A.transpose() + b_vec.transpose().replicate(sampleCount, 1);

            // Fit a new k-dim flat to those points
            auto newFlat = std::make_unique<AffineFit>(k, n);
            newFlat->fit(ambientPoints);
            flats.push_back(std::move(newFlat));
        }

        return flats;
    }

    /*------------------------------------------------------------
     *             sampleFlatSeparated
     *------------------------------------------------------------*/
    /// @brief Samples points on a flat and separates the inliers (including noisy ones) from the outliers.
    /// @param model The flat model from which to sample.
    /// @param N Total number of points to generate.
    /// @param noise Strength of the noise.
    /// @param outlierRatio Fraction of points to be given outlier offsets.
    /// @param outlierStrength Strength of the outlier offsets.
    /// @param saltAndPepper Whether to use salt-and-pepper noise.
    /// @return A pair of matrices: (inliers, outliers).
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
    sampleFlatSeparated(FlatModel &model, int N, double noise,
                        double outlierRatio, double outlierStrength,
                        bool saltAndPepper)
    {
        // Scale parameters by sqrt(d) if you want the same approach as sampleFlat
        int d = model.get_dimension();
        noise *= std::sqrt(d);
        outlierStrength *= std::sqrt(d);

        // Orthonormalize
        model.orthonormalize();
        auto [A, b] = model.get_parametric_repr();

        // Base points from Gaussian -> mapped to the flat
        Eigen::MatrixXd points = sampleGaussianPoints(N, d);
        points = points * A.transpose() + b.transpose().replicate(N, 1);

        // Get normals for noise/outliers
        auto [N_model, c] = model.get_implicit_repr();

        // Add noise
        addNoiseToPoints(points, N_model, noise, generator);

        // Add outliers while keeping track via outlierMask
        std::vector<bool> outlierMask(points.rows(), false);
        addOutlierToPoints(points, N_model, outlierStrength, outlierRatio,
                           saltAndPepper, outlierMask, generator);

        // Separate them
        int outlierCount = 0;
        for (bool isOut : outlierMask) {
            if (isOut) outlierCount++;
        }
        int inlierCount = N - outlierCount;

        Eigen::MatrixXd inliers(inlierCount, points.cols());
        Eigen::MatrixXd outliers(outlierCount, points.cols());

        int inlierIdx = 0, outlierIdx = 0;
        for (int i = 0; i < points.rows(); ++i)
        {
            if (outlierMask[i])
            {
                outliers.row(outlierIdx++) = points.row(i);
            }
            else
            {
                inliers.row(inlierIdx++) = points.row(i);
            }
        }

        return std::make_pair(inliers, outliers);
    }

} // namespace FlatSampler
