#include "core/flat_sampler.hpp"
#include <Eigen/Dense>
#include <random>
#include <iostream>

namespace FlatSampler {
    std::random_device rd;
    std::mt19937 generator(rd());

    Eigen::MatrixXd sampleFlat(FlatModel& model, int N, double noise, double outlierRatio, double outlierStrength, double volume, bool saltAndPepper) {
        int d = model.get_dimension();
        int n = model.get_ambient_dimension();


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

    void addNoise(Eigen::MatrixXd& points, const Eigen::MatrixXd& N, double noise) {
        std::normal_distribution<double> gaussianDist(0.0, noise);
        for (auto normal : N.rowwise()) {
            for (int i = 0; i < points.rows(); ++i) {
                points.row(i) += normal * gaussianDist(generator);
            }
        }
    }

    void addOutlier(Eigen::MatrixXd& points, const Eigen::MatrixXd& N, double strength, double ratio, bool saltAndPepper) {
        std::uniform_real_distribution<double> dist;
        double outlierPosNegRatio = dist(generator);
        for (auto normal : N.rowwise()) {
            for (int i = 0; i < points.rows(); ++i) {
                if (dist(generator) < ratio) {
                    Eigen::VectorXd outlierOffset = normal * strength;

                    if (!saltAndPepper) {
                        outlierOffset *= dist(generator);
                    }

                    if (dist(generator) < outlierPosNegRatio) {
                        outlierOffset *= -1;
                    }

                    points.row(i) += outlierOffset;
                }
            }
        }
    }

    Eigen::MatrixXd sampleGaussianPoints(int N, int d) {
        std::normal_distribution<double> gaussianDist(0.0, 1.0);

        Eigen::MatrixXd points = Eigen::MatrixXd::Zero(N, d);

        std::generate(points.data(), points.data() + points.size(), [&]() {
            return gaussianDist(generator);
        });

        return points;
    }

}