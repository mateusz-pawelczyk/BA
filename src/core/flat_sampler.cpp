#include "core/flat_sampler.hpp"
#include <Eigen/Dense>
#include <random>
#include <iostream>

namespace FlatSampler {
    std::random_device rd;
    std::mt19937 generator(rd());

    Eigen::MatrixXd sampleFlat(FlatModel& model, int N, double noise, double outlierRatio, double outlierStrength, double volume) {
        int d = model.get_dimension();
        int n = model.get_ambient_dimension();


        model.orthonormalize();
        
        auto [A, b] = model.get_parametric_repr();
        // auto [w, bLocal] = model.get_explicit_repr();
        
        // // Print the explicit form
        // std::cout << "Explicit form: " << std::endl;
        // std::cout << "w: " << w.transpose() << std::endl;
        // std::cout << "b: " << bLocal << std::endl;

        // Generate points on the flat
        Eigen::MatrixXd points = sampleGaussianPoints(N, d);

        points = points * A.transpose() + b.transpose().replicate(N, 1); 

        // Add noise
        auto [N_model, c] = model.get_implicit_repr();
        addNoise(points, N_model, noise);

        // Add outliers (salt-and-pepper noise)
        std::uniform_real_distribution<double> dist;
        addOutlier(points, N_model, outlierStrength, outlierRatio);

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

    void addOutlier(Eigen::MatrixXd& points, const Eigen::MatrixXd& N, double strength, double ratio) {
        std::uniform_real_distribution<double> dist;
        double outlierPosNegRatio = dist(generator);
        for (auto normal : N.rowwise()) {
            for (int i = 0; i < points.rows(); ++i) {
                if (dist(generator) < ratio) {
                    points.row(i) += normal * dist(generator) * strength * (dist(generator) < outlierPosNegRatio ? 1.0 : -1.0);
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