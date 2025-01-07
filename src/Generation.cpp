#ifndef GENERATION_CPP
#define GENERATION_CPP

#include <Eigen/Core>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

#include "Generation.h"
#include "Definitions.h"

namespace Generation {
    Eigen::MatrixXd generateHyperPlane3d(const Eigen::Vector2d& w, double b, int N, double noise, double outlierRatio, double outlierStrength) {
        int d = 2;   // hyperplane dimensions
        int n = 3;   // space dimensions


        // Side length
        int sidePoints = static_cast<int>(std::round(std::pow(N, 1.0 / d)));
        N = sidePoints * sidePoints;
        double spacing = SIDE_LEN / (double) sidePoints;
        
        // Generates a vector of size `sidePoints` with evenly spaced out values from 0 to `SIDE_LEN`
        Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(sidePoints, 0.0, SIDE_LEN);

        // Create all possible x and y values.
        Eigen::MatrixXd x = grid.replicate(1, sidePoints);
        Eigen::MatrixXd y = grid.transpose().replicate(sidePoints, 1);

        // Add Random Noise to the Grid
        Eigen::MatrixXd gridNoiseX = Eigen::MatrixXd::Random(sidePoints, sidePoints) * (spacing / 2.0);
        Eigen::MatrixXd gridNoiseY = Eigen::MatrixXd::Random(sidePoints, sidePoints) * (spacing / 2.0);
        Eigen::MatrixXd x_noised = x + gridNoiseX;
        Eigen::MatrixXd y_noised = y + gridNoiseY;


        // Flatten 2D Grid to 3D Points
        Eigen::MatrixXd points(sidePoints * sidePoints, 3);
        points.col(0) = Eigen::Map<Eigen::VectorXd>(x_noised.data(), x_noised.size());
        points.col(1) = Eigen::Map<Eigen::VectorXd>(y_noised.data(), y_noised.size());
        points.col(2).setZero();

        // Project to Hyperplane
        Eigen::VectorXd b_vec(points.rows());
        b_vec.setConstant(b);
        points.col(2) = points.leftCols(2) * w + b_vec;

        // Add Noise
        std::random_device rd;
        std::default_random_engine generator(rd());

        std::normal_distribution<double> gaussianDist(0.0, noise * spacing);
        Eigen::VectorXd noiseVector(points.rows());
        for (int i = 0; i < N; ++i) {
            noiseVector(i) = gaussianDist(generator);
        }
        points.col(2) += noiseVector;


        return points;

    }

    Eigen::MatrixXd generateHyperPlane(const Eigen::VectorXd& w, double b, int N, double noise, double outlierRatio, double outlierStrength) {
        int d = w.size();   // hyperplane dimensions
        int n = d + 1;   // space dimensions

        // Side length
        int sidePoints = static_cast<int>(std::round(std::pow(N, 1.0 / d)));
        N = std::pow(sidePoints, d);
        double spacing = SIDE_LEN / (double) sidePoints;
        
        // Generates a vector of size `sidePoints` with evenly spaced out values from 0 to `SIDE_LEN`
        Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(sidePoints, 0.0, SIDE_LEN);

        // Create all possible x_1, x_2, ..., x_d points
        Eigen::MatrixXd points(N, n);

        for (int i = 0; i < d; ++i) {
            Eigen::MatrixXd col = grid.replicate(std::pow(sidePoints, i), std::pow(sidePoints, d - i - 1)).transpose();
            Eigen::MatrixXd gridNoise = Eigen::MatrixXd::Random(col.rows(), col.cols()) * (spacing / 2.0);
            col += gridNoise;

            points.col(i) = Eigen::Map<Eigen::VectorXd>(col.data(), col.size());

        }

        // Project to Hyperplane
        Eigen::VectorXd b_vec(points.rows());
        b_vec.setConstant(b);
        points.col(d) = points.leftCols(d) * w + b_vec;

        // Add Noise
        std::random_device rd;
        std::default_random_engine generator(rd());

        std::normal_distribution<double> gaussianDist(0.0, noise * spacing);
        Eigen::VectorXd noiseVector(points.rows());
        for (int i = 0; i < N; ++i) {
            noiseVector(i) = gaussianDist(generator);
        }
        points.col(d) += noiseVector;


        return points;

    }
}

#endif