#ifndef GENERATION_CPP
#define GENERATION_CPP

#include <Eigen/Core>
#include <vector>
#include <random>
#include <cmath>
#include <format>
#include <iostream>

#include "polyscope/polyscope.h"
#include "core/generation.h"
#include "Definitions.h"

namespace Generation {
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

        std::normal_distribution<double> gaussianDist(0.0, noise);
        std::uniform_real_distribution<double> dist;
        double outlierPosNegRatio = dist(generator);
        Eigen::VectorXd noiseVector(points.rows());
        for (int i = 0; i < N; ++i) {
            if (dist(generator) > outlierRatio)
                noiseVector(i) = gaussianDist(generator);
            else
                noiseVector(i) = dist(generator) * outlierStrength * (dist(generator) < outlierPosNegRatio ? 1.0 : -1.0 );
        }
        points.col(d) += noiseVector;


        return points;

    }

    void flatParameterGUI(Eigen::VectorXf& w_ui, float& b_ui) {
        ImGui::Text("Flat Parameters");
            
        for (int i = 0; i < w_ui.size(); ++i) {
            ImGui::SliderFloat(std::format("w[{}]", i).c_str(), &w_ui[i], -5.0f, 5.0f);
        }

        ImGui::SliderFloat("b", &b_ui, -10.0f, 10.0f);
    }
}

#endif