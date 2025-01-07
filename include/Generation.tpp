#ifndef GENERATION_TPP
#define GENERATION_TPP

#include <Eigen/Core>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

#include "Definitions.h"

namespace {
    template <typename Vector>
    void generateGaussianNoiseVector(Vector& result, float mean, float std) {
        std::default_random_engine generator;
        std::normal_distribution<float> dist(mean, std);

        for (int i = 0; i < result.size(); ++i) {
            result[i] = dist(generator);
        }

    }
}

namespace Generation {
    template <typename Vector>
    std::vector<Vector> generateHyperPlane3f(const Vector& w, float b, int N, bool outliers) {
        // TODO:
        // 1. Generate Samples with gaussian noise
        // 2. Generate Outliers
        int d = w.size();   // hyperplane dimensions
        int n = d + 1;      // space dimensions

        // Side length
        int sidePoints = static_cast<int>(std::pow(N, 1.0 / d));
        N = std::pow(sidePoints, d);
        float spacing = SIDE_LEN / (float) sidePoints;
        std::cout << "N: " << N << ", d: " << d << ", spacing: " << spacing << std::endl;

        std::default_random_engine gaussianGenerator;
        std::normal_distribution<float> dist(0.0, spacing / 2.0);

        std::vector<Vector> points;
        for (int i = 0; i < d; ++i) {
            if (i == 0) {
                for (int j = 0; j < sidePoints; ++j) {
                    Vector v(n);
                    v.setZero();
                    v[i] = static_cast<float>(j) * spacing;

                    Vector randomOffset(n);
                    randomOffset.setRandom();
                    randomOffset *= spacing / 2.0;

                    points.push_back(v + randomOffset);
                }
            } else {
                std::vector<Vector> tempPoints = points;  // Copy of current points
                for (auto supportVec : tempPoints) {
                    for (int j = 1; j < sidePoints; ++j) {
                        Vector v(n);
                        v = supportVec;
                        v[i] = float(j) * spacing;
                        
                        Vector randomOffset(n);
                        randomOffset.setRandom();
                        randomOffset *= spacing / 2.0;

                        points.push_back(v + randomOffset);
                    }
                }
            }
        }
        std::vector<Vector> result;
        std::vector<Vector> tempPoints = points;  // Copy of current points
        for (auto supportVec : tempPoints) {
            for (int j = 1; j < sidePoints; ++j) {
                Vector v(n);
                v = supportVec;
                v[d] = v.dot(w) + b;
                

                result.push_back(v);
            }
        }


        return result;


    }
}

#endif