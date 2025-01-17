#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <Eigen/Core>
#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>


class Visualizer {
public:
    static void plotPoints(const Eigen::MatrixXd& points, const std::string& name, const std::string& pointRenderMode, double pointRadius);
    static void plotFittingFlat(const Eigen::VectorXd& w, double b, const std::string& name, double sideLen = 0.5, double lineRadius = 0.01, float flatAlpha=1.0);
};


#endif