#include "visualization/visualizer.h"
#include <iostream>

void Visualizer::plotPoints(const Eigen::MatrixXd& points, const std::string& name, const std::string& pointRenderMode, double pointRadius)  {
    std::vector<glm::vec3> glmPoints;
    for (const auto& v : points.rowwise()) {
        glmPoints.push_back({v.coeff(0), v.coeff(1), v.size() >= 3 ? v.coeff(2) : 0.0});
    }

    polyscope::PointRenderMode renderMode;
    if (pointRenderMode == "Sphere") {
        renderMode = polyscope::PointRenderMode::Sphere;
    } else {
        renderMode = polyscope::PointRenderMode::Quad;
    }

    polyscope::registerPointCloud(name, glmPoints)->setPointRenderMode(renderMode)->setPointRadius(pointRadius);
}


void Visualizer::plotFittingFlat(const Eigen::VectorXd& w_star, double b_star, const std::string& name, double sideLen, double lineRadius, float flatAlpha) {
    int d = w_star.size();
    if (d < 1 || d > 2) return;

    constexpr int numPoints = 2;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    double offset = 0.1;

    if (d == 1) {
        Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(numPoints, 0.0 - offset, sideLen + offset);
        vertices = Eigen::MatrixXd::Zero(numPoints, 3);
        vertices.col(0) = x;
        vertices.col(1) = x * w_star(0) + Eigen::VectorXd::Ones(numPoints) * b_star;

        faces.resize(numPoints - 1, 2);
        faces.col(0) = Eigen::VectorXi::LinSpaced(numPoints - 1, 0, numPoints - 2);
        faces.col(1) = faces.col(0).array() + 1;

        polyscope::registerCurveNetwork(name, vertices, faces)->setRadius(lineRadius);
    } else {
        Eigen::Matrix<double, 4, 2> xy;
        xy << 0.0 - offset, 0.0 - offset,
              sideLen + offset, 0.0 - offset,
              0.0 - offset, sideLen + offset,
              sideLen + offset, sideLen + offset;

        vertices = Eigen::MatrixXd::Zero(4, 3);
        vertices.leftCols(2) = xy;
        vertices.col(2) = xy * w_star + Eigen::Vector4d::Ones() * b_star;

        faces.resize(1, 4);
        faces << 0, 1, 3, 2;

        polyscope::registerSurfaceMesh(name, vertices, faces)->setTransparency(flatAlpha);
    }
}