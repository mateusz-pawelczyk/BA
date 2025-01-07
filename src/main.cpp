#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include <polyscope/surface_mesh.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <random>
#include <variant>
#include <stdexcept>
#include <string>
#include <format>

#include "Generation.h"
#include "Regression.h"
#include "Definitions.h"

Eigen::VectorXd w;
double b;
int points = 200;
double noise = 1.0;
double outlierRatio = 0.0;
double outlierStrength = 5.0f;
bool sphereRepr = false;


// Temporary float variables for UI
Eigen::VectorXf w_ui;
float b_ui;
float noise_ui = static_cast<float>(noise);
float outlierRatio_ui = static_cast<float>(outlierRatio);
float outlierStrength_ui = static_cast<float>(outlierStrength);

Eigen::MatrixXd hyperplanePoints;


void visualizeFittingPlane() {
    // TODO: CHANGE SO THAT 1-D HYPERPLANES ARE VISUALIZED AS LINES
    auto [w_star, b_star] = Regression::OLS(hyperplanePoints);
    int d = w_star.size();

    Eigen::Vector2d xy0 = {0.0, 0.0};
    Eigen::Vector2d xy1 = {SIDE_LEN, 0.0};
    Eigen::Vector2d xy2 = {0.0, SIDE_LEN};
    Eigen::Vector2d xy3 = {SIDE_LEN, SIDE_LEN};

    Eigen::Matrix<double, 4, 2> xy;
    xy << 0.0, 0.0,
          SIDE_LEN, 0.0,
          0.0, SIDE_LEN,
          SIDE_LEN, SIDE_LEN;
    
    Eigen::Vector4d one = Eigen::Vector4d::Ones();
    Eigen::Vector4d z = xy * w_star.head(2) + b_star * one; // xy matrix multiplied by w to get the lin. comb. for each vertex, then add the same b everywhere

    // Concatenate xy and z to form 3D points
    Eigen::MatrixXd vertices(4, 3);
    vertices.setZero();
    vertices.leftCols(2) = xy.cast<double>();  // Copy xy to first 2 columns
    vertices.col(2) = z.cast<double>();        // Copy z to 3rd column


    Eigen::MatrixXi faces(1, 4);
    faces << 0, 1, 3, 2; 

    polyscope::registerSurfaceMesh("Quad Mesh", vertices, faces);
}

void generatePointCloud() {
    //hyperplanePoints = Generation::generateHyperPlane3d(w, b, points, noise, outlierRatio, outlierStrength);
    hyperplanePoints = Generation::generateHyperPlane(w, b, points, noise, outlierRatio, outlierStrength);

    // print mean of points
    std::cout << "Mean of points: " << hyperplanePoints.colwise().mean().transpose() << std::endl;

    // Convert Eigen vectors to glm::vec3
    std::vector<glm::vec3> glmPoints;
    for (const auto& v : hyperplanePoints.rowwise()) {
        glmPoints.push_back({v.coeff(0), v.coeff(1), v.size() >= 3 ? v.coeff(2) : 0.0});
    }

    
    polyscope::PointRenderMode pointRenderMode;
    if (sphereRepr) {
        pointRenderMode = polyscope::PointRenderMode::Sphere;
    } else {
        pointRenderMode = polyscope::PointRenderMode::Quad;

    }

    float sidePoints = std::pow(points, 1.0 / 2);
    polyscope::registerPointCloud("Cube Point Cloud", glmPoints)->setPointRenderMode(pointRenderMode)->setPointRadius(std::max(0.0001, std::min(0.01, SIDE_LEN / sidePoints / 2.0)));;

    visualizeFittingPlane();
}


int main() {
    polyscope::init();

    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist;


    // Innitialize w and b random
    w = Eigen::VectorXd::Random(3);
    w_ui = w.cast<float>();
    b = dist(gen);
    
    generatePointCloud();


    // UI for sliders
    polyscope::state::userCallback = [&]() {
        ImGui::Text("Adjust Line Parameters");
        
        for (int i = 0; i < w_ui.size(); ++i) {
            ImGui::SliderFloat(std::format("w[{}]", i).c_str(), &w_ui[i], -5.0f, 5.0f);
        }

        ImGui::SliderFloat("b", &b_ui, -10.0f, 10.0f);
        ImGui::SliderFloat("noise", &noise_ui, 0.0f, 5.0f);
        ImGui::SliderFloat("outlier fraction", &outlierRatio_ui, 0.0f, 1.0f);
        ImGui::SliderFloat("outlier strength", &outlierStrength_ui, 0.0f, 4.0f);
        ImGui::SliderInt("Number of Points", &points, 20, 20000);
        ImGui::Checkbox("Represent Points as Spheres? (less efficient)", &sphereRepr);

        // Update double variables when "Regenerate Line" is clicked
        if (ImGui::Button("Regenerate Line")) {
            w = w_ui.cast<double>();
            b = static_cast<double>(b_ui);
            noise = static_cast<double>(noise_ui);
            outlierRatio = static_cast<double>(outlierRatio_ui);
            outlierStrength = static_cast<double>(outlierStrength_ui);
            
            generatePointCloud();
        }
    };

    polyscope::show();

    return 0;
}
