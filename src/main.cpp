#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <random>
#include <variant>
#include <stdexcept>
#include <string>
#include <format>
#include <unordered_map>
#include <cmath>
#include <Eigen/Dense>

#include "core/generation.h"
#include "core/flat_sampler.hpp"
#include "core/ransac.hpp"
#include "core/flat_model.hpp"
#include "models/ols.hpp"
#include "models/affine_fit.hpp"
#include "Definitions.h"
#include "visualization/visualizer.h"

Eigen::VectorXd w;
double b;
int points = 20;
double noise = 0.01;
double outlierRatio = 0.0;
double outlierStrength = 1.0f;
bool sphereRepr = false;
int d = 2; // Hyperplane Dimensions
int n = 3; // Ambient Space Dimensions
int average_contributions = 10;

int ransac_max_iterations = 1000;
double ransac_threshold = 0.0001;
double ransac_train_data_percenatge = 0.2;
int ransac_min_inliners = points * ransac_train_data_percenatge * 0.3;

std::unordered_map<char*, std::function<std::unique_ptr<Model>(int, int)>> label_to_model;

// Function to register types
template <typename T>
void registerType(char* typeName) {
    label_to_model[typeName] = [](int dimension, int ambient_dimension) -> std::unique_ptr<Model> {
        return std::make_unique<T>(dimension, ambient_dimension);
    };
}

// Function to create an object by type
std::unique_ptr<Model> createObject(char* typeName, int dimension, int ambient_dimension) {
    auto it = label_to_model.find(typeName);
    if (it != label_to_model.end()) {
        return it->second(dimension, ambient_dimension); // Call the factory function
    } else {
        throw std::runtime_error("Type not registered: " + std::string(typeName));
    }
}

std::unique_ptr<Model> model;


// Temporary float variables for UI
Eigen::VectorXf w_ui;
float b_ui;
float noise_ui = static_cast<float>(noise);
float outlierRatio_ui = static_cast<float>(outlierRatio);
float outlierStrength_ui = static_cast<float>(outlierStrength);
float flatAlpha = 0.7;

float ransac_threshold_ui = static_cast<float>(ransac_threshold);
float ransac_train_data_percenatge_ui = static_cast<float>(ransac_train_data_percenatge);


Eigen::MatrixXd hyperplanePoints;

float calculatePointRadius() {
    float sidePoints = std::round(std::pow(points, 1.0 / (float)d));
    float gridCellWidth = SIDE_LEN / sidePoints;
    float pointRadius = std::min(gridCellWidth / 2.0, 0.01);

    return pointRadius;
}

double r2_metric(Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred) {
    double mean = Y_true.mean();
    double ss_tot = (Y_true.array() - mean).square().sum();
    double ss_res = (Y_true - Y_pred).squaredNorm();
    return ss_res / ss_tot;
}


void visualizeFittingPlane() {
    // Loss is per-point and metric is overall
    auto loss_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred){return Eigen::VectorXd((Y_true - Y_pred).array().square().matrix());};
    auto metric_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred){return (Y_true - Y_pred).squaredNorm() / Y_true.size();};

    // Maybe move to global to avoid reinitialization???
    RANSAC ransac(ransac_max_iterations, ransac_threshold, ransac_train_data_percenatge, ransac_min_inliners, loss_fn, r2_metric);
    
    Eigen::MatrixXd X = hyperplanePoints.leftCols(d);
    Eigen::VectorXd Y = hyperplanePoints.col(d);

    std::unique_ptr<Model> singleBestModel = ransac.run(X, Y, model.get());
    std::unique_ptr<FlatModel> averagedBestModel = ransac.run(hyperplanePoints, (FlatModel*)model.get(), average_contributions);

    if (singleBestModel == nullptr) {
        std::cout << "No good Model with these parameters could be found." << std::endl;
        return;
    }

    float pointRadius = calculatePointRadius();

    FlatModel* singleBestModel_ptr = (FlatModel*)singleBestModel.get();
    FlatModel* averagedBestModel_ptr = (FlatModel*)averagedBestModel.get();
    
    double singleBestModelMSE = singleBestModel_ptr->MSE(X, Y);
    double averagedBestModelMSE = averagedBestModel_ptr->MSE(X, Y);

    std::cout << "Single Best Model performance: " << singleBestModelMSE << std::endl;
    std::cout << "Averaged Best Model performance: " << averagedBestModelMSE << std::endl;
    std::cout << "Single Best Model outperformed by: " << (averagedBestModelMSE - singleBestModelMSE) / (averagedBestModelMSE + singleBestModelMSE) << std::endl;
    if (d != 1 && d != 2) return;

    singleBestModel_ptr->visualize("Single Best Flat", 6.0, pointRadius / 2.0, 0.6);
    averagedBestModel_ptr->visualize("Averaged Best Flat", 6.0, pointRadius / 2.0, 0.6);

    
}


void testFlatModel(bool split) {
    int iterations = 10000;
    double singleBestModelMSE = 0.0;
    double averagedBestModelMSE = 0.0;

    int test_points = 10;
    double test_noise = 0.0;
    double test_outlierRatio = 0.0;
    double outlierStrength = 1.0;

    std::default_random_engine generator;
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < iterations; ++i) {
        model = createObject("Affine Fit", d, n);
        w = Eigen::VectorXd::Random(d);
        b = dist(generator);
        hyperplanePoints = Generation::generateHyperPlane(w, b, test_points, test_noise, test_outlierRatio, outlierStrength);
        AffineFit* m = new AffineFit(n - 1, n);
        if (split) {
            m->fit(hyperplanePoints.leftCols(d), hyperplanePoints.col(d));
        } else {
            m->fit(hyperplanePoints);
        }
        hyperplanePoints = FlatSampler::sampleFlat(*m, points, noise, outlierRatio, outlierStrength);

        // Loss is per-point and metric is overall
        auto loss_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred){return Eigen::VectorXd((Y_true - Y_pred).array().square().matrix());};
        auto metric_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred){return (Y_true - Y_pred).squaredNorm() / Y_true.size();};

        RANSAC ransac(ransac_max_iterations, ransac_threshold, ransac_train_data_percenatge, ransac_min_inliners, loss_fn, r2_metric);
        
        Eigen::MatrixXd X = hyperplanePoints.leftCols(d);
        Eigen::VectorXd Y = hyperplanePoints.col(d);

        std::unique_ptr<Model> singleBestModel = ransac.run(X, Y, model.get());
        std::unique_ptr<FlatModel> averagedBestModel = ransac.run(hyperplanePoints, (FlatModel*)model.get(), average_contributions);

        if (singleBestModel == nullptr) {
            std::cout << "No good Model with these parameters could be found." << std::endl;
            return;
        }

        float pointRadius = calculatePointRadius();

        FlatModel* singleBestModel_ptr = (FlatModel*)singleBestModel.get();
        FlatModel* averagedBestModel_ptr = (FlatModel*)averagedBestModel.get();
        
        singleBestModelMSE += singleBestModel_ptr->MSE(X, Y);
        averagedBestModelMSE += averagedBestModel_ptr->MSE(X, Y);

        // Increment all parameters
        noise += 0.3 / iterations;
        outlierRatio += 1.0 / iterations;
        outlierStrength += 10.0 / iterations;
        test_points += 10;
    }

    std::cout << "Single Best Model performance: " << singleBestModelMSE / iterations << std::endl;
    std::cout << "Averaged Best Model performance: " << averagedBestModelMSE / iterations << std::endl;


}


void generatePointCloud() {
    hyperplanePoints = Generation::generateHyperPlane(w, b, points, noise, outlierRatio, outlierStrength);

    // Compare datasets	
    AffineFit* m = new AffineFit(n - 1, n);
    m->fit(hyperplanePoints.leftCols(d), hyperplanePoints.col(d));
    // Eigen::MatrixXd D1 = FlatSampler::sampleFlat(*m, points, noise, outlierRatio, outlierStrength);

    // m = new AffineFit(n - 1);
    // m->fit(hyperplanePoints);
    hyperplanePoints = FlatSampler::sampleFlat(*m, points, noise, outlierRatio, outlierStrength);

    float pointRadius = calculatePointRadius();
    Visualizer::plotPoints(hyperplanePoints, "Hyperplane Point Cloud", sphereRepr ? "Sphere" : "Quad", pointRadius);

    visualizeFittingPlane();
}

void dataParameterGUI() {
    ImGui::Text("Data Parameters");

    ImGui::SliderFloat("noise", &noise_ui, 0.0f, 0.3f, "%.4f");
    ImGui::SliderFloat("outlier fraction", &outlierRatio_ui, 0.0f, 1.0f);
    ImGui::SliderFloat("outlier strength", &outlierStrength_ui, 1.0f, 10.0f, "%.4f");
    ImGui::SliderInt("Number of Points", &points, 20, 20000);
    ImGui::SliderInt("Flat dimensions", &d, 1, 3);
}

void dataReprGUI() {
    ImGui::Text("Representation");
    ImGui::Checkbox("Represent Points as Spheres? (less efficient)", &sphereRepr);
    ImGui::SliderFloat("Flat Transparency", &flatAlpha, 0.0, 1.0);
}


void ransacParameterGUI() {
    ImGui::Text("RANSAC Hyper Parameter");
    ImGui::SliderInt("Max Iterations", &ransac_max_iterations, 100, 10000);
    ImGui::SliderFloat("Threshold", &ransac_threshold_ui, 0.0f, 0.1f, "%.6f");
    ImGui::SliderFloat("Train-Data Percentage", &ransac_train_data_percenatge_ui, 0.0f, 1.0f);
    ImGui::SliderInt("Minimum Inliners", &ransac_min_inliners, points * ransac_train_data_percenatge * 0.1, points * ransac_train_data_percenatge * 0.9);
    ImGui::SliderInt("Model Average Count", &average_contributions, 1, static_cast<int>(std::sqrt(points)));

}


void initializeLabelToModel() {
    registerType<OLS>("OLS");
    registerType<AffineFit>("Affine Fit");
    model = createObject("Affine Fit", d, n);
}

void modelSelectionGUI() {
    std::vector<char*> items;
    for (auto& [key, value] : label_to_model) {
        items.push_back(key);
    }
    static const char* current_item = items[0];

    ImGui::Text("Model:");
    if (ImGui::BeginCombo("##combo", current_item)) 
    {
        for (int i = 0; i < items.size(); i++)
        {
            bool is_selected = (current_item == items[i]);
            if (ImGui::Selectable(items[i], is_selected)) {
                current_item = items[i];
                model = createObject(items[i], d, n);
                // model->fit(hyperplanePoints.leftCols(d), hyperplanePoints.col(d));
            }
            if (is_selected)
                ImGui::SetItemDefaultFocus();  
        }
        ImGui::EndCombo();
}
}



int main() {
    polyscope::init();

    w = Eigen::VectorXd::Random(d);
    w_ui = w.cast<float>();
    b = 0.0;

    initializeLabelToModel();
    // std::cout << "Test model with split:" << std::endl;
    // testFlatModel(true);
    // std::cout << "Test model without split:" << std::endl;
    // testFlatModel(false);
    generatePointCloud();

    
    // UI for sliders
    polyscope::state::userCallback = [&]() {
        Generation::flatParameterGUI(w_ui, b_ui);
        modelSelectionGUI();
        dataParameterGUI();
        ransacParameterGUI();
        dataReprGUI();

        if (ImGui::Button("Regenerate Line")) {
            polyscope::removeAllStructures();
            w_ui.conservativeResize(d);
            w.conservativeResize(d);
            w = w_ui.cast<double>();
            b = static_cast<double>(b_ui);
            noise = static_cast<double>(noise_ui);
            outlierRatio = static_cast<double>(outlierRatio_ui);
            outlierStrength = static_cast<double>(outlierStrength_ui);

            ransac_threshold = static_cast<double>(ransac_threshold_ui);
            ransac_train_data_percenatge = static_cast<double>(ransac_train_data_percenatge_ui);
            
            generatePointCloud();
        }
    };


    polyscope::show();

    return 0;
}
