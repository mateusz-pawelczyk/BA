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

#include "evaluation/evaluator.hpp"

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
bool saltAndPepper = false;
MetricType metric = MetricType::MSE;

int ransac_max_iterations = 50;
double ransac_threshold = 0.0001;
double ransac_train_data_percenatge = 0.2;
int ransac_min_inliners = points * ransac_train_data_percenatge * 0.3;

std::unordered_map<char *, std::function<std::unique_ptr<Model>(int, int)>> label_to_model;

// Function to register types
template <typename T>
void registerType(char *typeName)
{
    label_to_model[typeName] = [](int dimension, int ambient_dimension) -> std::unique_ptr<Model>
    {
        return std::make_unique<T>(dimension, ambient_dimension);
    };
}

// Function to create an object by type
std::unique_ptr<Model> createObject(char *typeName, int dimension, int ambient_dimension)
{
    auto it = label_to_model.find(typeName);
    if (it != label_to_model.end())
    {
        return it->second(dimension, ambient_dimension); // Call the factory function
    }
    else
    {
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

float calculatePointRadius()
{
    float sidePoints = std::round(std::pow(points, 1.0 / (float)d));
    float gridCellWidth = SIDE_LEN / sidePoints;
    float pointRadius = std::min(gridCellWidth / 2.0, 0.01);

    return pointRadius;
}

double r2_metric(Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred)
{
    double mean = Y_true.mean();
    double ss_tot = (Y_true.array() - mean).square().sum();
    double ss_res = (Y_true - Y_pred).squaredNorm();
    return ss_res / ss_tot;
}

void visualizeFittingPlane()
{
    // Loss is per-point and metric is overall
    auto loss_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred)
    { return Eigen::VectorXd((Y_true - Y_pred).array().square().matrix()); };
    auto metric_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred)
    { return (Y_true - Y_pred).squaredNorm() / Y_true.size(); };

    // Maybe move to global to avoid reinitialization???
    RANSAC ransac(ransac_max_iterations, ransac_threshold, ransac_train_data_percenatge, ransac_min_inliners, metric);

    Eigen::MatrixXd X = hyperplanePoints.leftCols(d);
    Eigen::VectorXd Y = hyperplanePoints.col(d);

    // std::unique_ptr<Model> singleBestModel = ransac.run(X, Y, model.get(), loss_fn, metric_fn);

    std::unique_ptr<FlatModel> averagedBestModel = ransac.run2(hyperplanePoints, (FlatModel *)model.get(), average_contributions);

    AffineFit *tmpModel = new AffineFit(2, 3);
    std::unique_ptr<FlatModel> lineAveragedBestModel = ransac.run_slow(hyperplanePoints, tmpModel, average_contributions);

    // if (singleBestModel == nullptr)
    // {
    //     std::cout << "No good Model with these parameters could be found." << std::endl;
    //     return;
    // }

    float pointRadius = calculatePointRadius();

    // FlatModel *singleBestModel_ptr = (FlatModel *)singleBestModel.get();
    FlatModel *averagedBestModel_ptr = (FlatModel *)averagedBestModel.get();
    FlatModel *lineAveragedBestModel_ptr = (FlatModel *)lineAveragedBestModel.get();

    // double singleBestModelMSE = singleBestModel_ptr->MSE(X, Y);
    double averagedBestModelMSE = averagedBestModel_ptr->MSE(X, Y);
    double lineAveragedBestModelMSE = lineAveragedBestModel_ptr->MSE(X, Y);

    // std::cout << "Single Best Model performance: " << singleBestModelMSE << std::endl;
    std::cout << "Averaged Best Model performance: " << averagedBestModelMSE << std::endl;
    std::cout << "Line-Averaged Best Flat performance: " << lineAveragedBestModelMSE << std::endl;
    // std::cout << "Single Best Model outperformed by: " << (averagedBestModelMSE - singleBestModelMSE) / (averagedBestModelMSE + singleBestModelMSE) << std::endl;
    if (d != 1 && d != 2)
        return;

    // singleBestModel_ptr->visualize("Single Best Flat", 6.0, pointRadius / 2.0, 0.6);
    averagedBestModel_ptr->visualize("Averaged Best Flat", 6.0, pointRadius / 2.0, 0.6);
    lineAveragedBestModel_ptr->visualize("Line-Averaged Best Flat", 6.0, pointRadius / 2.0, 0.6);
}

void generatePointCloud()
{
    // Compare datasets
    AffineFit *m = new AffineFit(n - 1, n);

    m->override_explicit(w, b);

    hyperplanePoints = FlatSampler::sampleFlat(*m, points, noise, outlierRatio, outlierStrength, 1.0, saltAndPepper);

    float pointRadius = calculatePointRadius();
    Visualizer::plotPoints(hyperplanePoints, "Hyperplane Point Cloud", sphereRepr ? "Sphere" : "Quad", pointRadius);

    visualizeFittingPlane();
}

void dataParameterGUI()
{
    ImGui::Text("Data Parameters");

    ImGui::SliderFloat("noise", &noise_ui, 0.0f, 0.3f, "%.4f");
    ImGui::SliderFloat("outlier fraction", &outlierRatio_ui, 0.0f, 1.0f);
    ImGui::SliderFloat("outlier strength", &outlierStrength_ui, 1.0f, 10.0f, "%.4f");
    ImGui::SliderInt("Number of Points", &points, 20, 20000);
    ImGui::SliderInt("Flat dimensions", &d, 1, 3);
    ImGui::Checkbox("Salt and Pepper Noise", &saltAndPepper);
}

void dataReprGUI()
{
    ImGui::Text("Representation");
    ImGui::Checkbox("Represent Points as Spheres? (less efficient)", &sphereRepr);
    ImGui::SliderFloat("Flat Transparency", &flatAlpha, 0.0, 1.0);
}

void ransacParameterGUI()
{
    ImGui::Text("RANSAC Hyper Parameter");
    ImGui::SliderInt("Max Iterations", &ransac_max_iterations, 100, 10000);
    ImGui::SliderFloat("Threshold", &ransac_threshold_ui, 0.0f, 0.1f, "%.6f");
    ImGui::SliderFloat("Train-Data Percentage", &ransac_train_data_percenatge_ui, 0.0f, 1.0f);
    ImGui::SliderInt("Minimum Inliners", &ransac_min_inliners, points * ransac_train_data_percenatge * 0.1, points * ransac_train_data_percenatge * 0.9);
    ImGui::SliderInt("Model Average Count", &average_contributions, 1, static_cast<int>(std::sqrt(points)));
    ImGui::Combo("Metric", (int *)&metric, "MSE\0R2\0RSS\0RMSE\0");
}

void initializeLabelToModel()
{
    registerType<OLS>("OLS");
    registerType<AffineFit>("Affine Fit");
    model = createObject("Affine Fit", d, n);
}

void modelSelectionGUI()
{
    std::vector<char *> items;
    for (auto &[key, value] : label_to_model)
    {
        items.push_back(key);
    }
    static const char *current_item = items[0];

    ImGui::Text("Model:");
    if (ImGui::BeginCombo("##combo", current_item))
    {
        for (int i = 0; i < items.size(); i++)
        {
            bool is_selected = (current_item == items[i]);
            if (ImGui::Selectable(items[i], is_selected))
            {
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

void evaluate()
{
    // 1) Prepare data for Variation 1
    //    Suppose we have NxD design matrix X and Nx1 label vector Y

    // ... fill X and Y ...

    // 2) Prepare data for Variation 2 & 3
    //    Nx(D+1), last col is label
    AffineFit *m = new AffineFit(n - 1, n);
    w = Eigen::VectorXd::Random(d);
    b = Eigen::VectorXd::Random(1)[0];
    m->override_explicit(w, b);

    Eigen::MatrixXd D = FlatSampler::sampleFlat(*m, points, noise, outlierRatio, outlierStrength, 1.0, saltAndPepper);
    // ... fill D, where D.leftCols(2) = X, D.col(2) = Y, for example ...

    // 3) Define your RANSAC param grid
    RansacParameterGrid grid;
    grid.maxIterations = {100, 1000, 10000};
    grid.thresholds = {0.0001, 0.001, 0.01, 0.1};
    grid.trainDataPercentages = {0.2, 0.3, 0.5};
    grid.minInliers = {10, 100, 1000, 10000};
    grid.bestModelCounts = {5, 10, 50};

    DataParameterGrid dataGrid;
    dataGrid.numPoints = {50, 500, 5000, 50000, 500000};
    dataGrid.subspaceDimentions = {2};
    dataGrid.ambientDimentions = {3, 5, 10, 20};
    dataGrid.noiseLevels = {0.01, 0.1, 0.3, 0.5};
    dataGrid.outlierRatios = {0.0, 0.1, 0.3, 0.5};
    dataGrid.outlierStrengths = {1.0, 3.0, 10.0};
    dataGrid.volumes = {1.0};
    dataGrid.saltAndPepper = {true, false};

    // 4) Define a factory for RANSAC with your chosen (loss_fn, metric_fn)
    std::function<RANSAC(int, double, double, int)> ransacFactory = [&](int maxIt, double thresh, double trainPct, int minInl)
    {
        // e.g. define the RANSAC constructor
        // you might have something like:
        auto loss_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred)
        { return Eigen::VectorXd((Y_true - Y_pred).array().square().matrix()); };
        auto metric_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred)
        { return (Y_true - Y_pred).squaredNorm() / Y_true.size(); };

        return RANSAC(maxIt, thresh, trainPct, minInl);
    };

    // 5) Define your Model factory (for Variation 1)
    std::function<std::unique_ptr<Model>(int, int)> modelFactory = [](int d, int n) -> std::unique_ptr<Model>
    {
        // Could be OLS or something else
        return std::make_unique<AffineFit>(d, n);
    };

    // 6) Define your FlatModel factory (for Variation 2 & 3)
    std::function<std::unique_ptr<FlatModel>(int, int)> flatModelFactory = [](int d, int n) -> std::unique_ptr<FlatModel>
    {
        // Could be AffineFit
        return std::make_unique<AffineFit>(d, n);
    };

    std::string outputCsvPath = "ransac_evaluation_results.csv";

    // 7) Run the full grid search, results go to CSV
    Evaluator::evaluateAllParamCombinations(
        grid,
        dataGrid,
        modelFactory,
        flatModelFactory,
        ransacFactory,
        outputCsvPath // saves partial results
    );

    std::cout << "Done evaluating all parameter combinations.\n";
}

int main()
{

    polyscope::init();
    // test();

    w = Eigen::VectorXd::Random(d);
    w_ui = w.cast<float>();
    b = Eigen::VectorXd::Random(1)(0);

    initializeLabelToModel();

    generatePointCloud();

    // UI for sliders
    polyscope::state::userCallback = [&]()
    {
        Generation::flatParameterGUI(w_ui, b_ui);
        modelSelectionGUI();
        dataParameterGUI();
        ransacParameterGUI();
        dataReprGUI();

        if (ImGui::Button("Regenerate Line"))
        {
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
