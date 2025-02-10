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
#include "models/median_sdf.hpp"
#include "models/mean_sdf.hpp"
#include "models/huber_regression.hpp"
#include "Definitions.h"
#include "visualization/visualizer.h"
#include "evaluation/evaluator.hpp"

enum class SampleFlatRepresentation
{
    Parametric,
    Implicit,
    Count
};

SampleFlatRepresentation repr = SampleFlatRepresentation::Parametric;

Eigen::VectorXd w;
double b;
int points = 20;
double noise = 0.01;
double outlierRatio = 0.0;
double outlierStrength = 1.0f;
bool sphereRepr = false;
int k = 0; // Input Data Dimension (k = 0 represents a point)
int d = 2; // Hyperplane Dimensions
int n = 3; // Ambient Space Dimensions
int average_contributions = 10;
bool saltAndPepper = false;
MetricType metric = MetricType::MSE;

int ransac_max_iterations = 50;
double ransac_threshold = 0.0001;
double ransac_train_data_percenatge = 0.2;
int ransac_min_inliners = points * ransac_train_data_percenatge * 0.3;

std::unordered_map<char *, std::function<std::unique_ptr<FlatModel>(int, int)>> label_to_model;

std::unique_ptr<FlatModel> model;

// Temporary float variables for UI
Eigen::MatrixXf A_ui = Eigen::MatrixXf::Random(n, d);
Eigen::VectorXf b_vec_ui = Eigen::VectorXf::Random(n);
Eigen::MatrixXf N_ui = Eigen::MatrixXf::Random(n - d, n);
Eigen::VectorXf c_ui = Eigen::VectorXf::Random(n - d);

Eigen::VectorXf w_ui;
float b_ui;
float noise_ui = static_cast<float>(noise);
float outlierRatio_ui = static_cast<float>(outlierRatio);
float outlierStrength_ui = static_cast<float>(outlierStrength);
float flatAlpha = 0.7;

float ransac_threshold_ui = static_cast<float>(ransac_threshold);
float ransac_train_data_percenatge_ui = static_cast<float>(ransac_train_data_percenatge);

Eigen::MatrixXd hyperplanePoints;

// Function to register types
template <typename T>
void registerType(char *typeName)
{
    label_to_model[typeName] = [](int dimension, int ambient_dimension) -> std::unique_ptr<FlatModel>
    {
        return std::make_unique<T>(dimension, ambient_dimension);
    };
}

// Function to create an object by type
std::unique_ptr<FlatModel> createObject(char *typeName, int dimension, int ambient_dimension)
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

float calculatePointRadius()
{
    float sidePoints = std::round(std::pow(points, 1.0 / (float)d));
    float gridCellWidth = SIDE_LEN / sidePoints;
    float pointRadius = std::max(std::min(gridCellWidth / 2.0, 0.01), 0.001);

    return pointRadius;
}

double r2_metric(Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred)
{
    double mean = Y_true.mean();
    double ss_tot = (Y_true.array() - mean).square().sum();
    double ss_res = (Y_true - Y_pred).squaredNorm();
    return ss_res / ss_tot;
}

void visualizeSampleFlats(std::vector<std::unique_ptr<FlatModel>> &sampled_flats)
{
    //// TODO: If d == 0, put all `b_vec` into a point matrix and visualize them (way more efficient)
    for (int i = 0; i < sampled_flats.size(); ++i)
    {
        sampled_flats[i]->visualize("Flat " + std::to_string(i), 6.0, 0.01, flatAlpha);
    }
}

void visualizeFittingPlane()
{
    if (k > 0)
    {
        AffineFit *m = new AffineFit(d, n);

        m->override_parametric(A_ui.cast<double>(), b_vec_ui.cast<double>());
        auto sampled_flats = FlatSampler::sampleFlat2(*m, points, 1);
        for (int i = 0; i < sampled_flats.size(); ++i)
        {
            sampled_flats[i]->visualize("Flat " + std::to_string(i), 6.0, 0.005, flatAlpha);
        }

        MedianSDF flatFitter = MedianSDF(d, n, 0.01, 1000);
        flatFitter.fit(sampled_flats);
        flatFitter.visualize("Median Flat", 6.0, 0.01, flatAlpha);
    }
    else
    {

        // Loss is per-point and metric is overall
        auto loss_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred)
        { return Eigen::VectorXd((Y_true - Y_pred).array().square().matrix()); };
        auto metric_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred)
        { return (Y_true - Y_pred).squaredNorm() / Y_true.size(); };

        // Maybe move to global to avoid reinitialization???
        RANSAC ransac(ransac_max_iterations, ransac_threshold, ransac_train_data_percenatge, ransac_min_inliners, metric);

        // Eigen::MatrixXd X = hyperplanePoints.leftCols(d);
        // Eigen::VectorXd Y = hyperplanePoints.col(d);

        // std::unique_ptr<Model> singleBestModel = ransac.run(X, Y, model.get(), loss_fn, metric_fn);

        MedianSDF *averager = new MedianSDF(n - 1, n, 0.01, 1000);
        std::unique_ptr<FlatModel> averagedBestModel = ransac.run2(hyperplanePoints, (FlatModel *)model.get(), average_contributions, averager);
        averager->reset();
        std::unique_ptr<FlatModel> lineAveragedBestModel = ransac.run_slow(hyperplanePoints, (FlatModel *)model.get(), average_contributions, averager);

        HuberRegression *huber = new HuberRegression(n - 1, n);
        huber->fit(hyperplanePoints);
        // if (singleBestModel == nullptr)
        // {
        //     std::cout << "No good Model with these parameters could be found." << std::endl;
        //     return;
        // }

        // FlatModel *singleBestModel_ptr = (FlatModel *)singleBestModel.get();
        FlatModel *averagedBestModel_ptr = (FlatModel *)averagedBestModel.get();
        FlatModel *lineAveragedBestModel_ptr = (FlatModel *)lineAveragedBestModel.get();

        // double singleBestModelMSE = singleBestModel_ptr->MSE(X, Y);
        double averagedBestModelMSE = averagedBestModel_ptr->quadratic_loss(hyperplanePoints).mean();
        double lineAveragedBestModelMSE = lineAveragedBestModel_ptr->quadratic_loss(hyperplanePoints).mean();
        double huberMSE = huber->quadratic_loss(hyperplanePoints).mean();

        std::cout << "[PERFORMANCE:MSE]" << std::endl;
        std::cout << "Averaged Best Model [1]: " << averagedBestModelMSE << std::endl;
        std::cout << "Averaged Best Model [2]: " << lineAveragedBestModelMSE << std::endl;
        std::cout << "Huber Regression: " << huberMSE << std::endl;

        double averagedBestModelR2 = averagedBestModel_ptr->R2(hyperplanePoints);
        double lineAveragedBestModelR2 = lineAveragedBestModel_ptr->R2(hyperplanePoints);
        double huberR2 = huber->R2(hyperplanePoints);

        std::cout << "[PERFORMANCE:R2]" << std::endl;
        std::cout << "Averaged Best Model [1]: " << averagedBestModelR2 << std::endl;
        std::cout << "Averaged Best Model [2]: " << lineAveragedBestModelR2 << std::endl;
        std::cout << "Huber Regression: " << huberR2 << std::endl;

        std::cout << "--------------------------------" << std::endl;

        float pointRadius = calculatePointRadius();
        Visualizer::plotPoints(hyperplanePoints, "Hyperplane Point Cloud", sphereRepr ? "Sphere" : "Quad", pointRadius);

        // singleBestModel_ptr->visualize("Single Best Flat", 6.0, pointRadius / 2.0, 0.6);
        averagedBestModel_ptr->visualize("Averaged Best Flat", 6.0, pointRadius / 2.0, 0.6);
        lineAveragedBestModel_ptr->visualize("Line-Averaged Best Flat", 6.0, pointRadius / 2.0, 0.6);
        huber->visualize("Huber Flat", 6.0, pointRadius / 2.0, 0.6);
    }
}

void generatePointCloud()
{
    int subspaceNum = 1;
    int points_per_subspace = points / subspaceNum;

    // Generate points for each subspace and concatenate them together in hyperplanePoints
    hyperplanePoints = Eigen::MatrixXd::Zero(points, n);
    for (int i = 0; i < subspaceNum; ++i)
    {
        AffineFit *m = new AffineFit(d, n);
        // override with random values
        m->override_parametric(Eigen::MatrixXf::Random(n, d).cast<double>(), Eigen::VectorXf::Random(n).cast<double>());
        Eigen::MatrixXd subspacePoints = FlatSampler::sampleFlat(*m, points_per_subspace, noise, outlierRatio, outlierStrength, saltAndPepper);

        hyperplanePoints.block(i * points_per_subspace, 0, points_per_subspace, n) = subspacePoints;
    }

    visualizeFittingPlane();
}

void editDynamicMatrixGui(Eigen::MatrixXf &matrix, std::string name)
{
    // Get the available width for the matrix
    float windowWidth = ImGui::GetContentRegionAvail().x;
    float cellWidth = windowWidth / matrix.cols() - 5.0f; // Subtract some padding for spacing

    // Iterate through rows and columns
    for (int row = 0; row < matrix.rows(); ++row)
    {
        for (int col = 0; col < matrix.cols(); ++col)
        {
            // Create a unique label for each matrix element
            std::string label = name + "(" + std::to_string(row) + "," + std::to_string(col) + ")";

            // Set width for the next item
            ImGui::PushID(label.c_str());                                       // Avoid ID conflicts
            ImGui::PushItemWidth(cellWidth);                                    // Adjust cell width dynamically
            ImGui::DragFloat("", &matrix(row, col), 0.01f, 0.0f, 0.0f, "%.2f"); // Drag-only field
            ImGui::PopItemWidth();
            ImGui::PopID();

            // Align cells in the same row
            if (col < matrix.cols() - 1)
            {
                ImGui::SameLine();
            }
        }
    }
}

void editDynamicVectorGui(Eigen::VectorXf &vec, std::string name)
{
    // Get the available width for the matrix
    float windowWidth = ImGui::GetContentRegionAvail().x;
    float cellWidth = windowWidth / vec.size() - 5.0f; // Subtract some padding for spacing

    // Iterate through rows and columns

    for (int col = 0; col < vec.size(); ++col)
    {
        // Create a unique label for each matrix element
        std::string label = name + "(" + std::to_string(col) + ")";

        // Set width for the next item
        ImGui::PushID(label.c_str());                               // Avoid ID conflicts
        ImGui::PushItemWidth(cellWidth);                            // Adjust cell width dynamically
        ImGui::DragFloat("", &vec(col), 0.01f, 0.0f, 0.0f, "%.2f"); // Drag-only field
        ImGui::PopItemWidth();
        ImGui::PopID();

        // Align cells in the same row
        if (col < vec.size() - 1)
        {
            ImGui::SameLine();
        }
    }
}

void parametricRepresentationGUI()
{
    ImGui::Text("Parametric Representation");
    ImGui::Text("A");
    editDynamicMatrixGui(A_ui, "A");
    ImGui::Text("b_vec");
    editDynamicVectorGui(b_vec_ui, "b_vec");

    // ImGui::DragFloat("b_vec", &b_vec_ui, 0.01f, 0.0f, 0.0f, "%.2f"); // Drag-only field
}

void implicitRepresentationGUI()
{
    ImGui::Text("Implicit Representation");
    ImGui::Text("N");
    editDynamicMatrixGui(N_ui, "N");
    ImGui::Text("c");
    editDynamicVectorGui(c_ui, "c");
}

void flatParameterGUI()
{
    const char *representationNames[] = {
        "Parametric",
        "Implicit"};

    ImGui::Text("Flat Parameters");

    int currentIndex = static_cast<int>(repr);
    if (ImGui::Combo("Sample Flat Representation", &currentIndex, representationNames, static_cast<int>(SampleFlatRepresentation::Count)))
    {
        repr = static_cast<SampleFlatRepresentation>(currentIndex);
    }

    switch (repr)
    {
    case SampleFlatRepresentation::Parametric:
        parametricRepresentationGUI();
        break;
    case SampleFlatRepresentation::Implicit:
        implicitRepresentationGUI();
        break;
    default:
        break;
    }

    // for (int i = 0; i < w_ui.size(); ++i)
    // {
    //     ImGui::SliderFloat(std::format("w[{}]", i).c_str(), &w_ui[i], -5.0f, 5.0f);
    // }

    // ImGui::SliderFloat("b", &b_ui, -10.0f, 10.0f);
}

void dataParameterGUI()
{
    ImGui::Text("Data Parameters");

    ImGui::SliderFloat("noise", &noise_ui, 0.0f, 0.3f, "%.4f");
    ImGui::SliderFloat("outlier fraction", &outlierRatio_ui, 0.0f, 1.0f);
    ImGui::SliderFloat("outlier strength", &outlierStrength_ui, 1.0f, 10.0f, "%.4f");
    ImGui::SliderInt("Number of Points", &points, 20, 20000);
    ImGui::SliderInt("Ambient Dimension", &n, 2, 10);
    ImGui::SliderInt("Fitting Flat Dimension", &d, 1, n - 1);
    ImGui::SliderInt("Input Flat Dimension", &k, 0, d - 1);
    ImGui::Checkbox("Salt and Pepper Noise", &saltAndPepper);

    d = std::min(d, n - 1);
    k = std::min(k, d - 1);

    int last_n = A_ui.rows();
    int last_d = A_ui.cols();

    if (last_n == n && last_d == d)
    {
        return;
    }

    // Conservative Resive of all UI Matrices and Vectors
    A_ui.conservativeResize(n, d);
    b_vec_ui.conservativeResize(n);
    N_ui.conservativeResize(n - d, n);
    c_ui.conservativeResize(n - d);

    if (last_n >= n && last_d >= d)
    {
        return;
    }

    // Set the new values to random
    A_ui.block(0, last_d, n, d - last_d).setRandom();
    A_ui.block(last_n, 0, n - last_n, d).setRandom();
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

    Eigen::MatrixXd D = FlatSampler::sampleFlat(*m, points, noise, outlierRatio, outlierStrength, saltAndPepper);

    RansacParameterGrid grid;
    grid.maxIterations = {100};
    grid.thresholds = {0.1};
    grid.trainDataPercentages = {0.2};
    grid.minInliers = {100};
    grid.bestModelCounts = {1, 500};
    grid.metrics = {MetricType::R2};
    grid.weightedAverages = {false};

    DataParameterGrid dataGrid;
    dataGrid.numPoints = {500};
    dataGrid.subspaceDimentions = {1, 2, 4, 9, 19, 49};
    dataGrid.ambientDimentions = {2, 3, 5, 10, 20, 50};
    dataGrid.noiseLevels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
    dataGrid.outlierRatios = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
    dataGrid.outlierStrengths = {2.5};
    dataGrid.saltAndPepper = {true, false};

    // Am ende immer zu hyperebene
    // also d=1,..,n-1 zu hyperebene immer
    // r2 gegen n visualisieren
    // ab wann wird schlechter
    //

    std::function<RANSAC(int, double, double, int, MetricType)> ransacFactory = [&](int maxIt, double thresh, double trainPct, int minInl, MetricType metric) -> RANSAC
    {
        // e.g. define the RANSAC constructor
        // you might have something like:
        auto loss_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred)
        { return Eigen::VectorXd((Y_true - Y_pred).array().square().matrix()); };
        auto metric_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred)
        { return (Y_true - Y_pred).squaredNorm() / Y_true.size(); };

        return RANSAC(maxIt, thresh, trainPct, minInl, metric);
    };

    // 5) Define your Model factory (for Variation 1)
    std::function<std::unique_ptr<FlatModel>(int, int)> modelFactory = [](int d, int n) -> std::unique_ptr<FlatModel>
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
    evaluate();
    // polyscope::init();

    // initializeLabelToModel();
    // model->override_parametric(A_ui.cast<double>(), b_vec_ui.cast<double>());
    // generatePointCloud();

    // // UI for sliders
    // polyscope::state::userCallback = [&]()
    // {
    //     flatParameterGUI();
    //     modelSelectionGUI();
    //     dataParameterGUI();
    //     ransacParameterGUI();
    //     dataReprGUI();

    //     if (ImGui::Button("Regenerate Line"))
    //     {
    //         polyscope::removeAllStructures();
    //         w_ui.conservativeResize(d);
    //         w.conservativeResize(d);
    //         w = w_ui.cast<double>();
    //         b = static_cast<double>(b_ui);
    //         noise = static_cast<double>(noise_ui);
    //         outlierRatio = static_cast<double>(outlierRatio_ui);
    //         outlierStrength = static_cast<double>(outlierStrength_ui);

    //         ransac_threshold = static_cast<double>(ransac_threshold_ui);
    //         ransac_train_data_percenatge = static_cast<double>(ransac_train_data_percenatge_ui);

    //         d = std::min(d, n - 1);
    //         k = std::min(k, d - 1);

    //         Eigen::MatrixXd A_double;
    //         switch (repr)
    //         {
    //         case SampleFlatRepresentation::Parametric:
    //             model->override_parametric(A_ui.cast<double>(), b_vec_ui.cast<double>());
    //             break;
    //         case SampleFlatRepresentation::Implicit:
    //             model->override_implicit(N_ui.cast<double>(), c_ui.cast<double>());
    //             break;
    //         default:
    //             break;
    //         }

    //         generatePointCloud();
    //     }
    // };

    // polyscope::show();

    // return 0;
}
