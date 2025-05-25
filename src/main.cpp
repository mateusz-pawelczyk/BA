
#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <string>
#include <numeric>
#include <iostream>

#include "core/flat_sampler.hpp"
#include "core/ransac.hpp"
#include "core/normalization_utils.hpp"
#include "models/affine_fit.hpp"
#include "models/mean_sdf.hpp"
#include "models/median_sdf.hpp"
#include "models/huber_regression.hpp"

#include "evaluation/evaluator.hpp" 

enum class SampleFlatRepresentation
{
    Parametric,
    Implicit,
    Explicit,
    Count
};


struct FlatParams {
    SampleFlatRepresentation repr; // Representation of the flat
    Eigen::MatrixXd A;           // Parametric representation
    Eigen::VectorXd b_vec;           // Parametric representation
    Eigen::MatrixXd N;           // Implicit representation
    Eigen::VectorXd c;           // Implicit representation
    Eigen::MatrixXd W;           // Explicit representation
    Eigen::VectorXd b;       // Explicit representation
};

struct DataParams {
    int n;                  // Number of ambient dimensions
    int d;                  // Dimension of the fitting flat
    int    numPoints;        // Number of points
    double noise;           // Noise level
    double outlierRatio;    // Ratio of outliers
    double outlierStrength; // Strength of outliers
    bool   saltAndPepper;   // Whether to use salt-and-pepper noise
};

struct RansacParams {
    int    maxIterations;      // Maximum RANSAC iterations
    double threshold;         // Threshold for inlier classification
    double trainPct;          // Percentage of data used for training
    int    minInliers;        // Minimum number of inliers
    int    bestModelCount;    // Number of best models to average
    bool   weightedAverage;   // Whether to use weighted averaging
};
struct MetricParams {
    MetricType metric;        // Metric type
    DistanceType distance;    // Distance type
    double     huberDelta;   // Huber delta
};

struct UIParams {
    float noise;           // Noise level
    float outlierRatio;    // Ratio of outliers
    float outlierStrength; // Strength of outliers

    float ransacThreshold; // RANSAC threshold
    float ransacTrainPct; // Percentage of data used for training
    int   ransacMaxIterations; // Maximum RANSAC iterations

    float averagerErrTolerance; // Error tolerance for MedianSDF

    Eigen::MatrixXf A_ui; // UI representation of A
    Eigen::VectorXf b_vec_ui; // UI representation of b_vec
    Eigen::MatrixXf N_ui; // UI representation of N
    Eigen::VectorXf c_ui; // UI representation of c
    Eigen::MatrixXf W_ui; // UI representation of W
    Eigen::VectorXf b_ui; // UI representation of b
    float flatAlpha;      // Alpha value for flat visualization
    bool  sphereRepr;     // Whether to use sphere representation
};

struct AveragerParams {
    int     modelCount; // Number of models to average
    bool    weightedAverage; // Whether to use weighted averaging
    double  errTolerance; // Error tolerance for MedianSDF
    int     iterations; // Number of iterations for MedianSDF
};

struct GeneratedFlats {
    std::unique_ptr<FlatModel> meanSDF_Flat;
    std::unique_ptr<FlatModel> medianSDF_Flat;
    std::unique_ptr<FlatModel> huber_Flat;
};

struct MetricEvaluation {
    double inlier;
    double all;
};

struct MetricSet  {
    MetricEvaluation r2;
    MetricEvaluation mse;
};

struct DistanceEvaluation {
    MetricSet orthogonal;
    MetricSet vertical;
};

struct MethodEvaluation {
    DistanceEvaluation meanSDF;
    DistanceEvaluation medianSDF;
    DistanceEvaluation huber;
};

MethodEvaluation methodEval;
DataParams dataParams;
RansacParams ransacParams;
MetricParams metricParams;
UIParams uiParams;
FlatParams flatParams;
AveragerParams averagerParams;
GeneratedFlats generatedFlats;

std::unique_ptr<FlatModel> trueFlat;

static Eigen::MatrixXd        D_inliers, D_outliers, D_all;
static std::vector<bool>      isInlierGlobal;

static std::random_device       rd;
static std::mt19937           rng(rd());

// --- Helpers --------------------------------------------------------------
// Random double in [minVal, maxVal]
double randomDouble(double minVal, double maxVal) {
    std::uniform_real_distribution<double> dist(minVal, maxVal);
    return dist(rng);
}
// Random unit vector in R^dim
Eigen::VectorXd randomUnitVector(int dim) {
    Eigen::VectorXd v(dim);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < dim; ++i) v(i) = dist(rng);
    v.normalize();
    return v;
}

#include "polyscope/curve_network.h"

void visualizeCurrentCase() {
    auto ps = polyscope::registerPointCloud("All Points", D_all);
    ps->setPointRadius(0.005);
    std::vector<glm::vec3> colors(D_all.rows());
    for (int i = 0; i < (int)D_all.rows(); ++i) {
        colors[i] = isInlierGlobal[i]
            ? glm::vec3(0.0f, 0.5f, 1.0f)
            : glm::vec3(1.0f, 0.1f, 0.1f);
    }
    ps->addColorQuantity("Inlier/Outlier", colors)->setEnabled(true);

    // === Draw coordinate axes manually (CORRECT for your polyscope version) ===
    std::vector<glm::vec3> nodes = {
        {0.0f, 0.0f, 0.0f},
        {5.0f, 0.0f, 0.0f},  // X axis
        {0.0f, 0.0f, 0.0f},
        {0.0f, 5.0f, 0.0f},  // Y axis
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 5.0f}   // Z axis
    };
    std::vector<std::array<size_t, 2>> edges = {
        {0, 1}, // X
        {2, 3}, // Y
        {4, 5}  // Z
    };

    polyscope::CurveNetwork* axis = polyscope::registerCurveNetwork("CoordinateAxes", nodes, edges);
    axis->setRadius(0.005);

    // Per-edge color
    std::vector<glm::vec3> edgeColors = {
        {0.6f, 0.6f, 0.6f}, // X = gray
        {0.6f, 0.6f, 0.6f}, // Y = gray
        {1.0f, 0.0f, 0.0f}  // Z = red (dependent variable)
    };
    axis->addEdgeColorQuantity("Axis Colors", edgeColors)->setEnabled(true);

    // Visualize the flats
    if (generatedFlats.meanSDF_Flat)      generatedFlats.meanSDF_Flat->visualize("MeanSDF Plane",   6.0, 0.01, 0.6);
    if (generatedFlats.medianSDF_Flat)    generatedFlats.medianSDF_Flat->visualize("MedianSDF Plane", 6.0, 0.01, 0.6);
    if (generatedFlats.huber_Flat)         generatedFlats.huber_Flat->visualize("Huber Plane", 6.0, 0.01, 0.6);
    trueFlat->visualize("True Flat", 6.0, 0.01, 0.5);
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
    editDynamicMatrixGui(uiParams.A_ui, "A");
    ImGui::Text("b_vec");
    editDynamicVectorGui(uiParams.b_vec_ui, "b_vec");
}

void implicitRepresentationGUI()
{
    ImGui::Text("Implicit Representation");
    ImGui::Text("N");
    editDynamicMatrixGui(uiParams.N_ui, "N");
    ImGui::Text("c");
    editDynamicVectorGui(uiParams.c_ui, "c");
}

void explicitRepresentationGUI()
{
    ImGui::Text("Explicit Representation");
    ImGui::Text("W");
    editDynamicMatrixGui(uiParams.W_ui, "W");
    ImGui::Text("b");
    editDynamicVectorGui(uiParams.b_ui, "b");
}

void flatParameterGUI()
{
    const char *representationNames[] = {
        "Parametric",
        "Implicit",
        "Explicit"};

    ImGui::Text("Flat Parameters");

    int currentIndex = static_cast<int>(flatParams.repr);
    if (ImGui::Combo("Sample Flat Representation", &currentIndex, representationNames, static_cast<int>(SampleFlatRepresentation::Count)))
    {
        flatParams.repr = static_cast<SampleFlatRepresentation>(currentIndex);
    }

    switch (flatParams.repr)
    {
    case SampleFlatRepresentation::Parametric:
        parametricRepresentationGUI();
        break;
    case SampleFlatRepresentation::Implicit:
        implicitRepresentationGUI();
        break;
    case SampleFlatRepresentation::Explicit:
        explicitRepresentationGUI();
        break;
    default:
        break;
    }

}

void dataParameterGUI()
{
    ImGui::Text("Data Parameters");

    ImGui::SliderFloat("noise", &uiParams.noise, 0.0f, 0.5f, "%.4f");
    ImGui::SliderFloat("outlier fraction", &uiParams.outlierRatio, 0.0f, 1.0f);
    ImGui::SliderFloat("outlier strength", &uiParams.outlierStrength, 1.0f, 100.0f, "%.4f");
    ImGui::SliderInt("Number of Points", &dataParams.numPoints, 20, 20000);
    ImGui::SliderInt("Ambient Dimension", &dataParams.n, 2, 10);
    ImGui::SliderInt("Input Flat Dimension", &dataParams.d, 1, dataParams.n - 1);
    ImGui::Checkbox("Salt and Pepper Noise", &dataParams.saltAndPepper);

    dataParams.d = std::min(dataParams.d, dataParams.n - 1);

    int n = dataParams.n;
    int d = dataParams.d;

    int last_n = uiParams.A_ui.rows();
    int last_d = uiParams.A_ui.cols();

    if (last_n == n && last_d == d)
    {
        return;
    }

    // Conservative Resize of all UI Matrices and Vectors
    uiParams.A_ui.conservativeResize(n, d);
    uiParams.b_vec_ui.conservativeResize(n);
    uiParams.N_ui.conservativeResize(n - d, n);
    uiParams.c_ui.conservativeResize(n - d);
    uiParams.W_ui.conservativeResize(d, n - d);
    uiParams.b_ui.conservativeResize(n - d);

    if (last_n >= n && last_d >= d)
    {
        return;
    }

    // Set the new values to random
    uiParams.A_ui.block(0, last_d, n, d - last_d).setRandom();
    uiParams.A_ui.block(last_n, 0, n - last_n, d).setRandom();
}

void dataReprGUI()
{
    ImGui::Text("Representation");
    ImGui::Checkbox("Represent Points as Spheres? (less efficient)", &uiParams.sphereRepr);
    ImGui::SliderFloat("Flat Transparency", &uiParams.flatAlpha, 0.0, 1.0);
}

void ransacParameterGUI()
{
    ImGui::Text("RANSAC Hyper Parameter");
    ImGui::SliderInt("Max Iterations", &ransacParams.maxIterations, 100, 10000);
    ImGui::SliderFloat("Threshold", &uiParams.ransacThreshold, 0.0f, 0.1f, "%.6f");
    ImGui::SliderFloat("Train-Data Percentage", &uiParams.ransacTrainPct, 0.0f, 1.0f);
    ImGui::SliderInt("Minimum Inliners", &ransacParams.minInliers, dataParams.numPoints * uiParams.ransacTrainPct * 0.1, dataParams.numPoints * uiParams.ransacTrainPct * 0.9);
    ImGui::SliderInt("Model Average Count", &averagerParams.modelCount, 1, static_cast<int>(std::sqrt(dataParams.numPoints)));
    // Metric selection
    ImGui::Combo("Metric", reinterpret_cast<int*>(&metricParams.metric), "R2\0MSE\0");
    // Distance selection
    ImGui::Combo("Distance", reinterpret_cast<int*>(&metricParams.distance), "Regression\0Orthogonal\0");
}

void evaluationGUI() {
    ImGui::Separator();
    ImGui::Text("Evaluation Results");
    const char* metricLabel = (metricParams.metric == MetricType::R2 ? "R²" : "MSE");
    const char* distanceLabel = (metricParams.distance == DistanceType::Orthogonal ? "Orthogonal" : "Regression");
    ImGui::Text("Using %s on %s distance", metricLabel, distanceLabel);

    // Helper to select the appropriate MetricSet
    auto selectMetricSet = [&](const DistanceEvaluation &distEval) -> const MetricSet& {
        return (metricParams.distance == DistanceType::Orthogonal ? distEval.orthogonal : distEval.vertical);
    };

    // MeanSDF evaluation
    {
        const auto &ms = selectMetricSet(methodEval.meanSDF);
        const MetricEvaluation &me = (metricParams.metric == MetricType::R2 ? ms.r2 : ms.mse);
        ImGui::Text("MeanSDF - Inliers: %.4f, All: %.4f", me.inlier, me.all);
    }
    // MedianSDF evaluation
    {
        const auto &ms = selectMetricSet(methodEval.medianSDF);
        const MetricEvaluation &me = (metricParams.metric == MetricType::R2 ? ms.r2 : ms.mse);
        ImGui::Text("MedianSDF - Inliers: %.4f, All: %.4f", me.inlier, me.all);
    }
    // HuberRegression evaluation
    {
        const auto &ms = selectMetricSet(methodEval.huber);
        const MetricEvaluation &me = (metricParams.metric == MetricType::R2 ? ms.r2 : ms.mse);
        ImGui::Text("HuberRegression - Inliers: %.4f, All: %.4f", me.inlier, me.all);
    }
    ImGui::Separator();
}

void mergeAndShuffle()
{
    const int numInliers = D_inliers.rows();
    const int numOutliers = D_outliers.rows();
    const int totalRows = numInliers + numOutliers;
    const int numCols = D_inliers.cols();

    // Step 1: Merge inliers and outliers into D_all
    D_all.resize(totalRows, numCols);
    D_all << D_inliers, D_outliers;

    // Step 2: Create raw inlier labels (true for inliers, false for outliers)
    std::vector<bool> isInlierRaw(totalRows);
    for (int i = 0; i < totalRows; ++i) {
        isInlierRaw[i] = (i < numInliers);
    }

    // Step 3: Generate a random permutation of row indices
    std::vector<int> perm(totalRows);
    std::iota(perm.begin(), perm.end(), 0);  // fill with 0, 1, 2, ..., totalRows-1

    for (int i = 0; i < totalRows; ++i) {
        int j = std::uniform_int_distribution<int>(0, totalRows - 1)(rng);
        std::swap(perm[i], perm[j]);
    }

    // Step 4: Apply permutation to shuffle data and inlier labels
    Eigen::MatrixXd D_shuffled = D_all;
    isInlierGlobal.resize(totalRows);

    for (int i = 0; i < totalRows; ++i) {
        D_shuffled.row(i) = D_all.row(perm[i]);
        isInlierGlobal[i] = isInlierRaw[perm[i]];
    }

    // Step 5: Update D_all with shuffled data
    D_all = D_shuffled;

}

void generatePointCloud()
{  
    AffineFit *m =  (AffineFit *)trueFlat.get();

    // override with random values
    auto [D_inlier, D_outlier] = FlatSampler::sampleFlatSeparated(
        *m, 
        dataParams.numPoints, 
        dataParams.noise, 
        dataParams.outlierRatio, 
        dataParams.outlierStrength, 
        dataParams.saltAndPepper
    );

    D_inliers = D_inlier;
    D_outliers = D_outlier;

    std::random_device rd;
    std::mt19937 gen(rd());

    mergeAndShuffle();
}

#include <chrono> // Required for timing

void fitFlats() {
    // Fit flats using RANSAC
    RANSAC ransac(
        ransacParams.maxIterations,
        ransacParams.threshold,
        ransacParams.trainPct,
        ransacParams.minInliers,
        metricParams.metric,
        metricParams.distance
    );

    int n = dataParams.n;
    int d = dataParams.d;

    auto proto = std::make_unique<AffineFit>(d, n);

    std::cout << "--- Fitting Models ---" << std::endl;

        // MeanSDF
        MeanSDF meanAvg(n - 1, n);
        std::cout << "MeanSDF:" << std::endl;

        auto start_slow_mean = std::chrono::high_resolution_clock::now();
        auto meanSDF_slow_result = ransac.run_slow(
            D_all, proto.get(), averagerParams.modelCount, &meanAvg, averagerParams.weightedAverage
        );
        auto end_slow_mean = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_slow_mean = end_slow_mean - start_slow_mean;
        std::cout << "  run_slow time: " << duration_slow_mean.count() << " ms";
        if (meanSDF_slow_result) {
            double mse_slow_mean = meanSDF_slow_result->MSE(D_inliers);
            std::cout << ", MSE (Orthogonal, Inliers): " << mse_slow_mean;
        }
        std::cout << std::endl;

        auto start_fast_mean = std::chrono::high_resolution_clock::now();
        generatedFlats.meanSDF_Flat = ransac.run_fast(
            D_all, proto.get(), averagerParams.modelCount, &meanAvg, averagerParams.weightedAverage
        );
        auto end_fast_mean = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_fast_mean = end_fast_mean - start_fast_mean;
        std::cout << "  run_fast time: " << duration_fast_mean.count() << " ms";
        if (generatedFlats.meanSDF_Flat) {
            double mse_fast_mean = generatedFlats.meanSDF_Flat->MSE(D_inliers);
            std::cout << ", MSE (Orthogonal, Inliers): " << mse_fast_mean;
        }
        std::cout << std::endl;


        // MedianSDF
        MedianSDF medAvg(n - 1, n, averagerParams.errTolerance, averagerParams.iterations);
        std::cout << "MedianSDF:" << std::endl;

        auto start_slow_median = std::chrono::high_resolution_clock::now();
        auto medianSDF_slow_result = ransac.run_slow(
            D_all, proto.get(), averagerParams.modelCount, &medAvg, averagerParams.weightedAverage
        );
        auto end_slow_median = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_slow_median = end_slow_median - start_slow_median;
        std::cout << "  run_slow time: " << duration_slow_median.count() << " ms";
        if (medianSDF_slow_result) {
            double mse_slow_median = medianSDF_slow_result->MSE(D_all);
            std::cout << ", MSE (Orthogonal, All): " << mse_slow_median;
        }
        std::cout << std::endl;

        auto start_fast_median = std::chrono::high_resolution_clock::now();
        generatedFlats.medianSDF_Flat = ransac.run_fast(
            D_all, proto.get(), averagerParams.modelCount, &medAvg, averagerParams.weightedAverage
        );
        auto end_fast_median = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_fast_median = end_fast_median - start_fast_median;
        std::cout << "  run_fast time: " << duration_fast_median.count() << " ms";
        if (generatedFlats.medianSDF_Flat) {
            double mse_fast_median = generatedFlats.medianSDF_Flat->MSE(D_all);
            std::cout << ", MSE (Orthogonal, All): " << mse_fast_median;
        }
        std::cout << std::endl;


    // Huber Regression
    // This part does not use ransac.run_fast or ransac.run_slow, so it remains unchanged
    // unless you also want to time its 'fit' method.
    std::cout << "Huber Regression:" << std::endl;
    auto start_huber = std::chrono::high_resolution_clock::now();
    HuberRegression huber(n - 1, n, metricParams.distance);
    huber.fit(D_all);
    generatedFlats.huber_Flat = std::make_unique<HuberRegression>(std::move(huber));
    auto end_huber = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_huber = end_huber - start_huber;
    std::cout << "  fit time: " << duration_huber.count() << " ms" << std::endl;

    std::cout << "--- Fitting Complete ---" << std::endl;
}

void evaluateFlats() {
    // Evaluate flats using R² and MSE
    Eigen::MatrixXd Xa = D_all.leftCols(dataParams.d);
    Eigen::VectorXd Ya = D_all.col(dataParams.n - 1);
    Eigen::MatrixXd Xi = D_inliers.leftCols(dataParams.d);
    Eigen::VectorXd Yi = D_inliers.col(dataParams.n - 1);

    // MeanSDF evaluation
    methodEval.meanSDF.orthogonal.r2.inlier = generatedFlats.meanSDF_Flat->R2(D_inliers);
    methodEval.meanSDF.orthogonal.r2.all = generatedFlats.meanSDF_Flat->R2(D_all);
    methodEval.meanSDF.orthogonal.mse.inlier = generatedFlats.meanSDF_Flat->MSE(D_inliers);
    methodEval.meanSDF.orthogonal.mse.all = generatedFlats.meanSDF_Flat->MSE(D_all);

    methodEval.meanSDF.vertical.r2.inlier = generatedFlats.meanSDF_Flat->R2(Xi, Yi);
    methodEval.meanSDF.vertical.r2.all = generatedFlats.meanSDF_Flat->R2(Xa, Ya);
    methodEval.meanSDF.vertical.mse.inlier = generatedFlats.meanSDF_Flat->MSE(Xi, Yi);
    methodEval.meanSDF.vertical.mse.all = generatedFlats.meanSDF_Flat->MSE(Xa, Ya);

    // MedianSDF evaluation
    methodEval.medianSDF.orthogonal.r2.inlier = generatedFlats.medianSDF_Flat->R2(D_inliers);
    methodEval.medianSDF.orthogonal.r2.all = generatedFlats.medianSDF_Flat->R2(D_all);
    methodEval.medianSDF.orthogonal.mse.inlier = generatedFlats.medianSDF_Flat->MSE(D_inliers);
    methodEval.medianSDF.orthogonal.mse.all = generatedFlats.medianSDF_Flat->MSE(D_all);

    methodEval.medianSDF.vertical.r2.inlier = generatedFlats.medianSDF_Flat->R2(Xi, Yi);
    methodEval.medianSDF.vertical.r2.all = generatedFlats.medianSDF_Flat->R2(Xa, Ya);
    methodEval.medianSDF.vertical.mse.inlier = generatedFlats.medianSDF_Flat->MSE(Xi, Yi);
    methodEval.medianSDF.vertical.mse.all = generatedFlats.medianSDF_Flat->MSE(Xa, Ya);

    // Huber evaluation
    methodEval.huber.orthogonal.r2.inlier = generatedFlats.huber_Flat->R2(D_inliers);
    methodEval.huber.orthogonal.r2.all = generatedFlats.huber_Flat->R2(D_all);
    methodEval.huber.orthogonal.mse.inlier = generatedFlats.huber_Flat->MSE(D_inliers);
    methodEval.huber.orthogonal.mse.all = generatedFlats.huber_Flat->MSE(D_all);

    methodEval.huber.vertical.r2.inlier = generatedFlats.huber_Flat->R2(Xi, Yi);
    methodEval.huber.vertical.r2.all = generatedFlats.huber_Flat->R2(Xa, Ya);
    methodEval.huber.vertical.mse.inlier = generatedFlats.huber_Flat->MSE(Xi, Yi);
    methodEval.huber.vertical.mse.all = generatedFlats.huber_Flat->MSE(Xa, Ya);
}

void runCurrentCase () {
    // Debug print after each function call
    generatePointCloud();
    fitFlats();
    evaluateFlats();
    visualizeCurrentCase();
}

void ui() {
    polyscope::state::userCallback = [&]()
    {
        flatParameterGUI();
        dataParameterGUI();
        ransacParameterGUI();
        dataReprGUI();

        if (ImGui::Button("Regenerate Line"))
        {
            polyscope::removeAllStructures();

            // Assign values
            dataParams.noise = static_cast<double>(uiParams.noise);
            dataParams.outlierRatio = static_cast<double>(uiParams.outlierRatio);
            dataParams.outlierStrength = static_cast<double>(uiParams.outlierStrength);

            ransacParams.threshold = static_cast<double>(uiParams.ransacThreshold);
            ransacParams.trainPct = static_cast<double>(uiParams.ransacTrainPct);

            dataParams.d = std::min(dataParams.d, dataParams.n - 1);

            Eigen::MatrixXd A_double;
            switch (flatParams.repr)
            {
            case SampleFlatRepresentation::Parametric:
                trueFlat->override_parametric(uiParams.A_ui.cast<double>(), uiParams.b_vec_ui.cast<double>());
                break;
            case SampleFlatRepresentation::Implicit:
                trueFlat->override_implicit(uiParams.N_ui.cast<double>(), uiParams.c_ui.cast<double>());
                break;
            case SampleFlatRepresentation::Explicit:
                trueFlat->override_explicit(uiParams.W_ui.cast<double>(), uiParams.b_ui.cast<double>());
                break;
            default:
                break;
            }

            runCurrentCase();
        }
        evaluationGUI();
    };
}

void initializeParameters() {
    // Initialize parameters
    dataParams.n = 3;
    dataParams.d = 2;
    dataParams.numPoints = 1000;
    dataParams.noise = 0.1;
    dataParams.outlierRatio = 0.1;
    dataParams.outlierStrength = 10.0;
    dataParams.saltAndPepper = false;

    ransacParams.maxIterations = 500;
    ransacParams.threshold = 0.001;
    ransacParams.trainPct = 0.5;
    ransacParams.minInliers = 50;
    ransacParams.bestModelCount = 10;
    ransacParams.weightedAverage = true;

    averagerParams.modelCount = 10;
    averagerParams.weightedAverage = true;
    averagerParams.errTolerance = 0.01;
    averagerParams.iterations = 1000;

    metricParams.metric = MetricType::R2;
    metricParams.distance = DistanceType::Orthogonal;
    metricParams.huberDelta = 0.1;

    uiParams.flatAlpha = 0.5f;
    uiParams.sphereRepr = true;
    uiParams.noise = static_cast<float>(dataParams.noise);
    uiParams.outlierRatio = static_cast<float>(dataParams.outlierRatio);
    uiParams.outlierStrength = static_cast<float>(dataParams.outlierStrength);
    uiParams.ransacThreshold = static_cast<float>(ransacParams.threshold);
    uiParams.ransacTrainPct = static_cast<float>(ransacParams.trainPct);
    uiParams.ransacMaxIterations = ransacParams.maxIterations;
    uiParams.A_ui = Eigen::MatrixXf::Random(dataParams.n, dataParams.d);
    uiParams.b_vec_ui = Eigen::VectorXf::Random(dataParams.n);
    uiParams.N_ui = Eigen::MatrixXf::Random(dataParams.n - dataParams.d, dataParams.n);
    uiParams.c_ui = Eigen::VectorXf::Random(dataParams.n - dataParams.d);
    uiParams.W_ui = Eigen::MatrixXf::Random(dataParams.d, dataParams.n - dataParams.d);
    uiParams.b_ui = Eigen::VectorXf::Random(dataParams.n - dataParams.d);

    flatParams.A = static_cast<Eigen::MatrixXd>(uiParams.A_ui.cast<double>());
    flatParams.b_vec = static_cast<Eigen::VectorXd>(uiParams.b_vec_ui.cast<double>());
    uiParams.N_ui = Eigen::MatrixXf::Random(dataParams.n - dataParams.d, dataParams.n);
    uiParams.c_ui = Eigen::VectorXf::Random(dataParams.n - dataParams.d);
    uiParams.W_ui = Eigen::MatrixXf::Random(dataParams.d, dataParams.n - dataParams.d);
    uiParams.b_ui = Eigen::VectorXf::Random(dataParams.n - dataParams.d);
    
    trueFlat = std::make_unique<AffineFit>(dataParams.n - 1, dataParams.n);
    trueFlat->override_parametric(flatParams.A, flatParams.b_vec);


}

// --- main() ---------------------------------------------------------------
int main() {
    // Evaluator::Grid grid;

    // Evaluator::Config cfg;
    // cfg.csvPath = "evaluationBetter01.csv";
    // cfg.logPath = "evaluationBetter01.log";
    // std::cout << "Starting grid search...\n";
    // Evaluator::runGridSearch(grid, cfg);
    polyscope::init();
    initializeParameters();
    runCurrentCase();
    ui();
    polyscope::show();
    return 0;
}
