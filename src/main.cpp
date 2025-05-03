
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
#include "models/affine_fit.hpp"
#include "models/mean_sdf.hpp"
#include "models/median_sdf.hpp"
#include "models/huber_regression.hpp"

#include "evaluation/evaluator.hpp" 

// --- Settings -------------------------------------------------------------
constexpr int AMBIENT_DIM   = 3;
constexpr int FLAT_DIM      = 1;
constexpr int NUM_POINTS    = 200;
constexpr int NUM_CASES     = 20;
double      MAX_TILT_MAG   = 10.0;   // max slope magnitude for tilt = 100%



// --- Globals --------------------------------------------------------------
struct TestCase {
    double tiltFrac;        // Fraction of tilt (0.0 to 1.0)
    double noise;           // Noise level
    double outlierRatio;    // Ratio of outliers
    double outlierStrength; // Strength of outliers
    bool saltAndPepper;     // Whether to use salt-and-pepper noise
    int maxIterations;      // Maximum RANSAC iterations
    double threshold;       // Threshold for inlier classification
    double trainPct;        // Percentage of data used for training
    int minInliers;         // Minimum number of inliers
    int bestModelCount;     // Number of best models to average
    bool weightedAverage;   // Whether to use weighted averaging
};

static std::vector<TestCase> testCases;
static int                    currentCase = 0;
static Eigen::MatrixXd        D_inliers, D_outliers, D_all;
static std::vector<bool>      isInlierGlobal;
static Eigen::VectorXd        trueW, trueb;
static double                 trueTilt = 0.0;


static double                 meanR2Inlier = 0, meanR2All = 0;
static double                 medianR2Inlier = 0, medianR2All = 0;
static double                 huberR2Inlier = 0, huberR2All = 0;

static double                 meanMSEInlier = 0, meanMSEAll = 0;
static double                 medianMSEInlier = 0, medianMSEAll = 0;
static double                 huberMSEInlier = 0, huberMSEAll = 0;



static std::mt19937           rng(std::random_device{}());

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
}


// --- Run one test case ----------------------------------------------------
void runCurrentCase() {
    polyscope::removeAllStructures();
    const auto &tc = testCases[currentCase];

    // 1) True flat based on tilt fraction
    auto trueFlat = std::make_unique<AffineFit>(AMBIENT_DIM - 1, AMBIENT_DIM);
    Eigen::VectorXd dir = randomUnitVector(AMBIENT_DIM - 1);
    trueW = dir * (tc.tiltFrac * MAX_TILT_MAG);
    trueb = Eigen::VectorXd::Random(1) * 10.0;
    trueFlat->override_explicit(trueW, trueb);
    trueTilt = trueW.norm();

    // 2) Sample inliers/outliers
    std::tie(D_inliers, D_outliers) = FlatSampler::sampleFlatSeparated(
        *trueFlat,
        NUM_POINTS,
        tc.noise,
        tc.outlierRatio,
        tc.outlierStrength,
        tc.saltAndPepper
    );

    // 3) Merge & synchronized shuffle
    D_all.resize(D_inliers.rows() + D_outliers.rows(), D_inliers.cols());
    D_all << D_inliers, D_outliers;
    std::vector<bool> isInlierRaw(D_all.rows());
    for (int i = 0; i < (int)D_all.rows(); ++i)
        isInlierRaw[i] = (i < (int)D_inliers.rows());
    std::vector<int> perm(D_all.rows());
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = 0; i < (int)D_all.rows(); ++i) {
        int j = std::uniform_int_distribution<int>(0, D_all.rows()-1)(rng);
        std::swap(perm[i], perm[j]);
    }
    Eigen::MatrixXd D_shuf = D_all;
    isInlierGlobal.resize(D_all.rows());
    for (int i = 0; i < (int)D_all.rows(); ++i) {
        D_shuf.row(i) = D_all.row(perm[i]);
        isInlierGlobal[i] = isInlierRaw[perm[i]];
    }
    D_all = D_shuf;

    // 4) RANSAC + MeanSDF & MedianSDF
    RANSAC ransac(
        tc.maxIterations,
        tc.threshold,
        tc.trainPct,
        tc.minInliers,
        MetricType::R2
    );
    auto proto = std::make_unique<AffineFit>(FLAT_DIM, AMBIENT_DIM);
    
    // MeanSDF
    MeanSDF meanAvg(AMBIENT_DIM - 1, AMBIENT_DIM);
    auto flatMean = ransac.run_slow(
        D_all, proto.get(), tc.bestModelCount, &meanAvg, tc.weightedAverage
    );

    // MedianSDF
    MedianSDF medAvg(AMBIENT_DIM - 1, AMBIENT_DIM, 0.01, 1000);
    auto flatMed = ransac.run_slow(
        D_all, proto.get(), tc.bestModelCount, &medAvg, tc.weightedAverage
    );

    // Huber Regression
    HuberRegression huber(AMBIENT_DIM - 1, AMBIENT_DIM);
    huber.fit(D_all);

    // 5) Compute R²
    Eigen::MatrixXd Xa = D_all.leftCols(AMBIENT_DIM - 1);
    Eigen::VectorXd Ya = D_all.col(AMBIENT_DIM - 1);
    Eigen::MatrixXd Xi = D_inliers.leftCols(AMBIENT_DIM - 1);
    Eigen::VectorXd Yi = D_inliers.col(AMBIENT_DIM - 1);

    meanR2All    = flatMean ? flatMean->R2(Xa, Ya) : NAN;
    meanR2Inlier = flatMean ? flatMean->R2(Xi, Yi) : NAN;
    medianR2All    = flatMed ? flatMed->R2(Xa, Ya) : NAN;
    medianR2Inlier = flatMed ? flatMed->R2(Xi, Yi) : NAN;
    huberR2All    = huber.R2(Xa, Ya);
    huberR2Inlier = huber.R2(Xi, Yi);

    meanMSEAll    = flatMean ? flatMean->MSE(Xa, Ya) : NAN;
    meanMSEInlier = flatMean ? flatMean->MSE(Xi, Yi) : NAN;
    medianMSEAll    = flatMed ? flatMed->MSE(Xa, Ya) : NAN;
    medianMSEInlier = flatMed ? flatMed->MSE(Xi, Yi) : NAN;
    huberMSEAll    = huber.MSE(Xa, Ya);
    huberMSEInlier = huber.MSE(Xi, Yi);

    // 6) Visualize
    visualizeCurrentCase();
    if (flatMean)   flatMean->visualize("MeanSDF Plane",   6.0, 0.01, 0.6);
    if (flatMed)    flatMed->visualize("MedianSDF Plane", 6.0, 0.01, 0.6);
    huber.visualize("Huber Plane", 6.0, 0.01, 0.6);
    trueFlat->visualize("True Flat", 6.0, 0.01, 0.5);
}

// --- UI callback ----------------------------------------------------------
void uiCallback() {
    ImGui::Begin("Test Case Navigator");
    ImGui::Text("Case %d / %d", currentCase+1, (int)testCases.size());
    const auto &tc = testCases[currentCase];
    ImGui::Separator();
    ImGui::Text("Tilt frac      : %.2f%%", tc.tiltFrac*100.0);
    ImGui::Text("noise          : %.3f", tc.noise);
    ImGui::Text("outlier ratio  : %.3f", tc.outlierRatio);
    ImGui::Text("outlier str    : %.2f", tc.outlierStrength);
    ImGui::Text("salt&pepper    : %s", tc.saltAndPepper?"yes":"no");
    ImGui::Text("iters          : %d", tc.maxIterations);
    ImGui::Text("threshold      : %.5f", tc.threshold);
    ImGui::Text("trainPct       : %.3f", tc.trainPct);
    ImGui::Text("minInliers     : %d", tc.minInliers);
    ImGui::Text("bestCount      : %d", tc.bestModelCount);
    ImGui::Text("weightedAvg    : %s", tc.weightedAverage?"yes":"no");
    ImGui::Separator();
    ImGui::Text("True W (slope):");
    for (int i=0;i<trueW.size();++i) ImGui::Text(" W[%d]=%.3f",i, trueW(i));
    ImGui::Text("Tilt magnitude: %.3f", trueTilt);
    ImGui::Separator();
    ImGui::Text("[MeanSDF] R2 inlier: %.3f | all: %.3f", meanR2Inlier, meanR2All);
    ImGui::Text("[MedianSDF]R2 inlier: %.3f | all: %.3f", medianR2Inlier, medianR2All);
    ImGui::Text("[Huber]    R2 inlier: %.3f | all: %.3f", huberR2Inlier, huberR2All);
    ImGui::Text("[MeanSDF] MSE inlier: %.3f | all: %.3f", meanMSEInlier, meanMSEAll);
    ImGui::Text("[MedianSDF] MSE inlier: %.3f | all: %.3f", medianMSEInlier, medianMSEAll);
    ImGui::Text("[Huber]    MSE inlier: %.3f | all: %.3f", huberMSEInlier, huberMSEAll);
    
    ImGui::Separator();
    if (ImGui::Button("Previous") && currentCase>0) { currentCase--; runCurrentCase(); }
    ImGui::SameLine();
    if (ImGui::Button("Next")     && currentCase+1<(int)testCases.size()) { currentCase++; runCurrentCase(); }
    ImGui::End();
}



// --- main() ---------------------------------------------------------------
int main() {
    Evaluator::Grid grid;

  

    // …(fill others as desired)…

    Evaluator::Config cfg;
    cfg.csvPath = "mySweepNew2.csv";
    cfg.logPath = "mySweepNew2.log";

    // Evaluator::runGridSearch(grid, cfg);

    polyscope::init();

    // Generate test cases with tilt from 0% to 100%
    testCases.reserve(NUM_CASES);
    for (int i = 0; i < NUM_CASES; ++i) {
        double tf = (NUM_CASES>1) ? double(i)/(NUM_CASES-1) : 0.0;
        TestCase t;
        t.tiltFrac        = tf;
        t.noise           = randomDouble(0.01, 0.4);
        t.outlierRatio    = randomDouble(0.0, 0.5);
        t.outlierStrength = randomDouble(5.0, 10.0);
        t.saltAndPepper   = (rng()%2)==0;
        t.maxIterations   = 300;
        t.threshold       = 0.0001;
        t.trainPct        = 0.2;
        t.minInliers      = 50;
        t.bestModelCount  = 10;
        t.weightedAverage = true;
        testCases.push_back(t);
    }

    runCurrentCase();
    polyscope::state::userCallback = uiCallback;
    polyscope::show();
    return 0;
}
