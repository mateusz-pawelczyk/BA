#pragma once
// ───────────────────────────────────────────────────────────────────────────────
//   Evaluator – simple, self-contained grid-search runner for your project
//   * Generates synthetic data for each parameter combo
//   * Runs RANSAC+MeanSDF   and   RANSAC+MedianSDF
//   * Computes only two metrics you care about:
//       – r2_regression       (all points)
//       – r2_regression_inlier(inliers only)
//   * Streams results to     <csvPath>
//   * Detailed step log to   <logPath>
// ───────────────────────────────────────────────────────────────────────────────
#include <Eigen/Core>
#include <string>
#include <vector>
#include <fstream>
#include <random>  
#include <core/types.hpp>

// ── Grid of independent parameters ────────────────────────────────────────────
namespace Evaluator {

struct Grid {
    // ─ data generation ─
    std::vector<int>    numPoints      {200, 500, 1000, 2000, 5000};
    std::vector<int>    ambientDims    {2, 4, 6, 8, 10};
    std::vector<double> noiseLevels    {0.1, 0.3, 0.6, 0.8, 1.1};
    std::vector<double> outlierRatios  {0.1, 0.2, 0.3, 0.4};
    std::vector<double> outlierStr     {20.0, 100.0};
    std::vector<bool>   saltPepper     {true, false};
    std::vector<double> tiltFractions  {1.0}; // 0-100 %

    // ─ RANSAC ─
    std::vector<int>    maxIterations  {300};
    std::vector<double> thresholds     {1e-2};
    std::vector<double> trainPcts      {0.2, 0.6, 1.0};
    std::vector<double> minInliers     {0.3, 0.5, 0.95};
    std::vector<int>    bestModelCnt   {1, 10, 50, 1000};
    std::vector<bool>   weightedAvg    {true, false};
    std::vector<MetricType> metricTypes {MetricType::R2, MetricType::MSE};
    std::vector<DistanceType> distanceTypes {DistanceType::Orthogonal};

    // MedianQDF
    std::vector<double> medianErrTols  {1e-3, 1e-2, 1e-1};
    std::vector<int>    medianMaxIters {1000, 2000, 5000};
};

// ── paths & misc ──────────────────────────────────────────────────────────────
struct Config {
    std::string csvPath {"results.csv"};
    std::string logPath {"evaluator.log"};
    double      maxTiltMag = 1.0;          // |W| at tilt=100 %
    int         huberMaxIter = 1000;         
    double      huberErrTol  = 1e-2;
    unsigned    seed = std::random_device{}();
};

// ── API ───────────────────────────────────────────────────────────────────────
void runGridSearch(const Grid& grid, const Config& cfg = {});

} // namespace Evaluator
