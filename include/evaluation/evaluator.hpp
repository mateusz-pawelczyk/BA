#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <Eigen/Core>
#include <memory>
#include <vector>
#include <string>
#include <functional>

// Forward declarations from your project
class Model;
class FlatModel;

enum class MetricType;

// Forward declaration of your RANSAC class
class RANSAC;

/**
 * @brief Collection of parameter ranges for RANSAC grid search.
 *
 * Define each parameter as a list of possible values.
 * For example:
 *   maxIterations = {100, 500, 1000}
 *   thresholds    = {0.001, 0.01, 0.1}
 *   etc.
 *
 * You can fill these however you like in your main or elsewhere.
 */
struct RansacParameterGrid
{
    std::vector<int> maxIterations;           ///< e.g. {100, 500, 1000}
    std::vector<double> thresholds;           ///< e.g. {0.0001, 0.001, 0.01}
    std::vector<double> trainDataPercentages; ///< e.g. {0.2, 0.5, 0.8}
    std::vector<int> minInliers;              ///< e.g. {5, 10, 20}
    std::vector<int> bestModelCounts;         ///< e.g. {1, 5, 10} (used in run(..., best_model_count) variants)
    std::vector<MetricType> metrics;          ///< e.g. {MetricType::r2, MetricType::R2}
    std::vector<bool> weightedAverages;       ///< e.g. {true, false}
};

struct DataParameterGrid
{
    std::vector<int> numPoints;           ///< e.g. {100, 500, 1000}
    std::vector<int> subspaceDimentions;  ///< e.g. {2, 3, 4}
    std::vector<int> ambientDimentions;   ///< e.g. {3, 4, 5}
    std::vector<double> noiseLevels;      ///< e.g. {0.01, 0.1, 0.5}
    std::vector<double> outlierRatios;    ///< e.g. {0.0, 0.1, 0.2}
    std::vector<double> outlierStrengths; ///< e.g. {1.0, 2.0, 5.0}
    std::vector<double> volumes;          ///< e.g. {1.0, 2.0, 5.0}
    std::vector<bool> saltAndPepper;      ///< e.g. {true, false}
};

/**
 * @brief A single evaluation record (one row in the CSV).
 */
struct EvaluationRecord
{
    int iterationIndex; ///< Counter for which combination of parameters we're on
    int maxIterations;
    double threshold;
    double trainDataPercentage;
    int minInliers;
    int bestModelCount;
    double r2;                  ///< Computed r2
    double mse;                 ///< Computed MSE
    double elapsedMilliseconds; ///< Timing for that RANSAC run
    int variation;              ///< 1, 2
};

/**
 * @brief Contains static functions for enumerating grid search parameters and running all RANSAC variations.
 */
namespace Evaluator
{

    /**
     * @brief Evaluate a single combination of parameters using all three RANSAC variations:
     *  1) `run(X, Y, Model*)`
     *  2) `run(D, FlatModel*, best_model_count)`
     *  3) `run2(D, FlatModel*, best_model_count)`
     *
     * This is invoked internally by the grid search; you typically won't call it directly.
     */
    std::vector<EvaluationRecord> evaluateSingleCombo(
        int iterationIndex,
        int maxIt,
        double threshold,
        double trainDataPct,
        int minInl,
        int bestModelCount,
        bool weighted_average,
        double median_err_tol,
        int median_max_iter,
        const Eigen::MatrixXd &D, // for Variation 2 & 3 (last col is Y)
        int n,                    // ambient dimension
        int d,                    // flat dimension
        MetricType metric,
        std::function<std::unique_ptr<Model>(int, int)> modelFactory,             // to create fresh Model*
        std::function<std::unique_ptr<FlatModel>(int, int)> flatModelFactory,     // to create fresh FlatModel*
        std::function<RANSAC(int, double, double, int, MetricType)> ransacFactory // how to create RANSAC object
    );

    /**
     * @brief Perform a full grid search over all parameter combinations, running all three RANSAC variations.
     *        Writes partial results to CSV after each variation so you don't lose data on crash.
     *
     * @param grid              A struct with vectors of possible parameter values for RANSAC
     * @param X                 The design matrix for Variation 1 (dimensions NxD).
     * @param Y                 The labels vector for Variation 1 (length N).
     * @param D                 Data matrix for Variation 2 & 3 (dimensions Nx(D+1)), last column is label.
     * @param modelFactory      Creates a fresh `Model*` for Variation 1 each time.
     * @param flatModelFactory  Creates a fresh `FlatModel*` for Variation 2 & 3 each time.
     * @param ransacFactory     Creates a fresh RANSAC object with (maxIt, threshold, trainPct, minInliers).
     * @param outputCsvPath     Where to write CSV results (will append if file exists).
     */
    void evaluateAllParamCombinations(
        const RansacParameterGrid &grid,
        const DataParameterGrid &dataGrid,
        std::function<std::unique_ptr<Model>(int, int)> modelFactory,
        std::function<std::unique_ptr<FlatModel>(int, int)> flatModelFactory,
        std::function<RANSAC(int, double, double, int, MetricType)> ransacFactory,
        const std::string &outputCsvPath);

} // namespace Evaluator

#endif // EVALUATOR_H
