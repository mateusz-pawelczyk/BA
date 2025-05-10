#include "evaluation/evaluator_old.hpp"

// Your RANSAC, Model, FlatModel definitions
#include "core/ransac.hpp" // For the RANSAC constructor & run methods
#include <fstream>
#include <iostream>
#include <chrono>
#include <cassert>
#include <core/flat_sampler.hpp>
#include <core/flat_model.hpp>
#include <core/model.hpp>
#include <models/affine_fit.hpp>
#include <models/ols.hpp>
#include <models/mean_sdf.hpp>
#include <models/median_sdf.hpp>
#include <models/huber_regression.hpp>

/**
 * @brief Utility for measuring the milliseconds elapsed while running a function.
 */
static double timeFunctionMS(const std::function<void()> &fn)
{
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    fn();
    auto end = clock::now();
    auto dur = std::chrono::duration<double, std::milli>(end - start);
    return dur.count();
}

/**
 * @brief Compute mse for Variation 1 (Model*).
 *
 * You likely have something similar in your own code (e.g., model->MSE(X, Y)).
 */
static double computeMSE_Model(Model *model, const Eigen::MatrixXd &X, const Eigen::VectorXd &Y)
{
    // Example: If your Model class has MSE(...) directly:
    return model->MSE(X, Y);
}

/**
 * @brief Compute MSE for Variation 2 and 3 (FlatModel*),
 *        where the last column of D is the target.
 */
static double computeMSE_FlatModel(FlatModel *fmodel, const Eigen::MatrixXd &D)
{

    // If your FlatModel implements MSE(X, Y), do:
    return fmodel->quadratic_loss(D).mean();
}

Eigen::MatrixXd mergeAndShuffleEval(const Eigen::MatrixXd &mat1, const Eigen::MatrixXd &mat2, std::mt19937 &gen)
{
    // Ensure both matrices have the same number of columns.
    if (mat1.cols() != mat2.cols())
    {
        throw std::runtime_error("Matrices must have the same number of columns to merge.");
    }

    if (mat1.rows() == 0) {
        if (mat2.rows() == 0) {
            throw std::runtime_error("Both matrices are empty.");
        }
        return mat2;
    }
    if (mat2.rows() == 0) {
        return mat1;
    }
    
    int totalRows = mat1.rows() + mat2.rows();
    int cols = mat1.cols();

    // Concatenate matrices vertically.
    Eigen::MatrixXd merged(totalRows, cols);
    merged << mat1, mat2;

    // Create an index vector for the rows.
    std::vector<int> indices(totalRows);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices.
    std::shuffle(indices.begin(), indices.end(), gen);

    // Build the output matrix using the shuffled indices.
    Eigen::MatrixXd shuffled(totalRows, cols);
    for (int i = 0; i < totalRows; ++i)
    {
        shuffled.row(i) = merged.row(indices[i]);
    }


    return shuffled;
}

namespace Evaluator
{

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
        const Eigen::MatrixXd &D_inlier,
        const Eigen::MatrixXd &D_outlier,
        int n,
        int d,
        MetricType metric,
        std::function<std::unique_ptr<Model>(int, int)> modelFactory,
        std::function<std::unique_ptr<FlatModel>(int, int)> flatModelFactory,
        std::function<RANSAC(int, double, double, int, MetricType)> ransacFactory)
    {
        std::vector<EvaluationRecord> results;

        std::random_device rd;
        std::mt19937 gen(rd());
        Eigen::MatrixXd D = mergeAndShuffleEval(D_inlier, D_outlier, gen);

        Eigen::MatrixXd X_inlier = D_inlier.leftCols(n - 1);
        Eigen::VectorXd Y_inlier = D_inlier.col(n - 1);

        Eigen::MatrixXd X = D.leftCols(n - 1);
        Eigen::VectorXd Y = D.col(n - 1);

        auto loss_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred)
        { return Eigen::VectorXd((Y_true - Y_pred).array().square().matrix()); };
        auto metric_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred)
        { return (Y_true - Y_pred).squaredNorm() / Y_true.size(); };

        std::unique_ptr localFlatModel = std::make_unique<AffineFit>(d, n);

        // 1) Variation 1: MeanSDF
        {
            MeanSDF *averager = new MeanSDF(n - 1, n);

            std::unique_ptr localFlatModel = std::make_unique<AffineFit>(d, n);
            auto ransacObject = ransacFactory(maxIt, threshold, trainDataPct, minInl, metric);

            double elapsedMs = 0.0;
            std::unique_ptr<FlatModel> bestFlat;
            {
                elapsedMs = timeFunctionMS([&]()
                                           { bestFlat = ransacObject.run_slow(D, localFlatModel.get(), bestModelCount, averager, weighted_average); });
            }

            double r2_regression_inlier = bestFlat ? bestFlat->R2(X_inlier, Y_inlier) : 999999.0;
            double mse_regression_inlier = bestFlat ? bestFlat->MSE(X_inlier, Y_inlier) : 999999.0;
            double r2_orthogonal_inlier = bestFlat ? bestFlat->R2(D_inlier) : 999999.0;
            double mse_orthogonal_inlier = bestFlat ? bestFlat->MSE(D_inlier) : 999999.0;

            double r2_regression = bestFlat ? bestFlat->R2(X, Y) : 999999.0;
            double mse_regression = bestFlat ? bestFlat->MSE(X, Y) : 999999.0;
            double r2_orthogonal = bestFlat ? bestFlat->R2(D) : 999999.0;
            double mse_orthogonal = bestFlat ? bestFlat->MSE(D) : 999999.0;

            

            EvaluationRecord rec;
            rec.iterationIndex = iterationIndex;
            rec.maxIterations = maxIt;
            rec.threshold = threshold;
            rec.trainDataPercentage = trainDataPct;
            rec.minInliers = minInl;
            rec.bestModelCount = bestModelCount;
            rec.r2_regression_inlier = r2_regression_inlier;
            rec.r2_orthogonal_inlier = r2_orthogonal_inlier;
            rec.mse_regression_inlier = mse_regression_inlier;
            rec.mse_orthogonal_inlier = mse_orthogonal_inlier;
            rec.r2_regression = r2_regression;
            rec.r2_orthogonal = r2_orthogonal;
            rec.mse_regression = mse_regression;
            rec.mse_orthogonal = mse_orthogonal;
            rec.elapsedMilliseconds = elapsedMs;
            rec.variation = 1;

            results.push_back(rec);
        }

        // 2) Variation 2: MedianSDF
        {
            MedianSDF *averager = new MedianSDF(n - 1, n, median_err_tol, median_max_iter);

            std::unique_ptr localFlatModel = std::make_unique<AffineFit>(d, n);
            auto ransacObject = ransacFactory(maxIt, threshold, trainDataPct, minInl, metric);

            double elapsedMs = 0.0;
            std::unique_ptr<FlatModel> bestFlat;
            {
                elapsedMs = timeFunctionMS([&]()
                                           { bestFlat = ransacObject.run_slow(D, localFlatModel.get(), bestModelCount, averager, weighted_average); });
            }

            double r2_regression_inlier = bestFlat ? bestFlat->R2(X_inlier, Y_inlier) : 999999.0;
            double mse_regression_inlier = bestFlat ? bestFlat->MSE(X_inlier, Y_inlier) : 999999.0;
            double r2_orthogonal_inlier = bestFlat ? bestFlat->R2(D_inlier) : 999999.0;
            double mse_orthogonal_inlier = bestFlat ? bestFlat->MSE(D_inlier) : 999999.0;

            double r2_regression = bestFlat ? bestFlat->R2(X, Y) : 999999.0;
            double mse_regression = bestFlat ? bestFlat->MSE(X, Y) : 999999.0;
            double r2_orthogonal = bestFlat ? bestFlat->R2(D) : 999999.0;
            double mse_orthogonal = bestFlat ? bestFlat->MSE(D) : 999999.0;

            EvaluationRecord rec;
            rec.iterationIndex = iterationIndex;
            rec.maxIterations = maxIt;
            rec.threshold = threshold;
            rec.trainDataPercentage = trainDataPct;
            rec.minInliers = minInl;
            rec.bestModelCount = bestModelCount;
            rec.r2_regression_inlier = r2_regression_inlier;
            rec.r2_orthogonal_inlier = r2_orthogonal_inlier;
            rec.mse_regression_inlier = mse_regression_inlier;
            rec.mse_orthogonal_inlier = mse_orthogonal_inlier;
            rec.r2_regression = r2_regression;
            rec.r2_orthogonal = r2_orthogonal;
            rec.mse_regression = mse_regression;
            rec.mse_orthogonal = mse_orthogonal;
            rec.elapsedMilliseconds = elapsedMs;
            rec.variation = 2;

            results.push_back(rec);
        }

        // 3) Variation 3: Huber Regression
        {

            std::unique_ptr localFlatModel = std::make_unique<AffineFit>(d, n);
            auto ransacObject = ransacFactory(maxIt, threshold, trainDataPct, minInl, metric);

            double elapsedMs = 0.0;
            // bestFlat is huber directly
            std::unique_ptr<FlatModel> bestFlat = std::make_unique<HuberRegression>(n - 1, n);
            {
                elapsedMs = timeFunctionMS([&]()
                                           { bestFlat->fit(D); });
            }

            double r2_regression_inlier = bestFlat ? bestFlat->R2(X_inlier, Y_inlier) : 999999.0;
            double mse_regression_inlier = bestFlat ? bestFlat->MSE(X_inlier, Y_inlier) : 999999.0;
            double r2_orthogonal_inlier = bestFlat ? bestFlat->R2(D_inlier) : 999999.0;
            double mse_orthogonal_inlier = bestFlat ? bestFlat->MSE(D_inlier) : 999999.0;

            double r2_regression = bestFlat ? bestFlat->R2(X, Y) : 999999.0;
            double mse_regression = bestFlat ? bestFlat->MSE(X, Y) : 999999.0;
            double r2_orthogonal = bestFlat ? bestFlat->R2(D) : 999999.0;
            double mse_orthogonal = bestFlat ? bestFlat->MSE(D) : 999999.0;

            EvaluationRecord rec;
            rec.iterationIndex = iterationIndex;
            rec.maxIterations = maxIt;
            rec.threshold = threshold;
            rec.trainDataPercentage = trainDataPct;
            rec.minInliers = minInl;
            rec.bestModelCount = bestModelCount;
            rec.r2_regression_inlier = r2_regression_inlier;
            rec.r2_orthogonal_inlier = r2_orthogonal_inlier;
            rec.mse_regression_inlier = mse_regression_inlier;
            rec.mse_orthogonal_inlier = mse_orthogonal_inlier;
            rec.r2_regression = r2_regression;
            rec.r2_orthogonal = r2_orthogonal;
            rec.mse_regression = mse_regression;
            rec.mse_orthogonal = mse_orthogonal;
            rec.elapsedMilliseconds = elapsedMs;
            rec.variation = 3;

            results.push_back(rec);
        }

        return results;
    }

#include <memory>
#include <sstream>

    void evaluateAllParamCombinations(
        const RansacParameterGrid &grid,
        const DataParameterGrid &dataGrid,
        std::function<std::unique_ptr<Model>(int, int)> modelFactory,
        std::function<std::unique_ptr<FlatModel>(int, int)> flatModelFactory,
        std::function<RANSAC(int, double, double, int, MetricType)> ransacFactory,
        const std::string &outputCsvPath)
    {
        std::ofstream ofs(outputCsvPath, std::ios::app);
        if (!ofs.is_open())
        {
            throw std::runtime_error("Could not open output CSV file: " + outputCsvPath);
        }

        ofs.seekp(0, std::ios::end);
        if (ofs.tellp() == 0)
        {
            ofs << "iteration,maxIt,threshold,trainPct,minInliers,bestModelCount,numPoints,n,d,noise,outlierRatio,outlierStrength,saltAndPepper,metric,variation,weighted_average,r2_regression_inlier,r2_orthogonal_inlier,mse_regression_inlier,mse_orthogonal_inlier,r2_regression,r2_orthogonal,mse_regression,mse_orthogonal,timeMs\n";
        }
        ofs.flush();

        int iterationCounter = 0;
        double progress = 0.0;
        double totalIterations = grid.maxIterations.size() * grid.thresholds.size() * grid.trainDataPercentages.size() * grid.minInliers.size() * grid.bestModelCounts.size() * dataGrid.numPoints.size() * dataGrid.ambientDimentions.size() * 2 * dataGrid.noiseLevels.size() * dataGrid.outlierRatios.size() * dataGrid.outlierStrengths.size() * dataGrid.saltAndPepper.size() * grid.metrics.size() * grid.weightedAverages.size();
        std::cout << "Total iterations: " << totalIterations << std::endl;

        for (int maxIt : grid.maxIterations)
        {
            for (double thresh : grid.thresholds)
            {
                for (double tdp : grid.trainDataPercentages)
                {
                    for (int inl : grid.minInliers)
                    {
                        for (int bmc : grid.bestModelCounts)
                        {
                            for (int numPoints : dataGrid.numPoints)
                            {
                                for (int n : dataGrid.ambientDimentions)
                                {
                                    for (int d : {1, n - 1})
                                    {
                                        for (double noise : dataGrid.noiseLevels)
                                        {
                                            for (double outlierRatio : dataGrid.outlierRatios)
                                            {
                                                for (double outlierStrength : dataGrid.outlierStrengths)
                                                {
                                                    for (bool saltAndPepper : dataGrid.saltAndPepper)
                                                    {
                                                        for (MetricType metric : grid.metrics)
                                                        {
                                                            for (bool weighted_average : grid.weightedAverages)
                                                            {
                                                                progress++;
                                                                int metricInt = static_cast<int>(metric);
                                                                if (inl > numPoints * tdp || d >= n || numPoints <= d + 1 || n <= 1 || d <= 0 || noise < 0.0 || outlierRatio < 0.0 || outlierStrength < 0.0 || outlierRatio > 1.0 || numPoints <= 0 || (noise > 0.3 && n <= 3))
                                                                {
                                                                    continue;
                                                                }

                                                                try
                                                                {
                                                                    auto m = std::make_unique<AffineFit>(n - 1, n);
                                                                    Eigen::MatrixXd W = Eigen::MatrixXd::Random(n - 1, 1) * 0.05;
                                                                    Eigen::VectorXd b = Eigen::VectorXd::Random(1) * 0.05;

                                                                    m->override_explicit(W, b);

                                                                    auto [D_inlier, D_outlier] = FlatSampler::sampleFlatSeparated(*m, numPoints, noise, outlierRatio, outlierStrength, saltAndPepper);
                                                                    if (D_inlier.rows() == 0) {
                                                                        std::cout << "Skipping... No inlier found. Outlier Count: " << D_outlier.rows() << std::endl;
                                                                        std::ostringstream oss;
                                                                        oss << "Parameters:\n"
                                                                            << "  maxIt: " << maxIt << "\n"
                                                                            << "  thresh: " << thresh << "\n"
                                                                            << "  tdp: " << tdp << "\n"
                                                                            << "  inl: " << inl << "\n"
                                                                            << "  bmc: " << bmc << "\n"
                                                                            << "  numPoints: " << numPoints << "\n"
                                                                            << "  n: " << n << "\n"
                                                                            << "  d: " << d << "\n"
                                                                            << "  noise: " << noise << "\n"
                                                                            << "  outlierRatio: " << outlierRatio << "\n"
                                                                            << "  outlierStrength: " << outlierStrength << "\n"
                                                                            << "  saltAndPepper: " << (saltAndPepper ? "true" : "false") << "\n"
                                                                            << "  metric: " << static_cast<int>(metric) << "\n"
                                                                            << "  weighted_average: " << (weighted_average ? "true" : "false") << "\n";
                                                                        std::cerr << oss.str() << std::endl;
                                                                            continue;
                                                                    }
                                                                    iterationCounter++;

                                                                    std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << (progress / totalIterations) * 100 << "%" << std::flush;

                                                                    auto records = evaluateSingleCombo(
                                                                        iterationCounter,
                                                                        maxIt, thresh, tdp, inl, bmc, weighted_average,
                                                                        0.01, 1000,
                                                                        D_inlier, D_outlier,
                                                                        n, d, metric,
                                                                        modelFactory,
                                                                        flatModelFactory,
                                                                        ransacFactory);

                                                                    for (auto &rec : records)
                                                                    {
                                                                        ofs << rec.iterationIndex << ","
                                                                            << rec.maxIterations << ","
                                                                            << rec.threshold << ","
                                                                            << rec.trainDataPercentage << ","
                                                                            << rec.minInliers << ","
                                                                            << rec.bestModelCount << ","
                                                                            << numPoints << ","
                                                                            << n << ","
                                                                            << d << ","
                                                                            << noise << ","
                                                                            << outlierRatio << ","
                                                                            << outlierStrength << ","
                                                                            << saltAndPepper << ","
                                                                            << static_cast<int>(metric) << ","
                                                                            << rec.variation << ","
                                                                            << weighted_average << ","
                                                                            << rec.r2_regression_inlier << ","
                                                                            << rec.r2_orthogonal_inlier << ","
                                                                            << rec.mse_regression_inlier << ","
                                                                            << rec.mse_orthogonal_inlier << ","
                                                                            << rec.r2_regression << ","
                                                                            << rec.r2_orthogonal << ","
                                                                            << rec.mse_regression << ","
                                                                            << rec.mse_orthogonal << ","
                                                                            << rec.elapsedMilliseconds
                                                                            << "\n";
                                                                        ofs.flush();
                                                                    }
                                                                }
                                                                catch (const std::exception &e)
                                                                {
                                                                    std::ostringstream oss;
                                                                    oss << "Error encountered during evaluation:\n"
                                                                        << "Parameters:\n"
                                                                        << "  maxIt: " << maxIt << "\n"
                                                                        << "  thresh: " << thresh << "\n"
                                                                        << "  tdp: " << tdp << "\n"
                                                                        << "  inl: " << inl << "\n"
                                                                        << "  bmc: " << bmc << "\n"
                                                                        << "  numPoints: " << numPoints << "\n"
                                                                        << "  n: " << n << "\n"
                                                                        << "  d: " << d << "\n"
                                                                        << "  noise: " << noise << "\n"
                                                                        << "  outlierRatio: " << outlierRatio << "\n"
                                                                        << "  outlierStrength: " << outlierStrength << "\n"
                                                                        << "  saltAndPepper: " << (saltAndPepper ? "true" : "false") << "\n"
                                                                        << "  metric: " << static_cast<int>(metric) << "\n"
                                                                        << "  weighted_average: " << (weighted_average ? "true" : "false") << "\n"
                                                                        << "Error message: " << e.what() << "\n\n";
                                                                    std::cerr << oss.str() << std::endl;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        std::cout << std::endl;
        ofs.close();
    }

} // namespace Evaluator
