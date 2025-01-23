#include "evaluation/evaluator.hpp"

// Your RANSAC, Model, FlatModel definitions
#include "core/ransac.hpp"    // For the RANSAC constructor & run methods
#include <fstream>
#include <iostream>
#include <chrono>
#include <cassert>
#include <core/flat_sampler.hpp>
#include <core/flat_model.hpp>
#include <core/model.hpp>
#include <models/affine_fit.hpp>
#include <models/ols.hpp>

/**
 * @brief Utility for measuring the milliseconds elapsed while running a function.
 */
static double timeFunctionMS(const std::function<void()>& fn) {
    using clock = std::chrono::high_resolution_clock;
    auto start  = clock::now();
    fn();
    auto end    = clock::now();
    auto dur    = std::chrono::duration<double, std::milli>(end - start);
    return dur.count();
}

/**
 * @brief Compute MSE for Variation 1 (Model*).
 * 
 * You likely have something similar in your own code (e.g., model->MSE(X, Y)).
 */
static double computeMSE_Model(Model* model, const Eigen::MatrixXd& X, const Eigen::VectorXd& Y) {
    // Example: If your Model class has MSE(...) directly:
    return model->MSE(X, Y); 
}

/**
 * @brief Compute MSE for Variation 2 and 3 (FlatModel*), 
 *        where the last column of D is the target.
 */
static double computeMSE_FlatModel(FlatModel* fmodel, const Eigen::MatrixXd& D) {
    // For Variation 2 & 3, the last col of D is Y, first columns are X
    int d = D.cols() - 1;
    Eigen::MatrixXd X = D.leftCols(d);
    Eigen::VectorXd Y = D.col(d);

    // If your FlatModel implements MSE(X, Y), do:
    return fmodel->MSE(X, Y);
}

namespace Evaluator {

std::vector<EvaluationRecord> evaluateSingleCombo(
    int iterationIndex,
    int maxIt,
    double threshold,
    double trainDataPct,
    int minInl,
    int bestModelCount,
    const Eigen::MatrixXd& D,
    int n,
    int d,
    std::function<std::unique_ptr<Model>(int, int)>     modelFactory,
    std::function<std::unique_ptr<FlatModel>(int, int)> flatModelFactory,
    std::function<RANSAC(int,double,double,int)> ransacFactory
) {
    std::vector<EvaluationRecord> results;
    Eigen::MatrixXd X = D.leftCols(D.cols() - 1);
    Eigen::VectorXd Y = D.col(D.cols() - 1);

    auto loss_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred){return Eigen::VectorXd((Y_true - Y_pred).array().square().matrix());};
    auto metric_fn = [](Eigen::VectorXd Y_true, Eigen::VectorXd Y_pred){return (Y_true - Y_pred).squaredNorm() / Y_true.size();};

    // 1) Variation 1: run(X, Y, Model*)
    if (n == d + 1)
    {
        auto localModel   = modelFactory(d, n);  // new instance each time
        auto ransacObject = ransacFactory(maxIt, threshold, trainDataPct, minInl);

        double elapsedMs = 0.0;
        std::unique_ptr<Model> bestModel;
        {
            elapsedMs = timeFunctionMS([&]() {
                bestModel = ransacObject.run(X, Y, localModel.get(), loss_fn, metric_fn);
            });
        }
        double mse = bestModel ? computeMSE_Model(bestModel.get(), X, Y) : 999999.0;
        
        EvaluationRecord rec;
        rec.iterationIndex       = iterationIndex;
        rec.variationIndex       = 1;
        rec.maxIterations        = maxIt;
        rec.threshold            = threshold;
        rec.trainDataPercentage  = trainDataPct;
        rec.minInliers           = minInl;
        rec.bestModelCount       = bestModelCount;  // not used in Variation 1, but we store it anyway
        rec.mse                  = mse;
        rec.elapsedMilliseconds  = elapsedMs;

        results.push_back(rec);
    }

    // 2) Variation 2: run(D, FlatModel*, best_model_count)
    if (n == d + 1)
    {
        auto localFlatModel = flatModelFactory(d, n);
        auto ransacObject   = ransacFactory(maxIt, threshold, trainDataPct, minInl);

        double elapsedMs = 0.0;
        std::unique_ptr<FlatModel> bestFlat;
        {
            elapsedMs = timeFunctionMS([&]() {
                bestFlat = ransacObject.run(D, localFlatModel.get(), bestModelCount, loss_fn, metric_fn);
            });
        }

        double mse = bestFlat ? computeMSE_FlatModel(bestFlat.get(), D) : 999999.0;

        EvaluationRecord rec;
        rec.iterationIndex       = iterationIndex;
        rec.variationIndex       = 2;
        rec.maxIterations        = maxIt;
        rec.threshold            = threshold;
        rec.trainDataPercentage  = trainDataPct;
        rec.minInliers           = minInl;
        rec.bestModelCount       = bestModelCount;
        rec.mse                  = mse;
        rec.elapsedMilliseconds  = elapsedMs;

        results.push_back(rec);
    }

    // 3) Variation 3: run2(D, FlatModel*, best_model_count)
    {
        auto localFlatModel = flatModelFactory(d, n);
        auto ransacObject   = ransacFactory(maxIt, threshold, trainDataPct, minInl);

        double elapsedMs = 0.0;
        std::unique_ptr<FlatModel> bestFlat;
        {
            elapsedMs = timeFunctionMS([&]() {
                bestFlat = ransacObject.run2(D, localFlatModel.get(), bestModelCount);
            });
        }

        double mse = bestFlat ? computeMSE_FlatModel(bestFlat.get(), D) : 999999.0;

        EvaluationRecord rec;
        rec.iterationIndex       = iterationIndex;
        rec.variationIndex       = 3;
        rec.maxIterations        = maxIt;
        rec.threshold            = threshold;
        rec.trainDataPercentage  = trainDataPct;
        rec.minInliers           = minInl;
        rec.bestModelCount       = bestModelCount;
        rec.mse                  = mse;
        rec.elapsedMilliseconds  = elapsedMs;

        results.push_back(rec);
    }

    return results;
}

void evaluateAllParamCombinations(
    const RansacParameterGrid& grid,
    const DataParameterGrid& dataGrid,
    std::function<std::unique_ptr<Model>(int, int)>     modelFactory,
    std::function<std::unique_ptr<FlatModel>(int, int)> flatModelFactory,
    std::function<RANSAC(int,double,double,int)> ransacFactory,
    const std::string& outputCsvPath
) {
    // Open the CSV in append mode (so if you crash, partial results remain).
    // If you want to overwrite each time, use std::ios::out instead.
    std::ofstream ofs(outputCsvPath, std::ios::app);
    if(!ofs.is_open()) {
        throw std::runtime_error("Could not open output CSV file: " + outputCsvPath);
    }

    // If file is empty, write a header row
    // (You might want a more robust check if file already has data).
    ofs.seekp(0, std::ios::end);
    if (ofs.tellp() == 0) {
        // old: ofs << "iteration,variation,maxIt,threshold,trainPct,minInliers,bestModelCount,mse,timeMs\n";
        ofs << "iteration,variation,maxIt,threshold,trainPct,minInliers,bestModelCount,numPoints,n,d,noise,outlierRatio,outlierStrength,saltAndPepper,mse,timeMs\n";
    }
    ofs.flush();

    int iterationCounter = 0;

    double progress = 0.0;
    double totalIterations = grid.maxIterations.size() * grid.thresholds.size() * grid.trainDataPercentages.size() * grid.minInliers.size() * grid.bestModelCounts.size() * dataGrid.numPoints.size() * dataGrid.ambientDimentions.size() * dataGrid.noiseLevels.size() * dataGrid.outlierRatios.size() * dataGrid.outlierStrengths.size() * dataGrid.saltAndPepper.size();
    std::cout << "Total iterations: " << totalIterations << std::endl;
    for (int maxIt : grid.maxIterations) {
        for (double thresh : grid.thresholds) {
            for (double tdp : grid.trainDataPercentages) {
                for (int inl : grid.minInliers) {
                    for (int bmc : grid.bestModelCounts) {
                        for (int numPoints : dataGrid.numPoints) {
                            for (int n : dataGrid.ambientDimentions) {
                                for (int d = 1; d < n; ++d) {
                                    for (double noise : dataGrid.noiseLevels) {
                                        for (double outlierRatio : dataGrid.outlierRatios) {
                                            for (double outlierStrength : dataGrid.outlierStrengths) {
                                                for (bool saltAndPepper : dataGrid.saltAndPepper) {
                                                    if (inl > numPoints * tdp) {
                                                        continue; // skip impossible cases
                                                    }

                                                    // Create the data matrix D
                                                    AffineFit* m = new AffineFit(d, n);
                                                    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, d);
                                                    Eigen::VectorXd b = Eigen::VectorXd::Random(n);

                                                    m->override_parametric(A, b);
                                                    Eigen::MatrixXd D = FlatSampler::sampleFlat(*m, numPoints, noise, outlierRatio, outlierStrength, 1.0, saltAndPepper);

                                                    // Increment the iteration counter
                                                    iterationCounter++;
                                                    progress += 1.0;
                                                    std::cout << "Progress: " << (progress / totalIterations) * 100 << "%" << std::endl;

                                                    // Evaluate one combination: run all 3 variations
                                                    auto records = evaluateSingleCombo(
                                                        iterationCounter, 
                                                        maxIt, thresh, tdp, inl, bmc, 
                                                        D, 
                                                        n, d,
                                                        modelFactory, 
                                                        flatModelFactory, 
                                                        ransacFactory
                                                    );


                                                    // Write them to CSV (3 lines, one per variation)
                                                    for (auto& rec : records) {
                                                        ofs << rec.iterationIndex << ","
                                                            << rec.variationIndex << ","
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
                                                            << rec.mse << ","
                                                            << rec.elapsedMilliseconds 
                                                            << "\n";
                                                        ofs.flush(); // ensure each line is written
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

    ofs.close();
}

} // namespace Evaluator
