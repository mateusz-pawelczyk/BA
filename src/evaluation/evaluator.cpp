#include "evaluation/evaluator.hpp"

#include <iomanip>
#include <iostream>
#include <random>
#include <chrono>

// ─ Project headers ─
#include "core/flat_sampler.hpp"
#include "core/ransac.hpp"
#include "models/affine_fit.hpp"
#include "models/mean_sdf.hpp"
#include "models/median_sdf.hpp"
#include "models/huber_regression.hpp"

using Clock = std::chrono::high_resolution_clock;

namespace {

// ────────── helpers ──────────
double randomDouble(std::mt19937& g,double a,double b){
    std::uniform_real_distribution<double> d(a,b); return d(g);
}
Eigen::VectorXd randomUnitVec(int dim,std::mt19937& g){
    std::normal_distribution<double> N(0,1);
    Eigen::VectorXd v(dim); for(int i=0;i<dim;++i) v(i)=N(g);
    return v.normalized();
}

// ─ CSV header once ─
void writeCsvHeader(std::ofstream& csv){
    csv << "idx,n,d,numPts,tiltFrac,noise,outRatio,outStr,sp,"
         << "maxIt,thr,trainPct,minInl,bestCnt,weighted,"
         << "r2_all_mean,r2_inl_mean,r2_all_med,r2_inl_med,"
         << "mse_all_mean,mse_inl_mean,mse_all_med,mse_inl_med,"
         << "mse_huber,mse_huber_inl,timeMs\n";
}

} // <anon ns>

namespace Evaluator {

void runGridSearch(const Grid& g,const Config& cfg)
{
    std::ofstream csv(cfg.csvPath, std::ios::app);
    std::ofstream log(cfg.logPath, std::ios::app);
    if(!csv) throw std::runtime_error("Cannot open CSV file");
    if(csv.tellp()==0) writeCsvHeader(csv);
    if(!log)  throw std::runtime_error("Cannot open log file");

    std::mt19937 rng(cfg.seed);
    std::size_t idx=0;

    // ────────── nested loops – explicit for readability & logging ──────────
    for(int N         : g.numPoints      )
    for(int n         : g.ambientDims    )
    for(int d         : { 1, n - 1}      )
    for(double tiltF  : g.tiltFractions  )
    for(double noise  : g.noiseLevels    )
    for(double oRatio : g.outlierRatios  )
    for(double oStr   : g.outlierStr     )
    for(bool   sPep   : g.saltPepper     )
    for(int    maxIt  : g.maxIterations  )
    for(double thr    : g.thresholds     )
    for(double tPct   : g.trainPcts      )
    for(int    minInl : g.minInliers     )
    for(int    bestK  : g.bestModelCnt   )
    for(bool   wAvg   : g.weightedAvg    )
    {
        ++idx;
        log << "\n──────────────── Case "<<idx<<" ────────────────\n";

        // 1) ---------- ground-truth flat ----------
        Eigen::VectorXd W = randomUnitVec(n - 1,rng) * (tiltF*cfg.maxTiltMag);
        Eigen::VectorXd B = Eigen::VectorXd::Zero(1);
        auto trueFlat = std::make_unique<AffineFit>(n - 1,n);
        trueFlat->override_explicit(W,B);

        // 2) ---------- sample data ----------
        Eigen::MatrixXd Din,Dout;
        std::tie(Din,Dout)=FlatSampler::sampleFlatSeparated(
            *trueFlat, N, noise, oRatio, oStr, sPep);

        // guard: need inliers
        if(Din.rows()<d+1){ log<<"Skipped (no inliers)\n"; continue; }

        // merge
        Eigen::MatrixXd D(Din.rows()+Dout.rows(), n);
        D << Din, Dout;

        // 3) ---------- RANSAC object ----------
        RANSAC ransac(maxIt,thr,tPct,minInl,MetricType::R2);
        auto proto = std::make_unique<AffineFit>(d,n);

        // 4) ---------- MeanSDF ----------
        MeanSDF meanAvg(n - 1,n);
        auto t0=Clock::now();
        auto flatMean = ransac.run_slow(D,proto.get(),bestK,&meanAvg,wAvg);

        // 5) ---------- MedianSDF ----------
        MedianSDF medAvg(n - 1,n,cfg.huberErrTol,cfg.huberMaxIter);
        auto flatMed  = ransac.run_slow(D,proto.get(),bestK,&medAvg ,wAvg);
        auto t1=Clock::now();
        double ms=std::chrono::duration<double, std::milli>(t1-t0).count();

        // 6) ---------- Huber regression ----------
        HuberRegression huber(n - 1,n);
        huber.fit(D);

        // 7) ---------- metrics ----------
        auto Xa = D.leftCols(n - 1);
        auto Ya = D.col(n - 1);
        auto Xi = Din.leftCols(n - 1);
        auto Yi = Din.col(n - 1);

        auto r2   = [&](const std::unique_ptr<FlatModel>& f,
                        const Eigen::MatrixXd& X,const Eigen::VectorXd& Y)
                        { return f? f->R2(X,Y):std::numeric_limits<double>::quiet_NaN(); };

        

        // R² 
        double r2AllMean  = r2(flatMean ,Xa,Ya);
        double r2InlMean  = r2(flatMean ,Xi,Yi);
        double r2AllMed   = r2(flatMed  ,Xa,Ya);
        double r2InlMed   = r2(flatMed  ,Xi,Yi);
        double r2Huber    = huber.R2(Xa,Ya);
        double r2HuberInl = huber.R2(Xi,Yi);

        // MSE
        double mseAllMean  = flatMean ? flatMean->MSE(Xa,Ya) : std::numeric_limits<double>::quiet_NaN();
        double mseInlMean  = flatMean ? flatMean->MSE(Xi,Yi) : std::numeric_limits<double>::quiet_NaN();
        double mseAllMed   = flatMed  ? flatMed->MSE(Xa,Ya) : std::numeric_limits<double>::quiet_NaN();
        double mseInlMed   = flatMed  ? flatMed->MSE(Xi,Yi) : std::numeric_limits<double>::quiet_NaN();
        double mseHuber    = huber.MSE(Xa,Ya);
        double mseHuberInl = huber.MSE(Xi,Yi);

        // 8) ---------- logging ----------
        log<<std::fixed<<std::setprecision(4);
        log<<"  n="<<n<<" d="<<d<<" N="<<N<<" tilt="<<tiltF*100<<"%\n";
        log<<"  noise="<<noise<<" outRatio="<<oRatio<<" outStr="<<oStr
           <<" saltPep="<<(sPep?"yes":"no")<<"\n";
        log<<"  RANSAC: maxIt="<<maxIt<<" thr="<<thr
           <<" trainPct="<<tPct<<" minInl="<<minInl<<"\n";
        log<<"  Heap bestK="<<bestK<<" weighted="<<(wAvg?"yes":"no")<<"\n";
        log<<"  r2  (Mean)  inl="<<r2InlMean<<"  all="<<r2AllMean<<"\n";
        log<<"  r2  (Median)inl="<<r2InlMed <<"  all="<<r2AllMed<<"\n";
        log<<"  r2  (Huber) inl="<<r2HuberInl<<"  all="<<r2Huber<<"\n";
        log<<"  mse (Mean)  inl="<<mseInlMean<<"  all="<<mseAllMean<<"\n";
        log<<"  mse (Median)inl="<<mseInlMed <<"  all="<<mseAllMed<<"\n";
        log<<"  mse (Huber) inl="<<mseHuberInl<<"  all="<<mseHuber<<"\n";
        log<<"  runtime "<<ms<<" ms\n";

        // 8) ---------- CSV ----------
        csv<<idx<<','<<n<<','<<d<<','<<N<<','<<tiltF<<','<<noise<<','
           <<oRatio<<','<<oStr<<','<<sPep<<','
           <<maxIt<<','<<thr<<','<<tPct<<','<<minInl<<','<<bestK<<','
           <<wAvg<<','
           <<r2AllMean<<','<<r2InlMean<<','
           <<r2AllMed <<','<<r2InlMed <<','
           <<mseAllMean<<','<<mseInlMean<<','
           <<mseAllMed <<','<<mseInlMed <<','
           <<mseHuber    <<','<<mseHuberInl <<','<<ms<<"\n";
        csv.flush();
    }

    log<<"\n===== Grid search finished =====\n";
}

} // namespace Evaluator
