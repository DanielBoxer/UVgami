#include <algorithm>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <limits>
#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>
#include <thread>

#include <tbb/global_control.h>

#include "uvgami.h"
#include "uvgami_config.h"
#include "Flatten.hpp"
#include "IglUtils.hpp"
#include "Optimizer.hpp"
#include "SymDirichletEnergy.hpp"

#include <igl/cut_to_disk.h>
#include <igl/default_num_threads.h>
#include <igl/cut_mesh.h>
#include <igl/readOFF.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/arap.h>
#include <igl/avg_edge_length.h>
#include <igl/edge_lengths.h>
#include <igl/is_vertex_manifold.h>
#include <igl/is_edge_manifold.h>
#include <igl/facet_components.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#define TCLAP_NAMESTARTSTRING "-"
#include "tclap/CmdLine.h"

Eigen::MatrixXd V, UV, N;
Eigen::MatrixXi F, FUV, FN;

// optimization
std::vector<const uvgami::TriMesh *> triSoup;
int vertAmt_input;
uvgami::TriMesh triSoup_backup;
uvgami::Optimizer *optimizer;
std::vector<uvgami::Energy *> energyTerms;
std::vector<double> energyParams;

bool rand1PInitCut = false;
// pinned runs can chase an unreachable distortion bound forever, capped below
bool pinnedMode = false;
// relax the kept map without any split/merge, stops at the first stationary point
bool noCutMode = false;
// greedily merge islands along shared mesh edges, no splits ever
bool stitchMode = false;
int stationaryCount = 0;
double lambda_init = 0.999;
bool optimization_on = false;
int iterNum = 0;
int converged = 0;
double fracThres = 0.0;
bool topoLineSearch = true;
int initCutOption = 0;
// one cut per inverted piece per round about halves its depth
const int MAX_DEEPEN_ROUNDS = 30;
// below this 1 / area^2 exceeds the other energy terms' precision
const double NEAR_ZERO_INIT_RATIO = 1e-16;
bool outerLoopFinished = false;
double upperBound = 4.1;
const double convTol_upperBound = 1.0e-3;

std::vector<std::pair<double, double>> energyChanges_bSplit,
    energyChanges_iSplit, energyChanges_merge;
std::vector<std::vector<int>> paths_bSplit, paths_iSplit, paths_merge;
std::vector<Eigen::MatrixXd> newVertPoses_bSplit, newVertPoses_iSplit,
    newVertPoses_merge;

int opType_queried = -1;
std::vector<int> path_queried;
Eigen::MatrixXd newVertPos_queried;
bool reQuery = false;
double filterExp_in = 0.6;
int inSplitTotalAmt;

// std::ofstream logFile;
std::string outputFolderPath;
std::string meshName;

const int channel_initial = 0;
const int channel_result = 1;
bool canSaveMesh = false;
bool mute = true;
std::atomic<bool> forceQuit = false;
std::atomic<bool> forceQuitSave = false;
std::atomic<bool> snapshot = false;
int maxSeamWeight = 100;
int maxFaceWeight = 10;

const char *pathSeparator() {
#ifdef _WIN32
    return "\\";
#else
    return "/";
#endif
}
uvgami::ChronoTimer mainTimer("unwrap time");

std::string outputDirArg;

// reset per mesh in resetMeshState like the globals above
double minRestEdgeLength = std::numeric_limits<double>::quiet_NaN();
int iterNum_bestFeasible = -1;
uvgami::TriMesh triSoup_bestFeasible;
double E_se_bestFeasible = DBL_MAX;
// boundary and interior queries share an iterNum
int lastStationaryIterNum = 0;
std::map<double, std::vector<std::pair<double, double>>> configs_stationaryV;
int oscillated_infeasible = 0;
int stalledSolveIterations = 0;
double stalledSolveEnergy = -1.0;
int noProgressCount = 0;
std::deque<std::pair<double, Eigen::Index>> recentSeamSets;

// every value a fresh process starts with
void resetMeshState() {
    V = Eigen::MatrixXd();
    UV = Eigen::MatrixXd();
    N = Eigen::MatrixXd();
    F = Eigen::MatrixXi();
    FUV = Eigen::MatrixXi();
    FN = Eigen::MatrixXi();
    triSoup.clear();
    vertAmt_input = 0;
    triSoup_backup = uvgami::TriMesh();
    optimizer = nullptr;
    energyTerms.clear();
    energyParams.clear();
    rand1PInitCut = false;
    pinnedMode = false;
    noCutMode = false;
    stitchMode = false;
    stationaryCount = 0;
    optimization_on = false;
    iterNum = 0;
    converged = 0;
    outerLoopFinished = false;
    energyChanges_bSplit.clear();
    energyChanges_iSplit.clear();
    energyChanges_merge.clear();
    paths_bSplit.clear();
    paths_iSplit.clear();
    paths_merge.clear();
    newVertPoses_bSplit.clear();
    newVertPoses_iSplit.clear();
    newVertPoses_merge.clear();
    opType_queried = -1;
    path_queried.clear();
    newVertPos_queried = Eigen::MatrixXd();
    reQuery = false;
    filterExp_in = 0.6;
    inSplitTotalAmt = 0;
    canSaveMesh = false;
    forceQuit = false;
    forceQuitSave = false;
    snapshot = false;
    minRestEdgeLength = std::numeric_limits<double>::quiet_NaN();
    iterNum_bestFeasible = -1;
    triSoup_bestFeasible = uvgami::TriMesh();
    E_se_bestFeasible = DBL_MAX;
    lastStationaryIterNum = 0;
    configs_stationaryV.clear();
    oscillated_infeasible = 0;
    stalledSolveIterations = 0;
    stalledSolveEnergy = -1.0;
    noProgressCount = 0;
    recentSeamSets.clear();
}

void releaseMesh() {
    for (auto &eI : energyTerms)
        delete eI;
    energyTerms.clear();
    delete optimizer;
    optimizer = nullptr;
    if (!triSoup.empty())
        delete triSoup[0];
    triSoup.clear();
}

// "unwrap <path>" stdin lines, taken by main in order
std::mutex pathQueueMutex;
std::condition_variable pathQueued;
std::deque<std::string> pathQueue;
bool stdinClosed = false;

void stdin_listener() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line == "stop") {
            forceQuit = true;
            forceQuitSave = true;
        } else if (line == "cancel") {
            forceQuit = true;
            forceQuitSave = false;
        } else if (line == "snapshot") {
            snapshot = true;
        } else if (line.rfind("unwrap ", 0) == 0) {
            std::lock_guard<std::mutex> lock(pathQueueMutex);
            pathQueue.push_back(line.substr(7));
            pathQueued.notify_one();
        }
    }
    std::lock_guard<std::mutex> lock(pathQueueMutex);
    stdinClosed = true;
    pathQueued.notify_one();
}

// empty once stdin closes
std::string nextMeshPath() {
    std::unique_lock<std::mutex> lock(pathQueueMutex);
    pathQueued.wait(lock, [] { return !pathQueue.empty() || stdinClosed; });
    if (pathQueue.empty())
        return "";
    std::string path = pathQueue.front();
    pathQueue.pop_front();
    return path;
}

void proceedOptimization(int proceedNum) {
    for (int proceedI = 0; (proceedI < proceedNum) && converged == 0;
         proceedI++) {
        converged = optimizer->solve(1);
        iterNum = optimizer->getIterNum();
    }
}

// emits the progress line the addon parses and serves the snapshot command
void reportProgress(void) {
    Eigen::VectorXd distortionPerElem;
    energyTerms[0]->getEnergyValPerElem(*triSoup[channel_result],
                                        distortionPerElem, true);
    uvgami::IglUtils::reportDistortion(distortionPerElem, 4.0, 8.5);

    if (snapshot) {
        triSoup[channel_result]->saveAsMesh(F, true);
        snapshot = false;
    }
}

bool postDrawFunc(void) {
    if (iterNum == 0) {
        optimization_on = !optimization_on;
        if (optimization_on && converged)
            optimization_on = false;
    }
    if (forceQuit) {
        canSaveMesh = forceQuitSave;
        outerLoopFinished = true;
    }
    if (canSaveMesh) {
        // save mesh
        if (outerLoopFinished) {
            if (!triSoup[channel_result]->saveAsMesh(outputFolderPath, F, true))
                std::cerr << "Unable to save mesh" << std::endl;
            mainTimer.finish();
        }
        canSaveMesh = false;
    }

    if (outerLoopFinished)
        return true;

    return false;
}

int computeOptPicked(
    const std::vector<std::pair<double, double>> &energyChanges0,
    const std::vector<std::pair<double, double>> &energyChanges1,
    double lambda) {
    assert(!energyChanges0.empty());
    assert(!energyChanges1.empty());
    assert((lambda >= 0.0) && (lambda <= 1.0));

    double minEChange0 = DBL_MAX;
    for (int ecI = 0; ecI < energyChanges0.size(); ecI++) {
        if ((energyChanges0[ecI].first == DBL_MAX) ||
            (energyChanges0[ecI].second == DBL_MAX))
            continue;
        double EwChange = energyChanges0[ecI].first * (1.0 - lambda) +
                          energyChanges0[ecI].second * lambda;
        if (EwChange < minEChange0)
            minEChange0 = EwChange;
    }
    double minEChange1 = DBL_MAX;
    for (int ecI = 0; ecI < energyChanges1.size(); ecI++) {
        if ((energyChanges1[ecI].first == DBL_MAX) ||
            (energyChanges1[ecI].second == DBL_MAX))
            continue;
        double EwChange = energyChanges1[ecI].first * (1.0 - lambda) +
                          energyChanges1[ecI].second * lambda;
        if (EwChange < minEChange1)
            minEChange1 = EwChange;
    }

    assert((minEChange0 != DBL_MAX) || (minEChange1 != DBL_MAX));
    return (minEChange0 > minEChange1);
}

int computeBestCand(const std::vector<std::pair<double, double>> &energyChanges,
                    double lambda, double &bestEChange) {
    assert((lambda >= 0.0) && (lambda <= 1.0));

    bestEChange = DBL_MAX;
    int id_minEChange = -1;
    for (int ecI = 0; ecI < energyChanges.size(); ecI++) {
        if ((energyChanges[ecI].first == DBL_MAX) ||
            (energyChanges[ecI].second == DBL_MAX))
            continue;
        double EwChange = energyChanges[ecI].first * (1.0 - lambda) +
                          energyChanges[ecI].second * lambda;
        if (EwChange < bestEChange) {
            bestEChange = EwChange;
            id_minEChange = ecI;
        }
    }

    return id_minEChange;
}

// with a pinned border there may be no boundary-split candidates at all
bool hasValidCand(const std::vector<std::pair<double, double>> &energyChanges) {
    for (const auto &candI : energyChanges) {
        if ((candI.first != DBL_MAX) && (candI.second != DBL_MAX))
            return true;
    }
    return false;
}

bool checkCand(const std::vector<std::pair<double, double>> &energyChanges) {
    for (const auto &candI : energyChanges) {
        if ((candI.first < 0.0) || (candI.second < 0.0))
            return true;
    }
    double minEChange = DBL_MAX;
    for (const auto &candI : energyChanges) {
        if (candI.first < minEChange)
            minEChange = candI.first;
        if (candI.second < minEChange)
            minEChange = candI.second;
    }
    // DISABLE std::cout << "candidates not valid, minEChange: " << minEChange
    // << std::endl;

    return false;
}

double updateLambda(double measure_bound, double lambda_SD = energyParams[0],
                    double kappa = 1.0, double kappa2 = 1.0) {
    lambda_SD =
        (std::max)(0.0, kappa * (measure_bound -
                                 (upperBound - convTol_upperBound / 2.0)) +
                            kappa2 * lambda_SD / (1.0 - lambda_SD));
    return lambda_SD / (1.0 + lambda_SD);
}

bool updateLambda_stationaryV(bool cancelMomentum = true,
                              bool checkConvergence = false) {
    // splits and merges only re-index the soup, rest edge lengths never change
    if (std::isnan(minRestEdgeLength)) {
        Eigen::MatrixXd edgeLengths;
        igl::edge_lengths(triSoup[channel_result]->V_rest,
                          triSoup[channel_result]->F, edgeLengths);
        minRestEdgeLength = edgeLengths.minCoeff();
    }
    const double eps_E_se =
        1.0e-3 * minRestEdgeLength / triSoup[channel_result]->virtualRadius;

    // measurement and energy value computation
    const double E_SD = optimizer->getLastEnergyVal(true) / energyParams[0];
    double E_se;
    triSoup[channel_result]->computeSeamSparsity(E_se);
    E_se /= triSoup[channel_result]->virtualRadius;
    double measure_bound = E_SD;
    const double eps_lambda =
        (std::min)(1.0e-3,
                   std::abs(updateLambda(measure_bound) - energyParams[0]));

    // TODO?: stop when first violates bounds from feasible, don't go to best
    // feasible. check after each merge whether distortion is violated
    //  oscillation detection
    if (iterNum != lastStationaryIterNum) {
        // not a roll back config
        const double lambda = 1.0 - energyParams[0];
        bool oscillate = false;
        const auto low = configs_stationaryV.lower_bound(E_se);
        if (low == configs_stationaryV.end()) {
            // all less than E_se
            if (!configs_stationaryV.empty()) {
                // use largest element
                if (std::abs(configs_stationaryV.rbegin()->first - E_se) <
                    eps_E_se) {
                    for (const auto &lambdaI :
                         configs_stationaryV.rbegin()->second) {
                        if ((std::abs(lambdaI.first - lambda) < eps_lambda) &&
                            (std::abs(lambdaI.second - E_SD) < eps_E_se)) {
                            oscillate = true;
                            // DISABLE logFile <<
                            // configs_stationaryV.rbegin()->first << ", " <<
                            // lambdaI.second << std::endl; DISABLE logFile <<
                            // E_se << ", " << lambda << ", " << E_SD <<
                            // std::endl;
                            break;
                        }
                    }
                }
            }
        } else if (low == configs_stationaryV.begin()) {
            // all not less than E_se
            if (std::abs(low->first - E_se) < eps_E_se) {
                for (const auto &lambdaI : low->second) {
                    if ((std::abs(lambdaI.first - lambda) < eps_lambda) &&
                        (std::abs(lambdaI.second - E_SD) < eps_E_se)) {
                        oscillate = true;
                        // DISABLE logFile << low->first << ", " <<
                        // lambdaI.first << ", " << lambdaI.second << std::endl;
                        // DISABLE logFile << E_se << ", " << lambda << ", " <<
                        // E_SD << std::endl;
                        break;
                    }
                }
            }
        } else {
            const auto prev = std::prev(low);
            if (std::abs(low->first - E_se) < eps_E_se) {
                for (const auto &lambdaI : low->second) {
                    if ((std::abs(lambdaI.first - lambda) < eps_lambda) &&
                        (std::abs(lambdaI.second - E_SD) < eps_E_se)) {
                        oscillate = true;
                        // DISABLE logFile << low->first << ", " <<
                        // lambdaI.first << ", " << lambdaI.second << std::endl;
                        // DISABLE logFile << E_se << ", " << lambda << ", " <<
                        // E_SD << std::endl;
                        break;
                    }
                }
            }
            if ((!oscillate) && (std::abs(prev->first - E_se) < eps_E_se)) {
                for (const auto &lambdaI : prev->second) {
                    if ((std::abs(lambdaI.first - lambda) < eps_lambda) &&
                        (std::abs(lambdaI.second - E_SD) < eps_E_se)) {
                        oscillate = true;
                        // DISABLE logFile << prev->first << ", " <<
                        // lambdaI.first << ", " << lambdaI.second << std::endl;
                        // DISABLE logFile << E_se << ", " << lambda << ", " <<
                        // E_SD << std::endl;
                        break;
                    }
                }
            }
        }
        // record best feasible UV map
        if ((measure_bound <= upperBound) && (E_se < E_se_bestFeasible)) {
            iterNum_bestFeasible = iterNum;
            triSoup_bestFeasible = *triSoup[channel_result];
            E_se_bestFeasible = E_se;
        }
        if (oscillate && (iterNum_bestFeasible >= 0)) {
            // arrive at the best feasible config again
            // DISABLE logFile << "oscillation detected at measure = " <<
            // measure_bound << ", b = " << upperBound <<
            //    "lambda = " << energyParams[0] << std::endl;
            // DISABLE logFile << lastStationaryIterNum << ", " << iterNum <<
            // std::endl;
            if (iterNum_bestFeasible != iterNum) {
                optimizer->setConfig(triSoup_bestFeasible, iterNum,
                                     optimizer->getTopoIter());
                // DISABLE logFile << "rolled back to best feasible in iter " <<
                // iterNum_bestFeasible << std::endl;
            }
            return false;
        } else if (oscillate) {
            // revisiting the same stationary state means the split/merge pair is cycling
            if (++oscillated_infeasible >= 3)
                return false;
        } else {
            configs_stationaryV[E_se].emplace_back(
                std::pair<double, double>(lambda, E_SD));
        }
    }
    lastStationaryIterNum = iterNum;
    // convergence check
    if (checkConvergence) {
        if (measure_bound <= upperBound) {
            if (measure_bound >= upperBound - convTol_upperBound) {
                // DISABLE logFile << "all converged at measure = " <<
                // measure_bound << ", b = " << upperBound <<
                //    " lambda = " << energyParams[0] << std::endl;
                if (iterNum_bestFeasible != iterNum) {
                    assert(iterNum_bestFeasible >= 0);
                    optimizer->setConfig(triSoup_bestFeasible, iterNum,
                                         optimizer->getTopoIter());
                    // DISABLE logFile << "rolled back to best feasible in iter
                    // " << iterNum_bestFeasible << std::endl;
                }
                return false;
            }
        }
    }

    // lambda update (dual update)
    energyParams[0] = updateLambda(measure_bound);
    // TODO: needs to be careful on lambda update space

    // critical lambda scheme
    if (checkConvergence) {
        // update lambda until feasible update on T might be triggered
        if (measure_bound > upperBound) {
            // need to cut further, increase energyParams[0]
            // DISABLE logFile << "curUpdated = " << energyParams[0] << ",
            // increase" << std::endl;
            if ((!energyChanges_merge.empty()) &&
                hasValidCand(energyChanges_bSplit) &&
                (computeOptPicked(energyChanges_bSplit, energyChanges_merge,
                                  1.0 - energyParams[0]) == 1)) {
                // still picking merge, the dual update saturates and x/(1+x) sticks at 1
                double lambda_last = -1.0;
                do {
                    energyParams[0] = updateLambda(measure_bound);
                    if (energyParams[0] == lambda_last)
                        break;
                    lambda_last = energyParams[0];
                } while (
                    (computeOptPicked(energyChanges_bSplit, energyChanges_merge,
                                      1.0 - energyParams[0]) == 1));
                // DISABLE logFile << "iterativelyUpdated = " << energyParams[0]
                // << ", increase for switch" << std::endl;
            }

            if (!checkCand(energyChanges_iSplit) &&
                !checkCand(energyChanges_bSplit)) {
                // if filtering too strong
                reQuery = true;
                // DISABLE logFile << "enlarge filtering!" << std::endl;
            } else {
                double eDec_b, eDec_i;
                assert(!(energyChanges_bSplit.empty() &&
                         energyChanges_iSplit.empty()));
                int id_pickingBSplit = computeBestCand(
                    energyChanges_bSplit, 1.0 - energyParams[0], eDec_b);
                int id_pickingISplit = computeBestCand(
                    energyChanges_iSplit, 1.0 - energyParams[0], eDec_i);
                // pins can leave no split profitable at any lambda
                double lambda_last = -1.0;
                while ((eDec_b > 0.0) && (eDec_i > 0.0)) {
                    if (energyParams[0] == lambda_last)
                        break;
                    lambda_last = energyParams[0];
                    energyParams[0] = updateLambda(measure_bound);
                    id_pickingBSplit = computeBestCand(
                        energyChanges_bSplit, 1.0 - energyParams[0], eDec_b);
                    id_pickingISplit = computeBestCand(
                        energyChanges_iSplit, 1.0 - energyParams[0], eDec_i);
                }
                if (id_pickingBSplit < 0 && id_pickingISplit < 0) {
                    // no pickable split at all, widen the filter instead
                    reQuery = true;
                } else if ((id_pickingISplit < 0) || (eDec_b <= 0.0) ||
                           ((id_pickingBSplit >= 0) && (eDec_b <= eDec_i))) {
                    opType_queried = 0;
                    path_queried = paths_bSplit[id_pickingBSplit];
                    newVertPos_queried = newVertPoses_bSplit[id_pickingBSplit];
                } else {
                    opType_queried = 1;
                    path_queried = paths_iSplit[id_pickingISplit];
                    newVertPos_queried = newVertPoses_iSplit[id_pickingISplit];
                }
                // DISABLE logFile << "iterativelyUpdated = " << energyParams[0]
                // << ", increased, current eDec = " <<
                //   eDec_b << ", " << eDec_i << "; id: " << id_pickingBSplit <<
                //   ", " << id_pickingISplit << std::endl;
            }
        } else {
            bool noOp = true;
            for (const auto ecI : energyChanges_merge) {
                if (ecI.first != DBL_MAX) {
                    noOp = false;
                    break;
                }
            }
            if (noOp) {
                // DISABLE logFile << "No merge operation available, end
                // process!" << std::endl;
                energyParams[0] = 1.0 - eps_lambda;
                optimizer->updateEnergyData(true, false, false);
                if (iterNum_bestFeasible != iterNum)
                    optimizer->setConfig(triSoup_bestFeasible, iterNum,
                                         optimizer->getTopoIter());
                return false;
            }
            // DISABLE logFile << "curUpdated = " << energyParams[0] << ",
            // decrease" << std::endl;
            //!!! also account for iSplit for this switch?
            if (hasValidCand(energyChanges_bSplit) &&
                computeOptPicked(energyChanges_bSplit, energyChanges_merge,
                                 1.0 - energyParams[0]) == 0) {
                // still picking split, break at the dual update's fixed point
                double lambda_last = -1.0;
                do {
                    energyParams[0] = updateLambda(measure_bound);
                    if (energyParams[0] == lambda_last)
                        break;
                    lambda_last = energyParams[0];
                } while (computeOptPicked(energyChanges_bSplit,
                                          energyChanges_merge,
                                          1.0 - energyParams[0]) == 0);

                // DISABLE logFile << "iterativelyUpdated = " << energyParams[0]
                // << ", decrease for switch" << std::endl;
            }

            double eDec_m;
            assert(!energyChanges_merge.empty());
            int id_pickingMerge = computeBestCand(
                energyChanges_merge, 1.0 - energyParams[0], eDec_m);
            // break at the dual update's fixed point
            double lambda_last = -1.0;
            while (eDec_m > 0.0) {
                if (energyParams[0] == lambda_last)
                    break;
                lambda_last = energyParams[0];
                energyParams[0] = updateLambda(measure_bound);
                id_pickingMerge = computeBestCand(
                    energyChanges_merge, 1.0 - energyParams[0], eDec_m);
            }
            if (id_pickingMerge < 0) {
                // a merge can be listed but unpickable, a partial DBL_MAX sentinel
                energyParams[0] = 1.0 - eps_lambda;
                optimizer->updateEnergyData(true, false, false);
                if (iterNum_bestFeasible != iterNum)
                    optimizer->setConfig(triSoup_bestFeasible, iterNum,
                                         optimizer->getTopoIter());
                return false;
            }
            opType_queried = 2;
            path_queried = paths_merge[id_pickingMerge];
            newVertPos_queried = newVertPoses_merge[id_pickingMerge];

            // DISABLE logFile << "iterativelyUpdated = " << energyParams[0] <<
            // ", decreased, current eDec = " << eDec_m << std::endl;
        }
    }
    // lambda value sanity check
    if (energyParams[0] > 1.0 - eps_lambda)
        energyParams[0] = 1.0 - eps_lambda;
    if (energyParams[0] < eps_lambda)
        energyParams[0] = eps_lambda;

    optimizer->updateEnergyData(true, false, false);

    // DISABLE logFile << "measure = " << measure_bound << ", b = " <<
    // upperBound << ", updated lambda = " << energyParams[0] << std::endl;
    return true;
}

void converge_preDrawFunc(void) {
    reportProgress();
    optimization_on = false;
    // std::cout << "optimization converged, in " << secPast << "s." <<
    // std::endl;
    outerLoopFinished = true;
}

// a dead solve decreases by exactly zero, a healthy one never below ~4e-8
const int SOLVE_STALL_ITERATION_CAP = 100;
const double SOLVE_STALL_RELATIVE_TOLERANCE = 1.0e-12;

// a window of 1 misses a split alternating with its own merge
const int NO_PROGRESS_ROUNDS = 50;
const size_t REVISIT_WINDOW = 8;

bool preDrawFunc(void) {
    if (optimization_on) {
        while (!converged) {
            proceedOptimization(1);
            // a stop during a long solve must not wait for convergence
            if (forceQuit)
                // postDrawFunc saves the current map and exits
                return false;
            if (snapshot)
                reportProgress();
            const double energy = optimizer->getLastEnergyVal(true);
            if (std::abs(energy - stalledSolveEnergy) <=
                SOLVE_STALL_RELATIVE_TOLERANCE * std::abs(stalledSolveEnergy)) {
                if (++stalledSolveIterations >= SOLVE_STALL_ITERATION_CAP)
                    converged = 1;
            } else {
                stalledSolveIterations = 0;
                stalledSolveEnergy = energy;
            }
        }
        stalledSolveIterations = 0;
        stalledSolveEnergy = -1.0;
        reportProgress();

        // give postDraw option to save mesh
        canSaveMesh = true;

        double measure_bound =
            optimizer->getLastEnergyVal(true) / energyParams[0];
        if (converged == 2) {
            converged = 0;
            return false;
        }
        // if necessary, turn on scaffolding for random one point initial cut
        if (!optimizer->isScaffolding() && rand1PInitCut)
            optimizer->setScaffolding(true);

        // everything past this point queries cuts
        if (noCutMode) {
            converge_preDrawFunc();
            return false;
        }

        // re-converge between placements so the zip and relaxation settle
        if (stitchMode) {
            bool changed = optimizer->zipStitched();
            if (optimizer->stitchIslands())
                changed = true;
            if (changed)
                converged = 0;
            else
                converge_preDrawFunc();
            return false;
        }

        // a pinned border leaves no productive merge for the full search to make
        if (pinnedMode && measure_bound <= upperBound) {
            converge_preDrawFunc();
            return false;
        }

        double E_se;
        triSoup[channel_result]->computeSeamSparsity(E_se);
        E_se /= triSoup[channel_result]->virtualRadius;

        if (pinnedMode && ++stationaryCount > 500) {
            converge_preDrawFunc();
            return false;
        }

        // distortion stays out, the lambda renormalization wobbles it on a frozen map
        const Eigen::Index V_now = triSoup[channel_result]->V_rest.rows();
        bool revisited = false;
        for (const auto &seamSet : recentSeamSets) {
            if (std::abs(E_se - seamSet.first) <=
                    1.0e-9 * std::abs(seamSet.first) &&
                V_now == seamSet.second) {
                revisited = true;
                break;
            }
        }
        if (revisited) {
            // lambda creep per frozen round is too small to flip a pick pins blocked
            if (++noProgressCount >= (pinnedMode ? 3 : NO_PROGRESS_ROUNDS)) {
                converge_preDrawFunc();
                return false;
            }
        } else {
            noProgressCount = 0;
            recentSeamSets.emplace_back(E_se, V_now);
            if (recentSeamSets.size() > REVISIT_WINDOW)
                recentSeamSets.pop_front();
        }

        // continue to split boundary
        if (!updateLambda_stationaryV()) {
            // oscillation detected
            converge_preDrawFunc();
        } else {
            // DISABLE logFile << "boundary op V " <<
            // triSoup[channel_result]->V_rest.rows() << std::endl;
            if (optimizer->createFracture(fracThres, false, topoLineSearch)) {
                converged = 0;
            } else {
                // if no boundary op, try interior split if split is the current
                // best boundary op
                if ((measure_bound > upperBound) &&
                    optimizer->createFracture(fracThres, false, topoLineSearch,
                                              true)) {
                    // DISABLE logFile << "interior split " <<
                    // triSoup[channel_result]->V_rest.rows() << std::endl;
                    converged = 0;
                } else {
                    if (!updateLambda_stationaryV(false, true)) {
                        // all converged
                        converge_preDrawFunc();
                    } else {
                        // split or merge after lambda update
                        if (reQuery) {
                            bool found = false;
                            do {
                                // log(0) and log(1) would make this step 0 or inf
                                if (inSplitTotalAmt >= 2) {
                                    filterExp_in +=
                                        std::log(2.0) /
                                        std::log(inSplitTotalAmt);
                                    filterExp_in =
                                        (std::min)(1.0, filterExp_in);
                                } else {
                                    filterExp_in = 1.0;
                                }
                                found = optimizer->createFracture(
                                    fracThres, false, topoLineSearch, true);
                            } while (!found && filterExp_in < 1.0);
                            reQuery = false;
                            // TODO: set filtering param back?
                            if (!found) {
                                // a pinned border can leave nothing left to split
                                converge_preDrawFunc();
                                return false;
                            }
                        } else {
                            optimizer->createFracture(
                                opType_queried, path_queried,
                                newVertPos_queried, topoLineSearch);
                        }
                        opType_queried = -1;
                        converged = 0;
                    }
                }
            }
        }
    }
    return false;
}

static std::vector<float> split(const std::string &str, char sep) {
    std::vector<float> tokens;

    float i;
    std::stringstream ss(str);
    while (ss >> i) {
        tokens.push_back(i);
        if (ss.peek() == sep) {
            ss.ignore();
        }
    }

    return tokens;
}

// reads a one-line "index,weight,..." sidecar, false when the file doesn't exist
static bool loadWeightSidecar(const std::string &filePath,
                              Eigen::VectorXd &out) {
    std::ifstream file(filePath);
    if (!file.is_open())
        return false;
    std::string line;
    getline(file, line);
    std::vector<float> tokens = split(line, ',');
    for (uint32_t i = 0; i + 1 < tokens.size(); i += 2) {
        const int selected = (int)tokens[i];
        if (selected >= 0 && selected < out.size())
            out[selected] = tokens[i + 1];
    }
    return true;
}

// a chart is disk-topology when its euler characteristic is 1
static std::vector<bool> chartDiskFlags(const Eigen::MatrixXi &F,
                                        int n_components,
                                        const Eigen::VectorXi &C) {
    std::vector<std::set<int>> verts(n_components);
    std::vector<std::set<std::pair<int, int>>> edges(n_components);
    std::vector<int> faces(n_components, 0);
    for (int triI = 0; triI < F.rows(); ++triI) {
        int c = C[triI];
        ++faces[c];
        for (int i = 0; i < 3; ++i) {
            int a = F(triI, i), b = F(triI, (i + 1) % 3);
            verts[c].insert(a);
            edges[c].insert(std::pair<int, int>(std::min(a, b), std::max(a, b)));
        }
    }
    std::vector<bool> isDisk(n_components);
    for (int c = 0; c < n_components; ++c) {
        isDisk[c] = static_cast<int>(verts[c].size()) -
                        static_cast<int>(edges[c].size()) + faces[c] ==
                    1;
    }
    return isDisk;
}

// the input arrives packed, so measure at the minimum of s^2*grow + shrink/s^2
static double importedMapMeasure(const uvgami::TriMesh &mesh) {
    double grow = 0.0, shrink = 0.0, total = 0.0;
    for (int triI = 0; triI < mesh.F.rows(); ++triI) {
        const Eigen::RowVector3i &tri = mesh.F.row(triI);
        const Eigen::RowVector3d e1 =
            mesh.V_rest.row(tri[1]) - mesh.V_rest.row(tri[0]);
        const Eigen::RowVector3d e2 =
            mesh.V_rest.row(tri[2]) - mesh.V_rest.row(tri[0]);
        const double l1 = e1.norm();
        const double area = e1.cross(e2).norm() / 2;
        // a zero-area rest triangle has no distortion to measure
        if (area <= 0.0)
            continue;
        const double x2 = e1.dot(e2) / l1;
        const double y2 = 2 * area / l1;
        const Eigen::RowVector2d u1 = mesh.V.row(tri[1]) - mesh.V.row(tri[0]);
        const Eigen::RowVector2d u2 = mesh.V.row(tri[2]) - mesh.V.row(tri[0]);
        const double a = u1[0] / l1;
        const double b = (u2[0] - x2 * a) / y2;
        const double c = u1[1] / l1;
        const double d = (u2[1] - x2 * c) / y2;
        const double det = a * d - b * c;
        if (det <= 0.0)
            return DBL_MAX;
        const double frob2 = a * a + b * b + c * c + d * d;
        const double weighted = area * mesh.faceWeight[triI];
        grow += weighted * frob2;
        shrink += weighted * frob2 / (det * det);
        total += area;
    }
    if (total <= 0.0)
        return DBL_MAX;
    return 2 * std::sqrt(grow * shrink) / total;
}

// partedFacePairs are never joined, a vertex whose fans span two components is left alone
static int splitBowtieVertices(
    Eigen::MatrixXd &V, Eigen::MatrixXi &F, const Eigen::VectorXi &triComponent,
    std::vector<bool> &bowtieComponent,
    const std::set<std::pair<int, int>> &partedFacePairs = {}) {
    std::vector<std::vector<int>> vertTris(V.rows());
    for (int triI = 0; triI < F.rows(); ++triI) {
        for (int i = 0; i < 3; ++i) {
            vertTris[F(triI, i)].emplace_back(triI);
        }
    }
    int bowtieAmt = 0;
    const int vAmt = static_cast<int>(vertTris.size());
    for (int vI = 0; vI < vAmt; ++vI) {
        const std::vector<int> &tris = vertTris[vI];
        if (tris.size() < 2) {
            continue;
        }
        // fans join when two incident triangles share an edge through vI
        std::map<int, std::vector<int>> edgeTris;
        for (const auto triI : tris) {
            for (int i = 0; i < 3; ++i) {
                if (F(triI, i) != vI) {
                    edgeTris[F(triI, i)].emplace_back(triI);
                }
            }
        }
        std::set<int> left(tris.begin(), tris.end());
        std::vector<std::vector<int>> fans;
        while (!left.empty()) {
            std::vector<int> fan({*left.begin()});
            left.erase(fan[0]);
            for (size_t fanI = 0; fanI < fan.size(); ++fanI) {
                for (int i = 0; i < 3; ++i) {
                    const int u = F(fan[fanI], i);
                    if (u == vI) {
                        continue;
                    }
                    if (edgeTris[u].size() != 2) {
                        continue;
                    }
                    const int lowTriI = (std::min)(edgeTris[u][0], edgeTris[u][1]);
                    const int highTriI = (std::max)(edgeTris[u][0], edgeTris[u][1]);
                    if (partedFacePairs.count({lowTriI, highTriI})) {
                        continue;
                    }
                    for (const auto nbTriI : edgeTris[u]) {
                        if (left.erase(nbTriI)) {
                            fan.emplace_back(nbTriI);
                        }
                    }
                }
            }
            fans.emplace_back(fan);
        }
        if (fans.size() < 2) {
            continue;
        }
        if (triComponent.size()) {
            const int component = triComponent[fans[0][0]];
            bool oneComponent = true;
            for (const auto &fan : fans) {
                oneComponent = oneComponent && triComponent[fan[0]] == component;
            }
            if (!oneComponent) {
                continue;
            }
            bowtieComponent[component] = true;
        }
        for (size_t fanI = 1; fanI < fans.size(); ++fanI) {
            const int nV = static_cast<int>(V.rows());
            V.conservativeResize(nV + 1, V.cols());
            V.row(nV) = V.row(vI);
            for (const auto triI : fans[fanI]) {
                for (int i = 0; i < 3; ++i) {
                    if (F(triI, i) == vI) {
                        F(triI, i) = nV;
                    }
                }
            }
            ++bowtieAmt;
        }
    }
    return bowtieAmt;
}

// each pair low index first
static std::set<std::pair<int, int>>
sameDirectionFacePairs(const Eigen::MatrixXi &F) {
    std::map<std::pair<int, int>, int> directedEdgeFace;
    std::set<std::pair<int, int>> pairs;
    for (int triI = 0; triI < F.rows(); ++triI) {
        for (int i = 0; i < 3; ++i) {
            const auto inserted = directedEdgeFace.emplace(
                std::make_pair(F(triI, i), F(triI, (i + 1) % 3)), triI);
            if (!inserted.second) {
                pairs.emplace(inserted.first->second, triI);
            }
        }
    }
    return pairs;
}

// one cut round has cleared every twisted piece seen so far
static const int WINDING_CUT_ROUNDS = 3;

// keeps the side most of a component's faces had, igl::bfs_orient re-flips visited ones
static int orientComponents(Eigen::MatrixXi &F) {
    std::map<std::pair<int, int>, std::vector<int>> edgeTris;
    for (int triI = 0; triI < F.rows(); ++triI) {
        for (int i = 0; i < 3; ++i) {
            const int a = F(triI, i);
            const int b = F(triI, (i + 1) % 3);
            edgeTris[{(std::min)(a, b), (std::max)(a, b)}].emplace_back(triI);
        }
    }
    const auto hasDirectedEdge = [&F](int triI, int a, int b) {
        for (int i = 0; i < 3; ++i) {
            if (F(triI, i) == a && F(triI, (i + 1) % 3) == b) {
                return true;
            }
        }
        return false;
    };
    const int UNVISITED = -1;
    std::vector<int> reversed(F.rows(), UNVISITED);
    int reorientedAmt = 0;
    for (int seed = 0; seed < F.rows(); ++seed) {
        if (reversed[seed] != UNVISITED) {
            continue;
        }
        reversed[seed] = 0;
        std::vector<int> component{seed};
        for (size_t queueI = 0; queueI < component.size(); ++queueI) {
            const int triI = component[queueI];
            for (int i = 0; i < 3; ++i) {
                int a = F(triI, i);
                int b = F(triI, (i + 1) % 3);
                if (reversed[triI]) {
                    std::swap(a, b);
                }
                const auto &tris =
                    edgeTris[{(std::min)(a, b), (std::max)(a, b)}];
                if (tris.size() != 2) {
                    continue;
                }
                const int nbTriI = tris[0] == triI ? tris[1] : tris[0];
                if (reversed[nbTriI] != UNVISITED) {
                    continue;
                }
                reversed[nbTriI] = hasDirectedEdge(nbTriI, a, b);
                component.emplace_back(nbTriI);
            }
        }
        int reversedAmt = 0;
        for (const auto triI : component) {
            reversedAmt += reversed[triI];
        }
        const int size = static_cast<int>(component.size());
        if (reversedAmt * 2 > size) {
            for (const auto triI : component) {
                reversed[triI] = !reversed[triI];
            }
            reversedAmt = size - reversedAmt;
        }
        reorientedAmt += reversedAmt;
    }
    for (int triI = 0; triI < F.rows(); ++triI) {
        if (reversed[triI]) {
            F.row(triI) = F.row(triI).reverse().eval();
        }
    }
    return reorientedAmt;
}

// hop count is the distance because that is what a tutte map rounds away
static std::vector<int> deepestPath(const Eigen::MatrixXi &F_component) {
    std::map<std::pair<int, int>, int> edgeUse;
    std::map<int, std::vector<int>> adjacency;
    for (int triI = 0; triI < F_component.rows(); ++triI) {
        for (int i = 0; i < 3; ++i) {
            int a = F_component(triI, i), b = F_component(triI, (i + 1) % 3);
            if (++edgeUse[{(std::min)(a, b), (std::max)(a, b)}] == 1) {
                adjacency[a].emplace_back(b);
                adjacency[b].emplace_back(a);
            }
        }
    }
    std::map<int, int> parent;
    std::deque<int> queue;
    for (const auto &[edge, uses] : edgeUse) {
        if (uses == 1) {
            for (int vI : {edge.first, edge.second}) {
                if (parent.emplace(vI, -1).second)
                    queue.emplace_back(vI);
            }
        }
    }
    int deepest = -1;
    while (!queue.empty()) {
        deepest = queue.front();
        queue.pop_front();
        for (int nbI : adjacency[deepest]) {
            if (parent.emplace(nbI, deepest).second)
                queue.emplace_back(nbI);
        }
    }
    std::vector<int> path;
    for (int vI = deepest; vI >= 0; vI = parent[vI])
        path.emplace_back(vI);
    std::reverse(path.begin(), path.end());
    return path;
}

// cutPath opens an interior path only while its middle stays clear of the boundary
static std::vector<std::vector<int>> seamSegments(const uvgami::TriMesh &mesh,
                                                  std::vector<int> seam) {
    if (seam.size() < 2)
        return {};
    const bool closed = seam.front() == seam.back();
    if (closed)
        seam.pop_back();
    const auto firstBoundary =
        std::find_if(seam.begin(), seam.end(),
                     [&](int vI) { return mesh.isBoundaryVert(vI); });
    if (firstBoundary == seam.end()) {
        if (closed)
            seam.emplace_back(seam.front());
        return {seam};
    }
    if (closed) {
        std::rotate(seam.begin(), firstBoundary, seam.end());
        seam.emplace_back(seam.front());
    }
    std::vector<std::vector<int>> segments;
    std::vector<int> segment;
    for (int vI : seam) {
        segment.emplace_back(vI);
        if (mesh.isBoundaryVert(vI) && segment.size() >= 2) {
            segments.emplace_back(segment);
            segment = {vI};
        }
    }
    if (segment.size() >= 2)
        segments.emplace_back(segment);
    return segments;
}

// cut_to_disk can return only boundary edges, which cutPath skips
static std::vector<int> boundaryLoopConnector(const uvgami::TriMesh &mesh,
                                              const Eigen::MatrixXi &F_component) {
    std::vector<std::vector<int>> loops;
    igl::boundary_loop(F_component, loops);
    if (loops.size() < 2)
        return {};

    std::map<int, std::vector<int>> adjacency;
    for (int triI = 0; triI < F_component.rows(); ++triI) {
        for (int i = 0; i < 3; ++i) {
            int a = F_component(triI, i), b = F_component(triI, (i + 1) % 3);
            if (mesh.edge2Tri.count({a, b}) && mesh.edge2Tri.count({b, a})) {
                adjacency[a].emplace_back(b);
                adjacency[b].emplace_back(a);
            }
        }
    }

    std::set<int> firstLoop(loops[0].begin(), loops[0].end());
    std::map<int, int> parent;
    std::vector<int> queue;
    for (int vI : loops[0]) {
        parent[vI] = vI;
        queue.emplace_back(vI);
    }
    for (size_t head = 0; head < queue.size(); ++head) {
        for (int next : adjacency[queue[head]]) {
            if (parent.emplace(next, queue[head]).second)
                queue.emplace_back(next);
        }
    }

    for (size_t loopI = 1; loopI < loops.size(); ++loopI) {
        for (int vI : loops[loopI]) {
            if (firstLoop.count(vI) || !parent.count(vI))
                continue;
            std::vector<int> path{vI};
            while (parent[path.back()] != path.back())
                path.emplace_back(parent[path.back()]);
            return path;
        }
    }
    return {};
}

// an escaped exception fast-fails with no message otherwise
static void reportTerminate() {
    if (auto e = std::current_exception()) {
        try {
            std::rethrow_exception(e);
        } catch (const std::exception &ex) {
            std::cerr << "fatal: " << ex.what() << std::endl;
        } catch (...) {
            std::cerr << "fatal: non-std exception" << std::endl;
        }
    } else {
        std::cerr << "fatal: terminate without exception" << std::endl;
    }
    std::_Exit(90);
}

static std::string meshStem(const std::string &meshFilePath) {
    const std::string fileName =
        meshFilePath.substr(meshFilePath.find_last_of(pathSeparator()) + 1);
    return fileName.substr(0, fileName.find_last_of('.'));
}

// the mesh name is appended to outputFolderPath later
static int prepareOutputFolder(const std::string &meshFilePath) {
    if (outputDirArg.empty()) {
        const std::filesystem::path inputFolderPath =
            std::filesystem::path(meshFilePath).parent_path();
        outputFolderPath =
            std::string(inputFolderPath.parent_path().u8string()) +
            pathSeparator() + "output" + pathSeparator();
    } else {
        outputFolderPath = outputDirArg;
    }
    if (!std::filesystem::exists(outputFolderPath) &&
        !std::filesystem::create_directory(outputFolderPath)) {
        printf("Failed to create output directory %s\n",
               outputFolderPath.c_str());
        return -1;
    }
    return 0;
}

static int unwrapMeshOrThrow(const std::string &meshFilePath, bool ignoreUV);

// a rest mesh collapsing to a point throws from TriMesh construction
static int unwrapMesh(const std::string &meshFilePath, bool ignoreUV) {
    try {
        return unwrapMeshOrThrow(meshFilePath, ignoreUV);
    } catch (UvgamiElementInversionException &) {
        return UVGAMI_RC_ELEMENT_INVERSION;
    }
}

int main(int argc, char *argv[]) {
    std::set_terminate(reportTerminate);
    // igl::parallel_for spawns a thread per core on every call
    igl::default_num_threads(1);
    std::string meshFileName;
    lambda_init = 0.999;
    bool ignoreUV = false;
    bool flattenMode = false;
    bool packOnlyMode = false;
    int flattenIters = 30;
    // the cap holds only while this exists
    std::unique_ptr<tbb::global_control> threadCap;

    try {
        TCLAP::CmdLine cmd("uvgami command line", ' ', UVGAMI_VERSION);
        TCLAP::ValueArg<std::string> inputArg(
            "i", "input",
            "Input mesh. Without it, mesh paths are read from stdin as "
            "\"unwrap <path>\" lines until it closes",
            false, "", "string", cmd);
        TCLAP::ValueArg<std::string> outputArg(
            "o", "output", "Output directory", false, "", "string", cmd);
        TCLAP::ValueArg<double> lambdaInitArg("L", "lambda_init",
                                              "Lambda initial value", false, 0,
                                              "double", cmd);
        TCLAP::ValueArg<double> upperBoundArg("u", "upper_bound", "Upper bound",
                                              false, 0, "double", cmd);
        TCLAP::ValueArg<uint32_t> maxSeamWeightArg("s", "max_seam_weight",
                                                   "Maximum seam weight", false,
                                                   0, "uint32_t", cmd);
        TCLAP::ValueArg<uint32_t> maxFaceWeightArg(
            "w", "max_face_weight", "Maximum face importance weight", false, 0,
            "uint32_t", cmd);
        TCLAP::SwitchArg ignoreUVArg("g", "ignore_uv", "Ignore UV map", cmd);
        TCLAP::SwitchArg flattenArg(
            "", "flatten",
            "Flatten and pack along the _seams sidecar, no optimization", cmd);
        TCLAP::ValueArg<int> flattenItersArg(
            "", "flatten_iters",
            "Most SLIM iterations per island in flatten mode, the solve stops "
            "early once the energy settles",
            false, 30, "int", cmd);
        TCLAP::SwitchArg packOnlyArg(
            "", "pack_only", "Repack the input UV map without solving", cmd);
        TCLAP::ValueArg<int> threadsArg(
            "t", "threads",
            "Most worker threads for the parallel loops, default every core",
            false, 0, "int", cmd);
        cmd.parse(argc, argv);

        if (threadsArg.getValue() > 0)
            threadCap = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism,
                static_cast<size_t>(threadsArg.getValue()));

        flattenMode = flattenArg.getValue() || packOnlyArg.getValue();
        packOnlyMode = packOnlyArg.getValue();
        if (flattenItersArg.getValue() > 0)
            flattenIters = flattenItersArg.getValue();

        if (maxSeamWeightArg.isSet())
            maxSeamWeight = maxSeamWeightArg.getValue();
        if (maxFaceWeightArg.isSet())
            maxFaceWeight = maxFaceWeightArg.getValue();
        if (ignoreUVArg.isSet())
            ignoreUV = ignoreUVArg.getValue();
        meshFileName = inputArg.getValue();
        if (outputArg.isSet())
            outputDirArg = outputArg.getValue();
        if (lambdaInitArg.isSet()) {
            lambda_init = lambdaInitArg.getValue();
            if (lambda_init < 0.0 || lambda_init >= 1.0)
                lambda_init = 0.999;
        }
        if (upperBoundArg.isSet())
            upperBound = upperBoundArg.getValue();
    } catch (TCLAP::ArgException &e) // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId()
                  << std::endl;
        return 1;
    }
    if (flattenMode) {
        if (meshFileName.empty()) {
            std::cerr << "error: flatten needs -i" << std::endl;
            return 1;
        }
        const int folderCode = prepareOutputFolder(meshFileName);
        if (folderCode != 0)
            return folderCode;
        return uvgami::runFlatten(meshFileName, outputFolderPath, flattenIters,
                                  packOnlyMode);
    }

    // returning from main destroys cin under the listener and crashes
    std::thread(&stdin_listener).detach();

    if (!meshFileName.empty()) {
        const int code = unwrapMesh(meshFileName, ignoreUV);
        releaseMesh();
        std::cout.flush();
        std::_Exit(code);
    }

    // the addon sends a path each time this process is idle
    for (std::string path = nextMeshPath(); !path.empty();
         path = nextMeshPath()) {
        const std::string stem = meshStem(path);
        std::cout << "start: " << stem << std::endl;
        const int code = unwrapMesh(path, ignoreUV);
        releaseMesh();
        if (code == UVGAMI_RC_SUCCESS)
            std::cout << "done: " << stem << std::endl;
        else
            std::cout << "failed: " << stem << " " << code << std::endl;
    }
    std::cout.flush();
    std::_Exit(0);
}

static int unwrapMeshOrThrow(const std::string &meshFilePath, bool ignoreUV) {
    resetMeshState();
    mainTimer.start();
    const int folderCode = prepareOutputFolder(meshFilePath);
    if (folderCode != 0)
        return folderCode;
    const std::filesystem::path inputFolderPath =
        std::filesystem::path(meshFilePath).parent_path();
    meshName = meshStem(meshFilePath);
    const std::string suffix =
        meshFilePath.substr(meshFilePath.find_last_of('.'));
    bool loadSucceed = false;
    if (suffix == ".off") {
        loadSucceed = igl::readOFF(meshFilePath, V, F);
    } else if (suffix == ".obj") {
        loadSucceed = igl::readOBJ(meshFilePath, V, UV, N, F, FUV, FN);
    } else {
        std::cout << "unkown mesh file format" << std::endl;
        return UVGAMI_RC_UNKNOWN_MESH_FORMAT;
    }
    if (!loadSucceed) {
        std::cerr << "failed to load mesh" << std::endl;
        return UVGAMI_RC_FAILED_TO_LOAD_MESH;
    }
    // nan or absurd coordinates hang the overlap grid and the solver
    if (V.rows() != 0 &&
        (!V.allFinite() || V.cwiseAbs().maxCoeff() > 1e15 ||
         (UV.rows() != 0 &&
          (!UV.allFinite() || UV.cwiseAbs().maxCoeff() > 1e15)))) {
        std::cerr << "input has non-finite or extreme coordinates" << std::endl;
        return UVGAMI_RC_INVALID_COORDS;
    }
    //    //DEBUG
    //    uvgami::TriMesh squareMesh(uvgami::P_SQUARE, 1.0, 0.1, false);
    //    V = squareMesh.V_rest;
    //    F = squareMesh.F;

    const bool hasUV = !ignoreUV && (UV.rows() != 0);
    if (!hasUV) {
        std::vector<bool> noComponentFlags;
        const int bowtieAmt =
            splitBowtieVertices(V, F, Eigen::VectorXi(), noComponentFlags);
        if (bowtieAmt) {
            std::cerr << "split " << bowtieAmt
                      << " vertices at bowties and non-manifold edges into "
                         "per-fan copies"
                      << std::endl;
        }
        vertAmt_input = V.rows();
        Eigen::VectorXi B;
        bool isManifoldVertices = igl::is_vertex_manifold(F, B);
        if (!isManifoldVertices) {
            std::cerr << "input mesh contains non-manifold vertices"
                      << std::endl;
            return UVGAMI_RC_NON_MANIFOLD_VERTICES;
        }
        bool isManifoldEdges = igl::is_edge_manifold(F);
        if (!isManifoldEdges) {
            std::cerr << "input mesh contains non-manifold edges" << std::endl;
            return UVGAMI_RC_NON_MANIFOLD_EDGES;
        }
        const int reorientedAmt = orientComponents(F);
        if (reorientedAmt) {
            std::cerr << "flipped " << reorientedAmt
                      << " faces to match their component" << std::endl;
        }
        // a sheet sewn to itself with a twist has no consistent winding
        std::set<std::pair<int, int>> partedFacePairs;
        auto twisted = sameDirectionFacePairs(F);
        for (int round = 0; round < WINDING_CUT_ROUNDS && !twisted.empty();
             ++round) {
            partedFacePairs.insert(twisted.begin(), twisted.end());
            const int cutAmt = splitBowtieVertices(
                V, F, Eigen::VectorXi(), noComponentFlags, partedFacePairs);
            std::cerr << "cut " << twisted.size()
                      << " edges whose faces cannot be wound alike, split "
                      << cutAmt << " vertices" << std::endl;
            orientComponents(F);
            twisted = sameDirectionFacePairs(F);
        }
        // a repeated directed edge corrupts the edge2Tri adjacency
        if (!twisted.empty()) {
            std::cerr << "input mesh has inconsistently oriented faces"
                      << std::endl;
            return UVGAMI_RC_FLIPPED_FACES;
        }
    }

    // a repeated uv corner puts a diagonal entry in igl's adjacency matrix
    if (hasUV) {
        int splitCorners = 0;
        for (int triI = 0; triI < FUV.rows(); ++triI) {
            for (int i = 0; i < 3; ++i) {
                for (int j = i + 1; j < 3; ++j) {
                    if (FUV(triI, i) == FUV(triI, j)) {
                        const int nV = static_cast<int>(UV.rows());
                        UV.conservativeResize(nV + 1, UV.cols());
                        UV.row(nV) = UV.row(FUV(triI, j));
                        FUV(triI, j) = nV;
                        ++splitCorners;
                    }
                }
            }
        }
        if (splitCorners)
            std::cerr << "split " << splitCorners
                      << " repeated uv corners into their own vertices"
                      << std::endl;
    }

    // with input UV the components are the UV charts
    Eigen::VectorXi C;
    igl::facet_components(hasUV ? FUV : F, C);
    int n_components = C.maxCoeff() + 1;

    // a chart pinched at a uv vertex reads as one short of a disk
    std::vector<bool> bowtieChart(n_components, false);
    if (hasUV) {
        const int bowtieAmt = splitBowtieVertices(UV, FUV, C, bowtieChart);
        if (bowtieAmt) {
            std::cerr << "split " << bowtieAmt
                      << " pinched uv vertices into per-fan copies"
                      << std::endl;
        }
    }

    uvgami::TriMesh temp =
        hasUV ? uvgami::TriMesh(V, F, UV, FUV, false)
              : uvgami::TriMesh(V, F, Eigen::MatrixXd(), Eigen::MatrixXi(),
                                false);

    // the initial cut below seeds from these weights
    std::string weightsFileName = std::string(inputFolderPath.u8string()) +
                                  pathSeparator() + meshName + "_weights";
    Eigen::VectorXd seamAvoidance = Eigen::VectorXd::Zero(temp.V.rows());
    loadWeightSidecar(weightsFileName, seamAvoidance);

    // read before the keep decision, stitch runs relax the disk requirement below
    std::string stitchFileName = std::string(inputFolderPath.u8string()) +
                                 pathSeparator() + meshName + "_stitch";
    stitchMode = std::ifstream(stitchFileName).is_open();

    // read before the cut-to-disk fallback so a pinned run fails with this reason
    std::set<int> fixedVerts;
    std::string fixedFileName = std::string(inputFolderPath.u8string()) +
                                pathSeparator() + meshName + "_fixed";
    std::ifstream fixedFile(fixedFileName);
    if (fixedFile.is_open()) {
        std::string line;
        getline(fixedFile, line);
        for (float token : split(line, ','))
            fixedVerts.insert((int)token);
        // an empty pin line is valid, a whole-island relax holds nothing
        if (getline(fixedFile, line)) {
            while (!line.empty() &&
                   (line.back() == '\r' || line.back() == '\n'))
                line.pop_back();
            noCutMode = line == "nocut";
        }
        fixedFile.close();
    }

    // a chart is kept when it has no flipped or overlapping triangles and is a disk
    std::vector<bool> keepChart(n_components, false);
    int keptCharts = 0;
    bool keepInputUV = false;
    if (hasUV) {
        std::vector<std::vector<int>> chartTris(n_components);
        for (int triI = 0; triI < temp.F.rows(); ++triI) {
            chartTris[C[triI]].emplace_back(triI);
        }

        std::vector<bool> isDisk = chartDiskFlags(temp.F, n_components, C);

        std::vector<std::vector<int>> bnd_all;
        igl::boundary_loop(temp.F, bnd_all);
        std::set<int> crossingVerts;
        uvgami::IglUtils::checkUVBoundaryOverlap(temp.V, bnd_all,
                                                 &crossingVerts);
        // a crossing condemns both charts it touches
        std::vector<bool> overlaps(n_components, false);
        for (int triI = 0; triI < temp.F.rows(); ++triI) {
            for (int i = 0; i < 3; ++i) {
                if (crossingVerts.count(temp.F(triI, i))) {
                    overlaps[C[triI]] = true;
                }
            }
        }

        // a pinched vertex belongs to two charts at once, pinning it pulls two ways
        std::vector<int> vertChart(temp.V.rows(), -1);
        std::vector<bool> pinched(n_components, false);
        for (int triI = 0; triI < temp.F.rows(); ++triI) {
            for (int i = 0; i < 3; ++i) {
                int &owner = vertChart[temp.F(triI, i)];
                if (owner == -1) {
                    owner = C[triI];
                } else if (owner != C[triI]) {
                    pinched[owner] = true;
                    pinched[C[triI]] = true;
                }
            }
        }

        std::vector<bool> inverted(n_components);
        for (int c = 0; c < n_components; ++c) {
            inverted[c] = !temp.checkInversion(true, chartTris[c]);
        }

        // a point boundary evades the crossing test
        std::vector<bool> degenerate(n_components, false);
        for (int triI = 0; triI < temp.F.rows(); ++triI) {
            const Eigen::RowVector3i &tri = temp.F.row(triI);
            const Eigen::RowVector2d e1 = temp.V.row(tri[1]) - temp.V.row(tri[0]);
            const Eigen::RowVector2d e2 = temp.V.row(tri[2]) - temp.V.row(tri[0]);
            if (e1[0] * e2[1] - e1[1] * e2[0] <= 0.0)
                degenerate[C[triI]] = true;
        }

        bool anyInversion = false, allDisks = true, anyDegenerate = false;
        for (int c = 0; c < n_components; ++c) {
            anyInversion = anyInversion || inverted[c];
            allDisks = allDisks && isDisk[c];
            anyDegenerate = anyDegenerate || degenerate[c];
        }
        // stitch is hole-safe, a pinched boundary is not, the corner air loop cannot represent it
        bool stitchKeepable = stitchMode || noCutMode;
        if (stitchKeepable && !allDisks) {
            std::map<std::pair<int, int>, int> edgeCount;
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                for (int i = 0; i < 3; ++i) {
                    int a = temp.F(triI, i), b = temp.F(triI, (i + 1) % 3);
                    edgeCount[{std::min(a, b), std::max(a, b)}]++;
                }
            }
            std::vector<int> bndDeg(temp.V.rows(), 0);
            for (const auto &ec : edgeCount) {
                if (ec.second == 1) {
                    bndDeg[ec.first.first]++;
                    bndDeg[ec.first.second]++;
                }
            }
            std::vector<bool> pinchFree(n_components, true);
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                for (int i = 0; i < 3; ++i) {
                    if (bndDeg[temp.F(triI, i)] > 2)
                        pinchFree[C[triI]] = false;
                }
            }
            for (int c = 0; c < n_components; ++c)
                stitchKeepable = stitchKeepable && pinchFree[c];
        }

        // whole-map decision first, a map kept before is still kept byte for byte
        bool anyBowtie = false;
        for (int c = 0; c < n_components; ++c) {
            anyBowtie = anyBowtie || bowtieChart[c];
        }
        keepInputUV = (allDisks || stitchKeepable) && !anyBowtie &&
                      !anyInversion && !anyDegenerate && crossingVerts.empty();

        int badInverted = 0, badOverlapping = 0, badNonDisk = 0,
            badDegenerate = 0;
        if (keepInputUV) {
            keepChart.assign(n_components, true);
            keptCharts = n_components;
        } else {
            for (int c = 0; c < n_components; ++c) {
                keepChart[c] = isDisk[c] && !bowtieChart[c] && !overlaps[c] &&
                               !inverted[c] && !pinched[c] && !degenerate[c];
                keptCharts += keepChart[c];
                if (inverted[c]) {
                    ++badInverted;
                } else if (degenerate[c]) {
                    ++badDegenerate;
                } else if (overlaps[c]) {
                    ++badOverlapping;
                } else if (!isDisk[c] || bowtieChart[c]) {
                    ++badNonDisk;
                }
            }
        }

        if (!keepInputUV && keptCharts == 0) {
            std::cout << (anyInversion             ? "local injectivity violated"
                          : !crossingVerts.empty() ? "self-intersecting UV islands"
                                                   : "charts are not disk-topology")
                      << " in input UV map, cutting to disk-topology and "
                         "applying Tutte's embedding..."
                      << std::endl;
        } else if (!keepInputUV) {
            std::cout << "kept " << keptCharts << " of " << n_components
                      << " input UV charts, re-cutting " << badInverted
                      << " inverted, " << badDegenerate << " degenerate, "
                      << badOverlapping << " self-intersecting, " << badNonDisk
                      << " not disk-topology" << std::endl;
        }
    }

    if (!fixedVerts.empty() && !keepInputUV) {
        std::cerr << "pinned vertices need the input UV map kept" << std::endl;
        return UVGAMI_RC_PINNED_UV_NOT_KEPT;
    }

    if (stitchMode && !keepInputUV) {
        std::cerr << "stitching needs the input UV map kept" << std::endl;
        return UVGAMI_RC_STITCH_UV_NOT_KEPT;
    }
    if (noCutMode && !keepInputUV) {
        std::cerr << "nocut needs the input UV map kept" << std::endl;
        return UVGAMI_RC_NOCUT_UV_NOT_KEPT;
    }
    if (!fixedVerts.empty()) {
        // pins block the global rescale the solver would otherwise start with
        double uvArea = 0.0, restArea = 0.0;
        for (int triI = 0; triI < temp.F.rows(); ++triI) {
            const Eigen::RowVector3i &tri = temp.F.row(triI);
            const Eigen::RowVector2d e1 = temp.V.row(tri[1]) - temp.V.row(tri[0]);
            const Eigen::RowVector2d e2 = temp.V.row(tri[2]) - temp.V.row(tri[0]);
            uvArea += std::abs(e1[0] * e2[1] - e1[1] * e2[0]) / 2;
            const Eigen::RowVector3d p0 = temp.V_rest.row(tri[0]);
            const Eigen::RowVector3d p1 = temp.V_rest.row(tri[1]);
            const Eigen::RowVector3d p2 = temp.V_rest.row(tri[2]);
            restArea += (p1 - p0).cross(p2 - p0).norm() / 2;
        }
        if (uvArea > 0.0 && restArea > 0.0)
            temp.V *= std::sqrt(restArea / uvArea);
    }

    uvgami::TriMesh *keptInputMesh = nullptr;
    if (keepInputUV) {
        keptInputMesh = new uvgami::TriMesh(temp);
        triSoup.emplace_back(keptInputMesh);
    } else {
        // in each pass, make one cut on each piece if needed, until all disk-topology
        Eigen::VectorXi pieceOf;
        int n_pieces = 0;
        std::vector<Eigen::MatrixXi> F_component;
        std::vector<std::set<int>> V_ind_component;
        // every pass cuts at least one edge open and an edge only cuts once
        int passLimit = 3 * static_cast<int>(temp.F.rows());
        while (true) {
            igl::facet_components(temp.F, pieceOf);
            n_pieces = pieceOf.maxCoeff() + 1;
            F_component.assign(n_pieces, Eigen::MatrixXi());
            V_ind_component.assign(n_pieces, std::set<int>());
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                F_component[pieceOf[triI]].conservativeResize(
                    F_component[pieceOf[triI]].rows() + 1, 3);
                F_component[pieceOf[triI]].bottomRows(1) = temp.F.row(triI);
                for (int i = 0; i < 3; ++i) {
                    V_ind_component[pieceOf[triI]].insert(temp.F(triI, i));
                }
            }

            std::vector<int> components_to_cut;
            for (int componentI = 0; componentI < n_pieces; ++componentI) {
                std::set<std::pair<int, int>> edges;
                for (int triI = 0; triI < F_component[componentI].rows();
                     ++triI) {
                    for (int i = 0; i < 3; ++i) {
                        int a = F_component[componentI](triI, i);
                        int b = F_component[componentI](triI, (i + 1) % 3);
                        edges.emplace((std::min)(a, b), (std::max)(a, b));
                    }
                }
                int EC = static_cast<int>(V_ind_component[componentI].size()) -
                         static_cast<int>(edges.size()) +
                         static_cast<int>(F_component[componentI].rows());
                if (EC < 1) {
                    // treat as higher-genus surfaces using cut_to_disk()
                    components_to_cut.emplace_back(-componentI - 1);
                } else if (EC == 2) {
                    // closed genus-0 surface
                    components_to_cut.emplace_back(componentI);
                } else if (EC != 1) {
                    std::cerr << "unsupported single-connected component"
                              << std::endl;
                    return UVGAMI_RC_UNSUPPORTED_TOPOLOGY;
                }
            }

            if (components_to_cut.empty()) {
                break;
            }
            if (--passLimit < 0) {
                std::cerr << "cutting input geometry to disk-topology did "
                             "not converge"
                          << std::endl;
                return UVGAMI_RC_CUT_FAILED;
            }

            try {
                for (auto componentI : components_to_cut) {
                    if (componentI < 0) {
                        // cut high genus
                        componentI = -componentI - 1;

                        // meshes with boundary are supported; boundary edges
                        // will be included as cuts
                        std::vector<std::vector<int>> cuts;
                        igl::cut_to_disk(F_component[componentI], cuts);

                        // only cut one seam each time to avoid seam vertex id
                        // inconsistency
                        int cuts_made = 0;
                        for (auto &seamI : cuts) {
                            // a cut renumbers one side of its end vertex
                            for (auto &segment : seamSegments(temp, seamI)) {
                                if (segment.front() == segment.back() &&
                                    !temp.isBoundaryVert(segment.front())) {
                                    // cutPath() does not support closed-loop
                                    // cuts, split it into two cuts
                                    cuts_made += temp.cutPath(
                                        std::vector<int>(
                                            {segment[segment.size() - 3],
                                             segment[segment.size() - 2],
                                             segment[segment.size() - 1]}),
                                        true);
                                    temp.initSeams = temp.cohE;
                                    segment.resize(segment.size() - 2);
                                }
                                cuts_made += temp.cutPath(segment, true);
                                temp.initSeams = temp.cohE;
                                if (cuts_made) {
                                    break;
                                }
                            }
                            if (cuts_made) {
                                break;
                            }
                        }

                        if (!cuts_made) {
                            std::vector<int> connector = boundaryLoopConnector(
                                temp, F_component[componentI]);
                            if (connector.size() >= 2) {
                                cuts_made += temp.cutPath(connector, true);
                                temp.initSeams = temp.cohE;
                            }
                        }

                        if (!cuts_made) {
                            std::cerr << "no cuts made when cutting input "
                                         "geometry to disk-topology"
                                      << std::endl;
                            return UVGAMI_RC_CUT_FAILED;
                        }
                    } else {
                        // cut the topological sphere into a topological
                        // disk; the seed cut never merges away
                        auto avoidance = [&](int vI) {
                            return vI < seamAvoidance.size() ? seamAvoidance[vI]
                                                             : 0.0;
                        };
                        int seedVI = *V_ind_component[componentI].begin();
                        for (int vI : V_ind_component[componentI]) {
                            if (avoidance(vI) < avoidance(seedVI))
                                seedVI = vI;
                        }
                        switch (initCutOption) {
                        case 0:
                            temp.onePointCut(seedVI);
                            rand1PInitCut = (n_pieces == 1);
                            break;
                        case 1:
                            temp.farthestPointCut(seedVI);
                            break;
                        default:
                            assert(0);
                            break;
                        }
                    }
                }
            } catch (const std::exception &e) {
                std::cerr << "initial cut failed: " << e.what()
                          << std::endl;
                return UVGAMI_RC_CUT_FAILED;
            }
        }

        // the layout and pinning below work per piece
        std::vector<bool> keepPiece(n_pieces, false);
        for (int triI = 0; triI < temp.F.rows(); ++triI) {
            if (keepChart[C[triI]]) {
                keepPiece[pieceOf[triI]] = true;
            }
        }
        keepChart = keepPiece;
        keptCharts = 0;
        for (int pieceI = 0; pieceI < n_pieces; ++pieceI) {
            keptCharts += keepPiece[pieceI];
        }
        C = pieceOf;
        n_components = n_pieces;

        // a bowtie vertex pinned for one piece drags the other across the layout grid
        {
            std::vector<int> owner(temp.V.rows(), -1);
            std::map<std::pair<int, int>, int> copyOf;
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                for (int i = 0; i < 3; ++i) {
                    int vI = temp.F(triI, i);
                    if (owner[vI] == -1) {
                        owner[vI] = C[triI];
                    } else if (owner[vI] != C[triI]) {
                        auto [it, inserted] = copyOf.emplace(
                            std::make_pair(vI, C[triI]),
                            static_cast<int>(temp.V.rows()));
                        if (inserted) {
                            int nV = static_cast<int>(temp.V.rows());
                            temp.V_rest.conservativeResize(nV + 1, 3);
                            temp.V_rest.row(nV) = temp.V_rest.row(vI);
                            temp.V.conservativeResize(nV + 1, 2);
                            temp.V.row(nV) = temp.V.row(vI);
                            temp.vertWeight.conservativeResize(nV + 1);
                            temp.vertWeight[nV] = temp.vertWeight[vI];
                        }
                        temp.F(triI, i) = it->second;
                    }
                }
            }
        }

        // an orphan row's zero laplacian makes the tutte solve singular
        auto dropOrphanVertices = [&]() {
            std::vector<int> vMap(temp.V.rows(), -1);
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                for (int i = 0; i < 3; ++i) {
                    vMap[temp.F(triI, i)] = 0;
                }
            }
            int vNew = 0;
            for (int vI = 0; vI < temp.V.rows(); ++vI) {
                if (vMap[vI] == 0) {
                    vMap[vI] = vNew++;
                }
            }
            if (vNew < temp.V.rows()) {
                for (int vI = 0; vI < temp.V.rows(); ++vI) {
                    if (vMap[vI] >= 0 && vMap[vI] != vI) {
                        temp.V_rest.row(vMap[vI]) = temp.V_rest.row(vI);
                        temp.V.row(vMap[vI]) = temp.V.row(vI);
                        temp.vertWeight[vMap[vI]] = temp.vertWeight[vI];
                    }
                }
                temp.V_rest.conservativeResize(vNew, 3);
                temp.V.conservativeResize(vNew, 2);
                temp.vertWeight.conservativeResize(vNew);
                for (int triI = 0; triI < temp.F.rows(); ++triI) {
                    for (int i = 0; i < 3; ++i) {
                        temp.F(triI, i) = vMap[temp.F(triI, i)];
                    }
                }
            }
        };
        dropOrphanVertices();

        F_component.assign(n_components, Eigen::MatrixXi());
        V_ind_component.assign(n_components, std::set<int>());
        for (int triI = 0; triI < temp.F.rows(); ++triI) {
            F_component[C[triI]].conservativeResize(
                F_component[C[triI]].rows() + 1, 3);
            F_component[C[triI]].bottomRows(1) = temp.F.row(triI);
            for (int i = 0; i < 3; ++i) {
                V_ind_component[C[triI]].insert(temp.F(triI, i));
            }
        }

        int UVGridDim = 0;
        do {
            ++UVGridDim;
        } while (UVGridDim * UVGridDim < n_components);

        // a unit circle beside charts kept from a packed map is far too big
        std::vector<double> chartRadius(n_components, 1.0);
        double gridCell = 2.1, gridOriginX = 0.0;
        if (keptCharts) {
            double keptUV = 0.0, kept3D = 0.0;
            std::vector<double> area3D(n_components, 0.0);
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                const Eigen::RowVector3i &tri = temp.F.row(triI);
                // the raw cross product gives a zero-area chart radius 0
                double a3 = temp.triArea[triI];
                area3D[C[triI]] += a3;
                if (keepChart[C[triI]]) {
                    const Eigen::RowVector2d e1 =
                        temp.V.row(tri[1]) - temp.V.row(tri[0]);
                    const Eigen::RowVector2d e2 =
                        temp.V.row(tri[2]) - temp.V.row(tri[0]);
                    keptUV += std::abs(e1[0] * e2[1] - e1[1] * e2[0]) / 2;
                    kept3D += a3;
                }
            }
            double uvPerRest =
                (keptUV > 0.0 && kept3D > 0.0) ? std::sqrt(keptUV / kept3D) : 1.0;
            gridCell = 0.0;
            for (int c = 0; c < n_components; ++c) {
                chartRadius[c] = uvPerRest * std::sqrt(area3D[c] / M_PI);
                if (!keepChart[c]) {
                    gridCell = (std::max)(gridCell, 2.1 * chartRadius[c]);
                }
            }
            // the output layout is only normalized, never packed
            for (int componentI = 0; componentI < n_components; ++componentI) {
                if (!keepChart[componentI]) {
                    continue;
                }
                for (const auto &vI : V_ind_component[componentI]) {
                    gridOriginX =
                        (std::max)(gridOriginX, temp.V(vI, 0) + gridCell / 2);
                }
            }
        }

        // compute boundary UV coordinates, using a grid layout for multiComp,
        // then the harmonic map with uniform weights
        auto solveTutte = [&]() {
            Eigen::MatrixXd UV_Tutte;
            Eigen::VectorXi bnd_stacked;
            Eigen::MatrixXd bnd_uv_stacked;
            for (int componentI = 0; componentI < n_components; ++componentI) {
                if (keepChart[componentI]) {
                    // pinning every vertex reproduces the chart's input UV exactly
                    const std::set<int> &chartV = V_ind_component[componentI];
                    int base = bnd_stacked.size();
                    bnd_stacked.conservativeResize(base + chartV.size());
                    bnd_uv_stacked.conservativeResize(base + chartV.size(), 2);
                    for (const auto &vI : chartV) {
                        bnd_stacked[base] = vI;
                        bnd_uv_stacked.row(base) = temp.V.row(vI);
                        ++base;
                    }
                    continue;
                }
                std::vector<std::vector<int>> bnd_all;
                igl::boundary_loop(F_component[componentI], bnd_all);

                int longest_bnd_id = 0;
                for (int bnd_id = 1; bnd_id < bnd_all.size(); ++bnd_id) {
                    if (bnd_all[longest_bnd_id].size() < bnd_all[bnd_id].size()) {
                        longest_bnd_id = bnd_id;
                    }
                }

                bnd_stacked.conservativeResize(bnd_stacked.size() +
                                               bnd_all[longest_bnd_id].size());
                bnd_stacked.tail(bnd_all[longest_bnd_id].size()) =
                    Eigen::VectorXi::Map(bnd_all[longest_bnd_id].data(),
                                         bnd_all[longest_bnd_id].size());

                Eigen::MatrixXd bnd_uv;
                if (n_components == 1) {
                    // multiComp keeps unit circles so the 2.1 grid offsets hold
                    uvgami::IglUtils::map_vertices_to_circle(
                        temp.V_rest,
                        bnd_stacked.tail(bnd_all[longest_bnd_id].size()), bnd_uv);
                } else {
                    igl::map_vertices_to_circle(
                        temp.V_rest,
                        bnd_stacked.tail(bnd_all[longest_bnd_id].size()), bnd_uv);
                }
                double xOffset = gridOriginX + componentI % UVGridDim * gridCell,
                       yOffset = componentI / UVGridDim * gridCell;
                for (int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {
                    bnd_uv(bnd_uvI, 0) =
                        bnd_uv(bnd_uvI, 0) * chartRadius[componentI] + xOffset;
                    bnd_uv(bnd_uvI, 1) =
                        bnd_uv(bnd_uvI, 1) * chartRadius[componentI] + yOffset;
                }
                bnd_uv_stacked.conservativeResize(
                    bnd_uv_stacked.rows() + bnd_uv.rows(), 2);
                bnd_uv_stacked.bottomRows(bnd_uv.rows()) = bnd_uv;
            }

            Eigen::SparseMatrix<double> A, M;
            uvgami::IglUtils::computeUniformLaplacian(temp.F, A);
            igl::harmonic(A, M, bnd_stacked, bnd_uv_stacked, 1, UV_Tutte);
            return UV_Tutte;
        };
        triSoup.emplace_back(
            new uvgami::TriMesh(V, F, solveTutte(), temp.F, false));

        const auto nearZeroInitCharts = [&]() {
            std::set<int> flagged;
            const uvgami::TriMesh &init = *triSoup.back();
            for (int triI = 0; triI < init.F.rows(); ++triI) {
                if (keepChart[C[triI]])
                    continue;
                // an area too small for the energy's 1 / area^2 is as dead as an inversion
                const Eigen::RowVector2d e1 =
                    init.V.row(init.F(triI, 1)) - init.V.row(init.F(triI, 0));
                const Eigen::RowVector2d e2 =
                    init.V.row(init.F(triI, 2)) - init.V.row(init.F(triI, 0));
                const double dbArea = e1[0] * e2[1] - e1[1] * e2[0];
                if (!init.checkInversion(triI, true) ||
                    dbArea < NEAR_ZERO_INIT_RATIO * 2.0 * init.triArea[triI])
                    flagged.insert(C[triI]);
            }
            return flagged;
        };

        // temp's adjacency is stale by now, the cut runs on a rebuild
        for (int round = 0; round < MAX_DEEPEN_ROUNDS; ++round) {
            const std::set<int> inverted = nearZeroInitCharts();
            if (inverted.empty())
                break;
            uvgami::TriMesh deeper(temp.V_rest, temp.F, temp.V, temp.F, false);
            int cuts = 0;
            for (const int componentI : inverted) {
                const std::vector<int> path =
                    deepestPath(F_component[componentI]);
                if (path.size() < 2)
                    continue;
                // glued charts leave a chart boundary mesh-interior, cutPath throws there
                if (!deeper.isBoundaryVert(path.front()) &&
                    !deeper.isBoundaryVert(path.back()))
                    continue;
                cuts += deeper.cutPath(path, true);
                deeper.initSeams = deeper.cohE;
            }
            if (!cuts)
                break;
            std::cerr << "element inversion during UV init, cutting deeper"
                      << std::endl;
            temp = deeper;
            dropOrphanVertices();
            rand1PInitCut = false;
            F_component.assign(n_components, Eigen::MatrixXi());
            V_ind_component.assign(n_components, std::set<int>());
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                F_component[C[triI]].conservativeResize(
                    F_component[C[triI]].rows() + 1, 3);
                F_component[C[triI]].bottomRows(1) = temp.F.row(triI);
                for (int i = 0; i < 3; ++i)
                    V_ind_component[C[triI]].insert(temp.F(triI, i));
            }
            delete triSoup.back();
            triSoup.back() =
                new uvgami::TriMesh(V, F, solveTutte(), temp.F, false);
        }
        // a near-zero area passes the strict inversion check below
        if (!nearZeroInitCharts().empty()) {
            std::cerr << "UV init stuck at near-zero area after deepening"
                      << std::endl;
            return UVGAMI_RC_ELEMENT_INVERSION;
        }
    }
    if (!fixedVerts.empty()) {
        if (*fixedVerts.begin() < 0 ||
            *fixedVerts.rbegin() >= keptInputMesh->V.rows()) {
            std::cerr << "pinned vertex index out of range" << std::endl;
            return UVGAMI_RC_INVALID_UV;
        }
        keptInputMesh->resetFixedVert(fixedVerts);
        pinnedMode = true;
    }

    // loaded before the optimizer copies the mesh, so the initial energy is weighted
    if (maxFaceWeight > 1) {
        const Eigen::MatrixXi &F_in = hasUV ? FUV : F;
        const int nVW = hasUV ? (int)UV.rows() : (int)V.rows();
        Eigen::VectorXd vW = Eigen::VectorXd::Zero(nVW);
        if (loadWeightSidecar(std::string(inputFolderPath.u8string()) +
                                  pathSeparator() + meshName + "_importance",
                              vW)) {
            // the initial mesh is heap-allocated above, triSoup stores it as const
            uvgami::TriMesh &mesh0 = *const_cast<uvgami::TriMesh *>(triSoup[0]);
            if (F_in.rows() == mesh0.F.rows()) {
                for (int triI = 0; triI < F_in.rows(); triI++) {
                    const double avg = (vW[F_in(triI, 0)] + vW[F_in(triI, 1)] +
                                        vW[F_in(triI, 2)]) /
                                       3.0;
                    mesh0.faceWeight[triI] = 1.0 + avg * (maxFaceWeight - 1);
                }
                // area-weighted mean 1 keeps the -u bound comparable
                mesh0.faceWeight /=
                    mesh0.faceWeight.dot(mesh0.triArea) / mesh0.surfaceArea;
            } else {
                std::cerr << "importance weights skipped, face count mismatch"
                          << std::endl;
            }
        }
    }

    outputFolderPath += meshName;
    energyParams.emplace_back(1.0 - lambda_init);
    energyTerms.emplace_back(new uvgami::SymDirichletEnergy());

    // for random one point initial cut, don't need air meshes in the
    // beginning since it's impossible for a quad to intersect itself
    optimizer = new uvgami::Optimizer(
        *triSoup[0], energyTerms, energyParams, 0, true, !rand1PInitCut);
    optimizer->precompute();
    triSoup.emplace_back(&optimizer->getResult());
    triSoup_backup = optimizer->getResult();
    triSoup.emplace_back(
        &optimizer->getData_findExtrema()); // for visualizing UV map for
                                            // finding extrema

    // regional seam placement
    uvgami::TriMesh &result = optimizer->getResult();
    Eigen::VectorXd sW = Eigen::VectorXd::Zero(result.vertWeight.size());
    if (loadWeightSidecar(weightsFileName, sW)) {
        result.vertWeight =
            (1.0 + sW.array() * (maxSeamWeight - 1)).matrix();
        uvgami::IglUtils::smoothVertField(result, result.vertWeight);
    }

    // the search spends slack under the bound on shorter seams, drifting a preseed
    if (keepInputUV && !pinnedMode && !stitchMode && !noCutMode &&
        importedMapMeasure(optimizer->getResult()) <= upperBound) {
        canSaveMesh = true;
        converge_preDrawFunc();
    }

    while (true) {
        preDrawFunc();
        if (postDrawFunc())
            break;
    }
    return UVGAMI_RC_SUCCESS;
}
