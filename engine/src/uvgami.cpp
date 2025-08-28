#include <cfloat>
#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <thread>

#include "uvgami.h"
#include "IglUtils.hpp"
#include "Optimizer.hpp"
#include "SymDirichletEnergy.hpp"

#include <igl/cut_to_disk.h>
#include <igl/cut_mesh.h>
#include <igl/readOFF.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/arap.h>
#include <igl/avg_edge_length.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/png/writePNG.h>
#include <igl/euler_characteristic.h>
#include <igl/edge_lengths.h>
#include <igl/is_vertex_manifold.h>
#include <igl/is_edge_manifold.h>
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
double lambda_init = 0.999;
bool optimization_on = false;
int iterNum = 0;
int converged = 0;
double fracThres = 0.0;
bool topoLineSearch = true;
int initCutOption = 0;
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

// visualization
bool headlessMode = false;
igl::opengl::glfw::Viewer viewer_;
const int channel_initial = 0;
const int channel_result = 1;
const int channel_findExtrema = 2;
int viewChannel = channel_result;
bool viewUV = true; // view UV or 3D model
double texScale = 1.0;
bool showSeam = true;
Eigen::MatrixXd seamColor;
bool showTexture = true; // show checkerboard
bool isLighting = false;
bool showFracTail = true;
float fracTailSize = 15.0f;
bool canSaveMesh = false;
std::string infoName = "";
bool isCapture3D = false;
int capture3DI = 0;
bool mute = true;
std::atomic<bool> forceQuit = false;
std::atomic<bool> forceQuitSave = false;
std::atomic<bool> snapshot = false;
int maxSeamWeight = 100;

const char *pathSeparator() {
#ifdef _WIN32
    return "\\";
#else
    return "/";
#endif
}
uvgami::ChronoTimer mainTimer("unwrap time");

void stdin_listener() {
    std::string line;
    do {
        std::cin >> line;
        if (line == "stop") {
            forceQuit = true;
            forceQuitSave = true;
        } else if (line == "cancel") {
            forceQuit = true;
            forceQuitSave = false;
        } else if (line == "snapshot") {
            snapshot = true;
        }
    } while (!line.empty());
}

void proceedOptimization(int proceedNum) {
    for (int proceedI = 0; (proceedI < proceedNum) && converged == 0;
         proceedI++) {
        converged = optimizer->solve(1);
        iterNum = optimizer->getIterNum();
    }
}

void updateViewerData_meshEdges(void) {
    viewer_.data().show_lines = !showSeam;
    viewer_.data().set_edges(Eigen::MatrixXd(0, 3), Eigen::MatrixXi(0, 2),
                             Eigen::RowVector3d(0.0, 0.0, 0.0));
    if (showSeam) {
        // only draw air mesh edges
        if (optimizer->isScaffolding() && viewUV &&
            (viewChannel == channel_result)) {
            const Eigen::MatrixXd V_airMesh =
                optimizer->getAirMesh().V * texScale;
            for (int triI = 0; triI < optimizer->getAirMesh().F.rows();
                 triI++) {
                const Eigen::RowVector3i &triVInd =
                    optimizer->getAirMesh().F.row(triI);
                for (int eI = 0; eI < 3; eI++)
                    viewer_.data().add_edges(
                        V_airMesh.row(triVInd[eI]),
                        V_airMesh.row(triVInd[(eI + 1) % 3]),
                        Eigen::RowVector3d::Zero());
            }
        }
    }
}

void updateViewerData_seam(Eigen::MatrixXd &V, Eigen::MatrixXi &F,
                           Eigen::MatrixXd &UV) {
    if (showSeam) {
        const Eigen::VectorXd cohIndices =
            Eigen::VectorXd::LinSpaced(triSoup[viewChannel]->cohE.rows(), 0,
                                       triSoup[viewChannel]->cohE.rows() - 1);
        Eigen::MatrixXd color;
        color.resize(cohIndices.size(), 3);
        color.rowwise() = Eigen::RowVector3d(1.0, 0.5, 0.0);

        seamColor.resize(0, 3);
        double seamThickness =
            (viewUV ? (triSoup[viewChannel]->virtualRadius * 0.0007 /
                       viewer_.core().camera_zoom * texScale)
                    : (triSoup[viewChannel]->virtualRadius * 0.006));
        for (int eI = 0; eI < triSoup[viewChannel]->cohE.rows(); eI++) {
            const Eigen::RowVector4i &cohE = triSoup[viewChannel]->cohE.row(eI);
            const auto finder = triSoup[viewChannel]->edge2Tri.find(
                std::pair<int, int>(cohE[0], cohE[1]));
            assert(finder != triSoup[viewChannel]->edge2Tri.end());
            const Eigen::RowVector3d &sn =
                triSoup[viewChannel]->triNormal.row(finder->second);

            // seam edge
            uvgami::IglUtils::addThickEdge(
                V, F, UV, seamColor, color.row(eI), V.row(cohE[0]),
                V.row(cohE[1]), seamThickness, texScale, !viewUV, sn);
            if (viewUV)
                uvgami::IglUtils::addThickEdge(
                    V, F, UV, seamColor, color.row(eI), V.row(cohE[2]),
                    V.row(cohE[3]), seamThickness, texScale, !viewUV, sn);
        }
    }
}

void updateViewerData_distortion(const std::string &meshName) {
    Eigen::MatrixXd color_distortionVis;
    Eigen::VectorXd distortionPerElem;
    energyTerms[0]->getEnergyValPerElem(*triSoup[viewChannel],
                                        distortionPerElem, true);
    uvgami::IglUtils::mapScalarToColor(meshName, distortionPerElem,
                                       color_distortionVis, 4.0, 8.5, 0);

    if (optimizer->isScaffolding() && viewUV && (viewChannel == channel_result))
        optimizer->getScaffold().augmentFColorwithAirMesh(color_distortionVis);

    if (showSeam) {
        color_distortionVis.conservativeResize(
            color_distortionVis.rows() + seamColor.rows(), 3);
        color_distortionVis.bottomRows(seamColor.rows()) = seamColor;
    }

    viewer_.data().set_colors(color_distortionVis);
}

void updateViewerData(const std::string &meshName) {
    Eigen::MatrixXd UV_vis = triSoup[viewChannel]->V * texScale;
    Eigen::MatrixXi F_vis = triSoup[viewChannel]->F;
    if (viewUV) {
        if (optimizer->isScaffolding() && (viewChannel == channel_result)) {
            optimizer->getScaffold().augmentUVwithAirMesh(UV_vis, texScale);
            optimizer->getScaffold().augmentFwithAirMesh(F_vis);
        }
        UV_vis.conservativeResize(UV_vis.rows(), 3);
        UV_vis.rightCols(1) = Eigen::VectorXd::Zero(UV_vis.rows());
        viewer_.core().align_camera_center(UV_vis, F_vis);
        updateViewerData_seam(UV_vis, F_vis, UV_vis);

        if ((UV_vis.rows() != viewer_.data().V.rows()) ||
            (F_vis.rows() != viewer_.data().F.rows())) {
            viewer_.data().clear();
        }
        viewer_.data().set_mesh(UV_vis, F_vis);
        viewer_.data().show_texture = false;
        viewer_.core().lighting_factor = 0.0;

        updateViewerData_meshEdges();

        viewer_.data().set_points(Eigen::MatrixXd::Zero(0, 3),
                                  Eigen::RowVector3d(0.0, 0.0, 0.0));
        if (showFracTail) {
            for (const auto &tailVI : triSoup[viewChannel]->fracTail)
                viewer_.data().add_points(UV_vis.row(tailVI),
                                          Eigen::RowVector3d(0.0, 0.0, 0.0));
        }
    } else {
        Eigen::MatrixXd V_vis = triSoup[viewChannel]->V_rest;
        viewer_.core().align_camera_center(V_vis, F_vis);
        updateViewerData_seam(V_vis, F_vis, UV_vis);

        if ((V_vis.rows() != viewer_.data().V.rows()) ||
            (UV_vis.rows() != viewer_.data().V_uv.rows()) ||
            (F_vis.rows() != viewer_.data().F.rows())) {
            viewer_.data().clear();
        }
        viewer_.data().set_mesh(V_vis, F_vis);

        if (showTexture) {
            viewer_.data().set_uv(UV_vis);
            viewer_.data().show_texture = true;
        } else {
            viewer_.data().show_texture = false;
        }

        if (isLighting) {
            viewer_.core().lighting_factor = 1.0;
        } else {
            viewer_.core().lighting_factor = 0.0;
        }
        updateViewerData_meshEdges();

        viewer_.data().set_points(Eigen::MatrixXd::Zero(0, 3),
                                  Eigen::RowVector3d(0.0, 0.0, 0.0));
        if (showFracTail) {
            for (const auto &tailVI : triSoup[viewChannel]->fracTail)
                viewer_.data().add_points(V_vis.row(tailVI),
                                          Eigen::RowVector3d(0.0, 0.0, 0.0));
        }
    }
    updateViewerData_distortion(meshName);
    viewer_.data().compute_normals();

    if (snapshot) {
        triSoup[channel_result]->saveAsMesh(F, true);
        snapshot = false;
    }
}
bool postDrawFunc(igl::opengl::glfw::Viewer &viewer) {
    if (iterNum == 0) {
        optimization_on = !optimization_on;
        if (optimization_on && converged)
            optimization_on = false;
    }
    if (forceQuit) {
        canSaveMesh = forceQuitSave;
        outerLoopFinished = true;
        isCapture3D = true;
        capture3DI = 2;
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

    if (outerLoopFinished) {
        if (!isCapture3D) {
            viewer.core().is_animating = true;
            isCapture3D = true;
        } else {
            if (capture3DI < 2) {
                capture3DI++;
            } else {
                return true;
            }
        }
    }

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
    Eigen::MatrixXd edgeLengths;
    igl::edge_lengths(triSoup[channel_result]->V_rest,
                      triSoup[channel_result]->F, edgeLengths);
    const double eps_E_se = 1.0e-3 * edgeLengths.minCoeff() /
                            triSoup[channel_result]->virtualRadius;

    // measurement and energy value computation
    const double E_SD = optimizer->getLastEnergyVal(true) / energyParams[0];
    double E_se;
    triSoup[channel_result]->computeSeamSparsity(E_se);
    E_se /= triSoup[channel_result]->virtualRadius;
    double stretch_l2, stretch_inf, stretch_shear, compress_inf;
    triSoup[channel_result]->computeStandardStretch(
        stretch_l2, stretch_inf, stretch_shear, compress_inf);
    double measure_bound = E_SD;
    const double eps_lambda =
        (std::min)(1.0e-3,
                   std::abs(updateLambda(measure_bound) - energyParams[0]));

    // TODO?: stop when first violates bounds from feasible, don't go to best
    // feasible. check after each merge whether distortion is violated
    //  oscillation detection
    static int iterNum_bestFeasible = -1;
    static uvgami::TriMesh triSoup_bestFeasible;
    static double E_se_bestFeasible = DBL_MAX;
    static int lastStationaryIterNum =
        0; // still necessary because boundary and interior query are with same
           // iterNum
    static std::map<double, std::vector<std::pair<double, double>>>
        configs_stationaryV;
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
        } else {
            configs_stationaryV[E_se].emplace_back(
                std::pair<double, double>(lambda, E_SD));
        }
    }
    lastStationaryIterNum = iterNum;
    // convergence check
    if (checkConvergence) {
        if (measure_bound <= upperBound) {
            // save info at first feasible stationaryVT for comparison
            static bool saved = false;
            if (!saved) {
                //                logFile << "saving firstFeasibleS..." <<
                //                std::endl; saveScreenshot(outputFolderPath +
                //                "firstFeasibleS.png", 0.5, false, true);
                //                //TODO: saved is before roll back...
                //                triSoup[channel_result]->saveAsMesh(outputFolderPath
                //                + "firstFeasibleS_mesh.obj", F);
                saved = true;
                //              logFile << "firstFeasibleS saved" << std::endl;
            }
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
                (computeOptPicked(energyChanges_bSplit, energyChanges_merge,
                                  1.0 - energyParams[0]) == 1)) {
                // still picking merge
                do {
                    energyParams[0] = updateLambda(measure_bound);
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
                while ((eDec_b > 0.0) && (eDec_i > 0.0)) {
                    energyParams[0] = updateLambda(measure_bound);
                    id_pickingBSplit = computeBestCand(
                        energyChanges_bSplit, 1.0 - energyParams[0], eDec_b);
                    id_pickingISplit = computeBestCand(
                        energyChanges_iSplit, 1.0 - energyParams[0], eDec_i);
                }
                if (eDec_b <= 0.0) {
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
            if (computeOptPicked(energyChanges_bSplit, energyChanges_merge,
                                 1.0 - energyParams[0]) == 0) {
                // still picking split
                do {
                    energyParams[0] = updateLambda(measure_bound);
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
            while (eDec_m > 0.0) {
                energyParams[0] = updateLambda(measure_bound);
                id_pickingMerge = computeBestCand(
                    energyChanges_merge, 1.0 - energyParams[0], eDec_m);
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

void converge_preDrawFunc(igl::opengl::glfw::Viewer &viewer) {
    updateViewerData(meshName);
    optimization_on = false;
    viewer.core().is_animating = false;
    // std::cout << "optimization converged, in " << secPast << "s." <<
    // std::endl;
    outerLoopFinished = true;
}

bool preDrawFunc(igl::opengl::glfw::Viewer &viewer) {
    if (optimization_on) {
        while (!converged)
            proceedOptimization(1);
        updateViewerData(meshName);

        // give postDraw option to save mesh
        canSaveMesh = true;

        double stretch_l2, stretch_inf, stretch_shear, compress_inf;
        triSoup[channel_result]->computeStandardStretch(
            stretch_l2, stretch_inf, stretch_shear, compress_inf);
        double measure_bound =
            optimizer->getLastEnergyVal(true) / energyParams[0];
        if (converged == 2) {
            converged = 0;
            return false;
        }
        // if necessary, turn on scaffolding for random one point initial cut
        if (!optimizer->isScaffolding() && rand1PInitCut)
            optimizer->setScaffolding(true);

        double E_se;
        triSoup[channel_result]->computeSeamSparsity(E_se);
        E_se /= triSoup[channel_result]->virtualRadius;
        const double E_SD = optimizer->getLastEnergyVal(true) / energyParams[0];

        // std::cout << iterNum << ": " << E_SD << " " << E_se << " " <<
        // triSoup[channel_result]->V_rest.rows() << std::endl;
        //  DISABLE logFile << iterNum << ": " << E_SD << " " << E_se << " " <<
        //  triSoup[channel_result]->V_rest.rows() << std::endl;

        // continue to split boundary
        if (!updateLambda_stationaryV()) {
            // oscillation detected
            converge_preDrawFunc(viewer);
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
                        converge_preDrawFunc(viewer);
                    } else {
                        // split or merge after lambda update
                        if (reQuery) {
                            filterExp_in +=
                                std::log(2.0) / std::log(inSplitTotalAmt);
                            filterExp_in = (std::min)(1.0, filterExp_in);
                            while (!optimizer->createFracture(
                                fracThres, false, topoLineSearch, true)) {
                                filterExp_in +=
                                    std::log(2.0) / std::log(inSplitTotalAmt);
                                filterExp_in = (std::min)(1.0, filterExp_in);
                            }
                            reQuery = false;
                            // TODO: set filtering param back?
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
    } else {
        if (isCapture3D && (capture3DI < 2)) {
            // change view accordingly
            double rotDeg =
                ((capture3DI < 8) ? (M_PI_2 * (capture3DI / 2)) : M_PI_2);
            Eigen::Vector3f rotAxis = Eigen::Vector3f::UnitY();
            if ((capture3DI / 2) == 4) {
                rotAxis = Eigen::Vector3f::UnitX();
            } else if ((capture3DI / 2) == 5) {
                rotAxis = -Eigen::Vector3f::UnitX();
            }
            viewer.core().trackball_angle =
                Eigen::Quaternionf(Eigen::AngleAxisf(rotDeg, rotAxis));
            viewChannel = channel_result;
            viewUV = false;
            showSeam = true;
            isLighting = false;
            showTexture = capture3DI % 2;
            updateViewerData(meshName);
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

int main(int argc, char *argv[]) {
    int progMode = 100;
    std::string meshFileName;
    lambda_init = 0.999;
    std::filesystem::path inputFolderPath;
    bool hasUV = false;
    bool ignoreUV = false;

    try {
        TCLAP::CmdLine cmd("uvgami command line", ' ', "1.1.2");
        TCLAP::ValueArg<uint32_t> programModeArg("p", "program_mode",
                                                 "Program mode", false, 0,
                                                 "unsigned integer", cmd);
        TCLAP::ValueArg<std::string> inputArg("i", "input", "Input mesh", true,
                                              "", "string", cmd);
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
        TCLAP::SwitchArg ignoreUVArg("g", "ignore_uv", "Ignore UV map", cmd);
        cmd.parse(argc, argv);

        if (maxSeamWeightArg.isSet())
            maxSeamWeight = maxSeamWeightArg.getValue();
        if (ignoreUVArg.isSet())
            ignoreUV = ignoreUVArg.getValue();
        meshFileName = inputArg.getValue();
        inputFolderPath = std::filesystem::path(meshFileName).parent_path();
        if (outputArg.isSet())
            outputFolderPath = outputArg.getValue();
        else
            outputFolderPath =
                std::string(inputFolderPath.parent_path().u8string()) +
                pathSeparator() + "output" + pathSeparator();
        if (programModeArg.isSet())
            progMode = programModeArg.getValue();
        switch (progMode) {
        case 10:
            headlessMode = false;
            break;
        case 100:
            headlessMode = true;
            break;
        default: {
            std::cout << "Invalid program mode " << progMode << std::endl;
            return 0;
        }
        }
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
    // create output folder
    if (!std::filesystem::exists(outputFolderPath) &&
        !std::filesystem::create_directory(outputFolderPath)) {
        printf("Failed to create output directory %s\n",
               outputFolderPath.c_str());
        return -1;
    }
    // Load mesh
    std::string meshFilePath = meshFileName;
    meshFileName =
        meshFileName.substr(meshFileName.find_last_of(pathSeparator()) + 1);
    meshName = meshFileName.substr(0, meshFileName.find_last_of('.'));
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
    //    //DEBUG
    //    uvgami::TriMesh squareMesh(uvgami::P_SQUARE, 1.0, 0.1, false);
    //    V = squareMesh.V_rest;
    //    F = squareMesh.F;

    hasUV = !ignoreUV && (UV.rows() != 0);
    if (hasUV) {
        uvgami::TriMesh *temp = new uvgami::TriMesh(V, F, UV, FUV, false);
        std::vector<std::vector<int>> bnd_all;
        igl::boundary_loop(temp->F, bnd_all);
        int UVGridDim = std::ceil(std::sqrt(bnd_all.size()));
        // if (UVGridDim > 1)
        //	std::cout << "Multi-chart bijective UV map needs to be
        // validated." << std::endl;

        if (!temp->checkInversion() /*|| (UVGridDim > 1)*/) {
            std::cout << "local injectivity violated in input UV map, " <<
                //"or multi-chart bijective UV map needs to be ensured, " <<
                "obtaining new initial UV map by applying Tutte's embedding..."
                      << std::endl;
            Eigen::VectorXi bnd_stacked;
            Eigen::MatrixXd bnd_uv_stacked;
            int curBndVAmt = 0;
            for (int bndI = 0; bndI < bnd_all.size(); bndI++) {
                // map boundary to unit circle
                bnd_stacked.conservativeResize(curBndVAmt +
                                               bnd_all[bndI].size());
                bnd_stacked.tail(bnd_all[bndI].size()) = Eigen::VectorXi::Map(
                    bnd_all[bndI].data(), bnd_all[bndI].size());
                Eigen::MatrixXd bnd_uv;
                igl::map_vertices_to_circle(
                    temp->V_rest, bnd_stacked.tail(bnd_all[bndI].size()),
                    bnd_uv);
                double xOffset = bndI % UVGridDim * 2.1,
                       yOffset = bndI / UVGridDim * 2.1;
                for (int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {
                    bnd_uv(bnd_uvI, 0) += xOffset;
                    bnd_uv(bnd_uvI, 1) += yOffset;
                }
                bnd_uv_stacked.conservativeResize(curBndVAmt + bnd_uv.rows(),
                                                  2);
                bnd_uv_stacked.bottomRows(bnd_uv.rows()) = bnd_uv;
                curBndVAmt = bnd_stacked.size();
            }
            // Harmonic map with uniform weights
            Eigen::MatrixXd UV_Tutte;
            Eigen::SparseMatrix<double> A, M;
            uvgami::IglUtils::computeUniformLaplacian(temp->F, A);
            igl::harmonic(A, M, bnd_stacked, bnd_uv_stacked, 1, temp->V);
            if (!temp->checkInversion()) {
                std::cout << "local injectivity still violated in the computed "
                             "initial UV map, "
                          << "please carefully check UV topology for e.g. "
                             "non-manifold vertices. "
                          << "Exit program..." << std::endl;
                return UVGAMI_RC_INVALID_UV;
            }
        }
        triSoup.emplace_back(temp);
    } else {
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

        std::vector<std::vector<int>> bnd_all;
        igl::boundary_loop(F, bnd_all);
        Eigen::VectorXi bnd;
        if (bnd_all.size()) {
            // ASSUME: no disconnected closed surface present
            if (bnd_all.size() == 1) {
                // disk-topology
                bnd.resize(bnd_all[0].size());
                std::memcpy(bnd.data(), bnd_all[0].data(),
                            sizeof(int) * bnd.size());

                // Map the boundary to a circle, preserving edge proportions
                Eigen::MatrixXd bnd_uv;
                uvgami::IglUtils::map_vertices_to_circle(V, bnd, bnd_uv);

                Eigen::MatrixXd UV_Tutte;

                // Harmonic map with uniform weights
                if (bnd.size() == V.rows()) {
                    UV_Tutte.resize(V.rows(), 2);
                    for (int bndVI = 0; bndVI < bnd_uv.rows(); ++bndVI) {
                        UV_Tutte.row(bnd[bndVI]) = bnd_uv.row(bndVI);
                    }
                } else {
                    Eigen::SparseMatrix<double> A, M;
                    uvgami::IglUtils::computeUniformLaplacian(F, A);
                    igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);
                    //            uvgami::IglUtils::computeMVCMtr(V, F, A);
                    //            uvgami::IglUtils::fixedBoundaryParam_MVC(A,
                    //            bnd, bnd_uv, UV_Tutte);
                }

                triSoup.emplace_back(new uvgami::TriMesh(
                    V, F, UV_Tutte, Eigen::MatrixXi(), false));
            }

            else {
                // multiple disk topology surfaces
                int UVGridDim = std::ceil(std::sqrt(bnd_all.size()));
                Eigen::VectorXi bnd_stacked;
                Eigen::MatrixXd bnd_uv_stacked;
                int curBndVAmt = 0;
                for (int bndI = 0; bndI < bnd_all.size(); bndI++) {
                    // map boundary to unit circle
                    bnd_stacked.conservativeResize(curBndVAmt +
                                                   bnd_all[bndI].size());
                    bnd_stacked.tail(bnd_all[bndI].size()) =
                        Eigen::VectorXi::Map(bnd_all[bndI].data(),
                                             bnd_all[bndI].size());

                    Eigen::MatrixXd bnd_uv;
                    igl::map_vertices_to_circle(
                        V, bnd_stacked.tail(bnd_all[bndI].size()), bnd_uv);
                    double xOffset = bndI % UVGridDim * 2.1,
                           yOffset = bndI / UVGridDim * 2.1;
                    for (int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {
                        bnd_uv(bnd_uvI, 0) += xOffset;
                        bnd_uv(bnd_uvI, 1) += yOffset;
                    }
                    bnd_uv_stacked.conservativeResize(
                        curBndVAmt + bnd_uv.rows(), 2);
                    bnd_uv_stacked.bottomRows(bnd_uv.rows()) = bnd_uv;

                    curBndVAmt = bnd_stacked.size();
                }
                // Harmonic map with uniform weights
                Eigen::MatrixXd UV_Tutte;
                Eigen::SparseMatrix<double> A, M;
                uvgami::IglUtils::computeUniformLaplacian(F, A);
                igl::harmonic(A, M, bnd_stacked, bnd_uv_stacked, 1, UV_Tutte);

                triSoup.emplace_back(new uvgami::TriMesh(
                    V, F, UV_Tutte, Eigen::MatrixXi(), false));
            }
        } else {
            // closed surface
            int genus = 1 - igl::euler_characteristic(V, F) / 2;
            if (genus != 0) {
                // DISABLE std::cout << "Input surface genus = " +
                // std::to_string(genus) + " or has multiple connected
                // components!" << std::endl;
                std::vector<std::vector<int>> cuts;
                igl::cut_to_disk(F, cuts);

                // record cohesive edge information,
                // transfer information format for cut_mesh
                uvgami::TriMesh temp(V, F, Eigen::MatrixXd(), Eigen::MatrixXi(),
                                     false);
                Eigen::MatrixXi cutFlags(F.rows(), 3);
                Eigen::MatrixXi cohEdgeRecord;
                cutFlags.setZero();
                for (const auto &seamI : cuts) {
                    for (int segI = 0; segI + 1 < seamI.size(); segI++) {
                        std::pair<int, int> edge(seamI[segI], seamI[segI + 1]);
                        auto finder = temp.edge2Tri.find(edge);
                        assert(finder != temp.edge2Tri.end());
                        int i = 0;
                        for (; i < 3; i++) {
                            if (temp.F(finder->second, i) == edge.first) {
                                cutFlags(finder->second, i) = 1;
                                break;
                            }
                        }
                        int cohERI = cohEdgeRecord.rows();
                        cohEdgeRecord.conservativeResize(cohERI + 1, 4);
                        cohEdgeRecord(cohERI, 0) = finder->second;
                        cohEdgeRecord(cohERI, 1) = i;

                        edge.second = seamI[segI];
                        edge.first = seamI[segI + 1];
                        finder = temp.edge2Tri.find(edge);
                        assert(finder != temp.edge2Tri.end());
                        for (i = 0; i < 3; i++) {
                            if (temp.F(finder->second, i) == edge.first) {
                                cutFlags(finder->second, i) = 1;
                                break;
                            }
                        }
                        cohEdgeRecord(cohERI, 2) = finder->second;
                        cohEdgeRecord(cohERI, 3) = i;
                    }
                }
                Eigen::MatrixXd Vcut;
                Eigen::MatrixXi Fcut;
                igl::cut_mesh(temp.V_rest, temp.F, cutFlags, Vcut, Fcut);
                igl::writeOBJ(outputFolderPath + meshName + "_disk.obj", Vcut,
                              Fcut);
                V = Vcut;
                F = Fcut;

                igl::boundary_loop(F, bnd); // Find the open boundary
                assert(bnd.size());

                Eigen::MatrixXd bnd_uv;
                uvgami::IglUtils::map_vertices_to_circle(V, bnd, bnd_uv);

                Eigen::MatrixXd UV_Tutte;

                // Harmonic map with uniform weights
                Eigen::SparseMatrix<double> A, M;
                uvgami::IglUtils::computeUniformLaplacian(F, A);
                igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);
                //            uvgami::IglUtils::computeMVCMtr(V, F, A);
                //            uvgami::IglUtils::fixedBoundaryParam_MVC(A, bnd,
                //            bnd_uv, UV_Tutte);

                uvgami::TriMesh *ptr = new uvgami::TriMesh(
                    V, F, UV_Tutte, Eigen::MatrixXi(), false);
                ptr->buildCohEfromRecord(cohEdgeRecord);
                triSoup.emplace_back(ptr);
            } else {
                uvgami::TriMesh *temp = new uvgami::TriMesh(
                    V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), false);
                switch (initCutOption) {
                case 0:
                    temp->onePointCut();
                    rand1PInitCut = true;
                    break;
                case 1:
                    temp->farthestPointCut();
                    break;
                default:
                    assert(0);
                    break;
                }
                igl::boundary_loop(temp->F, bnd);
                assert(bnd.size());

                Eigen::MatrixXd bnd_uv;
                uvgami::IglUtils::map_vertices_to_circle(temp->V_rest, bnd,
                                                         bnd_uv);

                Eigen::SparseMatrix<double> A, M;
                uvgami::IglUtils::computeUniformLaplacian(temp->F, A);

                Eigen::MatrixXd UV_Tutte;
                igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);

                triSoup.emplace_back(new uvgami::TriMesh(
                    V, F, UV_Tutte, temp->F, false, temp->initSeamLen));

                // try initialize one-point cut with different vertices
                // until no inversion is detected
                int splitVI = 0;
                while (!triSoup.back()->checkInversion(true)) {
                    std::cout
                        << "element inversion detected during UV "
                           "initialization "
                        << "due to rounding errors, trying another vertex..."
                        << std::endl;

                    delete temp;
                    temp = new uvgami::TriMesh(V, F, Eigen::MatrixXd(),
                                               Eigen::MatrixXi(), false);
                    temp->onePointCut(++splitVI);

                    igl::boundary_loop(temp->F, bnd);
                    assert(bnd.size());
                    uvgami::IglUtils::map_vertices_to_circle(temp->V_rest, bnd,
                                                             bnd_uv);
                    uvgami::IglUtils::computeUniformLaplacian(temp->F, A);
                    igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);
                    delete triSoup.back();
                    triSoup.back() = new uvgami::TriMesh(
                        V, F, UV_Tutte, temp->F, false, temp->initSeamLen);
                }
                delete temp;
            }
        }
    }
    outputFolderPath += meshName;
    texScale =
        10.0 / (triSoup[0]->bbox.row(1) - triSoup[0]->bbox.row(0)).maxCoeff();
    energyParams.emplace_back(1.0 - lambda_init);
    energyTerms.emplace_back(new uvgami::SymDirichletEnergy());

    try {
        // for random one point initial cut, don't need air meshes in the
        // beginning since it's impossible for a quad to intersect itself
        optimizer = new uvgami::Optimizer(
            *triSoup[0], energyTerms, energyParams, 0, true, !rand1PInitCut);
    } catch (UvgamiElementInversionException &eie) {
        (void)eie;
        return UVGAMI_RC_ELEMENT_INVERSION;
    }
    optimizer->precompute();
    triSoup.emplace_back(&optimizer->getResult());
    triSoup_backup = optimizer->getResult();
    triSoup.emplace_back(
        &optimizer->getData_findExtrema()); // for visualizing UV map for
                                            // finding extrema

    // regional seam placement
    std::string weightsFileName = std::string(inputFolderPath.u8string()) +
                                  std::string(pathSeparator()) + meshName;
    weightsFileName += "_weights";
    std::ifstream vWFile(weightsFileName);
    if (vWFile.is_open()) {
        std::string line;
        getline(vWFile, line);
        char sep = ',';
        std::vector<float> tokens = split(line, sep);
        for (uint32_t i = 0; i < tokens.size(); i += 2) {
            uint32_t selected = (uint32_t)tokens[i];
            float weight = tokens[i + 1];
            if (selected < optimizer->getResult().vertWeight.size())
                optimizer->getResult().vertWeight[selected] =
                    1 + weight * (maxSeamWeight - 1);
        }
        vWFile.close();
        uvgami::IglUtils::smoothVertField(optimizer->getResult(),
                                          optimizer->getResult().vertWeight);
    }

    std::thread t(&stdin_listener);
    if (headlessMode) {
        while (true) {
            preDrawFunc(viewer_);
            if (postDrawFunc(viewer_))
                break;
        }
    } else {
        // Setup viewer and launch
        viewer_.core().background_color << 1.0f, 1.0f, 1.0f, 0.0f;
        viewer_.callback_pre_draw = &preDrawFunc;
        viewer_.callback_post_draw = &postDrawFunc;
        viewer_.data().show_lines = true;
        viewer_.core().orthographic = true;
        viewer_.core().camera_zoom *= 1.9;
        viewer_.core().animation_max_fps = 60.0;
        viewer_.data().point_size = fracTailSize;
        viewer_.data().show_overlay = true;
        updateViewerData(meshName);
        viewer_.launch();
    }
    // cleanup
    t.detach();
    for (auto &eI : energyTerms)
        delete eI;
    delete optimizer;
    delete triSoup[0];

    return 0;
}
