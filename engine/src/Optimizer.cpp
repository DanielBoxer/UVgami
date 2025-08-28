//  Created by Minchen Li on 8/31/17.

#include <fstream>
#include <iostream>
#include <string>
#include <numeric>

#include "uvgami.h"
#include "Optimizer.hpp"
#include "SymDirichletEnergy.hpp"
#include "IglUtils.hpp"
#include "EigenLibSolver.hpp"
#include <igl/avg_edge_length.h>

namespace uvgami {

Optimizer::Optimizer(const TriMesh &p_data0,
                     const std::vector<Energy *> &p_energyTerms,
                     const std::vector<double> &p_energyParams,
                     int p_propagateFracture, bool p_mute, bool p_scaffolding,
                     const Eigen::MatrixXd &UV_bnds, const Eigen::MatrixXi &E,
                     const Eigen::VectorXi &bnd, bool p_useDense)
    : data0(p_data0), energyTerms(p_energyTerms), energyParams(p_energyParams) {
    assert(energyTerms.size() == energyParams.size());
    useDense = p_useDense;
    energyParamSum = 0.0;
    for (const auto &ePI : energyParams)
        energyParamSum += ePI;
    gradient_ET.resize(energyTerms.size());
    energyVal_ET.resize(energyTerms.size());
    allowEDecRelTol = true;
    propagateFracture = p_propagateFracture;
    mute = p_mute;
    if (!data0.checkInversion())
        throw UvgamiElementInversionException();
    globalIterNum = 0;
    relGL2Tol = 1.0e-12;
    topoIter = 0;
    needRefactorize = false;
    for (const auto &energyTermI : energyTerms) {
        if (energyTermI->getNeedRefactorize()) {
            needRefactorize = true;
            break;
        }
    }
    pardisoThreadAmt = 4;
    scaffolding = p_scaffolding;
    UV_bnds_scaffold = UV_bnds;
    E_scaffold = E;
    bnd_scaffold = bnd;
    w_scaf = energyParams[0] * 0.01;
    linSysSolver = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
}

Optimizer::~Optimizer(void) { delete linSysSolver; }
TriMesh &Optimizer::getResult(void) { return result; }

const Scaffold &Optimizer::getScaffold(void) const { return scaffold; }

const TriMesh &Optimizer::getAirMesh(void) const { return scaffold.airMesh; }

bool Optimizer::isScaffolding(void) const { return scaffolding; }

const TriMesh &Optimizer::getData_findExtrema(void) const {
    return data_findExtrema;
}

int Optimizer::getIterNum(void) const { return globalIterNum; }

int Optimizer::getTopoIter(void) const { return topoIter; }

void Optimizer::setRelGL2Tol(double p_relTol) {
    assert(p_relTol > 0.0);
    relGL2Tol = p_relTol;
    updateTargetGRes();
}

void Optimizer::setAllowEDecRelTol(bool p_allowEDecRelTol) {
    allowEDecRelTol = p_allowEDecRelTol;
}

void Optimizer::precompute(void) {
    result = data0;
    if (scaffolding) {
        scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
        result.scaffold = &scaffold;
        scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
        scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
    }
    computeHessian(result, scaffold);
    if (useDense) {
        if (!needRefactorize)
            denseSolver = Hessian.ldlt();
    } else {
        linSysSolver->set_type(pardisoThreadAmt, -2);
        linSysSolver->set_pattern(
            scaffolding ? vNeighbor_withScaf : result.vNeighbor,
            scaffolding ? fixedV_withScaf : result.fixedVert);
        linSysSolver->update_a(I_mtr, J_mtr, V_mtr);
        linSysSolver->analyze_pattern();
        if (!needRefactorize) {
            try {
                linSysSolver->factorize();
            } catch (std::exception e) {
                exit(-1);
            }
        }
    }
    lastEDec = 0.0;
    data_findExtrema = data0;
    updateTargetGRes();
    computeEnergyVal(result, scaffold, lastEnergyVal);
    // std::cout << "E_initial = " << lastEnergyVal << std::endl;
}

int Optimizer::solve(int maxIter) {
    static bool lastPropagate = false;
    for (int iterI = 0; iterI < maxIter; iterI++) {
        computeGradient(result, scaffold, gradient);
        const double sqn_g = gradient.squaredNorm();
        // std::cout << "||gradient||^2 = " << sqn_g << ", targetGRes = " <<
        // targetGRes << std::endl;
        if (sqn_g < targetGRes) {
            // converged
            lastEDec = 0.0;
            globalIterNum++;
            return 1;
        } else {
            if (solve_oneStep()) {
                globalIterNum++;
                return 1;
            }
        }
        globalIterNum++;
        if (propagateFracture > 0) {
            if (!createFracture(lastEDec, propagateFracture)) {
                // always perform the one decreasing E_w more
                if (scaffolding) {
                    scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold,
                                        bnd_scaffold);
                    result.scaffold = &scaffold;
                    scaffold.mergeVNeighbor(result.vNeighbor,
                                            vNeighbor_withScaf);
                    scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
                }

                if (lastPropagate) {
                    lastPropagate = false;
                    return 2; // for saving screenshots
                }
            } else {
                lastPropagate = true;
            }
        } else {
            if (scaffolding) {
                scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold,
                                    bnd_scaffold);
                result.scaffold = &scaffold;
                scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
                scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
            }
        }
    }
    return 0;
}

void Optimizer::updatePrecondMtrAndFactorize(void) {
    if (needRefactorize) {
        // don't need to call this function
        return;
    }
    // std::cout << "recompute proxy/Hessian matrix and factorize..." <<
    // std::endl;
    computeHessian(result, scaffold);
    if (useDense) {
        denseSolver = Hessian.ldlt();
    } else {
        linSysSolver->update_a(I_mtr, J_mtr, V_mtr);
        linSysSolver->factorize();
    }
}

void Optimizer::setConfig(const TriMesh &config, int iterNum, int p_topoIter) {
    topoIter = p_topoIter;
    globalIterNum = iterNum;
    result = config;
    if (scaffolding) {
        scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
        result.scaffold = &scaffold;
        scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
        scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
    }
    updateEnergyData();
}
void Optimizer::setScaffolding(bool p_scaffolding) {
    scaffolding = p_scaffolding;
    if (scaffolding) {
        scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
        result.scaffold = &scaffold;
        scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
        scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
    }
}

void Optimizer::updateEnergyData(bool updateEVal, bool updateGradient,
                                 bool updateHessian) {
    energyParamSum = 0.0;
    for (const auto &ePI : energyParams)
        energyParamSum += ePI;
    updateTargetGRes();
    if (updateEVal) {
        // compute energy and output
        computeEnergyVal(result, scaffold, lastEnergyVal);
    }
    if (updateGradient) {
        // compute gradient and output
        computeGradient(result, scaffold, gradient);
        if (gradient.squaredNorm() < targetGRes) {
            // DISABLE logFile << "||g||^2 = " << gradient.squaredNorm() << "
            // after fracture initiation!" << std::endl;
        }
    }
    if (updateHessian) {
        // for the changing hessian
        // std::cout << "recompute proxy/Hessian matrix and factorize..." <<
        // std::endl;
        computeHessian(result, scaffold);
        if (useDense) {
            if (!needRefactorize)
                denseSolver = Hessian.ldlt();
        } else {
            linSysSolver->set_pattern(
                scaffolding ? vNeighbor_withScaf : result.vNeighbor,
                scaffolding ? fixedV_withScaf : result.fixedVert);
            linSysSolver->update_a(I_mtr, J_mtr, V_mtr);
            linSysSolver->analyze_pattern();
            if (!needRefactorize)
                linSysSolver->factorize();
        }
    }
}

bool Optimizer::createFracture(int opType, const std::vector<int> &path,
                               const Eigen::MatrixXd &newVertPos,
                               bool allowPropagate) {
    topoIter++;
    bool isMerge = false;
    data_findExtrema = result; // potentially time-consuming
    switch (opType) {
    case 0: // boundary split
            // std::cout << "boundary split without querying again" <<
            // std::endl;
        result.splitEdgeOnBoundary(std::pair<int, int>(path[0], path[1]),
                                   newVertPos);
        // DISABLE logFile << "boundary edge split without querying again" <<
        // std::endl;
        // TODO: process fractail here!
        result.updateFeatures();
        break;
    case 1: // interior split
            // std::cout << "Interior split without querying again" <<
            // std::endl;
        result.cutPath(path, true, 1, newVertPos);
        // DISABLE logFile << "interior edge split without querying again" <<
        // std::endl;
        result.fracTail.insert(path[0]);
        result.fracTail.insert(path[2]);
        result.curInteriorFracTails.first = path[0];
        result.curInteriorFracTails.second = path[2];
        result.curFracTail = -1;
        break;
    case 2: // merge
            // std::cout << "corner edge merged without querying again" <<
            // std::endl;
        result.mergeBoundaryEdges(std::pair<int, int>(path[0], path[1]),
                                  std::pair<int, int>(path[1], path[2]),
                                  newVertPos.row(0));
        // DISABLE logFile << "corner edge merged without querying again" <<
        // std::endl;

        result.computeFeatures(); // TODO: only update locally
        isMerge = true;
        break;
    default:
        assert(0);
        break;
    }
    if (scaffolding) {
        scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
        result.scaffold = &scaffold;
        scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
        scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
    }
    updateEnergyData(true, false, true);
    fractureInitiated = true;
    if (allowPropagate)
        propagateFracture = 1 + isMerge;

    return true;
}

bool Optimizer::createFracture(double stressThres, int propType,
                               bool allowPropagate, bool allowInSplit) {
    if (propType == 0) {
        topoIter++;
    }
    bool changed = false;
    bool isMerge = false;
    data_findExtrema = result;
    switch (propType) {
    case 0: // initiation
        changed = result.splitOrMerge(1.0 - energyParams[0], stressThres, false,
                                      allowInSplit, isMerge);
        break;
    case 1: // propagate split
        changed = result.splitEdge(1.0 - energyParams[0], stressThres, true,
                                   allowInSplit);
        break;
    case 2: // propagate merge
        changed = result.mergeEdge(1.0 - energyParams[0], stressThres, true);
        isMerge = true;
        break;
    }
    if (changed) {
        if (scaffolding) {
            scaffold =
                Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
            result.scaffold = &scaffold;
            scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
            scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
        }
        updateEnergyData(true, false, true);
        fractureInitiated = true;
        if (allowPropagate && (propType == 0)) {
            propagateFracture = 1 + isMerge;
        }
    }
    return changed;
}

bool Optimizer::solve_oneStep(void) {
    if (needRefactorize) {
        // for the changing hessian
        // std::cout << "recompute proxy/Hessian matrix..." << std::endl;
        if (!fractureInitiated)
            computeHessian(result, scaffold);
        // std::cout << "factorizing proxy/Hessian matrix..." << std::endl;
        if (!fractureInitiated) {
            if (!useDense) {
                if (scaffolding) {
                    linSysSolver->set_pattern(
                        scaffolding ? vNeighbor_withScaf : result.vNeighbor,
                        scaffolding ? fixedV_withScaf : result.fixedVert);
                    linSysSolver->update_a(I_mtr, J_mtr, V_mtr);
                    linSysSolver->analyze_pattern();
                } else {
                    linSysSolver->update_a(I_mtr, J_mtr, V_mtr);
                }
            }
        }
        try {
            if (useDense)
                denseSolver = Hessian.ldlt();
            else
                linSysSolver->factorize();
        } catch (std::exception e) {
            exit(-1);
        }
    }
    Eigen::VectorXd minusG = -gradient;
    if (useDense)
        searchDir = denseSolver.solve(minusG);
    else
        linSysSolver->solve(minusG, searchDir);
    fractureInitiated = false;

    return lineSearch();
}

bool Optimizer::lineSearch(void) {
    bool stopped = false;
    double stepSize = 1.0;
    initStepSize(result, stepSize);
    stepSize *= 0.99; // producing degenerated element is not allowed
    double lastEnergyVal_scaffold = 0.0;
    Eigen::MatrixXd resultV0 = result.V;
    Eigen::MatrixXd scaffoldV0;
    if (scaffolding) {
        scaffoldV0 = scaffold.airMesh.V;
        computeEnergyVal(
            result, scaffold,
            lastEnergyVal); // this update is necessary since scaffold changes
        lastEnergyVal_scaffold = energyVal_scaffold;
    }
    stepForward(resultV0, scaffoldV0, result, scaffold, stepSize);
    double testingE;
    computeEnergyVal(result, scaffold, testingE);
    while (testingE > lastEnergyVal) // ensure energy decrease
    {
        stepSize /= 2.0;
        if (stepSize == 0.0) {
            stopped = true;
            if (!mute) {
                // DISABLE logFile << "testingE" << globalIterNum << " " <<
                // testingE << " > " << lastEnergyVal << std::endl;
            }
            break;
        }
        stepForward(resultV0, scaffoldV0, result, scaffold, stepSize);
        computeEnergyVal(result, scaffold, testingE);
    }
    // std::cout << stepSize << "(armijo) ";
    while ((!result.checkInversion()) ||
           ((scaffolding) && (!scaffold.airMesh.checkInversion()))) {
        assert(0 && "element inversion after armijo shouldn't happen!");
        stepSize /= 2.0;
        if (stepSize == 0.0) {
            assert(0 && "line search failed!");
            stopped = true;
            break;
        }
        stepForward(resultV0, scaffoldV0, result, scaffold, stepSize);
        computeEnergyVal(result, scaffold, testingE);
    }
    lastEDec = lastEnergyVal - testingE;
    if (scaffolding)
        lastEDec += (-lastEnergyVal_scaffold + energyVal_scaffold);
    if (allowEDecRelTol && (lastEDec / lastEnergyVal < 1.0e-6 * stepSize) &&
        (stepSize > 1.0e-3)) { // avoid stopping in hard situations
        stopped = true;
    }
    lastEnergyVal = testingE;
    // std::cout << stepSize << std::endl;
    // std::cout << "stepLen = " << (stepSize * searchDir).squaredNorm() <<
    // std::endl; std::cout << "E_cur_smooth = " << testingE -
    // energyVal_scaffold << std::endl;

    return stopped;
}

void Optimizer::stepForward(const Eigen::MatrixXd &dataV0,
                            const Eigen::MatrixXd &scaffoldV0, TriMesh &data,
                            Scaffold &scaffoldData, double stepSize) const {
    assert(dataV0.rows() == data.V.rows());
    if (scaffolding)
        assert(data.V.rows() + scaffoldData.airMesh.V.rows() -
                   scaffoldData.bnd.size() ==
               searchDir.size() / 2);
    else
        assert(data.V.rows() * 2 == searchDir.size());
    assert(data.V.rows() == result.V.rows());

    for (int vI = 0; vI < data.V.rows(); vI++) {
        data.V(vI, 0) = dataV0(vI, 0) + stepSize * searchDir[vI * 2];
        data.V(vI, 1) = dataV0(vI, 1) + stepSize * searchDir[vI * 2 + 1];
    }
    if (scaffolding)
        scaffoldData.stepForward(scaffoldV0, searchDir, stepSize);
}

void Optimizer::updateTargetGRes(void) {
    targetGRes =
        energyParamSum *
        static_cast<double>(data0.V_rest.rows() - data0.fixedVert.size()) /
        static_cast<double>(data0.V_rest.rows()) * relGL2Tol;
}

void Optimizer::getGradientVisual(Eigen::MatrixXd &arrowVec) const {
    assert(result.V.rows() * 2 == gradient.size());
    arrowVec.resize(result.V.rows(), result.V.cols());
    for (int vI = 0; vI < result.V.rows(); vI++) {
        arrowVec(vI, 0) = gradient[vI * 2];
        arrowVec(vI, 1) = gradient[vI * 2 + 1];
        arrowVec.row(vI).normalize();
    }
    arrowVec *= igl::avg_edge_length(result.V, result.F);
}

void Optimizer::initStepSize(const TriMesh &data, double &stepSize) const {
    for (int eI = 0; eI < energyTerms.size(); eI++)
        energyTerms[eI]->initStepSize(data, searchDir, stepSize);
    if (scaffolding) {
        Eigen::VectorXd searchDir_scaffold;
        scaffold.wholeSearchDir2airMesh(searchDir, searchDir_scaffold);
        SymDirichletEnergy SD;
        SD.initStepSize(scaffold.airMesh, searchDir_scaffold, stepSize);
    }
}
void Optimizer::computeEnergyVal(const TriMesh &data,
                                 const Scaffold &scaffoldData,
                                 double &energyVal, bool excludeScaffold) {
    energyTerms[0]->computeEnergyVal(data, energyVal_ET[0]);
    energyVal = energyParams[0] * energyVal_ET[0];
    for (int eI = 1; eI < energyTerms.size(); eI++) {
        energyTerms[eI]->computeEnergyVal(data, energyVal_ET[eI]);
        energyVal += energyParams[eI] * energyVal_ET[eI];
    }
    if (scaffolding && (!excludeScaffold)) {
        SymDirichletEnergy SD;
        SD.computeEnergyVal(scaffoldData.airMesh, energyVal_scaffold, true);
        energyVal_scaffold *= w_scaf / scaffold.airMesh.F.rows();
        energyVal += energyVal_scaffold;
    } else {
        energyVal_scaffold = 0.0;
    }
}
void Optimizer::computeGradient(const TriMesh &data,
                                const Scaffold &scaffoldData,
                                Eigen::VectorXd &gradient,
                                bool excludeScaffold) {
    energyTerms[0]->computeGradient(data, gradient_ET[0]);
    gradient = energyParams[0] * gradient_ET[0];
    for (int eI = 1; eI < energyTerms.size(); eI++) {
        energyTerms[eI]->computeGradient(data, gradient_ET[eI]);
        gradient += energyParams[eI] * gradient_ET[eI];
    }
    if (scaffolding) {
        SymDirichletEnergy SD;
        SD.computeGradient(scaffoldData.airMesh, gradient_scaffold, true);
        scaffoldData.augmentGradient(
            gradient, gradient_scaffold,
            (excludeScaffold ? 0.0 : (w_scaf / scaffold.airMesh.F.rows())));
    }
}
void Optimizer::computeHessian(const TriMesh &data,
                               const Scaffold &scaffoldData) {
    if (useDense) {
        energyTerms[0]->computeHessian(data, Hessian);
        Hessian *= energyParams[0];
        for (int eI = 1; eI < energyTerms.size(); eI++) {
            Eigen::MatrixXd HessianI;
            energyTerms[eI]->computeHessian(data, HessianI);
            Hessian += energyParams[eI] * HessianI;
        }
        if (scaffolding) {
            SymDirichletEnergy SD;
            Eigen::MatrixXd Hessian_scaf;
            SD.computeHessian(scaffoldData.airMesh, Hessian_scaf, true);
            scaffoldData.augmentProxyMatrix(Hessian, Hessian_scaf,
                                            w_scaf / scaffold.airMesh.F.rows());
        }
    } else {
        I_mtr.resize(0);
        J_mtr.resize(0);
        V_mtr.resize(0);
        for (int eI = 0; eI < energyTerms.size(); eI++) {
            Eigen::VectorXi I, J;
            Eigen::VectorXd V;
            energyTerms[eI]->computeHessian(data, &V, &I, &J);
            V *= energyParams[eI];
            I_mtr.conservativeResize(I_mtr.size() + I.size());
            I_mtr.bottomRows(I.size()) = I;
            J_mtr.conservativeResize(J_mtr.size() + J.size());
            J_mtr.bottomRows(J.size()) = J;
            V_mtr.conservativeResize(V_mtr.size() + V.size());
            V_mtr.bottomRows(V.size()) = V;
        }
        if (scaffolding) {
            SymDirichletEnergy SD;
            Eigen::VectorXi I, J;
            Eigen::VectorXd V;
            SD.computeHessian(scaffoldData.airMesh, &V, &I, &J, true);
            scaffoldData.augmentProxyMatrix(I_mtr, J_mtr, V_mtr, I, J, V,
                                            w_scaf / scaffold.airMesh.F.rows());
        }
    }
}

double Optimizer::getLastEnergyVal(bool excludeScaffold) const {
    return ((excludeScaffold && scaffolding)
                ? (lastEnergyVal - energyVal_scaffold)
                : lastEnergyVal);
}
} // namespace uvgami
