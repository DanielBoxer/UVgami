//  Created by Minchen Li on 9/3/17.

#include <limits>
#include <fstream>
#include <cfloat>

#include "uvgami.h"
#include "SymDirichletEnergy.hpp"
#include "IglUtils.hpp"

#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>

#include "ParallelFor.hpp"

namespace uvgami {
SymDirichletEnergy::SymDirichletEnergy(void) : Energy(true) {}
void SymDirichletEnergy::getEnergyValPerElem(const TriMesh &data,
                                             Eigen::VectorXd &energyValPerElem,
                                             bool uniformWeight) const {
    const double normalizer_div = data.surfaceArea;

    energyValPerElem.resize(data.F.rows());
    parallelFor((int)data.F.rows(), [&](int triI) {
        const Eigen::Vector3i &triVInd = data.F.row(triI);

        const Eigen::RowVector2d &U1 = data.V.row(triVInd[0]);
        const Eigen::RowVector2d &U2 = data.V.row(triVInd[1]);
        const Eigen::RowVector2d &U3 = data.V.row(triVInd[2]);

        const Eigen::RowVector2d U2m1 = U2 - U1;
        const Eigen::RowVector2d U3m1 = U3 - U1;

        const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);

        // a flat rest triangle with a flat uv one gives 0 * inf
        if (area_U <= 0.0) {
            energyValPerElem[triI] = DBL_MAX;
            return;
        }
        const double w =
            (uniformWeight ? 1.0
                           : (data.faceWeight[triI] * data.triArea[triI] /
                              normalizer_div));
        energyValPerElem[triI] =
            w * (1.0 + data.triAreaSq[triI] / area_U / area_U) *
            ((U3m1.squaredNorm() * data.e0SqLen[triI] +
              U2m1.squaredNorm() * data.e1SqLen[triI]) /
                 4 / data.triAreaSq[triI] -
             U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI]);
    });
}

void SymDirichletEnergy::getEnergyValByElemID(const TriMesh &data, int elemI,
                                              double &energyVal,
                                              bool uniformWeight) const {
    const double normalizer_div = data.surfaceArea;

    int triI = elemI;
    const Eigen::Vector3i &triVInd = data.F.row(triI);

    const Eigen::Vector2d &U1 = data.V.row(triVInd[0]);
    const Eigen::Vector2d &U2 = data.V.row(triVInd[1]);
    const Eigen::Vector2d &U3 = data.V.row(triVInd[2]);

    const Eigen::Vector2d U2m1 = U2 - U1;
    const Eigen::Vector2d U3m1 = U3 - U1;

    const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
    if (area_U <= 0.0) {
        energyVal = DBL_MAX;
        return;
    }

    const double w =
        (uniformWeight
             ? 1.0
             : (data.faceWeight[triI] * data.triArea[triI] / normalizer_div));
    energyVal =
        w * (1.0 + data.triAreaSq[triI] / area_U / area_U) *
        ((U3m1.squaredNorm() * data.e0SqLen[triI] +
          U2m1.squaredNorm() * data.e1SqLen[triI]) /
             4 / data.triAreaSq[triI] -
         U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI]);
}
void SymDirichletEnergy::computeDivGradPerVert(
    const TriMesh &data, Eigen::VectorXd &divGradPerVert) const {
    Eigen::MatrixXd localGradients;
    computeLocalGradient(data, localGradients);

#define STANDARD_DEVIATION_FILTERING 1
#ifdef STANDARD_DEVIATION_FILTERING
    // NOTE: no need to weight by area since gradient already contains area
    // information
    // TODO: don't need to compute mean because it's always zero at stationary,
    // go back to the simpler version?
    Eigen::MatrixXd mean = Eigen::MatrixXd::Zero(data.V_rest.rows(), 2);
    Eigen::VectorXi incTriAmt = Eigen::VectorXi::Zero(data.V_rest.rows());
    for (int triI = 0; triI < data.F.rows(); triI++) {
        const Eigen::RowVector3i &triVInd = data.F.row(triI);
        int locGradStartInd = triI * 3;
        for (int i = 0; i < 3; i++) {
            mean.row(triVInd[i]) += localGradients.row(locGradStartInd + i);
            incTriAmt[triVInd[i]]++;
        }
    }
    for (int vI = 0; vI < data.V_rest.rows(); vI++) {
        mean.row(vI) /= incTriAmt[vI];
    }

    Eigen::VectorXd standardDeviation =
        Eigen::VectorXd::Zero(data.V_rest.rows());
    for (int triI = 0; triI < data.F.rows(); triI++) {
        const Eigen::RowVector3i &triVInd = data.F.row(triI);
        int locGradStartInd = triI * 3;
        for (int i = 0; i < 3; i++) {
            standardDeviation[triVInd[i]] +=
                (localGradients.row(locGradStartInd + i) - mean.row(triVInd[i]))
                    .squaredNorm();
        }
    }

    divGradPerVert = Eigen::VectorXd::Zero(data.V_rest.rows());
    for (int vI = 0; vI < data.V_rest.rows(); vI++) {
        if (incTriAmt[vI] == 1) {
            // impossible to be split
            divGradPerVert[vI] = 0.0;
        } else {
            divGradPerVert[vI] =
                std::sqrt(standardDeviation[vI] / (incTriAmt[vI] - 1.0));
        }
    }
#else
    divGradPerVert = Eigen::VectorXd::Zero(data.V_rest.rows());
    for (int triI = 0; triI < data.F.rows(); triI++) {
        const Eigen::RowVector3i &triVInd = data.F.row(triI);
        const Eigen::RowVector2d eDir[3] = {
            (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).normalized(),
            (data.V.row(triVInd[2]) - data.V.row(triVInd[1])).normalized(),
            (data.V.row(triVInd[0]) - data.V.row(triVInd[2])).normalized()
            //                (data.V.row(triVInd[1]) - data.V.row(triVInd[0])),
            //                (data.V.row(triVInd[2]) - data.V.row(triVInd[1])),
            //                (data.V.row(triVInd[0]) - data.V.row(triVInd[2]))
        };
        int locGradStartInd = triI * 3;
        for (int i = 0; i < 3; i++) {
            const Eigen::RowVector2d centralDir =
                (eDir[i] - eDir[(i + 2) % 3]).normalized();
            //                const Eigen::RowVector2d centralDir(eDir[(i+1) %
            //                3][1], -eDir[(i+1) % 3][0]);
            const double w = std::acos(
                (std::max)(-1.0,
                           (std::min)(1.0, eDir[i].dot(-eDir[(i + 2) % 3]))));
            divGradPerVert[triVInd[i]] +=
                w * -localGradients.row(locGradStartInd + i).dot(centralDir);
            //                divGradPerVert[triVInd[i]] +=
            //                -localGradients.row(locGradStartInd +
            //                i).dot(centralDir);

            // TODO: when querying boundary verts, no need to compute for
            // interior verts
            // TODO: if want to compare boundary with interior, w needs to be
            // /pi or /2pi
        }
    }

    //        double minDiv = divGradPerVert.minCoeff();
    //        minDiv -= std::abs(minDiv) * 1.0e-3;
    for (int vI = 0; vI < data.V_rest.rows(); vI++) {
        // prefer to split stretched regions:
        //            if(data.vNeighbor[vI].size() <= 2) {
        //                // impossible to be split
        //                divGradPerVert[vI] = minDiv;
        //            }

        // if want to split both compressed and stretched regions:
        if (data.vNeighbor[vI].size() <= 2) {
            // impossible to be split
            divGradPerVert[vI] = 0.0;
        } else {
            divGradPerVert[vI] = std::abs(divGradPerVert[vI]);
        }
    }
#endif
}
void SymDirichletEnergy::computeLocalGradient(
    const TriMesh &data, Eigen::MatrixXd &localGradients) const {
    const double normalizer_div = data.surfaceArea;

    localGradients.resize(data.F.rows() * 3, 2);
    for (int triI = 0; triI < data.F.rows(); triI++) {
        const Eigen::Vector3i &triVInd = data.F.row(triI);

        const Eigen::Vector2d &U1 = data.V.row(triVInd[0]);
        const Eigen::Vector2d &U2 = data.V.row(triVInd[1]);
        const Eigen::Vector2d &U3 = data.V.row(triVInd[2]);

        const Eigen::Vector2d U2m1 = U2 - U1;
        const Eigen::Vector2d U3m1 = U3 - U1;

        const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);

        const double leftTerm = 1.0 + data.triAreaSq[triI] / area_U / area_U;
        const double rightTerm =
            (U3m1.squaredNorm() * data.e0SqLen[triI] +
             U2m1.squaredNorm() * data.e1SqLen[triI]) /
                4 / data.triAreaSq[triI] -
            U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI];

        const double areaRatio =
            data.triAreaSq[triI] / area_U / area_U / area_U;
        const double w =
            data.faceWeight[triI] * data.triArea[triI] / normalizer_div;
        const int startRowI = triI * 3;

        const Eigen::Vector2d edge_oppo1 = U3 - U2;
        const Eigen::Vector2d dLeft1 =
            areaRatio * Eigen::Vector2d(edge_oppo1[1], -edge_oppo1[0]);
        const Eigen::Vector2d dRight1 =
            ((data.e0dote1[triI] - data.e0SqLen[triI]) * U3m1 +
             (data.e0dote1[triI] - data.e1SqLen[triI]) * U2m1) /
            2.0 / data.triAreaSq[triI];
        localGradients.row(startRowI) =
            w * (dLeft1 * rightTerm + dRight1 * leftTerm);

        const Eigen::Vector2d edge_oppo2 = U1 - U3;
        const Eigen::Vector2d dLeft2 =
            areaRatio * Eigen::Vector2d(edge_oppo2[1], -edge_oppo2[0]);
        const Eigen::Vector2d dRight2 =
            (data.e1SqLen[triI] * U2m1 - data.e0dote1[triI] * U3m1) / 2.0 /
            data.triAreaSq[triI];
        localGradients.row(startRowI + 1) =
            w * (dLeft2 * rightTerm + dRight2 * leftTerm);

        const Eigen::Vector2d edge_oppo3 = U2 - U1;
        const Eigen::Vector2d dLeft3 =
            areaRatio * Eigen::Vector2d(edge_oppo3[1], -edge_oppo3[0]);
        const Eigen::Vector2d dRight3 =
            (data.e0SqLen[triI] * U3m1 - data.e0dote1[triI] * U2m1) / 2.0 /
            data.triAreaSq[triI];
        localGradients.row(startRowI + 2) =
            w * (dLeft3 * rightTerm + dRight3 * leftTerm);
    }
}

void SymDirichletEnergy::computeGradient(const TriMesh &data,
                                         Eigen::VectorXd &gradient,
                                         bool uniformWeight) const {
    const double normalizer_div = data.surfaceArea;

    gradient.resize(data.V.rows() * 2);
    gradient.setZero();

    std::vector<Eigen::Matrix<double, 6, 1>> triGrads(data.F.rows());
    parallelFor((int)data.F.rows(), [&](int triI) {
        const Eigen::Vector3i &triVInd = data.F.row(triI);

        const Eigen::Vector2d &U1 = data.V.row(triVInd[0]);
        const Eigen::Vector2d &U2 = data.V.row(triVInd[1]);
        const Eigen::Vector2d &U3 = data.V.row(triVInd[2]);

        const Eigen::Vector2d U2m1 = U2 - U1;
        const Eigen::Vector2d U3m1 = U3 - U1;

        const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);

        const double leftTerm = 1.0 + data.triAreaSq[triI] / area_U / area_U;
        const double rightTerm =
            (U3m1.squaredNorm() * data.e0SqLen[triI] +
             U2m1.squaredNorm() * data.e1SqLen[triI]) /
                4 / data.triAreaSq[triI] -
            U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI];

        const double areaRatio =
            data.triAreaSq[triI] / area_U / area_U / area_U;
        const double w =
            (uniformWeight ? 1.0
                           : (data.faceWeight[triI] * data.triArea[triI] /
                              normalizer_div));

        Eigen::Matrix<double, 6, 1> &triGrad = triGrads[triI];

        const Eigen::Vector2d edge_oppo1 = U3 - U2;
        const Eigen::Vector2d dLeft1 =
            areaRatio * Eigen::Vector2d(edge_oppo1[1], -edge_oppo1[0]);
        const Eigen::Vector2d dRight1 =
            ((data.e0dote1[triI] - data.e0SqLen[triI]) * U3m1 +
             (data.e0dote1[triI] - data.e1SqLen[triI]) * U2m1) /
            2.0 / data.triAreaSq[triI];
        triGrad.segment(0, 2) = w * (dLeft1 * rightTerm + dRight1 * leftTerm);

        const Eigen::Vector2d edge_oppo2 = U1 - U3;
        const Eigen::Vector2d dLeft2 =
            areaRatio * Eigen::Vector2d(edge_oppo2[1], -edge_oppo2[0]);
        const Eigen::Vector2d dRight2 =
            (data.e1SqLen[triI] * U2m1 - data.e0dote1[triI] * U3m1) / 2.0 /
            data.triAreaSq[triI];
        triGrad.segment(2, 2) = w * (dLeft2 * rightTerm + dRight2 * leftTerm);

        const Eigen::Vector2d edge_oppo3 = U2 - U1;
        const Eigen::Vector2d dLeft3 =
            areaRatio * Eigen::Vector2d(edge_oppo3[1], -edge_oppo3[0]);
        const Eigen::Vector2d dRight3 =
            (data.e0SqLen[triI] * U3m1 - data.e0dote1[triI] * U2m1) / 2.0 /
            data.triAreaSq[triI];
        triGrad.segment(4, 2) = w * (dLeft3 * rightTerm + dRight3 * leftTerm);
    });

    // serial scatter keeps the fp accumulation order fixed
    for (int triI = 0; triI < data.F.rows(); triI++) {
        const Eigen::Vector3i &triVInd = data.F.row(triI);
        gradient.block(triVInd[0] * 2, 0, 2, 1) += triGrads[triI].segment(0, 2);
        gradient.block(triVInd[1] * 2, 0, 2, 1) += triGrads[triI].segment(2, 2);
        gradient.block(triVInd[2] * 2, 0, 2, 1) += triGrads[triI].segment(4, 2);
    }

    for (const auto fixedVI : data.fixedVert) {
        gradient[2 * fixedVI] = 0.0;
        gradient[2 * fixedVI + 1] = 0.0;
    }
}


// the uv hessian is zero along the two translations
static const Eigen::Matrix<double, 6, 4> TRANSLATION_FREE_BASIS = [] {
    Eigen::Matrix<double, 6, 4> basis;
    const double half = 1.0 / std::sqrt(2.0), sixth = 1.0 / std::sqrt(6.0);
    basis << half, 0.0, sixth, 0.0, 0.0, half, 0.0, sixth, -half, 0.0, sixth,
        0.0, 0.0, -half, 0.0, sixth, 0.0, 0.0, -2.0 * sixth, 0.0, 0.0, 0.0,
        0.0, -2.0 * sixth;
    return basis;
}();
const int SECULAR_NEWTON_STEPS = 30;
const double SECULAR_F_TOLERANCE = 1.0e-12;
const double SECULAR_STEP_TOLERANCE = 1.0e-13;
const double NEGLIGIBLE_EIGENVALUE = 1.0e-13;
const double NEGATIVE_EIGENPAIR_RESIDUAL = 1.0e-12;

static Eigen::Vector4d fourDimensionalCross(const Eigen::Vector4d &a,
                                            const Eigen::Vector4d &b,
                                            const Eigen::Vector4d &c) {
    Eigen::Vector4d n;
    double sign = 1.0;
    for (int i = 0; i < 4; i++) {
        Eigen::Matrix3d minor;
        int column = 0;
        for (int j = 0; j < 4; j++) {
            if (j == i)
                continue;
            minor(0, column) = a[j];
            minor(1, column) = b[j];
            minor(2, column) = c[j];
            column++;
        }
        n[i] = sign * minor.determinant();
        sign = -sign;
    }
    return n;
}

// cholesky, P - shift I is positive definite for any shift below zero
static Eigen::Vector4d solveShifted(const Eigen::Matrix4d &P, double shift,
                                    const Eigen::Vector4d &rhs) {
    Eigen::Matrix4d L;
    for (int j = 0; j < 4; j++) {
        double diagonal = P(j, j) - shift;
        for (int k = 0; k < j; k++)
            diagonal -= L(j, k) * L(j, k);
        L(j, j) = std::sqrt(diagonal);
        for (int i = j + 1; i < 4; i++) {
            double sum = P(i, j);
            for (int k = 0; k < j; k++)
                sum -= L(i, k) * L(j, k);
            L(i, j) = sum / L(j, j);
        }
    }
    Eigen::Vector4d x;
    for (int i = 0; i < 4; i++) {
        double sum = rhs[i];
        for (int k = 0; k < i; k++)
            sum -= L(i, k) * x[k];
        x[i] = sum / L(i, i);
    }
    for (int i = 3; i >= 0; i--) {
        double sum = x[i];
        for (int k = i + 1; k < 4; k++)
            sum -= L(k, i) * x[k];
        x[i] = sum / L(i, i);
    }
    return x;
}

// the one negative eigenvalue is the root of f(mu) = 1 + beta t^T (P - mu I)^-1 t
static void clampNegativeEigenvalue(Eigen::Matrix<double, 6, 6> &hessian,
                                    const Eigen::Matrix<double, 6, 1> modes[4],
                                    const double scaledEigenvalues[4]) {
    const Eigen::Matrix<double, 6, 4> &Q = TRANSLATION_FREE_BASIS;
    Eigen::Matrix4d P = Eigen::Matrix4d::Zero();
    Eigen::Vector4d reducedModes[3];
    for (int modeI = 0; modeI < 3; modeI++) {
        reducedModes[modeI] = Q.transpose() * modes[modeI];
        P += scaledEigenvalues[modeI] * reducedModes[modeI] *
             reducedModes[modeI].transpose();
    }
    const Eigen::Vector4d t = Q.transpose() * modes[3];
    const double beta = scaledEigenvalues[3];
    // a rayleigh quotient is always above the eigenvalue
    const Eigen::Vector4d nullDirection = fourDimensionalCross(
        reducedModes[0], reducedModes[1], reducedModes[2]).normalized();
    const double nullQuotient = beta * std::pow(t.dot(nullDirection), 2);
    const double twistQuotient =
        (t.dot(P * t) + beta * std::pow(t.squaredNorm(), 2)) / t.squaredNorm();
    double low = beta * t.squaredNorm(), high = 0.0;
    // a twist term below rounding is noise
    if (-low <= NEGLIGIBLE_EIGENVALUE * P.norm())
        return;
    double mu = (std::min)(nullQuotient, twistQuotient);
    if (!(mu < 0.0))
        mu = 0.5 * low;
    Eigen::Vector4d y = t;
    for (int stepI = 0; stepI < SECULAR_NEWTON_STEPS; stepI++) {
        y = solveShifted(P, mu, t);
        const double f = 1.0 + beta * t.dot(y);
        if (std::abs(f) <= SECULAR_F_TOLERANCE)
            break;
        if (f > 0.0)
            low = mu;
        else
            high = mu;
        double next = mu - f / (beta * y.squaredNorm());
        if (std::abs(next - mu) <= SECULAR_STEP_TOLERANCE * std::abs(mu))
            break;
        if (!(next > low && next < high))
            next = 0.5 * (low + high);
        mu = next;
    }
    const Eigen::Matrix4d reduced = P + beta * t * t.transpose();
    const Eigen::Vector4d x = y.normalized();
    const double residual = (reduced * x - mu * x).norm();
    if (mu < 0.0 && residual <= NEGATIVE_EIGENPAIR_RESIDUAL * reduced.norm()) {
        const Eigen::Matrix<double, 6, 1> eigenvector = Q * x;
        hessian -= mu * eigenvector * eigenvector.transpose();
        return;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigenSolver(reduced);
    Eigen::Vector4d clamped = eigenSolver.eigenvalues();
    for (int i = 0; i < 4; ++i)
        if (clamped[i] < 0.0)
            clamped[i] = 0.0;
    const Eigen::Matrix<double, 6, 4> QV = Q * eigenSolver.eigenvectors();
    hessian = QV * clamped.asDiagonal() * QV.transpose();
}

// analytic eigensystem from Smith, De Goes, Kim 2019, eq. 31
static void projectedTriangleHessian(const TriMesh &data, int triI,
                                     bool uniformWeight,
                                     Eigen::Matrix<double, 6, 6> &hessian) {
    const Eigen::Vector3i &triVInd = data.F.row(triI);
    const Eigen::Vector2d &U1 = data.V.row(triVInd[0]);
    const Eigen::Vector2d &U2 = data.V.row(triVInd[1]);
    const Eigen::Vector2d &U3 = data.V.row(triVInd[2]);

    // rest triangle laid flat with vertex 2 on the x axis
    const double e0Len = std::sqrt(data.e0SqLen[triI]);
    const double doubleArea = 2.0 * data.triArea[triI];
    Eigen::Matrix2d restInverse;
    restInverse << 1.0 / e0Len, -data.e0dote1[triI] / (doubleArea * e0Len),
        0.0, e0Len / doubleArea;
    Eigen::Matrix<double, 3, 2> vertexGradient;
    vertexGradient.row(1) = restInverse.row(0);
    vertexGradient.row(2) = restInverse.row(1);
    vertexGradient.row(0) = -(vertexGradient.row(1) + vertexGradient.row(2));

    Eigen::Matrix2d uvEdges;
    uvEdges.col(0) = U2 - U1;
    uvEdges.col(1) = U3 - U1;
    const Eigen::Matrix2d deformation = uvEdges * restInverse;

    // polar decomposition, det F > 0 since the line search rejects inversions
    const double a = deformation(0, 0), b = deformation(0, 1);
    const double c = deformation(1, 0), d = deformation(1, 1);
    const double rotationNorm = std::sqrt((a + d) * (a + d) + (b - c) * (b - c));
    Eigen::Matrix2d rotation;
    rotation << a + d, b - c, c - b, a + d;
    rotation /= rotationNorm;
    const Eigen::Matrix2d stretch = rotation.transpose() * deformation;
    const double angle =
        0.5 * std::atan2(2.0 * stretch(0, 1), stretch(0, 0) - stretch(1, 1));
    const double cosine = std::cos(angle), sine = std::sin(angle);
    Eigen::Matrix2d V;
    V << cosine, -sine, sine, cosine;
    const Eigen::Matrix2d U = rotation * V;
    const double s1 = stretch(0, 0) * cosine * cosine +
                      2.0 * stretch(0, 1) * sine * cosine +
                      stretch(1, 1) * sine * sine;
    const double s2 = stretch(0, 0) * sine * sine -
                      2.0 * stretch(0, 1) * sine * cosine +
                      stretch(1, 1) * cosine * cosine;

    const double I2 = s1 * s1 + s2 * s2;
    const double I3 = s1 * s2;
    const double I3Sq = I3 * I3;
    double eigenvalues[4] = {1.0 + 3.0 / (s1 * s1 * s1 * s1),
                             1.0 + 3.0 / (s2 * s2 * s2 * s2),
                             1.0 + 1.0 / I3Sq + I2 / (I3Sq * I3),
                             1.0 + 1.0 / I3Sq - I2 / (I3Sq * I3)};

    const Eigen::Vector2d u1 = U.col(0), u2 = U.col(1);
    const Eigen::Vector2d v1 = V.col(0), v2 = V.col(1);
    const double invSqrt2 = 1.0 / std::sqrt(2.0);
    const Eigen::Matrix2d eigenmatrices[4] = {
        u1 * v1.transpose(), u2 * v2.transpose(),
        invSqrt2 * (u1 * v2.transpose() + u2 * v1.transpose()),
        invSqrt2 * (u2 * v1.transpose() - u1 * v2.transpose())};

    // the triangle energy is twice the psi of Smith, De Goes, Kim 2019
    const double weight =
        2.0 * (uniformWeight ? 1.0
                             : data.faceWeight[triI] * data.triArea[triI] /
                                   data.surfaceArea);
    Eigen::Matrix<double, 6, 1> modes[4];
    double scaledEigenvalues[4];
    hessian.setZero();
    for (int modeI = 0; modeI < 4; modeI++) {
        for (int vI = 0; vI < 3; vI++)
            modes[modeI].segment<2>(vI * 2) =
                eigenmatrices[modeI] * vertexGradient.row(vI).transpose();
        scaledEigenvalues[modeI] = weight * eigenvalues[modeI];
        hessian += scaledEigenvalues[modeI] * modes[modeI] * modes[modeI].transpose();
    }
    // only the twist term can make it indefinite
    if (scaledEigenvalues[3] < 0.0)
        clampNegativeEigenvalue(hessian, modes, scaledEigenvalues);
}

void SymDirichletEnergy::computeHessian(const TriMesh &data,
                                        Eigen::MatrixXd &Hessian,
                                        bool uniformWeight) const {
    Hessian.setZero(data.V.rows() * 2, data.V.rows() * 2);

    std::vector<char> isFixedVert(data.V.rows(), 0);
    for (const auto fixedVI : data.fixedVert)
        isFixedVert[fixedVI] = 1;

    Eigen::Matrix<double, 6, 6> triHessian;
    for (int triI = 0; triI < data.F.rows(); triI++) {
        projectedTriangleHessian(data, triI, uniformWeight, triHessian);
        Eigen::Vector3i vInd = data.F.row(triI);
        for (int vI = 0; vI < 3; vI++) {
            if (isFixedVert[vInd[vI]]) {
                vInd[vI] = -1;
            }
        }
        IglUtils::addBlockToMatrix(triHessian, vInd, 2, Hessian);
    }

    Eigen::VectorXi fixedVertInd;
    fixedVertInd.resize(data.fixedVert.size());
    int fVI = 0;
    for (const auto fixedVI : data.fixedVert)
        fixedVertInd[fVI++] = fixedVI;
    IglUtils::addDiagonalToMatrix(
        Eigen::VectorXd::Ones(data.fixedVert.size() * 2), fixedVertInd, 2,
        Hessian);
}

void SymDirichletEnergy::computeHessian(const TriMesh &data, Eigen::VectorXd *V,
                                        Eigen::VectorXi *I, Eigen::VectorXi *J,
                                        bool uniformWeight) const {
    //        std::cout << "computing entry value..." << std::endl;
    //        clock_t start = clock();
    std::vector<char> isFixedVert(data.V.rows(), 0);
    for (const auto fixedVI : data.fixedVert)
        isFixedVert[fixedVI] = 1;

    std::vector<Eigen::Vector3i> vInds(data.F.rows());
    for (int triI = 0; triI < data.F.rows(); triI++) {
        Eigen::Vector3i &vInd = vInds[triI];
        vInd = data.F.row(triI);
        for (int vI = 0; vI < 3; vI++) {
            if (isFixedVert[vInd[vI]]) {
                vInd[vI] = -1;
            }
        }
    }
    // the per-triangle offsets keep the exact order of serial appends
    std::vector<int> triTripletStart(data.F.rows() + 1);
    triTripletStart[0] = static_cast<int>(V->size());
    for (int triI = 0; triI < data.F.rows(); triI++) {
        int numFree = 0;
        for (int vI = 0; vI < 3; vI++) {
            if (vInds[triI][vI] >= 0) {
                numFree++;
            }
        }
        triTripletStart[triI + 1] = triTripletStart[triI] + 4 * numFree * numFree;
    }
    V->conservativeResize(triTripletStart[data.F.rows()]);
    if (I) {
        I->conservativeResize(triTripletStart[data.F.rows()]);
        J->conservativeResize(triTripletStart[data.F.rows()]);
    }
    parallelFor((int)data.F.rows(), [&](int triI) {
        Eigen::Matrix<double, 6, 6> triHessian;
        projectedTriangleHessian(data, triI, uniformWeight, triHessian);
        IglUtils::addBlockToMatrix(triHessian, vInds[triI], 2, V, I, J,
                                   triTripletStart[triI]);
    });
    //        std::cout << static_cast<double>(clock() - start) / CLOCKS_PER_SEC
    //        << "s" << std::endl;

    Eigen::VectorXi fixedVertInd;
    fixedVertInd.resize(data.fixedVert.size());
    int fVI = 0;
    for (const auto fixedVI : data.fixedVert) {
        fixedVertInd[fVI++] = fixedVI;
    }
    IglUtils::addDiagonalToMatrix(
        Eigen::VectorXd::Ones(data.fixedVert.size() * 2), fixedVertInd, 2, V, I,
        J);
}

void SymDirichletEnergy::initStepSize(const TriMesh &data,
                                      const Eigen::VectorXd &searchDir,
                                      double &stepSize) const {
    assert(stepSize > 0.0);

    const double stepSizeInit = stepSize;
    // min reduce is fp-order-independent
    stepSize = parallelReduce(
        (int)data.F.rows(), stepSizeInit,
        [&](const tbb::blocked_range<int> &range, double localStep) -> double {
            for (int triI = range.begin(); triI != range.end(); triI++) {
                const Eigen::Vector3i &triVInd = data.F.row(triI);

                const Eigen::Vector2d &U1 = data.V.row(triVInd[0]);
                const Eigen::Vector2d &U2 = data.V.row(triVInd[1]);
                const Eigen::Vector2d &U3 = data.V.row(triVInd[2]);

                const Eigen::Vector2d V1(searchDir[triVInd[0] * 2],
                                         searchDir[triVInd[0] * 2 + 1]);
                const Eigen::Vector2d V2(searchDir[triVInd[1] * 2],
                                         searchDir[triVInd[1] * 2 + 1]);
                const Eigen::Vector2d V3(searchDir[triVInd[2] * 2],
                                         searchDir[triVInd[2] * 2 + 1]);

                const Eigen::Vector2d U2m1 = U2 - U1;
                const Eigen::Vector2d U3m1 = U3 - U1;
                const Eigen::Vector2d V2m1 = V2 - V1;
                const Eigen::Vector2d V3m1 = V3 - V1;

                const double a = V2m1[0] * V3m1[1] - V2m1[1] * V3m1[0];
                const double b = U2m1[0] * V3m1[1] - U2m1[1] * V3m1[0] +
                                 V2m1[0] * U3m1[1] - V2m1[1] * U3m1[0];
                const double c = U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0];
                assert(c > 0.0);
                const double delta = b * b - 4.0 * a * c;

                double bound = stepSizeInit;
                if (a > 0.0) {
                    if ((b < 0.0) && (delta >= 0.0)) {
                        bound = 2.0 * c / (-b + sqrt(delta));
                        // (same in math as (-b - sqrt(delta)) / 2.0 / a
                        //  but smaller numerical error when b < 0.0)
                        assert(bound > 0.0);
                    }
                } else if (a < 0.0) {
                    assert(delta > 0.0);
                    if (b < 0.0) {
                        bound = 2.0 * c / (-b + sqrt(delta));
                        // (same in math as (-b - sqrt(delta)) / 2.0 / a
                        //  but smaller numerical error when b < 0.0)
                    } else {
                        bound = (-b - sqrt(delta)) / 2.0 / a;
                    }
                    assert(bound > 0.0);
                } else {
                    if (b < 0.0) {
                        bound = -c / b;
                        assert(bound > 0.0);
                    }
                }

                if (bound < localStep) {
                    localStep = bound;
                }
            }
            return localStep;
        },
        [](double a, double b) { return (std::min)(a, b); });
}
} // namespace uvgami
