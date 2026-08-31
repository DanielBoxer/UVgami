//  Created by Minchen Li on 8/30/17.

#include <igl/cotmatrix.h>
#include <igl/avg_edge_length.h>
#include <igl/writeOBJ.h>
#include <igl/list_to_matrix.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>

#include <tbb/tbb.h>

#include "TriMesh.hpp"
#include <climits>
#include "IglUtils.hpp"
#include "SymDirichletEnergy.hpp"
#include "Optimizer.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <set>
#include <stdexcept>
#include <streambuf>

extern std::vector<std::pair<double, double>> energyChanges_bSplit,
    energyChanges_iSplit, energyChanges_merge;
extern std::vector<std::vector<int>> paths_bSplit, paths_iSplit, paths_merge;
extern std::vector<Eigen::MatrixXd> newVertPoses_bSplit, newVertPoses_iSplit,
    newVertPoses_merge;
extern double filterExp_in;
extern int inSplitTotalAmt;
extern bool pinnedMode;
extern std::atomic<bool> forceQuit;

namespace uvgami {

// stand-in rest area for a flat or needle triangle, a fraction of its own
// longest edge squared so a local query mesh sees the same shape it does.
// below it the energy is rounding noise and can come out negative
static const double ZERO_AREA_STAND_IN = 1e-6;

TriMesh::TriMesh(void) : surfaceArea(0), curFracTail(0), initSeamLen(0) {}

TriMesh::TriMesh(const Eigen::MatrixXd &V_mesh, const Eigen::MatrixXi &F_mesh,
                 const Eigen::MatrixXd &UV_mesh,
                 const Eigen::MatrixXi &FUV_mesh, bool separateTri,
                 double p_initSeamLen, double p_areaThres_AM) {
    initSeamLen = p_initSeamLen;
    areaThres_AM = p_areaThres_AM;

    bool multiComp = false; // TODO: detect whether the mesh is multi-component
    if (separateTri) {
        // duplicate vertices and edges, use new face vertex indices,
        // construct cohesive edge pairs,
        // compute triangle matrix to save rest shapes
        V_rest.resize(F_mesh.rows() * F_mesh.cols(), 3);
        V.resize(F_mesh.rows() * F_mesh.cols(), 2);
        F.resize(F_mesh.rows(), F_mesh.cols());
        std::map<std::pair<int, int>, Eigen::Vector3i> edge2DupInd;
        int cohEAmt = 0;
        for (int triI = 0; triI < F_mesh.rows(); triI++) {
            int vDupIndStart = triI * 3;

            if (UV_mesh.rows() == V_mesh.rows()) {
                // bijective map without seams, usually Tutte
                V.row(vDupIndStart) = UV_mesh.row(F_mesh.row(triI)[0]);
                V.row(vDupIndStart + 1) = UV_mesh.row(F_mesh.row(triI)[1]);
                V.row(vDupIndStart + 2) = UV_mesh.row(F_mesh.row(triI)[2]);
            }

            V_rest.row(vDupIndStart) = V_mesh.row(F_mesh.row(triI)[0]);
            V_rest.row(vDupIndStart + 1) = V_mesh.row(F_mesh.row(triI)[1]);
            V_rest.row(vDupIndStart + 2) = V_mesh.row(F_mesh.row(triI)[2]);

            F(triI, 0) = vDupIndStart;
            F(triI, 1) = vDupIndStart + 1;
            F(triI, 2) = vDupIndStart + 2;

            for (int vI = 0; vI < 3; vI++) {
                int vsI = F_mesh.row(triI)[vI],
                    veI = F_mesh.row(triI)[(vI + 1) % 3];
                auto cohEFinder =
                    edge2DupInd.find(std::pair<int, int>(veI, vsI));
                if (cohEFinder == edge2DupInd.end()) {
                    cohEAmt++;
                    edge2DupInd[std::pair<int, int>(vsI, veI)] =
                        Eigen::Vector3i(cohEAmt, F(triI, vI),
                                        F(triI, (vI + 1) % 3));
                } else {
                    edge2DupInd[std::pair<int, int>(vsI, veI)] =
                        Eigen::Vector3i(-cohEFinder->second[0], F(triI, vI),
                                        F(triI, (vI + 1) % 3));
                }
            }
        }

        cohE.resize(cohEAmt, 4);
        cohE.setConstant(-1);
        for (const auto &cohPI : edge2DupInd) {
            if (cohPI.second[0] > 0) {
                cohE.row(cohPI.second[0] - 1)[0] = cohPI.second[1];
                cohE.row(cohPI.second[0] - 1)[1] = cohPI.second[2];
            } else {
                cohE.row(-cohPI.second[0] - 1)[2] = cohPI.second[2];
                cohE.row(-cohPI.second[0] - 1)[3] = cohPI.second[1];
            }
        }

        if (UV_mesh.rows() == 0) {
            // no input UV
            initRigidUV();
        } else if (UV_mesh.rows() != V_mesh.rows()) {
            // input UV with seams
            assert(
                0 &&
                "TODO: separate each triangle in UV space according to FUV!");
        }
    } else {
        // deal with mesh
        if (UV_mesh.rows() == V_mesh.rows()) {
            // same vertex and uv index
            V_rest = V_mesh;
            V = UV_mesh;
            F = F_mesh;
        } else if (UV_mesh.rows() != 0) {
            // different vertex and uv index, split 3D surface according to UV
            // and merge back while saving into files
            assert(F_mesh.rows() == FUV_mesh.rows());
            // UV map contains seams
            // Split triangles along the seams on the surface (construct
            // cohesive edges there) to construct a bijective map
            std::set<std::pair<int, int>> HE_UV;
            std::map<std::pair<int, int>, std::pair<int, int>> HE;
            for (int triI = 0; triI < FUV_mesh.rows(); triI++) {
                const Eigen::RowVector3i &triVInd_UV = FUV_mesh.row(triI);
                HE_UV.insert(std::pair<int, int>(triVInd_UV[0], triVInd_UV[1]));
                HE_UV.insert(std::pair<int, int>(triVInd_UV[1], triVInd_UV[2]));
                HE_UV.insert(std::pair<int, int>(triVInd_UV[2], triVInd_UV[0]));
                const Eigen::RowVector3i &triVInd = F_mesh.row(triI);
                HE[std::pair<int, int>(triVInd[0], triVInd[1])] =
                    std::pair<int, int>(triI, 0);
                HE[std::pair<int, int>(triVInd[1], triVInd[2])] =
                    std::pair<int, int>(triI, 1);
                HE[std::pair<int, int>(triVInd[2], triVInd[0])] =
                    std::pair<int, int>(triI, 2);
            }
            std::vector<std::vector<int>> cohEdges;
            for (int triI = 0; triI < FUV_mesh.rows(); triI++) {
                const Eigen::RowVector3i &triVInd_UV = FUV_mesh.row(triI);
                const Eigen::RowVector3i &triVInd = F_mesh.row(triI);
                for (int eI = 0; eI < 3; eI++) {
                    int vI = eI, vI_post = (eI + 1) % 3;
                    if (HE_UV.find(std::pair<int, int>(triVInd_UV[vI_post],
                                                       triVInd_UV[vI])) ==
                        HE_UV.end()) {
                        // boundary edge in UV space
                        const auto finder = HE.find(
                            std::pair<int, int>(triVInd[vI_post], triVInd[vI]));
                        if (finder != HE.end()) {
                            // non-boundary edge on the surface
                            // construct cohesive edge pair
                            cohEdges.resize(cohEdges.size() + 1);
                            cohEdges.back().emplace_back(triVInd_UV[vI]);
                            cohEdges.back().emplace_back(triVInd_UV[vI_post]);
                            cohEdges.back().emplace_back(
                                FUV_mesh(finder->second.first,
                                         (finder->second.second + 1) % 3));
                            cohEdges.back().emplace_back(FUV_mesh(
                                finder->second.first, finder->second.second));
                            HE.erase(std::pair<int, int>(
                                triVInd[vI],
                                triVInd[vI_post])); // prevent from inserting
                                                    // again
                        }
                    }
                }
            }
            bool makeCoh = true;
            if (makeCoh)
                igl::list_to_matrix(cohEdges, cohE);

            V_rest.resize(UV_mesh.rows(), 3);
            V = UV_mesh;
            F = FUV_mesh;
            std::vector<bool> updated(UV_mesh.rows(), false);
            for (int triI = 0; triI < F_mesh.rows(); triI++) {
                const Eigen::RowVector3i &triVInd = F_mesh.row(triI);
                const Eigen::RowVector3i &triVInd_UV = FUV_mesh.row(triI);
                for (int vI = 0; vI < 3; vI++) {
                    if (!updated[triVInd_UV[vI]]) {
                        V_rest.row(triVInd_UV[vI]) = V_mesh.row(triVInd[vI]);
                        updated[triVInd_UV[vI]] = true;
                    }
                }
            }

            if (!makeCoh) {
                for (const auto &cohI : cohEdges)
                    initSeamLen +=
                        (V_rest.row(cohI[0]) - V_rest.row(cohI[1])).norm();
            } else {
                initSeams = cohE;
            }
        } else {
            assert(V_mesh.rows() > 0);
            assert(F_mesh.rows() > 0);
            V_rest = V_mesh;
            F = F_mesh;
            V = Eigen::MatrixXd::Zero(V_rest.rows(), 2);
            // DISABLE std::cout << "No UV provided, initialized to all 0" <<
            // std::endl;
        }
    }

    computeFeatures(false, true);

    vertWeight = Eigen::VectorXd::Ones(V.rows());
}

void initCylinder(double r1_x, double r1_y, double r2_x, double r2_y,
                  double height, int circle_res, int height_resolution,
                  Eigen::MatrixXd &V, Eigen::MatrixXi &F,
                  Eigen::MatrixXd *uv_coords_per_face = NULL,
                  Eigen::MatrixXi *uv_coords_face_ids = NULL) {
    int nvertices = circle_res * (height_resolution + 1);
    int nfaces = 2 * circle_res * height_resolution;

    V.resize(nvertices, 3);
    if (uv_coords_per_face)
        uv_coords_per_face->resize(nvertices, 2);
    F.resize(nfaces, 3);
    for (int j = 0; j < height_resolution + 1; j++) {
        for (int i = 0; i < circle_res; i++) {
            double t = (double)j / (double)height_resolution;
            double h = height * t;
            double theta = i * 2 * M_PI / circle_res;
            double r_x = r1_x * t + r2_x * (1 - t);
            double r_y = r1_y * t + r2_y * (1 - t);
            V.row(j * circle_res + i) =
                Eigen::Vector3d(r_x * cos(theta), height - h, r_y * sin(theta));
            if (uv_coords_per_face)
                uv_coords_per_face->row(j * circle_res + i) =
                    Eigen::Vector2d(r_x * cos(theta), r_y * sin(theta));
            if (j < height_resolution) {
                int vl0 = j * circle_res + i;
                int vl1 = j * circle_res + (i + 1) % circle_res;
                int vu0 = (j + 1) * circle_res + i;
                int vu1 = (j + 1) * circle_res + (i + 1) % circle_res;
                F.row(2 * (j * circle_res + i) + 0) =
                    Eigen::Vector3i(vl0, vl1, vu1);
                F.row(2 * (j * circle_res + i) + 1) =
                    Eigen::Vector3i(vu0, vl0, vu1);
            }
        }
    }
}

void TriMesh::computeLaplacianMtr(void) {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(V_rest, F, L);
    LaplacianMtr.resize(L.rows(), L.cols());
    LaplacianMtr.setZero();
    LaplacianMtr.reserve(L.nonZeros());
    for (int k = 0; k < L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
            if ((fixedVert.find(static_cast<int>(it.row())) ==
                 fixedVert.end()) &&
                (fixedVert.find(static_cast<int>(it.col())) ==
                 fixedVert.end())) {
                LaplacianMtr.insert(it.row(), it.col()) = -it.value();
            }
        }
    }
    for (const auto fixedVI : fixedVert)
        LaplacianMtr.insert(fixedVI, fixedVI) = 1.0;
    LaplacianMtr.makeCompressed();
}

void TriMesh::computeTriangleFeatures(void) {
    boundaryEdge.resize(cohE.rows());
    edgeLen.resize(cohE.rows());
    for (int cohI = 0; cohI < cohE.rows(); cohI++) {
        if (cohE.row(cohI).minCoeff() >= 0)
            boundaryEdge[cohI] = 0;
        else
            boundaryEdge[cohI] = 1;
        edgeLen[cohI] =
            (V_rest.row(cohE(cohI, 0)) - V_rest.row(cohE(cohI, 1))).norm();
    }

    // triangle count never changes after construction, so only initialize
    if (faceWeight.size() != F.rows())
        faceWeight = Eigen::VectorXd::Ones(F.rows());

    triNormal.resize(F.rows(), 3);
    triArea.resize(F.rows());
    surfaceArea = 0.0;
    triAreaSq.resize(F.rows());
    e0SqLen.resize(F.rows());
    e1SqLen.resize(F.rows());
    e0dote1.resize(F.rows());
    e0SqLen_div_dbAreaSq.resize(F.rows());
    e1SqLen_div_dbAreaSq.resize(F.rows());
    e0dote1_div_dbAreaSq.resize(F.rows());
    Eigen::VectorXd longestSq(F.rows());
    int zeroAreaAmt = 0;
    for (int triI = 0; triI < F.rows(); triI++) {
        const Eigen::Vector3i &triVInd = F.row(triI);
        const Eigen::Vector3d P1 = V_rest.row(triVInd[0]);
        const Eigen::Vector3d P2 = V_rest.row(triVInd[1]);
        const Eigen::Vector3d P3 = V_rest.row(triVInd[2]);
        const Eigen::RowVector3d normalVec = (P2 - P1).cross(P3 - P1);
        triNormal.row(triI) = normalVec.normalized();
        triArea[triI] = 0.5 * normalVec.norm();
        longestSq[triI] =
            (std::max)({(P2 - P1).squaredNorm(), (P3 - P1).squaredNorm(),
                        (P3 - P2).squaredNorm()});
        zeroAreaAmt += triArea[triI] == 0.0;
    }
    // a zero-area rest triangle has no shape to measure distortion against,
    // and a local query mesh can be nothing but those
    const double meanArea = zeroAreaAmt < F.rows()
                                ? triArea.sum() / (F.rows() - zeroAreaAmt)
                                : longestSq.mean();
    if (zeroAreaAmt && meanArea == 0.0)
        throw std::runtime_error("every rest triangle is a point");
    for (int triI = 0; triI < F.rows(); triI++) {
        const Eigen::Vector3i &triVInd = F.row(triI);

        const Eigen::Vector3d &P1 = V_rest.row(triVInd[0]);
        const Eigen::Vector3d &P2 = V_rest.row(triVInd[1]);
        const Eigen::Vector3d &P3 = V_rest.row(triVInd[2]);

        const Eigen::Vector3d P2m1 = P2 - P1;
        const Eigen::Vector3d P3m1 = P3 - P1;
        const Eigen::RowVector3d normalVec = P2m1.cross(P3m1);

        const double floorArea = (std::max)(
            areaThres_AM,
            ZERO_AREA_STAND_IN *
                (longestSq[triI] > 0.0 ? longestSq[triI] : meanArea));
        if (triArea[triI] < floorArea) {
            // air mesh triangle degeneracy prevention, and the stand-in
            triArea[triI] = floorArea;
            surfaceArea += floorArea;
            triAreaSq[triI] = floorArea * floorArea;
            e0SqLen[triI] = e1SqLen[triI] = 4.0 / std::sqrt(3.0) * floorArea;
            e0dote1[triI] = e0SqLen[triI] / 2.0;

            e0SqLen_div_dbAreaSq[triI] = e1SqLen_div_dbAreaSq[triI] =
                2.0 / std::sqrt(3.0) / floorArea;
            e0dote1_div_dbAreaSq[triI] = e0SqLen_div_dbAreaSq[triI] / 2.0;
        } else {
            surfaceArea += triArea[triI];
            triAreaSq[triI] = triArea[triI] * triArea[triI];
            e0SqLen[triI] = P2m1.squaredNorm();
            e1SqLen[triI] = P3m1.squaredNorm();
            e0dote1[triI] = P2m1.dot(P3m1);

            e0SqLen_div_dbAreaSq[triI] = e0SqLen[triI] / 2. / triAreaSq[triI];
            e1SqLen_div_dbAreaSq[triI] = e1SqLen[triI] / 2. / triAreaSq[triI];
            e0dote1_div_dbAreaSq[triI] = e0dote1[triI] / 2. / triAreaSq[triI];
        }
    }
    avgEdgeLen = igl::avg_edge_length(V_rest, F);
    virtualRadius = std::sqrt(surfaceArea / M_PI);

    // std::cout << "avg edge length = " << avgEdgeLen << std::endl;
    // std::cout << "triNormal validity: " << !!triNormal.rowwise().sum().sum()
    // << std::endl; std::cout << "surfaceArea = " << surfaceArea << std::endl;
    // std::cout << "avg triAreaSq = " << triAreaSq.sum() / triAreaSq.size() <<
    // std::endl; std::cout << "avg e0SqLen =" << e0SqLen.sum() / e0SqLen.size()
    // << std::endl; std::cout << "avg e1SqLen =" << e1SqLen.sum() /
    // e1SqLen.size() << std::endl; std::cout << "avg e0dote1 =" <<
    // e0dote1.sum() / e0dote1.size() << std::endl; std::cout << "avg
    // e0SqLen_div_dbAreaSq =" << e0SqLen_div_dbAreaSq.sum() /
    // e0SqLen_div_dbAreaSq.size() << std::endl; std::cout << "avg
    // e1SqLen_div_dbAreaSq =" << e1SqLen_div_dbAreaSq.sum() /
    // e1SqLen_div_dbAreaSq.size() << std::endl; std::cout << "avg
    // e0dote1_div_dbAreaSq =" << e0dote1_div_dbAreaSq.sum() /
    // e0dote1_div_dbAreaSq.size() << std::endl;

    // computeLaplacianMtr();

    bbox.block(0, 0, 1, 3) = V_rest.row(0);
    bbox.block(1, 0, 1, 3) = V_rest.row(0);
    for (int vI = 1; vI < V_rest.rows(); vI++) {
        const Eigen::RowVector3d &v = V_rest.row(vI);
        for (int dimI = 0; dimI < 3; dimI++) {
            if (v[dimI] < bbox(0, dimI))
                bbox(0, dimI) = v[dimI];
            if (v[dimI] > bbox(1, dimI))
                bbox(1, dimI) = v[dimI];
        }
    }
}

void TriMesh::computeFeatures(bool multiComp, bool resetFixedV) {
    if (resetFixedV) {
        fixedVert.clear();
        fixedVert.insert(0);
    }

    computeTriangleFeatures();

    edge2Tri.clear();
    vNeighbor.resize(0);
    vNeighbor.resize(V_rest.rows());
    for (int triI = 0; triI < F.rows(); triI++) {
        const Eigen::RowVector3i &triVInd = F.row(triI);
        for (int vI = 0; vI < 3; vI++) {
            int vI_post = (vI + 1) % 3;
            edge2Tri[std::pair<int, int>(triVInd[vI], triVInd[vI_post])] = triI;
            vNeighbor[triVInd[vI]].insert(triVInd[vI_post]);
            vNeighbor[triVInd[vI_post]].insert(triVInd[vI]);
        }
    }
    cohEIndex.clear();
    for (int cohI = 0; cohI < cohE.rows(); cohI++) {
        const Eigen::RowVector4i &cohEI = cohE.row(cohI);
        if (cohEI.minCoeff() >= 0) {
            cohEIndex[std::pair<int, int>(cohEI[0], cohEI[1])] = cohI;
            cohEIndex[std::pair<int, int>(cohEI[3], cohEI[2])] = -cohI - 1;
        }
    }

    // validSplit.resize(V_rest.rows());
    // for (int vI = 0; vI < V_rest.rows(); vI++) {
    //     validSplit[vI].clear();
    //     if (isBoundaryVert(vI))
    //         continue;
    //
    //     std::vector<int> nbVs(vNeighbor[vI].begin(), vNeighbor[vI].end());
    //     std::vector<Eigen::RowVector3d> projectedEdge(nbVs.size());
    //     for (int nbI = 0; nbI < nbVs.size(); nbI++) {
    //         const Eigen::RowVector3d edge =
    //             V_rest.row(nbVs[nbI]) - V_rest.row(vI);
    //         projectedEdge[nbI] =
    //             (edge - edge.dot(vertNormals[vI]) * vertNormals[vI])
    //                 .normalized();
    //     }
    //     for (int nbI = 0; nbI + 1 < nbVs.size(); nbI++) {
    //         for (int nbJ = nbI + 1; nbJ < nbVs.size(); nbJ++) {
    //             if (projectedEdge[nbI].dot(projectedEdge[nbJ]) <= 0.0) {
    //                 validSplit[vI].insert(
    //                     std::pair<int, int>(nbVs[nbI], nbVs[nbJ]));
    //                 validSplit[vI].insert(
    //                     std::pair<int, int>(nbVs[nbJ], nbVs[nbI]));
    //             }
    //         }
    //     }
    // }

    // init fracture tail record
    fracTail.clear();
    for (int cohI = 0; cohI < cohE.rows(); cohI++) {
        if (cohE(cohI, 0) == cohE(cohI, 2))
            fracTail.insert(cohE(cohI, 0));
        else if (cohE(cohI, 1) == cohE(cohI, 3))
            fracTail.insert(cohE(cohI, 1));
    }
    // tails of initial seams doesn't count as fracture tails for propagation
    for (int initSeamI = 0; initSeamI < initSeams.rows(); initSeamI++) {
        if (initSeams(initSeamI, 0) == initSeams(initSeamI, 2))
            fracTail.erase(initSeams(initSeamI, 0));
        else if (initSeams(initSeamI, 1) == initSeams(initSeamI, 3))
            fracTail.erase(initSeams(initSeamI, 1));
    }
}

void TriMesh::updateFeatures(void) {
    const int nCE = static_cast<int>(boundaryEdge.size());
    boundaryEdge.conservativeResize(cohE.rows());
    edgeLen.conservativeResize(cohE.rows());
    for (int cohI = nCE; cohI < cohE.rows(); cohI++) {
        if (cohE.row(cohI).minCoeff() >= 0)
            boundaryEdge[cohI] = 0;
        else
            boundaryEdge[cohI] = 1;
        edgeLen[cohI] =
            (V_rest.row(cohE(cohI, 0)) - V_rest.row(cohE(cohI, 1))).norm();
    }

    // computeLaplacianMtr();
}

void TriMesh::resetFixedVert(const std::set<int> &p_fixedVert) {
    for (const auto &vI : p_fixedVert)
        assert(vI < V.rows());

    fixedVert = p_fixedVert;
    // computeLaplacianMtr();
}

void TriMesh::buildCohEfromRecord(const Eigen::MatrixXi &cohERecord) {
    assert(cohERecord.cols() == 4);

    cohE.resize(cohERecord.rows(), 4);
    for (int cohEI = 0; cohEI < cohERecord.rows(); cohEI++) {
        int triI1 = cohERecord(cohEI, 0);
        int vI11 = F(triI1, cohERecord(cohEI, 1));
        int vI12 = F(triI1, (cohERecord(cohEI, 1) + 1) % 3);
        int triI2 = cohERecord(cohEI, 2);
        int vI21 = F(triI2, cohERecord(cohEI, 3));
        int vI22 = F(triI2, (cohERecord(cohEI, 3) + 1) % 3);
        cohE.row(cohEI) << vI11, vI12, vI22, vI21;
    }

    computeFeatures();
}

// a cut whose path runs boundary to boundary splits the chart in two. in
// pinned mode a piece without a pinned vert floats away from the held
// border, so such split candidates are vetoed in querySplit
bool TriMesh::cutLeavesPinlessPiece(const std::vector<int> &path) const {
    if (path.size() < 2 || !isBoundaryVert(path.front()) ||
        !isBoundaryVert(path.back()))
        return false;

    std::set<std::pair<int, int>> cut;
    for (int i = 0; i + 1 < static_cast<int>(path.size()); ++i) {
        cut.insert(std::pair<int, int>(path[i], path[i + 1]));
        cut.insert(std::pair<int, int>(path[i + 1], path[i]));
    }

    // flood faces over shared edges without crossing the cut
    auto seed = edge2Tri.find(*cut.begin());
    if (seed == edge2Tri.end())
        return false;
    std::vector<bool> visited(F.rows(), false);
    std::vector<int> stack = {seed->second};
    visited[seed->second] = true;
    while (!stack.empty()) {
        int triI = stack.back();
        stack.pop_back();
        for (int i = 0; i < 3; ++i) {
            int a = F(triI, i), b = F(triI, (i + 1) % 3);
            if (cut.count(std::pair<int, int>(a, b)))
                continue;
            const auto nb = edge2Tri.find(std::pair<int, int>(b, a));
            if (nb != edge2Tri.end() && !visited[nb->second]) {
                visited[nb->second] = true;
                stack.push_back(nb->second);
            }
        }
    }

    bool unvisited = false;
    bool pinned[2] = {false, false};
    for (int triI = 0; triI < F.rows(); ++triI) {
        if (!visited[triI])
            unvisited = true;
        for (int i = 0; i < 3; ++i) {
            if (fixedVert.count(F(triI, i))) {
                pinned[visited[triI] ? 0 : 1] = true;
                break;
            }
        }
    }
    if (!unvisited)
        return false;
    return !pinned[0] || !pinned[1];
}

void TriMesh::querySplit(double lambda_t, bool propagate, bool splitInterior,
                         double &EwDec_max, std::vector<int> &path_max,
                         Eigen::MatrixXd &newVertPos_max,
                         std::pair<double, double> &energyChanges_max) const {
    const double filterExp_b = 0.8, filterMult_b = 1.0; // TODO: better use
                                                        // ratio

    std::vector<int> bestCandVerts;
    if (!propagate) {
        SymDirichletEnergy SD;
        Eigen::VectorXd divGradPerVert;
        SD.computeDivGradPerVert(*this, divGradPerVert);

        std::map<double, int> sortedCandVerts_b, sortedCandVerts_in;
        if (splitInterior) {
            for (int vI = 0; vI < V_rest.rows(); vI++) {
                if (fixedVert.count(vI)) {
                    // pinned verts must not be duplicated
                    continue;
                }
                if (vNeighbor[vI].size() <= 2) {
                    // this vertex is impossible to be split further
                    continue;
                }

                if (!isBoundaryVert(vI)) {
                    bool connectToBound = false;
                    for (const auto &nbVI : vNeighbor[vI]) {
                        if (isBoundaryVert(nbVI)) {
                            connectToBound = true;
                            break;
                        }
                    }
                    if (!connectToBound) {
                        // don't split vertices connected to boundary here
                        sortedCandVerts_in[-divGradPerVert[vI] /
                                           vertWeight[vI]] = vI;
                    }
                }
            }
            inSplitTotalAmt = sortedCandVerts_in.size();
        } else {
            for (int vI = 0; vI < V_rest.rows(); vI++) {
                if (fixedVert.count(vI)) {
                    // pinned verts must not be duplicated
                    continue;
                }
                if (vNeighbor[vI].size() <= 2) {
                    // this vertex is impossible to be split further
                    continue;
                }
                if (isBoundaryVert(vI))
                    sortedCandVerts_b[-divGradPerVert[vI] / vertWeight[vI]] =
                        vI;
            }
        }

        // the lists outlive this call, updateLambda_stationaryV queues an
        // op from them rounds later, so stale positions invert triangles
        if (!splitInterior) {
            if (sortedCandVerts_b.empty()) {
                paths_bSplit.clear();
                newVertPoses_bSplit.clear();
                energyChanges_bSplit.clear();
                EwDec_max = 0.0;
                return;
            }
            int bestCandAmt_b = static_cast<int>(
                std::pow(sortedCandVerts_b.size(), filterExp_b) * filterMult_b);
            if (bestCandAmt_b < 2)
                bestCandAmt_b = 2;
            bestCandVerts.reserve(bestCandAmt_b);
            for (const auto &candI : sortedCandVerts_b) {
                bestCandVerts.emplace_back(candI.second);
                if (bestCandVerts.size() >= bestCandAmt_b)
                    break;
            }
        } else {
            if (sortedCandVerts_in.empty()) {
                paths_iSplit.clear();
                newVertPoses_iSplit.clear();
                energyChanges_iSplit.clear();
                EwDec_max = 0.0;
                return;
            }
            int bestCandAmt_in = static_cast<int>(
                std::pow(sortedCandVerts_in.size(), filterExp_in));
            if (bestCandAmt_in < 2) {
                bestCandAmt_in = 2;
            }
            bestCandVerts.reserve(bestCandVerts.size() + bestCandAmt_in);
            for (const auto &candI : sortedCandVerts_in) {
                bestCandVerts.emplace_back(candI.second);
                if (bestCandVerts.size() >= bestCandAmt_in)
                    break;
            }
        }
    } else {
        // see whether fracture could be propagated from each fracture tail
// #define PROPAGATE_MULTIPLE_TAIL 1
#ifdef PROPAGATE_MULTIPLE_TAIL
        if (fracTail.empty()) {
#else
        if (curFracTail < 0) {
#endif
            if (curInteriorFracTails.first < 0) {
                EwDec_max = -DBL_MAX;
                path_max.resize(0);
                newVertPos_max.resize(0, 2);
                return;
            } else {
                assert(curInteriorFracTails.second >= 0);
                splitInterior = false;
                bestCandVerts.emplace_back(curInteriorFracTails.first);
                bestCandVerts.emplace_back(curInteriorFracTails.second);
            }
        } else {
            splitInterior = false;
#ifdef PROPAGATE_MULTIPLE_TAIL
            bestCandVerts.insert(bestCandVerts.end(), fracTail.begin(),
                                 fracTail.end());
#else
            bestCandVerts.emplace_back(curFracTail);
#endif
        }
    }

    assert(!bestCandVerts.empty());

    // a split path through a pinned vert would duplicate it, invalidate
    // those, and in pinned mode also cuts that would strand a pinless piece
    auto invalidateFixed = [this](const std::vector<int> &path, double &EwDec,
                                  std::pair<double, double> &energyChange) {
        for (const auto &vI : path) {
            if (fixedVert.count(vI)) {
                EwDec = -DBL_MAX;
                energyChange.first = DBL_MAX;
                energyChange.second = DBL_MAX;
                return;
            }
        }
        if (pinnedMode && cutLeavesPinlessPiece(path)) {
            EwDec = -DBL_MAX;
            energyChange.first = DBL_MAX;
            energyChange.second = DBL_MAX;
        }
    };

    // evaluate local energy decrease
    // DISABLE std::cout << "evaluate vertex splits, " << bestCandVerts.size()
    // << " candidate verts" << std::endl; run in parallel:
    static std::vector<double> EwDecs;
    EwDecs.resize(bestCandVerts.size());
    static std::vector<std::vector<int>> paths_p;
    static std::vector<Eigen::MatrixXd> newVertPoses_p;
    static std::vector<std::pair<double, double>> energyChanges_p;
    int operationType = -1;
    // query boundary splits
    if (!splitInterior) {
        if (propagate) {
            paths_p.resize(0);
            paths_p.resize(bestCandVerts.size());
            newVertPoses_p.resize(bestCandVerts.size());
            energyChanges_p.resize(bestCandVerts.size());
            tbb::parallel_for(0, (int)bestCandVerts.size(), 1, [&](int candI) {
                EwDecs[candI] = computeLocalLDec(
                    bestCandVerts[candI], lambda_t, paths_p[candI],
                    newVertPoses_p[candI], energyChanges_p[candI]);
                invalidateFixed(paths_p[candI], EwDecs[candI],
                                energyChanges_p[candI]);
            });
        } else {
            operationType = 0;
            paths_bSplit.resize(0);
            paths_bSplit.resize(bestCandVerts.size());
            newVertPoses_bSplit.resize(bestCandVerts.size());
            energyChanges_bSplit.resize(bestCandVerts.size());
            tbb::parallel_for(0, (int)bestCandVerts.size(), 1, [&](int candI) {
                EwDecs[candI] = computeLocalLDec(
                    bestCandVerts[candI], lambda_t, paths_bSplit[candI],
                    newVertPoses_bSplit[candI], energyChanges_bSplit[candI]);
                invalidateFixed(paths_bSplit[candI], EwDecs[candI],
                                energyChanges_bSplit[candI]);
            });
        }
    } else {
        assert(!propagate);
        operationType = 1;
        // query interior splits
        paths_iSplit.resize(0);
        paths_iSplit.resize(bestCandVerts.size());
        newVertPoses_iSplit.resize(bestCandVerts.size());
        energyChanges_iSplit.resize(bestCandVerts.size());
        tbb::parallel_for(0, (int)bestCandVerts.size(), 1, [&](int candI) {
            EwDecs[candI] = computeLocalLDec(
                bestCandVerts[candI], lambda_t, paths_iSplit[candI],
                newVertPoses_iSplit[candI], energyChanges_iSplit[candI]);
            invalidateFixed(paths_iSplit[candI], EwDecs[candI],
                            energyChanges_iSplit[candI]);
            if (EwDecs[candI] != -DBL_MAX)
                EwDecs[candI] *= 0.5;
        });
    }

    int candI_max = 0;
    for (int candI = 1; candI < bestCandVerts.size(); candI++) {
        if (EwDecs[candI] > EwDecs[candI_max])
            candI_max = candI;
    }

    EwDec_max = EwDecs[candI_max];
    switch (operationType) {
    case -1:
        path_max = paths_p[candI_max];
        newVertPos_max = newVertPoses_p[candI_max];
        energyChanges_max = energyChanges_p[candI_max];
        break;

    case 0:
        path_max = paths_bSplit[candI_max];
        newVertPos_max = newVertPoses_bSplit[candI_max];
        energyChanges_max = energyChanges_bSplit[candI_max];
        break;

    case 1:
        path_max = paths_iSplit[candI_max];
        newVertPos_max = newVertPoses_iSplit[candI_max];
        energyChanges_max = energyChanges_iSplit[candI_max];
        break;

    default:
        assert(0);
        break;
    }
}

bool TriMesh::splitEdge(double lambda_t, double thres, bool propagate,
                        bool splitInterior) {
    double EwDec_max;
    std::vector<int> path_max;
    Eigen::MatrixXd newVertPos_max;
    std::pair<double, double> energyChanges_max;
    querySplit(lambda_t, propagate, splitInterior, EwDec_max, path_max,
               newVertPos_max, energyChanges_max);

    // DISABLE std::cout << "E_dec threshold = " << thres << std::endl;
    if (EwDec_max > thres) {
        if (!splitInterior) {
            // boundary split
            // DISABLE std::cout << "boundary split E_dec = " << EwDec_max <<
            // std::endl;
            splitEdgeOnBoundary(std::pair<int, int>(path_max[0], path_max[1]),
                                newVertPos_max);
            // TODO: process fractail here!
            updateFeatures();
        } else {
            // interior split
            assert(!propagate);
            // DISABLE std::cout << "interior split E_dec = " << EwDec_max <<
            // std::endl;
            cutPath(path_max, true, 1, newVertPos_max);
            // DISABLE logFile << "interior edge split" << std::endl;
            fracTail.insert(path_max[0]);
            fracTail.insert(path_max[2]);
            curInteriorFracTails.first = path_max[0];
            curInteriorFracTails.second = path_max[2];
            curFracTail = -1;
        }
        return true;
    } else {
        // DISABLE std::cout << "max E_dec = " << EwDec_max << " < thres " <<
        // thres << std::endl;
        return false;
    }
}

void TriMesh::queryMerge(double lambda, bool propagate, double &localEwDec_max,
                         std::vector<int> &path_max,
                         Eigen::MatrixXd &newVertPos_max,
                         std::pair<double, double> &energyChanges_max) {
    // TODO: local index updates in mergeBoundaryEdge()
    // TODO: parallelize the query

    // DISABLE std::cout << "evaluate edge merge, " << cohE.rows() << " cohesive
    // edge pairs." << std::endl;
    localEwDec_max = -DBL_MAX;
    if (!propagate) {
        paths_merge.resize(0);
        newVertPoses_merge.resize(0);
        energyChanges_merge.resize(0);
    }
    for (int cohI = 0; cohI < cohE.rows(); cohI++) {
        int forkVI = 0;
        if (cohE(cohI, 0) == cohE(cohI, 2))
            forkVI = 0;
        else if (cohE(cohI, 1) == cohE(cohI, 3))
            forkVI = 1;
        else
            // only consider "zipper bottom" edge pairs for now
            continue;

        if (propagate) {
            if (cohE(cohI, forkVI) != curFracTail)
                continue;
        }

        // merging moves both verts, pinned ones must stay put
        if (fixedVert.count(cohE(cohI, 1 - forkVI)) ||
            fixedVert.count(cohE(cohI, 3 - forkVI)))
            continue;

        // find incident triangles for inversion check and local energy decrease
        // evaluation
        std::vector<int> triangles;
        int firstVertIncTriAmt = 0;
        for (int mergeVI = 1; mergeVI <= 3; mergeVI += 2) {
            for (const auto &nbVI : vNeighbor[cohE(cohI, mergeVI - forkVI)]) {
                auto finder = edge2Tri.find(
                    std::pair<int, int>(cohE(cohI, mergeVI - forkVI), nbVI));
                if (finder != edge2Tri.end())
                    triangles.emplace_back(finder->second);
            }
            if (mergeVI == 1) {
                firstVertIncTriAmt = static_cast<int>(triangles.size());
                assert(firstVertIncTriAmt >= 1);
            }
        }

        Eigen::RowVector2d mergedPos =
            (V.row(cohE(cohI, 1 - forkVI)) + V.row(cohE(cohI, 3 - forkVI))) /
            2.0;
        const Eigen::RowVector2d backup0 = V.row(cohE(cohI, 1 - forkVI));
        const Eigen::RowVector2d backup1 = V.row(cohE(cohI, 3 - forkVI));
        V.row(cohE(cohI, 1 - forkVI)) = mergedPos;
        V.row(cohE(cohI, 3 - forkVI)) = mergedPos;
        if (checkInversion(true, triangles)) {
            V.row(cohE(cohI, 1 - forkVI)) = backup0;
            V.row(cohE(cohI, 3 - forkVI)) = backup1;
        } else {
            // project mergedPos to feasible set via Relaxation method for
            // linear inequalities

            // find inequality constraints by opposite edge in incident
            // triangles
            Eigen::MatrixXd inequalityConsMtr;
            Eigen::VectorXd inequalityConsVec;
            for (int triII = 0; triII < triangles.size(); triII++) {
                int triI = triangles[triII];
                int vI_toMerge =
                    ((triII < firstVertIncTriAmt) ? cohE(cohI, 1 - forkVI)
                                                  : cohE(cohI, 3 - forkVI));
                for (int i = 0; i < 3; i++) {
                    if (F(triI, i) == vI_toMerge) {
                        const Eigen::RowVector2d &v1 =
                            V.row(F(triI, (i + 1) % 3));
                        const Eigen::RowVector2d &v2 =
                            V.row(F(triI, (i + 2) % 3));
                        Eigen::RowVector2d coef(v2[1] - v1[1], v1[0] - v2[0]);
                        inequalityConsMtr.conservativeResize(
                            inequalityConsMtr.rows() + 1, 2);
                        inequalityConsMtr.row(inequalityConsMtr.rows() - 1) =
                            coef / coef.norm();
                        inequalityConsVec.conservativeResize(
                            inequalityConsVec.size() + 1);
                        inequalityConsVec[inequalityConsVec.size() - 1] =
                            (v1[0] * v2[1] - v1[1] * v2[0]) / coef.norm();
                        break;
                    }
                }
            }
            assert(inequalityConsMtr.rows() == triangles.size());
            assert(inequalityConsVec.size() == triangles.size());

            // Relaxation method for linear inequalities
            int maxIter = 70;
            const double eps_IC = 1.0e-6 * avgEdgeLen;
            for (int iterI = 0; iterI < maxIter; iterI++) {
                double maxRes = -DBL_MAX;
                for (int consI = 0; consI < inequalityConsMtr.rows(); consI++) {
                    double res =
                        inequalityConsMtr.row(consI) * mergedPos.transpose() -
                        inequalityConsVec[consI];
                    if (res > -eps_IC) {
                        // project
                        mergedPos -= (res + eps_IC) *
                                     inequalityConsMtr.row(consI).transpose();
                    }
                    if (res > maxRes)
                        maxRes = res;
                }

                if (maxRes < 0.0) {
                    // converged (non-inversion satisfied)
                    // NOTE: although this maxRes is 1 iteration behind, it is
                    // OK for a convergence check
                    break;
                }
            }

            V.row(cohE(cohI, 1 - forkVI)) = mergedPos;
            V.row(cohE(cohI, 3 - forkVI)) = mergedPos;
            bool noInversion = checkInversion(true, triangles);
            V.row(cohE(cohI, 1 - forkVI)) = backup0;
            V.row(cohE(cohI, 3 - forkVI)) = backup1;
            if (!noInversion) {
                // because propagation is not at E_SD stationary, so it's
                // possible to have no feasible region
                continue;
            }
        }

        // optimize local distortion
        std::vector<int> path;
        if (forkVI) {
            path.emplace_back(cohE(cohI, 0));
            path.emplace_back(cohE(cohI, 1));
            path.emplace_back(cohE(cohI, 2));
        } else {
            path.emplace_back(cohE(cohI, 3));
            path.emplace_back(cohE(cohI, 2));
            path.emplace_back(cohE(cohI, 1));
        }
        Eigen::MatrixXd newVertPos;
        std::pair<double, double> energyChanges;
        double localEwDec = computeLocalLDec(
            0, lambda, path, newVertPos, energyChanges, triangles, mergedPos);
        if (!propagate) {
            paths_merge.emplace_back(path);
            newVertPoses_merge.emplace_back(newVertPos);
            energyChanges_merge.emplace_back(energyChanges);
        }

        if (localEwDec > localEwDec_max) {
            localEwDec_max = localEwDec;
            newVertPos_max = newVertPos;
            path_max = path;
            energyChanges_max = energyChanges;
        }
    }
}

bool TriMesh::mergeEdge(double lambda, double EDecThres, bool propagate) {
    double localEwDec_max;
    std::vector<int> path_max;
    Eigen::MatrixXd newVertPos_max;
    std::pair<double, double> energyChanges_max;
    queryMerge(lambda, propagate, localEwDec_max, path_max, newVertPos_max,
               energyChanges_max);

    // DISABLE std::cout << "E_dec threshold = " << EDecThres << std::endl;
    if (localEwDec_max > EDecThres) {
        // DISABLE std::cout << "merge edge E_dec = " << localEwDec_max <<
        // std::endl;
        mergeBoundaryEdges(std::pair<int, int>(path_max[0], path_max[1]),
                           std::pair<int, int>(path_max[1], path_max[2]),
                           newVertPos_max.row(0));
        // DISABLE logFile << "edge merged" << std::endl;

        computeFeatures(); // TODO: only update locally
        return true;
    } else {
        // DISABLE std::cout << "max E_dec = " << localEwDec_max << " < thres "
        // << EDecThres << std::endl;
        return false;
    }
}

// after a weld swap-deletes vertex removed into kept and renames the last
// vertex backI into the freed slot, map an index to its new name. every
// vertex-indexed structure must go through this after a stitch weld
static int renameAfterWeld(int vI, int removed, int kept, int backI) {
    if (vI == removed)
        return kept;
    if (vI == backI)
        return removed;
    return vI;
}

static void renameSetAfterWeld(std::set<int> &verts, int removed, int kept,
                               int backI) {
    std::set<int> renamed;
    for (const int vI : verts)
        renamed.insert(renameAfterWeld(vI, removed, kept, backI));
    verts.swap(renamed);
}

bool TriMesh::stitchIsland(void) {
    // island id per vertex
    std::vector<int> vertComp(V.rows(), -1);
    int compAmt = 0;
    for (int seedVI = 0; seedVI < V.rows(); seedVI++) {
        if (vertComp[seedVI] >= 0)
            continue;
        std::vector<int> stack = {seedVI};
        vertComp[seedVI] = compAmt;
        while (!stack.empty()) {
            int vI = stack.back();
            stack.pop_back();
            for (const auto &nbVI : vNeighbor[vI]) {
                if (vertComp[nbVI] < 0) {
                    vertComp[nbVI] = compAmt;
                    stack.emplace_back(nbVI);
                }
            }
        }
        compAmt++;
    }
    if (compAmt < 2)
        return false;

    std::vector<int> compSize(compAmt, 0);
    std::vector<bool> compFixed(compAmt, false);
    for (int vI = 0; vI < V.rows(); vI++)
        compSize[vertComp[vI]]++;
    for (const auto &vI : fixedVert)
        compFixed[vertComp[vI]] = true;

    // rank island pairs by total shared rest length, the strongest link is
    // the most seam removed per stitch
    auto compKey = [&](int cohI) -> std::pair<int, int> {
        return std::minmax(vertComp[cohE(cohI, 0)], vertComp[cohE(cohI, 2)]);
    };
    std::map<std::pair<int, int>, double> sharedLen;
    std::vector<int> candidates;
    for (int cohI = 0; cohI < cohE.rows(); cohI++) {
        if (cohE.row(cohI).minCoeff() < 0)
            continue;
        if (vertComp[cohE(cohI, 0)] == vertComp[cohE(cohI, 2)])
            continue;
        if (fixedVert.count(cohE(cohI, 0)) || fixedVert.count(cohE(cohI, 1)) ||
            fixedVert.count(cohE(cohI, 2)) || fixedVert.count(cohE(cohI, 3)))
            continue;
        candidates.emplace_back(cohI);
        sharedLen[compKey(cohI)] += edgeLen[cohI];
    }
    std::sort(candidates.begin(), candidates.end(), [&](int cohA, int cohB) {
        double lenA = sharedLen[compKey(cohA)], lenB = sharedLen[compKey(cohB)];
        if (lenA != lenB)
            return lenA > lenB;
        return edgeLen[cohA] > edgeLen[cohB];
    });

    // the raw boundary only depends on F, shared by every candidate's test
    std::vector<std::vector<int>> bnd_raw;
    igl::boundary_loop(F, bnd_raw);

    for (const auto &cohI : candidates) {
        int a0 = cohE(cohI, 0), a1 = cohE(cohI, 1);
        int b0 = cohE(cohI, 2), b1 = cohE(cohI, 3);

        // move the smaller island, but never one holding a fixed vertex
        int movingComp = vertComp[b0];
        int from0 = b0, from1 = b1, to0 = a0, to1 = a1;
        if (compFixed[movingComp] ||
            (!compFixed[vertComp[a0]] &&
             compSize[vertComp[a0]] < compSize[movingComp])) {
            movingComp = vertComp[a0];
            from0 = a0, from1 = a1, to0 = b0, to1 = b1;
        }
        if (compFixed[movingComp])
            continue;

        // rigid transform mapping from0 exactly onto to0 and the shared edge
        // directions together. the two uv edge lengths can differ, the slack
        // stays at the far endpoint pair for the merge machinery to close
        const Eigen::RowVector2d eF = V.row(from1) - V.row(from0);
        const Eigen::RowVector2d eT = V.row(to1) - V.row(to0);
        const double nF = eF.norm(), nT = eT.norm();
        if ((nF == 0.0) || (nT == 0.0))
            continue;
        const double cosR = eF.dot(eT) / (nF * nT);
        const double sinR = (eF[0] * eT[1] - eF[1] * eT[0]) / (nF * nT);
        Eigen::Matrix2d rotT; // transposed, for row vectors
        rotT << cosR, sinR, -sinR, cosR;
        const Eigen::RowVector2d pFrom = V.row(from0);
        const Eigen::RowVector2d pTo = V.row(to0);

        // test the placement on a scratch copy with every endpoint pair the
        // zip will weld already identified, spreading from the stitched edge
        // through shared endpoints. the whole shared run then reads interior
        // and only real crossings remain, a run continuation landing exactly
        // collinear would otherwise read as an overlap
        Eigen::MatrixXd V_test = V;
        for (int vI = 0; vI < V.rows(); vI++) {
            if (vertComp[vI] == movingComp)
                V_test.row(vI) = (V.row(vI) - pFrom) * rotT + pTo;
        }
        std::map<int, int> parent;
        auto findRep = [&parent](int vI) {
            while (true) {
                auto it = parent.find(vI);
                if ((it == parent.end()) || (it->second == vI))
                    return vI;
                vI = it->second;
            }
        };
        auto unite = [&](int vA, int vB) {
            int repA = findRep(vA), repB = findRep(vB);
            if (repA != repB)
                parent[std::max(repA, repB)] = std::min(repA, repB);
        };
        unite(a0, b0);
        unite(a1, b1);
        const std::pair<int, int> stitchKey = compKey(cohI);
        bool grew = true;
        while (grew) {
            grew = false;
            for (const auto &cohI2 : candidates) {
                if (compKey(cohI2) != stitchKey)
                    continue;
                bool weld0 = findRep(cohE(cohI2, 0)) == findRep(cohE(cohI2, 2));
                bool weld1 = findRep(cohE(cohI2, 1)) == findRep(cohE(cohI2, 3));
                if (weld0 != weld1) {
                    unite(cohE(cohI2, 0), cohE(cohI2, 2));
                    unite(cohE(cohI2, 1), cohE(cohI2, 3));
                    grew = true;
                }
            }
        }
        Eigen::MatrixXi F_test = F;
        for (int triI = 0; triI < F_test.rows(); triI++) {
            for (int vI = 0; vI < 3; vI++)
                F_test(triI, vI) = findRep(F_test(triI, vI));
        }
        std::vector<std::vector<int>> bnd_all;
        igl::boundary_loop(F_test, bnd_all);
        if (IglUtils::checkUVBoundaryOverlap(V_test, bnd_all, NULL))
            continue;

        // the pre-welded test reads every open zipper as interior, but a
        // zipper blocked mid-seam stays open, so also test the raw boundary
        // for real crossings. the flush run reads as touching, not crossing,
        // and a candidate whose slack would drag through anything is refused
        // here rather than jamming the zip later
        if (IglUtils::checkUVBoundaryOverlap(V_test, bnd_raw, NULL, true))
            continue;

        // crossing-free placement can still bury one island inside another,
        // test a representative point of each island against the material of
        // every other (even-odd over all its loops, so holes stay legal)
        const int mergedComp = vertComp[a0];
        auto compOf = [&](int vI) {
            return (vertComp[vI] == vertComp[b0]) ? mergedComp : vertComp[vI];
        };
        auto insideComp = [&](const Eigen::RowVector2d &p, int compI) {
            bool inside = false;
            for (const auto &loop : bnd_all) {
                if (compOf(loop[0]) != compI)
                    continue;
                for (int i = 0; i < loop.size(); i++) {
                    const Eigen::RowVector2d &s = V_test.row(loop[i]);
                    const Eigen::RowVector2d &e =
                        V_test.row(loop[(i + 1) % loop.size()]);
                    if (((s[1] > p[1]) != (e[1] > p[1])) &&
                        (p[0] < s[0] + (e[0] - s[0]) * (p[1] - s[1]) /
                                           (e[1] - s[1])))
                        inside = !inside;
                }
            }
            return inside;
        };
        std::map<int, int> compRepr;
        for (const auto &loop : bnd_all)
            compRepr.emplace(compOf(loop[0]), loop[0]);
        bool buried = false;
        for (const auto &reprI : compRepr) {
            for (const auto &reprJ : compRepr) {
                if (reprI.first == reprJ.first)
                    continue;
                if (insideComp(V_test.row(reprJ.second), reprI.first)) {
                    buried = true;
                    break;
                }
            }
            if (buried)
                break;
        }
        if (buried)
            continue;

        // apply the placement
        for (int vI = 0; vI < V.rows(); vI++) {
            if (vertComp[vI] == movingComp)
                V.row(vI) = (V.row(vI) - pFrom) * rotT + pTo;
        }

        // weld b0 into a0, turning this pair into a zipper bottom. same
        // swap-delete scheme as mergeBoundaryEdges
        int vBackI = static_cast<int>(V.rows()) - 1;
        const int keepVI = ((a0 == vBackI) && (b0 < vBackI)) ? b0 : a0;
        if (b0 < vBackI) {
            V_rest.row(b0) = V_rest.row(vBackI);
            vertWeight[b0] = vertWeight[vBackI];
            V.row(b0) = V.row(vBackI);
        }
        V_rest.conservativeResize(vBackI, 3);
        vertWeight.conservativeResize(vBackI);
        V.conservativeResize(vBackI, 2);

        for (int triI = 0; triI < F.rows(); triI++) {
            for (int vI = 0; vI < 3; vI++)
                F(triI, vI) = renameAfterWeld(F(triI, vI), b0, keepVI, vBackI);
        }
        for (int cohI2 = 0; cohI2 < cohE.rows(); cohI2++) {
            for (int pI = 0; pI < 4; pI++)
                cohE(cohI2, pI) =
                    renameAfterWeld(cohE(cohI2, pI), b0, keepVI, vBackI);
        }
        renameSetAfterWeld(stitchFronts, b0, keepVI, vBackI);
        stitchFronts.insert(keepVI);

        computeFeatures(); // rebuilds fracTail from the new zipper bottom
        curFracTail = keepVI;
        zipStitchedSeam();
        return true;
    }
    return false;
}

bool TriMesh::zipStitchedSeam(void) {
    bool changed = false;
    std::set<int> active = stitchFronts;
    while (!active.empty()) {
        const int fork = *active.begin();
        active.erase(active.begin());

        // a zipper bottom hinged at this front
        int zipI = -1, forkVI = -1;
        for (int cohI = 0; cohI < cohE.rows(); cohI++) {
            if (cohE.row(cohI).minCoeff() < 0)
                continue;
            if ((cohE(cohI, 0) == cohE(cohI, 2)) && (cohE(cohI, 0) == fork) &&
                (cohE(cohI, 1) != cohE(cohI, 3))) {
                zipI = cohI;
                forkVI = 0;
                break;
            }
            if ((cohE(cohI, 1) == cohE(cohI, 3)) && (cohE(cohI, 1) == fork) &&
                (cohE(cohI, 0) != cohE(cohI, 2))) {
                zipI = cohI;
                forkVI = 1;
                break;
            }
        }
        if (zipI < 0) {
            // zipped past or blocked for good, either way not a front anymore
            stitchFronts.erase(fork);
            continue;
        }

        const int u = cohE(zipI, 1 - forkVI), w = cohE(zipI, 3 - forkVI);
        if (fixedVert.count(u) || fixedVert.count(w))
            continue;

        std::vector<int> triangles;
        for (const auto &nbVI : vNeighbor[u]) {
            auto finder = edge2Tri.find(std::pair<int, int>(u, nbVI));
            if (finder != edge2Tri.end())
                triangles.emplace_back(finder->second);
        }
        for (const auto &nbVI : vNeighbor[w]) {
            auto finder = edge2Tri.find(std::pair<int, int>(w, nbVI));
            if (finder != edge2Tri.end())
                triangles.emplace_back(finder->second);
        }
        const Eigen::RowVector2d mergedPos = (V.row(u) + V.row(w)) / 2.0;
        const Eigen::RowVector2d backupU = V.row(u), backupW = V.row(w);
        V.row(u) = mergedPos;
        V.row(w) = mergedPos;
        const bool feasible = checkInversion(true, triangles);
        V.row(u) = backupU;
        V.row(w) = backupW;
        if (!feasible)
            // the front stays, relaxation may make room for a later attempt
            continue;

        // inversion is only checked locally, but pulling u and w together can
        // drag the outline across a thin section far away. transversal only,
        // so the still-coincident runs of open zippers don't read as
        // crossings while a real drag-through still blocks the weld
        Eigen::MatrixXi F_test = F;
        for (int triI = 0; triI < F_test.rows(); triI++) {
            for (int vI = 0; vI < 3; vI++) {
                if (F_test(triI, vI) == w)
                    F_test(triI, vI) = u;
            }
        }
        Eigen::MatrixXd V_test_zip = V;
        V_test_zip.row(u) = mergedPos;
        std::vector<std::vector<int>> bnd_zip;
        igl::boundary_loop(F_test, bnd_zip);
        if (IglUtils::checkUVBoundaryOverlap(V_test_zip, bnd_zip, NULL, true))
            continue;

        std::vector<int> path;
        if (forkVI) {
            path = {cohE(zipI, 0), cohE(zipI, 1), cohE(zipI, 2)};
        } else {
            path = {cohE(zipI, 3), cohE(zipI, 2), cohE(zipI, 1)};
        }
        const int vBackI = static_cast<int>(V.rows()) - 1;
        // path[2] is welded away and the last vertex takes its index, the
        // merged vertex ends up at path[0] unless that rename moved it
        const int mergedVI =
            ((path[0] == vBackI) && (path[2] < vBackI)) ? path[2] : path[0];
        mergeBoundaryEdges(std::pair<int, int>(path[0], path[1]),
                           std::pair<int, int>(path[1], path[2]), mergedPos);
        computeFeatures();
        changed = true;

        renameSetAfterWeld(stitchFronts, path[2], mergedVI, vBackI);
        renameSetAfterWeld(active, path[2], mergedVI, vBackI);
        stitchFronts.insert(mergedVI);
        // the same front can hinge another zipper, a stitch lands mid-run
        active.insert(mergedVI);
        active.insert(renameAfterWeld(fork, path[2], mergedVI, vBackI));
    }
    return changed;
}

bool TriMesh::splitOrMerge(double lambda_t, double EDecThres, bool propagate,
                           bool splitInterior, bool &isMerge) {
    assert((!propagate) &&
           "propagation is supported separately for split and merge!");

    double EwDec_max;
    std::vector<int> path_max;
    Eigen::MatrixXd newVertPos_max;
    isMerge = false;
    std::pair<double, double> energyChanes_split, energyChanes_merge;
    if (splitInterior) {
        querySplit(lambda_t, propagate, splitInterior, EwDec_max, path_max,
                   newVertPos_max, energyChanes_split);
    } else {
        double EwDec_max_split, EwDec_max_merge;
        std::vector<int> path_max_split, path_max_merge;
        Eigen::MatrixXd newVertPos_max_split, newVertPos_max_merge;
        querySplit(lambda_t, propagate, splitInterior, EwDec_max_split,
                   path_max_split, newVertPos_max_split, energyChanes_split);
        queryMerge(lambda_t, propagate, EwDec_max_merge, path_max_merge,
                   newVertPos_max_merge, energyChanes_merge);

        if (EwDec_max_merge > EwDec_max_split) {
            isMerge = true;
            EwDec_max = EwDec_max_merge;
            path_max = path_max_merge;
            newVertPos_max = newVertPos_max_merge;
        } else {
            EwDec_max = EwDec_max_split;
            path_max = path_max_split;
            newVertPos_max = newVertPos_max_split;
        }
    }

    if (EwDec_max > EDecThres) {
        if (isMerge) {
            // DISABLE std::cout << "merge edge E_dec = " << EwDec_max <<
            // std::endl;
            mergeBoundaryEdges(std::pair<int, int>(path_max[0], path_max[1]),
                               std::pair<int, int>(path_max[1], path_max[2]),
                               newVertPos_max.row(0));
            // DISABLE logFile << "edge merged" << std::endl;
            computeFeatures(); // TODO: only update locally
        } else {
            if (!splitInterior) {
                // boundary split
                // DISABLE std::cout << "boundary split E_dec = " << EwDec_max
                // << std::endl;
                splitEdgeOnBoundary(
                    std::pair<int, int>(path_max[0], path_max[1]),
                    newVertPos_max);
                // DISABLE logFile << "boundary edge split" << std::endl;
                // TODO: process fractail here!
                updateFeatures();
            } else {
                // interior split
                // DISABLE  std::cout << "Interior split E_dec = " << EwDec_max
                // << std::endl;
                cutPath(path_max, true, 1, newVertPos_max);
                // DISABLE logFile << "interior edge split" << std::endl;
                fracTail.insert(path_max[0]);
                fracTail.insert(path_max[2]);
                curInteriorFracTails.first = path_max[0];
                curInteriorFracTails.second = path_max[2];
                curFracTail = -1;
            }
        }
        return true;
    } else {
        // DISABLE  std::cout << "max E_dec = " << EwDec_max << " < thres " <<
        // EDecThres << std::endl;
        return false;
    }
}

void TriMesh::onePointCut(int vI) {
    assert((vI >= 0) && (vI < V_rest.rows()));
    std::vector<int> path(vNeighbor[vI].begin(), vNeighbor[vI].end());
    assert(path.size() >= 3);
    path[1] = vI;
    path.resize(3);

    bool makeCoh = true;
    if (!makeCoh) {
        for (int pI = 0; pI + 1 < path.size(); pI++)
            initSeamLen +=
                (V_rest.row(path[pI]) - V_rest.row(path[pI + 1])).norm();
    }
    cutPath(path, makeCoh);
    if (makeCoh)
        initSeams = cohE;
}

void TriMesh::highCurvOnePointCut(void) {
    std::vector<double> gaussianCurv(V.rows(), 2.0 * M_PI);
    for (int triI = 0; triI < F.rows(); triI++) {
        const Eigen::RowVector3i &triVInd = F.row(triI);
        const Eigen::RowVector3d v[3] = {V_rest.row(triVInd[0]),
                                         V_rest.row(triVInd[1]),
                                         V_rest.row(triVInd[2])};
        for (int vI = 0; vI < 3; vI++) {
            int vI_post = (vI + 1) % 3;
            int vI_pre = (vI + 2) % 3;
            const Eigen::RowVector3d e0 = v[vI_pre] - v[vI];
            const Eigen::RowVector3d e1 = v[vI_post] - v[vI];
            gaussianCurv[triVInd[vI]] -= std::acos((
                std::max)(-1.0,
                          (std::min)(1.0, e0.dot(e1) / e0.norm() / e1.norm())));
        }
    }

    for (auto &gcI : gaussianCurv) {
        if (gcI < 0)
            gcI = -gcI;
    }
    int vI_maxGC = 0;
    for (int vI = 1; vI < gaussianCurv.size(); vI++) {
        if (gaussianCurv[vI] > gaussianCurv[vI_maxGC])
            vI_maxGC = vI;
    }

    assert(vNeighbor[vI_maxGC].size() >= 3);
    int vJ_maxGC = *vNeighbor[vI_maxGC].begin();
    for (const auto &vINb : vNeighbor[vI_maxGC]) {
        if (gaussianCurv[vINb] > gaussianCurv[vJ_maxGC])
            vJ_maxGC = vINb;
    }

    assert(vNeighbor[vJ_maxGC].size() >= 3);
    int vK_maxGC = -1;
    double gc_vK = -DBL_MAX;
    for (const auto &vJNb : vNeighbor[vJ_maxGC]) {
        if ((gaussianCurv[vJNb] > gc_vK) && (vJNb != vI_maxGC)) {
            vK_maxGC = vJNb;
            gc_vK = gaussianCurv[vJNb];
        }
    }

    std::vector<int> path(3);
    path[0] = vI_maxGC;
    path[1] = vJ_maxGC;
    path[2] = vK_maxGC;

    bool makeCoh = true;
    if (!makeCoh) {
        for (int pI = 0; pI + 1 < path.size(); pI++)
            initSeamLen +=
                (V_rest.row(path[pI]) - V_rest.row(path[pI + 1])).norm();
    }
    cutPath(path, makeCoh);
    if (makeCoh)
        initSeams = cohE;
}

// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
int minDistance(const std::vector<double> &dist,
                const std::vector<bool> &sptSet) {
    // Initialize min value
    double min = DBL_MAX;
    int min_index = -1;

    for (int v = 0; v < dist.size(); v++) {
        if ((!sptSet[v]) && (dist[v] <= min))
            min = dist[v], min_index = v;
    }

    return min_index;
}

// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
void dijkstra(const std::vector<std::map<int, double>> &graph, int src,
              std::vector<double> &dist, std::vector<int> &parent) {
    int nV = static_cast<int>(graph.size());

    dist.resize(0);
    dist.resize(nV, DBL_MAX);
    // The output array.  dist[i] will hold the shortest
    // distance from src to i

    std::vector<bool> sptSet(
        nV, false); // sptSet[i] will true if vertex i is included in shortest
    // path tree or shortest distance from src to i is finalized

    parent.resize(0);
    parent.resize(nV, -1);

    // Distance of source vertex from itself is always 0
    dist[src] = 0.0;

    // Find shortest path for all vertices
    for (int count = 0; count + 1 < nV; count++) {
        // Pick the minimum distance vertex from the set of vertices not
        // yet processed. u is always equal to src in first iteration.
        int u = minDistance(dist, sptSet);

        // Mark the picked vertex as processed
        sptSet[u] = true;

        for (const auto v : graph[u]) {
            // Update dist[v] only if is not in sptSet, there is an edge from
            // u to v, and total weight of path from src to  v through u is
            // smaller than current value of dist[v]
            if ((!sptSet[v.first]) && (dist[u] != DBL_MAX) &&
                (dist[u] + v.second < dist[v.first])) {
                dist[v.first] = dist[u] + v.second;
                parent[v.first] = u;
            }
        }
    }
}

int getFarthestPoint(const std::vector<std::map<int, double>> &graph, int src) {
    int nV = static_cast<int>(graph.size());
    std::vector<double> dist;
    std::vector<int> parent;
    dijkstra(graph, src, dist, parent);

    double maxDist = 0.0;
    int vI_maxDist = -1;
    for (int vI = 0; vI < nV; vI++) {
        if (dist[vI] > maxDist) {
            maxDist = dist[vI];
            vI_maxDist = vI;
        }
    }
    assert(vI_maxDist >= 0);
    return vI_maxDist;
}

void getFarthestPointPath(const std::vector<std::map<int, double>> &graph,
                          int src, std::vector<int> &path) {
    int nV = static_cast<int>(graph.size());
    std::vector<double> dist;
    std::vector<int> parent;
    dijkstra(graph, src, dist, parent);

    double maxDist = 0.0;
    int vI_maxDist = -1;
    for (int vI = 0; vI < nV; vI++) {
        if (dist[vI] > maxDist) {
            maxDist = dist[vI];
            vI_maxDist = vI;
        }
    }
    assert(vI_maxDist >= 0);
    path.resize(0);
    while (vI_maxDist >= 0) {
        path.emplace_back(vI_maxDist);
        vI_maxDist = parent[vI_maxDist];
    }
    std::reverse(path.begin(), path.end());
}

void TriMesh::farthestPointCut(int p_vI) {
    assert(vNeighbor.size() == V_rest.rows());

    std::vector<std::map<int, double>> graph(vNeighbor.size());
    for (int vI = 0; vI < vNeighbor.size(); vI++) {
        for (const auto nbI : vNeighbor[vI]) {
            if (nbI > vI)
                graph[nbI][vI] = graph[vI][nbI] =
                    (V_rest.row(vI) - V_rest.row(nbI)).norm();
        }
    }

    std::vector<int> path;
    getFarthestPointPath(graph, getFarthestPoint(graph, p_vI), path);

    bool makeCoh = true;
    if (!makeCoh) {
        for (int pI = 0; pI + 1 < path.size(); pI++)
            initSeamLen +=
                (V_rest.row(path[pI]) - V_rest.row(path[pI + 1])).norm();
    }
    cutPath(path, makeCoh);
    if (makeCoh)
        initSeams = cohE;
}

// void TriMesh::geomImgCut(TriMesh &data_findExtrema) {
//     // compute UV map for find extremal point (interior)
//     data_findExtrema = *this;
//     const int mapType = 1; // 0: SD, 1: harmonic (uniform), 2: harmonic
//                            // (cotangent), 3: harmonic (MVC)
//     if (mapType) {
//         Eigen::VectorXi bnd;
//         igl::boundary_loop(this->F, bnd); // Find the open boundary
//         assert(bnd.size());
//         // TODO: ensure it doesn't have multiple boundaries? or
//         // multi-components?
//
//         // Map the boundary to a circle, preserving edge proportions
//         Eigen::MatrixXd bnd_uv;
//         uvgami::IglUtils::map_vertices_to_circle(this->V_rest, bnd, bnd_uv);
//
//         Eigen::MatrixXd UV_Tutte;
//
//         switch (mapType) {
//         case 1: {
//             // Harmonic map with uniform weights
//             Eigen::SparseMatrix<double> A, M;
//             uvgami::IglUtils::computeUniformLaplacian(this->F, A);
//             igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);
//             break;
//         }
//
//         case 2: {
//             // Harmonic parametrization
//             igl::harmonic(V, F, bnd, bnd_uv, 1, UV_Tutte);
//             break;
//         }
//
//         case 3: {
//             // Shape Preserving Mesh Parameterization
//             // (Harmonic map with MVC weights)
//             Eigen::SparseMatrix<double> A;
//             uvgami::IglUtils::computeMVCMtr(this->V_rest, this->F, A);
//             uvgami::IglUtils::fixedBoundaryParam_MVC(A, bnd, bnd_uv, UV_Tutte);
//             break;
//         }
//
//         default:
//             assert(0 &&
//                    "Unknown map specified for finding Geometry Image cuts");
//             break;
//         }
//
//         data_findExtrema.V = UV_Tutte;
//     }
//
//     // pick the vertex with largest L2 stretch
//     int vI_extremal = -1;
//     Eigen::VectorXd L2stretchPerElem, vertScores;
//     data_findExtrema.computeL2StretchPerElem(L2stretchPerElem);
//     vertScores.resize(V_rest.rows());
//     vertScores.setZero();
//     for (int triI = 0; triI < F.rows(); triI++) {
//         for (int i = 0; i < 3; i++) {
//             if (vertScores[F(triI, i)] < L2stretchPerElem[triI])
//                 vertScores[F(triI, i)] = L2stretchPerElem[triI];
//         }
//     }
//     double extremal = 0.0;
//     for (int vI = 0; vI < vertScores.size(); vI++) {
//         if (!isBoundaryVert(vI)) {
//             if (extremal < vertScores[vI]) {
//                 extremal = vertScores[vI];
//                 vI_extremal = vI;
//             }
//         }
//     }
//     assert(vI_extremal >= 0);
//
//     // construct mesh graph
//     assert(vNeighbor.size() == V_rest.rows());
//     std::vector<std::map<int, double>> graph(vNeighbor.size());
//     for (int vI = 0; vI < vNeighbor.size(); vI++) {
//         for (const auto nbI : vNeighbor[vI]) {
//             if (nbI > vI)
//                 graph[nbI][vI] = graph[vI][nbI] =
//                     (V_rest.row(vI) - V_rest.row(nbI)).norm();
//         }
//     }
//
//     // find closest point on boundary
//     int nV = static_cast<int>(graph.size());
//     std::vector<double> dist;
//     std::vector<int> parent;
//     dijkstra(graph, vI_extremal, dist, parent);
//     double minDistToBound = DBL_MAX;
//     int vI_minDistToBound = -1;
//     for (int vI = 0; vI < nV; vI++) {
//         if (isBoundaryVert(vI)) {
//             if (dist[vI] < minDistToBound) {
//                 minDistToBound = dist[vI];
//                 vI_minDistToBound = vI;
//             }
//         }
//     }
//     assert((vI_minDistToBound >= 0) && "No boundary on the mesh!");
//
//     // find shortest path to closest point on boundary
//     std::vector<int> path;
//     while (vI_minDistToBound >= 0) {
//         path.emplace_back(vI_minDistToBound);
//         vI_minDistToBound = parent[vI_minDistToBound];
//     }
//     std::reverse(path.begin(), path.end());
//     cutPath(path, true);
// }

int TriMesh::cutPath(std::vector<int> path, bool makeCoh, int changePos,
                     const Eigen::MatrixXd &newVertPos, bool allowCutThrough) {
    int cuts_made = 0;
    assert(path.size() >= 2);
    if (changePos) {
        assert((changePos == 1) &&
               "right now only support change 1"); //!!! still only allow 1?
        assert(newVertPos.cols() == 2);
        assert(changePos * 2 == newVertPos.rows());
    }

    // for (int pI = 1; pI + 1 < path.size(); pI++) {
    //     assert(!isBoundaryVert(path[pI]) &&
    //            "Boundary vertices detected on split path, please split "
    //            "multiple times!");
    // }

    bool isFromBound = isBoundaryVert(path[0]);
    bool isToBound = isBoundaryVert(path.back());
    if (allowCutThrough && (isFromBound || isToBound)) {
        bool cutThrough = false;
        if (isFromBound && isToBound) {
            cutThrough = true;
        } else if (isToBound) {
            std::reverse(path.begin(), path.end());
            // always start cut from boundary
        }

        for (int vI = 0; vI + 1 < path.size(); vI++) {
            int vInd_s = path[vI];
            int vInd_e = path[vI + 1];
            if ((edge2Tri.find(std::pair<int, int>(vInd_s, vInd_e)) ==
                 edge2Tri.end()) ||
                (edge2Tri.find(std::pair<int, int>(vInd_e, vInd_s)) ==
                 edge2Tri.end())) {
                continue;
            }
            Eigen::MatrixXd newVertPos;
            if (cutThrough) {
                newVertPos.resize(4, 2);
                newVertPos << V.row(vInd_s), V.row(vInd_s), V.row(vInd_e),
                    V.row(vInd_e);
            } else {
                newVertPos.resize(2, 2);
                newVertPos << V.row(vInd_s), V.row(vInd_s);
            }
            splitEdgeOnBoundary(std::pair<int, int>(vInd_s, vInd_e), newVertPos,
                                true); //!!! make coh?
            ++cuts_made;
            updateFeatures();
        }
    } else {
        // path is interior
        assert(path.size() >= 3);

        std::vector<int> tri_left;
        int vI = path[1];
        int vI_new = path[0];
        while (1) {
            auto finder = edge2Tri.find(std::pair<int, int>(vI, vI_new));
            if (finder == edge2Tri.end())
                throw std::runtime_error("cut path leaves the surface");
            tri_left.emplace_back(finder->second);
            const Eigen::RowVector3i &triVInd = F.row(finder->second);
            for (int i = 0; i < 3; i++) {
                if ((triVInd[i] != vI) && (triVInd[i] != vI_new)) {
                    vI_new = triVInd[i];
                    break;
                }
            }

            if (vI_new == path[2])
                break;
            if (vI_new == path[0])
                throw std::runtime_error(
                    "cut path circles its vertex without splitting it");
        }

        int nV = static_cast<int>(V_rest.rows());
        V_rest.conservativeResize(nV + 1, 3);
        V_rest.row(nV) = V_rest.row(path[1]);
        vertWeight.conservativeResize(nV + 1);
        vertWeight[nV] = vertWeight[path[1]];
        V.conservativeResize(nV + 1, 2);
        if (changePos) {
            V.row(nV) = newVertPos.block(0, 0, 1, 2);
            V.row(path[1]) = newVertPos.block(1, 0, 1, 2);
        } else {
            V.row(nV) = V.row(path[1]);
        }
        for (const auto triI : tri_left) {
            for (int vI = 0; vI < 3; vI++) {
                if (F(triI, vI) == path[1]) {
                    F(triI, vI) = nV;
                    break;
                }
            }
        }
        if (makeCoh) {
            int nCoh = static_cast<int>(cohE.rows());
            cohE.conservativeResize(nCoh + 2, 4);
            cohE.row(nCoh) << nV, path[0], path[1], path[0];
            cohE.row(nCoh + 1) << path[2], nV, path[2], path[1];
        }
        computeFeatures(); // TODO: only update locally
        for (int vI = 2; vI + 1 < path.size(); vI++) {
            int vInd_s = path[vI];
            int vInd_e = path[vI + 1];
            if ((edge2Tri.find(std::pair<int, int>(vInd_s, vInd_e)) ==
                 edge2Tri.end()) ||
                (edge2Tri.find(std::pair<int, int>(vInd_e, vInd_s)) ==
                 edge2Tri.end())) {
                continue;
            }
            Eigen::Matrix2d newVertPos;
            newVertPos << V.row(vInd_s), V.row(vInd_s);
            splitEdgeOnBoundary(std::pair<int, int>(vInd_s, vInd_e), newVertPos,
                                true, allowCutThrough); //!!! make coh?
            ++cuts_made;
            updateFeatures();
        }
    }

    return cuts_made;
}

// void TriMesh::computeSeamScore(Eigen::VectorXd &seamScore) const {
//     seamScore.resize(cohE.rows());
//     for (int cohI = 0; cohI < cohE.rows(); cohI++) {
//         if (boundaryEdge[cohI]) {
//             seamScore[cohI] = -1.0;
//         } else {
//             seamScore[cohI] =
//                 (std::max)((V.row(cohE(cohI, 0)) - V.row(cohE(cohI, 2))).norm(),
//                            (V.row(cohE(cohI, 1)) - V.row(cohE(cohI, 3)))
//                                .norm()) /
//                 avgEdgeLen;
//         }
//     }
// }
// void TriMesh::computeBoundaryLen(double &boundaryLen) const {
//     boundaryLen = 0.0;
//     for (const auto &e : edge2Tri) {
//         if (edge2Tri.find(std::pair<int, int>(e.first.second, e.first.first)) ==
//             edge2Tri.end())
//             boundaryLen +=
//                 (V_rest.row(e.first.second) - V_rest.row(e.first.first)).norm();
//     }
// }
void TriMesh::computeSeamSparsity(double &sparsity, bool triSoup) const {
    const double thres = 1.0e-2;
    sparsity = 0.0;
    for (int cohI = 0; cohI < cohE.rows(); cohI++) {
        if (!boundaryEdge[cohI]) {
            if ((!triSoup) ||
                ((V.row(cohE(cohI, 0)) - V.row(cohE(cohI, 2))).norm() /
                     avgEdgeLen >
                 thres) ||
                ((V.row(cohE(cohI, 1)) - V.row(cohE(cohI, 3))).norm() /
                     avgEdgeLen >
                 thres)) {
                sparsity += edgeLen[cohI];
            }
        }
    }
    sparsity += initSeamLen;
}
// void TriMesh::computeL2StretchPerElem(Eigen::VectorXd &L2StretchPerElem) const {
//     L2StretchPerElem.resize(F.rows());
//     for (int triI = 0; triI < F.rows(); triI++) {
//         const Eigen::Vector3i &triVInd = F.row(triI);
//         const Eigen::Vector3d x_3D[3] = {V_rest.row(triVInd[0]),
//                                          V_rest.row(triVInd[1]),
//                                          V_rest.row(triVInd[2])};
//         const Eigen::Vector2d uv[3] = {V.row(triVInd[0]), V.row(triVInd[1]),
//                                        V.row(triVInd[2])};
//         Eigen::Matrix2d dg;
//         IglUtils::computeDeformationGradient(x_3D, uv, dg);
//
//         const double a = Eigen::Vector2d(dg.block(0, 0, 2, 1)).squaredNorm();
//         const double c = Eigen::Vector2d(dg.block(0, 1, 2, 1)).squaredNorm();
//         const double t0 = a + c;
//
//         L2StretchPerElem[triI] = std::sqrt(t0 / 2.0);
//     }
// }
// void TriMesh::computeStandardStretch(double &stretch_l2, double &stretch_inf,
//                                      double &stretch_shear,
//                                      double &compress_inf) const {
//     stretch_l2 = 0.0;
//     stretch_inf = -DBL_MAX;
//     stretch_shear = 0.0;
//     compress_inf = DBL_MAX;
//     for (int triI = 0; triI < F.rows(); triI++) {
//         const Eigen::Vector3i &triVInd = F.row(triI);
//         const Eigen::Vector3d x_3D[3] = {V_rest.row(triVInd[0]),
//                                          V_rest.row(triVInd[1]),
//                                          V_rest.row(triVInd[2])};
//         const Eigen::Vector2d uv[3] = {V.row(triVInd[0]), V.row(triVInd[1]),
//                                        V.row(triVInd[2])};
//         Eigen::Matrix2d dg;
//         IglUtils::computeDeformationGradient(x_3D, uv, dg);
//
//         const double a = Eigen::Vector2d(dg.block(0, 0, 2, 1)).squaredNorm();
//         const double b = Eigen::Vector2d(dg.block(0, 0, 2, 1))
//                              .dot(Eigen::Vector2d(dg.block(0, 1, 2, 1)));
//         const double c = Eigen::Vector2d(dg.block(0, 1, 2, 1)).squaredNorm();
//         const double t0 = a + c;
//         const double t1 = std::sqrt((a - c) * (a - c) + 4. * b * b);
//         const double tau = std::sqrt((t0 + t1) / 2.);
//         const double gamma = std::sqrt((t0 - t1) / 2.);
//
//         stretch_l2 += t0 / 2.0 * triArea[triI];
//         if (tau > stretch_inf)
//             stretch_inf = tau;
//         stretch_shear += b * b / a / c * triArea[triI];
//         if (gamma < compress_inf)
//             compress_inf = gamma;
//     }
//     stretch_l2 /= surfaceArea;
//     stretch_l2 = std::sqrt(stretch_l2);
//     stretch_shear /= surfaceArea;
//     stretch_shear = std::sqrt(stretch_shear);
//
//     double surfaceArea_UV = 0.0;
//     for (int triI = 0; triI < F.rows(); triI++) {
//         const Eigen::Vector3i &triVInd = F.row(triI);
//
//         const Eigen::Vector2d &U1 = V.row(triVInd[0]);
//         const Eigen::Vector2d &U2 = V.row(triVInd[1]);
//         const Eigen::Vector2d &U3 = V.row(triVInd[2]);
//
//         const Eigen::Vector2d U2m1 = U2 - U1;
//         const Eigen::Vector2d U3m1 = U3 - U1;
//
//         surfaceArea_UV += 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
//     }
//
//     // area scaling:
//     const double scaleFactor = std::sqrt(surfaceArea_UV / surfaceArea);
//     stretch_l2 *= scaleFactor;
//     stretch_inf *= scaleFactor;
//     compress_inf *= scaleFactor; // not meaningful now...
//     // stretch_shear won't be affected by area scaling
// }
// void TriMesh::outputStandardStretch(std::ofstream &file) const {
//     double stretch_l2, stretch_inf, stretch_shear, compress_inf;
//     computeStandardStretch(stretch_l2, stretch_inf, stretch_shear,
//                            compress_inf);
//     file << stretch_l2 << " " << stretch_inf << " " << stretch_shear << " "
//          << compress_inf << std::endl;
// }
// void TriMesh::computeAbsGaussianCurv(double &absGaussianCurv) const {
//     std::vector<double> weights(V.rows(), 0.0);
//     std::vector<double> gaussianCurv(V.rows(), 2.0 * M_PI);
//     for (int triI = 0; triI < F.rows(); triI++) {
//         const Eigen::RowVector3i &triVInd = F.row(triI);
//         const Eigen::RowVector3d v[3] = {V_rest.row(triVInd[0]),
//                                          V_rest.row(triVInd[1]),
//                                          V_rest.row(triVInd[2])};
//         for (int vI = 0; vI < 3; vI++) {
//             int vI_post = (vI + 1) % 3;
//             int vI_pre = (vI + 2) % 3;
//             const Eigen::RowVector3d e0 = v[vI_pre] - v[vI];
//             const Eigen::RowVector3d e1 = v[vI_post] - v[vI];
//             gaussianCurv[triVInd[vI]] -= std::acos((
//                 std::max)(-1.0,
//                           (std::min)(1.0, e0.dot(e1) / e0.norm() / e1.norm())));
//             weights[triVInd[vI]] += triArea[triI];
//         }
//     }
//
//     absGaussianCurv = 0.0;
//     for (int vI = 0; vI < V.rows(); vI++) {
//         if (!isBoundaryVert(vI))
//             absGaussianCurv += std::abs(gaussianCurv[vI]) * weights[vI];
//     }
//     absGaussianCurv /= surfaceArea * 3.0;
// }

void TriMesh::initRigidUV(void) {
    V.resize(V_rest.rows(), 2);
    for (int triI = 0; triI < F.rows(); triI++) {
        const Eigen::Vector3i &triVInd = F.row(triI);

        const Eigen::Vector3d x_3D[3] = {V_rest.row(triVInd[0]),
                                         V_rest.row(triVInd[1]),
                                         V_rest.row(triVInd[2])};
        Eigen::Vector2d x[3];
        IglUtils::mapTriangleTo2D(x_3D, x);

        V.row(triVInd[0]) = x[0];
        V.row(triVInd[1]) = x[1];
        V.row(triVInd[2]) = x[2];
    }
}

bool TriMesh::checkInversion(int triI, bool mute) const {
    assert(triI < F.rows());

    const double eps = 0.0;

    const Eigen::Vector3i &triVInd = F.row(triI);

    const Eigen::Vector2d e_u[2] = {V.row(triVInd[1]) - V.row(triVInd[0]),
                                    V.row(triVInd[2]) - V.row(triVInd[0])};

    // a flat triangle counts too, its distortion is infinite
    const double dbArea = e_u[0][0] * e_u[1][1] - e_u[0][1] * e_u[1][0];
    if (dbArea <= eps) {
        if (!mute) {
            std::cout << "***Element inversion detected: " << dbArea << " < "
                      << eps << std::endl;
            std::cout << "mesh triangle count: " << F.rows() << std::endl;
            // DISABLE logFile << "***Element inversion detected: " << dbArea <<
            // " < " << eps << std::endl;
        }
        return false;
    } else {
        return true;
    }
}
bool TriMesh::checkInversion(bool mute,
                             const std::vector<int> &triangles) const {
    if (triangles.empty()) {
        for (int triI = 0; triI < F.rows(); triI++) {
            if (!checkInversion(triI, mute))
                return false;
        }
    } else {
        for (const auto &triI : triangles) {
            if (!checkInversion(triI, mute))
                return false;
        }
    }

    return true;
}

bool TriMesh::save(const std::string &filePath, const Eigen::MatrixXd &V,
                   const Eigen::MatrixXi &F, const Eigen::MatrixXd UV,
                   const Eigen::MatrixXi &FUV) const {
    std::ofstream out;
    std::string meshPath = filePath + ".obj";
    out.open(meshPath);
    if (!out.is_open())
        return false;
    for (int vI = 0; vI < V.rows(); vI++) {
        const Eigen::RowVector3d &v = V.row(vI);
        out << "v " << std::scientific << v[0] << " " << std::scientific << v[1]
            << " " << std::scientific << v[2] << std::endl;
    }

    for (int vI = 0; vI < UV.rows(); vI++) {
        const Eigen::RowVector2d &uv = UV.row(vI);
        out << "vt " << std::scientific << uv[0] << " " << std::scientific
            << uv[1] << std::endl;
    }

    if (FUV.rows() == F.rows()) {
        for (int triI = 0; triI < F.rows(); triI++) {
            const Eigen::RowVector3i &tri = F.row(triI);
            const Eigen::RowVector3i &tri_UV = FUV.row(triI);
            out << "f " << tri[0] + 1 << "/" << tri_UV[0] + 1 << " "
                << tri[1] + 1 << "/" << tri_UV[1] + 1 << " " << tri[2] + 1
                << "/" << tri_UV[2] + 1 << std::endl;
        }
    } else {
        for (int triI = 0; triI < F.rows(); triI++) {
            const Eigen::RowVector3i &tri = F.row(triI);
            out << "f " << tri[0] + 1 << "/" << tri[0] + 1 << " " << tri[1] + 1
                << "/" << tri[1] + 1 << " " << tri[2] + 1 << "/" << tri[2] + 1
                << std::endl;
        }
    }

    out.close();
    return true;
}

bool TriMesh::save(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                   const Eigen::MatrixXd UV, const Eigen::MatrixXi &FUV) const {

    std::cout << "visual_begin:" << std::endl;
    for (int vI = 0; vI < UV.rows(); vI++) {
        const Eigen::RowVector2d &uv = UV.row(vI);
        std::cout << "vt " << std::scientific << uv[0] << " " << std::scientific
                  << uv[1] << std::endl;
    }

    if (FUV.rows() == F.rows()) {
        for (int triI = 0; triI < F.rows(); triI++) {
            const Eigen::RowVector3i &tri = F.row(triI);
            const Eigen::RowVector3i &tri_UV = FUV.row(triI);
            std::cout << "f " << tri_UV[0] << " " << tri_UV[1] << " "
                      << tri_UV[2] << std::endl;
        }
    }
    std::cout << "visual_end:" << std::endl;
    return true;
}

bool TriMesh::saveAsMesh(const Eigen::MatrixXi &F0, bool scaleUV) const {
    assert(F0.rows() == F.rows());
    assert(F0.cols() == 3);

    Eigen::MatrixXd V_mesh;
    for (int fI = 0; fI < F0.rows(); ++fI) {
        for (int localVI = 0; localVI < 3; ++localVI) {
            int vI = F0(fI, localVI);
            if (vI >= V_mesh.rows())
                V_mesh.conservativeResize(vI + 1, 3);
            V_mesh.row(vI) = V_rest.row(F(fI, localVI));
        }
    }

    Eigen::MatrixXd UV_mesh = V;
    if (scaleUV) {
        const Eigen::VectorXd &u = UV_mesh.col(0);
        const Eigen::VectorXd &v = UV_mesh.col(1);
        const double uMin = u.minCoeff();
        const double vMin = v.minCoeff();
        const double uScale = u.maxCoeff() - uMin;
        const double vScale = v.maxCoeff() - vMin;
        const double scale = (std::max)(uScale, vScale);
        for (int uvI = 0; uvI < UV_mesh.rows(); uvI++) {
            UV_mesh(uvI, 0) = (UV_mesh(uvI, 0) - uMin) / scale;
            UV_mesh(uvI, 1) = (UV_mesh(uvI, 1) - vMin) / scale;
        }
    }

    return save(V_mesh, F0, UV_mesh, F);
}

bool TriMesh::saveAsMesh(const std::string &filePath, const Eigen::MatrixXi &F0,
                         bool scaleUV) const {
    assert(F0.rows() == F.rows());
    assert(F0.cols() == 3);

    Eigen::MatrixXd V_mesh;
    for (int fI = 0; fI < F0.rows(); ++fI) {
        for (int localVI = 0; localVI < 3; ++localVI) {
            int vI = F0(fI, localVI);
            if (vI >= V_mesh.rows())
                V_mesh.conservativeResize(vI + 1, 3);
            V_mesh.row(vI) = V_rest.row(F(fI, localVI));
        }
    }

    Eigen::MatrixXd UV_mesh = V;
    if (scaleUV) {
        const Eigen::VectorXd &u = UV_mesh.col(0);
        const Eigen::VectorXd &v = UV_mesh.col(1);
        const double uMin = u.minCoeff();
        const double vMin = v.minCoeff();
        const double uScale = u.maxCoeff() - uMin;
        const double vScale = v.maxCoeff() - vMin;
        const double scale = (std::max)(uScale, vScale);
        for (int uvI = 0; uvI < UV_mesh.rows(); uvI++) {
            UV_mesh(uvI, 0) = (UV_mesh(uvI, 0) - uMin) / scale;
            UV_mesh(uvI, 1) = (UV_mesh(uvI, 1) - vMin) / scale;
        }
    }

    return save(filePath, V_mesh, F0, UV_mesh, F);
}

bool TriMesh::findBoundaryEdge(int vI, const std::pair<int, int> &startEdge,
                               std::pair<int, int> &boundaryEdge) {
    auto finder = edge2Tri.find(startEdge);
    assert(finder != edge2Tri.end());
    bool proceed = (startEdge.first != vI);
    const int vI_neighbor = (proceed ? startEdge.first : startEdge.second);
    int vI_new = vI_neighbor;
    while (1) {
        const Eigen::RowVector3i &triVInd = F.row(finder->second);
        for (int i = 0; i < 3; i++) {
            if ((triVInd[i] != vI) && (triVInd[i] != vI_new)) {
                vI_new = triVInd[i];
                break;
            }
        }
        if (vI_new == vI_neighbor)
            return false;
        finder = edge2Tri.find(proceed ? std::pair<int, int>(vI_new, vI)
                                       : std::pair<int, int>(vI, vI_new));
        if (finder == edge2Tri.end()) {
            boundaryEdge.first = (proceed ? vI : vI_new);
            boundaryEdge.second = (proceed ? vI_new : vI);
            return true;
        }
    }
}

bool TriMesh::insideTri(int triI, const Eigen::RowVector2d &pos) const {
    const Eigen::RowVector3i &triVInd = F.row(triI);
    const Eigen::RowVector2d e01 = V.row(triVInd[1]) - V.row(triVInd[0]);
    const Eigen::RowVector2d e02 = V.row(triVInd[2]) - V.row(triVInd[0]);
    const Eigen::RowVector2d e0p = pos - V.row(triVInd[0]);

    // represent ep0 using e01 and e02
    Eigen::Matrix2d A;
    A << e01.transpose(), e02.transpose();
    Eigen::Vector2d coef = A.colPivHouseholderQr().solve(e0p.transpose());

    // check inside
    return ((coef[0] >= 0.0) && (coef[1] >= 0.0) && (coef[0] + coef[1] <= 1.0));
}

bool TriMesh::insideUVRegion(const std::vector<int> &triangles,
                             const Eigen::RowVector2d &pos) const {
    for (const auto &triI : triangles) {
        if (insideTri(triI, pos))
            return true;
    }
    return false;
}

bool TriMesh::isBoundaryVert(int vI, int vI_neighbor,
                             std::vector<int> &tri_toSep,
                             std::pair<int, int> &boundaryEdge,
                             bool toBound) const {
    tri_toSep.resize(0);
    auto finder = edge2Tri.find(toBound ? std::pair<int, int>(vI_neighbor, vI)
                                        : std::pair<int, int>(vI, vI_neighbor));
    if (finder == edge2Tri.end()) {
        boundaryEdge.first = (toBound ? vI : vI_neighbor);
        boundaryEdge.second = (toBound ? vI_neighbor : vI);
        return true;
    }

    int vI_new = vI_neighbor;
    do {
        tri_toSep.emplace_back(finder->second);
        const Eigen::RowVector3i &triVInd = F.row(finder->second);
        for (int i = 0; i < 3; i++) {
            if ((triVInd[i] != vI) && (triVInd[i] != vI_new)) {
                vI_new = triVInd[i];
                break;
            }
        }
        if (vI_new == vI_neighbor)
            return false;
        finder = edge2Tri.find(toBound ? std::pair<int, int>(vI_new, vI)
                                       : std::pair<int, int>(vI, vI_new));
        if (finder == edge2Tri.end()) {
            boundaryEdge.first = (toBound ? vI : vI_new);
            boundaryEdge.second = (toBound ? vI_new : vI);
            return true;
        }
    } while (1);
}

// same loops and vertex order as igl::boundary_loop, Triangle's output
// depends on the order
void TriMesh::boundaryLoops(std::vector<std::vector<int>> &loops) const {
    loops.clear();
    // outgoing boundary edges per vertex as (triangle, end vertex)
    std::vector<std::vector<std::pair<int, int>>> outgoing(V.rows());
    std::vector<char> unvisited(V.rows(), 0);
    for (const auto &edgeTri : edge2Tri) {
        const std::pair<int, int> &edge = edgeTri.first;
        if (edge2Tri.count(std::pair<int, int>(edge.second, edge.first)))
            continue;
        outgoing[edge.first].emplace_back(edgeTri.second, edge.second);
        unvisited[edge.first] = unvisited[edge.second] = 1;
    }
    for (int start = 0; start < V.rows(); start++) {
        if (!unvisited[start])
            continue;
        std::vector<int> loop(1, start);
        unvisited[start] = 0;
        int vI = start;
        while (true) {
            // igl walks the lowest indexed triangle first
            int next = -1, nextTri = INT_MAX;
            for (const auto &triEnd : outgoing[vI]) {
                if (unvisited[triEnd.second] && triEnd.first < nextTri) {
                    nextTri = triEnd.first;
                    next = triEnd.second;
                }
            }
            if (next < 0)
                break;
            loop.push_back(next);
            unvisited[next] = 0;
            vI = next;
        }
        loops.emplace_back(std::move(loop));
    }
}

bool TriMesh::isBoundaryVert(int vI) const {
    assert(vNeighbor.size() == V.rows());
    assert(vI < vNeighbor.size());

    for (const auto vI_neighbor : vNeighbor[vI]) {
        if ((edge2Tri.find(std::pair<int, int>(vI, vI_neighbor)) ==
             edge2Tri.end()) ||
            (edge2Tri.find(std::pair<int, int>(vI_neighbor, vI)) ==
             edge2Tri.end())) {
            return true;
        }
    }

    return false;
}

void TriMesh::compute2DInwardNormal(int vI, Eigen::RowVector2d &normal) const {
    std::vector<int> incTris[2];
    std::pair<int, int> boundaryEdge[2];
    if (!isBoundaryVert(vI, *vNeighbor[vI].begin(), incTris[0], boundaryEdge[0],
                        0))
        return;
    isBoundaryVert(vI, *vNeighbor[vI].begin(), incTris[1], boundaryEdge[1], 1);
    assert(!(incTris[0].empty() && incTris[1].empty()));

    Eigen::RowVector2d boundaryEdgeDir[2] = {
        (V.row(boundaryEdge[0].first) - V.row(boundaryEdge[0].second))
            .normalized(),
        (V.row(boundaryEdge[1].second) - V.row(boundaryEdge[1].first))
            .normalized(),
    };
    normal = (boundaryEdgeDir[0] + boundaryEdgeDir[1]).normalized();
    if (boundaryEdgeDir[1][0] * normal[1] - boundaryEdgeDir[1][1] * normal[0] <
        0.0)
        normal *= -1.0;
}

// concave gets the deeper discount, a groove hides a seam best
static const double CONCAVE_RELIEF = 0.5;
static const double CONVEX_RELIEF = 0.3;
// below this a curved surface reads as flat, so a sculpt is left alone
static const double RELIEF_LOW_ANGLE = 20.0;
static const double RELIEF_FULL_ANGLE = 45.0;

double TriMesh::creaseReliefTris(int triA, int triB, int baseVI) const {
    const double normalDot = std::max(
        -1.0,
        std::min(1.0, triNormal.row(triA).dot(triNormal.row(triB))));
    const double angle = std::acos(normalDot) * 180.0 / M_PI;
    if (angle <= RELIEF_LOW_ANGLE)
        return 1.0;
    // a neighbour risen above the face plane means concave
    const Eigen::RowVector3d centroidOpp =
        (V_rest.row(F(triB, 0)) + V_rest.row(F(triB, 1)) +
         V_rest.row(F(triB, 2))) /
        3.0;
    const double lift =
        triNormal.row(triA).dot(centroidOpp - V_rest.row(baseVI));
    const double depth = std::min(
        (angle - RELIEF_LOW_ANGLE) / (RELIEF_FULL_ANGLE - RELIEF_LOW_ANGLE),
        1.0);
    return 1.0 - (lift > 0.0 ? CONCAVE_RELIEF : CONVEX_RELIEF) * depth;
}

double TriMesh::creaseRelief(int vI, int nbVI) const {
    const auto tri = edge2Tri.find(std::pair<int, int>(vI, nbVI));
    const auto triOpp = edge2Tri.find(std::pair<int, int>(nbVI, vI));
    if ((tri == edge2Tri.end()) || (triOpp == edge2Tri.end()))
        return 1.0;
    return creaseReliefTris(tri->second, triOpp->second, vI);
}

int TriMesh::edgeOwnerTri(int vI, int nbVI) const {
    auto finder = edge2Tri.find(std::pair<int, int>(vI, nbVI));
    if (finder == edge2Tri.end())
        finder = edge2Tri.find(std::pair<int, int>(nbVI, vI));
    return (finder == edge2Tri.end()) ? -1 : finder->second;
}

// a seam edge's two sides are separate copies, so edge2Tri holds one of them
double TriMesh::creaseReliefSeam(int vI, int nbVI, int vTwinI) const {
    const int tri = edgeOwnerTri(vI, nbVI);
    const int triTwin = edgeOwnerTri(vTwinI, nbVI);
    if ((tri < 0) || (triTwin < 0) || (tri == triTwin))
        return 1.0;
    return creaseReliefTris(tri, triTwin, nbVI);
}

double
TriMesh::computeLocalLDec(int vI, double lambda_t, std::vector<int> &path_max,
                          Eigen::MatrixXd &newVertPos_max,
                          std::pair<double, double> &energyChanges_max,
                          const std::vector<int> &incTris,
                          const Eigen::RowVector2d &initMergedPos) const {
    if (forceQuit) {
        // a stop mid-query drains the remaining candidates as invalid so the
        // outer loop can save the current map instead of finishing the query
        energyChanges_max.first = DBL_MAX;
        energyChanges_max.second = DBL_MAX;
        return -DBL_MAX;
    }
    if (!path_max.empty()) {
        // merge query
        assert(path_max.size() >= 3);
        for (const auto &pI : path_max)
            assert(isBoundaryVert(pI));
        if (path_max.size() == 3) {
            // zipper merge
            // the same discount as seInc, so a split and the merge undoing
            // it net zero
            double seDec =
                (V_rest.row(path_max[0]) - V_rest.row(path_max[1])).norm() /
                virtualRadius *
                (vertWeight[path_max[0]] + vertWeight[path_max[1]]) / 2.0 *
                creaseReliefSeam(path_max[0], path_max[1], path_max[2]);
            // closing up split diamond
            bool closeup = false;
            for (const auto &nbVI : vNeighbor[path_max[0]]) {
                if (nbVI != path_max[1]) {
                    if (isBoundaryVert(nbVI)) {
                        if (vNeighbor[path_max[2]].find(nbVI) !=
                            vNeighbor[path_max[2]].end()) {
                            seDec +=
                                (V_rest.row(path_max[0]) - V_rest.row(nbVI))
                                    .norm() /
                                virtualRadius *
                                (vertWeight[path_max[0]] + vertWeight[nbVI]) /
                                2.0 *
                                creaseReliefSeam(path_max[0], nbVI,
                                                 path_max[2]);
                            closeup = true;
                            break;
                        }
                    }
                }
            }

            assert(incTris.size() >= 2);
            std::set<int> freeVert;
            freeVert.insert(path_max[0]);
            freeVert.insert(path_max[2]);
            std::map<int, int> mergeVert;
            mergeVert[path_max[0]] = path_max[2];
            mergeVert[path_max[2]] = path_max[0];
            std::map<int, Eigen::RowVector2d> newVertPos;
            const double SDInc = -computeLocalEdDec_merge(
                path_max, incTris, freeVert, newVertPos, mergeVert,
                initMergedPos, closeup);
            energyChanges_max.first = SDInc;
            energyChanges_max.second = -seDec;
            if (SDInc == DBL_MAX) {
                return -DBL_MAX;
            } else {
                auto finder = newVertPos.find(path_max[0]);
                assert(finder != newVertPos.end());
                newVertPos_max.resize(1, 2);
                newVertPos_max.row(0) = finder->second;

                return lambda_t * seDec - (1.0 - lambda_t) * SDInc;
            }
        } else {
            assert(0 && "currently not considering \"interior\" merge!");
        }
    }

    // split:
    // a full reversal triples an edge's seam cost
    const double turnPenalty = 2.0;
    std::vector<int> umbrella;
    std::pair<int, int> boundaryEdge;
    if (isBoundaryVert(vI, *(vNeighbor[vI].begin()), umbrella, boundaryEdge,
                       false)) {
        // boundary split
        double maxEwDec = -DBL_MAX;
        energyChanges_max.first = DBL_MAX;
        energyChanges_max.second = DBL_MAX;
        path_max.resize(2);
        // at a cut tip the two boundary neighbors are copies of one vertex
        std::vector<int> umbrella_other;
        std::pair<int, int> boundaryEdge_other;
        isBoundaryVert(vI, *(vNeighbor[vI].begin()), umbrella_other,
                       boundaryEdge_other, true);
        const int bndNb0 = boundaryEdge.first;
        const int bndNb1 = boundaryEdge_other.second;
        const bool atCutTip =
            (V_rest.row(bndNb0) - V_rest.row(bndNb1)).squaredNorm() == 0.0;
        Eigen::RowVector3d arrivalDir;
        if (atCutTip)
            arrivalDir = (V_rest.row(vI) - V_rest.row(bndNb0)).normalized();
        for (const auto &nbVI : vNeighbor[vI]) {
            const std::pair<int, int> edge(vI, nbVI);
            if ((edge2Tri.find(edge) != edge2Tri.end()) &&
                (edge2Tri.find(std::pair<int, int>(nbVI, vI)) !=
                 edge2Tri.end())) {
                // interior edge

                Eigen::MatrixXd newVertPosI;
                const double SDDec = queryLocalEdDec_bSplit(edge, newVertPosI);

                double seInc =
                    (V_rest.row(vI) - V_rest.row(nbVI)).norm() / virtualRadius *
                    (vertWeight[vI] + vertWeight[nbVI]) / 2.0 *
                    creaseRelief(vI, nbVI);
                if (atCutTip) {
                    const Eigen::RowVector3d extendDir =
                        (V_rest.row(nbVI) - V_rest.row(vI)).normalized();
                    seInc *= 1.0 + turnPenalty * 0.5 *
                                       (1.0 - arrivalDir.dot(extendDir));
                }
                const double curEwDec =
                    (1.0 - lambda_t) * SDDec - lambda_t * seInc;
                if (curEwDec > maxEwDec) {
                    maxEwDec = curEwDec;
                    path_max[0] = vI;
                    path_max[1] = nbVI;
                    newVertPos_max = newVertPosI;
                    energyChanges_max.first = -SDDec;
                    energyChanges_max.second = seInc;
                }
            }
        }
        return maxEwDec;
    } else {
        // interior split
        for (const auto &nbVI : vNeighbor[vI]) {
            if (isBoundaryVert(nbVI)) {
                energyChanges_max.first = DBL_MAX;
                energyChanges_max.second = DBL_MAX;
                assert(0 && "should have prevented this case outside");
                return -DBL_MAX; // don't split vertices connected to boundary
                                 // here
            }
        }

        if (umbrella.size() > 10) {
            // DISABLE std::cout << "large degree vert, " << umbrella.size() <<
            // " incident tris" << std::endl; DISABLE logFile << "large degree
            // vert, " << umbrella.size() << " incident tris" << std::endl;
        }

        path_max.resize(3);
        double EwDec_max = -DBL_MAX;
        energyChanges_max.first = DBL_MAX;
        energyChanges_max.second = DBL_MAX;
        std::set<int> freeVert;
        freeVert.insert(vI);
        std::map<int, Eigen::RowVector2d> newVertPosMap;
        std::vector<int> path(3);
        path[1] = vI;
        for (int startI = 0; startI + 1 < umbrella.size(); startI++) {
            for (int i = 0; i < 3; i++) {
                if (F(umbrella[startI], i) == vI) {
                    path[0] = F(umbrella[startI], (i + 1) % 3);
                    break;
                }
            }

            for (int endI = startI + 1; endI < umbrella.size(); endI++) {
                for (int i = 0; i < 3; i++) {
                    if (F(umbrella[endI], i) == vI) {
                        path[2] = F(umbrella[endI], (i + 1) % 3);
                        break;
                    }
                }

                double SDDec = 0.0;
                Eigen::MatrixXd newVertPos;

                SDDec += computeLocalEdDec_inSplit(umbrella, freeVert, path,
                                                   newVertPos);
                // TODO: share local mesh before split, also for boundary splits

                const double legLen0 =
                    (V_rest.row(path[0]) - V_rest.row(path[1])).norm() *
                    (vertWeight[path[0]] + vertWeight[path[1]]);
                const double legLen1 =
                    (V_rest.row(path[1]) - V_rest.row(path[2])).norm() *
                    (vertWeight[path[1]] + vertWeight[path[2]]);
                // a bent starting path seeds a zigzag
                const Eigen::RowVector3d legDir0 =
                    (V_rest.row(path[1]) - V_rest.row(path[0])).normalized();
                const Eigen::RowVector3d legDir1 =
                    (V_rest.row(path[2]) - V_rest.row(path[1])).normalized();
                const double turnFactor =
                    1.0 + turnPenalty * 0.5 * (1.0 - legDir0.dot(legDir1));
                const double seInc =
                    (legLen0 * creaseRelief(path[0], path[1]) +
                     legLen1 * creaseRelief(path[1], path[2])) /
                    virtualRadius / 2.0 * turnFactor;
                const double EwDec =
                    (1.0 - lambda_t) * SDDec - lambda_t * seInc;
                if (EwDec > EwDec_max) {
                    EwDec_max = EwDec;
                    newVertPos_max = newVertPos;
                    path_max = path;
                    energyChanges_max.first = -SDDec;
                    energyChanges_max.second = seInc;
                }
            }
        }
        return EwDec_max;
    }
}

double TriMesh::computeLocalEdDec_inSplit(const std::vector<int> &triangles,
                                          const std::set<int> &freeVert,
                                          const std::vector<int> &path,
                                          Eigen::MatrixXd &newVertPos,
                                          int maxIter) const {
    assert(triangles.size() && freeVert.size());

    // construct local mesh
    Eigen::MatrixXi localF;
    localF.resize(triangles.size(), 3);
    Eigen::MatrixXd localV_rest, localV;
    std::set<int> fixedVert;
    std::map<int, int> globalVI2local;
    int localTriI = 0;
    for (const auto triI : triangles) {
        for (int vI = 0; vI < 3; vI++) {
            int globalVI = F(triI, vI);
            auto localVIFinder = globalVI2local.find(globalVI);
            if (localVIFinder == globalVI2local.end()) {
                int localVI = static_cast<int>(localV_rest.rows());
                if (freeVert.find(globalVI) == freeVert.end()) {
                    fixedVert.insert(localVI);
                }
                localV_rest.conservativeResize(localVI + 1, 3);
                localV_rest.row(localVI) = V_rest.row(globalVI);
                localV.conservativeResize(localVI + 1, 2);
                localV.row(localVI) = V.row(globalVI);
                localF(localTriI, vI) = localVI;
                globalVI2local[globalVI] = localVI;
            } else {
                localF(localTriI, vI) = localVIFinder->second;
            }
        }
        localTriI++;
    }
    TriMesh localMesh(localV_rest, localF, localV, Eigen::MatrixXi(), false);
    localMesh.resetFixedVert(fixedVert);
    // keep the local energy estimate consistent with the weighted global one
    for (int fwI = 0; fwI < static_cast<int>(triangles.size()); fwI++)
        localMesh.faceWeight[fwI] = faceWeight[triangles[fwI]];

    SymDirichletEnergy SD;
    double initE = 0.0;
    for (const auto &triI : triangles) {
        double energyValI;
        SD.getEnergyValByElemID(*this, triI, energyValI);
        initE += energyValI;
    }
    initE *= surfaceArea / localMesh.surfaceArea;

    // convert split path global index to local index
    std::vector<int> path_local;
    path_local.reserve(path.size());
    for (const auto &pvI : path) {
        const auto finder = globalVI2local.find(pvI);
        assert(finder != globalVI2local.end());
        path_local.emplace_back(finder->second);
    }
    // split
    localMesh.cutPath(path_local, true, 0, Eigen::MatrixXd(), false);

    bool isBijective = !!scaffold;

    // construct air mesh
    Eigen::MatrixXd UV_bnds;
    Eigen::MatrixXi E;
    Eigen::VectorXi bnd;
    if (isBijective) {
        // separate vertex
        // TODO: write into a function
        Eigen::RowVector2d splittedV[2] = {
            localMesh.V.row(path_local[1]),
            localMesh.V.row(localMesh.V.rows() - 1)};
        Eigen::RowVector2d sepDir_oneV[2];
        localMesh.compute2DInwardNormal(path_local[1], sepDir_oneV[0]);
        localMesh.compute2DInwardNormal(localMesh.V.rows() - 1, sepDir_oneV[1]);
        Eigen::VectorXd sepDir[2] = {
            Eigen::VectorXd::Zero(localMesh.V.rows() * 2),
            Eigen::VectorXd::Zero(localMesh.V.rows() * 2)};
        sepDir[0].block(path_local[1] * 2, 0, 2, 1) =
            sepDir_oneV[0].transpose();
        sepDir[1].block((localMesh.V.rows() - 1) * 2, 0, 2, 1) =
            sepDir_oneV[1].transpose();
        const double eps_sep =
            (V.row(path[1]) - V.row(path[0])).squaredNorm() * 1.0e-4;
        double curSqDist = (splittedV[0] - splittedV[1]).squaredNorm();
        // the step size can collapse to zero with coincident copies, and the
        // relative-progress break below is NaN when lastSqDist is zero, so cap
        int sepIter = 0;
        while (curSqDist < eps_sep && ++sepIter <= 100) {
            for (int i = 0; i < 2; i++) {
                double stepSize_sep = 1.0;
                SD.initStepSize(localMesh, sepDir[i], stepSize_sep);
                splittedV[i] += 0.1 * stepSize_sep * sepDir_oneV[i];
            }
            localMesh.V.row(path_local[1]) = splittedV[0];
            localMesh.V.row(localMesh.V.rows() - 1) = splittedV[1];

            double lastSqDist = curSqDist;
            curSqDist = (splittedV[0] - splittedV[1]).squaredNorm();
            if (std::abs(curSqDist - lastSqDist) / lastSqDist < 1.0e-3) {
                break;
            }
            //                    // may update search dir, and accelerate
            //                    localMesh.compute2DInwardNormal(splitPath_local[0],
            //                    sepDir_oneV[0]);
            //                    localMesh.compute2DInwardNormal(localMesh.V.rows()
            //                    - 1, sepDir_oneV[1]);
            //                    sepDir[0].block(splitPath_local[0] * 2, 0, 2,
            //                    1) = sepDir_oneV[0].transpose();
            //                    sepDir[1].bottomRows(2) =
            //                    sepDir_oneV[1].transpose();
        }
        if (curSqDist < eps_sep * 1.0e-6) {
            // the copies would not separate, a degenerate air mesh here
            // crashes the scaffold, reject the candidate instead
            return -DBL_MAX;
        }

        // establish air mesh information
        UV_bnds.resize(4, 2);
        E.resize(4, 2);
        E << 0, 1, 1, 2, 2, 3, 3, 0;
        bnd.resize(4);
        bnd << path_local[2], path_local[1], path_local[0],
            static_cast<int>(localMesh.V.rows()) - 1;
    }

    // the optimizer throws on a flat or folded start
    if (!localMesh.checkInversion(true))
        return -DBL_MAX;

    // conduct optimization on local mesh
    std::vector<uvgami::Energy *> energyTerms(1, &SD);
    std::vector<double> energyParams(1, 1.0);
    Optimizer optimizer(localMesh, energyTerms, energyParams, 0, true,
                        isBijective, UV_bnds, E, bnd, true);
    optimizer.precompute();
    optimizer.setRelGL2Tol(1.0e-6);
    optimizer.solve(maxIter); // do not output, the other part
    double curE;
    optimizer.computeEnergyVal(optimizer.getResult(), optimizer.getScaffold(),
                               curE, true);
    const double eDec = (initE - curE) * localMesh.surfaceArea / surfaceArea;

    // get new vertex positions
    newVertPos.resize(2, 2);
    newVertPos.row(0) = optimizer.getResult().V.bottomRows(1);
    newVertPos.row(1) = optimizer.getResult().V.row(path_local[1]);

    return eDec;
}

double TriMesh::computeLocalEdDec_merge(
    const std::vector<int> &path, const std::vector<int> &triangles,
    const std::set<int> &freeVert,
    std::map<int, Eigen::RowVector2d> &newVertPos,
    const std::map<int, int> &mergeVert,
    const Eigen::RowVector2d &initMergedPos, bool closeup, int maxIter) const {
    assert(triangles.size() && freeVert.size());
    assert(!mergeVert.empty());

    bool isBijective = ((!!scaffold) && (!closeup));

    // construct local mesh
    Eigen::MatrixXi localF;
    localF.resize(triangles.size(), 3);
    Eigen::MatrixXd localV_rest, localV;
    std::set<int> fixedVert;
    std::map<int, int> globalVI2local;
    int localTriI = 0;
    for (const auto triI : triangles) {
        for (int vI = 0; vI < 3; vI++) {
            int globalVI = F(triI, vI);
            auto mergeFinder = mergeVert.find(globalVI);
            if (mergeFinder == mergeVert.end()) {
                // normal vertices
                auto localVIFinder = globalVI2local.find(globalVI);
                if (localVIFinder == globalVI2local.end()) {
                    int localVI = static_cast<int>(localV_rest.rows());
                    if (freeVert.find(globalVI) == freeVert.end())
                        fixedVert.insert(localVI);
                    localV_rest.conservativeResize(localVI + 1, 3);
                    localV_rest.row(localVI) = V_rest.row(globalVI);
                    localV.conservativeResize(localVI + 1, 2);
                    localV.row(localVI) = V.row(globalVI);
                    localF(localTriI, vI) = localVI;
                    globalVI2local[globalVI] = localVI;
                } else {
                    localF(localTriI, vI) = localVIFinder->second;
                }
            } else {
                // one of the vertices to be merged
                auto localVIFinder = globalVI2local.find(globalVI);
                auto localVIFinder_mergePair =
                    globalVI2local.find(mergeFinder->second);
                bool selfAdded = (localVIFinder != globalVI2local.end());
                bool mergePairAdded =
                    (localVIFinder_mergePair != globalVI2local.end());
                if (selfAdded) {
                    assert(mergePairAdded);
                    localF(localTriI, vI) = localVIFinder->second;
                } else {
                    assert(!mergePairAdded);
                    int localVI = static_cast<int>(localV_rest.rows());
                    if (freeVert.find(globalVI) == freeVert.end())
                        fixedVert.insert(localVI);
                    localV_rest.conservativeResize(localVI + 1, 3);
                    localV_rest.row(localVI) = V_rest.row(globalVI);
                    localV.conservativeResize(localVI + 1, 2);
                    localV.row(localVI) = initMergedPos;
                    localF(localTriI, vI) = localVI;
                    globalVI2local[globalVI] = localVI;

                    globalVI2local[mergeFinder->second] = localVI;
                }
            }
        }
        localTriI++;
    }
    TriMesh localMesh(localV_rest, localF, localV, Eigen::MatrixXi(), false);
    localMesh.resetFixedVert(fixedVert);
    // keep the local energy estimate consistent with the weighted global one
    for (int fwI = 0; fwI < static_cast<int>(triangles.size()); fwI++)
        localMesh.faceWeight[fwI] = faceWeight[triangles[fwI]];

    SymDirichletEnergy SD;
    double initE = 0.0;
    for (const auto &triI : triangles) {
        double energyValI;
        SD.getEnergyValByElemID(*this, triI, energyValI);
        initE += energyValI;
    }
    initE *= surfaceArea / localMesh.surfaceArea;

    // construct air mesh
    Eigen::MatrixXd UV_bnds;
    Eigen::MatrixXi E;
    Eigen::VectorXi bnd;
    if (isBijective) {
        if (!scaffold->getCornerAirLoop(path, initMergedPos, UV_bnds, E, bnd)) {
            // if initPos causes the composite loop to self-intersect, or the
            // loop is totally inverted (potentially violating bijectivity),
            // abandon this query
            return -DBL_MAX;
        }

        for (int bndI = 0; bndI < bnd.size(); bndI++) {
            const auto finder = globalVI2local.find(bnd[bndI]);
            assert(finder != globalVI2local.end());
            bnd[bndI] = finder->second;
        }
    }

    // the optimizer throws on a flat or folded start
    if (!localMesh.checkInversion(true))
        return -DBL_MAX;

    // conduct optimization on local mesh
    std::vector<uvgami::Energy *> energyTerms(1, &SD);
    std::vector<double> energyParams(1, 1.0);
    Optimizer optimizer(localMesh, energyTerms, energyParams, 0, true,
                        isBijective, UV_bnds, E, bnd, true);
    optimizer.precompute();
    optimizer.setRelGL2Tol(1.0e-6);
    optimizer.solve(maxIter); // do not output, the other part
    double curE;
    optimizer.computeEnergyVal(optimizer.getResult(), optimizer.getScaffold(),
                               curE, true);
    const double eDec = (initE - curE) * localMesh.surfaceArea / surfaceArea;

    // get new vertex positions
    newVertPos.clear();
    for (const auto &vI_free : freeVert)
        newVertPos[vI_free] =
            optimizer.getResult().V.row(globalVI2local[vI_free]);

    return eDec;
}

double TriMesh::computeLocalEdDec_bSplit(const std::vector<int> &triangles,
                                         const std::set<int> &freeVert,
                                         const std::vector<int> &splitPath,
                                         Eigen::MatrixXd &newVertPos,
                                         int maxIter) const {
    assert(triangles.size() && freeVert.size());

    // construct local mesh
    Eigen::MatrixXi localF;
    localF.resize(triangles.size(), 3);
    Eigen::MatrixXd localV_rest, localV;
    std::set<int> fixedVert;
    std::map<int, int> globalVI2local;
    int localTriI = 0;
    for (const auto triI : triangles) {
        for (int vI = 0; vI < 3; vI++) {
            int globalVI = F(triI, vI);
            auto localVIFinder = globalVI2local.find(globalVI);
            if (localVIFinder == globalVI2local.end()) {
                int localVI = static_cast<int>(localV_rest.rows());
                if (freeVert.find(globalVI) == freeVert.end()) {
                    fixedVert.insert(localVI);
                }
                localV_rest.conservativeResize(localVI + 1, 3);
                localV_rest.row(localVI) = V_rest.row(globalVI);
                localV.conservativeResize(localVI + 1, 2);
                localV.row(localVI) = V.row(globalVI);
                localF(localTriI, vI) = localVI;
                globalVI2local[globalVI] = localVI;
            } else {
                localF(localTriI, vI) = localVIFinder->second;
            }
        }
        localTriI++;
    }
    TriMesh localMesh(localV_rest, localF, localV, Eigen::MatrixXi(), false);
    localMesh.resetFixedVert(fixedVert);
    // keep the local energy estimate consistent with the weighted global one
    for (int fwI = 0; fwI < static_cast<int>(triangles.size()); fwI++)
        localMesh.faceWeight[fwI] = faceWeight[triangles[fwI]];

    // compute initial symmetric Dirichlet Energy value
    SymDirichletEnergy SD;
    double initE = 0.0;
    for (const auto &triI : triangles) {
        double energyValI;
        SD.getEnergyValByElemID(*this, triI, energyValI);
        initE += energyValI;
    }
    initE *= surfaceArea / localMesh.surfaceArea;

    // split edge
    Eigen::MatrixXd UV_bnds;
    Eigen::MatrixXi E;
    Eigen::VectorXi bnd;
    bool cutThrough = false;
    switch (splitPath.size()) {
    case 0: // nothing to split
        assert(0 && "currently we don't use this function without splitting!");
        break;

    case 2: { // boundary split
        assert(freeVert.find(splitPath[0]) != freeVert.end());
        if (freeVert.find(splitPath[1]) != freeVert.end()) {
            cutThrough = true;
        }

        // convert splitPath global index to local index
        std::vector<int> splitPath_local;
        splitPath_local.reserve(splitPath.size());
        for (const auto &pvI : splitPath) {
            const auto finder = globalVI2local.find(pvI);
            assert(finder != globalVI2local.end());
            splitPath_local.emplace_back(finder->second);
        }

        // split
        localMesh.splitEdgeOnBoundary(
            std::pair<int, int>(splitPath_local[0], splitPath_local[1]),
            Eigen::Matrix2d(), false, cutThrough);

        if (scaffold) {
            // separate the split vertices to leave room for airmesh
            Eigen::RowVector2d splittedV[2] = {
                localMesh.V.row(splitPath_local[0]),
                localMesh.V.row(localMesh.V.rows() - 1 - cutThrough)};
            Eigen::RowVector2d sepDir_oneV[2];
            localMesh.compute2DInwardNormal(splitPath_local[0], sepDir_oneV[0]);
            localMesh.compute2DInwardNormal(localMesh.V.rows() - 1 - cutThrough,
                                            sepDir_oneV[1]);
            Eigen::VectorXd sepDir[2] = {
                Eigen::VectorXd::Zero(localMesh.V.rows() * 2),
                Eigen::VectorXd::Zero(localMesh.V.rows() * 2)};
            sepDir[0].block(splitPath_local[0] * 2, 0, 2, 1) =
                sepDir_oneV[0].transpose();
            sepDir[1].block((localMesh.V.rows() - 1 - cutThrough) * 2, 0, 2,
                            1) = sepDir_oneV[1].transpose();
            const double eps_sep =
                (V.row(splitPath[1]) - V.row(splitPath[0])).squaredNorm() *
                1.0e-4;
            double curSqDist = (splittedV[0] - splittedV[1]).squaredNorm();
            // capped for the same zero-step / NaN-break reason as in
            // computeLocalEdDec_inSplit
            int sepIter = 0;
            while (curSqDist < eps_sep && ++sepIter <= 100) {
                for (int i = 0; i < 2; i++) {
                    double stepSize_sep = 1.0;
                    SD.initStepSize(localMesh, sepDir[i], stepSize_sep);
                    splittedV[i] += 0.1 * stepSize_sep * sepDir_oneV[i];
                }
                localMesh.V.row(splitPath_local[0]) = splittedV[0];
                localMesh.V.row(localMesh.V.rows() - 1 - cutThrough) =
                    splittedV[1];

                double lastSqDist = curSqDist;
                curSqDist = (splittedV[0] - splittedV[1]).squaredNorm();
                if (std::abs(curSqDist - lastSqDist) / lastSqDist < 1.0e-3) {
                    break;
                }
                //                    // may update search dir, and accelerate
                //                    localMesh.compute2DInwardNormal(splitPath_local[0],
                //                    sepDir_oneV[0]);
                //                    localMesh.compute2DInwardNormal(localMesh.V.rows()
                //                    - 1, sepDir_oneV[1]);
                //                    sepDir[0].block(splitPath_local[0] * 2, 0,
                //                    2, 1) = sepDir_oneV[0].transpose();
                //                    sepDir[1].bottomRows(2) =
                //                    sepDir_oneV[1].transpose();
            }
            assert(localMesh.checkInversion());

            if (cutThrough) {
                splittedV[0] = localMesh.V.row(splitPath_local[1]);
                splittedV[1] = localMesh.V.bottomRows(1);
                localMesh.compute2DInwardNormal(splitPath_local[1],
                                                sepDir_oneV[0]);
                localMesh.compute2DInwardNormal(localMesh.V.rows() - 1,
                                                sepDir_oneV[1]);
                sepDir[0] = sepDir[1] =
                    Eigen::VectorXd::Zero(localMesh.V.rows() * 2);
                sepDir[0].block(splitPath_local[1] * 2, 0, 2, 1) =
                    sepDir_oneV[0].transpose();
                sepDir[1].bottomRows(2) = sepDir_oneV[1].transpose();
                double curSqDist = (splittedV[0] - splittedV[1]).squaredNorm();
                // capped for the same zero-step / NaN-break reason as above
                int sepIter2 = 0;
                while (curSqDist < eps_sep && ++sepIter2 <= 100) {
                    for (int i = 0; i < 2; i++) {
                        double stepSize_sep = 1.0;
                        SD.initStepSize(localMesh, sepDir[i], stepSize_sep);
                        splittedV[i] += 0.1 * stepSize_sep * sepDir_oneV[i];
                    }
                    localMesh.V.row(splitPath_local[1]) = splittedV[0];
                    localMesh.V.bottomRows(1) = splittedV[1];

                    double lastSqDist = curSqDist;
                    curSqDist = (splittedV[0] - splittedV[1]).squaredNorm();
                    if (std::abs(curSqDist - lastSqDist) / lastSqDist <
                        1.0e-3) {
                        break;
                    }
                    // may update search dir, and accelerate
                }
                assert(localMesh.checkInversion());
            }

            // prepare local air mesh boundary
            Eigen::MatrixXd UV_temp;
            Eigen::VectorXi bnd_temp;
            std::set<int> loop_AMVI;
            if (!scaffold->get1RingAirLoop(splitPath[0], UV_temp, E,
                                           bnd_temp, loop_AMVI))
                return -DBL_MAX;
            int loopVAmt_beforeSplit = E.rows();
            if (!cutThrough) {
                E.bottomRows(1) << loopVAmt_beforeSplit - 1,
                    loopVAmt_beforeSplit;
                E.conservativeResize(loopVAmt_beforeSplit + 2, 2);
                E.bottomRows(2) << loopVAmt_beforeSplit,
                    loopVAmt_beforeSplit + 1, loopVAmt_beforeSplit + 1, 0;

                UV_bnds.resize(loopVAmt_beforeSplit + 2, 2);
                UV_bnds.bottomRows(loopVAmt_beforeSplit - 3) =
                    UV_temp.bottomRows(loopVAmt_beforeSplit - 3);
                // NOTE: former vertices will be filled with mesh coordinates
                // while constructing the local air mesh

                bnd.resize(bnd_temp.size() + 2);
                bnd[0] = bnd_temp[0];
                bnd[1] = localMesh.V.rows() - 1 - cutThrough;
                bnd[2] = splitPath[1];
                bnd.bottomRows(2) = bnd_temp.bottomRows(2);
                for (int bndI = 0; bndI < bnd.size(); bndI++) {
                    if (bndI != 1) {
                        const auto finder = globalVI2local.find(bnd[bndI]);
                        assert(finder != globalVI2local.end());
                        bnd[bndI] = finder->second;
                    }
                }
            } else {
                Eigen::MatrixXd UV_temp1;
                Eigen::VectorXi bnd_temp1;
                Eigen::MatrixXi E1;
                std::set<int> loop1_AMVI;
                if (!scaffold->get1RingAirLoop(splitPath[1], UV_temp1, E1,
                                               bnd_temp1, loop1_AMVI))
                    return -DBL_MAX;
                // avoid generating air mesh with duplicated vertices
                // NOTE: this also avoid forming tiny charts
                for (const auto &i : loop1_AMVI) {
                    if (loop_AMVI.find(i) != loop_AMVI.end()) {
                        return -DBL_MAX;
                    }
                }
                int loopVAmt1_beforeSplit = E1.rows();

                UV_bnds.resize(loopVAmt_beforeSplit + loopVAmt1_beforeSplit + 2,
                               2);
                UV_bnds.bottomRows(UV_bnds.rows() - 8)
                    << UV_temp1.bottomRows(loopVAmt1_beforeSplit - 3),
                    UV_temp.bottomRows(loopVAmt_beforeSplit - 3);
                // NOTE: former vertices will be filled with mesh coordinates
                // while constructing the local air mesh

                bnd.resize(8);
                bnd[0] = bnd_temp[0];
                bnd[1] = localMesh.V.rows() - 2;
                bnd[2] = splitPath[1];
                bnd[3] = bnd_temp1[2];
                bnd[4] = bnd_temp1[0];
                bnd[5] = localMesh.V.rows() - 1;
                bnd[6] = splitPath[0];
                bnd[7] = bnd_temp[2];
                for (int bndI = 0; bndI < bnd.size(); bndI++) {
                    if ((bndI != 1) && (bndI != 5)) {
                        const auto finder = globalVI2local.find(bnd[bndI]);
                        assert(finder != globalVI2local.end());
                        bnd[bndI] = finder->second;
                    }
                }

                E.resize(UV_bnds.rows(), 2);
                E.row(0) << 0, 1;
                E.row(1) << 1, 2;
                E.row(2) << 2, 3;
                if (loopVAmt1_beforeSplit - 3 == 0) {
                    E.row(3) << 3, 4;
                } else {
                    E.row(3) << 3, 8;
                    for (int i = 0; i < loopVAmt1_beforeSplit - 3; i++) {
                        E.row(4 + i) << 8 + i, 9 + i;
                    }
                    E(loopVAmt1_beforeSplit, 1) = 4;
                }
                E.row(loopVAmt1_beforeSplit + 1) << 4, 5;
                E.row(loopVAmt1_beforeSplit + 2) << 5, 6;
                E.row(loopVAmt1_beforeSplit + 3) << 6, 7;
                if (loopVAmt_beforeSplit - 3 == 0) {
                    E.row(loopVAmt1_beforeSplit + 4) << 7, 0;
                } else {
                    E.row(loopVAmt1_beforeSplit + 4) << 7,
                        loopVAmt1_beforeSplit + 5;
                    for (int i = 0; i < loopVAmt_beforeSplit - 3; i++) {
                        E.row(loopVAmt1_beforeSplit + 5 + i)
                            << loopVAmt1_beforeSplit + 5 + i,
                            loopVAmt1_beforeSplit + 6 + i;
                    }
                    E(loopVAmt1_beforeSplit + loopVAmt_beforeSplit + 1, 1) = 0;
                }
            }
        }

        break;
    }

    case 3: // interior split
        // not processed here
        break;

    default:
        assert(0 && "invalid split path!");
        break;
    }

    // the optimizer throws on a flat or folded start
    if (!localMesh.checkInversion(true))
        return -DBL_MAX;

    // conduct optimization on local mesh
    std::vector<uvgami::Energy *> energyTerms(1, &SD);
    std::vector<double> energyParams(1, 1.0);
    Optimizer optimizer(localMesh, energyTerms, energyParams, 0, true,
                        !!scaffold, UV_bnds, E, bnd, true);
    optimizer.precompute();
    optimizer.setRelGL2Tol(1.0e-6);
    optimizer.solve(maxIter);
    double curE;
    optimizer.computeEnergyVal(optimizer.getResult(), optimizer.getScaffold(),
                               curE, true);
    const double eDec = (initE - curE) * localMesh.surfaceArea / surfaceArea;

    // get new vertex positions
    newVertPos.resize(2, 2);
    newVertPos << optimizer.getResult().V.row(globalVI2local[splitPath[0]]),
        optimizer.getResult().V.row(localMesh.V.rows() - 1 - cutThrough);
    if (cutThrough) {
        newVertPos.conservativeResize(4, 2);
        newVertPos.row(2) = optimizer.getResult().V.bottomRows(1);
        newVertPos.row(3) =
            optimizer.getResult().V.row(globalVI2local[splitPath[1]]);
    }

    return eDec;
}

double TriMesh::queryLocalEdDec_bSplit(const std::pair<int, int> &edge,
                                       Eigen::MatrixXd &newVertPos) const {
    assert(vNeighbor.size() == V.rows());
    auto edgeTriIndFinder = edge2Tri.find(edge);
    auto edgeTriIndFinder_dual =
        edge2Tri.find(std::pair<int, int>(edge.second, edge.first));
    assert(edgeTriIndFinder != edge2Tri.end());
    assert(edgeTriIndFinder_dual != edge2Tri.end());

    int vI_boundary = edge.first, vI_interior = edge.second;
    bool cutThrough = false;
    if (isBoundaryVert(edge.first)) {
        if (isBoundaryVert(edge.second))
            cutThrough = true;
    } else {
        assert(isBoundaryVert(edge.second) &&
               "Input edge must attach mesh boundary!");

        vI_boundary = edge.second;
        vI_interior = edge.first;
    }

    if (cutThrough)
        newVertPos.resize(4, 2);
    else
        newVertPos.resize(2, 2);

    std::set<int> freeVertGID;
    freeVertGID.insert(vI_boundary);
    if (cutThrough)
        freeVertGID.insert(vI_interior);

    std::vector<int> tri_toSep, tri_toSep1;
    std::pair<int, int> boundaryEdge;
    isBoundaryVert(vI_boundary, vI_interior, tri_toSep, boundaryEdge, 0);
    assert(!tri_toSep.empty());
    isBoundaryVert(vI_boundary, vI_interior, tri_toSep1, boundaryEdge, 1);
    assert(!tri_toSep1.empty());
    tri_toSep.insert(tri_toSep.end(), tri_toSep1.begin(), tri_toSep1.end());
    if (cutThrough) {
        for (int clockwise = 0; clockwise < 2; clockwise++) {
            std::vector<int> tri_interior;
            std::pair<int, int> boundaryEdge_interior;
            isBoundaryVert(vI_interior, vI_boundary, tri_interior,
                           boundaryEdge_interior, clockwise);
            for (const auto &triI : tri_interior) {
                bool newTri = true;
                for (const auto &triI_b : tri_toSep) {
                    if (triI_b == triI) {
                        newTri = false;
                        break;
                    }
                }
                if (newTri)
                    tri_toSep.emplace_back(triI);
            }
        }
    }

    std::vector<int> splitPath(2);
    splitPath[0] = vI_boundary;
    splitPath[1] = vI_interior;
    return computeLocalEdDec_bSplit(tri_toSep, freeVertGID, splitPath,
                                    newVertPos);
}

bool TriMesh::queriedOpFits(int opType, const std::vector<int> &path,
                            const Eigen::MatrixXd &newVertPos) const {
    for (const auto &vI : path)
        if (vI < 0 || vI >= V.rows())
            return false;
    auto hasEdge = [this](int a, int b) {
        return edge2Tri.find(std::pair<int, int>(a, b)) != edge2Tri.end();
    };
    switch (opType) {
    case 0: {
        if (path.size() != 2 || !hasEdge(path[0], path[1]) ||
            !hasEdge(path[1], path[0]))
            return false;
        const int boundaryEnds = isBoundaryVert(path[0]) + isBoundaryVert(path[1]);
        // one boundary end is a plain split, two is a cut-through
        return boundaryEnds == (newVertPos.rows() == 4 ? 2 : 1);
    }
    case 1:
        return path.size() == 3 && hasEdge(path[0], path[1]) &&
               hasEdge(path[1], path[0]) && hasEdge(path[1], path[2]) &&
               hasEdge(path[2], path[1]) && !isBoundaryVert(path[0]) &&
               !isBoundaryVert(path[1]) && !isBoundaryVert(path[2]);
    case 2:
        return path.size() == 3 && hasEdge(path[0], path[1]) &&
               !hasEdge(path[1], path[0]) && hasEdge(path[1], path[2]) &&
               !hasEdge(path[2], path[1]);
    default:
        return false;
    }
}

void TriMesh::splitEdgeOnBoundary(const std::pair<int, int> &edge,
                                  const Eigen::MatrixXd &newVertPos,
                                  bool changeVertPos, bool allowCutThrough) {
    assert(vNeighbor.size() == V.rows());
    auto edgeTriIndFinder = edge2Tri.find(edge);
    auto edgeTriIndFinder_dual =
        edge2Tri.find(std::pair<int, int>(edge.second, edge.first));
    assert(edgeTriIndFinder != edge2Tri.end());
    assert(edgeTriIndFinder_dual != edge2Tri.end());

    bool duplicateBoth = false;
    int vI_boundary = edge.first, vI_interior = edge.second;
    if (isBoundaryVert(edge.first)) {
        if (allowCutThrough && isBoundaryVert(edge.second)) {
            if (changeVertPos)
                assert(newVertPos.rows() == 4);
            duplicateBoth = true;
        }
    } else {
        assert(isBoundaryVert(edge.second) &&
               "Input edge must attach mesh boundary!");

        vI_boundary = edge.second;
        vI_interior = edge.first;
    }

    fracTail.erase(vI_boundary);
    if (!duplicateBoth) {
        fracTail.insert(vI_interior);
        curFracTail = vI_interior;
    } else {
        curFracTail = -1;
    }
    curInteriorFracTails.first = curInteriorFracTails.second = -1;

    // duplicate vI_boundary
    std::vector<int> tri_toSep[2];
    std::pair<int, int> boundaryEdge[2];
    for (int toBound = 0; toBound < 2; toBound++) { //!!! why?
        isBoundaryVert(vI_boundary, vI_interior, tri_toSep[1], boundaryEdge[1],
                       toBound);
        assert(!tri_toSep[1].empty());
    }
    if (duplicateBoth) {
        isBoundaryVert(vI_interior, vI_boundary, tri_toSep[0], boundaryEdge[0],
                       true);
        assert(!tri_toSep[0].empty());
    }

    int nV = static_cast<int>(V_rest.rows());
    V_rest.conservativeResize(nV + 1, 3);
    V_rest.row(nV) = V_rest.row(vI_boundary);
    vertWeight.conservativeResize(nV + 1);
    vertWeight[nV] = vertWeight[vI_boundary];
    V.conservativeResize(nV + 1, 2);
    if (changeVertPos) {
        V.row(nV) = newVertPos.block(1, 0, 1, 2);
        V.row(vI_boundary) = newVertPos.block(0, 0, 1, 2);
    } else {
        V.row(nV) = V.row(vI_boundary);
    }

    for (const auto triI : tri_toSep[1]) {
        for (int vI = 0; vI < 3; vI++) {
            if (F(triI, vI) == vI_boundary) {
                // update triangle vertInd, edge2Tri and vNeighbor
                int vI_post = F(triI, (vI + 1) % 3);
                int vI_pre = F(triI, (vI + 2) % 3);

                F(triI, vI) = nV;

                edge2Tri.erase(std::pair<int, int>(vI_boundary, vI_post));
                edge2Tri[std::pair<int, int>(nV, vI_post)] = triI;
                edge2Tri.erase(std::pair<int, int>(vI_pre, vI_boundary));
                edge2Tri[std::pair<int, int>(vI_pre, nV)] = triI;

                vNeighbor[vI_pre].erase(vI_boundary);
                vNeighbor[vI_pre].insert(nV);
                vNeighbor[vI_post].erase(vI_boundary);
                vNeighbor[vI_post].insert(nV);
                vNeighbor[vI_boundary].erase(vI_pre);
                vNeighbor[vI_boundary].erase(vI_post);
                vNeighbor.resize(nV + 1);
                vNeighbor[nV].insert(vI_pre);
                vNeighbor[nV].insert(vI_post);

                break;
            }
        }
    }
    vNeighbor[vI_boundary].insert(vI_interior);
    vNeighbor[vI_interior].insert(vI_boundary);

    // add cohesive edge pair and update cohEIndex
    const int nCE = static_cast<int>(cohE.rows());
    cohE.conservativeResize(nCE + 1, 4);
    cohE.row(nCE) << vI_interior, nV, vI_interior, vI_boundary;
    cohEIndex[std::pair<int, int>(vI_interior, nV)] = nCE;
    cohEIndex[std::pair<int, int>(vI_boundary, vI_interior)] = -nCE - 1;
    auto CEIfinder = cohEIndex.find(boundaryEdge[1]);
    if (CEIfinder != cohEIndex.end()) {
        if (CEIfinder->second >= 0)
            cohE(CEIfinder->second, 0) = nV;
        else
            cohE(-CEIfinder->second - 1, 3) = nV;
        cohEIndex[std::pair<int, int>(nV, boundaryEdge[1].second)] =
            CEIfinder->second;
        cohEIndex.erase(CEIfinder);
    }

    if (duplicateBoth) {
        int nV = static_cast<int>(V_rest.rows());
        V_rest.conservativeResize(nV + 1, 3);
        V_rest.row(nV) = V_rest.row(vI_interior);
        vertWeight.conservativeResize(nV + 1);
        vertWeight[nV] = vertWeight[vI_interior];
        V.conservativeResize(nV + 1, 2);
        if (changeVertPos) {
            V.row(nV) = newVertPos.block(2, 0, 1, 2);
            V.row(vI_interior) = newVertPos.block(3, 0, 1, 2);
        } else {
            V.row(nV) = V.row(vI_interior);
        }

        for (const auto triI : tri_toSep[0]) {
            for (int vI = 0; vI < 3; vI++) {
                if (F(triI, vI) == vI_interior) {
                    // update triangle vertInd, edge2Tri and vNeighbor
                    int vI_post = F(triI, (vI + 1) % 3);
                    int vI_pre = F(triI, (vI + 2) % 3);

                    F(triI, vI) = nV;

                    edge2Tri.erase(std::pair<int, int>(vI_interior, vI_post));
                    edge2Tri[std::pair<int, int>(nV, vI_post)] = triI;
                    edge2Tri.erase(std::pair<int, int>(vI_pre, vI_interior));
                    edge2Tri[std::pair<int, int>(vI_pre, nV)] = triI;

                    vNeighbor[vI_pre].erase(vI_interior);
                    vNeighbor[vI_pre].insert(nV);
                    vNeighbor[vI_post].erase(vI_interior);
                    vNeighbor[vI_post].insert(nV);
                    vNeighbor[vI_interior].erase(vI_pre);
                    vNeighbor[vI_interior].erase(vI_post);
                    vNeighbor.resize(nV + 1);
                    vNeighbor[nV].insert(vI_pre);
                    vNeighbor[nV].insert(vI_post);

                    break;
                }
            }
        }

        // update cohesive edge pair and update cohEIndex
        cohE(nCE, 2) = nV;
        cohEIndex.erase(std::pair<int, int>(vI_boundary, vI_interior));
        cohEIndex[std::pair<int, int>(vI_boundary, nV)] = -nCE - 1;
        auto CEIfinder = cohEIndex.find(boundaryEdge[0]);
        if (CEIfinder != cohEIndex.end()) {
            if (CEIfinder->second >= 0)
                cohE(CEIfinder->second, 0) = nV;
            else
                cohE(-CEIfinder->second - 1, 3) = nV;
            cohEIndex[std::pair<int, int>(nV, boundaryEdge[0].second)] =
                CEIfinder->second;
            cohEIndex.erase(CEIfinder);
        }
    }
}

void TriMesh::mergeBoundaryEdges(const std::pair<int, int> &edge0,
                                 const std::pair<int, int> &edge1,
                                 const Eigen::RowVectorXd &mergedPos) {
    assert(edge0.second == edge1.first);
    assert(edge2Tri.find(std::pair<int, int>(edge0.second, edge0.first)) ==
           edge2Tri.end());
    assert(edge2Tri.find(std::pair<int, int>(edge1.second, edge1.first)) ==
           edge2Tri.end());
    assert(vNeighbor.size() == V.rows());

    fracTail.erase(edge0.second);
    fracTail.insert(edge0.first);
    curFracTail = edge0.first;

    V.row(edge0.first) = mergedPos;
    int vBackI = static_cast<int>(V.rows()) - 1;
    if (edge1.second < vBackI) {
        V_rest.row(edge1.second) = V_rest.row(vBackI);
        vertWeight[edge1.second] = vertWeight[vBackI];
        V.row(edge1.second) = V.row(vBackI);

        auto finder = fracTail.find(vBackI);
        if (finder != fracTail.end()) {
            fracTail.erase(finder);
            fracTail.insert(edge1.second);
        }
    } else {
        assert(edge1.second == vBackI);
    }
    V_rest.conservativeResize(vBackI, 3);
    vertWeight.conservativeResize(vBackI);
    V.conservativeResize(vBackI, 2);

    //        for(const auto& nbI : vNeighbor[edge1.second]) {
    //            std::pair<int, int> edgeToFind[2] = {
    //                std::pair<int, int>(edge1.second, nbI),
    //                std::pair<int, int>(nbI, edge1.second)
    //            };
    //            for(int eI = 0; eI < 2; eI++) {
    //                auto edgeTri = edge2Tri.find(edgeToFind[eI]);
    //                if(edgeTri != edge2Tri.end()) {
    //                    for(int vI = 0; vI < 3; vI++) {
    //                        if(F(edgeTri->second, vI) == edge1.second) {
    //                            F(edgeTri->second, vI) = edge0.first;
    //                            break;
    //                        }
    //                    }
    //                }
    //            }
    //        }

    for (int triI = 0; triI < F.rows(); triI++) {
        for (int vI = 0; vI < 3; vI++) {
            if (F(triI, vI) == edge1.second) {
                F(triI, vI) = edge0.first;
                break;
            }
        }
    }

    if (edge1.second < vBackI) {
        for (int triI = 0; triI < F.rows(); triI++) {
            for (int vI = 0; vI < 3; vI++) {
                if (F(triI, vI) == vBackI)
                    F(triI, vI) = edge1.second;
            }
        }
        //            // not valid because vNeighbor is not updated
        //            for(const auto& nbI : vNeighbor[vBackI]) {
        //                std::pair<int, int> edgeToFind[2] = {
        //                    std::pair<int, int>(vBackI, nbI),
        //                    std::pair<int, int>(nbI, vBackI)
        //                };
        //                for(int eI = 0; eI < 2; eI++) {
        //                    auto edgeTri = edge2Tri.find(edgeToFind[eI]);
        //                    if(edgeTri != edge2Tri.end()) {
        //                        for(int vI = 0; vI < 3; vI++) {
        //                            if(F(edgeTri->second, vI) == vBackI) {
        //                                F(edgeTri->second, vI) = edge1.second;
        //                                break;
        //                            }
        //                        }
        //                    }
        //                }
        //            }
    }

    auto cohEFinder = cohEIndex.find(edge0);
    assert(cohEFinder != cohEIndex.end());
    int cohEBackI = static_cast<int>(cohE.rows()) - 1;
    if (cohEFinder->second >= 0) {
        if (cohEFinder->second < cohEBackI)
            cohE.row(cohEFinder->second) = cohE.row(cohEBackI);
        else
            assert(cohEFinder->second == cohEBackI);
    } else {
        if (-cohEFinder->second - 1 < cohEBackI)
            cohE.row(-cohEFinder->second - 1) = cohE.row(cohEBackI);
        else
            assert(-cohEFinder->second - 1 == cohEBackI);
    }
    cohE.conservativeResize(cohEBackI, 4);

    for (int cohI = 0; cohI < cohE.rows(); cohI++) {
        for (int pI = 0; pI < 4; pI++) {
            if (cohE(cohI, pI) == edge1.second)
                cohE(cohI, pI) = edge0.first;
        }
    }
    if (edge1.second < vBackI) {
        for (int cohI = 0; cohI < cohE.rows(); cohI++) {
            for (int pI = 0; pI < 4; pI++) {
                if (cohE(cohI, pI) == vBackI)
                    cohE(cohI, pI) = edge1.second;
            }
        }
    }
    // closeup just interior split diamond
    // TODO: do it faster by knowing the edge in advance and locate using
    // cohIndex
    for (int cohI = 0; cohI < cohE.rows(); cohI++) {
        if ((cohE(cohI, 0) == cohE(cohI, 2)) &&
            (cohE(cohI, 1) == cohE(cohI, 3))) {
            fracTail.erase(cohE(cohI, 0));
            fracTail.erase(cohE(cohI, 1));
            curFracTail = -1;

            if (cohI < cohE.rows() - 1)
                cohE.row(cohI) = cohE.row(cohE.rows() - 1);
            cohE.conservativeResize(cohE.rows() - 1, 4);
            break;
        }
    }
    // TODO: locally update edge2Tri, vNeighbor, cohEIndex
}
} // namespace uvgami
