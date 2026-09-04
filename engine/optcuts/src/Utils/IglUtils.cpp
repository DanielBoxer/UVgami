//  Created by Minchen Li on 8/30/17.

#include "IglUtils.hpp"

#include "ParallelFor.hpp"

#include <set>
#include <unordered_map>

namespace uvgami {

void IglUtils::computeUniformLaplacian(const Eigen::MatrixXi &F,
                                       Eigen::SparseMatrix<double> &graphL) {
    int vertAmt = F.maxCoeff() + 1;
    std::vector<Eigen::Triplet<double>> triplet(F.rows() * 9);
    parallelFor((int)F.rows(), [&](int rowI) {
        int startInd = rowI * 9;

        triplet[startInd] = Eigen::Triplet<double>(F(rowI, 0), F(rowI, 1), 1.0);
        triplet[startInd + 1] =
            Eigen::Triplet<double>(F(rowI, 1), F(rowI, 0), 1.0);
        triplet[startInd + 2] =
            Eigen::Triplet<double>(F(rowI, 1), F(rowI, 2), 1.0);
        triplet[startInd + 3] =
            Eigen::Triplet<double>(F(rowI, 2), F(rowI, 1), 1.0);
        triplet[startInd + 4] =
            Eigen::Triplet<double>(F(rowI, 2), F(rowI, 0), 1.0);
        triplet[startInd + 5] =
            Eigen::Triplet<double>(F(rowI, 0), F(rowI, 2), 1.0);

        triplet[startInd + 6] =
            Eigen::Triplet<double>(F(rowI, 0), F(rowI, 0), -2.0);
        triplet[startInd + 7] =
            Eigen::Triplet<double>(F(rowI, 1), F(rowI, 1), -2.0);
        triplet[startInd + 8] =
            Eigen::Triplet<double>(F(rowI, 2), F(rowI, 2), -2.0);
    });
    graphL.resize(vertAmt, vertAmt);
    graphL.setFromTriplets(triplet.begin(), triplet.end());
}

double getHETan(const std::map<std::pair<int, int>, double> &HETan, int v0,
                int v1) {
    auto finder = HETan.find(std::pair<int, int>(v0, v1));
    if (finder == HETan.end()) {
        return 0.0;
    } else {
        return finder->second;
    }
}

void IglUtils::computeMVCMtr(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                             Eigen::SparseMatrix<double> &MVCMtr) {
    std::map<std::pair<int, int>, double> HETan;
    std::map<std::pair<int, int>, int> thirdPoint;
    std::vector<std::set<int>> vvNeighbor(V.rows());
    for (int triI = 0; triI < F.rows(); triI++) {
        int v0I = F(triI, 0);
        int v1I = F(triI, 1);
        int v2I = F(triI, 2);

        Eigen::Vector3d e01 = V.row(v1I) - V.row(v0I);
        Eigen::Vector3d e12 = V.row(v2I) - V.row(v1I);
        Eigen::Vector3d e20 = V.row(v0I) - V.row(v2I);
        double dot0102 = -e01.dot(e20);
        double dot1210 = -e12.dot(e01);
        double dot2021 = -e20.dot(e12);
        double cos0102 = dot0102 / (e01.norm() * e20.norm());
        double cos1210 = dot1210 / (e01.norm() * e12.norm());
        double cos2021 = dot2021 / (e12.norm() * e20.norm());

        HETan[std::pair<int, int>(v0I, v1I)] =
            sqrt(1.0 - cos0102 * cos0102) / (1.0 + cos0102);
        HETan[std::pair<int, int>(v1I, v2I)] =
            sqrt(1.0 - cos1210 * cos1210) / (1.0 + cos1210);
        HETan[std::pair<int, int>(v2I, v0I)] =
            sqrt(1.0 - cos2021 * cos2021) / (1.0 + cos2021);

        thirdPoint[std::pair<int, int>(v0I, v1I)] = v2I;
        thirdPoint[std::pair<int, int>(v1I, v2I)] = v0I;
        thirdPoint[std::pair<int, int>(v2I, v0I)] = v1I;

        vvNeighbor[v0I].insert(v1I);
        vvNeighbor[v0I].insert(v2I);
        vvNeighbor[v1I].insert(v0I);
        vvNeighbor[v1I].insert(v2I);
        vvNeighbor[v2I].insert(v0I);
        vvNeighbor[v2I].insert(v1I);
    }

    MVCMtr.resize(V.rows(), V.rows());
    MVCMtr.setZero();
    MVCMtr.reserve(V.rows() * 7);
    for (int rowI = 0; rowI < V.rows(); rowI++) {
        for (const auto &nbVI : vvNeighbor[rowI]) {
            double weight = getHETan(HETan, rowI, nbVI);
            auto finder = thirdPoint.find(std::pair<int, int>(nbVI, rowI));
            if (finder != thirdPoint.end()) {
                weight += getHETan(HETan, rowI, finder->second);
            }
            weight /= (V.row(rowI) - V.row(nbVI)).norm();

            MVCMtr.coeffRef(rowI, rowI) -= weight;
            MVCMtr.insert(rowI, nbVI) = weight;

            //                // symmetrized version
            //                MVCMtr.coeffRef(rowI, rowI) -= weight;
            //                MVCMtr.coeffRef(rowI, nbVI) += weight;
            //                MVCMtr.coeffRef(nbVI, nbVI) -= weight;
            //                MVCMtr.coeffRef(nbVI, rowI) += weight;
        }
    }
    //        writeSparseMatrixToFile("/Users/mincli/Desktop/meshes/mtr",
    //        MVCMtr);
}

void IglUtils::fixedBoundaryParam_MVC(Eigen::SparseMatrix<double> A,
                                      const Eigen::VectorXi &bnd,
                                      const Eigen::MatrixXd &bnd_uv,
                                      Eigen::MatrixXd &UV_Tutte) {
    assert(bnd.size() == bnd_uv.rows());
    assert(bnd.maxCoeff() < A.rows());
    assert(A.rows() == A.cols());

    int vN = static_cast<int>(A.rows());
    A.conservativeResize(vN + bnd.size(), vN + bnd.size());
    A.reserve(A.nonZeros() + bnd.size() * 2);
    for (int pcI = 0; pcI < bnd.size(); pcI++) {
        A.insert(vN + pcI, bnd[pcI]) = 1.0;
        A.insert(bnd[pcI], vN + pcI) = 1.0;
    }

    Eigen::SparseLU<Eigen::SparseMatrix<double>> spLUSolver;
    spLUSolver.compute(A);
    if (spLUSolver.info() == Eigen::Success) {
        UV_Tutte.resize(A.rows(), 2);
        Eigen::VectorXd rhs;
        rhs.resize(A.rows());

        for (int dimI = 0; dimI < 2; dimI++) {
            rhs << Eigen::VectorXd::Zero(vN), bnd_uv.col(dimI);
            UV_Tutte.col(dimI) = spLUSolver.solve(rhs);
            if (spLUSolver.info() != Eigen::Success) {
                assert("LU back solve failed!");
            }
        }

        UV_Tutte.conservativeResize(vN, 2);
    } else {
        assert("LU decomposition on MVC matrix (with Langrange Multiplier) "
               "failed!");
    }
}

void IglUtils::mapTriangleTo2D(const Eigen::Vector3d v[3],
                               Eigen::Vector2d u[3]) {
    const Eigen::Vector3d e[2] = {v[1] - v[0], v[2] - v[0]};
    u[0] << 0.0, 0.0;
    u[1] << e[0].norm(), 0.0;
    u[2] << e[0].dot(e[1]) / u[1][0], e[0].cross(e[1]).norm() / u[1][0];
}

void IglUtils::computeDeformationGradient(const Eigen::Vector3d v[3],
                                          const Eigen::Vector2d u[3],
                                          Eigen::Matrix2d &F) {
    Eigen::Vector2d x[3];
    IglUtils::mapTriangleTo2D(v, x);

    const Eigen::Vector2d u01 = u[1] - u[0];
    const Eigen::Vector2d u02 = u[2] - u[0];
    const double u01Len = u01.norm();

    Eigen::Matrix2d U;
    U << u01Len, u01.dot(u02) / u01Len, 0.0,
        (u01[0] * u02[1] - u01[1] * u02[0]) / u01Len;
    Eigen::Matrix2d V;
    V << x[1], x[2];
    F = V * U.inverse();
}

void IglUtils::map_vertices_to_circle(const Eigen::MatrixXd &V,
                                      const Eigen::VectorXi &bnd,
                                      Eigen::MatrixXd &UV) {
    // Get sorted list of boundary vertices
    std::vector<int> interior, map_ij;
    map_ij.resize(V.rows());

    std::vector<bool> isOnBnd(V.rows(), false);
    for (int i = 0; i < bnd.size(); i++) {
        isOnBnd[bnd[i]] = true;
        map_ij[bnd[i]] = i;
    }

    for (int i = 0; i < (int)isOnBnd.size(); i++) {
        if (!isOnBnd[i]) {
            map_ij[i] = static_cast<int>(interior.size());
            interior.push_back(i);
        }
    }

    // Map boundary to circle
    std::vector<double> len(bnd.size());
    len[0] = 0.;

    for (int i = 1; i < bnd.size(); i++) {
        len[i] = len[i - 1] + (V.row(bnd[i - 1]) - V.row(bnd[i])).norm();
    }
    double total_len = len[len.size() - 1] +
                       (V.row(bnd[0]) - V.row(bnd[bnd.size() - 1])).norm();

    UV.resize(bnd.size(), 2);
    const double radius = total_len / 2.0 / M_PI;
    for (int i = 0; i < bnd.size(); i++) {
        double frac = len[i] * 2. * M_PI / total_len;
        UV.row(map_ij[bnd[i]]) << radius * cos(frac), radius * sin(frac);
    }
}

// the addon parses this line for its progress bar
void IglUtils::reportDistortion(const Eigen::VectorXd &scalar,
                                double lowerBound, double upperBound) {
    const double rangeDelta = upperBound - lowerBound;
    const double rangeLow = lowerBound + 0.2 * rangeDelta;
    const double rangeMedium = lowerBound + 0.4 * rangeDelta;

    int distortionLow = 0;
    int distortionMedium = 0;
    int distortionHigh = 0;
    for (int elemI = 0; elemI < scalar.size(); elemI++) {
        if (scalar[elemI] < rangeLow)
            distortionLow++;
        else if (scalar[elemI] < rangeMedium)
            distortionMedium++;
        else
            distortionHigh++;
    }

    std::cout << "progress: " << (float)distortionLow / scalar.size() << " "
              << (float)distortionMedium / scalar.size() << " "
              << (float)distortionHigh / scalar.size() << std::endl
              << std::flush;
}

void IglUtils::addBlockToMatrix(Eigen::SparseMatrix<double> &mtr,
                                Eigen::Ref<const Eigen::MatrixXd> block,
                                Eigen::Ref<const Eigen::VectorXi> index,
                                int dim) {
    assert(block.rows() == block.cols());
    assert(index.size() * dim == block.rows());
    assert(mtr.rows() == mtr.cols());
    assert(index.maxCoeff() * dim + dim - 1 < mtr.rows());

    for (int indI = 0; indI < index.size(); indI++) {
        if (index[indI] < 0) {
            continue;
        }
        int startIndI = index[indI] * dim;
        int startIndI_block = indI * dim;

        for (int indJ = 0; indJ < index.size(); indJ++) {
            if (index[indJ] < 0) {
                continue;
            }
            int startIndJ = index[indJ] * dim;
            int startIndJ_block = indJ * dim;

            for (int dimI = 0; dimI < dim; dimI++) {
                for (int dimJ = 0; dimJ < dim; dimJ++) {
                    mtr.coeffRef(startIndI + dimI, startIndJ + dimJ) +=
                        block(startIndI_block + dimI, startIndJ_block + dimJ);
                }
            }
        }
    }
}

void IglUtils::addDiagonalToMatrix(Eigen::Ref<const Eigen::VectorXd> diagonal,
                                   Eigen::Ref<const Eigen::VectorXi> index,
                                   int dim, Eigen::VectorXd *V,
                                   Eigen::VectorXi *I, Eigen::VectorXi *J) {
    assert(index.size() * dim == diagonal.size());

    assert(V);
    int tripletInd = static_cast<int>(V->size());
    const int entryAmt = static_cast<int>(diagonal.size());
    V->conservativeResize(tripletInd + entryAmt);
    if (I) {
        assert(J);
        assert(I->size() == tripletInd);
        assert(J->size() == tripletInd);
        I->conservativeResize(tripletInd + entryAmt);
        J->conservativeResize(tripletInd + entryAmt);
    }

    for (int indI = 0; indI < index.size(); indI++) {
        if (index[indI] < 0) {
            assert(0 && "currently doesn't support fixed vertices here!");
            continue;
        }
        int startIndI = index[indI] * dim;
        int startIndI_diagonal = indI * dim;

        for (int dimI = 0; dimI < dim; dimI++) {
            (*V)[tripletInd] = diagonal(startIndI_diagonal + dimI);
            if (I) {
                (*I)[tripletInd] = (*J)[tripletInd] = startIndI + dimI;
            }
            tripletInd++;
        }
    }
}

void IglUtils::addBlockToMatrix(Eigen::Ref<const Eigen::MatrixXd> block,
                                Eigen::Ref<const Eigen::VectorXi> index, int dim,
                                Eigen::VectorXd *V, Eigen::VectorXi *I,
                                Eigen::VectorXi *J, int tripletInd) {
    int num_free = 0;
    for (int indI = 0; indI < index.size(); indI++) {
        if (index[indI] >= 0) {
            num_free++;
        }
    }
    if (!num_free) {
        return;
    }

    assert(block.rows() == block.cols());
    assert(index.size() * dim == block.rows());

    assert(V);
    const int entryAmt = static_cast<int>(dim * dim * num_free * num_free);
    assert(V->size() >= tripletInd + entryAmt);
    if (I) {
        assert(J);
        assert(I->size() == V->size());
        assert(J->size() == V->size());
    }

    for (int indI = 0; indI < index.size(); indI++) {
        if (index[indI] < 0) {
            continue;
        }
        int startIndI = index[indI] * dim;
        int startIndI_block = indI * dim;

        for (int indJ = 0; indJ < index.size(); indJ++) {
            if (index[indJ] < 0) {
                continue;
            }
            int startIndJ = index[indJ] * dim;
            int startIndJ_block = indJ * dim;

            for (int dimI = 0; dimI < dim; dimI++) {
                for (int dimJ = 0; dimJ < dim; dimJ++) {
                    (*V)[tripletInd] =
                        block(startIndI_block + dimI, startIndJ_block + dimJ);
                    if (I) {
                        (*I)[tripletInd] = startIndI + dimI;
                        (*J)[tripletInd] = startIndJ + dimJ;
                    }
                    tripletInd++;
                }
            }
        }
    }
}

void IglUtils::addBlockToMatrix(Eigen::Ref<const Eigen::MatrixXd> block,
                                Eigen::Ref<const Eigen::VectorXi> index, int dim,
                                Eigen::MatrixXd &mtr) {
    int num_free = 0;
    for (int indI = 0; indI < index.size(); indI++) {
        if (index[indI] >= 0) {
            num_free++;
        }
    }
    if (!num_free) {
        return;
    }

    assert(block.rows() == block.cols());
    assert(index.size() * dim == block.rows());
    assert(mtr.rows() == mtr.cols());
    assert(mtr.rows() % dim == 0);

    for (int indI = 0; indI < index.size(); indI++) {
        if (index[indI] < 0) {
            continue;
        }
        int startIndI = index[indI] * dim;
        int startIndI_block = indI * dim;

        for (int indJ = 0; indJ < index.size(); indJ++) {
            if (index[indJ] < 0) {
                continue;
            }
            int startIndJ = index[indJ] * dim;
            int startIndJ_block = indJ * dim;

            mtr.block(startIndI, startIndJ, dim, dim) +=
                block.block(startIndI_block, startIndJ_block, dim, dim);
        }
    }
}
void IglUtils::addDiagonalToMatrix(Eigen::Ref<const Eigen::VectorXd> diagonal,
                                   Eigen::Ref<const Eigen::VectorXi> index,
                                   int dim, Eigen::MatrixXd &mtr) {
    assert(index.size() * dim == diagonal.size());
    assert(mtr.rows() == mtr.cols());
    assert(mtr.rows() % dim == 0);

    for (int indI = 0; indI < index.size(); indI++) {
        if (index[indI] < 0) {
            assert(0 && "currently doesn't support fixed vertices here!");
            continue;
        }
        int startIndI = index[indI] * dim;
        int startIndI_diagonal = indI * dim;

        mtr(startIndI, startIndI) = diagonal(startIndI_diagonal);
        mtr(startIndI + 1, startIndI + 1) = diagonal(startIndI_diagonal + 1);
    }
}
double IglUtils::computeRotAngle(const Eigen::RowVector2d &from,
                                 const Eigen::RowVector2d &to) {
    double angle =
        std::acos((std::max)(-1.0, (std::min)(1.0, from.dot(to) / from.norm() /
                                                       to.norm())));
    return ((from[0] * to[1] - from[1] * to[0] < 0.0) ? -angle : angle);
}

/////////////////////////////////////////////////////////////////
// 2D line segments intersection checking code
// based on Real-Time Collision Detection by Christer Ericson
// (Morgan Kaufmaan Publishers, 2005 Elvesier Inc)
double Signed2DTriArea(const Eigen::RowVector2d &a, const Eigen::RowVector2d &b,
                       const Eigen::RowVector2d &c) {
    return (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]);
}

bool IglUtils::Test2DSegmentSegment(const Eigen::RowVector2d &a,
                                    const Eigen::RowVector2d &b,
                                    const Eigen::RowVector2d &c,
                                    const Eigen::RowVector2d &d, double eps) {
    double eps_quad = 0.0, eps_sq = 0.0;
    if (eps) {
        eps = std::abs(eps);
        eps_sq =
            eps * eps * ((a - b).squaredNorm() + (c - d).squaredNorm()) / 2.0;
        eps_quad = eps_sq * eps_sq;
    }

    // signs of areas correspond to which side of ab points c and d are
    double a1 = Signed2DTriArea(a, b, d); // Compute winding of abd (+ or -)
    double a2 =
        Signed2DTriArea(a, b, c); // To intersect, must have sign opposite of a1

    // If c and d are on different sides of ab, areas have different signs
    if (a1 * a2 <= eps_quad) // require unsigned x & y values.
    {
        double a3 = Signed2DTriArea(c, d, a); // Compute winding of cda (+ or -)
        double a4 = a3 + a2 - a1; // Since area is constant a1 - a2 = a3 - a4,
                                  // or a4 = a3 + a2 - a1

        // Points a and b on different sides of cd if areas have different signs
        if (a3 * a4 <= eps_quad) {
            if ((std::abs(a1) <= eps_sq) && (std::abs(a2) <= eps_sq)) {
                // colinear
                const Eigen::RowVector2d ab = b - a;
                const double sqnorm_ab = ab.squaredNorm();
                const Eigen::RowVector2d ac = c - a;
                const Eigen::RowVector2d ad = d - a;
                double coef_c = ac.dot(ab) / sqnorm_ab;
                double coef_d = ad.dot(ab) / sqnorm_ab;
                assert(coef_c != coef_d);

                if (coef_c > coef_d) {
                    std::swap(coef_c, coef_d);
                }

                if ((coef_c > 1.0 + eps) || (coef_d < -eps)) {
                    return false;
                } else {
                    return true;
                }
            } else {
                // Segments intersect.
                return true;
            }
        }
    }

    // Segments not intersecting
    return false;
}

static bool transversalCross(const Eigen::RowVector2d &a,
                             const Eigen::RowVector2d &b,
                             const Eigen::RowVector2d &c,
                             const Eigen::RowVector2d &d) {
    const double a1 = Signed2DTriArea(a, b, d);
    const double a2 = Signed2DTriArea(a, b, c);
    if (a1 * a2 >= 0.0)
        return false;
    const double a3 = Signed2DTriArea(c, d, a);
    const double a4 = a3 + a2 - a1;
    return a3 * a4 < 0.0;
}

bool IglUtils::checkUVBoundaryOverlap(
    const Eigen::MatrixXd &UV,
    const std::vector<std::vector<int>> &bnd_all,
    std::set<int> *crossingVerts, bool transversalOnly) {
    std::vector<std::pair<int, int>> edges;
    for (const auto &loop : bnd_all) {
        int n = loop.size();
        for (int i = 0; i < n; ++i) {
            edges.emplace_back(loop[i], loop[(i + 1) % n]);
        }
    }
    if (edges.size() < 2) {
        return false;
    }

    double totalLen = 0.0;
    for (const auto &e : edges) {
        totalLen += (UV.row(e.first) - UV.row(e.second)).norm();
    }
    double cellSize = totalLen / edges.size();
    if (cellSize <= 0.0) {
        return false;
    }

    // hash collisions only add narrow-phase tests
    auto cellKey = [](int cx, int cy) -> int64_t {
        return ((int64_t)cx * 73856093) ^ ((int64_t)cy * 19349663);
    };
    auto cellRange = [&](int ei, int &cxMin, int &cxMax, int &cyMin,
                         int &cyMax) {
        const auto a = UV.row(edges[ei].first);
        const auto b = UV.row(edges[ei].second);
        cxMin = std::floor(std::min(a[0], b[0]) / cellSize);
        cxMax = std::floor(std::max(a[0], b[0]) / cellSize);
        cyMin = std::floor(std::min(a[1], b[1]) / cellSize);
        cyMax = std::floor(std::max(a[1], b[1]) / cellSize);
    };

    std::unordered_map<int64_t, std::vector<int>> grid;
    for (int ei = 0; ei < (int)edges.size(); ++ei) {
        int cxMin, cxMax, cyMin, cyMax;
        cellRange(ei, cxMin, cxMax, cyMin, cyMax);
        for (int cx = cxMin; cx <= cxMax; ++cx) {
            for (int cy = cyMin; cy <= cyMax; ++cy) {
                grid[cellKey(cx, cy)].push_back(ei);
            }
        }
    }

    bool overlapped = false;
    for (int ei = 0; ei < (int)edges.size(); ++ei) {
        const Eigen::RowVector2d a = UV.row(edges[ei].first).head<2>();
        const Eigen::RowVector2d b = UV.row(edges[ei].second).head<2>();
        int cxMin, cxMax, cyMin, cyMax;
        cellRange(ei, cxMin, cxMax, cyMin, cyMax);
        std::set<int> candidates;
        for (int cx = cxMin; cx <= cxMax; ++cx) {
            for (int cy = cyMin; cy <= cyMax; ++cy) {
                for (int ej : grid[cellKey(cx, cy)]) {
                    if (ej > ei) {
                        candidates.insert(ej);
                    }
                }
            }
        }
        for (int ej : candidates) {
            // adjacent boundary edges share a vertex
            if (edges[ei].first == edges[ej].first ||
                edges[ei].first == edges[ej].second ||
                edges[ei].second == edges[ej].first ||
                edges[ei].second == edges[ej].second) {
                continue;
            }
            const Eigen::RowVector2d c = UV.row(edges[ej].first).head<2>();
            const Eigen::RowVector2d d = UV.row(edges[ej].second).head<2>();
            if (transversalOnly ? transversalCross(a, b, c, d)
                                : Test2DSegmentSegment(a, b, c, d)) {
                if (!crossingVerts) {
                    return true;
                }
                overlapped = true;
                crossingVerts->insert(edges[ei].first);
                crossingVerts->insert(edges[ei].second);
                crossingVerts->insert(edges[ej].first);
                crossingVerts->insert(edges[ej].second);
            }
        }
    }
    return overlapped;
}
////////////////////////////////////////////////////////////

void IglUtils::smoothVertField(const TriMesh &mesh, Eigen::VectorXd &field) {
    assert(field.size() == mesh.V.rows());
    Eigen::VectorXd field_copy = field;
    for (int vI = 0; vI < field.size(); vI++) {
        for (const auto nbVI : mesh.vNeighbor[vI]) {
            field[vI] += field_copy[nbVI];
        }
        field[vI] /= mesh.vNeighbor[vI].size() + 1;
    }
}
} // namespace uvgami
