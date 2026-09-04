//  Created by Minchen Li on 8/30/17.

#ifndef IglUtils_hpp
#define IglUtils_hpp

#include "TriMesh.hpp"

#include <Eigen/Eigen>

#include <iostream>
#include <fstream>
#include <set>
#include <vector>

namespace uvgami {

// a static class implementing basic geometry processing operations that are not
// provided in libIgl
class IglUtils {
  public:
    // graph laplacian with half-weighted boundary edge, the computation is also
    // faster
    static void computeUniformLaplacian(const Eigen::MatrixXi &F,
                                        Eigen::SparseMatrix<double> &graphL);

    static void computeMVCMtr(const Eigen::MatrixXd &V,
                              const Eigen::MatrixXi &F,
                              Eigen::SparseMatrix<double> &MVCMtr);

    static void fixedBoundaryParam_MVC(Eigen::SparseMatrix<double> A,
                                       const Eigen::VectorXi &bnd,
                                       const Eigen::MatrixXd &bnd_uv,
                                       Eigen::MatrixXd &UV_Tutte);

    static void mapTriangleTo2D(const Eigen::Vector3d v[3],
                                Eigen::Vector2d u[3]);
    static void computeDeformationGradient(const Eigen::Vector3d v[3],
                                           const Eigen::Vector2d u[3],
                                           Eigen::Matrix2d &F);

    // to a circle with the perimeter equal to the length of the boundary on the
    // mesh
    static void map_vertices_to_circle(const Eigen::MatrixXd &V,
                                       const Eigen::VectorXi &bnd,
                                       Eigen::MatrixXd &UV);

    static void reportDistortion(const Eigen::VectorXd &scalar,
                                 double lowerBound, double upperBound);

    static void addBlockToMatrix(Eigen::SparseMatrix<double> &mtr,
                                 Eigen::Ref<const Eigen::MatrixXd> block,
                                 Eigen::Ref<const Eigen::VectorXi> index,
                                 int dim);
    // writes into presized V, I, J at tripletInd so callers can fill disjoint slices
    static void addBlockToMatrix(Eigen::Ref<const Eigen::MatrixXd> block,
                                 Eigen::Ref<const Eigen::VectorXi> index, int dim,
                                 Eigen::VectorXd *V, Eigen::VectorXi *I,
                                 Eigen::VectorXi *J, int tripletInd);
    static void addDiagonalToMatrix(Eigen::Ref<const Eigen::VectorXd> diagonal,
                                    Eigen::Ref<const Eigen::VectorXi> index,
                                    int dim, Eigen::VectorXd *V,
                                    Eigen::VectorXi *I = NULL,
                                    Eigen::VectorXi *J = NULL);
    static void addBlockToMatrix(Eigen::Ref<const Eigen::MatrixXd> block,
                                 Eigen::Ref<const Eigen::VectorXi> index, int dim,
                                 Eigen::MatrixXd &mtr);
    static void addDiagonalToMatrix(Eigen::Ref<const Eigen::VectorXd> diagonal,
                                    Eigen::Ref<const Eigen::VectorXi> index,
                                    int dim, Eigen::MatrixXd &mtr);

    static double computeRotAngle(const Eigen::RowVector2d &from,
                                  const Eigen::RowVector2d &to);

    // test whether 2D segments ab intersect with cd
    static bool Test2DSegmentSegment(const Eigen::RowVector2d &a,
                                     const Eigen::RowVector2d &b,
                                     const Eigen::RowVector2d &c,
                                     const Eigen::RowVector2d &d,
                                     double eps = 0.0);

    // true if any two non-adjacent UV boundary edges cross, full containment is missed
    static bool checkUVBoundaryOverlap(
        const Eigen::MatrixXd &UV,
        const std::vector<std::vector<int>> &bnd_all,
        std::set<int> *crossingVerts = nullptr,
        // a mid-zip stitch's coincident runs must not read as crossings
        bool transversalOnly = false);

    static void smoothVertField(const TriMesh &mesh, Eigen::VectorXd &field);
};
} // namespace uvgami

#endif
