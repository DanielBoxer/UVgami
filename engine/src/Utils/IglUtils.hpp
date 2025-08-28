//  Created by Minchen Li on 8/30/17.

#ifndef IglUtils_hpp
#define IglUtils_hpp

#include "TriMesh.hpp"

#include <Eigen/Eigen>

#include <iostream>
#include <fstream>

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

    static void mapScalarToColor_bin(const Eigen::VectorXd &scalar,
                                     Eigen::MatrixXd &color, double thres);
    static void mapScalarToColor(const std::string &meshName,
                                 const Eigen::VectorXd &scalar,
                                 Eigen::MatrixXd &color, double lowerBound,
                                 double upperBound, int opt = 0);

    static void addBlockToMatrix(Eigen::SparseMatrix<double> &mtr,
                                 const Eigen::MatrixXd &block,
                                 const Eigen::VectorXi &index, int dim);
    static void addBlockToMatrix(const Eigen::MatrixXd &block,
                                 const Eigen::VectorXi &index, int dim,
                                 Eigen::VectorXd *V, Eigen::VectorXi *I = NULL,
                                 Eigen::VectorXi *J = NULL);
    static void addDiagonalToMatrix(const Eigen::VectorXd &diagonal,
                                    const Eigen::VectorXi &index, int dim,
                                    Eigen::VectorXd *V,
                                    Eigen::VectorXi *I = NULL,
                                    Eigen::VectorXi *J = NULL);
    static void addBlockToMatrix(const Eigen::MatrixXd &block,
                                 const Eigen::VectorXi &index, int dim,
                                 Eigen::MatrixXd &mtr);
    static void addDiagonalToMatrix(const Eigen::VectorXd &diagonal,
                                    const Eigen::VectorXi &index, int dim,
                                    Eigen::MatrixXd &mtr);

    // project a symmetric real matrix to the nearest SPD matrix
    template <typename Scalar, int size>
    static void makePD(Eigen::Matrix<Scalar, size, size> &symMtr) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>>
            eigenSolver(symMtr);
        if (eigenSolver.eigenvalues()[0] >= 0.0) {
            return;
        }
        Eigen::DiagonalMatrix<Scalar, size> D(eigenSolver.eigenvalues());
        int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
        int i = 0;
        for (; i < rows; i++) {
            if (D.diagonal()[i] < 0.0) {
                D.diagonal()[i] = 0.0;
            } else {
                break;
            }
        }
        //            D.diagonal().segment(0, i).array() += 1.0e-3 *
        //            D.diagonal()[i];
        symMtr = eigenSolver.eigenvectors() * D *
                 eigenSolver.eigenvectors().transpose();
    }

    static double computeRotAngle(const Eigen::RowVector2d &from,
                                  const Eigen::RowVector2d &to);

    // test whether 2D segments ab intersect with cd
    static bool Test2DSegmentSegment(const Eigen::RowVector2d &a,
                                     const Eigen::RowVector2d &b,
                                     const Eigen::RowVector2d &c,
                                     const Eigen::RowVector2d &d,
                                     double eps = 0.0);

    static void
    addThickEdge(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &UV,
                 Eigen::MatrixXd &seamColor, const Eigen::RowVector3d &color,
                 const Eigen::RowVector3d &v0, const Eigen::RowVector3d &v1,
                 double halfWidth, double texScale, bool UVorSurface = false,
                 const Eigen::RowVector3d &normal = Eigen::RowVector3d());

    static void smoothVertField(const TriMesh &mesh, Eigen::VectorXd &field);
};
} // namespace uvgami

#endif
