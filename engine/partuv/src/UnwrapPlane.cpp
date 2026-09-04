#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/per_face_normals.h>
#include <igl/adjacency_list.h>
#include <igl/lscm.h>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <utility>

// LSCM
#include "Mesh.h"
#include "FormTrait.h"
#include "LSCM.h"
#include "Component.h"
#include "UnwrapMerge.h"
#include "UnwrapBB.h"
#include "UnwrapPlane.h"
// Triangle intersection
#include "triangleHelper.hpp"

// Distortion
#include "Distortion.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <set>

#include "UnwrapBB.h"
#include "merge.h"
#include "IO.h"
#include "UnwrapBB.h"





// ---------------- CGAL headers ----------------
#include <CGAL/Simple_cartesian.h>
#include <CGAL/convex_hull_2.h>

using namespace MeshLib;





typedef CGAL::Simple_cartesian<double> Kernel;



//============================================================
// A helper to compute the intersection of one triangle with a plane.
void trianglePlaneIntersection(const Eigen::Vector3d &v0,
                               const Eigen::Vector3d &v1,
                               const Eigen::Vector3d &v2,
                               const Eigen::Vector3d &plane_point,
                               const Eigen::Vector3d &plane_normal,
                               std::vector<Eigen::Vector3d> &pts)
{
    const double eps = 1e-6;
    // Compute signed distances from the plane
    double d0 = (v0 - plane_point).dot(plane_normal);
    double d1 = (v1 - plane_point).dot(plane_normal);
    double d2 = (v2 - plane_point).dot(plane_normal);

    // A lambda to process one edge
    auto processEdge = [&](const Eigen::Vector3d &va, const Eigen::Vector3d &vb,
                           double da, double db)
    {
        if (std::abs(da) < eps && std::abs(db) < eps)
        {
            // Both endpoints are (nearly) on the plane
            pts.push_back(va);
            pts.push_back(vb);
        }
        else if ((da > 0 && db < 0) || (da < 0 && db > 0))
        {
            double t = da / (da - db);
            Eigen::Vector3d p = va + t * (vb - va);
            pts.push_back(p);
        }
        else if (std::abs(da) < eps)
        {
            pts.push_back(va);
        }
        else if (std::abs(db) < eps)
        {
            pts.push_back(vb);
        }
    };

    // Process each triangle edge
    processEdge(v0, v1, d0, d1);
    processEdge(v1, v2, d1, d2);
    processEdge(v2, v0, d2, d0);
}

//============================================================
// A helper to compute the area of a polygon from its 2D hull.
double polygonArea2D(const std::vector<Point_2> &poly)
{
    double area = 0.0;
    const size_t n = poly.size();
    if(n < 3) return 0.0;

    for(size_t i = 0; i < n; ++i)
    {
        size_t j = (i + 1) % n;
        double x_i = CGAL::to_double(poly[i].x());
        double y_i = CGAL::to_double(poly[i].y());
        double x_j = CGAL::to_double(poly[j].x());
        double y_j = CGAL::to_double(poly[j].y());
        area += (x_i * y_j - y_i * x_j);
    }
    return 0.5 * std::fabs(area);
}

//============================================================
// A struct to hold the result
struct CutPlaneResult {
  std::vector<int> faces_side1;
  std::vector<int> faces_side2;
  Eigen::Vector3d optimal_normal;
  Eigen::Vector3d optimal_point;
};

//============================================================
// The main function: find_max_cut_plane_box_multiplane
CutPlaneResult find_max_cut_plane_box_multiplane(const Eigen::MatrixXd &V_rotated,
                                                 const Eigen::MatrixXi &F)
{
    // 1) Rotate mesh to OBB frame


    // 2) Compute bounding box
    Eigen::RowVector3d bbox_min = V_rotated.colwise().minCoeff();
    Eigen::RowVector3d bbox_max = V_rotated.colwise().maxCoeff();

    // 3) Parameters for sampling
    const int n_samples = 20;
    double max_area = -std::numeric_limits<double>::infinity();
    Eigen::Vector3d optimal_normal;
    double optimal_d = 0.0;
    int optimal_axis = -1;

    // 4) The three principal axes
    std::vector<std::pair<Eigen::Vector3d,int>> axes = {
        { Eigen::Vector3d(1,0,0), 0 },
        { Eigen::Vector3d(0,1,0), 1 },
        { Eigen::Vector3d(0,0,1), 2 }
    };

    // 5) For each axis, sample parallel planes
    for(const auto &ax : axes)
    {
        Eigen::Vector3d normal = ax.first;
        int axis_index = ax.second;
        double d_min = bbox_min(axis_index);
        double d_max = bbox_max(axis_index);

        // Sample d values in [d_min, d_max]
        std::vector<double> d_samples(n_samples);
        for(int i = 0; i < n_samples; ++i)
        {
            double t = double(i) / (n_samples - 1);
            d_samples[i] = (1.0 - t)*d_min + t*d_max;
        }

        // For each plane
        for(double d : d_samples)
        {
            // Construct plane point: (0,0,0) plus d in the chosen axis
            Eigen::Vector3d plane_point = Eigen::Vector3d::Zero();
            plane_point(axis_index) = d;

            // Collect intersection points
            std::vector<Eigen::Vector3d> intersection_points;
            intersection_points.reserve(F.rows()*2); // A guess (2 points per face intersection)

            for(int f = 0; f < F.rows(); ++f)
            {
                Eigen::Vector3d v0 = V_rotated.row(F(f,0));
                Eigen::Vector3d v1 = V_rotated.row(F(f,1));
                Eigen::Vector3d v2 = V_rotated.row(F(f,2));

                // Get intersection of this face with the plane
                std::vector<Eigen::Vector3d> pts;
                trianglePlaneIntersection(v0, v1, v2, plane_point, normal, pts);
                if(!pts.empty())
                {
                    // Remove duplicates within this face intersection
                    const double tol = 1e-7;
                    for(const auto &p : pts)
                    {
                        bool found = false;
                        for(const auto &q : intersection_points)
                        {
                            if((p - q).norm() < tol) { found = true; break; }
                        }
                        if(!found) intersection_points.push_back(p);
                    }
                }
            } // end for faces

            if(intersection_points.size() < 3) 
                continue;  // Not enough points to form an area

            // 6) Project intersection points to 2D
            Eigen::MatrixXd pts2D(intersection_points.size(), 2);
            for(size_t i = 0; i < intersection_points.size(); ++i)
            {
                const Eigen::Vector3d &p = intersection_points[i];
                if(axis_index == 0)       // normal = (1, 0, 0)
                {
                    pts2D(i,0) = p.y();
                    pts2D(i,1) = p.z();
                }
                else if(axis_index == 1)  // normal = (0, 1, 0)
                {
                    pts2D(i,0) = p.x();
                    pts2D(i,1) = p.z();
                }
                else                     // normal = (0, 0, 1)
                {
                    pts2D(i,0) = p.x();
                    pts2D(i,1) = p.y();
                }
            }

            // 7) Use CGAL to compute the 2D convex hull
            std::vector<Point_2> cgal_points;
            cgal_points.reserve(pts2D.rows());

            for(int i = 0; i < pts2D.rows(); ++i)
                cgal_points.emplace_back(pts2D(i,0), pts2D(i,1));

            std::vector<Point_2> hull;
            hull.reserve(cgal_points.size());
            CGAL::convex_hull_2(cgal_points.begin(), cgal_points.end(),
                                std::back_inserter(hull));

            if(hull.size() < 3)
                continue;

            // 8) Compute area of that hull
            double area = polygonArea2D(hull);

            // 9) Track maximum cross-sectional area
            if(area > max_area)
            {
                max_area = area;
                optimal_normal = normal;
                optimal_d = d;
                optimal_axis = axis_index;
            }
        } // end for d_samples
    } // end for each axis

    if(max_area < 0.0)
        throw std::runtime_error("Failed to find a valid cutting plane.");

    // 10) Build final plane point
    Eigen::Vector3d optimal_point = Eigen::Vector3d::Zero();
    optimal_point(optimal_axis) = optimal_d;

    // 11) Partition faces
    std::vector<int> faces_side1, faces_side2;
    faces_side1.reserve(F.rows());
    faces_side2.reserve(F.rows());

    const double tol = 1e-6;
    for(int f = 0; f < F.rows(); ++f)
    {
        Eigen::Vector3d v0 = V_rotated.row(F(f,0));
        Eigen::Vector3d v1 = V_rotated.row(F(f,1));
        Eigen::Vector3d v2 = V_rotated.row(F(f,2));
        double s0 = (v0 - optimal_point).dot(optimal_normal);
        double s1 = (v1 - optimal_point).dot(optimal_normal);
        double s2 = (v2 - optimal_point).dot(optimal_normal);

        int pos_count = 0, neg_count = 0;
        auto check_sign = [&](double val){
            if(val >= -tol) ++pos_count; else ++neg_count;
        };
        check_sign(s0); check_sign(s1); check_sign(s2);

        if(pos_count >= neg_count)
            faces_side1.push_back(f);
        else
            faces_side2.push_back(f);
    }

    // 12) Return the result
    CutPlaneResult result;
    result.faces_side1 = faces_side1;
    result.faces_side2 = faces_side2;
    result.optimal_normal = optimal_normal;
    result.optimal_point  = optimal_point;
    return result;
}
 


    // THIS IS CURRENTLY DEBUGGING WRAPPER FOR PAMO AND AGG_ALL, PROCEED WITH CAUTION
  std::vector<Component> unwrap_aligning_plane(const Eigen::MatrixXd &V,   const  Eigen::MatrixXi &F,double threshold, bool check_overlap, int chart_limit){
    MatrixX3R FN;  
    std::vector<int> faceAssignment;
    std::vector<std::vector<int>> faceAdj;

    Eigen::MatrixXd V_rotated = V;
    std::vector<std::vector<double>> edge_lengths;
    prepareOBBData(V_rotated, F, FN, faceAssignment, faceAdj, edge_lengths);

    CutPlaneResult cut_plane_result = find_max_cut_plane_box_multiplane(V_rotated, F);

    // Partition the mesh faces with respect to the optimal plane.
    std::vector<int> faces_side1 = cut_plane_result.faces_side1;
    std::vector<int> faces_side2 = cut_plane_result.faces_side2;

    
    std::vector<std::vector<int>> partedFaces = { faces_side1, faces_side2 };
    smoothComponentEdge(partedFaces, faceAdj);
    // Create two components from the partition.

    std::vector<Component> ret_comp(2);
    if (process_submesh(partedFaces[0], V, F, 0, ret_comp[0],check_overlap) != 0)
    {
        return {};

    }
    if (process_submesh(partedFaces[1], V, F, 1, ret_comp[1],check_overlap) != 0)
    {
        return {};
    }



    return ret_comp;
    
}
  

int process_submesh(std::vector<int> faces, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, int id, Component &comp, bool check_overlap, cudaStream_t stream){
    if(faces.size() == 0){
        return -1;
    }
    Eigen::MatrixXd Vc;
    Eigen::MatrixXi Fc;
    Eigen::MatrixXd UVc;

    std::vector<int> local2global;
    ExtractSubmesh(faces,F, V, Fc, Vc, &local2global);

    
    EASY_BLOCK("Unwrapin submesh", profiler::colors::Orange);
    auto start = std::chrono::high_resolution_clock::now();
    int lscm_success = -1;
    double unwrap_distortion;
    Eigen::MatrixXd V_simp;
    Eigen::MatrixXi F_simp;
    if (CONFIG_unwrapPamo && stream != nullptr){
        V_simp = Vc;
        F_simp = Fc;
        lscm_success = unwrap_pamo(V_simp, F_simp, UVc, stream);
        if (lscm_success == 0){
            unwrap_distortion = calculate_distortion_area(V_simp, F_simp, UVc);
        }

    }else{
        lscm_success = unwrap(Vc, Fc, UVc);
        if (lscm_success == 0){
            unwrap_distortion = calculate_distortion_area(Vc, Fc, UVc);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Write to log file
    if(CONFIG_saveStuff){
        std::ofstream logfile("abf_timings.txt", std::ios::app);
        if (logfile.is_open()) {
            logfile << Vc.rows() << ", " << Fc.rows() << ", " << "Two part" <<  " : " << elapsed.count() << "\n";
        }
    }
    EASY_END_BLOCK; 
    // save mesh

    if(lscm_success != 0){
        if(CONFIG_verbose)
            std::cout << "LSCM projection failed in UnwrapPlane for component: "<< std::to_string(id) << std::endl;
        return -1;
    }

    comp.index = id;
    comp.faces = faces; 

    if(CONFIG_unwrapPamo){
        comp.V = V_simp;
        comp.F = F_simp;
    }else{
        comp.V = Vc;
        comp.F = Fc;
    }
    if (check_overlap){
        std::vector<std::pair<int, int>> overlap_triangles;
        auto num_overlaps = computeOverlapingTrianglesFast(UVc, comp.F, overlap_triangles);
        if (num_overlaps > 0){
            if(CONFIG_verbose) std::cout << overlap_triangles.size() <<  " Overlapping triangles found in UnwrapSubmesh for component: " <<  std::to_string(id) << std::endl;
            return -1;
        }
    }
    
    comp.V_original = Vc;
    comp.F_original = Fc;
    comp.source_vid_original = local2global;
    // pamo may have replaced V with a simplified mesh
    if (comp.V.rows() == (Eigen::Index)local2global.size())
        comp.source_vid = local2global;
    comp.UV = UVc;
    comp.distortion = unwrap_distortion;
    return 0;
}

  