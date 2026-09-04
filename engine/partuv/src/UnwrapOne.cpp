#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/per_face_normals.h>
#include <igl/adjacency_list.h>
#include <igl/lscm.h>


// LSCM
#include "Mesh.h"
#include "FormTrait.h"
#include "LSCM.h"
#include "Component.h"
#include "UnwrapMerge.h"
#include "UnwrapBB.h"

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
#include <numeric>
#include <filesystem>

#include "UnwrapBB.h"
#include "merge.h"
#include "IO.h"




#define ENABLE_PROFILING






#include <easy/profiler.h>


using namespace MeshLib;


std::vector<Component> unwrap_aligning_one(const Eigen::MatrixXd &V,   const  Eigen::MatrixXi &F, double threshold, bool check_overlap, int chart_limit){
    Component comp;
    comp.V = V;
    comp.F = F;
    // V is used as-is, so provenance is the identity
    comp.source_vid.resize(V.rows());
    std::iota(comp.source_vid.begin(), comp.source_vid.end(), 0);

    if (CONFIG_verbose)
        std::cout << "Unwrap One: part  With " << V.rows() << " vertices and " << F.rows() << " faces" << std::endl;    

    if (F.rows() == 0){
        std::cerr << "No faces in the mesh." << std::endl;
        return {};
    }
    
    Eigen::MatrixXd UV;
    EASY_BLOCK("Unwrapin One", profiler::colors::Blue);
    auto start = std::chrono::high_resolution_clock::now();
    int success = -1;
    double unwrap_distortion;

    success = unwrap(V , F, UV);
    unwrap_distortion = calculate_distortion_area(V, F, UV);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Write to log file
    if(CONFIG_saveStuff)
    {    std::ofstream logfile("abf_timings.txt", std::ios::app);
        if (logfile.is_open()) {
            logfile << V.rows() << ", " << F.rows() << ", " << "One part" <<  " : " << elapsed.count() << "\n";
        }
    }
    EASY_END_BLOCK; 
    
    if (success != 0){
        std::cerr << "LSCM projection failed." << std::endl;
        return {};
    }else if (check_overlap){
        std::vector<std::pair<int, int>> overlap_triangles;
        auto num_overlaps = computeOverlapingTrianglesFast(UV,  F, overlap_triangles);
        if (num_overlaps > 0){
            if(CONFIG_verbose) std::cout << overlap_triangles.size() <<  " Overlapping triangles found in UnwrapOne " << std::endl;
            // int folder_file_count = std::distance(std::filesystem::directory_iterator("/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/meshes/overlapped_meshes_2"), std::filesystem::directory_iterator{});
            // igl::write_triangle_mesh("/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/meshes/overlapped_meshes_2/mesh_" + std::to_string(folder_file_count) + ".obj", V, F);

            return {};
        }
    }
    comp.UV = UV;
    comp.distortion = unwrap_distortion;
    
    std::vector<Component> components;
    components.push_back(comp);

    return components;

}

