
#ifndef Component_H
#define Component_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>

#include <iostream>
#include <igl/write_triangle_mesh.h>
#include <chrono>

// LSCM
#include "Mesh.h"
#include "FormTrait.h"
#include "LSCM.h"
#include "OpenABF/OpenABF.hpp"

#include <easy/profiler.h>
#include <easy/arbitrary_value.h> // EASY_VALUE, EASY_ARRAY are defined here

#include  "Config.h"
#include "stream_pool.h"
#include <cuda_runtime.h>

#define DISTORTION_THRESHOLD 0.5


typedef Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> MatrixX3R;
typedef Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> MatrixX3I;


struct ComponentEdge
{
    int comp_u;
    int comp_v;
    int weight; // e.g. how many face-face edges connect comp_u to comp_v
};


struct Component
{
    int index;
    std::vector<int> faces;
    int cube_face_idx;
    Eigen::MatrixXd face_normals;
    Eigen::RowVector3d avg_normal;
    
    MatrixX3I F;
    MatrixX3R V;
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> UV;

    // These holds the original mesh if the component is simplified
    MatrixX3I F_original;
    MatrixX3R V_original;

    // source_vid[i] is the processed-input vertex index V.row(i) came from
    std::vector<int> source_vid;
    std::vector<int> source_vid_original;

    

    int original_vertex_count = 0;
    double distortion = 1000.0;

    Component() = default;

    // Constructor with F and V as optional parameters
    Component(int idx, const std::vector<int> &fcs, int cube_idx, const Eigen::MatrixXd &fn, const MatrixX3I& faces_matrix = MatrixX3I(), const MatrixX3R& vertices_matrix = MatrixX3R())
            : index(idx),  faces(fcs), cube_face_idx(cube_idx), face_normals(fn), F(faces_matrix), V(vertices_matrix), F_original(faces_matrix), V_original(vertices_matrix)
    {
        avg_normal = face_normals.colwise().mean().normalized();
        original_vertex_count = V.rows();
    }
    /**
     * @brief Concatenate another Component into this one to create a new Component.
     *
     * - Concats `V`, `F`, and `UV`.
     * - Offsets the faces in `other.F` by the row-count of `V`.
     * - Appends `face_normals` and recalculates `avg_normal`.
     * - Merges `faces` and sets `distortion` to the max of both.
     */
    Component operator+(const Component &other) const;

    void save_mesh(const std::string &filename, bool save_uv = false) const
    {
        if (save_uv)
        {
            if (UV.rows() > 0)
            {
                Eigen::MatrixXi FNc; // (empty) face normals for OBJ
                Eigen::MatrixXd Nc; // (empty) vertex normals for OBJ
                // The face indices for UV coordinates are assumed to be the same as for geometry.
                Eigen::MatrixXi FUVc = F; 
                if (!igl::writeOBJ(filename, V, F, Nc, FNc, UV, FUVc)) {
                    std::cerr << "Failed to write mesh OBJ: " << filename << std::endl;
                } else {
                    std::cout << "Wrote mesh OBJ with UV: " << filename << std::endl;
                }
            }else{
                std::cerr << "UV is empty, cannot save mesh with UV coordinates, trying to save without uv instead." << std::endl;
            }
        }

        igl::writeOBJ(filename, V, F);
        
    }
    
    
};

inline void writetotext( const std::string &text)
{
    std::ofstream out("profile.txt", std::ios::app);
    if (out)
    {
        out << text << std::endl;
    }
}

extern double total_time_spent_lscm;

using ABF = OpenABF::ABFPlusPlus<double>;
using ABFLSCM = OpenABF::AngleBasedLSCM<double, ABF::Mesh>;

int lscm_unwrap(const Eigen::MatrixXd V,
    const Eigen::MatrixXi F,
    Eigen::MatrixXd &UVc);


extern double total_time_spent_abf;

int abf_unwrap(const Eigen::MatrixXd V,
                 const Eigen::MatrixXi F,
                 Eigen::MatrixXd &UVc,
                const int num_iter = 10);


// extern std::string lscm_method;
int unwrap(const Eigen::MatrixXd V,
                 const Eigen::MatrixXi F,
                 Eigen::MatrixXd &UVc);

int abf_unwrap_fix_boundary(const Eigen::MatrixXd V,
                 const Eigen::MatrixXi F,
                 Eigen::MatrixXd &UVc,
                const int num_iter,
                const std::unordered_map<std::size_t, std::pair<double,double>> pinned_uvs);

int unwrap_pamo( Eigen::MatrixXd& V,
                  Eigen::MatrixXi& F,
                 Eigen::MatrixXd &UVc,
                 cudaStream_t stream = nullptr);

std::vector<int> findConnectedComponent( int startFace, const std::vector<std::vector<int>> &faceAdj, const std::vector<bool> &visitedMask, std::vector<bool> &globalVisited);


void ExtractSubmesh(const std::vector<int>& faces_in_comp, const Eigen::MatrixXi& F, const Eigen::MatrixXd& V, Eigen::MatrixXi& Fc, Eigen::MatrixXd& Vc, std::vector<int>* local2global = nullptr);
std::unordered_map<std::size_t, std::pair<double,double>> boundary_uv_dict(const Eigen::MatrixXd&  V,const Eigen::MatrixXi& F,const Eigen::MatrixXd&  Vs,    const Eigen::MatrixXi&  Fs,    const Eigen::MatrixXd&  UVs,    double tol = 1e-8);

std::vector<std::vector<std::pair<int,int>>> buildComponentAdjacency(
    const std::vector<std::vector<int>> &faceAdj,         // face-level adjacency
    const std::map<int, int> &faceToComponent,              // face -> component mapping
    const std::vector<Component> &componentsMap,              
    int numComponents
);

std::vector<std::vector<std::pair<int,double>>> buildComponentAdjacencyEdgeLength(
    const std::vector<std::vector<int>> &faceAdj,         // face-level adjacency
    const std::map<int, int> &faceToComponent,              // face -> component mapping
    const std::vector<Component> &componentsMap,              
    int numComponents,
    const std::vector<std::vector<double>> &edge_lengths
);

std::vector<int> largestComponent(const std::vector<std::vector<int>> &components);

// template <typename T>
int ComputeOriginalUV(Component &component);

#endif