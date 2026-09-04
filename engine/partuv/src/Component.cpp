// main.cpp
#include "Component.h"
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <set>

#include <cassert>

#include <igl/doublearea.h>
#include "Distortion.h"
#include <igl/write_triangle_mesh.h>
#include <igl/read_triangle_mesh.h>

#include "CuMeshSimplifier.h"
#include "Config.h"

#include <filesystem>


#include <igl/boundary_loop.h>
#include <igl/AABB.h>

#include "stream_pool.h"
#include "triangleHelper.hpp"


double total_time_spent_lscm = 0.0;
double total_time_spent_abf = 0.0;
std::string lscm_method = "abf";
// BFS (or DFS) to find connected components in the face graph
std::vector<int> findConnectedComponent(
    int startFace,
    const std::vector<std::vector<int>> &faceAdj,
    const std::vector<bool> &visitedMask,
    std::vector<bool> &globalVisited)
{
    std::vector<int> component;
    std::queue<int> Q;
    Q.push(startFace);
    globalVisited[startFace] = true;

    while(!Q.empty())
    {
        int f = Q.front();
        Q.pop();
        component.push_back(f);
        // Visit neighbors
        for(int nbr : faceAdj[f])
        {
            if(!globalVisited[nbr] && !visitedMask[nbr])
            {
                globalVisited[nbr] = true;
                Q.push(nbr);
            }
        }
    }
    return component;
}

void ExtractSubmesh(
    const std::vector<int>& faces_in_comp,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& V,
    Eigen::MatrixXi& Fc,
    Eigen::MatrixXd& Vc,
    std::vector<int>* local2global)
{
    
    // Remove invalid faces
    std::vector<int> valid_faces_comp = faces_in_comp;
    // print invalid faces
    for(int i = 0; i < valid_faces_comp.size(); i++){
        if(valid_faces_comp[i] < 0 || valid_faces_comp[i] >= F.rows()){
            std::cout << "Invalid face: " << valid_faces_comp[i] << std::endl;
        }
    }
    
    
    
    valid_faces_comp.erase(
        std::remove_if(
            valid_faces_comp.begin(), valid_faces_comp.end(),
            [&F](int face) { return  face < 0 || face >= F.rows(); }
        ),
        valid_faces_comp.end()
    );




    if (faces_in_comp.size() != valid_faces_comp.size())
    {
        std::cerr << "Warning: " << faces_in_comp.size() - valid_faces_comp.size() << " invalid faces found in the component.\n";
    }
    int num_faces = valid_faces_comp.size();
    
    // Step 1: Collect all unique vertex indices from the given faces
    std::set<int> verts_in_comp_set;

    for (int f_i : valid_faces_comp)
    {
        verts_in_comp_set.insert(F(f_i, 0));
        verts_in_comp_set.insert(F(f_i, 1));
        verts_in_comp_set.insert(F(f_i, 2));
    }

    // Step 2: Create a mapping from global vertex indices to local submesh vertex indices
    std::unordered_map<int, int> global2local;
    global2local.reserve(verts_in_comp_set.size());
    int local_index = 0;
    if (local2global)
    {
        local2global->clear();
        local2global->reserve(verts_in_comp_set.size());
    }
    for (int vi : verts_in_comp_set)
    {
        global2local[vi] = local_index++;
        if (local2global) local2global->push_back(vi);
    }

    // Step 3: Build the submesh vertex matrix (Vc)
    Vc.resize(verts_in_comp_set.size(), V.cols());
    for (const auto& kv : global2local)
    {
        int old_idx = kv.first;
        int new_idx = kv.second;
        Vc.row(new_idx) = V.row(old_idx);
    }

    // Step 4: Build the submesh face matrix (Fc) with re-indexed vertex indices
    Fc.resize(valid_faces_comp.size(), F.cols());
    for (size_t i = 0; i < valid_faces_comp.size(); ++i)
    {
        int f_old = valid_faces_comp[i];
        // if (f_old < 0 || f_old >= F.rows()) continue;
        for (int j = 0; j < F.cols(); ++j)
        {
            int global_vertex = F(f_old, j);
            Fc(i, j) = global2local[global_vertex];
        }
    }
}


std::unordered_map<std::size_t, std::pair<double,double>> boundary_uv_dict(const Eigen::MatrixXd&  V,
                 const Eigen::MatrixXi&  F,
                 const Eigen::MatrixXd&  Vs,
                 const Eigen::MatrixXi&  Fs,
                 const Eigen::MatrixXd&  UVs,
                 double tol)
{
    if(UVs.rows() != Vs.rows()){
        std::cerr << "UVs and Vs have different number of rows" << std::endl;
        return std::unordered_map<std::size_t, std::pair<double,double>>();
    }
    // ---------------- 1. boundary loops ----------------
    Eigen::VectorXi bnd_orig, bnd_simp;
    igl::boundary_loop(F,  bnd_orig);
    igl::boundary_loop(Fs, bnd_simp);

    // ---------------- 2. KD-tree on original boundary --
    Eigen::MatrixXd Vb(bnd_orig.size(),3);
    for (int i = 0; i < bnd_orig.size(); ++i)
        Vb.row(i) = V.row(bnd_orig[i]);

    // ► element list: one index per point
    Eigen::MatrixXi Eb(bnd_orig.size(),1);
    for (int i = 0; i < bnd_orig.size(); ++i)  Eb(i,0) = i;

    igl::AABB<Eigen::MatrixXd,3> tree;
    tree.init(Vb,Eb);

    // ---------------- 3. build the map -----------------
    std::unordered_map<std::size_t, std::pair<double,double>> fixed_uv;
    fixed_uv.reserve(bnd_orig.size());

    for (int j = 0; j < bnd_simp.size(); ++j) {
        const auto& p = Vs.row(bnd_simp[j]);

        int    idx_orig;
        Eigen::Matrix<double,1,3> closest_p;
        double   sqr_d = tree.squared_distance(Vb,Eb,p,idx_orig,closest_p);

        if (sqr_d > tol ) {
            // throw std::runtime_error("Boundary mismatch exceeds tolerance");
            if(CONFIG_verbose)
                std::cerr << "Boundary mismatch exceeds tolerance: " << sqr_d << std::endl;
            tol = sqr_d;
        }

        std::size_t v_orig = static_cast<std::size_t>(bnd_orig[idx_orig]);


        fixed_uv.emplace(v_orig,
                         std::make_pair(UVs(bnd_simp[j], 0),
                                        UVs(bnd_simp[j], 1)));
    }
    return fixed_uv;
}

// Build adjacency between components (charts), returning a structure that allows you to find neighbors
std::vector<std::vector<std::pair<int,int>>> buildComponentAdjacency(
    const std::vector<std::vector<int>> &faceAdj,         // face-level adjacency
    const std::map<int, int> &faceToComponent,              // face -> component mapping
    const std::vector<Component> &componentsMap,              
    int numComponents
)
{
    // We will store the adjacency in a format:
    //   compAdjList[u] = { {v, weight}, {v2, weight2}, ... }
    // meaning there's an edge u->v with a certain 'weight'.
    std::vector<std::vector<std::pair<int,int>>> compAdjList(numComponents);

    // A helper map to store edges: (minComp, maxComp) -> weight
    // to avoid duplicating edges in both directions.
    std::map<std::pair<int,int>, int> edgeMap;

    // For each face, look at its adjacent faces
    // If adjacency crosses components, increment the edge weight for those components

    for(Component component : componentsMap)
    {
    
        for(int f = 0; f < component.faces.size(); ++f)
        {
            int comp_u = component.faces[f];
            for(int n = 0; n < faceAdj[comp_u].size(); ++n)
            {
                int fNeighbor = faceAdj[comp_u][n];
                auto it = faceToComponent.find(fNeighbor);
                if(it == faceToComponent.end())
                    continue; // skip if not found

                int comp_v_index = it->second;
                if(component.index != comp_v_index)
                {
                    auto key = std::minmax(component.index, comp_v_index);
                    edgeMap[key]++;
                }
            }
        }
    }

    // Now fill compAdjList from edgeMap
    for(const auto &kv : edgeMap)
    {
        int u = kv.first.first;
        int v = kv.first.second;
        int w = kv.second;
        compAdjList[u].push_back({v, w});
        compAdjList[v].push_back({u, w});
    }

    return compAdjList;
}




// Build adjacency between components (charts), returning a structure that allows you to find neighbors
std::vector<std::vector<std::pair<int,double>>> buildComponentAdjacencyEdgeLength(
    const std::vector<std::vector<int>> &faceAdj,         // face-level adjacency
    const std::map<int, int> &faceToComponent,              // face -> component mapping
    const std::vector<Component> &componentsMap,              
    int numComponents,
    const std::vector<std::vector<double>> &edge_lengths
)
{
    // We will store the adjacency in a format:
    //   compAdjList[u] = { {v, weight}, {v2, weight2}, ... }
    // meaning there's an edge u->v with a certain 'weight'.
    std::vector<std::vector<std::pair<int,double>>> compAdjList(numComponents);

    // A helper map to store edges: (minComp, maxComp) -> weight
    // to avoid duplicating edges in both directions.
    std::map<std::pair<int,int>, double> edgeMap;

    // For each face, look at its adjacent faces
    // If adjacency crosses components, increment the edge weight for those components

    for(Component component : componentsMap)
    {
    
        for(int f = 0; f < component.faces.size(); ++f)
        {
            int comp_u = component.faces[f];
            for(int n = 0; n < faceAdj[comp_u].size(); ++n)
            {
                int fNeighbor = faceAdj[comp_u][n];
                auto it = faceToComponent.find(fNeighbor);
                if(it == faceToComponent.end())
                    continue; // skip if not found

                int comp_v_index = it->second;
                if(component.index != comp_v_index)
                {
                    auto key = std::minmax(component.index, comp_v_index);
                    edgeMap[key] = edgeMap[key] + (edge_lengths.size() == 0 ? 1 : edge_lengths[comp_u][n]);
                }
            }
        }
    }

    // Now fill compAdjList from edgeMap
    for(const auto &kv : edgeMap)
    {
        int u = kv.first.first;
        int v = kv.first.second;        
        double w = kv.second;
        compAdjList[u].push_back({v, w});
        compAdjList[v].push_back({u, w});
    }

    return compAdjList;
}




// Choose whichever connected component is largest (by number of faces)
std::vector<int> largestComponent(const std::vector<std::vector<int>> &components)
{
    size_t largestIndex = 0;
    size_t largestSize  = 0;
    for(size_t i = 0; i < components.size(); ++i)
    {
        if(components[i].size() > largestSize)
        {
            largestSize = components[i].size();
            largestIndex = i;
        }
    }
    return (largestSize > 0) ? components[largestIndex] : std::vector<int>();
}





int lscm_unwrap(const Eigen::MatrixXd V,
    const Eigen::MatrixXi F,
    Eigen::MatrixXd &UVc)
{
    EASY_BLOCK("LSCM"); 


    // --- Actual function body ---
    MeshLib::Mesh mesh;
    mesh.from_igl(V, F);

    MeshLib::FormTrait formTrait(&mesh);
    MeshLib::LSCM lscm(&mesh);
    int success = lscm.project(UVc);
    // ----------------------------


    // // writetotext("LSCM took " + std::to_string(elapsed.count()) + " seconds.");
    return success;
}



template <typename Scalar>
inline Eigen::VectorXi
RemoveUnreferencedVertices(Eigen::Matrix<Scalar, -1, -1> &V,
                           Eigen::MatrixXi               &F)
{
    const int nV = V.rows();
    const int d  = V.cols();
    const int nF = F.rows();
    const int k  = F.cols();

    // 1) Mark used vertices
    Eigen::VectorXi used = Eigen::VectorXi::Zero(nV);
    for (int i = 0; i < nF; ++i) {
        for (int j = 0; j < k; ++j) {
            const int idx = F(i, j);
            if (idx < 0) continue;                // allow sentinel if present
            if (idx >= nV) throw std::runtime_error("F index out of range");
            used(idx) = 1;
        }
    }

    // 2) Early exit if nothing to remove
    const int nKeep = used.sum();
    Eigen::VectorXi removed(nV - nKeep);
    if (nKeep == nV) {
        removed.resize(0);
        return removed;
    }

    // 3) Build remaps
    Eigen::VectorXi old2new = Eigen::VectorXi::Constant(nV, -1);
    Eigen::VectorXi new2old(nKeep);
    for (int i = 0, cur = 0; i < nV; ++i) {
        if (used(i)) { old2new(i) = cur; new2old(cur) = i; ++cur; }
    }

    // 4) Compact V (stable)
    Eigen::Matrix<Scalar, -1, -1> Vnew(nKeep, d);
    for (int r = 0; r < nKeep; ++r) Vnew.row(r) = V.row(new2old(r));
    V.swap(Vnew);

    // 5) Remap F
    for (int i = 0; i < nF; ++i) {
        for (int j = 0; j < k; ++j) {
            int idx = F(i, j);
            if (idx < 0) continue;
            const int m = old2new(idx);
            if (m < 0) throw std::logic_error("Face refers to removed vertex");
            F(i, j) = m;
        }
    }


    for (int i = 0, r = 0; i < nV; ++i) if (!used(i)) removed(r++) = i;
    return removed;
}
int abf_unwrap_pamo( Eigen::MatrixXd &V,
                  Eigen::MatrixXi &F,
                 Eigen::MatrixXd &UVc,
                const int num_iter,
                cudaStream_t stream){


            EASY_BLOCK("ABF++ PAMO");

            EASY_BLOCK("Simplify",profiler::colors::Grey);
            // int cpuid = omp_get_thread_num();
            // cudaSetDevice(cpuid);
            auto [V_simp, F_simp] = CuMeshSimplifier::simplify(V.cast<float>(), F.cast<int>(), 0.0001f, 1000, stream);
            auto removed = RemoveUnreferencedVertices(V_simp, F_simp);
            if(removed.size() > 0)
            {
                std::cout << "PAMO created " << removed.size() << " unreferenced vertices!!! " << std::endl;   
            }


            EASY_END_BLOCK;

            // Cast V_simp from float to double
            Eigen::MatrixXd V_simp_double = V_simp.cast<double>();
            
            int success = abf_unwrap(V_simp_double, F_simp, UVc, num_iter);


            V = V_simp_double;
            F = F_simp;

            EASY_END_BLOCK;
            return success;
            
        }


int abf_unwrap(const Eigen::MatrixXd V,
                 const Eigen::MatrixXi F,
                 Eigen::MatrixXd &UVc,
                const int num_iter)
{

    EASY_BLOCK("ABF++"); 
    EASY_VALUE("# F", F.rows());    
    EASY_VALUE("# V", V.rows());



    auto startAbf = std::chrono::high_resolution_clock::now();

    auto mesh = ABF::Mesh::New();
    for (int i = 0; i < V.rows(); i++)
    {
        mesh->insert_vertex(V(i, 0), V(i, 1), V(i, 2));
    }
    for (int i = 0; i < F.rows(); i++)
    {
        mesh->insert_face(F(i, 0), F(i, 1), F(i, 2));
    }
    


    double max_length = 0;
    auto loops = mesh->find_boundary_loops();
    if(loops.size() == 0)
    {
        max_length = 0.001;
        return -2;
    }
    else{
        auto maxLoopIdx = mesh->find_longest_boundary_loop(loops, &max_length);
    }

    std::size_t iters{0};
    double grad{OpenABF::INF<double>};

    EASY_BLOCK("ABF++"); 
    ABF::Compute(mesh, iters, grad, num_iter);
    EASY_END_BLOCK;


    EASY_BLOCK("ABF++ LSCM");
    int success = ABFLSCM::Compute(mesh);
    EASY_END_BLOCK;

    if (success != 0)
    {
        return -1;
    }
    
    UVc.resize(V.rows(), 2);


    for (const auto& v : mesh->vertices()) {
        UVc(v->idx, 0) = v->pos[0];
        UVc(v->idx, 1) = v->pos[1];
    }

    auto endAbf = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endAbf - startAbf;
    total_time_spent_abf += elapsed.count();

    if(CONFIG_saveStuff ){
        // std::string output_dir = "/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/meshes/overlapped_meshes_2";
        std::string output_dir = CONFIG_outputPath + "/ABF_parts";
        std::filesystem::create_directories(output_dir);
        int folder_file_count = std::distance(std::filesystem::directory_iterator(output_dir), std::filesystem::directory_iterator{});
        igl::write_triangle_mesh(output_dir + "/part_" + std::to_string(folder_file_count) + ".obj", V, F);
        
        double distortion = calculate_distortion_area(V, F, UVc);
        std::ofstream logfile(output_dir + "/abf_timings.txt", std::ios::app);
        if (logfile.is_open()) {
            logfile << "index: " << folder_file_count << " , " << "time : " << elapsed.count() << " distortion: " << distortion << "\n";
        }
    }
    return 0;
    
}


// int abf_unwrap(const Eigen::MatrixXd V,
//               const Eigen::MatrixXi F,
//               Eigen::MatrixXd &UVc,
//               const int num_iter)
// {
//     EASY_BLOCK("LABF"); 
//     // EASY_VALUE("# F", F.rows());    
//     // EASY_VALUE("# V", V.rows());

//     // auto mesh = ABF::Mesh::New();
//     // for (int i = 0; i < V.rows(); i++)

//     LinABF solver(V, F);
//     UVc = solver.getUVs();
//     EASY_END_BLOCK;
//     return solver.success;
// }



int abf_unwrap_fix_boundary(const Eigen::MatrixXd V,
                 const Eigen::MatrixXi F,
                 Eigen::MatrixXd &UVc,
                const int num_iter,
                const std::unordered_map<std::size_t, std::pair<double,double>> pinned_uvs)
{

    EASY_BLOCK("ABF++"); 
    EASY_VALUE("# F", F.rows());    
    EASY_VALUE("# V", V.rows());



    auto startAbf = std::chrono::high_resolution_clock::now();

    auto mesh = ABF::Mesh::New();
    for (int i = 0; i < V.rows(); i++)
    {
        mesh->insert_vertex(V(i, 0), V(i, 1), V(i, 2));
    }
    for (int i = 0; i < F.rows(); i++)
    {
        mesh->insert_face(F(i, 0), F(i, 1), F(i, 2));
    }

    double max_length = 0;
    auto loops = mesh->find_boundary_loops();
    if(loops.size() == 0)
    {
        max_length = 0.001;
        return -2;
    }
    else{
        auto maxLoopIdx = mesh->find_longest_boundary_loop(loops, &max_length);
    }

    // mesh->fill_holes();
    std::size_t iters{0};
    double grad{OpenABF::INF<double>};

    EASY_BLOCK("ABF++"); 
    ABF::Compute(mesh, iters, grad, num_iter);
    EASY_END_BLOCK;


    EASY_BLOCK("ABF++ LSCM");
    int success = ABFLSCM::Compute_FixBoundary(mesh, pinned_uvs);
    EASY_END_BLOCK;

    if (success != 0)
    {
        return -1;
    }
    
    UVc.resize(V.rows(), 2);


    for (const auto& v : mesh->vertices()) {
        UVc(v->idx, 0) = v->pos[0];
        UVc(v->idx, 1) = v->pos[1];
    }

    auto endAbf = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endAbf - startAbf;
    total_time_spent_abf += elapsed.count();

    if(CONFIG_saveStuff ){
        // std::string output_dir = "/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/meshes/overlapped_meshes_2";
        std::string output_dir = CONFIG_outputPath + "/ABF_parts";
        std::filesystem::create_directories(output_dir);
        int folder_file_count = std::distance(std::filesystem::directory_iterator(output_dir), std::filesystem::directory_iterator{});
        igl::write_triangle_mesh(output_dir + "/part_" + std::to_string(folder_file_count) + ".obj", V, F);
        
        double distortion = calculate_distortion_area(V, F, UVc);
        std::ofstream logfile(output_dir + "/abf_timings.txt", std::ios::app);
        if (logfile.is_open()) {
            logfile << "index: " << folder_file_count << " , " << "time : " << elapsed.count() << " distortion: " << distortion << "\n";
        }
    }
    return 0;
    
}




// extern std::string lscm_method;
int unwrap_pamo( Eigen::MatrixXd &V,
                  Eigen::MatrixXi &F,
                 Eigen::MatrixXd &UVc,
                 cudaStream_t stream)
{
    // check if the method is set
    if (CONFIG_unwrapMethod.empty())
    {
        CONFIG_unwrapMethod = "abf";
    }

    if(stream == nullptr && CONFIG_unwrapPamo){
        stream = StreamPool::getStream(0);
    }
    std::string lscm_method = CONFIG_unwrapMethod;
    auto start = std::chrono::high_resolution_clock::now();
    int success;
    if(lscm_method == "lscm")
    {
        success =  lscm_unwrap(V, F, UVc);
    }
    else if(lscm_method == "abf")
    {
        try {
            if(CONFIG_unwrapPamo && F.rows() > CONFIG_unwrapUsePamoFaceThreshold){
                success =  abf_unwrap_pamo(V, F, UVc, CONFIG_unwrapAbfIters, stream);
            }else{
                success =  abf_unwrap(V, F, UVc, CONFIG_unwrapAbfIters);
            }
        } catch (const std::runtime_error &ex) {
            if(CONFIG_verbose)
                std::cerr << "Caught exception in ABF: " << ex.what() << " , " << std::endl;
            
            success =  -1;
            return success;
        }

        if (success == -1)
        {
            if (CONFIG_verbose)
                std::cerr << "ABF failed, Falling back to LSCM" << std::endl;
            success =  lscm_unwrap(V, F, UVc);
        }
        else if (success == -2)
        {
            if (CONFIG_verbose)
                std::cerr << "Estimated distortion is too large, skipping ABF" << std::endl;
            success =  -1;
        }
    }else{
        std::cerr << "Invalid method: " << lscm_method << std::endl;
        success =  -1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    total_time_spent_lscm += elapsed.count();
    return success;
}


// extern std::string lscm_method;
int unwrap(const  Eigen::MatrixXd V,
                 const  Eigen::MatrixXi F,
                 Eigen::MatrixXd &UVc)
{
    // check if the method is set
    if (CONFIG_unwrapMethod.empty())
    {
        CONFIG_unwrapMethod = "abf";
    }
    std::string lscm_method = CONFIG_unwrapMethod;
    auto start = std::chrono::high_resolution_clock::now();
    int success;
    if(lscm_method == "lscm")
    {
        success =  lscm_unwrap(V, F, UVc);
    }
    else if(lscm_method == "abf")
    {
        try {
            success =  abf_unwrap(V, F, UVc, CONFIG_unwrapAbfIters);
            // Code that may throw
        } catch (const std::runtime_error &ex) {
            if(CONFIG_verbose)
            std::cerr << "Caught exception in ABF: " << ex.what() << " , ";
            
            success =  -1;
        }

        if (success == -1)
        {
            if(CONFIG_verbose)
                std::cerr << "ABF failed, Falling back to LSCM" << std::endl;
            success =  lscm_unwrap(V, F, UVc);
        }
        else if (success == -2)
        {
            if(CONFIG_verbose)
                std::cerr << "Estimated distortion is too large, skipping ABF" << std::endl;
            success =  -1;
        }
    }else{
        std::cerr << "Invalid method: " << lscm_method << std::endl;
        success =  -1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    total_time_spent_lscm += elapsed.count();
    return success;
}






template<typename DerivedA, typename DerivedB>
void appendRows(
    Eigen::PlainObjectBase<DerivedA> &A,
    Eigen::MatrixBase<DerivedB>   const &B)
{
    // 1) Scalar‐type check
    static_assert(
      std::is_same<typename DerivedA::Scalar,
                   typename DerivedB::Scalar>::value,
      "A and B must have the same Scalar type");

    // 2) Column‐count check (either both same fixed, or one/both dynamic)
    constexpr int CA = DerivedA::ColsAtCompileTime;
    constexpr int CB = DerivedB::ColsAtCompileTime;
    static_assert(
      CA == CB || CA == Eigen::Dynamic || CB == Eigen::Dynamic,
      "A and B must have the same number of columns");

    // 3) Remember sizes
    const int rowsA = A.rows();
    const int rowsB = B.rows();

    // 4) Grow A, preserving its old contents
    A.derived().conservativeResize(rowsA + rowsB,
                                   A.cols());

    // 5) Copy B into the newly added block
    A.derived().block(rowsA, 0,
                      rowsB,  A.cols()) = B;
}

template<typename DerivedA, typename DerivedB>
auto concatenateRows(
    const Eigen::MatrixBase<DerivedA> & A,
    const Eigen::MatrixBase<DerivedB> & B)
-> Eigen::Matrix<
      typename DerivedA::Scalar,
      Eigen::Dynamic,
      DerivedA::ColsAtCompileTime>
{
    using Scalar     = typename DerivedA::Scalar;
    constexpr int Cols = DerivedA::ColsAtCompileTime;  // may be Dynamic (-1)

    // build a (A.rows()+B.rows())×A.cols() matrix of the same scalar type
    Eigen::Matrix<Scalar, Eigen::Dynamic, Cols> R(A.rows() + B.rows(),
                                                 A.cols());

    // copy blocks
    R.template block< Eigen::Dynamic, Cols >(0,         0, A.rows(), A.cols()) = A;
    R.template block< Eigen::Dynamic, Cols >(A.rows(), 0, B.rows(), B.cols()) = B;

    return R;
}


Component Component::operator+(const Component &other) const
 {
        // Create a new Component to store the result
        Component result;

        // ---------- Basic data ----------
        // Here we arbitrarily decide to use the `index` of the *left* component,
        // but you can modify this logic as suits your application.
        result.index         = index;
        result.cube_face_idx = cube_face_idx;

        // ---------- Distortion ----------
        // Set to the max of both
        result.distortion = std::max(distortion, other.distortion);

        // ---------- faces ----------
        // If `faces` is just a list of face IDs, we can simply concatenate.
        result.faces = faces; 
        result.faces.insert(result.faces.end(), other.faces.begin(), other.faces.end());

        // ---------- face_normals ----------
        // We assume both Components have face_normals with the same number of columns (usually 3).
        // If not, you may need to handle that case carefully.
        if (face_normals.cols() == other.face_normals.cols() && 
            face_normals.cols() > 0)
        {
            result.face_normals.resize(face_normals.rows() + other.face_normals.rows(),
                                       face_normals.cols());
            // Copy the top part
            result.face_normals.block(0, 0, face_normals.rows(), face_normals.cols()) = face_normals;
            // Copy the bottom part
            result.face_normals.block(face_normals.rows(), 0,
                                      other.face_normals.rows(), other.face_normals.cols()) 
                = other.face_normals;
        }
        else
        {
            // Fallback if they differ or are empty
            result.face_normals = face_normals.rows() > 0 ? face_normals : other.face_normals;
        }

        // After concatenating, compute the new avg_normal if we have any face normals
        if (result.face_normals.size() > 0) {
            result.avg_normal = result.face_normals.colwise().mean().normalized();
        } else {
            result.avg_normal = Eigen::RowVector3d::Zero();
        }

        // ---------- V and F ----------
        // 1) Concatenate the vertex arrays
        const int rowsV1 = V.rows();
        result.V = concatenateRows(V, other.V);

        result.F = concatenateRows(F, (other.F.array() + rowsV1).matrix());

        // ---------- UV ----------
        // We assume a 1:1 mapping between the rows in V and rows in UV. 
        // If your data structure differs, you will need to adjust logic here.
        result.UV = concatenateRows(UV, other.UV);


        result.F_original = concatenateRows(F_original, other.F_original);
        result.V_original = concatenateRows(V_original, other.V_original);
        // component's original mesh).
        result.original_vertex_count = original_vertex_count + other.original_vertex_count;

        // a mismatch on either side would put wrong indices in the result
        if (source_vid.size() == (size_t)V.rows() &&
            other.source_vid.size() == (size_t)other.V.rows())
        {
            result.source_vid = source_vid;
            result.source_vid.insert(result.source_vid.end(),
                                     other.source_vid.begin(), other.source_vid.end());
        }
        if (source_vid_original.size() == (size_t)V_original.rows() &&
            other.source_vid_original.size() == (size_t)other.V_original.rows())
        {
            result.source_vid_original = source_vid_original;
            result.source_vid_original.insert(result.source_vid_original.end(),
                                              other.source_vid_original.begin(),
                                              other.source_vid_original.end());
        }

        // Done
        return result;
    }


/**
 * @brief Computes the UV coordinates for a component using its original mesh data
 * 
 * This function checks if a component has original mesh data (V_original and F_original)
 * and if the UV coordinates need to be recomputed. If the component's UV coordinates
 * don't match the original vertex count, it recomputes them using LSCM projection.
 * 
 * @param component The component for which to compute UV coordinates
 * @return int Status code:
 *         -1: No original mesh data available
 *         -2: UV coordinates already match original vertex count
 *          0: Successfully computed new UV coordinates
 */

int ComputeOriginalUV(Component &component)
{
    if (component.V_original.rows()== 0){
        return -1;
    }

    if (component.UV.rows() == component.V_original.rows()){
        return -2;
    }

    std::unordered_map<std::size_t, std::pair<double,double>>  boundary_uv = boundary_uv_dict(component.V_original, component.F_original, component.V, component.F, component.UV,1e-8);
    component.V = component.V_original;
    component.F = component.F_original;
    component.source_vid = component.source_vid_original;
    Eigen::MatrixXd uv;

    int success = -1;
    try{
        success = abf_unwrap_fix_boundary(component.V, component.F, uv, CONFIG_unwrapAbfIters, boundary_uv);
    }catch(const std::runtime_error &ex){
        std::cerr << "ABF failed, Falling back to LSCM" << std::endl;
        success =  lscm_unwrap(component.V, component.F, uv);
    }

    std::vector<std::pair<int,int>> overlappingTriangles;
    if(success == -1 || computeOverlapingTrianglesFast(uv, component.F,overlappingTriangles)){
        try{
            abf_unwrap(component.V, component.F, uv, CONFIG_unwrapAbfIters);
        }catch(const std::runtime_error &ex){
            std::cerr << "ABF Free Boundary failed, using LSCM, overlap may occur at this point" << std::endl;
            success =  lscm_unwrap(component.V, component.F, uv);
        }
    }
    component.distortion = calculate_distortion_area(component.V, component.F, uv);
    component.UV         = std::move(uv); 

    

    return 0;
}
