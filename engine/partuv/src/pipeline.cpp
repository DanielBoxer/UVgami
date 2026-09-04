#include "pipeline.h"
#include "IO.h"
#include "Component.h"


#include <igl/edge_flaps.h>
#include <igl/edge_topology.h>
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/facet_components.h>
#include <igl/slice_mask.h>

#include <Eigen/Dense>
#include <filesystem>  

#include <algorithm>
#include <iostream>
#include <limits>
#include <functional>

#include <execution>
#include <vector>
#include <mutex>
#include <iostream>
#include <chrono>
#include <string>

#include "UnwrapBB.h"
#include "UnwrapMerge.h"
#include "UnwrapPlane.h"
#include "UnwrapOne.h"
#include "UnwrapAgg.h"

#include "triangleHelper.hpp"


#include  "Config.h"
#include "Pack.h"

#include <easy/profiler.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <omp.h>
#include <cmath>

#include <iostream>
#include <fstream>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3>                      Mesh;
typedef boost::graph_traits<Mesh>::face_descriptor          face_descriptor;

#define ENABLE_PROFILING


void verify_omp() {
    std::cout.setf(std::ios::boolalpha);

    std::cout << "dynamic=" << static_cast<bool>(omp_get_dynamic())
              << " nested=" << static_cast<bool>(omp_get_nested())
              << " thread_limit=" << omp_get_thread_limit()
              << " max_active_levels=" << omp_get_max_active_levels()
              << '\n';

    #pragma omp parallel
    {
        #pragma omp master
        {
            std::cout << "[level " << omp_get_level()
                      << "] outer team size=" << omp_get_num_threads()
                      << " (max_threads=" << omp_get_max_threads() << ")\n";
            std::cout.flush();
        }

        #pragma omp parallel
        {
            #pragma omp master
            {
                std::cout << "[level " << omp_get_level()
                          << "] inner team size=" << omp_get_num_threads()
                          << " (max_threads=" << omp_get_max_threads()
                          << ", parent_tid=" << omp_get_ancestor_thread_num(1)
                          << ")\n";
                std::cout.flush();
            }
        }
    }
}

namespace
{
    Eigen::MatrixXd gV;
    Eigen::MatrixXi gF;
    std::vector<std::vector<int>> gFaceAdj;
    double gThreshold;
    static int num_parts = -1;
    static std::vector<UVParts> allParts;
    static std::unordered_map<int, int> map_root_to_part;

    static int level = 0;

    static std::ofstream global_log_file;
    static std::string log_node_mesh_files_dir = "";

    static Hierarchy g_hierarchy;

    // emit progress: lines on stdout for the blender add-on's progress bar
    static bool gVisual = false;
    static int gVisualFacesDone = 0;

}
#ifdef ENABLE_PROFILING
using Clock = std::chrono::high_resolution_clock;
static double total_time_spent_s = 0.0;
static double total_time_spent_unwrap = 0.0;

#endif

void clear_global_data(){
    gV.setZero();
    gF.setZero();
    gFaceAdj.clear();
    gThreshold = 0;
    num_parts = -1;
    allParts.clear();
    map_root_to_part.clear();
    level = 0;
    g_hierarchy = Hierarchy();
    gVisual = false;
    gVisualFacesDone = 0;
}

void set_global_mesh(const Eigen::MatrixXd& V,
                     const Eigen::MatrixXi& F,
                     double threshold)
{
    gV = V;
    gF = F;
    gThreshold = threshold;
}

void init_log_files(){
    if (CONFIG_log_traverse_csv){
        global_log_file.open(CONFIG_outputPath + "/traverse_log.csv");
        global_log_file << "node_id,parent_id,distortion,overlaps,num_components,color" << std::endl;
        log_node_mesh_files_dir = CONFIG_outputPath + "/node_mesh_files/";
        std::filesystem::create_directories(log_node_mesh_files_dir);
    }

}

void log_traverse_csv(const NodeRecord *node, const UVParts *uv_parts, int parent_id, bool overlaps, std::string color){
    // CSV row: node_id, parent_id, distortion, overlaps, num_components, color
    if(global_log_file.is_open()){
        double max_distortion = -1;
        if (uv_parts->distortion > 1000){
            for (Component comp : uv_parts->components){
                if(std::isnan(comp.distortion)){
                    comp.distortion = calculate_distortion_area(comp.V, comp.F, comp.UV);
                }
                if(comp.distortion > max_distortion){
                    max_distortion = comp.distortion;
                }
            }
        }else{
            max_distortion = uv_parts->distortion;
        }


        int overlap_count = 0;
        for (Component comp : uv_parts->components){
            if (comp.UV.rows() == 0){
                overlap_count = -1;
                break;
            }
            std::vector<std::pair<int,int>> overlapping_triangles;
            computeOverlapingTrianglesFast(comp.UV, comp.F, overlapping_triangles);
            overlap_count += overlapping_triangles.size();
        }

        global_log_file << node->id << "," << parent_id << "," << max_distortion << "," << overlap_count << "," << uv_parts->num_components << "," << color << std::endl;
        std::string mesh_filename = log_node_mesh_files_dir + std::to_string(node->id) + ".obj";
        if (uv_parts->num_components > 0){
            Component comp = uv_parts->to_components();
            igl::write_triangle_mesh(mesh_filename, comp.V, comp.F);
        }
    }
}


bool check_self_intersection(std::string mesh_filename){
    Mesh mesh;
    if(!PMP::IO::read_polygon_mesh(mesh_filename, mesh) || !CGAL::is_triangle_mesh(mesh))
    {
      std::cerr << "Invalid input." << std::endl;
      return 1;
    }

    std::vector<std::pair<face_descriptor, face_descriptor> > intersected_tris;
    PMP::self_intersections<CGAL::Parallel_if_available_tag>(faces(mesh), mesh, std::back_inserter(intersected_tris));
    std::cout << intersected_tris.size() << " pairs of triangles intersect." << std::endl;
    if(CONFIG_verbose){
        for (const auto& pair : intersected_tris){
            std::cout << "Triangles (" << pair.first << ", " << pair.second << ") intersect." << std::endl;
        }
    }

    return intersected_tris.size() > 0;
}

// make sure V and F is within bounds
bool check_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F){
    if(V.rows() <=3  || F.rows() <= 1){
        return false;
    }
    for (int i = 0; i < F.rows(); i++){
        if(F(i, 0) == F(i, 1) || F(i, 1) == F(i, 2) || F(i, 0) == F(i, 2)){
            return false;
        }
        // check F indices are within bounds
        if(F(i, 0) >= V.rows() || F(i, 1) >= V.rows() || F(i, 2) >= V.rows()){
            return false;
        }
    }

    return true;
}

bool check_2_manifold(Eigen::MatrixXi &F,Eigen::MatrixXd &V){
    auto mesh = ABF::Mesh::New();
    for (int i = 0; i < V.rows(); i++)
    {
        mesh->insert_vertex(V(i, 0), V(i, 1), V(i, 2));
    }
    for (int i = 0; i < F.rows(); i++)
    {
        try
        {
            mesh->insert_face(F(i, 0), F(i, 1), F(i, 2));
        }
        catch (const std::runtime_error &ex)
        {
            // Current our ABF only throws when detecting non-2-manifold mesh
            std::cerr << "Error: " << ex.what() << '\n';
            return false;
        }
    }
    return true;

}


// Function to check if the mesh is valid
bool load_mesh_with_validation(std::string mesh_path, Eigen::MatrixXi &F,Eigen::MatrixXd &V){
    if(!igl::read_triangle_mesh(mesh_path, V, F)){
        std::cerr << "Failed to read mesh: " << mesh_path << std::endl;
        return false;
    }
    if (!check_mesh(V, F)){
        std::cerr << "Invalid mesh: " << mesh_path << " , V or F out of bounds." << std::endl;
        return false;
    }

    if(CONFIG_checkSelfIntersection){
        if(check_self_intersection(mesh_path)){
            std::cerr << "Self-intersection detected in mesh: " << mesh_path << std::endl;
            std::cerr << " toggle checkSelfIntersection in config to ignore."  << std::endl;

            return false;
        }
    }
    if (CONFIG_checkNon2Manifold){
        if(!check_2_manifold(F, V)){
            std::cerr << "Non-2-manifold mesh detected in mesh: " << mesh_path << std::endl;
            std::cerr << " toggle checkNon2Manifold in config to ignore."  << std::endl;
            return false;
        }
    }
    return true;
}


// each extraction level lifts source_vid one space up through its local2global
static void remap_source_vids(Component &comp, const std::vector<int> &l2g)
{
    auto remap = [&l2g](std::vector<int> &vids) {
        for (int &vid : vids)
            vid = (vid >= 0 && vid < (int)l2g.size()) ? l2g[vid] : -1;
    };
    remap(comp.source_vid);
    remap(comp.source_vid_original);
}

static void remap_source_vids(UVParts &part, const std::vector<int> &l2g)
{
    for (Component &comp : part.components)
        remap_source_vids(comp, l2g);
}

static void remap_source_vids(std::vector<UVParts> &parts, const std::vector<int> &l2g)
{
    for (UVParts &part : parts)
        remap_source_vids(part, l2g);
}

std::vector<std::vector<UVParts>>  get_uv_wrapper( const Eigen::MatrixXi &F,const Eigen::MatrixXd &V, double threshold,bool check_overlap, bool use_full, int chart_limit){


    if(!check_mesh(V, F)){
        if(F.rows() == 1){
            std::vector<Component> components = unwrap_aligning_one(V, F, threshold, check_overlap, 1);
            return {{UVParts(components)}};
        }else{
            std::cerr << "Invalid mesh. Returning empty UVParts." << std::endl;
            return {{UVParts(MAX_CHARTS)}};
        }
    }
    // Compute the connected components of the faces
    Eigen::VectorXi component;
    igl::facet_components(F, component);
    int num_components = component.maxCoeff() + 1;
    std::vector<std::vector<UVParts>> allUVParts(num_components);
    std::vector<std::vector<int>> allFaces(num_components);
    for(int i = 0; i < component.size(); ++i) {
        int c = component[i];
        allFaces[c].push_back(i);
    }

    for(int c = 0; c < num_components; ++c) {
        std::vector<int> component_faces = allFaces[c];
        Eigen::MatrixXd Vc; Eigen::MatrixXi Fc;

        std::vector<int> local2global;
        ExtractSubmesh(component_faces, F, V, Fc, Vc, &local2global);
        std::vector<UVParts> uv_parts;
        if(use_full){
            uv_parts = get_uv(Fc, Vc, threshold, check_overlap, use_full, chart_limit);
        }else{
            uv_parts= unwrap_aligning_Agglomerative_all(Vc, Fc, threshold, check_overlap, chart_limit, /*check_break*/true);
        }
        remap_source_vids(uv_parts, local2global);

        allUVParts[c] = uv_parts;

    }

    return allUVParts;

}

std::vector<UVParts> get_uv( const Eigen::MatrixXi &F,const Eigen::MatrixXd &V, double threshold,bool check_overlap, bool use_full, int chart_limit)
{
    EASY_FUNCTION();
    #ifdef ENABLE_PROFILING

    if(!check_mesh(V, F)){
        if(F.rows() == 1){
            std::vector<Component> components = unwrap_aligning_one(V, F, threshold, check_overlap, 1);
            return {UVParts(components)};
        }else{
            std::cerr << "Invalid mesh. Returning empty UVParts." << std::endl;
            return {UVParts(MAX_CHARTS)};
        }
    }
    auto start = Clock::now();
    #endif

    std::vector<std::pair<std::string, UnwrapFunction>> unwrap_methods;
    std::vector<std::string> unwrap_method_names = {"unwrap_aligning_merge"};

    std::vector<UVParts> all_candidates;

    for (const auto &methodName : unwrap_method_names) {
        auto it = available_unwrap_methods.find(methodName);
        if (it != available_unwrap_methods.end()) {
            unwrap_methods.emplace_back(methodName, it->second);
        }
        else {
            std::cerr << "Warning: Unknown unwrap method (simple): " << methodName << std::endl;
        }
    }
    auto start_unwrap = Clock::now();

        
    std::mutex output_mutex;
    std::mutex time_mutex;
    
    double local_total_time = 0.0;

    // Parallel processing of each unwrap method.
    std::for_each(std::execution::par, unwrap_methods.begin(), unwrap_methods.end(),
        [&](const std::pair<std::string, UnwrapFunction>& method_pair) {
            const std::string &func_name = method_pair.first;
            const UnwrapFunction &method = method_pair.second;
            {
                std::lock_guard<std::mutex> lock(output_mutex);
            }      
            
            auto start_unwrap = Clock::now();
            std::vector<Component> components = method(V, F, threshold, false, chart_limit);

            auto end_unwrap = Clock::now();
            std::chrono::duration<double> elapsed_unwrap = end_unwrap - start_unwrap;
            
            {
                std::lock_guard<std::mutex> lock(time_mutex);
                local_total_time += elapsed_unwrap.count();
            }
            
            int total_faces = 0;
            for (const auto &comp : components) {
                total_faces += comp.F.rows();
            }

            if (components.empty()) {
                return;
            }
            
            UVParts uv_parts(components);
            
            {
                std::lock_guard<std::mutex> lock(output_mutex);
                if (CONFIG_verbose) {
                    std::cout << "[GET_UV] Method " << func_name << " has " << uv_parts.num_components
                            << " components and distortion " << uv_parts.distortion << std::endl;
                }
                all_candidates.push_back(uv_parts);
            }
        }
    );

    total_time_spent_unwrap += local_total_time;
    
    if (all_candidates.empty()) {
        return {UVParts(MAX_CHARTS)};
    }else{
        return all_candidates;
    }

}


bool check_part_overlap(UVParts &part){
    bool has_overlaps = false;
    std::vector<std::pair<int,int>> all_overlappingTriangles;
    for (Component chart : part.components){
        std::vector<std::pair<int,int>> overlappingTriangles;
        if(computeOverlapingTrianglesFast(chart.UV, chart.F,overlappingTriangles)){
            return true;
        }
    }
    return false;
}

UVParts get_best_part( std::vector<std::vector<UVParts>> all_candidates, double threshold, bool check_overlap, int* debug_index , bool use_dummy_best_part){

    std::vector<UVParts> valid_candidates(all_candidates.size());

    for (int i = 0; i < all_candidates.size(); ++i)
    {
        std::vector<UVParts> candidates = all_candidates[i];
        UVParts dummy_best_part;
        auto best_part = get_best_part(candidates, threshold, check_overlap, &dummy_best_part, debug_index );
        if (best_part.num_components  == MAX_CHARTS)
        {
            if(use_dummy_best_part){
                valid_candidates[i] = dummy_best_part;
            }else{
                return best_part;
            }
        }else{
            valid_candidates[i] = best_part;
        }
    }
    
    return std::accumulate(valid_candidates.begin() + 1, valid_candidates.end(), valid_candidates[0]);
}

UVParts get_best_part( std::vector<UVParts> all_candidates, double threshold, bool check_overlap, UVParts *best_part_in_list,  int *debug_index ){
      // If we have valid candidates (distortion < threshold),
    // choose the one with the fewest parts.
    std::vector<UVParts> valid_candidates;

    

    // -1 means all invalid, 0 -> not the last one,  1 -> last one, 
    // ONLY WORKS FOR SINGLE COMPONENT, MULTIPLE COMPONENTS NOT SUPPORTED YET
    if (debug_index) *debug_index = 0;              
    for (auto &candidate : all_candidates)
    {
        if (CONFIG_verbose){
            std::cout << "[INFO] Candidate # of charts: " << candidate.num_components << " distortion: " << candidate.distortion << std::endl;
        }
        if (check_overlap && check_part_overlap(candidate))
        {
            if (CONFIG_verbose)std::cout << "[INFO] Candidate # of charts: " << candidate.num_components << "  has overlaps. Skipping. distortion: " << candidate.distortion << std::endl;
            continue;
        }

        if (candidate.distortion < 0 || candidate.distortion != candidate.distortion) {
            if (CONFIG_verbose)std::cout << "[INFO] Candidate # of charts: " << candidate.num_components << " distortion: " << candidate.distortion << " is NaN/negative. Skipping." << std::endl;
            continue;
        }

        if (candidate.distortion >= threshold)
        {
            if(best_part_in_list != nullptr){
                if(candidate.distortion < best_part_in_list->distortion){
                    *best_part_in_list = candidate;
                }
            }else{
                *best_part_in_list = candidate;
            }
            if (CONFIG_verbose)std::cout << "[INFO] Candidate # of charts: " << candidate.num_components << " distortion: " << candidate.distortion << " is above threshold " << threshold << std::endl;
            continue;
        }

        if (CONFIG_verbose)std::cout << "[INFO] Candidate # of charts: " << candidate.num_components << " distortion: " << candidate.distortion << " is below threshold " << threshold << std::endl;
        
        #pragma omp critical
        valid_candidates.push_back(candidate);
    
    }

    
    if (!valid_candidates.empty())
    {
        auto best_it = std::min_element(valid_candidates.begin(), valid_candidates.end(),
            [](const UVParts &a, const UVParts &b)
            {
                if (a.components.size() == b.components.size())
                {
                    return a.distortion < b.distortion;
                }
                return a.components.size() < b.components.size();
            }
        );
        
        if (debug_index) *debug_index = (*best_it == all_candidates.back()) ? 1 : 0;
        
        return *best_it;

    }
    else
    {
        if (debug_index) *debug_index = -1;
        return UVParts(MAX_CHARTS);
    }
}




bool verify_part(const UVParts &part){
    return part.num_components > 0 && part.distortion >= 0 && part.components.size() == part.num_components;
}


UVParts compare_new_parts_log(UVParts &curr_uv_parts, const UVParts &new_uv)
{
    if(not verify_part(curr_uv_parts)) return new_uv;
    if(not verify_part(new_uv)) return curr_uv_parts;

    // Log the first condition: new_uv.distortion < gThreshold
    bool cond1 = new_uv.distortion < gThreshold;
    std::cout << "[COMPARE] new_uv.distortion (" << new_uv.distortion 
              << ") < gThreshold (" << gThreshold << ") is " 
              << std::boolalpha << cond1 << std::endl;

    // Log the second condition: new_uv.num_components < curr_uv_parts.num_components
    bool cond2 = new_uv.num_components < curr_uv_parts.num_components;
    std::cout << "[COMPARE] new_uv.num_components (" << new_uv.num_components 
              << ") < curr_uv_parts.num_components (" << curr_uv_parts.num_components 
              << ") is " << std::boolalpha << cond2 << std::endl;

    // Log the third condition: new_uv.num_components == curr_uv_parts.num_components
    bool cond3 = new_uv.num_components == curr_uv_parts.num_components;
    std::cout << "[COMPARE] new_uv.num_components (" << new_uv.num_components 
              << ") == curr_uv_parts.num_components (" << curr_uv_parts.num_components 
              << ") is " << std::boolalpha << cond3 << std::endl;

    // Log the fourth condition: new_uv.distortion < curr_uv_parts.distortion
    bool cond4 = new_uv.distortion < curr_uv_parts.distortion;
    std::cout << "[COMPARE] new_uv.distortion (" << new_uv.distortion 
              << ") < curr_uv_parts.distortion (" << curr_uv_parts.distortion 
              << ") is " << std::boolalpha << cond4 << std::endl;

    // Determine if new_uv should replace curr_uv_parts
    if (cond1 && (cond2 || (cond3 && cond4)))
    {
        std::cout << "[COMPARE] Conditions met. Updating curr_uv_parts to new_uv." << std::endl;
        curr_uv_parts = new_uv;
    }
    else
    {
        std::cout << "[COMPARE] Conditions not met. Keeping current uv parts." << std::endl;
    }

    std::cout << "Using current parts with " << curr_uv_parts.num_components 
              << " components and distortion " << curr_uv_parts.distortion << std::endl;
    return curr_uv_parts;
}

UVParts compare_new_parts(UVParts &curr_uv_parts, const UVParts &new_uv)
{
    if(not verify_part(curr_uv_parts)) return new_uv;
    if(not verify_part(new_uv)) return curr_uv_parts;

    if(  new_uv.distortion < gThreshold && 
        (new_uv.num_components < curr_uv_parts.num_components || 
        (new_uv.num_components == curr_uv_parts.num_components && new_uv.distortion < curr_uv_parts.distortion)))
    {
        curr_uv_parts = new_uv;
    }
    return curr_uv_parts;
}

void mock_pipeline(const std::string &mesh_filename, double threshold){
    if (!igl::read_triangle_mesh(mesh_filename, gV, gF)){
        std::cerr << "Failed to load mesh from the given path." << std::endl; 
        return ;
    }
    gThreshold = threshold;
    std::vector<std::vector<double>> edge_lengths;
    gFaceAdj = computeFaceAdjacency(gF, gV, edge_lengths);
}

void get_individual_parts(std::vector<UVParts> &individual_parts){
    individual_parts = allParts;
}

// blue is finished faces, red is the rest. partuv finishing a chart is
// binary, so there is no green in-progress segment.
// assumes the caller holds the visual critical section.
void print_visual_progress()
{
    if (gF.rows() == 0) return;
    double done = (double)gVisualFacesDone / gF.rows();
    std::cout << "progress: " << done << " 0 " << (1.0 - done) << std::endl;
}

// counts a finished part's faces toward progress and emits a progress line.
// must not be called from inside another unnamed critical section.
void advance_visual_progress(const UVParts &part)
{
    if (!gVisual) return;
    #pragma omp critical
    {
        for (const Component &chart : part.components)
            gVisualFacesDone += chart.F.rows();
        print_visual_progress();
    }
}


UVParts pipeline_helper(std::vector<int> leaves, Tree tree, int root, double chart_limit, int stack_level)
{   
    if(leaves.size() == 0){
        std::cerr << "[PIPELINE] WARNING 0 faces in the submesh " << root << " Returnning." << std::endl;
        return UVParts({});
    }else if(leaves.size() == 1){
        Eigen::MatrixXi Fc; Eigen::MatrixXd Vc;
        std::vector<int> local2global;
        ExtractSubmesh(leaves, gF, gV, Fc, Vc, &local2global);
        UVParts single_part(unwrap_aligning_one(Vc, Fc, gThreshold, false, 1));
        remap_source_vids(single_part, local2global);
        // at part level this single face never reaches save_part below
        if (chart_limit == NO_CHART_LIMIT) advance_visual_progress(single_part);
        return single_part;
    }
    if(chart_limit != NO_CHART_LIMIT & chart_limit < 1)
    {
        // returns a dummy UVParts object with infinite parts and distortion
        #ifdef VERBOSE
        std::cout << "[INFO] Chart limit " << chart_limit << " is less than 1. Returning dummy UVParts object." << std::endl;
        #endif
        
        return UVParts(MAX_CHARTS);
    }
    
    #pragma omp critical
    {
        level++;
    }

    bool check_overlap = (chart_limit != NO_CHART_LIMIT);

    if (CONFIG_verbose) {
        std::cout << "[INFO] Processing mesh " << root << " with " << leaves.size() << " faces" << std::endl;
    }
    
    
    Eigen::MatrixXd Vc; Eigen::MatrixXi Fc;
    UVParts curr_uv_parts = UVParts(MAX_CHARTS);
    
    
    bool save_part = false;
    std::vector<int> local2global;
    ExtractSubmesh(leaves, gF, gV, Fc, Vc, &local2global);

    auto candidates = get_uv_wrapper(Fc, Vc, gThreshold, /*check_overlap*/false, /*use_full*/false, (int)chart_limit);
    for (auto &parts : candidates) remap_source_vids(parts, local2global);



    curr_uv_parts = get_best_part(candidates, gThreshold, check_overlap);

    if(CONFIG_verbose){
        std::cout << "[INFO] Submesh " << root << " has " << curr_uv_parts.num_components << " parts, " << chart_limit << " chart limit,  and distortion " << curr_uv_parts.distortion << std::endl;
    }

    if(curr_uv_parts.distortion < gThreshold || chart_limit != NO_CHART_LIMIT)
    {
        // Either current part is below threshold or it's parent is below threshold (chart limit is not NO_CHART_LIMIT, meaning we are recursing)
        check_overlap = CONFIG_pipelineOverlaps;

        std::vector<std::vector<UVParts>>  new_candidates =  get_uv_wrapper(Fc, Vc, gThreshold, false, true, (int)chart_limit);
        for (auto &parts : new_candidates) remap_source_vids(parts, local2global);

        for(int i = 0; i < candidates.size(); ++i){
            candidates[i].insert(candidates[i].end(), new_candidates[i].begin(), new_candidates[i].end());
        }
        int debug_index;
        curr_uv_parts = get_best_part(candidates, gThreshold, check_overlap, &debug_index);
        
        if(CONFIG_log_traverse_csv){
            log_traverse_csv(&tree[root], &curr_uv_parts, find_parent_with_child(tree, root), curr_uv_parts.num_components == MAX_CHARTS, "pink");
        }

        if (chart_limit == NO_CHART_LIMIT) {
            // This is first time getting here, resembling a PART here
            num_parts++;
            save_part = true;
            map_root_to_part[root] = num_parts;
            check_overlap = true;
            chart_limit = curr_uv_parts.num_components ;
            if (CONFIG_verbose) {
                std::cout << "[INFO] Part created with " << root << " with distortion of "  << curr_uv_parts.distortion << std::endl;
            }
        }else{
            chart_limit = std::min(chart_limit, (double)curr_uv_parts.num_components);
        }
        
        if(CONFIG_verbose){
            std::cout << "[RECURSIVE] Submesh " << root << " has " << curr_uv_parts.num_components << " parts, " << chart_limit << " chart limit,  and distortion " << curr_uv_parts.distortion << std::endl;
        }
        if (chart_limit > 2) {


            assert(chart_limit > 1);

            if (CONFIG_verbose) {
                std::cout << "[INFO] Chart limit: " << chart_limit << std::endl;
                std::cout << "[INFO] Splitting mesh " << root << " with " << leaves.size() << " faces" << std::endl;
            }
            std::vector<std::vector<int>> childFaces = { get_tree_leaves(tree, tree[root].left), get_tree_leaves(tree, tree[root].right) };
            const auto childFacesOrig = childFaces; // keep a copy
            const Tree treeOrig = tree;     

            // DEBUGGING
            if(CONFIG_verbose){
                std::cout << "[INFO] left and right children RIGHT AFTER GETTING TREE LEAVES " << tree[root].left << ", " <<childFaces[0].size()  << " faces "<< tree[root].right << ", " << childFaces[1].size() << " faces" << std::endl;
            }

            // smoothComponentEdge(childFaces, gFaceAdj);
            smoothComponentEdge(childFaces, gFaceAdj, &tree);
            if (childFaces[0].size() == 0 || childFaces[1].size() == 0) {
                // One of the children has no faces, so we don't smooth the edge
                // childFaces = { get_tree_leaves(tree, tree[root].left), get_tree_leaves(tree, tree[root].right) };
                childFaces = childFacesOrig;
                tree.~Tree();              // destroy current state
                new (&tree) Tree(treeOrig); // copy-construct in place (no operator=)
            }

            if(CONFIG_verbose){
                std::cout << "[INFO] left and right children " << tree[root].left << ", " <<childFaces[0].size()  << " faces "<< tree[root].right << ", " << childFaces[1].size() << " faces" << std::endl;
            }

            std::vector<int> leaves_left = childFaces[0];
            std::vector<int> leaves_right = childFaces[1];


            // This function is used to handle the case when there is only one or less face in the submesh
            auto handle_one = [](std::vector<int> leaves) -> std::vector<Component> {
                assert(leaves.size() <= 1);
                if (leaves.size() == 0) return {};
                Eigen::MatrixXd Vc; Eigen::MatrixXi Fc;
                std::vector<int> local2global;
                ExtractSubmesh(leaves, gF, gV, Fc, Vc, &local2global);
                std::vector<Component> components = unwrap_aligning_one(Vc, Fc, gThreshold, false, 1);
                for (Component &comp : components) remap_source_vids(comp, local2global);
                return components;
            };
            
            UVParts left_uv_components ;
            UVParts right_uv_components;
            
            if (stack_level < CONFIG_parallelDepth){
                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        left_uv_components = leaves_left.size() <= 1 ?  UVParts(handle_one(leaves_left)) : pipeline_helper(leaves_left,tree, tree[root].left,  chart_limit - 1, stack_level + 1);
                    }
                    
                    #pragma omp section
                    {
                        right_uv_components = leaves_right.size() <= 1  ?  UVParts(handle_one(leaves_right)) :  pipeline_helper(leaves_right, tree, tree[root].right, chart_limit - 1 - left_uv_components.num_components, stack_level + 1);
                    }
                }
            }else{
                left_uv_components = leaves_left.size() <= 1 ?  UVParts(handle_one(leaves_left)) : pipeline_helper(leaves_left,tree, tree[root].left,  chart_limit - 1, stack_level + 1);
                right_uv_components = leaves_right.size() <= 1  ?  UVParts(handle_one(leaves_right)) :  pipeline_helper(leaves_right, tree, tree[root].right, chart_limit - 1 - left_uv_components.num_components, stack_level + 1);
            }
                
                
            UVParts combined_uv = left_uv_components + right_uv_components;

            if(CONFIG_verbose){
                std::cout << "[BEFORE COMPARE] left_uv_components: " << left_uv_components.num_components << " " << left_uv_components.distortion << std::endl;
                std::cout << "[BEFORE COMPARE] right_uv_components: " << right_uv_components.num_components << " " << right_uv_components.distortion << std::endl;
                UVParts new_curr_uv_parts = compare_new_parts_log(curr_uv_parts, combined_uv);
                if (new_curr_uv_parts != curr_uv_parts){
                    debug_index = 2;
                }
                curr_uv_parts = new_curr_uv_parts;
            }else{
                UVParts new_curr_uv_parts = compare_new_parts(curr_uv_parts, combined_uv);
                if (new_curr_uv_parts != curr_uv_parts){
                    debug_index = 2;
                }
                curr_uv_parts = new_curr_uv_parts;
            }

        }
        
        if(save_part) {
            for(auto & chart  : curr_uv_parts.components){
                ComputeOriginalUV(chart);

                std::vector<std::pair<int,int>> overlappingTriangles;
                if(computeOverlapingTrianglesFast(chart.UV, chart.F,overlappingTriangles)){
                    std::cerr << "part " << allParts.size() << " has un-resolvable overlapping triangles" << std::endl;
                }
            }

            #pragma omp critical
            {
                allParts.push_back(curr_uv_parts);
                g_hierarchy.addLeaf(root, allParts.size()-1, curr_uv_parts.to_components().F.rows());
                if (CONFIG_verbose) {
                    std::cout << "pushing part with faces:" << curr_uv_parts.to_components().F.rows() << std::endl;
                }
            }
            std::cout << "PART: " << allParts.size()-1 << " created with charts: " << curr_uv_parts.num_components << std::endl;
            advance_visual_progress(curr_uv_parts);


            if (CONFIG_verbose) {
                std::cout << "debug index: " << debug_index << " for the FINAL PART  " << allParts.size()-1 << " with num components " << curr_uv_parts.num_components  << std::endl;
            }

            if (CONFIG_log_traverse_csv){
                log_traverse_csv(&tree[root], &curr_uv_parts, find_parent_with_child(tree, root), false, "green");
            }
        }
        

    }
    else
    {
        if(not tree.contains(root)) {
            std::cerr << "[PIPELINE] WARNING: Tree does not contain node " << root << ". Returning Empty parts." << std::endl;
            return UVParts({});
        }
        
        std::vector<std::vector<int>> childFaces = { get_tree_leaves(tree, tree[root].left), get_tree_leaves(tree, tree[root].right) };
        const auto childFacesOrig = childFaces; // keep a copy
        const Tree treeOrig = tree;     

        if(CONFIG_verbose){
            std::cout << "Keep Splitting in node " << root << "  with distortion " << curr_uv_parts.distortion << std::endl;
        }

        if(CONFIG_log_traverse_csv){
            log_traverse_csv(&tree[root], &curr_uv_parts, find_parent_with_child(tree, root), curr_uv_parts.num_components == MAX_CHARTS, "blue");
        }


        // smoothComponentEdge(childFaces, gFaceAdj);
        smoothComponentEdge(childFaces, gFaceAdj, &tree);

        if (childFaces[0].size() == 0 || childFaces[1].size() == 0) {
            // One of the children has no faces, so we don't smooth the edge
            // childFaces = { get_tree_leaves(tree, tree[root].left), get_tree_leaves(tree, tree[root].right) };
            childFaces = childFacesOrig;
            tree.~Tree();              // destroy current state
            new (&tree) Tree(treeOrig); // copy-construct in place (no operator=)
        }

        if(CONFIG_verbose){
            std::cout << "[INFO] left and right children " << tree[root].left << ", " <<childFaces[0].size()  << " faces "<< tree[root].right << ", " << childFaces[1].size() << " faces" << std::endl;
        }



        g_hierarchy.addInner(root, tree[root].left, tree[root].right, leaves.size());
        std::vector<int> leaves_left = childFaces[0];
        std::vector<int> leaves_right = childFaces[1];
        UVParts left_uv_components, right_uv_components;
        if (stack_level < CONFIG_parallelDepth){

            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    left_uv_components = pipeline_helper(leaves_left,tree, tree[root].left,  NO_CHART_LIMIT, stack_level + 1);
                }
                
                #pragma omp section
                {
                    right_uv_components = pipeline_helper(leaves_right,tree, tree[root].right, NO_CHART_LIMIT, stack_level + 1);
                }
            }
        }else{
            left_uv_components = pipeline_helper(leaves_left,tree, tree[root].left,  NO_CHART_LIMIT, stack_level + 1);
            right_uv_components = pipeline_helper(leaves_right,tree, tree[root].right, NO_CHART_LIMIT, stack_level + 1);
        }

        curr_uv_parts = left_uv_components + right_uv_components;

    }


    level --;
    return curr_uv_parts;

}



int parse_component_from_tree(Tree &tree,int root, int max_depth,  std::vector<std::vector<int>> &all_faces, std::vector<int> &all_roots){
    // if (max_depth == 1) {
    //     all_roots.push_back(root);
    //     all_faces.push_back(get_tree_leaves(tree, root));
    //     return 1;
    // }

    std::vector<int> leaves = get_tree_leaves(tree, root);

    Eigen::MatrixXd Vc; Eigen::MatrixXi Fc;
    ExtractSubmesh(leaves, gF, gV, Fc, Vc);
    Eigen::VectorXi component;
    igl::facet_components(Fc, component);
    int num_components = component.maxCoeff() + 1;

    if (max_depth == 1 || num_components == 1) {
        all_roots.push_back(root);
        all_faces.push_back(leaves);

        return num_components;
    }


    // here we keep split components and thus it is white 
    if (CONFIG_log_traverse_csv){
        UVParts dummy_part = UVParts(MAX_CHARTS);
        Eigen::MatrixXd comp_V; Eigen::MatrixXi comp_F;
        ExtractSubmesh(get_tree_leaves(tree, root), gF, gV, comp_F, comp_V);

        Component comp = Component(0, get_tree_leaves(tree, root), 0, Eigen::MatrixXd::Zero(1,3), comp_F, comp_V);
        dummy_part.components.push_back(comp);

        log_traverse_csv(&tree[root],&dummy_part, find_parent_with_child(tree, root), false, "gray");
    }
    std::vector<int> left_roots;
    std::vector<int> right_roots;
    std::vector<std::vector<int>> left_parts;
    std::vector<std::vector<int>> right_parts;
    int left_size = parse_component_from_tree(tree, tree[root].left, max_depth - 1, left_parts, left_roots);
    int right_size = parse_component_from_tree(tree, tree[root].right, max_depth - 1, right_parts, right_roots);

    if (left_size == num_components || right_size == num_components){
        all_roots.push_back(root);
        all_faces.push_back(get_tree_leaves(tree, root));
        g_hierarchy.removeInner(tree[root].left);
        g_hierarchy.removeInner(tree[root].right);

        return num_components;
    }

    g_hierarchy.addInner(root, tree[root].left, tree[root].right, leaves.size());

    all_faces.insert(all_faces.end(), left_parts.begin(), left_parts.end());
    all_faces.insert(all_faces.end(), right_parts.begin(), right_parts.end());
    all_roots.insert(all_roots.end(), left_roots.begin(), left_roots.end());
    all_roots.insert(all_roots.end(), right_roots.begin(), right_roots.end());
    
    return left_size + right_size;


}

UVParts pipeline(const std::string &tree_filename, const std::string &mesh_filename, double threshold, std::vector<UVParts> & individual_parts)
{

    EASY_PROFILER_ENABLE;
    EASY_BLOCK("pipeline");
    
    // Only load mesh from file if mesh_filename is not empty
    if (!mesh_filename.empty()) {
        bool mesh_valid = load_mesh_with_validation(mesh_filename, gF, gV);
        if(!mesh_valid){
            std::cerr << "Invalid mesh. Exiting." << std::endl;
            exit(3);
        }
    } else {
        // Mesh data should already be loaded in gV and gF from numpy arrays
        // Just validate the mesh data
        if (!check_mesh(gV, gF)) {
            std::cerr << "Invalid mesh data from numpy arrays. Exiting." << std::endl;
            exit(3);
        }
    }
    

    
    Tree part_field_tree(tree_filename);
    
    validate_tree_leaves(part_field_tree, gF.rows());
    if (part_field_tree.size() != gF.rows()-1){
        std::cerr << "Invalid tree. Expected " << gF.rows()-1 << " (number of faces - 1) nodes, but got " << part_field_tree.size() << " nodes with mesh having " << gF.rows() << " faces" << std::endl;
        std::cerr << "Exiting." << std::endl;
        exit(3);
    }
    


    std::cout << "tree validated with " << part_field_tree.size() << " nodes and " << gF.rows() << " faces" << std::endl;
    
    int root = part_field_tree.root();
    
    auto start = Clock::now();
    
    // Read the input mesh from file
    std::vector<std::vector<double>> edge_lengths;
    gFaceAdj = computeFaceAdjacency(gF, gV, edge_lengths);
    gThreshold = threshold;

    if(CONFIG_unwrapPamo){
        StreamPool::asyncInitializeStreams(CONFIG_num_cuda_streams);
    }
    
    init_log_files();

    omp_set_dynamic(0);
    omp_set_num_threads(CONFIG_num_omp_threads);  

    omp_set_nested(1);
    verify_omp();



    int max_depth = CONFIG_componentMaxDepth;
    UVParts final_parts;
    if(max_depth != 0){
        std::vector<int> all_roots;
        std::vector<std::vector<int>> allFaces;

        int num_components = parse_component_from_tree(part_field_tree, root, max_depth, allFaces, all_roots);
        std::cout << "parsed " << allFaces.size() << " parts" << " with " << num_components << " components" << std::endl;

        std::cout << "all roots: ";
        for(int i = 0; i < all_roots.size(); ++i){
            std::cout << all_roots[i] << " ";
        }
        std::cout << std::endl;

        std::vector<UVParts> all_parts(allFaces.size());    
        #pragma omp parallel for
        for(int c = 0; c < allFaces.size(); ++c) {
            // Access face indices for component 'c' via allParts[c]
            std::vector<int> component_faces = allFaces[c];
            int curr_root = all_roots[c];
            UVParts curr_part = pipeline_helper(component_faces, part_field_tree, curr_root, NO_CHART_LIMIT);
            all_parts[c] = curr_part;
        }

        final_parts = all_parts[0];
        for(int i = 1; i < all_parts.size(); ++i){
            final_parts = final_parts + all_parts[i];
        }
        
    }else{
        std::vector<int> leaves = get_tree_leaves(part_field_tree, root);
        final_parts = pipeline_helper(leaves, part_field_tree, root, NO_CHART_LIMIT);

    }
    


    #ifdef ENABLE_PROFILING
    auto end = Clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "[Profiler] Total time spent in pipeline: " << elapsed.count() << " seconds." << std::endl;
    // std::cout << "[Profiler] Total time spent in get_uv: " << total_time_spent_s << " seconds." << std::endl;
    // std::cout << "[Profiler] Method call took " << total_time_spent_unwrap << " seconds." << std::endl;
    // std::cout << "[Profiler] Total time spent in unwrap project: " << total_time_spent_lscm << " seconds." << std::endl;
    // std::cout << "[Profiler] Total time spent in computeOverlapingTrianglesFast: " << total_time_spent_overlap << " seconds." << std::endl;
    #endif

    individual_parts = allParts;
    EASY_END_BLOCK;
    profiler::dumpBlocksToFile("test_profile.prof");
    // std::cout << "saving hierarchy to " << CONFIG_outputPath + "/hierarchy.json" << std::endl;
    // g_hierarchy.save(CONFIG_outputPath + "/hierarchy.json");
    final_parts.hierarchy = g_hierarchy;
    return final_parts;

}

UVParts pipeline(const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<NodeRecord>& tree_nodes,
    const std::string& configPath,
    double threshold,
    bool pack_final_mesh,
    bool visual,
    std::vector<UVParts>& individual_parts){

        if (pack_final_mesh){
            std::cerr << "[Warning] Packing final mesh within the pipeline with uvpackmaster SDK is not supported right now. Please use the blender packing script instead." << std::endl;
        }

        // Uncomment this to enable profiling
        // EASY_PROFILER_ENABLE;

        EASY_BLOCK("pipeline");
        clear_global_data();
        gVisual = visual;
        
        ConfigManager::instance().loadFromFile(configPath);
        gV = V;
        gF = F;
        // ConfigManager::instance().printConfigs();
        

        if (!check_mesh(gV, gF)) {
            std::cerr << "Invalid mesh data from numpy arrays. Exiting." << std::endl;
            exit(3);
        }

        
        Tree part_field_tree(tree_nodes);
        
        validate_tree_leaves(part_field_tree, gF.rows());
        if (part_field_tree.size() != gF.rows()-1){
            std::cerr << "Invalid tree. Expected " << gF.rows()-1 << " (number of faces - 1) nodes, but got " << part_field_tree.size() << " nodes with mesh having " << gF.rows() << " faces" << std::endl;
            std::cerr << "Exiting." << std::endl;
            exit(3);
        }
        
        std::cout << "tree validated with " << part_field_tree.size() << " nodes and " << gF.rows() << " faces" << std::endl;
        
        int root = part_field_tree.root();
        
        auto start = Clock::now();
        
        // Read the input mesh from file
        std::vector<std::vector<double>> edge_lengths;
        gFaceAdj = computeFaceAdjacency(gF, gV, edge_lengths);
        gThreshold = threshold;

    
        if(CONFIG_unwrapPamo){
            StreamPool::asyncInitializeStreams(CONFIG_num_cuda_streams);
        }
        
        init_log_files();
    
        omp_set_dynamic(0);        
        omp_set_num_threads(CONFIG_num_omp_threads);  
        
        omp_set_nested(1);
        verify_omp();
    
    
    
        
    
        int max_depth = CONFIG_componentMaxDepth;
        UVParts final_parts;
        if(max_depth != 0){
            std::vector<int> all_roots;
            std::vector<std::vector<int>> allFaces;
    
            int num_components = parse_component_from_tree(part_field_tree, root, max_depth, allFaces, all_roots);
            std::cout << "parsed " << allFaces.size() << " parts" << " with " << num_components << " components" << std::endl;
    
            std::cout << "all roots: ";
            for(int i = 0; i < all_roots.size(); ++i){
                std::cout << all_roots[i] << " ";
            }
            std::cout << std::endl;
    
            std::vector<UVParts> all_parts(allFaces.size());    
            #pragma omp parallel for
            for(int c = 0; c < allFaces.size(); ++c) {
                // Access face indices for component 'c' via allParts[c]
                std::vector<int> component_faces = allFaces[c];
                int curr_root = all_roots[c];
                UVParts curr_part = pipeline_helper(component_faces, part_field_tree, curr_root, NO_CHART_LIMIT);
                all_parts[c] = curr_part;
            }
    
            final_parts = all_parts[0];
            for(int i = 1; i < all_parts.size(); ++i){
                final_parts = final_parts + all_parts[i];
            }
        }else{
            std::vector<int> leaves = get_tree_leaves(part_field_tree, root);
            final_parts = pipeline_helper(leaves, part_field_tree, root, NO_CHART_LIMIT);
    
        }
        
    
    
        #ifdef ENABLE_PROFILING
        auto end = Clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        std::cout << "[Profiler] Total time spent in pipeline: " << elapsed.count() << " seconds." << std::endl;
        // std::cout << "[Profiler] Total time spent in get_uv: " << total_time_spent_s << " seconds." << std::endl;
        // std::cout << "[Profiler] Method call took " << total_time_spent_unwrap << " seconds." << std::endl;
        // std::cout << "[Profiler] Total time spent in unwrap project: " << total_time_spent_lscm << " seconds." << std::endl;
        // std::cout << "[Profiler] Total time spent in computeOverlapingTrianglesFast: " << total_time_spent_overlap << " seconds." << std::endl;
        #endif
    
        individual_parts = allParts;
        EASY_END_BLOCK;
        // profiler::dumpBlocksToFile("test_profile.prof");
        // std::cout << "saving hierarchy to " << CONFIG_outputPath + "/hierarchy.json" << std::endl;
        // g_hierarchy.save(CONFIG_outputPath + "/hierarchy.json");
        for (auto& comp : final_parts.components) {
            normalize_uv_by_3d_area(comp);
        }
        
        Component UV_component = final_parts.to_components();
        // if(pack_final_mesh){
        //     // Pack  
        //     IglUvWrapper wrapper;

        //     wrapper.add_component(UV_component, 0, false);
        //     UV_component.UV = wrapper.runPacking();
        // }
        UVParts return_parts = UVParts({UV_component}); 
        return_parts.hierarchy = g_hierarchy;
        return return_parts;
}