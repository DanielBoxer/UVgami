#include "Flatten.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/flip_avoiding_line_search.h>
#include <igl/slim.h>
#include <tbb/parallel_for.h>

#include "uvgami.h"

// igl::slim internals, redeclared so slimSolve can run the iteration itself
namespace igl {
namespace slim {
void update_weights_and_closest_rotations(igl::SLIMData &s,
                                          Eigen::MatrixXd &uv);
void build_linear_system(igl::SLIMData &s, Eigen::SparseMatrix<double> &L);
double compute_energy(igl::SLIMData &s, const Eigen::MatrixXd &V_new);
}  // namespace slim
}  // namespace igl

namespace {

struct PolyMesh {
    std::vector<Eigen::Vector3d> verts;
    std::vector<std::vector<int>> faces;    // vertex indices per polygon
    std::vector<std::vector<int>> corners;  // uv-vertex id per polygon corner
    std::vector<Eigen::Vector2d> uvs;       // per uv-vertex
    std::vector<int> uvVert;                // uv-vertex id -> source vertex
};

struct UnionFind {
    std::vector<int> parent;
    explicit UnionFind(size_t n) : parent(n) {
        for (size_t i = 0; i < n; ++i) parent[i] = static_cast<int>(i);
    }
    int find(int a) {
        while (parent[a] != a) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    }
    void unite(int a, int b) { parent[find(a)] = find(b); }
};

// f-token forms "v", "v/vt", "v//vn", "v/vt/vn", negative indices are relative
bool parseFaceToken(const std::string &token, int vertCount, int uvCount,
                    int &vertOut, int &uvOut) {
    size_t slash = token.find('/');
    std::string vertStr =
        slash == std::string::npos ? token : token.substr(0, slash);
    uvOut = -1;
    try {
        size_t consumed = 0;
        long v = std::stol(vertStr, &consumed);
        if (consumed != vertStr.size()) return false;
        v = v < 0 ? vertCount + v : v - 1;
        if (v < 0 || v >= vertCount) return false;
        vertOut = static_cast<int>(v);
    } catch (const std::exception &) {
        return false;
    }
    if (slash == std::string::npos) return true;
    size_t second = token.find('/', slash + 1);
    std::string uvStr = second == std::string::npos
                            ? token.substr(slash + 1)
                            : token.substr(slash + 1, second - slash - 1);
    if (uvStr.empty()) return true;
    try {
        size_t consumed = 0;
        long t = std::stol(uvStr, &consumed);
        if (consumed != uvStr.size()) return false;
        t = t < 0 ? uvCount + t : t - 1;
        if (t < 0 || t >= uvCount) return false;
        uvOut = static_cast<int>(t);
    } catch (const std::exception &) {
        return false;
    }
    return true;
}

// hasUV comes back true only when every corner carries a vt index
bool readObj(const std::string &path, PolyMesh &mesh, bool &hasUV) {
    std::ifstream file(path);
    if (!file) return false;
    std::vector<Eigen::Vector2d> fileUvs;
    hasUV = true;
    std::string line;
    while (std::getline(file, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && (line[1] == ' ' || line[1] == '\t')) {
            std::istringstream stream(line.substr(1));
            double x, y, z;
            if (!(stream >> x >> y >> z)) return false;
            mesh.verts.emplace_back(x, y, z);
        } else if (line[0] == 'v' && line[1] == 't' && line.size() > 2 &&
                   (line[2] == ' ' || line[2] == '\t')) {
            std::istringstream stream(line.substr(2));
            double u, v;
            if (!(stream >> u >> v)) return false;
            fileUvs.emplace_back(u, v);
        } else if (line[0] == 'f' && (line[1] == ' ' || line[1] == '\t')) {
            std::istringstream stream(line.substr(1));
            std::string token;
            std::vector<int> face, faceUv;
            while (stream >> token) {
                int vert = 0, uv = -1;
                if (!parseFaceToken(token, static_cast<int>(mesh.verts.size()),
                                    static_cast<int>(fileUvs.size()), vert, uv))
                    return false;
                face.push_back(vert);
                faceUv.push_back(uv);
                if (uv < 0) hasUV = false;
            }
            if (face.size() < 3) return false;
            mesh.faces.push_back(std::move(face));
            mesh.corners.push_back(std::move(faceUv));
        }
    }
    if (mesh.faces.empty()) return false;
    if (fileUvs.empty()) hasUV = false;
    if (hasUV) {
        mesh.uvs = std::move(fileUvs);
        mesh.uvVert.assign(mesh.uvs.size(), -1);
        for (size_t f = 0; f < mesh.faces.size(); ++f)
            for (size_t k = 0; k < mesh.faces[f].size(); ++k)
                mesh.uvVert[static_cast<size_t>(mesh.corners[f][k])] =
                    mesh.faces[f][k];
    }
    return true;
}

// "a b" per line, 0-based vertex indices. a missing file means no seams.
bool readSeams(const std::string &path, int vertCount,
               std::set<std::pair<int, int>> &seams) {
    std::ifstream file(path);
    if (!file) return true;
    int a, b;
    while (file >> a >> b) {
        if (a < 0 || b < 0 || a >= vertCount || b >= vertCount) return false;
        seams.emplace(std::min(a, b), std::max(a, b));
    }
    return file.eof();
}

// corners weld across interior edges that are neither seams nor non-manifold
void weldCorners(PolyMesh &mesh, const std::set<std::pair<int, int>> &seams) {
    std::vector<int> offset(mesh.faces.size() + 1, 0);
    for (size_t f = 0; f < mesh.faces.size(); ++f)
        offset[f + 1] = offset[f] + static_cast<int>(mesh.faces[f].size());
    UnionFind uf(static_cast<size_t>(offset.back()));

    std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> owners;
    for (size_t f = 0; f < mesh.faces.size(); ++f) {
        const std::vector<int> &face = mesh.faces[f];
        for (size_t k = 0; k < face.size(); ++k) {
            int a = face[k], b = face[(k + 1) % face.size()];
            owners[{std::min(a, b), std::max(a, b)}].emplace_back(
                static_cast<int>(f), static_cast<int>(k));
        }
    }
    for (const auto &entry : owners) {
        if (entry.second.size() != 2 || seams.count(entry.first)) continue;
        auto [fA, kA] = entry.second[0];
        auto [fB, kB] = entry.second[1];
        const std::vector<int> &faceA = mesh.faces[static_cast<size_t>(fA)];
        const std::vector<int> &faceB = mesh.faces[static_cast<size_t>(fB)];
        int kA2 = (kA + 1) % static_cast<int>(faceA.size());
        int kB2 = (kB + 1) % static_cast<int>(faceB.size());
        // winding can differ between the two faces
        int cornerA[2] = {offset[static_cast<size_t>(fA)] + kA,
                          offset[static_cast<size_t>(fA)] + kA2};
        int vertA[2] = {faceA[static_cast<size_t>(kA)],
                        faceA[static_cast<size_t>(kA2)]};
        int cornerB[2] = {offset[static_cast<size_t>(fB)] + kB,
                          offset[static_cast<size_t>(fB)] + kB2};
        int vertB[2] = {faceB[static_cast<size_t>(kB)],
                        faceB[static_cast<size_t>(kB2)]};
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                if (vertA[i] == vertB[j]) uf.unite(cornerA[i], cornerB[j]);
    }

    std::map<int, int> compact;
    mesh.corners.assign(mesh.faces.size(), {});
    mesh.uvVert.clear();
    for (size_t f = 0; f < mesh.faces.size(); ++f) {
        mesh.corners[f].resize(mesh.faces[f].size());
        for (size_t k = 0; k < mesh.faces[f].size(); ++k) {
            int root = uf.find(offset[f] + static_cast<int>(k));
            auto [it, inserted] =
                compact.emplace(root, static_cast<int>(mesh.uvVert.size()));
            if (inserted) mesh.uvVert.push_back(mesh.faces[f][k]);
            mesh.corners[f][k] = it->second;
        }
    }
    mesh.uvs.assign(mesh.uvVert.size(), Eigen::Vector2d::Zero());
}

// island id per face: components of faces connected through shared uv-verts
std::vector<int> faceIslands(const PolyMesh &mesh, int &islandCount) {
    UnionFind uf(mesh.uvs.size());
    for (const std::vector<int> &corners : mesh.corners)
        for (size_t k = 1; k < corners.size(); ++k)
            uf.unite(corners[0], corners[k]);
    std::map<int, int> compact;
    std::vector<int> island(mesh.faces.size());
    for (size_t f = 0; f < mesh.faces.size(); ++f) {
        int root = uf.find(mesh.corners[f][0]);
        auto [it, inserted] =
            compact.emplace(root, static_cast<int>(compact.size()));
        island[f] = it->second;
    }
    islandCount = static_cast<int>(compact.size());
    return island;
}

struct Island {
    std::vector<int> faces;      // input face indices
    std::vector<int> globalUv;   // local uv id -> global uv id
    Eigen::MatrixXd V;           // local rest positions
    Eigen::MatrixXi T;           // fan triangles, local uv ids
    std::vector<std::pair<int, int>> polyEdges;  // local, undirected unique
    Eigen::MatrixXd UV;          // local solution
    double area3d = 0.0;
};

// fills localOf (global uv id -> local), the caller resets those entries to -1
Island buildIsland(const PolyMesh &mesh, std::vector<int> faces,
                   std::vector<int> &localOf) {
    Island island;
    island.faces = std::move(faces);
    for (int f : island.faces)
        for (int c : mesh.corners[static_cast<size_t>(f)])
            if (localOf[static_cast<size_t>(c)] < 0) {
                localOf[static_cast<size_t>(c)] =
                    static_cast<int>(island.globalUv.size());
                island.globalUv.push_back(c);
            }

    island.V.resize(static_cast<Eigen::Index>(island.globalUv.size()), 3);
    for (size_t i = 0; i < island.globalUv.size(); ++i)
        island.V.row(static_cast<Eigen::Index>(i)) =
            mesh.verts[static_cast<size_t>(
                           mesh.uvVert[static_cast<size_t>(island.globalUv[i])])]
                .transpose();

    std::set<std::pair<int, int>> edges;
    int triCount = 0;
    for (int f : island.faces)
        triCount +=
            static_cast<int>(mesh.faces[static_cast<size_t>(f)].size()) - 2;
    island.T.resize(triCount, 3);
    int t = 0;
    for (int f : island.faces) {
        const std::vector<int> &corners = mesh.corners[static_cast<size_t>(f)];
        const std::vector<int> &face = mesh.faces[static_cast<size_t>(f)];
        std::vector<int> local(corners.size());
        for (size_t k = 0; k < corners.size(); ++k)
            local[k] = localOf[static_cast<size_t>(corners[k])];
        for (size_t k = 1; k + 1 < local.size(); ++k) {
            island.T(t, 0) = local[0];
            island.T(t, 1) = local[k];
            island.T(t, 2) = local[k + 1];
            ++t;
        }
        for (size_t k = 0; k < local.size(); ++k) {
            int a = local[k], b = local[(k + 1) % local.size()];
            edges.emplace(std::min(a, b), std::max(a, b));
        }
        // polygon 3d area by fan, matching the triangles the solve sees
        const Eigen::Vector3d &p0 = mesh.verts[static_cast<size_t>(face[0])];
        for (size_t k = 1; k + 1 < face.size(); ++k) {
            const Eigen::Vector3d &p1 = mesh.verts[static_cast<size_t>(face[k])];
            const Eigen::Vector3d &p2 =
                mesh.verts[static_cast<size_t>(face[k + 1])];
            island.area3d += (p1 - p0).cross(p2 - p0).norm() / 2.0;
        }
    }
    island.polyEdges.assign(edges.begin(), edges.end());
    return island;
}

// walked along face winding so the tutte map comes out positively oriented
std::vector<int> outerBoundaryLoop(const Island &island, const PolyMesh &mesh,
                                   const std::vector<int> &localOf) {
    std::map<std::pair<int, int>, int> count;
    for (int f : island.faces) {
        const std::vector<int> &corners = mesh.corners[static_cast<size_t>(f)];
        for (size_t k = 0; k < corners.size(); ++k) {
            int a = localOf[static_cast<size_t>(corners[k])];
            int b = localOf[static_cast<size_t>(
                corners[(k + 1) % corners.size()])];
            ++count[{std::min(a, b), std::max(a, b)}];
        }
    }
    std::multimap<int, int> next;
    for (int f : island.faces) {
        const std::vector<int> &corners = mesh.corners[static_cast<size_t>(f)];
        for (size_t k = 0; k < corners.size(); ++k) {
            int a = localOf[static_cast<size_t>(corners[k])];
            int b = localOf[static_cast<size_t>(
                corners[(k + 1) % corners.size()])];
            if (count[{std::min(a, b), std::max(a, b)}] == 1)
                next.emplace(a, b);
        }
    }

    std::set<std::pair<int, int>> used;
    std::vector<int> best;
    double bestLength = -1.0;
    for (const auto &start : next) {
        if (used.count({start.first, start.second})) continue;
        std::vector<int> loop{start.first};
        int cur = start.second;
        used.insert({start.first, cur});
        double length = (island.V.row(start.first) - island.V.row(cur)).norm();
        while (cur != loop.front()) {
            loop.push_back(cur);
            auto range = next.equal_range(cur);
            int chosen = -1;
            for (auto it = range.first; it != range.second; ++it) {
                if (used.count({cur, it->second})) continue;
                chosen = it->second;
                break;
            }
            if (chosen < 0) break;  // open chain from a non-manifold weld
            used.insert({cur, chosen});
            length += (island.V.row(cur) - island.V.row(chosen)).norm();
            cur = chosen;
        }
        if (cur == loop.front() && loop.size() >= 3 && length > bestLength) {
            bestLength = length;
            best = loop;
        }
    }
    return best;
}

// closed islands have nothing to map to a circle
void projectToPlane(Island &island) {
    Eigen::RowVector3d normal = Eigen::RowVector3d::Zero();
    for (Eigen::Index t = 0; t < island.T.rows(); ++t) {
        Eigen::RowVector3d e1 =
            island.V.row(island.T(t, 1)) - island.V.row(island.T(t, 0));
        Eigen::RowVector3d e2 =
            island.V.row(island.T(t, 2)) - island.V.row(island.T(t, 0));
        normal += e1.cross(e2);
    }
    if (normal.norm() < 1e-12) normal = Eigen::RowVector3d::UnitZ();
    normal.normalize();
    Eigen::RowVector3d u = normal.unitOrthogonal();
    Eigen::RowVector3d v = normal.cross(u);
    island.UV.resize(island.V.rows(), 2);
    for (Eigen::Index i = 0; i < island.V.rows(); ++i) {
        island.UV(i, 0) = island.V.row(i).dot(u);
        island.UV(i, 1) = island.V.row(i).dot(v);
    }
}

// tutte, injective for disks and best effort otherwise
bool tutteInit(Island &island, const std::vector<int> &loop) {
    Eigen::Index n = island.V.rows();
    island.UV = Eigen::MatrixXd::Zero(n, 2);

    std::vector<double> cumulative(loop.size(), 0.0);
    double total = 0.0;
    for (size_t i = 0; i < loop.size(); ++i) {
        cumulative[i] = total;
        int a = loop[i], b = loop[(i + 1) % loop.size()];
        total += (island.V.row(a) - island.V.row(b)).norm();
    }
    if (total <= 0.0) return false;
    double radius = std::sqrt(std::max(island.area3d, 1e-12) / M_PI);
    std::vector<bool> fixed(static_cast<size_t>(n), false);
    for (size_t i = 0; i < loop.size(); ++i) {
        double angle = 2.0 * M_PI * cumulative[i] / total;
        island.UV(loop[i], 0) = radius * std::cos(angle);
        island.UV(loop[i], 1) = radius * std::sin(angle);
        fixed[static_cast<size_t>(loop[i])] = true;
    }

    std::vector<int> interior;
    std::vector<int> interiorOf(static_cast<size_t>(n), -1);
    for (Eigen::Index i = 0; i < n; ++i)
        if (!fixed[static_cast<size_t>(i)]) {
            interiorOf[static_cast<size_t>(i)] =
                static_cast<int>(interior.size());
            interior.push_back(static_cast<int>(i));
        }
    if (interior.empty()) return true;

    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::MatrixXd rhs =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(interior.size()), 2);
    std::vector<int> degree(static_cast<size_t>(n), 0);
    for (const auto &edge : island.polyEdges) {
        ++degree[static_cast<size_t>(edge.first)];
        ++degree[static_cast<size_t>(edge.second)];
    }
    for (const auto &edge : island.polyEdges) {
        for (int flip = 0; flip < 2; ++flip) {
            int a = flip ? edge.second : edge.first;
            int b = flip ? edge.first : edge.second;
            int row = interiorOf[static_cast<size_t>(a)];
            if (row < 0) continue;
            if (interiorOf[static_cast<size_t>(b)] >= 0)
                triplets.emplace_back(row, interiorOf[static_cast<size_t>(b)],
                                      -1.0);
            else
                rhs.row(row) += island.UV.row(b);
        }
    }
    for (size_t i = 0; i < interior.size(); ++i)
        triplets.emplace_back(
            static_cast<int>(i), static_cast<int>(i),
            static_cast<double>(degree[static_cast<size_t>(interior[i])]));
    Eigen::SparseMatrix<double> L(static_cast<Eigen::Index>(interior.size()),
                                  static_cast<Eigen::Index>(interior.size()));
    L.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(L);
    if (solver.info() != Eigen::Success) return false;
    Eigen::MatrixXd solution = solver.solve(rhs);
    if (solver.info() != Eigen::Success) return false;
    for (size_t i = 0; i < interior.size(); ++i)
        island.UV.row(interior[i]) = solution.row(static_cast<Eigen::Index>(i));
    return true;
}

// symmetric dirichlet per unit area, 4 is isometric
const double SLIM_ISOMETRIC = 4.0;
const double SLIM_SETTLED = 1e-3;
const double SLIM_STOP_SHARE = 0.1;

// damping starts at this share of the largest diagonal entry, tenfold steps up
const double DAMPING_FIRST_SHARE = 1e-12;
const double DAMPING_STEP = 10.0;

using SparseSolver = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;

// a zero pivot comes from an init with collapsed triangles, energy 1e25 and up
double factorizeDamped(SparseSolver &solver, const Eigen::SparseMatrix<double> &L) {
    solver.factorize(L);
    if (solver.info() == Eigen::Success) return 0.0;
    double largest = L.diagonal().cwiseAbs().maxCoeff();
    double shift = largest * DAMPING_FIRST_SHARE;
    for (; shift <= largest; shift *= DAMPING_STEP) {
        solver.setShift(shift);
        solver.factorize(L);
        if (solver.info() == Eigen::Success) break;
    }
    solver.setShift(0.0);
    return shift <= largest ? shift : -1.0;
}

// igl::slim_solve's iteration with the symbolic factorization done once
void slimSolve(Island &island, int maxIterations) {
    igl::SLIMData data;
    Eigen::VectorXi b;
    Eigen::MatrixXd bc(0, 2);
    igl::slim_precompute(island.V, island.T, island.UV, data,
                         igl::MappingEnergyType::SYMMETRIC_DIRICHLET, b, bc,
                         0.0);
    SparseSolver solver;
    Eigen::SparseMatrix<double> L;
    Eigen::VectorXd rhs, solution;
    std::function<double(Eigen::MatrixXd &)> energyOf =
        [&](Eigen::MatrixXd &uv) { return igl::slim::compute_energy(data, uv); };
    for (int i = 0; i < maxIterations; ++i) {
        double before = data.energy - SLIM_ISOMETRIC;
        Eigen::MatrixXd next = data.V_o;
        igl::slim::update_weights_and_closest_rotations(data, next);
        igl::slim::build_linear_system(data, L);
        if (i == 0) solver.analyzePattern(L);
        double damping = factorizeDamped(solver, L);
        if (damping < 0.0) break;
        rhs = data.rhs;
        for (int d = 0; d < 2; ++d)
            rhs.segment(d * data.v_n, data.v_n) += damping * data.V_o.col(d);
        solution = solver.solve(rhs);
        for (int d = 0; d < 2; ++d)
            next.col(d) = solution.segment(d * data.v_n, data.v_n);
        data.energy = igl::flip_avoiding_line_search(
                          data.F, data.V_o, next, energyOf,
                          data.energy * data.mesh_area) /
                      data.mesh_area;
        double excess = data.energy - SLIM_ISOMETRIC;
        if (excess < SLIM_SETTLED || before - excess < SLIM_STOP_SHARE * before)
            break;
    }
    if (data.V_o.rows() == island.UV.rows() && data.V_o.allFinite())
        island.UV = data.V_o;
}

// the biggest island must not be the one left running alone at the end
void solveIslands(std::vector<Island> &islands,
                  const std::vector<std::vector<int>> &loops,
                  int maxIterations) {
    std::vector<size_t> order(islands.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return islands[a].T.rows() > islands[b].T.rows();
    });
    std::mutex progressMutex;
    int done = 0;
    double lastProgress = -1.0;
    tbb::parallel_for(0, static_cast<int>(order.size()), 1, [&](int k) {
        Island &island = islands[order[static_cast<size_t>(k)]];
        const std::vector<int> &loop = loops[order[static_cast<size_t>(k)]];
        if (loop.empty() || !tutteInit(island, loop)) projectToPlane(island);
        slimSolve(island, maxIterations);

        std::lock_guard<std::mutex> lock(progressMutex);
        ++done;
        double progress =
            std::round(100.0 * done / static_cast<double>(islands.size())) /
            100.0;
        if (progress > lastProgress) {
            lastProgress = progress;
            std::printf("progress: %.2f 0 %.2f\n", progress, 1.0 - progress);
            std::fflush(stdout);
        }
    });
}

// a closed island projected to a plane cancels to zero when signed
double absUvArea(const Island &island) {
    double area = 0.0;
    for (Eigen::Index t = 0; t < island.T.rows(); ++t) {
        Eigen::RowVector2d a = island.UV.row(island.T(t, 0));
        Eigen::RowVector2d b = island.UV.row(island.T(t, 1));
        Eigen::RowVector2d c = island.UV.row(island.T(t, 2));
        area += std::abs((b - a).x() * (c - a).y() - (b - a).y() * (c - a).x()) / 2.0;
    }
    return area;
}

// a mostly negative sum means the island came out mirrored
double signedUvArea(const Island &island) {
    double area = 0.0;
    for (Eigen::Index t = 0; t < island.T.rows(); ++t) {
        Eigen::RowVector2d a = island.UV.row(island.T(t, 0));
        Eigen::RowVector2d b = island.UV.row(island.T(t, 1));
        Eigen::RowVector2d c = island.UV.row(island.T(t, 2));
        area += ((b - a).x() * (c - a).y() - (b - a).y() * (c - a).x()) / 2.0;
    }
    return area;
}

// uv area == 3d area per island, blender's average_islands_scale equivalent
void normalizeIsland(Island &island) {
    double signedArea = signedUvArea(island);
    if (signedArea < 0.0) island.UV.col(0) *= -1.0;
    double area = std::abs(signedArea);
    if (area > 0.0 && island.area3d > 0.0)
        island.UV *= std::sqrt(island.area3d / area);
    Eigen::RowVector2d minCorner = island.UV.colwise().minCoeff();
    island.UV.rowwise() -= minCorner;
}

// coarser than blender's pack but overlap-free, which the keep-map check needs
void packIslands(std::vector<Island> &islands) {
    if (islands.empty()) return;
    std::vector<int> order(islands.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::vector<Eigen::RowVector2d> sizes(islands.size());
    double totalArea = 0.0;
    for (size_t i = 0; i < islands.size(); ++i) {
        sizes[i] = islands[i].UV.colwise().maxCoeff();
        totalArea += sizes[i].x() * sizes[i].y();
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return sizes[static_cast<size_t>(a)].y() >
               sizes[static_cast<size_t>(b)].y();
    });
    double width = std::sqrt(std::max(totalArea, 1e-12));
    double margin = 0.001 * width;

    double x = margin, y = margin, shelf = 0.0, extentX = 0.0;
    for (int i : order) {
        const Eigen::RowVector2d &size = sizes[static_cast<size_t>(i)];
        if (x > margin && x + size.x() > width) {
            x = margin;
            y += shelf + margin;
            shelf = 0.0;
        }
        islands[static_cast<size_t>(i)].UV.rowwise() +=
            Eigen::RowVector2d(x, y);
        x += size.x() + margin;
        shelf = std::max(shelf, size.y());
        extentX = std::max(extentX, x);
    }
    double extent = std::max(extentX, y + shelf + margin);
    if (extent <= 0.0) return;
    for (Island &island : islands) island.UV /= extent;
}

bool writeObj(const std::string &path, const PolyMesh &mesh) {
    std::ofstream file(path);
    if (!file) return false;
    file << std::setprecision(9);
    for (const Eigen::Vector3d &v : mesh.verts)
        file << "v " << v.x() << ' ' << v.y() << ' ' << v.z() << '\n';
    for (const Eigen::Vector2d &uv : mesh.uvs)
        file << "vt " << uv.x() << ' ' << uv.y() << '\n';
    for (size_t f = 0; f < mesh.faces.size(); ++f) {
        file << 'f';
        for (size_t k = 0; k < mesh.faces[f].size(); ++k)
            file << ' ' << mesh.faces[f][k] + 1 << '/'
                 << mesh.corners[f][k] + 1;
        file << '\n';
    }
    return static_cast<bool>(file);
}

void emitFailed(const std::string &stem, int code) {
    std::printf("failed: %s %d\n", stem.c_str(), code);
    std::fflush(stdout);
}

}  // namespace

namespace uvgami {

int runFlatten(const std::string &inputPath, const std::string &outputDir,
               int maxIterations, bool packOnly) {
    std::filesystem::path input(inputPath);
    std::string stem = input.stem().string();
    std::printf("start: %s\n", stem.c_str());
    std::fflush(stdout);

    PolyMesh mesh;
    bool hasUV = false;
    if (!readObj(inputPath, mesh, hasUV)) {
        std::fprintf(stderr, "cannot read or parse input: %s\n",
                     inputPath.c_str());
        emitFailed(stem, UVGAMI_RC_FAILED_TO_LOAD_MESH);
        return UVGAMI_RC_FAILED_TO_LOAD_MESH;
    }
    // a nan vertex flows through tutte init and the pack sort into a nan uv
    for (const Eigen::Vector3d &v : mesh.verts) {
        if (!v.allFinite() || v.cwiseAbs().maxCoeff() > 1e15) {
            std::fprintf(stderr, "input has non-finite coordinates\n");
            emitFailed(stem, UVGAMI_RC_INVALID_COORDS);
            return UVGAMI_RC_INVALID_COORDS;
        }
    }

    if (packOnly) {
        if (!hasUV) {
            std::fprintf(stderr, "pack-only needs a uv map in the input\n");
            emitFailed(stem, UVGAMI_RC_INVALID_UV);
            return UVGAMI_RC_INVALID_UV;
        }
    } else {
        std::set<std::pair<int, int>> seams;
        std::string seamsPath =
            (input.parent_path() / (stem + "_seams")).string();
        if (!readSeams(seamsPath, static_cast<int>(mesh.verts.size()), seams)) {
            std::fprintf(stderr, "cannot parse seams file: %s\n",
                         seamsPath.c_str());
            emitFailed(stem, UVGAMI_RC_FAILED_TO_LOAD_MESH);
            return UVGAMI_RC_FAILED_TO_LOAD_MESH;
        }
        weldCorners(mesh, seams);
    }

    int islandCount = 0;
    std::vector<int> faceIsland = faceIslands(mesh, islandCount);
    std::vector<std::vector<int>> islandFaces(static_cast<size_t>(islandCount));
    for (size_t f = 0; f < faceIsland.size(); ++f)
        islandFaces[static_cast<size_t>(faceIsland[f])].push_back(
            static_cast<int>(f));
    std::vector<int> localOf(mesh.uvs.size(), -1);
    std::vector<Island> islands;
    islands.reserve(static_cast<size_t>(islandCount));
    std::vector<std::vector<int>> loops(static_cast<size_t>(islandCount));
    for (int islandId = 0; islandId < islandCount; ++islandId) {
        Island island = buildIsland(
            mesh, std::move(islandFaces[static_cast<size_t>(islandId)]),
            localOf);
        if (packOnly) {
            island.UV.resize(static_cast<Eigen::Index>(island.globalUv.size()),
                             2);
            for (size_t i = 0; i < island.globalUv.size(); ++i)
                island.UV.row(static_cast<Eigen::Index>(i)) =
                    mesh.uvs[static_cast<size_t>(island.globalUv[i])]
                        .transpose();
        } else {
            loops[static_cast<size_t>(islandId)] =
                outerBoundaryLoop(island, mesh, localOf);
        }
        for (int c : island.globalUv) localOf[static_cast<size_t>(c)] = -1;
        islands.push_back(std::move(island));
    }

    if (!packOnly) solveIslands(islands, loops, maxIterations);

    double solvedUvArea = 0.0, restArea = 0.0;
    for (Island &island : islands) {
        // before the normalize, which rescales even a noise-area island
        solvedUvArea += absUvArea(island);
        restArea += island.area3d;
        normalizeIsland(island);
    }

    // pack-only uvs are unit-square scale, not mesh scale
    if (!packOnly && restArea > 0.0 && solvedUvArea <= restArea * 1e-9) {
        std::fprintf(stderr, "flatten came back collapsed\n");
        emitFailed(stem, UVGAMI_RC_FLATTEN_FAILED);
        return UVGAMI_RC_FLATTEN_FAILED;
    }

    packIslands(islands);
    for (const Island &island : islands)
        for (size_t i = 0; i < island.globalUv.size(); ++i)
            mesh.uvs[static_cast<size_t>(island.globalUv[i])] =
                island.UV.row(static_cast<Eigen::Index>(i)).transpose();

    std::filesystem::path outPath =
        std::filesystem::path(outputDir) / (stem + ".obj");
    if (!writeObj(outPath.string(), mesh)) {
        std::fprintf(stderr, "cannot write output: %s\n",
                     outPath.string().c_str());
        emitFailed(stem, UVGAMI_RC_FLATTEN_FAILED);
        return UVGAMI_RC_FLATTEN_FAILED;
    }
    std::printf("done: %s\n", stem.c_str());
    std::fflush(stdout);
    return UVGAMI_RC_SUCCESS;
}

}  // namespace uvgami
