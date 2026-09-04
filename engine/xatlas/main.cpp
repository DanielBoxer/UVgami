// dependency-free cli wrapping xatlas as a uvgami unwrap engine.
// stdout is a parsed protocol (start:/progress:/done:/failed:), diagnostics go
// to stderr. see AGENTS.md in the repo root.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "xatlas.h"

namespace {

constexpr int EXIT_OK = 0;
constexpr int CODE_BAD_ARGS = 2;      // bad args or unreadable/unparsable input
constexpr int CODE_INVALID_GEOMETRY = 3; // no faces, degenerate, AddMesh error
constexpr int CODE_GENERATE_FAILED = 4;  // generate/pack failed or empty atlas

struct Vec3 {
    float x, y, z;
};

// weights per ProgressCategory (AddMesh, ComputeCharts, PackCharts,
// BuildOutputMeshes); chart computation dominates. must sum to 1.
constexpr double kCategoryWeight[4] = {0.05, 0.70, 0.20, 0.05};

// the progress callback fires from xatlas worker threads, so stdout writes and
// the emit state are serialized here.
struct ProgressState {
    std::mutex mtx;
    double category[4] = {0.0, 0.0, 0.0, 0.0};
    double last_emitted = -1.0;
};

void emit_progress_locked(ProgressState &state) {
    double overall = 0.0;
    for (int i = 0; i < 4; ++i) {
        overall += kCategoryWeight[i] * state.category[i];
    }
    double rounded = std::round(overall * 100.0) / 100.0;
    if (rounded > state.last_emitted) {
        state.last_emitted = rounded;
        std::printf("progress: %.2f 0 %.2f\n", rounded, 1.0 - rounded);
        std::fflush(stdout);
    }
}

bool progress_callback(xatlas::ProgressCategory category, int progress,
                       void *user_data) {
    auto *state = static_cast<ProgressState *>(user_data);
    std::lock_guard<std::mutex> lock(state->mtx);
    int index = static_cast<int>(category);
    if (index >= 0 && index < 4) {
        state->category[index] = static_cast<double>(progress) / 100.0;
    }
    emit_progress_locked(*state);
    return true;
}

// finite positive float from an argument value, false on anything else
bool parse_positive_float(const std::string &text, float &out) {
    try {
        size_t consumed = 0;
        float value = std::stof(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value) || value <= 0.0f) {
            return false;
        }
        out = value;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

// one input vertex index from an f-token: "v", "v/vt", "v//vn", "v/vt/vn".
// negative indices are relative to the vertices seen so far. returns false on
// parse failure or out-of-range.
bool parse_face_index(const std::string &token, int vertex_count, uint32_t &out) {
    size_t slash = token.find('/');
    std::string index_str =
        slash == std::string::npos ? token : token.substr(0, slash);
    if (index_str.empty()) {
        return false;
    }
    long idx = 0;
    try {
        size_t consumed = 0;
        idx = std::stol(index_str, &consumed);
        if (consumed != index_str.size()) {
            return false;
        }
    } catch (const std::exception &) {
        return false;
    }
    if (idx < 0) {
        idx = vertex_count + idx;  // relative to current vertex count
    } else {
        idx -= 1;  // 1-based to 0-based
    }
    if (idx < 0 || idx >= vertex_count) {
        return false;
    }
    out = static_cast<uint32_t>(idx);
    return true;
}

// minimal obj reader: v and f only, everything else ignored. faces with more
// than three vertices are fan-triangulated. the whole file is one mesh.
bool read_obj(const std::string &path, std::vector<Vec3> &positions,
              std::vector<uint32_t> &indices) {
    std::ifstream file(path);
    if (!file) {
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.size() < 2) {
            continue;
        }
        if (line[0] == 'v' && (line[1] == ' ' || line[1] == '\t')) {
            std::istringstream stream(line.substr(1));
            Vec3 v{};
            if (!(stream >> v.x >> v.y >> v.z)) {
                return false;
            }
            positions.push_back(v);
        } else if (line[0] == 'f' && (line[1] == ' ' || line[1] == '\t')) {
            std::istringstream stream(line.substr(1));
            std::string token;
            std::vector<uint32_t> face;
            int vertex_count = static_cast<int>(positions.size());
            while (stream >> token) {
                uint32_t idx = 0;
                if (!parse_face_index(token, vertex_count, idx)) {
                    return false;
                }
                face.push_back(idx);
            }
            if (face.size() < 3) {
                return false;
            }
            for (size_t i = 1; i + 1 < face.size(); ++i) {
                indices.push_back(face[0]);
                indices.push_back(face[i]);
                indices.push_back(face[i + 1]);
            }
        }
    }
    return true;
}

// a v per atlas vertex would split every chart into disconnected geometry
bool write_obj(const std::string &path, const std::vector<Vec3> &positions,
               const xatlas::Mesh &mesh, float width, float height) {
    std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
    }
    std::ofstream file(path);
    if (!file) {
        return false;
    }
    file << std::setprecision(9);
    for (const Vec3 &p : positions) {
        file << "v " << p.x << ' ' << p.y << ' ' << p.z << '\n';
    }
    for (uint32_t i = 0; i < mesh.vertexCount; ++i) {
        const xatlas::Vertex &vertex = mesh.vertexArray[i];
        file << "vt " << vertex.uv[0] / width << ' ' << vertex.uv[1] / height
             << '\n';
    }
    for (uint32_t i = 0; i + 2 < mesh.indexCount; i += 3) {
        file << 'f';
        for (uint32_t k = 0; k < 3; ++k) {
            uint32_t t = mesh.indexArray[i + k];
            file << ' ' << mesh.vertexArray[t].xref + 1 << '/' << t + 1;
        }
        file << '\n';
    }
    return static_cast<bool>(file);
}

// xatlas drops faces under kAreaEpsilon, so shrinking loses the thinnest triangles
std::vector<Vec3> grown_positions(const std::vector<Vec3> &positions) {
    Vec3 low = positions[0];
    Vec3 high = positions[0];
    for (const Vec3 &p : positions) {
        low.x = std::min(low.x, p.x);
        low.y = std::min(low.y, p.y);
        low.z = std::min(low.z, p.z);
        high.x = std::max(high.x, p.x);
        high.y = std::max(high.y, p.y);
        high.z = std::max(high.z, p.z);
    }
    float extent = std::max({high.x - low.x, high.y - low.y, high.z - low.z});
    if (!std::isfinite(extent) || extent <= 0.0f || extent >= 1.0f) {
        return positions;
    }
    std::vector<Vec3> scaled;
    scaled.reserve(positions.size());
    for (const Vec3 &p : positions) {
        scaled.push_back({p.x / extent, p.y / extent, p.z / extent});
    }
    return scaled;
}

void emit_failed(const std::string &stem, int code) {
    std::printf("failed: %s %d\n", stem.c_str(), code);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char **argv) {
    std::string input;
    std::string output;
    xatlas::ChartOptions chart_options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            input = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output = argv[++i];
        } else if (arg == "--max-cost" && i + 1 < argc) {
            if (!parse_positive_float(argv[++i], chart_options.maxCost)) {
                std::fprintf(stderr, "--max-cost must be a positive number\n");
                return CODE_BAD_ARGS;
            }
        } else {
            std::fprintf(stderr, "unknown or malformed argument: %s\n",
                         arg.c_str());
            return CODE_BAD_ARGS;
        }
    }
    if (input.empty() || output.empty()) {
        std::fprintf(stderr,
                     "usage: xatlas -i <input.obj> -o <output.obj>"
                     " [--max-cost <float>]\n");
        return CODE_BAD_ARGS;
    }

    std::string stem = std::filesystem::path(input).stem().string();
    std::printf("start: %s\n", stem.c_str());
    std::fflush(stdout);

    std::vector<Vec3> positions;
    std::vector<uint32_t> indices;
    if (!read_obj(input, positions, indices)) {
        std::fprintf(stderr, "cannot read or parse input: %s\n", input.c_str());
        emit_failed(stem, CODE_BAD_ARGS);
        return CODE_BAD_ARGS;
    }
    if (indices.empty() || positions.empty()) {
        std::fprintf(stderr, "input has no faces or vertices\n");
        emit_failed(stem, CODE_INVALID_GEOMETRY);
        return CODE_INVALID_GEOMETRY;
    }

    xatlas::Atlas *atlas = xatlas::Create();
    ProgressState progress_state;
    xatlas::SetProgressCallback(atlas, progress_callback, &progress_state);

    // write_obj still writes the input positions, only xatlas sees the grown ones
    const std::vector<Vec3> scaled = grown_positions(positions);

    xatlas::MeshDecl decl;
    decl.vertexCount = static_cast<uint32_t>(scaled.size());
    decl.vertexPositionData = scaled.data();
    decl.vertexPositionStride = sizeof(Vec3);
    decl.indexCount = static_cast<uint32_t>(indices.size());
    decl.indexData = indices.data();
    decl.indexFormat = xatlas::IndexFormat::UInt32;

    xatlas::AddMeshError add_error = xatlas::AddMesh(atlas, decl, 1);
    if (add_error != xatlas::AddMeshError::Success) {
        std::fprintf(stderr, "AddMesh failed: %s\n",
                     xatlas::StringForEnum(add_error));
        xatlas::Destroy(atlas);
        emit_failed(stem, CODE_INVALID_GEOMETRY);
        return CODE_INVALID_GEOMETRY;
    }

    // the parameterization leaves whole charts mirrored without this
    chart_options.fixWinding = true;
    xatlas::Generate(atlas, chart_options);

    if (atlas->meshCount == 0 || atlas->width == 0 || atlas->height == 0) {
        std::fprintf(stderr, "generate produced an empty atlas\n");
        xatlas::Destroy(atlas);
        emit_failed(stem, CODE_GENERATE_FAILED);
        return CODE_GENERATE_FAILED;
    }
    const xatlas::Mesh &mesh = atlas->meshes[0];
    if (mesh.vertexCount == 0 || mesh.indexCount == 0) {
        std::fprintf(stderr, "generate produced no output geometry\n");
        xatlas::Destroy(atlas);
        emit_failed(stem, CODE_GENERATE_FAILED);
        return CODE_GENERATE_FAILED;
    }

    bool written = write_obj(output, positions, mesh,
                             static_cast<float>(atlas->width),
                             static_cast<float>(atlas->height));
    xatlas::Destroy(atlas);
    if (!written) {
        std::fprintf(stderr, "cannot write output: %s\n", output.c_str());
        emit_failed(stem, CODE_GENERATE_FAILED);
        return CODE_GENERATE_FAILED;
    }

    std::printf("done: %s\n", stem.c_str());
    std::fflush(stdout);
    return EXIT_OK;
}
