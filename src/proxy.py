"""Unwrap a decimated copy, then cut the original along its seams.

The engine only ever sees the proxy, and so does the repair, which is what
makes this fast: every cut is decided on a few thousand triangles instead of
the whole mesh. What comes out is a cut network, redrawn on the original by
snapping each cut edge to a path of real edges, and the dense mesh is then
flattened and packed once. Texel density follows the original, not the proxy,
because the original is really unwrapped.

Chart labels cannot carry a cut. Most of what the engine makes is a slit
inside one chart, which separates nothing and so has no boundary to label,
and a single chart output is nothing but slit.

The pipeline is seams.proxy_transfer, plain data only. This module reads the
meshes into arrays and applies the results, so the work between can run in a
worker thread."""

import bmesh
import bpy
import numpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

from .hard_surface import (
    apply_face_uvs,
    apply_seams,
    flatten_engine,
    seam_restrictions,
)
from .seams import proxy_transfer
from .utils.mesh import (
    corner_uvs,
    face_vertices,
    new_bmesh,
    set_bmesh,
)


def triangle_count(obj):
    """Every face fans into loop_total - 2 triangles."""
    return len(obj.data.loops) - 2 * len(obj.data.polygons)


def make_proxy(obj, target_faces):
    """Decimate obj in place to roughly target_faces triangles.

    obj has to be visible in the view layer and carry no other modifiers,
    since the decimate is baked by evaluating the whole object."""
    triangles = triangle_count(obj)
    if triangles <= target_faces:
        return False
    modifier = obj.modifiers.new("UVgami Proxy", "DECIMATE")
    modifier.ratio = target_faces / triangles
    depsgraph = bpy.context.evaluated_depsgraph_get()
    baked = bpy.data.meshes.new_from_object(
        obj.evaluated_get(depsgraph),
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )

    stale = obj.data
    obj.data = baked
    obj.modifiers.remove(modifier)
    bpy.data.meshes.remove(stale)

    if triangle_count(obj) == triangles:
        # a hidden object is left out of the depsgraph
        raise RuntimeError(f"{obj.name} was not decimated, it has to be visible")

    drop_loose_vertices(obj)
    return True


def drop_loose_vertices(obj):
    """Collapsing leaves vertices with no face behind, which the engine reads
    as non-manifold vertices and refuses."""
    bm = new_bmesh(obj)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    set_bmesh(bm, obj)


def bounds_frame(obj):
    """The space of the mesh's own bounding box, so two copies of a model
    line up wherever each one is placed and whatever size it is."""
    corners = [Vector(corner) for corner in obj.bound_box]
    low = Vector([min(corner[axis] for corner in corners) for axis in range(3)])
    high = Vector([max(corner[axis] for corner in corners) for axis in range(3)])
    return Matrix.Scale(1 / max(high - low), 4) @ Matrix.Translation(-(low + high) / 2)


def _vertex_array(data, attribute):
    flat = numpy.empty(len(data.vertices) * 3)
    data.vertices.foreach_get(attribute, flat)
    return flat.reshape(-1, 3)


def _edge_array(data):
    pairs = numpy.empty(len(data.edges) * 2, dtype=numpy.int64)
    data.edges.foreach_get("vertices", pairs)
    return pairs.reshape(-1, 2)


# candidates to pick a facing match from when snapping a cut vertex
NEAREST_VERTS = 8


def _rotation_array(matrix):
    return numpy.array(Matrix(matrix.tolist()).to_3x3().inverted_safe().transposed())


def facing_matcher(
    input_positions,
    input_normals,
    input_matrix,
    output_positions,
    output_normals,
    output_matrix,
):
    """nearest(input_indices, output_indices): each listed output vertex to
    the nearest listed input vertex facing the same way, matched in world
    space.

    Thin walls put the far side of the wall nearest, and a cut snapped
    through the wall would seam both sides at once."""
    matrix = numpy.asarray(input_matrix, dtype=numpy.float64)
    positions = numpy.asarray(input_positions).reshape(-1, 3)
    positions = positions @ matrix[:3, :3].T + matrix[:3, 3]

    normals = numpy.asarray(input_normals).reshape(-1, 3) @ _rotation_array(matrix).T
    lengths = numpy.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    normals /= lengths[:, None]

    out_matrix = numpy.asarray(output_matrix, dtype=numpy.float64)
    queries = numpy.asarray(output_positions).reshape(-1, 3)
    queries = queries @ out_matrix[:3, :3].T + out_matrix[:3, 3]
    # only the sign of the dot is read, so these stay unnormalized
    facings = numpy.asarray(output_normals).reshape(-1, 3)
    facings = facings @ _rotation_array(out_matrix).T

    def nearest(input_indices, output_indices):
        kd = KDTree(len(input_indices))
        for i in input_indices:
            kd.insert(positions[i], int(i))
        kd.balance()
        mapped = []
        for i in output_indices:
            found = kd.find_n(queries[i], NEAREST_VERTS)
            best = found[0][1]
            for _, index, _ in found:
                if normals[index] @ facings[i] > 0:
                    best = index
                    break
            mapped.append(best)
        return mapped

    return nearest


def vertex_map(input_mesh, output, matrix=None, out_matrix=None):
    """Every output vertex's nearest facing input vertex, in world space
    unless a frame is given for each."""
    if matrix is None:
        matrix = input_mesh.matrix_world
    if out_matrix is None:
        out_matrix = output.matrix_world
    nearest = facing_matcher(
        _vertex_array(input_mesh.data, "co"),
        _vertex_array(input_mesh.data, "normal"),
        numpy.array(matrix, dtype=numpy.float64),
        _vertex_array(output.data, "co"),
        _vertex_array(output.data, "normal"),
        numpy.array(out_matrix, dtype=numpy.float64),
    )
    return nearest(
        range(len(input_mesh.data.vertices)), range(len(output.data.vertices))
    )


def snap_cuts(input_mesh, mapped, cuts):
    """Another mesh's cut network redrawn along input_mesh's own edges."""
    data = input_mesh.data
    verts = _vertex_array(data, "co").tolist()
    return proxy_transfer.snap_cuts(verts, _edge_array(data), mapped, cuts)


def restriction_weights(input_mesh):
    """The painted seam restrictions when avoid seams is on."""
    if not bpy.context.scene.uvgami.avoid_seams:
        return None
    return seam_restrictions(input_mesh)


def transfer_inputs(input_mesh, output):
    """The (dense, proxy, weights) arrays the transfer pipeline reads."""
    data = input_mesh.data
    dense = {
        "positions": _vertex_array(data, "co"),
        "normals": _vertex_array(data, "normal"),
        "matrix": numpy.array(input_mesh.matrix_world, dtype=numpy.float64),
        "edges": _edge_array(data),
        "faces": face_vertices(data),
    }
    out_data = output.data
    proxy = {
        "positions": _vertex_array(out_data, "co"),
        "normals": _vertex_array(out_data, "normal"),
        "matrix": numpy.array(output.matrix_world, dtype=numpy.float64),
        "faces": face_vertices(out_data),
        "corner_uvs": corner_uvs(out_data),
    }
    return dense, proxy, restriction_weights(input_mesh)


def finish_transfer(dense, proxy, weights, engine, progress=None, cancelled=None):
    """Map, repair, snap and flatten extracted arrays. No bpy, so this is the
    half that runs off the main thread."""
    nearest = facing_matcher(
        dense["positions"],
        dense["normals"],
        dense["matrix"],
        proxy["positions"],
        proxy["normals"],
        proxy["matrix"],
    )
    return proxy_transfer.finish_proxy(
        dense, proxy, weights, engine, nearest, progress, cancelled
    )


def transfer_cuts(input_mesh, output):
    """Seam the original along the proxy's cuts and unwrap it there."""
    dense, proxy, weights = transfer_inputs(input_mesh, output)
    seams, uvs = finish_transfer(dense, proxy, weights, flatten_engine())
    data = input_mesh.data
    apply_seams(data, seams)
    if not data.uv_layers:
        data.uv_layers.new()
    apply_face_uvs(data, uvs)
