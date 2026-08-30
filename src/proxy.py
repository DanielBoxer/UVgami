"""Unwrap a decimated copy, then read its uv map onto the original.

The engine only ever sees the proxy, which is what makes this fast: every
cut is decided on a few thousand triangles instead of the whole mesh. The
original is never unwrapped, each of its vertices takes the uv of the nearest
proxy face, so the cuts land where the proxy's tears project onto it.

The pipeline is seams.proxy_transfer, plain data only. This module builds the
proxy and reads the meshes into arrays for it, so the work can run in a
worker thread."""

import bmesh
import bpy
import numpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from .seams import proxy_transfer
from .utils.mesh import corner_uvs, face_vertices, loop_totals, new_bmesh, set_bmesh


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

    clean_proxy(obj)
    return True


# neighbour normals this far apart are a triangle folded over the other
FOLD_DOT = -0.8
# a flip can fold another triangle
UNFOLD_ROUNDS = 5


def clean_proxy(obj):
    """Collapsing leaves vertices with no face behind, which the engine reads
    as non-manifold vertices and refuses, and now and then a triangle folded
    over its neighbour, which the transfer reads as two maps for one patch of
    surface."""
    bm = new_bmesh(obj)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    unfold(bm)
    set_bmesh(bm, obj)


def unfold(bm):
    """Flip the longest edge of every folded triangle. A collapse that drags
    a vertex across the edge opposite it leaves the triangle inverted on top
    of the neighbour across that edge, and the flip splits that neighbour
    through the vertex instead."""
    for _ in range(UNFOLD_ROUNDS):
        bm.normal_update()
        inverted = [face for face in bm.faces if _inverted(face)]
        if not inverted:
            return
        for face in inverted:
            # a flip next to it may have removed the face already
            if face.is_valid:
                _flip(max(face.edges, key=lambda edge: edge.calc_length()))


def _flip(edge):
    """The edge's two triangles replaced by the pair across the other
    diagonal. bmesh.ops.rotate_edges leaves a folded edge as it is."""
    if len(edge.link_faces) != 2:
        return
    apexes = [
        next(vert for vert in face.verts if vert not in edge.verts)
        for face in edge.link_faces
    ]
    # a doubled triangle has one apex
    if apexes[0] is apexes[1]:
        return
    if any(other.other_vert(apexes[0]) is apexes[1] for other in apexes[0].link_edges):
        return
    quad = bmesh.utils.face_join(edge.link_faces)
    bmesh.utils.face_split(quad, apexes[0], apexes[1])


def _inverted(face):
    """A face on a fold that points against its own vertices' normals, which
    the properly oriented neighbours around it decide."""
    folded = any(
        len(edge.link_faces) == 2
        and edge.link_faces[0].normal.dot(edge.link_faces[1].normal) < FOLD_DOT
        for edge in face.edges
    )
    if not folded:
        return False
    around = sum((vert.normal for vert in face.verts), Vector())
    return face.normal.dot(around) < 0


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
    # only the sign of the dot is read
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


# in mean proxy edge lengths, how far a ray along the vertex normal may travel
RAY_REACH_EDGES = 1.0
# how far to look for a facing face when the nearest faces away, a thin wall
FACING_SEARCH_EDGES = 2.0
# a hit past this many times the nearest distance skimmed off a wrinkle
HIT_OVER_NEAREST = 3.0


def face_locator(positions, faces):
    """nearest_faces(points, normals): for each point the proxy face it
    stands over and the point to read that face's map at, in the proxy's
    own space.

    A ray along the normal, either way, finds the face under a point that
    sits on a bulge over a crease, where the nearest face is one of the two
    and its plane continued off the face lands elsewhere than the other's.
    The map is read at the hit. Where the ray misses, or travels much
    further than the nearest point is, the nearest facing face is used and
    the map is read at the point itself: reading it at the nearest point
    would pinch every bulge onto the crease line."""
    positions = numpy.asarray(positions, dtype=numpy.float64).reshape(-1, 3)
    tree = BVHTree.FromPolygons(positions.tolist(), [list(face) for face in faces])
    first = numpy.array([face[:2] for face in faces], dtype=numpy.int64)
    edge_lengths = numpy.linalg.norm(
        positions[first[:, 1]] - positions[first[:, 0]], axis=1
    )
    reach = RAY_REACH_EDGES * edge_lengths.mean()
    radius = FACING_SEARCH_EDGES * edge_lengths.mean()

    def facing(normal, direction):
        return (
            normal.x * direction[0] + normal.y * direction[1] + normal.z * direction[2]
        )

    def nearest_faces(points, normals):
        found = numpy.empty(len(points), dtype=numpy.int64)
        surface = numpy.empty((len(points), 3))
        for i, (point, normal) in enumerate(zip(points.tolist(), normals.tolist())):
            _, face_normal, index, nearest_distance = tree.find_nearest(point)
            backward = tuple(-c for c in normal)
            hits = [
                tree.ray_cast(point, normal, reach),
                tree.ray_cast(point, backward, reach),
            ]
            hits = [
                hit for hit in hits if hit[2] is not None and facing(hit[1], normal) > 0
            ]
            if hits:
                location, _, hit_index, distance = min(hits, key=lambda hit: hit[3])
                if distance <= HIT_OVER_NEAREST * nearest_distance:
                    found[i] = hit_index
                    surface[i] = location[:]
                    continue
            if facing(face_normal, normal) < 0:
                nearby = sorted(
                    tree.find_nearest_range(point, radius), key=lambda hit: hit[3]
                )
                for _, other_normal, other, _ in nearby:
                    if facing(other_normal, normal) > 0:
                        index = other
                        break
            found[i] = index
            surface[i] = point
        return found, surface

    return nearest_faces


def transfer_inputs(input_mesh, output):
    """The (dense, proxy) arrays the transfer pipeline reads."""
    data = input_mesh.data
    corners = numpy.empty(len(data.loops), dtype=numpy.int64)
    data.loops.foreach_get("vertex_index", corners)
    dense = {
        "positions": _vertex_array(data, "co"),
        "normals": _vertex_array(data, "normal"),
        "matrix": numpy.array(input_mesh.matrix_world, dtype=numpy.float64),
        "corners": corners,
        "face_sizes": numpy.array(loop_totals(data), dtype=numpy.int64),
    }
    out_data = output.data
    proxy = {
        "positions": _vertex_array(out_data, "co"),
        "matrix": numpy.array(output.matrix_world, dtype=numpy.float64),
        "faces": face_vertices(out_data),
        "corner_uvs": corner_uvs(out_data),
    }
    return dense, proxy
