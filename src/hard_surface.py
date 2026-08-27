import sys

import bpy
import numpy

from .ops.guides import SEAM_RESTRICTIONS_GROUP
from .seams import (
    CREASE_ANGLE,
    FlattenEngine,
    FlattenError,
    check_cancelled,
    is_hard_surface,
    preseed_uvs,
    vertex_components,
)
from .utils.mesh import (
    face_vertices,
    loop_starts,
    loop_uvs,
    set_loop_uvs,
    vertex_positions,
)


def flatten_engine():
    """Client for the optcuts flatten mode, resolved like an unwrap run."""
    # imported here: engines imports this module back
    from .engines import get_engine
    from .utils.paths import get_extension_dir_path, get_preferences

    path, error = get_engine("OPTCUTS").validate(get_preferences())
    if error is not None:
        raise FlattenError(error)
    return FlattenEngine(path, get_extension_dir_path() / "preseed")


def seam_workers():
    """The interpreter the seam worker processes run on: Blender's own
    python, started the way Blender starts it (isolated from the user's
    site and PYTHONPATH)."""
    return sys.executable, list(bpy.app.python_args)


def seam_restrictions(obj):
    """Per-vertex weights from the painted guide, the same group the engine
    reads. Higher repels seams."""
    group = obj.vertex_groups.get(SEAM_RESTRICTIONS_GROUP)
    if group is None:
        return None
    weights = {}
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == group.index and g.weight:
                weights[v.index] = g.weight
                break
    return weights or None


def hard_faces(verts, faces, marks, marked="NONE", cancelled=None):
    """Face indices of the loose parts worth preseeding, for auto mode.

    Each part classifies on its own geometry. A part carrying marked seams
    counts as hard whenever marks are in use: the user placed seams there
    deliberately. With marked ONLY detection never runs, so the marked parts
    are the whole hard set. cancelled is checked per part."""
    marked_verts = {v for edge in marks for v in edge} if marked != "NONE" else set()
    hard = set()
    for comp in vertex_components(faces):
        check_cancelled(cancelled)
        if (marked_verts and marked_verts & {v for fi in comp for v in faces[fi]}) or (
            marked != "ONLY" and is_hard_surface(verts, [faces[fi] for fi in comp])
        ):
            hard.update(comp)
    return hard


def auto_hard_faces(obj, marked="NONE"):
    mesh = obj.data
    verts = vertex_positions(mesh)
    faces = face_vertices(mesh)
    return hard_faces(verts, faces, marked_seams(mesh), marked)


def _packed(pairs):
    """(low, high) index pairs as one integer each, so numpy can match them."""
    return numpy.fromiter(
        ((a << 32) | b for a, b in pairs), dtype=numpy.int64, count=len(pairs)
    )


def _edge_keys(mesh):
    pairs = numpy.empty(len(mesh.edges) * 2, dtype=numpy.int64)
    mesh.edges.foreach_get("vertices", pairs)
    pairs = pairs.reshape(-1, 2)
    return (pairs.min(axis=1) << 32) | pairs.max(axis=1)


def apply_seams(mesh, seams):
    """Mark exactly these edges as seams, given as (low, high) index pairs."""
    mesh.edges.foreach_set("use_seam", numpy.isin(_edge_keys(mesh), _packed(seams)))


def apply_seams_except_faces(mesh, faces, seams):
    """Like apply_seams, but the edges of these faces keep their marks."""
    loop_total = numpy.empty(len(mesh.polygons), dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_total", loop_total)
    loop_edges = numpy.empty(len(mesh.loops), dtype=numpy.int64)
    mesh.loops.foreach_get("edge_index", loop_edges)
    kept_faces = numpy.zeros(len(mesh.polygons), dtype=bool)
    kept_faces[list(faces)] = True
    kept = numpy.zeros(len(mesh.edges), dtype=bool)
    kept[loop_edges[numpy.repeat(kept_faces, loop_total)]] = True
    flags = numpy.isin(_edge_keys(mesh), _packed(seams))
    flags[kept] = seam_flags(mesh)[kept]
    mesh.edges.foreach_set("use_seam", flags)


def apply_interior_seams(mesh, interior, seams):
    """Mark the seams on the interior edges only, leaving every other edge's
    mark as it was."""
    keys = _edge_keys(mesh)
    flags = seam_flags(mesh)
    inside = numpy.isin(keys, _packed(interior))
    flags[inside] = numpy.isin(keys[inside], _packed(seams))
    mesh.edges.foreach_set("use_seam", flags)


def seam_flags(mesh):
    """Per edge, whether it carries a seam mark."""
    flags = numpy.empty(len(mesh.edges), dtype=bool)
    mesh.edges.foreach_get("use_seam", flags)
    return flags


def marked_seams(mesh):
    pairs = numpy.empty(len(mesh.edges) * 2, dtype=numpy.int64)
    mesh.edges.foreach_get("vertices", pairs)
    pairs = numpy.sort(pairs.reshape(-1, 2)[seam_flags(mesh)], axis=1)
    return {(a, b) for a, b in pairs.tolist()}


def apply_face_uvs(mesh, uvs, only=None):
    """Write per-face corner uvs into the active layer."""
    layer = mesh.uv_layers.active.data
    if only is None:
        flat = [c for face in uvs for uv in face for c in uv]
        layer.foreach_set("uv", flat)
        return

    coords = loop_uvs(mesh)
    starts = loop_starts(mesh)
    for f in only:
        start = int(starts[f])
        coords[start : start + len(uvs[f])] = uvs[f]
    set_loop_uvs(mesh, coords)


def preseed_work(obj, angle, marked="NONE", weights=None, auto=False, mirrors=None):
    """build_seam_uvs split for a worker thread: compute is bpy-free and safe
    off the main thread, apply(compute(cancelled)) writes the result back.
    compute returns None when there is nothing to preseed (no hard parts in
    auto, or an empty seam set on a closed mesh), which apply passes through
    as False, and raises seams.Cancelled when the caller gave up on it."""
    mesh = obj.data
    verts = vertex_positions(mesh)
    faces = face_vertices(mesh)
    marks = marked_seams(mesh) if (marked != "NONE" or auto) else frozenset()
    engine = flatten_engine()
    python = seam_workers()

    def compute(cancelled=None):
        only = None
        if auto:
            only = hard_faces(verts, faces, marks, marked, cancelled)
            if not only:
                return None
            if len(only) == len(faces):
                only = None
        result = preseed_uvs(
            engine,
            verts,
            faces,
            angle,
            marked,
            weights,
            only,
            marks,
            mirrors,
            cancelled,
            python,
        )
        if result is None:
            return None
        seams, uvs = result
        return seams, uvs, only

    def apply(result):
        if result is None:
            return False
        seams, uvs, only = result
        apply_seams(mesh, seams)
        if not mesh.uv_layers:
            mesh.uv_layers.new()
        apply_face_uvs(mesh, uvs, sorted(only) if only is not None else None)
        return True

    return compute, apply


def build_seam_uvs(obj, angle=CREASE_ANGLE, marked="NONE", weights=None, only=None):
    """Seam the strip-merged feature boundaries, then flatten into the uv map.

    marked says what the mesh's own seam marks do. ONLY takes them as the whole
    starting set with no detection, for hand-edited marks after a Mark Seams.
    ADD detects as usual and forces the marks on top, cutting from the
    partition on so no merge pass can dissolve one: a detected seam running
    beside a marked one gives way instead of leaving a ribbon between the two.
    An island the flatten ruins ships as-is: the engine rejects it and recuts
    it with its own search, so hand-placed seams on healthy islands come
    through untouched.

    weights are the painted restrictions, which steer every cut this module
    places, the same paint the engine reads for the cuts it makes itself.

    only restricts everything to those face indices, whole loose parts in
    auto mode: detection and the flatten skip the other parts, whose uvs are
    left exactly as they were.

    The solve runs outside Blender, in seams.preseed against the engine's
    flatten mode. This function only reads the mesh into arrays and writes
    the result back. Returns False when the seam set came out empty on a
    closed mesh, leaving the mesh untouched."""
    mesh = obj.data
    verts = vertex_positions(mesh)
    faces = face_vertices(mesh)
    result = preseed_uvs(
        flatten_engine(),
        verts,
        faces,
        angle,
        marked,
        weights,
        only,
        marked_seams(mesh) if marked != "NONE" else frozenset(),
        python=seam_workers(),
    )
    if result is None:
        return False
    seams, uvs = result
    apply_seams(mesh, seams)
    if not mesh.uv_layers:
        mesh.uv_layers.new()
    apply_face_uvs(mesh, uvs, sorted(only) if only is not None else None)
    return True
