import contextlib

import bmesh
import bmesh.utils
import bpy
import numpy

# re-export, callers import it from here
from ..seams.mesh import split_per_face as split_per_face


def new_bmesh(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    return bm


def loop_totals(mesh):
    totals = numpy.empty(len(mesh.polygons), dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_total", totals)
    return totals.tolist()


def face_vertices(mesh):
    """Per-face vertex index lists, read in bulk."""
    corners = numpy.empty(len(mesh.loops), dtype=numpy.int64)
    mesh.loops.foreach_get("vertex_index", corners)
    return split_per_face(corners.tolist(), loop_totals(mesh))


def vertex_positions(mesh):
    """Vertex positions in the mesh's own space, read in bulk."""
    flat = numpy.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", flat)
    return flat.reshape(-1, 3).tolist()


def loop_starts(mesh):
    """Each face's first loop index, read in bulk."""
    starts = numpy.empty(len(mesh.polygons), dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    return starts


def loop_uvs(mesh):
    """The active layer's uvs, one loop per row, written back with
    set_loop_uvs."""
    flat = numpy.empty(len(mesh.loops) * 2)
    mesh.uv_layers.active.data.foreach_get("uv", flat)
    return flat.reshape(-1, 2)


def set_loop_uvs(mesh, coords):
    mesh.uv_layers.active.data.foreach_set("uv", coords.ravel())


def corner_uvs(mesh):
    """Per-face loop uvs from the active layer, in face vertex order, read in
    bulk."""
    corners = [tuple(uv) for uv in loop_uvs(mesh).tolist()]
    return split_per_face(corners, loop_totals(mesh))


def face_uvs(mesh):
    """corner_uvs rounded, so float noise between loops of one vert doesn't
    read as a seam. Rounded in python: numpy rounds by scaling and lands a ulp
    off on some values, which would split an island this kept together."""
    return [[(round(u, 6), round(v, 6)) for u, v in face] for face in corner_uvs(mesh)]


def triangulate(bm, mesh):
    """Triangulate for engine input. bm has to be fresh from mesh, the quad
    scan reads mesh and splits by face index. BEAUTY alone can give two quads
    the same diagonal (Suzanne's mouth fold), leaving an edge with 4 faces that
    the engines reject as non-manifold, so split conflicting quads safely
    first."""
    _split_conflicting_quads(bm, mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces, quad_method="BEAUTY")


def _edge_keys(tail, head, vertex_count):
    """One integer per edge, the same for both directions."""
    return numpy.minimum(tail, head) * vertex_count + numpy.maximum(tail, head)


def _conflicting_quads(mesh):
    """(face index, diagonal keys, diagonal is a mesh edge) for each quad with
    a diagonal that is an existing edge or another quad's diagonal. Every
    other quad is safe for BEAUTY as is."""
    totals = numpy.empty(len(mesh.polygons), dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_total", totals)
    quads = numpy.flatnonzero(totals == 4)
    if len(quads) == 0:
        return []
    corners = numpy.empty(len(mesh.loops), dtype=numpy.int64)
    mesh.loops.foreach_get("vertex_index", corners)
    edge_vertices = numpy.empty(len(mesh.edges) * 2, dtype=numpy.int64)
    mesh.edges.foreach_get("vertices", edge_vertices)
    edge_vertices = edge_vertices.reshape(-1, 2)

    vertex_count = len(mesh.vertices)
    quad_corners = corners[loop_starts(mesh)[quads][:, None] + numpy.arange(4)]
    diagonals = numpy.stack(
        [
            _edge_keys(quad_corners[:, 0], quad_corners[:, 2], vertex_count),
            _edge_keys(quad_corners[:, 1], quad_corners[:, 3], vertex_count),
        ],
        axis=1,
    )
    edge_keys = numpy.sort(
        _edge_keys(edge_vertices[:, 0], edge_vertices[:, 1], vertex_count)
    )
    # searchsorted beats isin by 7x here
    slots = numpy.searchsorted(edge_keys, diagonals).clip(max=len(edge_keys) - 1)
    is_edge = edge_keys[slots] == diagonals
    _, inverse, counts = numpy.unique(
        diagonals.ravel(), return_inverse=True, return_counts=True
    )
    is_shared = counts[inverse].reshape(diagonals.shape) > 1
    flagged = numpy.flatnonzero((is_edge | is_shared).any(axis=1))
    return list(
        zip(
            quads[flagged].tolist(),
            diagonals[flagged].tolist(),
            is_edge[flagged].tolist(),
        )
    )


def _split_conflicting_quads(bm, mesh):
    conflicting = _conflicting_quads(mesh)
    if not conflicting:
        return
    bm.faces.ensure_lookup_table()
    conflicting = [(bm.faces[f], keys, flags) for f, keys, flags in conflicting]

    claims = {}
    for face, keys, _ in conflicting:
        for key in keys:
            claims.setdefault(key, []).append(face)

    split_keys = set()
    for face, keys, edge_flags in conflicting:
        verts = face.verts
        ends = {keys[0]: (verts[0], verts[2]), keys[1]: (verts[1], verts[3])}
        edge_of = dict(zip(keys, edge_flags))

        def conflicts(key):
            # a split face drops to 3 verts, so resolved partners don't count
            return (
                edge_of[key]
                or key in split_keys
                or any(
                    other is not face and len(other.verts) == 4 for other in claims[key]
                )
            )

        if not any(conflicts(key) for key in keys):
            continue
        safest = min(
            keys,
            key=lambda k: (conflicts(k), (ends[k][0].co - ends[k][1].co).length),
        )
        bmesh.utils.face_split(face, *ends[safest])
        split_keys.add(safest)


def set_bmesh(bm, obj):
    if obj.mode == "EDIT":
        bmesh.update_edit_mesh(obj.data)
    else:
        bm.to_mesh(obj.data)
    bm.free()


def move_to_collection(obj, target):
    for collection in obj.users_collection:
        collection.objects.unlink(obj)
    target.objects.link(obj)


def check_collection(name, parent):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if not bpy.context.scene.user_of_id(collection):
        parent.children.link(collection)
    return collection


NOT_UNWRAPPED_COLLECTION = "UVgami Not Unwrapped"


def mark_not_unwrapped(obj, reason, name=None):
    """Hide obj in the Not Unwrapped collection named reason first, the
    outliner cuts long names short."""
    collection = check_collection(
        NOT_UNWRAPPED_COLLECTION, bpy.context.scene.collection
    )
    move_to_collection(obj, collection)
    obj.name = f"{reason}: {name or obj.name}"
    obj.hide_set(True)
    return obj.name


def check_exists(reference):
    try:
        reference.name
        return True
    except ReferenceError:
        return False


def validate_obj(op, obj, report=False, check_uvs=False):
    if obj.type != "MESH":
        if report:
            op.report({"ERROR"}, "Selected object is not a mesh")
        return False
    if len(obj.data.polygons) == 0:
        if report:
            op.report({"ERROR"}, "Selected object has zero polygons")
        return False
    if check_uvs and not obj.data.uv_layers:
        if report:
            op.report({"ERROR"}, "Selected object doesn't have a UV map")
        return False
    return True


def deselect_all():
    for object in bpy.context.selected_objects:
        object.select_set(False)


def select_uvs():
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")


def set_active_any():
    for obj in bpy.data.objects:
        if obj.type == "MESH" and len(obj.users_collection) != 0 and obj.visible_get():
            bpy.context.view_layer.objects.active = obj
            return obj
    return None


@contextlib.contextmanager
def _shown(objects):
    """A hidden object is left out of objects_in_mode even after mode_set
    succeeds, so the uv operators return CANCELLED without a message."""
    view_layer = bpy.context.view_layer
    targets = set(objects)
    restore = []

    def clear(holder, attr):
        if getattr(holder, attr):
            restore.append((holder, attr, True))
            setattr(holder, attr, False)

    def walk(layer_collection):
        if targets & set(layer_collection.collection.objects):
            clear(layer_collection, "exclude")
            clear(layer_collection, "hide_viewport")
            clear(layer_collection.collection, "hide_viewport")
        for child in layer_collection.children:
            walk(child)

    walk(view_layer.layer_collection)
    for obj in targets:
        clear(obj, "hide_viewport")
        if obj.hide_get(view_layer=view_layer):
            restore.append((obj, None, True))
            obj.hide_set(False, view_layer=view_layer)

    try:
        yield
    finally:
        for holder, attr, value in reversed(restore):
            if attr is None:
                holder.hide_set(value, view_layer=view_layer)
            else:
                setattr(holder, attr, value)


def edit_restore(input, func, *args, **kwargs):
    old_selection = bpy.context.selected_objects
    old_active = bpy.context.view_layer.objects.active

    # a hidden active object (e.g. one just moved to the not unwrapped
    # collection) fails the mode_set poll the same as no active object
    if old_active is None or not old_active.visible_get():
        old_active = set_active_any()

    old_mode = old_active.mode if old_active is not None else "OBJECT"

    if old_active is not None:
        bpy.ops.object.mode_set(mode="OBJECT")

    with _shown(input):
        deselect_all()
        for obj in input:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = input[0]
        bpy.ops.object.mode_set(mode="EDIT")

        func(*args, **kwargs)

        bpy.ops.object.mode_set(mode="OBJECT")

    deselect_all()
    for obj in old_selection:
        obj.select_set(True)
    if old_active is not None:
        bpy.context.view_layer.objects.active = old_active
        bpy.ops.object.mode_set(mode=old_mode)


def in_object_mode(target, func, *args):
    """Run func with target in object mode, which mesh reads and writes
    need."""
    if target.mode == "OBJECT":
        return func(*args)
    old_active = bpy.context.view_layer.objects.active
    old_mode = target.mode
    bpy.ops.object.mode_set(mode="OBJECT")
    try:
        return func(*args)
    finally:
        if check_exists(old_active):
            bpy.context.view_layer.objects.active = old_active
            bpy.ops.object.mode_set(mode=old_mode)


AUTO_SMOOTH_MODIFIER_NAME = "Smooth by Angle"
WEIGHTED_NORMAL_PROPERTIES = (
    "weight",
    "mode",
    "thresh",
    "keep_sharp",
    "vertex_group",
    "invert_vertex_group",
    "use_face_influence",
)


def _is_auto_smooth(modifier):
    return modifier.type == "NODES" and AUTO_SMOOTH_MODIFIER_NAME in modifier.name


def is_shading_modifier(modifier):
    """Smooth by Angle and Weighted Normal only change normals."""
    return _is_auto_smooth(modifier) or modifier.type == "WEIGHTED_NORMAL"


def _node_input_identifiers(modifier):
    return [
        item.identifier
        for item in modifier.node_group.interface.items_tree
        if item.item_type == "SOCKET"
        and item.in_out == "INPUT"
        and item.socket_type != "NodeSocketGeometry"
    ]


# blender 5.2 moved geometry nodes inputs off id properties
def _node_input_values(modifier):
    identifiers = _node_input_identifiers(modifier)
    if hasattr(modifier, "properties"):
        inputs = modifier.properties.inputs
        return {key: getattr(inputs, key).value for key in identifiers}
    return {key: modifier[key] for key in identifiers}


def _set_node_input_values(modifier, values):
    if hasattr(modifier, "properties"):
        inputs = modifier.properties.inputs
        for key, value in values.items():
            getattr(inputs, key).value = value
        return
    for key, value in values.items():
        modifier[key] = value


def get_shading_modifiers(obj):
    records = []
    for modifier in obj.modifiers:
        if _is_auto_smooth(modifier) and modifier.node_group is not None:
            settings = (modifier.node_group.name, _node_input_values(modifier))
            records.append((modifier.type, modifier.name, settings))
        elif modifier.type == "WEIGHTED_NORMAL":
            settings = {
                key: getattr(modifier, key) for key in WEIGHTED_NORMAL_PROPERTIES
            }
            records.append((modifier.type, modifier.name, settings))
    return records


def add_shading_modifiers(obj, records):
    for kind, name, settings in records:
        if kind == "NODES":
            node_group = bpy.data.node_groups.get(settings[0])
            if node_group is None:
                continue
            modifier = obj.modifiers.new(name, kind)
            modifier.node_group = node_group
            modifier.use_pin_to_last = True
            _set_node_input_values(modifier, settings[1])
        else:
            modifier = obj.modifiers.new(name, kind)
            for key, value in settings.items():
                setattr(modifier, key, value)
