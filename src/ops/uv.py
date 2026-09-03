import bmesh
import bpy

# from ..hard_surface import apply_seams, marked_seams
# from ..proxy import bounds_frame, snap_cuts, vertex_map
from ..seams import face_edges, uv_island_groups
from ..similar import find_stacks
from ..utils.mesh import (
    corner_uvs,
    edit_restore,
    face_vertices,
    in_object_mode,
    select_uvs,
    validate_obj,
)


# the bulk reads come back empty for an object in edit mode
def island_stacks(obj, faces=None, edges=None):
    mesh = obj.data
    if mesh.uv_layers.active is None:
        return []
    if faces is None:
        faces = face_vertices(mesh)
    if edges is None:
        edges = face_edges(faces)
    uvs = corner_uvs(mesh)

    stacks = []
    groups = uv_island_groups(faces, uvs, edges)
    for kept, duplicates in find_stacks(groups, uvs):
        anchor_loop = (kept[0], 0)
        anchor = complex(*uvs[kept[0]][0])
        corners = ((fi, ci) for fi in kept for ci in range(len(uvs[fi])))
        reference_loop = max(
            corners, key=lambda fc: abs(complex(*uvs[fc[0]][fc[1]]) - anchor)
        )
        reference = complex(*uvs[reference_loop[0]][reference_loop[1]])
        if reference == anchor:
            # a zero size island can't recover a transform, pack it normally
            continue
        duplicate_faces = [fi for group in duplicates for fi in group]
        kept_loops = {}
        for fi in kept:
            for ci, uv in enumerate(uvs[fi]):
                kept_loops.setdefault(uv, (fi, ci))
        stacks.append(
            (
                kept_loops,
                anchor_loop,
                reference_loop,
                anchor,
                reference,
                duplicate_faces,
            )
        )
    return stacks


# a deselected face is invisible to every uv operator while sync is off
def _deselect_stacked_duplicates(obj, stacks):
    if not stacks:
        return
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    for stack in stacks:
        for fi in stack[-1]:
            bm.faces[fi].select = False


# replays the kept island's similarity transform onto its duplicates
def _restack(obj, stacks):
    if not stacks:
        return
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.active
    bm.faces.ensure_lookup_table()

    def loop_uv(fc):
        return complex(*bm.faces[fc[0]].loops[fc[1]][uv_layer].uv)

    for stack in stacks:
        kept_loops, anchor_loop, reference_loop, anchor, reference, duplicate_faces = (
            stack
        )
        new_anchor = loop_uv(anchor_loop)
        ratio = (loop_uv(reference_loop) - new_anchor) / (reference - anchor)
        # the next pack must still see the stack as bit equal copies
        exact = {
            uv: tuple(bm.faces[fi].loops[ci][uv_layer].uv)
            for uv, (fi, ci) in kept_loops.items()
        }
        for fi in duplicate_faces:
            for loop in bm.faces[fi].loops:
                old = tuple(loop[uv_layer].uv)
                snapped = exact.get(old)
                if snapped is None:
                    moved = new_anchor + ratio * (complex(*old) - anchor)
                    snapped = (moved.real, moved.imag)
                loop[uv_layer].uv = snapped
    bmesh.update_edit_mesh(obj.data)


# topology maps an object to its (faces, edges) for a caller that read them already
def pack_objects(objects, topology=None):
    if not objects:
        return
    topology = topology or {}

    def read_and_pack():
        stacks_of = {obj: island_stacks(obj, *topology.get(obj, ())) for obj in objects}
        edit_restore(objects, pack, stacks_of)

    in_object_mode(objects[0], read_and_pack)


# blender's merge_overlap can't keep stacks together
def pack(stacks_of):
    tool_settings = bpy.context.scene.tool_settings
    old_sync = tool_settings.use_uv_select_sync
    # with sync on, uv operators follow the mesh selection differently per version
    tool_settings.use_uv_select_sync = False
    try:
        select_uvs()
        # a rotated twin's 3d area computes a few ulps apart
        for obj, stacks in stacks_of.items():
            _deselect_stacked_duplicates(obj, stacks)
        if bpy.context.scene.uvgami.fix_scale:
            bpy.ops.uv.average_islands_scale()
        bpy.ops.uv.pack_islands(
            margin=bpy.context.scene.uvgami.margin, rotate_method="AXIS_ALIGNED"
        )
        for obj, stacks in stacks_of.items():
            _restack(obj, stacks)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.select_all(action="DESELECT")
    finally:
        tool_settings.use_uv_select_sync = old_sync


def show_seams():
    select_uvs()
    bpy.ops.uv.mark_seam(clear=True)
    bpy.ops.uv.seams_from_islands()
    bpy.ops.uv.select_all(action="DESELECT")
    bpy.ops.mesh.select_all(action="DESELECT")


# class UVGAMI_OT_transfer_seams(bpy.types.Operator):
#     bl_idname = "uvgami.transfer_seams"
#     bl_label = "Transfer Seams"
#     bl_description = (
#         "Copy the seams of the other selected object onto the active object"
#     )
#     bl_options = {"UNDO"}
#
#     @classmethod
#     def poll(cls, context):
#         return context.mode == "OBJECT"
#
#     def execute(self, context):
#         target = context.view_layer.objects.active
#         if target is None or not validate_obj(self, target, report=True):
#             return {"CANCELLED"}
#         others = [obj for obj in context.selected_objects if obj is not target]
#         if len(others) != 1:
#             self.report({"ERROR"}, "Select one other object to copy the seams from")
#             return {"CANCELLED"}
#         donor = others[0]
#         if not validate_obj(self, donor, report=True):
#             return {"CANCELLED"}
#         seams = marked_seams(donor.data)
#         if not seams:
#             self.report({"ERROR"}, f"{donor.name} has no seams")
#             return {"CANCELLED"}
#
#         mapped = vertex_map(target, donor, bounds_frame(target), bounds_frame(donor))
#         apply_seams(target.data, snap_cuts(target, mapped, seams))
#         self.report({"INFO"}, f"Transferred {len(seams)} seams from {donor.name}")
#         return {"FINISHED"}


class UVGAMI_OT_pack(bpy.types.Operator):
    bl_idname = "uvgami.pack"
    bl_label = "Pack"
    bl_description = "Pack UVs with Blender's packer"
    bl_options = {"UNDO"}

    def execute(self, context):
        combine_uvs = context.scene.uvgami.combine_uvs
        valid_objs = []

        for obj in context.selected_objects:
            if not validate_obj(self, obj, check_uvs=True):
                continue
            if combine_uvs:
                valid_objs.append(obj)
            else:
                pack_objects([obj])

        if combine_uvs:
            pack_objects(valid_objs)

        return {"FINISHED"}
