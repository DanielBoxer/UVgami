import bpy

from ..utils.mesh import validate_obj
from ..utils.ui import switch_shading


def make_grid_img():
    if "UVgami_UV_grid" not in bpy.data.images:
        bpy.ops.image.new(name="UVgami_UV_grid", generated_type="UV_GRID")
    grid_img = bpy.data.images.get("UVgami_UV_grid")

    props = bpy.context.scene.uvgami
    if props.grid_type == "UV":
        grid_img.generated_type = "UV_GRID"
    else:
        grid_img.generated_type = "COLOR_GRID"

    res = props.grid_res
    grid_img.generated_width = res
    grid_img.generated_height = res

    return grid_img


def make_grid_mat(grid_img):
    grid_mat = bpy.data.materials.get("UVgami_grid")
    if grid_mat is not None:
        return grid_mat

    grid_mat = bpy.data.materials.new("UVgami_grid")
    # 5.0 builds the node tree here and deprecates the flag
    if bpy.app.version < (5, 0, 0):
        grid_mat.use_nodes = True
    tree_nodes = grid_mat.node_tree.nodes
    nodes = (
        tree_nodes.new(type="ShaderNodeTexImage"),
        tree_nodes.get("Principled BSDF"),
        tree_nodes.get("Material Output"),
    )
    nodes[0].image = grid_img
    nodes[0].location = (-300, 300)
    grid_mat.node_tree.links.new(nodes[0].outputs[0], nodes[1].inputs[0])
    for node in nodes:
        node.select = False
        node.hide = True

    return grid_mat


SAVED_INDICES = "uvgami_material_indices"


def add_grid(obj, grid_mat):
    if any(slot.material is grid_mat for slot in obj.material_slots):
        return

    # appending and pointing every face at the new slot leaves the existing
    # slots alone, where rebuilding them drops each one's object level material
    mesh = obj.data
    indices = [0] * len(mesh.polygons)
    mesh.polygons.foreach_get("material_index", indices)
    mesh[SAVED_INDICES] = indices

    mesh.materials.append(grid_mat)
    grid_index = len(mesh.materials) - 1
    mesh.polygons.foreach_set("material_index", [grid_index] * len(mesh.polygons))
    obj.active_material_index = grid_index


def remove_grid(obj):
    mesh = obj.data
    for material_idx, material in enumerate(mesh.materials):
        if material is not None and material.name == "UVgami_grid":
            mesh.materials.pop(index=material_idx)
            break

    indices = mesh.get(SAVED_INDICES)
    if indices is None:
        return
    del mesh[SAVED_INDICES]

    if len(indices) == len(mesh.polygons):
        mesh.polygons.foreach_set("material_index", indices)


class UVGAMI_OT_add_grid(bpy.types.Operator):
    bl_idname = "uvgami.add_grid"
    bl_label = "Add Grid"
    bl_description = "Add grid material to all selected meshes"
    bl_options = {"UNDO"}

    def execute(self, context):
        grid_img = make_grid_img()
        grid_mat = make_grid_mat(grid_img)

        valid_count = 0
        for obj in context.selected_objects:
            if validate_obj(self, obj):
                add_grid(obj, grid_mat)
                valid_count += 1

        if valid_count > 0:
            switch_shading("MATERIAL")

        self.report({"INFO"}, "Added UV grid")
        return {"FINISHED"}


class UVGAMI_OT_remove_grid(bpy.types.Operator):
    bl_idname = "uvgami.remove_grid"
    bl_label = "Remove Grid"
    bl_description = "Remove grid material from all selected meshes"
    bl_options = {"UNDO"}

    def execute(self, context):
        for obj in context.selected_objects:
            if validate_obj(self, obj):
                remove_grid(obj)

        switch_shading("SOLID")

        return {"FINISHED"}
