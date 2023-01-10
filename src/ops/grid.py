# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
from ..utils import validate_obj, switch_shading


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
    grid_mat = None
    if "UVgami_grid" in bpy.data.materials:
        grid_mat = bpy.data.materials.get("UVgami_grid")
    else:
        # create new material
        grid_mat = bpy.data.materials.new("UVgami_grid")
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


def add_grid(obj, grid_mat):
    # store all materials
    materials = [slot.material for slot in obj.material_slots]

    if grid_mat not in materials:
        obj.data.materials.clear()

        # add materials back with grid as the first active material
        obj.data.materials.append(grid_mat)
        for m in materials:
            obj.data.materials.append(m)


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
                for material_idx, material in enumerate(obj.data.materials):
                    if material.name == "UVgami_grid":
                        obj.data.materials.pop(index=material_idx)

        switch_shading("SOLID")

        return {"FINISHED"}
