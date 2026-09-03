"""The painted weight map: what Avoid Seams and Reduce Stretching do to the
unwrap, and what the seed restriction generators write into it."""

import bpy
import pytest
from bl_ext.user_default.UVgami.src.ops.guides import SEAM_RESTRICTIONS_GROUP
from bl_ext.user_default.UVgami.src.seams.rectify import flatten_distortion
from bl_ext.user_default.UVgami.src.utils.mesh import (
    face_uvs,
    face_vertices,
    vertex_positions,
)
from blender_fixtures import needs_engine

pytestmark = [needs_engine, pytest.mark.smoke]

# a sphere cannot flatten without stretch, so a region has something to win
SEGMENTS = 32
RINGS = 16
OUTPUT_NAME = "sphere_unwrapped"


def add_sphere(name="sphere"):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=SEGMENTS, ring_count=RINGS)
    obj = bpy.context.active_object
    obj.name = name
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


def paint_top_half(obj):
    """The map is a vertex group, so a script paints it the same way the modal
    brush does."""
    group = obj.vertex_groups.new(name=SEAM_RESTRICTIONS_GROUP)
    group.add([v.index for v in obj.data.vertices if v.co.z > 0], 1.0, "REPLACE")


def unwrap_again(sphere, unwrap, outputs):
    """A second run on the same mesh, with the first output gone so the name
    is free and the input is back the way the operator found it."""
    bpy.data.objects.remove(outputs()[OUTPUT_NAME])
    sphere.hide_set(False)
    bpy.ops.object.select_all(action="DESELECT")
    sphere.select_set(True)
    bpy.context.view_layer.objects.active = sphere
    unwrap()
    return outputs()[OUTPUT_NAME]


def region_distortion(obj):
    """Symmetric Dirichlet over the top half and over the bottom, 4.0 at
    isometry. The output is a new mesh, so the halves come back by height."""
    mesh = obj.data
    verts = vertex_positions(mesh)
    faces = face_vertices(mesh)
    uvs = face_uvs(mesh)
    top, bottom = [], []
    for index, face in enumerate(faces):
        height = sum(verts[v][2] for v in face) / len(face)
        (top if height > 0 else bottom).append(index)
    return (
        flatten_distortion(verts, faces, uvs, top),
        flatten_distortion(verts, faces, uvs, bottom),
    )


def top_half_seams(obj):
    positions = vertex_positions(obj.data)
    return sum(
        1
        for edge in obj.data.edges
        if edge.use_seam and sum(positions[v][2] for v in edge.vertices) / 2 > 0
    )


def group_weights(obj):
    group = obj.vertex_groups[SEAM_RESTRICTIONS_GROUP]
    weights = {}
    for vertex in obj.data.vertices:
        for entry in vertex.groups:
            if entry.group == group.index:
                weights[vertex.index] = entry.weight
    return weights


def test_reduce_stretching_lowers_the_painted_region_distortion(unwrap, outputs):
    props = bpy.context.scene.uvgami
    sphere = add_sphere()
    unwrap()
    plain_top, _ = region_distortion(outputs()[OUTPUT_NAME])

    paint_top_half(sphere)
    props.use_weights = True
    props.weight_mode = "STRETCH"
    painted_top, painted_bottom = region_distortion(
        unwrap_again(sphere, unwrap, outputs)
    )

    assert painted_top < plain_top
    # the unpainted half pays for it
    assert painted_top < painted_bottom


def test_avoid_seams_moves_the_seams_off_the_painted_region(unwrap, outputs):
    props = bpy.context.scene.uvgami
    sphere = add_sphere()
    unwrap()
    plain = top_half_seams(outputs()[OUTPUT_NAME])

    paint_top_half(sphere)
    props.use_weights = True
    props.weight_mode = "SEAMS"
    painted = top_half_seams(unwrap_again(sphere, unwrap, outputs))

    assert plain > 0
    assert painted == 0


def test_crevices_weights_the_exposed_side_above_the_hole():
    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.3)
    torus = bpy.context.active_object

    bpy.ops.uvgami.seed_restrictions(mode="CREVICES")

    weights = group_weights(torus)
    outer = [
        weights.get(v.index, 0.0) for v in torus.data.vertices if v.co.xy.length > 1
    ]
    inner = [
        weights.get(v.index, 0.0) for v in torus.data.vertices if v.co.xy.length < 1
    ]
    assert bpy.context.scene.uvgami.use_weights
    assert sum(outer) / len(outer) > 2 * sum(inner) / len(inner)


def test_from_view_refuses_outside_the_3d_viewport():
    bpy.ops.mesh.primitive_cube_add()
    with pytest.raises(RuntimeError, match="Only available in the 3D viewport"):
        bpy.ops.uvgami.seed_restrictions(mode="VIEW")
