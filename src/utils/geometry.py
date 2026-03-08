import math

import bmesh
import mathutils
import numpy

from .mesh import new_bmesh, set_bmesh


def set_origin(obj, point):
    mw = obj.matrix_world
    obj.data.transform(mathutils.Matrix.Translation(-(mw.inverted() @ point)))
    mw.translation += point - mw.translation


def calc_center(obj):
    lc = 0.125 * sum((mathutils.Vector(co) for co in obj.bound_box), mathutils.Vector())
    center = obj.matrix_world @ lc
    return center


def apply_transforms(obj):
    location, _, scale = obj.matrix_basis.decompose()
    actual = (
        mathutils.Matrix.Translation(location)
        @ obj.matrix_basis.to_3x3().normalized().to_4x4()
        @ mathutils.Matrix.Diagonal(scale).to_4x4()
    )
    obj.data.transform(actual)
    for c in obj.children:
        c.matrix_local = actual @ c.matrix_local
    obj.matrix_basis = mathutils.Matrix()


def cut(num, center, length, dim, bm):
    start = center[dim] - length / 2
    end = center[dim] + length / 2
    rot = [0, 0, 0]
    rot[dim] = math.radians(90)
    # n + 2 for endpoints, 1:-1 to remove endpoints
    for s in numpy.linspace(start, end, num + 2)[1:-1]:
        loc = center.copy()
        loc[dim] = s
        cut = bmesh.ops.bisect_plane(
            bm,
            geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
            plane_co=loc,
            plane_no=rot,
            dist=1e-7,
        )["geom_cut"]
        bmesh.ops.split_edges(
            bm, edges=[e for e in cut if isinstance(e, bmesh.types.BMEdge)]
        )


def cut_on_axes(obj, obj_center, axes):
    bm = new_bmesh(obj)
    cuts = []
    if "X" in axes:
        cuts.append((1, 0, 0))
    if "Y" in axes:
        cuts.append((0, 1, 0))
    if "Z" in axes:
        cuts.append((0, 0, 1))

    for direction in cuts:
        bmesh.ops.bisect_plane(
            bm,
            geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
            plane_co=obj_center,
            plane_no=direction,
            clear_inner=True,
        )
    # if the object already had vertices down its center plane
    # there will be duplicates
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    set_bmesh(bm, obj)
