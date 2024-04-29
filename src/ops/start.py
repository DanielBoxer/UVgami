# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import pathlib
import platform
import shutil
import subprocess

import bmesh
import bpy
import numpy

from ..handler import handle_error
from ..job import Cleanup, Join, Preserve, Symmetrise
from ..logger import logger
from ..manager import manager
from ..ui.panels import expand
from ..unwrap import Unwrap
from ..utils import (
    apply_transforms,
    calc_center,
    check_collection,
    cut,
    deselect_all,
    get_dir_path,
    get_linux_path,
    get_preferences,
    move_to_collection,
    new_bmesh,
    set_bmesh,
)


class UVGAMI_OT_start(bpy.types.Operator):
    bl_idname = "uvgami.start"
    bl_label = "Unwrap"
    bl_description = "Start UV unwrap process"

    def execute(self, context):
        start_objects = set(bpy.data.objects)
        try:
            logger.new_info()
            prefs = get_preferences()
            props = context.scene.uvgami
            engine_path = pathlib.Path(prefs.engine_path)
            if prefs.autosave:
                if bpy.data.is_saved:
                    bpy.ops.wm.save_mainfile()
                else:
                    bpy.ops.wm.save_as_mainfile("INVOKE_DEFAULT")
                    self.report(
                        {"WARNING"},
                        "Autosave is turned on. Save the file before starting UVgami",
                    )
                    return {"CANCELLED"}

            if str(engine_path) == ".":
                self.report(
                    {"ERROR"},
                    "Engine path is not set. Set the path in the add-on preferences",
                )
                return {"CANCELLED"}

            if not engine_path.is_file():
                self.report(
                    {"ERROR"},
                    "Engine path doesn't exist",
                )
                return {"CANCELLED"}

            if engine_path.stem != "uvgami":
                self.report(
                    {"ERROR"},
                    "Engine path is incorrect",
                )
                return {"CANCELLED"}

            # platform check
            if (
                platform.system() == "Windows"
                and engine_path.suffix == ""
                and not prefs.is_wsl_setup
            ):
                # wsl check
                if shutil.which("wsl") is None:
                    self.report(
                        {"ERROR"},
                        (
                            "WSL is not installed."
                            " Either install WSL or use UVgami for Windows"
                        ),
                    )
                    return {"CANCELLED"}

                r = subprocess.run(["bash", "-c", "test -e ~/uvgami"]).returncode
                if r == 1:
                    # copy uvgami to wsl
                    subprocess.run(
                        ["bash", "-c", f"cp {get_linux_path(engine_path)} ~/"]
                    )
                    prefs.is_wsl_setup = True
                elif r == 0:
                    prefs.is_wsl_setup = True
                else:
                    self.report({"ERROR"}, ("Unknown error configuring engine in WSL"))
                    return {"CANCELLED"}

            elif platform.system() == "Darwin":
                self.report({"ERROR"}, "Mac is not supported")
                return {"CANCELLED"}

            # check if there is an active object selected
            old_active = context.active_object
            input = context.selected_objects
            if not (old_active and old_active in input):
                self.report({"ERROR"}, "No active object selected")
                return {"CANCELLED"}

            old_mode = old_active.mode

            objects = []
            names = {}
            messages = [False, False]
            for obj in input:
                if obj.type != "MESH":
                    messages[0] = True
                    continue
                if len(obj.data.polygons) == 0:
                    messages[1] = True
                    continue

                # make unlinked duplicate of object
                copy_object = obj.copy()
                copy_object.data = obj.data.copy()
                copy_object.animation_data_clear()

                # link to scene
                object_collection = obj.users_collection[0]
                object_collection.objects.link(copy_object)

                # apply all modifiers
                context.view_layer.objects.active = copy_object
                for modifier in copy_object.modifiers:
                    try:
                        bpy.ops.object.modifier_apply(modifier=modifier.name)
                    except:
                        # if the modifier is disabled, don't apply
                        pass

                # cuts
                if props.use_cuts and not props.use_symmetry:
                    bm = new_bmesh(copy_object)

                    if props.cut_type == "EVEN":
                        # make even cuts on axes
                        apply_transforms(copy_object)
                        props = context.scene.uvgami
                        axes = props.cut_axes
                        cuts = props.cuts

                        a_length = len(axes) if len(axes) != 0 else 3
                        d = cuts // a_length
                        r = cuts % a_length

                        # distribute cuts
                        x_num = d if r == 0 else d + 1
                        y_num = d if r != 2 else d + 1
                        z_num = d

                        center = calc_center(obj)
                        if not axes or "X" in axes:
                            cut(x_num, center, copy_object.dimensions.x, 0, bm)
                        if not axes or "Y" in axes:
                            cut(y_num, center, copy_object.dimensions.y, 1, bm)
                        if not axes or "Z" in axes:
                            cut(z_num, center, copy_object.dimensions.z, 2, bm)

                    else:
                        # cut on seams
                        seams = numpy.zeros(len(bm.edges), dtype=numpy.bool)
                        obj.data.edges.foreach_get("use_seam", seams)
                        bm_seams = numpy.array(bm.edges)[seams]
                        bmesh.ops.split_edges(bm, edges=bm_seams)

                    set_bmesh(bm, copy_object)

                # save name, format: input name, unwrap name
                names[copy_object.name] = [obj.name, obj.name]
                objects.append(copy_object)

            report_msg = "Input contains"
            if messages[0]:
                report_msg += " non mesh objects,"
            if messages[1]:
                report_msg += " objects with zero polygons "
            # remove comma or space at end
            report_msg = report_msg[:-1]

            if len(objects) == 0:
                # there are no valid meshes
                self.report({"ERROR"}, report_msg)
                return {"CANCELLED"}
            else:
                input_path = get_dir_path() / "input"
                input_path.mkdir(exist_ok=True)
                output_path = input_path.parent / "output"
                output_path.mkdir(exist_ok=True)
                # io folder clean up
                if not manager.is_active:
                    for file in input_path.iterdir():
                        file.unlink()
                    for file in output_path.iterdir():
                        file.unlink()

                jobs = {}
                separated_objects = []

                # objects can't be in edit mode
                context.view_layer.objects.active = old_active
                if old_mode == "EDIT":
                    bpy.ops.object.mode_set(mode="OBJECT")

                for object_idx, obj in enumerate(objects):
                    deselect_all()
                    obj.select_set(True)

                    symmetrize_job = None
                    # bisect if symmetry on
                    axes = props.sym_axes
                    if props.use_symmetry:
                        apply_transforms(obj)

                        bm = new_bmesh(obj)
                        obj_center = calc_center(obj)
                        symmetrize_job = Symmetrise(
                            1, axes, obj_center, props.sym_merge
                        )
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
                        # if the object already had vertices down it's center plane
                        # there will be duplicates
                        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
                        set_bmesh(bm, obj)

                    # separate objects
                    bpy.ops.mesh.separate(type="LOOSE")
                    s = context.selected_objects
                    if len(s) > 1:
                        # get input name
                        unwrap_name = names[obj.name][0]
                        join_job = Join(len(s))
                        cleanup_job = None

                        # the delete job can come after join because it doesn't depend
                        # on the unwrapped objects
                        if prefs.cleanup == "HIDE" or prefs.cleanup == "DELETE":
                            # the count is > 1 because all the separated objs need to
                            # finish before deleting the original
                            cleanup_job = Cleanup(len(s), prefs.cleanup)
                            manager.input[cleanup_job] = input[object_idx]

                        for obj_idx, o in enumerate(s):
                            # check for 0 polygons again
                            if len(o.data.polygons) == 0:
                                join_job.count -= 1
                                if cleanup_job is not None:
                                    cleanup_job.count -= 1
                                collection = check_collection(
                                    "UVgami Invalid Input", context.scene.collection
                                )
                                move_to_collection(o, collection)
                                o.name = f"{unwrap_name}: No Polygons"
                            else:
                                # add ids to separated objects
                                jobs[o] = {
                                    "join": join_job,
                                    "preserve": None,
                                    "cleanup": cleanup_job,
                                    "symmetrize": symmetrize_job,
                                }
                                separated_objects.append(o)
                                names[o.name] = [
                                    unwrap_name,
                                    f"{unwrap_name}_{obj_idx + 1}",
                                ]
                    else:
                        # object didn't need to be separated
                        jobs[obj] = {
                            "join": None,
                            "preserve": None,
                            "cleanup": None,
                            "symmetrize": symmetrize_job,
                        }
                        if prefs.cleanup == "HIDE" or prefs.cleanup == "DELETE":
                            cleanup_job = Cleanup(1, prefs.cleanup)
                            jobs[obj]["cleanup"] = cleanup_job
                            manager.input[cleanup_job] = input[object_idx]
                        separated_objects.append(obj)

                deselect_all()

                for obj in separated_objects:
                    # get unwrap name
                    unwrap_name = names[obj.name][1]
                    path = input_path / f"{bpy.path.clean_name(unwrap_name)}.obj"
                    # if path to file already exists, find a unique name
                    while path.is_file():
                        path = path.parent / (f"{path.stem}1.obj")

                    # make sure the mesh is triangulated
                    new_edges = []
                    bm = new_bmesh(obj)

                    must_triangulate = False
                    ngon_dict = {}
                    for face_idx, face in enumerate(bm.faces):
                        if len(face.edges) > 3:
                            must_triangulate = True
                            # n-gon vertices are only needed in full mode
                            if props.maintain_mode == "PARTIAL":
                                break

                        if len(face.edges) > 4:
                            # found n-gon
                            for vert in face.verts:
                                if vert.index not in ngon_dict:
                                    ngon_dict[vert.index] = set()
                                ngon_dict[vert.index].add(face_idx)

                    edge_path = None
                    if must_triangulate:
                        if props.untriangulate:
                            jobs[obj]["preserve"] = Preserve(1)
                            old_edges = set(bm.edges)

                        bmesh.ops.triangulate(bm, faces=bm.faces, quad_method="BEAUTY")

                        if props.untriangulate:
                            # write added edges to file
                            edge_path = path.parent / f"{path.stem}_edges"
                            with edge_path.open("w") as f:
                                for bm_e in set(bm.edges).difference(old_edges):
                                    edge = (bm_e.verts[0].index, bm_e.verts[1].index)
                                    if (
                                        # if both vertices are ngon vertices
                                        edge[0] in ngon_dict
                                        and edge[1] in ngon_dict
                                        # and they are from the same ngon
                                        and len(
                                            ngon_dict[edge[0]].intersection(
                                                ngon_dict[edge[1]]
                                            )
                                        )
                                        > 0
                                    ):
                                        # edge is inside ngon, don't dissolve
                                        # because ngons aren't rerouted
                                        continue
                                    new_edges.append(edge)
                                    f.write(f"{edge[0]} {edge[1]}\n")

                        set_bmesh(bm, obj)

                    obj.select_set(True)

                    if bpy.app.version >= (3, 1, 0):
                        # new obj exporter
                        args = {
                            "filepath": str(path),
                            "export_selected_objects": True,
                            "export_normals": False,
                            "export_uv": props.import_uvs,
                            "export_materials": False,
                            "apply_modifiers": False,
                            "forward_axis": "Y",
                            "up_axis": "Z",
                        }

                        if bpy.app.version < (3, 2, 0):
                            # apply_modifiers doesn't exist in 3.1
                            del args["apply_modifiers"]

                        if bpy.app.version < (3, 3, 0):
                            # axis enums were renamed
                            args["forward_axis"] = "Y_FORWARD"
                            args["up_axis"] = "Z_UP"

                        bpy.ops.wm.obj_export("EXEC_DEFAULT", **args)

                    else:
                        # old
                        bpy.ops.export_scene.obj(
                            "EXEC_DEFAULT",
                            filepath=str(path),
                            use_selection=True,
                            use_normals=False,
                            use_uvs=props.import_uvs,
                            use_materials=False,
                            use_blen_objects=False,
                            use_mesh_modifiers=False,
                            axis_forward="Y",
                            axis_up="Z",
                        )

                    obj.select_set(False)

                    guide_path = None
                    if (
                        props.use_guided_mode
                        and "UVgami_seam_restrictions" in obj.vertex_groups
                    ):
                        # get seam guide
                        guide = ""
                        group_idx = obj.vertex_groups["UVgami_seam_restrictions"].index
                        for v in obj.data.vertices:
                            for g in v.groups:
                                if g.group == group_idx:
                                    guide += f"{v.index},{g.weight},"
                                    break
                        # remove last comma
                        guide = guide[:-1]

                        guide_path = path.parent / f"{path.stem}_weights"
                        with guide_path.open("w") as f:
                            f.write(f"{guide}\n")

                    # get materials
                    materials = [
                        slot.material.name
                        for slot in obj.material_slots
                        if slot.material
                    ]

                    # check smooth and auto smooth shading
                    shade_smooth = True if obj.data.polygons[0].use_smooth else False
                    angle = (
                        obj.data.auto_smooth_angle if obj.data.use_auto_smooth else -1
                    )

                    unwrap = Unwrap(
                        name=unwrap_name,
                        input_name=names[obj.name][0],
                        path=path,
                        guide_path=guide_path,
                        edge_path=edge_path,
                        jobs=(
                            jobs[obj]["preserve"],
                            jobs[obj]["join"],
                            jobs[obj]["cleanup"],
                            jobs[obj]["symmetrize"],
                        ),
                        origin=obj.matrix_world.translation,
                        materials=materials,
                        added_edges=new_edges,
                        vertex_count=len(obj.data.vertices),
                        shade_smooth=shade_smooth,
                        auto_smooth=angle,
                        merge_cuts=props.use_cuts and not props.use_symmetry,
                    )
                    manager.active.append(unwrap)

                    bpy.data.objects.remove(obj, do_unlink=True)

                if not manager.is_active:
                    expand.clear()
                    manager.engine_path = engine_path
                    manager.start()
                else:
                    # fix progress bar ratio
                    manager.starting_count += len(separated_objects)
                context.view_layer.objects.active = old_active
                bpy.ops.object.mode_set(mode=old_mode)

                if report_msg == "Input contain":
                    self.report({"INFO"}, "UV unwrap in progress")
                else:
                    self.report({"WARNING"}, f"UV unwrap in progress. {report_msg}")

        except Exception as e:
            handle_error(e, "START", objects=start_objects)

        return {"FINISHED"}
