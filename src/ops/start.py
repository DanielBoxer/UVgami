import contextlib
import functools
import time
from collections import deque, namedtuple

import bmesh
import bpy
import numpy

from ..engines import active_engine
from ..handler import handle_error
from ..job import (
    HideInput,
    Join,
    Preserve,
    ProxyCopyUVs,
    ProxyUVs,
    Result,
    # Symmetrise,
    TransferUVs,
)
from ..logger import logger
from ..manager import Preparing, manager
from ..proxy import make_proxy, needs_proxy, triangle_count
from ..seams import (
    face_edges,
    island_layout,
    islands_overlap,
    uv_island_groups,
)
from ..similar import find_twins, write_twin_output

# from ..similar import mirror_matches
from ..ui.panels import describe_settings, unwrap_settings
from ..unwrap import Unwrap

# from ..utils.geometry import apply_transforms, calc_center
from ..utils.io import export_obj
from ..utils.mesh import (
    check_exists,
    deselect_all,
    get_shading_modifiers,
    is_shading_modifier,
    mark_not_unwrapped,
    new_bmesh,
    set_bmesh,
    triangulate,
)
from ..utils.paths import (
    clear_io_dir,
    engine_file_stem,
    get_io_dir_paths,
    get_preferences,
)
from ..utils.task import BackgroundTask
from ..utils.ui import tag_redraw
from .guides import SEAM_RESTRICTIONS_GROUP

Preseeding = namedtuple("Preseeding", ["task", "apply", "obj", "symmetrize_job"])

# seconds of work per tick before yielding to the event loop
TICK_BUDGET = 0.033


# two faces walking one shared edge in the same direction
def has_inconsistent_winding(mesh):
    corners = numpy.empty(len(mesh.loops), dtype=numpy.int64)
    mesh.loops.foreach_get("vertex_index", corners)
    totals = numpy.empty(len(mesh.polygons), dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_total", totals)
    ends = numpy.cumsum(totals)
    next_corner = numpy.arange(1, len(corners) + 1)
    next_corner[ends - 1] = ends - totals
    directed = corners * (len(mesh.vertices) + 1) + corners[next_corner]
    return len(numpy.unique(directed)) < len(directed)


# mixed winding is optcuts exit 115
def fix_inconsistent_winding(obj):
    if not has_inconsistent_winding(obj.data):
        return
    bm = new_bmesh(obj)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    set_bmesh(bm, obj)


# the engine re-cuts a mirrored or stacked chart, losing its seams
def normalize_uvs(mesh):
    layer = mesh.uv_layers.active
    face_count = len(mesh.polygons)
    totals = numpy.empty(face_count, dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_total", totals)
    starts = numpy.empty(face_count, dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    loop_verts = numpy.empty(len(mesh.loops), dtype=numpy.int64)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    coords = numpy.empty(len(mesh.loops) * 2)
    layer.uv.foreach_get("vector", coords)
    coords = coords.reshape(-1, 2)

    # the grouping helpers walk plain lists
    vert_list = loop_verts.tolist()
    uv_list = coords.tolist()
    faces = []
    uvs = []
    for start, total in zip(starts.tolist(), totals.tolist()):
        faces.append(tuple(vert_list[start : start + total]))
        uvs.append(uv_list[start : start + total])

    groups = uv_island_groups(faces, uvs, face_edges(faces))

    # shoelace per face, each loop paired with the next one around its own face
    following = numpy.arange(1, len(coords) + 1)
    following[starts + totals - 1] = starts
    cross = coords[:, 0] * coords[following, 1] - coords[following, 0] * coords[:, 1]
    face_areas = 0.5 * numpy.add.reduceat(cross, starts) if face_count else []

    island_of_face = numpy.empty(face_count, dtype=numpy.int64)
    for index, group in enumerate(groups):
        island_of_face[group] = index
    loop_island = numpy.repeat(island_of_face, totals)
    order = numpy.argsort(loop_island, kind="stable")
    counts = numpy.bincount(loop_island, minlength=len(groups))
    island_loops = numpy.split(order, numpy.cumsum(counts)[:-1])

    boxes = []
    areas = []
    for group, loops in zip(groups, island_loops):
        points = coords[loops]
        boxes.append(
            (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )
        )
        areas.append(float(face_areas[group].sum()))

    if all(a >= 0 for a in areas) and not islands_overlap(boxes):
        return

    # nan marks an island that keeps its handedness
    flips = numpy.full(face_count, numpy.nan)
    offsets = numpy.empty((face_count, 2))
    for group, (flip, du, dv) in zip(groups, island_layout(boxes, areas)):
        if flip is not None:
            flips[group] = flip
        offsets[group] = (du, dv)

    loop_flip = numpy.repeat(flips, totals)
    mirrored = ~numpy.isnan(loop_flip)
    coords[mirrored, 0] = loop_flip[mirrored] - coords[mirrored, 0]
    coords += numpy.repeat(offsets, totals, axis=0)
    layer.uv.foreach_set("vector", coords.ravel())
    mesh.update()


# writes across timer ticks so the UI stays responsive
class InputExporter:
    def __init__(self, engine, engine_ctx, pieces, start_objects, temp_collection):
        self.engine = engine
        self.engine_ctx = engine_ctx
        self.pieces = pieces
        self.remaining = deque(pieces)
        self.start_objects = start_objects
        self.temp_collection = temp_collection

    def tick(self):
        # exports need object mode, the user may have entered edit mode between ticks
        if bpy.context.mode != "OBJECT":
            return 0.2

        # the session we joined was cancelled mid-export (Cancel All)
        if self.pieces and not manager.is_active:
            # reported first so a failure below can't leave the count stuck
            manager.finished_adding()
            for obj, _ in self.remaining:
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(self.temp_collection)
            return None

        try:
            # frozen, so a slider moved during the export can't reach the engine
            props = manager.props
            start = time.monotonic()
            while self.remaining:
                self._export_object(*self.remaining.popleft(), props)
                if time.monotonic() - start >= TICK_BUDGET:
                    # yield to the event loop
                    return 0.0

            self._finish()
            return None

        except Exception as e:
            handle_error(e, "START", objects=self.start_objects)
            # handle_error leaves these alone during a live session
            for piece in list(self.temp_collection.objects):
                bpy.data.objects.remove(piece, do_unlink=True)
            bpy.data.collections.remove(self.temp_collection)
            manager.finished_adding()
            # settle the pieces that never got exported or the session can't finish
            for _, unwrap in self.pieces:
                if unwrap.result is None and not unwrap.is_exported:
                    manager.record_result(unwrap, Result.INVALID)
            return None

    def _export_object(self, obj, unwrap, props):
        if unwrap.result is not None:
            # cancelled while waiting to export
            bpy.data.objects.remove(obj, do_unlink=True)
            return
        if unwrap.copy_of is not None:
            self._export_twin(obj, unwrap, props)
            return

        path = unwrap.path
        fix_inconsistent_winding(obj)
        edge_path, new_edges = self._triangulate_mesh(obj, unwrap, path, props)

        matrix = unwrap.matrix

        # seams and uvs were built before separation, see create_jobs
        if unwrap.has_uvs:
            normalize_uvs(obj.data)
        # a transfer writes onto the original, where the winding doesn't matter
        vt_verts = export_obj(
            obj, path, unwrap.has_uvs, flip_mirrored=unwrap.keeps_output, matrix=matrix
        )

        guide_path = self._create_guide_file(obj, path, props, vt_verts)

        materials, material_indices, vertex_groups, face_smooth = (
            self._get_mesh_metadata(obj)
        )

        unwrap.set_export_data(
            guide_path=guide_path,
            edge_path=edge_path,
            origin=matrix.translation,
            materials=materials,
            added_edges=new_edges,
            vertex_count=len(obj.data.vertices),
            material_indices=material_indices,
            vertex_groups=vertex_groups,
            face_smooth=face_smooth,
            shading_modifiers=get_shading_modifiers(obj),
        )

        bpy.data.objects.remove(obj, do_unlink=True)

    # a reordered twin's own indices point at the wrong vertices
    def _export_twin(self, obj, unwrap, props):
        representative = unwrap.copy_of
        if unwrap.copy_reordered:
            unwrap.set_export_data(
                edge_path=representative.edge_path,
                origin=representative.origin,
                materials=representative.materials,
                added_edges=representative.added_edges,
                vertex_count=representative.vertex_count,
                material_indices=representative.material_indices,
                vertex_groups=representative.vertex_groups,
                face_smooth=representative.face_smooth,
                shading_modifiers=representative.shading_modifiers,
            )
        else:
            edge_path, new_edges = self._triangulate_mesh(
                obj, unwrap, unwrap.path, props
            )
            materials, material_indices, vertex_groups, face_smooth = (
                self._get_mesh_metadata(obj)
            )
            unwrap.set_export_data(
                edge_path=edge_path,
                # pieces of one object share a transform
                origin=representative.origin,
                materials=materials,
                added_edges=new_edges,
                vertex_count=len(obj.data.vertices),
                material_indices=material_indices,
                vertex_groups=vertex_groups,
                face_smooth=face_smooth,
                shading_modifiers=get_shading_modifiers(obj),
            )

        bpy.data.objects.remove(obj, do_unlink=True)

        # the representative finished before this twin was ready
        if representative.result is Result.FINISHED:
            write_twin_output(
                representative.output_path, unwrap.output_path, unwrap.copy_matrix
            )
            manager.record_result(unwrap, Result.FINISHED)

    def _finish(self):
        bpy.data.collections.remove(self.temp_collection)
        manager.finished_adding()

    def _triangulate_mesh(self, obj, unwrap, path, props):
        new_edges = []
        bm = new_bmesh(obj)

        must_triangulate = False
        ngon_dict = {}
        for face_idx, face in enumerate(bm.faces):
            if len(face.edges) > 3:
                must_triangulate = True
                # if props.maintain_mode == "PARTIAL":
                #     break

            if len(face.edges) > 4:
                for vert in face.verts:
                    if vert.index not in ngon_dict:
                        ngon_dict[vert.index] = set()
                    ngon_dict[vert.index].add(face_idx)

        # the panel hides the setting where it doesn't apply but the value persists
        untriangulate = props.preserve_mesh

        edge_path = None
        if must_triangulate:
            if untriangulate:
                unwrap.preserve_job = Preserve()
                old_edges = set(bm.edges)

            triangulate(bm, obj.data)

            if untriangulate:
                edge_path = path.parent / f"{path.stem}_edges"
                with edge_path.open("w") as f:
                    for bm_e in set(bm.edges).difference(old_edges):
                        edge = (bm_e.verts[0].index, bm_e.verts[1].index)
                        if (
                            edge[0] in ngon_dict
                            and edge[1] in ngon_dict
                            and len(ngon_dict[edge[0]].intersection(ngon_dict[edge[1]]))
                            > 0
                        ):
                            # both ends sit on the same ngon, the edge is inside it
                            continue
                        new_edges.append(edge)
                        f.write(f"{edge[0]} {edge[1]}\n")

            set_bmesh(bm, obj)
        else:
            bm.free()

        return edge_path, new_edges

    # _weights repels seams, _importance protects faces from stretching
    def _create_guide_file(self, obj, path, props, vt_verts):
        weights = {}
        if (
            self.engine.supports_guided
            and (props.avoid_seams or props.reduce_stretching)
            and SEAM_RESTRICTIONS_GROUP in obj.vertex_groups
        ):
            group_idx = obj.vertex_groups[SEAM_RESTRICTIONS_GROUP].index
            for v in obj.data.vertices:
                for g in v.groups:
                    if g.group == group_idx:
                        weights[v.index] = g.weight
                        break

        if not weights:
            return None

        if vt_verts is not None:
            # optcuts rebuilds a UV-carrying obj with one vertex per vt
            weights = {
                vt: weights[v] for vt, v in enumerate(vt_verts.tolist()) if v in weights
            }

        guide = ",".join(f"{index},{weight}" for index, weight in weights.items())
        guide_path = None
        if props.avoid_seams:
            guide_path = path.parent / f"{path.stem}_weights"
            with guide_path.open("w") as f:
                f.write(f"{guide}\n")
        if props.reduce_stretching:
            importance_path = path.parent / f"{path.stem}_importance"
            with importance_path.open("w") as f:
                f.write(f"{guide}\n")

        return guide_path

    def _get_mesh_metadata(self, obj):
        # keep empty slots as None so the per-face indices below stay valid
        materials = [
            slot.material.name if slot.material else None for slot in obj.material_slots
        ]

        material_indices = numpy.empty(len(obj.data.polygons), dtype=numpy.int32)
        obj.data.polygons.foreach_get("material_index", material_indices)

        face_smooth = numpy.empty(len(obj.data.polygons), dtype=bool)
        obj.data.polygons.foreach_get("use_smooth", face_smooth)

        vertex_groups = {}
        for group in obj.vertex_groups:
            weights = {}
            for v in obj.data.vertices:
                for g in v.groups:
                    if g.group == group.index:
                        weights[v.index] = g.weight
                        break
            vertex_groups[group.name] = weights

        return materials, material_indices.tolist(), vertex_groups, face_smooth.tolist()


# proxied says the mesh was decimated, one under Proxy Faces never is
def input_job(props, proxied):
    if proxied:
        return ProxyUVs() if props.transfer_uvs else ProxyCopyUVs()
    if props.transfer_uvs:
        return TransferUVs()
    return None


# the preseed runs on a worker thread and the rest across timer ticks
class SessionBuilder:
    def __init__(
        self,
        engine,
        engine_ctx,
        input_path,
        names,
        input_for,
        objects,
        proxied_objects,
        start_objects,
        temp_collection,
    ):
        self.engine = engine
        self.engine_ctx = engine_ctx
        self.input_path = input_path
        self.names = names
        self.input_for = input_for
        self.proxied_objects = proxied_objects
        self.remaining = deque(objects)
        self.start_objects = start_objects
        self.temp_collection = temp_collection
        self.pieces = []
        self.piece_unwrap = {}
        self.pending = None
        # cancelled mid preseed, dropped when the thread unwinds
        self.cancelled = set()
        # queue ui placeholders until each object's pieces exist
        self.preparing = {
            obj: Preparing(
                names[obj.name][0], functools.partial(self.cancel_object, obj)
            )
            for obj in self.remaining
        }
        manager.preparing.extend(self.preparing.values())

    def tick(self):
        # separation and uv writes need object mode
        if bpy.context.mode != "OBJECT":
            return 0.2
        try:
            return self._advance()
        except Exception as e:
            handle_error(e, "START", objects=self.start_objects)
            # an undo past the session start kills the collection datablock
            if check_exists(self.temp_collection):
                # handle_error leaves these alone during a live session
                for piece in list(self.temp_collection.objects):
                    bpy.data.objects.remove(piece, do_unlink=True)
                bpy.data.collections.remove(self.temp_collection)
            manager.finished_adding()
            for entry in self.preparing.values():
                manager.drop_preparing(entry)
            self.preparing.clear()
            # settle the pieces already added, none of them exported yet
            for unwrap in self.piece_unwrap.values():
                if unwrap.result is None:
                    manager.record_result(unwrap, Result.INVALID)
            return None

    def _drop_preparing(self, obj):
        entry = self.preparing.pop(obj, None)
        if entry is not None:
            manager.drop_preparing(entry)

    # a running preseed only stops at its next check, so it is left to unwind
    def cancel_object(self, obj):
        if self.pending is not None and self.pending.obj is obj:
            self.pending.task.cancel()
            self.cancelled.add(obj)
        else:
            self.remaining = deque(
                entry for entry in self.remaining if entry is not obj
            )
            self._drop_object(obj)

    def _drop_object(self, obj):
        self._drop_preparing(obj)
        if check_exists(obj):
            bpy.data.objects.remove(obj, do_unlink=True)

    def _advance(self):
        if self.pending is not None:
            task, apply, obj, symmetrize_job = self.pending
            if not task.done():
                return 0.1
            self.pending = None
            if obj in self.cancelled:
                # it may have finished before it saw the flag
                self.cancelled.discard(obj)
                self._drop_object(obj)
                return 0.0
            result = task.result()
            if not check_exists(obj):
                raise RuntimeError("Undo removed the working copy mid unwrap")
            props = manager.props
            preseeded = apply(result)
            if symmetrize_job is not None:
                if preseeded:
                    # the seams are mirrored, so the mesh ships whole
                    symmetrize_job.kept_whole = True
                    symmetrize_job.prepare_half(obj)
                else:
                    logger.add_data(
                        "errors",
                        f"{self.names[obj.name][0]}: no seams to mirror,"
                        " the mesh was cut and mirrored",
                    )
                    symmetrize_job.cut(obj)
            has_uvs = preseeded or self.engine.uses_import_uvs(props)
            self._separate(obj, has_uvs, symmetrize_job, preseeded=preseeded)
            return 0.0
        if not self.remaining:
            return self._finish()

        obj = self.remaining.popleft()
        props = manager.props
        symmetrize_job = None
        mirrors = None
        # if props.use_symmetry:
        #     apply_transforms(obj)
        #     center = calc_center(obj)
        #     symmetrize_job = Symmetrise(props.sym_axes, center, props.sym_merge)
        #     mirrors = mirror_matches(obj.data, center, sorted(props.sym_axes))
        #     symmetrize_job.mirrors = mirrors

        # the seams package reads region widths off the full model
        work = self.engine.preseed_work(obj, props, mirrors)
        if work is None:
            has_uvs = self.engine.prepare_uvs(obj, props)
            if symmetrize_job is not None:
                symmetrize_job.cut(obj)
            self._separate(obj, has_uvs, symmetrize_job)
            return 0.0
        compute, apply = work
        self.pending = Preseeding(BackgroundTask(compute), apply, obj, symmetrize_job)
        return 0.1

    def _input_jobs(self, props, obj, proxied):
        transfer_uvs_job = input_job(props, proxied)
        hide_job = None
        if transfer_uvs_job is not None:
            manager.input[transfer_uvs_job] = self.input_for[obj]
        else:
            # the hide job doesn't depend on the unwrapped objects
            hide_job = HideInput()
            manager.input[hide_job] = self.input_for[obj]
        return hide_job, transfer_uvs_job

    def _add_piece(
        self, obj, input_name, piece_name, jobs, has_uvs, props, preseeded, keeps_output
    ):
        # created before the input file exists, so the queue ui is complete upfront
        path = self.input_path / f"{engine_file_stem(piece_name)}.obj"
        # names can repeat across pieces, and the output file is keyed by stem
        claimed = {u.path for u in manager.active}
        claimed.update(u.path for u, _ in manager.results)
        while path.is_file() or path in claimed:
            path = path.parent / f"{path.stem}1.obj"

        uses_uvs = (
            self.engine.piece_uses_uvs(obj, props, has_uvs)
            and obj.data.uv_layers.active is not None
        )
        unwrap = Unwrap(
            name=piece_name,
            input_name=input_name,
            path=path,
            jobs=jobs,
            # maintain_mode=props.maintain_mode,
            preseeded=preseeded and uses_uvs,
        )
        unwrap.has_uvs = uses_uvs
        unwrap.keeps_output = keeps_output
        self.piece_unwrap[obj] = unwrap
        manager.add(unwrap)

    def _separate(self, obj, has_uvs, symmetrize_job, preseeded=False):
        props = manager.props
        proxied = obj in self.proxied_objects
        # relink for the ops selection, all within one tick so nothing shows
        bpy.context.scene.collection.objects.link(obj)
        self.temp_collection.objects.unlink(obj)
        bpy.context.view_layer.update()
        # the pieces share this transform, their own matrix is never updated
        matrix = obj.matrix_world.copy()
        deselect_all()
        obj.select_set(True)

        bpy.ops.mesh.separate(type="LOOSE")
        s = bpy.context.selected_objects
        added = []
        if len(s) > 1:
            unwrap_name = self.names[obj.name][0]
            valid = []
            for o in s:
                if len(o.data.polygons) == 0:
                    mark_not_unwrapped(o, "No Polygons", unwrap_name)
                    manager.moved_to_invalid = True
                else:
                    valid.append(o)

            join_job = Join(len(valid))
            hide_job, transfer_uvs_job = self._input_jobs(props, obj, proxied)
            for obj_idx, o in enumerate(valid):
                jobs = (None, join_job, hide_job, symmetrize_job, transfer_uvs_job)
                piece_name = f"{unwrap_name}_{obj_idx + 1}"
                self._add_piece(
                    o,
                    unwrap_name,
                    piece_name,
                    jobs,
                    has_uvs,
                    props,
                    preseeded,
                    transfer_uvs_job is None,
                )
                added.append(o)
            if props.stack_similar:
                # the new pieces' matrix_world needs an evaluation first
                bpy.context.view_layer.update()
                # a representative always precedes its twins in valid order
                for twin_obj, (rep_obj, twin_matrix, exact) in find_twins(
                    valid
                ).items():
                    twin = self.piece_unwrap[twin_obj]
                    representative = self.piece_unwrap[rep_obj]
                    twin.copy_of = representative
                    twin.copy_matrix = twin_matrix
                    twin.copy_reordered = not exact
                    representative.twins.append(twin)
        else:
            unwrap_name = self.names[obj.name][0]
            hide_job, transfer_uvs_job = self._input_jobs(props, obj, proxied)
            jobs = (None, None, hide_job, symmetrize_job, transfer_uvs_job)
            self._add_piece(
                obj,
                unwrap_name,
                unwrap_name,
                jobs,
                has_uvs,
                props,
                preseeded,
                transfer_uvs_job is None,
            )
            added.append(obj)

        self._drop_preparing(obj)

        deselect_all()
        for piece in added:
            unwrap = self.piece_unwrap[piece]
            unwrap.matrix = matrix
            self.pieces.append((piece, unwrap))
            for coll in piece.users_collection:
                coll.objects.unlink(piece)
            self.temp_collection.objects.link(piece)

    def _finish(self):
        exporter = InputExporter(
            engine=self.engine,
            engine_ctx=self.engine_ctx,
            pieces=self.pieces,
            start_objects=self.start_objects,
            temp_collection=self.temp_collection,
        )
        # the exporter takes over the count entry this builder was carrying
        bpy.app.timers.register(exporter.tick)
        return None


class UVGAMI_OT_start(bpy.types.Operator):
    bl_idname = "uvgami.start"
    bl_label = "Unwrap"
    bl_description = "Start UV unwrap process"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_variables()

    def reset_variables(self):
        self.engine = None
        self.engine_ctx = None
        self.input_path = None

        self.input_objs = None

        self.objects = None
        self.names = None
        self.input_for = None
        self.proxied_objects = None
        self.reports = []

        self.temp_collection = None

    def execute(self, context):
        start_objects = set(bpy.data.objects)
        builder_registered = False

        try:
            info = logger.new_info()
            self.engine = active_engine(context.scene.uvgami.engine)
            if self.engine is None:
                self.report({"ERROR"}, "No engine installed")
                logger.discard_info()
                return {"CANCELLED"}
            info.engine = self.engine.describe()
            info.settings = describe_settings(
                context.scene.uvgami, unwrap_settings(context.scene.uvgami)
            )

            # a mesh added to a running session would take the first one's settings
            if manager.is_active:
                self.report({"ERROR"}, "Finish or cancel the current unwrap first")
                logger.discard_info()
                return {"CANCELLED"}

            if self.check_for_errors() is not None:
                logger.discard_info()
                return {"CANCELLED"}
            if self._prepare_unwrap_session(context) is not None:
                logger.discard_info()
                return {"CANCELLED"}

            builder = SessionBuilder(
                engine=self.engine,
                engine_ctx=self.engine_ctx,
                input_path=self.input_path,
                names=self.names,
                input_for=self.input_for,
                objects=self.objects,
                proxied_objects=self.proxied_objects,
                start_objects=start_objects,
                temp_collection=self.temp_collection,
            )
            bpy.app.timers.register(builder.tick)
            builder_registered = True

            # before the first piece exists, the bar is up while the builder works
            manager.engine = self.engine
            manager.engine_ctx = self.engine_ctx
            manager.pieces_still_arriving += 1
            manager.start()
            tag_redraw()

            for level, text in self.reports:
                self.report({level}, text)

        except Exception as e:
            handle_error(e, "START", objects=start_objects)
            # once the builder is registered tick() removes the collection
            if not builder_registered and self.temp_collection is not None:
                bpy.data.collections.remove(self.temp_collection)

        # only valid while the operator is running
        self.reset_variables()
        return {"FINISHED"}

    def _prepare_unwrap_session(self, context):
        # leaving edit mode flushes the mesh the copies below take
        active = context.active_object
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        self.input_objs = context.selected_objects
        self.objects, self.names, self.reports, self.proxied_objects, self.input_for = (
            self.prepare_meshes(context)
        )
        if len(self.objects) == 0:
            reason = self.reports[0][1] if self.reports else "No object selected"
            self.report({"ERROR"}, reason)
            return {"CANCELLED"}

        self.input_path, _ = self.prepare_io_folders()
        deselect_all()

        # an unlinked collection doesn't flash in the viewport between ticks
        self.temp_collection = bpy.data.collections.new("UVgami Temp")
        for obj in self.objects:
            for coll in obj.users_collection:
                coll.objects.unlink(obj)
            self.temp_collection.objects.link(obj)
        return None

    def check_for_errors(self):
        prefs = get_preferences()

        if self._run_autosave_check(prefs) is not None:
            return {"CANCELLED"}

        engine_ctx, error = self.engine.validate(prefs)
        if error is not None:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        self.engine_ctx = engine_ctx

        return None

    def _run_autosave_check(self, prefs):
        if not prefs.autosave:
            return None

        if bpy.data.is_saved:
            bpy.ops.wm.save_mainfile()
            return None

        bpy.ops.wm.save_as_mainfile("INVOKE_DEFAULT")
        self.report(
            {"WARNING"},
            "Autosave is turned on. Save the file before starting UVgami",
        )
        return {"CANCELLED"}

    def prepare_meshes(self, context):
        props = context.scene.uvgami

        objects = []
        names = {}
        input_for = {}
        input_sizes = []
        skipped = set()
        applied_modifiers = False
        proxied_objects = set()
        warn = get_preferences().show_warnings
        depsgraph = context.evaluated_depsgraph_get()
        for obj in self.input_objs:
            if obj.type != "MESH":
                skipped.add("non mesh objects")
                continue
            if len(obj.data.polygons) == 0:
                skipped.add("objects with zero polygons")
                continue

            # unlinked duplicate, so the original is never touched
            copy_object = obj.copy()
            copy_object.data = obj.data.copy()
            copy_object.animation_data_clear()

            obj.users_collection[0].objects.link(copy_object)

            # the count is the modifier bake, what a plain run would unwrap
            evaluated = obj.evaluated_get(depsgraph)
            triangles = triangle_count(evaluated)
            input_sizes.append((obj.name, triangles))
            proxied = self.engine.uses_proxy(props) and needs_proxy(
                evaluated, props.proxy_faces
            )
            # the engine has to see the input mesh itself, not a modifier bake of it
            if input_job(props, proxied) is not None:
                copy_object.modifiers.clear()
            elif self._apply_modifiers(context, copy_object):
                applied_modifiers = True

            if proxied:
                make_proxy(copy_object, props.proxy_faces)
                proxied_objects.add(copy_object)

            # format: input name, unwrap name
            names[copy_object.name] = [obj.name, obj.name]
            input_for[copy_object] = obj
            objects.append(copy_object)

        logger.get_latest().objects = input_sizes

        reports = []
        if skipped:
            reports.append(("WARNING", f"Input contains {', '.join(sorted(skipped))}"))
        if warn and applied_modifiers:
            reports.append(
                ("WARNING", "Modifiers will be applied on the unwrapped copy")
            )

        return objects, names, reports, proxied_objects, input_for

    def _apply_modifiers(self, context, obj):
        context.view_layer.objects.active = obj
        applied = False
        for modifier in obj.modifiers:
            if is_shading_modifier(modifier):
                continue

            # a disabled modifier can't be applied
            with contextlib.suppress(RuntimeError):
                bpy.ops.object.modifier_apply(modifier=modifier.name)
                applied = True
        return applied

    def prepare_io_folders(self):
        input_path, output_path = get_io_dir_paths()
        if not manager.is_active:
            clear_io_dir(input_path)
            clear_io_dir(output_path)

        return input_path, output_path
