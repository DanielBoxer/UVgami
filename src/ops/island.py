import collections

import bmesh
import bpy
import numpy

from ..engines import get_engine
from ..hard_surface import apply_seams
from ..job import AreaUVs, IslandUVs, ProxyIslandUVs
from ..logger import logger
from ..manager import manager
from ..proxy import make_proxy, triangle_count
from ..seams import (
    FLIP_NOISE,
    face_edges,
    flatten_distortion,
    island_groups,
    pair,
    rectify_targets,
    signed_area,
    split_moves,
    uv_island_groups,
    uv_seams,
)
from ..ui.panels import describe_settings, fix_settings
from ..unwrap import Unwrap
from ..utils.io import export_obj
from ..utils.mesh import (
    face_uvs,
    face_vertices,
    loop_starts,
    loop_uvs,
    new_bmesh,
    set_bmesh,
    set_loop_uvs,
    triangulate,
    vertex_positions,
)
from ..utils.paths import (
    clear_io_dir,
    engine_file_stem,
    get_io_dir_paths,
    get_preferences,
)

# blender's default is 10, 50 flattens the stubborn folds
REPAIR_ITERATIONS = 50

# a rectified island reverts when its solved area fell under this fraction
RECTIFY_COLLAPSE = 0.5
# or when its scale-free Dirichlet rose more than this over the engine's map
RECTIFY_DISTORTION = 1.0


# bpy.ops pins uvs, which the engine's flatten mode has no channel for
def unwrap(obj, only, iterations, method="MINIMUM_STRETCH"):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for fi in only:
        obj.data.polygons[fi].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.unwrap(method=method, margin=0.001, iterations=iterations)
    bpy.ops.object.mode_set(mode="OBJECT")


# with uv sync off the editor leaves stale uv flags on faces it isn't drawing
def selected_faces(mesh):
    selected = {p.index for p in mesh.polygons if p.select}
    if bpy.context.scene.tool_settings.use_uv_select_sync:
        return selected
    uv_select = mesh.attributes.get(".uv_select_face")
    if uv_select is None:
        return set()
    return {fi for fi in selected if uv_select.data[fi].value}


# a fix carries none of the material or vertex group state a full unwrap does
def queue_fix(obj, job, name, path, vertex_count, props):
    manager.input[job] = obj
    unwrap = Unwrap(
        name=name,
        input_name=name,
        path=path,
        jobs=(None, None, None, None, job),
        # maintain_mode=props.maintain_mode,
    )
    unwrap.set_export_data(
        origin=obj.matrix_world.translation, vertex_count=vertex_count
    )
    manager.add(unwrap)


def target_islands(obj):
    mesh = obj.data
    if not mesh.uv_layers.active:
        return None, "Mesh has no uv map"
    selected = selected_faces(mesh)
    if not selected:
        return None, "Select the faces of the islands to fix"

    faces = face_vertices(mesh)
    uvs = face_uvs(mesh)
    targets = []
    for group in uv_island_groups(faces, uvs, face_edges(faces)):
        if selected.isdisjoint(group):
            continue
        points = [uv for fi in group for uv in uvs[fi]]
        xs = [u for u, _ in points]
        ys = [v for _, v in points]
        area = sum(abs(signed_area(uvs[fi])) for fi in group)
        targets.append((group, (min(xs), min(ys), max(xs), max(ys)), area))
    return targets, None


def queue_island(obj, group, bbox, area, k, input_path, props):
    mesh = obj.data
    used = sorted({v for fi in group for v in mesh.polygons[fi].vertices})
    local = {v: i for i, v in enumerate(used)}
    island_mesh = bpy.data.meshes.new("uvgami_island")
    island_mesh.from_pydata(
        [mesh.vertices[v].co.copy() for v in used],
        [],
        [[local[v] for v in mesh.polygons[fi].vertices] for fi in group],
    )
    temp = bpy.data.objects.new("uvgami_island", island_mesh)
    bpy.context.scene.collection.objects.link(temp)
    temp.matrix_world = obj.matrix_world.copy()

    bm = new_bmesh(temp)
    if any(len(f.verts) > 3 for f in bm.faces):
        triangulate(bm, temp.data)
        set_bmesh(bm, temp)
    else:
        bm.free()

    proxied = props.use_proxy and make_proxy(temp, props.proxy_faces)

    name = f"{obj.name}_island_{k}"
    path = input_path / f"{engine_file_stem(name)}.obj"
    while path.is_file():
        path = path.parent / f"{path.stem}1.obj"
    export_obj(temp, path, False)
    vertex_count = len(temp.data.vertices)
    # make_proxy swaps in a mesh of its own
    mesh = temp.data
    bpy.data.objects.remove(temp, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    job = (
        ProxyIslandUVs(list(group), bbox, area)
        if proxied
        else IslandUVs(list(group), bbox, area)
    )
    queue_fix(obj, job, name, path, vertex_count, props)


def edge_splits_uv(edge, uvl):
    uv_of = {}
    for loop in edge.link_loops:
        for corner in (loop, loop.link_loop_next):
            uv = (round(corner[uvl].uv[0], 6), round(corner[uvl].uv[1], 6))
            if uv_of.setdefault(corner.vert.index, uv) != uv:
                return True
    return False


def folded_faces(bm, uvl):
    total = 0.0
    fans = []
    for face in bm.faces:
        pts = [loop[uvl].uv for loop in face.loops]
        areas = [
            signed_area([pts[0], pts[i], pts[i + 1]]) for i in range(1, len(pts) - 1)
        ]
        total += sum(areas)
        fans.append((face, areas))
    orientation = 1 if total >= 0 else -1
    return [face for face, areas in fans if any(a * orientation < 0 for a in areas)]


# the uv discontinuities are marked as seams so the island's own cuts survive
def repair_flipped_island(obj, temp):
    bm = new_bmesh(temp)
    uvl = bm.loops.layers.uv.active

    flipped = folded_faces(bm, uvl)
    if not flipped:
        bm.free()
        return

    for edge in bm.edges:
        edge.seam = edge_splits_uv(edge, uvl)

    free = {v for face in flipped for v in face.verts}
    for _ in range(2):
        grown = [f for f in bm.faces if not free.isdisjoint(f.verts)]
        free |= {v for face in grown for v in face.verts}
    for face in bm.faces:
        for loop in face.loops:
            loop[uvl].pin_uv = loop.vert not in free
    set_bmesh(bm, temp)

    # keep the input mesh out of the edit session, like repair_flipped
    obj.select_set(False)
    unwrap(temp, range(len(temp.data.polygons)), REPAIR_ITERATIONS)
    obj.select_set(True)


MeshReads = collections.namedtuple("MeshReads", "coords faces edges uvs seams groups")


def read_mesh(mesh):
    faces = face_vertices(mesh)
    uvs = face_uvs(mesh)
    edges = face_edges(faces)
    seams = uv_seams(faces, uvs, edges)
    groups = island_groups(faces, seams, edges)
    return MeshReads(vertex_positions(mesh), faces, edges, uvs, seams, groups)


# returns the reads as they are after the moves
def finish_preseed(obj, reads, ranges=None):
    mesh = obj.data
    starts = loop_starts(mesh)
    moves, seams, groups = split_moves(
        reads.coords,
        reads.faces,
        reads.uvs,
        starts.tolist(),
        ranges,
        reads.edges,
        reads.seams,
        reads.groups,
    )
    if not moves:
        return reads
    coords = loop_uvs(mesh)
    for loop_index, u, v in moves:
        coords[loop_index] = (u, v)
    set_loop_uvs(mesh, coords)

    # blender stores uvs as float32
    loops = numpy.fromiter((m[0] for m in moves), numpy.int64, len(moves))
    stored = coords[loops].astype(numpy.float32).astype(numpy.float64).tolist()
    face_of = numpy.searchsorted(starts, loops, "right") - 1
    corners = (loops - starts[face_of]).tolist()
    uvs = list(reads.uvs)
    copied = set()
    for fi, corner, (u, v) in zip(face_of.tolist(), corners, stored):
        if fi not in copied:
            uvs[fi] = list(uvs[fi])
            copied.add(fi)
        uvs[fi][corner] = (round(u, 6), round(v, 6))
    return reads._replace(uvs=uvs, seams=seams, groups=groups)


def _fan_triangle_has_area(coords, face, i):
    x0, y0, z0 = coords[face[0]]
    ax, ay, az = coords[face[i]]
    bx, by, bz = coords[face[i + 1]]
    ux, uy, uz = ax - x0, ay - y0, az - z0
    vx, vy, vz = bx - x0, by - y0, bz - z0
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    return cx * cx + cy * cy + cz * cz > 0.0


# blender's unwrap reinitializes from scratch, a pinned solve can't unbend a curl
def rectify_islands(obj, reads):
    mesh = obj.data
    coords, faces, edges, uvs, seams, groups = reads
    plans = rectify_targets(uvs, groups)
    if not plans:
        return
    before_distortion = [
        flatten_distortion(coords, faces, uvs, group) for group, _, _ in plans
    ]

    # the unwrap splits charts by seam marks, not by the uv map
    apply_seams(mesh, seams)
    bm = new_bmesh(obj)
    uvl = bm.loops.layers.uv.active
    bm.faces.ensure_lookup_table()
    for group, targets, inner in plans:
        for fi in group:
            for corner, loop in enumerate(bm.faces[fi].loops):
                target = targets.get(uvs[fi][corner])
                if target is None and inner is not None:
                    target = inner.get(uvs[fi][corner])
                if target is not None:
                    loop[uvl].uv = target
                    loop[uvl].pin_uv = inner is None
    set_bmesh(bm, obj)

    solver_faces = [fi for group, _, inner in plans if inner is None for fi in group]
    if solver_faces:
        unwrap(obj, solver_faces, REPAIR_ITERATIONS, method="CONFORMAL")
        unwrap(obj, solver_faces, REPAIR_ITERATIONS)

    bm = new_bmesh(obj)
    uvl = bm.loops.layers.uv.active
    bm.faces.ensure_lookup_table()
    for (group, _, _), before in zip(plans, before_distortion):
        total = sum(signed_area(uvs[fi]) for fi in group)
        orientation = 1 if total >= 0 else -1
        area_before = abs(total)
        # the distortion measure skips faces at or under this floor
        floor = FLIP_NOISE * area_before
        area_after = 0.0
        crushed = False
        solved = {}
        for fi in group:
            pts = [tuple(loop[uvl].uv) for loop in bm.faces[fi].loops]
            solved[fi] = pts
            face = faces[fi]
            for i in range(1, len(pts) - 1):
                a = signed_area([pts[0], pts[i], pts[i + 1]]) * orientation
                area_after += a
                if a <= floor and _fan_triangle_has_area(coords, face, i):
                    crushed = True
        after = flatten_distortion(coords, faces, solved, group)
        bad = (
            crushed
            or not RECTIFY_COLLAPSE * area_before
            <= area_after
            <= area_before / RECTIFY_COLLAPSE
            or after > before + RECTIFY_DISTORTION
        )
        for fi in group:
            for corner, loop in enumerate(bm.faces[fi].loops):
                if bad:
                    loop[uvl].uv = uvs[fi][corner]
                loop[uvl].pin_uv = False
    set_bmesh(bm, obj)


def island_copy(obj, group):
    mesh = obj.data
    layer = mesh.uv_layers.active
    used = sorted({v for fi in group for v in mesh.polygons[fi].vertices})
    local = {v: i for i, v in enumerate(used)}
    island_mesh = bpy.data.meshes.new("uvgami_island")
    island_mesh.from_pydata(
        [mesh.vertices[v].co.copy() for v in used],
        [],
        [[local[v] for v in mesh.polygons[fi].vertices] for fi in group],
    )
    island_layer = island_mesh.uv_layers.new()
    li = 0
    for fi in group:
        poly = mesh.polygons[fi]
        for c in range(poly.loop_total):
            island_layer.uv[li].vector = layer.uv[poly.loop_start + c].vector
            li += 1
    temp = bpy.data.objects.new("uvgami_island", island_mesh)
    bpy.context.scene.collection.objects.link(temp)
    temp.matrix_world = obj.matrix_world.copy()
    return temp


def remove_temp(temp):
    mesh = temp.data
    bpy.data.objects.remove(temp, do_unlink=True)
    bpy.data.meshes.remove(mesh)


# the engine keeps the map, only the stretch moves
def queue_nocut(obj, temp, group, bbox, area, k, input_path, props):
    layer = temp.data.uv_layers.active
    total = 0.0
    for poly in temp.data.polygons:
        pts = [
            tuple(layer.uv[poly.loop_start + c].vector) for c in range(poly.loop_total)
        ]
        total += signed_area(pts)
    mirrored = total < 0
    if mirrored:
        for corner in layer.uv:
            u, w = corner.vector
            corner.vector = (-u, w)

    bm = new_bmesh(temp)
    if any(len(f.verts) > 3 for f in bm.faces):
        triangulate(bm, temp.data)
        set_bmesh(bm, temp)
    else:
        bm.free()

    repair_flipped_island(obj, temp)

    name = f"{obj.name}_island_{k}"
    path = input_path / f"{engine_file_stem(name)}.obj"
    while path.is_file():
        path = path.parent / f"{path.stem}1.obj"
    export_obj(temp, path, True)
    # empty pin line, nothing is held
    with (path.parent / f"{path.stem}_fixed").open("w") as f:
        f.write("\nnocut")
    vertex_count = len(temp.data.vertices)
    remove_temp(temp)

    queue_fix(
        obj,
        IslandUVs(list(group), bbox, area, mirrored),
        name,
        path,
        vertex_count,
        props,
    )


def queue_relax(obj, group, bbox, area, k, input_path, props):
    queue_nocut(obj, island_copy(obj, group), group, bbox, area, k, input_path, props)


# 1 for a disk, one less per extra hole or handle
def euler_characteristic(temp, cut_edges):
    bm = new_bmesh(temp)
    bm.edges.ensure_lookup_table()
    bmesh.ops.split_edges(bm, edges=[bm.edges[i] for i in cut_edges])
    chi = len(bm.verts) - len(bm.edges) + len(bm.faces)
    bm.free()
    return chi


# the runs between two of the islands stay unmarked, which is what the unwrap welds
def weld_shared_seams(temp, island_of, island_count):
    bm = new_bmesh(temp)
    uvl = bm.loops.layers.uv.active
    splits = []
    seams = []
    for edge in bm.edges:
        if not edge_splits_uv(edge, uvl):
            continue
        splits.append(edge.index)
        shared = len({island_of[face.index] for face in edge.link_faces}) > 1
        edge.seam = not shared
        if not shared:
            seams.append(edge.index)
    set_bmesh(bm, temp)

    # joining n sheets along n - 1 single runs costs exactly n - 1
    expected = euler_characteristic(temp, splits) - (island_count - 1)
    if euler_characteristic(temp, seams) < expected:
        return (
            "The islands touch along more than one seam,"
            " so they can't be one flat island"
        )
    return None


# shared mesh edges are the only places the engine can weld
def islands_connected(mesh, targets):
    parent = list(range(len(targets)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    edge_owner = {}
    for i, (group, _, _) in enumerate(targets):
        for fi in group:
            for key in mesh.polygons[fi].edge_keys:
                j = edge_owner.setdefault(key, i)
                if j != i:
                    parent[find(i)] = find(j)
    return len({find(i) for i in range(len(targets))}) == 1


# a disconnected selection gives one area per connected piece
def target_areas(obj, rings):
    mesh = obj.data
    if not mesh.uv_layers.active:
        return None, 0, 0, "Mesh has no uv map"
    selected = selected_faces(mesh)
    if not selected:
        return None, 0, 0, "Select the faces of the area to fix"

    faces = face_vertices(mesh)
    uvs = face_uvs(mesh)
    edges = face_edges(faces)
    targets = []
    whole = ring_skipped = 0
    for group in uv_island_groups(faces, uvs, edges):
        group_set = set(group)
        island_selected = selected & group_set
        if not island_selected:
            continue
        # each uv-connected piece of the selection grows into its own area
        taken = set(island_selected)
        emitted = set()
        absorbed = set()
        for seed in sorted(
            face_components(faces, uvs, edges, island_selected), key=min
        ):
            patch = set(seed) - absorbed
            if not patch:
                continue
            # grow so the fix blends out instead of stopping at the selection edge
            uv_of = {
                faces[fi][i]: uvs[fi][i] for fi in patch for i in range(len(faces[fi]))
            }
            for _ in range(rings):
                ring = set(uv_of.items())
                grew = False
                for fi in sorted(group_set - taken):
                    corners = list(zip(faces[fi], uvs[fi]))
                    if ring.isdisjoint(corners):
                        continue
                    if any(uv_of.get(v, uv) != uv for v, uv in corners):
                        continue
                    patch.add(fi)
                    taken.add(fi)
                    uv_of.update(corners)
                    grew = True
                if not grew:
                    break
            # a pocket can hold another piece's faces
            enclosed = enclosed_faces(faces, edges, group_set, patch) - emitted
            patch |= enclosed
            taken |= enclosed
            absorbed |= enclosed
            if patch == group_set:
                # a fully covered island has no border to hold
                whole += 1
                continue
            emitted |= patch
            for comp in face_components(faces, uvs, edges, patch):
                comp_verts = {v for fi in comp for v in faces[fi]}
                comp_edges = {
                    pair(faces[fi][i], faces[fi][(i + 1) % len(faces[fi])])
                    for fi in comp
                    for i in range(len(faces[fi]))
                }
                if len(comp_verts) - len(comp_edges) + len(comp) != 1:
                    # the area rings a real hole, no selection can make it a disk
                    ring_skipped += 1
                    continue
                outside = group_set - comp
                border = comp_verts & {v for fi in outside for v in faces[fi]}
                targets.append((sorted(comp), border))
    return targets, whole, ring_skipped, None


# pieces are joined by edges whose corner uvs agree on both faces
def face_components(faces, uvs, edges, patch):
    def corner_uv(f, v):
        return uvs[f][faces[f].index(v)]

    unvisited = set(patch)
    components = []
    while unvisited:
        seed = unvisited.pop()
        component = {seed}
        stack = [seed]
        while stack:
            fi = stack.pop()
            face = faces[fi]
            for i in range(len(face)):
                u, v = face[i], face[(i + 1) % len(face)]
                for nb in edges[pair(u, v)]:
                    if (
                        nb in unvisited
                        and corner_uv(nb, u) == corner_uv(fi, u)
                        and corner_uv(nb, v) == corner_uv(fi, v)
                    ):
                        unvisited.discard(nb)
                        component.add(nb)
                        stack.append(nb)
        components.append(component)
    return components


# a ring selection strands these, and the engine can only keep a disk
def enclosed_faces(faces, edges, group_set, patch):
    boundary = {
        e
        for e, owners in edges.items()
        if sum(1 for fi in owners if fi in group_set) == 1
    }
    unvisited = group_set - patch
    stranded = []
    while unvisited:
        seed = unvisited.pop()
        component = {seed}
        stack = [seed]
        touches_boundary = False
        while stack:
            fi = stack.pop()
            face = faces[fi]
            for i in range(len(face)):
                e = pair(face[i], face[(i + 1) % len(face)])
                if e in boundary:
                    touches_boundary = True
                for nb in edges[e]:
                    if nb in unvisited:
                        unvisited.discard(nb)
                        component.add(nb)
                        stack.append(nb)
        if not touches_boundary:
            stranded.append(component)
    if not boundary and stranded:
        # a closed island has no boundary edges, the largest piece is the outside
        stranded.remove(max(stranded, key=len))
    enclosed = set()
    for component in stranded:
        enclosed |= component
    return enclosed


# a twisted quad hides a flipped triangle behind a positive polygon area
def has_flipped(mesh, patch, uv_of):
    for fi in patch:
        pts = [uv_of[v] for v in mesh.polygons[fi].vertices]
        for i in range(1, len(pts) - 1):
            if signed_area([pts[0], pts[i], pts[i + 1]]) <= 0:
                return True
    return False


# queue_area's export gives one uv per vert, so such a patch would weld the seam
def spans_own_seam(mesh, patch):
    layer = mesh.uv_layers.active
    uv_of = {}
    for fi in patch:
        poly = mesh.polygons[fi]
        for c, v in enumerate(poly.vertices):
            vec = layer.uv[poly.loop_start + c].vector
            uv = (round(vec[0], 6), round(vec[1], 6))
            if uv_of.setdefault(v, uv) != uv:
                return True
    return False


# runs on the exported copy, so the visible map moves once
def repair_flipped(obj, area_mesh, used, uv_of, border):
    temp = bpy.data.objects.new("uvgami_area", area_mesh)
    bpy.context.scene.collection.objects.link(temp)

    bm = bmesh.new()
    bm.from_mesh(area_mesh)
    uvl = bm.loops.layers.uv.new()
    for face in bm.faces:
        for loop in face.loops:
            v = used[loop.vert.index]
            loop[uvl].uv = uv_of[v]
            loop[uvl].pin_uv = v in border
    bm.to_mesh(area_mesh)
    bm.free()

    # multi-object edit would pull the input mesh in and clear its faces
    obj.select_set(False)
    unwrap(temp, range(len(area_mesh.polygons)), REPAIR_ITERATIONS)
    obj.select_set(True)

    layer = area_mesh.uv_layers.active
    for poly in area_mesh.polygons:
        for c, v in enumerate(poly.vertices):
            uv_of[used[v]] = tuple(layer.uv[poly.loop_start + c].vector)
    bpy.data.objects.remove(temp, do_unlink=True)


# inside one island every vert has one uv, so vt indices mirror v indices
def queue_area(obj, patch, border, k, input_path, props, nocut):
    mesh = obj.data
    layer = mesh.uv_layers.active
    used = sorted({v for fi in patch for v in mesh.polygons[fi].vertices})
    local = {v: i for i, v in enumerate(used)}

    uv_of = {}
    pins = []
    for fi in patch:
        poly = mesh.polygons[fi]
        for c, v in enumerate(poly.vertices):
            uv = tuple(layer.uv[poly.loop_start + c].vector)
            uv_of[v] = uv
            if v in border:
                pins.append((fi, c, uv))

    # a pinned solve can only fill the border's own orientation
    total = 0.0
    for fi in patch:
        pts = [uv_of[v] for v in mesh.polygons[fi].vertices]
        for i in range(1, len(pts) - 1):
            total += signed_area([pts[0], pts[i], pts[i + 1]])
    mirrored = total < 0
    if mirrored:
        uv_of = {v: (-u, w) for v, (u, w) in uv_of.items()}

    area_mesh = bpy.data.meshes.new("uvgami_area")
    area_mesh.from_pydata(
        [mesh.vertices[v].co.copy() for v in used],
        [],
        [[local[v] for v in mesh.polygons[fi].vertices] for fi in patch],
    )

    # a flipped area would make the engine reject the map
    if has_flipped(mesh, patch, uv_of):
        repair_flipped(obj, area_mesh, used, uv_of, border)

    bm = bmesh.new()
    bm.from_mesh(area_mesh)
    if any(len(f.verts) > 3 for f in bm.faces):
        triangulate(bm, area_mesh)
        bm.to_mesh(area_mesh)
    bm.free()

    name = f"{obj.name}_area_{k}"
    path = input_path / f"{engine_file_stem(name)}.obj"
    while path.is_file():
        path = path.parent / f"{path.stem}1.obj"

    matrix = obj.matrix_world
    with path.open("w") as f:
        for v in area_mesh.vertices:
            x, y, z = matrix @ v.co
            f.write(f"v {x} {y} {z}\n")
        for v in area_mesh.vertices:
            u, w = uv_of[used[v.index]]
            f.write(f"vt {u} {w}\n")
        for poly in area_mesh.polygons:
            corners = " ".join(f"{v + 1}/{v + 1}" for v in poly.vertices)
            f.write(f"f {corners}\n")
    with (path.parent / f"{path.stem}_fixed").open("w") as f:
        f.write(",".join(str(local[v]) for v in sorted(border)))
        if nocut:
            f.write("\nnocut")

    vertex_count = len(area_mesh.vertices)
    bpy.data.meshes.remove(area_mesh)

    queue_fix(obj, AreaUVs(patch, pins, mirrored), name, path, vertex_count, props)


# optcuts is the only engine that can pin a border or stitch islands
def validate_engine(op):
    engine = get_engine("OPTCUTS")
    if manager.is_active and manager.engine is not engine:
        op.report(
            {"ERROR"},
            "Finish or cancel the current unwrap first",
        )
        return None, None
    engine_ctx, error = engine.validate(get_preferences())
    if error is not None:
        op.report({"ERROR"}, error)
        return None, None
    return engine, engine_ctx


def queue_targets(engine, engine_ctx, obj, count, queue_one):
    input_path, output_path = get_io_dir_paths()
    if not manager.is_active:
        clear_io_dir(input_path)
        clear_io_dir(output_path)

    queued = len(manager._queue)
    try:
        for k in range(count):
            queue_one(k, input_path)
    except Exception:
        # drop the partial batch or it silently runs with the next session
        while len(manager._queue) > queued:
            manager._queue.pop()
        raise

    if not manager.is_active:
        props = bpy.context.scene.uvgami
        info = logger.new_info()
        info.engine = engine.describe()
        info.objects = [(obj.name, triangle_count(obj))]
        info.settings = describe_settings(props, fix_settings(props))
        manager.engine = engine
        manager.engine_ctx = engine_ctx
        # these operators are run from the uv editor, so the bar belongs there
        manager.start(uv_editor=True)


# registering a subclass of a registered operator unregisters the parent
class FixOperator:
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"

    def execute(self, context):
        engine, engine_ctx = validate_engine(self)
        if engine is None:
            return {"CANCELLED"}

        obj = context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")
        try:
            return self.queue_selection(obj, context.scene.uvgami, engine, engine_ctx)
        finally:
            obj.select_set(True)
            context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="EDIT")


class IslandOperator(FixOperator):
    queue_target = None
    verb = ""

    def queue_selection(self, obj, props, engine, engine_ctx):
        targets, error = target_islands(obj)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}

        def queue_one(k, input_path):
            group, bbox, area = targets[k]
            self.queue_target(obj, group, bbox, area, k + 1, input_path, props)

        queue_targets(engine, engine_ctx, obj, len(targets), queue_one)
        self.report({"INFO"}, f"{self.verb} {len(targets)} island(s)")
        return {"FINISHED"}


class UVGAMI_OT_unwrap_island(IslandOperator, bpy.types.Operator):
    bl_idname = "uvgami.unwrap_island"
    bl_label = "Unwrap Island"
    bl_description = "Re-unwrap the island under the selected face(s)"
    queue_target = staticmethod(queue_island)
    verb = "Unwrapping"


class UVGAMI_OT_relax_island(IslandOperator, bpy.types.Operator):
    bl_idname = "uvgami.relax_island"
    bl_label = "Relax Island"
    bl_description = (
        "Relax the island under the selected faces to reduce stretching"
        " without changing its seams"
    )
    queue_target = staticmethod(queue_relax)
    verb = "Relaxing"


class UVGAMI_OT_combine_islands(FixOperator, bpy.types.Operator):
    bl_idname = "uvgami.combine_islands"
    bl_label = "Combine Islands"
    bl_description = "Select a face on two islands to re-unwrap them as one island"

    def queue_selection(self, obj, props, engine, engine_ctx):
        targets, error = target_islands(obj)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        if len(targets) < 2:
            self.report({"ERROR"}, "Select faces on at least two islands")
            return {"CANCELLED"}
        if not islands_connected(obj.data, targets):
            self.report(
                {"ERROR"},
                "The selected islands don't touch, select a face on each island between them",
            )
            return {"CANCELLED"}

        group = sorted({fi for g, _, _ in targets for fi in g})
        bbox = (
            min(b[0] for _, b, _ in targets),
            min(b[1] for _, b, _ in targets),
            max(b[2] for _, b, _ in targets),
            max(b[3] for _, b, _ in targets),
        )
        area = sum(a for _, _, a in targets)

        # blender unwraps the union with only the shared runs welded
        island_of = {fi: i for i, (g, _, _) in enumerate(targets) for fi in g}
        temp = island_copy(obj, group)
        error = weld_shared_seams(temp, [island_of[fi] for fi in group], len(targets))
        if error:
            remove_temp(temp)
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        obj.select_set(False)
        unwrap(temp, range(len(temp.data.polygons)), REPAIR_ITERATIONS)
        obj.select_set(True)
        bm = new_bmesh(temp)
        folded = bool(folded_faces(bm, bm.loops.layers.uv.active))
        bm.free()

        if folded:
            # the engine can't unfold a kept map, so it cuts the union fresh
            remove_temp(temp)

            def queue_one(k, input_path):
                queue_island(obj, group, bbox, area, k + 1, input_path, props)

            message = "The combined island folds, unwrapping it from scratch instead"
            level = "WARNING"
        else:

            def queue_one(k, input_path):
                queue_nocut(obj, temp, group, bbox, area, k + 1, input_path, props)

            message = f"Combining {len(targets)} islands"
            level = "INFO"

        queue_targets(engine, engine_ctx, obj, 1, queue_one)
        self.report({level}, message)
        return {"FINISHED"}


class AreaOperator(FixOperator):
    nocut = False

    def queue_selection(self, obj, props, engine, engine_ctx):
        targets, whole, rings, error = target_areas(obj, props.area_expand)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        seam_skipped = 0
        for target in list(targets):
            if spans_own_seam(obj.data, target[0]):
                targets.remove(target)
                seam_skipped += 1
        if not targets:
            if whole:
                error = "The whole island is selected, use Unwrap Island instead"
            elif seam_skipped:
                error = (
                    "The area covers both sides of a seam,"
                    " deselect the faces on one side"
                )
            else:
                error = "The area rings a hole, deselect a face to break the ring"
            self.report({"ERROR"}, error)
            return {"CANCELLED"}

        def queue_one(k, input_path):
            patch, border = targets[k]
            queue_area(obj, patch, border, k + 1, input_path, props, self.nocut)

        queue_targets(engine, engine_ctx, obj, len(targets), queue_one)

        notes = []
        if whole:
            notes.append(f"{whole} whole island(s) skipped")
        if rings:
            notes.append(f"{rings} area(s) around holes skipped")
        if seam_skipped:
            notes.append(f"{seam_skipped} area(s) on both sides of a seam skipped")
        if notes:
            self.report({"WARNING"}, ", ".join(notes))
        return {"FINISHED"}


class UVGAMI_OT_unwrap_area(AreaOperator, bpy.types.Operator):
    bl_idname = "uvgami.unwrap_area"
    bl_label = "Unwrap Area"
    bl_description = "Re-unwrap the selected faces with cuts if necessary"


class UVGAMI_OT_relax_area(AreaOperator, bpy.types.Operator):
    bl_idname = "uvgami.relax_area"
    bl_label = "Relax Area"
    bl_description = "Relax the selected faces to reduce stretching"
    nocut = True
