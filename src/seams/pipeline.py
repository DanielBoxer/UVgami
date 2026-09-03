import collections

from .boundaries import boundary_edges, flatten_teeth, reroute_boundaries
from .cancel import check_cancelled
from .cuts import crease_relief, disk_cuts
from .mesh import LOW_ANGLE, build, diagonal, norm, pair, turn_angle
from .regions import (
    CREASE_ANGLE,
    PANEL_SHARE,
    absorb,
    close_rings,
    detect_width,
    merge_flat,
    merge_smooth,
    panel_share,
    partition,
    unfold_hinges,
)
from .sweeps import split_sweeps, sweep_rims

# what a loose part run reads off the whole mesh (face_ids: see straight_runs)
WholeMesh = collections.namedtuple("WholeMesh", "min_width model_area face_ids")


# the forced edges with the sweep rims added, and the wall faces
def rim_cuts(verts, faces, rims, forced=None, whole=None, built=None):
    if not rims:
        return forced, None
    model_area, face_ids = (None, None) if whole is None else whole[1:]
    rim_edges, walls = sweep_rims(verts, faces, model_area, face_ids, built)
    if rim_edges:
        return rim_edges | (forced or set()), walls
    return forced, walls


# walls partition as one region however sharply the coarse wall turns
def low_partition(verts, faces, forced=None, walls=None, built=None):
    weighted, areas, edges = built or build(verts, faces)
    smooth = None
    if walls:
        smooth = {
            key
            for key, owners in edges.items()
            if len(owners) == 2 and owners[0] in walls and owners[1] in walls
        }
    root = partition(faces, weighted, edges, LOW_ANGLE, forced, smooth)
    return weighted, areas, edges, root


# a run over one loose part gets the seams the whole run would
def whole_mesh_inputs(verts, faces, rims=True, forced=None):
    built = build(verts, faces)
    cut_from_start, walls = rim_cuts(verts, faces, rims, forced, built=built)
    weighted, areas, edges, root = low_partition(
        verts, faces, cut_from_start, walls, built
    )
    min_width = detect_width(verts, faces, areas, edges, root, diagonal(verts))
    return min_width, sum(areas)


# what survives the merges is the feature structure the seams will trace
def feature_labels(
    verts,
    faces,
    angle=CREASE_ANGLE,
    rims=True,
    forced=None,
    scale=None,
    walls=None,
    cancelled=None,
    whole=None,
    built=None,
):
    weighted, areas, edges, root = low_partition(verts, faces, forced, walls, built)
    if whole is None:
        if scale is None:
            scale = diagonal(verts)
        min_width = detect_width(verts, faces, areas, edges, root, scale)
        model_area = face_ids = None
    else:
        min_width, model_area, face_ids = whole
    label, bounds = absorb(
        verts, faces, weighted, areas, edges, root, min_width, forced
    )
    check_cancelled(cancelled)
    label = merge_smooth(edges, label, bounds, min_width, angle, forced)
    label = merge_flat(weighted, areas, edges, label, angle, forced)
    label = close_rings(verts, weighted, areas, edges, label, angle, forced)
    presplit = label
    check_cancelled(cancelled)
    if rims:
        label = split_sweeps(
            verts,
            faces,
            weighted,
            areas,
            edges,
            label,
            min_width,
            model_area,
            face_ids,
        )
    locked = set()
    for key, owners in edges.items():
        if len(owners) != 2:
            continue
        a, b = owners
        if label[a] == label[b]:
            continue
        if (forced and key in forced) or presplit[a] == presplit[b]:
            locked.add(pair(label[a], label[b]))
    return weighted, areas, edges, label, min_width, locked


# one region covering the part means no structure, a smooth blob
ORGANIC_SHARE = 0.9
# regions averaging under this many faces mean the partition found noise
FRAGMENT_FACES = 8
# spread curvature, read away from the region boundaries
SPREAD_ANGLE = 25
SPREAD_SHARE = 0.21
# the boundaries must mostly be creased or a rim split_sweeps placed
BOUNDARY_ANGLE = 20
BOUNDARY_CREASED = 0.6


# ties fall organic, seams on sculpt ridges cost more than a slow unwrap
def is_hard_surface(verts, faces):
    # verts can be the whole mesh with faces one loose part
    used = {v for face in faces for v in face}
    part_scale = diagonal([verts[v] for v in used])
    built = build(verts, faces)
    rims, walls = sweep_rims(verts, faces, built=built)
    weighted, areas, edges, presweep, _, _ = feature_labels(
        verts,
        faces,
        rims=False,
        forced=rims or None,
        scale=part_scale,
        walls=walls,
        built=built,
    )
    label = split_sweeps(verts, faces, weighted, areas, edges, presweep)
    total = sum(areas)
    if total <= 0:
        return False
    region = collections.defaultdict(float)
    for i, r in label.items():
        region[r] += areas[i]
    # a smooth cylinder's wall is one big region, so the organic guard skips it
    wall_regions = {label[i] for i in walls}
    top = max(region, key=region.get)
    if region[top] / total >= ORGANIC_SHARE and top not in wall_regions:
        return False
    if len(faces) / len(region) < FRAGMENT_FACES:
        # a low poly box trips FRAGMENT_FACES too, only curled regions are noise
        root = partition(faces, weighted, edges, LOW_ANGLE)
        panels = collections.defaultdict(list)
        for i in range(len(faces)):
            panels[root(i)].append(i)
        if panel_share(weighted, panels.values()) < PANEL_SHARE:
            return False

    near = set()
    for key, owners in edges.items():
        if len(owners) == 2 and label[owners[0]] != label[owners[1]]:
            near.update(owners)
    # two rings, so a dissolved bevel band beside a seam stays out
    for _ in range(2):
        grown = set(near)
        for owners in edges.values():
            if len(owners) == 2 and not near.isdisjoint(owners):
                grown.update(owners)
        near = grown

    spread = interior = boundary = boundary_creased = 0.0
    for (a, b), owners in edges.items():
        if len(owners) != 2:
            continue
        length = norm([verts[a][i] - verts[b][i] for i in range(3)])
        turn = turn_angle(weighted, owners)
        if label[owners[0]] != label[owners[1]]:
            boundary += length
            if turn >= BOUNDARY_ANGLE or presweep[owners[0]] == presweep[owners[1]]:
                boundary_creased += length
        elif owners[0] not in near and owners[1] not in near:
            # wall curvature is explained by the sweep, not sculpt detail
            if owners[0] in walls and owners[1] in walls:
                continue
            interior += length
            if LOW_ANGLE < turn < SPREAD_ANGLE:
                spread += length
    if not boundary:
        return False
    return (
        not interior or spread / interior < SPREAD_SHARE
    ) and boundary_creased / boundary >= BOUNDARY_CREASED


# a closed mesh with every feature under the angle reruns at the CREASE_ANGLE floor
def seam_edges(
    verts,
    faces,
    angle=CREASE_ANGLE,
    rims=True,
    weights=None,
    forced=None,
    cancelled=None,
    whole=None,
):
    built = build(verts, faces)
    seams, closed = seams_at_angle(
        verts, faces, angle, rims, weights, forced, cancelled, whole, built
    )
    if not seams and angle > CREASE_ANGLE and closed:
        seams, _ = seams_at_angle(
            verts, faces, CREASE_ANGLE, rims, weights, forced, cancelled, whole, built
        )
    return seams


# a caller joining parts decides the rerun for all of them
def seams_at_angle(
    verts, faces, angle, rims, weights, forced, cancelled, whole, built=None
):
    if built is None:
        built = build(verts, faces)
    cut_from_start, walls = rim_cuts(verts, faces, rims, forced, whole, built)
    check_cancelled(cancelled)
    weighted, areas, edges, label, min_width, locked = feature_labels(
        verts,
        faces,
        angle,
        rims,
        cut_from_start,
        walls=walls,
        cancelled=cancelled,
        whole=whole,
        built=built,
    )
    label = flatten_teeth(weighted, faces, edges, label, angle, forced)
    relief = crease_relief(verts, faces, weighted, edges)
    check_cancelled(cancelled)
    label = reroute_boundaries(verts, faces, areas, edges, label, relief, forced)
    # everything since absorb can leave a region under its width floor
    label, _ = absorb(
        verts,
        faces,
        weighted,
        areas,
        edges,
        label.__getitem__,
        min_width,
        forced,
        locked,
    )
    check_cancelled(cancelled)
    hinges = (
        unfold_hinges(verts, faces, weighted, edges, label, cut_from_start)
        if rims
        else set()
    )
    check_cancelled(cancelled)
    # disk_cuts counts in-region forced edges as slits, so it gets only user marks
    seams = (boundary_edges(edges, label) - hinges) | disk_cuts(
        verts, edges, label, weights, relief, forced
    )
    if forced:
        seams |= forced
    return seams, all(len(owners) != 1 for owners in edges.values())
