"""The entry points. seam_edges runs the whole thing: partition and
merges (feature_labels), boundary cleanup, then the surviving region
boundaries plus disk cuts are the seams. is_hard_surface reads the
same region structure to decide whether a loose part has features
worth preseeding."""

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


def rim_cuts(verts, faces, rims, forced=None, whole=None):
    """The forced edges with the sweep rims added, and the wall faces."""
    if not rims:
        return forced, None
    model_area, face_ids = (None, None) if whole is None else whole[1:]
    rim_edges, walls = sweep_rims(verts, faces, model_area, face_ids)
    if rim_edges:
        return rim_edges | (forced or set()), walls
    return forced, walls


def low_partition(verts, faces, forced=None, walls=None):
    """The over-segmented partition every pass starts from, with the
    per-face normals, areas and edge map it was built on. walls are faces
    sweep_rims verified as one swept wall: they partition as one region
    however sharply the coarse wall turns, so an annulus never depends on
    the merges rebuilding it from columns."""
    weighted, areas, edges = build(verts, faces)
    smooth = None
    if walls:
        smooth = {
            key
            for key, owners in edges.items()
            if len(owners) == 2 and owners[0] in walls and owners[1] in walls
        }
    root = partition(faces, weighted, edges, LOW_ANGLE, forced, smooth)
    return weighted, areas, edges, root


def whole_mesh_inputs(verts, faces, rims=True, forced=None):
    """The two numbers the passes read off the whole mesh, the auto width
    and the model area. Given to a run over one loose part, it gets the
    seams the whole run would."""
    cut_from_start, walls = rim_cuts(verts, faces, rims, forced)
    weighted, areas, edges, root = low_partition(verts, faces, cut_from_start, walls)
    min_width = detect_width(verts, faces, areas, edges, root, diagonal(verts))
    return min_width, sum(areas)


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
):
    """Region labels from the merge passes: partition at auto width, the three
    merges, sweep rims. What survives is the feature structure the seams will
    trace, before the boundary cleanup passes move any edge. scale is the
    model size the width cap reads, the full diagonal of verts by default.
    whole is the WholeMesh a loose part run alone reads, None reads the
    width off faces.

    Also returns the absorb width and the deliberate boundary pairs, region
    pairs a forced edge or the sweep split separates, so a later width pass
    can absorb leftovers without dissolving structure."""
    weighted, areas, edges, root = low_partition(verts, faces, forced, walls)
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


# auto mode guards, tuned on the bench sets.
# a part is hard surface only when every one holds:
# one region covering the part means no structure, a smooth blob,
ORGANIC_SHARE = 0.9
# regions averaging under this many faces mean the partition found noise
# (chainmail reads 1 face per region),
FRAGMENT_FACES = 8
# turn between LOW_ANGLE and this is spread curvature. a bevel is spread
# turn beside a boundary and a sculpt is spread turn everywhere, so the
# share is read away from the region boundaries,
SPREAD_ANGLE = 25
SPREAD_SHARE = 0.21
# and the boundaries must mostly be deliberate: creased, or a rim
# split_sweeps placed
BOUNDARY_ANGLE = 20
BOUNDARY_CREASED = 0.6


def is_hard_surface(verts, faces):
    """Whether a loose part's features are worth preseeding.

    Reads the merged region structure at the CREASE_ANGLE floor, not the
    feature angle knob: the question is whether structure exists at all.
    Detected sweep walls are forced apart and count as structure, so a
    smooth cylinder still reads hard, and split_sweeps runs so a filleted
    one does too. Misreading organic costs a slow from-scratch unwrap,
    misreading hard costs seams on sculpt ridges, so ties fall organic.
    """
    # verts can be the whole mesh with faces one loose part, so the width cap
    # must read the part's own size or nearby geometry changes the label
    used = {v for face in faces for v in face}
    part_scale = diagonal([verts[v] for v in used])
    rims, walls = sweep_rims(verts, faces)
    weighted, areas, edges, presweep, _, _ = feature_labels(
        verts, faces, rims=False, forced=rims or None, scale=part_scale, walls=walls
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
    # two rings, so a dissolved bevel band beside a seam stays out of the
    # interior read
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
    """The full pipeline at auto width: partition, merges, cleanup, seams.

    angle is what counts as a feature: boundaries turning less than it
    merge away, so lower keeps more shallow-feature seams at the cost of
    shattering coarse curved walls. rims off skips the sweep passes and the
    unfold, so a smooth cylinder keeps its end caps. weights are painted
    restrictions the cuts avoid, region boundaries sit where the shape says
    and paint does not move them. forced edges cut from the partition on,
    so the merges route around them, and they are seams in the end.

    Sweep rims cut the partition the same way but stay out of the cleanup
    passes' forced set: a rim traces a per-face wall/cap or panel split, so
    it staircases, and only flatten_teeth and reroute_boundaries can settle
    it. A moved rim stays a seam through boundary_edges, no re-adding.

    cancelled is checked between passes, so a cancel lands at the next one.

    whole is the WholeMesh a loose part run alone reads (seam_edges_parallel
    runs the parts that way), None on a whole mesh. A closed mesh with every feature under the angle
    cannot flatten, so it reruns at the CREASE_ANGLE floor.
    """
    seams, closed = seams_at_angle(
        verts, faces, angle, rims, weights, forced, cancelled, whole
    )
    if not seams and angle > CREASE_ANGLE and closed:
        seams, _ = seams_at_angle(
            verts, faces, CREASE_ANGLE, rims, weights, forced, cancelled, whole
        )
    return seams


def seams_at_angle(verts, faces, angle, rims, weights, forced, cancelled, whole):
    """seam_edges at one angle with no rerun, plus whether the mesh is
    closed, so a caller joining parts decides the rerun for all of them."""
    cut_from_start, walls = rim_cuts(verts, faces, rims, forced, whole)
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
    # disk_cuts counts in-region forced edges as slits, so it gets only the
    # user marks: a rim the cleanup moved would otherwise linger as one
    seams = (boundary_edges(edges, label) - hinges) | disk_cuts(
        verts, edges, label, weights, relief, forced
    )
    if forced:
        seams |= forced
    return seams, all(len(owners) != 1 for owners in edges.values())
