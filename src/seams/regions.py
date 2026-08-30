"""Partition and merge, the core of the mode.

A per-edge angle test cannot seam a beveled model: a bevel splits one
crease into several small turns, and what separates it from a corner
is width, not angle. So partition at a low angle, which over-segments
bevels and curved surfaces into narrow bands, then merge back: absorb
dissolves anything narrower than the auto width, merge_smooth takes
any boundary that turns less than a crease, merge_flat joins
neighbours whose union is still nearly flat, and close_rings rejoins
disk pairs whose union is a short annulus. Every merge refuses to
leave a region with a hole, the engine throws non-disk charts away."""

import collections
import functools
import heapq
import math

from .islands import SPLIT_ASPECT
from .mesh import LOW_ANGLE, cross, find, norm, pair, turn_angle

# auto width: at the low partition angle region widths are dominated by
# narrow bands, real surfaces are the top few percent, and no clean gap
# separates them, so take a high quantile with clearance on top. the cap, a
# fraction of the diagonal, stops sparse cases like a beveled cube from
# getting an oversized width
WIDTH_CAP = 0.05
WIDTH_QUANTILE = 0.9
WIDTH_FACTOR = 2.0
# flat merge: |sum of weighted normals| / (2 * area) is 1 on a plane. merge
# while the union stays above a spherical cap of this half-angle, so shallow
# creases merge and real corners (~0.7 for a 90 degree pair) survive
FLAT_ANGLE = 30
# smooth merge: how far a boundary must turn to count as a crease. read at
# the boundary because flatness cannot pass a cylinder, whose panels spread
# as far as a corner once enough of them merge
CREASE_ANGLE = 30
# unfold: a region whose mass sums flat panel by panel is paper, it unfolds
# rigidly however far a hinge turns. curved mass drops the share: a
# hemisphere reads 0.5, a quarter-bent strip about 0.9
PANEL_SHARE = 0.95
# every edge of a hinge must lie on one line, within this cosine of the
# first edge's direction, or the fold cannot open rigidly
HINGE_LINE_COS = 0.996


# per region boundary, the length-weighted sums the smooth merge reads: turn
# carried across dissolved bands, turn at the boundary's own edges, the
# width that carry crossed, and length to divide by
Boundaries = collections.namedtuple("Boundaries", "turn step spread length")


def partition(faces, weighted, edges, angle, forced=None, smooth=None):
    """Union-find over faces, cutting every edge sharper than angle. Edges in
    forced cut whatever they turn, edges in smooth merge whatever they turn,
    and forced wins when an edge is in both."""
    parent = list(range(len(faces)))

    for key, owners in edges.items():
        if len(owners) != 2:
            continue
        if forced and key in forced:
            continue
        if turn_angle(weighted, owners) > angle and not (smooth and key in smooth):
            continue
        a, b = owners
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[ra] = rb
    return functools.partial(find, parent)


def boundary_turns(verts, weighted, edges, label):
    """Per region boundary, its turn summed over the edges, weighted by length.

    Divided by the boundary's length this is the angle the surface turns
    crossing it. Kept as a sum so merges can add boundaries together."""
    total = collections.defaultdict(float)
    for (v0, v1), owners in edges.items():
        if len(owners) != 2:
            continue
        ra, rb = label[owners[0]], label[owners[1]]
        if ra == rb:
            continue
        length = norm([verts[v0][i] - verts[v1][i] for i in range(3)])
        total[pair(ra, rb)] += length * turn_angle(weighted, owners)
    return total


def region_topology(edges, label):
    """Per-region euler characteristic, vertex sets and shared edge counts.

    EC is 1 for a disk, 0 once a region has a hole. Both merge passes refuse
    anything that lowers it, the engine throws non-disk charts away.
    """
    face_count = collections.Counter(label.values())
    rverts = collections.defaultdict(set)
    edge_count = collections.Counter()
    shared = collections.Counter()
    for (v0, v1), owners in edges.items():
        regions = {label[o] for o in owners}
        for r in regions:
            rverts[r].update((v0, v1))
            edge_count[r] += 1
        if len(regions) == 2:
            ra, rb = sorted(regions)
            shared[(ra, rb)] += 1
    ec = {r: len(rverts[r]) - edge_count[r] + face_count[r] for r in face_count}
    return ec, rverts, shared


def joint_count(rverts, shared, a, b):
    """Shared vertices minus shared edges: 1 when the two regions meet along a
    single path, 2 when they meet twice (so the union has a hole), 0 when the
    contact is a closed loop (so the union is closed)."""
    return len(rverts[a] & rverts[b]) - shared[pair(a, b)]


def keeps_topology(ec, rverts, shared, a, b):
    """A merge is allowed when the union is no worse than the worse of the two
    and not a closed surface: meeting along one path keeps a disk, meeting
    twice would open a hole, and swallowing the region inside a hole closes
    one.
    """
    merged = ec[a] + ec[b] - joint_count(rverts, shared, a, b)
    return min(ec[a], ec[b]) <= merged <= 1


def locked_pairs(edges, label, forced):
    """Region pairs whose shared boundary holds a forced seam. No merge may
    take one, so a hand-marked edge survives as a region boundary and the
    passes route around it.
    """
    locked = set()
    if not forced:
        return locked
    for key, owners in edges.items():
        if len(owners) == 2 and key in forced:
            ra, rb = label[owners[0]], label[owners[1]]
            if ra != rb:
                locked.add(pair(ra, rb))
    return locked


def region_stats(verts, areas, edges, label):
    area = collections.defaultdict(float)
    shared = collections.defaultdict(float)  # (ra, rb) -> boundary length
    perimeter = collections.defaultdict(float)
    for i, a in enumerate(areas):
        area[label[i]] += a
    for (v0, v1), owners in edges.items():
        length = norm([verts[v0][i] - verts[v1][i] for i in range(3)])
        regions = {label[o] for o in owners}
        if len(regions) == 2:
            ra, rb = sorted(regions)
            shared[(ra, rb)] += length
            perimeter[ra] += length
            perimeter[rb] += length
        elif len(owners) == 1:
            perimeter[label[owners[0]]] += length
    return area, shared, perimeter


def detect_width(verts, faces, areas, edges, root, scale):
    """Absolute merge width from a high quantile of the region widths."""
    label = {i: root(i) for i in range(len(faces))}
    area, _, perimeter = region_stats(verts, areas, edges, label)
    ws = sorted(2 * area[r] / perimeter[r] for r in area if perimeter[r] > 0)
    if not ws:
        return 0.0
    quantile = ws[min(len(ws) - 1, int(WIDTH_QUANTILE * len(ws)))]
    return min(WIDTH_FACTOR * quantile, WIDTH_CAP * scale)


def absorb(
    verts, faces, weighted, areas, edges, root, min_width, forced=None, locked=None
):
    """Repeatedly merge the narrowest region into its longest-shared neighbour.

    Stats update incrementally per merge and the heap is lazy, an entry
    whose width no longer matches its region is stale. Dissolving a band
    moves its turn and width onto the boundaries it leaves, so an absorbed
    bevel still reads as the crease it was, while step, the turn at a
    boundary's own edges, never moves. Returns labels plus the per-boundary
    sums the smooth merge reads. locked adds region pairs that may never
    merge on top of the pairs forced edges lock, for a caller whose
    deliberate boundaries no longer sit on their original edges.
    """
    label = {i: root(i) for i in range(len(faces))}
    area, shared, perimeter = region_stats(verts, areas, edges, label)
    turns = boundary_turns(verts, weighted, edges, label)
    steps = collections.defaultdict(float, turns)
    spread = collections.defaultdict(float)
    ec, rverts, shared_edges = region_topology(edges, label)
    locked = locked_pairs(edges, label, forced) | (locked or set())
    neighbors = collections.defaultdict(set)
    for ra, rb in shared:
        neighbors[ra].add(rb)
        neighbors[rb].add(ra)
    alive = set(area)
    parent = {r: r for r in area}

    def width(r):
        return 2 * area[r] / perimeter[r] if perimeter[r] > 0 else math.inf

    heap = [(width(r), r) for r in alive if width(r) < min_width]
    heapq.heapify(heap)
    while heap:
        w, region = heapq.heappop(heap)
        if region not in alive or width(region) != w:
            continue
        partners = [
            (shared[pair(region, n)], n) for n in neighbors[region] if n in alive
        ]
        if not partners:
            continue
        # prefer a neighbour already over the threshold: a bevel should
        # dissolve into its surface, not accrete into a wider strip that
        # then survives the threshold itself
        wide = [p for p in partners if width(p[1]) >= min_width]
        rest = [p for p in partners if width(p[1]) < min_width]
        best = next(
            (
                n
                for _, n in sorted(wide, reverse=True) + sorted(rest, reverse=True)
                if pair(region, n) not in locked
                and keeps_topology(ec, rverts, shared_edges, region, n)
            ),
            None,
        )
        # nothing can take this strip without opening a hole
        if best is None:
            continue

        contact = shared[pair(region, best)]
        # the turn crossing the band that is about to disappear, and how wide
        # it is, so the far side can tell a crease from spread out curvature
        crossed = turns[pair(region, best)] / contact if contact > 0 else 0.0
        crossed_width = width(region) + (
            spread[pair(region, best)] / contact if contact > 0 else 0.0
        )
        area[best] += area[region]
        perimeter[best] += perimeter[region] - 2 * contact
        ec[best] += ec[region] - joint_count(rverts, shared_edges, region, best)
        rverts[best] |= rverts[region]
        for n in list(neighbors[region]):
            if n != best and n in alive:
                turns[pair(best, n)] += (
                    turns[pair(region, n)] + shared[pair(region, n)] * crossed
                )
                spread[pair(best, n)] += (
                    spread[pair(region, n)] + shared[pair(region, n)] * crossed_width
                )
                steps[pair(best, n)] += steps[pair(region, n)]
                shared[pair(best, n)] = (
                    shared.get(pair(best, n), 0.0) + shared[pair(region, n)]
                )
                shared_edges[pair(best, n)] += shared_edges[pair(region, n)]
                if pair(region, n) in locked:
                    locked.add(pair(best, n))
                neighbors[n].add(best)
                neighbors[best].add(n)
            shared.pop(pair(region, n), None)
            turns.pop(pair(region, n), None)
            steps.pop(pair(region, n), None)
            spread.pop(pair(region, n), None)
            shared_edges.pop(pair(region, n), None)
            locked.discard(pair(region, n))
            neighbors[n].discard(region)
        neighbors[best].discard(best)
        del neighbors[region]
        alive.discard(region)
        parent[region] = best
        new_width = width(best)
        if new_width < min_width:
            heapq.heappush(heap, (new_width, best))

    bounds = Boundaries(turns, steps, spread, shared)
    return {i: find(parent, r) for i, r in label.items()}, bounds


def merge_smooth(edges, label, bounds, min_width, angle=CREASE_ANGLE, forced=None):
    """Merge neighbours whose shared boundary is not a crease.

    A crease is a turn with no width: every boundary on a cylinder wall
    turns one segment angle however much has merged, while a corner turns
    the whole corner. The turn absorb carried over counts only while its
    spread stays band-narrow, past that the boundary is judged by its own
    edges alone. Least turn first, lazy heap.
    """
    turns, steps, spread, shared = bounds
    neighbors = collections.defaultdict(set)
    for ra, rb in shared:
        neighbors[ra].add(rb)
        neighbors[rb].add(ra)
    ec, rverts, shared_edges = region_topology(edges, label)
    locked = locked_pairs(edges, label, forced)
    alive = set(label.values())
    parent = {r: r for r in alive}

    def crease(a, b):
        """How sharply the surface turns crossing this boundary: its own edges
        always count, a dissolved band's carry only while it is still band
        width.
        """
        key = pair(a, b)
        length = shared[key]
        if length <= 0:
            return math.inf
        step = steps[key] / length
        if spread[key] / length > min_width:
            return step
        return max(step, turns[key] / length)

    heap = [
        (crease(a, b), a, b)
        for a in neighbors
        for b in neighbors[a]
        if a < b and crease(a, b) < angle
    ]
    heapq.heapify(heap)

    while heap:
        value, a, b = heapq.heappop(heap)
        if (
            a not in alive
            or b not in alive
            or crease(a, b) != value
            or pair(a, b) in locked
            or not keeps_topology(ec, rverts, shared_edges, a, b)
        ):
            continue
        ec[a] += ec[b] - joint_count(rverts, shared_edges, a, b)
        rverts[a] |= rverts[b]
        for n in list(neighbors[b]):
            if n != a:
                # both sides of this boundary stay put, so no turn is crossed
                turns[pair(a, n)] += turns[pair(b, n)]
                steps[pair(a, n)] += steps[pair(b, n)]
                spread[pair(a, n)] += spread[pair(b, n)]
                shared[pair(a, n)] += shared[pair(b, n)]
                shared_edges[pair(a, n)] += shared_edges[pair(b, n)]
                if pair(b, n) in locked:
                    locked.add(pair(a, n))
                neighbors[n].add(a)
                neighbors[a].add(n)
            turns.pop(pair(b, n), None)
            steps.pop(pair(b, n), None)
            spread.pop(pair(b, n), None)
            shared.pop(pair(b, n), None)
            shared_edges.pop(pair(b, n), None)
            locked.discard(pair(b, n))
            neighbors[n].discard(b)
        neighbors[a].discard(a)
        del neighbors[b]
        alive.discard(b)
        parent[b] = a
        for n in neighbors[a]:
            value = crease(a, n)
            if value < angle:
                heapq.heappush(heap, (value, *((a, n) if a < n else (n, a))))

    return {i: find(parent, r) for i, r in label.items()}


def merge_flat(weighted, areas, edges, label, angle=FLAT_ANGLE, forced=None):
    """Merge adjacent regions while their union stays nearly flat, flattest
    pair first, with the same lazy heap as absorb.
    """
    total = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    mass = collections.defaultdict(float)
    for i, r in label.items():
        for k in range(3):
            total[r][k] += weighted[i][k]
        mass[r] += 2 * areas[i]

    neighbors = collections.defaultdict(set)
    for owners in edges.values():
        if len(owners) == 2:
            ra, rb = label[owners[0]], label[owners[1]]
            if ra != rb:
                neighbors[ra].add(rb)
                neighbors[rb].add(ra)

    def ratio(a, b):
        m = mass[a] + mass[b]
        s = [total[a][k] + total[b][k] for k in range(3)]
        return norm(s) / m if m > 0 else 1.0

    bound = (1 + math.cos(math.radians(angle))) / 2
    ec, rverts, shared_edges = region_topology(edges, label)
    locked = locked_pairs(edges, label, forced)
    alive = set(total)
    parent = {r: r for r in total}
    heap = []
    for a in neighbors:
        for b in neighbors[a]:
            if a < b and ratio(a, b) >= bound:
                heap.append((-ratio(a, b), a, b))
    heapq.heapify(heap)

    while heap:
        negr, a, b = heapq.heappop(heap)
        if (
            a not in alive
            or b not in alive
            or ratio(a, b) != -negr
            or pair(a, b) in locked
            or not keeps_topology(ec, rverts, shared_edges, a, b)
        ):
            continue
        for k in range(3):
            total[a][k] += total[b][k]
        mass[a] += mass[b]
        ec[a] += ec[b] - joint_count(rverts, shared_edges, a, b)
        rverts[a] |= rverts[b]
        for n in neighbors[b]:
            if n != a:
                shared_edges[pair(a, n)] += shared_edges[pair(b, n)]
                if pair(b, n) in locked:
                    locked.add(pair(a, n))
                neighbors[n].add(a)
                neighbors[a].add(n)
            shared_edges.pop(pair(b, n), None)
            locked.discard(pair(b, n))
            neighbors[n].discard(b)
        neighbors[a].discard(a)
        del neighbors[b]
        alive.discard(b)
        parent[b] = a
        for n in neighbors[a]:
            r = ratio(a, n)
            if r >= bound:
                heapq.heappush(heap, (-r, *((a, n) if a < n else (n, a))))

    return {i: find(parent, r) for i, r in label.items()}


def close_rings(verts, weighted, areas, edges, label, angle=CREASE_ANGLE, forced=None):
    """Merge disk pairs whose union is a short annulus, for one cut not two.

    keeps_topology leaves a coarse tube wall as two half shells with two
    seams where a fine one gets a single disk_cuts seam. Closing the ring
    is safe exactly when the cut-open wall unrolls into a compact strip,
    and the boundary must be smooth at its own edges, a lid meeting a
    channel twice turns a corner there and keeps both seams.
    """
    area, shared, perimeter = region_stats(verts, areas, edges, label)
    turns = boundary_turns(verts, weighted, edges, label)
    ec, rverts, shared_edges = region_topology(edges, label)
    locked = locked_pairs(edges, label, forced)

    candidates = []
    for (a, b), length in shared.items():
        if ec[a] != 1 or ec[b] != 1:
            continue
        if (a, b) in locked:
            continue
        if joint_count(rverts, shared_edges, a, b) != 2:
            continue
        if length <= 0 or turns[(a, b)] / length >= angle:
            continue
        strip = (perimeter[a] + perimeter[b]) / 2 - length
        size = area[a] + area[b]
        if size <= 0 or strip * strip / size > SPLIT_ASPECT:
            continue
        candidates.append((strip * strip / size, a, b))

    taken = set()
    closed = {}
    for _, a, b in sorted(candidates):
        if a in taken or b in taken:
            continue
        taken.update((a, b))
        closed[b] = a
    if not closed:
        return label
    return {i: closed.get(r, r) for i, r in label.items()}


def panel_share(weighted, groups):
    """How much of these faces' mass sums to flat panels: 1 when every group
    is planar, lower the more each one curls."""
    flat = mass = 0.0
    for group in groups:
        resultant = [0.0, 0.0, 0.0]
        for i in group:
            for k in range(3):
                resultant[k] += weighted[i][k]
            mass += norm(weighted[i])
        flat += norm(resultant)
    return flat / mass if mass > 0 else 0.0


def straight_path(verts, keys):
    """Whether these edges form one connected path along one line. Rigid
    unfolding needs exactly that: the sides swing about a single fold axis,
    a bent or split contact cannot open flat."""
    reference = None
    ends = collections.Counter()
    parent = {}
    for v0, v1 in keys:
        step = [verts[v1][k] - verts[v0][k] for k in range(3)]
        length = norm(step)
        if length <= 0:
            return False
        step = [x / length for x in step]
        if reference is None:
            reference = step
        elif abs(sum(x * y for x, y in zip(reference, step))) < HINGE_LINE_COS:
            return False
        ends.update((v0, v1))
        parent.setdefault(v0, v0)
        parent.setdefault(v1, v1)
        ra, rb = find(parent, v0), find(parent, v1)
        if ra != rb:
            parent[ra] = rb
    if any(count > 2 for count in ends.values()):
        return False
    return len({find(parent, v) for v in parent}) == 1


def path_ends(keys):
    """The two endpoint vertices of a connected open path of edges."""
    counts = collections.Counter(v for key in keys for v in key)
    ends = [v for v, count in counts.items() if count == 1]
    return ends if len(ends) == 2 else None


def panel_basis(verts, faces, weighted, members):
    """Origin and in-plane axes projecting a flat panel to 2D, or None when
    the panel has no usable normal."""
    normal = [0.0, 0.0, 0.0]
    for i in members:
        for k in range(3):
            normal[k] += weighted[i][k]
    scale = norm(normal)
    if scale <= 0:
        return None
    normal = [x / scale for x in normal]
    face = faces[members[0]]
    origin = verts[face[0]]
    for other in face[1:]:
        direction = [verts[other][k] - origin[k] for k in range(3)]
        lift = sum(d * n for d, n in zip(direction, normal))
        axis_u = [d - lift * n for d, n in zip(direction, normal)]
        length = norm(axis_u)
        if length > 0:
            axis_u = [x / length for x in axis_u]
            return origin, axis_u, cross(normal, axis_u)
    return None


def project(basis, vert):
    origin, axis_u, axis_v = basis
    d = [vert[k] - origin[k] for k in range(3)]
    return (
        sum(x * y for x, y in zip(d, axis_u)),
        sum(x * y for x, y in zip(d, axis_v)),
    )


IDENTITY = (1.0, 0.0, 0.0, 0.0)


def apply_transform(transform, point):
    c, s, x, y = transform
    return (c * point[0] - s * point[1] + x, s * point[0] + c * point[1] + y)


def compose_transforms(outer, inner):
    c2, s2, x2, y2 = outer
    c1, s1, x1, y1 = inner
    return (
        c2 * c1 - s2 * s1,
        s2 * c1 + c2 * s1,
        c2 * x1 - s2 * y1 + x2,
        s2 * x1 + c2 * y1 + y2,
    )


def glue_transform(a0, a1, b0, b1):
    """Rigid 2D map taking segment b0-b1 onto a0-a1, rotation only: both
    panels project with their normal up, so an unfold never mirrors."""
    dax, day = a1[0] - a0[0], a1[1] - a0[1]
    dbx, dby = b1[0] - b0[0], b1[1] - b0[1]
    la, lb = math.hypot(dax, day), math.hypot(dbx, dby)
    if la <= 0 or lb <= 0:
        return None
    dax, day, dbx, dby = dax / la, day / la, dbx / lb, dby / lb
    c = dbx * dax + dby * day
    s = dbx * day - dby * dax
    return (
        c,
        s,
        a0[0] - (c * b0[0] - s * b0[1]),
        a0[1] - (s * b0[0] + c * b0[1]),
    )


def segments_cross(a, b, c, d):
    """Whether the two segments properly cross. Touching at an endpoint or
    running collinear is not a crossing, so glued neighbours sharing their
    hinge line stay clean."""

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2 = orient(a, b, c), orient(a, b, d)
    if o1 == 0 or o2 == 0 or (o1 > 0) == (o2 > 0):
        return False
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return o3 != 0 and o4 != 0 and (o3 > 0) != (o4 > 0)


def point_in_polygon(point, segments):
    """Ray parity over an unordered boundary, the half-open rule so a vertex
    on the ray counts once."""
    px, py = point
    hits = 0
    for (x0, y0), (x1, y1) in segments:
        if (y0 > py) != (y1 > py) and x0 + (py - y0) * (x1 - x0) / (y1 - y0) > px:
            hits += 1
    return hits % 2 == 1


def _boxes_meet(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _span(boxes):
    """The one box covering all of them."""
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def panels_overlap(placed_a, placed_b):
    """Whether two placed panels share area. The caller box-rejects first."""
    _, segs_a, _, inner_a = placed_a
    _, segs_b, _, inner_b = placed_b
    for sa in segs_a:
        for sb in segs_b:
            if segments_cross(sa[0], sa[1], sb[0], sb[1]):
                return True
    return point_in_polygon(inner_a, segs_b) or point_in_polygon(inner_b, segs_a)


def unfold_hinges(verts, faces, weighted, edges, label, forced=None):
    """Boundary edges to leave uncut so flat panels unfold as one island,
    the way an artist opens a box into a cross.

    A hinge is the contact between two flat panels when it is one straight
    path: the sides then swing rigidly open like paper and the flatten
    stays isometric. Hinges are picked longest first into a spanning
    forest over the regions, each glue along one path keeps every island a
    disk, and a contact holding a forced seam never hinges. Each candidate
    is test-placed in the flat net first, and one whose side would land on
    panels already placed is dropped, so the boundary ships as a seam
    there instead of an overlapping net the engine recuts as one blob."""
    # forced cuts here too, or a panel spans two regions and the forest
    # cannot tell which one a contact connects
    root = partition(faces, weighted, edges, LOW_ANGLE, forced)
    contacts = collections.defaultdict(list)
    outline = collections.defaultdict(list)
    poisoned = set()
    for key, owners in edges.items():
        panels = sorted({root(o) for o in owners})
        if len(owners) == 2 and len(panels) == 1:
            continue
        for p in panels:
            outline[p].append(key)
        if len(owners) == 2:
            panel_pair = (panels[0], panels[1])
            contacts[panel_pair].append(key)
            if forced and key in forced:
                poisoned.add(panel_pair)

    members = collections.defaultdict(list)
    for i in range(len(faces)):
        members[root(i)].append(i)
    flat = {
        p: panel_share(weighted, [members[p]]) >= PANEL_SHARE
        for p in dict.fromkeys(p for panel_pair in contacts for p in panel_pair)
    }
    ec, _, _ = region_topology(edges, label)
    region = {p: label[members[p][0]] for p in flat}
    basis = {
        p: panel_basis(verts, faces, weighted, members[p])
        for p, is_flat in flat.items()
        if is_flat
    }

    def length(keys):
        return sum(
            norm([verts[v1][k] - verts[v0][k] for k in range(3)]) for v0, v1 in keys
        )

    candidates = sorted(
        (
            (length(keys), panel_pair)
            for panel_pair, keys in contacts.items()
            if region[panel_pair[0]] != region[panel_pair[1]]
            and panel_pair not in poisoned
            and all(flat[p] for p in panel_pair)
            and all(ec[region[p]] == 1 for p in panel_pair)
            and straight_path(verts, keys)
        ),
        reverse=True,
    )

    local_cache = {}

    def placed_panel(move, p):
        if p not in local_cache:
            face = faces[members[p][0]]
            centroid = [sum(verts[v][k] for v in face) / len(face) for k in range(3)]
            local_cache[p] = (
                [
                    (project(basis[p], verts[v0]), project(basis[p], verts[v1]))
                    for v0, v1 in outline[p]
                ],
                project(basis[p], centroid),
            )
        segments, inner = local_cache[p]
        moved = [
            (apply_transform(move, a), apply_transform(move, b)) for a, b in segments
        ]
        xs = [c[0] for seg in moved for c in seg]
        ys = [c[1] for seg in moved for c in seg]
        box = (min(xs), min(ys), max(xs), max(ys))
        return move, moved, box, apply_transform(move, inner)

    # pre-glue panels within a region across straight flat contacts: those
    # folds are never cut
    cluster = {}
    placement = {}
    glue_adjacency = collections.defaultdict(list)
    for panel_pair, keys in contacts.items():
        pa, pb = panel_pair
        if (
            pa in region
            and region[pa] == region[pb]
            and panel_pair not in poisoned
            and flat[pa]
            and flat[pb]
            and basis.get(pa)
            and basis.get(pb)
            and straight_path(verts, keys)
            and path_ends(keys)
        ):
            glue_adjacency[pa].append((pb, keys))
            glue_adjacency[pb].append((pa, keys))
    for p in basis:
        if basis[p] is None or p in placement:
            continue
        cluster[p] = p
        placement[p] = IDENTITY
        stack = [p]
        while stack:
            current = stack.pop()
            for neighbor, keys in glue_adjacency[current]:
                if neighbor in placement:
                    continue
                e0, e1 = path_ends(keys)
                move = glue_transform(
                    apply_transform(
                        placement[current], project(basis[current], verts[e0])
                    ),
                    apply_transform(
                        placement[current], project(basis[current], verts[e1])
                    ),
                    project(basis[neighbor], verts[e0]),
                    project(basis[neighbor], verts[e1]),
                )
                if move is None:
                    continue
                cluster[neighbor] = p
                placement[neighbor] = move
                stack.append(neighbor)

    # clusters merged so far, each with its panels laid out in one shared
    # frame. only panels inside a group can be overlap-tested
    group_parent = {}
    layouts = {}

    def group_layout(p):
        seed = cluster[p]
        group_parent.setdefault(seed, seed)
        top = find(group_parent, seed)
        if top not in layouts:
            layouts[top] = {
                q: placed_panel(placement[q], q)
                for q, s in cluster.items()
                if s == seed
            }
        return top

    parent = {r: r for r in set(label.values())}
    hinges = set()
    for _, panel_pair in candidates:
        pa, pb = panel_pair
        ra, rb = find(parent, region[pa]), find(parent, region[pb])
        if ra == rb:
            continue
        ends = path_ends(contacts[panel_pair])
        if ends and basis.get(pa) and basis.get(pb):
            ga, gb = group_layout(pa), group_layout(pb)
            if ga != gb:
                side_a = layouts[ga][pa]
                side_b = layouts[gb][pb]
                e0, e1 = ends
                move = glue_transform(
                    apply_transform(side_a[0], project(basis[pa], verts[e0])),
                    apply_transform(side_a[0], project(basis[pa], verts[e1])),
                    apply_transform(side_b[0], project(basis[pb], verts[e0])),
                    apply_transform(side_b[0], project(basis[pb], verts[e1])),
                )
                if move is not None:
                    arriving = {
                        q: placed_panel(compose_transforms(move, placed[0]), q)
                        for q, placed in layouts[gb].items()
                    }
                    # one box for the arrivals drops a panel clear of them all
                    span = _span([placed[2] for placed in arriving.values()])
                    if any(
                        panels_overlap(placed, arrival)
                        for placed in layouts[ga].values()
                        if _boxes_meet(placed[2], span)
                        for arrival in arriving.values()
                        if _boxes_meet(placed[2], arrival[2])
                    ):
                        continue
                    layouts[ga].update(arriving)
                    del layouts[gb]
                    group_parent[gb] = ga
        parent[ra] = rb
        hinges.update(contacts[panel_pair])
    return hinges
