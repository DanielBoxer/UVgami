"""Boundary targets that straighten near-rectangular islands.

The solve itself stays in the addon: the caller pins these targets and runs
a pinned unwrap over the island interior. Everything walks uv points, not
mesh vertices: an island bordering its own cut carries two uvs on each cut
vertex and the slit is part of the boundary."""

import bisect
import itertools
import math

from .mesh import signed_area

# uv area over fitted rectangle area an island needs to rectify. a circle
# fills 0.785 of its square, so blobs stay under the gate
RECTANGLE_SHARE = 0.8
# boundary length squared over circle length squared at equal area. strips
# measure 3 and up, blobs 1 to 2.5, so this admits the wavy and curled
# strips the share gate misses
STRIP_ELONGATION = 3.0
# corner picking by boundary turning: a corner concentrates about 90
# degrees inside a window this share of the perimeter, a bend spreads its
# turn thin and never reaches the floor. the window must stay under a
# slender strip's short side or one end's corners crowd each other out
CORNER_WINDOW = 0.02
CORNER_TURN = 45
# opposite sides of a real strip match in arc length, and the rectangle
# the corners cut must hold about the island's own area. picks that break
# either caught a tooth or a jag, not a corner
CORNER_SIDE_RATIO = 2.0
CORNER_FIT_AREA = 1.6
# corner windows tried against the fit, so a jagged edge full of sharp
# turns cannot crowd out a strip's real end corners
CORNER_CANDIDATES = 12
SPINE_SAMPLES = 64
# rings around a flipped face that move to the neighbor average, and the cap
# on repeats
RELAX_RING = 2
RELAX_ROUNDS = 200


def island_area(group, uvs):
    return abs(sum(signed_area(uvs[fi]) for fi in group))


# uv areas under this fraction of the island count as degenerate in the
# distortion measure: a boundary triangle pinned collinear reads as flipped
# at float noise, a real flip is orders of magnitude bigger
FLIP_NOISE = 1e-6


def flatten_distortion(verts, faces, uvs, group, uv_areas=None):
    """Scale-free symmetric Dirichlet of the island's map, 4.0 at isometry.
    A face flipped against the island's own orientation, above noise scale,
    is infinity: the map is broken there, not just stretched. A mirrored
    island measures like its source, orientation is the island's majority.
    uv_areas are the per-face signed uv areas, when the caller has them."""
    if uv_areas is None:
        signed_total = sum(signed_area(uvs[fi]) for fi in group)
    else:
        signed_total = sum(uv_areas[fi] for fi in group)
    orientation = 1.0 if signed_total >= 0 else -1.0
    floor = FLIP_NOISE * abs(signed_total)
    grow = shrink = total = 0.0
    for fi in group:
        face = faces[fi]
        face_uv = uvs[fi]
        p0 = verts[face[0]]
        u0 = face_uv[0]
        for i in range(1, len(face) - 1):
            p1 = verts[face[i]]
            p2 = verts[face[i + 1]]
            e1x, e1y, e1z = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
            e2x, e2y, e2z = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
            length = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
            sx = e1y * e2z - e1z * e2y
            sy = e1z * e2x - e1x * e2z
            sz = e1x * e2y - e1y * e2x
            area = math.sqrt(sx * sx + sy * sy + sz * sz) / 2
            u1 = face_uv[i]
            u2 = face_uv[i + 1]
            uv_area = (
                (
                    (u0[0] * u1[1] - u1[0] * u0[1])
                    + (u1[0] * u2[1] - u2[0] * u1[1])
                    + (u2[0] * u0[1] - u0[0] * u2[1])
                )
                / 2
                * orientation
            )
            if length <= 0 or area <= 0 or abs(uv_area) <= floor:
                continue
            if uv_area < 0:
                return math.inf
            x2 = (e1x * e2x + e1y * e2y + e1z * e2z) / length
            y2 = 2 * area / length
            a = (u1[0] - u0[0]) * orientation / length
            b = ((u2[0] - u0[0]) * orientation - x2 * a) / y2
            c = (u1[1] - u0[1]) / length
            d = (u2[1] - u0[1] - x2 * c) / y2
            det = a * d - b * c
            if det <= 0:
                return math.inf
            frob2 = a * a + b * b + c * c + d * d
            grow += area * frob2
            shrink += area * frob2 / (det * det)
            total += area
    if total <= 0:
        return math.inf
    return 2 * math.sqrt(grow * shrink) / total


def _boundary_loop(group, uvs):
    """The island's single boundary loop as uv points in walk order. None
    when the boundary splits into several loops (holes) or pinches through
    a point."""
    counts = {}
    for fi in group:
        face = uvs[fi]
        for i in range(len(face)):
            a, b = face[i], face[(i + 1) % len(face)]
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    boundary = [edge for edge, count in counts.items() if count == 1]
    if len(boundary) < 4:
        return None

    neighbors = {}
    for a, b in boundary:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    if any(len(around) != 2 for around in neighbors.values()):
        return None

    start = boundary[0][0]
    loop = [start]
    previous, current = None, start
    while True:
        a, b = neighbors[current]
        following = b if a == previous else a
        if following == start:
            break
        loop.append(following)
        previous, current = current, following
    if len(loop) != len(neighbors):
        return None
    return loop


def _arc_lengths(points, start, stop):
    """Cumulative distance along points from index start to index stop,
    cyclic, one entry per point of the segment including both ends."""
    lengths = [0.0]
    i = start
    while i != stop:
        after = (i + 1) % len(points)
        lengths.append(lengths[-1] + math.dist(points[i], points[after]))
        i = after
    return lengths


def _turning_corners(points, area):
    """Four boundary indices where turning concentrates, in loop order, None
    when four clear corners do not stand out. Nearest-to-box picking twists
    a curled strip, its outer bulge sits closer to the box corner than the
    strip's real end does, so corners are read from the boundary itself: a
    corner keeps its turn sharp, a bend spreads it, and a tooth's zigzag
    cancels inside the window."""
    n = len(points)
    if n < 4:
        return None
    positions = []
    total = 0.0
    for i in range(n):
        positions.append(total)
        total += math.dist(points[i], points[(i + 1) % n])
    if total <= 0:
        return None
    turns = []
    for i in range(n):
        before, here, after = points[i - 1], points[i], points[(i + 1) % n]
        v0 = (here[0] - before[0], here[1] - before[1])
        v1 = (after[0] - here[0], after[1] - here[1])
        turns.append(
            math.atan2(v0[0] * v1[1] - v0[1] * v1[0], v0[0] * v1[0] + v0[1] * v1[1])
        )
    window = total * CORNER_WINDOW
    doubled = positions + [p + total for p in positions]
    prefix = [0.0]
    for i in range(2 * n):
        prefix.append(prefix[-1] + turns[i % n])

    def window_turn(i):
        center = positions[i] + total
        lo = bisect.bisect_left(doubled, center - window / 2)
        hi = bisect.bisect_right(doubled, center + window / 2)
        return prefix[hi] - prefix[lo]

    floor = math.radians(CORNER_TURN)
    candidates = []
    for score, i in sorted(((window_turn(i), i) for i in range(n)), reverse=True):
        if score < floor or len(candidates) == CORNER_CANDIDATES:
            break
        apart = all(
            min(
                (positions[i] - positions[j]) % total,
                (positions[j] - positions[i]) % total,
            )
            >= window
            for j in candidates
        )
        if apart:
            candidates.append(i)
    if len(candidates) < 4:
        return None

    # take the four candidates that partition the loop into a rectangle
    # fitting the island, matching area and opposite sides. picking the
    # sharpest four instead lets a jagged seam edge outscore a real corner
    best, best_fit = None, math.inf
    for combo in itertools.combinations(sorted(candidates), 4):
        arcs = [
            (positions[combo[(s + 1) % 4]] - positions[combo[s]]) % total
            for s in range(4)
        ]
        if any(arc <= 0 for arc in arcs):
            continue
        rectangle = (arcs[0] + arcs[2]) / 2 * (arcs[1] + arcs[3]) / 2
        ratio_across = max(arcs[0], arcs[2]) / min(arcs[0], arcs[2])
        ratio_along = max(arcs[1], arcs[3]) / min(arcs[1], arcs[3])
        if (
            not area / CORNER_FIT_AREA <= rectangle <= area * CORNER_FIT_AREA
            or ratio_across > CORNER_SIDE_RATIO
            or ratio_along > CORNER_SIDE_RATIO
        ):
            continue
        fit = (
            abs(math.log(rectangle / area))
            + math.log(ratio_across)
            + math.log(ratio_along)
        )
        if fit < best_fit:
            best, best_fit = list(combo), fit
    return best


def _spine_targets(rotated, picks, sides, width, queries):
    """Straightened positions for uv points of a strip: each keeps its
    fraction along the strip's spine and its signed offset across it, so the
    strip unbends but keeps its own width profile. A direct placement
    because blender's unwrap reinitializes from scratch, so a pinned solve
    cannot unbend a deep curl without folding, however the pins are
    staged."""
    n = len(rotated)

    def along(side, lengths, fraction):
        distance = fraction * lengths[-1]
        k = bisect.bisect_right(lengths, distance) - 1
        if k >= len(lengths) - 1:
            return rotated[side[-1]]
        a, b = rotated[side[k]], rotated[side[k + 1]]
        span = lengths[k + 1] - lengths[k]
        t = (distance - lengths[k]) / span if span > 0 else 0.0
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def walk(a, b):
        out = [a]
        i = a
        while i != b:
            i = (i + 1) % n
            out.append(i)
        return out

    side0 = walk(picks[0], picks[1])
    side2 = walk(picks[2], picks[3])
    spine = []
    for k in range(SPINE_SAMPLES + 1):
        fraction = k / SPINE_SAMPLES
        p0 = along(side0, sides[0], fraction)
        # the far side walks the loop backward relative to the near one
        p2 = along(side2, sides[2], 1.0 - fraction)
        spine.append(((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2))

    placed = []
    for q in queries:
        best, best_k = math.inf, 0
        for k, s in enumerate(spine):
            d = (q[0] - s[0]) ** 2 + (q[1] - s[1]) ** 2
            if d < best:
                best, best_k = d, k
        # project onto the straighter of the two segments at the sample
        fraction = best_k / SPINE_SAMPLES
        offset = 0.0
        for k in (best_k - 1, best_k):
            if not 0 <= k < SPINE_SAMPLES:
                continue
            a, b = spine[k], spine[k + 1]
            dx, dy = b[0] - a[0], b[1] - a[1]
            span2 = dx * dx + dy * dy
            if span2 <= 0:
                continue
            t = ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / span2
            if 0.0 <= t <= 1.0:
                fraction = (k + t) / SPINE_SAMPLES
                offset = ((q[0] - a[0]) * dy - (q[1] - a[1]) * dx) / math.sqrt(span2)
                break
        else:
            a = spine[max(best_k - 1, 0)]
            b = spine[min(best_k + 1, SPINE_SAMPLES)]
            dx, dy = b[0] - a[0], b[1] - a[1]
            span = math.hypot(dx, dy)
            if span > 0:
                offset = ((q[0] - a[0]) * dy - (q[1] - a[1]) * dx) / span
        x = -width / 2 + fraction * width
        placed.append((x, -offset))
    return placed


def _rectangle_targets(loop, area, interior=None):
    points = loop
    if signed_area(points) < 0:
        points = points[::-1]

    perimeter = sum(
        math.dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points))
    )
    elongation = perimeter**2 / (4 * math.pi * area)

    mean_x = sum(p[0] for p in points) / len(points)
    mean_y = sum(p[1] for p in points) / len(points)
    centered = [(x - mean_x, y - mean_y) for x, y in points]
    xx = sum(x * x for x, y in centered)
    xy = sum(x * y for x, y in centered)
    yy = sum(y * y for x, y in centered)
    angle = 0.5 * math.atan2(2 * xy, xx - yy)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotated = [(x * cos_a + y * sin_a, y * cos_a - x * sin_a) for x, y in centered]

    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    box_width = max(xs) - min(xs)
    box_height = max(ys) - min(ys)
    if box_width <= 0 or box_height <= 0:
        return None
    share = area / (box_width * box_height)
    if share < RECTANGLE_SHARE and elongation < STRIP_ELONGATION:
        return None

    def sides_for(picks):
        if picks is None or len(set(picks)) != 4:
            return None
        offsets = [(k - picks[0]) % len(points) for k in picks]
        if not offsets[1] < offsets[2] < offsets[3]:
            return None
        sides = [_arc_lengths(rotated, picks[s], picks[(s + 1) % 4]) for s in range(4)]
        if any(side[-1] == 0 for side in sides):
            return None
        return sides

    picks = _turning_corners(points, area)
    sides = sides_for(picks)
    if sides is not None and sides[0][-1] + sides[2][-1] < sides[1][-1] + sides[3][-1]:
        # the longer side pair maps onto the rectangle's width along the
        # fitted axis, so the island does not land turned a quarter over
        picks = picks[1:] + picks[:1]
        sides = sides[1:] + sides[:1]
    turned = sides is not None
    if sides is None:
        box_corners = [
            (min(xs), min(ys)),
            (max(xs), min(ys)),
            (max(xs), max(ys)),
            (min(xs), max(ys)),
        ]
        picks = []
        for corner_x, corner_y in box_corners:
            picks.append(
                min(
                    range(len(rotated)),
                    key=lambda i: (
                        ((rotated[i][0] - corner_x) / box_width) ** 2
                        + ((rotated[i][1] - corner_y) / box_height) ** 2
                    ),
                )
            )
        sides = sides_for(picks)
        if sides is None:
            return None
    # side lengths come from the boundary, not the box. a strip that still
    # curls unrolls to its real length, which the box undershoots
    width = (sides[0][-1] + sides[2][-1]) / 2
    height = (sides[1][-1] + sides[3][-1]) / 2
    rectangle = [
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ]

    def restore(x, y):
        return (
            x * cos_a - y * sin_a + mean_x,
            x * sin_a + y * cos_a + mean_y,
        )

    if turned and interior:
        # the boundary shares the spine mapping: arc length instead forces a
        # constant width, stretching a tapered strip past the distortion gate
        rotated_interior = [
            (
                (q[0] - mean_x) * cos_a + (q[1] - mean_y) * sin_a,
                (q[1] - mean_y) * cos_a - (q[0] - mean_x) * sin_a,
            )
            for q in interior
        ]
        placed = _spine_targets(
            rotated, picks, sides, width, rotated + rotated_interior
        )
        targets = {point: restore(x, y) for point, (x, y) in zip(points, placed)}
        inner = {q: restore(x, y) for q, (x, y) in zip(interior, placed[len(points) :])}
        return targets, inner

    targets = {}
    for s in range(4):
        ax, ay = rectangle[s]
        bx, by = rectangle[(s + 1) % 4]
        lengths = sides[s]
        for step, distance in enumerate(lengths[:-1]):
            t = distance / lengths[-1]
            x = ax + (bx - ax) * t
            y = ay + (by - ay) * t
            targets[points[(picks[s] + step) % len(points)]] = restore(x, y)
    return targets, None


def _relax_flips(group, uvs, targets, inner):
    """Untangle the triangles a direct placement flipped, moving their uv
    points and a ring around them onto the average of their neighbors. The
    spine projection can jump between samples where the strip wiggles, and one
    flipped sliver would revert the whole island at the distortion gate.
    Updates targets and inner in place."""
    position = dict(targets)
    position.update(inner)
    neighbors = {}
    faces_at = {}
    for fi in group:
        face = uvs[fi]
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
            faces_at.setdefault(a, set()).add(fi)

    def placed(fi):
        return [position.get(uv, uv) for uv in uvs[fi]]

    def flipped_face(fi):
        pts = placed(fi)
        return any(
            signed_area([pts[0], pts[i], pts[i + 1]]) * orientation < -floor
            for i in range(1, len(pts) - 1)
        )

    total = sum(signed_area(placed(fi)) for fi in group)
    orientation = 1.0 if total >= 0 else -1.0
    floor = FLIP_NOISE * abs(total)
    # only a face touching a moved uv can change. iterating group, not the
    # candidate set, keeps the order fixed
    candidates = set(group)
    for _ in range(RELAX_ROUNDS):
        flipped = [fi for fi in group if fi in candidates and flipped_face(fi)]
        if not flipped:
            break
        free = {uv for fi in flipped for uv in uvs[fi]}
        for _ in range(RELAX_RING):
            free |= {other for uv in free for other in neighbors[uv]}
        for uv in free:
            around = neighbors[uv]
            position[uv] = (
                sum(position.get(o, o)[0] for o in around) / len(around),
                sum(position.get(o, o)[1] for o in around) / len(around),
            )
        candidates = {fi for uv in free for fi in faces_at[uv]}
    for uv, p in position.items():
        if uv in inner:
            inner[uv] = p
        else:
            targets[uv] = p


def rectify_targets(uvs, groups):
    """Per qualifying island, the faces, the boundary targets, and the
    interior positions. Corners found by boundary turning make a strip: every
    uv is placed directly by its spine coordinates, so no solve is needed.
    Without four turning corners the boundary maps onto the fitted rectangle
    by arc length (corners nearest the bounding box) and the interior is None,
    for the pinned unwrap."""
    plans = []
    for group in groups:
        loop = _boundary_loop(group, uvs)
        if loop is None:
            continue
        area = island_area(group, uvs)
        if area <= 0:
            continue
        interior = list({uv for fi in group for uv in uvs[fi]} - set(loop))
        result = _rectangle_targets(loop, area, interior)
        if result is not None:
            targets, inner = result
            if inner is not None:
                _relax_flips(group, uvs, targets, inner)
            plans.append((group, targets, inner))
    return plans
