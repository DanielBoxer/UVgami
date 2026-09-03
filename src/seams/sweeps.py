import bisect
import collections
import math

from .islands import SPLIT_ASPECT
from .mesh import build, cross, face_keys, find, norm
from .regions import CREASE_ANGLE, partition

# a measured elbow fills 0.31 of the middle band, a screwdriver handle 0.07
SWEEP_BAND = 0.1
SWEEP_CAP_MIN = 0.02
# cutting every screw and pin explodes the chart count
SWEEP_MIN_SHARE = 0.01
# an axis fit over fewer faces than this is noise
SWEEP_MIN_FACES = 8
# the normals' resultant over the mass, 1 on a plate, 0.7 asks for a half turn
WALL_ROUND = 0.7
# two walls sweeping the same axis need no rim, a grooved ring unrolls together
SHARED_AXIS_COS = 0.95
# a flat side spikes the normal-direction histogram, a round profile stays flat
PROFILE_BINS = 60
PROFILE_PEAK = 3.0
PROFILE_FLAT = 2.0
# a coarse round tube spikes too, but between its facet spikes is empty
PROFILE_CORNER = 0.02
# the flat sides must face different ways, a multi-bump shell is one panel
PROFILE_TURN = 45
# how finely a wall with a handle through it is trimmed back along its axis
GENUS_TRIM_LEVELS = 24
# wall/cap boundary and the band edges, as squared sines of the tilt
CAP_SPLIT = 0.5  # 45 degrees
BAND_LO = 0.25  # 30 degrees
BAND_HI = 0.75  # 60 degrees
SWEEP_FIT_ROUNDS = 10
# in faces, not growth rings, a spiral-strip triangulation grows one face a ring
RUN_SEED_FACES = 512
# 0.2 admits a straight tube about four diameters long
RUN_SLENDER = 0.2
# growth rings searched behind a run cut for a concave boundary to snap to
VALLEY_SNAP_LAYERS = 6


# the cross of the two most independent rows of the shifted matrix
def min_eigenvector(xx, yy, zz, xy, xz, yz):
    p1 = xy * xy + xz * xz + yz * yz
    if p1 == 0:
        lam = min(xx, yy, zz)
    else:
        q = (xx + yy + zz) / 3
        p2 = (xx - q) ** 2 + (yy - q) ** 2 + (zz - q) ** 2 + 2 * p1
        p = math.sqrt(p2 / 6)
        bxx, byy, bzz = (xx - q) / p, (yy - q) / p, (zz - q) / p
        bxy, bxz, byz = xy / p, xz / p, yz / p
        det = (
            bxx * (byy * bzz - byz * byz)
            - bxy * (bxy * bzz - byz * bxz)
            + bxz * (bxy * byz - byy * bxz)
        )
        r = max(-1.0, min(1.0, det / 2))
        phi = math.acos(r) / 3
        lam = q + 2 * p * math.cos(phi + 2 * math.pi / 3)
    rows = [(xx - lam, xy, xz), (xy, yy - lam, yz), (xz, yz, zz - lam)]
    best, best_norm = None, 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            c = cross(rows[i], rows[j])
            n = norm(c)
            if n > best_norm:
                best, best_norm = c, n
    if best is None:
        return (1.0, 0.0, 0.0)  # degenerate: any direction is an eigenvector
    return tuple(x / best_norm for x in best)


# largest first
def eigenvalues3(xx, yy, zz, xy, xz, yz):
    p1 = xy * xy + xz * xz + yz * yz
    if p1 == 0:
        return tuple(sorted((xx, yy, zz), reverse=True))
    q = (xx + yy + zz) / 3
    p2 = (xx - q) ** 2 + (yy - q) ** 2 + (zz - q) ** 2 + 2 * p1
    p = math.sqrt(p2 / 6)
    bxx, byy, bzz = (xx - q) / p, (yy - q) / p, (zz - q) / p
    bxy, bxz, byz = xy / p, xz / p, yz / p
    det = (
        bxx * (byy * bzz - byz * byz)
        - bxy * (bxy * bzz - byz * bxz)
        + bxz * (bxy * byz - byy * bxz)
    )
    r = max(-1.0, min(1.0, det / 2))
    phi = math.acos(r) / 3
    high = q + 2 * p * math.cos(phi)
    low = q + 2 * p * math.cos(phi + 2 * math.pi / 3)
    return high, 3 * q - high - low, low


# refitting with cap normals repelled pulls the axis perpendicular to the walls
def sweep_axis(normals):
    def fit(entries):
        m = [0.0] * 6
        for w, n in entries:
            m[0] += w * n[0] * n[0]
            m[1] += w * n[1] * n[1]
            m[2] += w * n[2] * n[2]
            m[3] += w * n[0] * n[1]
            m[4] += w * n[0] * n[2]
            m[5] += w * n[1] * n[2]
        return min_eigenvector(*m)

    axis = fit(normals)
    for _ in range(SWEEP_FIT_ROUNDS):
        signed = []
        for w, n in normals:
            axial = (n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2]) ** 2
            signed.append((w if axial < CAP_SPLIT else -w, n))
        refit = fit(signed)
        done = abs(abs(sum(x * y for x, y in zip(axis, refit))) - 1) < 1e-12
        axis = refit
        if done:
            break
    return axis


# membership is read with find
def class_components(group, is_cap, edges, areas):
    in_group = set(group)
    parent = {i: i for i in group}
    contact = collections.Counter()
    for owners in edges.values():
        if len(owners) != 2:
            continue
        a, b = owners
        if a not in in_group or b not in in_group:
            continue
        if is_cap[a] == is_cap[b]:
            ra, rb = find(parent, a), find(parent, b)
            if ra != rb:
                parent[ra] = rb
        else:
            contact[(a, b)] += 1
    comp_area = collections.defaultdict(float)
    for i in group:
        comp_area[find(parent, i)] += areas[i]
    return parent, contact, comp_area


# a sliver's two boundary seams run a single face apart with no crease
def merge_slivers(wall, parent, contact, comp_area, edges):
    in_wall = set(wall)
    while len(comp_area) > 1:
        exposed = set()
        for key, owners in edges.items():
            if len(owners) != 2:
                continue
            a, b = owners
            a_in, b_in = a in in_wall, b in in_wall
            if a_in != b_in:
                exposed.add(a if a_in else b)
            elif a_in and find(parent, a) != find(parent, b):
                exposed.update((a, b))
        interior = collections.Counter()
        for i in wall:
            if i not in exposed:
                interior[find(parent, i)] += 1
        slivers = [c for c in comp_area if not interior[c]]
        if not slivers:
            return
        sliver = min(slivers, key=comp_area.get)
        touch = collections.Counter()
        for (a, b), count in contact.items():
            ra, rb = find(parent, a), find(parent, b)
            if ra != rb and sliver in (ra, rb):
                touch[rb if ra == sliver else ra] += count
        if not touch:
            return
        into = touch.most_common(1)[0][0]
        parent[sliver] = into
        comp_area[into] += comp_area.pop(sliver)


# a rim seam is not worth a speck
def merge_specks(parent, contact, comp_area, floor):
    while True:
        speck = min(comp_area, key=comp_area.get)
        if comp_area[speck] >= floor or len(comp_area) < 2:
            break
        touch = collections.Counter()
        for (a, b), count in contact.items():
            ra, rb = find(parent, a), find(parent, b)
            if ra != rb and speck in (ra, rb):
                touch[rb if ra == speck else ra] += count
        if not touch:
            break
        into = touch.most_common(1)[0][0]
        parent[speck] = into
        comp_area[into] += comp_area.pop(speck)


# entries is {face: (area, normal)}
def normal_fit(entries):
    def fit(run):
        picked = [entries[i] for i in run if i in entries]
        if len(picked) < SWEEP_MIN_FACES:
            return None
        m = [0.0] * 6
        for w, n in picked:
            m[0] += w * n[0] * n[0]
            m[1] += w * n[1] * n[1]
            m[2] += w * n[2] * n[2]
            m[3] += w * n[0] * n[1]
            m[4] += w * n[0] * n[2]
            m[5] += w * n[1] * n[2]
        axis = min_eigenvector(*m)
        total = band = wall_mass = 0.0
        wall_sum = [0.0, 0.0, 0.0]
        for w, n in picked:
            axial = (n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2]) ** 2
            total += w
            if BAND_LO < axial < BAND_HI:
                band += w
            if axial < CAP_SPLIT:
                wall_mass += w
                for k in range(3):
                    wall_sum[k] += w * n[k]
        # an end run carries the tube tip, but a run may not be mostly cap
        if band > SWEEP_BAND * total or total - wall_mass > total / 2:
            return None
        if not wall_mass or norm(wall_sum) / wall_mass > WALL_ROUND:
            return None
        return axis

    return fit


# a corrugated hose reads straight where the normal test sees a bend at each rib
def slender_fit(verts, faces, areas, entries):
    centers = {}

    def center(i):
        if i not in centers:
            face = faces[i]
            centers[i] = [sum(verts[v][k] for v in face) / len(face) for k in range(3)]
        return centers[i]

    def fit(run):
        if len(run) < SWEEP_MIN_FACES:
            return None
        total = 0.0
        mean = [0.0, 0.0, 0.0]
        resultant = [0.0, 0.0, 0.0]
        for i in run:
            w = areas[i]
            total += w
            c = center(i)
            for k in range(3):
                mean[k] += w * c[k]
            if i in entries:
                n = entries[i][1]
                for k in range(3):
                    resultant[k] += w * n[k]
        if not total:
            return None
        if norm(resultant) / total > WALL_ROUND:
            return None
        mean = [x / total for x in mean]
        m = [0.0] * 6
        for i in run:
            w = areas[i]
            c = center(i)
            d0, d1, d2 = (c[k] - mean[k] for k in range(3))
            m[0] += w * d0 * d0
            m[1] += w * d1 * d1
            m[2] += w * d2 * d2
            m[3] += w * d0 * d1
            m[4] += w * d0 * d2
            m[5] += w * d1 * d2
        high, middle, low = eigenvalues3(*m)
        if high <= 0 or (middle + low) / high > RUN_SLENDER:
            return None
        return min_eigenvector(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5])

    return fit


# moves the cut to the deepest valley, the groove between ribs on a hose
def valley_snap(verts, faces, edges, entries):
    def centroid(i):
        face = faces[i]
        return [sum(verts[v][k] for v in face) / len(face) for k in range(3)]

    def score(layer, next_layer):
        ahead = set(next_layer)
        total = 0.0
        for f in layer:
            if f not in entries:
                continue
            normal = entries[f][1]
            base = centroid(f)
            for key in face_keys(faces[f]):
                owners = edges[key]
                if len(owners) != 2:
                    continue
                a, b = owners
                other = b if a == f else a
                if other not in ahead or other not in entries:
                    continue
                v0, v1 = key
                length = norm([verts[v1][k] - verts[v0][k] for k in range(3)])
                dot = sum(x * y for x, y in zip(normal, entries[other][1]))
                turn = math.acos(max(-1.0, min(1.0, dot)))
                across = centroid(other)
                toward = sum(normal[k] * (across[k] - base[k]) for k in range(3))
                total += length * turn * (1.0 if toward > 0 else -1.0)
        return total

    def snap(flat, ends, cut):
        best, best_score = cut, 0.0
        for c in range(max(2, cut - VALLEY_SNAP_LAYERS), cut + 1):
            valley = score(flat[ends[c - 2] : ends[c - 1]], flat[ends[c - 1] : ends[c]])
            if valley > best_score:
                best, best_score = c, valley
        return best

    return snap


# the walk runs after absorb and nothing width-checks its leftovers
def merge_shed(group, relabel, label, verts, faces, edges, areas, min_width):
    if not min_width:
        return
    in_group = set(group)
    shed = [i for i in group if i not in relabel]
    for comp in component_faces(shed, edges):
        in_comp = set(comp)
        area = perimeter = 0.0
        touch = collections.Counter()
        for i in comp:
            area += areas[i]
            for key in face_keys(faces[i]):
                owners = edges[key]
                other = owners[owners[0] == i] if len(owners) == 2 else None
                if other in in_comp:
                    continue
                v0, v1 = key
                length = norm([verts[v1][k] - verts[v0][k] for k in range(3)])
                perimeter += length
                if other is None:
                    continue
                if other in in_group:
                    if other in relabel:
                        touch[relabel[other]] += length
                else:
                    touch[relabel.get(other, label[other])] += length
        if perimeter <= 0 or 2 * area / perimeter >= min_width or not touch:
            continue
        into = touch.most_common(1)[0][0]
        for i in comp:
            relabel[i] = into


# end caps kept give a polar map no distortion measure catches
def split_sweeps(
    verts,
    faces,
    weighted,
    areas,
    edges,
    label,
    min_width=0,
    model_area=None,
    face_ids=None,
):
    members = collections.defaultdict(list)
    for i, r in label.items():
        members[r].append(i)
    if model_area is None:
        model_area = sum(areas)

    relabel = {}
    for r, group in members.items():
        axial = {}
        normals = []
        for i in group:
            n = weighted[i]
            length = norm(n)
            if length:
                normals.append((i, areas[i], tuple(x / length for x in n)))
        total = sum(w for _, w, _ in normals)
        if len(normals) < SWEEP_MIN_FACES or total < SWEEP_MIN_SHARE * model_area:
            continue
        axis = sweep_axis([(w, n) for _, w, n in normals])
        band = cap = 0.0
        for i, w, n in normals:
            axial[i] = (n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2]) ** 2
            if BAND_LO < axial[i] < BAND_HI:
                band += w
            if axial[i] >= CAP_SPLIT:
                cap += w
        if band > SWEEP_BAND * total:
            # normals catch a fat smooth tube, positions catch a corrugated hose
            entries = {i: (w, n) for i, w, n in normals}
            by_normals = normal_fit(entries)
            by_positions = slender_fit(verts, faces, areas, entries)

            def either(run):
                return by_normals(run) or by_positions(run)

            snap = valley_snap(verts, faces, edges, entries)
            for run, _ in straight_runs(group, entries, edges, either, snap, face_ids):
                for i in run:
                    relabel[i] = run[0]
            merge_shed(group, relabel, label, verts, faces, edges, areas, min_width)
            continue
        if not SWEEP_CAP_MIN * total <= cap <= total / 2:
            continue
        wall_sum = [0.0, 0.0, 0.0]
        for i, w, n in normals:
            if axial[i] < CAP_SPLIT:
                for k in range(3):
                    wall_sum[k] += w * n[k]
        if norm(wall_sum) / (total - cap) > WALL_ROUND:
            continue
        is_cap = {i: axial.get(i, 0.0) >= CAP_SPLIT for i in group}
        parent, contact, comp_area = class_components(group, is_cap, edges, areas)
        merge_specks(parent, contact, comp_area, SWEEP_CAP_MIN * total)
        if len(comp_area) < 2:
            continue
        for i in group:
            relabel[i] = find(parent, i)

    if not relabel:
        return label
    return {i: relabel.get(i, r) for i, r in label.items()}


def component_faces(subset, edges):
    in_set = set(subset)
    parent = {i: i for i in subset}
    for owners in edges.values():
        if len(owners) == 2 and owners[0] in in_set and owners[1] in in_set:
            ra, rb = find(parent, owners[0]), find(parent, owners[1])
            if ra != rb:
                parent[ra] = rb
    groups = collections.defaultdict(list)
    for i in subset:
        groups[find(parent, i)].append(i)
    return list(groups.values())


# above 0 a handle runs through the surface and no cut disk_cuts places opens it
def surface_genus(subset, faces, edges):
    in_set = set(subset)
    used = set()
    keys = set()
    for i in subset:
        used.update(faces[i])
        keys.update(face_keys(faces[i]))
    euler = len(used) - len(keys) + len(subset)
    parent = {}
    for key in keys:
        if sum(o in in_set for o in edges[key]) != 1:
            continue
        v0, v1 = key
        parent.setdefault(v0, v0)
        parent.setdefault(v1, v1)
        ra, rb = find(parent, v0), find(parent, v1)
        if ra != rb:
            parent[ra] = rb
    loops = len({find(parent, v) for v in parent})
    return (2 - loops - euler) // 2


# a handle through a wall makes a surface no flatten can open
def trim_genus(wall, verts, faces, edges, areas, axis):
    kept = set()
    for comp in component_faces(wall, edges):
        if surface_genus(comp, faces, edges) <= 0:
            kept.update(comp)
            continue
        position = {}
        for i in comp:
            face = faces[i]
            centroid = [sum(verts[v][k] for v in face) / len(face) for k in range(3)]
            position[i] = sum(centroid[k] * axis[k] for k in range(3))
        ordered = sorted(comp, key=position.get)
        best = None
        for direction in (ordered, ordered[::-1]):
            for level in range(1, GENUS_TRIM_LEVELS):
                cut = len(comp) * level // GENUS_TRIM_LEVELS
                rest = direction[cut:]
                if not rest:
                    break
                pieces = component_faces(rest, edges)
                if all(surface_genus(p, faces, edges) <= 0 for p in pieces):
                    loss = sum(areas[i] for i in direction[:cut])
                    if best is None or loss < best[0]:
                        best = (loss, rest)
                    break
        if best is not None and best[0] <= sum(areas[i] for i in comp) / 2:
            kept.update(best[1])
    return kept


# a hoop unrolls into a strip past the slicer bound and rims would shred it
def wall_hoops(wall, axis, verts, faces, edges):
    boundary = 0.0
    for i in wall:
        for key in face_keys(faces[i]):
            owners = edges[key]
            if len(owners) == 2 and (owners[0] in wall) != (owners[1] in wall):
                v0, v1 = key
                boundary += norm([verts[v1][k] - verts[v0][k] for k in range(3)])
    along = [
        sum(verts[v][k] * axis[k] for k in range(3)) for i in wall for v in faces[i]
    ]
    extent = max(along) - min(along) if along else 0.0
    return boundary > 0 and (extent <= 0 or boundary / 2 > SPLIT_ASPECT * extent)


# the group's wall faces after the cap split, the genus trim and the hoop check
def claim_wall(group, axial, axis, total, verts, faces, edges, areas):
    is_cap = {i: axial.get(i, 0.0) >= CAP_SPLIT for i in group}
    parent, contact, comp_area = class_components(group, is_cap, edges, areas)
    cls = {find(parent, i): is_cap[i] for i in group}
    merge_specks(parent, contact, comp_area, SWEEP_CAP_MIN * total)
    wall = {i for i in group if not cls[find(parent, i)]}
    wall = trim_genus(wall, verts, faces, edges, areas, axis)
    if not wall:
        return set()
    if wall_hoops(wall, axis, verts, faces, edges):
        return set()
    return wall


# reachable faces grouped by steps taken from the seed
def spread_rings(adjacency, seed, allowed):
    layers = [[seed]]
    seen = {seed}
    while True:
        grown = []
        for face in layers[-1]:
            for other in adjacency[face]:
                if other in allowed and other not in seen:
                    seen.add(other)
                    grown.append(other)
        if not grown:
            return layers
        layers.append(grown)


# narrow panels are what let rectify straighten a bowed shell
def profile_panels(wall, axis, entries, edges, areas):
    u = None
    for candidate in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        c = cross(axis, candidate)
        length = norm(c)
        if length > 0.1:
            u = [x / length for x in c]
            break
    v = cross(axis, u)
    theta = {}
    mass = [0.0] * PROFILE_BINS
    for i in wall:
        if i not in entries:
            continue
        w, n = entries[i]
        axial = sum(x * y for x, y in zip(n, axis))
        perp = [n[k] - axial * axis[k] for k in range(3)]
        if norm(perp) <= 0:
            continue
        angle = math.atan2(
            sum(x * y for x, y in zip(perp, v)),
            sum(x * y for x, y in zip(perp, u)),
        )
        theta[i] = angle
        b = int((angle + math.pi) / (2 * math.pi) * PROFILE_BINS)
        mass[min(b, PROFILE_BINS - 1)] += w
    total = sum(mass)
    uniform = total / PROFILE_BINS
    if uniform <= 0 or max(mass) < PROFILE_PEAK * uniform:
        return None
    flat = [m > PROFILE_FLAT * uniform for m in mass]
    starts = [b for b in range(PROFILE_BINS) if flat[b] and not flat[b - 1]]
    if len(starts) < 2:
        return None
    runs = []
    for start in starts:
        length = 1
        while flat[(start + length) % PROFILE_BINS]:
            length += 1
        runs.append((start, length))
    cuts = []
    for (start, length), (following, next_length) in zip(runs, runs[1:] + runs[:1]):
        gap = [
            (start + length + k) % PROFILE_BINS
            for k in range((following - start - length) % PROFILE_BINS)
        ]
        if not gap or sum(mass[g] for g in gap) < PROFILE_CORNER * total:
            continue
        mid = start + (length - 1) / 2
        next_mid = following + (next_length - 1) / 2
        turn = (next_mid - mid) % PROFILE_BINS * 360 / PROFILE_BINS
        if turn < PROFILE_TURN:
            continue
        center = (len(gap) - 1) / 2
        low = min(range(len(gap)), key=lambda k: (mass[gap[k]], abs(k - center)))
        cuts.append(-math.pi + (gap[low] + 0.5) * 2 * math.pi / PROFILE_BINS)
    if len(cuts) < 2:
        return None
    cuts.sort()
    panel = {i: -1 for i in wall}
    for i, angle in theta.items():
        panel[i] = bisect.bisect_left(cuts, angle) % len(cuts)
    wall_area = sum(areas[i] for i in wall)
    parent, contact, comp_area = class_components(wall, panel, edges, areas)
    merge_specks(parent, contact, comp_area, SWEEP_CAP_MIN * wall_area)
    merge_slivers(wall, parent, contact, comp_area, edges)
    if len(comp_area) < 2:
        return None
    return {i: find(parent, i) for i in wall}


# a coiled cable fails the whole-cluster test though every short piece is a tube
def straight_runs(group, entries, edges, fit_of=None, snap=None, face_ids=None):
    in_group = set(group)
    adjacency = collections.defaultdict(list)
    for owners in edges.values():
        if len(owners) == 2:
            a, b = owners
            if a in in_group and b in in_group:
                adjacency[a].append(b)
                adjacency[b].append(a)

    if fit_of is None:
        fit_of = normal_fit(entries)

    remaining = set(group)
    ids = list(group) if face_ids is None else [face_ids[i] for i in group]
    picks = set(ids)
    local = dict(zip(ids, group))
    runs = []
    seed = None
    while remaining:
        if seed is None or seed not in remaining:
            start = local[next(iter(picks))]
            seed = spread_rings(adjacency, start, remaining)[-1][0]
        layers = spread_rings(adjacency, seed, remaining)
        flat, ends = [], []
        for layer in layers:
            flat.extend(layer)
            ends.append(len(flat))
        count = len(layers)

        span = count
        for depth, end in enumerate(ends, 1):
            if end >= RUN_SEED_FACES:
                span = depth
                break
        low, axis = 0, None
        probe = 1
        while probe < span:
            fit = fit_of(flat[: ends[probe - 1]])
            if fit:
                low, axis = probe, fit
                break
            probe *= 2
        if axis is None:
            fit = fit_of(flat[: ends[span - 1]])
            if fit:
                low, axis = span, fit
            else:
                # nothing straight near this seed
                dropped = flat[: ends[span - 1]]
                remaining.difference_update(dropped)
                picks.difference_update(ids_of(dropped, face_ids))
                seed = layers[span][0] if span < count else None
                continue
        high = None
        while high is None and low < count:
            probe = min(low * 2, count)
            fit = fit_of(flat[: ends[probe - 1]])
            if fit:
                low, axis = probe, fit
            else:
                high = probe
        while high is not None and high - low > 1:
            mid = (low + high) // 2
            fit = fit_of(flat[: ends[mid - 1]])
            if fit:
                low, axis = mid, fit
            else:
                high = mid
        if snap and 1 < low < count:
            low = snap(flat, ends, low)
        run = flat[: ends[low - 1]]
        runs.append((run, axis))
        remaining.difference_update(run)
        picks.difference_update(ids_of(run, face_ids))
        seed = layers[low][0] if low < count else None
    return runs


def ids_of(faces, face_ids):
    return faces if face_ids is None else [face_ids[i] for i in faces]


# absorb never checks how far a boundary turns, so this reads before any merge
def sweep_rims(verts, faces, model_area=None, face_ids=None, built=None):
    weighted, areas, edges = built or build(verts, faces)
    root = partition(faces, weighted, edges, CREASE_ANGLE)
    groups = collections.defaultdict(list)
    for i in range(len(faces)):
        groups[root(i)].append(i)
    if model_area is None:
        model_area = sum(areas)

    walls = set()
    wall_axis = {}
    wall_run = {}
    wall_cluster = {}
    wall_panel = {}
    run_count = 0

    def claim(run, axis, cluster, axial, total, entries):
        nonlocal run_count
        wall = claim_wall(run, axial, axis, total, verts, faces, edges, areas)
        if not wall:
            return
        run_count += 1
        walls.update(wall)
        panels = profile_panels(wall, axis, entries, edges, areas)
        for i in wall:
            wall_axis[i] = axis
            wall_run[i] = run_count
            wall_cluster[i] = cluster
            if panels:
                wall_panel[i] = panels[i]

    for group in groups.values():
        normals = []
        for i in group:
            n = weighted[i]
            length = norm(n)
            if length:
                normals.append((i, areas[i], tuple(x / length for x in n)))
        total = sum(w for _, w, _ in normals)
        if len(normals) < SWEEP_MIN_FACES or total < SWEEP_MIN_SHARE * model_area:
            continue
        m = [0.0] * 6
        for _, w, n in normals:
            m[0] += w * n[0] * n[0]
            m[1] += w * n[1] * n[1]
            m[2] += w * n[2] * n[2]
            m[3] += w * n[0] * n[1]
            m[4] += w * n[0] * n[2]
            m[5] += w * n[1] * n[2]
        axis = min_eigenvector(*m)
        axial = {}
        band = off_wall = 0.0
        resultant = [0.0, 0.0, 0.0]
        for i, w, n in normals:
            axial[i] = (n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2]) ** 2
            if axial[i] > BAND_LO:
                off_wall += w
            if BAND_LO < axial[i] < BAND_HI:
                band += w
            for k in range(3):
                resultant[k] += w * n[k]
        # of those only a bent tube fills the middle band
        if off_wall > SWEEP_BAND * total or norm(resultant) / total > WALL_ROUND:
            if band > SWEEP_BAND * total:
                entries = {i: (w, n) for i, w, n in normals}
                for run, run_axis in straight_runs(
                    group, entries, edges, face_ids=face_ids
                ):
                    run_axial = {}
                    run_total = 0.0
                    for i in run:
                        if i in entries:
                            w, n = entries[i]
                            run_total += w
                            run_axial[i] = (
                                n[0] * run_axis[0]
                                + n[1] * run_axis[1]
                                + n[2] * run_axis[2]
                            ) ** 2
                    claim(run, run_axis, group[0], run_axial, run_total, entries)
            continue
        claim(group, axis, group[0], axial, total, {i: (w, n) for i, w, n in normals})

    if not walls:
        return set(), walls
    rims = set()
    for key, owners in edges.items():
        if len(owners) != 2:
            continue
        a, b = owners
        in_a, in_b = a in walls, b in walls
        if in_a != in_b:
            rims.add(key)
        elif in_a and wall_run[a] != wall_run[b]:
            # runs of one bent tube always part, that cut is what made them straight
            if wall_cluster[a] == wall_cluster[b]:
                rims.add(key)
            else:
                dot = abs(sum(x * y for x, y in zip(wall_axis[a], wall_axis[b])))
                if dot < SHARED_AXIS_COS:
                    rims.add(key)
        elif in_a and wall_panel.get(a) != wall_panel.get(b):
            rims.add(key)
    return rims, walls
