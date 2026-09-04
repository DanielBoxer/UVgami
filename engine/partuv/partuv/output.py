from pathlib import Path

import numpy as np

from .common import EXIT_BAD_OUTPUT, UnwrapError


def _reject(reason):
    raise UnwrapError(EXIT_BAD_OUTPUT, f"engine output rejected: {reason}")


def _canonical_faces(faces):
    order = (faces.argmin(axis=1)[:, None] + np.arange(3)) % 3
    return np.take_along_axis(faces, order, axis=1)


def validate_provenance(V, F, UV, source_vid, source_V, source_F):
    if len(source_vid) == 0:
        _reject("missing source vertex provenance")
    if len(source_vid) != len(V):
        _reject("provenance length does not match vertices")
    if len(UV) != len(V):
        _reject("uv count does not match vertices")
    if not np.isfinite(UV).all():
        _reject("non-finite uv coordinates")
    if source_vid.min() < 0 or source_vid.max() >= len(source_V):
        _reject("source vertex index out of range")
    if not np.array_equal(V, source_V[source_vid]):
        _reject("vertex positions disagree with provenance")

    # the mapped faces must be exactly the source faces as oriented cycles
    mapped = _canonical_faces(source_vid[F])
    source = _canonical_faces(np.asarray(source_F))
    mapped = mapped[np.lexsort(mapped.T[::-1])]
    source = source[np.lexsort(source.T[::-1])]
    if not np.array_equal(mapped, source):
        _reject("output faces do not cover the source faces exactly")


def _write_obj(path, source_V, UV, F, source_vid):
    lines = [f"v {x:.17g} {y:.17g} {z:.17g}" for x, y, z in source_V]
    lines += [f"vt {u:.17g} {v:.17g}" for u, v in UV[:, :2]]
    lines += [
        "f "
        + " ".join(f"{source_vid[corner] + 1}/{corner + 1}" for corner in face)
        for face in F
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def save_results(output_dir, final_parts, individual_parts, source_V, source_F):
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    combined = final_parts.to_components()
    V = np.asarray(combined.V)
    F = np.asarray(combined.F)
    UV = np.asarray(combined.UV)
    source_vid = np.asarray(combined.source_vid)
    source_V = np.asarray(source_V)

    validate_provenance(V, F, UV, source_vid, source_V, source_F)

    combined_mesh_path = output_dir / "final_components.obj"
    _write_obj(combined_mesh_path, source_V, UV, F, source_vid)

    return [combined_mesh_path]
