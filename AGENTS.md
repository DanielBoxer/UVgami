# UVgami

Blender addon that does automatic UV unwrapping. Three engines: optcuts (C++ binary), xatlas (C++ binary) and partuv (CUDA wheel).

## Layout

- `src/` is the addon, `src/engines/__init__.py` lists the engines. `dev/` is not part of the addon.
- `dev/tests/` is the addon and CLI tests. `dev/tests/blender/` runs inside a real Blender through its `run.py` and is skipped by the venv run.
- `docs/docs.md` (user guide) and `README.md` are human only. Never edit them, propose the change instead. Anything an agent writes goes in the agent notes folder.
- Keep this file and the agent notes short: only what an agent would get wrong without it.

## Commands

- Test: `uv run --no-sync pytest` (no GPU or Blender needed), then `uv run --no-sync python dev/tests/blender/run.py` for the bpy half
- Lint: `uv run --no-sync ruff check --fix` then `uv run --no-sync ruff format`

## Gotchas

- The dev venv is hand-built: the partuv CUDA stack was installed with `--extra partuv`, which is outside the default sync set. Bare `uv sync` uninstalls all of it, so sync with `uv sync --inexact`. Plain `uv run` is safe (inexact by default).
- Engine stdout is a parsed protocol (`start:`/`done:`/`failed:`/`progress:` lines). Don't print extra lines to stdout there, use stderr.
- Optcuts unwraps many meshes per process. A new global or static in `uvgami.cpp` must be reset in `resetMeshState`.
- `src/` imports bpy, so only its bpy-free modules are unit-testable. `dev/tests/` shows which ones.
- See the agent notes for more about `src/seams/` and `src/proxy.py`.
- The venv's numpy makes int32 arrays on Windows, Blender's makes int64, so pass `dtype=numpy.int64` to anything that feeds a key or a shift.
- The addon runs one engine job per loose part. The CLI and bench feed the mesh whole, so a multi-part model can fail there but not in Blender.
- The addon zip contains no engines. Each one downloads from its own GitHub release on first use, driven by `src/engines/binary_engine.py` (optcuts, xatlas) and `src/engines/partuv/install.py`.
- Changing an engine's code needs that engine's version bumped: `engine/optcuts/VERSION`, `engine/xatlas/VERSION`, `engine/partuv/pyproject.toml`. CI fails if the version constants mirrored in the addon drift. That rebuilds the engine only, releases trigger only from the version line in `blender_manifest.toml`. An older install keeps running and the addon offers the update. A release that adds something the addon then sends, a flag or a stdin command, raises `OPTCUTS_MINIMUM_VERSION` to its own version, and is a minor bump by convention.
- After building an engine, copy the binary into `engine-builds/windows/` (per-platform, gitignored) or the addon and CLI run the stale one. That local copy is used instead of the downloaded engine, which is how a dev build gets tested in Blender. If addon behavior contradicts the engine source, compare the binary's mtime to the latest engine commit first.
- `UVGAMI_ENGINE_DIR` points at another folder of engine binaries, checked before `engine-builds/`, so a worktree run doesn't need to copy over the shared one.
- GitHub Pages builds the site from the repo root, so anything tracked and not in the `exclude` list in `_config.yml` gets published. Adding a top level folder or a markdown file that isn't user docs means adding it there too.
