# Changelog

## 2.1.0

**Improvements:**

- Proxy mode rework (faster and better transfer)
- Preview proxy button
- Combine island stitches islands together when it can
- OptCuts speed improvements
- Better engine updating
- Better logs

**Bug Fixes:**
- Fix some OptCuts crashes and refusals
- Fix weights mode not always working for some seams
- Auto smooth and weighted normals are kept after unwrap
- Fix proxy mode on multi part objects

## 2.0.0

**New Features (see docs for more info)**

- Hard surface mode
- Proxy mode
- UV editor tools: Unwrap Island, Relax Island, Combine Islands, Unwrap Area, Relax Area
- xatlas engine
- Reduce Stretching weight mode
- Generate weights buttons
- Stack similar option
- Automatic updates repository
- Settings icon summary

**Improvements:**

- Speedups to OptCuts engine
- OptCuts can now handle messy meshes much better
- OptCuts better seam placement on creases and less zigzags
- Other speedups
- Better priority option which replaces old quality option
- Better UV viewer
- Better logs
- Better UI and various QOL improvments

**Bug Fixes:**

- Fix many OptCuts crashes, hangs and refusals
- Fix PartUV installing wrong package on windows
- Fix transfer UVs failing on small meshes
- Fix undo stack after unwrap
- Other various fixes

**Breaking changes**

- Blender 4.3 is now the minimum version
- Removed: symmetry (temporary), preserve mesh, input cleanup, finish percent, cuts, unwrap sharp, mark seams sharp
- No more bundled zip, use the download engine button instead
- Autosave defaults off
- Pack after unwrap defaults on
- Timeout defaults to 60 minutes
- Concurrent defaults on
