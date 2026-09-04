#pragma once

#include <string>

namespace uvgami {

// reads obj plus a <stem>_seams sidecar, writes obj with vt, faces stay polygons
int runFlatten(const std::string &inputPath, const std::string &outputDir,
               int maxIterations, bool packOnly);

}  // namespace uvgami
