#!/usr/bin/env bash
# Extract engine binaries from engine zip files and bundle them into the addon.
# Usage: bundle-engines.sh <addon_dir> <engine_zip_prefix>
# Example: bundle-engines.sh UVgami uvgami-engine

set -euo pipefail

ADDON_DIR="$1"
ENGINE_ZIP_PREFIX="$2"

engine_zips=( ${ENGINE_ZIP_PREFIX}-*.zip )
if [ ! -f "${engine_zips[0]}" ]; then
  echo "No engine zips found matching ${ENGINE_ZIP_PREFIX}-*.zip"
  exit 1
fi

mkdir -p "${ADDON_DIR}/engines"

for zip_file in "${engine_zips[@]}"; do
  # extract platform from filename (e.g., uvgami-engine-1.1.2-windows.zip -> windows)
  platform=$(echo "$zip_file" | sed "s/${ENGINE_ZIP_PREFIX}-[0-9.]*-//" | sed 's/\.zip//')

  tmp_dir=$(mktemp -d)
  unzip -o "$zip_file" -d "$tmp_dir"

  # find the engine binary (named uvgami or uvgami.exe)
  engine_bin=$(find "$tmp_dir" -name "uvgami" -o -name "uvgami.exe" | head -1)
  if [ -n "$engine_bin" ]; then
    mkdir -p "${ADDON_DIR}/engines/${platform}"
    cp "$engine_bin" "${ADDON_DIR}/engines/${platform}/"
    echo "Bundled engine for ${platform}"
  else
    echo "Warning: no engine binary found in ${zip_file}"
  fi

  rm -rf "$tmp_dir"
done
