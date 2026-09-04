import hashlib
import json
import pathlib
import sys
import tomllib
import zipfile

# blender's extension listing api version, not the manifest schema version
LISTING_VERSION = "v1"

MANIFEST_NAME = "blender_manifest.toml"

# added by `extension build`
GENERATED_SECTION = "build"


def build_entry(zip_path, archive_url):
    archive = zip_path.read_bytes()
    with zipfile.ZipFile(zip_path) as package:
        manifest = tomllib.loads(package.read(MANIFEST_NAME).decode())
    entry = {k: v for k, v in manifest.items() if k != GENERATED_SECTION}
    # extension server-generate can only emit a path relative to the index
    entry["archive_url"] = archive_url
    entry["archive_size"] = len(archive)
    entry["archive_hash"] = f"sha256:{hashlib.sha256(archive).hexdigest()}"
    return entry


def main(argv):
    zip_path, archive_url, output = argv
    listing = {
        "version": LISTING_VERSION,
        "blocklist": [],
        "data": [build_entry(pathlib.Path(zip_path), archive_url)],
    }
    pathlib.Path(output).write_text(json.dumps(listing, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
