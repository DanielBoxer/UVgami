"""Launch a background Blender with this checkout enabled as an extension.

Two halves, for a script that is both the launcher and what runs inside:
`launch` runs from the dev venv and starts Blender on a throwaway user profile
with the repo junctioned in, `enable_addon` runs in that Blender. The real
profile is never touched.

Blender's stdout is inherited, not captured, so a caller parsing its lines
sees them as they come.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
# the extension id the addon is registered under, which utils.paths derives
# its preferences key from, so the link name and repo folder both matter
EXTENSION_REPO = "user_default"
# the link name, which is what the module is named after, not the manifest id.
# matching the repo folder keeps this the same mixed case the vs code extension
# loads it under, so a lookup that assumes lowercase fails here too
EXTENSION_NAME = "UVgami"
ADDON_MODULE = f"bl_ext.{EXTENSION_REPO}.{EXTENSION_NAME}"
# how the launcher hands its own site-packages to blender's python, which has
# no pytest of its own
SITE_PACKAGES_VAR = "UVGAMI_TEST_SITE_PACKAGES"
WINDOWS_INSTALL_GLOB = "Blender Foundation/Blender */blender.exe"


def find_blender():
    override = os.environ.get("BLENDER")
    if override:
        return pathlib.Path(override)
    found = shutil.which("blender")
    if found:
        return pathlib.Path(found)
    installs = []
    for program_files in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(program_files)
        if root:
            installs += pathlib.Path(root).glob(WINDOWS_INSTALL_GLOB)
    if not installs:
        raise SystemExit("no blender found, set BLENDER to its executable")
    # newest version last, so the latest install is the default
    return sorted(installs)[-1]


def link_checkout(profile):
    """Junction the repo into a fresh profile's extensions folder. The addon is
    a valid extension folder as it stands (manifest plus __init__), so nothing
    has to be built, and the run is against the working tree."""
    link = profile / "extensions" / EXTENSION_REPO / EXTENSION_NAME
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # a junction, not a symlink: symlinks need developer mode or admin
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(REPO_ROOT)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(REPO_ROOT, target_is_directory=True)
    return link


def launch(script, args):
    """Run script inside a background Blender, returning its exit code."""
    blender = find_blender()
    profile = pathlib.Path(tempfile.mkdtemp(prefix="uvgami-blender-"))
    link = link_checkout(profile)
    try:
        return subprocess.run(
            [
                str(blender),
                "-b",
                "--python",
                str(pathlib.Path(script).resolve()),
                "--",
                *args,
            ],
            env={
                **os.environ,
                "BLENDER_USER_RESOURCES": str(profile),
                SITE_PACKAGES_VAR: sysconfig.get_paths()["purelib"],
            },
        ).returncode
    finally:
        # drop the link first: a recursive delete can follow a junction into
        # the checkout and take the working tree with it
        if os.name == "nt":
            os.rmdir(link)
        else:
            link.unlink()
        shutil.rmtree(profile, ignore_errors=True)


def script_args():
    """Blender hands the script its whole command line, ours is after the --."""
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def main(test_dir, script):
    """Both halves of a test runner script: launch Blender on script from the
    dev venv, then run pytest over test_dir once inside it."""
    if "bpy" not in sys.modules:
        sys.exit(launch(script, sys.argv[1:]))
    enable_addon()
    import pytest

    # faulthandler prints every access violation tbbmalloc_proxy probes and catches
    sys.exit(
        pytest.main(
            [
                str(test_dir),
                "-p",
                "no:cacheprovider",
                "-p",
                "no:faulthandler",
                *script_args(),
            ]
        )
    )


def enable_addon():
    """Inside Blender: put the launcher's packages, dev/ and this folder on the
    path, then enable the linked checkout."""
    import bpy

    # appended, not prepended, so blender's own numpy still wins over the
    # venv's build for a different python version. dev/ comes along because the
    # venv installs uvgami_cli through a .pth, which only site.py reads, and
    # the repo's pytest addopts load a plugin from it. this folder comes along
    # for timer_pump and the shared fixtures
    sys.path += [os.environ[SITE_PACKAGES_VAR], str(REPO_ROOT / "dev"), str(HERE)]
    bpy.ops.extensions.repo_refresh_all()
    bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
