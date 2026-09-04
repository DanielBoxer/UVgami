import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
# utils.paths derives its preferences key from this
EXTENSION_REPO = "user_default"
# the module is named after the link, not the manifest id
EXTENSION_NAME = "UVgami"
ADDON_MODULE = f"bl_ext.{EXTENSION_REPO}.{EXTENSION_NAME}"
# blender's python has no pytest of its own
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
    # version folder names sort newest last
    return sorted(installs)[-1]


def link_checkout(profile):
    link = profile / "extensions" / EXTENSION_REPO / EXTENSION_NAME
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # symlinks need developer mode or admin
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(REPO_ROOT)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(REPO_ROOT, target_is_directory=True)
    return link


def launch(script, args):
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
        # a recursive delete can follow the junction into the checkout
        if os.name == "nt":
            os.rmdir(link)
        else:
            link.unlink()
        shutil.rmtree(profile, ignore_errors=True)


def script_args():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def main(test_dir, script):
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
    import bpy

    # appended so blender's own numpy still wins over the venv's build
    sys.path += [os.environ[SITE_PACKAGES_VAR], str(REPO_ROOT / "dev"), str(HERE)]
    bpy.ops.extensions.repo_refresh_all()
    bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
