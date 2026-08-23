# must match the ai extra in engine/partuv/pyproject.toml
TORCH_VERSION = "2.3.0"
# pypi's windows torch wheel is cpu-only
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
# without this page torch-scatter builds from source
TORCH_SCATTER_FIND_LINKS = f"https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu121.html"
# independent of blender's python version
VENV_PYTHON = "3.11"


def _uv(uv, *args):
    """A uv command line. --no-config keeps a user's uv.toml, which can pin an
    index or an exclude-newer cutoff, out of the addon's venv."""
    return [uv, "--no-config", *args]


def build_install_commands(uv, venv_python, venv_path, wheel_url, ai, create_venv):
    """The uv command lines that put partuv and its deps in the managed venv."""
    commands = []
    if create_venv:
        commands.append(_uv(uv, "venv", "--python", VENV_PYTHON, venv_path))
    if ai:
        # uv keeps this over the extra's cpu pin
        commands.append(
            _uv(
                uv,
                "pip",
                "install",
                "--python",
                venv_python,
                "--index-url",
                TORCH_CUDA_INDEX,
                f"torch=={TORCH_VERSION}+cu121",
            )
        )
        requirement = f"partuv[ai] @ {wheel_url}"
        extra_args = ["-f", TORCH_SCATTER_FIND_LINKS]
    else:
        requirement = f"partuv @ {wheel_url}"
        extra_args = []
    commands.append(
        _uv(
            uv,
            "pip",
            "install",
            "--python",
            venv_python,
            "--upgrade",
            *extra_args,
            requirement,
        )
    )
    return commands
