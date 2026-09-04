import pytest


# the repo root __init__.py is the blender addon entry, not a python package
def pytest_collect_directory(path, parent):
    if path == parent.config.rootpath:
        return pytest.Dir.from_parent(parent, path=path)
    return None
