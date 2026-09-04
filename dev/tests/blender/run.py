import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
# blender's --python runs the script without its folder on the path
sys.path.append(str(HERE))

import background_blender  # noqa: E402

if __name__ == "__main__":
    background_blender.main(HERE, __file__)
