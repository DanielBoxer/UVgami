# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy
import traceback
from .manager import manager
from .logger import logger
from .utils import popup


def handle_error(error, location, **kwargs):
    msg = "Error: "
    if location == "START":
        msg = "Error starting unwrap: "
    elif location == "FINISH":
        msg = "Error finishing unwrap: "
    elif location == "MIDDLE":
        msg = "Error during unwrap: "
    logger.add_data("errors", msg)

    error_list = traceback.format_exc().split("\n")[:-1]
    for line in error_list:
        logger.add_data("errors", line)
    logger.change_status("Error")

    popup(error_list, msg + str(error), "ERROR")

    cleanup(location, **kwargs)


def cleanup(location, objects=set()):
    if location == "START":
        for obj in set(bpy.data.objects).difference(objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        manager.stop_all()

    elif location == "FINISH":
        manager.stop_all()

    elif location == "MIDDLE":
        manager.stop_all()

    manager.finish()
