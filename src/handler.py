import traceback

import bpy

from .logger import logger
from .manager import manager
from .utils.ui import popup

MESSAGES = {
    "START": "Error starting unwrap: ",
    "MIDDLE": "Error during unwrap: ",
}


def handle_error(error, location, **kwargs):
    msg = MESSAGES.get(location, "Error: ")
    logger.add_data("errors", msg)

    error_list = traceback.format_exc().split("\n")[:-1]
    for line in error_list:
        logger.add_data("errors", line)
        print(line)
    logger.update_time()
    logger.change_status("Error")
    logger.write_latest()

    popup(error_list, msg + str(error), "ERROR")

    cleanup(location, **kwargs)


def cleanup(location, objects=frozenset()):
    if location == "START":
        # the delete below would take a live session's objects too
        if manager.is_active:
            return
        # every object created since the operator started is temporary
        for obj in set(bpy.data.objects).difference(objects):
            bpy.data.objects.remove(obj, do_unlink=True)

    manager.stop_all()
    manager.finish()
