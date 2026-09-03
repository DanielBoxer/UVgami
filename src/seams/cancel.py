class Cancelled(Exception):
    pass


# a thread cannot be killed, so the passes call this where giving up is safe
def check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise Cancelled
