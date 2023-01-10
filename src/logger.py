# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import time
from .utils import get_preferences


class Info:
    def __init__(self):
        self.time = 0
        self.errors = []
        self.status = "In Progress"
        self.objects = []

    def get_info(self):
        return (
            [
                f"Status: {self.status}",
                f"Time: {self.time:.2f}s",
                "Objects:",
            ]
            + self.objects
            + ["Errors:"]
            + self.errors
        )


class Logger:
    def __init__(self):
        self.unwrap_info = []
        self.start_time = 0

    def new_info(self):
        if get_preferences().show_info:
            self.unwrap_info.append(Info())
            self.start_timer()

    def add_data(self, target, data):
        if get_preferences().show_info:
            # the spaces are for an indentation in the output text
            getattr(self.get_latest(), target).append("    " + data)

    def change_status(self, status):
        if get_preferences().show_info:
            self.get_latest().status = status

    def get_latest(self):
        if get_preferences().show_info:
            # if logs cleared during unwrap, add a new one
            if not self.unwrap_info:
                self.new_info()
            return self.unwrap_info[-1]

    def get_all(self):
        output = []
        # get the most recent ones first
        for info in reversed(self.unwrap_info):
            output.extend(info.get_info())
            # add empty string for newline
            output.append("")
        # remove final newline
        return output[:-1]

    def start_timer(self):
        self.start_time = time.perf_counter()

    def update_time(self):
        if get_preferences().show_info:
            self.get_latest().time = time.perf_counter() - self.start_time


logger = Logger()
