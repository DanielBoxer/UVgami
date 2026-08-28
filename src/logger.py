import time


class Info:
    def __init__(self):
        self.started = time.strftime("%H:%M:%S")
        self.time = 0
        # these three add up to self.time
        self.pre_processing_time = 0
        self.unwrap_time = 0
        self.post_processing_time = 0
        self.errors = []
        self.status = "In Progress"
        self.objects = []
        self.engine = ""
        self.settings = ""

    def get_time(self):
        text = f"{self.time:.2f}s"
        if self.unwrap_time:
            text += (
                f" (pre-processing {self.pre_processing_time:.2f}s,"
                f" unwrap {self.unwrap_time:.2f}s,"
                f" post-processing {self.post_processing_time:.2f}s)"
            )
        return text

    def get_info(self):
        """One line for the run."""
        fields = [self.started, self.status, self.get_time()]
        if self.engine:
            fields.append(self.engine)
        fields.append(", ".join(self.objects))
        if self.settings:
            fields.append(f"Settings: {self.settings}")
        if self.errors:
            fields.append(f"Errors: {'; '.join(self.errors)}")
        return " | ".join(fields)


class Logger:
    def __init__(self):
        self.unwrap_info = []
        self.start_time = 0
        self.unwrap_start = None
        self.unwrap_end = None

    def new_info(self):
        info = Info()
        self.unwrap_info.append(info)
        self.start_timer()
        return info

    def discard_info(self):
        """Drop the entry for a run that was refused before it started."""
        self.unwrap_info.pop()

    def add_data(self, target, data):
        getattr(self.get_latest(), target).append(data)

    def change_status(self, status):
        self.get_latest().status = status

    def get_latest(self):
        return self.unwrap_info[-1]

    def get_all(self):
        """One line per run, oldest first."""
        return [info.get_info() for info in self.unwrap_info]

    def start_timer(self):
        self.start_time = time.perf_counter()
        self.unwrap_start = None
        self.unwrap_end = None

    def mark_unwrapping(self):
        """Called while any engine still has work. Everything before the first
        call is pre-processing, everything after the last is post-processing."""
        self.unwrap_end = time.perf_counter()
        if self.unwrap_start is None:
            self.unwrap_start = self.unwrap_end

    def update_time(self):
        info = self.get_latest()
        info.time = time.perf_counter() - self.start_time
        if self.unwrap_start is None:
            info.pre_processing_time = info.time
            return
        info.pre_processing_time = self.unwrap_start - self.start_time
        info.unwrap_time = self.unwrap_end - self.unwrap_start
        info.post_processing_time = (
            info.time - info.pre_processing_time - info.unwrap_time
        )


logger = Logger()
