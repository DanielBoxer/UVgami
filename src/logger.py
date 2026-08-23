import time


class Info:
    def __init__(self):
        self.started = time.strftime("%H:%M:%S")
        self.time = 0
        self.errors = []
        self.status = "In Progress"
        self.objects = []
        self.engine = ""
        self.settings = ""

    def get_info(self):
        """One line for the run."""
        fields = [self.started, self.status, f"{self.time:.2f}s"]
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
        # if logs cleared during unwrap, add a new one
        if not self.unwrap_info:
            self.new_info()
        return self.unwrap_info[-1]

    def get_all(self):
        """One line per run, oldest first."""
        return [info.get_info() for info in self.unwrap_info]

    def start_timer(self):
        self.start_time = time.perf_counter()

    def update_time(self):
        self.get_latest().time = time.perf_counter() - self.start_time


logger = Logger()
