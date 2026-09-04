import importlib.util
import re
import threading
import tomllib
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(relpath, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# install.py imports bpy, download.py does not
download = _load("src/utils/download.py", "uvgami_download")

CONTENT = bytes(range(256)) * 40  # 10240 bytes
API = "http://127.0.0.1:1/releases"


class RangeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        server = self.server
        rng = self.headers.get("Range")
        server.requests.append(rng)
        if len(server.requests) <= server.fail_416:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{len(CONTENT)}")
            self.end_headers()
            return
        if server.ignore_range:
            self.send_response(200)
            self.send_header("Content-Length", str(len(CONTENT)))
            self.end_headers()
            self.wfile.write(CONTENT)
            return
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        body = CONTENT[start:]
        if start:
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(CONTENT) - 1}/{len(CONTENT)}"
            )
        else:
            self.send_response(200)
        # a full Content-Length with a truncated body is a mid-transfer drop
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if len(server.requests) <= server.drop_requests:
            self.wfile.write(body[: len(body) // 2])
            return
        self.wfile.write(body)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    srv.requests = []
    srv.ignore_range = False
    srv.drop_requests = 0
    srv.fail_416 = 0
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()


def url_for(server):
    host, port = server.server_address
    return f"http://{host}:{port}/model.ckpt"


def test_resume_via_range(server, tmp_path):
    server.drop_requests = 1
    dest = tmp_path / "model.ckpt"

    download.download_file(url_for(server), dest, backoff=0)

    assert dest.read_bytes() == CONTENT
    assert not dest.with_name(dest.name + ".part").exists()
    # first request carried no range, the retry resumed with one
    assert server.requests[0] is None
    assert server.requests[1].startswith("bytes=")


def test_size_mismatch_triggers_retry(server, tmp_path):
    server.drop_requests = 1
    dest = tmp_path / "model.ckpt"

    download.download_file(url_for(server), dest, backoff=0)

    # the truncated first response was rejected, forcing a second attempt
    assert len(server.requests) == 2


def test_200_response_restarts_file(server, tmp_path):
    server.ignore_range = True
    dest = tmp_path / "model.ckpt"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(b"stale-garbage-bytes")

    download.download_file(url_for(server), dest, backoff=0)

    # a range was requested but the 200 reply discarded the stale bytes
    assert server.requests[0].startswith("bytes=")
    assert dest.read_bytes() == CONTENT


def test_full_part_416_refetches_from_scratch(server, tmp_path):
    server.fail_416 = 1
    dest = tmp_path / "model.ckpt"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(CONTENT)  # full-size leftover makes the resume range unsatisfiable

    download.download_file(url_for(server), dest, backoff=0)

    assert dest.read_bytes() == CONTENT
    assert not part.exists()
    # first attempt ranged for the full file and got 416, clearing the part
    assert server.requests[0].startswith("bytes=")
    # the retry sent no range and pulled a fresh 200
    assert server.requests[1] is None


def test_progress_reports_cumulative_bytes_with_resume(server, tmp_path):
    server.drop_requests = 1
    dest = tmp_path / "model.ckpt"
    calls = []

    download.download_file(
        url_for(server),
        dest,
        backoff=0,
        progress=lambda done, total: calls.append((done, total)),
    )

    assert dest.read_bytes() == CONTENT
    # done counts from the file start, offset included
    resume_offset = len(CONTENT) // 2
    assert any(done > resume_offset for done, _ in calls)
    assert calls[-1] == (len(CONTENT), len(CONTENT))


def test_exhausts_attempts_and_raises(server, tmp_path):
    server.drop_requests = 99
    dest = tmp_path / "model.ckpt"

    with pytest.raises(download.DownloadError):
        download.download_file(url_for(server), dest, attempts=3, backoff=0)

    assert not dest.exists()
    assert len(server.requests) == 3


def test_with_retries_survives_a_transient_failure():
    calls = []

    def flaky():
        calls.append(None)
        if len(calls) < 3:
            raise urllib.error.HTTPError(API, 504, "Gateway Timeout", None, None)
        return "ok"

    assert download.with_retries(API, flaky, backoff=0) == "ok"
    assert len(calls) == 3


def test_with_retries_names_the_url_when_it_gives_up():
    def always_fails():
        raise urllib.error.HTTPError(API, 504, "Gateway Timeout", None, None)

    with pytest.raises(download.DownloadError, match=API):
        download.with_retries(API, always_fails, attempts=2, backoff=0)


def test_with_retries_gives_up_on_a_404():
    calls = []

    def missing():
        calls.append(None)
        raise urllib.error.HTTPError(API, 404, "Not Found", None, None)

    with pytest.raises(download.DownloadError, match="404"):
        download.with_retries(API, missing, backoff=0)
    assert len(calls) == 1


def test_partuv_version_matches_pyproject():
    pyproject = REPO_ROOT / "engine" / "partuv" / "pyproject.toml"
    version = tomllib.loads(pyproject.read_text())["project"]["version"]
    install_src = (REPO_ROOT / "src" / "engines" / "partuv" / "install.py").read_text()
    match = re.search(r'PARTUV_VERSION = "([^"]+)"', install_src)
    assert match, "PARTUV_VERSION constant not found in install.py"
    assert match.group(1) == version
