import http.client
import time
import urllib.error
import urllib.request
from pathlib import Path

# github releases honor Range, so a dropped transfer resumes from the .part
# file instead of restarting from zero (the ai checkpoint is over a gigabyte)
_CHUNK = 1 << 20
_ATTEMPTS = 4
_BACKOFF = 2.0
# the only 4xx a retry can clear: a stale range and rate limiting
_RETRIED_CLIENT_CODES = (416, 429)


class DownloadError(RuntimeError):
    pass


def _worth_retrying(error):
    if not isinstance(error, urllib.error.HTTPError):
        return True
    return error.code < 400 or error.code >= 500 or error.code in _RETRIED_CLIENT_CODES


def with_retries(url, call, attempts=_ATTEMPTS, backoff=_BACKOFF):
    """Run call until it returns, waiting longer after each failure.

    url is only for the error message.
    """
    error = None
    tried = 0
    while tried < attempts:
        if tried:
            time.sleep(backoff * tried)
        tried += 1
        try:
            return call()
        except (OSError, http.client.IncompleteRead, DownloadError) as e:
            error = e
            if not _worth_retrying(e):
                break
    tail = f" after {tried} attempts" if tried > 1 else ""
    raise DownloadError(f"failed to fetch {url}{tail}: {error}")


def download_file(
    url, dest, attempts=_ATTEMPTS, timeout=30, backoff=_BACKOFF, progress=None
):
    """Download url to dest, resuming a partial .part across retries.

    streams into <dest>.part and atomically replaces dest on success. each
    retry resumes with a Range request when bytes are already on disk; a 200
    reply to that range restarts the file from scratch. progress, if given, is
    called as progress(done_bytes, total_bytes_or_None) with done_bytes counting
    from the start of the file, including any resumed offset.
    """
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")

    def attempt():
        try:
            _fetch(url, part, timeout, progress)
        except urllib.error.HTTPError as error:
            # a full-size .part yields an unsatisfiable range, drop it to refetch
            if error.code == 416:
                part.unlink(missing_ok=True)
            raise
        part.replace(dest)

    with_retries(url, attempt, attempts, backoff)


def _fetch(url, part, timeout, progress=None):
    resume_from = part.stat().st_size if part.is_file() else 0
    request = urllib.request.Request(url)
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        resumed = response.status == 206
        if not resumed:
            # server ignored the range (or none was sent), start the file over
            resume_from = 0
        length = response.headers.get("Content-Length")
        expected = resume_from + int(length) if length is not None else None
        done = resume_from
        with open(part, "ab" if resumed else "wb") as file:
            while chunk := response.read(_CHUNK):
                file.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, expected)
    written = part.stat().st_size
    if expected is not None and written != expected:
        raise DownloadError(f"expected {expected} bytes from {url}, got {written}")
