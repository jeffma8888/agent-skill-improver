"""Run a child process with its output bounded while it is read.

Both external commands this tool runs (the conversation CLI and each model
role) used ``subprocess.run(capture_output=True)`` and compared the length of
stdout to a cap afterwards. A command that streamed without end therefore cost
the parent that much memory before the refusal fired. A cap applied after the
resource is spent is a log line, not a limit.

This module is the one place that reads a child's pipes:

* ``max_output_bytes`` is enforced as stdout arrives; the child is killed the
  moment it is exceeded and :class:`OutputOverflow` is raised;
* the timeout is a wall-clock deadline over the whole exchange, including the
  stdin write, and a child that outlives it is killed and reaped
  (:class:`RunTimedOut`);
* the child is started in its own session, so stopping it stops its whole
  process group: a role command is routinely a wrapper (a shell, a launcher, an
  isolation tool) whose real worker is a grandchild, and killing the wrapper
  alone would leave that worker running past the deadline or the cap;
* stdin is fed from a helper thread so a child that answers before it reads,
  or never reads, cannot wedge the parent on a full pipe;
* stderr is drained concurrently and only a bounded tail is kept, because it is
  diagnostic, not data.
* whatever a caller quotes from the child into a failure message goes through
  ``quotable``: redacted with the transcript's credential shapes, then cut to
  the documented length, because that message becomes a persisted ``reason``.

Callers map the two exceptions onto their own failure classes; this module
knows nothing about records, roles or sources.
"""

from __future__ import annotations

import os
import selectors
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..domain.redaction import redact

#: How much of the child's stderr is kept for the failure message.
STDERR_TAIL_BYTES = 4096

#: How many characters of that tail a failure message quotes. The two adapters
#: that quote a child's stderr share this number so a refusal record never
#: carries more of a child's output than the operator docs say it does. It must
#: not exceed ``STDERR_TAIL_BYTES``; the test suite holds the two in order.
QUOTED_TAIL_CHARS = 500

#: How many characters of a role's unparseable stdout a failure message quotes.
#: The head rather than the tail: it shows what the reply started as (prose, a
#: fence, an error page) and no more. Stated in the privacy doc as well.
QUOTED_OUTPUT_CHARS = 120


def quotable(text: str, *, limit: int, from_end: bool) -> str:
    """Child output as a failure message may quote it: redacted, then bounded.

    Quoted output lands in a deferral or quarantine ``reason`` and from there in
    the receipt, the checkpoint and the scheduler log. A CLI that dies printing
    its own authorization header, or a role that echoes a key instead of JSON,
    would otherwise persist that credential in four places. Redaction runs over
    the whole captured text *before* the cut, so a key is masked as a whole
    rather than sliced into a fragment the shapes no longer recognise. Best
    effort, like every redaction here: the shapes are the ones the transcript
    redactor uses.
    """
    masked = redact(text).text
    return masked[-limit:] if from_end else masked[:limit]


#: Pipe read size. Small enough that the overshoot past ``max_output_bytes``
#: before the kill is at most one chunk.
READ_CHUNK = 64 * 1024

#: The longest wait a configured ``timeout_seconds`` may ask for: seven days.
#: The platform's timer bounds the wait on the child's pipes, and a wait past
#: it is refused by the platform mid-run (measured on macOS: from 9223372037
#: seconds kqueue raised ``TypeError`` and a daily run fell, exit 70). Seven
#: days is far inside such a bound and far past any call a daily run waits on.
MAX_TIMEOUT_SECONDS = 7 * 24 * 60 * 60


class RunTimedOut(Exception):
    """The child did not finish within the deadline. It has been killed."""


class OutputOverflow(Exception):
    """The child wrote more than ``max_output_bytes`` to stdout. It has been killed."""


def program_problem(program: str) -> str:
    """Why ``program`` could not be started as a subprocess, or "" when it can.

    Mirrors how the process will be started: a bare name is looked up on PATH
    and nowhere else (a file of that name in the working directory does not
    count); a path is taken as given and must be a regular file this user may
    execute. Measured before: a conversation command at mode 644, and a directory
    given by mistake, both passed a preflight that asked only whether the path
    existed, and the run then died as an internal error; a model role command
    at mode 644 passed ``doctor`` as ready and did the same.
    """
    if shutil.which(program) is not None:
        return ""
    path = Path(program)
    if os.sep not in program:
        return "not found on PATH"
    if not path.exists():
        return "not found"
    if path.is_dir():
        return "is a directory, not a program"
    if not path.is_file():
        return "is not a regular file"
    if not os.access(path, os.X_OK):
        mode = path.stat().st_mode & 0o777
        return f"exists but is not executable (mode {mode:03o}); chmod +x it"
    return ""


@dataclass(frozen=True)
class BoundedResult:
    stdout: bytes
    stderr_tail: bytes
    returncode: int


def run_bounded(
    argv: Sequence[str],
    *,
    stdin: bytes = b"",
    env: Mapping[str, str] | None = None,
    timeout_seconds: float,
    max_output_bytes: int,
) -> BoundedResult:
    """Run ``argv`` with bounded output and a wall-clock deadline.

    ``env=None`` inherits the parent environment (the conversation CLI needs the
    operator's own variables and credentials); a mapping replaces it entirely (a
    model role gets the allowlisted minimum). The POSIX session is never
    inherited: the child leads its own, with no controlling terminal, so a tool
    that opened ``/dev/tty`` to prompt would fail at once rather than wait for
    an answer nobody can give through these pipes. Raises ``FileNotFoundError``
    if the executable is missing, :class:`RunTimedOut` or :class:`OutputOverflow`
    as described above.
    """
    deadline = time.monotonic() + timeout_seconds
    proc = subprocess.Popen(  # noqa: S603 - argv is constructed by the caller, never shell
        list(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=None if env is None else dict(env),
        start_new_session=True,
    )
    try:
        return _read_bounded(proc, stdin, deadline, timeout_seconds, max_output_bytes)
    except BaseException:
        # Whatever ends this call before the loop's own ``finally`` is set up
        # (a stop signal, a thread that cannot start) ends the child too.
        if proc.poll() is None:
            _kill(proc)
        raise


def _read_bounded(
    proc: subprocess.Popen[bytes],
    stdin: bytes,
    deadline: float,
    timeout_seconds: float,
    max_output_bytes: int,
) -> BoundedResult:
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    stdin_pipe, stdout_pipe, stderr_pipe = proc.stdin, proc.stdout, proc.stderr

    def feed() -> None:
        try:
            if stdin:
                stdin_pipe.write(stdin)
                stdin_pipe.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                stdin_pipe.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

    writer = threading.Thread(target=feed, name="bounded-stdin", daemon=True)
    writer.start()

    out = bytearray()
    err = bytearray()
    sel = selectors.DefaultSelector()
    sel.register(stdout_pipe, selectors.EVENT_READ, "out")
    sel.register(stderr_pipe, selectors.EVENT_READ, "err")
    failure: Exception | None = None
    try:
        open_streams = 2
        while open_streams and failure is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = RunTimedOut(f"timed out after {timeout_seconds:g}s")
                break
            for key, _events in sel.select(timeout=remaining):
                stream = key.fileobj
                chunk = os.read(stream.fileno(), READ_CHUNK)  # type: ignore[union-attr]
                if not chunk:
                    sel.unregister(stream)
                    open_streams -= 1
                    continue
                if key.data == "out":
                    out += chunk
                    if len(out) > max_output_bytes:
                        failure = OutputOverflow(
                            f"produced more than {max_output_bytes} bytes on stdout"
                        )
                        break
                else:
                    err += chunk
                    if len(err) > STDERR_TAIL_BYTES:
                        del err[: len(err) - STDERR_TAIL_BYTES]
        if failure is not None:
            _kill(proc)
            raise failure
        remaining = max(0.0, deadline - time.monotonic())
        try:
            returncode = proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            _kill(proc)
            raise RunTimedOut(f"timed out after {timeout_seconds:g}s") from exc
    finally:
        sel.close()
        for stream in (stdout_pipe, stderr_pipe):
            try:
                stream.close()
            except OSError:
                pass
        if proc.poll() is None:
            _kill(proc)
        writer.join(timeout=1.0)
    return BoundedResult(stdout=bytes(out), stderr_tail=bytes(err), returncode=returncode)


def _kill(proc: subprocess.Popen[bytes]) -> None:
    """Stop the child and everything it started, then reap it.

    The child leads its own process group (``start_new_session``), so the
    signal goes to the group: the wrapper and every worker it spawned. The
    direct kill is kept as the fallback for a platform without process groups
    or a child that already changed group.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        pass
