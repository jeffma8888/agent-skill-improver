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
* stdin is fed from a helper thread so a child that answers before it reads,
  or never reads, cannot wedge the parent on a full pipe;
* stderr is drained concurrently and only a bounded tail is kept, because it is
  diagnostic, not data.

Callers map the two exceptions onto their own failure classes; this module
knows nothing about records, roles or sources.
"""

from __future__ import annotations

import os
import selectors
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

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

#: Pipe read size. Small enough that the overshoot past ``max_output_bytes``
#: before the kill is at most one chunk.
READ_CHUNK = 64 * 1024


class RunTimedOut(Exception):
    """The child did not finish within the deadline. It has been killed."""


class OutputOverflow(Exception):
    """The child wrote more than ``max_output_bytes`` to stdout. It has been killed."""


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
    operator's own session); a mapping replaces it entirely (a model role gets
    the allowlisted minimum). Raises ``FileNotFoundError`` if the executable is
    missing, :class:`RunTimedOut` or :class:`OutputOverflow` as described above.
    """
    deadline = time.monotonic() + timeout_seconds
    proc = subprocess.Popen(  # noqa: S603 - argv is constructed by the caller, never shell
        list(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=None if env is None else dict(env),
    )
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
    """Stop the child and reap it; never leave a zombie or a writer blocked on us."""
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        pass
