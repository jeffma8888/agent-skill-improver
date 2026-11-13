"""Single-holder run lock.

``flock`` rather than a pid file: the kernel releases it when the process dies,
so a crashed run does not leave a stale lock that a human has to clear at 2am.
The lock file keeps the holder's identity for diagnosis only -- it is never the
thing that grants or denies the lock.
"""

from __future__ import annotations

import errno
import fcntl
import os
import stat
from pathlib import Path
from types import TracebackType
from typing import Any

from ..domain.candidate import one_line_label
from ..domain.redaction import why_unreadable
from ..errors import LockHeld, SafetyRefusal


def _close_quietly(fd: int) -> OSError | None:
    """Close ``fd``, returning an error the close reports rather than raising it.

    Every close here comes after the lock's outcome is settled: a refusal about
    to be raised, or the work done under the lock. Closing the descriptor ends
    the lock whatever the close reports (Linux's close(2) frees the descriptor
    even when it returns an error, and so does macOS's, read from their manuals
    and macOS's source, not measured here; and the ``asi`` program exits at the
    end of each command, which frees it in any case, though a caller of
    ``cli.main`` may go on), so the outcome, not the close, is what the operator
    is told. Only an ``OSError`` is returned; anything else (a stop signal's
    exception) is raised as it is, the close done. Measured
    before (3.11 and 3.13, the close made to report EIO after closing the
    descriptor): ``apply`` and ``daily`` refused while another holder had the
    lock exited 70, "internal error", with the holder's name lost.
    """
    try:
        os.close(fd)
    except OSError as exc:
        return exc
    return None


class RunLock:
    def __init__(self, path: Path, *, owner: str = "") -> None:
        self.path = Path(path)
        self.owner = owner or f"pid={os.getpid()}"
        self._fd: int | None = None
        # What the latest release of the lock reported as an OSError, kept and
        # never raised; None after a release that reported nothing (see __exit__).
        self.release_error: OSError | None = None

    @property
    def _shown(self) -> str:
        r"""The lock path as a refusal names it: on one line, a line break spelled.

        The lock sits under ``[run].state_dir``, whose name may hold a line break
        (a launchd job carries such a path exactly). Measured before (3.11 and
        3.13): a directory at the lock path under a state directory ending
        ``\nconfig error: forged`` made ``daily`` print its refusal as 3 lines.
        """
        return one_line_label(str(self.path))

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # O_NONBLOCK guards the open itself: a regular file ignores it, and a
        # fifo precreated at the path would otherwise park the process here.
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NONBLOCK", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        # ``os.path.islink`` rather than ``Path.is_symlink``: the latter raises on
        # an unreadable parent, and this is asked inside an error handler.
        if not nofollow and os.path.islink(self.path):
            raise SafetyRefusal(
                f"refusing to use {self._shown} as a run lock: the lock leaf is a symlink"
            )
        try:
            fd = os.open(str(self.path), flags | nofollow, 0o600)
        except OSError as exc:
            if os.path.islink(self.path):
                raise SafetyRefusal(
                    f"refusing to use {self._shown} as a run lock: the lock leaf is a symlink"
                ) from exc
            # A directory at the path (``IsADirectoryError``) is the same refusal
            # as a fifo: not this tool's file, and the advice is to remove it. A
            # path this process cannot open for another reason (no space, a
            # read-only or unreadable directory) is refused by name too, with
            # no advice to remove anything, since nothing foreign is there.
            # Measured before: a directory escaped as the bare ``OSError`` and
            # the run ended with exit 70 and a traceback.
            foreign = exc.errno in (errno.EISDIR, errno.ENOTDIR, errno.ELOOP)
            raise SafetyRefusal(
                f"refusing to use {self._shown} as a run lock: it could not be opened "
                f"({why_unreadable(exc)})."
                + (" Remove the entry at that path if you did not put it there." if foreign else "")
            ) from exc
        # Judged before anything is done to it: a fifo or a device at the lock
        # path is not this tool's file, so its mode is not ours to set, a read
        # of it could wait forever and a write to it would go who knows where.
        try:
            kind = os.fstat(fd).st_mode
        except OSError as exc:
            _close_quietly(fd)
            raise SafetyRefusal(
                f"refusing to use {self._shown} as a run lock: cannot stat it "
                f"({why_unreadable(exc)})"
            ) from exc
        if not stat.S_ISREG(kind):
            _close_quietly(fd)
            raise SafetyRefusal(
                f"refusing to use {self._shown} as a run lock: the lock leaf is not a regular "
                "file. Remove the entry at that path if you did not put it there."
            )
        # The 0o600 above applies only when the file is CREATED, and it is
        # further masked by the umask. An existing lock file from a run under a
        # permissive umask keeps its old mode, so set it explicitly. It carries
        # only a pid, but the state tree has one permission policy and an
        # exception in it is a thing somebody has to remember.
        try:
            os.fchmod(fd, 0o600)
            actual_mode = os.fstat(fd).st_mode & 0o777
        except OSError as exc:
            _close_quietly(fd)
            raise SafetyRefusal(
                f"refusing to use {self._shown} as a run lock: cannot enforce mode 0600 "
                f"({why_unreadable(exc)})"
            ) from exc
        if actual_mode != 0o600:
            _close_quietly(fd)
            raise SafetyRefusal(
                f"refusing to use {self._shown} as a run lock: mode is {oct(actual_mode)}, want 0600"
            )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            try:
                held = os.read(fd, 400).decode("utf-8", errors="replace").strip()
            except OSError:
                held = ""
            _close_quietly(fd)
            # The holder is a run or a promotion; its owner line says which.
            # Spelled onto one line: it is read back from a file.
            raise LockHeld(
                f"another asi process holds {self._shown}"
                + (f" ({one_line_label(held)})" if held else "")
                + ". Overlapping runs would double-count records and race on the checkpoint, "
                "and a promotion during a run would land in the tree the run is reading."
            ) from exc
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{self.owner}\n".encode("utf-8"))
            os.fsync(fd)
        except BaseException:
            _close_quietly(fd)
            raise
        self._fd = fd
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release the lock; an ``OSError`` the release reports is kept, never raised.

        The unlock and the close come after the work under the lock is done or
        refused, and the close ends the lock (see ``_close_quietly``) after an
        unlock that failed in any way, so neither replaces that outcome; the
        error is kept in ``release_error``, the unlock's if both fail, and each
        release starts it again at None. Anything raised that is no ``OSError``
        (a stop signal's exception, during the unlock or the close) is raised as
        it is, after the close: measured with the first form of this change (3.11
        and 3.13), a SIGTERM during the unlock in ``cli.main`` left the descriptor
        open and the next ``daily`` in that process refused ``LockHeld``, where
        before it took the lock. Measured before (3.11 and 3.13, the unlock or the
        close made to report EIO, the descriptor closed all the same): an
        ``apply`` refused at the write exited 70, "internal error", with its
        refusal lost; an ``apply`` that applied the candidate (record applied,
        skill written, backup taken) exited 70 too, and so did a ``rollback``
        that rolled it back and a committed ``daily`` that had written its
        receipt; the lock was free after each (the next command took it).
        """
        fd, self._fd = self._fd, None
        if fd is None:
            return
        self.release_error = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as exc:
            self.release_error = exc
        finally:
            closed = _close_quietly(fd)
            self.release_error = self.release_error or closed

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "owner": self.owner}
