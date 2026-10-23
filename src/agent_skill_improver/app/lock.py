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
from ..errors import LockHeld, SafetyRefusal


class RunLock:
    def __init__(self, path: Path, *, owner: str = "") -> None:
        self.path = Path(path)
        self.owner = owner or f"pid={os.getpid()}"
        self._fd: int | None = None

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
                f"refusing to use {self._shown} as a run lock: it could not be opened ({exc})."
                + (" Remove the entry at that path if you did not put it there." if foreign else "")
            ) from exc
        # Judged before anything is done to it: a fifo or a device at the lock
        # path is not this tool's file, so its mode is not ours to set, a read
        # of it could wait forever and a write to it would go who knows where.
        try:
            kind = os.fstat(fd).st_mode
        except OSError as exc:
            os.close(fd)
            raise SafetyRefusal(
                f"refusing to use {self._shown} as a run lock: cannot stat it ({exc})"
            ) from exc
        if not stat.S_ISREG(kind):
            os.close(fd)
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
            os.close(fd)
            raise SafetyRefusal(
                f"refusing to use {self._shown} as a run lock: cannot enforce mode 0600 ({exc})"
            ) from exc
        if actual_mode != 0o600:
            os.close(fd)
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
            os.close(fd)
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
            os.close(fd)
            raise
        self._fd = fd
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "owner": self.owner}
