"""Single-holder run lock.

``flock`` rather than a pid file: the kernel releases it when the process dies,
so a crashed run does not leave a stale lock that a human has to clear at 2am.
The lock file keeps the holder's identity for diagnosis only -- it is never the
thing that grants or denies the lock.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from types import TracebackType
from typing import Any

from ..errors import LockHeld, SafetyRefusal


class RunLock:
    def __init__(self, path: Path, *, owner: str = "") -> None:
        self.path = Path(path)
        self.owner = owner or f"pid={os.getpid()}"
        self._fd: int | None = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if not nofollow and self.path.is_symlink():
            raise SafetyRefusal(
                f"refusing to use {self.path} as a run lock: the lock leaf is a symlink"
            )
        try:
            fd = os.open(str(self.path), flags | nofollow, 0o600)
        except OSError as exc:
            if self.path.is_symlink():
                raise SafetyRefusal(
                    f"refusing to use {self.path} as a run lock: the lock leaf is a symlink"
                ) from exc
            raise
        # The 0o600 above applies only when the file is CREATED, and it is
        # further masked by the umask. An existing lock file from a run under a
        # permissive umask keeps its old mode, so set it explicitly. It carries
        # only a pid, but the state tree has one permission policy and an
        # exception in it is a thing somebody has to remember.
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            try:
                held = os.read(fd, 400).decode("utf-8", errors="replace").strip()
            except OSError:
                held = ""
            os.close(fd)
            raise LockHeld(
                f"another run holds {self.path}"
                + (f" ({held})" if held else "")
                + ". Overlapping runs would double-count records and race on the checkpoint."
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"{self.owner}\n".encode("utf-8"))
        os.fsync(fd)
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
