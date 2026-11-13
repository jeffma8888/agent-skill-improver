"""A run lock that fails to release leaves the outcome as it is.

``apply``, ``rollback`` and ``daily`` hold the run lock (``RunLock`` in
``app/lock.py``) while they read or write the tree, and close its descriptor
when they let it go, or when they refuse before taking it. Measured before
(3.11 and 3.13, the unlock or the close made to report EIO, the descriptor
closed all the same, as Linux's close does whatever it reports): an ``apply``
refused at the write (the skill's text changed after its own check) exited 70,
"internal error: this is a bug in asi", its refusal lost, the record
``validated`` and no backup; an ``apply`` that applied the candidate (record
``applied``, skill written, one backup) exited 70 as well; so did a
``rollback`` that rolled it back (record ``rolled-back``, skill as before) and
a committed ``daily`` whose receipt and candidate were written; and ``apply``
and ``daily`` refused while another holder had the lock (exit 3, naming it, as
it is) exited 70 with the holder's name lost when the close of their own
descriptor reported the error. The lock was free after each: the next command in the
same process took it.

What the release reports as an ``OSError`` is now kept on the lock
(``release_error``, the unlock's if both fail, started again at each release)
and never raised, and the closes before a refusal report nothing: the outcome
is what the command says and exits with. Anything else raised during the
release (a stop signal's exception) is raised as it is, after the close: with
the first form of this change, a SIGTERM during the unlock in ``cli.main`` left
the descriptor open and the next ``daily`` in that process refused ``LockHeld``.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import io
import json
import os
import re
import signal
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app import lock as lock_module
from agent_skill_improver.app.lock import RunLock
from agent_skill_improver.errors import LockHeld
from tests.support import build_env

DAY, NEXT_DAY, THIRD_DAY = "2026-02-01", "2026-02-02", "2026-02-03"
OK, REFUSED = 0, 3
CID = f"CAND-{DAY}-01-tdd"
EIO = OSError(errno.EIO, "Input/output error")
STOPPED = 128 + signal.SIGTERM


def eio() -> BaseException:
    return OSError(EIO.errno, EIO.strerror)


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


def first_line(said: str) -> str:
    return next(line for line in said.splitlines() if line.strip())


class _OsWhoseCloseFails:
    """``os`` for the lock module, its close closing the descriptor, then failing."""

    def __init__(self, error: Callable[[], BaseException] = eio) -> None:
        self.closed: list[int] = []
        self.error = error

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    def close(self, fd: int) -> None:
        os.close(fd)
        self.closed.append(fd)
        raise self.error()


class _FcntlWhoseUnlockFails:
    """``fcntl`` for the lock module, its unlock failing (the lock left to the close)."""

    def __init__(self, error: Callable[[], BaseException] = eio) -> None:
        self.unlocks = 0
        self.error = error

    def __getattr__(self, name: str) -> Any:
        return getattr(fcntl, name)

    def flock(self, fd: int, operation: int) -> None:
        if operation == fcntl.LOCK_UN:
            self.unlocks += 1
            raise self.error()
        fcntl.flock(fd, operation)


class _FcntlStoppedAtTheUnlock:
    """``fcntl`` for the lock module, a SIGTERM arriving as the lock is unlocked."""

    def __init__(self) -> None:
        self.held: tuple[int, os.stat_result] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(fcntl, name)

    def flock(self, fd: int, operation: int) -> None:
        if operation == fcntl.LOCK_UN:
            self.held = (fd, os.fstat(fd))
            os.kill(os.getpid(), signal.SIGTERM)
        fcntl.flock(fd, operation)


def still_open(fd: int, was: os.stat_result) -> bool:
    """Whether ``fd`` still names the file it named, not one opened since."""
    try:
        now = os.fstat(fd)
    except OSError as exc:
        assert exc.errno == errno.EBADF, exc
        return False
    return (now.st_dev, now.st_ino) == (was.st_dev, was.st_ino)


@contextlib.contextmanager
def failing(release: str) -> Iterator[Any]:
    if release == "close":
        stand_in: Any = _OsWhoseCloseFails()
        with mock.patch.object(lock_module, "os", stand_in):
            yield stand_in
    else:
        stand_in = _FcntlWhoseUnlockFails()
        with mock.patch.object(lock_module, "fcntl", stand_in):
            yield stand_in


RELEASES = ("close", "unlock")


class Validated:
    """A committed run's one validated candidate, its token taken."""

    def __init__(self, base: Path) -> None:
        self.base = base.resolve()
        self.config = build_env(self.base / "env", seed="c232")
        self.state = self.base / "env" / "state"
        rc, said = run(self.config, "daily", "--commit", "--day", DAY)
        assert rc == OK, said
        (self.record,) = sorted(self.state.glob("candidates/*.json"))
        assert self.record.stem == CID, self.record.stem
        self.skill = Path(json.loads(self.record.read_text())["target"]["canonical_path"])
        rc, shown = run(self.config, "review", CID)
        found = re.search(r"--approve (\S+)", shown)
        assert rc == OK and found, shown
        self.token = found.group(1)
        self.original = self.skill.read_bytes()
        (self.lock_path,) = [p for p in (self.state / "locks").iterdir() if p.is_file()]

    def apply(self) -> tuple[int, str]:
        return run(self.config, "apply", CID, "--approve", self.token, "--i-reviewed-this")

    def apply_with_the_text_changed_after_its_check(self) -> tuple[int, str]:
        real = FilesystemSkillStore.mirror_paths_of
        changed = self.original + b"\nchanged while apply ran\n"

        def change_then_list(store: Any, path: Path) -> tuple[str, ...]:
            if self.skill.read_bytes() == self.original:
                self.skill.write_bytes(changed)
            return real(store, path)

        with mock.patch.object(FilesystemSkillStore, "mirror_paths_of", change_then_list):
            return self.apply()

    def status(self) -> str:
        return str(json.loads(self.record.read_text())["status"])

    def backups(self) -> list[Path]:
        return sorted(self.state.glob("backups/*"))

    def the_lock_is_free(self) -> bool:
        try:
            with RunLock(self.lock_path, owner="pid=1 the next holder"):
                return True
        except LockHeld:
            return False

    @contextlib.contextmanager
    def held_by_another(self) -> Iterator[None]:
        holder = os.open(self.lock_path, os.O_RDWR)
        try:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(holder, 0)
            os.write(holder, b"daily pid=4242\n")
            yield
        finally:
            os.close(holder)


class AFailedReleaseLeavesTheOutcomeTests(unittest.TestCase):
    def test_an_apply_refused_at_the_write_keeps_its_refusal(self) -> None:
        for release in RELEASES:
            with self.subTest(release=release), tempfile.TemporaryDirectory() as tmp:
                case = Validated(Path(tmp))
                with failing(release) as stand_in:
                    rc, said = case.apply_with_the_text_changed_after_its_check()
                self.assertEqual(rc, REFUSED, said)
                self.assertTrue(
                    first_line(said).startswith(f"BaseHashMismatch: refusing to apply {CID}: "),
                    said,
                )
                self.assertNotIn("internal error", said)
                self.assertNotIn("Traceback", said)
                self.assertTrue(stand_in.closed if release == "close" else stand_in.unlocks)
                self.assertEqual(case.status(), "validated")
                self.assertEqual(case.backups(), [])
                self.assertTrue(case.the_lock_is_free())
                case.skill.write_bytes(case.original)
                rc, said = case.apply()
                self.assertEqual(rc, OK, said)
                self.assertEqual(case.status(), "applied")

    def test_an_apply_that_applies_exits_0(self) -> None:
        for release in RELEASES:
            with self.subTest(release=release), tempfile.TemporaryDirectory() as tmp:
                case = Validated(Path(tmp))
                with failing(release):
                    rc, said = case.apply()
                self.assertEqual(rc, OK, said)
                self.assertNotIn("internal error", said)
                self.assertEqual(case.status(), "applied")
                self.assertNotEqual(case.skill.read_bytes(), case.original)
                self.assertEqual(len(case.backups()), 1)
                self.assertTrue(case.the_lock_is_free())

    def test_a_rollback_that_rolls_back_exits_0(self) -> None:
        for release in RELEASES:
            with self.subTest(release=release), tempfile.TemporaryDirectory() as tmp:
                case = Validated(Path(tmp))
                rc, said = case.apply()
                self.assertEqual(rc, OK, said)
                with failing(release):
                    rc, said = run(case.config, "rollback", CID)
                self.assertEqual(rc, OK, said)
                self.assertNotIn("internal error", said)
                self.assertEqual(case.status(), "rolled-back")
                self.assertEqual(case.skill.read_bytes(), case.original)
                self.assertTrue(case.the_lock_is_free())

    def test_a_committed_daily_exits_as_it_would(self) -> None:
        for release in RELEASES:
            with self.subTest(release=release), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp).resolve()
                config = build_env(base / "env", seed="c232")
                with failing(release):
                    rc, said = run(config, "daily", "--commit", "--day", DAY)
                self.assertEqual(rc, OK, said)
                self.assertNotIn("internal error", said)
                self.assertEqual(len(list((base / "env" / "state").glob("candidates/*.json"))), 1)
                rc, said = run(config, "daily", "--commit", "--day", NEXT_DAY)
                self.assertEqual(rc, OK, said)

    def test_a_refusal_while_another_holds_the_lock_names_the_holder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            for argv in (
                ("apply", CID, "--approve", case.token, "--i-reviewed-this"),
                ("daily", "--commit", "--day", NEXT_DAY),
            ):
                with self.subTest(command=argv[0]):
                    with case.held_by_another(), failing("close") as stand_in:
                        rc, said = run(case.config, *argv)
                    self.assertEqual(rc, REFUSED, said)
                    self.assertTrue(first_line(said).startswith("LockHeld: "), said)
                    self.assertIn("(daily pid=4242)", first_line(said))
                    self.assertEqual(len(stand_in.closed), 1)
                    self.assertEqual(case.status(), "validated")
            rc, said = run(case.config, "daily", "--commit", "--day", THIRD_DAY)
            self.assertEqual(rc, OK, said)

    def test_a_stop_at_the_unlock_ends_the_run_and_lets_the_lock_go(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_env(Path(tmp).resolve() / "env", seed="c232")
            stand_in = _FcntlStoppedAtTheUnlock()
            try:
                with mock.patch.object(lock_module, "fcntl", stand_in):
                    rc, said = run(config, "daily", "--commit", "--day", DAY)
                self.assertEqual(rc, STOPPED, said)
                self.assertTrue(first_line(said).startswith("stopped by SIGTERM"), said)
                assert stand_in.held is not None, "the lock was never unlocked"
                self.assertFalse(still_open(*stand_in.held), "the descriptor was left open")
                rc, said = run(config, "daily", "--commit", "--day", NEXT_DAY)
                self.assertEqual(rc, OK, said)
            finally:
                if stand_in.held is not None and still_open(*stand_in.held):
                    os.close(stand_in.held[0])


class _OsFailingAt(_OsWhoseCloseFails):
    """The same, and one more call failing as ``__enter__`` makes it."""

    def __init__(self, at: str) -> None:
        super().__init__()
        self.at = at

    def fstat(self, fd: int) -> os.stat_result:
        if self.at == "fstat":
            raise OSError(errno.EACCES, "Permission denied")
        return os.fstat(fd)

    def fchmod(self, fd: int, mode: int) -> None:
        if self.at == "fchmod":
            raise OSError(errno.EPERM, "Operation not permitted")
        if self.at != "mode":
            os.fchmod(fd, mode)

    def write(self, fd: int, data: bytes) -> int:
        if self.at == "write":
            raise OSError(errno.ENOSPC, "No space left on device")
        return os.write(fd, data)


class EachRefusalBeforeTheLockIsTakenSurvivesAFailedCloseTests(unittest.TestCase):
    def test_each_refusal_is_raised_and_not_the_close(self) -> None:
        cases = (
            ("fstat", "cannot stat it"),
            ("fifo", "the lock leaf is not a regular file"),
            ("fchmod", "cannot enforce mode 0600"),
            ("mode", "mode is 0o644, want 0600"),
            ("held", "another asi process holds"),
            ("write", "No space left on device"),
        )
        for at, words in cases:
            with self.subTest(at=at), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "daily.lock"
                holder = None
                if at == "fifo":
                    os.mkfifo(path)
                elif at == "mode":
                    path.write_text("")
                    path.chmod(0o644)
                elif at == "held":
                    holder = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
                    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
                stand_in = _OsFailingAt(at)
                try:
                    with mock.patch.object(lock_module, "os", stand_in):
                        with self.assertRaises(Exception) as caught:
                            with RunLock(path):
                                self.fail("the lock was taken")
                finally:
                    if holder is not None:
                        os.close(holder)
                self.assertIn(words, str(caught.exception))
                self.assertNotEqual(getattr(caught.exception, "errno", None), errno.EIO)
                self.assertEqual(len(stand_in.closed), 1, "its descriptor was closed once")


class TheLockKeepsWhatItsReleaseReportedTests(unittest.TestCase):
    def test_the_release_error_is_kept_and_the_lock_is_free(self) -> None:
        for release in RELEASES:
            with self.subTest(release=release), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "locks" / "daily.lock"
                lock = RunLock(path, owner="pid=1 this one")
                with failing(release):
                    with lock:
                        pass
                self.assertIsInstance(lock.release_error, OSError)
                assert lock.release_error is not None
                self.assertEqual(lock.release_error.errno, errno.EIO)
                with RunLock(path, owner="pid=2 the next one"):
                    self.assertEqual(path.read_text(encoding="utf-8"), "pid=2 the next one\n")

    def test_a_failed_unlock_still_closes_the_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "daily.lock"
            lock = RunLock(path)
            with failing("unlock"):
                lock.__enter__()
                fd = lock._fd
                assert fd is not None
                lock.__exit__(None, None, None)
            with self.assertRaises(OSError) as caught:
                os.fstat(fd)
            self.assertEqual(caught.exception.errno, errno.EBADF)

    def test_a_release_after_a_failed_one_keeps_nothing(self) -> None:
        for release in RELEASES:
            with self.subTest(release=release), tempfile.TemporaryDirectory() as tmp:
                lock = RunLock(Path(tmp) / "daily.lock")
                with failing(release):
                    with lock:
                        pass
                self.assertIsInstance(lock.release_error, OSError)
                with lock:
                    pass
                self.assertIsNone(lock.release_error)

    def test_with_the_unlock_and_the_close_both_failing_the_unlock_is_kept(self) -> None:
        def enospc() -> BaseException:
            return OSError(errno.ENOSPC, "No space left on device")

        with tempfile.TemporaryDirectory() as tmp:
            lock = RunLock(Path(tmp) / "daily.lock")
            unlock, close = _FcntlWhoseUnlockFails(), _OsWhoseCloseFails(enospc)
            with mock.patch.object(lock_module, "fcntl", unlock):
                with mock.patch.object(lock_module, "os", close):
                    with lock:
                        pass
            self.assertEqual((unlock.unlocks, len(close.closed)), (1, 1))
            assert lock.release_error is not None
            self.assertEqual(lock.release_error.errno, errno.EIO)

    def test_a_release_that_reports_nothing_keeps_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock = RunLock(Path(tmp) / "daily.lock")
            with lock:
                pass
            self.assertIsNone(lock.release_error)
            with lock:
                pass
            self.assertIsNone(lock.release_error)

    def test_a_second_release_closes_nothing(self) -> None:
        # Released twice, the lock must not close a descriptor it no longer owns:
        # the number may already name another open file.
        with tempfile.TemporaryDirectory() as tmp:
            lock = RunLock(Path(tmp) / "daily.lock")
            lock.__enter__()
            lock.__exit__(None, None, None)
            other = os.open(Path(tmp) / "other", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                lock.__exit__(None, None, None)
                os.fstat(other)
            finally:
                os.close(other)
            self.assertIsNone(lock.release_error)

    def test_an_error_that_is_no_os_error_is_raised_and_the_lock_let_go(self) -> None:
        # Raised during the unlock, or by the close once it has closed (as a
        # stop signal's handler raises after the call it arrived in): either
        # way it is raised as it is, and the descriptor is closed.
        for release in RELEASES:
            with self.subTest(release=release), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "daily.lock"
                lock = RunLock(path)
                lock.__enter__()
                fd = lock._fd
                assert fd is not None
                was = os.fstat(fd)
                try:
                    stand_in: Any
                    if release == "close":
                        stand_in = _OsWhoseCloseFails(KeyboardInterrupt)
                        patched = mock.patch.object(lock_module, "os", stand_in)
                    else:
                        stand_in = _FcntlWhoseUnlockFails(KeyboardInterrupt)
                        patched = mock.patch.object(lock_module, "fcntl", stand_in)
                    with patched, self.assertRaises(KeyboardInterrupt):
                        lock.__exit__(None, None, None)
                    self.assertFalse(still_open(fd, was), "the descriptor was left open")
                    with RunLock(path, owner="pid=2 the next one"):
                        pass
                finally:
                    if still_open(fd, was):
                        os.close(fd)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
