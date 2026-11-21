"""A skill document swapped after the listing for an entry of a refused kind is passed over.

Discovery (``FilesystemSkillStore.snapshots`` in ``adapters/skill_fs.py``) lists each
``<root>/<skill>/SKILL.md`` and then reads it. Measured before (3.13.0 and 3.11.15), with one
of three swaps made between the listing and the read (the listed leaf replaced by a link out of
the root, the leaf replaced by a fifo, or its skill directory replaced by a link out of the
root), ``skills``, ``doctor``, ``daily`` (text and JSON), ``review`` and ``apply`` exited 70
having read the outside bytes or waited past a 5 s alarm, and ``rollback`` of an applied
candidate refused only after reading them or waited: 21 of 21, those three shapes by those
seven invocations. A link to another skill's document inside the root was a different defect:
with links not followed, the leaf replaced by a link to ``code-review/SKILL.md`` was read
through and inventoried as a second canonical copy of ``code-review``, under the wrong skill
identity, instead of being passed over; nothing exited 70 or read outside bytes
(``skills``/``doctor`` 0, ``daily`` 5, ``review``/``apply``/``rollback`` 3, refusing
``code-review`` as ambiguous).

Now each listed document is read through handles pinned from its root (``_read_discovered``):
each directory below the root is opened ``O_DIRECTORY|O_NOFOLLOW`` relative to the one above,
the document ``O_NOFOLLOW|O_NONBLOCK`` relative to its directory, and judged by ``fstat`` on
that handle. What is refused is the kind of entry, not every replacement: a symlink or a
non-directory on that walk, and a leaf that is not a regular file (a fifo, a socket, a
directory, ...); while links are not followed, a link anywhere on the path below the root is
refused. Any refusal passes the document over with why, as a copy of its skill, so its
canonical slot is withheld and ``review``/``apply``/``rollback`` refuse it by name; nothing
raises and nothing waits. What is read, not refused: an in-root regular file put in the
document's place before its open; and, with links followed, a link that resolves inside the
root at the one resolve, read at its resolved target. In both cases the bytes come from inside
the root.

The refusal and swap cases in tests 1-15 are each passed over. Of the swaps tested, test 16 is
the one that is read, and says so: with links followed, a link re-pointed after its one resolve
is read from the in-root file it named then, and the file it names afterwards is never opened.
Test 17 pins that containment is by path below the root, not by inode: a hard link at the
document's path is an in-root entry and is read. Test 18 holds links followed to the same
contract when the listed document is swapped for a link to the root itself, or to a regular
file directly in it: neither has a skill directory, and each is passed over by name as a copy
of ``tdd``. Measured on the first c248a fix: an IndexError from discovery, and all six commands
exit 70.

Every "no outside read" assertion uses a spy on ``os.open`` (``fstat`` of each descriptor,
counted when it is the outside directory or a file in it), ``os.read`` and ``Path.read_bytes``
(counted when the outside marker is read). Every command that could wait is run under a 5 s
alarm that fails the test.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import signal
import stat
import tempfile
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters import skill_fs
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app.wiring import build_skill_store
from agent_skill_improver.config import load_config
from agent_skill_improver.domain.skills import group_for
from agent_skill_improver.hashing import sha256_text

from tests.support import TDD_SKILL, build_env

MARK = b"OUTSIDE-MARKER"
NOT_FOLLOWED = "a symlink when it was read; this store does not follow links"
A_FIFO = "not a regular file when it was read (a fifo)"
BELOW_TAIL = ", which was a symlink or not a directory when the document was read"
OUT_OF_ROOT = "a symlink that resolved outside its root when it was read"
LOOP_AT_READ = "a path that could not be resolved when it was read (a symlink loop)"
LOOP_AT_WALK = "a path that could not be resolved (a symlink loop)"
NOT_READ = (
    "not read: this platform cannot open a document through a directory handle pinned to its root"
)
SHALLOW = "a symlink that resolved to its root or to an entry directly in it when it was read"
SHALLOW_UNFOLLOWED = "a document listed at its root or directly in it, not in a skill directory"
LOOSE = "notes.md"
WITHHELD = "the copy in editable root 'canonical' was passed over by discovery"
ALARM = "LIVE SKILL MUTATION DETECTED"
ORIGINAL_ITER = FilesystemSkillStore._iter_skill_files
ORIGINAL_RESOLVE = Path.resolve
ORIGINAL_IS_SYMLINK = Path.is_symlink
ORIGINAL_OPEN = os.open
ORIGINAL_READ = os.read
ORIGINAL_READ_BYTES = Path.read_bytes
ORIGINAL_INIT = FilesystemSkillStore.__init__


class _Hang(BaseException):
    """Raised by the alarm; turned into an assertion by ``_bounded``."""


def _on_alarm(signum: int, frame: object) -> None:
    raise _Hang()


@contextlib.contextmanager
def _bounded(seconds: int = 5) -> Iterator[None]:
    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        yield
    except _Hang:
        raise AssertionError(f"waited past the {seconds} s alarm") from None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _resolve_raises_on_a_loop() -> bool:
    """3.11's ``Path.resolve()`` raises ``RuntimeError`` on a loop; 3.13's does not."""
    with tempfile.TemporaryDirectory() as tmp:
        loop = Path(tmp) / "loop"
        loop.symlink_to(loop)
        try:
            (loop / "SKILL.md").resolve()
        except RuntimeError:
            return True
        except OSError:
            return False
    return False


RESOLVE_RAISES_ON_A_LOOP = _resolve_raises_on_a_loop()


class _OutsideSpy:
    """Counts descriptors opened on the outside directory or a file in it, and marker reads."""

    def __init__(self, outside: Path) -> None:
        self.inodes = {(outside.stat().st_dev, outside.stat().st_ino)}
        for entry in outside.rglob("*"):
            found = entry.lstat()
            self.inodes.add((found.st_dev, found.st_ino))
        self.opened = 0
        self.marker_read = False
        self._stack = contextlib.ExitStack()

    def _open(self, path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        fd = ORIGINAL_OPEN(path, flags, mode, dir_fd=dir_fd)
        found = os.fstat(fd)
        self.opened += int((found.st_dev, found.st_ino) in self.inodes)
        return fd

    def _read(self, fd: int, n: int) -> bytes:
        raw = ORIGINAL_READ(fd, n)
        self.marker_read |= MARK in raw
        return raw

    def _read_bytes(self, path: Path) -> bytes:
        raw = ORIGINAL_READ_BYTES(path)
        self.marker_read |= MARK in raw
        return raw

    def __enter__(self) -> _OutsideSpy:
        self._stack.enter_context(mock.patch.object(os, "open", self._open))
        self._stack.enter_context(mock.patch.object(os, "read", self._read))
        spy = self

        def read_bytes(path: Path) -> bytes:
            return spy._read_bytes(path)

        self._stack.enter_context(mock.patch.object(Path, "read_bytes", read_bytes))
        return self

    def __exit__(self, *exc: object) -> None:
        self._stack.close()


def _snapshots(store: FilesystemSkillStore) -> list[Any]:
    """``store.snapshots(refresh=True)``; an exception from it fails the test."""
    try:
        return list(store.snapshots(refresh=True))
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"discovery raised {type(exc).__name__}: {exc}") from None


def _swapper(kind: str, live: Path, outside: Path) -> Callable[[], None]:
    """The swap of ``kind``, made on ``live`` (``<root>/tdd/SKILL.md``)."""

    def swap() -> None:
        if kind == "dir-out":
            live.parent.rename(live.parent.with_name("tdd.real"))
            live.parent.symlink_to(outside, target_is_directory=True)
        elif kind == "parent-loop":
            live.parent.rename(live.parent.with_name("tdd.real"))
            live.parent.symlink_to(live.parent)
        elif kind == "fifo-aside":
            live.rename(live.with_name("SKILL.md.aside"))
            os.mkfifo(live)
        else:
            live.unlink()
            if kind == "link-out":
                live.symlink_to(outside / "SKILL.md")
            elif kind == "link-in":
                live.symlink_to(live.parent.parent / "code-review" / "SKILL.md")
            elif kind == "fifo":
                os.mkfifo(live)
            else:
                raise AssertionError(kind)

    return swap


@contextlib.contextmanager
def _after_the_listing(live: Path, swap: Callable[[], None]) -> Iterator[dict[str, int]]:
    """Make ``swap`` once, right after the walk has listed ``live``."""
    seen = {"swapped": 0}

    def race(self: FilesystemSkillStore, *args: Any, **kwargs: Any) -> Any:
        found = ORIGINAL_ITER(self, *args, **kwargs)
        if live in found and not seen["swapped"]:
            swap()
            seen["swapped"] = 1
        return found

    with mock.patch.object(FilesystemSkillStore, "_iter_skill_files", race):
        yield seen


def _restore(live: Path, original: bytes) -> None:
    real = live.parent.with_name("tdd.real")
    if live.parent.is_symlink():
        live.parent.unlink()
        real.rename(live.parent)
    if live.is_symlink() or (live.exists() and not live.is_file()):
        live.unlink()
    aside = live.with_name("SKILL.md.aside")
    if aside.exists():
        aside.unlink()
    if not live.exists():
        live.write_bytes(original)


def _run(config: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue(), err.getvalue()


def _find_key(obj: Any, key: str) -> Iterator[Any]:
    if isinstance(obj, dict):
        for name, value in obj.items():
            if name == key:
                yield value
            yield from _find_key(value, key)
    elif isinstance(obj, list):
        for value in obj:
            yield from _find_key(value, key)


def _tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    if directory.exists():
        for entry in sorted(directory.rglob("*")):
            if entry.is_file() and not entry.is_symlink():
                digest.update(str(entry.relative_to(directory)).encode() + b"\0")
                digest.update(ORIGINAL_READ_BYTES(entry))
    return digest.hexdigest()


class _Tree:
    """A fresh environment, its canonical ``tdd/SKILL.md`` and an outside directory."""

    def __init__(self, base: Path, name: str) -> None:
        self.config = build_env(base / name, dry_run=False, seed="c248a")
        self.root = base / name / "skills-canonical"
        self.live = self.root / "tdd" / "SKILL.md"
        self.mirror = base / name / "skills-mirror" / "tdd" / "SKILL.md"
        self.backups = base / name / "state" / "backups"
        self.outside = base / f"{name}-outside"
        self.outside.mkdir()
        (self.outside / "SKILL.md").write_bytes(TDD_SKILL.encode() + b"\n" + MARK + b"\n")
        self.original = self.live.read_bytes()

    def store(self, *, follow: bool = False) -> FilesystemSkillStore:
        store = build_skill_store(load_config(self.config))
        assert isinstance(store, FilesystemSkillStore)
        store._follow_symlinks = follow
        return store

    def swap(self, kind: str) -> Callable[[], None]:
        return _swapper(kind, self.live, self.outside)


class _StoreCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self._count = 0

    def tree(self) -> _Tree:
        self._count += 1
        return _Tree(self.base, f"t{self._count}")

    def passed_over_after(
        self, kind: str, *, follow: bool = False
    ) -> tuple[_Tree, FilesystemSkillStore, list[Any], list[Any], _OutsideSpy]:
        tree = self.tree()
        store = tree.store(follow=follow)
        with _OutsideSpy(tree.outside) as spy, _bounded():
            with _after_the_listing(tree.live, tree.swap(kind)) as seen:
                snaps = _snapshots(store)
        self.assertEqual(seen["swapped"], 1, kind)
        passed = [p for p in store.passed_over() if p.path == tree.live]
        return tree, store, snaps, passed, spy

    def assert_passed_over(
        self, kind: str, reason: str | re.Pattern[str], *, follow: bool = False
    ) -> list[Any]:
        tree, _store, snaps, passed, spy = self.passed_over_after(kind, follow=follow)
        where = f"{kind} follow={follow}"
        self.assertEqual([s for s in snaps if s.path == tree.live], [], where)
        self.assertEqual(
            [s for s in snaps if s.skill_id == "tdd" and s.root_id == "canonical"], [], where
        )
        self.assertEqual(len(passed), 1, where)
        if isinstance(reason, str):
            self.assertEqual(passed[0].reason, reason, where)
        else:
            self.assertRegex(passed[0].reason, reason, where)
        self.assertEqual(passed[0].skill_id, "tdd", where)
        self.assertEqual(spy.opened, 0, f"{where}: the outside directory was opened")
        self.assertFalse(spy.marker_read, f"{where}: the outside bytes were read")
        return passed


BELOW = re.compile(r"^below .*/skills-canonical/tdd" + re.escape(BELOW_TAIL) + "$")


class TheSwapsAreRefusedAtTheStoreTests(_StoreCase):
    def test_a_leaf_swapped_for_a_link_out_of_the_root_is_passed_over_and_not_read(self) -> None:
        self.assert_passed_over("link-out", NOT_FOLLOWED)

    def test_a_leaf_swapped_for_a_link_to_another_skills_document_is_passed_over(self) -> None:
        tree, _store, snaps, passed, _spy = self.passed_over_after("link-in")
        self.assertEqual([s.skill_id for s in snaps if s.path == tree.live], [])
        self.assertEqual([p.reason for p in passed], [NOT_FOLLOWED])
        code_review = [s for s in snaps if s.skill_id == "code-review"]
        self.assertEqual([s.path.name for s in code_review], ["SKILL.md"])
        self.assertEqual(code_review[0].path.parent.name, "code-review")

    def test_a_leaf_swapped_for_a_fifo_is_passed_over_without_waiting(self) -> None:
        self.assert_passed_over("fifo", A_FIFO)

    def test_a_skill_directory_swapped_for_a_link_out_of_the_root_is_passed_over(self) -> None:
        self.assert_passed_over("dir-out", BELOW, follow=False)
        self.assert_passed_over("dir-out", OUT_OF_ROOT, follow=True)
        self.assert_passed_over("link-out", OUT_OF_ROOT, follow=True)

    def test_a_double_ancestor_swap_around_a_resolve_reads_nothing_outside(self) -> None:
        for follow in (False, True):
            tree = self.tree()
            store = tree.store(follow=follow)
            parent = tree.live.parent
            real, saved = parent.with_name("tdd.real"), parent.with_name("link.saved")
            swapped = {"swapped": 0}

            def swap() -> None:
                parent.rename(real)
                parent.symlink_to(tree.outside, target_is_directory=True)
                swapped["swapped"] = 1

            def resolve_with_the_real_directory_back(p: Path, *a: Any, **k: Any) -> Path:
                if swapped["swapped"] and str(p).startswith(str(parent) + os.sep):
                    parent.rename(saved)
                    real.rename(parent)
                    try:
                        return ORIGINAL_RESOLVE(p, *a, **k)
                    finally:
                        parent.rename(real)
                        saved.rename(parent)
                return ORIGINAL_RESOLVE(p, *a, **k)

            with _OutsideSpy(tree.outside) as spy, _bounded():
                with (
                    _after_the_listing(tree.live, swap),
                    mock.patch.object(Path, "resolve", resolve_with_the_real_directory_back),
                ):
                    snaps = _snapshots(store)
            where = f"follow={follow}"
            self.assertEqual(swapped["swapped"], 1, where)
            self.assertEqual(
                [s for s in snaps if s.skill_id == "tdd" and s.root_id == "canonical"], [], where
            )
            reasons = [p.reason for p in store.passed_over() if p.path == tree.live]
            self.assertEqual(len(reasons), 1, where)
            self.assertRegex(reasons[0], BELOW, where)
            self.assertEqual(spy.opened, 0, where)
            self.assertFalse(spy.marker_read, where)
        # The same double swap on the read itself, around the one resolve it may make.
        fn = self.base / "fn"
        root = fn / "root"
        inside = root / "tdd"
        inside.mkdir(parents=True)
        path = inside / "SKILL.md"
        path.write_bytes(b"INSIDE")
        outside = fn / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_bytes(MARK)
        real = root / "tdd.real"
        inside.rename(real)
        inside.symlink_to(outside, target_is_directory=True)
        saved_link = root / "link.saved"

        def resolve_around(p: Path, *a: Any, **k: Any) -> Path:
            if p == path:
                inside.rename(saved_link)
                real.rename(inside)
                try:
                    return ORIGINAL_RESOLVE(p, *a, **k)
                finally:
                    inside.rename(real)
                    saved_link.rename(inside)
            return ORIGINAL_RESOLVE(p, *a, **k)

        for follow in (False, True):
            read = getattr(skill_fs, "_read_discovered", None)
            self.assertIsNotNone(read, "discovery has no read through pinned handles")
            assert read is not None
            with _OutsideSpy(outside) as spy, mock.patch.object(Path, "resolve", resolve_around):
                try:
                    taken = read(path, ORIGINAL_RESOLVE(root), follow=follow)
                except Exception as exc:  # noqa: BLE001
                    raise AssertionError(f"the read raised {type(exc).__name__}: {exc}") from None
            self.assertIsInstance(taken, str, f"follow={follow}")
            self.assertRegex(
                str(taken), r"^below .*/root/tdd" + re.escape(BELOW_TAIL) + "$", f"{follow}"
            )
            self.assertEqual(spy.opened, 0, f"follow={follow}")
            self.assertFalse(spy.marker_read, f"follow={follow}")

    def test_a_parent_loop_after_the_listing_is_passed_over_on_every_python(self) -> None:
        self.assert_passed_over("parent-loop", BELOW, follow=False)
        followed = LOOP_AT_READ if RESOLVE_RAISES_ON_A_LOOP else BELOW
        self.assert_passed_over("parent-loop", followed, follow=True)

    def test_a_loop_at_the_walks_own_resolve_is_skipped_not_raised(self) -> None:
        tree = self.tree()
        store = tree.store()
        live = tree.live
        swapped = {"swapped": 0}

        def is_symlink_then_loop(p: Path) -> bool:
            got = ORIGINAL_IS_SYMLINK(p)
            if p == live and not swapped["swapped"]:
                live.parent.rename(live.parent.with_name("tdd.real"))
                live.parent.symlink_to(live.parent)
                swapped["swapped"] = 1
            return got

        with _bounded(), mock.patch.object(Path, "is_symlink", is_symlink_then_loop):
            snaps = _snapshots(store)
        self.assertEqual(swapped["swapped"], 1)
        self.assertEqual([s for s in snaps if s.skill_id == "tdd" and s.root_id == "canonical"], [])
        reasons = [p.reason for p in store.passed_over() if p.path == live]
        expected = LOOP_AT_WALK if RESOLVE_RAISES_ON_A_LOOP else "not a regular file (unreadable)"
        self.assertEqual(reasons, [expected])


class TheControlsTests(_StoreCase):
    def test_the_control_a_followed_link_inside_the_root_is_still_inventoried(self) -> None:
        tree = self.tree()
        text = tree.live.with_name("text.md")
        tree.live.rename(text)
        tree.live.symlink_to(text)
        store = tree.store(follow=True)
        snaps = _snapshots(store)
        taken = [s for s in snaps if s.skill_id == "tdd" and s.root_id == "canonical"]
        self.assertEqual(len(taken), 1)
        self.assertEqual(taken[0].sha256, sha256_text(text.read_text(encoding="utf-8")))
        self.assertEqual(store.passed_over(), [])

    def test_the_control_an_unswapped_tree_gives_the_same_snapshots_and_the_same_reasons(
        self,
    ) -> None:
        tree = self.tree()
        store = tree.store()
        snaps = _snapshots(store)
        self.assertEqual(
            sorted((s.skill_id, s.root_id) for s in snaps),
            [
                ("broken-frontmatter", "canonical"),
                ("code-review", "canonical"),
                ("mirror-only", "mirror"),
                ("tdd", "canonical"),
                ("tdd", "mirror"),
            ],
        )
        for snap in snaps:
            self.assertEqual(
                snap.sha256, sha256_text(snap.path.read_text(encoding="utf-8")), snap.path
            )
        self.assertEqual(store.passed_over(), [])
        if os.geteuid() != 0:
            locked = self.tree()
            locked.live.chmod(0)
            self.addCleanup(locked.live.chmod, 0o644)
            store = locked.store()
            _snapshots(store)
            self.assertEqual(
                [(p.skill_id, p.reason) for p in store.passed_over()],
                [("tdd", "a document this process cannot read (Permission denied)")],
            )
        vanished = self.tree()
        store = vanished.store()

        def vanish() -> None:
            vanished.live.rename(vanished.live.with_name("gone.md"))

        with _after_the_listing(vanished.live, vanish):
            _snapshots(store)
        self.assertEqual(
            [(p.skill_id, p.reason) for p in store.passed_over()],
            [("tdd", "a document this process cannot read (No such file or directory)")],
        )
        followed_out = self.tree()
        followed_out.live.rename(followed_out.live.with_name("text.md"))
        followed_out.live.symlink_to(followed_out.mirror)
        store = followed_out.store(follow=True)
        snaps = _snapshots(store)
        self.assertEqual([s for s in snaps if s.root_id == "canonical" and s.skill_id == "tdd"], [])
        self.assertEqual(
            [(p.skill_id, p.path.name, p.reason) for p in store.passed_over()],
            [
                (
                    None,
                    "SKILL.md",
                    "at the root itself, not in a skill directory; a skill is"
                    " <root>/<name>/SKILL.md",
                )
            ],
        )

    def test_without_dir_fd_opens_every_document_is_passed_over_and_without_pinned_writes_discovery_still_reads(  # noqa: E501
        self,
    ) -> None:
        tree = self.tree()
        store = tree.store()
        with mock.patch.object(skill_fs, "PINNED_READ_SUPPORTED", False, create=True):
            snaps = _snapshots(store)
        self.assertEqual(snaps, [])
        passed = store.passed_over()
        self.assertEqual(
            sorted({p.skill_id for p in passed}),
            ["broken-frontmatter", "code-review", "mirror-only", "tdd"],
        )
        self.assertEqual(len(passed), 5)
        self.assertEqual({p.reason for p in passed}, {NOT_READ})
        tree = self.tree()
        store = tree.store()
        with mock.patch.object(skill_fs, "PINNED_IO_SUPPORTED", False):
            snaps = _snapshots(store)
        self.assertEqual(len(snaps), 5)
        self.assertEqual(store.passed_over(), [])

    def test_a_followed_link_repointed_after_its_resolve_is_read_from_the_in_root_file_it_named(
        self,
    ) -> None:
        tree = self.tree()
        live = tree.live
        old = live.with_name("old.md")
        old.write_bytes(TDD_SKILL.encode() + b"\nOLD-IN-ROOT")
        foreign = tree.outside / "SKILL.md"
        foreign.write_bytes(TDD_SKILL.encode() + b"\nOUTSIDE-DIFFERENT")
        live.rename(live.with_name("saved-original"))
        live.symlink_to(old)
        store = tree.store(follow=True)
        state = {"listed": False, "repointed": False, "outside_opens": 0}
        out_inode = (foreign.stat().st_dev, foreign.stat().st_ino)

        def list_then_arm(self: FilesystemSkillStore, *a: Any, **k: Any) -> Any:
            found = ORIGINAL_ITER(self, *a, **k)
            state["listed"] = True
            return found

        def resolve_then_repoint(p: Path, *a: Any, **k: Any) -> Path:
            got = ORIGINAL_RESOLVE(p, *a, **k)
            if p == live and state["listed"] and not state["repointed"]:
                live.rename(live.with_name("saved-link"))
                live.symlink_to(foreign)
                state["repointed"] = True
            return got

        def counting_open(
            path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
        ) -> int:
            fd = ORIGINAL_OPEN(path, flags, mode, dir_fd=dir_fd)
            found = os.fstat(fd)
            state["outside_opens"] += int((found.st_dev, found.st_ino) == out_inode)
            return fd

        with (
            mock.patch.object(FilesystemSkillStore, "_iter_skill_files", list_then_arm),
            mock.patch.object(Path, "resolve", resolve_then_repoint),
            mock.patch.object(os, "open", counting_open),
        ):
            snaps = _snapshots(store)
        self.assertTrue(state["repointed"])
        self.assertEqual(ORIGINAL_RESOLVE(live), ORIGINAL_RESOLVE(foreign))
        taken = [s for s in snaps if s.path == live]
        self.assertEqual([s.skill_id for s in taken], ["tdd"])
        self.assertEqual(taken[0].sha256, sha256_text(old.read_text(encoding="utf-8")))
        self.assertNotEqual(taken[0].sha256, sha256_text(foreign.read_text(encoding="utf-8")))
        self.assertEqual(state["outside_opens"], 0)
        self.assertEqual([p for p in store.passed_over() if p.path == live], [])

    def test_the_control_a_hardlink_at_the_documents_path_is_an_in_root_entry_and_is_inventoried(
        self,
    ) -> None:
        tree = self.tree()
        live = tree.live
        foreign = tree.outside / "SKILL.md"
        foreign.write_bytes(TDD_SKILL.encode() + b"\nHARDLINK-MARKER")
        live.rename(live.with_name("saved-original"))
        os.link(foreign, live)
        found, outside = os.lstat(live), os.lstat(foreign)
        self.assertEqual((found.st_dev, found.st_ino), (outside.st_dev, outside.st_ino))
        self.assertEqual(found.st_nlink, 2)
        store = tree.store()
        snaps = _snapshots(store)
        taken = [s for s in snaps if s.path == live]
        self.assertEqual([s.skill_id for s in taken], ["tdd"])
        self.assertEqual(taken[0].sha256, sha256_text(foreign.read_text(encoding="utf-8")))
        self.assertEqual([p for p in store.passed_over() if p.path == live], [])


class TheCanonicalSlotIsWithheldTests(_StoreCase):
    def test_a_passed_over_copy_withholds_its_skills_canonical_slot(self) -> None:
        for kind in ("link-out", "fifo", "dir-out"):
            tree = self.tree()
            store = tree.store()
            with _OutsideSpy(tree.outside) as spy, _bounded():
                with _after_the_listing(tree.live, tree.swap(kind)):
                    _snapshots(store)
                    groups = store.groups()
            passed = [p for p in store.passed_over() if p.path == tree.live]
            self.assertEqual([p.skill_id for p in passed], ["tdd"], kind)
            group = group_for(groups, "tdd")
            self.assertIsNotNone(group, kind)
            assert group is not None
            self.assertIsNone(group.canonical, kind)
            self.assertIn(WITHHELD, group.canonical_reason or "", kind)
            self.assertEqual(spy.opened, 0, kind)
            self.assertFalse(spy.marker_read, kind)
        tree = self.tree()
        with _bounded(), _after_the_listing(tree.live, tree.swap("fifo")):
            rc, out, err = _run(tree.config, "daily", "--day", "2026-01-02")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            "skill 'tdd' has no canonical editable copy (the copy in editable root 'canonical'"
            " was passed over by discovery (",
            out + err,
        )
        self.assertNotIn("Traceback", err)


class _Applied:
    """A committed run with a validated ``tdd`` candidate, and one with it applied."""

    def __init__(self, base: Path, name: str, *, apply: bool) -> None:
        self.tree = _Tree(base, name)
        config = self.tree.config
        rc, out, err = _run(config, "daily", "--commit", "--day", "2026-01-01")
        assert rc == 0, err[-1500:]
        rc, out, err = _run(config, "--json", "candidates")
        found = [c for c in json.loads(out)["candidates"] if c["skill_id"] == "tdd"]
        self.cid = found[0]["id"]
        rc, out, err = _run(config, "--json", "review", self.cid)
        self.token = str(json.loads(out).get("approval_token"))
        if apply:
            rc, out, err = _run(
                config,
                "apply",
                self.cid,
                *("--approve", self.token, "--i-reviewed-this", "--operator", "alice"),
            )
            assert rc == 0, err[-1500:]
        self.original = self.tree.live.read_bytes()


class _CommandCase(unittest.TestCase):
    applied: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.env = _Applied(Path(cls._tmp.name).resolve(), "env", apply=cls.applied)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        _restore(self.env.tree.live, self.env.original)

    def run_swapped(self, kind: str, *argv: str) -> tuple[int, str, str, _OutsideSpy, bool]:
        tree = self.env.tree
        _restore(tree.live, self.env.original)
        before = (tree.mirror.read_bytes(), _tree_hash(tree.backups))
        with _OutsideSpy(tree.outside) as spy, _bounded():
            with _after_the_listing(tree.live, tree.swap(kind)) as seen:
                rc, out, err = _run(tree.config, *argv)
        unchanged = before == (tree.mirror.read_bytes(), _tree_hash(tree.backups))
        where = f"{kind} {argv[:2]}"
        self.assertEqual(seen["swapped"], 1, where)
        self.assertNotIn("Traceback", err, where)
        self.assertNotEqual(rc, 70, f"{where}: {err[-800:]}")
        self.assertEqual(spy.opened, 0, f"{where}: the outside directory was opened")
        self.assertFalse(spy.marker_read, f"{where}: the outside bytes were read")
        return rc, out, err, spy, unchanged


FRAGMENT = {"link-out": NOT_FOLLOWED, "fifo": A_FIFO, "dir-out": BELOW_TAIL}


class TheCommandsNameTheSwappedDocumentTests(_CommandCase):
    def test_skills_doctor_review_and_apply_name_a_swapped_document_and_none_exits_70(
        self,
    ) -> None:
        env = self.env
        approve = ("--approve", env.token, "--i-reviewed-this", "--operator", "alice")
        for kind in ("link-out", "fifo", "dir-out"):
            for argv, expected in (
                (("skills",), 0),
                (("doctor",), 0),
                (("review", env.cid), 3),
                (("apply", env.cid, *approve), 3),
            ):
                rc, out, err, _spy, unchanged = self.run_swapped(kind, *argv)
                where = f"{kind} {argv[0]}"
                said = out + err
                self.assertEqual(rc, expected, f"{where}: {said[-800:]}")
                self.assertIn(FRAGMENT[kind], said, where)
                self.assertIn("skills-canonical/tdd", said, where)
                if expected == 3:
                    self.assertIn("MirrorEditRefused", said, where)
                self.assertTrue(unchanged, f"{where}: the mirror or a backup changed")

    def test_daily_passes_a_link_or_fifo_over_and_exits_0(self) -> None:
        for kind in ("link-out", "fifo"):
            rc, out, err, _spy, _unchanged = self.run_swapped(kind, "daily", "--day", "2026-01-02")
            self.assertEqual(rc, 0, f"{kind}: {(out + err)[-800:]}")
            self.assertNotIn(ALARM, out + err, kind)
            rc, out, err, _spy, _unchanged = self.run_swapped(
                kind, "--json", "daily", "--day", "2026-01-03"
            )
            self.assertEqual(rc, 0, f"{kind}: {(out + err)[-800:]}")
            doc = json.loads(out)
            self.assertNotIn(ALARM, out + err, kind)
            entries = [e for v in _find_key(doc, "skill_documents_passed_over") for e in (v or [])]
            named = [e for e in entries if "tdd/SKILL.md" in json.dumps(e)]
            self.assertEqual(len(named), 1, f"{kind}: {entries}")
            self.assertIn(FRAGMENT[kind], json.dumps(named[0]), kind)
            self.assertIn(False, list(_find_key(doc, "mutation_detected")), kind)
            self.assertNotIn(True, list(_find_key(doc, "mutation_detected")), kind)

    def test_daily_blocks_with_5_when_the_skill_directory_is_swapped_and_names_both(
        self,
    ) -> None:
        rc, out, err, _spy, _unchanged = self.run_swapped("dir-out", "daily", "--day", "2026-01-04")
        said = out + err
        self.assertEqual(rc, 5, said[-800:])
        self.assertIn("status=blocked", said)
        self.assertIn("live_mutation=True", said)
        self.assertIn(ALARM, said)
        self.assertIn("added: canonical:tdd.real/SKILL.md; removed: canonical:tdd/SKILL.md", said)
        rc, out, err, _spy, _unchanged = self.run_swapped(
            "dir-out", "--json", "daily", "--day", "2026-01-05"
        )
        self.assertEqual(rc, 5, (out + err)[-800:])
        doc = json.loads(out)
        self.assertIn(ALARM, out)
        entries = [e for v in _find_key(doc, "skill_documents_passed_over") for e in (v or [])]
        named = [e for e in entries if "tdd/SKILL.md" in json.dumps(e)]
        self.assertEqual(len(named), 1, entries)
        self.assertRegex(str(named[0].get("reason", "")), BELOW)
        self.assertIn(True, list(_find_key(doc, "mutation_detected")))


class TheRollbackRefusesTheSwappedCopyTests(_CommandCase):
    applied = True

    def test_rollback_of_an_applied_candidate_refuses_a_canonical_copy_swapped_for_a_fifo(
        self,
    ) -> None:
        env = self.env
        live = env.tree.live
        rc, out, err, _spy, unchanged = self.run_swapped("fifo-aside", "rollback", env.cid)
        said = out + err
        self.assertEqual(rc, 3, said[-800:])
        self.assertIn("MirrorEditRefused", said)
        self.assertIn("skills-canonical/tdd/SKILL.md", said)
        self.assertTrue(stat.S_ISFIFO(os.lstat(live).st_mode), "the fifo was not left in place")
        self.assertEqual(live.with_name("SKILL.md.aside").read_bytes(), env.original)
        self.assertTrue(unchanged, "the mirror or a backup changed")
        for kind in ("link-out", "dir-out"):
            rc, out, err, _spy, unchanged = self.run_swapped(kind, "rollback", env.cid)
            said = out + err
            self.assertEqual(rc, 3, f"{kind}: {said[-800:]}")
            self.assertIn("MirrorEditRefused", said, kind)
            self.assertTrue(unchanged, f"{kind}: the mirror or a backup changed")
            if kind == "dir-out":
                self.assertTrue(live.parent.is_symlink(), kind)
                real = live.parent.with_name("tdd.real") / "SKILL.md"
                self.assertEqual(real.read_bytes(), env.original, kind)
            else:
                self.assertTrue(live.is_symlink(), kind)


def _shallow_swapper(shape: str, live: Path) -> Callable[[], None]:
    """Replace ``live`` by a link to its root, or to the regular file ``<root>/notes.md``."""
    root = live.parent.parent

    def swap() -> None:
        live.unlink()
        if shape == "root":
            live.symlink_to(root, target_is_directory=True)
        elif shape == "root-file":
            live.symlink_to(root / LOOSE)
        else:
            raise AssertionError(shape)

    return swap


def _kept(tree: _Tree) -> tuple[bytes, str, bytes]:
    """The mirror's bytes, the backups' digest and the bytes of ``<root>/notes.md``."""
    return tree.mirror.read_bytes(), _tree_hash(tree.backups), (tree.root / LOOSE).read_bytes()


@contextlib.contextmanager
def _every_store_follows() -> Iterator[None]:
    """The commands build their stores without following links; make each one follow."""

    def following(self: FilesystemSkillStore, *args: Any, **kwargs: Any) -> None:
        ORIGINAL_INIT(self, *args, **kwargs)
        self._follow_symlinks = True

    with mock.patch.object(FilesystemSkillStore, "__init__", following):
        yield


class TheFollowedLinkWithNoSkillDirectoryTests(_StoreCase):
    def test_a_followed_link_to_its_root_or_a_file_directly_in_it_is_passed_over_by_name(
        self,
    ) -> None:
        commands = _Applied(self.base, "shallow-commands", apply=False)
        applied = _Applied(self.base, "shallow-rollback", apply=True)
        for env in (commands, applied):
            (env.tree.root / LOOSE).write_bytes(env.original)
        approve = ("--approve", commands.token, "--i-reviewed-this", "--operator", "alice")
        for shape in ("root", "root-file"):
            with self.subTest(shape=shape, part="store"):
                tree = self.tree()
                (tree.root / LOOSE).write_bytes(tree.original)
                store = tree.store(follow=True)
                swap = _shallow_swapper(shape, tree.live)
                with _bounded(), _after_the_listing(tree.live, swap) as seen:
                    snaps = _snapshots(store)
                    groups = store.groups()
                self.assertEqual(seen["swapped"], 1)
                self.assertEqual([s for s in snaps if s.path == tree.live], [])
                self.assertEqual(
                    [s for s in snaps if s.skill_id == "tdd" and s.root_id == "canonical"], []
                )
                passed = [p for p in store.passed_over() if p.path == tree.live]
                self.assertEqual([(p.reason, p.skill_id) for p in passed], [(SHALLOW, "tdd")])
                group = group_for(groups, "tdd")
                self.assertIsNotNone(group)
                assert group is not None
                self.assertIsNone(group.canonical)
                self.assertIn(WITHHELD, group.canonical_reason or "")
            for env, argv, expected in (
                (commands, ("skills",), 0),
                (commands, ("doctor",), 0),
                (commands, ("--json", "daily", "--day", "2026-01-02"), 5),
                (commands, ("review", commands.cid), 3),
                (commands, ("apply", commands.cid, *approve), 3),
                (applied, ("rollback", applied.cid), 3),
            ):
                with self.subTest(shape=shape, part=next(a for a in argv if a[0] != "-")):
                    tree = env.tree
                    _restore(tree.live, env.original)
                    before = _kept(tree)
                    swap = _shallow_swapper(shape, tree.live)
                    with _bounded(), _every_store_follows():
                        with _after_the_listing(tree.live, swap) as seen:
                            rc, out, err = _run(tree.config, *argv)
                    said = out + err
                    self.assertEqual(seen["swapped"], 1)
                    self.assertNotIn("Traceback", err)
                    self.assertEqual(rc, expected, said[-800:])
                    self.assertIn(SHALLOW, said)
                    self.assertIn("skills-canonical/tdd/SKILL.md", said)
                    if expected == 3:
                        self.assertIn("MirrorEditRefused", said)
                    if expected == 5:
                        # Blocked, not 0: the run's closing walk follows the swapped-in
                        # link to the root's own level, where no skill is, so it sees
                        # the tree changed mid-run (as for a swapped directory, above).
                        doc = json.loads(out)
                        self.assertIn(ALARM, out)
                        self.assertEqual(
                            list(_find_key(doc, "changes")),
                            [[{"key": "canonical:tdd/SKILL.md", "kind": "removed"}]],
                        )
                        entries = [
                            e
                            for v in _find_key(doc, "skill_documents_passed_over")
                            for e in v or []
                        ]
                        named = [e for e in entries if e.get("path") == str(tree.live)]
                        self.assertEqual(
                            [(e.get("reason"), e.get("skill_id")) for e in named],
                            [(SHALLOW, "tdd")],
                        )
                    target = tree.root if shape == "root" else tree.root / LOOSE
                    self.assertEqual(os.readlink(tree.live), str(target))
                    self.assertEqual(
                        _kept(tree), before, "the mirror, a backup or notes.md changed"
                    )
        with self.subTest(shape="unfollowed, called directly"):
            # The listing passes a document shallower than ``<skill>/SKILL.md``
            # over, so without links followed only a direct call reaches this.
            read = getattr(skill_fs, "_read_discovered", None)
            self.assertIsNotNone(read, "no _read_discovered")
            assert read is not None
            root = self.base / "direct"
            root.mkdir()
            (root / LOOSE).write_bytes(TDD_SKILL.encode())
            for listed in (root, root / LOOSE):
                self.assertEqual(read(listed, root, follow=False), SHALLOW_UNFOLLOWED)


if __name__ == "__main__":
    unittest.main()
