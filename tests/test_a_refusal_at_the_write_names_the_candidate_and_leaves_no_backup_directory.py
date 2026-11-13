"""A refusal at the write names the candidate and leaves no backup directory.

``apply`` judges its target (``assert_write_target``) and writes
(``write_canonical``), which judges the target again and checks the file's hash
before it takes its backup; before, ``apply`` made the backup directory
(``backup_dir_for``) between the two. Measured
before (3.11 and 3.13, a committed run with one validated candidate, its token
taken, then, at the step ``apply`` takes after its own check, a second copy of
the skill made in the editable root, the skill removed, replaced by a fifo or by
a link to an identical copy, the skill's folder replaced by a link to an
identical copy of it, its text changed or its bytes made other than UTF-8):
``apply`` exited 3, the record stayed ``validated`` and the backup directory was
left behind, empty; each refusal of the write's own target check named no
candidate ("refusing to write <path>: ..." for the second copy, the skill
removed and the fifo; "refusing to replace <path>: ..." for the two links),
where ``apply``'s own check names it and how to close it; the changed text was
refused naming it, and the bytes that are not UTF-8 naming none, as before. With
each change undone, ``apply`` with the same token applied the candidate.

The write's own target check is now given the candidate, so it words a refusal
as ``apply``'s own check does. And ``apply`` no longer makes the backup
directory: it judges it where it made it (``backup_dir_for(..., make=False)``),
so each refusal of it comes where it came, and gives the write the call that
makes it, which the write makes only as it takes its backup, after its target
check, its read and its hash check, so a refusal of the making itself (the
mkdir failing, or the directory found not a real one after it) comes there
now, after the write's own. Nothing is removed. Since then the bytes made other
than UTF-8 are refused naming the candidate too (the module
``test_a_skill_made_other_than_utf8_while_apply_runs_is_refused_with_the_candidate``).
"""

from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import re
import shutil
import signal
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters import skill_fs
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app import state as state_module
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.config import load_config
from agent_skill_improver.errors import MirrorEditRefused, SafetyRefusal

from tests.support import build_env

DAY = "2026-02-01"
REFUSED = 3
CID = "CAND-2026-02-01-01-tdd"
CLOSE = (
    "The candidate stays open, and counts against [caps].max_open_candidates, until it is "
    f"closed: to close it, delete its record ({CID}.json in the state directory's candidates/)."
)


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            rc = cli.main(["--config", str(config), *argv])
        except SystemExit as exc:  # pragma: no cover - the CLI returns its code
            rc = int(exc.code or 0)
    return rc, out.getvalue()


class Validated:
    """A committed run's one validated candidate, its token taken."""

    def __init__(self, base: Path) -> None:
        self.base = base.resolve()
        self.config = build_env(base / "env", seed="c229")
        self.state = self.base / "env" / "state"
        rc, said = run(self.config, "daily", "--commit", "--day", DAY)
        assert rc == 0, said
        (self.record,) = sorted(self.state.glob("candidates/*.json"))
        assert self.record.stem == CID, self.record.stem
        self.skill = Path(json.loads(self.record.read_text())["target"]["canonical_path"])
        rc, shown = run(self.config, "review", CID)
        found = re.search(r"--approve (\S+)", shown)
        assert rc == 0 and found, shown
        self.token = found.group(1)
        self.original = self.skill.read_bytes()

    def apply_after(self, change: Callable[[], None]) -> tuple[int, str]:
        """``apply``, with ``change`` made at the step it takes after its own check."""
        real = FilesystemSkillStore.mirror_paths_of
        done: list[bool] = []

        def change_then_list(store: Any, path: Path) -> tuple[str, ...]:
            if not done:
                done.append(True)
                change()
            return real(store, path)

        with mock.patch.object(FilesystemSkillStore, "mirror_paths_of", change_then_list):
            return self.apply()

    def apply(self) -> tuple[int, str]:
        return run(self.config, "apply", CID, "--approve", self.token, "--i-reviewed-this")

    def status(self) -> str:
        return str(json.loads(self.record.read_text())["status"])

    def backups(self) -> list[Path]:
        return sorted(self.state.glob("backups/*"))

    def put_back(self) -> None:
        if self.skill.parent.is_symlink():
            twin = self.skill.parent.resolve()
            self.skill.parent.unlink()
            shutil.copytree(twin, self.skill.parent)
        if self.skill.is_symlink() or self.skill.is_fifo():
            self.skill.unlink()
        self.skill.write_bytes(self.original)


def first_line(said: str) -> str:
    return next(line for line in said.splitlines() if line.strip())


class TheWritesOwnCheckNamesTheCandidateTests(unittest.TestCase):
    def test_a_skill_removed_after_applys_own_check_is_refused_with_the_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            rc, said = case.apply_after(case.skill.unlink)
            self.assertEqual(rc, REFUSED, said)
            self.assertEqual(
                first_line(said),
                f"MirrorEditRefused: refusing to apply {CID} at {case.skill}: it is not a skill "
                "document in any configured root, so this tool cannot tell whether it is "
                "canonical. An unrecognised target is refused rather than assumed safe. The file "
                f"and the candidate are unchanged. {CLOSE} If that file is meant to be the skill's "
                "canonical copy, make it that again: while it holds the text the candidate was "
                "written against, `review` and `apply` take the candidate as before.",
            )
            self.assertEqual(case.status(), "validated")
            self.assertEqual(case.backups(), [])
            case.put_back()
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            self.assertEqual(case.status(), "applied")

    def test_a_fifo_or_a_link_put_there_after_applys_own_check_is_refused_with_the_candidate(
        self,
    ) -> None:
        # The file replaced by a fifo, the file by a link to an identical copy,
        # or the skill's folder by a link to an identical copy of it.
        def fifo(case: Validated) -> None:
            case.skill.unlink()
            os.mkfifo(case.skill)

        def link(case: Validated) -> None:
            twin = case.base / "twin.md"
            twin.write_bytes(case.original)
            case.skill.unlink()
            case.skill.symlink_to(twin)

        def folder(case: Validated) -> None:
            twin = case.base / "tdd-copy"
            shutil.copytree(case.skill.parent, twin)
            shutil.rmtree(case.skill.parent)
            case.skill.parent.symlink_to(twin, target_is_directory=True)

        for name, change in (("fifo", fifo), ("link", link), ("folder", folder)):
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                case = Validated(Path(tmp))
                rc, said = case.apply_after(lambda: change(case))
                self.assertEqual(rc, REFUSED, said)
                self.assertIn(f"refusing to apply {CID}", first_line(said))
                self.assertIn(CLOSE, first_line(said))
                self.assertEqual(case.status(), "validated")
                self.assertEqual(case.backups(), [])
                case.put_back()
                rc, said = case.apply()
                self.assertEqual(rc, 0, said)

    def test_without_a_candidate_the_writes_own_refusal_is_worded_as_before(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_env(Path(tmp) / "env", seed="c229")
            skill = (Path(tmp) / "env").resolve() / "skills-canonical" / "tdd" / "SKILL.md"
            extra = skill.parent.parent / "group" / "tdd" / "SKILL.md"
            extra.parent.mkdir(parents=True)
            shutil.copy(skill, extra)
            store = FilesystemSkillStore(load_config(config).skill_roots, read_only=False)
            with self.assertRaises(MirrorEditRefused) as ctx:
                store.write_canonical(
                    skill,
                    "new text\n",
                    expected_sha256="0" * 64,
                    backup_dir=Path(tmp) / "backup",
                    skill_id="tdd",
                )
            self.assertTrue(
                str(ctx.exception).startswith(
                    f"refusing to write {skill}: skill 'tdd' has no canonical copy"
                ),
                str(ctx.exception),
            )
            self.assertNotIn("candidate", str(ctx.exception))


class TheBackupDirectoryTests(unittest.TestCase):
    def test_a_skill_edited_after_applys_own_check_leaves_no_backup_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            rc, said = case.apply_after(
                lambda: case.skill.write_bytes(case.original + b"\nedited while apply ran\n")
            )
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                first_line(said).startswith(f"BaseHashMismatch: refusing to apply {CID}: "), said
            )
            self.assertEqual(case.status(), "validated")
            self.assertEqual(case.backups(), [])

    def test_bytes_made_other_than_utf8_after_applys_own_check_name_it_and_leave_no_backup(
        self,
    ) -> None:
        # The write's read words them with the candidate; its words are
        # test_a_skill_made_other_than_utf8_while_apply_runs_is_refused_with_the_candidate's.
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            rc, said = case.apply_after(lambda: case.skill.write_bytes(b"\xff" + case.original))
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                first_line(said).startswith(
                    f"UndecodableDocument: refusing to apply {CID}: its canonical target "
                    f"{case.skill} is not valid UTF-8 at byte 0"
                ),
                said,
            )
            self.assertEqual(case.status(), "validated")
            self.assertEqual(case.backups(), [])
            case.put_back()
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)

    def test_a_write_that_took_its_backup_and_then_failed_keeps_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))

            def refuse(*args: Any, **kwargs: Any) -> None:
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES))

            with mock.patch.object(skill_fs, "_write_pinned", refuse):
                rc, said = case.apply()
            self.assertEqual(rc, REFUSED, said)
            (backup,) = case.backups()
            kept = backup / f"{case.skill.name}.bak"
            self.assertIn(f"the copy taken before the write is at {kept}", said)
            self.assertEqual(kept.read_bytes(), case.original)
            self.assertEqual(case.skill.read_bytes(), case.original)
            self.assertEqual(case.status(), "validated")

    def test_a_stop_before_the_write_took_its_backup_leaves_no_backup_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))

            def stop(*args: Any, **kwargs: Any) -> Any:
                raise cli.Stopped(signal.SIGTERM)

            with mock.patch.object(FilesystemSkillStore, "_open_parent_pinned", stop):
                rc, said = case.apply()
            self.assertNotEqual(rc, 0, said)
            self.assertEqual(case.skill.read_bytes(), case.original)
            self.assertEqual(case.status(), "validated")
            self.assertEqual(case.backups(), [])

    def test_backups_made_a_link_at_the_writes_own_check_is_refused_and_nothing_goes_outside(
        self,
    ) -> None:
        # Measured before: the directory was made before the write, and the write
        # took its copy and wrote backup.json through the link, outside state.
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            outside = case.base / "outside-state"
            outside.mkdir()
            real = FilesystemSkillStore.assert_write_target
            asked: list[list[Path]] = []

            def swap_at_the_writes_check(store: Any, path: Path, **kwargs: Any) -> Path:
                asked.append(case.backups())
                if len(asked) == 2:
                    backups = case.state / "backups"
                    backups.rename(case.state / "backups-original")
                    backups.symlink_to(outside, target_is_directory=True)
                return real(store, path, **kwargs)

            with mock.patch.object(
                FilesystemSkillStore, "assert_write_target", swap_at_the_writes_check
            ):
                rc, said = case.apply()
            self.assertEqual(asked, [[], []])
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                first_line(said).startswith(
                    f"SafetyRefusal: refusing to use {case.state / 'backups'} as a state directory"
                ),
                said,
            )
            self.assertEqual(list(outside.iterdir()), [])
            self.assertEqual(list((case.state / "backups-original").iterdir()), [])
            self.assertEqual(case.skill.read_bytes(), case.original)
            self.assertEqual(case.status(), "validated")


class TheWriteMakesTheDirectoryTests(unittest.TestCase):
    """``apply`` judges the backup directory where it made it; the write makes it."""

    def test_no_backup_directory_is_there_until_the_write_takes_its_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            real_check = FilesystemSkillStore.assert_write_target
            real_copy = skill_fs.write_bytes_atomic
            at_checks: list[list[Path]] = []
            at_copy: list[list[list[str]]] = []

            def check(store: Any, path: Path, **kwargs: Any) -> Path:
                at_checks.append(case.backups())
                return real_check(store, path, **kwargs)

            def copy(path: Path, data: bytes, **kwargs: Any) -> None:
                if not at_copy:
                    at_copy.append([sorted(os.listdir(d)) for d in case.backups()])
                real_copy(path, data, **kwargs)

            with mock.patch.object(FilesystemSkillStore, "assert_write_target", check):
                with mock.patch.object(skill_fs, "write_bytes_atomic", copy):
                    rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            self.assertEqual(at_checks, [[], []])
            self.assertEqual(at_copy, [[[]]])
            (backup,) = case.backups()
            self.assertEqual(sorted(os.listdir(backup)), ["SKILL.md.bak", "backup.json"])
            self.assertEqual(case.status(), "applied")

    def test_a_refusal_of_the_backup_directory_still_comes_before_the_writes_own(self) -> None:
        # Both made after apply's own check: backups/ a link outside state, and
        # a second copy of the skill in the editable root. The backup directory
        # is refused first, as before, and the write is not reached.
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            outside = case.base / "outside-state"
            outside.mkdir()
            written: list[bool] = []
            real_write = FilesystemSkillStore.write_canonical

            def both() -> None:
                backups = case.state / "backups"
                backups.rename(case.state / "backups-original")
                backups.symlink_to(outside, target_is_directory=True)
                extra = case.skill.parent.parent / "group" / "tdd" / "SKILL.md"
                extra.parent.mkdir(parents=True)
                shutil.copy(case.skill, extra)

            def write(store: Any, *args: Any, **kwargs: Any) -> Any:
                written.append(True)
                return real_write(store, *args, **kwargs)

            with mock.patch.object(FilesystemSkillStore, "write_canonical", write):
                rc, said = case.apply_after(both)
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                first_line(said).startswith(
                    f"SafetyRefusal: refusing to use {case.state / 'backups'} as a state directory"
                ),
                said,
            )
            self.assertEqual(written, [])
            self.assertEqual(list(outside.iterdir()), [])
            self.assertEqual(case.status(), "validated")

    def test_a_refusal_of_the_making_itself_comes_at_the_write_after_its_own(self) -> None:
        # Not as before: a refusal of the making itself (the mkdir failing, or
        # the directory found not a real one after it) came before the write;
        # it comes at the write now, as the backup is taken, after the write's
        # target check, its read and its hash check. Alone, it is given as
        # before; with the text also changed after apply's own check, the
        # write's refusal of that comes instead, and the making is not tried.
        for making in ("the mkdir refused", "a file made in its place"):
            for changed in (False, True):
                with self.subTest(making=making, changed=changed):
                    with tempfile.TemporaryDirectory() as tmp:
                        case = Validated(Path(tmp))
                        events: list[str] = []
                        real_check = FilesystemSkillStore.assert_write_target
                        real_make = state_module._mkdir_or_refuse

                        def check(store: Any, path: Path, **kwargs: Any) -> Path:
                            events.append("target check")
                            return real_check(store, path, **kwargs)

                        def make(path: Path, root: Path, **kwargs: Any) -> None:
                            if not path.name.startswith(f"{CID}--"):
                                return real_make(path, root, **kwargs)
                            events.append("making")
                            if making == "a file made in its place":
                                path.write_text("")
                                return None
                            full = OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
                            with mock.patch.object(Path, "mkdir", side_effect=full):
                                return real_make(path, root, **kwargs)

                        edited = case.original + b"\nedited while apply ran\n"
                        with (
                            mock.patch.object(FilesystemSkillStore, "assert_write_target", check),
                            mock.patch.object(state_module, "_mkdir_or_refuse", make),
                        ):
                            if changed:
                                rc, said = case.apply_after(lambda: case.skill.write_bytes(edited))
                            else:
                                rc, said = case.apply()
                        self.assertEqual(rc, REFUSED, said)
                        self.assertEqual(case.status(), "validated")
                        if changed:
                            self.assertTrue(
                                first_line(said).startswith(
                                    f"BaseHashMismatch: refusing to apply {CID}: "
                                ),
                                said,
                            )
                            self.assertEqual(events, ["target check", "target check"])
                            self.assertEqual(case.skill.read_bytes(), edited)
                            self.assertEqual(case.backups(), [])
                            continue
                        self.assertEqual(events, ["target check", "target check", "making"])
                        self.assertEqual(case.skill.read_bytes(), case.original)
                        if making == "the mkdir refused":
                            self.assertTrue(first_line(said).startswith("ConfigError: "), said)
                            self.assertIn("cannot be created (No space left on device)", said)
                            self.assertEqual(case.backups(), [])
                        else:
                            self.assertTrue(
                                first_line(said).startswith("SafetyRefusal: refusing to use "), said
                            )
                            self.assertIn("after creation", said)
                            self.assertEqual([p.is_file() for p in case.backups()], [True])

    def test_a_write_that_fails_before_its_copy_is_in_the_directory_leaves_it_empty(self) -> None:
        # Not covered: as before, the directory the write made stays, empty.
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))

            def full(*args: Any, **kwargs: Any) -> None:
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

            with mock.patch.object(skill_fs, "write_bytes_atomic", full):
                rc, said = case.apply()
            self.assertNotEqual(rc, 0, said)
            (backup,) = case.backups()
            self.assertEqual(os.listdir(backup), [])
            self.assertEqual(case.skill.read_bytes(), case.original)
            self.assertEqual(case.status(), "validated")

    def test_judged_without_making_it_is_refused_as_making_it_is_and_makes_no_backup_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            state = StateStore(base / "state")
            backups = state.dir("backups")
            self.assertEqual(state.backup_dir_for("CAND-a", "1", make=False), backups / "CAND-a--1")
            self.assertEqual(list(backups.iterdir()), [])
            elsewhere = base / "elsewhere"
            elsewhere.mkdir()
            (backups / "CAND-a--2").symlink_to(elsewhere, target_is_directory=True)
            (backups / "CAND-a--3").write_text("a file\n")
            for stamp in ("2", "3"):
                refused = []
                for make in (False, True):
                    with self.assertRaises(SafetyRefusal) as ctx:
                        state.backup_dir_for("CAND-a", stamp, make=make)
                    refused.append(str(ctx.exception))
                self.assertEqual(refused[0], refused[1])
                self.assertIn(f"refusing to use {backups / f'CAND-a--{stamp}'}", refused[0])
            for make in (False, True):
                with self.assertRaises(ValueError):
                    state.backup_dir_for("CAND-a", "1--2", make=make)
            dry = StateStore(base / "state", dry_run=True)
            for make in (False, True):
                with self.assertRaises(SafetyRefusal) as ctx:
                    dry.backup_dir_for("CAND-a", "1", make=make)
                self.assertIn("withholds writes (dry run)", str(ctx.exception))
            # The name ``backup_dir_for`` builds cannot hold a separator, so the
            # helper the judgement and the making share is asked directly.
            before_root = sorted(p.name for p in state.root.iterdir())
            for name in ("a/b", "../CAND-a--1"):
                refused = []
                for judge in (state_module._judge_child_dir, state_module._ensure_child_dir):
                    with self.assertRaises(SafetyRefusal) as ctx:
                        judge(backups, name, state.root)
                    refused.append(str(ctx.exception))
                self.assertEqual(refused[0], refused[1])
                self.assertIn("must be a single path component", refused[0])
            self.assertEqual(sorted(p.name for p in state.root.iterdir()), before_root)
            self.assertEqual(sorted(p.name for p in backups.iterdir()), ["CAND-a--2", "CAND-a--3"])
            self.assertEqual(list(elsewhere.iterdir()), [])

    def test_backups_moved_outside_with_a_file_left_as_the_directory_is_made_makes_nothing(
        self,
    ) -> None:
        # Measured with an earlier version of this change, which made the
        # directory through a handle on backups/: the directory was made in
        # backups/ where it had been moved, outside state.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            state = StateStore(base / "state")
            backups = state.dir("backups")
            outside = base / "outside-state"
            real = state_module._ensure_child_dir

            def moved_then_made(parent: Path, name: str, root: Path) -> Path:
                backups.rename(outside)
                backups.write_text("a file\n")
                return real(parent, name, root)

            with mock.patch.object(state_module, "_ensure_child_dir", moved_then_made):
                with self.assertRaises(SafetyRefusal):
                    state.backup_dir_for("CAND-a", "1")
            self.assertEqual(list(outside.iterdir()), [])
