"""A backup that cannot be taken is refused with the candidate.

``apply`` writes (``write_canonical`` in ``adapters/skill_fs.py``) only after it
takes a backup under ``state/backups/``: it makes the backup directory, writes
the copy of the skill there, reads the copy's hash back and writes its manifest
(``backup.json``), all before the skill is touched. Measured before (3.11 and
3.13, a committed run with one validated candidate, its token taken, then the
copy's write or the manifest's raising ENOSPC): ``apply`` exited 70 with
"internal error: this is a bug in asi", the record ``validated``, the skill
unchanged, and under ``backups/`` an empty directory (the copy) or one holding
only the copy (the manifest); with the space back, ``apply`` with the same token
applied the candidate.

An ``OSError`` taking the backup is now the refusal ``BackupNotTaken``
(``backup_not_taken``): with the candidate and the same token's way on, or,
without one (a direct write), naming the skill. Nothing is removed, and the
words say only that this apply wrote no ``backup.json``: the directory is named
for the candidate and the second, so one an earlier apply of the candidate,
stopped before its skill write within the same second, left there is kept, and
``rollback`` reads it as that apply's.
"""

from __future__ import annotations

import contextlib
import datetime
import errno
import io
import json
import os
import re
import signal
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters import skill_fs
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.config import load_config
from agent_skill_improver.errors import BackupNotTaken, SafetyRefusal
from tests.support import build_env

DAY = "2026-02-01"
REFUSED = 3
CID = f"CAND-{DAY}-01-tdd"
NO_SPACE = "[Errno 28] No space left on device"


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


def first_line(said: str) -> str:
    return next(line for line in said.splitlines() if line.strip())


def no_space(*_args: Any, **_kwargs: Any) -> None:
    raise OSError(errno.ENOSPC, "No space left on device")


def refused_with_the_candidate(skill: Path, left: Path, why: str) -> str:
    return (
        f"BackupNotTaken: refusing to apply {CID}: the backup of its canonical target {skill} "
        f"(the copy and its manifest) could not be written before the write ({why}), so the "
        "skill was not written and the candidate is as it was. Make room for the backup, or "
        "make the backup directory writable again, then run the same apply: its token still "
        f"holds. Whatever was written of the backup stays under {left}, and this apply wrote "
        "no backup.json there."
    )


class Validated:
    """A committed run's one validated candidate, its token taken."""

    def __init__(self, base: Path) -> None:
        self.base = base.resolve()
        self.config = build_env(base / "env", seed="c233")
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

    def apply(self) -> tuple[int, str]:
        return run(self.config, "apply", CID, "--approve", self.token, "--i-reviewed-this")

    def apply_with(self, name: str, fake: Any) -> tuple[int, str]:
        with mock.patch.object(skill_fs, name, fake):
            return self.apply()

    def status(self) -> str:
        return str(json.loads(self.record.read_text())["status"])

    def backups(self) -> list[Path]:
        return sorted(self.state.glob("backups/*"))


class ApplyRefusesTests(unittest.TestCase):
    def test_no_space_for_the_copy_is_refused_with_the_candidate_and_the_skill_unchanged(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            rc, said = case.apply_with("write_bytes_atomic", no_space)
            self.assertEqual(rc, REFUSED, said)
            (left,) = case.backups()
            self.assertEqual(
                first_line(said), refused_with_the_candidate(case.skill, left, NO_SPACE)
            )
            self.assertEqual(case.status(), "validated")
            self.assertEqual(case.skill.read_bytes(), case.original)
            self.assertEqual(sorted(left.iterdir()), [])

    def test_no_space_for_the_manifest_leaves_the_copy_and_no_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            rc, said = case.apply_with("write_json", no_space)
            self.assertEqual(rc, REFUSED, said)
            (left,) = case.backups()
            self.assertEqual(
                first_line(said), refused_with_the_candidate(case.skill, left, NO_SPACE)
            )
            self.assertEqual(case.status(), "validated")
            self.assertEqual(case.skill.read_bytes(), case.original)
            self.assertEqual([p.name for p in left.iterdir()], ["SKILL.md.bak"])
            self.assertEqual((left / "SKILL.md.bak").read_bytes(), case.original)

    def test_the_copys_hash_read_failing_is_refused_the_same_way(self) -> None:
        def unreadable(_path: Path) -> str:
            raise OSError(errno.EIO, "Input/output error")

        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            rc, said = case.apply_with("sha256_file", unreadable)
            self.assertEqual(rc, REFUSED, said)
            (left,) = case.backups()
            self.assertEqual(
                first_line(said),
                refused_with_the_candidate(case.skill, left, "[Errno 5] Input/output error"),
            )
            self.assertEqual(case.status(), "validated")
            self.assertEqual(case.skill.read_bytes(), case.original)
            self.assertFalse((left / "backup.json").exists())

    def test_with_the_space_back_the_same_token_applies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            rc, said = case.apply_with("write_bytes_atomic", no_space)
            self.assertEqual(rc, REFUSED, said)
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            self.assertEqual(case.status(), "applied")
            self.assertNotEqual(case.skill.read_bytes(), case.original)
            manifests = sorted(case.state.glob("backups/*/backup.json"))
            self.assertEqual(len(manifests), 1, manifests)
            self.assertEqual((manifests[0].parent / "SKILL.md.bak").read_bytes(), case.original)

    def test_rollback_passes_over_what_is_left(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            nothing_applied = run(case.config, "rollback", CID)
            rc, said = case.apply_with("write_json", no_space)
            self.assertEqual(rc, REFUSED, said)
            (left,) = case.backups()
            self.assertEqual(run(case.config, "rollback", CID), nothing_applied)
            self.assertNotEqual(nothing_applied[0], 0, nothing_applied)
            self.assertEqual(case.skill.read_bytes(), case.original)
            self.assertEqual([p.name for p in left.iterdir()], ["SKILL.md.bak"])
            self.assertEqual(case.status(), "validated")

    def test_a_manifest_an_apply_stopped_in_the_same_second_left_is_kept_and_read(
        self,
    ) -> None:
        def stopped(*_args: Any, **_kwargs: Any) -> None:
            raise cli.Stopped(signal.SIGTERM)

        moment = datetime.datetime(2026, 2, 1, 9, 0, 0, tzinfo=datetime.timezone.utc)
        # The retry's copy fails, or its copy is taken and its manifest fails:
        # either way the stopped apply's backup.json is the same file after it,
        # not removed and not written again, even with the same bytes.
        for failing in ("write_bytes_atomic", "write_json"):
            with self.subTest(failing=failing), tempfile.TemporaryDirectory() as tmp:
                case = Validated(Path(tmp))
                with mock.patch.object(cli, "SystemClock", return_value=FrozenClock(moment)):
                    rc, said = case.apply_with("_write_pinned", stopped)
                    self.assertEqual(rc, 128 + signal.SIGTERM, said)
                    (manifest,) = case.state.glob("backups/*/backup.json")
                    written = manifest.read_bytes()
                    after_the_stop = run(case.config, "rollback", CID)
                    kept = manifest.stat()
                    rc, said = case.apply_with(failing, no_space)
                self.assertEqual(rc, REFUSED, said)
                self.assertEqual(case.backups(), [manifest.parent])
                self.assertEqual(
                    first_line(said),
                    refused_with_the_candidate(case.skill, manifest.parent, NO_SPACE),
                )
                self.assertEqual(manifest.read_bytes(), written)
                now = manifest.stat()
                self.assertEqual((now.st_ino, now.st_mtime_ns), (kept.st_ino, kept.st_mtime_ns))
                self.assertEqual(run(case.config, "rollback", CID), after_the_stop)
                self.assertIn(f"{CID} was never applied", after_the_stop[1])
                self.assertEqual(case.status(), "validated")
                self.assertEqual(case.skill.read_bytes(), case.original)
                rc, said = case.apply()
                self.assertEqual(rc, 0, said)
                self.assertEqual(case.status(), "applied")


class TheWriteRefusesTests(unittest.TestCase):
    def _store_and_skill(self, tmp: str) -> tuple[FilesystemSkillStore, Path, str]:
        config = build_env(Path(tmp) / "env", seed="c233")
        skill = (Path(tmp) / "env").resolve() / "skills-canonical" / "tdd" / "SKILL.md"
        store = FilesystemSkillStore(load_config(config).skill_roots, read_only=False)
        current = FilesystemSkillStore._document_sha256(skill.read_bytes(), skill)
        return store, skill, current

    def test_without_a_candidate_a_backup_directory_that_cannot_be_made_names_the_skill(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, skill, current = self._store_and_skill(tmp)
            original = skill.read_bytes()
            blocker = Path(tmp).resolve() / "a-file"
            blocker.write_text("not a directory\n")
            backup_dir = blocker / "backup"
            with self.assertRaises(BackupNotTaken) as ctx:
                store.write_canonical(
                    skill,
                    "new text\n",
                    expected_sha256=current,
                    backup_dir=backup_dir,
                    skill_id="tdd",
                )
            self.assertEqual(
                str(ctx.exception),
                f"refusing to write {skill}: its backup (the copy and its manifest) could not be "
                "written before the write ([Errno 20] Not a directory), so the skill was not "
                f"written. Whatever was written of the backup stays under {backup_dir}, and this "
                "write made no backup.json there.",
            )
            self.assertIsInstance(ctx.exception, SafetyRefusal)
            self.assertIsInstance(ctx.exception.__cause__, NotADirectoryError)
            self.assertEqual(skill.read_bytes(), original)

    def test_a_backup_directory_whose_mode_cannot_be_set_is_refused_with_the_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, skill, current = self._store_and_skill(tmp)
            original = skill.read_bytes()
            backup_dir = Path(tmp).resolve() / "backup"
            real_chmod = os.chmod
            denied = OSError(errno.EPERM, "Operation not permitted")

            def chmod(path: Any, mode: int, *args: Any, **kwargs: Any) -> None:
                if Path(path) == backup_dir:
                    raise denied
                real_chmod(path, mode, *args, **kwargs)

            with mock.patch.object(skill_fs.os, "chmod", chmod):
                with self.assertRaises(BackupNotTaken) as ctx:
                    store.write_canonical(
                        skill,
                        "new text\n",
                        expected_sha256=current,
                        backup_dir=backup_dir,
                        skill_id="tdd",
                        candidate_id=CID,
                    )
            self.assertTrue(
                str(ctx.exception).startswith(
                    f"refusing to apply {CID}: the backup of its canonical target {skill} (the "
                    "copy and its manifest) could not be written before the write ([Errno 1] "
                    "Operation not permitted), so the skill was not written"
                ),
                str(ctx.exception),
            )
            self.assertIs(ctx.exception.__cause__, denied)
            self.assertEqual(skill.read_bytes(), original)
            self.assertEqual(sorted(backup_dir.iterdir()), [])

    def test_the_manifest_not_written_keeps_its_error_as_the_cause(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, skill, current = self._store_and_skill(tmp)
            original = skill.read_bytes()
            backup_dir = Path(tmp).resolve() / "backup"
            full = OSError(errno.ENOSPC, "No space left on device")

            def write_json(*_args: Any, **_kwargs: Any) -> str:
                raise full

            with mock.patch.object(skill_fs, "write_json", write_json):
                with self.assertRaises(BackupNotTaken) as ctx:
                    store.write_canonical(
                        skill,
                        "new text\n",
                        expected_sha256=current,
                        backup_dir=backup_dir,
                        skill_id="tdd",
                        candidate_id=CID,
                    )
            self.assertIs(ctx.exception.__cause__, full)
            self.assertIn(f"({NO_SPACE})", str(ctx.exception))
            self.assertEqual(skill.read_bytes(), original)
            self.assertEqual([p.name for p in backup_dir.iterdir()], ["SKILL.md.bak"])


if __name__ == "__main__":
    unittest.main()
