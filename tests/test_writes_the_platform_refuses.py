"""Rollback writes only where apply writes, and a refused write is named.

Measured before (3.11 and 3.13): a backup manifest edited to name the mirror's
copy of a skill, with ``new_sha256`` set to that copy's hash, made ``rollback``
overwrite the mirror and report success (apply refuses that target as a
mirror); a manifest naming a backup outside its own directory whose hash
matched was restored from; and a skill directory at 0555 made ``apply`` and
``rollback`` fall with a raw ``PermissionError`` from the temp file, exit 70.
"""

from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.app.state import StateStore
from agent_skill_improver.cli import RC_OK, RC_REFUSED
from agent_skill_improver.domain.skills import SKILL_FILENAME
from agent_skill_improver.errors import (
    BackupUnusable,
    MirrorEditRefused,
    SafetyRefusal,
    SkillWriteFailed,
)
from agent_skill_improver.hashing import sha256_file

from tests.support import build_env, run_cli
from tests.test_skill_identity_and_payloads import GOOD, Harness, deadline

AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0
FD_DIR = "/dev/fd"


def open_descriptors() -> int:
    """How many descriptors this process holds, where the platform lists them."""
    return len(os.listdir(FD_DIR))


class _ReadOnlyDirectory:
    """``path`` at 0555 inside the block, back to 0755 after it."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> Path:
        os.chmod(self.path, 0o555)
        return self.path

    def __exit__(self, *exc) -> bool:
        os.chmod(self.path, 0o755)
        return False


class RollbackAndApplyStoreTests(Harness):
    def apply_a_change(self, backup: str = "bak") -> tuple[Path, Path, str]:
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        previous = target.read_text(encoding="utf-8")
        receipt = self.store(read_only=False).write_canonical(
            target,
            GOOD + "\n- an approved addition\n",
            expected_sha256=sha256_file(target),
            backup_dir=self.base / backup,
            skill_id="shared",
        )
        return target, Path(receipt.backup_manifest_path), previous

    def edit(self, manifest: Path, **fields: object) -> None:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        raw.update(fields)
        manifest.write_text(json.dumps(raw), encoding="utf-8")

    def leftovers(self, directory: Path) -> list[str]:
        return sorted(p.name for p in directory.iterdir() if p.name != SKILL_FILENAME)

    @unittest.skipIf(AS_ROOT, "root writes through directory permissions")
    def test_an_apply_into_a_directory_it_cannot_write_is_refused_with_the_skill_unchanged(self):
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        before = target.read_text(encoding="utf-8")
        with _ReadOnlyDirectory(target.parent):
            with self.assertRaises(SkillWriteFailed) as ctx:
                self.store(read_only=False).write_canonical(
                    target,
                    GOOD + "\n- an approved addition\n",
                    expected_sha256=sha256_file(target),
                    backup_dir=self.base / "bak",
                    skill_id="shared",
                )
        self.assertIn("The skill is unchanged", str(ctx.exception))
        self.assertIsInstance(ctx.exception, SafetyRefusal, "a refusal, exit 3, never exit 70")
        self.assertEqual(target.read_text(encoding="utf-8"), before)
        self.assertEqual(self.leftovers(target.parent), [], "no temp file is left behind")

    @unittest.skipIf(AS_ROOT, "root writes through directory permissions")
    def test_a_rollback_into_a_directory_it_cannot_write_is_refused_with_the_applied_text_kept(
        self,
    ):
        target, manifest, _previous = self.apply_a_change()
        applied = target.read_text(encoding="utf-8")
        with _ReadOnlyDirectory(target.parent):
            with self.assertRaises(SkillWriteFailed) as ctx:
                self.store(read_only=False).rollback(manifest)
        self.assertIn("still holds the text this apply wrote", str(ctx.exception))
        self.assertEqual(target.read_text(encoding="utf-8"), applied)
        self.assertEqual(self.leftovers(target.parent), [])
        # The control: with the directory writable again, the same manifest restores.
        self.store(read_only=False).rollback(manifest)
        self.assertEqual(target.read_text(encoding="utf-8"), _previous)

    def test_a_rollback_aimed_at_a_mirror_copy_is_refused_as_apply_refuses_it(self):
        _target, manifest, _previous = self.apply_a_change()
        mirror = self.mirror_dir / "shared" / SKILL_FILENAME
        before = mirror.read_text(encoding="utf-8")
        self.edit(manifest, target_path=str(mirror.resolve()), new_sha256=sha256_file(mirror))
        with self.assertRaises(MirrorEditRefused):
            self.store(read_only=False).rollback(manifest)
        self.assertEqual(mirror.read_text(encoding="utf-8"), before)

    def test_a_rollback_aimed_outside_every_root_is_refused(self):
        _target, manifest, _previous = self.apply_a_change()
        stray = self.base / "elsewhere" / "shared" / SKILL_FILENAME
        stray.parent.mkdir(parents=True)
        stray.write_text(GOOD + "\n- an approved addition\n", encoding="utf-8")
        before = stray.read_text(encoding="utf-8")
        self.edit(manifest, target_path=str(stray.resolve()), new_sha256=sha256_file(stray))
        with self.assertRaises(MirrorEditRefused):
            self.store(read_only=False).rollback(manifest)
        self.assertEqual(stray.read_text(encoding="utf-8"), before)

    def test_a_backup_outside_its_manifests_directory_is_refused(self):
        target, manifest, _previous = self.apply_a_change()
        applied = target.read_text(encoding="utf-8")
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        elsewhere = self.base / "elsewhere" / "SKILL.md.bak"
        elsewhere.parent.mkdir(parents=True)
        shutil.copyfile(raw["backup_path"], elsewhere)
        self.edit(manifest, backup_path=str(elsewhere))
        with self.assertRaises(BackupUnusable) as ctx:
            self.store(read_only=False).rollback(manifest)
        self.assertIn("is not in the directory of its manifest", str(ctx.exception))
        self.assertEqual(target.read_text(encoding="utf-8"), applied)

    def test_the_control_a_backup_spelled_another_way_in_that_directory_is_restored(self):
        target, manifest, previous = self.apply_a_change()
        backup = Path(json.loads(manifest.read_text(encoding="utf-8"))["backup_path"])
        respelled = backup.parent / ".." / backup.parent.name / backup.name
        self.assertNotEqual(respelled.parent, manifest.parent, "spelled apart, same directory")
        self.edit(manifest, backup_path=str(respelled))
        self.store(read_only=False).rollback(manifest)
        self.assertEqual(target.read_text(encoding="utf-8"), previous)

    def test_a_backup_that_is_not_utf8_is_refused_before_anything_is_written(self):
        """Read with its bad bytes replaced, a backup that is not UTF-8 hashed as
        the text it stood in for whenever that text held U+FFFD at the spot."""
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        target.write_text(GOOD + "\n- a mark kept as written: \ufffd\n", encoding="utf-8")
        receipt = self.store(read_only=False).write_canonical(
            target,
            GOOD + "\n- an approved addition\n",
            expected_sha256=sha256_file(target),
            backup_dir=self.base / "bak",
            skill_id="shared",
        )
        manifest = Path(receipt.backup_manifest_path)
        applied = target.read_bytes()
        backup = Path(json.loads(manifest.read_text(encoding="utf-8"))["backup_path"])
        mark = "\ufffd".encode("utf-8")
        self.assertEqual(backup.read_bytes().count(mark), 1)
        backup.write_bytes(backup.read_bytes().replace(mark, b"\xff"))
        with self.assertRaises(BackupUnusable) as ctx:
            self.store(read_only=False).rollback(manifest)
        self.assertIn("is not valid UTF-8 at byte", str(ctx.exception))
        self.assertEqual(target.read_bytes(), applied)

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_a_backup_that_is_now_a_symbolic_link_is_refused(self):
        target, manifest, _previous = self.apply_a_change()
        applied = target.read_text(encoding="utf-8")
        backup = Path(json.loads(manifest.read_text(encoding="utf-8"))["backup_path"])
        twin = self.base / "twin.bak"
        shutil.copyfile(backup, twin)
        backup.unlink()
        os.symlink(twin, backup)
        with self.assertRaises(BackupUnusable) as ctx:
            self.store(read_only=False).rollback(manifest)
        self.assertIn("a symbolic link", str(ctx.exception))
        self.assertEqual(target.read_text(encoding="utf-8"), applied)

    def test_the_bytes_restored_are_the_bytes_whose_hash_was_checked(self):
        """The backup was hashed, then read again for the write: a backup that
        changed between the two was written into the skill, and only the
        verification after the write noticed, with the skill already rewritten.
        Changed straight after its one read, so a hash or a write that went
        back to the path would meet the changed bytes."""
        from unittest import mock

        from agent_skill_improver.adapters import skill_fs

        target, manifest, previous = self.apply_a_change()
        backup = Path(json.loads(manifest.read_text(encoding="utf-8"))["backup_path"])
        real = skill_fs._read_regular_file
        reads: list[Path] = []

        def read_then_change_the_backup(path: Path) -> bytes | None:
            reads.append(Path(path))
            data = real(path)
            backup.write_text("changed after it was read\n", encoding="utf-8")
            return data

        with mock.patch.object(skill_fs, "_read_regular_file", read_then_change_the_backup):
            result = self.store(read_only=False).rollback(manifest)
        self.assertEqual(reads, [backup], "the backup is read once")
        self.assertFalse(result["already_restored"])
        self.assertEqual(target.read_text(encoding="utf-8"), previous)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "platform has no fifos")
    def test_a_link_or_fifo_swapped_in_after_the_checks_is_never_read_through(self):
        """The checks before the read go by path, and the entry can change after
        them; the read opens without following a link and judges the open
        handle, so what is judged is what is read."""
        from unittest import mock

        real_is_symlink, real_is_file = Path.is_symlink, Path.is_file
        for i, kind in enumerate(("a link", "a fifo")):
            with self.subTest(kind):
                target, manifest, previous = self.apply_a_change(f"bak-{i}")
                try:
                    applied = target.read_text(encoding="utf-8")
                    raw = json.loads(manifest.read_text(encoding="utf-8"))
                    backup = Path(raw["backup_path"])
                    if kind == "a link":
                        twin = self.base / f"twin-{i}.bak"
                        shutil.copyfile(backup, twin)
                        backup.unlink()
                        os.symlink(twin, backup)
                    else:
                        backup.unlink()
                        os.mkfifo(backup)

                    def is_symlink(path: Path) -> bool:
                        return False if path == backup else real_is_symlink(path)

                    def is_file(path: Path) -> bool:
                        return True if path == backup else real_is_file(path)

                    with (
                        mock.patch.object(Path, "is_symlink", is_symlink),
                        mock.patch.object(Path, "is_file", is_file),
                    ):
                        with deadline(5.0), self.assertRaises(BackupUnusable) as ctx:
                            self.store(read_only=False).rollback(manifest)
                    self.assertRegex(
                        str(ctx.exception), "could not be checked|is not a regular file"
                    )
                    self.assertEqual(target.read_text(encoding="utf-8"), applied)
                finally:
                    target.write_text(previous, encoding="utf-8")

    @unittest.skipUnless(os.path.isdir(FD_DIR), "platform does not list its descriptors")
    def test_a_directory_swapped_in_for_the_backup_is_refused_and_no_descriptor_is_left_open(
        self,
    ):
        from unittest import mock

        target, manifest, _previous = self.apply_a_change()
        applied = target.read_text(encoding="utf-8")
        backup = Path(json.loads(manifest.read_text(encoding="utf-8"))["backup_path"])
        backup.unlink()
        backup.mkdir()
        real_is_file = Path.is_file

        def is_file(path: Path) -> bool:
            return True if path == backup else real_is_file(path)

        store = self.store(read_only=False)
        before = open_descriptors()
        with mock.patch.object(Path, "is_file", is_file):
            with self.assertRaises(BackupUnusable) as ctx:
                store.rollback(manifest)
        self.assertEqual(open_descriptors(), before, "the handle on the directory was closed")
        self.assertIn("is not a regular file", str(ctx.exception))
        self.assertEqual(target.read_text(encoding="utf-8"), applied)


@unittest.skipIf(AS_ROOT, "root writes through directory permissions")
class TheApplyCommandOnADirectoryItCannotWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name).resolve()
        self.cfg = [
            "--config",
            str(
                build_env(
                    base,
                    rubric_ids=("instruction-compliance",),
                    use_subprocess_model=True,
                    dry_run=False,
                    seed="unwritable-skill-dir",
                )
            ),
        ]
        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        listed = json.loads(run_cli([*self.cfg, "--json", "candidates"]).stdout)["candidates"]
        self.candidate_id = sorted(c["id"] for c in listed)[0]
        self.state = StateStore(base / "state")

    def _record(self) -> dict:
        return json.loads(self.state.candidate_path(self.candidate_id).read_text("utf-8"))

    def test_the_command_names_the_refused_write_and_the_candidate_stays_validated(self):
        review = json.loads(run_cli([*self.cfg, "--json", "review", self.candidate_id]).stdout)
        skill = Path(self._record()["target"]["canonical_path"])
        before = skill.read_text(encoding="utf-8")
        apply = [
            *self.cfg,
            "apply",
            self.candidate_id,
            "--approve",
            review["approval_token"],
            "--i-reviewed-this",
            "--operator",
            "tester",
        ]
        with _ReadOnlyDirectory(skill.parent):
            proc = run_cli(apply)
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
        self.assertIn("SkillWriteFailed", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(skill.read_text(encoding="utf-8"), before)
        self.assertEqual(self._record()["status"], "validated")
        # The control: with the directory writable again, the same command applies.
        self.assertEqual(run_cli(apply).returncode, RC_OK)
        self.assertEqual(self._record()["status"], "applied")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
