"""State directory permissions, measured from ``st_mode`` rather than asserted.

Gitignoring the state tree stops it reaching a remote. It does nothing about the
other account on the machine, and on a shared or managed host that account is the
normal case rather than the exotic one. Everything under the state root is
derived from private conversation content -- raw exports first of all -- so the
tree is 0700 and every artifact in it is 0600.

Two things make these tests real rather than decorative.

**They run under ``umask(0)``.** A permissive umask is exactly the condition that
produces the defect: ``mkdir`` lands 0755 and ``open`` lands 0666. A test run
under the developer's ordinary 022 umask would pass against code that sets no
mode at all, which is the version of this test that proves nothing.

**They measure the mode.** ``chmod`` succeeding is not evidence the mode changed:
on a filesystem that carries no POSIX modes it is accepted and does nothing. So
every assertion reads ``stat().st_mode`` back off disk.
"""

from __future__ import annotations

import datetime
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.app.state import (
    DIR_MODE,
    STATE_FILE_MODE,
    SUBDIRS,
    StateStore,
)
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.errors import SafetyRefusal
from agent_skill_improver.jsonio import FILE_MODE, write_json, write_text_atomic


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class PermissiveUmask:
    """Force the worst realistic umask for the duration of a test."""

    def __enter__(self) -> "PermissiveUmask":
        self._previous = os.umask(0)
        return self

    def __exit__(self, *exc: object) -> None:
        os.umask(self._previous)


class StateTreeHarness(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name) / "state"
        self._umask = PermissiveUmask().__enter__()
        self.addCleanup(lambda: self._umask.__exit__())
        self.addCleanup(self._tmp.cleanup)

    def store(self) -> StateStore:
        return StateStore(
            self.root,
            clock=FrozenClock(datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)),
        )


class CreatedTreeIsOwnerOnlyTests(StateTreeHarness):
    def test_the_state_root_is_0700_even_under_a_permissive_umask(self):
        store = self.store()
        store.ensure()
        self.assertEqual(oct(mode_of(self.root)), oct(DIR_MODE))

    def test_every_declared_subdirectory_is_0700(self):
        store = self.store()
        store.ensure()
        wrong = {
            name: oct(mode_of(self.root / name))
            for name in SUBDIRS
            if mode_of(self.root / name) != DIR_MODE
        }
        self.assertEqual(wrong, {}, f"expected {oct(DIR_MODE)} on every subdirectory")

    def test_the_readme_is_0600(self):
        store = self.store()
        store.ensure()
        self.assertEqual(oct(mode_of(self.root / "README.txt")), oct(STATE_FILE_MODE))

    def test_the_umask_really_was_permissive_during_the_test(self):
        """The control. Without it these tests could pass on a 077 umask against
        code that sets no mode at all, which is the version that proves nothing.
        """
        control = self.root.parent / "control-dir"
        control.mkdir()
        (control / "f.txt").write_text("x", encoding="utf-8")
        self.assertEqual(oct(mode_of(control)), oct(0o777))
        self.assertEqual(oct(mode_of(control / "f.txt")), oct(0o666))


class EveryArtifactIsOwnerOnlyTests(StateTreeHarness):
    def test_a_raw_export_is_0600(self):
        """The most sensitive artifact this tool writes."""
        store = self.store()
        store.ensure()
        path = store.write_raw("thread-1", '{"thread": {"id": "thread-1"}}')
        self.assertEqual(oct(mode_of(path)), oct(STATE_FILE_MODE))

    def test_conversations_scorecards_failures_and_receipts_are_all_0600(self):
        store = self.store()
        store.ensure()
        written = [
            store.write_conversation("c1", {"schema": "x"}),
            store.write_scorecard("c1", "r1", {"schema": "x"}),
            store.write_failures("c1", "r1", [{"criterion_id": "a"}]),
            store.write_receipt("2026-01-02", {"schema": "x"}, dry_run=True),
            store.write_checkpoint("2026-01-02", {"schema": "x"}),
            store.quarantine("c1", stage="score", reason="because"),
            store.refuse("c1", stage="propose", reason="because"),
            store.defer("c1", stage="ingest", reason="because"),
        ]
        wrong = {str(p): oct(mode_of(p)) for p in written if mode_of(p) != STATE_FILE_MODE}
        self.assertEqual(wrong, {})

    def test_a_backup_directory_is_0700(self):
        store = self.store()
        store.ensure()
        path = store.backup_dir_for("cand-1", "20260102-000000")
        self.assertEqual(oct(mode_of(path)), oct(DIR_MODE))

    def test_the_atomic_writer_defaults_to_0600_and_not_to_the_umask(self):
        target = self.root / "plain.txt"
        write_text_atomic(target, "hello")
        self.assertEqual(oct(mode_of(target)), oct(FILE_MODE))

    def test_the_atomic_writer_honours_an_explicit_mode(self):
        """The control for the default: the parameter must actually be wired.

        A writer that hard-coded 0600 would pass the test above while silently
        tightening a skill file it was asked to leave alone.
        """
        target = self.root / "explicit.txt"
        write_text_atomic(target, "hello", mode=0o644)
        self.assertEqual(oct(mode_of(target)), oct(0o644))

    def test_write_json_also_lands_0600(self):
        target = self.root / "obj.json"
        write_json(target, {"a": 1})
        self.assertEqual(oct(mode_of(target)), oct(FILE_MODE))

    def test_no_temp_file_is_left_behind_at_a_wider_mode(self):
        """The rename preserves the temp file's mode, so the temp file is where a
        wider mode would come from. Nothing may survive the write."""
        target = self.root / "obj.json"
        write_json(target, {"a": 1})
        leftovers = [p.name for p in self.root.iterdir() if p.name.startswith(".tmp-")]
        self.assertEqual(leftovers, [])


class ExistingStateIsRepairedTests(StateTreeHarness):
    def _make_wide_tree(self) -> tuple[Path, Path]:
        """A tree that looks like one created before the policy existed."""
        self.root.mkdir(parents=True)
        os.chmod(self.root, 0o755)
        sub = self.root / "raw"
        sub.mkdir()
        os.chmod(sub, 0o755)
        artifact = sub / "old-export.json"
        artifact.write_text("{}", encoding="utf-8")
        os.chmod(artifact, 0o644)
        return sub, artifact

    def test_a_pre_existing_world_readable_tree_is_detected(self):
        sub, artifact = self._make_wide_tree()
        store = self.store()
        problems = store.permission_problems()
        joined = "\n".join(problems)
        self.assertIn("raw", joined)
        self.assertIn("old-export.json", joined)

    def test_it_is_repaired_and_the_repair_is_reported(self):
        sub, artifact = self._make_wide_tree()
        store = self.store()
        changed = store.repair_permissions()
        self.assertIn(".", changed)
        self.assertIn("raw", changed)
        self.assertIn(str(Path("raw") / "old-export.json"), changed)
        self.assertEqual(oct(mode_of(self.root)), oct(DIR_MODE))
        self.assertEqual(oct(mode_of(sub)), oct(DIR_MODE))
        self.assertEqual(oct(mode_of(artifact)), oct(STATE_FILE_MODE))
        self.assertEqual(store.permission_problems(), [])

    def test_ensure_repairs_a_pre_existing_tree_rather_than_refusing_it(self):
        sub, artifact = self._make_wide_tree()
        self.store().ensure()
        self.assertEqual(oct(mode_of(artifact)), oct(STATE_FILE_MODE))

    def test_a_repair_that_cannot_take_effect_is_a_refusal_not_a_warning(self):
        """The fail-closed half.

        Simulated by a store whose chmod does nothing, which is exactly what a
        filesystem carrying no POSIX modes looks like. Without this test the
        verification step could be deleted and everything else would still pass.
        """
        sub, artifact = self._make_wide_tree()

        class UnchmodableStore(StateStore):
            def repair_permissions(self) -> list[str]:
                return []

        store = UnchmodableStore(self.root)
        with self.assertRaises(SafetyRefusal) as ctx:
            store.ensure()
        message = str(ctx.exception)
        self.assertIn("owner-only", message)
        self.assertIn("private conversation content", message)

    def test_the_control_shows_a_healthy_tree_does_not_refuse(self):
        store = self.store()
        store.ensure()
        store.verify_permissions()
        self.assertEqual(store.permission_problems(), [])

    def test_a_symlink_in_the_tree_is_skipped_rather_than_followed(self):
        """chmod follows symlinks, so a symlink pointing outside the state tree
        would let a repair walk change a file the tool does not own."""
        store = self.store()
        store.ensure()
        outside = self.root.parent / "outside.txt"
        outside.write_text("x", encoding="utf-8")
        os.chmod(outside, 0o644)
        (self.root / "raw" / "link.json").symlink_to(outside)
        store.repair_permissions()
        self.assertEqual(oct(mode_of(outside)), oct(0o644))


class LockFileTests(StateTreeHarness):
    def test_the_run_lock_is_0600_even_when_it_already_existed_wider(self):
        from agent_skill_improver.app.lock import RunLock

        store = self.store()
        store.ensure()
        path = store.dir("locks") / "daily.lock"
        path.write_text("stale\n", encoding="utf-8")
        os.chmod(path, 0o666)
        with RunLock(path):
            self.assertEqual(oct(mode_of(path)), oct(0o600))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
