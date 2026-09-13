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

The last class here measures a different property: that the tree being made
private is the tree that was configured. ``mkdir(exist_ok=True)``, ``chmod`` and
every later write follow a symlink, so a state root or a mandatory subdirectory
precreated as a link redirects the private tree somewhere else entirely -- and a
mode measured through the link then describes a directory this tool does not own.
Those tests therefore assert the refusal *and* that the outside target's mode and
contents are exactly what they were, because a refusal that has already chmodded
the target is not a refusal.
"""

from __future__ import annotations

import datetime
import json
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

from tests.support import build_env, run_cli


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
        # Healthy callers pass a canonical path. StateStore deliberately rejects
        # every existing symlink component, including macOS's /var alias.
        self.root = Path(self._tmp.name).resolve() / "state"
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
    def test_distinct_external_ids_never_collide_on_one_state_leaf(self):
        store = self.store()
        store.ensure()
        first = store.write_raw("a/b", "first")
        second = store.write_raw("a_b", "second")
        third = store.write_raw("a?b", "third")
        fourth = store.write_raw("a:b", "fourth")

        self.assertEqual(len({first, second, third, fourth}), 4)
        self.assertEqual(first.read_text(encoding="utf-8"), "first")
        self.assertEqual(second.read_text(encoding="utf-8"), "second")
        self.assertEqual(third.read_text(encoding="utf-8"), "third")
        self.assertEqual(fourth.read_text(encoding="utf-8"), "fourth")

    def test_distinct_keys_never_collide_in_any_keyed_state_namespace(self):
        store = self.store()
        store.ensure()
        pairs = [
            (
                store.quarantine("a/b", stage="x", reason="first"),
                store.quarantine("a_b", stage="x", reason="second"),
            ),
            (
                store.refuse("a?b", stage="x", reason="first"),
                store.refuse("a:b", stage="x", reason="second"),
            ),
            (store.checkpoint_path("a/b"), store.checkpoint_path("a_b")),
            (store.receipt_path("a?b", dry_run=False), store.receipt_path("a:b", dry_run=False)),
        ]
        for first, second in pairs:
            self.assertNotEqual(first, second)

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
        path = store.backup_dir_for("CAND-1", "20260102-000000")
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


class StateLayoutIsConfinedToItsRootTests(StateTreeHarness):
    """A state directory must be a real directory, checked before it is followed.

    Gitignoring and chmodding the tree are both worthless if the tree is not the
    one that was configured. These are the cases where an entry the store treats
    as its own directory is really a link to somewhere else.
    """

    def _outside_dir(self) -> Path:
        outside = self.root.parent / "outside"
        outside.mkdir()
        os.chmod(outside, 0o755)
        (outside / "keep.txt").write_text("keep me", encoding="utf-8")
        os.chmod(outside / "keep.txt", 0o644)
        return outside

    def _assert_outside_untouched(self, outside: Path) -> None:
        self.assertEqual(oct(mode_of(outside)), oct(0o755))
        self.assertEqual(oct(mode_of(outside / "keep.txt")), oct(0o644))
        self.assertEqual(sorted(p.name for p in outside.iterdir()), ["keep.txt"])

    def test_an_existing_symlink_in_a_state_root_ancestor_is_refused_before_creation(self):
        outside = self._outside_dir()
        alias = self.root.parent / "alias"
        alias.symlink_to(outside)
        redirected_root = alias / "state"

        with self.assertRaises(SafetyRefusal) as ctx:
            StateStore(redirected_root).ensure()

        self.assertIn("ancestor", str(ctx.exception))
        self.assertFalse((outside / "state").exists())
        self._assert_outside_untouched(outside)

    def test_a_state_root_that_is_a_symlink_is_refused_before_anything_is_created(self):
        outside = self._outside_dir()
        self.root.symlink_to(outside)
        store = self.store()
        with self.assertRaises(SafetyRefusal) as ctx:
            store.ensure()
        self.assertIn("real directory", str(ctx.exception))
        self._assert_outside_untouched(outside)
        self.assertEqual(
            store.layout_problems(),
            [f". is a symlink to {str(outside)!r}, not a real directory"],
        )

    def test_a_symlinked_state_root_receives_no_layout_and_no_artifact(self):
        outside = self._outside_dir()
        self.root.symlink_to(outside)
        store = self.store()
        with self.assertRaises(SafetyRefusal):
            store.ensure()
        with self.assertRaises(SafetyRefusal):
            store.write_raw("thread-1", '{"thread": {"id": "thread-1"}}')
        self._assert_outside_untouched(outside)

    def test_a_mandatory_subdirectory_that_is_a_symlink_is_refused(self):
        outside = self._outside_dir()
        self.root.mkdir(parents=True)
        os.chmod(self.root, DIR_MODE)
        (self.root / "raw").symlink_to(outside)
        store = self.store()
        with self.assertRaises(SafetyRefusal) as ctx:
            store.ensure()
        message = str(ctx.exception)
        self.assertIn("raw", message)
        self.assertIn("real directory", message)
        self._assert_outside_untouched(outside)

    def test_a_write_through_a_symlinked_mandatory_subdirectory_is_refused(self):
        """The write path, not only ``ensure``. ``dir()`` is what every write goes
        through, so a store that was never ``ensure``d must refuse there too."""
        outside = self._outside_dir()
        self.root.mkdir(parents=True)
        os.chmod(self.root, DIR_MODE)
        (self.root / "raw").symlink_to(outside)
        store = self.store()
        with self.assertRaises(SafetyRefusal):
            store.write_raw("thread-1", '{"thread": {"id": "thread-1"}}')
        self.assertFalse((outside / "thread-1.json").exists())
        self._assert_outside_untouched(outside)

    def test_a_repair_walk_does_not_chmod_through_a_redirected_layout(self):
        """The repair is the dangerous half: it exists to chmod, and it must not
        chmod a directory reached through a link the tool does not own."""
        outside = self._outside_dir()
        self.root.mkdir(parents=True)
        os.chmod(self.root, 0o755)
        (self.root / "raw").symlink_to(outside)
        store = self.store()
        self.assertEqual(store.repair_permissions(), [])
        self.assertEqual(store.permission_problems(), [])
        self.assertEqual(len(store.layout_problems()), 1)
        self._assert_outside_untouched(outside)
        self.assertEqual(oct(mode_of(self.root)), oct(0o755), "the root was not chmodded either")

    def test_a_mandatory_state_directory_that_is_a_regular_file_is_refused(self):
        self.root.mkdir(parents=True)
        os.chmod(self.root, DIR_MODE)
        (self.root / "raw").write_text("not a directory", encoding="utf-8")
        store = self.store()
        self.assertEqual(store.layout_problems(), ["raw exists and is not a directory"])
        with self.assertRaises(SafetyRefusal):
            store.ensure()

    def test_a_precreated_backup_directory_symlink_is_refused(self):
        """The dynamically named case. The stamp is guessable, so an entry that
        already exists under ``backups/`` must be proven rather than accepted."""
        store = self.store()
        store.ensure()
        outside = self._outside_dir()
        (self.root / "backups" / "CAND-1--20260102-000000").symlink_to(outside)
        with self.assertRaises(SafetyRefusal) as ctx:
            store.backup_dir_for("CAND-1", "20260102-000000")
        self.assertIn("real directory", str(ctx.exception))
        self._assert_outside_untouched(outside)

    def test_a_backup_directory_is_created_as_a_direct_child_of_backups(self):
        """The control for the case above: the healthy path still creates a real
        0700 directory, directly beneath the mandatory ``backups`` directory."""
        store = self.store()
        store.ensure()
        path = store.backup_dir_for("CAND-1", "20260102-000000")
        self.assertFalse(path.is_symlink())
        self.assertTrue(path.is_dir())
        self.assertEqual(path.parent, self.root / "backups")
        self.assertEqual(path.parent.resolve(), (self.root / "backups").resolve())
        self.assertEqual(oct(mode_of(path)), oct(DIR_MODE))

    def test_a_backup_manifest_is_not_read_through_a_symlinked_backup_directory(self):
        store = self.store()
        store.ensure()
        outside = self._outside_dir()
        write_json(outside / "backup.json", {"schema": "asi.backup.v1"})
        (self.root / "backups" / "CAND-1--20260102-000000").symlink_to(outside)
        self.assertIsNone(store.latest_backup_manifest("CAND-1"))

    def test_a_backup_manifest_leaf_symlink_is_not_selected(self):
        store = self.store()
        store.ensure()
        outside = self.root.parent / "outside-manifest.json"
        outside.write_text('{"schema": "asi.backup.v1"}\n', encoding="utf-8")
        backup = store.backup_dir_for("CAND-1", "20260102-000000")
        (backup / "backup.json").symlink_to(outside)

        self.assertIsNone(store.latest_backup_manifest("CAND-1"))
        self.assertEqual(outside.read_text(encoding="utf-8"), '{"schema": "asi.backup.v1"}\n')

    def test_the_control_shows_a_healthy_layout_reports_no_problems(self):
        store = self.store()
        store.ensure()
        self.assertEqual(store.layout_problems(), [])
        store.verify_layout()
        for name in SUBDIRS:
            path = self.root / name
            self.assertTrue(path.is_dir() and not path.is_symlink(), name)

    def test_an_arbitrary_leaf_symlink_is_still_skipped_rather_than_refused(self):
        """The confinement check covers the root and the mandatory directories.
        A link some other tool left in the tree must still be walked past without
        touching its target, which is the behaviour that already existed."""
        store = self.store()
        store.ensure()
        outside = self.root.parent / "outside.txt"
        outside.write_text("x", encoding="utf-8")
        os.chmod(outside, 0o644)
        (self.root / "raw" / "link.json").symlink_to(outside)
        self.assertEqual(store.layout_problems(), [])
        store.repair_permissions()
        store.verify_permissions()
        self.assertEqual(oct(mode_of(outside)), oct(0o644))


class DoctorReportsAnUnsafeStateLayoutTests(unittest.TestCase):
    """``doctor`` runs on a tree nobody has ensured yet, so it is the command most
    likely to meet this layout first. It must report it and repair nothing."""

    def test_doctor_reports_a_symlinked_state_subdirectory_without_chmodding_it(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = build_env(base, dry_run=True)
            outside = base / "outside"
            outside.mkdir()
            os.chmod(outside, 0o755)
            (outside / "keep.txt").write_text("keep me", encoding="utf-8")
            os.chmod(outside / "keep.txt", 0o644)
            state = base / "state"
            state.mkdir()
            os.chmod(state, DIR_MODE)
            (state / "raw").symlink_to(outside)

            proc = run_cli(["--config", str(config), "--json", "doctor"])

            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            unsafe = payload["state_permissions"]["unsafe_layout"]
            self.assertEqual(len(unsafe), 1, unsafe)
            self.assertIn("raw", unsafe[0])
            self.assertTrue(
                any("not confined" in problem for problem in payload["problems"]),
                payload["problems"],
            )
            self.assertEqual(payload["state_permissions"]["repaired"], [])
            self.assertEqual(oct(mode_of(outside)), oct(0o755))
            self.assertEqual(oct(mode_of(outside / "keep.txt")), oct(0o644))
            self.assertEqual(sorted(p.name for p in outside.iterdir()), ["keep.txt"])


class LockFileTests(StateTreeHarness):
    def test_a_regular_lock_still_contends_and_releases_normally(self):
        from agent_skill_improver.app.lock import RunLock
        from agent_skill_improver.errors import LockHeld

        store = self.store()
        store.ensure()
        path = store.dir("locks") / "daily.lock"
        with RunLock(path, owner="first"):
            with self.assertRaises(LockHeld):
                with RunLock(path, owner="second"):
                    pass
        with RunLock(path, owner="second"):
            self.assertEqual(path.read_text(encoding="utf-8"), "second\n")

    def test_a_symlinked_lock_leaf_is_refused_without_touching_its_target(self):
        from agent_skill_improver.app.lock import RunLock

        store = self.store()
        store.ensure()
        outside = self.root.parent / "outside.lock"
        outside.write_text("keep me\n", encoding="utf-8")
        os.chmod(outside, 0o644)
        path = store.dir("locks") / "daily.lock"
        path.symlink_to(outside)

        with self.assertRaises(SafetyRefusal) as ctx:
            with RunLock(path):
                pass

        self.assertIn("symlink", str(ctx.exception))
        self.assertEqual(outside.read_text(encoding="utf-8"), "keep me\n")
        self.assertEqual(oct(mode_of(outside)), oct(0o644))

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
