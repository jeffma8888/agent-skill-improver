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
    UnreadableRecord,
    keyed_name,
)
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.errors import SafetyRefusal, StateUnreadable
from agent_skill_improver.jsonio import FILE_MODE, write_json, write_text_atomic

from tests.support import build_env, run_cli
from tests.test_candidate_integrity import build_candidate


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

    def test_a_reset_made_by_dir_before_the_walk_is_still_reported_once(self):
        """``dir()`` runs before ``ensure()`` in every command (wiring takes the
        sandbox scratch root first). Measured before this test existed, that
        early access reset a world-readable root and the walk that followed,
        finding it already fine, reported nothing."""
        sub, artifact = self._make_wide_tree()
        store = self.store()
        store.dir("raw")
        self.assertEqual(oct(mode_of(self.root)), oct(DIR_MODE), "dir() reset the root")
        self.assertEqual(oct(mode_of(sub)), oct(DIR_MODE), "and the directory it opened")
        changed = store.repair_permissions()
        self.assertEqual(changed[:2], [".", "raw"], "reported first, in the order they happened")
        self.assertEqual(changed.count("."), 1)
        self.assertIn(str(Path("raw") / "old-export.json"), changed)
        self.assertEqual(store.repair_permissions(), [], "and not reported twice")

    def test_a_directory_dir_creates_is_not_reported_as_a_repair(self):
        store = self.store()
        store.dir("raw")
        self.assertEqual(store.repair_permissions(), [])

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

    def test_a_dotted_path_through_a_symlinked_ancestor_is_refused_not_collapsed(self):
        """``parent/link/../state`` names ``parent/state`` on paper and
        ``<target's parent>/state`` on disk. The check used to collapse the
        dots first and never look at the link; the tree was then created
        through it."""
        deep = self.root.parent / "deep"
        deep.mkdir()
        outside = deep / "outside"
        outside.mkdir()
        os.chmod(outside, 0o755)
        (self.root.parent / "link").symlink_to(outside)
        dotted = self.root.parent / "link" / ".." / "state"
        store = StateStore(
            dotted,
            clock=FrozenClock(datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)),
        )

        with self.assertRaises(SafetyRefusal) as ctx:
            store.ensure()
        self.assertIn(f"ancestor {self.root.parent / 'link'} is a symlink to", str(ctx.exception))
        self.assertFalse((deep / "state").exists(), "the tree was created through the link")
        self.assertFalse(self.root.exists(), "nothing was created at the lexical location either")
        self.assertEqual(sorted(p.name for p in deep.iterdir()), ["outside"])
        self.assertEqual(oct(mode_of(outside)), oct(0o755))

    def test_a_dotted_path_that_steps_back_over_an_absent_component_still_meets_the_link(self):
        """``parent/missing/../link/state``: the walk must not stop at
        ``missing``, because ``..`` steps back onto ``parent`` and the link
        after it is exactly what the kernel will follow. Checked on both the
        eager and the lazy entry point, since ``dir()`` has no later verify to
        catch what the walk missed."""
        outside = self._outside_dir()
        (self.root.parent / "link").symlink_to(outside)
        dotted = self.root.parent / "missing" / ".." / "link" / "state"
        clock = FrozenClock(datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc))

        expected = f"ancestor {self.root.parent / 'link'} is a symlink to"
        with self.assertRaises(SafetyRefusal) as ctx:
            StateStore(dotted, clock=clock).ensure()
        self.assertIn(expected, str(ctx.exception))
        with self.assertRaises(SafetyRefusal) as ctx:
            StateStore(dotted, clock=clock).dir("conversations")
        self.assertIn(expected, str(ctx.exception))
        self._assert_outside_untouched(outside)
        self.assertFalse((self.root.parent / "missing").exists())

    def test_the_control_a_dotted_path_through_real_directories_is_accepted(self):
        """The same shape with a real directory in place of the link is just a
        long way of writing the lexical path, and lands there."""
        real = self.root.parent / "real"
        real.mkdir()
        dotted = real / ".." / "state"
        store = StateStore(
            dotted,
            clock=FrozenClock(datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)),
        )
        store.ensure()
        self.assertTrue(self.root.is_dir() and not self.root.is_symlink())
        self.assertEqual(oct(mode_of(self.root)), oct(DIR_MODE))

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

    def test_a_lazy_subdirectory_access_refuses_a_symlinked_ancestor_too(self):
        """``dir()`` is how every command that does not call ``ensure()`` reaches
        the tree (`candidates`, `review`, `apply`, wiring's sandbox root).
        Measured before this test existed, it checked the root and the
        subdirectory but not the ancestors, so `asi candidates` created the
        layout through an aliased parent that ``ensure()`` refuses."""
        outside = self._outside_dir()
        alias = self.root.parent / "alias"
        alias.symlink_to(outside)
        redirected_root = alias / "state"

        with self.assertRaises(SafetyRefusal) as ctx:
            StateStore(redirected_root).dir("candidates")

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

    def test_a_backup_belongs_to_exactly_the_candidate_whose_id_names_it(self):
        """The backup directory is ``<candidate id>--<stamp>``, and a candidate
        id may itself contain ``--``. Measured before this test, the lookup for
        ``CAND-a`` was a string-prefix glob, so it returned the manifest of
        ``CAND-a--b``'s apply: ``asi rollback CAND-a``, for a candidate that was
        never applied, would have restored another candidate's file and marked
        the wrong record rolled back. A rollback must find only its own apply."""
        store = self.store()
        store.ensure()
        other = store.backup_dir_for("CAND-a--b", "20260102-000000")
        write_json(other / "backup.json", {"schema": "asi.backup.v1"})
        self.assertIsNone(store.latest_backup_manifest("CAND-a"))
        self.assertEqual(store.latest_backup_manifest("CAND-a--b"), other / "backup.json")
        own = store.backup_dir_for("CAND-a", "20260103-000000")
        write_json(own / "backup.json", {"schema": "asi.backup.v1"})
        self.assertEqual(store.latest_backup_manifest("CAND-a"), own / "backup.json")
        self.assertEqual(store.latest_backup_manifest("CAND-a--b"), other / "backup.json")

    def test_a_stamp_that_would_blur_the_owner_is_refused_at_write_time(self):
        store = self.store()
        store.ensure()
        with self.assertRaises(ValueError):
            store.backup_dir_for("CAND-1", "2026--0102")
        self.assertEqual(list((self.root / "backups").iterdir()), [])

    def test_the_latest_backup_is_the_newest_stamp_for_that_candidate(self):
        store = self.store()
        store.ensure()
        older = store.backup_dir_for("CAND-1", "20260102-000000")
        newer = store.backup_dir_for("CAND-1", "20260105-000000")
        write_json(older / "backup.json", {"schema": "asi.backup.v1"})
        write_json(newer / "backup.json", {"schema": "asi.backup.v1"})
        self.assertEqual(store.latest_backup_manifest("CAND-1"), newer / "backup.json")

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

    def test_every_entry_the_mode_walk_skips_is_named_rather_than_silently_passed(self):
        """Skipping a link is the safe move; saying nothing about it is not. The
        store names what it stepped over so ``doctor`` can, and names nothing on
        a healthy tree."""
        store = self.store()
        store.ensure()
        self.assertEqual(store.foreign_entries(), [])
        outside = self._outside_dir()
        (self.root / "raw" / "link.json").symlink_to(outside / "keep.txt")
        (self.root / "receipts" / "elsewhere").symlink_to(outside)
        (self.root / "receipts" / "real.json").write_text("{}", encoding="utf-8")

        self.assertEqual(
            store.foreign_entries(),
            ["raw/link.json is a symlink", "receipts/elsewhere is a symlink"],
        )
        self.assertEqual(store.layout_problems(), [])
        store.repair_permissions()
        store.verify_permissions()
        self._assert_outside_untouched(outside)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "fifos are POSIX")
    def test_a_fifo_in_the_tree_is_named_left_alone_and_never_opened_for_long(self):
        """The one shape of foreign entry that is not a link. Its mode is not
        content, so the walk must not touch it, and a read must refuse it at
        once: ``open`` on a fifo with no writer otherwise never returns."""
        import signal

        from agent_skill_improver.app.state import _read_state_json
        from agent_skill_improver.errors import StateUnreadable

        store = self.store()
        store.ensure()
        pipe = self.root / "deferred" / "conv-9.json"
        os.mkfifo(pipe)
        os.chmod(pipe, 0o644)

        self.assertEqual(
            store.foreign_entries(), ["deferred/conv-9.json is not a regular file or directory"]
        )
        self.assertEqual(store.repair_permissions(), [])
        store.verify_permissions()
        self.assertEqual(oct(mode_of(pipe)), oct(0o644), "the fifo's mode is not ours to set")
        self.assertEqual(store.count("deferred"), 0)

        def hung(signum, frame):  # pragma: no cover - only fires on regression
            raise AssertionError("opening the fifo blocked instead of refusing")

        previous = signal.signal(signal.SIGALRM, hung)
        signal.alarm(5)
        try:
            with self.assertRaises(StateUnreadable) as ctx:
                _read_state_json(pipe)
            self.assertEqual(store.deferred_attempts("conv-9"), 0)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        self.assertIn("not a regular file", str(ctx.exception))

    def test_a_symlinked_leaf_is_not_counted_as_a_record(self):
        """Every read refuses a symlinked leaf, so a count that includes one
        promises a record no command will ever return."""
        store = self.store()
        store.ensure()
        outside = self._outside_dir()
        store.quarantine("conv-1", stage="scoring", reason="unit test")
        (self.root / "quarantine" / "planted.json").symlink_to(outside / "keep.txt")

        self.assertEqual(store.count("quarantine"), 1)
        self._assert_outside_untouched(outside)


class UnreadableCandidateRecordsAreNamedTests(StateTreeHarness):
    """``scan_candidates`` returns what read and what did not; nothing is dropped."""

    def test_a_truncated_record_is_returned_as_unreadable_with_its_reason(self):
        store = self.store()
        store.ensure()
        folder = store.dir("candidates")
        write_text_atomic(
            folder / "CAND-2026-01-01-01-tdd.json", '{"id": "CAND-2026-01-01-01-tdd", "sta'
        )
        write_text_atomic(folder / "CAND-2026-01-01-02-tdd.json", "[]")
        scan = store.scan_candidates()
        self.assertEqual(scan.readable, ())
        self.assertEqual(
            [u.path.name for u in scan.unreadable],
            ["CAND-2026-01-01-01-tdd.json", "CAND-2026-01-01-02-tdd.json"],
        )
        self.assertIsInstance(scan.unreadable[0], UnreadableRecord)
        self.assertIn("could not be read", scan.unreadable[0].reason)
        self.assertIn("not a JSON object", scan.unreadable[1].reason)
        self.assertEqual(store.list_candidates(), [], "the readable listing is unchanged")
        self.assertEqual(
            [u.to_dict()["path"] for u in store.unreadable_candidates()],
            [
                str(folder / "CAND-2026-01-01-01-tdd.json"),
                str(folder / "CAND-2026-01-01-02-tdd.json"),
            ],
        )

    def test_a_link_in_the_directory_is_a_foreign_entry_not_an_unreadable_record(self):
        store = self.store()
        store.ensure()
        outside = self.root.parent / "keep.txt"
        outside.write_text("keep me", encoding="utf-8")
        (store.dir("candidates") / "planted.json").symlink_to(outside)
        scan = store.scan_candidates()
        self.assertEqual((scan.readable, scan.unreadable), ((), ()))
        self.assertEqual(store.foreign_entries(), ["candidates/planted.json is a symlink"])

    def test_a_record_whose_filename_is_not_derived_from_its_id_is_unreadable_by_the_same_rule_as_read(
        self,
    ):
        store = self.store()
        store.ensure()
        folder = store.dir("candidates")
        candidate = build_candidate(candidate_id="CAND-2026-01-01-01-tdd")
        write_json(folder / "CAND-2026-01-01-01-tdd.json", candidate.to_dict())
        write_json(folder / "CAND-2026-01-01-09-tdd.json", candidate.to_dict())
        scan = store.scan_candidates()
        self.assertEqual([c.id for c in scan.readable], ["CAND-2026-01-01-01-tdd"])
        self.assertEqual([u.path.name for u in scan.unreadable], ["CAND-2026-01-01-09-tdd.json"])
        reason = scan.unreadable[0].reason
        self.assertIn("declares id 'CAND-2026-01-01-01-tdd'", reason)
        self.assertIn("does not derive this filename", reason)
        with self.assertRaises(StateUnreadable) as refused:
            store.read_candidate("CAND-2026-01-01-09-tdd")
        self.assertIn("declares id 'CAND-2026-01-01-01-tdd'", str(refused.exception))
        self.assertEqual(
            store.read_candidate("CAND-2026-01-01-01-tdd").id, "CAND-2026-01-01-01-tdd"
        )

    def test_the_control_an_empty_queue_scans_to_nothing_on_either_side(self):
        store = self.store()
        store.ensure()
        scan = store.scan_candidates()
        self.assertEqual((scan.readable, scan.unreadable), ((), ()))


class DeferralBookkeepingStepsOverForeignEntriesTests(StateTreeHarness):
    """A link planted where a deferral counter lives is read as no streak,
    written over (never through) and never removed.

    Measured before: ``deferred_attempts`` and ``defer`` raised
    ``StateUnreadable`` on the link, and ``clear_deferred`` unlinked it.
    """

    def plant(self, store: StateStore, record_id: str) -> tuple[Path, Path]:
        outside = self.root.parent / "keep.json"
        outside.write_text(json.dumps({"record_id": record_id, "attempts": 99}), encoding="utf-8")
        path = store.dir("deferred") / f"{keyed_name(record_id)}.json"
        path.symlink_to(outside)
        return path, outside

    def test_a_link_reads_as_no_streak_and_claims_no_priority(self):
        store = self.store()
        store.ensure()
        path, outside = self.plant(store, "rec-1")
        self.assertEqual(store.deferred_attempts("rec-1"), 0)
        self.assertEqual(store.deferred_ids(), [])
        self.assertTrue(path.is_symlink(), "a read changes nothing")

    def test_a_fresh_deferral_replaces_the_link_with_a_file_of_our_own(self):
        store = self.store()
        store.ensure()
        path, outside = self.plant(store, "rec-1")
        store.defer("rec-1", stage="ingest", reason="down")
        self.assertFalse(path.is_symlink())
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["attempts"], 1)
        self.assertEqual(
            json.loads(outside.read_text(encoding="utf-8"))["attempts"], 99, "not followed"
        )
        self.assertEqual(store.deferred_attempts("rec-1"), 1)
        self.assertEqual(store.deferred_ids(), ["rec-1"])

    def test_clearing_a_deferral_leaves_a_link_where_it_found_it(self):
        store = self.store()
        store.ensure()
        path, outside = self.plant(store, "rec-1")
        store.clear_deferred("rec-1")
        self.assertTrue(path.is_symlink(), "a foreign entry is not ours to remove")
        self.assertTrue(outside.is_file())

    def test_a_record_that_does_not_parse_reads_as_no_streak_and_is_rewritten(self):
        store = self.store()
        store.ensure()
        store.defer("rec-1", stage="ingest", reason="down")
        path = store.dir("deferred") / f"{keyed_name('rec-1')}.json"
        path.write_text("{garbage", encoding="utf-8")
        self.assertEqual(store.deferred_attempts("rec-1"), 0)
        self.assertEqual(store.deferred_ids(), [])
        store.defer("rec-1", stage="ingest", reason="down")
        self.assertEqual(store.deferred_attempts("rec-1"), 1)

    def test_a_counter_holding_json_that_is_not_an_object_reads_as_no_streak_and_is_rewritten(
        self,
    ):
        # Valid JSON, wrong shape: ``.get`` on a list, a string, a number or
        # ``null`` raised ``AttributeError`` out of ``defer`` (measured by the
        # cycle-67 audit, end to end: the daily run ended with no receipt).
        store = self.store()
        store.ensure()
        path = store.dir("deferred") / f"{keyed_name('rec-1')}.json"
        for planted in ("[]", '"x"', "5", "null", '{"attempts": "many"}'):
            with self.subTest(planted=planted):
                path.write_text(planted, encoding="utf-8")
                self.assertEqual(store.deferred_attempts("rec-1"), 0)
                self.assertEqual(store.deferred_ids(), [])
                store.defer("rec-1", stage="ingest", reason="down")
                self.assertEqual(store.deferred_attempts("rec-1"), 1)
                self.assertEqual(store.deferred_ids(), ["rec-1"])

    def test_the_control_a_genuine_deferral_counts_clears_and_is_removed(self):
        store = self.store()
        store.ensure()
        path = store.dir("deferred") / f"{keyed_name('rec-1')}.json"
        store.defer("rec-1", stage="ingest", reason="down")
        store.defer("rec-1", stage="ingest", reason="down")
        self.assertEqual(store.deferred_attempts("rec-1"), 2)
        self.assertEqual(store.deferred_ids(), ["rec-1"])
        store.clear_deferred("rec-1")
        self.assertFalse(path.exists())
        self.assertEqual(store.deferred_attempts("rec-1"), 0)


class CheckpointShapeIsCheckedOnReadTests(StateTreeHarness):
    """``read_checkpoint`` hands back only the shapes ``run_daily`` merges:
    ``runs`` an int, ``records`` and ``skills`` objects, ``candidates`` a list.
    A missing field is filled with the empty value; a present field of another
    shape is refused by name, like a truncated record.
    """

    def test_missing_fields_are_filled_with_empty_values(self):
        store = self.store()
        store.ensure()
        store.checkpoint_path("2026-01-02").write_text('{"schema": "asi.checkpoint.v1"}')
        data = store.read_checkpoint("2026-01-02")
        self.assertEqual(data["runs"], 0)
        self.assertEqual(data["records"], {})
        self.assertEqual(data["skills"], {})
        self.assertEqual(data["candidates"], [])

    def test_each_default_is_a_fresh_object_not_shared_between_reads(self):
        store = self.store()
        store.ensure()
        store.checkpoint_path("2026-01-02").write_text("{}")
        store.checkpoint_path("2026-01-03").write_text("{}")
        first = store.read_checkpoint("2026-01-02")
        first["records"]["x"] = {"status": "scored"}
        first["candidates"].append("c")
        second = store.read_checkpoint("2026-01-03")
        self.assertEqual(second["records"], {})
        self.assertEqual(second["candidates"], [])

    def test_a_field_of_the_wrong_shape_is_refused_naming_file_and_field(self):
        store = self.store()
        store.ensure()
        path = store.checkpoint_path("2026-01-02")
        for field, planted in (
            ("runs", "1"),
            ("runs", True),
            ("records", []),
            ("skills", 5),
            ("candidates", {}),
            ("candidates", None),
        ):
            with self.subTest(field=field, planted=planted):
                path.write_text(json.dumps({field: planted}), encoding="utf-8")
                with self.assertRaises(StateUnreadable) as caught:
                    store.read_checkpoint("2026-01-02")
                message = str(caught.exception)
                self.assertIn(str(path), message)
                self.assertIn(repr(field), message)
                self.assertIn(type(planted).__name__, message)
                with self.assertRaises(StateUnreadable):
                    store.processed_record_ids("2026-01-02")

    def test_a_candidate_entry_that_is_not_a_string_is_refused_with_the_container(self):
        # The ids are sorted and set-merged at the end of a pass; measured by the
        # cycle-68 audit: [1, "a"] and [null] raised TypeError there, after the
        # receipt was on disk, and [{"id": "a"}] was unhashable.
        store = self.store()
        store.ensure()
        path = store.checkpoint_path("2026-01-02")
        for planted in ([1, "a"], [None], [{"id": "a"}], ["a", 2, 3]):
            with self.subTest(planted=planted):
                path.write_text(json.dumps({"candidates": planted}), encoding="utf-8")
                with self.assertRaises(StateUnreadable) as caught:
                    store.read_checkpoint("2026-01-02")
                message = str(caught.exception)
                self.assertIn(str(path), message)
                self.assertIn("'candidates'", message)
                self.assertIn("not strings", message)
        path.write_text(json.dumps({"candidates": ["a", "b"]}), encoding="utf-8")
        self.assertEqual(store.read_checkpoint("2026-01-02")["candidates"], ["a", "b"])

    def test_the_control_a_well_formed_checkpoint_reads_back_as_written(self):
        store = self.store()
        store.ensure()
        payload = {
            "schema": "asi.checkpoint.v1",
            "runs": 2,
            "records": {"a": {"status": "scored"}},
            "skills": {"tdd": {"status": "proposed"}},
            "candidates": ["c1"],
        }
        store.write_checkpoint("2026-01-02", payload)
        self.assertEqual(store.read_checkpoint("2026-01-02"), payload)
        self.assertEqual(store.processed_record_ids("2026-01-02"), {"a"})


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

    def test_a_configured_state_root_that_is_a_symlink_is_refused_through_the_cli(self):
        """The store's root check is worthless if config resolves the link away
        before the store sees the path. Measured through the real commands: the
        configured ``state`` is a link to a 0755 directory with a file in it;
        ``doctor`` must report the layout and ``daily`` must refuse, and the
        target must keep its mode, its contents and nothing else."""
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            config = build_env(base, dry_run=True)
            outside = base / "outside"
            outside.mkdir()
            os.chmod(outside, 0o755)
            (outside / "keep.txt").write_text("keep me", encoding="utf-8")
            os.chmod(outside / "keep.txt", 0o644)
            (base / "state").symlink_to(outside)

            doctor = run_cli(["--config", str(config), "--json", "doctor"])
            daily = run_cli(["--config", str(config), "daily"])
            listing = run_cli(["--config", str(config), "candidates"])

            self.assertEqual(doctor.returncode, 1, doctor.stdout + doctor.stderr)
            payload = json.loads(doctor.stdout)
            self.assertEqual(payload["state_dir"], str(base / "state"), "the link, as configured")
            unsafe = payload["state_permissions"]["unsafe_layout"]
            self.assertEqual(len(unsafe), 1, unsafe)
            self.assertIn("is a symlink to", unsafe[0])
            self.assertEqual(daily.returncode, 3, daily.stdout + daily.stderr)
            self.assertIn("SafetyRefusal", daily.stderr)
            self.assertEqual(listing.returncode, 3, listing.stdout + listing.stderr)
            self.assertEqual(oct(mode_of(outside)), oct(0o755))
            self.assertEqual(oct(mode_of(outside / "keep.txt")), oct(0o644))
            self.assertEqual(sorted(p.name for p in outside.iterdir()), ["keep.txt"])

    def test_doctor_names_a_link_inside_a_confined_tree_as_a_warning_not_a_problem(self):
        """A link below the mandatory layout redirects nothing this tool writes
        (a write replaces it, a read refuses it), so the tree is still usable.
        It is still not this tool's entry, so ``doctor`` says it is there and
        that it was left alone, instead of calling the tree clean."""
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            config = build_env(base, dry_run=True)
            outside = base / "outside"
            outside.mkdir()
            os.chmod(outside, 0o755)
            (outside / "keep.txt").write_text("keep me", encoding="utf-8")
            os.chmod(outside / "keep.txt", 0o644)
            state = base / "state"
            StateStore(
                state,
                clock=FrozenClock(datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)),
            ).ensure()
            (state / "candidates" / "planted.json").symlink_to(outside / "keep.txt")
            (state / "receipts" / "elsewhere").symlink_to(outside)

            proc = run_cli(["--config", str(config), "--json", "doctor"])
            text = run_cli(["--config", str(config), "doctor"])

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            perms = payload["state_permissions"]
            self.assertEqual(
                perms["left_alone"],
                ["candidates/planted.json is a symlink", "receipts/elsewhere is a symlink"],
            )
            self.assertEqual(perms["unsafe_layout"], [])
            self.assertEqual(perms["still_wrong"], [])
            self.assertEqual(payload["problems"], [])
            self.assertTrue(
                any(
                    "left alone" in w and "candidates/planted.json" in w
                    for w in payload["warnings"]
                ),
                payload["warnings"],
            )
            self.assertIn("2 left alone", text.stdout)
            self.assertIn("WARN", text.stdout)
            self.assertIn("verdict: ready", text.stdout)
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

    def test_a_lock_mode_that_cannot_be_tightened_refuses_before_truncation(self):
        from unittest.mock import patch

        from agent_skill_improver.app.lock import RunLock

        store = self.store()
        store.ensure()
        path = store.dir("locks") / "daily.lock"
        path.write_text("keep me\n", encoding="utf-8")
        os.chmod(path, 0o666)

        with patch("agent_skill_improver.app.lock.os.fchmod", return_value=None):
            with self.assertRaises(SafetyRefusal) as ctx:
                with RunLock(path):
                    pass

        self.assertIn("0600", str(ctx.exception))
        self.assertEqual(path.read_text(encoding="utf-8"), "keep me\n")
        self.assertEqual(oct(mode_of(path)), oct(0o666))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "fifos are POSIX")
    def test_a_fifo_at_the_lock_path_is_refused_without_being_waited_on(self):
        """A fifo at the lock path used to be opened, chmodded and then read,
        and the read never returned. It is judged first now: a named refusal,
        its mode untouched, nothing written to it."""
        import signal

        from agent_skill_improver.app.lock import RunLock

        store = self.store()
        store.ensure()
        path = store.dir("locks") / "daily.lock"
        os.mkfifo(path)
        os.chmod(path, 0o644)

        def hung(signum, frame):  # pragma: no cover - only fires on regression
            raise AssertionError("the lock blocked on the fifo instead of refusing")

        previous = signal.signal(signal.SIGALRM, hung)
        signal.alarm(5)
        try:
            with self.assertRaises(SafetyRefusal) as ctx:
                with RunLock(path):
                    pass
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        self.assertIn("not a regular file", str(ctx.exception))
        self.assertTrue(stat.S_ISFIFO(os.lstat(path).st_mode), "the entry was not replaced")
        self.assertEqual(oct(mode_of(path)), oct(0o644), "the fifo's mode is not ours to set")

    def test_the_run_lock_is_0600_even_when_it_already_existed_wider(self):
        from agent_skill_improver.app.lock import RunLock

        store = self.store()
        store.ensure()
        path = store.dir("locks") / "daily.lock"
        path.write_text("stale\n", encoding="utf-8")
        os.chmod(path, 0o666)
        with RunLock(path):
            self.assertEqual(oct(mode_of(path)), oct(0o600))


class ConfigPathsFailClosedTests(unittest.TestCase):
    """A configured path may use ``~`` and ``$VARIABLE``. Measured before this
    class existed, an unset variable was kept as literal text, so ``state_dir =
    "$ASI_HOME/state"`` under a scheduler without ``ASI_HOME`` created a second
    private tree at ``<base>/$ASI_HOME/state``: the scheduled run's candidates
    landed where ``asi review`` from a shell would never look. Refused by name.
    """

    VAR = "ASI_TEST_UNSET_VARIABLE_7F3A"

    def _config_with_state_dir(self, base: Path, value: str) -> Path:
        config = Path(build_env(base, dry_run=True))
        text = config.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.startswith("state_dir = ")]
        assert len(lines) == 1, lines
        config.write_text(text.replace(lines[0], f'state_dir = "{value}"'), encoding="utf-8")
        return config

    def setUp(self) -> None:
        os.environ.pop(self.VAR, None)

    def test_a_path_naming_an_unset_variable_is_refused_with_the_name(self):
        from agent_skill_improver.config import load_config
        from agent_skill_improver.errors import ConfigError

        with TemporaryDirectory() as tmp:
            config = self._config_with_state_dir(Path(tmp), f"${self.VAR}/state")
            with self.assertRaises(ConfigError) as ctx:
                load_config(config)
            self.assertIn(self.VAR, str(ctx.exception))
            self.assertFalse((Path(tmp) / f"${self.VAR}").exists(), "nothing was created")

    def test_the_braced_form_is_refused_too(self):
        from agent_skill_improver.config import load_config
        from agent_skill_improver.errors import ConfigError

        with TemporaryDirectory() as tmp:
            config = self._config_with_state_dir(Path(tmp), f"${{{self.VAR}}}/state")
            with self.assertRaises(ConfigError) as ctx:
                load_config(config)
            self.assertIn(self.VAR, str(ctx.exception))

    def test_a_braced_form_with_a_shell_modifier_is_refused_even_when_the_variable_is_set(self):
        """``${NAME:-default}`` is what a shell user writes; ``expandvars`` does
        not understand it and keeps it literally whether or not NAME is set."""
        from agent_skill_improver.config import load_config
        from agent_skill_improver.errors import ConfigError

        with TemporaryDirectory() as tmp:
            os.environ[self.VAR] = str(Path(tmp) / "elsewhere")
            try:
                config = self._config_with_state_dir(Path(tmp), f"${{{self.VAR}:-/tmp}}/state")
                with self.assertRaises(ConfigError) as ctx:
                    load_config(config)
            finally:
                os.environ.pop(self.VAR, None)
            self.assertIn(self.VAR, str(ctx.exception))

    def test_a_tilde_naming_no_user_is_refused(self):
        from agent_skill_improver.config import load_config
        from agent_skill_improver.errors import ConfigError

        with TemporaryDirectory() as tmp:
            config = self._config_with_state_dir(Path(tmp), "~no-such-user-7f3a/state")
            with self.assertRaises(ConfigError) as ctx:
                load_config(config)
            self.assertIn("~", str(ctx.exception))

    def test_the_control_a_set_variable_expands(self):
        from agent_skill_improver.config import load_config

        with TemporaryDirectory() as tmp:
            os.environ[self.VAR] = str(Path(tmp) / "elsewhere")
            try:
                config = self._config_with_state_dir(Path(tmp), f"${self.VAR}/state")
                loaded = load_config(config)
            finally:
                os.environ.pop(self.VAR, None)
            self.assertEqual(loaded.state_dir, (Path(tmp) / "elsewhere" / "state").resolve())

    def test_doctor_reports_it_as_a_config_error_rather_than_creating_the_tree(self):
        with TemporaryDirectory() as tmp:
            config = self._config_with_state_dir(Path(tmp), f"${self.VAR}/state")
            proc = run_cli(["--config", str(config), "doctor"])
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertIn("config error", proc.stderr)
            self.assertIn(self.VAR, proc.stderr)
            self.assertFalse((Path(tmp) / f"${self.VAR}").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
