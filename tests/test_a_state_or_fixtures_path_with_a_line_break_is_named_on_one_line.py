"""A refusal naming a state or fixtures path with a line break in it stays on one line.

A directory name may hold a line break, and such a config is supported: a
launchd job carries the path exactly. The refusals the state store and the
fixtures source raise named every path under ``[run].state_dir`` and
``[source].fixtures_dir`` as written, so the text after the break stood at
column 0 of stderr as a line of its own. Measured before (3.11 and 3.13), each
path ending ``\\nconfig error: forged``: a ``state_dir`` that cannot be created
made a 2-line refusal (1 forged) in ``doctor`` and ``daily``; a ``state_dir``
with a mandatory subdirectory replaced by a file made ``daily``'s refusal 3
lines (2 forged); a missing ``fixtures_dir`` made ``daily``'s 2 lines (1
forged). Both modules now spell a line break in each path they name.
"""

from __future__ import annotations

import json
import os
import re
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver import SCHEMA_RECEIPT
from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.app import lock as lock_module
from agent_skill_improver.app import promote as promote_module
from agent_skill_improver.app import state as state_module
from agent_skill_improver.app.state import SUBDIRS, StateStore
from agent_skill_improver.app.lock import RunLock
from agent_skill_improver.errors import LockHeld, RecordDefect, SafetyRefusal, SourceUnavailable
from tests.support import build_env, run_cli

DAY = "2026-03-01"
FORGED = "config error: forged"


class StateAndFixturesPathOnOneLineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base / "env", dry_run=False, seed="paths"))

    def set_path(self, key: str, value: str) -> None:
        text = self.config_path.read_text(encoding="utf-8")
        new, count = re.subn(
            rf"^{re.escape(key)} = .*$",
            lambda _: f"{key} = {json.dumps(value)}",
            text,
            count=1,
            flags=re.M,
        )
        self.assertEqual(count, 1, key)
        self.config_path.write_text(new, encoding="utf-8")

    def run_refused(self, *command: str) -> str:
        """The command refuses with exit 3 on one stderr line that spells the break."""
        proc = run_cli(["--config", str(self.config_path), *command])
        self.assertEqual(proc.returncode, 3, proc.stderr)
        lines = proc.stderr.splitlines()
        self.assertEqual(len(lines), 1, proc.stderr)
        self.assertIn(f"\\n{FORGED}", lines[0])
        self.assertFalse(lines[0].startswith(FORGED), proc.stderr)
        return lines[0]

    def test_a_state_dir_that_cannot_be_created_is_refused_on_one_line(self):
        # The break in the state directory's own name, then in the ancestor
        # that is in the way, which the reason names too.
        for where in ("leaf", "ancestor"):
            holder = self.base / where
            if where == "ancestor":
                holder = holder / f"d\n{FORGED}"
            holder.mkdir(parents=True)
            blocker = holder / "a-file"
            blocker.write_text("x", encoding="utf-8")
            leaf = f"s\n{FORGED}" if where == "leaf" else "s"
            self.set_path("state_dir", f"{blocker}/{leaf}")
            for command in (["doctor"], ["daily", "--day", DAY, "--commit"]):
                with self.subTest(where=where, command=command[0]):
                    line = self.run_refused(*command)
                    self.assertTrue(line.startswith("SafetyRefusal: refusing to use "), line)
                    if where == "ancestor":
                        self.assertEqual(line.count(f"\\n{FORGED}"), 2, line)

    def test_a_state_dir_with_a_subdirectory_replaced_by_a_file_is_refused_on_one_line(
        self,
    ):
        state = self.base / f"st\n{FORGED}"
        self.set_path("state_dir", str(state))
        self.assertEqual(run_cli(["--config", str(self.config_path), "doctor"]).returncode, 0)
        replaced = sorted(p for p in state.iterdir() if p.is_dir())[0]
        replaced.rmdir()
        replaced.write_text("x", encoding="utf-8")
        before = sorted(p.name for p in state.rglob("*"))
        line = self.run_refused("daily", "--day", DAY, "--commit")
        self.assertIn(f"/{replaced.name} as a state directory: ", line)
        self.assertEqual(sorted(p.name for p in state.rglob("*")), before)

    def test_the_layout_refusal_keeps_its_own_lines_and_spells_the_path_inside_them(
        self,
    ):
        # A fresh tree whose first subdirectory is a file meets the refusal that
        # lists its reasons one per line: those lines stay, the path is spelled.
        state = self.base / f"st\n{FORGED}"
        state.mkdir()
        (state / SUBDIRS[0]).write_text("x", encoding="utf-8")
        self.set_path("state_dir", str(state))
        proc = run_cli(["--config", str(self.config_path), "daily", "--day", DAY, "--commit"])
        self.assertEqual(proc.returncode, 3, proc.stderr)
        lines = proc.stderr.splitlines()
        self.assertFalse(any(ln.lstrip().startswith(FORGED) for ln in lines), proc.stderr)
        head = [ln for ln in lines if ln.startswith("SafetyRefusal: ")]
        self.assertEqual(len(head), 1, proc.stderr)
        self.assertTrue(head[0].endswith(f"under {state.parent}/st\\n{FORGED}."), head[0])
        self.assertIn(f"  - {SUBDIRS[0]} exists and is not a directory", lines)
        self.assertTrue((state / SUBDIRS[0]).is_file())

    def test_a_missing_fixtures_dir_is_refused_on_one_line(self):
        self.set_path("fixtures_dir", str(self.base / f"nope\n{FORGED}"))
        line = self.run_refused("daily", "--day", DAY, "--commit")
        self.assertTrue(line.startswith("SourceUnavailable: fixtures_dir "), line)
        # doctor reports it as a finding; its listing was already one line per entry.
        proc = run_cli(["--config", str(self.config_path), "doctor"])
        self.assertEqual(proc.returncode, 1, proc.stderr)
        for text in (proc.stdout, proc.stderr):
            self.assertFalse(any(ln.lstrip().startswith(FORGED) for ln in text.splitlines()))

    def test_an_unreadable_state_record_under_such_a_path_is_named_on_one_line(self):
        records = self.base / f"st\n{FORGED}"
        records.mkdir()
        with self.assertRaises(SafetyRefusal) as ctx:
            state_module._read_state_object(
                self.write(records / "cut.json", '{"id": '), what="receipt"
            )
        self.assert_spelled(ctx.exception)
        for name, content in (
            ("truncated", '{"id": '),
            ("list", "[]"),
            ("shapeless", '{"id": 1}'),
        ):
            path = records / f"{name}.json"
            path.write_text(content, encoding="utf-8")
            with self.subTest(record=name):
                with self.assertRaises(SafetyRefusal) as ctx:
                    state_module._candidate_from_file(path)
                self.assertNotIn("\n", str(ctx.exception))
                self.assertIn(f"\\n{FORGED}", str(ctx.exception))
        with self.assertRaises(SafetyRefusal) as ctx:
            state_module._read_state_object(records / "list.json", what="receipt")
        self.assertNotIn("\n", str(ctx.exception))
        (records / "link.json").symlink_to(records / "list.json")
        (records / "dir.json").mkdir()
        for leaf in ("link.json", "dir.json"):
            with self.subTest(leaf=leaf):
                with self.assertRaises(SafetyRefusal) as ctx:
                    state_module._read_state_json(records / leaf)
                self.assertNotIn("\n", str(ctx.exception))
                self.assertIn(f"\\n{FORGED}/{leaf}: the leaf is ", str(ctx.exception))

    def write(self, path: Path, text: str) -> Path:
        path.write_text(text, encoding="utf-8")
        return path

    def assert_spelled(self, error: object) -> None:
        message = str(error)
        self.assertNotIn("\n", message)
        self.assertIn(f"\\n{FORGED}", message)

    def test_each_other_state_refusal_under_such_a_root_is_named_on_one_line(self):
        root = self.base / f"st\n{FORGED}"
        root.mkdir(mode=0o700)
        store = StateStore(root)
        record = root / "record.json"
        record.mkdir()
        with self.subTest(site="a directory at a record path"):
            with self.assertRaises(SafetyRefusal) as ctx:
                store._persist(record, lambda: None)
            self.assert_spelled(ctx.exception)
        with self.subTest(site="modes a repair could not reset"):
            reasons = ["x is 0o644", "y is 0o755"]
            with mock.patch.object(store, "permission_problems", return_value=reasons):
                with self.assertRaises(SafetyRefusal) as ctx:
                    store.verify_permissions()
            # A refusal that lists its reasons one per line keeps them; the root is spelled.
            lines = str(ctx.exception).splitlines()
            self.assertEqual(len(lines), 4, lines)
            self.assertEqual(lines[1:3], ["  - x is 0o644", "  - y is 0o755"])
            self.assertTrue(lines[3].startswith("  Expected "), lines)
            self.assertFalse(any(ln.lstrip().startswith(FORGED) for ln in lines), lines)
            self.assertIn(f"under {self.base}/st\\n{FORGED}. ", str(ctx.exception))
        with self.subTest(site="a record that claims another id"):
            cand = f"CAND-{DAY}-01-asked"
            self.write(store.candidate_path(cand), "{}")
            other = SimpleNamespace(id=f"CAND-{DAY}-01-other")
            with mock.patch.object(state_module, "_candidate_from_file", return_value=other):
                with self.assertRaises(SafetyRefusal) as ctx:
                    store.read_candidate(cand)
            self.assert_spelled(ctx.exception)
        run = SimpleNamespace(path=root / "receipts" / "absent.json")
        with self.subTest(site="moving a blocked receipt in a dry run"):
            with self.assertRaises(SafetyRefusal) as ctx:
                StateStore(root, dry_run=True).keep_blocked_receipt(run)
            self.assert_spelled(ctx.exception)
        with self.subTest(site="a blocked receipt that cannot be moved"):
            kept = root / "receipts" / "kept.json"
            with mock.patch.object(StateStore, "kept_blocked_receipt_path", return_value=kept):
                with self.assertRaises(SafetyRefusal) as ctx:
                    store.keep_blocked_receipt(run)
            self.assert_spelled(ctx.exception)
            self.assertEqual(str(ctx.exception).count(f"\\n{FORGED}"), 2)
        with self.subTest(site="a symlinked leaf where O_NOFOLLOW is missing"):
            (root / "link.json").symlink_to(record)
            with mock.patch.object(state_module.os, "O_NOFOLLOW", 0):
                with self.assertRaises(SafetyRefusal) as ctx:
                    state_module._read_state_json(root / "link.json")
            self.assert_spelled(ctx.exception)
        with self.subTest(site="a child name that is a path"):
            with self.assertRaises(SafetyRefusal) as ctx:
                state_module._ensure_child_dir(root, "a/b", root)
            self.assert_spelled(ctx.exception)
        with self.subTest(site="a child that is not a real directory after creation"):
            elsewhere = self.base / "elsewhere"
            elsewhere.mkdir()

            def make_a_link(path: Path, _root: Path, **_: object) -> None:
                path.symlink_to(elsewhere)

            with (
                mock.patch.object(state_module, "_mkdir_or_refuse", make_a_link),
                mock.patch.object(state_module, "_chmod_to", lambda *_: True),
            ):
                with self.assertRaises(SafetyRefusal) as ctx:
                    state_module._ensure_child_dir(root, "child", root)
            self.assert_spelled(ctx.exception)
            self.assertEqual(str(ctx.exception).count(f"\\n{FORGED}"), 2)

    def test_an_ancestor_reason_under_such_a_directory_is_named_on_one_line(self):
        target = self.base / "real"
        target.mkdir()
        link = self.base / f"l\n{FORGED}"
        link.symlink_to(target)
        reason = state_module._ancestor_layout_reason(link / "state")
        self.assertIsNotNone(reason)
        self.assertTrue(reason.startswith("ancestor "), reason)
        self.assert_spelled(reason)
        if os.geteuid() == 0:  # pragma: no cover - root reads a mode-000 directory
            return
        shut = self.base / f"d\n{FORGED}"
        shut.mkdir()
        shut.chmod(0)
        self.addCleanup(shut.chmod, 0o700)
        reason = state_module._ancestor_layout_reason(shut / "x" / "state")
        self.assertIsNotNone(reason)
        self.assertIn(" cannot be inspected ", reason)
        self.assert_spelled(reason)

    def test_a_fixtures_dir_that_cannot_be_listed_is_named_on_one_line(self):
        if os.geteuid() == 0:  # pragma: no cover - root lists a mode-000 directory
            self.skipTest("root can list a mode-000 directory")
        fixtures = self.base / f"fx\n{FORGED}"
        fixtures.mkdir()
        fixtures.chmod(0)
        self.addCleanup(fixtures.chmod, 0o700)
        with self.assertRaises(SourceUnavailable) as ctx:
            FixtureConversationSource(fixtures).preflight()
        self.assertIn(" cannot be listed ", str(ctx.exception))
        self.assert_spelled(ctx.exception)

    def test_a_checkpoint_of_the_wrong_shape_under_such_a_state_dir_is_named_on_one_line(self):
        state = self.base / f"st\n{FORGED}"
        self.set_path("state_dir", str(state))
        self.assertEqual(run_cli(["--config", str(self.config_path), "doctor"]).returncode, 0)
        self.write(StateStore(state).checkpoint_path(DAY), '{"records": "x"}').chmod(0o600)
        line = self.run_refused("daily", "--day", DAY, "--commit")
        self.assertIn("refusing to use the checkpoint record ", line)

    def test_each_reason_a_state_scan_returns_names_its_path_on_one_line(self):
        root = self.base / f"st\n{FORGED}"
        root.mkdir(mode=0o700)
        store = StateStore(root)
        with self.subTest(reason="the run lock path"):
            store.lock_path().mkdir(parents=True)
            self.assert_spelled(store.lock_entry_problem())
        with self.subTest(reason="an action receipt path, as apply refuses it"):
            cand = SimpleNamespace(id=f"CAND-{DAY}-01-skill")
            store.action_receipt_path("apply", cand.id).mkdir(parents=True)
            with self.assertRaises(SafetyRefusal) as ctx:
                promote_module._refuse_unwritable_receipt(store, "apply", cand)
            self.assert_spelled(ctx.exception)
        with self.subTest(reason="a receipt of the wrong shape"):
            found = store._run_at(
                self.write(root / "r.json", json.dumps({"schema": SCHEMA_RECEIPT}))
            )
            self.assert_spelled(found.reason)
        with self.subTest(reason="a set-aside record of the wrong shape"):
            self.write(store.dir("quarantine") / "q.json", '{"stage": "propose"}')
            (bad,) = store.scan_set_aside().unreadable
            self.assert_spelled(bad.reason)
        with self.subTest(reason="a candidate filed under another name"):
            self.write(store.dir("candidates") / f"CAND-{DAY}-01-filed.json", "{}")
            other = SimpleNamespace(id=f"CAND-{DAY}-01-other")
            with mock.patch.object(state_module, "_candidate_from_file", return_value=other):
                (bad,) = store.scan_candidates().unreadable
            self.assert_spelled(bad.reason)
        with self.subTest(reason="a mode a repair could not reset"):
            self.write(root / f"x\n{FORGED}.json", "{}")
            with mock.patch.object(state_module, "_mode_ok", return_value=False):
                problems = store.permission_problems()
            self.assertTrue(any(f"x\\n{FORGED}.json is " in p for p in problems), problems)
            self.assertFalse(any("\n" in p for p in problems), problems)

    def test_a_line_break_and_a_carriage_return_are_both_spelled(self):
        self.assertEqual(state_module._shown(Path("a\rb\nc")), "a\\rb\\nc")
        self.assertEqual(RunLock(Path("a\rb\nc"))._shown, "a\\rb\\nc")

    def test_each_run_lock_refusal_under_such_a_state_dir_is_named_on_one_line(self):
        locks = self.base / f"st\n{FORGED}" / "locks"
        locks.mkdir(parents=True)

        def refused(path: Path, error: type = SafetyRefusal) -> None:
            with self.assertRaises(error) as ctx:
                with RunLock(path):
                    pass
            self.assert_spelled(ctx.exception)

        with self.subTest(site="a symlinked leaf"):
            (locks / "link.lock").symlink_to(self.base)
            refused(locks / "link.lock")
            with mock.patch.object(lock_module.os, "O_NOFOLLOW", 0):
                refused(locks / "link.lock")
        with self.subTest(site="a directory at the path"):
            (locks / "dir.lock").mkdir()
            refused(locks / "dir.lock")
        with self.subTest(site="a fifo at the path"):
            os.mkfifo(locks / "fifo.lock")
            refused(locks / "fifo.lock")
        with self.subTest(site="a lock file that cannot be inspected"):
            with mock.patch.object(lock_module.os, "fstat", side_effect=OSError(5, "EIO")):
                refused(locks / "stat.lock")
        with self.subTest(site="a mode that cannot be set"):
            with mock.patch.object(lock_module.os, "fchmod", side_effect=OSError(1, "EPERM")):
                refused(locks / "chmod.lock")
        with self.subTest(site="a mode that does not stick"):
            self.write(locks / "mode.lock", "").chmod(0o644)
            with mock.patch.object(lock_module.os, "fchmod", lambda *_: None):
                refused(locks / "mode.lock")
        with self.subTest(site="another holder"):
            with RunLock(locks / "held.lock"):
                refused(locks / "held.lock", LockHeld)

    def test_reviewing_an_absent_candidate_under_such_a_state_dir_is_named_on_one_line(
        self,
    ):
        self.set_path("state_dir", str(self.base / f"st\n{FORGED}"))
        self.assertEqual(run_cli(["--config", str(self.config_path), "doctor"]).returncode, 0)
        proc = run_cli(["--config", str(self.config_path), "review", f"CAND-{DAY}-01-absent"])
        self.assertEqual(proc.returncode, 4, proc.stderr)
        lines = proc.stderr.splitlines()
        self.assertEqual(len(lines), 1, proc.stderr)
        self.assertIn(f"\\n{FORGED}/candidates/CAND-{DAY}-01-absent.json", lines[0])

    def test_a_fixture_lookup_under_such_a_directory_is_named_on_one_line(self):
        fixtures = self.base / f"fx\n{FORGED}"
        fixtures.mkdir()
        with self.assertRaises(RecordDefect) as ctx:
            FixtureConversationSource(fixtures).fetch("absent-thread")
        self.assertNotIn("\n", str(ctx.exception))
        (fixtures / f"bad\n{FORGED}.json").write_text("not json", encoding="utf-8")
        with self.assertRaises(RecordDefect) as ctx:
            FixtureConversationSource(fixtures).fetch("absent-thread")
        message = str(ctx.exception)
        self.assertNotIn("\n", message)
        self.assertEqual(message.count(f"\\n{FORGED}"), 3, message)

    def test_the_control_a_path_without_a_line_break_is_named_exactly_as_written(self):
        blocker = self.base / "a-file"
        blocker.write_text("x", encoding="utf-8")
        state = f"{blocker}/s\tcafé"
        self.set_path("state_dir", state)
        proc = run_cli(["--config", str(self.config_path), "daily", "--day", DAY, "--commit"])
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertIn(f"refusing to use {state} as the state directory", proc.stderr)


if __name__ == "__main__":
    unittest.main()
