"""``daily`` creates nothing beside a state layout it refuses.

Wiring the model runner opened the sandbox scratch root through
``StateStore.dir``, which checks only the root, its ancestors and ``sandbox``
itself before it creates the directory. A layout with any other mandatory entry
that is not a real directory is refused later, by ``ensure``. Measured before
(3.11 and 3.13): a fresh state directory whose ``raw`` (or ``locks``) was a
file was left with a new ``sandbox`` directory by the run that then refused its
layout, and the root's mode was reset under a note that said "the refusal below
wrote nothing else". The runner now gets no scratch root when the layout is
one the run will refuse, unless ``dir`` would have refused first, in its own
words, as before.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import stat
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver import cli as cli_module
from agent_skill_improver.app.state import SUBDIRS, StateStore
from agent_skill_improver.errors import SafetyRefusal
from tests.support import build_env, run_cli

DAY = "2026-03-01"


class DailyCreatesNothingBesideARefusedLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()

    def fresh_tree(self, name: str, entry: str) -> tuple[Path, Path]:
        """A config whose state directory holds only a file named ``entry``."""
        config = Path(build_env(self.base / name / "env", dry_run=False, seed="layout"))
        state = self.base / name / "st"
        state.mkdir()
        (state / entry).write_text("x", encoding="utf-8")
        text = config.read_text(encoding="utf-8")
        new, count = re.subn(
            r"^state_dir = .*$",
            lambda _: f"state_dir = {json.dumps(str(state))}",
            text,
            count=1,
            flags=re.M,
        )
        self.assertEqual(count, 1)
        config.write_text(new, encoding="utf-8")
        return config, state

    def snapshot(self, state: Path) -> tuple[list[str], int]:
        return sorted(str(p.relative_to(state)) for p in state.rglob("*")), state.stat().st_mode

    def test_a_fresh_tree_whose_raw_is_a_file_gets_no_sandbox(self):
        for flags in (["--commit"], []):
            with self.subTest(commit=bool(flags)):
                config, state = self.fresh_tree(f"raw{len(flags)}", "raw")
                before = self.snapshot(state)
                proc = run_cli(["--config", str(config), "daily", "--day", DAY, *flags])
                self.assertEqual(proc.returncode, 3, proc.stderr)
                lines = proc.stderr.splitlines()
                self.assertTrue(lines[0].startswith("SafetyRefusal: refusing to use the state "))
                self.assertIn("  - raw exists and is not a directory", lines)
                self.assertFalse(any(ln.startswith("note: ") for ln in lines), proc.stderr)
                self.assertEqual(self.snapshot(state), before)

    def test_each_other_mandatory_entry_that_is_a_file_leaves_the_tree_as_found(self):
        for entry in SUBDIRS:
            if entry == "sandbox":
                continue
            with self.subTest(entry=entry):
                config, state = self.fresh_tree(entry, entry)
                before = self.snapshot(state)
                proc = run_cli(["--config", str(config), "daily", "--day", DAY, "--commit"])
                self.assertEqual(proc.returncode, 3, proc.stderr)
                self.assertIn(f"  - {entry} exists and is not a directory", proc.stderr)
                self.assertEqual(self.snapshot(state), before)

    def test_the_control_a_sandbox_that_is_a_file_is_refused_in_the_words_of_dir(self):
        config, state = self.fresh_tree("sandbox", "sandbox")
        before = self.snapshot(state)
        proc = run_cli(["--config", str(config), "daily", "--day", DAY, "--commit"])
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertEqual(len(proc.stderr.splitlines()), 1, proc.stderr)
        self.assertTrue(
            proc.stderr.startswith(
                f"SafetyRefusal: refusing to use {state}/sandbox as a state directory: "
                "sandbox exists and is not a directory."
            ),
            proc.stderr,
        )
        self.assertEqual(self.snapshot(state), before)

    def test_the_control_a_sound_tree_still_runs(self):
        config, state = self.fresh_tree("sound", "README.txt")
        proc = run_cli(["--config", str(config), "daily", "--day", DAY])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue((state / "sandbox").is_dir())

    def test_a_built_tree_with_drifted_modes_beside_such_an_entry_is_left_as_found(self):
        # A built tree's root and scratch root at 0755 beside a file at `raw`:
        # the run used to reset both through the scratch root, then refuse.
        config, state = self.fresh_tree("built", "README.txt")
        StateStore(state).ensure()
        shutil.rmtree(state / "raw")
        (state / "raw").write_text("x", encoding="utf-8")
        os.chmod(state, 0o755)
        os.chmod(state / "sandbox", 0o755)
        before = sorted(str(p.relative_to(state)) for p in state.rglob("*"))
        proc = run_cli(["--config", str(config), "daily", "--day", DAY, "--commit"])
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertIn("  - raw exists and is not a directory", proc.stderr.splitlines())
        self.assertFalse(any(ln.startswith("note: ") for ln in proc.stderr.splitlines()))
        self.assertEqual(stat.S_IMODE(state.stat().st_mode), 0o755)
        self.assertEqual(stat.S_IMODE((state / "sandbox").stat().st_mode), 0o755)
        self.assertEqual(sorted(str(p.relative_to(state)) for p in state.rglob("*")), before)

    def test_the_runner_gets_its_scratch_root_unless_the_layout_will_be_refused(self):
        given: list[object] = []
        real = cli_module.build_runner

        def recording(config, state=None):
            given.append(state)
            return real(config, state)

        for entry, rc in (("README.txt", 0), ("raw", 3)):
            with self.subTest(entry=entry):
                given.clear()
                config, state = self.fresh_tree(f"wired-{entry}", entry)
                with mock.patch.object(cli_module, "build_runner", side_effect=recording):
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        code = cli_module.main(["--config", str(config), "daily", "--day", DAY])
                self.assertEqual(code, rc)
                self.assertEqual(len(given), 1, given)
                if rc == 0:
                    self.assertIsInstance(given[0], StateStore)
                    self.assertEqual(given[0].root, state)
                else:
                    self.assertIsNone(given[0])

    def test_check_dir_creates_nothing_and_refuses_as_dir_does(self):
        absent = StateStore(self.base / "absent")
        self.assertEqual(absent.check_dir("sandbox"), self.base / "absent" / "sandbox")
        self.assertFalse((self.base / "absent").exists())
        with self.assertRaises(KeyError):
            absent.check_dir("nope")
        (self.base / "real").mkdir()
        (self.base / "link").symlink_to(self.base / "real")
        (self.base / "held").mkdir()
        (self.base / "held" / "sandbox").write_text("x", encoding="utf-8")
        for root in (self.base / "link", self.base / "held"):
            with self.subTest(root=root.name):
                store = StateStore(root)
                with self.assertRaises(SafetyRefusal) as checked:
                    store.check_dir("sandbox")
                with self.assertRaises(SafetyRefusal) as opened:
                    store.dir("sandbox")
                self.assertEqual(str(checked.exception), str(opened.exception))
        self.assertEqual(sorted(p.name for p in (self.base / "real").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
