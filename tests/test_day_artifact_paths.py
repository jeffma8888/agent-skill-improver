"""A non-file where the day's receipt or checkpoint goes is refused before the run.

A committed ``daily`` ends by writing two files: the day's receipt and the
day's checkpoint. Measured before: a directory at either path left ``doctor``
at ``verdict: ready`` (a directory is not a foreign entry) and the run exited
70 from its final write, after every model call, leaving the day without a
receipt or a checkpoint; a fifo or a symlink at either path was replaced
without a word. Now the two paths are checked before the lock, before any
read and before the source is touched, and ``doctor`` names today's.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app.daily import run_daily
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.cli import RC_NEGATIVE, RC_OK, RC_REFUSED
from agent_skill_improver.clock import FrozenClock, day_key
from agent_skill_improver.config import load_config
from agent_skill_improver.errors import SafetyRefusal

from tests.support import InProcessRunner, build_env, run_cli

DAY = "2026-01-02"


def _clock() -> FrozenClock:
    return FrozenClock(datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc))


def _fingerprint(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_symlink():
            out[rel] = "link"
        elif path.is_dir():
            out[rel] = f"dir {oct(path.stat().st_mode & 0o777)}"
        elif path.is_file():
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            out[rel] = "other"
    return out


class _Untouched:
    """A source the refused run must never reach; any use is the failure."""

    def __getattr__(self, name: str):
        raise AssertionError(f"the source was touched ({name}) by a run that should have refused")


def _plant(path: Path, kind: str) -> None:
    if kind == "directory":
        # Owner-only, as the tool's own directories are, so the run's mode
        # repair has nothing to do and "nothing was written" can include modes.
        path.mkdir(mode=0o700)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        path.symlink_to(path.parent / "elsewhere.json")
    else:  # pragma: no cover - a typo in the table
        raise AssertionError(kind)


class TheDaysArtifactPathsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config = load_config(
            build_env(
                self.base,
                dry_run=False,
                rubric_ids=("instruction-compliance",),
                use_subprocess_model=False,
            )
        )
        self.state = StateStore(self.config.state_dir, clock=_clock())
        self.state.ensure()
        self.store = FilesystemSkillStore(
            self.config.skill_roots, read_only=True, protected_paths=self.config.protected_paths
        )

    def _run(self, *, source=None, commit=True):
        return run_daily(
            self.config,
            source=source or FixtureConversationSource(self.config.source.fixtures_dir),
            runner=InProcessRunner(),
            skill_store=self.store,
            state=self.state,
            clock=_clock(),
            day=DAY,
            commit=commit,
        )

    def _paths(self) -> dict[str, Path]:
        return {
            "receipt": self.state.receipt_path(DAY, dry_run=False),
            "checkpoint": self.state.checkpoint_path(DAY),
        }

    def test_a_non_file_at_either_path_is_refused_before_the_source_the_lock_and_any_write(
        self,
    ) -> None:
        for what, path in self._paths().items():
            for kind, spelled in (
                ("directory", "is a directory"),
                ("fifo", "is not a regular file"),
                ("symlink", "is a symlink"),
            ):
                with self.subTest(what=what, kind=kind):
                    _plant(path, kind)
                    try:
                        before = _fingerprint(self.config.state_dir)
                        with self.assertRaises(SafetyRefusal) as caught:
                            self._run(source=_Untouched())
                        message = str(caught.exception)
                        self.assertTrue(message.startswith("refusing to run: the day's "), message)
                        self.assertIn(f"{what} path {path} {spelled}", message)
                        self.assertIn("where this run would write", message)
                        self.assertIn("remove it if you did not put it there", message)
                        self.assertEqual(
                            _fingerprint(self.config.state_dir), before, "nothing written"
                        )
                        self.assertFalse(self.state.lock_path().exists(), "the lock never taken")
                    finally:
                        if path.is_dir() and not path.is_symlink():
                            path.rmdir()
                        else:
                            path.unlink()

    def test_a_state_directory_named_with_a_line_break_is_spelled_onto_one_line(self) -> None:
        # The entry's name comes from the day, which is validated; the only way
        # a line break reaches this message is the state directory's own name.
        state = StateStore(self.base / "state\nsecond line", clock=_clock())
        state.ensure()
        path = state.receipt_path(DAY, dry_run=False)
        path.mkdir(mode=0o700)
        with self.assertRaises(SafetyRefusal) as caught:
            run_daily(
                self.config,
                source=_Untouched(),
                runner=InProcessRunner(),
                skill_store=self.store,
                state=state,
                clock=_clock(),
                day=DAY,
                commit=True,
            )
        message = str(caught.exception)
        self.assertNotIn("\n", message)
        self.assertIn("second line", message, "folded, not cut")
        self.assertIn("is a directory", message)

    def test_a_dry_run_writes_neither_file_and_is_not_refused_over_them(self) -> None:
        for path in self._paths().values():
            path.mkdir(mode=0o700)
        receipt = self._run(commit=False)
        self.assertEqual(receipt.status, "ok", receipt.to_dict()["notes"])
        self.assertTrue(receipt.dry_run)
        for path in self._paths().values():
            self.assertTrue(path.is_dir(), f"{path} is still the planted directory")
        withheld = {Path(w).parent.name for w in self.state.withheld_writes}
        self.assertLessEqual({"receipts", "checkpoints"}, withheld, self.state.withheld_writes)

    def test_a_regular_file_at_both_paths_is_the_previous_run_and_is_rewritten(self) -> None:
        first = self._run()
        self.assertEqual(first.status, "ok", first.to_dict()["notes"])
        for path in self._paths().values():
            self.assertTrue(path.is_file(), path)
        second = self._run()
        self.assertEqual(second.day, DAY)
        self.assertEqual(self.state.read_checkpoint(DAY)["runs"], 2)

    def test_the_store_names_the_receipt_first_then_the_checkpoint(self) -> None:
        self.assertIsNone(self.state.day_artifact_problem(DAY))
        paths = self._paths()
        paths["checkpoint"].mkdir()
        self.assertEqual(
            self.state.day_artifact_problem(DAY),
            f"the day's checkpoint path {paths['checkpoint']} is a directory",
        )
        paths["receipt"].mkdir()
        self.assertEqual(
            self.state.day_artifact_problem(DAY),
            f"the day's receipt path {paths['receipt']} is a directory",
        )
        # The dry-run receipt path is never written and is not asked about.
        paths["receipt"].rmdir()
        paths["checkpoint"].rmdir()
        self.state.receipt_path(DAY, dry_run=True).mkdir()
        self.assertIsNone(self.state.day_artifact_problem(DAY))


class ThroughTheCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.cfg = ["--config", str(build_env(self.base, seed="paths", use_subprocess_model=True))]
        self.today = day_key(datetime.datetime.now(datetime.timezone.utc))

    def _problems(self, proc) -> list[str]:
        return [line for line in proc.stdout.splitlines() if line.startswith("PROBLEM")]

    def test_doctor_names_todays_receipt_path_and_the_committed_run_exits_three(self) -> None:
        ready = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(ready.returncode, RC_OK, ready.stderr)
        state = StateStore(self.base / "state", clock=_clock())
        receipt = state.receipt_path(self.today, dry_run=False)
        receipt.mkdir()

        doctor = run_cli([*self.cfg, "doctor"])
        self.assertEqual(doctor.returncode, RC_NEGATIVE, doctor.stdout)
        (problem,) = self._problems(doctor)
        self.assertIn(f"today's receipt path {receipt} is a directory", problem)
        self.assertIn("where today's committed run would write and so refuses to start", problem)
        self.assertIn("remove it if you did not put it there", problem)
        as_json = json.loads(run_cli([*self.cfg, "doctor", "--json"]).stdout)
        self.assertEqual(as_json["problems"], [problem.split(None, 1)[1]])

        # The checkpoint path is asked the same question, in the same words.
        checkpoint = state.checkpoint_path(self.today)
        os.mkfifo(checkpoint)
        both = self._problems(run_cli([*self.cfg, "doctor"]))
        self.assertEqual(len(both), 2, both)
        self.assertIn(f"today's checkpoint path {checkpoint} is not a regular file", both[0])
        self.assertIn(f"today's receipt path {receipt} is a directory", both[1])
        checkpoint.unlink()

        refused = run_cli([*self.cfg, "daily", "--commit"])
        self.assertEqual(refused.returncode, RC_REFUSED, refused.stderr)
        self.assertIn("SafetyRefusal: refusing to run: the day's receipt path", refused.stderr)
        self.assertIn("is a directory, where this run would write", refused.stderr)
        self.assertTrue(receipt.is_dir(), "the entry is left for the operator")

        # A dry run writes no receipt, so it is neither refused nor named.
        dry = run_cli([*self.cfg, "daily"])
        self.assertEqual(dry.returncode, RC_OK, dry.stderr)
        receipt.rmdir()
        state.receipt_path(self.today, dry_run=True).mkdir()
        self.assertEqual(run_cli([*self.cfg, "doctor"]).returncode, RC_OK)

    def test_doctor_says_nothing_about_todays_receipt_when_a_file_or_nothing_is_there(
        self,
    ) -> None:
        ready = run_cli([*self.cfg, "daily", "--commit"])
        self.assertEqual(ready.returncode, RC_OK, ready.stderr)
        doctor = run_cli([*self.cfg, "doctor"])
        self.assertEqual(doctor.returncode, RC_OK, doctor.stdout)
        self.assertEqual(self._problems(doctor), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
