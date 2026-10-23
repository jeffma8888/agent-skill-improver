"""A rubric version below 1 is refused when the rubric loads, by its file.

A candidate records the version of each rubric it was validated under, and the
record reader takes only a version of at least 1; the rubric loader took any
integer. Measured before (3.11 and 3.13): with ``version = 0`` or ``-5``,
``doctor`` said ready, ``daily --commit`` exited 0 and wrote a validated
candidate, ``candidates`` listed it with 0 unreadable records, and ``review``
exited 3 with ``CandidateTampered`` ("the record has changed since it was
written"), which it had not.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.config import load_config
from agent_skill_improver.errors import ConfigError
from agent_skill_improver.jsonio import MAX_STORED_INT

from tests.support import build_env, run_cli

DAY = "2026-01-02"


class ARubricVersionBelowOneTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base, dry_run=False, seed="rubric-version"))
        self.rubric = sorted((self.base / "rubrics").glob("*.toml"))[0]
        self.original = self.rubric.read_text(encoding="utf-8")
        self.assertEqual(self.original.count("version = 1\n"), 1)
        self.cfg = ["--config", str(self.config_path)]

    def _version(self, value: str) -> None:
        self.rubric.write_text(
            self.original.replace("version = 1\n", f"version = {value}\n"), encoding="utf-8"
        )

    def test_a_rubric_version_below_one_is_refused_at_load_by_its_file(self):
        for value in ("0", "-5", str(-MAX_STORED_INT)):
            with self.subTest(value=value[:8]):
                self._version(value)
                with self.assertRaises(ConfigError) as ctx:
                    load_config(self.config_path)
                message = str(ctx.exception)
                self.assertIn(self.rubric.name, message)
                self.assertIn("[rubric].version must be a whole number of at least 1", message)
                self.assertLess(len(message), 600)

    def test_doctor_and_daily_refuse_a_version_below_one_before_anything_is_written(self):
        self._version("0")
        for proc in (
            run_cli([*self.cfg, "doctor"]),
            run_cli([*self.cfg, "daily", "--commit", "--day", DAY]),
        ):
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertIn("[rubric].version must be a whole number of at least 1", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
        state = self.base / "state"
        self.assertEqual(sorted(p for p in state.rglob("*") if not p.is_dir()), [])

    def test_the_control_a_version_of_one_or_more_is_reviewed_as_written(self):
        for value in ("1", "2"):
            with self.subTest(value=value), TemporaryDirectory() as tmp:
                # A fresh tree each: one open proposal per skill.
                base = Path(tmp).resolve()
                cfg = ["--config", str(build_env(base, dry_run=False, seed="rubric-version"))]
                rubric = sorted((base / "rubrics").glob("*.toml"))[0]
                rubric.write_text(
                    self.original.replace("version = 1\n", f"version = {value}\n"),
                    encoding="utf-8",
                )
                daily = run_cli([*cfg, "daily", "--commit", "--day", DAY])
                self.assertEqual(daily.returncode, 0, daily.stderr)
                listed = run_cli([*cfg, "candidates"])
                ids = re.findall(rf"CAND-{DAY}-[A-Za-z0-9._-]+", listed.stdout)
                self.assertTrue(ids, listed.stdout)
                review = run_cli([*cfg, "review", ids[0]])
                self.assertEqual(review.returncode, 0, review.stderr)
                self.assertNotIn("CandidateTampered", review.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
