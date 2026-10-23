"""``doctor`` is not ready when today's run could not name a candidate.

``daily`` forms every known skill's longest candidate id before it reads a
conversation, and refuses the whole run with exit 3 when one would not fit:
a skill directory name too long for the id beside a date-sized day, or a
``[caps].max_candidates_per_run`` with more digits than the id has room for.
``doctor`` never asked. Measured before (3.11 and 3.13): with a 186-character
skill directory name, or a 170-digit cap, ``doctor`` exited 0 and said ready
while ``daily`` exited 3; at 185 characters and 169 digits both ran. ``doctor``
now asks the run's own question for today's day.
"""

from __future__ import annotations

import json
import re
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.clock import SystemClock, day_key
from agent_skill_improver.errors import ConfigError
from tests.support import build_env, run_cli

PREFIX = "PROBLEM  today's run would be refused before any conversation is read: "
NOT_DONE = " No conversation was read or scored."
A_DATE = re.compile(r"'\d{4}-\d{2}-\d{2}'")


def today() -> str:
    return day_key(SystemClock().now())


class DoctorAsksWhetherTodaysRunCanNameACandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base / "env", dry_run=False, seed="naming"))
        self.canonical = self.base / "env" / "skills-canonical"

    def add_skill(self, source: str, length: int) -> str:
        """A copy of a canonical skill under a directory name this long."""
        name = "s" * length
        shutil.copytree(self.canonical / source, self.canonical / name)
        document = self.canonical / name / "SKILL.md"
        text = document.read_text(encoding="utf-8")
        self.assertIn(f"name: {source}", text)
        document.write_text(text.replace(f"name: {source}", f"name: {name}", 1), "utf-8")
        return name

    def set_cap(self, digits: int) -> None:
        text = self.config_path.read_text(encoding="utf-8")
        new, count = re.subn(
            r"^max_candidates_per_run = .*$",
            f"max_candidates_per_run = {'9' * digits}",
            text,
            count=1,
            flags=re.M,
        )
        self.assertEqual(count, 1)
        self.config_path.write_text(new, encoding="utf-8")

    def doctor(self, *extra: str):
        return run_cli(["--config", str(self.config_path), "doctor", *extra])

    def naming_problems(self, proc) -> list[str]:
        return [ln[len(PREFIX) :] for ln in proc.stdout.splitlines() if ln.startswith(PREFIX)]

    def assert_ready(self) -> None:
        proc = self.doctor()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.naming_problems(proc), [])
        self.assertIn("verdict: ready", proc.stdout.splitlines())

    def assert_doctor_and_daily_agree(self) -> str:
        """``doctor`` exits 1 with one naming problem, said once and as a
        problem only, and today's run refuses with exit 3 for the same reason,
        adding only what it did not do."""
        days = {today()}
        proc = self.doctor()
        days.add(today())
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        found = self.naming_problems(proc)
        self.assertEqual(len(found), 1, proc.stdout)
        reason = found[0]
        self.assertEqual(proc.stdout.count(reason), 1, proc.stdout)
        self.assertIn("verdict: problems found", proc.stdout.splitlines())
        self.assertTrue(any(f"'{day}'" in reason for day in days), (reason, days))
        run = run_cli(["--config", str(self.config_path), "daily"])
        self.assertEqual(run.returncode, 3, run.stdout + run.stderr)
        refusals = [ln for ln in run.stderr.splitlines() if ln.startswith("ConfigError: ")]
        self.assertEqual(len(refusals), 1, run.stderr)
        # Each names the day it asked about; a UTC midnight between the two
        # commands changes that day and nothing else.
        self.assertEqual(
            A_DATE.sub("'DAY'", refusals[0]),
            A_DATE.sub("'DAY'", f"ConfigError: {reason}{NOT_DONE}"),
        )
        return reason

    def assert_a_name_too_long_is_a_problem(self, source: str) -> None:
        name = self.add_skill(source, 186)
        reason = self.assert_doctor_and_daily_agree()
        self.assertIn(f"cannot name a candidate for skill {name!r} (186 characters)", reason)
        self.assertIn("Use a shorter --day or a shorter skill directory name.", reason)

    def test_a_skill_directory_name_too_long_to_name_a_candidate_is_a_problem(self):
        self.assert_a_name_too_long_is_a_problem("tdd")

    def test_a_skill_that_does_not_load_is_asked_about_too(self):
        # The run asks about every known skill, one whose document does not
        # load included, so the report does too.
        self.assert_a_name_too_long_is_a_problem("broken-frontmatter")

    def test_the_control_a_name_one_character_shorter_leaves_doctor_ready(self):
        self.add_skill("tdd", 185)
        self.assert_ready()

    def test_a_candidate_cap_too_wide_to_number_an_id_is_a_problem(self):
        self.set_cap(170)
        reason = self.assert_doctor_and_daily_agree()
        self.assertTrue(
            reason.startswith("[caps].max_candidates_per_run has 170 digits, too many"), reason
        )
        skill = re.search(r"for skill '([^']+)'", reason).group(1)
        fits = 204 - len(f"CAND-{today()}--{skill}")
        self.assertEqual(fits, 169, skill)
        self.assertTrue(reason.endswith(f"Lower the cap to at most {fits} digits."), reason)
        self.assertNotIn("shorter --day", reason)

    def test_the_control_a_cap_one_digit_narrower_leaves_doctor_ready(self):
        self.set_cap(169)
        self.assert_ready()

    def test_the_json_report_lists_the_same_problem(self):
        self.set_cap(170)
        proc = self.doctor("--json")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)
        named = [p for p in report["problems"] if p.startswith(PREFIX[len("PROBLEM  ") :])]
        self.assertEqual(len(named), 1, report["problems"])
        self.assertIn("[caps].max_candidates_per_run has 170 digits", named[0])
        self.assertFalse(any("too many to number" in w for w in report["warnings"]), report)

    def test_the_run_refusal_is_the_reason_plus_what_was_not_done(self):
        # Imported here so the CLI tests above still run, and fail, without it.
        from agent_skill_improver.app.daily import (
            _assert_every_skill_can_be_named,
            candidate_naming_problem,
        )

        day = "2026-03-01"
        # 204 characters at a two-digit sequence: CAND-, the day, 99, the name.
        self.assertIsNone(candidate_naming_problem(day, ["tdd", "s" * 185], 3))
        # A 20-character name leaves room for exactly this many sequence digits.
        fits = 204 - len(f"CAND-{day}--{'x' * 20}")
        self.assertIsNone(candidate_naming_problem(day, ["x" * 20], 10 ** (fits - 1)))
        for case, skills, cap in (
            ("a name too long", ["tdd", "s" * 186], 3),
            ("a cap one digit too wide", ["tdd", "x" * 20], 10**fits),
        ):
            with self.subTest(case=case):
                reason = candidate_naming_problem(day, skills, cap)
                self.assertIsNotNone(reason)
                with self.assertRaises(ConfigError) as ctx:
                    _assert_every_skill_can_be_named(day, skills, cap)
                self.assertEqual(str(ctx.exception), f"{reason}{NOT_DONE}")
        self.assertTrue(
            candidate_naming_problem(day, ["x" * 20], 10**fits).endswith(
                f"Lower the cap to at most {fits} digits."
            )
        )
        self.assertIsNone(_assert_every_skill_can_be_named(day, ["tdd"], 3))


if __name__ == "__main__":
    unittest.main()
