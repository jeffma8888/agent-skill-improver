"""A cap that holds a skill or the queue says what releases it.

``daily`` refuses a skill that already has ``caps.max_open_candidates_per_skill``
open candidates, and stands down when ``caps.max_open_candidates`` are open, and
both said to "review or close" them. Measured before (3.11 and 3.13, a committed
run with one validated candidate, which ``review`` then showed): the run for the
next day refused the skill, or stood down, the same way, because a reviewed
candidate stays ``validated`` and still counts; there is no command that closes
one. Applying it released the skill, and deleting its record released the
queue. With ``max_open_candidates = 0`` every run stood down with nothing open,
saying the same. Both now say that a review leaves a candidate open, and that
applying it or deleting its record closes it (``RELEASE_ADVICE`` in
``app/daily.py``); a queue cap of 0 says only raising it resumes.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from agent_skill_improver import cli
from agent_skill_improver.app import daily

from tests.support import build_env

DAY, NEXT_DAY, THIRD_DAY = "2026-02-01", "2026-02-02", "2026-02-03"
OLD = "review or close"
RELEASE = (
    "a review leaves a candidate open, and one is closed by applying it (`asi apply`) or by "
    "deleting its record (its <id>.json in the state directory's candidates/; "
    "`asi candidates` lists the open ones)"
)


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


class _Held:
    """A committed run with one validated candidate, which ``review`` has shown."""

    def __init__(self, base: Path, **caps: int):
        self.config = build_env(base / "env", seed="c219", **caps)
        self.state = base / "env" / "state"
        self.first = run(self.config, "daily", "--commit", "--day", DAY)
        (self.record_file,) = sorted(self.state.glob("candidates/*.json"))
        self.cid = self.record_file.stem
        rc, shown = run(self.config, "review", self.cid)
        assert rc == 0, shown
        found = re.search(r"--approve (\S+)", shown)
        assert found, shown
        self.token = found.group(1)

    def day(self, day: str) -> tuple[int, str, dict[str, str]]:
        rc, out = run(self.config, "daily", "--commit", "--day", day)
        records = {
            p.stem: json.loads(p.read_text(encoding="utf-8"))["status"]
            for p in sorted(self.state.glob("candidates/*.json"))
        }
        return rc, out, records

    def stand_down_reason(self, day: str) -> str:
        (receipt,) = self.state.glob(f"receipts/{day}--*.json")
        return str(json.loads(receipt.read_text(encoding="utf-8"))["stand_down_reason"])

    def refused_reason(self, skill_id: str) -> str:
        rc, listed = run(self.config, "set-aside", "--json")
        assert rc == 0, listed
        (entry,) = [
            r
            for r in json.loads(listed)["records"]
            if r["kind"] == "refused" and r["record_id"] == skill_id
        ]
        return str(entry["reason"])


class TheWordsTests(unittest.TestCase):
    """The phrase and the cap clause, reached through ``app/daily.py``."""

    def test_the_release_advice_names_apply_and_the_record_to_delete(self) -> None:
        self.assertEqual(daily.RELEASE_ADVICE, RELEASE)

    def test_a_cap_of_one_waits_for_none_and_a_wider_cap_for_fewer(self) -> None:
        self.assertEqual(daily._once_fewer_than(1), "once none is open")
        self.assertEqual(daily._once_fewer_than(2), "once fewer than 2 are open")
        self.assertEqual(daily._once_fewer_than(12), "once fewer than 12 are open")


class APerSkillCapTests(unittest.TestCase):
    """``max_open_candidates_per_skill`` 1: review, another day, then apply and a third day."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.case = _Held(Path(cls._tmp.name))
        cls.after_review = cls.case.day(NEXT_DAY)
        cls.held_reason = cls.case.refused_reason("tdd")
        cls.applied = run(
            cls.case.config, "apply", cls.case.cid, "--approve", cls.case.token, "--i-reviewed-this"
        )
        cls.after_apply = cls.case.day(THIRD_DAY)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_a_reviewed_candidate_still_holds_its_skill_and_the_refusal_says_so(self) -> None:
        rc, said, records = self.after_review
        self.assertEqual(rc, 0, said)
        self.assertEqual(records, {self.case.cid: "validated"})
        self.assertEqual(
            self.held_reason,
            "1 open candidate(s) already target 'tdd' (cap 1 per skill). The next run proposes "
            f"for it again once none is open: {RELEASE}",
        )
        self.assertNotIn(OLD, said)

    def test_applying_the_candidate_releases_its_skill(self) -> None:
        self.assertEqual(self.applied[0], 0, self.applied[1])
        rc, said, records = self.after_apply
        self.assertEqual(rc, 0, said)
        self.assertEqual(records[self.case.cid], "applied")
        self.assertIn(f"CAND-{THIRD_DAY}-01-tdd", records)
        self.assertNotIn("open candidate(s) already target", said)


class AQueueCapTests(unittest.TestCase):
    """``max_open_candidates`` 1: review, another day, then delete the record and a third day."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.case = _Held(Path(cls._tmp.name), max_open_candidates=1)
        cls.after_review = cls.case.day(NEXT_DAY)
        cls.held_reason = cls.case.stand_down_reason(NEXT_DAY)
        cls.case.record_file.unlink()
        cls.after_delete = cls.case.day(THIRD_DAY)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_a_reviewed_candidate_still_fills_the_queue_and_the_stand_down_says_so(self) -> None:
        rc, said, records = self.after_review
        self.assertEqual(rc, 0, said)
        self.assertIn("status=stood-down", said)
        self.assertEqual(records, {self.case.cid: "validated"})
        self.assertEqual(
            self.held_reason,
            "backlog guard: 1 candidate(s) already awaiting review, cap is 1. Nothing was "
            f"discarded. The next run resumes once none is open: {RELEASE}.",
        )
        self.assertNotIn(OLD, said)

    def test_deleting_the_record_releases_the_queue(self) -> None:
        rc, said, records = self.after_delete
        self.assertEqual(rc, 0, said)
        self.assertNotIn("status=stood-down", said)
        self.assertEqual(list(records), [f"CAND-{THIRD_DAY}-01-tdd"])


class AQueueCapOfZeroTests(unittest.TestCase):
    """``max_open_candidates`` 0 holds with nothing open: only raising it resumes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        base = Path(cls._tmp.name)
        cls.config = build_env(base / "env", seed="c219", max_open_candidates=0)
        cls.state = base / "env" / "state"
        cls.ran = run(cls.config, "daily", "--commit", "--day", DAY)
        (receipt,) = cls.state.glob(f"receipts/{DAY}--*.json")
        cls.reason = json.loads(receipt.read_text(encoding="utf-8"))["stand_down_reason"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_a_queue_cap_of_zero_says_only_raising_it_resumes(self) -> None:
        rc, said = self.ran
        self.assertEqual(rc, 0, said)
        self.assertIn("status=stood-down", said)
        self.assertEqual(
            self.reason,
            "backlog guard: 0 candidate(s) already awaiting review, cap is 0. Nothing was "
            "discarded. No run proposes while [caps].max_open_candidates is 0; raise it to "
            "resume.",
        )
        self.assertEqual(list(self.state.glob("candidates/*.json")), [])


if __name__ == "__main__":
    unittest.main()
