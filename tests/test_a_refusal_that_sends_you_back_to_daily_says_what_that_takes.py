"""A refusal that sends you back to ``daily`` says what proposing again takes.

``review`` and ``apply`` refuse a candidate whose record was edited after it was
written (``CandidateTampered``) or whose skill file changed, moved or cannot be
named (``BaseHashMismatch``), and each refusal told the operator to re-run
``asi daily --commit``. Measured before (3.11 and 3.13, a committed run, then
the advice followed): a run for the same day stood down, "every discovered
conversation was already processed for this day", and proposed nothing; and a
run for another day, the refused record left in place as the base-hash advice
implied, refused the skill because the record still counted against its
open-candidate cap ("review or close them"), while ``review`` refused the
record again with the same advice. Only deleting the record and running another
day proposed the change again. ``repropose_advice`` in ``domain/integrity.py``
now says that, and every such refusal uses it.

``daily`` gives the same ``CandidateTampered`` check, at validate, to a
candidate whose record it has not yet written (the skill file changed between
the read that proposed it and the read that validates it), and that refusal
said to delete a candidate that was never written. It now says no record exists
(``set_aside_advice``), and to run another day.

That advice is for an open record. ``review`` read the candidate's file before
it asked its status, so an ``applied`` or ``rolled-back`` candidate whose skill
folder had moved was refused as a base mismatch with it, telling the operator
to delete the record ``rollback`` restores from. ``review`` now asks the status
first, as ``apply`` does, and names such a candidate closed.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent_skill_improver import cli
from agent_skill_improver.app import daily
from agent_skill_improver.app.promote import review_report
from agent_skill_improver.app.state import CLOSED_STATUSES
from agent_skill_improver.domain import integrity
from agent_skill_improver.domain.candidate import STATUS_PROPOSED
from agent_skill_improver.errors import CandidateTampered, GateFailed

from tests.support import build_env
from tests.test_candidate_integrity import BASE_TEXT, build_candidate

DAY, NEXT_DAY, THIRD_DAY = "2026-02-01", "2026-02-02", "2026-02-03"
OLD_ADVICE = "re-run `asi daily --commit` to"
SAME_DAY = "every discovered conversation was already processed for this day"
HELD = "open candidate(s) already target"
NO_RECORD = (
    "No record of it was written, since `daily` writes one only after validating it, so "
    "there is nothing to delete or to review. This run sets its skill aside; to propose "
    "again, run `asi daily --commit` for another day, because a run for the same day skips "
    "every conversation that day already scored, the ones behind this candidate included"
)


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


def advice_for(candidate_id: str) -> str:
    return (
        f"delete its record ({candidate_id}.json in the state directory's candidates/), "
        "which counts against its skill's open-candidate cap while it exists, then run "
        "`asi daily --commit` for another day"
    )


class _Proposed:
    """One committed daily run with one validated candidate, and its approval token."""

    def __init__(self, base: Path):
        self.config = build_env(base / "env", seed="c218")
        self.state = base / "env" / "state"
        self.daily_rc, self.daily = run(self.config, "daily", "--commit", "--day", DAY)
        (self.record_file,) = sorted(self.state.glob("candidates/*.json"))
        self.record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self.cid = self.record["id"]
        rc, shown = run(self.config, "review", self.cid)
        assert rc == 0, shown
        found = re.search(r"--approve (\S+)", shown)
        assert found, shown
        self.token = found.group(1)

    def refusals(self) -> dict[str, tuple[int, str]]:
        return {
            "review": run(self.config, "review", self.cid),
            "apply": run(
                self.config, "apply", self.cid, "--approve", self.token, "--i-reviewed-this"
            ),
        }

    def daily_on(self, day: str) -> tuple[int, str, list[str]]:
        rc, out = run(self.config, "daily", "--commit", "--day", day)
        status = next((line for line in out.splitlines() if line.startswith("status=")), "")
        return rc, out + status, sorted(p.stem for p in self.state.glob("candidates/*.json"))


class TheAdviceTests(unittest.TestCase):
    """The phrase itself, reached through its module so on a tree without it only these error."""

    def test_the_advice_names_the_record_to_delete_and_another_day(self) -> None:
        said = integrity.repropose_advice("CAND-2026-02-01-01-tdd")
        self.assertTrue(said.startswith(advice_for("CAND-2026-02-01-01-tdd")), said)

    def test_the_advice_says_why_the_same_day_does_not_do(self) -> None:
        said = integrity.repropose_advice("CAND-2026-02-01-01-tdd")
        self.assertIn(
            "a run for the same day skips every conversation that day already scored, the "
            "ones behind this candidate included",
            said,
        )
        self.assertNotIn("\n", said)

    def test_both_tamper_refusals_give_it(self) -> None:
        tampered = replace(build_candidate(), approval_challenge="f" * 24)
        for name, check in (
            ("verify_candidate_integrity", integrity.verify_candidate_integrity),
            ("verify_full_integrity", integrity.verify_full_integrity),
        ):
            with self.subTest(name):
                with self.assertRaises(CandidateTampered) as caught:
                    check(tampered, base_text=BASE_TEXT, action="review")
                said = str(caught.exception)
                self.assertIn(
                    "To propose and validate the change again, " + advice_for(tampered.id), said
                )
                self.assertNotIn(OLD_ADVICE, said)
                self.assertIn("nothing has been written to any skill file", said)

    def test_a_refusal_at_validate_names_no_record_to_delete(self) -> None:
        tampered = replace(build_candidate(), approval_challenge="f" * 24)
        with self.assertRaises(CandidateTampered) as caught:
            integrity.verify_candidate_integrity(tampered, base_text=BASE_TEXT, action="validate")
        said = str(caught.exception)
        self.assertIn(f"refusing to validate candidate {tampered.id}", said)
        self.assertIn(NO_RECORD, said)
        for record_words in ("delete its record", "Delete this candidate", "since it was written"):
            self.assertNotIn(record_words, said)
        self.assertIn("Nothing has been written to any skill file", said)


class AnEditedRecordTests(unittest.TestCase):
    """``CandidateTampered``: the record edited after it was written, then the advice followed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.case = _Proposed(Path(cls._tmp.name))
        record = dict(cls.case.record)
        record["patch"] = dict(record["patch"], added_lines=record["patch"]["added_lines"] + 1)
        cls.case.record_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
        cls.refused = cls.case.refusals()
        cls.case.record_file.unlink()
        cls.same_day = cls.case.daily_on(DAY)
        cls.next_day = cls.case.daily_on(NEXT_DAY)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_review_and_apply_of_an_edited_record_give_the_advice(self) -> None:
        for action, (rc, said) in self.refused.items():
            with self.subTest(action):
                self.assertEqual(rc, 3, said)
                self.assertIn(f"CandidateTampered: refusing to {action} candidate", said)
                self.assertIn(advice_for(self.case.cid), said)
                self.assertNotIn(OLD_ADVICE, said)

    def test_a_run_for_the_same_day_proposes_nothing(self) -> None:
        rc, said, records = self.same_day
        self.assertEqual(rc, 0, said)
        self.assertIn("status=stood-down", said)
        self.assertIn(SAME_DAY, said)
        self.assertEqual(records, [])

    def test_another_day_after_deleting_the_edited_record_proposes_again(self) -> None:
        rc, said, records = self.next_day
        self.assertEqual(rc, 0, said)
        self.assertEqual(records, [f"CAND-{NEXT_DAY}-01-tdd"])


class AnEditedSkillTests(unittest.TestCase):
    """``BaseHashMismatch``: the skill edited after the proposal, then the advice followed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.case = _Proposed(Path(cls._tmp.name))
        skill = Path(cls.case.record["target"]["canonical_path"])
        skill.write_text(
            skill.read_text(encoding="utf-8") + "\nEdited by hand after the proposal.\n",
            encoding="utf-8",
        )
        cls.refused = cls.case.refusals()
        cls.kept = cls.case.daily_on(NEXT_DAY)
        cls.case.record_file.unlink()
        cls.deleted = cls.case.daily_on(THIRD_DAY)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_review_and_apply_of_an_edited_skill_give_the_advice(self) -> None:
        for action, (rc, said) in self.refused.items():
            with self.subTest(action):
                self.assertEqual(rc, 3, said)
                self.assertIn(f"BaseHashMismatch: refusing to {action} {self.case.cid}", said)
                self.assertIn(
                    "To propose and validate against the new base, " + advice_for(self.case.cid),
                    said,
                )
                self.assertNotIn(OLD_ADVICE, said)

    def test_a_run_for_another_day_with_the_record_kept_refuses_the_skill(self) -> None:
        rc, said, records = self.kept
        self.assertEqual(rc, 0, said)
        self.assertIn(HELD, said)
        self.assertEqual(records, [self.case.cid])

    def test_another_day_after_deleting_the_stale_record_proposes_again(self) -> None:
        rc, said, records = self.deleted
        self.assertEqual(rc, 0, said)
        self.assertEqual(records, [f"CAND-{THIRD_DAY}-01-tdd"])


class AMovedSkillTests(unittest.TestCase):
    """``BaseHashMismatch`` for a skill file that is no longer where the record says."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.case = _Proposed(Path(cls._tmp.name))
        folder = Path(cls.case.record["target"]["canonical_path"]).parent
        folder.rename(folder.with_name(folder.name + "-moved"))
        cls.refused = cls.case.refusals()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_review_and_apply_of_a_moved_skill_give_the_advice(self) -> None:
        for action, (rc, said) in self.refused.items():
            with self.subTest(action):
                self.assertEqual(rc, 3, said)
                self.assertIn("could not be read", said)
                self.assertIn(
                    "To propose against the configured tree, " + advice_for(self.case.cid), said
                )
                self.assertNotIn(OLD_ADVICE, said)


class ARefusalAtValidateTests(unittest.TestCase):
    """``CandidateTampered`` at validate: the skill changed between propose's read and daily's."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        base = Path(cls._tmp.name)
        cls.config = build_env(base / "env", seed="c218")
        cls.state = base / "env" / "state"
        read = daily.read_canonical_text
        daily.read_canonical_text = lambda *a, **k: read(*a, **k) + "\nRaced edit.\n"
        try:
            cls.raced = run(cls.config, "daily", "--commit", "--day", DAY)
        finally:
            daily.read_canonical_text = read
        cls.raced_records = sorted(p.stem for p in cls.state.glob("candidates/*.json"))
        rc, listed = run(cls.config, "set-aside", "--json")
        assert rc == 0, listed
        cls.set_aside = json.loads(listed)["records"]
        cls.same_day = run(cls.config, "daily", "--commit", "--day", DAY)
        cls.next_day = run(cls.config, "daily", "--commit", "--day", NEXT_DAY)
        cls.next_records = sorted(p.stem for p in cls.state.glob("candidates/*.json"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_a_refusal_at_validate_says_no_record_was_written(self) -> None:
        rc, said = self.raced
        self.assertEqual(rc, 0, said)
        self.assertIn("skills_deferred=1 candidates=0", said)
        self.assertEqual(self.raced_records, [])
        (entry,) = self.set_aside
        self.assertEqual(
            (entry["kind"], entry["record_id"], entry["stage"]), ("deferred", "tdd", "validate")
        )
        self.assertIn(f"refusing to validate candidate CAND-{DAY}-01-tdd", entry["reason"])
        self.assertIn(NO_RECORD, entry["reason"])
        self.assertNotIn("delete its record", entry["reason"])
        self.assertNotIn("Delete this candidate", entry["reason"])

    def test_a_run_for_the_same_day_after_a_refusal_at_validate_proposes_nothing(self) -> None:
        rc, said = self.same_day
        self.assertEqual(rc, 0, said)
        self.assertIn("status=stood-down", said)
        self.assertIn(SAME_DAY, said)

    def test_another_day_after_a_refusal_at_validate_proposes_again(self) -> None:
        rc, said = self.next_day
        self.assertEqual(rc, 0, said)
        self.assertEqual(self.next_records, [f"CAND-{NEXT_DAY}-01-tdd"])


class AClosedCandidateTests(unittest.TestCase):
    """``review`` of a closed candidate whose skill folder moved: named closed, never told to delete."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.case = _Proposed(Path(cls._tmp.name))
        cls.applied = run(
            cls.case.config, "apply", cls.case.cid, "--approve", cls.case.token, "--i-reviewed-this"
        )
        folder = Path(cls.case.record["target"]["canonical_path"]).parent
        moved = folder.with_name(folder.name + "-moved")
        folder.rename(moved)
        cls.applied_review = run(cls.case.config, "review", cls.case.cid)
        moved.rename(folder)
        cls.rolled = run(cls.case.config, "rollback", cls.case.cid)
        cls.rolled_status = json.loads(cls.case.record_file.read_text(encoding="utf-8"))["status"]
        folder.rename(moved)
        cls.rolled_review = run(cls.case.config, "review", cls.case.cid)
        moved.rename(folder)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _closed(self, refused: tuple[int, str], status: str) -> None:
        rc, said = refused
        self.assertEqual(rc, 3, said)
        self.assertIn(
            f"GateFailed: refusing to review candidate {self.case.cid}: status is '{status}'", said
        )
        self.assertNotIn("BaseHashMismatch", said)
        self.assertNotIn("delete its record", said)
        self.assertNotIn(OLD_ADVICE, said)

    def test_review_of_an_applied_candidate_whose_skill_moved_is_refused_as_closed(self) -> None:
        self._closed(self.applied_review, "applied")

    def test_review_of_a_rolled_back_candidate_whose_skill_moved_is_refused_as_closed(
        self,
    ) -> None:
        self._closed(self.rolled_review, "rolled-back")

    def test_review_report_still_refuses_every_status_but_validated(self) -> None:
        for status in (STATUS_PROPOSED, *CLOSED_STATUSES):
            with self.subTest(status):
                with self.assertRaises(GateFailed) as caught:
                    review_report(build_candidate(status=status), base_text=BASE_TEXT)
                self.assertIn(f"status is '{status}', not 'validated'", str(caught.exception))

    def test_the_applied_record_kept_still_rolls_back(self) -> None:
        self.assertEqual(self.applied[0], 0, self.applied[1])
        self.assertEqual(self.rolled[0], 0, self.rolled[1])
        self.assertEqual(self.rolled_status, "rolled-back")


class TheRunsBeforeTheRefusalTests(unittest.TestCase):
    """Controls: the run each case starts from is healthy and reviewable."""

    def test_the_first_run_proposes_one_validated_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = _Proposed(Path(tmp))
            self.assertEqual(first.daily_rc, 0, first.daily)
            self.assertEqual(first.record["status"], "validated")
            self.assertEqual(first.cid, f"CAND-{DAY}-01-tdd")
            self.assertTrue(first.token)


if __name__ == "__main__":
    unittest.main()
