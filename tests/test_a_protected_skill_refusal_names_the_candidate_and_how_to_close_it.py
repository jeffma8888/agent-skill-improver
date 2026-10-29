"""A skill listed as protected is refused with the candidate and how to close it.

``review`` and ``apply`` read ``[guard].protected_skill_ids`` again, so a
candidate validated before its skill was listed is refused by the target check
(``assert_write_target`` in ``adapters/skill_fs.py``). Measured before (3.11 and
3.13, a committed run with one validated candidate, then its skill listed):
``review`` and ``apply`` each exited 3 with ``SelfMutationRefused`` "refusing to
review <path>" and "refusing to write <path>", which named no candidate and said
"nothing was touched", while ``apply`` had written the run lock's owner line.
The record stayed ``validated``, so it stayed open: with
``[caps].max_open_candidates`` at 1 the next day stood down on it, and with the
record deleted the next day ran. With the skill taken off the list, ``review``
and ``apply`` with the same token took it. ``listed_as_protected`` now words
the refusal with the candidate, for the command that asked, and says how it is
closed; without a candidate the words are as before.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import signal
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters import skill_fs

from tests.support import build_env

DAY, NEXT_DAY, THIRD_DAY = "2026-02-01", "2026-02-02", "2026-02-03"
REFUSED = 3
NO_GUARD = "protected_skill_ids = []"


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


def how_to_close(cid: str) -> str:
    return (
        "The skill and the candidate are unchanged. The candidate stays open, and counts "
        "against [caps].max_open_candidates, until it is closed: to close it, delete its "
        f"record ({cid}.json in the state directory's candidates/). If the skill is not "
        "meant to stay listed, take it off the list, and `review` and `apply` take the "
        "candidate as before."
    )


class _Listed:
    """One validated candidate, its token taken, then its skill listed as protected."""

    def __init__(self, base: Path, *, listed: bool = True, **env: int) -> None:
        self.config = build_env(base / "env", seed="c222", **env)
        self.state = base / "env" / "state"
        rc, said = run(self.config, "daily", "--commit", "--day", DAY)
        assert rc == 0, said
        (self.record_file,) = sorted(self.state.glob("candidates/*.json"))
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self.cid = record["id"]
        self.skill_id = record["target"]["skill_id"]
        self.skill = Path(record["target"]["canonical_path"])
        rc, shown = run(self.config, "review", self.cid)
        assert rc == 0, shown
        found = re.search(r"--approve (\S+)", shown)
        assert found, shown
        self.token = found.group(1)
        self.unlisted = self.config.read_text(encoding="utf-8")
        assert self.unlisted.count(NO_GUARD) == 1
        if listed:
            self.list_it()

    def list_it(self) -> None:
        self.config.write_text(
            self.unlisted.replace(NO_GUARD, f'protected_skill_ids = ["{self.skill_id}"]'),
            encoding="utf-8",
        )

    def review(self) -> tuple[int, str]:
        return run(self.config, "review", self.cid)

    def apply(self) -> tuple[int, str]:
        return run(self.config, "apply", self.cid, "--approve", self.token, "--i-reviewed-this")

    def unlist(self) -> None:
        self.config.write_text(self.unlisted, encoding="utf-8")

    def left_as_it_is(self) -> tuple[bytes, bytes, list[Path]]:
        """The skill's bytes, the record's bytes, and the backups made."""
        backups = sorted(self.state.glob("backups/*"))
        return self.skill.read_bytes(), self.record_file.read_bytes(), backups


class TheWordsTests(unittest.TestCase):
    """The refusal itself, reached through its module so on a tree without it only this errors."""

    def test_the_words_name_the_candidate_for_the_command_that_asked(self) -> None:
        cid = "CAND-2026-02-01-01-tdd"
        path = Path("/skills/tdd/SKILL.md")
        for verb, command in (("review", "review"), ("write", "apply")):
            with self.subTest(verb):
                self.assertEqual(
                    skill_fs.listed_as_protected(path, "tdd", cid, verb),
                    f"refusing to {command} {cid}: its skill 'tdd' at {path} is listed in "
                    "[guard].protected_skill_ids, so `apply` does not write it while it is "
                    "listed. "
                    f"{how_to_close(cid)}",
                )

    def test_without_a_candidate_the_words_are_as_before(self) -> None:
        path = Path("/skills/tdd/SKILL.md")
        self.assertEqual(
            skill_fs.listed_as_protected(path, "tdd", "", "write"),
            f"refusing to write {path}: skill 'tdd' is listed in [guard].protected_skill_ids, "
            "so it is never a proposal target and never applied. A candidate validated before "
            "the skill was listed, or a record edited to name it, is refused here; nothing was "
            "touched.",
        )


class ReviewAndApplyOfAListedSkillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.case = _Listed(Path(cls._tmp.name))
        cls.left = cls.case.left_as_it_is()
        cls.refused = {"review": cls.case.review(), "apply": cls.case.apply()}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_review_and_apply_name_the_candidate_and_how_to_close_it(self) -> None:
        for action, (rc, said) in self.refused.items():
            with self.subTest(action):
                self.assertEqual(rc, REFUSED, said)
                self.assertTrue(
                    said.startswith(
                        f"SelfMutationRefused: refusing to {action} {self.case.cid}: its skill "
                        f"'{self.case.skill_id}' at {self.case.skill} is listed in "
                        "[guard].protected_skill_ids"
                    ),
                    said,
                )
                self.assertIn(how_to_close(self.case.cid), said)
                self.assertNotIn("nothing was touched", said)
                self.assertEqual(said.count("\n"), 1, said)

    def test_the_skill_the_record_and_the_backups_are_as_the_refusal_says(self) -> None:
        self.assertEqual(self.case.left_as_it_is(), self.left)
        self.assertEqual(self.left[2], [])
        record = json.loads(self.case.record_file.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "validated")


class WhatTheRefusalSaysToDoTests(unittest.TestCase):
    def test_kept_the_open_candidate_holds_the_queue_and_deleted_it_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _Listed(Path(tmp), max_open_candidates=1)
            rc, said = run(case.config, "daily", "--commit", "--day", NEXT_DAY)
            self.assertEqual(rc, 0, said)
            self.assertIn("status=stood-down", said)
            self.assertIn("1 candidate(s) already awaiting review, cap is 1", said)
            case.record_file.unlink()
            rc, said = run(case.config, "daily", "--commit", "--day", THIRD_DAY)
            self.assertEqual(rc, 0, said)
            self.assertIn("status=ok", said)
            self.assertNotIn("awaiting review", said)
            self.assertIn(f"skill '{case.skill_id}' is listed in [guard].protected_skill_ids", said)

    def test_the_skill_taken_off_the_list_reviews_and_applies_with_the_same_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _Listed(Path(tmp))
            self.assertEqual(case.review()[0], REFUSED)
            case.unlist()
            rc, said = case.review()
            self.assertEqual(rc, 0, said)
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            record = json.loads(case.record_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "applied")

    def test_an_apply_finishing_an_earlier_write_records_it_and_writes_nothing(self) -> None:
        # The words say `apply` does not write a listed skill, not that its candidate
        # is never applied: an apply stopped after it wrote the skill is finished by
        # running it again, which records that write before the target is judged.
        with tempfile.TemporaryDirectory() as tmp:
            case = _Listed(Path(tmp), listed=False)
            real = skill_fs._write_pinned

            def write_then_stop(*args: Any, **kwargs: Any) -> None:
                real(*args, **kwargs)
                raise cli.Stopped(signal.SIGTERM)

            with mock.patch.object(skill_fs, "_write_pinned", write_then_stop):
                rc, said = case.apply()
            self.assertNotEqual(rc, 0, said)
            written = case.skill.read_bytes()
            record = json.loads(case.record_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "validated")
            self.assertEqual(written.decode("utf-8"), record["patch"]["new_text"])
            case.list_it()
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            record = json.loads(case.record_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "applied")
            self.assertEqual(case.skill.read_bytes(), written)
            receipt = case.state / "receipts" / f"apply--{case.cid}.json"
            self.assertIs(json.loads(receipt.read_text())["finishes_an_earlier_run"], True)


if __name__ == "__main__":
    unittest.main()
