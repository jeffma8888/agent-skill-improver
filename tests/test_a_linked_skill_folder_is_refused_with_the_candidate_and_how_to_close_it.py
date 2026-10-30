"""A skill folder that became a link is refused with the candidate and how to close it.

``review`` and ``apply`` judge a candidate's target through the target check
(``assert_write_target`` in ``adapters/skill_fs.py``), which refuses a target
reached through a symbolic link below its root. Measured before (3.11 and
3.13, a committed run with one validated candidate, then its skill folder
replaced by a link to an identical copy, beside it in the root or outside
every root): ``review`` and ``apply`` each exited 3 with ``LinkedTargetRefused``
"refusing to review <path>" and "refusing to replace <path>", which named no
candidate and no way to close it. The record stayed ``validated``, so it stayed
open: with ``[caps].max_open_candidates`` at 1 the next day stood down on it,
and with the record deleted the next day ran. With the folder put back in place
of the link, ``review`` and ``apply`` with the same token took it.
``reached_through_a_link`` now words the refusal with the candidate, for the
command that asked, and says how it is closed; without a candidate
(``rollback``) the words are as before.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
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


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


def how_to_close(cid: str, itself: str = "the directory itself") -> str:
    return (
        "The link, what it points to and the candidate are unchanged. The candidate stays "
        "open, and counts against [caps].max_open_candidates, until it is closed: to close "
        f"it, delete its record ({cid}.json in the state directory's candidates/). If the "
        f"link is not meant to be there, put {itself} in its place: while the skill file "
        "there holds the text the candidate was written against, `review` and `apply` take "
        "the candidate as before."
    )


class _Linked:
    """One validated candidate, its token taken, then its skill folder made a link."""

    def __init__(self, base: Path, *, where: str = "beside", linked: bool = True, **env: int):
        self.base = base
        self.config = build_env(base / "env", seed="c223", **env)
        self.state = base / "env" / "state"
        rc, said = run(self.config, "daily", "--commit", "--day", DAY)
        assert rc == 0, said
        (self.record_file,) = sorted(self.state.glob("candidates/*.json"))
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self.cid = record["id"]
        self.skill = Path(record["target"]["canonical_path"])
        self.folder = self.skill.parent
        rc, shown = run(self.config, "review", self.cid)
        assert rc == 0, shown
        found = re.search(r"--approve (\S+)", shown)
        assert found, shown
        self.token = found.group(1)
        if where == "beside":
            self.copy = self.folder.with_name(self.folder.name + "-copy")
        else:
            self.copy = base / "outside" / self.folder.name
        if linked:
            self.link_it()

    def link_it(self) -> None:
        self.copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.folder), str(self.copy))
        self.folder.symlink_to(self.copy, target_is_directory=True)

    def put_back(self) -> None:
        self.folder.unlink()
        shutil.move(str(self.copy), str(self.folder))

    def review(self) -> tuple[int, str]:
        return run(self.config, "review", self.cid)

    def apply(self) -> tuple[int, str]:
        return run(self.config, "apply", self.cid, "--approve", self.token, "--i-reviewed-this")

    def left_as_it_is(self) -> tuple[str, bytes, bytes, list[Path]]:
        """Where the link points, the copy's skill bytes, the record's bytes, the backups made."""
        backups = sorted(self.state.glob("backups/*"))
        return (
            os.readlink(self.folder),
            (self.copy / "SKILL.md").read_bytes(),
            self.record_file.read_bytes(),
            backups,
        )


class TheWordsTests(unittest.TestCase):
    """The refusal itself, reached through its module so on a tree without it only this errors."""

    def test_the_link_words_name_the_candidate_for_the_command_that_asked(self) -> None:
        cid = "CAND-2026-02-01-01-tdd"
        skill = Path("/skills/tdd/SKILL.md")
        for verb, command in (("review", "review"), ("replace", "apply")):
            for link, itself in (
                (skill.parent, "the directory itself"),
                (skill, "the file itself"),
            ):
                with self.subTest(verb=verb, link=str(link)):
                    self.assertEqual(
                        skill_fs.reached_through_a_link(skill, link, cid, verb),
                        f"refusing to {command} {cid}: its canonical target {skill} is reached "
                        f"through {link}, a symbolic link, so `apply` does not write through it: "
                        "the replacement would land wherever the link points rather than on the "
                        "skill the candidate was reviewed against. "
                        f"{how_to_close(cid, itself)}",
                    )

    def test_without_a_candidate_the_link_words_are_as_before(self) -> None:
        skill = Path("/skills/tdd/SKILL.md")
        self.assertEqual(
            skill_fs.reached_through_a_link(skill, skill.parent, "", "replace"),
            f"refusing to replace {skill}: {skill.parent} is a symbolic link, so the replacement "
            "would land wherever it points rather than on the skill the candidate was reviewed "
            "against. Neither the link nor its target was touched.",
        )


class ReviewAndApplyOfALinkedFolderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.cases = {}
        cls.left = {}
        cls.refused = {}
        for where in ("beside", "outside"):
            case = _Linked(Path(cls._tmp.name) / where, where=where)
            cls.cases[where] = case
            cls.left[where] = case.left_as_it_is()
            cls.refused[where] = {"review": case.review(), "apply": case.apply()}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_review_and_apply_of_a_linked_folder_name_the_candidate_and_how_to_close_it(
        self,
    ) -> None:
        for where, case in self.cases.items():
            for action, (rc, said) in self.refused[where].items():
                with self.subTest(where=where, action=action):
                    self.assertEqual(rc, REFUSED, said)
                    self.assertTrue(
                        said.startswith(
                            f"LinkedTargetRefused: refusing to {action} {case.cid}: its canonical "
                            f"target {case.skill} is reached through {case.folder}, a symbolic "
                            "link, so `apply` does not write through it"
                        ),
                        said,
                    )
                    self.assertIn(how_to_close(case.cid), said)
                    self.assertNotIn("was touched", said)
                    self.assertEqual(said.count("\n"), 1, said)

    def test_the_link_its_copy_the_record_and_the_backups_are_as_the_refusal_says(self) -> None:
        for where, case in self.cases.items():
            with self.subTest(where):
                self.assertEqual(case.left_as_it_is(), self.left[where])
                self.assertEqual(self.left[where][3], [])
                record = json.loads(case.record_file.read_text(encoding="utf-8"))
                self.assertEqual(record["status"], "validated")


class WhatTheRefusalSaysToDoTests(unittest.TestCase):
    def test_kept_the_candidate_of_a_linked_folder_holds_the_queue_and_deleted_it_does_not(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _Linked(Path(tmp), max_open_candidates=1)
            rc, said = run(case.config, "daily", "--commit", "--day", NEXT_DAY)
            self.assertEqual(rc, 0, said)
            self.assertIn("status=stood-down", said)
            self.assertIn("1 candidate(s) already awaiting review, cap is 1", said)
            case.record_file.unlink()
            rc, said = run(case.config, "daily", "--commit", "--day", THIRD_DAY)
            self.assertEqual(rc, 0, said)
            self.assertNotIn("status=stood-down", said)
            self.assertNotIn("awaiting review", said)
            self.assertIn("scored=5", said)

    def test_the_directory_put_back_reviews_and_applies_with_the_same_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _Linked(Path(tmp))
            self.assertEqual(case.review()[0], REFUSED)
            case.put_back()
            rc, said = case.review()
            self.assertEqual(rc, 0, said)
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            record = json.loads(case.record_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "applied")


class TheNeighbouringPathsTests(unittest.TestCase):
    def test_rollback_past_a_linked_folder_keeps_its_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _Linked(Path(tmp), linked=False)
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            case.link_it()
            copied = (case.copy / "SKILL.md").read_bytes()
            rc, said = run(case.config, "rollback", case.cid)
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(said.startswith("LinkedTargetRefused: refusing to replace "), said)
            self.assertIn(
                f"/{case.folder.name} is a symbolic link, so the replacement would land", said
            )
            self.assertIn("Neither the link nor its target was touched.", said)
            self.assertNotIn(case.cid, said)
            self.assertEqual((case.copy / "SKILL.md").read_bytes(), copied)

    def test_an_apply_finishing_an_earlier_write_writes_nothing_through_the_link(self) -> None:
        # The words say `apply` does not write through the link, not that the
        # candidate is never applied: an apply stopped after it wrote the skill is
        # finished by running it again, which records that write before the target
        # is judged.
        with tempfile.TemporaryDirectory() as tmp:
            case = _Linked(Path(tmp), linked=False)
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
            case.link_it()
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            record = json.loads(case.record_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "applied")
            self.assertEqual((case.copy / "SKILL.md").read_bytes(), written)
            receipt = case.state / "receipts" / f"apply--{case.cid}.json"
            self.assertIs(json.loads(receipt.read_text())["finishes_an_earlier_run"], True)


if __name__ == "__main__":
    unittest.main()
