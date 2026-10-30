"""A target that is no longer the canonical copy is refused with the candidate and how to close it.

``review`` and ``apply`` judge a candidate's target through the target check
(``assert_write_target`` in ``adapters/skill_fs.py``), which refuses a target
that is not the canonical copy of its skill (``assert_canonical_target``).
Measured before (3.11 and 3.13, a committed run with one validated candidate,
its token taken, then a second copy of the skill made in the editable root, the
canonical and mirror roots swapped, or the canonical root re-pointed at a copy
of the tree): ``review`` and ``apply`` each exited 3 with ``MirrorEditRefused``
"refusing to review <path>: ..." and "refusing to write <path>: ...", which
named no candidate and no way to close it. ``review`` changed nothing and
``apply`` only the run lock; the record stayed ``validated``, so it stayed
open: with ``[caps].max_open_candidates`` at 1 the next day stood down on it,
and with the record deleted the next day ran. With the change undone,
``review`` and ``apply`` with the same token took it.
``canonical_target_refused`` now words the refusal with the candidate, for
the command that asked, and says how it is closed; without a candidate
(``rollback``, and the write's own check) the words are as before.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import signal
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters import skill_fs

from tests.support import build_env

DAY, NEXT_DAY, THIRD_DAY = "2026-02-01", "2026-02-02", "2026-02-03"
REFUSED = 3
#: The config's two roots, as ``build_env`` writes them, swapped by ``mirror``.
SWAP = (
    ("precedence = 10\ncanonical = true\neditable = true", "@A@"),
    ("precedence = 90\ncanonical = false\neditable = false", "@B@"),
    ("@A@", "precedence = 90\ncanonical = false\neditable = false"),
    ("@B@", "precedence = 10\ncanonical = true\neditable = true"),
    ('kind = "custom"', "@K@"),
    ('kind = "mirror"', 'kind = "custom"'),
    ("@K@", 'kind = "mirror"'),
)
#: How each change's refusal gives its reason, from its start.
WHY = {
    "second copy": "skill 'tdd' has no canonical copy (editable root 'canonical' holds 2 copies",
    "mirror": "it is a mirror copy. The canonical source is ",
    "re-pointed": "it is not a skill document in any configured root, so this tool cannot tell",
}


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


def how_to_close(cid: str) -> str:
    return (
        "The file and the candidate are unchanged. The candidate stays open, and counts "
        "against [caps].max_open_candidates, until it is closed: to close it, delete its "
        f"record ({cid}.json in the state directory's candidates/). If that file is meant to "
        "be the skill's canonical copy, make it that again: while it holds the text the "
        "candidate was written against, `review` and `apply` take the candidate as before."
    )


class _NoLongerCanonical:
    """One validated candidate, its token taken, then its target made no canonical copy."""

    def __init__(self, base: Path, change: str = "", **env: int) -> None:
        self.base = base
        self.config = build_env(base / "env", seed="c224", **env)
        self.text = self.config.read_text(encoding="utf-8")
        self.state = base / "env" / "state"
        rc, said = run(self.config, "daily", "--commit", "--day", DAY)
        assert rc == 0, said
        (self.record_file,) = sorted(self.state.glob("candidates/*.json"))
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self.cid = record["id"]
        self.skill = Path(record["target"]["canonical_path"])
        self.root = self.skill.parent.parent
        rc, shown = run(self.config, "review", self.cid)
        assert rc == 0, shown
        found = re.search(r"--approve (\S+)", shown)
        assert found, shown
        self.token = found.group(1)
        self.undo: Callable[[], None] = lambda: None
        if change:
            self.make(change)

    def make(self, change: str) -> None:
        if change == "second copy":
            extra = self.root / "group" / self.skill.parent.name / "SKILL.md"
            extra.parent.mkdir(parents=True)
            shutil.copy(self.skill, extra)
            self.undo = lambda: shutil.rmtree(self.root / "group")
        elif change == "mirror":
            swapped = self.text
            for old, new in SWAP:
                swapped = swapped.replace(old, new)
            assert swapped != self.text
            self.config.write_text(swapped, encoding="utf-8")
            self.undo = lambda: self.config.write_text(self.text, encoding="utf-8")
        else:
            main = self.root.with_name("skills-main")
            shutil.copytree(self.root, main)
            repointed = self.text.replace("skills-canonical", "skills-main")
            assert repointed != self.text
            self.config.write_text(repointed, encoding="utf-8")

            def undo() -> None:
                self.config.write_text(self.text, encoding="utf-8")
                shutil.rmtree(main)

            self.undo = undo

    def review(self) -> tuple[int, str]:
        return run(self.config, "review", self.cid)

    def apply(self) -> tuple[int, str]:
        return run(self.config, "apply", self.cid, "--approve", self.token, "--i-reviewed-this")

    def left_as_it_is(self) -> tuple[bytes, bytes, list[Path]]:
        """The skill's bytes, the record's bytes, the backups made."""
        return (
            self.skill.read_bytes(),
            self.record_file.read_bytes(),
            sorted(self.state.glob("backups/*")),
        )


class TheCanonicalTargetWordsTests(unittest.TestCase):
    """The refusal itself, reached through its module so on a tree without it only this errors."""

    def test_the_canonical_target_words_name_the_candidate_for_the_command_that_asked(
        self,
    ) -> None:
        cid = "CAND-2026-02-01-01-tdd"
        skill = Path("/skills/tdd/SKILL.md")
        why = "it is a mirror copy."
        for action, command in (("review", "review"), ("write", "apply")):
            with self.subTest(action=action):
                self.assertEqual(
                    skill_fs.canonical_target_refused(skill, why, cid, action),
                    f"refusing to {command} {cid} at {skill}: {why} {how_to_close(cid)}",
                )

    def test_without_a_candidate_the_canonical_target_words_are_as_before(self) -> None:
        skill = Path("/skills/tdd/SKILL.md")
        why = "it is a mirror copy."
        for action in ("write", "roll back", "review"):
            with self.subTest(action=action):
                self.assertEqual(
                    skill_fs.canonical_target_refused(skill, why, "", action),
                    f"refusing to {action} {skill}: {why}",
                )

    def test_the_canonical_target_words_name_a_path_with_a_break_on_their_line(self) -> None:
        skill = Path("/skills/tdd\nconfig error: forged/SKILL.md")
        for cid in ("CAND-2026-02-01-01-tdd", ""):
            with self.subTest(cid=cid):
                said = skill_fs.canonical_target_refused(
                    skill, "it is a mirror copy.", cid, "write"
                )
                self.assertNotIn("\n", said)
                self.assertIn(repr(str(skill)), said)


class ReviewAndApplyOfANonCanonicalTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.cases = {}
        cls.left = {}
        cls.refused = {}
        for index, change in enumerate(WHY):
            case = _NoLongerCanonical(Path(cls._tmp.name) / f"case{index}", change)
            cls.cases[change] = case
            cls.left[change] = case.left_as_it_is()
            cls.refused[change] = {"review": case.review(), "apply": case.apply()}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_review_and_apply_of_a_target_no_longer_canonical_name_the_candidate(self) -> None:
        for change, case in self.cases.items():
            for action, (rc, said) in self.refused[change].items():
                with self.subTest(change=change, action=action):
                    self.assertEqual(rc, REFUSED, said)
                    self.assertTrue(
                        said.startswith(
                            f"MirrorEditRefused: refusing to {action} {case.cid} at "
                            f"{case.skill}: {WHY[change]}"
                        ),
                        said,
                    )
                    self.assertTrue(said.endswith(f" {how_to_close(case.cid)}\n"), said)
                    self.assertEqual(said.count("\n"), 1, said)

    def test_a_non_canonical_target_its_record_and_the_backups_are_as_the_refusal_says(
        self,
    ) -> None:
        for change, case in self.cases.items():
            with self.subTest(change):
                self.assertEqual(case.left_as_it_is(), self.left[change])
                self.assertEqual(self.left[change][2], [])
                record = json.loads(case.record_file.read_text(encoding="utf-8"))
                self.assertEqual(record["status"], "validated")


class WhatTheCanonicalRefusalSaysToDoTests(unittest.TestCase):
    def test_kept_the_candidate_of_a_second_copy_holds_the_queue_and_deleted_it_does_not(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _NoLongerCanonical(Path(tmp), "second copy", max_open_candidates=1)
            self.assertEqual(case.review()[0], REFUSED)
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

    def test_each_change_undone_reviews_and_applies_with_the_same_token(self) -> None:
        for index, change in enumerate(WHY):
            with self.subTest(change), tempfile.TemporaryDirectory() as tmp:
                case = _NoLongerCanonical(Path(tmp) / f"case{index}", change)
                self.assertEqual(case.review()[0], REFUSED)
                case.undo()
                rc, said = case.review()
                self.assertEqual(rc, 0, said)
                rc, said = case.apply()
                self.assertEqual(rc, 0, said)
                record = json.loads(case.record_file.read_text(encoding="utf-8"))
                self.assertEqual(record["status"], "applied")


class TheCanonicalNeighbouringPathsTests(unittest.TestCase):
    def test_rollback_of_a_target_become_a_mirror_keeps_its_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _NoLongerCanonical(Path(tmp))
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            written = case.skill.read_bytes()
            case.make("mirror")
            rc, said = run(case.config, "rollback", case.cid)
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                said.startswith(
                    f"MirrorEditRefused: refusing to roll back {case.skill}: it is a mirror copy."
                ),
                said,
            )
            self.assertNotIn(case.cid, said)
            self.assertNotIn("max_open_candidates", said)
            self.assertEqual(case.skill.read_bytes(), written)

    def test_a_second_copy_made_after_applys_own_check_is_refused_by_the_write_as_before(
        self,
    ) -> None:
        # The write's own target check is not given the candidate: a second
        # copy made after `apply` judged the target, and before its write, is
        # refused in the old words, and the backup directory `apply` made for
        # the write is left behind, empty.
        with tempfile.TemporaryDirectory() as tmp:
            case = _NoLongerCanonical(Path(tmp))
            before = case.skill.read_bytes()
            real = skill_fs.FilesystemSkillStore.mirror_paths_of

            def copy_then_list(store: Any, path: Path) -> tuple[str, ...]:
                if not (case.root / "group").exists():
                    case.make("second copy")
                return real(store, path)

            with mock.patch.object(
                skill_fs.FilesystemSkillStore, "mirror_paths_of", copy_then_list
            ):
                rc, said = case.apply()
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                said.startswith(
                    f"MirrorEditRefused: refusing to write {case.skill}: skill 'tdd' has no "
                    "canonical copy"
                ),
                said,
            )
            self.assertNotIn(case.cid, said)
            self.assertEqual(case.skill.read_bytes(), before)
            record = json.loads(case.record_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "validated")
            (backup,) = sorted(case.state.glob("backups/*"))
            self.assertEqual(list(backup.iterdir()), [])

    def test_an_apply_finishing_an_earlier_write_is_recorded_past_a_second_copy(self) -> None:
        # The words are of `review` and a first `apply`: an apply stopped after
        # it wrote the skill is finished by running it again, which records
        # that write before the target is judged, so a second copy made since
        # does not stop it, as before.
        with tempfile.TemporaryDirectory() as tmp:
            case = _NoLongerCanonical(Path(tmp))
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
            case.make("second copy")
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            record = json.loads(case.record_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "applied")
            self.assertEqual(case.skill.read_bytes(), written)
            receipt = case.state / "receipts" / f"apply--{case.cid}.json"
            self.assertIs(json.loads(receipt.read_text())["finishes_an_earlier_run"], True)


if __name__ == "__main__":
    unittest.main()
