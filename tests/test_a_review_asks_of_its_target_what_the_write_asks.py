"""Review and apply ask of a target what the write asks before it opens the file.

Measured before (3.11 and 3.13), with a real validated candidate: listing its
skill in ``[guard].protected_skill_ids`` after it was validated left ``review``
printing a token and an apply command, and ``apply`` wrote the skill, the list
being read only when a proposal is made; and with the record's path spelled
through a symbolic link (a skill folder that is a link below its root, a folder
outside every root that links into one, or an alias of the root), the diff
relabelled to match, ``review`` printed a token and an apply command that
``apply`` then refused with ``LinkedTargetRefused``, after it had made a backup
directory that it left empty; and with the path relabelled to a copy of the skill
inside a protected path, ``review`` refused it as an unrecognised target and
``apply`` as protected, again after making a backup directory it left empty.

Each test that runs the commands has a candidate of its own, so a regression
that lets one ``apply`` write is reported by that test alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.cli import RC_REFUSED
from agent_skill_improver.config import load_config
from agent_skill_improver.domain.candidate import make_patch
from agent_skill_improver.errors import SelfMutationRefused
from tests.support import build_env, run_cli

NO_GUARD = "protected_skill_ids = []"


def _validated(base: Path) -> tuple[list[str], str, str]:
    cfg = ["--config", str(build_env(base, dry_run=False, seed="review-asks-the-write"))]
    proc = run_cli([*cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
    cid = next(
        c["candidate_id"]
        for c in json.loads(proc.stdout)["candidates"]
        if c.get("status") == "validated"
    )
    token = json.loads(run_cli([*cfg, "--json", "review", cid]).stdout)["approval_token"]
    return cfg, cid, token


class _Target(unittest.TestCase):
    """A fresh validated candidate per test, against the ``tdd`` skill."""

    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.cfg, self.cid, self.token = _validated(self.base)
        self.record = self.base / "state" / "candidates" / f"{self.cid}.json"
        self.config = self.base / "config" / "config.toml"
        self.root = self.base / "skills-canonical"
        self.skill = self.root / "tdd" / "SKILL.md"
        self.skill_bytes = self.skill.read_bytes()

    def guarded(self, *skill_ids: str) -> None:
        text = self.config.read_text(encoding="utf-8")
        self.assertEqual(text.count(NO_GUARD), 1)
        listed = "protected_skill_ids = [" + ", ".join(f'"{s}"' for s in skill_ids) + "]"
        self.config.write_text(text.replace(NO_GUARD, listed), encoding="utf-8")

    def spelled(self, path: Path, skill_id: str) -> None:
        """The record's skill spelled as ``path``, its id and diff to match."""
        record = json.loads(self.record.read_text(encoding="utf-8"))
        record["target"]["canonical_path"] = str(path)
        record["target"]["skill_id"] = skill_id
        record["patch"]["unified_diff"] = make_patch(
            self.skill_bytes.decode("utf-8"), record["patch"]["new_text"], path_label=str(path)
        ).unified_diff
        self.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", "utf-8")

    def review_and_apply(self) -> tuple[Any, Any]:
        review = run_cli([*self.cfg, "review", self.cid])
        apply = run_cli(
            [*self.cfg, "apply", self.cid, "--approve", self.token, "--i-reviewed-this"]
            + ["--operator", "t"]
        )
        return review, apply

    def assert_both_refuse(self, refusal: str, review_says: str, apply_says: str) -> tuple:
        review, apply = self.review_and_apply()
        self.assertEqual(review.returncode, RC_REFUSED, review.stdout + review.stderr)
        self.assertNotIn("--approve", review.stdout, "review printed an apply command")
        self.assertIn(f"{refusal}: refusing to {review_says} ", review.stderr)
        self.assertEqual(apply.returncode, RC_REFUSED, apply.stdout + apply.stderr)
        self.assertIn(f"{refusal}: refusing to {apply_says} ", apply.stderr)
        self.assertEqual(self.skill.read_bytes(), self.skill_bytes, "the skill was written")
        backups = self.base / "state" / "backups"
        self.assertEqual(list(backups.glob("*")) if backups.exists() else [], [])
        return review, apply


class AProtectedSkillIsNeverReviewedOrWrittenTests(_Target):
    def test_a_skill_listed_as_protected_after_validation_gets_no_token_and_is_not_written(self):
        self.guarded("tdd")
        for proc in self.assert_both_refuse("SelfMutationRefused", "review", "write"):
            self.assertIn("'tdd' is listed in [guard].protected_skill_ids", proc.stderr)

    def test_a_skill_not_listed_is_reviewed_as_before(self):
        self.guarded("example-guardrails")
        review = run_cli([*self.cfg, "review", self.cid])
        self.assertEqual(review.returncode, 0, review.stderr)
        self.assertIn(f"--approve {self.token}", review.stdout)


class AReviewAsksOfALinkWhatTheWriteAsksTests(_Target):
    def test_a_skill_folder_that_is_a_link_below_its_root_gets_no_token(self):
        os.symlink(self.root / "tdd", self.root / "y")
        self.spelled(self.root / "y" / "SKILL.md", "y")
        self.assert_both_refuse("LinkedTargetRefused", "review", "replace")

    def test_a_folder_outside_every_root_that_links_into_one_gets_no_token(self):
        (self.base / "outside").mkdir()
        os.symlink(self.root / "tdd", self.base / "outside" / "x")
        self.spelled(self.base / "outside" / "x" / "SKILL.md", "x")
        self.assert_both_refuse("LinkedTargetRefused", "review", "replace")

    def test_an_alias_of_the_root_gets_no_token(self):
        os.symlink(self.root, self.base / "alias")
        self.spelled(self.base / "alias" / "tdd" / "SKILL.md", "tdd")
        self.assert_both_refuse("LinkedTargetRefused", "review", "replace")


class AProtectedPathIsRefusedAsProtectedByBothTests(_Target):
    def test_a_copy_of_the_skill_inside_a_protected_path_is_refused_as_protected_by_both(self):
        inside = self.base / "rubrics" / "tdd"
        inside.mkdir(parents=True)
        shutil.copy(self.skill, inside / "SKILL.md")
        self.spelled(inside / "SKILL.md", "tdd")
        for proc in self.assert_both_refuse("SelfMutationRefused", "review", "write"):
            self.assertIn("it is inside a protected path", proc.stderr)


class AWriteTargetJudgedAsTheWriteJudgesItTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        base = Path(cls._tmp.name).resolve()
        cls.config = load_config(build_env(base, dry_run=False, seed="review-asks-the-write"))
        cls.skill = base / "skills-canonical" / "tdd" / "SKILL.md"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def store(self, *protected: str) -> FilesystemSkillStore:
        return FilesystemSkillStore(
            self.config.skill_roots,
            protected_paths=self.config.protected_paths,
            protected_skill_ids=protected,
        )

    def test_the_skill_the_inventory_finds_is_judged(self):
        with self.assertRaises(SelfMutationRefused) as caught:
            self.store("tdd").assert_write_target(self.skill, action="review")
        self.assertIn("refusing to review ", str(caught.exception))

    def test_the_skill_the_record_names_is_judged(self):
        with self.assertRaises(SelfMutationRefused) as caught:
            self.store("guarded").assert_write_target(
                self.skill, action="write", skill_id="guarded"
            )
        self.assertIn("refusing to write ", str(caught.exception))
        self.assertIn("'guarded' is listed", str(caught.exception))

    def test_a_skill_path_that_does_not_read_as_one_is_named_quoted_on_one_line(self):
        canonical = next(r for r in self.config.skill_roots if r.canonical)
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "skills\nroot"
            (root / "tdd").mkdir(parents=True)
            (root / "tdd" / "SKILL.md").write_bytes(self.skill.read_bytes())
            store = FilesystemSkillStore(
                [replace(canonical, path=root)], protected_skill_ids=("tdd",)
            )
            with self.assertRaises(SelfMutationRefused) as caught:
                store.assert_write_target(root / "tdd" / "SKILL.md", action="review")
        message = str(caught.exception)
        self.assertNotIn("\n", message)
        self.assertIn("skills\\nroot", message)
        self.assertIn("skill 'tdd' is listed", message)

    def test_the_write_itself_refuses_the_skill_the_record_names_and_makes_no_backup(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            config = load_config(build_env(base, dry_run=False, seed="review-asks-the-write"))
            skill = base / "skills-canonical" / "tdd" / "SKILL.md"
            before = skill.read_bytes()
            store = FilesystemSkillStore(
                config.skill_roots,
                read_only=False,
                protected_paths=config.protected_paths,
                protected_skill_ids=("guarded",),
            )
            backup_dir = base / "backup"
            with self.assertRaises(SelfMutationRefused) as caught:
                store.write_canonical(
                    skill,
                    "replaced\n",
                    expected_sha256=hashlib.sha256(before).hexdigest(),
                    backup_dir=backup_dir,
                    skill_id="guarded",
                )
            self.assertEqual(skill.read_bytes(), before)
            self.assertFalse(backup_dir.exists())
        self.assertIn("refusing to write ", str(caught.exception))
        self.assertIn("'guarded' is listed", str(caught.exception))

    def test_an_unlisted_skill_is_the_resolved_target(self):
        target = self.store("other").assert_write_target(
            self.skill, action="review", skill_id="tdd"
        )
        self.assertEqual(target, self.skill.resolve())


class ACleanCandidateIsStillReviewedAndAppliedTests(_Target):
    def test_a_clean_candidate_is_reviewed_with_a_token_and_applied(self):
        review, apply = self.review_and_apply()
        self.assertEqual(review.returncode, 0, review.stderr)
        self.assertIn(f"--approve {self.token}", review.stdout)
        self.assertEqual(apply.returncode, 0, apply.stderr)
        self.assertNotEqual(self.skill.read_bytes(), self.skill_bytes)
        self.assertEqual(len(list((self.base / "state" / "backups").glob("*"))), 1)


if __name__ == "__main__":
    unittest.main()
