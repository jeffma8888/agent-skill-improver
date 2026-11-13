"""A target inside a protected path is refused with the candidate and how to close it.

The config's protected paths (this tool's package, the config file and its
directory, the state directory, the rubrics directory) are read again by
``review`` and ``apply``, so a candidate validated before one of them came to
hold its skill is refused by the target check (``assert_write_target`` in
``adapters/skill_fs.py``). Measured before (3.11 and 3.13, a committed run with
one validated candidate, its token taken, then the config file moved into the
skill's folder, or the rubrics directory pointed at it): ``review`` and
``apply`` each exited 3 with ``SelfMutationRefused`` "refusing to review
<path>: it is inside a protected path (<dir>)" and "refusing to write <path>:
...", which named no candidate and no way on; the record stayed ``validated``,
open, the skill unchanged and no backup taken. ``inside_a_protected_path`` now
words the refusal with the candidate, for the command that asked, and says how
it is closed; without a candidate the words are as before.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters import skill_fs
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore

from tests.support import build_env

DAY, NEXT_DAY, THIRD_DAY = "2026-02-01", "2026-02-02", "2026-02-03"
REFUSED = 3
#: Where the protected path comes to hold the skill after its candidate is validated.
CONFIG_MOVED_IN, RUBRICS_POINTED_AT_IT = "the config file", "the rubrics directory"


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


def how_to_close(cid: str) -> str:
    return (
        "The skill and the candidate are unchanged. The candidate stays open, and counts "
        "against [caps].max_open_candidates, until it is closed: to close it, delete its "
        f"record ({cid}.json in the state directory's candidates/). If that protected path "
        "is not meant to be where the skill is (a config file or a rubrics directory placed "
        "in the skill's folder), move it out, and `review` and `apply` take the candidate as "
        "before."
    )


class _Protected:
    """One validated candidate, its token taken, then its skill's folder made protected."""

    def __init__(self, base: Path, where: str, **env: int) -> None:
        self.env = base / "env"
        self.config = build_env(self.env, seed="c237", **env)
        text = self.config.read_text(encoding="utf-8")
        assert text.count('state_dir = "state"\n') == 1 and text.count('dir = "../rubrics"\n') == 1
        self.rubrics = self.env / "rubrics"
        text = text.replace(
            'state_dir = "state"\n', f"state_dir = {json.dumps(str(self.env / 'state'))}\n"
        )
        self.text = text.replace('dir = "../rubrics"\n', f"dir = {json.dumps(str(self.rubrics))}\n")
        self.config.write_text(self.text, encoding="utf-8")
        self.state = self.env / "state"
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
        self.where = where
        self.protect()

    def protect(self) -> None:
        if self.where == CONFIG_MOVED_IN:
            moved = self.folder / "asi.toml"
            moved.write_text(self.text, encoding="utf-8")
            self.config.unlink()
            self.config = moved
        else:
            for rubric in self.rubrics.glob("*.toml"):
                shutil.copy(rubric, self.folder / rubric.name)
            pointed = self.text.replace(
                f"dir = {json.dumps(str(self.rubrics))}\n",
                f"dir = {json.dumps(str(self.folder))}\n",
            )
            self.config.write_text(pointed, encoding="utf-8")

    def move_it_out(self) -> None:
        """What the refusal advises: the protected path out of the skill's folder."""
        if self.where == CONFIG_MOVED_IN:
            back = self.env / "config" / "asi.toml"
            back.parent.mkdir(exist_ok=True)
            back.write_text(self.text, encoding="utf-8")
            self.config.unlink()
            self.config = back
        else:
            for rubric in self.rubrics.glob("*.toml"):
                (self.folder / rubric.name).unlink()
            self.config.write_text(self.text, encoding="utf-8")

    def review(self) -> tuple[int, str]:
        return run(self.config, "review", self.cid)

    def apply(self) -> tuple[int, str]:
        return run(self.config, "apply", self.cid, "--approve", self.token, "--i-reviewed-this")

    def everything(self) -> dict[str, bytes | None]:
        """Every entry under the environment, a file's bytes or None for a directory.

        All but the run lock: `apply` writes its owner line before it judges the
        target, as it did before this row.
        """
        lock = self.state / "locks" / "daily.lock"
        return {
            str(entry.relative_to(self.env)): (None if entry.is_dir() else entry.read_bytes())
            for entry in sorted(self.env.rglob("*"))
            if entry != lock
        }

    def left_as_it_is(self) -> tuple[bytes, bytes, list[Path]]:
        """The skill's bytes, the record's bytes, and the backups made."""
        return (
            self.skill.read_bytes(),
            self.record_file.read_bytes(),
            sorted(self.state.glob("backups/*")),
        )


class TheWordsTests(unittest.TestCase):
    """The refusal itself, reached through its module so on a tree without it only this errors."""

    def test_the_protected_path_words_name_the_candidate_for_the_command_that_asked(self) -> None:
        cid = "CAND-2026-02-01-01-tdd"
        path, guard = Path("/skills/tdd/SKILL.md"), Path("/skills/tdd")
        for verb, command in (("review", "review"), ("write", "apply")):
            with self.subTest(verb):
                self.assertEqual(
                    skill_fs.inside_a_protected_path(path, guard, cid, verb),
                    f"refusing to {command} {cid}: its target {path} lies inside a protected "
                    f"path ({guard}), so `apply` does not write it: this tool never edits its own "
                    f"source, config, state or rubrics. {how_to_close(cid)}",
                )

    def test_without_a_candidate_the_protected_path_words_are_as_before(self) -> None:
        path, guard = Path("/skills/tdd/SKILL.md"), Path("/skills/tdd")
        self.assertEqual(
            skill_fs.inside_a_protected_path(path, guard, "", "write"),
            f"refusing to write {path}: it is inside a protected path ({guard}). This tool never "
            "edits its own source, config, or rubrics -- a loop that can rewrite its own scoring "
            "criteria has no gate at all.",
        )

    def test_the_target_and_the_protected_path_are_named_on_one_line(self) -> None:
        guard = Path("/skills/a\nconfig error: forged")
        said = skill_fs.inside_a_protected_path(guard / "SKILL.md", guard, "CAND-x", "review")
        self.assertNotIn("\n", said)
        self.assertIn(skill_fs.named_path(guard), said)


class ReviewAndApplyOfATargetInsideAProtectedPathTests(unittest.TestCase):
    def test_review_and_apply_of_a_target_inside_a_protected_path_name_the_candidate(self) -> None:
        for where in (CONFIG_MOVED_IN, RUBRICS_POINTED_AT_IT):
            with self.subTest(where=where), tempfile.TemporaryDirectory() as tmp:
                case = _Protected(Path(tmp).resolve(), where)
                left = case.left_as_it_is()
                for action in ("review", "apply"):
                    with self.subTest(action=action):
                        rc, said = case.review() if action == "review" else case.apply()
                        self.assertEqual(rc, REFUSED, said)
                        self.assertTrue(
                            said.startswith(
                                f"SelfMutationRefused: refusing to {action} {case.cid}: its target "
                                f"{case.skill} lies inside a protected path ({case.folder}), so "
                                "`apply` does not write it"
                            ),
                            said,
                        )
                        self.assertIn(how_to_close(case.cid), said)
                        self.assertEqual(said.count("\n"), 1, said)
                        self.assertEqual(case.left_as_it_is(), left)
                self.assertEqual(left[2], [], "a backup was taken")
                record = json.loads(case.record_file.read_text(encoding="utf-8"))
                self.assertEqual(record["status"], "validated")

    def test_a_refusal_writes_nothing_but_the_run_locks_owner_line(self) -> None:
        for where in (CONFIG_MOVED_IN, RUBRICS_POINTED_AT_IT):
            with self.subTest(where=where), tempfile.TemporaryDirectory() as tmp:
                case = _Protected(Path(tmp).resolve(), where)
                before = case.everything()
                # Each command is run, and the environment compared, before the
                # next runs: a refusal that wrote something the next one removed
                # would otherwise compare equal.
                for action in ("review", "apply"):
                    with self.subTest(action=action):
                        rc, said = case.review() if action == "review" else case.apply()
                        self.assertEqual(rc, REFUSED, said)
                        self.assertEqual(case.everything(), before)


class TheWritesOwnChecksTests(unittest.TestCase):
    """The checks after `apply`'s own: the write's, and `rollback`'s, which names no candidate."""

    def test_a_path_protected_after_applys_own_check_is_refused_by_the_write_with_the_candidate(
        self,
    ) -> None:
        # The store reads its protected paths once, when the config is loaded,
        # so this puts the folder on the store's list at the step `apply` takes
        # after its own check, to reach the write's own check with it. A config
        # file really moved into the skill's folder while `apply` runs is not
        # seen by either check (ACCEPTANCE, the row's Not covered).
        with tempfile.TemporaryDirectory() as tmp:
            case = _Protected(Path(tmp).resolve(), RUBRICS_POINTED_AT_IT)
            case.move_it_out()
            real = FilesystemSkillStore.mirror_paths_of
            done: list[bool] = []

            def protect_then_list(store: FilesystemSkillStore, path: Any) -> Any:
                if not done:
                    done.append(True)
                    store._protected = (*store._protected, case.folder)
                return real(store, path)

            left = case.left_as_it_is()
            with mock.patch.object(FilesystemSkillStore, "mirror_paths_of", protect_then_list):
                rc, said = case.apply()
            self.assertEqual(done, [True])
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                said.startswith(
                    f"SelfMutationRefused: refusing to apply {case.cid}: its target {case.skill} "
                    f"lies inside a protected path ({case.folder}), so `apply` does not write it"
                ),
                said,
            )
            self.assertIn(how_to_close(case.cid), said)
            self.assertEqual(case.left_as_it_is(), left)

    def test_rollback_of_a_skill_made_protected_after_its_apply_names_no_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _Protected(Path(tmp).resolve(), RUBRICS_POINTED_AT_IT)
            case.move_it_out()
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            case.protect()
            left = case.left_as_it_is()
            rc, said = run(case.config, "rollback", case.cid)
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                said.startswith(
                    f"SelfMutationRefused: refusing to write {case.skill}: it is inside a "
                    f"protected path ({case.folder}). This tool never edits"
                ),
                said,
            )
            self.assertNotIn(case.cid, said)
            self.assertEqual(case.left_as_it_is()[:2], left[:2])


class WhatTheRefusalSaysToDoTests(unittest.TestCase):
    def test_kept_inside_a_protected_path_the_candidate_holds_the_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _Protected(Path(tmp).resolve(), CONFIG_MOVED_IN, max_open_candidates=1)
            rc, said = run(case.config, "daily", "--commit", "--day", NEXT_DAY)
            self.assertEqual(rc, 0, said)
            self.assertIn("status=stood-down", said)
            self.assertIn("1 candidate(s) already awaiting review, cap is 1", said)
            case.record_file.unlink()
            rc, said = run(case.config, "daily", "--commit", "--day", THIRD_DAY)
            self.assertEqual(rc, 0, said)
            self.assertIn("status=ok", said)
            self.assertNotIn("awaiting review", said)

    def test_moved_out_the_same_token_reviews_and_applies(self) -> None:
        for where in (CONFIG_MOVED_IN, RUBRICS_POINTED_AT_IT):
            with self.subTest(where=where), tempfile.TemporaryDirectory() as tmp:
                case = _Protected(Path(tmp).resolve(), where)
                self.assertEqual(case.review()[0], REFUSED)
                case.move_it_out()
                rc, said = case.review()
                self.assertEqual(rc, 0, said)
                rc, said = case.apply()
                self.assertEqual(rc, 0, said)
                record = json.loads(case.record_file.read_text(encoding="utf-8"))
                self.assertEqual(record["status"], "applied")


if __name__ == "__main__":
    unittest.main()
