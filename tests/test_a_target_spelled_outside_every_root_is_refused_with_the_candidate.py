"""A target spelled outside every root is refused with the candidate and how to close it.

``review`` and ``apply`` judge a candidate's target through the target check
(``assert_write_target`` in ``adapters/skill_fs.py``), whose link check refuses
a target spelled outside every configured root that resolves into one.
Measured before (3.11 and 3.13, a committed run with one validated candidate,
its token taken, then its root moved to ``skills-main``, a link left in its
place, and the config re-pointed at ``skills-main``): ``review`` and ``apply``
each exited 3 with ``LinkedTargetRefused`` "refusing to review <path>: it is
not spelled under any configured skill root yet resolves to <path> inside
root canonical, ... Spell the target as the root does. Nothing was touched."
(from ``apply``, "refusing to replace <path>: ..."), which named no candidate
and told the operator to respell a path only the record holds. ``review``
changed nothing and ``apply`` only the run lock; the record stayed
``validated``, so it stayed open: with ``[caps].max_open_candidates`` at 1 the
next day stood down on it, and with the record deleted the next day ran. With
the change undone, ``review`` and ``apply`` with the same token took it.
``spelled_outside_every_root`` now words the refusal with the candidate, for
the command that asked, and says how it is closed; without a candidate
(``rollback``, and the write's own check) the words are as before. When the
candidate's path gives a spelling of the root (``_root_as_spelled``), and the
target check's later judgments take what the path resolves to
(``_assert_resolved_target``), the refusal names it as the one to configure; a
path that gives none, through a link no spelling of the root lies ahead of, is
given no such advice, and nor is one those judgments refuse, since so
configured the root's path is refused by them in turn, or one whose
judgments' discovery cannot finish.
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
from agent_skill_improver.domain.candidate import make_patch
from agent_skill_improver.domain.skills import SkillRoot

from tests.support import build_env

DAY, NEXT_DAY, THIRD_DAY = "2026-02-01", "2026-02-02", "2026-02-03"
REFUSED = 3


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


def how_to_close(cid: str, as_spelled: Path | None = None) -> str:
    closed = (
        "The skill and the candidate are unchanged. The candidate stays open, and counts "
        "against [caps].max_open_candidates, until it is closed: to close it, delete its "
        f"record ({cid}.json in the state directory's candidates/)."
    )
    if as_spelled is None:
        return closed
    return (
        f"{closed} If root canonical is meant to be configured as {as_spelled}, as the "
        "candidate's path spells it, configure it so: while the skill file there holds the "
        "text the candidate was written against, `review` and `apply` take the candidate as "
        "before."
    )


class _Moved:
    """One validated candidate, its token taken, then its root moved and re-pointed."""

    def __init__(self, base: Path, *, moved: bool = True, **env: int) -> None:
        self.base = base
        self.config = build_env(base / "env", seed="c225", **env)
        self.text = self.config.read_text(encoding="utf-8")
        self.state = base / "env" / "state"
        rc, said = run(self.config, "daily", "--commit", "--day", DAY)
        assert rc == 0, said
        (self.record_file,) = sorted(self.state.glob("candidates/*.json"))
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self.cid = record["id"]
        self.skill = Path(record["target"]["canonical_path"])
        self.root = self.skill.parent.parent
        self.main = self.root.with_name("skills-main")
        rc, shown = run(self.config, "review", self.cid)
        assert rc == 0, shown
        found = re.search(r"--approve (\S+)", shown)
        assert found, shown
        self.token = found.group(1)
        if moved:
            self.move()

    def move(self) -> None:
        shutil.move(str(self.root), str(self.main))
        self.root.symlink_to(self.main, target_is_directory=True)
        repointed = self.text.replace(self.root.name, self.main.name)
        assert repointed != self.text
        self.config.write_text(repointed, encoding="utf-8")

    def configure_as_the_path_spells_it(self) -> None:
        """The root configured as the record spells it: resolved when it was proposed.

        Under a temporary directory reached through a link (``/var`` on
        macOS) that is not the config's own spelling, which names the link.
        """
        self.configure_root_as(self.root)

    def configure_root_as(self, path: Path) -> None:
        """The canonical root configured as ``path``, the rest of the config as built."""
        configured = str(self.base / "env" / self.root.name)
        assert self.text.count(f'"{configured}"') == 1, configured
        self.config.write_text(self.text.replace(f'"{configured}"', f'"{path}"'), encoding="utf-8")

    def reconfigure(self, old: str, new: str) -> None:
        """One passage of the config as it now stands changed, the rest kept."""
        text = self.config.read_text(encoding="utf-8")
        assert text.count(old) == 1, old
        self.config.write_text(text.replace(old, new), encoding="utf-8")

    def respell(self, path: Path) -> None:
        """The record's skill spelled as ``path``, its id and diff to match."""
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        record["target"]["canonical_path"] = str(path)
        record["target"]["skill_id"] = path.parent.name
        record["patch"]["unified_diff"] = make_patch(
            self.skill.read_text(encoding="utf-8"),
            record["patch"]["new_text"],
            path_label=str(path),
        ).unified_diff
        self.record_file.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def review(self) -> tuple[int, str]:
        return run(self.config, "review", self.cid)

    def apply(self) -> tuple[int, str]:
        return run(self.config, "apply", self.cid, "--approve", self.token, "--i-reviewed-this")

    def left_as_it_is(self) -> tuple[str, bytes, bytes, list[Path]]:
        """Where the link points, the skill's bytes, the record's bytes, the backups made."""
        return (
            os.readlink(self.root),
            (self.main / self.skill.parent.name / "SKILL.md").read_bytes(),
            self.record_file.read_bytes(),
            sorted(self.state.glob("backups/*")),
        )


class TheOutsideWordsTests(unittest.TestCase):
    """The refusal itself, reached through its module so on a tree without it only this errors."""

    ROOT = SkillRoot(id="canonical", path=Path("/skills-main"), canonical=True, editable=True)

    def test_the_outside_words_name_the_candidate_for_the_command_that_asked(self) -> None:
        cid = "CAND-2026-02-01-01-tdd"
        spelled = Path("/skills/tdd/SKILL.md")
        resolved = Path("/skills-main/tdd/SKILL.md")
        root = Path("/skills")
        for verb, command in (("review", "review"), ("replace", "apply")):
            with self.subTest(verb=verb):
                self.assertEqual(
                    skill_fs.spelled_outside_every_root(
                        spelled, resolved, self.ROOT, cid, verb, as_spelled=root
                    ),
                    f"refusing to {command} {cid}: its canonical target {spelled} is not spelled "
                    f"under any configured skill root, yet resolves to {resolved} inside root "
                    "canonical, so it reaches the skill through a symbolic link, and `apply` "
                    f"does not write through it. {how_to_close(cid, root)}",
                )

    def test_the_outside_words_give_no_root_to_configure_when_the_path_spells_none(self) -> None:
        cid = "CAND-2026-02-01-01-tdd"
        spelled = Path("/outside/x/SKILL.md")
        resolved = Path("/skills-main/tdd/SKILL.md")
        said = skill_fs.spelled_outside_every_root(spelled, resolved, self.ROOT, cid, "review")
        self.assertTrue(said.endswith(f"through it. {how_to_close(cid)}"), said)
        self.assertNotIn("configure it so", said)

    def test_without_a_candidate_the_outside_words_are_as_before(self) -> None:
        spelled = Path("/skills/tdd/SKILL.md")
        resolved = Path("/skills-main/tdd/SKILL.md")
        for verb in ("replace", "review"):
            with self.subTest(verb=verb):
                self.assertEqual(
                    skill_fs.spelled_outside_every_root(
                        spelled, resolved, self.ROOT, "", verb, as_spelled=Path("/skills")
                    ),
                    f"refusing to {verb} {spelled}: it is not spelled under any configured skill "
                    f"root yet resolves to {resolved} inside root canonical, so it reaches the "
                    "skill through a symbolic link. Spell the target as the root does. Nothing "
                    "was touched.",
                )

    def test_the_outside_words_name_a_path_with_a_break_on_their_line(self) -> None:
        root = Path("/skills\nconfig error: forged")
        spelled = root / "tdd" / "SKILL.md"
        resolved = Path("/skills-main\nconfig error: forged/tdd/SKILL.md")
        for cid in ("CAND-2026-02-01-01-tdd", ""):
            with self.subTest(cid=cid):
                said = skill_fs.spelled_outside_every_root(
                    spelled, resolved, self.ROOT, cid, "replace", as_spelled=root
                )
                self.assertNotIn("\n", said)
                self.assertIn(repr(str(spelled)), said)
                self.assertIn(repr(str(resolved)), said)
                if cid:
                    self.assertIn(f"configured as {repr(str(root))}, as", said)


class ReviewAndApplyPastAMovedRootTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.case = _Moved(Path(cls._tmp.name))
        cls.left = cls.case.left_as_it_is()
        cls.refused = {"review": cls.case.review(), "apply": cls.case.apply()}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_review_and_apply_past_a_moved_root_name_the_candidate_and_how_to_close_it(
        self,
    ) -> None:
        case = self.case
        resolved = case.main / case.skill.parent.name / "SKILL.md"
        for action, (rc, said) in self.refused.items():
            with self.subTest(action=action):
                self.assertEqual(rc, REFUSED, said)
                self.assertEqual(
                    said,
                    f"LinkedTargetRefused: refusing to {action} {case.cid}: its canonical target "
                    f"{case.skill} is not spelled under any configured skill root, yet resolves "
                    f"to {resolved} inside root canonical, so it reaches the skill through a "
                    f"symbolic link, and `apply` does not write through it. "
                    f"{how_to_close(case.cid, case.root)}\n",
                )

    def test_the_link_the_moved_skill_the_record_and_the_backups_are_as_the_refusal_says(
        self,
    ) -> None:
        self.assertEqual(self.case.left_as_it_is(), self.left)
        self.assertEqual(self.left[3], [])
        record = json.loads(self.case.record_file.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "validated")


class WhatTheOutsideRefusalSaysToDoTests(unittest.TestCase):
    def test_kept_the_candidate_past_a_moved_root_holds_the_queue_and_deleted_it_does_not(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _Moved(Path(tmp), max_open_candidates=1)
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

    def test_the_root_configured_as_the_path_spells_it_reviews_and_applies_with_the_token(
        self,
    ) -> None:
        # Only the config is changed back: the root is configured as the
        # candidate's path spells it, now a link to where it is, which the
        # check allows (a root may itself be a link).
        with tempfile.TemporaryDirectory() as tmp:
            case = _Moved(Path(tmp))
            self.assertEqual(case.review()[0], REFUSED)
            case.configure_as_the_path_spells_it()
            self.assertTrue(case.root.is_symlink())
            rc, said = case.review()
            self.assertEqual(rc, 0, said)
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            record = json.loads(case.record_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "applied")
            self.assertEqual(
                (case.main / case.skill.parent.name / "SKILL.md").read_text(encoding="utf-8"),
                record["patch"]["new_text"],
            )


class TheRootTheAdviceNamesTests(unittest.TestCase):
    """The spelling of the root the refusal names, and a path that gives none."""

    def test_a_path_through_an_alias_of_the_root_names_it_and_configured_so_is_taken(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _Moved(Path(tmp), moved=False)
            alias = case.root.with_name("skills-alias")
            alias.symlink_to(case.root, target_is_directory=True)
            case.respell(alias / case.skill.parent.name / "SKILL.md")
            rc, said = case.review()
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(said.endswith(f"through it. {how_to_close(case.cid, alias)}\n"), said)
            case.configure_root_as(alias)
            rc, shown = case.review()
            self.assertEqual(rc, 0, shown)
            found = re.search(r"--approve (\S+)", shown)
            assert found, shown
            rc, said = run(
                case.config, "apply", case.cid, "--approve", found.group(1), "--i-reviewed-this"
            )
            self.assertEqual(rc, 0, said)
            record = json.loads(case.record_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "applied")
            self.assertEqual(case.skill.read_text(encoding="utf-8"), record["patch"]["new_text"])

    def test_a_path_through_a_link_no_spelling_of_the_root_lies_ahead_of_names_none(
        self,
    ) -> None:
        # A folder outside every root linked to the skill's, under another name
        # and under its own; and the skill's folder linked under another name
        # inside the root, reached through an alias of the root.
        with tempfile.TemporaryDirectory() as tmp:
            case = _Moved(Path(tmp), moved=False)
            folder = case.skill.parent
            (case.base / "outside").mkdir()
            (case.base / "outside" / "x").symlink_to(folder, target_is_directory=True)
            (case.base / "elsewhere").mkdir()
            (case.base / "elsewhere" / folder.name).symlink_to(folder, target_is_directory=True)
            alias = case.root.with_name("skills-alias")
            alias.symlink_to(case.root, target_is_directory=True)
            for spelled in (
                case.base / "outside" / "x" / "SKILL.md",
                case.base / "elsewhere" / folder.name / "SKILL.md",
                alias / "y" / "SKILL.md",
            ):
                with self.subTest(spelled=spelled.parent.parent.name):
                    if spelled.parent.name == "y":
                        (case.root / "y").symlink_to(folder, target_is_directory=True)
                    case.respell(spelled)
                    for action, (rc, said) in (("review", case.review()), ("apply", case.apply())):
                        self.assertEqual(rc, REFUSED, said)
                        self.assertEqual(
                            said,
                            f"LinkedTargetRefused: refusing to {action} {case.cid}: its canonical "
                            f"target {spelled} is not spelled under any configured skill root, "
                            f"yet resolves to {case.skill} inside root canonical, so it reaches "
                            "the skill through a symbolic link, and `apply` does not write "
                            f"through it. {how_to_close(case.cid)}\n",
                        )

    def test_a_root_configured_ahead_of_such_a_link_or_at_it_is_refused_in_turn(self) -> None:
        # Why a path through such a link is given no root to configure: the
        # root configured as the folder ahead of the link meets the link below
        # it, and configured as the link itself holds the skill file at its
        # top, where discovery finds no skill.
        with tempfile.TemporaryDirectory() as tmp:
            case = _Moved(Path(tmp), moved=False)
            outside = case.base / "outside"
            outside.mkdir()
            (outside / "x").symlink_to(case.skill.parent, target_is_directory=True)
            case.respell(outside / "x" / "SKILL.md")
            for root, refusal in (
                (outside, "LinkedTargetRefused"),
                (outside / "x", "MirrorEditRefused"),
            ):
                with self.subTest(root=root.name):
                    case.configure_root_as(root)
                    rc, said = case.review()
                    self.assertEqual(rc, REFUSED, said)
                    self.assertTrue(said.startswith(f"{refusal}: refusing to review "), said)
            self.assertEqual(
                json.loads(case.record_file.read_text(encoding="utf-8"))["status"], "validated"
            )


class TheAdviceIsGivenOnlyWhereTheLaterJudgmentsTakeThePathTests(unittest.TestCase):
    """A spelling of the root is named only where, so configured, the root's path is taken."""

    def refused_with_no_root_named(self, case: _Moved, spelled: Path, root_id: str) -> None:
        rc, said = case.review()
        self.assertEqual(rc, REFUSED, said)
        self.assertEqual(
            said,
            f"LinkedTargetRefused: refusing to review {case.cid}: its canonical target "
            f"{spelled} is not spelled under any configured skill root, yet resolves to "
            f"{spelled.resolve()} inside root {root_id}, so it reaches the skill through a "
            f"symbolic link, and `apply` does not write through it. {how_to_close(case.cid)}\n",
        )

    def test_a_path_through_an_alias_to_a_mirror_copy_names_none_and_so_configured_is_refused(
        self,
    ) -> None:
        # The mirror root made editable after the candidate was validated: its
        # copy of the skill is still a mirror, the canonical root being marked.
        with tempfile.TemporaryDirectory() as tmp:
            case = _Moved(Path(tmp), moved=False)
            mirror = case.base / "env" / "skills-mirror"
            alias = mirror.with_name("mirror-alias")
            alias.symlink_to(mirror, target_is_directory=True)
            spelled = alias / case.skill.parent.name / "SKILL.md"
            case.respell(spelled)
            case.reconfigure(
                'kind = "mirror"\nprecedence = 90\ncanonical = false\neditable = false\n',
                'kind = "mirror"\nprecedence = 90\ncanonical = false\neditable = true\n',
            )
            self.refused_with_no_root_named(case, spelled, "mirror")
            case.reconfigure(f'path = "{mirror}"', f'path = "{alias}"')
            rc, said = case.review()
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                said.startswith(f"MirrorEditRefused: refusing to review {case.cid} at "), said
            )
            self.assertIn("it is a mirror copy", said)

    def test_a_path_through_an_alias_to_a_listed_skill_names_none_and_so_configured_is_refused(
        self,
    ) -> None:
        # The skill listed after the candidate was validated.
        with tempfile.TemporaryDirectory() as tmp:
            case = _Moved(Path(tmp), moved=False)
            alias = case.root.with_name("skills-alias")
            alias.symlink_to(case.root, target_is_directory=True)
            spelled = alias / case.skill.parent.name / "SKILL.md"
            case.respell(spelled)
            case.reconfigure(
                "protected_skill_ids = []", f'protected_skill_ids = ["{case.skill.parent.name}"]'
            )
            self.refused_with_no_root_named(case, spelled, "canonical")
            configured = case.base / "env" / case.root.name
            case.reconfigure(f'path = "{configured}"', f'path = "{alias}"')
            rc, said = case.review()
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                said.startswith(f"SelfMutationRefused: refusing to review {case.cid}"), said
            )
            self.assertIn("is listed in [guard].protected_skill_ids", said)

    def test_a_path_through_an_alias_in_another_case_names_none_where_the_filesystem_folds_it(
        self,
    ) -> None:
        # The skill's folder spelled in another case than discovery's, which
        # resolves to the folder where the filesystem folds case (macOS).
        with tempfile.TemporaryDirectory() as tmp:
            case = _Moved(Path(tmp), moved=False)
            folded = case.root / case.skill.parent.name.upper()
            if not (folded / "SKILL.md").exists():
                self.skipTest("this filesystem does not fold case")
            alias = case.root.with_name("skills-alias")
            alias.symlink_to(case.root, target_is_directory=True)
            spelled = alias / folded.name / "SKILL.md"
            case.respell(spelled)
            self.refused_with_no_root_named(case, spelled, "canonical")
            case.configure_root_as(alias)
            rc, said = case.review()
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                said.startswith(f"MirrorEditRefused: refusing to review {case.cid} at "), said
            )
            self.assertIn("it is not a skill document in any configured root", said)

    @unittest.skipIf(os.geteuid() == 0, "a directory of mode 000 is read by root")
    def test_a_path_through_an_alias_names_none_where_discovery_cannot_finish(self) -> None:
        # Another SKILL.md in the root, a link to a file in a directory this
        # process cannot read: the judgments' discovery raises PermissionError,
        # and the link is refused as it is, without a spelling.
        with tempfile.TemporaryDirectory() as tmp:
            case = _Moved(Path(tmp), moved=False)
            alias = case.root.with_name("skills-alias")
            alias.symlink_to(case.root, target_is_directory=True)
            spelled = alias / case.skill.parent.name / "SKILL.md"
            case.respell(spelled)
            hidden = case.base / "hidden"
            hidden.mkdir()
            (hidden / "notes").write_text("text", encoding="utf-8")
            (case.root / "bad").mkdir()
            (case.root / "bad" / "SKILL.md").symlink_to(hidden / "notes")
            hidden.chmod(0)
            try:
                self.refused_with_no_root_named(case, spelled, "canonical")
            finally:
                hidden.chmod(0o700)


class TheOutsideNeighbouringPathsTests(unittest.TestCase):
    def test_rollback_past_a_moved_root_keeps_its_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _Moved(Path(tmp), moved=False)
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            written = case.skill.read_bytes()
            case.move()
            rc, said = run(case.config, "rollback", case.cid)
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                said.startswith(
                    f"LinkedTargetRefused: refusing to replace {case.skill}: it is not spelled "
                    "under any configured skill root yet resolves to "
                ),
                said,
            )
            self.assertIn("Spell the target as the root does.", said)
            self.assertNotIn(case.cid, said)
            self.assertEqual(
                (case.main / case.skill.parent.name / "SKILL.md").read_bytes(), written
            )

    def test_a_root_re_pointed_after_applys_own_check_is_refused_by_the_write_as_before(
        self,
    ) -> None:
        # The write's own target check is not given the candidate: a root
        # configured as a link, re-pointed after `apply` judged the target and
        # before its write (the folder the record spells left as a link to the
        # new place), is refused in the old words, and the backup directory
        # `apply` made for the write is left behind, empty.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = build_env(base / "env", seed="c225")
            state = base / "env" / "state"
            root = base / "env" / "skills-canonical"
            first, second = root.with_name("skills-a"), root.with_name("skills-b")
            shutil.move(str(root), str(first))
            root.symlink_to(first, target_is_directory=True)
            rc, said = run(config, "daily", "--commit", "--day", DAY)
            self.assertEqual(rc, 0, said)
            (record_file,) = sorted(state.glob("candidates/*.json"))
            record = json.loads(record_file.read_text(encoding="utf-8"))
            cid = record["id"]
            skill = Path(record["target"]["canonical_path"])
            self.assertEqual(skill.parent.parent.name, first.name)
            rc, shown = run(config, "review", cid)
            self.assertEqual(rc, 0, shown)
            found = re.search(r"--approve (\S+)", shown)
            assert found, shown
            before = skill.read_bytes()
            real = skill_fs.FilesystemSkillStore.mirror_paths_of

            def re_point_then_list(store: Any, path: Path) -> tuple[str, ...]:
                if not second.exists():
                    shutil.move(str(first), str(second))
                    root.unlink()
                    root.symlink_to(second, target_is_directory=True)
                    first.symlink_to(second, target_is_directory=True)
                return real(store, path)

            with mock.patch.object(
                skill_fs.FilesystemSkillStore, "mirror_paths_of", re_point_then_list
            ):
                rc, said = run(
                    config, "apply", cid, "--approve", found.group(1), "--i-reviewed-this"
                )
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(
                said.startswith(
                    f"LinkedTargetRefused: refusing to replace {skill}: it is not spelled under "
                    "any configured skill root yet resolves to "
                ),
                said,
            )
            self.assertIn("Spell the target as the root does.", said)
            self.assertNotIn(cid, said)
            self.assertEqual((second / skill.parent.name / "SKILL.md").read_bytes(), before)
            record = json.loads(record_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "validated")
            (backup,) = sorted(state.glob("backups/*"))
            self.assertEqual(list(backup.iterdir()), [])

    def test_an_apply_finishing_an_earlier_write_is_recorded_past_a_moved_root(self) -> None:
        # The words say `apply` does not write through the link, not that the
        # candidate is never applied: an apply stopped after it wrote the skill
        # is finished by running it again, which records that write before the
        # target is judged.
        with tempfile.TemporaryDirectory() as tmp:
            case = _Moved(Path(tmp), moved=False)
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
            case.move()
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            record = json.loads(case.record_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "applied")
            self.assertEqual(
                (case.main / case.skill.parent.name / "SKILL.md").read_bytes(), written
            )
            receipt = case.state / "receipts" / f"apply--{case.cid}.json"
            self.assertIs(json.loads(receipt.read_text())["finishes_an_earlier_run"], True)


if __name__ == "__main__":
    unittest.main()
