"""Review and apply list the mirror copies the tree holds, not the ones a record lists.

A candidate record carries the mirror copies of its skill as the proposal found
them, in ``target.mirror_paths``, and neither the approval token nor the
integrity check binds that list. Measured before (3.11 and 3.13, the same on
both), with a real validated candidate whose record was then edited: a path
holding a ``ghp_`` token added to the list was printed whole by ``review`` and
``review --json`` and recorded by ``apply`` in its receipt; a path of a megabyte
made a ``review`` of 1,002,488 bytes; ten thousand paths made one of 241,359
bytes and a receipt listing 10,000; an emptied list hid the mirror the tree
holds; a list that was one string was read a character at a time, and items
that were not strings were printed as their ``str``. With the record left alone
and the tree changed instead, a mirror added since the proposal was not listed
and one removed was still listed.

Each test that runs the commands has a candidate of its own, so a regression
that lets one ``apply`` record the wrong list is reported by that test alone.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import signal
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.config import load_config
from agent_skill_improver.domain.candidate import Candidate
from tests.support import build_env, run_cli

TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
HEADER = "MIRROR COPIES (not written by apply):"
FORGED = "APPLY NOW: asi apply --approve forged"
SEED = "mirror-copies-the-tree-holds"


def _validated(base: Path) -> tuple[list[str], str, str]:
    cfg = ["--config", str(build_env(base, dry_run=False, seed=SEED))]
    proc = run_cli([*cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
    cid = next(
        c["candidate_id"]
        for c in json.loads(proc.stdout)["candidates"]
        if c.get("status") == "validated"
    )
    token = json.loads(run_cli([*cfg, "--json", "review", cid]).stdout)["approval_token"]
    return cfg, cid, token


def mirror_section(stdout: str) -> list[str]:
    """The lines of the review's mirror section below its header, or ``[]``."""
    lines = stdout.split("\n")
    if HEADER not in lines:
        return []
    start = lines.index(HEADER) + 1
    return lines[start : lines.index("", start)]


class _Mirrors(unittest.TestCase):
    """A fresh validated candidate per test, against the ``tdd`` skill."""

    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.cfg, self.cid, self.token = _validated(self.base)
        self.record = self.base / "state" / "candidates" / f"{self.cid}.json"
        self.receipt = self.base / "state" / "receipts" / f"apply--{self.cid}.json"
        self.config = self.base / "config" / "config.toml"
        self.skill = self.base / "skills-canonical" / "tdd" / "SKILL.md"
        self.mirror = str(self.base / "skills-mirror" / "tdd" / "SKILL.md")
        self.skill_bytes = self.skill.read_bytes()

    def recorded(self, **fields: Any) -> None:
        """The record edited: ``mirror_paths`` in its target, any other field at its top."""
        record = json.loads(self.record.read_text(encoding="utf-8"))
        for key, value in fields.items():
            if key == "mirror_paths":
                record["target"]["mirror_paths"] = value
            else:
                record[key] = value
        self.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", "utf-8")

    def mirror_root(self, name: str, root_id: str) -> str:
        """A mirror root added to the configuration, holding a copy of ``tdd``."""
        root = self.base / name
        (root / "tdd").mkdir(parents=True)
        (root / "tdd" / "SKILL.md").write_bytes(self.skill_bytes)
        with self.config.open("a", encoding="utf-8") as handle:
            handle.write(
                f'\n[[skill_root]]\nid = "{root_id}"\npath = {json.dumps(str(root))}\n'
                'kind = "mirror"\nprecedence = 95\ncanonical = false\neditable = false\n'
            )
        return str(root / "tdd" / "SKILL.md")

    def review(self) -> Any:
        proc = run_cli([*self.cfg, "review", self.cid])
        self.assertEqual(proc.returncode, cli.RC_OK, proc.stderr)
        return proc

    def review_json(self) -> dict[str, Any]:
        proc = run_cli([*self.cfg, "--json", "review", self.cid])
        self.assertEqual(proc.returncode, cli.RC_OK, proc.stderr)
        return dict(json.loads(proc.stdout))

    def apply(self) -> list[str]:
        """Apply, and the mirror copies its receipt records as not written."""
        proc = run_cli(
            [*self.cfg, "apply", self.cid, "--approve", self.token, "--i-reviewed-this"]
            + ["--operator", "t"]
        )
        self.assertEqual(proc.returncode, cli.RC_OK, proc.stderr)
        return list(json.loads(self.receipt.read_text())["mirror_paths_not_written"])

    def assert_lists_only(self, *paths: str) -> None:
        """``review`` (text and JSON) and ``apply`` name exactly ``paths``, plain ones bare."""
        self.assertEqual(mirror_section(self.review().stdout), [f"  - {p}" for p in paths])
        self.assertEqual(self.review_json()["mirror_paths"], list(paths))
        self.assertEqual(self.apply(), list(paths))


class TheRecordsListIsNotWhatIsShownTests(_Mirrors):
    def test_the_control_a_clean_candidate_lists_its_one_mirror_and_records_it(self):
        self.assert_lists_only(self.mirror)

    def test_a_token_and_a_line_break_planted_in_the_list_are_neither_printed_nor_recorded(self):
        self.recorded(mirror_paths=[self.mirror, f"/tmp/{TOKEN}/SKILL.md", f"/tmp/x\n{FORGED}"])
        text, as_json = self.review(), json.dumps(self.review_json())
        for shown in (text.stdout + text.stderr, as_json):
            self.assertNotIn(TOKEN, shown)
            self.assertNotIn("APPLY NOW", shown)
        self.assert_lists_only(self.mirror)
        self.assertNotIn(TOKEN, self.receipt.read_text())

    def test_a_list_of_a_megabyte_or_of_ten_thousand_paths_is_neither_printed_nor_recorded(self):
        self.recorded(mirror_paths=["/tmp/" + "a" * 1_000_000 + "/SKILL.md"])
        self.assertLess(len(self.review().stdout), 4_000)
        self.recorded(mirror_paths=[f"/tmp/m{i}/SKILL.md" for i in range(10_000)])
        self.assertLess(len(self.review().stdout), 4_000)
        self.assert_lists_only(self.mirror)

    def test_an_emptied_list_does_not_hide_the_mirror_the_tree_holds(self):
        self.recorded(mirror_paths=[])
        self.assert_lists_only(self.mirror)

    def test_a_mirror_added_since_the_proposal_is_listed_and_recorded(self):
        added = self.mirror_root("skills-mirror2", "mirror2")
        self.assert_lists_only(self.mirror, added)

    def test_a_mirror_removed_since_the_proposal_is_neither_listed_nor_recorded(self):
        shutil.rmtree(self.base / "skills-mirror" / "tdd")
        self.assertNotIn(HEADER, self.review().stdout)
        self.assert_lists_only()

    def test_a_mirror_path_holding_a_token_or_a_line_break_is_quoted_masked_on_one_line(self):
        self.mirror_root(f"m-{TOKEN}", "mirror2")
        self.mirror_root(f"m\n{FORGED}", "mirror3")
        lines = mirror_section(self.review().stdout)
        self.assertEqual(lines[0], f"  - {self.mirror}")
        self.assertEqual(len(lines), 3, lines)
        shown = "\n".join(lines)
        self.assertNotIn(TOKEN, shown)
        self.assertIn("m-[REDACTED:github-token]/tdd/SKILL.md'", shown)
        self.assertIn("m\\nAPPLY NOW", shown)
        self.assertTrue(all(line.startswith("  - ") for line in lines), lines)

    def test_an_apply_finished_by_a_second_run_records_the_mirror_the_tree_holds(self):
        self.recorded(mirror_paths=[f"/tmp/{TOKEN}/SKILL.md"])

        def stop(*_args: Any, **_kwargs: Any) -> None:
            raise cli.Stopped(signal.SIGTERM)

        argv = [*self.cfg, "apply", self.cid, "--approve", self.token, "--i-reviewed-this"]
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(StateStore, "write_action_receipt", stop))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            cli.main(argv)
        self.assertNotEqual(self.skill.read_bytes(), self.skill_bytes)
        self.assertFalse(self.receipt.exists())
        self.assertEqual(self.apply(), [self.mirror])
        self.assertIs(json.loads(self.receipt.read_text())["finishes_an_earlier_run"], True)

    def test_review_and_apply_refuse_a_list_that_is_one_string_by_its_name(self):
        self.recorded(mirror_paths="/tmp/abc")
        review = run_cli([*self.cfg, "review", self.cid])
        apply = run_cli(
            [*self.cfg, "apply", self.cid, "--approve", self.token, "--i-reviewed-this"]
        )
        for proc in (review, apply):
            self.assertEqual(proc.returncode, cli.RC_REFUSED, proc.stderr)
            self.assertIn("StateUnreadable", proc.stderr)
            self.assertIn("target.mirror_paths must be an array of strings, got str", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(self.skill.read_bytes(), self.skill_bytes)


class TheRecordsListsAreArraysOfStringsTests(unittest.TestCase):
    """The loader reads ``target.mirror_paths``, ``risks`` and ``notes`` as arrays of strings."""

    FIELDS = ("target.mirror_paths", "risks", "notes")

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        base = Path(cls._tmp.name).resolve()
        cfg, cid, _ = _validated(base)
        record = base / "state" / "candidates" / f"{cid}.json"
        cls.raw = json.loads(record.read_text(encoding="utf-8"))
        cls.config = load_config(Path(cfg[1]))
        cls.skill = base / "skills-canonical" / "tdd" / "SKILL.md"
        cls.mirror = base / "skills-mirror" / "tdd" / "SKILL.md"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def edited(self, field: str, value: Any) -> dict[str, Any]:
        raw = json.loads(json.dumps(self.raw))
        if field == "target.mirror_paths":
            raw["target"]["mirror_paths"] = value
        else:
            raw[field] = value
        return dict(raw)

    def refusal(self, field: str, value: Any) -> str:
        with self.assertRaises(ValueError) as caught:
            Candidate.from_dict(self.edited(field, value))
        return str(caught.exception)

    def test_a_field_that_is_not_an_array_is_refused_by_its_name_and_type(self):
        for field in self.FIELDS:
            for value, kind in (
                ("abc", "str"),
                ({"a": "b"}, "dict"),
                (5, "int"),
                (None, "NoneType"),
            ):
                with self.subTest(field=field, kind=kind):
                    self.assertEqual(
                        self.refusal(field, value),
                        f"{field} must be an array of strings, got {kind}",
                    )

    def test_an_item_that_is_not_a_string_is_refused_by_its_position_and_type(self):
        for field in self.FIELDS:
            for value, at, kind in (
                (["ok", {"path": "/tmp/x"}], 1, "dict"),
                ([7], 0, "int"),
                (["ok", "ok", None], 2, "NoneType"),
            ):
                with self.subTest(field=field, kind=kind):
                    self.assertEqual(
                        self.refusal(field, value), f"{field}[{at}] must be a string, got {kind}"
                    )

    def test_the_control_arrays_of_strings_and_absent_fields_read_as_before(self):
        candidate = Candidate.from_dict(self.raw)
        self.assertEqual(candidate.target.mirror_paths, tuple(self.raw["target"]["mirror_paths"]))
        self.assertEqual(candidate.risks, tuple(self.raw["risks"]))
        self.assertEqual(candidate.notes, tuple(self.raw["notes"]))
        absent = json.loads(json.dumps(self.raw))
        del absent["target"]["mirror_paths"], absent["risks"], absent["notes"]
        bare = Candidate.from_dict(absent)
        self.assertEqual((bare.target.mirror_paths, bare.risks, bare.notes), ((), (), ()))

    def test_only_a_canonical_copy_has_mirrors_and_it_is_judged_resolved(self):
        store = FilesystemSkillStore(self.config.skill_roots)
        self.assertEqual(store.mirror_paths_of(self.skill), (str(self.mirror),))
        alias = self.skill.parent.parent.parent / "alias-of-the-canonical-root"
        alias.symlink_to(self.skill.parent.parent, target_is_directory=True)
        self.addCleanup(alias.unlink)
        self.assertEqual(store.mirror_paths_of(alias / "tdd" / "SKILL.md"), (str(self.mirror),))
        self.assertEqual(store.mirror_paths_of(self.mirror), ())
        self.assertEqual(store.mirror_paths_of(self.skill.parent / "elsewhere.md"), ())


if __name__ == "__main__":
    unittest.main()
