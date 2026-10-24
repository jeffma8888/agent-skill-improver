"""The candidate loader quotes the schema it refuses and names a number it cannot read.

Measured before (3.11 and 3.13, the same on both), with a real validated
candidate whose record was then edited: a ``schema`` holding a ``ghp_`` token
and a line break was printed with the token whole by ``review``, ``apply``,
``candidates`` and ``doctor``, text and JSON (the line break spelled by
``repr``); a ``schema`` of a megabyte made each of them a megabyte or more
(``doctor --json`` 2,005,641 bytes); and a token in ``patch.added_lines``, a
source's ``position`` or a rubric failure's ``rubric_version`` was printed whole
by all six, in the text ``int()`` quotes when it cannot read a number.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_skill_improver import cli
from agent_skill_improver.domain.candidate import Candidate
from tests.support import build_env, run_cli

TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
FORGED = "config error: forged"
SEED = "loader-names-a-field"


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


class TheLoaderQuotesWhatItRefusesTests(unittest.TestCase):
    """``Candidate.from_dict`` on a real record, one field edited per case."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        base = Path(cls._tmp.name).resolve()
        _, cid, _ = _validated(base)
        record = base / "state" / "candidates" / f"{cid}.json"
        cls.raw = json.loads(record.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def edited(self, *path: Any, value: Any) -> dict[str, Any]:
        raw = json.loads(json.dumps(self.raw))
        node = raw
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return dict(raw)

    def refusal(self, *path: Any, value: Any) -> str:
        with self.assertRaises(ValueError) as caught:
            Candidate.from_dict(self.edited(*path, value=value))
        return str(caught.exception)

    def test_a_schema_holding_a_token_and_a_line_break_is_quoted_masked_on_one_line(self):
        message = self.refusal("schema", value=f"{TOKEN}\n{FORGED}")
        self.assertNotIn(TOKEN, message)
        self.assertNotIn("\n", message)
        self.assertEqual(
            message, "unexpected candidate schema '[REDACTED:github-token]\\nconfig error: forged'"
        )

    def test_a_schema_of_a_megabyte_is_cut(self):
        message = self.refusal("schema", value="s" * 1_000_000)
        self.assertLess(len(message), 300)
        self.assertIn("more chars not quoted]", message)

    def test_a_schema_that_is_not_a_string_is_quoted_masked(self):
        message = self.refusal("schema", value={"k": TOKEN})
        self.assertEqual(message, "unexpected candidate schema {'k': '[REDACTED:github-token]'}")

    def test_a_plain_wrong_schema_reads_as_before(self):
        self.assertEqual(
            self.refusal("schema", value="asi.candidate.v0"),
            "unexpected candidate schema 'asi.candidate.v0'",
        )

    def test_each_number_field_it_cannot_read_is_named_by_field_and_type(self):
        fields = (
            (("patch", "added_lines"), "patch.added_lines"),
            (("patch", "removed_lines"), "patch.removed_lines"),
            (("rubric_failures", 0, "rubric_version"), "rubric_failures[0].rubric_version"),
            (("sources", 0, "position"), "sources[0].position"),
        )
        for path, name in fields:
            for value, kind in ((f"{TOKEN}\n{FORGED}", "str"), ([1], "list"), (None, "NoneType")):
                with self.subTest(field=name, kind=kind):
                    message = self.refusal(*path, value=value)
                    self.assertEqual(message, f"{name} must be an integer, got {kind}")

    def test_a_later_entry_is_named_by_its_own_index(self):
        raw = json.loads(json.dumps(self.raw))
        raw["sources"] = raw["sources"] + [dict(raw["sources"][0], position=TOKEN)]
        raw["rubric_failures"] = raw["rubric_failures"] + [
            dict(raw["rubric_failures"][0], rubric_version=TOKEN)
        ]
        last_failure = len(raw["rubric_failures"]) - 1
        with self.assertRaises(ValueError) as caught:
            Candidate.from_dict(raw)
        self.assertEqual(
            str(caught.exception),
            f"rubric_failures[{last_failure}].rubric_version must be an integer, got str",
        )
        raw["rubric_failures"] = raw["rubric_failures"][:-1]
        with self.assertRaises(ValueError) as caught:
            Candidate.from_dict(raw)
        self.assertEqual(
            str(caught.exception),
            f"sources[{len(raw['sources']) - 1}].position must be an integer, got str",
        )

    def test_what_int_read_before_it_still_reads(self):
        candidate = Candidate.from_dict(self.edited("patch", "added_lines", value="7"))
        self.assertEqual(candidate.patch.added_lines, 7)
        candidate = Candidate.from_dict(self.edited("patch", "removed_lines", value="4"))
        self.assertEqual(candidate.patch.removed_lines, 4)
        candidate = Candidate.from_dict(
            self.edited("rubric_failures", 0, "rubric_version", value="2")
        )
        self.assertEqual(candidate.rubric_failures[0].rubric_version, 2)
        candidate = Candidate.from_dict(self.edited("sources", 0, "position", value="3"))
        self.assertEqual(candidate.sources[0].position, 3)
        clean = Candidate.from_dict(self.raw)
        self.assertEqual(clean.patch.added_lines, self.raw["patch"]["added_lines"])
        self.assertEqual(clean.patch.removed_lines, self.raw["patch"]["removed_lines"])

    def test_an_absent_count_or_version_reads_as_its_default(self):
        raw = json.loads(json.dumps(self.raw))
        del raw["patch"]["added_lines"], raw["patch"]["removed_lines"]
        del raw["rubric_failures"][0]["rubric_version"]
        candidate = Candidate.from_dict(raw)
        self.assertEqual((candidate.patch.added_lines, candidate.patch.removed_lines), (0, 0))
        self.assertEqual(candidate.rubric_failures[0].rubric_version, 1)


class EveryCommandThatReadsTheRecordNamesTheFieldTests(unittest.TestCase):
    """``review``, ``apply``, ``candidates`` and ``doctor``, each with a record of its own."""

    COMMANDS = (
        ("review",),
        ("apply",),
        ("candidates",),
        ("--json", "candidates"),
        ("doctor",),
        ("--json", "doctor"),
    )

    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.cfg, self.cid, self.token = _validated(self.base)
        self.record = self.base / "state" / "candidates" / f"{self.cid}.json"
        self.skill = self.base / "skills-canonical" / "tdd" / "SKILL.md"
        self.skill_bytes = self.skill.read_bytes()

    def edit(self, *path: str, value: Any) -> None:
        record = json.loads(self.record.read_text(encoding="utf-8"))
        node = record
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        self.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", "utf-8")

    def run_all(self) -> dict[str, Any]:
        runs = {}
        for command in self.COMMANDS:
            argv = list(command)
            if argv[-1] in ("review", "apply"):
                argv.append(self.cid)
            if argv[-2:] == ["apply", self.cid]:
                argv += ["--approve", self.token, "--i-reviewed-this"]
            runs[" ".join(command)] = run_cli([*self.cfg, *argv])
        return runs

    def assert_named_without_the_token(self, words: str) -> None:
        runs = self.run_all()
        for name, proc in runs.items():
            with self.subTest(command=name):
                out = proc.stdout + proc.stderr
                self.assertNotIn(TOKEN, out)
                self.assertNotIn("Traceback", out)
                self.assertFalse(any(line.startswith(FORGED) for line in out.split("\n")), out)
                self.assertIn(words, out)
        for name in ("review", "apply"):
            self.assertEqual(runs[name].returncode, cli.RC_REFUSED, runs[name].stderr)
            self.assertIn("StateUnreadable", runs[name].stderr)
        self.assertEqual(self.skill.read_bytes(), self.skill_bytes)

    def test_a_number_field_holding_a_token_is_named_by_every_command_without_it(self):
        self.edit("patch", "added_lines", value=f"{TOKEN}\n{FORGED}")
        self.assert_named_without_the_token("patch.added_lines must be an integer, got str")

    def test_a_schema_holding_a_token_is_quoted_masked_by_every_command(self):
        self.edit("schema", value=f"{TOKEN}\n{FORGED}")
        self.assert_named_without_the_token("unexpected candidate schema")

    def test_the_control_the_unedited_record_is_reviewed_and_listed(self):
        runs = self.run_all()
        self.assertEqual(runs["review"].returncode, cli.RC_OK, runs["review"].stderr)
        self.assertEqual(runs["apply"].returncode, cli.RC_OK, runs["apply"].stderr)
        self.assertIn("1 closed not listed", runs["candidates"].stdout)
        self.assertIn("0 unreadable record(s)", runs["candidates"].stdout)


if __name__ == "__main__":
    unittest.main()
