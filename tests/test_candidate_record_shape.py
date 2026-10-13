"""A candidate record of the wrong shape is refused by name, never a traceback.

A candidate leaf is a file a hand can edit. ``Candidate.from_dict`` used to read
every nested field with ``.get`` straight off whatever the file held, and the
reader classified ``ValueError``, ``KeyError`` and ``TypeError`` only. Measured
before this suite: a record whose ``patch`` was a string, a list or null escaped
as a bare ``AttributeError`` and took ``candidates``, ``review``, ``apply`` and
``doctor`` down together with exit 70 and a traceback, where the contract is
that an unreadable record is listed as unreadable, by name, and the healthy
records beside it are still served. Found by mutating every field of a real
record to every other JSON type (2,445 variants); this file keeps the shapes
that escaped and the commands that fell.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agent_skill_improver.app.state import StateStore
from agent_skill_improver.cli import RC_OK, RC_REFUSED
from agent_skill_improver.domain.candidate import Candidate
from agent_skill_improver.errors import StateUnreadable
from tests.support import build_env, run_cli

DAY = "2026-03-03"


class CandidateRecordShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        base = Path(cls._tmp.name).resolve()
        cls.config_path = build_env(base / "env")
        proc = run_cli(["--config", str(cls.config_path), "daily", "--commit", "--day", DAY])
        assert proc.returncode == RC_OK, proc.stderr[-800:]
        leaves = sorted((base / "env").rglob("candidates/*.json"))
        assert len(leaves) == 1, f"the run must propose exactly one candidate, found {leaves}"
        cls.leaf = leaves[0]
        cls.pristine = cls.leaf.read_text(encoding="utf-8")
        cls.record = json.loads(cls.pristine)
        cls.candidate_id = cls.record["id"]
        cls.state = StateStore(cls.leaf.parent.parent)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self.addCleanup(self.leaf.write_text, self.pristine, encoding="utf-8")

    def _spoil(self, **fields) -> None:
        record = json.loads(self.pristine)
        record.update(fields)
        self.leaf.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def _cli(self, *args: str):
        return run_cli(["--config", str(self.config_path), *args])

    # -- the reader ----------------------------------------------------------

    def test_the_control_shows_the_untouched_record_reads_and_lists(self):
        self.assertEqual(self.state.read_candidate(self.candidate_id).id, self.candidate_id)
        proc = self._cli("candidates")
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertIn("1 open candidate(s); 0 unreadable record(s)", proc.stdout)

    def test_a_patch_that_is_not_an_object_is_a_named_refusal(self):
        for value in (None, "x", [], 0, 1.5, True):
            with self.subTest(value=value):
                self._spoil(patch=value)
                with self.assertRaises(StateUnreadable) as ctx:
                    self.state.read_candidate(self.candidate_id)
                message = str(ctx.exception)
                self.assertIn(str(self.leaf), message)
                self.assertIn(f"patch must be an object, got {type(value).__name__}", message)
                self.assertNotIn("AttributeError", message)

    def test_every_nested_object_is_checked_before_it_is_read(self):
        shapes = {
            "target": ("target must be an object", "x"),
            "sources": ("sources must be an array", "x"),
            "sources[0]": ("sources[0] must be an object", ["x"]),
            "rubric_failures": ("rubric_failures must be an array", {"a": 1}),
            "rubric_failures[1]": ("rubric_failures[1] must be an object", [{}, 7]),
            "validation": ("validation must be an object", "x"),
            "validation.gate": ("validation.gate must be an object", {"gate": "x"}),
        }
        for label, (expected, value) in shapes.items():
            with self.subTest(field=label):
                key = label.split("[")[0].split(".")[0]
                self._spoil(**{key: value})
                with self.assertRaises(StateUnreadable) as ctx:
                    self.state.read_candidate(self.candidate_id)
                self.assertIn(expected, str(ctx.exception))

    def test_from_dict_refuses_the_shapes_as_a_value_error_of_its_own(self):
        """The domain parser names the field itself; the reader only wraps it."""
        for field, value in (("patch", None), ("target", []), ("validation", 3)):
            with self.subTest(field=field):
                record = json.loads(self.pristine)
                record[field] = value
                with self.assertRaises(ValueError) as ctx:
                    Candidate.from_dict(record)
                self.assertIn(f"{field} must be an object", str(ctx.exception))

    def test_a_shape_error_the_parser_does_not_yet_name_is_still_one_refusal(self):
        """Belt and braces: an ``AttributeError`` from a field this check does not
        cover is wrapped by the reader rather than escaping as exit 70."""
        with mock.patch.object(Candidate, "from_dict", side_effect=AttributeError("x.get")):
            with self.assertRaises(StateUnreadable) as ctx:
                self.state.read_candidate(self.candidate_id)
        self.assertIn("AttributeError: x.get", str(ctx.exception))

    # -- the commands --------------------------------------------------------

    def test_the_listing_exits_zero_and_names_the_record_as_unreadable(self):
        self._spoil(patch="x")
        proc = self._cli("candidates")
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("0 open candidate(s); 1 unreadable record(s)", proc.stdout)
        self.assertIn("patch must be an object, got str", proc.stdout)
        as_json = self._cli("candidates", "--json")
        self.assertEqual(as_json.returncode, RC_OK, as_json.stderr)
        payload = json.loads(as_json.stdout)
        self.assertEqual(len(payload["unreadable"]), 1, payload)

    def test_review_and_apply_refuse_by_name(self):
        self._spoil(patch=[])
        for args in (["review", self.candidate_id], ["apply", self.candidate_id, "--approve", "t"]):
            with self.subTest(command=args[0]):
                proc = self._cli(*args)
                self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn("StateUnreadable", proc.stderr)
                self.assertIn("patch must be an object, got list", proc.stderr)

    def test_doctor_reports_the_record_as_a_problem_instead_of_falling(self):
        self._spoil(patch=None)
        proc = self._cli("doctor")
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        problems = [line for line in proc.stdout.splitlines() if line.startswith("PROBLEM")]
        self.assertTrue(
            any("patch must be an object, got NoneType" in line for line in problems), problems
        )

    def test_a_null_gate_lists_the_candidate_as_gate_fail_rather_than_falling(self):
        """A never-gated record carries ``validation`` as null and a gated one a
        ``gate`` object, so a null gate is reachable only by hand-edit; the
        listing still used to call ``.get`` on it and fall with exit 70."""
        record = json.loads(self.pristine)
        record["validation"]["gate"] = None
        self.leaf.write_text(json.dumps(record) + "\n", encoding="utf-8")
        proc = self._cli("candidates")
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertIn("gate=fail", proc.stdout)
        as_json = self._cli("candidates", "--json")
        self.assertEqual(as_json.returncode, RC_OK, as_json.stderr)
        self.assertFalse(json.loads(as_json.stdout)["candidates"][0]["gate_passed"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
