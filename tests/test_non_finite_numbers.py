"""A number no double can hold is an unreadable record, not an aborted run.

``json.loads`` accepts the non-standard tokens ``NaN``, ``Infinity`` and
``-Infinity`` by default, and parses a legal literal past the double range
(``1e400``) to ``inf`` without a word. Measured before the guard in
``loads_untrusted``: one fixture export whose element ``position`` was ``1e400``
reached ``int()`` inside the normalizer and aborted a committed ``daily`` run as
an internal error (exit 70, a traceback, no receipt), with the healthy export
beside it never processed. Every untrusted parse now holds numbers to the same
standard as nesting: finite, or the text is a ``ValueError`` the caller already
classifies (passed over, quarantined, malformed, unreadable).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.adapters import transcript_export
from agent_skill_improver.adapters.model_command import extract_json_object
from agent_skill_improver.adapters.transcript_export import normalize_export
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.errors import MalformedModelOutput, StateUnreadable
from agent_skill_improver.jsonio import dumps, loads_untrusted
from tests.support import FIXTURE_EXPORTS, build_env, run_cli

HEALTHY_FILE = "failing"
BAD = "conv-bad-number"
DAY = "2026-03-01"


class UntrustedNumberBoundaryTests(unittest.TestCase):
    """``loads_untrusted`` is the one parse every untrusted text goes through."""

    def test_the_non_standard_tokens_are_a_value_error(self):
        for text in ("NaN", "Infinity", "-Infinity", '{"confidence": NaN}', "[1, [Infinity]]"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as ctx:
                    loads_untrusted(text)
                self.assertIn("not a JSON number", str(ctx.exception))

    def test_a_literal_past_the_double_range_is_a_value_error(self):
        for text, token in (("1e400", "1e400"), ("-1e400", "-1e400"), ('{"a": [1e999]}', "1e999")):
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as ctx:
                    loads_untrusted(text)
                self.assertIn(token, str(ctx.exception))
                self.assertIn("does not round to a finite double", str(ctx.exception))

    def test_the_control_shows_ordinary_numbers_still_read_as_before(self):
        cases = {
            "1e308": 1e308,
            "0.5": 0.5,
            "-2": -2,
            "1.5e-300": 1.5e-300,
            "1" + "0" * 30: 10**30,
            '{"a": [1, 2.25, -0.0]}': {"a": [1, 2.25, -0.0]},
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                value = loads_untrusted(text)
                self.assertEqual(value, expected)
                self.assertEqual(value, json.loads(text))

    def test_a_refused_number_is_a_plain_value_error_like_deep_nesting(self):
        """So every caller's existing ``except ValueError`` classifies it."""
        for text in ("NaN", "1e400", "[" * 100_000 + "]" * 100_000):
            with self.subTest(text=text[:10]):
                try:
                    loads_untrusted(text)
                except ValueError as exc:
                    self.assertNotIsInstance(exc, RecursionError)
                    self.assertNotIsInstance(exc, OverflowError)
                else:
                    self.fail("accepted")


class WriterMirrorsTheReaderTests(unittest.TestCase):
    """``dumps`` refuses what ``loads_untrusted`` could never read back."""

    def test_the_writer_refuses_a_non_finite_float_rather_than_emitting_a_token_no_reader_takes(
        self,
    ):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    dumps({"confidence": value})
                with self.assertRaises(ValueError):
                    dumps([1, {"deep": [value]}])

    def test_the_control_shows_the_writer_still_writes_ordinary_floats(self):
        text = dumps({"ratio": 0.25, "big": 1e308, "zero": -0.0, "n": 3})
        self.assertEqual(loads_untrusted(text), {"ratio": 0.25, "big": 1e308, "zero": -0.0, "n": 3})


class NormalizerToleratesNonFiniteNumbersTests(unittest.TestCase):
    """``normalize_export`` is public and takes any payload; a non-finite float
    handed to it directly is a surprising element, not a reason to raise."""

    def _export(self, position):
        return {
            "thread": {"id": "t", "createdAt": position, "updatedAt": math.nan},
            "elements": [
                {"id": "m", "position": position, "kind": "message", "role": "user"},
                {"id": "n", "position": 2, "kind": "message", "role": "assistant"},
                {"id": "x", "position": 1, "kind": "text", "parentId": "m", "content": "hi"},
                {
                    "id": "tool",
                    "position": 3,
                    "kind": "tool",
                    "parentId": "n",
                    "name": "run",
                    "status": "done",
                    "durationMs": position,
                },
            ],
        }

    def test_a_non_finite_position_sorts_as_the_default_rather_than_raising(self):
        for position in (math.inf, -math.inf, math.nan):
            with self.subTest(position=position):
                conversation = normalize_export(self._export(position))
                self.assertEqual([m.element_id for m in conversation.messages], ["m", "n"])
                self.assertEqual(conversation.messages[0].position, 0)

    def test_a_non_finite_optional_int_is_absent_rather_than_raising(self):
        for value in (math.inf, math.nan):
            with self.subTest(value=value):
                self.assertIsNone(transcript_export._opt_int(value))
                self.assertEqual(transcript_export._as_int(value, 7), 7)
        self.assertEqual(transcript_export._opt_int(3.0), 3)
        self.assertEqual(transcript_export._as_int(3.9, 7), 3)


class OneNonFiniteExportDoesNotAbortTheRunTests(unittest.TestCase):
    """The measured failure, end to end through the CLI."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        base = Path(cls._tmp.name).resolve()
        fixtures = base / "exports"
        fixtures.mkdir()
        healthy = (FIXTURE_EXPORTS / f"{HEALTHY_FILE}.json").read_text(encoding="utf-8")
        (fixtures / f"{HEALTHY_FILE}.json").write_text(healthy, encoding="utf-8")
        healthy_id = json.loads(healthy)["thread"]["id"]
        needle = f'"id": "{healthy_id}"'
        assert healthy.count(needle) == 1, "the thread's own id must be the only such string"
        bad = healthy.replace(needle, f'"id": "{BAD}"', 1)
        bad, replaced = re.subn(r'"position": \d+', '"position": 1e400', bad, count=1)
        assert replaced == 1, "the healthy fixture carries a position to spoil"
        (fixtures / f"{BAD}.json").write_text(bad, encoding="utf-8")
        cls.config_path = build_env(base / "env", fixtures_dir=fixtures)
        cls.proc = run_cli(["--config", str(cls.config_path), "daily", "--commit", "--day", DAY])
        receipts = sorted((base / "env").rglob("receipts/*.json"))
        assert len(receipts) == 1, f"one committed run leaves one receipt, found {receipts}"
        cls.receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
        candidates = next((base / "env").rglob("candidates"), None)
        assert candidates is not None, "the run proposed no candidate"
        cls.state_root = candidates.parent

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_run_completes_and_scores_the_healthy_record(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-800:])
        self.assertNotIn("Traceback", self.proc.stderr)
        counts = self.receipt["counts"]
        self.assertEqual(counts["conversations_ingested"], 1, counts)
        self.assertGreater(counts["records_scored"], 0, counts)

    def test_the_spoiled_export_is_named_in_the_receipt_with_its_number(self):
        passed_over = self.receipt["discovery"]["passed_over"]
        self.assertEqual(len(passed_over), 1, passed_over)
        self.assertTrue(passed_over[0]["entry"].endswith(f"{BAD}.json"), passed_over)
        self.assertIn("1e400", passed_over[0]["reason"])
        self.assertIn("does not round to a finite double", passed_over[0]["reason"])
        self.assertTrue(
            any("1e400" in note for note in self.receipt["notes"]), self.receipt["notes"]
        )

    def test_a_model_reply_with_a_non_finite_confidence_is_malformed_not_a_crash(self):
        for reply in ('{"confidence": NaN}', 'Sure:\n{"confidence": Infinity, "verdict": "PASS"}'):
            with self.subTest(reply=reply):
                with self.assertRaises(MalformedModelOutput):
                    extract_json_object(reply)
        self.assertEqual(extract_json_object('{"confidence": 0.5}'), {"confidence": 0.5})

    def test_a_state_leaf_with_a_non_finite_number_is_a_named_refusal(self):
        state = StateStore(self.state_root)
        leaves = sorted((self.state_root / "candidates").glob("*.json"))
        self.assertTrue(leaves, "the run needs to have proposed a candidate")
        leaf = leaves[0]
        raw = json.loads(leaf.read_text(encoding="utf-8"))
        candidate_id = raw["id"]
        self.assertEqual(state.read_candidate(candidate_id).id, candidate_id)
        spoiled = leaf.read_text(encoding="utf-8").replace("{\n", '{\n  "zz_planted": NaN,\n', 1)
        leaf.write_text(spoiled, encoding="utf-8")
        with self.assertRaises(StateUnreadable) as ctx:
            state.read_candidate(candidate_id)
        self.assertIn("NaN", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
