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
from agent_skill_improver.config import load_config
from agent_skill_improver.errors import ConfigError
from agent_skill_improver.jsonio import MAX_STORED_INT, dumps, loads_untrusted
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

    def test_a_model_reply_with_an_integer_no_double_holds_is_malformed(self):
        """As ``1e400`` already was: the reply is refused whole, so a scorer's
        conversation is set aside rather than scored with that confidence."""
        for number in (MAX_STORED_INT + 1, 10**400, -(10**400)):
            with self.subTest(digits=len(str(number))):
                with self.assertRaises(MalformedModelOutput):
                    extract_json_object('Sure:\n{"confidence": %d, "verdict": "PASS"}' % number)

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


def _set_field(config_path: Path, table: str, key: str, value: str) -> None:
    """Write ``key = value`` into ``[table]`` of a test config, replacing any line there."""
    out: list[str] = []
    current, done = None, False
    for line in config_path.read_text(encoding="utf-8").split("\n"):
        header = re.match(r"^\[\[?([^\]]+)\]\]?\s*$", line)
        if header:
            if current == table and not done:
                out.append(f"{key} = {value}")
                done = True
            current = header.group(1)
        if current == table and re.match(rf"^{re.escape(key)}\s*=", line):
            continue
        out.append(line)
    if not done:
        out += ([] if current == table else ["", f"[{table}]"]) + [f"{key} = {value}"]
    config_path.write_text("\n".join(out), encoding="utf-8")


#: An integer with 401 digits: past the largest double (about 1.8e308).
PAST_A_DOUBLE = "1" + "0" * 400
#: More digits than Python reads from text (4300 by default).
PAST_THE_DIGIT_LIMIT = "1" * 5000


class AnIntegerNoDoubleHoldsTests(unittest.TestCase):
    """An integer literal past the double range is refused like ``1e400``.

    Measured before (3.11 and 3.13): it parsed as a Python int, and a stored
    validation's ``min_judge_preference_ratio`` of 401 digits made ``review``
    fall with ``OverflowError`` from ``float()``, exit 70.
    """

    def test_an_integer_literal_past_the_double_range_is_a_value_error(self):
        for text in (
            PAST_A_DOUBLE,
            "-" + PAST_A_DOUBLE,
            str(MAX_STORED_INT + 1),
            str(-MAX_STORED_INT - 1),
            '{"a": [%s]}' % PAST_THE_DIGIT_LIMIT,
        ):
            with self.subTest(text=text[:12], digits=len(text)):
                with self.assertRaises(ValueError) as ctx:
                    loads_untrusted(text)
                self.assertIn("is larger than any double", str(ctx.exception))
                self.assertNotIn("set_int_max_str_digits", str(ctx.exception))

    def test_the_control_the_largest_integers_a_double_holds_still_read_exactly(self):
        largest = int(sys.float_info.max)
        self.assertEqual(MAX_STORED_INT, largest)
        self.assertEqual(len(str(largest)), 309)
        for number in (largest, -largest, 2**63, -(2**63), 0, 7):
            with self.subTest(number=number):
                self.assertEqual(loads_untrusted(str(number)), number)
                self.assertIs(type(loads_untrusted(str(number))), int)
        self.assertEqual(loads_untrusted("-0"), 0)

    def test_a_policy_built_in_code_with_such_an_integer_is_a_value_error(self):
        from agent_skill_improver.domain.validation import parse_gate_policy

        policy = {
            "min_fixes": 1,
            "max_regressions": 0,
            "max_holdout_degradations": 0,
            "min_holdout_cases": 1,
            "min_judge_preference_ratio": int(PAST_A_DOUBLE),
            "require_judge": True,
            "require_independent_judge": True,
        }
        with self.assertRaises(ValueError) as ctx:
            parse_gate_policy(policy)
        self.assertIn("min_judge_preference_ratio must be a finite number", str(ctx.exception))


class AStoredNumberNoDoubleHoldsTests(unittest.TestCase):
    def test_a_stored_gate_ratio_past_the_double_range_is_a_named_refusal(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            cfg = ["--config", str(build_env(base, dry_run=False, seed="past-a-double"))]
            daily = run_cli([*cfg, "--json", "daily", "--commit", "--day", DAY])
            self.assertEqual(daily.returncode, 0, daily.stderr)
            listed = json.loads(run_cli([*cfg, "--json", "candidates"]).stdout)["candidates"]
            validated = sorted(c["id"] for c in listed if c.get("status") == "validated")
            self.assertTrue(validated, "the run needs a validated candidate")
            leaf = base / "state" / "candidates" / f"{validated[0]}.json"
            text, replaced = re.subn(
                r'("min_judge_preference_ratio":\s*)[0-9.]+',
                lambda m: m.group(1) + PAST_A_DOUBLE,
                leaf.read_text(encoding="utf-8"),
            )
            self.assertEqual(replaced, 1)
            leaf.write_text(text, encoding="utf-8")
            review = run_cli([*cfg, "review", validated[0]])
        self.assertEqual(review.returncode, 3, review.stderr)
        self.assertIn("is larger than any double", review.stderr)
        self.assertNotIn("Traceback", review.stderr)


class AConfigNumberThePlatformCannotHoldTests(unittest.TestCase):
    """A config number past what the platform holds is a ``ConfigError`` at load.

    Measured before (3.11 and 3.13): a 401-digit ``[gate]`` ratio made every
    command that loads the config fall there with ``OverflowError``; a 5000-digit integer
    anywhere in the config or a rubric fell with tomllib's digit-limit
    ``ValueError``; and a model ``timeout_seconds`` of 10000000000 passed
    ``doctor``, then made ``daily`` fall with ``TypeError`` from the wait on
    the child's pipes (macOS kqueue refuses a timeout from 9223372037 s), each
    exit 70 with a traceback.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base, seed="platform-numbers"))

    def run_doctor_and_daily(self) -> list:
        cfg = ["--config", str(self.config_path)]
        return [run_cli([*cfg, "doctor"]), run_cli([*cfg, "daily", "--day", DAY])]

    def assert_refused_at_load(self, needle: str) -> None:
        for proc in self.run_doctor_and_daily():
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertIn(needle, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)

    def test_a_gate_ratio_past_the_double_range_is_a_config_error(self):
        _set_field(self.config_path, "gate", "min_judge_preference_ratio", PAST_A_DOUBLE)
        self.assert_refused_at_load("gate.min_judge_preference_ratio is an integer of 401 digits")
        _set_field(self.config_path, "gate", "min_judge_preference_ratio", str(MAX_STORED_INT))
        self.assert_refused_at_load("min_judge_preference_ratio must be a number in [0, 1]")

    def test_the_control_a_gate_ratio_of_zero_or_one_written_as_an_integer_still_loads(self):
        for value in ("0", "1", "0.5"):
            with self.subTest(value=value):
                _set_field(self.config_path, "gate", "min_judge_preference_ratio", value)
                doctor = run_cli(["--config", str(self.config_path), "doctor"])
                self.assertEqual(doctor.returncode, 0, doctor.stderr)
                self.assertNotIn("min_judge_preference_ratio", doctor.stderr)
                self.assertNotIn("Traceback", doctor.stderr)

    def test_an_integer_with_more_digits_than_python_reads_is_a_config_error(self):
        _set_field(self.config_path, "gate", "min_fixes", PAST_THE_DIGIT_LIMIT)
        self.assert_refused_at_load("invalid TOML")

    def test_the_same_integer_in_a_rubric_is_a_config_error(self):
        rubrics = sorted((self.base / "rubrics").glob("*.toml"))
        self.assertTrue(rubrics)
        rubrics[0].write_text(
            f"zz = {PAST_THE_DIGIT_LIMIT}\n" + rubrics[0].read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self.assert_refused_at_load("invalid TOML")

    def test_a_timeout_past_seven_days_is_a_config_error_not_a_fall_mid_run(self):
        from agent_skill_improver.adapters.bounded_process import MAX_TIMEOUT_SECONDS

        self.assertEqual(MAX_TIMEOUT_SECONDS, 7 * 24 * 60 * 60)
        for table in ("model.defaults", "model.scorer", "source"):
            for value in ("10000000000", str(MAX_TIMEOUT_SECONDS + 1), str(MAX_STORED_INT)):
                with self.subTest(table=table, value=value[:12]):
                    _set_field(self.config_path, table, "timeout_seconds", value)
                    self.assert_refused_at_load("timeout_seconds must be")
                    _set_field(self.config_path, table, "timeout_seconds", "30")

    def test_the_control_a_timeout_of_exactly_seven_days_still_loads(self):
        for table in ("model.defaults", "source"):
            with self.subTest(table=table):
                _set_field(self.config_path, table, "timeout_seconds", "604800")
                doctor = run_cli(["--config", str(self.config_path), "doctor"])
                self.assertEqual(doctor.returncode, 0, doctor.stderr)
                self.assertNotIn("timeout_seconds", doctor.stderr)
                self.assertNotIn("Traceback", doctor.stderr)

    def test_a_timeout_of_zero_or_less_is_a_config_error_too(self):
        for table in ("model.defaults", "model.scorer", "source"):
            for value in ("0", "-1"):
                with self.subTest(table=table, value=value):
                    _set_field(self.config_path, table, "timeout_seconds", value)
                    with self.assertRaises(ConfigError) as ctx:
                        load_config(self.config_path)
                    self.assertIn("timeout_seconds must be", str(ctx.exception))
                    _set_field(self.config_path, table, "timeout_seconds", "30")

    def test_a_bad_value_a_role_takes_from_model_defaults_names_that_table(self):
        _set_field(self.config_path, "model.defaults", "timeout_seconds", "0")
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        message = str(ctx.exception)
        self.assertIn("timeout_seconds must be a positive integer", message)
        self.assertRegex(
            message, r"takes [a-z_, ]*\btimeout_seconds\b[a-z_, ]* from \[model\.defaults\]"
        )
        _set_field(self.config_path, "model.defaults", "timeout_seconds", "30")
        _set_field(self.config_path, "model.scorer", "timeout_seconds", "0")
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        self.assertNotRegex(str(ctx.exception), r"takes [a-z_, ]*\btimeout_seconds\b")

    def test_the_note_names_only_the_inherited_key_that_failed(self):
        # A role's own bad value is its own: no inherited key is blamed for it.
        _set_field(self.config_path, "model.defaults", "timeout_seconds", "30")
        _set_field(self.config_path, "model.scorer", "max_output_bytes", "0")
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        self.assertIn("[model.scorer].max_output_bytes must be", str(ctx.exception))
        self.assertNotIn("from [model.defaults]", str(ctx.exception))
        # A role with its own argv takes none from the defaults, so its own bad
        # argv blames no inherited one; a role without one does take it.
        _set_field(self.config_path, "model.scorer", "max_output_bytes", "65536")
        _set_field(self.config_path, "model.defaults", "argv", '["", "model"]')
        _set_field(self.config_path, "model.scorer", "argv", '["", "model"]')
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        self.assertIn("[model.scorer].argv element #0", str(ctx.exception))
        self.assertNotIn("from [model.defaults]", str(ctx.exception))
        # Nor does a role that names its own command: its argv is that command.
        lines = self.config_path.read_text(encoding="utf-8").split("\n")
        at = lines.index("[model.scorer]")
        own = [i for i in range(at + 1, len(lines)) if lines[i].startswith("argv =")][0]
        lines[own] = "command = '\"\" model'"
        self.config_path.write_text("\n".join(lines), encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        self.assertIn("[model.scorer].argv element #0", str(ctx.exception))
        self.assertNotIn("from [model.defaults]", str(ctx.exception))
        del lines[own]
        self.config_path.write_text("\n".join(lines), encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        message = str(ctx.exception)
        self.assertIn("[model.scorer].argv element #0", message)
        self.assertIn("([model.scorer] takes argv from [model.defaults])", message)


#: The integer fields a run writes into a record, one per table that has them.
STORED_FIELDS = (
    ("source", "max_messages"),
    ("selection", "max_conversations_per_run"),
    ("caps", "max_open_candidates"),
    ("gate", "min_fixes"),
    ("prompt", "max_transcript_chars"),
)


class AConfigIntegerNoRecordHoldsTests(unittest.TestCase):
    """An integer no stored record holds is refused at load, wherever it is written.

    Measured before (3.11 and 3.13, with ``loads_untrusted`` already refusing
    one): a 401-digit ``[caps].max_open_candidates`` let ``doctor`` say ready
    and the first ``daily --commit`` exit 0, then every later ``daily`` exited
    3 over the receipt the first had written (``JSON integer of 401 digits``),
    even after the config was fixed; a 401-digit rubric ``version`` made the
    candidates it wrote unreadable.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base, dry_run=False, seed="stored-integers"))

    def test_an_integer_no_stored_record_holds_is_refused_in_any_config_field(self):
        original = self.config_path.read_text(encoding="utf-8")
        for table, key in STORED_FIELDS:
            for number in (MAX_STORED_INT + 1, -MAX_STORED_INT - 1, int(PAST_A_DOUBLE)):
                with self.subTest(field=f"{table}.{key}", digits=len(str(abs(number)))):
                    _set_field(self.config_path, table, key, str(number))
                    with self.assertRaises(ConfigError) as ctx:
                        load_config(self.config_path)
                    message = str(ctx.exception)
                    self.assertIn(
                        f"{table}.{key} is an integer of {len(str(abs(number)))} digits, larger "
                        "than any number a stored record holds",
                        message,
                    )
                    self.assertNotIn(str(abs(number))[:40], message)
            self.config_path.write_text(original, encoding="utf-8")

    def test_a_name_that_is_not_plain_is_quoted_bounded_and_no_key_is_said_to_be_stored(self):
        original = self.config_path.read_text(encoding="utf-8")
        for key, spelled in (
            ('"line\\nbreak"', "'caps.line\\nbreak'"),
            ('"red\\u001b[31m"', "'caps.red\\x1b[31m'"),
            ('"' + "k" * 5000 + '"', "more chars not quoted]"),
        ):
            with self.subTest(key=key[:12]):
                self.config_path.write_text(
                    original.replace("[caps]\n", f"[caps]\n{key} = {PAST_A_DOUBLE}\n", 1),
                    encoding="utf-8",
                )
                with self.assertRaises(ConfigError) as ctx:
                    load_config(self.config_path)
                message = str(ctx.exception)
                self.assertIn(spelled, message)
                self.assertIn("is an integer of 401 digits", message)
                self.assertIn("so no setting can take it", message)
                self.assertNotIn("\n", message)
                self.assertNotIn("\x1b", message)
                self.assertLess(len(message), 1000)
        self.config_path.write_text(original, encoding="utf-8")

    def test_doctor_and_daily_refuse_it_before_anything_is_written(self):
        _set_field(self.config_path, "caps", "max_open_candidates", PAST_A_DOUBLE)
        cfg = ["--config", str(self.config_path)]
        for proc in (run_cli([*cfg, "doctor"]), run_cli([*cfg, "daily", "--commit", "--day", DAY])):
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertIn("caps.max_open_candidates is an integer of 401 digits", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(list((self.base / "state").rglob("*.json")))

    def test_the_same_integer_in_a_rubric_is_refused_by_its_field(self):
        rubric = sorted((self.base / "rubrics").glob("*.toml"))[0]
        text = rubric.read_text(encoding="utf-8")
        for old, new in (
            ("version = 1\n", f"version = {PAST_A_DOUBLE}\n"),
            ("weight = 3\n", f"weight = {MAX_STORED_INT + 1}\n"),
        ):
            with self.subTest(field=new.split(" ")[0]):
                self.assertIn(old, text)
                rubric.write_text(text.replace(old, new, 1), encoding="utf-8")
                with self.assertRaises(ConfigError) as ctx:
                    load_config(self.config_path)
                where = "rubric.version" if old.startswith("version") else "criterion[0].weight"
                self.assertIn(f"{rubric.name}: {where} is an integer of", str(ctx.exception))
        rubric.write_text(text, encoding="utf-8")

    def test_the_control_the_largest_integer_a_record_holds_is_written_and_read_back(self):
        _set_field(self.config_path, "caps", "max_open_candidates", str(MAX_STORED_INT))
        cfg = ["--config", str(self.config_path)]
        for day in (DAY, "2026-03-02"):
            with self.subTest(day=day):
                proc = run_cli([*cfg, "daily", "--commit", "--day", day])
                self.assertEqual(proc.returncode, 0, proc.stderr)
        receipts = sorted((self.base / "state" / "receipts").glob(f"{DAY}--*.json"))
        self.assertTrue(receipts)
        self.assertIn(str(MAX_STORED_INT), receipts[0].read_text(encoding="utf-8"))
        listed = run_cli([*cfg, "candidates"])
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertNotIn("UNREADABLE", listed.stdout + listed.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
