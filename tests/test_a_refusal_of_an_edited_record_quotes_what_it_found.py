"""The parser and the integrity check quote what they found in an edited candidate record.

Measured before (3.11 and 3.13), with a real validated candidate whose stored
record was then edited and ``review`` and ``apply`` run on it: a key the strict
validation parser did not expect was listed as written, so one holding a line
break printed a second line of the refusal (``config error: forged``) and one
holding a ``ghp_`` token printed the token, in the gate, its metrics or policy,
a case or the record itself; a stored digest, schema, case key, case kind,
rubric id or candidate id the parser or the integrity check refused was shown
with ``repr``, so a token in it was printed whole; a recorded patch hash
beginning with a line break printed a second line through its 12-character
prefix; a key or digest of a megabyte made a refusal of a megabyte; and the
integrity check listed every problem and every case key it found, so 10,000
extra metrics keys made 599 KB of refusal and 10,000 extra claimed failures
1.16 MB.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Callable

from agent_skill_improver.cli import RC_REFUSED
from agent_skill_improver.domain.candidate import Candidate
from agent_skill_improver.domain.integrity import PROBLEMS_SHOWN, verify_full_integrity
from agent_skill_improver.domain.redaction import QUOTED_VALUES_SHOWN, quoted_values
from agent_skill_improver.domain.validation import ValidationRecord
from agent_skill_improver.errors import CandidateTampered
from agent_skill_improver.hashing import sha256_file
from tests.support import build_env, run_cli

TOKEN = "gh" + "p_" + "Zq8vLm3pXr7tWk2sQ1aB0123456789abcdef"
FORGED = "config error: forged"
PLANTED = TOKEN + "\n" + FORGED
#: The line break first: a 12-character prefix of it keeps the break and "config erro".
LEADING = "\n" + FORGED + " " + TOKEN
MASKED = "[REDACTED:github-token]"
RUBRIC = "instruction-compliance"
HUGE = 10**4000

_ENV: dict[str, Any] = {}


def setUpModule() -> None:
    tmp = TemporaryDirectory()
    base = Path(tmp.name).resolve()
    cfg = ["--config", str(build_env(base, dry_run=False, seed="quoted-refusals"))]
    proc = run_cli([*cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
    validated = [
        c["candidate_id"]
        for c in json.loads(proc.stdout)["candidates"]
        if c.get("status") == "validated"
    ]
    cid = validated[0]
    review = json.loads(run_cli([*cfg, "--json", "review", cid]).stdout)
    record = base / "state" / "candidates" / f"{cid}.json"
    skill = base / "skills-canonical" / "tdd" / "SKILL.md"
    _ENV.update(
        tmp=tmp,
        cfg=cfg,
        cid=cid,
        token=review["approval_token"],
        record=record,
        skill=skill,
        clean=json.loads(record.read_text(encoding="utf-8")),
    )


def tearDownModule() -> None:
    _ENV["tmp"].cleanup()


def _renamed(value: Any, old: str, new: str) -> Any:
    """``value`` with ``old`` replaced by ``new`` in every string and key."""
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_renamed(item, old, new) for item in value]
    if isinstance(value, dict):
        return {_renamed(k, old, new): _renamed(v, old, new) for k, v in value.items()}
    return value


def _rekey(case: dict[str, Any]) -> None:
    case["key"] = f"{case['conversation_id']}::{case['criterion_id']}"


def _index(v: dict[str, Any], kind: str) -> int:
    return next(i for i, o in enumerate(v["outcomes"]) if o["case"]["kind"] == kind)


def _target_renamed(v: dict[str, Any]) -> dict[str, Any]:
    """The first target conversation renamed to the planted text throughout."""
    old = v["outcomes"][_index(v, "target")]["case"]["conversation_id"]
    renamed = _renamed(v, old, PLANTED)
    for judgement in renamed["judgements"]:
        judgement["preferred_label"] = judgement["preferred_arm"] = "tie"
    return renamed


class _Quoting(unittest.TestCase):
    def assert_quoted(self, text: str) -> None:
        self.assertNotIn(TOKEN, text)
        self.assertIn(MASKED, text)
        self.assertFalse(
            [line for line in text.splitlines() if line.startswith(FORGED[:11])], "a forged line"
        )

    def parse_refusal(self, edit: Callable[[dict[str, Any]], Any]) -> str:
        v = copy.deepcopy(_ENV["clean"]["validation"])
        replaced = edit(v)
        with self.assertRaises(ValueError) as caught:
            ValidationRecord.parse(replaced if isinstance(replaced, dict) else v)
        message = str(caught.exception)
        self.assertNotIn("\n", message, "a parse refusal is one line")
        return message

    def tampered(self, edit: Callable[[dict[str, Any]], Any], rubrics: Any = None) -> str:
        raw = copy.deepcopy(_ENV["clean"])
        replaced = edit(raw)
        candidate = Candidate.from_dict(replaced if isinstance(replaced, dict) else raw)
        with self.assertRaises(CandidateTampered) as caught:
            verify_full_integrity(
                candidate,
                base_text=_ENV["skill"].read_text(encoding="utf-8"),
                rubrics=rubrics,
                action="review",
            )
        return str(caught.exception)


def _set(path: tuple[Any, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def edit(v: dict[str, Any]) -> None:
        node = v
        for step in path[:-1]:
            node = node[step]
        node[path[-1]] = value

    return edit


def _holdout_twice(v: dict[str, Any]) -> None:
    h = _index(v, "holdout")
    v["outcomes"][h]["case"]["conversation_id"] = PLANTED
    _rekey(v["outcomes"][h]["case"])
    v["outcomes"].append(copy.deepcopy(v["outcomes"][h]))


def _holdout_as_both_kinds(v: dict[str, Any]) -> None:
    h = _index(v, "holdout")
    v["outcomes"][h]["case"]["conversation_id"] = PLANTED
    _rekey(v["outcomes"][h]["case"])
    other = copy.deepcopy(v["outcomes"][h])
    other["case"]["kind"] = "target"
    other["rubric_id"] = "another-rubric"
    v["outcomes"].append(other)


def _judgement_about_no_outcome(v: dict[str, Any]) -> None:
    case = v["judgements"][0]["case"]
    case["conversation_id"] = PLANTED
    _rekey(case)
    v["judgements"][0]["preferred_label"] = v["judgements"][0]["preferred_arm"] = "tie"


def _judgement_twice(v: dict[str, Any]) -> dict[str, Any]:
    renamed = _target_renamed(v)
    planted = next(j for j in renamed["judgements"] if j["case"]["conversation_id"] == PLANTED)
    renamed["judgements"].append(copy.deepcopy(planted))
    return renamed


def _arm_of_a_renamed_case(v: dict[str, Any]) -> dict[str, Any]:
    renamed = _target_renamed(v)
    planted = next(j for j in renamed["judgements"] if j["case"]["conversation_id"] == PLANTED)
    planted["preferred_label"], planted["preferred_arm"] = "A", "neither"
    return renamed


def _recomputed_key_stored_as(value: Any) -> Callable[[dict[str, Any]], None]:
    """A recomputed gate whose case key holds the planted text, stored as ``value``."""

    def edit(r: dict[str, Any]) -> None:
        r["validation"] = _target_renamed(r["validation"])
        keys = r["validation"]["gate"]["metrics"]["fixed_case_keys"]
        keys[next(i for i, key in enumerate(keys) if PLANTED in key)] = value

    return edit


def _rubric_renamed_at_another_version(r: dict[str, Any]) -> dict[str, Any]:
    renamed = _renamed(r, RUBRIC, PLANTED)
    renamed["rubric_failures"][0]["rubric_version"] += 1
    return renamed


def _arm_and_seed(v: dict[str, Any]) -> None:
    v["seed"] = PLANTED
    v["judgements"][0]["preferred_arm"] = PLANTED


PARSE_CASES: tuple[tuple[str, Callable[[dict[str, Any]], Any], str], ...] = (
    ("an unknown key in the record", _set((PLANTED,), 1), "carries unknown key(s)"),
    ("an unknown key in the gate", _set(("gate", PLANTED), 1), "carries unknown key(s)"),
    ("an unknown key in the policy", _set(("gate", "policy", PLANTED), 1), "unknown key(s)"),
    ("an unknown key in a case", _set(("outcomes", 0, "case", PLANTED), 1), "unknown key(s)"),
    ("a digest", _set(("base_sha256",), PLANTED), "sha256 hex digest"),
    ("a rubric source digest", _set(("rubrics", 0, "source_sha256"), PLANTED), "hex digest"),
    ("the record schema", _set(("schema",), PLANTED), "validation.schema must be"),
    ("a case kind", _set(("outcomes", 0, "case", "kind"), PLANTED), ".kind must be"),
    ("a case key", _set(("outcomes", 0, "case", "key"), PLANTED), ".key is"),
    (
        "a case key recomputed from its conversation",
        _set(("outcomes", 0, "case", "conversation_id"), PLANTED),
        "recomputes to",
    ),
    ("a judgement schema", _set(("judgements", 0, "schema"), PLANTED), ".schema must be"),
    ("a preferred label", _set(("judgements", 0, "preferred_label"), PLANTED), "must be 'A'"),
    ("a preferred arm and the seed", _arm_and_seed, "blind mapping for seed"),
    ("a preferred arm for a renamed case", _arm_of_a_renamed_case, "blind mapping for seed"),
    (
        "a duplicate rubric id",
        lambda v: v.__setitem__("rubrics", [dict(v["rubrics"][0], rubric_id=PLANTED)] * 2),
        "duplicate rubric_id",
    ),
    (
        "a duplicate target ref",
        lambda v: v.__setitem__(
            "target_refs", [dict(v["target_refs"][0], conversation_id=PLANTED)] * 2
        ),
        "duplicate ref",
    ),
    ("a duplicate outcome", _holdout_twice, "duplicate case"),
    ("a case recorded as both kinds", _holdout_as_both_kinds, "as both"),
    (
        "a target case no ref names",
        lambda v: v["outcomes"][_index(v, "target")].__setitem__("rubric_id", PLANTED),
        "is not among validation.target_refs",
    ),
    (
        "a rubric no binding names",
        lambda v: v["outcomes"][_index(v, "holdout")].__setitem__("rubric_id", PLANTED),
        "missing binding(s) for",
    ),
    (
        "a binding no outcome uses",
        lambda v: v["rubrics"].append(dict(v["rubrics"][0], rubric_id=PLANTED)),
        "unused binding(s) for",
    ),
    ("a duplicate judgement", _judgement_twice, "has a duplicate case"),
    ("a judgement about no outcome", _judgement_about_no_outcome, "references case"),
)


class AParseRefusalQuotesWhatItFoundTests(_Quoting):
    def test_each_stored_value_a_parse_refusal_names_is_quoted_masked_on_one_line(self):
        for name, edit, expect in PARSE_CASES:
            with self.subTest(name):
                message = self.parse_refusal(edit)
                self.assertIn(expect, message)
                self.assert_quoted(message)

    def test_ten_thousand_unknown_keys_are_named_five_and_counted(self):
        def edit(v: dict[str, Any]) -> None:
            for i in range(10_000):
                v[f"u{i}"] = i

        message = self.parse_refusal(edit)
        self.assertIn(f"and {10_000 - QUOTED_VALUES_SHOWN} more", message)
        self.assertLess(len(message), 1_000)

    def test_a_key_that_is_a_credential_and_nothing_else_is_masked(self):
        # A bare token reads as a field name; only the masking keeps it hidden.
        self.assert_quoted(self.parse_refusal(_set((TOKEN,), 1)))
        self.assert_quoted(self.tampered(_set(("validation", "gate", "metrics", TOKEN), 1)))

    def test_a_megabyte_key_or_digest_is_cut(self):
        for edit in (_set(("y" * 1_000_000,), 1), _set(("base_sha256",), "z" * 1_000_000)):
            with self.subTest(edit=edit):
                message = self.parse_refusal(edit)
                self.assertIn("more chars not quoted", message)
                self.assertLess(len(message), 1_000)

    def test_a_count_or_version_of_four_thousand_digits_is_cut(self):
        for edit in (
            _set(("evidence_model_calls",), HUGE),
            _set(("gate", "policy", "min_fixes"), -HUGE),
            _set(("rubrics", 0, "version"), -HUGE),
        ):
            with self.subTest(edit=edit):
                message = self.parse_refusal(edit)
                self.assertIn("more chars not quoted", message)
                self.assertLess(len(message), 1_500)


class AnIntegrityProblemQuotesWhatItFoundTests(_Quoting):
    def test_a_key_on_one_side_only_is_quoted_and_a_plain_one_named_as_before(self):
        message = self.tampered(_set(("validation", "gate", "metrics", PLANTED), 1))
        self.assertIn(
            f"validation.gate.metrics['{MASKED}\\n{FORGED}']: present in one side only", message
        )
        self.assert_quoted(message)
        message = self.tampered(_set(("validation", "gate", "metrics", "extra_count"), 1))
        self.assertIn("  - validation.gate.metrics.extra_count: present in one side only", message)

    def test_each_stored_value_an_integrity_problem_names_is_quoted(self):
        canonical = Path(_ENV["clean"]["target"]["canonical_path"])
        binding = _ENV["clean"]["validation"]["rubrics"][0]
        cases: tuple[tuple[str, Callable[[dict[str, Any]], Any], Any, str], ...] = (
            (
                "a metrics value",
                _set(("validation", "gate", "metrics", "fixed_case_keys", 0), PLANTED),
                None,
                " != ",
            ),
            (
                "a metrics value of another type",
                _set(("validation", "gate", "metrics", "fixes"), PLANTED),
                None,
                "type int != str",
            ),
            (
                "the record's candidate id",
                _set(("validation", "candidate_id"), PLANTED),
                None,
                "validation.candidate_id is",
            ),
            ("the skill id", _set(("target", "skill_id"), PLANTED), None, "target.skill_id is"),
            (
                "the skill file's name",
                _set(("target", "canonical_path"), str(canonical.parent / PLANTED)),
                None,
                "target.canonical_path names",
            ),
            (
                "a recorded patch hash",
                _set(("patch", "new_text_sha256"), LEADING),
                None,
                "patch.new_text_sha256 records",
            ),
            (
                "a recorded base hash",
                _set(("target", "base_sha256"), LEADING),
                None,
                "target.base_sha256 records",
            ),
            (
                "the skill's folder",
                _set(
                    ("target", "canonical_path"),
                    str(canonical.parent.with_name(PLANTED) / "SKILL.md"),
                ),
                None,
                "sits in",
            ),
            (
                "a recomputed value",
                _recomputed_key_stored_as("other"),
                None,
                " != 'other'",
            ),
            (
                "a recomputed value of another type",
                _recomputed_key_stored_as(1),
                None,
                "type str != int",
            ),
            (
                "a claimed failure at another version of a renamed rubric",
                _rubric_renamed_at_another_version,
                None,
                "binds that rubric at version",
            ),
            (
                "a claimed failure no ref binds",
                _set(("rubric_failures", 0, "conversation_id"), PLANTED),
                None,
                "not bound by the validation",
            ),
            (
                "a bound failure no claim names",
                lambda r: r["validation"]["target_refs"].append(
                    dict(r["validation"]["target_refs"][0], conversation_id=PLANTED)
                ),
                None,
                "bound by the validation but not claimed",
            ),
            (
                "a target case no claim names",
                lambda r: r.__setitem__("validation", _target_renamed(r["validation"])),
                None,
                "that are not among this candidate's own",
            ),
            (
                "a rubric the configuration lacks",
                lambda r: _renamed(r, RUBRIC, PLANTED),
                {},
                "which is not among the currently configured rubrics",
            ),
            (
                "a rubric configured at another version, source and criteria",
                lambda r: _renamed(r, RUBRIC, PLANTED),
                {
                    PLANTED: SimpleNamespace(
                        version=99, source_sha256="0" * 64, criterion=lambda _: None
                    )
                },
                "has no criterion",
            ),
            (
                "a criterion the configured rubric lacks",
                lambda r: _renamed(r, "scope-respected", PLANTED),
                {
                    RUBRIC: SimpleNamespace(
                        version=binding["version"],
                        source_sha256=binding["source_sha256"],
                        criterion=lambda criterion_id: None if criterion_id == PLANTED else True,
                    )
                },
                "has no criterion",
            ),
        )
        for name, edit, rubrics, expect in cases:
            with self.subTest(name):
                message = self.tampered(edit, rubrics)
                self.assertIn(expect, message)
                self.assert_quoted(message)

    def test_a_claimed_failure_at_a_version_of_four_thousand_digits_is_cut(self):
        message = self.tampered(_set(("rubric_failures", 0, "rubric_version"), HUGE))
        self.assertIn("more chars not quoted", message)
        self.assertLess(len(message), 2_000)

    def test_ten_thousand_differences_are_listed_twenty_and_counted(self):
        def edit(r: dict[str, Any]) -> None:
            for i in range(10_000):
                r["validation"]["gate"]["metrics"][f"m{i}"] = i

        message = self.tampered(edit)
        listed = [line for line in message.splitlines() if line.startswith("  - ")]
        self.assertIn("10000 derived field(s)", message.splitlines()[0])
        self.assertEqual(len(listed), PROBLEMS_SHOWN + 1)
        self.assertEqual(listed[-1], f"  - and {10_000 - PROBLEMS_SHOWN} more not shown")
        self.assertLess(len(message), 5_000)

    def test_ten_thousand_claimed_failures_no_ref_binds_are_named_five_and_counted(self):
        def edit(r: dict[str, Any]) -> None:
            first = r["rubric_failures"][0]
            r["rubric_failures"] += [dict(first, conversation_id=f"c{i}") for i in range(10_000)]

        message = self.tampered(edit)
        self.assertIn(f"and {10_000 - QUOTED_VALUES_SHOWN} more]", message)
        self.assertLess(len(message), 5_000)


class TheCommandsRefuseInLinesThatNameNoSecretTests(_Quoting):
    def test_review_and_apply_refuse_an_edited_record_without_printing_what_it_holds(self):
        record, clean = _ENV["record"], _ENV["clean"]
        skill_sha = sha256_file(_ENV["skill"])
        edits = (
            _set(("validation", PLANTED), 1),
            _set(("validation", "gate", "metrics", PLANTED), 1),
            _set(("validation", "base_sha256"), PLANTED),
            _set(("patch", "new_text_sha256"), LEADING),
        )
        commands = (
            ("review", _ENV["cid"]),
            (
                "apply",
                _ENV["cid"],
                "--approve",
                _ENV["token"],
                "--i-reviewed-this",
                "--operator",
                "t",
            ),
        )
        try:
            for edit in edits:
                raw = copy.deepcopy(clean)
                edit(raw)
                record.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
                for args in commands:
                    with self.subTest(edit=edit, command=args[0]):
                        proc = run_cli([*_ENV["cfg"], *args])
                        self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout)
                        self.assertIn("CandidateTampered", proc.stderr)
                        self.assert_quoted(proc.stdout + proc.stderr)
                        self.assertEqual(sha256_file(_ENV["skill"]), skill_sha)
        finally:
            record.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")


class TheControlARecordAsWrittenTests(_Quoting):
    def test_the_control_the_record_as_written_parses_and_verifies(self):
        ValidationRecord.parse(copy.deepcopy(_ENV["clean"]["validation"]))
        verify_full_integrity(
            Candidate.from_dict(copy.deepcopy(_ENV["clean"])),
            base_text=_ENV["skill"].read_text(encoding="utf-8"),
            action="review",
        )

    def test_the_control_a_plain_value_or_key_reads_as_before(self):
        message = self.parse_refusal(_set(("schema",), "asi.validation.v2"))
        self.assertTrue(message.endswith("got 'asi.validation.v2'"), message)
        message = self.parse_refusal(_set(("extra",), 1))
        self.assertTrue(message.endswith("carries unknown key(s): extra"), message)
        message = self.tampered(_set(("patch", "new_text_sha256"), "0" * 64))
        self.assertIn("patch.new_text_sha256 records 000000000000 but", message)
        self.assertEqual(quoted_values(["a", "b"]), "'a', 'b'")
        self.assertEqual(quoted_values("abcdefg"), "'a', 'b', 'c', 'd', 'e', and 2 more")


if __name__ == "__main__":
    unittest.main()
