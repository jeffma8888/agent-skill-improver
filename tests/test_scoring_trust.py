"""Scorecard parsing and the local trust rules applied to model output.

These are the tests that decide whether the whole loop is honest, because
everything downstream treats a FAIL as ground truth.
"""

from __future__ import annotations

import json
import unittest

from agent_skill_improver import SCHEMA_SCORECARD
from agent_skill_improver.adapters.transcript_export import normalize_export
from agent_skill_improver.adapters.model_command import extract_json_object
from agent_skill_improver.config import load_config
from agent_skill_improver.domain.failure import (
    ATTRIB_NONE,
    ATTRIB_OBSERVED_MENTION,
    ATTRIB_OBSERVED_PATH,
    ATTRIB_RUBRIC_PRIOR,
    attribute,
    isolate_failures,
    passing_cases,
)
from agent_skill_improver.domain.rubric import Rubric, Verdict
from agent_skill_improver.domain.scoring import parse_scorecard
from agent_skill_improver.errors import MalformedModelOutput

from tests import support


def conversation(name: str):
    payload = json.loads((support.FIXTURE_EXPORTS / name).read_text(encoding="utf-8"))
    return normalize_export(payload)


def rubric(rubric_id: str = "instruction-compliance") -> Rubric:
    import tomllib

    text = (support.REPO_ROOT / "rubrics" / f"{rubric_id}.toml").read_text(encoding="utf-8")
    return Rubric.from_mapping(tomllib.loads(text), source_sha256="deadbeef")


def card(criteria: list[dict], *, thread: str | None = None, conv=None):
    conv = conv or conversation("failing.json")
    payload = {
        "schema": SCHEMA_SCORECARD,
        "conversation_id": thread if thread is not None else conv.thread_id,
        "criteria": criteria,
    }
    return parse_scorecard(payload, rubric=rubric(), conversation=conv, scorer_command_id="test")


def good_ref(conv) -> dict:
    message = conv.messages[0]
    return {"element_id": message.element_id, "position": message.position, "kind": "message"}


class VerdictTests(unittest.TestCase):
    def test_three_verdicts_are_distinct_and_only_two_are_decided(self) -> None:
        self.assertTrue(Verdict.PASS.is_decided)
        self.assertTrue(Verdict.FAIL.is_decided)
        self.assertFalse(Verdict.INSUFFICIENT.is_decided)
        self.assertEqual(len(set(Verdict)), 3)

    def test_parsing_is_case_insensitive_but_rejects_invented_verdicts(self) -> None:
        self.assertIs(Verdict.parse("pass"), Verdict.PASS)
        self.assertIs(Verdict.parse("  FAIL "), Verdict.FAIL)
        for bad in ("MAYBE", "", "0.5", None, 3, ["PASS"]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Verdict.parse(bad)


class ScorecardContractTests(unittest.TestCase):
    def test_a_fatal_contract_violation_raises(self) -> None:
        conv = conversation("failing.json")
        for payload in (
            "not a mapping",
            {"criteria": []},
            {"schema": "wrong", "criteria": []},
            {"schema": SCHEMA_SCORECARD},
            {"schema": SCHEMA_SCORECARD, "criteria": "not a list"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    parse_scorecard(payload, rubric=rubric(), conversation=conv)

    def test_a_scorecard_for_the_wrong_conversation_is_refused(self) -> None:
        """Guards against a cached or crossed answer being credited to this record."""
        with self.assertRaises(ValueError):
            card([], thread="some-other-thread")

    def test_an_unmentioned_criterion_becomes_insufficient_never_pass(self) -> None:
        """Silence is not a pass."""
        result = card([])
        self.assertEqual(len(result.scores), len(rubric().criterion_ids))
        self.assertTrue(all(s.verdict is Verdict.INSUFFICIENT for s in result.scores))
        self.assertEqual(len(result.passes), 0)

    def test_an_unknown_criterion_is_recorded_as_extraneous_not_dropped(self) -> None:
        conv = conversation("failing.json")
        result = card(
            [
                {
                    "criterion_id": "invented",
                    "verdict": "PASS",
                    "rationale": "x",
                    "evidence": [good_ref(conv)],
                }
            ]
        )
        self.assertEqual(result.extraneous_criterion_ids, ("invented",))
        self.assertTrue(any("not in rubric" in n for n in result.notes))

    def test_a_criterion_scored_twice_keeps_the_first_verdict(self) -> None:
        conv = conversation("failing.json")
        cid = rubric().criterion_ids[0]
        result = card(
            [
                {
                    "criterion_id": cid,
                    "verdict": "FAIL",
                    "rationale": "a",
                    "evidence": [good_ref(conv)],
                },
                {
                    "criterion_id": cid,
                    "verdict": "PASS",
                    "rationale": "b",
                    "evidence": [good_ref(conv)],
                },
            ]
        )
        self.assertIs(result.score_for(cid).verdict, Verdict.FAIL)
        self.assertTrue(any("more than once" in n for n in result.notes))

    def test_scores_come_back_in_rubric_order_regardless_of_input_order(self) -> None:
        conv = conversation("failing.json")
        ids = list(rubric().criterion_ids)
        payload = [
            {"criterion_id": cid, "verdict": "PASS", "rationale": "x", "evidence": [good_ref(conv)]}
            for cid in reversed(ids)
        ]
        result = card(payload)
        self.assertEqual([s.criterion_id for s in result.scores], ids)


class EvidenceTrustTests(unittest.TestCase):
    """A decided verdict without resolvable evidence is INSUFFICIENT, not FAIL."""

    def setUp(self) -> None:
        self.conv = conversation("failing.json")
        self.cid = rubric().criterion_ids[0]

    def _one(self, **overrides):
        item = {"criterion_id": self.cid, "verdict": "FAIL", "rationale": "r"}
        item.update(overrides)
        return card([item]).score_for(self.cid)

    def test_fail_with_no_evidence_is_downgraded(self) -> None:
        score = self._one(evidence=[])
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        self.assertIs(score.downgraded_from, Verdict.FAIL)
        self.assertIn("evidence ref", score.downgrade_reason)

    def test_fail_with_an_invented_element_id_is_downgraded(self) -> None:
        score = self._one(evidence=[{"element_id": "fabricated", "position": 3}])
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        self.assertIn("does not resolve", score.downgrade_reason)

    def test_fail_with_the_right_id_but_wrong_position_is_downgraded(self) -> None:
        message = self.conv.messages[0]
        score = self._one(
            evidence=[{"element_id": message.element_id, "position": message.position + 400}]
        )
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        self.assertIn("does not resolve", score.downgrade_reason)

    def test_fail_with_malformed_evidence_is_downgraded_with_a_reason(self) -> None:
        score = self._one(evidence=[{"element_id": "", "position": "nope"}])
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        self.assertIn("malformed", score.downgrade_reason)

    def test_pass_is_held_to_the_same_standard_as_fail(self) -> None:
        score = self._one(verdict="PASS", evidence=[])
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        self.assertIs(score.downgraded_from, Verdict.PASS)

    def test_a_resolvable_reference_is_trusted(self) -> None:
        score = self._one(evidence=[good_ref(self.conv)])
        self.assertIs(score.verdict, Verdict.FAIL)
        self.assertIsNone(score.downgraded_from)

    def test_an_unparseable_verdict_becomes_insufficient_not_an_exception(self) -> None:
        score = self._one(verdict="PROBABLY_FINE", evidence=[good_ref(self.conv)])
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        self.assertIn("unparseable verdict", score.downgrade_reason)

    def test_insufficient_needs_no_evidence(self) -> None:
        score = self._one(verdict="INSUFFICIENT", evidence=[])
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        self.assertIsNone(score.downgraded_from)

    def test_downgraded_verdicts_are_countable_for_the_receipt(self) -> None:
        conv = self.conv
        ids = list(rubric().criterion_ids)
        result = card(
            [
                {"criterion_id": ids[0], "verdict": "FAIL", "rationale": "r", "evidence": []},
                {
                    "criterion_id": ids[1],
                    "verdict": "FAIL",
                    "rationale": "r",
                    "evidence": [good_ref(conv)],
                },
            ]
        )
        self.assertEqual(len(result.downgraded), 1)
        self.assertEqual(len(result.failures), 1)


class IsolationTests(unittest.TestCase):
    def test_only_fail_becomes_a_failure_case(self) -> None:
        conv = conversation("failing.json")
        ids = list(rubric().criterion_ids)
        result = card(
            [
                {
                    "criterion_id": ids[0],
                    "verdict": "FAIL",
                    "rationale": "r",
                    "evidence": [good_ref(conv)],
                },
                {
                    "criterion_id": ids[1],
                    "verdict": "PASS",
                    "rationale": "r",
                    "evidence": [good_ref(conv)],
                },
                {
                    "criterion_id": ids[2],
                    "verdict": "INSUFFICIENT",
                    "rationale": "r",
                    "evidence": [],
                },
            ]
        )
        cases = isolate_failures(conv, result, rubric(), known_skill_ids=["tdd"])
        self.assertEqual([c.criterion_id for c in cases], [ids[0]])

    def test_insufficient_never_becomes_a_failure_case(self) -> None:
        """A loop that treats 'could not tell' as 'broken' manufactures its own work."""
        conv = conversation("failing.json")
        result = card([])  # everything INSUFFICIENT
        self.assertEqual(isolate_failures(conv, result, rubric(), known_skill_ids=["tdd"]), [])

    def test_failure_case_ids_are_deterministic(self) -> None:
        conv = conversation("failing.json")
        ids = list(rubric().criterion_ids)
        payload = [
            {
                "criterion_id": ids[0],
                "verdict": "FAIL",
                "rationale": "r",
                "evidence": [good_ref(conv)],
            }
        ]
        first = isolate_failures(conv, card(payload), rubric(), known_skill_ids=["tdd"])
        second = isolate_failures(conv, card(payload), rubric(), known_skill_ids=["tdd"])
        self.assertEqual([c.id for c in first], [c.id for c in second])

    def test_failure_cases_carry_evidence_and_reproducible_signals(self) -> None:
        conv = conversation("failing.json")
        ids = list(rubric().criterion_ids)
        cases = isolate_failures(
            conv,
            card(
                [
                    {
                        "criterion_id": ids[0],
                        "verdict": "FAIL",
                        "rationale": "r",
                        "evidence": [good_ref(conv)],
                    }
                ]
            ),
            rubric(),
            known_skill_ids=["tdd"],
        )
        case = cases[0]
        self.assertTrue(case.evidence)
        self.assertEqual(case.signals["failed_tool_events"], 2)
        self.assertEqual(case.signals["user_messages"], 3)
        self.assertEqual(case.workspace, "/tmp/asi-fixture-workspace-b")

    def test_passing_cases_become_the_holdout_set(self) -> None:
        conv = conversation("compliant.json")
        ids = list(rubric().criterion_ids)
        result = card(
            [
                {
                    "criterion_id": cid,
                    "verdict": "PASS",
                    "rationale": "r",
                    "evidence": [good_ref(conv)],
                }
                for cid in ids
            ],
            conv=conv,
        )
        pairs = passing_cases(conv, result)
        self.assertEqual(len(pairs), len(ids))
        self.assertTrue(all(p[0] == conv.thread_id for p in pairs))


class AttributionTests(unittest.TestCase):
    def test_a_skill_named_in_a_message_is_an_observed_mention(self) -> None:
        conv = conversation("failing.json")
        attributions = attribute(conv, criterion_implicates=(), known_skill_ids=["tdd"])
        bases = {a.basis for a in attributions}
        self.assertTrue(bases & {ATTRIB_OBSERVED_MENTION, ATTRIB_OBSERVED_PATH})
        self.assertEqual(attributions[0].skill_id, "tdd")
        self.assertTrue(attributions[0].evidence, "attribution must cite where it saw the skill")

    def test_a_skill_path_in_tool_io_outranks_a_bare_mention(self) -> None:
        conv = conversation("failing.json")
        attributions = attribute(conv, criterion_implicates=(), known_skill_ids=["tdd"])
        tdd = [a for a in attributions if a.skill_id == "tdd"][0]
        self.assertEqual(tdd.basis, ATTRIB_OBSERVED_PATH)

    def test_a_rubric_prior_alone_is_marked_as_a_prior_with_low_confidence(self) -> None:
        """A prior is a hypothesis about the skill, not an observation of it."""
        conv = conversation("compliant.json")
        attributions = attribute(
            conv, criterion_implicates=("code-review",), known_skill_ids=["code-review"]
        )
        prior = [a for a in attributions if a.skill_id == "code-review"][0]
        self.assertEqual(prior.basis, ATTRIB_RUBRIC_PRIOR)
        self.assertLess(prior.confidence, 0.5)
        self.assertIn("NOT observed", " ".join(prior.reasons))

    def test_nothing_observed_and_no_prior_yields_an_explicit_unattributed_marker(self) -> None:
        conv = conversation("compliant.json")
        attributions = attribute(conv, criterion_implicates=(), known_skill_ids=["nowhere-skill"])
        self.assertEqual([a.basis for a in attributions], [ATTRIB_NONE])
        self.assertEqual(attributions[0].skill_id, "")

    def test_matching_is_word_boundary_aware(self) -> None:
        """A substring match invents confident false attribution."""
        payload = {
            "thread": {"id": "1700000000200-1", "title": "t", "workspacePath": "/tmp/x"},
            "elements": [
                {"id": "1700000000200-2", "kind": "message", "role": "user", "position": 0},
                {
                    "id": "1700000000200-3",
                    "kind": "text",
                    "parentId": "1700000000200-2",
                    "content": "We should archive the tddsuite directory and untddify the build.",
                    "position": 1,
                },
            ],
        }
        conv = normalize_export(payload)
        attributions = attribute(conv, criterion_implicates=(), known_skill_ids=["tdd"])
        self.assertEqual([a.basis for a in attributions], [ATTRIB_NONE])

    def test_hyphen_and_underscore_are_treated_as_equivalent(self) -> None:
        payload = {
            "thread": {"id": "1700000000201-1", "title": "t", "workspacePath": "/tmp/x"},
            "elements": [
                {"id": "1700000000201-2", "kind": "message", "role": "user", "position": 0},
                {
                    "id": "1700000000201-3",
                    "kind": "text",
                    "parentId": "1700000000201-2",
                    "content": "Follow the code_review skill please.",
                    "position": 1,
                },
            ],
        }
        conv = normalize_export(payload)
        attributions = attribute(conv, criterion_implicates=(), known_skill_ids=["code-review"])
        self.assertEqual(attributions[0].skill_id, "code-review")

    def test_a_hyphenated_skill_id_is_matched_by_mention(self) -> None:
        """Regression: sequential replaces corrupted the pattern for any id with a hyphen.

        `re.escape("code-review")` produces `code\\-review`; replacing `\\-` inserts a
        character class that itself contains `_`, so a second replace of `_`
        mangled it. Almost every real skill id is hyphenated, so mention-based
        attribution silently matched nothing while still looking healthy.
        """
        payload = {
            "thread": {"id": "1700000000202-1", "title": "t", "workspacePath": "/tmp/x"},
            "elements": [
                {"id": "1700000000202-2", "kind": "message", "role": "user", "position": 0},
                {
                    "id": "1700000000202-3",
                    "kind": "text",
                    "parentId": "1700000000202-2",
                    "content": "Please apply the code-review skill to this diff.",
                    "position": 1,
                },
            ],
        }
        conv = normalize_export(payload)
        attributions = attribute(conv, criterion_implicates=(), known_skill_ids=["code-review"])
        self.assertEqual(attributions[0].skill_id, "code-review")
        self.assertEqual(attributions[0].basis, ATTRIB_OBSERVED_MENTION)

    def test_a_hyphenated_id_still_respects_word_boundaries(self) -> None:
        payload = {
            "thread": {"id": "1700000000203-1", "title": "t", "workspacePath": "/tmp/x"},
            "elements": [
                {"id": "1700000000203-2", "kind": "message", "role": "user", "position": 0},
                {
                    "id": "1700000000203-3",
                    "kind": "text",
                    "parentId": "1700000000203-2",
                    "content": "The code-reviewer bot and precode-review hooks are unrelated.",
                    "position": 1,
                },
            ],
        }
        conv = normalize_export(payload)
        attributions = attribute(conv, criterion_implicates=(), known_skill_ids=["code-review"])
        self.assertEqual([a.basis for a in attributions], [ATTRIB_NONE])

    def test_primary_skill_ignores_the_unattributed_marker(self) -> None:
        conv = conversation("compliant.json")
        ids = list(rubric().criterion_ids)
        cases = isolate_failures(
            conv,
            card(
                [
                    {
                        "criterion_id": ids[0],
                        "verdict": "FAIL",
                        "rationale": "r",
                        "evidence": [good_ref(conv)],
                    }
                ],
                conv=conv,
            ),
            rubric(),
            known_skill_ids=["nowhere-skill"],
        )
        self.assertIsNone(cases[0].primary_skill_id)
        self.assertFalse(cases[0].is_attributed)


class MalformedModelOutputTests(unittest.TestCase):
    """Non-JSON is a RECORD defect; an unreachable model is a TRANSIENT failure."""

    def test_plain_json_parses(self) -> None:
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})

    def test_json_wrapped_in_prose_is_recovered(self) -> None:
        text = 'Sure! Here you go:\n\n{"schema": "x", "n": 2}\n\nHope that helps.'
        self.assertEqual(extract_json_object(text)["n"], 2)

    def test_a_fenced_block_is_recovered(self) -> None:
        text = 'Here:\n```json\n{"ok": true}\n```\n'
        self.assertTrue(extract_json_object(text)["ok"])

    def test_the_last_balanced_object_wins_when_a_model_restates_itself(self) -> None:
        text = '{"attempt": 1}\nActually, corrected:\n{"attempt": 2}'
        self.assertEqual(extract_json_object(text)["attempt"], 2)

    def test_braces_inside_strings_do_not_confuse_the_extractor(self) -> None:
        text = 'note: "a } brace { inside"\n{"real": "yes {not a brace} here"}'
        self.assertEqual(extract_json_object(text)["real"], "yes {not a brace} here")

    def test_empty_output_is_malformed(self) -> None:
        for text in ("", "   ", "\n\n"):
            with self.subTest(text=repr(text)):
                with self.assertRaises(MalformedModelOutput):
                    extract_json_object(text)

    def test_a_json_array_is_refused_because_the_contract_says_object(self) -> None:
        with self.assertRaises(MalformedModelOutput):
            extract_json_object("[1,2,3]")

    def test_unparseable_output_is_refused_with_a_diagnostic(self) -> None:
        with self.assertRaises(MalformedModelOutput) as ctx:
            extract_json_object("I refuse to answer in JSON, sorry.")
        self.assertIn("no parseable JSON object", str(ctx.exception))

    def test_an_unbalanced_object_is_refused_rather_than_repaired(self) -> None:
        with self.assertRaises(MalformedModelOutput):
            extract_json_object('{"a": 1, "b": ')


if __name__ == "__main__":
    unittest.main()
