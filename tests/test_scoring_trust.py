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

    def test_a_refusal_quotes_the_models_value_bounded_and_redacted(self) -> None:
        """What the scorer sent is quoted so the operator can see it, but the
        quote is bounded and redacted: the value is the model's, limited only
        by max_output_bytes, and the message lands in a quarantine reason, the
        receipt, the text summary and the scheduler log."""
        from agent_skill_improver.domain.redaction import QUOTED_VALUE_CHARS

        huge = "x" * 10_000
        with self.assertRaises(ValueError) as ctx:
            card([], thread=huge)
        message = str(ctx.exception)
        self.assertLess(len(message), QUOTED_VALUE_CHARS + 200, len(message))
        self.assertIn("'" + "x" * (QUOTED_VALUE_CHARS - 1) + "...", message)
        self.assertIn(f"[{len(repr(huge)) - QUOTED_VALUE_CHARS} more chars not quoted]", message)
        self.assertIn("does not match the record being scored", message)

        key = "AK" + "IA" + "B" * 16
        with self.assertRaises(ValueError) as ctx:
            card([], thread=f"token {key}")
        self.assertNotIn(key, str(ctx.exception))
        self.assertIn("aws-access-key-id", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            parse_scorecard(
                {"schema": "y" * 5_000, "criteria": []},
                rubric=rubric(),
                conversation=conversation("failing.json"),
            )
        self.assertLess(len(str(ctx.exception)), QUOTED_VALUE_CHARS + 200)

        result = card([{"criterion_id": "z" * 5_000, "verdict": "PASS"}])
        note = next(n for n in result.notes if "is not in rubric" in n)
        self.assertLess(len(note), QUOTED_VALUE_CHARS + 200, len(note))

    def test_the_control_a_short_value_is_quoted_whole(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            card([], thread="some-other-thread")
        self.assertIn("('some-other-thread' != ", str(ctx.exception))

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

    def test_an_unresolved_element_id_is_quoted_bounded_redacted_and_not_stored(self) -> None:
        """The element_id the scorer cited is the scorer's text. When it does
        not resolve, the downgrade reason names it bounded and redacted, and
        the scorecard stores no ref for it: a claim that does not resolve is
        not evidence, and the scorecard's evidence feeds the improver prompt."""
        from agent_skill_improver.domain.redaction import QUOTED_VALUE_CHARS

        key = "gh" + "p_" + "A" * 24
        refs = [{"element_id": f"{key} {'y' * 5_000}", "position": 1}]
        refs += [{"element_id": f"made-up-{i}", "position": i} for i in range(5)]
        score = self._one(evidence=refs)
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        self.assertEqual(score.evidence, ())
        self.assertNotIn(key, score.downgrade_reason)
        self.assertIn("github-token", score.downgrade_reason)
        self.assertIn("more chars not quoted]@1", score.downgrade_reason)
        self.assertIn(" and 3 more", score.downgrade_reason)
        self.assertLess(len(score.downgrade_reason), 3 * QUOTED_VALUE_CHARS + 200)

    def test_an_unresolved_position_is_quoted_bounded_like_its_element_id(self) -> None:
        """A JSON integer has no bound, and the position was printed raw."""
        from agent_skill_improver.domain.redaction import QUOTED_VALUE_CHARS

        huge = int("9" * 4300)
        refs = [{"element_id": f"made-up-{i}", "position": huge} for i in range(5)]
        score = self._one(evidence=refs)
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        self.assertIn("made-up-0", score.downgrade_reason)
        self.assertIn("more chars not quoted]", score.downgrade_reason)
        self.assertIn(" and 2 more", score.downgrade_reason)
        self.assertEqual(score.downgrade_reason.count("@"), 3, "three refs spelled out")
        self.assertLess(len(score.downgrade_reason), 3 * 2 * QUOTED_VALUE_CHARS + 300)

    def test_the_control_a_small_unresolved_position_is_named_whole(self) -> None:
        score = self._one(evidence=[{"element_id": "fabricated", "position": 3}])
        self.assertIn("'fabricated'@3", score.downgrade_reason)

    def test_exactly_as_many_unresolved_refs_as_are_shown_says_no_more(self) -> None:
        from agent_skill_improver.domain.scoring import MAX_SHOWN_BAD_REFS

        refs = [{"element_id": f"made-up-{i}", "position": i} for i in range(MAX_SHOWN_BAD_REFS)]
        score = self._one(evidence=refs)
        self.assertEqual(score.downgrade_reason.count("@"), MAX_SHOWN_BAD_REFS)
        self.assertNotIn("more", score.downgrade_reason)

    def test_a_ref_that_resolves_is_stored_beside_one_that_does_not(self) -> None:
        message = self.conv.messages[0]
        good = {"element_id": message.element_id, "position": message.position}
        score = self._one(
            verdict="INSUFFICIENT", evidence=[good, {"element_id": "no", "position": 9}]
        )
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        self.assertIsNone(score.downgraded_from)
        self.assertEqual([e.element_id for e in score.evidence], [message.element_id])

    def test_an_unknown_verdict_is_quoted_bounded_and_redacted(self) -> None:
        """The verdict string is the scorer's too; its failure becomes a note
        and a downgrade_reason, both persisted in the scorecard."""
        from agent_skill_improver.domain.redaction import QUOTED_VALUE_CHARS

        key = "AK" + "IA" + "C" * 16
        conv = self.conv
        notes_payload = [
            {
                "criterion_id": self.cid,
                "verdict": f"MAYBE {key} {'z' * 5_000}",
                "rationale": "r",
                "evidence": [good_ref(conv)],
            }
        ]
        result = card(notes_payload)
        score = result.score_for(self.cid)
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        for text in (score.downgrade_reason, *result.notes):
            self.assertNotIn(key, text)
        self.assertIn("aws-access-key-id", score.downgrade_reason)
        self.assertIn("more chars not quoted]", score.downgrade_reason)
        self.assertLess(len(score.downgrade_reason), QUOTED_VALUE_CHARS + 200)
        self.assertTrue(any("unknown verdict 'MAYBE" in n for n in result.notes), result.notes)

    def test_a_rationale_that_pastes_a_key_is_kept_masked_and_still_clipped(self) -> None:
        """The scorer's rationale is its own prose, persisted in the scorecard
        and fed to the improver. The transcript it saw was masked; a model role
        that can read more than its prompt gets the same mask on the way back."""
        from agent_skill_improver.domain.scoring import MAX_RATIONALE_CHARS

        key = "gh" + "p_" + "H" * 24
        score = self._one(
            verdict="PASS",
            evidence=[good_ref(self.conv)],
            rationale=f"saw {key} then " + "r" * 5_000,
        )
        self.assertIs(score.verdict, Verdict.PASS)
        self.assertNotIn(key, score.rationale)
        self.assertTrue(score.rationale.startswith("saw [REDACTED:github-token] then "))
        self.assertTrue(score.rationale.endswith(" ...[clipped]"))
        self.assertLessEqual(len(score.rationale), MAX_RATIONALE_CHARS + len(" ...[clipped]"))

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


class EvidenceFieldShapeTests(unittest.TestCase):
    """The ``evidence`` field is the model's. It used to be iterated as it came:
    a number or a boolean raised ``TypeError`` out of the parser, which no caller
    classifies, so one such scorer reply aborted a committed run as an internal
    error with no receipt (measured end to end with the in-process scorer); a
    string iterated character by character and an object by its keys, each a
    malformed ref of its own, so one wrong field became hundreds of refusals in
    the persisted reason."""

    def setUp(self) -> None:
        self.cid = rubric().criterion_ids[0]

    def _one(self, **overrides):
        item = {"criterion_id": self.cid, "verdict": "FAIL", "rationale": "r"}
        item.update(overrides)
        return card([item]).score_for(self.cid)

    def test_a_non_array_evidence_field_is_one_named_malformed_ref_not_a_crash(self) -> None:
        for value in (
            7,
            1.5,
            True,
            "x" * 300,
            {"element_id": "m", "position": 1},
            0,
            False,
            "",
            {},
        ):
            with self.subTest(value=type(value).__name__):
                score = self._one(evidence=value)
                self.assertIs(score.verdict, Verdict.INSUFFICIENT)
                self.assertIs(score.downgraded_from, Verdict.FAIL)
                self.assertIn(
                    f"evidence must be an array of refs, got {type(value).__name__}",
                    score.downgrade_reason,
                )
                self.assertNotIn(" and ", score.downgrade_reason, "one field, one refusal")
                self.assertLess(len(score.downgrade_reason), 200, score.downgrade_reason)

    def test_a_null_evidence_field_reads_as_no_evidence(self) -> None:
        score = self._one(evidence=None)
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        self.assertIn("needs >= 1 evidence ref(s), got 0", score.downgrade_reason)

    def test_the_control_shows_an_array_of_refs_still_resolves(self) -> None:
        conv = conversation("failing.json")
        item = {"criterion_id": self.cid, "verdict": "FAIL", "rationale": "r"}
        item["evidence"] = [good_ref(conv)]
        score = card([item], conv=conv).score_for(self.cid)
        self.assertIs(score.verdict, Verdict.FAIL)
        self.assertEqual(len(score.evidence), 1)

    def test_many_malformed_refs_are_summarised_not_spelled_out(self) -> None:
        """The reason lands in the scorecard and the failure record; the model
        decides how many refs there are, so the count must not decide the size."""
        from agent_skill_improver.domain.scoring import MAX_SHOWN_BAD_REFS

        score = self._one(evidence=["not-a-ref"] * 50)
        self.assertIs(score.verdict, Verdict.INSUFFICIENT)
        reason = score.downgrade_reason
        self.assertEqual(reason.count("evidence ref must be an object"), MAX_SHOWN_BAD_REFS)
        self.assertIn(f" and {50 - MAX_SHOWN_BAD_REFS} more", reason)
        few = self._one(evidence=["not-a-ref"] * MAX_SHOWN_BAD_REFS)
        self.assertNotIn(" more", few.downgrade_reason)


class ConfidenceFieldTests(unittest.TestCase):
    """``confidence`` is stored with the score, never decided on, and the
    model's to fill. Measured before: an integer too large for a float raised
    ``OverflowError`` out of the parser, and ``true`` was stored as 1.0. A reply
    carrying one is now refused whole by ``loads_untrusted``; these cards are
    built in code, which no parse has seen."""

    def _confidence_of(self, value):
        cid = rubric().criterion_ids[0]
        item = {"criterion_id": cid, "verdict": "FAIL", "rationale": "r", "confidence": value}
        return card([item]).score_for(cid).confidence

    def test_a_confidence_too_large_for_a_float_is_none_not_a_crash(self) -> None:
        for value in (10**400, -(10**400)):
            with self.subTest(sign="-" if value < 0 else "+"):
                self.assertIsNone(self._confidence_of(value))

    def test_a_confidence_that_is_not_a_number_is_none(self) -> None:
        for value in (True, False, "0.9", [0.9], {"v": 0.9}, None):
            with self.subTest(value=repr(value)):
                self.assertIsNone(self._confidence_of(value))

    def test_the_control_a_numeric_confidence_is_kept_as_a_float(self) -> None:
        self.assertEqual(self._confidence_of(0.75), 0.75)
        self.assertEqual(self._confidence_of(1), 1.0)
        self.assertIsInstance(self._confidence_of(1), float)


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

    def test_pathologically_nested_output_is_malformed_not_a_crash(self) -> None:
        """``json.loads`` raises RecursionError on deep nesting; that must be a
        record defect like any other unparseable reply, not an escape from the
        per-record handlers that aborts the pass."""
        bomb = "{" + '"a":' * 50_000 + "1" + "}" * 50_000
        with self.assertRaises(MalformedModelOutput):
            extract_json_object(bomb)
        wrapped = "Here you go:\n" + "[" * 100_000 + "]" * 100_000
        with self.assertRaises(MalformedModelOutput):
            extract_json_object(wrapped)

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

    def test_the_diagnostic_quotes_only_the_documented_head_of_the_output(self) -> None:
        """The quote lands in a quarantine record, so its length is a privacy
        bound the docs state, not a formatting choice."""
        from agent_skill_improver.adapters.bounded_process import QUOTED_OUTPUT_CHARS

        head = "H" * QUOTED_OUTPUT_CHARS
        with self.assertRaises(MalformedModelOutput) as ctx:
            extract_json_object(head + " PRIVATE-TAIL " + "T" * 2000)
        message = str(ctx.exception)
        self.assertIn(f"first {QUOTED_OUTPUT_CHARS} chars: ", message)
        self.assertIn(repr(head), message)
        self.assertNotIn("PRIVATE-TAIL", message)
        self.assertNotIn("T" * 10, message)

    def test_the_quoted_head_is_redacted_before_it_lands_in_a_quarantine_record(self) -> None:
        planted = "".join(("AKIA", "ABCDEFGHIJKLMNOP"))
        with self.assertRaises(MalformedModelOutput) as ctx:
            extract_json_object("Here is the key you asked for: " + planted)
        message = str(ctx.exception)
        self.assertIn("[REDACTED:aws-access-key-id]", message)
        self.assertNotIn(planted, message)

    def test_an_unbalanced_object_is_refused_rather_than_repaired(self) -> None:
        with self.assertRaises(MalformedModelOutput):
            extract_json_object('{"a": 1, "b": ')


if __name__ == "__main__":
    unittest.main()
