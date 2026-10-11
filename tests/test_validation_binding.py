"""A validation is evidence about *named* failures, or it is not evidence.

Three defects were measured on a green suite, and each of them let a validation
record be re-attributed rather than forged outright:

* a target outcome was bound to the candidate only by
  ``conversation_id::criterion_id``. Rewriting ``rubric_id`` on the outcome and
  on the top-level binding to another rubric that is live with a matching
  version and source hash produced no complaint at all, although the
  candidate's failure ref still named the original rubric. Evidence from a
  different examiner counted as evidence for the claim.
* the claim set was never compared with the candidate. A candidate could gain or
  lose a failure ref while the persisted target set stayed as it was, so what
  the gate proved and what the record claimed drifted apart silently.
* ``model_calls`` was checked only against a lower bound, so any larger number
  passed. A field that cannot be recomputed is an assertion sitting next to
  evidence, and this one advertised work nobody did.

Two producer defects made the writer emit records its own strict parser
rejects: rubric bindings were derived from every *attempted* case rather than
from the outcomes that persisted, and the judge ran over every attempted target,
so one failed replay produced an unused binding and a judgement about a case
with no outcome. Worse, the gate still passed: a validated candidate whose
target set was incomplete, which review then had to call tampered.

Every test below is either a refusal that must happen or a healthy control that
must keep working. The CLI class drives the real process, because the property
that matters is that no token, backup, receipt or skill write is produced.
"""

from __future__ import annotations

import json
import re
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

from agent_skill_improver.cli import RC_OK
from agent_skill_improver.app.validate import validate_candidate
from agent_skill_improver.config import PromptConfig
from agent_skill_improver.domain.candidate import (
    STATUS_GATE_FAILED,
    STATUS_PROPOSED,
    STATUS_VALIDATED,
    RubricFailureRef,
)
from agent_skill_improver.domain.conversation import NormalizedConversation
from agent_skill_improver.domain.integrity import validation_mismatches, verify_full_integrity
from agent_skill_improver.domain.rubric import Criterion, Rubric, Verdict
from agent_skill_improver.domain.validation import (
    ARM_CANDIDATE,
    CASE_HOLDOUT,
    CASE_TARGET,
    GatePolicy,
    Judgement,
    ReplayCase,
    ReplayOutcome,
    RubricBinding,
    ValidationRecord,
    blind_arms,
    evidence_call_count,
)
from agent_skill_improver.errors import CandidateTampered, TransientFailure
from agent_skill_improver.hashing import sha256_file, sha256_text

from tests.support import FIX_MARKER, build_env, run_cli
from tests.test_candidate_integrity import BASE_TEXT, build_candidate, build_validation

SEED = "integrity-test-seed"
FIX_LINE = "watch it fail for the right reason"

IC = "instruction-compliance"
OTHER = "other-rubric"


# -- shared fixtures -----------------------------------------------------
def rubric(rubric_id: str, *, criteria: Sequence[str] = ("c1",), version: int = 1) -> Rubric:
    """A live rubric whose source hash is the one the record helper binds."""
    return Rubric(
        id=rubric_id,
        version=version,
        title=rubric_id,
        criteria=tuple(
            Criterion(id=c, title=c, question=f"is {c} satisfied?", requires_evidence=False)
            for c in criteria
        ),
        source_sha256=sha256_text(f"{rubric_id} rubric source"),
    )


#: The substituted rubric is deliberately *live*, versioned and hash-matched,
#: and it really does grade ``c1``. A refusal here cannot come from the rubric
#: being unknown, stale or unrelated: only the triple binding can produce it.
LIVE_RUBRICS = {IC: rubric(IC), OTHER: rubric(OTHER)}
OTHER_SOURCE_SHA = LIVE_RUBRICS[OTHER].source_sha256


def ref(rubric_id: str, conversation_id: str, criterion_id: str, *, version: int = 1):
    return RubricFailureRef(
        failure_case_id=f"FC-{conversation_id}-{criterion_id}",
        rubric_id=rubric_id,
        rubric_version=version,
        criterion_id=criterion_id,
        conversation_id=conversation_id,
    )


def target(rubric_id: str, conversation_id: str, criterion_id: str) -> ReplayOutcome:
    return ReplayOutcome(
        case=ReplayCase(conversation_id, criterion_id, CASE_TARGET),
        baseline=Verdict.FAIL,
        candidate=Verdict.PASS,
        rubric_id=rubric_id,
    )


def holdout(rubric_id: str, conversation_id: str, criterion_id: str) -> ReplayOutcome:
    return ReplayOutcome(
        case=ReplayCase(conversation_id, criterion_id, CASE_HOLDOUT),
        baseline=Verdict.PASS,
        candidate=Verdict.PASS,
        rubric_id=rubric_id,
    )


def binding(rubric_id: str, *, version: int = 1) -> RubricBinding:
    return RubricBinding(
        rubric_id=rubric_id,
        version=version,
        source_sha256=sha256_text(f"{rubric_id} rubric source"),
    )


def judgement_for(case: ReplayCase, *, seed: str = SEED, arm: str = ARM_CANDIDATE) -> Judgement:
    arms = blind_arms(case.key, seed)
    label = next(lbl for lbl, mapped in arms.items() if mapped == arm)
    return Judgement(case=case, preferred_label=label, preferred_arm=arm, rationale="because")


def candidate_for(
    refs,
    *,
    outcomes,
    judgements=(),
    rubrics=None,
    status: str = STATUS_VALIDATED,
    policy: GatePolicy | None = None,
    target_refs=None,
    evidence_model_calls: int | None = None,
    judge_command_id: str = "",
):
    """A candidate whose every derived field, gate and record really is derived."""
    base = replace(build_candidate(validation={}), rubric_failures=tuple(refs), status=status)
    record = build_validation(
        base,
        outcomes=tuple(outcomes),
        judgements=tuple(judgements),
        rubrics=rubrics if rubrics is not None else (binding(IC),),
        policy=policy,
        target_refs=target_refs,
        evidence_model_calls=evidence_model_calls,
        judge_command_id=judge_command_id,
    )
    return replace(base, validation=record)


def reblind(judgement: dict, *, seed: str) -> None:
    """Relabel a moved judgement so its arm still unblinds correctly.

    The blind mapping is a function of the case key and the seed, so moving a
    judgement to another case makes the recorded label/arm pair inconsistent and
    the arm check refuses first. Re-deriving the label keeps the record
    well-formed in every other respect, which is what leaves the
    judgement-without-an-outcome rule as the only thing that can refuse.
    """
    arms = blind_arms(judgement["case"]["key"], seed)
    judgement["preferred_label"] = next(
        label for label, arm in arms.items() if arm == judgement["preferred_arm"]
    )


def tampered(candidate, mutate):
    """A deep copy of the persisted validation with exactly one thing changed."""
    raw = json.loads(json.dumps(candidate.validation))
    mutate(raw)
    return replace(candidate, validation=raw)


ONE_REF = (ref(IC, "t1", "c1"),)
ONE_TARGET = (target(IC, "t1", "c1"),)


class TargetRefBindingTests(unittest.TestCase):
    """The (rubric_id, conversation_id, criterion_id) triple, in both directions."""

    def setUp(self) -> None:
        self.candidate = candidate_for(ONE_REF, outcomes=ONE_TARGET)

    def test_a_clean_record_reports_no_mismatch_with_the_live_rubrics(self):
        self.assertEqual(validation_mismatches(self.candidate, rubrics=LIVE_RUBRICS), [])

    def test_a_clean_record_reports_no_mismatch_without_the_live_rubrics(self):
        self.assertEqual(validation_mismatches(self.candidate), [])

    def test_substituting_a_live_matching_rubric_on_the_outcome_is_refused(self):
        def mutate(raw):
            raw["outcomes"][0]["rubric_id"] = OTHER
            raw["rubrics"] = [{"rubric_id": OTHER, "version": 1, "source_sha256": OTHER_SOURCE_SHA}]

        problems = validation_mismatches(tampered(self.candidate, mutate), rubrics=LIVE_RUBRICS)
        self.assertTrue(problems, "a re-attributed target outcome must be refused")
        self.assertTrue(
            any("target_refs" in p for p in problems),
            f"the refusal must name the claim-set binding, got {problems}",
        )

    def test_substituting_the_rubric_on_the_claim_set_too_is_still_refused(self):
        """The deeper forgery: the record is made self-consistent first.

        Rewriting the outcome, the binding *and* the claim set produces a record
        that parses cleanly on its own, so the only thing left to catch it is the
        comparison against the candidate's own failure refs. That comparison is
        the point of this test, so the parse is asserted to succeed first --
        otherwise a refusal here could come from a malformed record and the real
        binding would still be untested.
        """

        def mutate(raw):
            raw["outcomes"][0]["rubric_id"] = OTHER
            raw["target_refs"] = [
                {"rubric_id": OTHER, "conversation_id": "t1", "criterion_id": "c1"}
            ]
            raw["rubrics"] = [{"rubric_id": OTHER, "version": 1, "source_sha256": OTHER_SOURCE_SHA}]

        forged = tampered(self.candidate, mutate)
        ValidationRecord.parse(forged.validation)  # must not raise: it is well-formed
        problems = validation_mismatches(forged, rubrics=LIVE_RUBRICS)
        self.assertTrue(
            any("is not exactly this candidate's own rubric_failures" in p for p in problems),
            f"the candidate/record binding must refuse the substitution, got {problems}",
        )

    def test_an_added_candidate_failure_ref_invalidates_the_record(self):
        added = replace(
            self.candidate,
            rubric_failures=self.candidate.rubric_failures + (ref(IC, "t2", "c1"),),
        )
        problems = validation_mismatches(added, rubrics=LIVE_RUBRICS)
        self.assertTrue(
            any(f"{IC}::t2::c1" in p for p in problems),
            f"an unmeasured claimed failure must be named, got {problems}",
        )

    def test_an_omitted_candidate_failure_ref_invalidates_the_record(self):
        two = candidate_for(
            ONE_REF + (ref(IC, "t2", "c1"),),
            outcomes=ONE_TARGET + (target(IC, "t2", "c1"),),
        )
        omitted = replace(two, rubric_failures=ONE_REF)
        problems = validation_mismatches(omitted, rubrics=LIVE_RUBRICS)
        self.assertTrue(
            any("not claimed by the candidate" in p for p in problems),
            f"a target the candidate no longer claims must be refused, got {problems}",
        )

    def test_adding_a_ref_to_both_sides_without_an_outcome_fails_the_recomputed_gate(self):
        """The remaining direction: keep record and candidate in step, add no evidence."""
        extra = ref(IC, "t2", "c1")
        added = replace(self.candidate, rubric_failures=self.candidate.rubric_failures + (extra,))

        def mutate(raw):
            raw["target_refs"].append(
                {"rubric_id": IC, "conversation_id": "t2", "criterion_id": "c1"}
            )

        forged = tampered(added, mutate)
        problems = validation_mismatches(forged, rubrics=LIVE_RUBRICS)
        self.assertTrue(
            any(p.startswith("validation.gate") for p in problems),
            f"the recomputed gate must stop matching, got {problems}",
        )
        self.assertTrue(
            any("no persisted replay outcome" in p for p in problems),
            f"the missing coverage must be named, got {problems}",
        )

    def test_an_invented_target_outcome_is_refused_by_the_record_itself(self):
        def mutate(raw):
            invented = json.loads(json.dumps(raw["outcomes"][0]))
            invented["case"]["conversation_id"] = "t9"
            invented["case"]["key"] = "t9::c1"
            raw["outcomes"].append(invented)

        problems = validation_mismatches(tampered(self.candidate, mutate), rubrics=LIVE_RUBRICS)
        self.assertTrue(
            any("not among validation.target_refs" in p for p in problems),
            f"a target outcome outside the claim set must be refused, got {problems}",
        )

    def test_a_duplicate_claim_is_refused(self):
        def mutate(raw):
            raw["target_refs"].append(json.loads(json.dumps(raw["target_refs"][0])))

        problems = validation_mismatches(tampered(self.candidate, mutate), rubrics=LIVE_RUBRICS)
        self.assertTrue(
            any("duplicate ref" in p for p in problems),
            f"a duplicated claim must be refused, got {problems}",
        )

    def test_two_rubrics_may_grade_the_same_criterion_id_without_colliding(self):
        """Identity is the triple, so the same case key under two rubrics is legal."""
        candidate = candidate_for(
            (ref(IC, "t1", "c1"), ref(OTHER, "t1", "c1")),
            outcomes=(target(IC, "t1", "c1"), target(OTHER, "t1", "c1")),
            rubrics=(binding(IC), binding(OTHER)),
        )
        self.assertEqual(validation_mismatches(candidate, rubrics=LIVE_RUBRICS), [])

    def test_a_case_recorded_as_both_target_and_holdout_is_refused(self):
        def mutate(raw):
            both = json.loads(json.dumps(raw["outcomes"][0]))
            both["case"]["kind"] = CASE_HOLDOUT
            both["rubric_id"] = OTHER
            raw["outcomes"].append(both)
            raw["rubrics"].append(
                {"rubric_id": OTHER, "version": 1, "source_sha256": OTHER_SOURCE_SHA}
            )

        problems = validation_mismatches(tampered(self.candidate, mutate), rubrics=LIVE_RUBRICS)
        self.assertTrue(
            any("cannot be both" in p for p in problems),
            f"one case cannot be both the repair target and its own control, got {problems}",
        )

    def test_an_outcome_whose_named_rubric_cannot_grade_it_is_refused(self):
        live = {IC: rubric(IC), OTHER: rubric(OTHER, criteria=("c2",))}
        candidate = candidate_for(
            (ref(OTHER, "t1", "c1"),),
            outcomes=(target(OTHER, "t1", "c1"),),
            rubrics=(binding(OTHER),),
        )
        problems = validation_mismatches(candidate, rubrics=live)
        self.assertTrue(
            any("cannot have produced it" in p for p in problems),
            f"an outcome must be attributed to a rubric that grades it, got {problems}",
        )

    def test_a_failure_ref_version_that_disagrees_with_the_bound_rubric_is_refused(self):
        candidate = candidate_for(
            (ref(IC, "t1", "c1", version=2),),
            outcomes=ONE_TARGET,
            rubrics=(binding(IC, version=1),),
        )
        problems = validation_mismatches(candidate, rubrics=LIVE_RUBRICS)
        self.assertTrue(
            any("different version of the examiner" in p for p in problems),
            f"the ref and the binding must agree on the rubric version, got {problems}",
        )

    def test_a_judgement_about_a_case_with_no_target_outcome_is_refused(self):
        candidate = candidate_for(
            ONE_REF,
            outcomes=ONE_TARGET,
            judgements=(judgement_for(ReplayCase("t1", "c1", CASE_TARGET)),),
            policy=GatePolicy(min_holdout_cases=0, require_judge=True),
            judge_command_id="judge-cmd",
        )
        self.assertEqual(validation_mismatches(candidate, rubrics=LIVE_RUBRICS), [])

        def mutate(raw):
            judgement = raw["judgements"][0]
            judgement["case"]["conversation_id"] = "t9"
            judgement["case"]["key"] = "t9::c1"
            reblind(judgement, seed=raw["seed"])

        problems = validation_mismatches(tampered(candidate, mutate), rubrics=LIVE_RUBRICS)
        self.assertTrue(
            any("target outcomes" in p for p in problems),
            f"a judgement must be about a replayed target, got {problems}",
        )

    def test_a_gate_failed_record_with_an_unreplayed_claim_is_honest_not_tampered(self):
        """The error path must parse itself and must not be called a forgery."""
        candidate = candidate_for(
            ONE_REF + (ref(OTHER, "t2", "c2"),),
            outcomes=ONE_TARGET,
            status=STATUS_GATE_FAILED,
        )
        self.assertFalse(candidate.validation["gate"]["passed"])
        self.assertTrue(
            any(
                "no replay outcome" in reason for reason in candidate.validation["gate"]["reasons"]
            ),
            candidate.validation["gate"]["reasons"],
        )
        self.assertEqual(validation_mismatches(candidate, rubrics=LIVE_RUBRICS), [])

    def test_the_refusal_wrapper_names_the_action_and_the_binding(self):
        def mutate(raw):
            raw["outcomes"][0]["rubric_id"] = OTHER
            raw["rubrics"] = [{"rubric_id": OTHER, "version": 1, "source_sha256": OTHER_SOURCE_SHA}]

        with self.assertRaises(CandidateTampered) as ctx:
            verify_full_integrity(
                tampered(self.candidate, mutate),
                base_text=BASE_TEXT,
                rubrics=LIVE_RUBRICS,
                action="review",
            )
        self.assertIn("refusing to review", str(ctx.exception))


class EvidenceCallCountTests(unittest.TestCase):
    """The model-call count is derived from the record or it is not evidence."""

    def setUp(self) -> None:
        self.candidate = candidate_for(ONE_REF, outcomes=ONE_TARGET)

    def test_the_clean_count_is_exactly_two_calls_per_scored_pair(self):
        self.assertEqual(self.candidate.validation["evidence_model_calls"], 2)
        self.assertEqual(validation_mismatches(self.candidate, rubrics=LIVE_RUBRICS), [])

    def test_one_extra_call_is_refused(self):
        def mutate(raw):
            raw["evidence_model_calls"] += 1

        problems = validation_mismatches(tampered(self.candidate, mutate), rubrics=LIVE_RUBRICS)
        self.assertTrue(
            any("evidence_model_calls" in p for p in problems),
            f"a +1 tamper must be refused, got {problems}",
        )

    def test_one_missing_call_is_refused(self):
        def mutate(raw):
            raw["evidence_model_calls"] -= 1

        problems = validation_mismatches(tampered(self.candidate, mutate), rubrics=LIVE_RUBRICS)
        self.assertTrue(
            any("evidence_model_calls" in p for p in problems),
            f"an undercount must be refused too, got {problems}",
        )

    def test_a_second_criterion_on_a_scored_pair_costs_no_further_call(self):
        outcomes = (target(IC, "t1", "c1"), holdout(IC, "t1", "c2"))
        self.assertEqual(evidence_call_count(outcomes, ()), 2)

    def test_each_distinct_conversation_rubric_pair_costs_two_calls(self):
        outcomes = (target(IC, "t1", "c1"), target(OTHER, "t2", "c1"))
        self.assertEqual(evidence_call_count(outcomes, ()), 4)

    def test_each_judgement_costs_one_call(self):
        case = ReplayCase("t1", "c1", CASE_TARGET)
        self.assertEqual(evidence_call_count(ONE_TARGET, (judgement_for(case),)), 3)


# -- the producer --------------------------------------------------------
CONVERSATIONS = {
    "t1": NormalizedConversation(thread_id="t1", title="one"),
    "t2": NormalizedConversation(thread_id="t2", title="two"),
}
PRODUCER_RUBRICS = {
    IC: rubric(IC, criteria=("c1", "h1")),
    OTHER: rubric(OTHER, criteria=("c2",)),
}


class ReplayRunner:
    """A scorer and judge double that answers from the prompt, not from disk.

    Answering from the prompt is what makes it a real exercise of the writer:
    the arm is recognised only because the candidate text really is substituted
    into the scorer prompt, and the blind label only because both options really
    are rendered for the judge.
    """

    def __init__(self, *, fail_conversations: Sequence[str] = ()) -> None:
        self.fail_conversations = set(fail_conversations)
        self.calls: list[tuple[str, str]] = []

    def command_id(self, role: str) -> str:
        return f"{role}-command"

    def run(self, *, role: str, system: str, user: str, response_schema: str) -> Any:
        thread = self._thread(user)
        self.calls.append((role, thread))
        if role == "scorer":
            if thread in self.fail_conversations:
                raise TransientFailure(f"double refuses to score {thread}")
            fixed = FIX_LINE in user
            return {
                "schema": "asi.scorecard.v1",
                "conversation_id": thread,
                "criteria": [
                    {
                        "criterion_id": criterion_id,
                        "verdict": "PASS" if fixed else "FAIL",
                        "rationale": "double",
                    }
                    for criterion_id in self._criteria(user)
                ],
            }
        if role == "judge":
            return {
                "schema": "asi.judgement.v1",
                "preferred": self._fixed_label(user),
                "rationale": "one option states the rule explicitly",
            }
        raise AssertionError(f"unexpected role {role!r}")

    @staticmethod
    def _thread(user: str) -> str:
        scored = re.search(r"Score conversation (\S+) against", user)
        if scored:
            return scored.group(1)
        judged = re.search(r"^conversation_id: (\S+)$", user, re.M)
        return judged.group(1) if judged else ""

    @staticmethod
    def _criteria(user: str) -> list[str]:
        block = re.search(r"## RUBRIC \(trusted\)\n(.*?)(?=\n## )", user, re.S)
        return re.findall(r"^- id: (\S+)$", block.group(1), re.M) if block else []

    @staticmethod
    def _fixed_label(user: str) -> str:
        for label in ("A", "B"):
            marker = f"## OPTION {label} "
            if marker not in user:
                continue
            section = user.split(marker, 1)[1].split("## OPTION ", 1)[0]
            if FIX_LINE in section:
                return label
        return "tie"


def producer_candidate(refs):
    """A proposed candidate, exactly as the daily driver hands one to validation."""
    base = build_candidate(status=STATUS_PROPOSED, validation={})
    return replace(base, rubric_failures=tuple(refs), validation=None)


def run_producer(refs, *, runner: ReplayRunner, conversations=None, rubrics=None, holdouts=()):
    return validate_candidate(
        producer_candidate(refs),
        runner,
        rubrics=rubrics if rubrics is not None else PRODUCER_RUBRICS,
        conversations=conversations if conversations is not None else CONVERSATIONS,
        base_text=BASE_TEXT,
        cfg=PromptConfig(),
        policy=GatePolicy(min_holdout_cases=0, require_judge=True),
        seed="producer-seed",
        holdout_pairs=holdouts,
    )


class ProducerWritesRecordsThatParseThemselvesTests(unittest.TestCase):
    """Whatever happens to a replay, the record written must be readable back."""

    def test_a_full_replay_passes_the_gate_and_binds_exactly_what_it_claims(self):
        runner = ReplayRunner()
        result = run_producer(
            (ref(IC, "t1", "c1"), ref(OTHER, "t2", "c2")),
            runner=runner,
            holdouts=(("t1", "h1"),),
        )
        self.assertTrue(result.decision.passed, result.decision.reasons)
        record = ValidationRecord.parse(result.to_dict())
        self.assertEqual(
            sorted(r.key for r in record.target_refs),
            [f"{IC}::t1::c1", f"{OTHER}::t2::c2"],
        )
        self.assertEqual(sorted(b.rubric_id for b in record.rubrics), [IC, OTHER])
        self.assertEqual(len(record.judgements), 2)
        self.assertEqual(
            record.evidence_model_calls,
            len(runner.calls),
            "with nothing failing, every call made is a call behind the evidence",
        )
        self.assertEqual([n for n in record.notes if "attempted" in n], [])

    def test_a_failed_target_replay_fails_the_gate_and_writes_a_readable_record(self):
        runner = ReplayRunner(fail_conversations=("t2",))
        result = run_producer((ref(IC, "t1", "c1"), ref(OTHER, "t2", "c2")), runner=runner)
        self.assertFalse(
            result.decision.passed,
            "an unavailable target replay must fail the gate here, not at review time",
        )
        self.assertTrue(
            any("no replay outcome" in r for r in result.decision.reasons),
            result.decision.reasons,
        )
        record = ValidationRecord.parse(result.to_dict())
        self.assertEqual(
            sorted(r.key for r in record.target_refs),
            [f"{IC}::t1::c1", f"{OTHER}::t2::c2"],
            "a claim is recorded whether or not it could be measured",
        )

    def test_a_partial_multi_rubric_replay_binds_only_the_rubric_that_produced_evidence(self):
        runner = ReplayRunner(fail_conversations=("t2",))
        result = run_producer((ref(IC, "t1", "c1"), ref(OTHER, "t2", "c2")), runner=runner)
        self.assertEqual(
            [b.rubric_id for b in result.rubrics],
            [IC],
            "a rubric that produced no outcome must not be bound: there is nothing to check it "
            "against, and the record's own parser rejects the unused binding",
        )
        ValidationRecord.parse(result.to_dict())

    def test_no_judgement_is_written_for_a_target_that_produced_no_outcome(self):
        runner = ReplayRunner(fail_conversations=("t2",))
        result = run_producer((ref(IC, "t1", "c1"), ref(OTHER, "t2", "c2")), runner=runner)
        self.assertEqual([j.case.key for j in result.judgements], ["t1::c1"])
        self.assertNotIn(
            "judge",
            [role for role, thread in runner.calls if thread == "t2"],
            "the judge must not be asked about a case with no replay outcome",
        )
        ValidationRecord.parse(result.to_dict())

    def test_the_call_count_is_the_evidence_count_and_the_attempts_are_a_note(self):
        runner = ReplayRunner(fail_conversations=("t2",))
        result = run_producer((ref(IC, "t1", "c1"), ref(OTHER, "t2", "c2")), runner=runner)
        self.assertEqual(
            result.evidence_model_calls, evidence_call_count(result.outcomes, result.judgements)
        )
        self.assertLess(result.evidence_model_calls, len(runner.calls))
        self.assertTrue(
            any("attempted" in note for note in result.notes),
            f"the attempted calls must still be reported honestly, got {result.notes}",
        )

    def test_a_target_naming_a_rubric_that_cannot_grade_it_is_never_substituted(self):
        """The producer half of the substitution defect.

        ``c1`` exists only in ``instruction-compliance``, and the ref names
        ``other-rubric``. The old resolver fell back to whichever rubric graded
        the criterion, producing an outcome attributed to a rubric the candidate
        never claimed. It must now refuse to measure the case at all.
        """
        runner = ReplayRunner()
        result = run_producer((ref(OTHER, "t1", "c1"),), runner=runner)
        self.assertEqual(result.outcomes, ())
        self.assertFalse(result.decision.passed)
        self.assertTrue(
            any("not loaded or does" in note for note in result.notes),
            result.notes,
        )
        self.assertEqual(runner.calls, [], "nothing may be scored for an unresolvable target")
        ValidationRecord.parse(result.to_dict())

    def test_a_target_whose_conversation_is_gone_fails_the_gate_for_want_of_coverage(self):
        runner = ReplayRunner()
        result = run_producer(
            (ref(IC, "t1", "c1"), ref(IC, "t9", "c1")),
            runner=runner,
            conversations={"t1": CONVERSATIONS["t1"]},
        )
        self.assertFalse(result.decision.passed)
        self.assertTrue(any(f"{IC}::t9::c1" in r for r in result.decision.reasons))
        ValidationRecord.parse(result.to_dict())

    def test_a_duplicated_failure_ref_is_replayed_once_and_claimed_once(self):
        runner = ReplayRunner()
        result = run_producer((ref(IC, "t1", "c1"), ref(IC, "t1", "c1")), runner=runner)
        self.assertEqual([r.key for r in result.target_refs], [f"{IC}::t1::c1"])
        self.assertEqual(len(result.outcomes), 1)
        ValidationRecord.parse(result.to_dict())


# -- the real process ----------------------------------------------------
class PromotionRefusesABrokenBindingEndToEnd(unittest.TestCase):
    """Two live rubrics, one real validated candidate, the real CLI.

    Every scenario rewrites one aspect of the persisted record and then asserts
    the whole promotion surface stays shut: no token, no apply command, no
    backup, no receipt, no byte written to the skill file, and no advance of the
    candidate's own status. The substituted rubric is a *live* rubric of the
    configured set, with the version and source hash it really has, so a refusal
    cannot come from the rubric being unknown or stale.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name)
        cls.config = build_env(
            cls.base,
            rubric_ids=("instruction-compliance", "verification-honesty"),
            dry_run=False,
            seed="binding-matrix-seed",
        )
        cls.cfg = ["--config", str(cls.config)]
        payload = cls._json("--json", "daily", "--commit", "--day", "2026-05-01")
        validated = [
            c["candidate_id"] for c in payload.get("candidates", []) if c["status"] == "validated"
        ]
        assert validated, payload.get("notes")
        cls.candidate_id = validated[0]
        review = cls._json("--json", "review", cls.candidate_id)
        cls.token = review["approval_token"]
        cls.clean_record = cls._record_path().read_text(encoding="utf-8")
        cls.clean_raw = json.loads(cls.clean_record)
        cls.skill_sha_before = sha256_file(cls._skill_path())
        cls.backups_before = cls._backups()
        cls.receipts_before = cls._receipts()
        # The two rubrics really are both live, and both are bound by the record.
        assert sorted(b["rubric_id"] for b in cls.clean_raw["validation"]["rubrics"]) == [
            "instruction-compliance",
            "verification-honesty",
        ], cls.clean_raw["validation"]["rubrics"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    @classmethod
    def _cli(cls, *args):
        return run_cli([*cls.cfg, *args])

    @classmethod
    def _json(cls, *args):
        proc = cls._cli(*args)
        assert proc.returncode == RC_OK, proc.stderr
        return json.loads(proc.stdout)

    @classmethod
    def _skill_path(cls) -> Path:
        return cls.base / "skills-canonical" / "tdd" / "SKILL.md"

    @classmethod
    def _record_path(cls) -> Path:
        return cls.base / "state" / "candidates" / f"{cls.candidate_id}.json"

    @classmethod
    def _backups(cls) -> list[str]:
        root = cls.base / "state" / "backups"
        return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.is_dir() else []

    @classmethod
    def _receipts(cls) -> list[str]:
        root = cls.base / "state" / "receipts"
        return sorted(p.name for p in root.glob("*.json")) if root.is_dir() else []

    def setUp(self) -> None:
        self._record_path().write_text(self.clean_record, encoding="utf-8")

    def tearDown(self) -> None:
        self._record_path().write_text(self.clean_record, encoding="utf-8")

    @staticmethod
    def _other_rubric(raw, rubric_id: str) -> str:
        others = [
            b["rubric_id"] for b in raw["validation"]["rubrics"] if b["rubric_id"] != rubric_id
        ]
        assert others, raw["validation"]["rubrics"]
        return others[0]

    def _first_target(self, raw) -> dict:
        for outcome in raw["validation"]["outcomes"]:
            if outcome["case"]["kind"] == "target":
                return outcome
        raise AssertionError("the fixture candidate has no target outcome")

    def assert_promotion_stays_shut(self, mutate, *, name: str, expect: str) -> None:
        raw = json.loads(json.dumps(self.clean_raw))
        mutate(raw)
        self._record_path().write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        review = self._cli("--json", "review", self.candidate_id)
        self.assertNotEqual(review.returncode, RC_OK, f"{name}: review must refuse")
        self.assertIn("CandidateTampered", review.stderr, f"{name}: refusal must be named")
        self.assertIn(expect, review.stderr, f"{name}: refusal must name the real reason")
        self.assertNotIn(self.token, review.stdout, f"{name}: no token may be printed")
        self.assertNotIn("apply_command", review.stdout, f"{name}: no apply command may be printed")
        self.assertNotIn("Traceback", review.stderr, f"{name}: no traceback")

        apply_proc = self._cli(
            "apply",
            self.candidate_id,
            "--approve",
            self.token,
            "--i-reviewed-this",
            "--operator",
            "tester",
        )
        self.assertNotEqual(apply_proc.returncode, RC_OK, f"{name}: apply must refuse")
        self.assertNotIn("Traceback", apply_proc.stderr, f"{name}: no traceback")
        self.assertEqual(
            sha256_file(self._skill_path()),
            self.skill_sha_before,
            f"{name}: the skill file must be untouched",
        )
        self.assertEqual(self._backups(), self.backups_before, f"{name}: no backup on refusal")
        self.assertEqual(self._receipts(), self.receipts_before, f"{name}: no receipt on refusal")
        on_disk = json.loads(self._record_path().read_text(encoding="utf-8"))
        self.assertEqual(on_disk["status"], raw["status"], f"{name}: status must not advance")
        self.assertNotIn("applied", " ".join(on_disk.get("notes", [])))

    def test_a_target_outcome_re_attributed_to_the_other_live_rubric_is_refused(self):
        def mutate(raw):
            outcome = self._first_target(raw)
            outcome["rubric_id"] = self._other_rubric(raw, outcome["rubric_id"])

        self.assert_promotion_stays_shut(
            mutate,
            name="target outcome re-attributed to a live rubric",
            expect="not among validation.target_refs",
        )

    def test_re_attributing_the_claim_set_as_well_is_still_refused(self):
        def mutate(raw):
            outcome = self._first_target(raw)
            substitute = self._other_rubric(raw, outcome["rubric_id"])
            for claim in raw["validation"]["target_refs"]:
                if (
                    claim["rubric_id"] == outcome["rubric_id"]
                    and claim["conversation_id"] == outcome["case"]["conversation_id"]
                    and claim["criterion_id"] == outcome["case"]["criterion_id"]
                ):
                    claim["rubric_id"] = substitute
            outcome["rubric_id"] = substitute

        self.assert_promotion_stays_shut(
            mutate,
            name="claim set re-attributed too",
            expect="is not exactly this candidate's own rubric_failures",
        )

    def test_an_added_candidate_failure_ref_is_refused(self):
        def mutate(raw):
            extra = json.loads(json.dumps(raw["rubric_failures"][0]))
            extra["failure_case_id"] = "FC-invented"
            extra["conversation_id"] = "1700000000009-9"
            raw["rubric_failures"].append(extra)

        self.assert_promotion_stays_shut(
            mutate,
            name="added candidate failure ref",
            expect="not bound by the validation",
        )

    def test_an_omitted_candidate_failure_ref_is_refused(self):
        def mutate(raw):
            raw["rubric_failures"] = raw["rubric_failures"][1:]

        self.assert_promotion_stays_shut(
            mutate,
            name="omitted candidate failure ref",
            expect="not claimed by the candidate",
        )

    def test_one_extra_evidence_model_call_is_refused(self):
        def mutate(raw):
            raw["validation"]["evidence_model_calls"] += 1

        self.assert_promotion_stays_shut(
            mutate,
            name="evidence_model_calls +1",
            expect="evidence_model_calls",
        )

    def test_a_judgement_about_an_unreplayed_case_is_refused(self):
        def mutate(raw):
            judgement = raw["validation"]["judgements"][0]
            judgement["case"]["conversation_id"] = "1700000000009-9"
            judgement["case"]["key"] = f"1700000000009-9::{judgement['case']['criterion_id']}"
            reblind(judgement, seed=raw["validation"]["seed"])

        self.assert_promotion_stays_shut(
            mutate,
            name="judgement about an unreplayed case",
            expect="target outcomes",
        )

    def test_zz_control_the_untampered_record_still_reviews_and_applies_with_two_rubrics(self):
        self._record_path().write_text(self.clean_record, encoding="utf-8")
        review = self._json("--json", "review", self.candidate_id)
        self.assertEqual(review["approval_token"], self.token)
        proc = self._cli(
            "apply",
            self.candidate_id,
            "--approve",
            self.token,
            "--i-reviewed-this",
            "--operator",
            "tester",
        )
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertIn(
            FIX_MARKER,
            self._skill_path().read_text(encoding="utf-8"),
            "the binding checks must not have broken the promotion path",
        )

    def test_zzz_control_the_daily_path_is_still_proposal_only(self):
        payload = self._json("--json", "daily", "--commit", "--day", "2026-05-02")
        tree = payload["skill_tree"]
        self.assertEqual(tree["digest_before"], tree["digest_after"])
        self.assertFalse(tree["mutation_detected"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
