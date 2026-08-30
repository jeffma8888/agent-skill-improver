"""The promotion gate, and the human interlock behind it.

The gate decides whether a candidate may be *offered* to a human. It is a floor on
evidence, not a substitute for judgement, and the interesting requirement is that
its numbers stay meaningful: a judge that shares a model with the improver is
grading its own homework, and the resulting agreement measures self-consistency
while looking exactly like evidence.
"""

from __future__ import annotations

import os
import unittest

from agent_skill_improver.domain.approval import (
    AUTOMATION_ENV_VAR,
    AutomationMarker,
    HumanApproval,
    automation_active,
)
from agent_skill_improver.domain.candidate import approval_token_for, challenge_for
from agent_skill_improver.domain.rubric import Verdict
from agent_skill_improver.domain.validation import (
    ARM_BASELINE,
    ARM_CANDIDATE,
    CASE_HOLDOUT,
    CASE_TARGET,
    GatePolicy,
    Judgement,
    ReplayCase,
    ReplayOutcome,
    blind_arms,
    evaluate_gate,
)
from agent_skill_improver.errors import ApprovalRequired, AutomationRefused


def outcome(name: str, baseline: Verdict, candidate: Verdict, kind: str = CASE_TARGET):
    return ReplayOutcome(
        case=ReplayCase(conversation_id=name, criterion_id="c1", kind=kind),
        baseline=baseline,
        candidate=candidate,
    )


def judgement(name: str, arm: str = ARM_CANDIDATE, label: str = "A"):
    return Judgement(
        case=ReplayCase(conversation_id=name, criterion_id="c1", kind=CASE_TARGET),
        preferred_label=label,
        preferred_arm=arm,
        rationale="because",
    )


FIX = outcome("t1", Verdict.FAIL, Verdict.PASS)
HOLDOUT_OK = outcome("h1", Verdict.PASS, Verdict.PASS, CASE_HOLDOUT)


class GateArithmeticTests(unittest.TestCase):
    def test_a_clean_fix_with_a_holdout_and_a_judge_passes(self):
        decision = evaluate_gate([FIX, HOLDOUT_OK], [judgement("t1")], GatePolicy())
        self.assertTrue(decision.passed, decision.reasons)

    def test_fail_to_insufficient_earns_no_credit(self):
        unresolved = outcome("t1", Verdict.FAIL, Verdict.INSUFFICIENT)
        decision = evaluate_gate([unresolved, HOLDOUT_OK], [judgement("t1")], GatePolicy())
        self.assertFalse(
            decision.passed,
            "a change that makes the scorer unable to decide is not a fix",
        )
        self.assertTrue(any("INSUFFICIENT" in r for r in decision.reasons))

    def test_a_regression_fails_the_gate(self):
        regression = outcome("h2", Verdict.PASS, Verdict.FAIL, CASE_HOLDOUT)
        decision = evaluate_gate([FIX, HOLDOUT_OK, regression], [judgement("t1")], GatePolicy())
        self.assertFalse(decision.passed)
        self.assertTrue(any("regression" in r for r in decision.reasons))

    def test_a_holdout_degradation_is_not_treated_as_neutral(self):
        degraded = outcome("h2", Verdict.PASS, Verdict.INSUFFICIENT, CASE_HOLDOUT)
        decision = evaluate_gate([FIX, HOLDOUT_OK, degraded], [judgement("t1")], GatePolicy())
        self.assertFalse(
            decision.passed,
            "PASS -> INSUFFICIENT on a holdout is a loss of evidence, not a tie",
        )

    def test_no_holdout_means_the_no_regression_claim_is_unsupported(self):
        decision = evaluate_gate([FIX], [judgement("t1")], GatePolicy())
        self.assertFalse(decision.passed)
        self.assertTrue(any("holdout" in r for r in decision.reasons))

    def test_replaying_nothing_is_not_a_pass(self):
        decision = evaluate_gate([], [], GatePolicy())
        self.assertFalse(decision.passed)
        self.assertTrue(any("nothing was measured" in r for r in decision.reasons))

    def test_a_judge_preferring_the_baseline_fails_the_ratio(self):
        decision = evaluate_gate([FIX, HOLDOUT_OK], [judgement("t1", ARM_BASELINE)], GatePolicy())
        self.assertFalse(decision.passed)

    def test_ties_are_excluded_from_the_ratio_rather_than_counted_as_wins(self):
        decision = evaluate_gate(
            [FIX, HOLDOUT_OK],
            [judgement("t1", "tie"), judgement("t2", ARM_CANDIDATE)],
            GatePolicy(min_judge_preference_ratio=1.0),
        )
        self.assertTrue(decision.passed, decision.reasons)

    def test_only_ties_means_no_decided_judgements_and_no_pass(self):
        decision = evaluate_gate([FIX, HOLDOUT_OK], [judgement("t1", "tie")], GatePolicy())
        self.assertFalse(decision.passed, "a ratio over zero decided judgements is not evidence")

    def test_every_failing_reason_is_reported_not_just_the_first(self):
        regression = outcome("h2", Verdict.PASS, Verdict.FAIL, CASE_HOLDOUT)
        decision = evaluate_gate(
            [outcome("t1", Verdict.FAIL, Verdict.INSUFFICIENT), regression],
            [judgement("t1", ARM_BASELINE)],
            GatePolicy(),
        )
        self.assertGreaterEqual(
            len(decision.reasons),
            3,
            "stopping at the first reason makes a human fix one thing at a time",
        )

    def test_the_policy_is_recorded_with_the_decision(self):
        decision = evaluate_gate([FIX, HOLDOUT_OK], [judgement("t1")], GatePolicy())
        self.assertEqual(decision.to_dict()["policy"]["min_fixes"], 1)


class CorrelatedEvaluatorTests(unittest.TestCase):
    """The quiet failure: the gate still emits a passing verdict."""

    def test_a_non_independent_judge_fails_the_gate_by_default(self):
        decision = evaluate_gate(
            [FIX, HOLDOUT_OK], [judgement("t1")], GatePolicy(), judge_independent=False
        )
        self.assertFalse(decision.passed)
        self.assertTrue(
            any("same command as the improver" in r for r in decision.reasons),
            decision.reasons,
        )

    def test_the_independence_requirement_can_be_switched_off_explicitly(self):
        decision = evaluate_gate(
            [FIX, HOLDOUT_OK],
            [judgement("t1")],
            GatePolicy(require_independent_judge=False),
            judge_independent=False,
        )
        self.assertTrue(
            decision.passed,
            "an operator may accept the weaker evidence, but only by saying so in "
            "configuration rather than by the check quietly not existing",
        )

    def test_independence_is_recorded_in_the_metrics_either_way(self):
        for independent in (True, False):
            with self.subTest(independent=independent):
                decision = evaluate_gate(
                    [FIX, HOLDOUT_OK],
                    [judgement("t1")],
                    GatePolicy(require_independent_judge=False),
                    judge_independent=independent,
                )
                self.assertEqual(decision.metrics["judge_independent"], independent)

    def test_dropping_the_judge_entirely_also_drops_the_independence_check(self):
        # Consistency: if no judge is required, there is no judge to be correlated.
        decision = evaluate_gate(
            [FIX, HOLDOUT_OK], [], GatePolicy(require_judge=False), judge_independent=False
        )
        self.assertTrue(decision.passed, decision.reasons)


class BlindJudgingTests(unittest.TestCase):
    def test_arms_are_assigned_from_the_seed_and_are_stable(self):
        first = blind_arms("conv::c1", "seed-a")
        self.assertEqual(first, blind_arms("conv::c1", "seed-a"))

    def test_both_arms_are_always_present_exactly_once(self):
        arms = blind_arms("conv::c1", "seed-a")
        self.assertEqual(sorted(arms.values()), sorted([ARM_BASELINE, ARM_CANDIDATE]))

    def test_the_assignment_varies_across_cases_so_position_is_not_a_signal(self):
        seen = {tuple(sorted(blind_arms(f"c{i}::x", "s").items())) for i in range(40)}
        self.assertGreater(
            len(seen), 1, "a constant assignment would let position stand in for quality"
        )


class ApprovalTokenTests(unittest.TestCase):
    def test_the_token_is_bound_to_the_candidate_the_base_and_the_patch(self):
        challenge = challenge_for("CAND-1", "b" * 64, "p" * 64)
        token = approval_token_for(challenge, "b" * 64, "p" * 64)
        self.assertTrue(token)
        self.assertNotEqual(
            token,
            approval_token_for(challenge_for("CAND-2", "b" * 64, "p" * 64), "b" * 64, "p" * 64),
        )

    def test_changing_the_base_hash_invalidates_the_token(self):
        c1 = challenge_for("CAND-1", "b" * 64, "p" * 64)
        first = approval_token_for(c1, "b" * 64, "p" * 64)
        c2 = challenge_for("CAND-1", "c" * 64, "p" * 64)
        second = approval_token_for(c2, "c" * 64, "p" * 64)
        self.assertNotEqual(
            first, second, "the base file changed, so the reviewed evidence no longer applies"
        )

    def test_changing_the_patch_invalidates_the_token(self):
        c1 = challenge_for("CAND-1", "b" * 64, "p" * 64)
        c2 = challenge_for("CAND-1", "b" * 64, "q" * 64)
        self.assertNotEqual(
            approval_token_for(c1, "b" * 64, "p" * 64),
            approval_token_for(c2, "b" * 64, "q" * 64),
        )

    def test_derivation_is_deterministic(self):
        self.assertEqual(
            approval_token_for("chal", "b" * 64, "p" * 64),
            approval_token_for("chal", "b" * 64, "p" * 64),
        )

    def test_a_wrong_token_is_refused_with_a_reason_that_says_why(self):
        approval = HumanApproval(token="deadbeef", acknowledged=True, origin="cli")
        with self.assertRaises(ApprovalRequired) as ctx:
            approval.verify(expected_token="cafef00d")
        self.assertIn("does not match this candidate", str(ctx.exception))

    def test_a_non_cli_origin_is_refused(self):
        approval = HumanApproval(token="t", acknowledged=True, origin="webhook")
        with self.assertRaises(ApprovalRequired):
            approval.verify(expected_token="t")

    def test_the_token_is_not_written_into_the_receipt(self):
        payload = HumanApproval(token="secretish", acknowledged=True, origin="cli").to_dict()
        self.assertNotIn("secretish", repr(payload))
        self.assertFalse(payload["token_recorded"])


class AutomationInterlockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.pop(AUTOMATION_ENV_VAR, None)

    def tearDown(self) -> None:
        os.environ.pop(AUTOMATION_ENV_VAR, None)
        if self._saved is not None:
            os.environ[AUTOMATION_ENV_VAR] = self._saved

    def test_no_marker_means_no_automation(self):
        self.assertFalse(automation_active())

    def test_the_marker_is_detected(self):
        with AutomationMarker():
            self.assertTrue(automation_active())
        self.assertFalse(automation_active(), "the marker must be restored on exit")

    def test_falsey_marker_values_do_not_count_as_automation(self):
        for value in ("", "0", "false", "False"):
            with self.subTest(value=value):
                os.environ[AUTOMATION_ENV_VAR] = value
                self.assertFalse(automation_active())

    def test_approval_from_cli_is_refused_while_the_marker_is_set(self):
        with AutomationMarker():
            with self.assertRaises(AutomationRefused):
                HumanApproval.from_cli(token="anything", acknowledged=True)

    def test_approval_from_cli_requires_the_acknowledgement(self):
        with self.assertRaises(ApprovalRequired):
            HumanApproval.from_cli(token="anything", acknowledged=False)

    def test_approval_from_cli_requires_a_token(self):
        with self.assertRaises(ApprovalRequired):
            HumanApproval.from_cli(token="", acknowledged=True)

    def test_a_pre_existing_marker_value_is_restored_not_clobbered(self):
        os.environ[AUTOMATION_ENV_VAR] = "outer"
        with AutomationMarker():
            self.assertTrue(automation_active())
        self.assertEqual(os.environ[AUTOMATION_ENV_VAR], "outer")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
