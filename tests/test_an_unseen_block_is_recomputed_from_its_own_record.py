"""The ``unseen`` block of a v4 validation record is recomputed from the record itself.

The domain, parser and verifier tests of c240 (DESIGN v3, 4.3, file 11). *The golden
v4* is a record built here with ``ValidationResult(...)`` and ``to_dict()`` over the
pool of the CLI fixture: five refs on conv-human-0001, 0006, 0007 and 0008, the seed
``example-seed:CAND-2026-01-02-01-example-evidence-check``, N = 4 and k = 1, 0006's two
refs withheld with unseen outcomes FAIL -> FAIL and FAIL -> PASS. Every refusal test
asserts its message, not only that something was refused.
"""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

from agent_skill_improver import SCHEMA_VALIDATION, SCHEMA_VALIDATION_V3
from agent_skill_improver.app.validate import ValidationResult, validate_candidate
from agent_skill_improver.cli import _unseen_line
from agent_skill_improver.config import PromptConfig, load_config
from agent_skill_improver.domain.candidate import Candidate, RubricFailureRef
from agent_skill_improver.domain.integrity import validation_mismatches, verify_full_integrity
from agent_skill_improver.domain.rubric import Verdict
from agent_skill_improver.domain.validation import (
    CASE_HOLDOUT,
    CASE_TARGET,
    CASE_UNSEEN,
    UNSEEN_MEASURED,
    UNSEEN_SCOPE,
    UNSEEN_SHORT_OF_K,
    GatePolicy,
    ReplayCase,
    ReplayOutcome,
    RubricBinding,
    TargetCaseRef,
    UnseenSplit,
    ValidationRecord,
    evaluate_gate,
    evidence_call_count,
    split_unseen,
)
from agent_skill_improver.errors import CandidateTampered
from agent_skill_improver.hashing import sha256_text

from tests.test_validation_binding import (
    BASE_TEXT,
    CONVERSATIONS,
    FIX_LINE,
    IC,
    PRODUCER_RUBRICS,
    ReplayRunner,
    producer_candidate,
    ref,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "validation-v3-record.json"
LIVE = {r.id: r for r in load_config(REPO_ROOT / "config" / "example.config.toml").rubrics}

ED = "evidence-discipline"
CCE = "cites-checkable-evidence"
SMA = "separates-measured-from-assumed"
SEED = "example-seed:CAND-2026-01-02-01-example-evidence-check"


def conv(n: str) -> str:
    return f"conv-human-{n}"


CLAIMS = tuple(TargetCaseRef(ED, conv(n), CCE) for n in ("0001", "0007", "0008"))
W_CCE = TargetCaseRef(ED, conv("0006"), CCE)
W_SMA = TargetCaseRef(ED, conv("0006"), SMA)
POOL = CLAIMS + (W_CCE, W_SMA)


def local_rank(seed: str, conversation_id: str) -> str:
    """The hash order, computed here and not with the function under test."""
    return hashlib.sha256(f"{seed}\x00unseen\x00{conversation_id}".encode()).hexdigest()


def frozen_dict() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8").replace("{TREE}", "/fixture-tree"))


def outcome(r: TargetCaseRef, kind: str, base: Verdict, cand: Verdict) -> ReplayOutcome:
    return ReplayOutcome(
        case=ReplayCase(r.conversation_id, r.criterion_id, kind),
        baseline=base,
        candidate=cand,
        rubric_id=r.rubric_id,
    )


class UncountableUnseen(ReplayOutcome):
    """An unseen outcome that fails the test if the gate reads a verdict of it.

    The refusal has to come *above* the counting, and a test that only asserts
    the message cannot see that: moved below the counting the refusal still
    refuses every real unseen outcome, with the same message, after having
    counted it. So every verdict-derived property the counting reads raises
    here, which asserts the position itself. ``case`` stays readable, because
    the refusal message quotes its key; ``baseline`` and ``candidate`` stay real
    PASS -> FAIL, so a gate that counted this outcome would count a regression.
    """

    def _counted(self, name: str) -> NoReturn:
        raise AssertionError(
            f"evaluate_gate read {name!r} of an unseen outcome: the refusal has to come "
            "before any counting, not after it"
        )

    @property
    def ref(self) -> TargetCaseRef:
        self._counted("ref")

    @property
    def fixed(self) -> bool:
        self._counted("fixed")

    @property
    def regressed(self) -> bool:
        self._counted("regressed")

    @property
    def degraded(self) -> bool:
        self._counted("degraded")

    @property
    def unresolved(self) -> bool:
        self._counted("unresolved")

    @property
    def change(self) -> str:
        self._counted("change")


POLICY = GatePolicy(
    min_fixes=1,
    max_regressions=0,
    max_holdout_degradations=0,
    min_holdout_cases=0,
    min_judge_preference_ratio=0.5,
    require_judge=False,
    require_independent_judge=True,
)


def binding(rubric_id: str, *, version: int | None = None) -> RubricBinding:
    live = LIVE[rubric_id]
    return RubricBinding(
        rubric_id=rubric_id,
        version=live.version if version is None else version,
        source_sha256=live.source_sha256,
    )


def golden_result(
    *,
    claims: tuple[TargetCaseRef, ...] = CLAIMS,
    pool: tuple[TargetCaseRef, ...] = POOL,
    split: UnseenSplit | None | str = "recompute",
    unseen_outcomes: tuple[ReplayOutcome, ...] | None = None,
    extra_outcomes: tuple[ReplayOutcome, ...] = (),
    rubrics: tuple[RubricBinding, ...] | None = None,
) -> ValidationResult:
    v = frozen_dict()["validation"]
    outcomes = tuple(outcome(c, CASE_TARGET, Verdict.FAIL, Verdict.PASS) for c in claims)
    outcomes += extra_outcomes
    if split == "recompute":
        split = split_unseen(pool, seed=SEED, min_failures=4, withhold=1)
    if unseen_outcomes is None:
        unseen_outcomes = (
            outcome(W_CCE, CASE_UNSEEN, Verdict.FAIL, Verdict.FAIL),
            outcome(W_SMA, CASE_UNSEEN, Verdict.FAIL, Verdict.PASS),
        )
    if split is None:
        unseen_outcomes = ()
    judge_independent = False  # no judge: the policy does not require one
    decision = evaluate_gate(
        outcomes, (), POLICY, judge_independent=judge_independent, target_refs=claims
    )
    return ValidationResult(
        candidate_id=v["candidate_id"],
        base_sha256=v["base_sha256"],
        patch_sha256=v["patch_sha256"],
        seed=SEED,
        improver_command_id=v["improver_command_id"],
        scorer_command_id=v["scorer_command_id"],
        judge_command_id="",
        rubrics=rubrics if rubrics is not None else (binding(ED),),
        target_refs=claims,
        outcomes=outcomes,
        judgements=(),
        decision=decision,
        evidence_model_calls=evidence_call_count(outcomes + unseen_outcomes, ()),
        notes=(),
        unseen=split if isinstance(split, UnseenSplit) else None,
        unseen_outcomes=unseen_outcomes,
    )


def golden() -> dict[str, Any]:
    return golden_result().to_dict()


def candidate_with(record: dict[str, Any], claims: tuple[TargetCaseRef, ...] = CLAIMS):
    base = Candidate.from_dict(frozen_dict())
    by_key = {
        f"{f.rubric_id}::{f.conversation_id}::{f.criterion_id}": f for f in base.rubric_failures
    }
    failures = tuple(
        by_key.get(c.key)
        or RubricFailureRef(
            failure_case_id=f"FC:{c.conversation_id}:{c.rubric_id}:{c.criterion_id}",
            rubric_id=c.rubric_id,
            rubric_version=1,
            criterion_id=c.criterion_id,
            conversation_id=c.conversation_id,
        )
        for c in claims
    )
    return replace(base, rubric_failures=failures, validation=record)


class AnUnseenBlockIsRecomputedFromItsOwnRecordTests(unittest.TestCase):
    def refused(self, record: dict[str, Any]) -> str:
        with self.assertRaises(ValueError) as caught:
            ValidationRecord.parse(record)
        return str(caught.exception)

    # -- the golden v4 and the frozen v3 ----------------------------------

    def test_the_golden_v4_is_a_coherent_record(self):
        record = golden()
        self.assertEqual(record["schema"], SCHEMA_VALIDATION)
        unseen = record["unseen"]
        self.assertEqual(unseen["status"], UNSEEN_MEASURED)
        self.assertEqual(unseen["withheld_refs"], [W_CCE.to_dict(), W_SMA.to_dict()])
        self.assertEqual(len(unseen["eligible_refs"]), 5)
        self.assertEqual((unseen["replayed"], unseen["fixed"]), (2, 1))
        parsed = ValidationRecord.parse(record)
        self.assertIsNotNone(parsed.unseen)
        self.assertEqual(validation_mismatches(candidate_with(record), rubrics=LIVE), [])

    def test_a_genuine_v3_record_still_parses_and_verifies(self):
        candidate = Candidate.from_dict(frozen_dict())
        self.assertEqual(candidate.validation["schema"], SCHEMA_VALIDATION_V3)
        record = ValidationRecord.parse(candidate.validation)
        self.assertIsNone(record.unseen)
        self.assertEqual(validation_mismatches(candidate, rubrics=LIVE), [])

    def test_a_validation_result_without_a_split_writes_v3(self):
        g = golden_result()
        fourteen = ValidationResult(
            candidate_id=g.candidate_id,
            base_sha256=g.base_sha256,
            patch_sha256=g.patch_sha256,
            seed=g.seed,
            improver_command_id=g.improver_command_id,
            scorer_command_id=g.scorer_command_id,
            judge_command_id=g.judge_command_id,
            rubrics=g.rubrics,
            target_refs=g.target_refs,
            outcomes=g.outcomes,
            judgements=g.judgements,
            decision=g.decision,
            evidence_model_calls=evidence_call_count(g.outcomes, ()),
            notes=g.notes,
        )
        record = fourteen.to_dict()
        self.assertEqual(record["schema"], SCHEMA_VALIDATION_V3)
        self.assertNotIn("unseen", record)
        with self.assertRaisesRegex(ValueError, "unseen outcomes without an unseen split"):
            replace(fourteen, unseen_outcomes=g.unseen_outcomes).to_dict()

    def test_a_v4_record_has_exactly_v3s_keys_plus_unseen(self):
        record = golden()
        v3 = golden_result(split=None).to_dict()
        self.assertEqual(set(record), set(v3) | {"unseen"})
        ValidationRecord.parse(record)
        without = copy.deepcopy(record)
        del without["unseen"]
        self.assertEqual(self.refused(without), "validation is missing key(s): unseen")
        as_v3 = copy.deepcopy(record)
        as_v3["schema"] = SCHEMA_VALIDATION_V3
        self.assertIn("validation carries unknown key(s): ", self.refused(as_v3))
        self.assertIn("unseen", self.refused(as_v3))

    def test_an_unseen_kind_in_outcomes_is_refused(self):
        record = frozen_dict()["validation"]
        case = record["outcomes"][5]["case"]
        self.assertEqual(case["key"], "conv-human-0001::separates-measured-from-assumed")
        self.assertEqual(case["kind"], CASE_HOLDOUT)
        self.assertEqual([o["case"]["kind"] for o in record["outcomes"][:5]].count(CASE_HOLDOUT), 0)
        case["kind"] = CASE_UNSEEN
        self.assertEqual(
            self.refused(record),
            "outcomes[5].case.kind must be 'target' or 'holdout', got 'unseen'",
        )

    def test_a_target_kind_in_unseen_outcomes_is_refused_by_its_label(self):
        record = golden()
        record["unseen"]["outcomes"][0]["case"]["kind"] = CASE_TARGET
        self.assertEqual(
            self.refused(record),
            "unseen.outcomes[0].case.kind must be 'unseen', got 'target'",
        )

    def test_a_claimed_failure_outside_the_pool_is_refused(self):
        record = golden()
        shown = CLAIMS[0].to_dict()
        record["unseen"]["eligible_refs"].remove(shown)
        self.assertEqual(len(record["unseen"]["eligible_refs"]), 4)
        self.assertEqual(
            self.refused(record),
            "validation.target_refs names "
            "'evidence-discipline::conv-human-0001::cites-checkable-evidence' but "
            "validation.unseen.eligible_refs does not: a claimed failure must come from the "
            "pool the split was made from",
        )

    def test_a_withheld_failure_on_the_claim_set_is_refused(self):
        record = golden()
        record["target_refs"].append(W_CCE.to_dict())
        self.assertEqual(
            self.refused(record),
            "validation.unseen.withheld_refs names "
            "'evidence-discipline::conv-human-0006::cites-checkable-evidence', which was "
            "withheld from the improver's prompt but is on the candidate's claim set",
        )

    def test_a_coherent_alternate_withheld_set_is_refused_naming_the_recomputed_split(self):
        claim_ranks = [local_rank(SEED, c.conversation_id) for c in CLAIMS]
        names = [f"conv-alt-{i:02d}" for i in range(64)]
        x = next(n for n in names if local_rank(SEED, n) < min(claim_ranks))
        y = next(n for n in names if n != x and local_rank(SEED, n) > local_rank(SEED, x))
        ref_x, ref_y = TargetCaseRef(ED, x, CCE), TargetCaseRef(ED, y, CCE)
        pool = tuple(sorted(CLAIMS + (ref_x, ref_y), key=lambda r: r.key))
        alternate = UnseenSplit(
            status=UNSEEN_MEASURED, eligible=pool, withheld=(ref_y,), min_failures=4, withhold=1
        )
        record = golden_result(
            split=alternate,
            unseen_outcomes=(outcome(ref_y, CASE_UNSEEN, Verdict.FAIL, Verdict.PASS),),
        ).to_dict()
        message = self.refused(record)
        self.assertIn(f"recomputes ['{ref_x.key}'] with status 'measured'", message)
        self.assertIn(f"validation.unseen.withheld_refs is ['{ref_y.key}']", message)

    def test_an_unseen_pass_to_fail_changes_no_gate_number(self):
        pool = (ref(IC, "t1", "c1"), ref(IC, "t2", "c1"))
        refs = tuple(TargetCaseRef(r.rubric_id, r.conversation_id, r.criterion_id) for r in pool)
        split = split_unseen(refs, seed="producer-seed", min_failures=2, withhold=1)
        self.assertEqual(split.status, UNSEEN_MEASURED)
        (withheld,) = split.withheld
        shown = [r for r in pool if r.conversation_id != withheld.conversation_id]

        class Regressing(ReplayRunner):
            def run(self, *, role, system, user, response_schema):
                reply = super().run(
                    role=role, system=system, user=user, response_schema=response_schema
                )
                if role == "scorer" and self._thread(user) == withheld.conversation_id:
                    verdict = "FAIL" if FIX_LINE in user else "PASS"
                    for row in reply["criteria"]:
                        row["verdict"] = verdict
                return reply

        result = validate_candidate(
            producer_candidate(shown),
            Regressing(),
            rubrics=PRODUCER_RUBRICS,
            conversations=CONVERSATIONS,
            base_text=BASE_TEXT,
            cfg=PromptConfig(),
            policy=GatePolicy(min_holdout_cases=0, require_judge=True),
            seed="producer-seed",
            unseen=split,
        )
        (unseen_outcome,) = result.unseen_outcomes
        self.assertEqual(
            (unseen_outcome.baseline, unseen_outcome.candidate), (Verdict.PASS, Verdict.FAIL)
        )
        self.assertEqual(result.decision.metrics["regressions"], 0)
        self.assertTrue(result.decision.passed)
        record = ValidationRecord.parse(result.to_dict())
        self.assertEqual(record.recompute_gate_mismatches(), [])

    def test_evaluate_gate_refuses_an_unseen_outcome(self):
        """A direct caller cannot hand the gate an unseen outcome either.

        Two-sided: the same call without it decides and passes, so the refusal is
        of the unseen PASS -> FAIL, which the gate would otherwise have counted as a
        regression and failed on.

        And the refusal comes before the counting, not merely before the return:
        ``UncountableUnseen`` raises ``AssertionError`` on every verdict-derived
        property the counting reads, so a refusal moved below it fails here
        instead of passing with the same message.
        """
        claimed = outcome(CLAIMS[0], CASE_TARGET, Verdict.FAIL, Verdict.PASS)
        unseen = outcome(W_CCE, CASE_UNSEEN, Verdict.PASS, Verdict.FAIL)
        policy = GatePolicy(min_holdout_cases=0, require_judge=False)
        decided = evaluate_gate((claimed,), (), policy)
        self.assertTrue(decided.passed)
        self.assertEqual(decided.metrics["regressions"], 0)
        for given in ((claimed, unseen), (unseen,), (unseen, claimed)):
            with self.subTest(n=len(given)):
                with self.assertRaises(ValueError) as caught:
                    evaluate_gate(given, (), policy)
                self.assertIn(
                    "evaluate_gate was given 1 unseen outcome(s), the first "
                    "'conv-human-0006::cites-checkable-evidence': an unseen case is "
                    "report-only, never gate evidence",
                    str(caught.exception),
                )
        poisoned = UncountableUnseen(
            case=ReplayCase(W_CCE.conversation_id, W_CCE.criterion_id, CASE_UNSEEN),
            baseline=Verdict.PASS,
            candidate=Verdict.FAIL,
            rubric_id=W_CCE.rubric_id,
        )
        for handed in ((poisoned,), (claimed, poisoned), (poisoned, claimed)):
            with self.subTest(n=len(handed), poisoned=True):
                with self.assertRaises(ValueError) as refused:
                    evaluate_gate(handed, (), policy)
                self.assertIn("never gate evidence", str(refused.exception))

    def test_a_withheld_case_passed_as_a_holdout_or_claimed_is_refused_before_any_model_call(
        self,
    ):
        pool = (ref(IC, "t1", "c1"), ref(IC, "t2", "c1"))
        refs = tuple(TargetCaseRef(r.rubric_id, r.conversation_id, r.criterion_id) for r in pool)
        split = split_unseen(refs, seed="producer-seed", min_failures=2, withhold=1)
        self.assertEqual(split.status, UNSEEN_MEASURED)
        (withheld,) = split.withheld
        shown = [r for r in pool if r.conversation_id != withheld.conversation_id]
        pair = (withheld.conversation_id, withheld.criterion_id)

        class Regressing(ReplayRunner):
            """The withheld case goes PASS -> FAIL: as a holdout, a regression."""

            def run(self, *, role, system, user, response_schema):
                reply = super().run(
                    role=role, system=system, user=user, response_schema=response_schema
                )
                if role == "scorer" and self._thread(user) == withheld.conversation_id:
                    verdict = "FAIL" if FIX_LINE in user else "PASS"
                    for row in reply["criteria"]:
                        row["verdict"] = verdict
                return reply

        def validate(runner, candidate_refs, holdout_pairs):
            return validate_candidate(
                producer_candidate(candidate_refs),
                runner,
                rubrics=PRODUCER_RUBRICS,
                conversations=CONVERSATIONS,
                base_text=BASE_TEXT,
                cfg=PromptConfig(),
                policy=GatePolicy(min_holdout_cases=0, require_judge=True),
                seed="producer-seed",
                holdout_pairs=holdout_pairs,
                unseen=split,
            )

        runner = Regressing()
        with self.assertRaises(ValueError) as caught:
            validate(runner, shown, (pair,))
        self.assertIn(f"holdout pair '{withheld.conversation_id}::c1'", str(caught.exception))
        self.assertIn(f"withheld case '{withheld.key}'", str(caught.exception))
        self.assertEqual(runner.calls, [])
        with self.assertRaisesRegex(ValueError, "cannot be a claimed target; no model was called"):
            validate(runner, pool, ())
        self.assertEqual(runner.calls, [])
        # Without the overlap the same call gates on the shown case alone: the
        # withheld case's PASS -> FAIL is in the unseen block, not a regression,
        # and the record parses and recomputes.
        result = validate(Regressing(), shown, ())
        self.assertEqual([o.case.kind for o in result.outcomes], [CASE_TARGET])
        (unseen_outcome,) = result.unseen_outcomes
        self.assertEqual(unseen_outcome.case.key, f"{pair[0]}::{pair[1]}")
        self.assertEqual(
            (unseen_outcome.baseline, unseen_outcome.candidate), (Verdict.PASS, Verdict.FAIL)
        )
        self.assertTrue(result.decision.passed)
        self.assertEqual(result.decision.metrics["regressions"], 0)
        record = ValidationRecord.parse(result.to_dict())
        self.assertEqual(record.recompute_gate_mismatches(), [])

    def test_an_unseen_only_rubric_must_be_bound(self):
        other = "unseen-only-rubric"
        w_cce, w_sma = (
            TargetCaseRef(other, r.conversation_id, r.criterion_id) for r in (W_CCE, W_SMA)
        )
        bound = RubricBinding(rubric_id=other, version=1, source_sha256=sha256_text("other"))
        result = golden_result(
            pool=CLAIMS + (w_cce, w_sma),
            unseen_outcomes=(outcome(w_cce, CASE_UNSEEN, Verdict.FAIL, Verdict.PASS),),
            rubrics=(binding(ED), bound),
        )
        record = result.to_dict()
        self.assertEqual(ValidationRecord.parse(record).unseen.withheld, (w_cce, w_sma))
        unbound = copy.deepcopy(record)
        unbound["rubrics"] = [binding(ED).to_dict()]
        self.assertEqual(
            self.refused(unbound),
            "validation.rubrics does not exactly match the rubric ids the outcomes claim to "
            "have been scored against: missing binding(s) for ['unseen-only-rubric']",
        )

    def _no_such_criterion_record(self, criterion: str) -> dict[str, Any]:
        withheld = TargetCaseRef(ED, conv("0006"), criterion)
        return golden_result(
            pool=CLAIMS + (withheld,),
            unseen_outcomes=(outcome(withheld, CASE_UNSEEN, Verdict.FAIL, Verdict.PASS),),
        ).to_dict()

    def test_an_unseen_outcome_on_a_criterion_the_live_rubric_does_not_grade_is_a_mismatch(self):
        record = self._no_such_criterion_record("no-such-criterion")
        parsed = ValidationRecord.parse(record)
        self.assertEqual([r.criterion_id for r in parsed.unseen.withheld], ["no-such-criterion"])
        self.assertNotIn("no-such-criterion", {c.id for c in LIVE[ED].criteria})
        problems = validation_mismatches(candidate_with(record), rubrics=LIVE)
        self.assertEqual(len(problems), 1, problems)
        self.assertTrue(problems[0].startswith("validation.unseen.outcomes case"), problems)
        self.assertIn("no-such-criterion", problems[0])
        control = self._no_such_criterion_record(CCE)
        self.assertEqual(validation_mismatches(candidate_with(control), rubrics=LIVE), [])

    def test_0_of_w_parses_and_fail_to_insufficient_is_no_fix(self):
        record = golden_result(
            unseen_outcomes=(
                outcome(W_CCE, CASE_UNSEEN, Verdict.FAIL, Verdict.FAIL),
                outcome(W_SMA, CASE_UNSEEN, Verdict.FAIL, Verdict.INSUFFICIENT),
            )
        ).to_dict()
        self.assertEqual(record["unseen"]["fixed"], 0)
        self.assertEqual(ValidationRecord.parse(record).unseen.fixed, 0)
        self.assertEqual(validation_mismatches(candidate_with(record), rubrics=LIVE), [])
        record["unseen"]["fixed"] = 1
        self.assertEqual(
            self.refused(record),
            "validation.unseen.fixed is 1 but 0 of the unseen outcome(s) went FAIL -> PASS "
            "(FAIL -> INSUFFICIENT is not a fix)",
        )

    def test_a_withheld_failure_not_replayed_counts_in_w_not_in_x(self):
        record = golden_result(
            unseen_outcomes=(outcome(W_SMA, CASE_UNSEEN, Verdict.FAIL, Verdict.PASS),)
        ).to_dict()
        parsed = ValidationRecord.parse(record)
        self.assertEqual((len(parsed.unseen.withheld), parsed.unseen.replayed), (2, 1))
        self.assertEqual(validation_mismatches(candidate_with(record), rubrics=LIVE), [])
        outside = copy.deepcopy(record)
        stray = outcome(
            TargetCaseRef(ED, conv("0001"), SMA), CASE_UNSEEN, Verdict.FAIL, Verdict.PASS
        )
        outside["unseen"]["outcomes"].append(stray.to_dict())
        self.assertIn(
            "validation.unseen.outcomes has case "
            "'evidence-discipline::conv-human-0001::separates-measured-from-assumed', which is "
            "not among validation.unseen.withheld_refs",
            self.refused(outside),
        )
        twice = copy.deepcopy(record)
        twice["unseen"]["outcomes"].append(copy.deepcopy(twice["unseen"]["outcomes"][0]))
        self.assertEqual(
            self.refused(twice),
            "validation.unseen.outcomes has a duplicate case "
            "'evidence-discipline::conv-human-0006::separates-measured-from-assumed'",
        )

    def test_a_withheld_case_whose_replay_fails_stays_in_w_with_a_note(self):
        claim = TargetCaseRef(IC, "t1", "c1")
        failing, missing = TargetCaseRef(IC, "t2", "c1"), TargetCaseRef(IC, "t3", "c1")
        self.assertNotIn("t3", CONVERSATIONS)
        # A seed whose order (computed here) walks t2, then t3, and leaves t1
        # last, so N = 3 and k = 2 genuinely withhold t2 and t3.
        seed = next(
            s
            for s in (f"failed-replay-{i}" for i in range(1000))
            if sorted(("t1", "t2", "t3"), key=lambda c: local_rank(s, c)) == ["t2", "t3", "t1"]
        )
        split = split_unseen((claim, failing, missing), seed=seed, min_failures=3, withhold=2)
        self.assertEqual((split.status, split.withheld), (UNSEEN_MEASURED, (failing, missing)))
        result = validate_candidate(
            producer_candidate([ref(IC, "t1", "c1")]),
            ReplayRunner(fail_conversations=["t2"]),
            rubrics=PRODUCER_RUBRICS,
            conversations=CONVERSATIONS,
            base_text=BASE_TEXT,
            cfg=PromptConfig(),
            policy=GatePolicy(min_holdout_cases=0, require_judge=True),
            seed=seed,
            unseen=split,
        )
        self.assertTrue(result.decision.passed)
        notes = "\n".join(result.notes)
        self.assertIn("replay of unseen t2::c1 failed", notes)
        self.assertIn("unseen instruction-compliance::t3::c1 was NOT replayed", notes)
        unseen = result.to_dict()["unseen"]
        self.assertEqual(len(unseen["withheld_refs"]), 2)
        self.assertEqual(unseen["replayed"], 0)
        self.assertEqual(unseen["fixed"], 0)
        parsed = ValidationRecord.parse(result.to_dict())
        self.assertEqual(parsed.recompute_gate_mismatches(), [])
        assert parsed.unseen is not None
        self.assertEqual(parsed.unseen.withheld, (failing, missing))
        self.assertEqual((parsed.unseen.replayed, parsed.unseen.fixed), (0, 0))
        report = {
            "unseen": {
                "status": unseen["status"],
                "withheld": unseen["withheld_refs"],
                "replayed": unseen["replayed"],
                "fixed": unseen["fixed"],
            }
        }
        self.assertEqual(
            _unseen_line(report),
            f"  unseen failures fixed: 0/2, 2 not replayed ({UNSEEN_SCOPE})",
        )

    def test_three_failures_on_two_conversations_with_n_3_k_2_is_short_of_k(self):
        seed = "short-of-k-seed"
        first, last = sorted(("conv-p", "conv-q"), key=lambda c: local_rank(seed, c))
        pool = (
            TargetCaseRef(ED, first, CCE),
            TargetCaseRef(ED, last, CCE),
            TargetCaseRef(ED, last, SMA),
        )
        split = split_unseen(pool, seed=seed, min_failures=3, withhold=2)
        self.assertEqual((split.status, split.withheld), (UNSEEN_SHORT_OF_K, ()))
        report = {"unseen": _unseen_report_of(split)}
        self.assertEqual(
            _unseen_line(report),
            "  unseen failures fixed: not measured (fewer than [caps].unseen_withhold = 2 "
            "failure(s) can be withheld while one whole conversation stays shown)",
        )

    def test_the_walk_stops_at_k(self):
        seed = "walk-seed"
        ids = ("conv-r", "conv-s", "conv-t")
        first = min(ids, key=lambda c: local_rank(seed, c))
        pool = tuple(TargetCaseRef(ED, c, CCE) for c in ids)
        split = split_unseen(pool, seed=seed, min_failures=3, withhold=1)
        self.assertEqual(split.status, UNSEEN_MEASURED)
        self.assertEqual(split.withheld, (TargetCaseRef(ED, first, CCE),))

    def test_the_split_does_not_depend_on_the_input_order(self):
        forward = split_unseen(POOL, seed=SEED, min_failures=4, withhold=1)
        reverse = split_unseen(
            sorted(POOL, key=lambda r: r.key, reverse=True), seed=SEED, min_failures=4, withhold=1
        )
        self.assertEqual(forward, reverse)
        self.assertEqual(list(reverse.eligible), sorted(POOL, key=lambda r: r.key))
        self.assertEqual(reverse.withheld, (W_CCE, W_SMA))
        self.assertEqual([r.key for r in reverse.withheld], sorted(r.key for r in reverse.withheld))

    def test_a_pool_missing_an_unclaimed_failure_still_verifies(self):
        record = golden_result(
            pool=CLAIMS + (W_SMA,),
            unseen_outcomes=(outcome(W_SMA, CASE_UNSEEN, Verdict.FAIL, Verdict.PASS),),
        ).to_dict()
        self.assertNotIn(W_CCE.to_dict(), record["unseen"]["eligible_refs"])
        self.assertEqual(record["unseen"]["withheld_refs"], [W_SMA.to_dict()])
        self.assertEqual(validation_mismatches(candidate_with(record), rubrics=LIVE), [])

    def test_a_v4_record_rewritten_as_v3_still_verifies(self):
        record = golden()
        del record["unseen"]
        record["schema"] = SCHEMA_VALIDATION_V3
        parsed_outcomes = ValidationRecord.parse(golden()).outcomes
        record["evidence_model_calls"] = evidence_call_count(parsed_outcomes, ())
        self.assertIsNone(ValidationRecord.parse(record).unseen)
        self.assertEqual(validation_mismatches(candidate_with(record), rubrics=LIVE), [])

    def test_the_unseen_block_refuses_unknown_missing_and_mistyped_fields(self):
        extra = golden()
        extra["unseen"]["scope"] = UNSEEN_SCOPE
        self.assertEqual(self.refused(extra), "validation.unseen carries unknown key(s): scope")
        for key in sorted(golden()["unseen"]):
            with self.subTest(missing=key):
                record = golden()
                del record["unseen"][key]
                self.assertEqual(
                    self.refused(record), f"validation.unseen is missing key(s): {key}"
                )
        for key in ("withhold", "replayed"):
            with self.subTest(mistyped=key):
                record = golden()
                record["unseen"][key] = str(record["unseen"][key])
                self.assertEqual(
                    self.refused(record), f"validation.unseen.{key} must be an integer, got str"
                )

    def test_unsorted_unseen_refs_are_refused(self):
        for field in ("eligible_refs", "withheld_refs"):
            with self.subTest(field=field):
                record = golden()
                self.assertGreaterEqual(len(record["unseen"][field]), 2)
                record["unseen"][field].reverse()
                self.assertEqual(
                    self.refused(record), f"validation.unseen.{field} must be sorted by ref key"
                )

    def test_a_recorded_status_the_split_does_not_recompute_is_refused(self):
        record = golden()
        self.assertEqual(record["unseen"]["status"], UNSEEN_MEASURED)
        record["unseen"]["status"] = "too-few"
        message = self.refused(record)
        self.assertIn("with status 'too-few' but seed", message)
        self.assertIn("recomputes", message)
        self.assertIn("with status 'measured'", message)

    def test_a_replayed_count_the_block_does_not_hold_is_refused(self):
        record = golden()
        self.assertEqual(record["unseen"]["replayed"], 2)
        record["unseen"]["replayed"] = 1
        self.assertEqual(
            self.refused(record),
            "validation.unseen.replayed is 1 but the block holds 2 unseen outcome(s)",
        )

    def test_a_duplicate_unseen_outcome_and_a_cross_kind_case_key_are_refused(self):
        twice = golden()
        twice["unseen"]["outcomes"].append(copy.deepcopy(twice["unseen"]["outcomes"][1]))
        self.assertEqual(
            self.refused(twice),
            "validation.unseen.outcomes has a duplicate case "
            "'evidence-discipline::conv-human-0006::separates-measured-from-assumed'",
        )
        holdout = outcome(W_CCE, CASE_HOLDOUT, Verdict.PASS, Verdict.PASS)
        cross = golden_result(extra_outcomes=(holdout,)).to_dict()
        self.assertEqual(
            self.refused(cross),
            "validation.unseen.outcomes case 'conv-human-0006::cites-checkable-evidence' is "
            "also a case in validation.outcomes: one case cannot be both withheld from the "
            "improver and evidence the gate reads",
        )

    def test_an_unseen_only_rubric_changed_after_the_run_is_a_mismatch(self):
        other = sorted(set(LIVE) - {ED})[0]
        criterion = LIVE[other].criteria[0].id
        w_ref = TargetCaseRef(other, conv("0006"), criterion)
        stale = binding(other, version=LIVE[other].version + 1)
        record = golden_result(
            pool=CLAIMS + (w_ref,),
            unseen_outcomes=(outcome(w_ref, CASE_UNSEEN, Verdict.FAIL, Verdict.PASS),),
            rubrics=(binding(ED), stale),
        ).to_dict()
        self.assertIsNotNone(ValidationRecord.parse(record).unseen)
        candidate = candidate_with(record)
        problems = validation_mismatches(candidate, rubrics=LIVE)
        self.assertTrue(problems)
        self.assertTrue(all(other in p for p in problems), problems)
        with self.assertRaises(CandidateTampered) as caught:
            verify_full_integrity(candidate, rubrics=LIVE, action="apply")
        self.assertIn(other, str(caught.exception))
        current = copy.deepcopy(record)
        current["rubrics"][1] = binding(other).to_dict()
        self.assertEqual(validation_mismatches(candidate_with(current), rubrics=LIVE), [])


def _unseen_report_of(split: UnseenSplit) -> dict[str, Any]:
    """The review report's unseen block for a split with no replayed outcome."""
    return {
        "status": split.status,
        "min_failures": split.min_failures,
        "withhold": split.withhold,
        "eligible": [r.to_dict() for r in split.eligible],
        "withheld": [r.to_dict() for r in split.withheld],
        "replayed": 0,
        "fixed": 0,
        "outcomes": [],
        "scope": UNSEEN_SCOPE,
    }


if __name__ == "__main__":
    unittest.main()
