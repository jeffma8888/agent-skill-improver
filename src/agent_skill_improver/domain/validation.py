"""Baseline-versus-candidate replay and the promotion gate.

Honest scope, stated here because it decides how much the gate is worth: this
is a *prompt-level* replay, not a re-run of the original agent session. The
scorer re-evaluates the same conversations with the baseline skill text and then
with the candidate skill text substituted. It measures whether the change makes
the graded criteria come out differently on the recorded evidence. It cannot
measure what a fresh agent would have done differently, and no offline
mechanism can.

What the gate does enforce:

* a minimum number of real fixes (FAIL -> PASS on the targeted cases);
* zero regressions (PASS -> FAIL anywhere, targeted or holdout);
* zero holdout degradations (PASS -> INSUFFICIENT is a loss of evidence, not a
  neutral outcome);
* a blind judge preference, where the judge never learns which side is which.

FAIL -> INSUFFICIENT earns no credit. A change that makes the scorer unable to
decide is not a fix.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .. import SCHEMA_JUDGEMENT
from .rubric import Verdict

ARM_BASELINE = "baseline"
ARM_CANDIDATE = "candidate"

CASE_TARGET = "target"
CASE_HOLDOUT = "holdout"


@dataclass(frozen=True)
class ReplayCase:
    conversation_id: str
    criterion_id: str
    kind: str  # CASE_TARGET | CASE_HOLDOUT

    @property
    def key(self) -> str:
        return f"{self.conversation_id}::{self.criterion_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "criterion_id": self.criterion_id,
            "kind": self.kind,
            "key": self.key,
        }


@dataclass(frozen=True)
class ReplayOutcome:
    case: ReplayCase
    baseline: Verdict
    candidate: Verdict

    @property
    def fixed(self) -> bool:
        return self.baseline is Verdict.FAIL and self.candidate is Verdict.PASS

    @property
    def regressed(self) -> bool:
        return self.baseline is Verdict.PASS and self.candidate is Verdict.FAIL

    @property
    def degraded(self) -> bool:
        """PASS became undecidable: evidence quality got worse, not better."""
        return self.baseline is Verdict.PASS and self.candidate is Verdict.INSUFFICIENT

    @property
    def unresolved(self) -> bool:
        return self.baseline is Verdict.FAIL and self.candidate is Verdict.INSUFFICIENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "baseline": self.baseline.value,
            "candidate": self.candidate.value,
            "fixed": self.fixed,
            "regressed": self.regressed,
            "degraded": self.degraded,
            "unresolved": self.unresolved,
        }


def blind_arms(case_key: str, seed: str) -> dict[str, str]:
    """Deterministically map presentation labels A/B onto the two arms.

    Deterministic so a replay is reproducible, and derived from a run seed so a
    judge cannot learn a fixed convention like "B is always the new one".
    """
    digest = hashlib.sha256(f"{seed}:{case_key}".encode("utf-8")).hexdigest()
    if int(digest[0], 16) % 2 == 0:
        return {"A": ARM_BASELINE, "B": ARM_CANDIDATE}
    return {"A": ARM_CANDIDATE, "B": ARM_BASELINE}


@dataclass(frozen=True)
class Judgement:
    case: ReplayCase
    preferred_label: str  # "A" | "B" | "tie"
    preferred_arm: str  # baseline | candidate | tie
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_JUDGEMENT,
            "case": self.case.to_dict(),
            "preferred_label": self.preferred_label,
            "preferred_arm": self.preferred_arm,
            "rationale": self.rationale,
        }


def parse_judgement(payload: Any, *, case: ReplayCase, arms: Mapping[str, str]) -> Judgement:
    """Parse a blind judge response and unblind it locally."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"judgement must be a JSON object, got {type(payload).__name__}")
    if payload.get("schema") != SCHEMA_JUDGEMENT:
        raise ValueError(
            f"judgement schema must be {SCHEMA_JUDGEMENT!r}, got {payload.get('schema')!r}"
        )
    label = payload.get("preferred")
    if label not in ("A", "B", "tie"):
        raise ValueError(f"judgement 'preferred' must be 'A', 'B' or 'tie', got {label!r}")
    arm = "tie" if label == "tie" else arms[label]
    rationale = payload.get("rationale")
    return Judgement(
        case=case,
        preferred_label=str(label),
        preferred_arm=arm,
        rationale=rationale if isinstance(rationale, str) else "",
    )


@dataclass(frozen=True)
class GatePolicy:
    min_fixes: int = 1
    max_regressions: int = 0
    max_holdout_degradations: int = 0
    min_holdout_cases: int = 1
    min_judge_preference_ratio: float = 0.5
    require_judge: bool = True
    #: Refuse the gate when the judge is the same command as the improver.
    #: A judge that shares a model with the thing it is grading is grading its
    #: own homework, and its preference is not independent evidence.
    require_independent_judge: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_fixes": self.min_fixes,
            "max_regressions": self.max_regressions,
            "max_holdout_degradations": self.max_holdout_degradations,
            "min_holdout_cases": self.min_holdout_cases,
            "min_judge_preference_ratio": self.min_judge_preference_ratio,
            "require_judge": self.require_judge,
            "require_independent_judge": self.require_independent_judge,
        }


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, Any]
    policy: GatePolicy = field(default_factory=GatePolicy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "policy": self.policy.to_dict(),
        }


def evaluate_gate(
    outcomes: Sequence[ReplayOutcome],
    judgements: Sequence[Judgement],
    policy: GatePolicy,
    *,
    judge_independent: bool = True,
) -> GateDecision:
    """Decide whether a candidate may be *offered* for human approval.

    Passing this gate is necessary and not sufficient: a human still has to
    approve, and apply re-checks the base hash at write time.

    ``judge_independent`` is passed in rather than inferred, because independence
    is a property of the *configuration* (which command runs which role) and the
    domain layer deliberately cannot read configuration.
    """
    targets = [o for o in outcomes if o.case.kind == CASE_TARGET]
    holdouts = [o for o in outcomes if o.case.kind == CASE_HOLDOUT]

    fixes = [o for o in targets if o.fixed]
    regressions = [o for o in outcomes if o.regressed]
    degradations = [o for o in holdouts if o.degraded]
    unresolved = [o for o in targets if o.unresolved]

    decided_judgements = [j for j in judgements if j.preferred_arm != "tie"]
    candidate_preferred = [j for j in decided_judgements if j.preferred_arm == ARM_CANDIDATE]
    ratio = len(candidate_preferred) / len(decided_judgements) if decided_judgements else 0.0

    reasons: list[str] = []
    if not targets:
        reasons.append("no target cases were replayed; nothing was measured")
    if len(fixes) < policy.min_fixes:
        reasons.append(
            f"fixes {len(fixes)} < required {policy.min_fixes} "
            f"(FAIL->INSUFFICIENT does not count; {len(unresolved)} case(s) became INSUFFICIENT)"
        )
    if len(regressions) > policy.max_regressions:
        keys = ", ".join(o.case.key for o in regressions[:5])
        reasons.append(f"{len(regressions)} regression(s) PASS->FAIL: {keys}")
    if len(degradations) > policy.max_holdout_degradations:
        keys = ", ".join(o.case.key for o in degradations[:5])
        reasons.append(f"{len(degradations)} holdout degradation(s) PASS->INSUFFICIENT: {keys}")
    if len(holdouts) < policy.min_holdout_cases:
        reasons.append(
            f"holdout set too small: {len(holdouts)} < required {policy.min_holdout_cases}; "
            "without holdouts a 'no regression' claim is unsupported"
        )
    if policy.require_judge:
        if policy.require_independent_judge and not judge_independent:
            reasons.append(
                "the judge role runs the same command as the improver role, so its preference "
                "is not independent evidence: it is the same model grading its own proposal. "
                "Configure a different [model.judge] command, or set "
                "[gate].require_independent_judge = false and accept that the judge signal "
                "is decoration"
            )
        if not decided_judgements:
            reasons.append("blind judge produced no decided preference")
        elif ratio < policy.min_judge_preference_ratio:
            reasons.append(
                f"blind judge preferred the candidate in {ratio:.2f} of decided cases "
                f"< required {policy.min_judge_preference_ratio:.2f}"
            )

    metrics = {
        "target_cases": len(targets),
        "holdout_cases": len(holdouts),
        "fixes": len(fixes),
        "regressions": len(regressions),
        "holdout_degradations": len(degradations),
        "target_unresolved": len(unresolved),
        "judgements": len(judgements),
        "judgements_decided": len(decided_judgements),
        "judge_candidate_preference_ratio": round(ratio, 4),
        "judge_independent": judge_independent,
        "fixed_case_keys": [o.case.key for o in fixes],
        "regressed_case_keys": [o.case.key for o in regressions],
        "degraded_case_keys": [o.case.key for o in degradations],
    }
    return GateDecision(passed=not reasons, reasons=tuple(reasons), metrics=metrics, policy=policy)
