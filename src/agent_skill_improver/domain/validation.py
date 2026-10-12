"""Baseline-versus-candidate replay and the promotion gate.

Honest scope, stated here because it decides how much the gate is worth: this
is a *prompt-level* replay, not a re-run of the original agent session. The
scorer re-evaluates the same conversations with the baseline skill text and then
with the candidate skill text substituted. It measures whether the change makes
the graded criteria come out differently on the recorded evidence. It cannot
measure what a fresh agent would have done differently, and no offline
mechanism can.

What the gate does enforce:

* every failure the candidate claims to fix was actually replayed, bound by the
  exact ``(rubric_id, conversation_id, criterion_id)`` triple rather than by a
  conversation/criterion pair two different rubrics could both answer to;
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
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .. import SCHEMA_JUDGEMENT, SCHEMA_VALIDATION
from .redaction import kept_prose, quoted_value
from .rubric import Verdict
from .scoring import MAX_RATIONALE_CHARS

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


_TARGET_REF_KEYS = frozenset({"rubric_id", "conversation_id", "criterion_id"})


@dataclass(frozen=True)
class TargetCaseRef:
    """One failure a candidate claims to fix: rubric, conversation, criterion.

    The identity is the whole triple, and that is the point. A
    ``conversation_id::criterion_id`` pair does not identify an examiner: two
    rubrics can grade criteria with the same id, and a rubric id sitting beside
    an outcome can be rewritten to name a *different* rubric that is live with a
    matching version and source hash. A binding that omits the rubric is
    therefore satisfiable by evidence some other examiner produced, which is
    exactly the substitution this type exists to make impossible.
    """

    rubric_id: str
    conversation_id: str
    criterion_id: str

    @property
    def key(self) -> str:
        return f"{self.rubric_id}::{self.conversation_id}::{self.criterion_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "conversation_id": self.conversation_id,
            "criterion_id": self.criterion_id,
        }

    @staticmethod
    def from_mapping(raw: Any, *, label: str) -> "TargetCaseRef":
        raw = _expect_mapping(raw, label=label)
        _expect_exact_keys(raw, _TARGET_REF_KEYS, label=label)
        return TargetCaseRef(
            rubric_id=_expect_nonempty_str(raw, "rubric_id", label=label),
            conversation_id=_expect_nonempty_str(raw, "conversation_id", label=label),
            criterion_id=_expect_nonempty_str(raw, "criterion_id", label=label),
        )


@dataclass(frozen=True)
class ReplayOutcome:
    case: ReplayCase
    baseline: Verdict
    candidate: Verdict
    #: The rubric that scored this exact case. Empty only for outcomes built
    #: in-memory before persistence (e.g. legacy call sites/tests); a parsed
    #: persisted record always carries a non-empty value, because a bare
    #: top-level list of rubrics cannot say which one produced which outcome
    #: when two rubrics grade overlapping criterion ids.
    rubric_id: str = ""

    @property
    def ref(self) -> TargetCaseRef:
        """The exact triple this outcome is evidence about.

        Built from the rubric that actually scored the case, so a target outcome
        can be compared against a candidate failure ref without either side
        being able to stand in for the other on a partial match.
        """
        return TargetCaseRef(
            rubric_id=self.rubric_id,
            conversation_id=self.case.conversation_id,
            criterion_id=self.case.criterion_id,
        )

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

    @property
    def change(self) -> str:
        """One word for what the replay showed: the gate's own vocabulary."""
        for label in ("fixed", "regressed", "degraded", "unresolved"):
            if getattr(self, label):
                return label
        return "unchanged"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "rubric_id": self.rubric_id,
            "baseline": self.baseline.value,
            "candidate": self.candidate.value,
            "fixed": self.fixed,
            "regressed": self.regressed,
            "degraded": self.degraded,
            "unresolved": self.unresolved,
        }


def evidence_call_count(outcomes: Sequence[ReplayOutcome], judgements: Sequence[Judgement]) -> int:
    """The number of model calls that produced this record's persisted evidence.

    Exactly derivable, and therefore checkable by a verifier that has only the
    record: each distinct ``(conversation, rubric)`` pair behind an outcome was
    scored once per arm, and each persisted judgement is one judge call.

    Calls that were attempted and produced nothing usable are deliberately *not*
    in this number; they are reported in ``notes`` instead. A field that mixed
    "work attempted" with "evidence kept" could not be recomputed from the
    record, and a number a verifier cannot recompute is not evidence -- it is an
    assertion that happens to sit next to evidence, which is precisely how a
    tampered count survives verification.
    """
    pairs = {(o.case.conversation_id, o.rubric_id) for o in outcomes}
    return 2 * len(pairs) + len(judgements)


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
            f"judgement schema must be {SCHEMA_JUDGEMENT!r}, got {quoted_value(payload.get('schema'))}"
        )
    label = payload.get("preferred")
    if label not in ("A", "B", "tie"):
        raise ValueError(
            f"judgement 'preferred' must be 'A', 'B' or 'tie', got {quoted_value(label)}"
        )
    arm = "tie" if label == "tie" else arms[label]
    return Judgement(
        case=case,
        preferred_label=str(label),
        preferred_arm=arm,
        rationale=kept_prose(payload.get("rationale"), limit=MAX_RATIONALE_CHARS),
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
    target_refs: Sequence[TargetCaseRef] = (),
) -> GateDecision:
    """Decide whether a candidate may be *offered* for human approval.

    Passing this gate is necessary and not sufficient: a human still has to
    approve, and apply re-checks the base hash at write time.

    ``judge_independent`` is passed in rather than inferred, because independence
    is a property of the *configuration* (which command runs which role) and the
    domain layer deliberately cannot read configuration.

    ``target_refs`` is the set of failures the candidate claims to fix. Every one
    of them must have a target outcome, matched on the whole
    ``(rubric_id, conversation_id, criterion_id)`` triple, or the gate fails for
    want of coverage. That is what makes an unavailable replay an honest gate
    failure at production time instead of a nominally passing record that a later
    verifier has to call tampered. It is a sequence rather than a boolean so the
    same decision can be recomputed byte-exact from a persisted record, which
    carries the same claim set.
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

    claimed = list(dict.fromkeys(target_refs))
    replayed_refs = {o.ref for o in targets}
    missing_refs = sorted((r for r in claimed if r not in replayed_refs), key=lambda r: r.key)

    reasons: list[str] = []
    if not targets:
        reasons.append("no target cases were replayed; nothing was measured")
    if missing_refs:
        keys = ", ".join(r.key for r in missing_refs[:5])
        reasons.append(
            f"{len(missing_refs)} claimed target case(s) have no replay outcome, so the change "
            f"was not measured against every failure this candidate says it fixes: {keys}"
        )
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
        "target_refs_claimed": len(claimed),
        "target_refs_replayed": len(claimed) - len(missing_refs),
        "unreplayed_target_ref_keys": [r.key for r in missing_refs],
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


# -- strict persisted-validation parsing and recomputation ----------------
#
# ``ValidationResult.to_dict()`` (in ``app/validate.py``) is the only writer of
# this shape, but the record is a plain JSON file like a candidate, so every
# field in it is *asserted* rather than proven once it is on disk. The schema
# below (``asi.validation.v3``) binds the primitives a verifier needs in order
# to recompute the gate from scratch rather than trust the stored decision:
# the candidate identity and hashes it was run against, the seed that decided
# blind arm assignment, the command that ran each role, and the rubric
# versions and source hashes that defined "pass". Parsing is strict -- an
# unknown key, a missing key, a wrong type, a duplicate case or a value that
# contradicts another field is a refusal, not a best-effort read, because a
# validation record a human never re-runs is exactly the kind of derived
# artifact that drifted silently before (see ``domain.integrity``).


def _expect_mapping(raw: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a JSON object, got {type(raw).__name__}")
    return raw


def _expect_exact_keys(raw: Mapping[str, Any], expected: frozenset, *, label: str) -> None:
    present = set(raw.keys())
    missing = expected - present
    unknown = present - expected
    if missing:
        raise ValueError(f"{label} is missing key(s): {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{label} carries unknown key(s): {', '.join(sorted(unknown))}")


def _expect_str(raw: Mapping[str, Any], key: str, *, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{label}.{key} must be a string, got {type(value).__name__}")
    return value


def _expect_bool(raw: Mapping[str, Any], key: str, *, label: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be a boolean, got {type(value).__name__}")
    return value


def _expect_int(raw: Mapping[str, Any], key: str, *, label: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be an integer, got {type(value).__name__}")
    return value


def _expect_number(raw: Mapping[str, Any], key: str, *, label: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{key} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        # Python's own JSON decoder accepts the non-standard tokens NaN,
        # Infinity and -Infinity by default. Neither the config loader nor this
        # writer ever emits one, so one here is either a hand-edit or a decoder
        # default nobody disabled -- either way, a non-finite policy value can
        # make every downstream comparison (ratio thresholds, equality checks)
        # behave unpredictably, so it is refused rather than propagated.
        raise ValueError(f"{label}.{key} must be a finite number, got {value!r}")
    return number


def _expect_str_list(raw: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or any(not isinstance(x, str) for x in raw):
        raise ValueError(f"{label} must be a list of strings")
    return tuple(raw)


def _expect_nonempty_str(raw: Mapping[str, Any], key: str, *, label: str) -> str:
    value = _expect_str(raw, key, label=label)
    if not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


_HEX_DIGITS = frozenset("0123456789abcdef")


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(c in _HEX_DIGITS for c in value)


def _expect_sha256(raw: Mapping[str, Any], key: str, *, label: str) -> str:
    value = _expect_str(raw, key, label=label)
    if not _is_sha256_hex(value):
        raise ValueError(
            f"{label}.{key} must be a 64-character lowercase sha256 hex digest, got {value!r}"
        )
    return value


_CASE_KEYS = frozenset({"conversation_id", "criterion_id", "kind", "key"})


def _parse_case(raw: Any, *, label: str) -> ReplayCase:
    raw = _expect_mapping(raw, label=label)
    _expect_exact_keys(raw, _CASE_KEYS, label=label)
    conversation_id = _expect_str(raw, "conversation_id", label=label)
    criterion_id = _expect_str(raw, "criterion_id", label=label)
    kind = _expect_str(raw, "kind", label=label)
    if kind not in (CASE_TARGET, CASE_HOLDOUT):
        raise ValueError(f"{label}.kind must be {CASE_TARGET!r} or {CASE_HOLDOUT!r}, got {kind!r}")
    case = ReplayCase(conversation_id=conversation_id, criterion_id=criterion_id, kind=kind)
    key = _expect_str(raw, "key", label=label)
    if key != case.key:
        raise ValueError(f"{label}.key is {key!r} but recomputes to {case.key!r}")
    return case


_OUTCOME_KEYS = frozenset(
    {
        "case",
        "rubric_id",
        "baseline",
        "candidate",
        "fixed",
        "regressed",
        "degraded",
        "unresolved",
    }
)


def _parse_outcome(raw: Any, *, index: int) -> ReplayOutcome:
    label = f"outcomes[{index}]"
    raw = _expect_mapping(raw, label=label)
    _expect_exact_keys(raw, _OUTCOME_KEYS, label=label)
    case = _parse_case(raw["case"], label=f"{label}.case")
    rubric_id = _expect_nonempty_str(raw, "rubric_id", label=label)
    try:
        baseline = Verdict.parse(raw.get("baseline"))
        candidate = Verdict.parse(raw.get("candidate"))
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from exc
    outcome = ReplayOutcome(case=case, baseline=baseline, candidate=candidate, rubric_id=rubric_id)
    for flag in ("fixed", "regressed", "degraded", "unresolved"):
        persisted = _expect_bool(raw, flag, label=label)
        recomputed = getattr(outcome, flag)
        if persisted != recomputed:
            raise ValueError(
                f"{label}.{flag} is {persisted!r} but baseline={baseline.value} "
                f"candidate={candidate.value} recomputes {flag}={recomputed!r}"
            )
    return outcome


_JUDGEMENT_KEYS = frozenset({"schema", "case", "preferred_label", "preferred_arm", "rationale"})


def _parse_judgement(raw: Any, *, index: int, seed: str) -> Judgement:
    label = f"judgements[{index}]"
    raw = _expect_mapping(raw, label=label)
    _expect_exact_keys(raw, _JUDGEMENT_KEYS, label=label)
    schema = _expect_str(raw, "schema", label=label)
    if schema != SCHEMA_JUDGEMENT:
        raise ValueError(f"{label}.schema must be {SCHEMA_JUDGEMENT!r}, got {schema!r}")
    case = _parse_case(raw["case"], label=f"{label}.case")
    if case.kind != CASE_TARGET:
        raise ValueError(
            f"{label}.case.kind is {case.kind!r}: only target cases are ever judged, so a "
            "judgement over a holdout case is a kind conflict, not a legitimate blind comparison"
        )
    preferred_label = _expect_str(raw, "preferred_label", label=label)
    if preferred_label not in ("A", "B", "tie"):
        raise ValueError(
            f"{label}.preferred_label must be 'A', 'B' or 'tie', got {preferred_label!r}"
        )
    preferred_arm = _expect_str(raw, "preferred_arm", label=label)
    # Kept masked and clipped on the way in (parse_judgement); the same on the
    # way out, so a record edited on disk cannot make review print more.
    rationale = kept_prose(_expect_str(raw, "rationale", label=label), limit=MAX_RATIONALE_CHARS)
    if preferred_label == "tie":
        expected_arm = "tie"
    else:
        arms = blind_arms(case.key, seed)
        expected_arm = arms[preferred_label]
    if preferred_arm != expected_arm:
        raise ValueError(
            f"{label}.preferred_arm is {preferred_arm!r} but the blind mapping for seed "
            f"{seed!r} and case {case.key!r} says label {preferred_label!r} is arm "
            f"{expected_arm!r}: the arm was unblinded to a different answer than it was assigned"
        )
    return Judgement(
        case=case, preferred_label=preferred_label, preferred_arm=preferred_arm, rationale=rationale
    )


_POLICY_KEYS = frozenset(
    {
        "min_fixes",
        "max_regressions",
        "max_holdout_degradations",
        "min_holdout_cases",
        "min_judge_preference_ratio",
        "require_judge",
        "require_independent_judge",
    }
)


def _expect_nonneg_int(raw: Mapping[str, Any], key: str, *, label: str) -> int:
    value = _expect_int(raw, key, label=label)
    if value < 0:
        raise ValueError(f"{label}.{key} must be >= 0, got {value}")
    return value


def parse_gate_policy(raw: Any, *, label: str = "gate.policy") -> GatePolicy:
    """Parse a persisted policy with the same domain the config loader enforces.

    ``config._load_gate`` refuses a negative count or a ratio outside ``[0, 1]``
    at load time; a persisted policy that skipped those checks could recompute
    a "passing" gate from nonsensical thresholds (a negative ``min_fixes`` is
    satisfied by doing nothing), so the same bounds are enforced here.
    """
    raw = _expect_mapping(raw, label=label)
    _expect_exact_keys(raw, _POLICY_KEYS, label=label)
    ratio = _expect_number(raw, "min_judge_preference_ratio", label=label)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"{label}.min_judge_preference_ratio must be in [0, 1], got {ratio}")
    return GatePolicy(
        min_fixes=_expect_nonneg_int(raw, "min_fixes", label=label),
        max_regressions=_expect_nonneg_int(raw, "max_regressions", label=label),
        max_holdout_degradations=_expect_nonneg_int(raw, "max_holdout_degradations", label=label),
        min_holdout_cases=_expect_nonneg_int(raw, "min_holdout_cases", label=label),
        min_judge_preference_ratio=ratio,
        require_judge=_expect_bool(raw, "require_judge", label=label),
        require_independent_judge=_expect_bool(raw, "require_independent_judge", label=label),
    )


_GATE_KEYS = frozenset({"passed", "reasons", "metrics", "policy"})


def _strict_equal(a: Any, b: Any, *, path: str) -> list[str]:
    """Deep equality that also refuses a type mismatch a loose ``==`` would miss.

    ``True == 1`` and ``1 == 1.0`` are both true in Python; neither is a value a
    JSON writer of this schema ever produces, and either would be a schema/type
    tamper this function exists to catch.
    """
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__} ({a!r} vs {b!r})"]
    if isinstance(a, dict):
        problems = []
        keys = set(a) | set(b)
        for key in sorted(keys):
            if key not in a or key not in b:
                problems.append(f"{path}.{key}: present in one side only")
                continue
            problems.extend(_strict_equal(a[key], b[key], path=f"{path}.{key}"))
        return problems
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        problems = []
        for i, (x, y) in enumerate(zip(a, b)):
            problems.extend(_strict_equal(x, y, path=f"{path}[{i}]"))
        return problems
    if a != b:
        return [f"{path}: {a!r} != {b!r}"]
    return []


_RUBRIC_BINDING_KEYS = frozenset({"rubric_id", "version", "source_sha256"})


@dataclass(frozen=True)
class RubricBinding:
    rubric_id: str
    version: int
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "version": self.version,
            "source_sha256": self.source_sha256,
        }

    @staticmethod
    def from_mapping(raw: Any, *, index: int) -> "RubricBinding":
        label = f"rubrics[{index}]"
        raw = _expect_mapping(raw, label=label)
        _expect_exact_keys(raw, _RUBRIC_BINDING_KEYS, label=label)
        version = _expect_int(raw, "version", label=label)
        if version < 1:
            raise ValueError(f"{label}.version must be >= 1, got {version}")
        return RubricBinding(
            rubric_id=_expect_nonempty_str(raw, "rubric_id", label=label),
            version=version,
            source_sha256=_expect_sha256(raw, "source_sha256", label=label),
        )


_RECORD_KEYS = frozenset(
    {
        "schema",
        "candidate_id",
        "base_sha256",
        "patch_sha256",
        "seed",
        "improver_command_id",
        "scorer_command_id",
        "judge_command_id",
        "rubrics",
        "target_refs",
        "outcomes",
        "judgements",
        "gate",
        "evidence_model_calls",
        "notes",
    }
)


@dataclass(frozen=True)
class ValidationRecord:
    """A persisted validation, parsed strictly and ready to recompute.

    ``judge_independent`` is deliberately not a stored field here: it is
    recomputed by the caller from ``judge_command_id`` and ``improver_command_id``
    (see :func:`agent_skill_improver.domain.integrity.validation_mismatches`),
    never read off a boolean the JSON file could assert on its own.

    ``target_refs`` is the claim set: the exact
    ``(rubric_id, conversation_id, criterion_id)`` triples this validation says it
    measured. Every target outcome must name one of them, and
    :func:`agent_skill_improver.domain.integrity.validation_mismatches` requires
    it to equal the candidate's own ``rubric_failures``, so a failure ref cannot
    be added to or removed from either side without the other disagreeing.

    ``evidence_model_calls`` is derived, not reported: it must equal
    :func:`evidence_call_count` over this record's own outcomes and judgements.
    """

    candidate_id: str
    base_sha256: str
    patch_sha256: str
    seed: str
    improver_command_id: str
    scorer_command_id: str
    judge_command_id: str
    rubrics: tuple[RubricBinding, ...]
    target_refs: tuple[TargetCaseRef, ...]
    outcomes: tuple[ReplayOutcome, ...]
    judgements: tuple[Judgement, ...]
    policy: GatePolicy
    gate_raw: Mapping[str, Any]
    evidence_model_calls: int
    notes: tuple[str, ...]

    @staticmethod
    def parse(raw: Any) -> "ValidationRecord":
        raw = _expect_mapping(raw, label="validation")
        _expect_exact_keys(raw, _RECORD_KEYS, label="validation")
        schema = _expect_str(raw, "schema", label="validation")
        if schema != SCHEMA_VALIDATION:
            raise ValueError(f"validation.schema must be {SCHEMA_VALIDATION!r}, got {schema!r}")
        gate_raw = _expect_mapping(raw["gate"], label="validation.gate")
        _expect_exact_keys(gate_raw, _GATE_KEYS, label="validation.gate")
        policy = parse_gate_policy(gate_raw["policy"], label="validation.gate.policy")
        if not isinstance(gate_raw["reasons"], list) or any(
            not isinstance(r, str) for r in gate_raw["reasons"]
        ):
            raise ValueError("validation.gate.reasons must be a list of strings")
        if not isinstance(gate_raw["metrics"], Mapping):
            raise ValueError("validation.gate.metrics must be a JSON object")
        if not isinstance(gate_raw["passed"], bool):
            raise ValueError("validation.gate.passed must be a boolean")

        seed = _expect_nonempty_str(raw, "seed", label="validation")

        rubrics_raw = raw["rubrics"]
        if not isinstance(rubrics_raw, list):
            raise ValueError("validation.rubrics must be a list")
        rubrics = tuple(
            RubricBinding.from_mapping(item, index=i) for i, item in enumerate(rubrics_raw)
        )
        seen_rubric_ids: set[str] = set()
        for binding in rubrics:
            if binding.rubric_id in seen_rubric_ids:
                raise ValueError(
                    f"validation.rubrics has a duplicate rubric_id {binding.rubric_id!r}"
                )
            seen_rubric_ids.add(binding.rubric_id)

        target_refs_raw = raw["target_refs"]
        if not isinstance(target_refs_raw, list):
            raise ValueError("validation.target_refs must be a list")
        target_refs = tuple(
            TargetCaseRef.from_mapping(item, label=f"target_refs[{i}]")
            for i, item in enumerate(target_refs_raw)
        )
        claimed_ref_keys: set[str] = set()
        for ref in target_refs:
            if ref.key in claimed_ref_keys:
                raise ValueError(f"validation.target_refs has a duplicate ref {ref.key!r}")
            claimed_ref_keys.add(ref.key)

        outcomes_raw = raw["outcomes"]
        if not isinstance(outcomes_raw, list):
            raise ValueError("validation.outcomes must be a list")
        outcomes = tuple(_parse_outcome(item, index=i) for i, item in enumerate(outcomes_raw))
        seen_outcome_refs: set[str] = set()
        kind_of_case: dict[str, str] = {}
        for outcome in outcomes:
            # Identity is the triple, not the case key: two rubrics may grade the
            # same criterion id, and collapsing them would let one rubric's
            # outcome silently stand in for another's.
            if outcome.ref.key in seen_outcome_refs:
                raise ValueError(f"validation.outcomes has a duplicate case {outcome.ref.key!r}")
            seen_outcome_refs.add(outcome.ref.key)
            first_kind = kind_of_case.setdefault(outcome.case.key, outcome.case.kind)
            if first_kind != outcome.case.kind:
                raise ValueError(
                    f"validation.outcomes records case {outcome.case.key!r} as both "
                    f"{first_kind!r} and {outcome.case.kind!r}: one case cannot be both the "
                    "failure under repair and untouched control evidence"
                )
            if outcome.case.kind == CASE_TARGET and outcome.ref.key not in claimed_ref_keys:
                raise ValueError(
                    f"validation.outcomes has a target case {outcome.ref.key!r} that is not among "
                    "validation.target_refs: a target outcome must be evidence about a failure "
                    "this record claims to fix, named by rubric as well as by conversation and "
                    "criterion"
                )

        # Rubric completeness: the top-level list must name exactly the rubrics
        # the outcomes actually claim to have been scored against -- no
        # omission (an outcome whose rubric is not bound at all) and no extra
        # (a rubric nobody's outcome used, which cannot be checked for staleness
        # against anything).
        required_rubric_ids = {o.rubric_id for o in outcomes}
        if required_rubric_ids != seen_rubric_ids:
            missing = sorted(required_rubric_ids - seen_rubric_ids)
            extra = sorted(seen_rubric_ids - required_rubric_ids)
            detail = []
            if missing:
                detail.append(f"missing binding(s) for {missing}")
            if extra:
                detail.append(f"unused binding(s) for {extra}")
            raise ValueError(
                "validation.rubrics does not exactly match the rubric ids the outcomes claim to "
                "have been scored against: " + "; ".join(detail)
            )

        judgements_raw = raw["judgements"]
        if not isinstance(judgements_raw, list):
            raise ValueError("validation.judgements must be a list")
        judgements = tuple(
            _parse_judgement(item, index=i, seed=seed) for i, item in enumerate(judgements_raw)
        )
        seen_judgement_keys: set[str] = set()
        target_outcome_keys = {o.case.key for o in outcomes if o.case.kind == CASE_TARGET}
        for judgement in judgements:
            if judgement.case.key in seen_judgement_keys:
                raise ValueError(
                    f"validation.judgements has a duplicate case {judgement.case.key!r}"
                )
            seen_judgement_keys.add(judgement.case.key)
            if judgement.case.key not in target_outcome_keys:
                raise ValueError(
                    f"validation.judgements references case {judgement.case.key!r}, which is not "
                    "among this record's own target outcomes: a judgement must be about a case "
                    "that was actually replayed as a target, not an absent or invented one"
                )

        evidence_model_calls = _expect_int(raw, "evidence_model_calls", label="validation")
        expected_calls = evidence_call_count(outcomes, judgements)
        if evidence_model_calls != expected_calls:
            distinct_scored_pairs = {(o.case.conversation_id, o.rubric_id) for o in outcomes}
            raise ValueError(
                f"validation.evidence_model_calls is {evidence_model_calls} but this record's own "
                f"evidence derives exactly {expected_calls}: {len(outcomes)} outcome(s) across "
                f"{len(distinct_scored_pairs)} distinct (conversation, rubric) pair(s) at a "
                f"baseline and a candidate call each, plus {len(judgements)} judgement(s). This "
                "count is derived rather than reported, so any other value is a rewritten field "
                "and not a different amount of work"
            )
        notes = _expect_str_list(raw["notes"], label="validation.notes")

        candidate_id = _expect_nonempty_str(raw, "candidate_id", label="validation")
        base_sha256 = _expect_sha256(raw, "base_sha256", label="validation")
        patch_sha256 = _expect_sha256(raw, "patch_sha256", label="validation")
        improver_command_id = _expect_nonempty_str(raw, "improver_command_id", label="validation")
        scorer_command_id = _expect_nonempty_str(raw, "scorer_command_id", label="validation")
        judge_command_id = _expect_str(raw, "judge_command_id", label="validation")
        if policy.require_judge:
            if not judge_command_id.strip():
                raise ValueError(
                    "validation.judge_command_id must be non-empty when "
                    "validation.gate.policy.require_judge is true: a judge that ran must be "
                    "identified by the command that ran it"
                )
        elif judge_command_id.strip():
            raise ValueError(
                "validation.judge_command_id is set but "
                "validation.gate.policy.require_judge is false: no judge command should be "
                "bound when the policy never required one"
            )
        return ValidationRecord(
            candidate_id=candidate_id,
            base_sha256=base_sha256,
            patch_sha256=patch_sha256,
            seed=seed,
            improver_command_id=improver_command_id,
            scorer_command_id=scorer_command_id,
            judge_command_id=judge_command_id,
            rubrics=rubrics,
            target_refs=target_refs,
            outcomes=outcomes,
            judgements=judgements,
            policy=policy,
            gate_raw=gate_raw,
            evidence_model_calls=evidence_model_calls,
            notes=notes,
        )

    def recompute_gate_mismatches(self) -> list[str]:
        """Recompute the gate from this record's own inputs and diff it byte-exact.

        ``judge_independent`` is derived from the command ids on this record --
        never from a stored boolean -- so a record whose judge and improver
        commands are identical can never claim independence by asserting it.

        The claim set is fed back in the same way, so a target ref added to the
        record without a matching replay outcome recomputes a *failing* gate and
        the persisted decision stops matching.
        """
        judge_independent = bool(self.judge_command_id) and (
            self.judge_command_id != self.improver_command_id
        )
        decision = evaluate_gate(
            self.outcomes,
            self.judgements,
            self.policy,
            judge_independent=judge_independent,
            target_refs=self.target_refs,
        )
        return _strict_equal(decision.to_dict(), dict(self.gate_raw), path="validation.gate")
