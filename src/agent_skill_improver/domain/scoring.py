"""Scorecard contract: parse a model's JSON, then *distrust* it correctly.

Two rules carry the honesty of the whole loop and both live here:

1. A ``FAIL`` whose cited evidence does not resolve against the conversation is
   downgraded to ``INSUFFICIENT``. A confident accusation with a bad citation is
   the single most dangerous output a scorer can produce, because everything
   downstream (isolation, proposal, promotion) treats a FAIL as ground truth.
2. Unknown criterion ids are recorded as *extraneous* rather than silently
   dropped, and criteria the model never mentioned come back as INSUFFICIENT
   rather than PASS. Silence is not a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .. import SCHEMA_SCORECARD
from .conversation import EvidenceRef, NormalizedConversation
from .redaction import kept_prose, quoted_value
from .rubric import Rubric, Verdict

MAX_RATIONALE_CHARS = 2000


@dataclass(frozen=True)
class CriterionScore:
    criterion_id: str
    verdict: Verdict
    rationale: str
    evidence: tuple[EvidenceRef, ...] = ()
    confidence: float | None = None
    # Set when the raw verdict was changed by a local rule (e.g. FAIL with
    # unresolvable evidence -> INSUFFICIENT). Kept so the receipt can be honest
    # about how many of the model's verdicts we refused to take at face value.
    downgraded_from: Verdict | None = None
    downgrade_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "verdict": self.verdict.value,
            "rationale": self.rationale,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence,
            "downgraded_from": self.downgraded_from.value if self.downgraded_from else None,
            "downgrade_reason": self.downgrade_reason,
        }


@dataclass(frozen=True)
class ScoreCard:
    conversation_id: str
    rubric_id: str
    rubric_version: int
    rubric_sha256: str
    scores: tuple[CriterionScore, ...]
    scorer_role: str = "scorer"
    scorer_command_id: str = ""
    extraneous_criterion_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def score_for(self, criterion_id: str) -> CriterionScore | None:
        for s in self.scores:
            if s.criterion_id == criterion_id:
                return s
        return None

    @property
    def failures(self) -> tuple[CriterionScore, ...]:
        return tuple(s for s in self.scores if s.verdict is Verdict.FAIL)

    @property
    def passes(self) -> tuple[CriterionScore, ...]:
        return tuple(s for s in self.scores if s.verdict is Verdict.PASS)

    @property
    def insufficient(self) -> tuple[CriterionScore, ...]:
        return tuple(s for s in self.scores if s.verdict is Verdict.INSUFFICIENT)

    @property
    def downgraded(self) -> tuple[CriterionScore, ...]:
        return tuple(s for s in self.scores if s.downgraded_from is not None)

    def weighted_fail_score(self, rubric: Rubric) -> int:
        """Sum of weights of decided failures. INSUFFICIENT contributes zero."""
        total = 0
        for s in self.failures:
            crit = rubric.criterion(s.criterion_id)
            total += crit.weight if crit else 1
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_SCORECARD,
            "conversation_id": self.conversation_id,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "rubric_sha256": self.rubric_sha256,
            "scorer_role": self.scorer_role,
            "scorer_command_id": self.scorer_command_id,
            "extraneous_criterion_ids": list(self.extraneous_criterion_ids),
            "notes": list(self.notes),
            "scores": [s.to_dict() for s in self.scores],
        }


def parse_scorecard(
    payload: Any,
    *,
    rubric: Rubric,
    conversation: NormalizedConversation,
    scorer_command_id: str = "",
) -> ScoreCard:
    """Turn raw model JSON into a :class:`ScoreCard`, applying trust rules.

    Raises ``ValueError`` when the payload violates the contract badly enough
    that no verdict at all can be recovered (caller quarantines the record).
    Individual bad criteria are *not* fatal -- they become INSUFFICIENT.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"scorecard must be a JSON object, got {type(payload).__name__}")
    schema = payload.get("schema")
    if schema != SCHEMA_SCORECARD:
        raise ValueError(
            f"scorecard schema must be {SCHEMA_SCORECARD!r}, got {quoted_value(schema)}"
        )
    reported_id = payload.get("conversation_id")
    if reported_id is not None and str(reported_id) != conversation.thread_id:
        raise ValueError(
            "scorecard conversation_id does not match the record being scored "
            f"({quoted_value(reported_id)} != {conversation.thread_id!r})"
        )
    raw_items = payload.get("criteria")
    if not isinstance(raw_items, list):
        raise ValueError("scorecard needs a 'criteria' array")

    known = set(rubric.criterion_ids)
    seen: dict[str, CriterionScore] = {}
    extraneous: list[str] = []
    notes: list[str] = []

    for idx, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            notes.append(f"criteria[{idx}] is not an object; ignored")
            continue
        cid = item.get("criterion_id")
        if not isinstance(cid, str) or not cid.strip():
            notes.append(f"criteria[{idx}] has no criterion_id; ignored")
            continue
        if cid not in known:
            extraneous.append(cid)
            notes.append(f"criterion {quoted_value(cid)} is not in rubric {rubric.id}; ignored")
            continue
        if cid in seen:
            notes.append(f"criterion {quoted_value(cid)} scored more than once; first verdict kept")
            continue
        seen[cid] = _build_score(item, rubric=rubric, conversation=conversation, notes=notes)

    # Criteria the model never mentioned are INSUFFICIENT, never PASS.
    for cid in rubric.criterion_ids:
        if cid not in seen:
            seen[cid] = CriterionScore(
                criterion_id=cid,
                verdict=Verdict.INSUFFICIENT,
                rationale="",
                downgrade_reason="criterion absent from scorer output",
            )
            notes.append(f"criterion {cid!r} absent from scorer output; recorded INSUFFICIENT")

    ordered = tuple(seen[cid] for cid in rubric.criterion_ids)
    return ScoreCard(
        conversation_id=conversation.thread_id,
        rubric_id=rubric.id,
        rubric_version=rubric.version,
        rubric_sha256=rubric.source_sha256,
        scores=ordered,
        scorer_command_id=scorer_command_id,
        extraneous_criterion_ids=tuple(sorted(set(extraneous))),
        notes=tuple(notes),
    )


#: How many malformed evidence refs a downgrade reason spells out. The reason is
#: persisted in the scorecard and the failure record, and the model decides how
#: many refs there are.
MAX_SHOWN_BAD_REFS = 3


def _evidence_list(raw: Any, bad_refs: list[str]) -> list[Any]:
    """The criterion's ``evidence`` as the array the contract asks for, or nothing.

    The field is the model's and used to be iterated as it came. A number or a
    boolean there raised ``TypeError`` out of the parser, which no caller
    classifies: measured before, one such scorer reply aborted a committed daily
    run as an internal error with no receipt. A string iterated character by
    character and an object by its keys, each a "malformed ref" of its own, so
    one wrong field became hundreds of refusals in the persisted reason. Any
    shape but an array (or null, read as empty) is now one named malformed ref,
    so a decided verdict on a criterion that requires evidence goes the way a
    malformed ref always sends it: INSUFFICIENT.
    """
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        bad_refs.append(f"evidence must be an array of refs, got {type(raw).__name__}")
        return []
    return list(raw)


def _confidence(raw: Any) -> float | None:
    """The criterion's ``confidence`` as a float, or None when it is not one.

    Stored with the score and never decided on, and the model's to fill. A
    boolean is not a confidence, though Python counts it as a number. JSON
    integers are unbounded and ``loads_untrusted`` leaves them alone, so one
    too large for a float raised ``OverflowError`` out of the parser, which no
    caller classifies: measured, a scorer reply with ``10**400`` there aborted
    a committed daily run with no receipt, as the ``evidence`` field did.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    try:
        value = float(raw)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def _build_score(
    item: Mapping[str, Any],
    *,
    rubric: Rubric,
    conversation: NormalizedConversation,
    notes: list[str],
) -> CriterionScore:
    cid = str(item["criterion_id"])
    criterion = rubric.criterion(cid)
    assert criterion is not None  # guarded by caller

    try:
        verdict = Verdict.parse(item.get("verdict"))
    except ValueError as exc:
        notes.append(f"criterion {cid}: {exc}; recorded INSUFFICIENT")
        return CriterionScore(
            criterion_id=cid,
            verdict=Verdict.INSUFFICIENT,
            rationale=_clip(item.get("rationale")),
            downgrade_reason=f"unparseable verdict: {exc}",
        )

    refs: list[EvidenceRef] = []
    bad_refs: list[str] = []
    for raw_ref in _evidence_list(item.get("evidence"), bad_refs):
        try:
            refs.append(EvidenceRef.from_dict(raw_ref))
        except ValueError as exc:
            bad_refs.append(str(exc))

    unresolved = conversation.unresolved(refs)
    # A ref that does not resolve is the model's claim, not evidence: it is
    # named in the reason (bounded, redacted) and stored nowhere else.
    resolved = tuple(r for r in refs if r not in unresolved)
    conf = _confidence(item.get("confidence"))
    rationale = _clip(item.get("rationale"))

    # Trust rule: a decided verdict needs resolvable evidence when the
    # criterion demands it. Otherwise it is INSUFFICIENT, not FAIL.
    if verdict.is_decided and criterion.requires_evidence:
        reason = ""
        if bad_refs:
            shown = "; ".join(bad_refs[:MAX_SHOWN_BAD_REFS])
            hidden = len(bad_refs) - MAX_SHOWN_BAD_REFS
            reason = f"malformed evidence refs: {shown}" + (
                f" and {hidden} more" if hidden > 0 else ""
            )
        elif len(refs) < max(rubric.min_evidence_refs, 1):
            reason = f"needs >= {max(rubric.min_evidence_refs, 1)} evidence ref(s), got {len(refs)}"
        elif unresolved:
            shown = ", ".join(f"{quoted_value(r.element_id)}@{r.position}" for r in unresolved[:3])
            more = f" and {len(unresolved) - 3} more" if len(unresolved) > 3 else ""
            reason = f"evidence does not resolve in conversation: {shown}{more}"
        if reason:
            notes.append(f"criterion {cid}: {verdict.value} downgraded to INSUFFICIENT ({reason})")
            return CriterionScore(
                criterion_id=cid,
                verdict=Verdict.INSUFFICIENT,
                rationale=rationale,
                evidence=resolved,
                confidence=conf,
                downgraded_from=verdict,
                downgrade_reason=reason,
            )

    return CriterionScore(
        criterion_id=cid,
        verdict=verdict,
        rationale=rationale,
        evidence=resolved,
        confidence=conf,
    )


def _clip(value: Any) -> str:
    return kept_prose(value, limit=MAX_RATIONALE_CHARS)
