"""Failure isolation and skill attribution.

Isolation turns a scorecard into discrete, evidence-carrying failure cases.
Two rules:

* Only ``FAIL`` becomes a failure case. ``INSUFFICIENT`` never does -- a loop
  that treats "could not tell" as "broken" manufactures its own workload.
* Every failure case carries its evidence refs, so a later reader can go back
  to the exact element and check the claim.

Attribution then asks *which skill is implicated*, and is deliberately
conservative. A rubric-declared prior alone is recorded as
``rubric-prior-only`` with reduced confidence, because a prior is a hypothesis
about the skill, not an observation of it. Unattributable failures land in
their own bucket instead of being blamed on the nearest plausible skill.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .conversation import EvidenceRef, NormalizedConversation
from .rubric import Rubric, Verdict
from .scoring import ScoreCard

ATTRIB_OBSERVED_MENTION = "observed-skill-mention"
ATTRIB_OBSERVED_PATH = "observed-skill-path"
ATTRIB_RUBRIC_PRIOR = "rubric-prior-only"
ATTRIB_NONE = "unattributed"


@dataclass(frozen=True)
class SkillAttribution:
    skill_id: str
    basis: str
    confidence: float
    reasons: tuple[str, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "basis": self.basis,
            "confidence": round(self.confidence, 3),
            "reasons": list(self.reasons),
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass(frozen=True)
class FailureCase:
    """One rubric criterion that FAILED on one conversation, with evidence."""

    id: str
    conversation_id: str
    rubric_id: str
    rubric_version: int
    criterion_id: str
    criterion_title: str
    weight: int
    rationale: str
    evidence: tuple[EvidenceRef, ...]
    signals: dict[str, int]
    attributions: tuple[SkillAttribution, ...]
    workspace: str = ""
    conversation_title: str = ""

    @property
    def is_attributed(self) -> bool:
        return any(a.basis != ATTRIB_NONE for a in self.attributions)

    @property
    def primary_skill_id(self) -> str | None:
        ranked = [a for a in self.attributions if a.basis != ATTRIB_NONE]
        if not ranked:
            return None
        return max(ranked, key=lambda a: (a.confidence, a.skill_id)).skill_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "conversation_title": self.conversation_title,
            "workspace": self.workspace,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "criterion_id": self.criterion_id,
            "criterion_title": self.criterion_title,
            "weight": self.weight,
            "verdict": Verdict.FAIL.value,
            "rationale": self.rationale,
            "evidence": [e.to_dict() for e in self.evidence],
            "signals": dict(self.signals),
            "attributions": [a.to_dict() for a in self.attributions],
            "primary_skill_id": self.primary_skill_id,
        }


def conversation_signals(conversation: NormalizedConversation) -> dict[str, int]:
    """Cheap, mechanical counts that a human can re-derive from the transcript."""
    tools = conversation.tool_events
    return {
        "messages": len(conversation.messages),
        "user_messages": len(conversation.user_messages),
        "tool_events": len(tools),
        "failed_tool_events": len(conversation.failed_tool_events),
        "total_elements": conversation.total_elements,
    }


def _mention_pattern(skill_id: str) -> re.Pattern[str]:
    """Word-boundary matcher for a skill id.

    Word boundaries are not optional: a substring search for a short skill id
    matches unrelated words and produces confident false attribution.
    """
    token = re.escape(skill_id)
    # Substitute the separator characters in ONE pass. Doing it as two
    # sequential str.replace calls corrupts the character class the first call
    # inserts, because that class itself contains "_" -- which silently broke
    # mention-matching for every hyphenated skill id, i.e. almost all of them.
    flexible = re.sub(r"\\-|_", "[-_ ]", token)
    return re.compile(rf"(?<![A-Za-z0-9]){flexible}(?![A-Za-z0-9])", re.IGNORECASE)


def attribute(
    conversation: NormalizedConversation,
    *,
    criterion_implicates: Sequence[str],
    known_skill_ids: Iterable[str],
) -> tuple[SkillAttribution, ...]:
    """Attribute a failure to skills, distinguishing observation from prior."""
    known = sorted({s for s in known_skill_ids if s})
    observed: dict[str, list[tuple[str, EvidenceRef]]] = {}

    for message in conversation.messages:
        for skill_id in known:
            pattern = _mention_pattern(skill_id)
            if pattern.search(message.text):
                observed.setdefault(skill_id, []).append(
                    (
                        f"named in {message.role} message at position {message.position}",
                        EvidenceRef(message.element_id, message.position, "message"),
                    )
                )
        for tool in message.tool_events:
            haystack = f"{tool.input_text}\n{tool.output_excerpt}"
            for skill_id in known:
                needle = f"skills/{skill_id}/"
                if needle in haystack or f"skills/{skill_id}" == haystack.strip():
                    observed.setdefault(skill_id, []).append(
                        (
                            f"skill path referenced by tool {tool.name} at position {tool.position}",
                            EvidenceRef(tool.element_id, tool.position, "tool"),
                        )
                    )

    results: list[SkillAttribution] = []
    for skill_id in sorted(observed):
        entries = observed[skill_id]
        reasons = tuple(dict.fromkeys(r for r, _ in entries))
        refs = tuple(dict.fromkeys(ref for _, ref in entries))
        basis = (
            ATTRIB_OBSERVED_PATH
            if any("skill path referenced" in r for r in reasons)
            else ATTRIB_OBSERVED_MENTION
        )
        prior_bonus = 0.15 if skill_id in criterion_implicates else 0.0
        base = 0.75 if basis == ATTRIB_OBSERVED_PATH else 0.6
        results.append(
            SkillAttribution(
                skill_id=skill_id,
                basis=basis,
                confidence=min(0.95, base + prior_bonus),
                reasons=reasons,
                evidence=refs,
            )
        )

    for skill_id in criterion_implicates:
        if skill_id in observed:
            continue
        if skill_id not in known:
            continue
        results.append(
            SkillAttribution(
                skill_id=skill_id,
                basis=ATTRIB_RUBRIC_PRIOR,
                confidence=0.3,
                reasons=(
                    "declared by the rubric criterion as an expected implicated skill; "
                    "NOT observed in this conversation",
                ),
            )
        )

    if not results:
        results.append(
            SkillAttribution(
                skill_id="",
                basis=ATTRIB_NONE,
                confidence=0.0,
                reasons=(
                    "no skill was observed in the conversation and the criterion "
                    "declares no implicated skill",
                ),
            )
        )
    return tuple(results)


def failure_case_id(conversation_id: str, rubric_id: str, criterion_id: str) -> str:
    """Deterministic id so a re-run does not create duplicate cases."""
    return f"FC:{conversation_id}:{rubric_id}:{criterion_id}"


def isolate_failures(
    conversation: NormalizedConversation,
    scorecard: ScoreCard,
    rubric: Rubric,
    *,
    known_skill_ids: Iterable[str],
) -> list[FailureCase]:
    """Extract FAIL verdicts as failure cases. INSUFFICIENT is never a failure."""
    known = list(known_skill_ids)
    signals = conversation_signals(conversation)
    cases: list[FailureCase] = []
    for score in scorecard.scores:
        if score.verdict is not Verdict.FAIL:
            continue
        criterion = rubric.criterion(score.criterion_id)
        if criterion is None:
            continue
        cases.append(
            FailureCase(
                id=failure_case_id(conversation.thread_id, rubric.id, score.criterion_id),
                conversation_id=conversation.thread_id,
                conversation_title=conversation.title,
                workspace=conversation.workspace,
                rubric_id=rubric.id,
                rubric_version=rubric.version,
                criterion_id=criterion.id,
                criterion_title=criterion.title,
                weight=criterion.weight,
                rationale=score.rationale,
                evidence=score.evidence,
                signals=dict(signals),
                attributions=attribute(
                    conversation,
                    criterion_implicates=criterion.implicates,
                    known_skill_ids=known,
                ),
            )
        )
    return cases


def passing_cases(
    conversation: NormalizedConversation,
    scorecard: ScoreCard,
) -> list[tuple[str, str]]:
    """``(conversation_id, criterion_id)`` pairs that PASSED.

    These become the holdout set: a candidate that fixes a failure while
    breaking one of these is a regression, not an improvement.
    """
    return [(conversation.thread_id, s.criterion_id) for s in scorecard.passes]
