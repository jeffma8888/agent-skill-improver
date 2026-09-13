"""Baseline-versus-candidate replay, blind judging, and the gate.

Scope, stated plainly because it bounds what the gate is worth: this replays the
*scorer* over the same recorded conversations, once with the current skill text
and once with the proposed text. It measures whether the graded criteria come out
differently on the recorded evidence. It does not re-run the original agent
session, and no offline mechanism can -- the tool calls, their outputs and the
user's replies are all fixed history.

Two arms are scored per (conversation, rubric) pair and cached, so a conversation
contributes its target case and all of its holdout criteria for the price of two
model calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .. import SCHEMA_VALIDATION
from ..config import ROLE_JUDGE, ROLE_SCORER, PromptConfig
from ..domain.candidate import Candidate
from ..domain.conversation import NormalizedConversation
from ..domain.integrity import verify_candidate_integrity
from ..domain.rubric import Rubric, Verdict
from ..domain.validation import (
    ARM_BASELINE,
    ARM_CANDIDATE,
    CASE_HOLDOUT,
    CASE_TARGET,
    GateDecision,
    GatePolicy,
    Judgement,
    ReplayCase,
    ReplayOutcome,
    RubricBinding,
    blind_arms,
    evaluate_gate,
    parse_judgement,
)
from ..errors import AsiError
from ..ports import ModelRunner
from . import prompts
from .pipeline import score_conversation


@dataclass(frozen=True)
class ValidationResult:
    candidate_id: str
    base_sha256: str
    patch_sha256: str
    seed: str
    improver_command_id: str
    scorer_command_id: str
    judge_command_id: str
    rubrics: tuple[RubricBinding, ...]
    outcomes: tuple[ReplayOutcome, ...]
    judgements: tuple[Judgement, ...]
    decision: GateDecision
    model_calls: int
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VALIDATION,
            "candidate_id": self.candidate_id,
            "base_sha256": self.base_sha256,
            "patch_sha256": self.patch_sha256,
            "seed": self.seed,
            "improver_command_id": self.improver_command_id,
            "scorer_command_id": self.scorer_command_id,
            "judge_command_id": self.judge_command_id,
            "rubrics": [r.to_dict() for r in self.rubrics],
            "gate": self.decision.to_dict(),
            "outcomes": [o.to_dict() for o in self.outcomes],
            "judgements": [j.to_dict() for j in self.judgements],
            "model_calls": self.model_calls,
            "notes": list(self.notes),
        }


def _rubric_for_pair(
    rubrics: Mapping[str, Rubric], rubric_id: str, criterion_id: str
) -> Rubric | None:
    rubric = rubrics.get(rubric_id)
    if rubric is not None and rubric.criterion(criterion_id) is not None:
        return rubric
    for candidate_rubric in rubrics.values():
        if candidate_rubric.criterion(criterion_id) is not None:
            return candidate_rubric
    return None


def validate_candidate(
    candidate: Candidate,
    runner: ModelRunner,
    *,
    rubrics: Mapping[str, Rubric],
    conversations: Mapping[str, NormalizedConversation],
    base_text: str,
    cfg: PromptConfig,
    policy: GatePolicy,
    seed: str,
    holdout_pairs: Sequence[tuple[str, str]] = (),
    max_extra_holdout_conversations: int = 3,
    known_skill_ids: Sequence[str] = (),
) -> ValidationResult:
    """Replay both arms, judge blind, and decide the gate.

    The candidate's derived fields are recomputed against ``base_text`` first. The
    candidate arm is scored with ``patch.new_text``, so a record whose stored hash,
    diff or challenge describes different text would produce a gate decision about
    one change and a token about another.

    Judge independence is derived from the actual commands that ran the judge
    and improver roles (``runner.command_id``), never from a caller-supplied
    label comparison: the persisted validation binds these two command ids, and
    a later verifier re-derives independence from them the same way.
    """
    verify_candidate_integrity(candidate, base_text=base_text, action="validate")
    notes: list[str] = []
    skill_id = candidate.target.skill_id
    new_text = candidate.patch.new_text
    scorer_command_id = runner.command_id(ROLE_SCORER)
    improver_command_id = candidate.improver_command_id
    judge_command_id = runner.command_id(ROLE_JUDGE) if policy.require_judge else ""
    judge_independent = bool(judge_command_id) and judge_command_id != improver_command_id

    targets: list[tuple[ReplayCase, str]] = []
    seen_keys: set[str] = set()
    for ref in candidate.rubric_failures:
        rubric = _rubric_for_pair(rubrics, ref.rubric_id, ref.criterion_id)
        if rubric is None:
            notes.append(
                f"criterion {ref.criterion_id!r} is not in any loaded rubric; case skipped"
            )
            continue
        if ref.conversation_id not in conversations:
            notes.append(
                f"conversation {ref.conversation_id} is not available for replay; case skipped"
            )
            continue
        case = ReplayCase(ref.conversation_id, ref.criterion_id, CASE_TARGET)
        if case.key in seen_keys:
            continue
        seen_keys.add(case.key)
        targets.append((case, rubric.id))

    target_conversations = {c.conversation_id for c, _ in targets}
    holdouts: list[tuple[ReplayCase, str]] = []
    extra_conversations: set[str] = set()
    for conversation_id, criterion_id in holdout_pairs:
        key = f"{conversation_id}::{criterion_id}"
        if key in seen_keys:
            continue
        if conversation_id not in conversations:
            continue
        rubric = _rubric_for_pair(rubrics, "", criterion_id)
        if rubric is None:
            continue
        if conversation_id not in target_conversations:
            if conversation_id not in extra_conversations:
                if len(extra_conversations) >= max_extra_holdout_conversations:
                    continue
                extra_conversations.add(conversation_id)
        seen_keys.add(key)
        holdouts.append((ReplayCase(conversation_id, criterion_id, CASE_HOLDOUT), rubric.id))

    cache: dict[tuple[str, str, str], dict[str, Verdict]] = {}
    calls = 0

    def verdicts(conversation_id: str, rubric_id: str, arm: str) -> dict[str, Verdict]:
        nonlocal calls
        key = (conversation_id, rubric_id, arm)
        if key in cache:
            return cache[key]
        rubric = rubrics[rubric_id]
        conversation = conversations[conversation_id]
        card, _ = score_conversation(
            conversation,
            rubric,
            runner,
            cfg=cfg,
            seed=f"{seed}:{arm}",
            known_skill_ids=known_skill_ids,
            skill_under_evaluation=skill_id,
            skill_text=base_text if arm == ARM_BASELINE else new_text,
        )
        calls += 1
        result = {s.criterion_id: s.verdict for s in card.scores}
        cache[key] = result
        return result

    outcomes: list[ReplayOutcome] = []
    for case, rubric_id in targets + holdouts:
        try:
            baseline = verdicts(case.conversation_id, rubric_id, ARM_BASELINE)
            proposed = verdicts(case.conversation_id, rubric_id, ARM_CANDIDATE)
        except AsiError as exc:
            notes.append(f"replay of {case.key} failed: {exc}")
            continue
        if case.criterion_id not in baseline or case.criterion_id not in proposed:
            notes.append(f"replay of {case.key} produced no verdict for that criterion")
            continue
        outcomes.append(
            ReplayOutcome(
                case=case,
                baseline=baseline[case.criterion_id],
                candidate=proposed[case.criterion_id],
                rubric_id=rubric_id,
            )
        )

    judgements: list[Judgement] = []
    if policy.require_judge:
        for case, rubric_id in targets:
            rubric = rubrics[rubric_id]
            criterion = rubric.criterion(case.criterion_id)
            arms = blind_arms(case.key, seed)
            texts = {
                label: (base_text if arm == ARM_BASELINE else new_text)
                for label, arm in arms.items()
            }
            rationale = ""
            for ref in candidate.rubric_failures:
                if (
                    ref.criterion_id == case.criterion_id
                    and ref.conversation_id == case.conversation_id
                ):
                    rationale = candidate.hypothesis
                    break
            user = prompts.judge_user_prompt(
                case=case,
                criterion_question=criterion.question if criterion else case.criterion_id,
                failure_rationale=rationale,
                texts_by_label=texts,
                cfg=cfg,
                seed=seed,
            )
            try:
                payload = runner.run(
                    role=ROLE_JUDGE,
                    system=prompts.judge_system_prompt(),
                    user=user,
                    response_schema="asi.judgement.v1",
                )
                calls += 1
                judgements.append(parse_judgement(payload, case=case, arms=arms))
            except (AsiError, ValueError) as exc:
                notes.append(f"judge for {case.key} produced no usable verdict: {exc}")

    if policy.require_judge and not judge_independent:
        notes.append(
            "judge and improver run the same command: the blind preference below is correlated "
            "with the proposal and is not independent evidence"
        )
    decision = evaluate_gate(outcomes, judgements, policy, judge_independent=judge_independent)
    used_rubric_ids = sorted({rid for _, rid in targets} | {rid for _, rid in holdouts})
    rubric_bindings = tuple(
        RubricBinding(
            rubric_id=rid,
            version=rubrics[rid].version,
            source_sha256=rubrics[rid].source_sha256,
        )
        for rid in used_rubric_ids
    )
    return ValidationResult(
        candidate_id=candidate.id,
        base_sha256=candidate.target.base_sha256,
        patch_sha256=candidate.recomputed_patch_sha256,
        seed=seed,
        improver_command_id=improver_command_id,
        scorer_command_id=scorer_command_id,
        judge_command_id=judge_command_id,
        rubrics=rubric_bindings,
        outcomes=tuple(outcomes),
        judgements=tuple(judgements),
        decision=decision,
        model_calls=calls,
        notes=tuple(notes),
    )
