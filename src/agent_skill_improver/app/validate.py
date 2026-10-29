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

Every record this module writes must parse under its own strict parser
(``ValidationRecord.parse``), on the error paths as much as on the happy one. So
the rubric bindings and the judgements are derived from the outcomes that were
actually *persisted*, never from the cases that were merely attempted, and a
target the candidate claims but that could not be replayed fails the gate here
rather than becoming a nominally validated record that review later has to call
tampered.
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
    TargetCaseRef,
    blind_arms,
    evaluate_gate,
    evidence_call_count,
    parse_judgement,
    runs_the_same_command,
)
from ..errors import AsiError, ConfigError
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
    target_refs: tuple[TargetCaseRef, ...]
    outcomes: tuple[ReplayOutcome, ...]
    judgements: tuple[Judgement, ...]
    decision: GateDecision
    evidence_model_calls: int
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
            "target_refs": [r.to_dict() for r in self.target_refs],
            "gate": self.decision.to_dict(),
            "outcomes": [o.to_dict() for o in self.outcomes],
            "judgements": [j.to_dict() for j in self.judgements],
            "evidence_model_calls": self.evidence_model_calls,
            "notes": list(self.notes),
        }


def _rubric_grading(rubrics: Mapping[str, Rubric], criterion_id: str) -> Rubric | None:
    """The first loaded rubric that grades ``criterion_id``, or ``None``.

    Used for holdout pairs only. A holdout is named by ``(conversation,
    criterion)`` with no rubric attached, so the rubric has to be looked up.

    A *target* is deliberately not resolved this way. A target names its rubric
    in the candidate's own failure ref, and the outcome is bound to that exact
    triple, so falling back to "whichever other rubric happens to grade a
    criterion with this id" would produce evidence attributed to an examiner the
    candidate never claimed -- the substitution the binding exists to prevent.
    """
    for rubric in rubrics.values():
        if rubric.criterion(criterion_id) is not None:
            return rubric
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

    Every target is bound to the candidate's own failure ref by the exact
    ``(rubric_id, conversation_id, criterion_id)`` triple: the rubric named by
    the ref must be loaded and must grade the criterion, with no fallback to
    another rubric that happens to use the same criterion id. A claimed target
    that cannot be replayed is still recorded in the claim set, so the gate
    fails here with a named coverage reason instead of producing a record that
    passes now and refuses itself at review time.
    """
    verify_candidate_integrity(candidate, base_text=base_text, action="validate")
    notes: list[str] = []
    skill_id = candidate.target.skill_id
    new_text = candidate.patch.new_text
    cap = cfg.max_skill_text_chars
    if 0 < cap < max(len(base_text), len(new_text)):
        # Both arms are shown to the scorer, and both options to the judge,
        # clipped at this cap. A decision about text no model read to the end
        # is not evidence about the candidate, so none is produced: the daily
        # pass refuses the replacement at propose, and this is the same rule
        # for a record proposed under a larger cap than the one in force now.
        readers = "the scorer and judge prompts" if policy.require_judge else "the scorer prompt"
        raise ConfigError(
            f"candidate {candidate.id}: the base is {len(base_text)} chars and the replacement "
            f"{len(new_text)} chars, and [prompt].max_skill_text_chars = {cap} would clip "
            f"whichever is longer in {readers}; a gate decision about text no model read to "
            "the end is not evidence, so no model was called. Raise the cap to at least "
            f"{max(len(base_text), len(new_text))} and run again."
        )
    scorer_command_id = runner.command_id(ROLE_SCORER)
    improver_command_id = candidate.improver_command_id
    judge_command_id = runner.command_id(ROLE_JUDGE) if policy.require_judge else ""
    # What runs, not the whole id: every id starts with its role, so comparing
    # whole ids called a judge running the improver's very command independent.
    judge_independent = bool(judge_command_id) and not runs_the_same_command(
        judge_command_id, improver_command_id
    )

    # The claim set is recorded whether or not a case turns out to be
    # replayable, because it is what the gate is asked to prove. A claimed
    # target with no outcome is a coverage failure with a named reason, not a
    # case that quietly disappears from the evidence.
    claimed_refs: list[TargetCaseRef] = []
    targets: list[tuple[ReplayCase, str]] = []
    seen_refs: set[str] = set()
    for ref in candidate.rubric_failures:
        claim = TargetCaseRef(
            rubric_id=ref.rubric_id,
            conversation_id=ref.conversation_id,
            criterion_id=ref.criterion_id,
        )
        if claim.key in seen_refs:
            continue
        seen_refs.add(claim.key)
        claimed_refs.append(claim)
        rubric = rubrics.get(ref.rubric_id)
        if rubric is None or rubric.criterion(ref.criterion_id) is None:
            notes.append(
                f"target {claim.key} names rubric {ref.rubric_id!r}, which is not loaded or does "
                f"not grade criterion {ref.criterion_id!r}; the case was NOT replayed, and the "
                "gate refuses for want of coverage rather than substituting another rubric"
            )
            continue
        if ref.conversation_id not in conversations:
            notes.append(
                f"conversation {ref.conversation_id} is not available for replay, so target "
                f"{claim.key} was NOT replayed and the gate refuses for want of coverage"
            )
            continue
        targets.append((ReplayCase(ref.conversation_id, ref.criterion_id, CASE_TARGET), rubric.id))

    target_conversations = {c.conversation_id for c, _ in targets}
    target_case_keys = {c.key for c, _ in targets}
    holdouts: list[tuple[ReplayCase, str]] = []
    seen_holdout_keys: set[str] = set()
    extra_conversations: set[str] = set()
    for conversation_id, criterion_id in holdout_pairs:
        key = f"{conversation_id}::{criterion_id}"
        # A case is either the failure under repair or untouched control
        # evidence, never both: the record refuses a case recorded as two kinds.
        if key in target_case_keys or key in seen_holdout_keys:
            continue
        if conversation_id not in conversations:
            continue
        rubric = _rubric_grading(rubrics, criterion_id)
        if rubric is None:
            continue
        if conversation_id not in target_conversations:
            if conversation_id not in extra_conversations:
                if len(extra_conversations) >= max_extra_holdout_conversations:
                    continue
                extra_conversations.add(conversation_id)
        seen_holdout_keys.add(key)
        holdouts.append((ReplayCase(conversation_id, criterion_id, CASE_HOLDOUT), rubric.id))

    cache: dict[tuple[str, str, str], dict[str, Verdict]] = {}
    attempted_calls = 0

    def verdicts(conversation_id: str, rubric_id: str, arm: str) -> dict[str, Verdict]:
        nonlocal attempted_calls
        key = (conversation_id, rubric_id, arm)
        if key in cache:
            return cache[key]
        rubric = rubrics[rubric_id]
        conversation = conversations[conversation_id]
        attempted_calls += 1
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
        # Only cases with a persisted target outcome are judged. A judgement
        # about a case that produced no outcome is evidence about nothing: the
        # record's own parser refuses it, so producing one would write a record
        # that cannot be read back.
        judged_keys: set[str] = set()
        for outcome in outcomes:
            if outcome.case.kind != CASE_TARGET:
                continue
            case = outcome.case
            if case.key in judged_keys:
                continue
            judged_keys.add(case.key)
            rubric = rubrics[outcome.rubric_id]
            criterion = rubric.criterion(case.criterion_id)
            arms = blind_arms(case.key, seed)
            texts = {
                label: (base_text if arm == ARM_BASELINE else new_text)
                for label, arm in arms.items()
            }
            rationale = ""
            for ref in candidate.rubric_failures:
                if (
                    ref.rubric_id == outcome.rubric_id
                    and ref.criterion_id == case.criterion_id
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
                attempted_calls += 1
                payload = runner.run(
                    role=ROLE_JUDGE,
                    system=prompts.judge_system_prompt(),
                    user=user,
                    response_schema="asi.judgement.v1",
                )
                judgements.append(parse_judgement(payload, case=case, arms=arms))
            except (AsiError, ValueError) as exc:
                notes.append(f"judge for {case.key} produced no usable verdict: {exc}")

    if policy.require_judge and not judge_independent:
        notes.append(
            "judge and improver run the same command: the blind preference below is correlated "
            "with the proposal and is not independent evidence"
        )
    decision = evaluate_gate(
        outcomes,
        judgements,
        policy,
        judge_independent=judge_independent,
        target_refs=tuple(claimed_refs),
    )
    # Bindings come from the outcomes that were persisted, not from the cases
    # that were attempted. A rubric whose replay produced nothing would
    # otherwise be bound with nothing to check it against, and the record's own
    # parser rejects such an unused binding.
    used_rubric_ids = sorted({o.rubric_id for o in outcomes})
    rubric_bindings = tuple(
        RubricBinding(
            rubric_id=rid,
            version=rubrics[rid].version,
            source_sha256=rubrics[rid].source_sha256,
        )
        for rid in used_rubric_ids
    )
    evidence_calls = evidence_call_count(outcomes, judgements)
    if attempted_calls != evidence_calls:
        notes.append(
            f"{attempted_calls} model call(s) were attempted and {evidence_calls} produced the "
            "evidence recorded here; evidence_model_calls counts only the calls behind the "
            "persisted outcomes and judgements, because that is the number a verifier can "
            "recompute from this record"
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
        target_refs=tuple(claimed_refs),
        outcomes=tuple(outcomes),
        judgements=tuple(judgements),
        decision=decision,
        evidence_model_calls=evidence_calls,
        notes=tuple(notes),
    )
