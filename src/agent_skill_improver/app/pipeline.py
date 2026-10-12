"""Stages one through three: score, isolate, propose.

Each stage is a function over ports, so every one is testable with a fake model
and a temporary skill tree. Exceptions are the control flow that keeps a batch
alive: :class:`RecordDefect` means quarantine this record, and
:class:`TransientFailure` means defer it. Neither ever aborts a batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..config import ROLE_IMPROVER, ROLE_SCORER, PromptConfig
from ..domain.redaction import credential_labels
from ..domain.candidate import (
    Candidate,
    STATUS_PROPOSED,
    challenge_for,
    candidate_defects,
    failure_refs_from_cases,
    make_patch,
    parse_proposal,
    sources_from_cases,
)
from ..domain.conversation import NormalizedConversation
from ..domain.failure import FailureCase, isolate_failures, passing_cases
from ..domain.rubric import Rubric
from ..domain.scoring import ScoreCard, parse_scorecard
from ..domain.skills import BOM, SkillGroup, parse_frontmatter
from ..errors import CredentialInSkill, MalformedModelOutput, RecordDefect, SelfMutationRefused
from ..hashing import sha256_text
from ..ports import ModelRunner
from . import prompts

MAX_GROWTH_FACTOR = 3.0
MAX_ABSOLUTE_GROWTH_CHARS = 20_000
#: A replacement that keeps less than this share of a base at least
#: ``SHRINK_RULE_MIN_BASE_CHARS`` long is a truncation or a wholesale rewrite,
#: not a targeted fix. A model that hits its output limit returns the first
#: part of the document and stops; measured before this rule, such a
#: replacement passed every structural check with thousands of removed lines.
MIN_RETAINED_FRACTION = 0.5
SHRINK_RULE_MIN_BASE_CHARS = 2_000


@dataclass(frozen=True)
class ScoreResult:
    scorecard: ScoreCard
    failures: tuple[FailureCase, ...]
    passes: tuple[tuple[str, str], ...]
    prompt_meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scorecard": self.scorecard.to_dict(),
            "failures": [f.to_dict() for f in self.failures],
            "passes": [list(p) for p in self.passes],
            "prompt": dict(self.prompt_meta),
        }


def score_conversation(
    conversation: NormalizedConversation,
    rubric: Rubric,
    runner: ModelRunner,
    *,
    cfg: PromptConfig,
    seed: str,
    known_skill_ids: Sequence[str] = (),
    skill_under_evaluation: str = "",
    skill_text: str = "",
) -> tuple[ScoreCard, dict[str, Any]]:
    """Run the scorer role and apply the local trust rules to its output."""
    user, render = prompts.scorer_user_prompt(
        conversation,
        rubric,
        cfg=cfg,
        seed=seed,
        known_skill_ids=known_skill_ids,
        skill_under_evaluation=skill_under_evaluation,
        skill_text=skill_text,
    )
    payload = runner.run(
        role=ROLE_SCORER,
        system=prompts.scorer_system_prompt(),
        user=user,
        response_schema="asi.scorecard.v1",
    )
    try:
        card = parse_scorecard(
            payload,
            rubric=rubric,
            conversation=conversation,
            scorer_command_id=runner.command_id(ROLE_SCORER),
        )
    except ValueError as exc:
        raise MalformedModelOutput(
            f"scorer output violated the scorecard contract: {exc}",
            record_id=conversation.thread_id,
        ) from exc
    return card, render.to_dict()


def score_and_isolate(
    conversation: NormalizedConversation,
    rubric: Rubric,
    runner: ModelRunner,
    *,
    cfg: PromptConfig,
    seed: str,
    known_skill_ids: Sequence[str] = (),
) -> ScoreResult:
    card, meta = score_conversation(
        conversation,
        rubric,
        runner,
        cfg=cfg,
        seed=seed,
        known_skill_ids=known_skill_ids,
    )
    failures = isolate_failures(conversation, card, rubric, known_skill_ids=known_skill_ids)
    return ScoreResult(
        scorecard=card,
        failures=tuple(failures),
        passes=tuple(passing_cases(conversation, card)),
        prompt_meta=meta,
    )


def skill_text_defects(new_text: str, *, base_text: str, skill_id: str) -> list[str]:
    """Reasons a proposed replacement must not be offered at all.

    These are structural, not stylistic. A file that loses its frontmatter never
    loads again -- silently -- and that is a worse outcome than the failure the
    proposal was trying to fix.
    """
    problems: list[str] = []
    new_fields, new_open, _ = parse_frontmatter(new_text)
    base_fields, base_open, _ = parse_frontmatter(base_text)
    if not new_open:
        problems.append(
            "replacement has no opening '---' delimiter: frontmatter would be unparseable "
            "and the skill would silently never load"
        )
    if not new_fields.get("name"):
        problems.append("replacement frontmatter has no 'name' field")
    if new_text.startswith(BOM) and not base_text.startswith(BOM):
        # Dropping a mark the base has is a visible first-line change, shown as
        # \ufeff in the diff; adding one the base lacks is an invisible byte
        # the model wrote, and nothing it fixes.
        problems.append(
            "replacement begins with a UTF-8 byte-order mark (U+FEFF) the base does not have; "
            "an invisible byte the model added is not an improvement"
        )
    if base_open and base_fields.get("name") and new_fields.get("name") != base_fields.get("name"):
        problems.append(
            f"replacement changes the frontmatter name from {base_fields.get('name')!r} to "
            f"{new_fields.get('name')!r}; renaming a skill is not an in-place improvement"
        )
    if base_fields.get("description") and not new_fields.get("description"):
        problems.append("replacement drops the frontmatter 'description' used for skill discovery")
    base_len = max(len(base_text), 1)
    if len(new_text) > base_len * MAX_GROWTH_FACTOR + MAX_ABSOLUTE_GROWTH_CHARS:
        problems.append(
            f"replacement is {len(new_text)} chars against a {len(base_text)}-char base; a rewrite "
            "that large cannot be reviewed as a targeted fix"
        )
    if (
        len(base_text) >= SHRINK_RULE_MIN_BASE_CHARS
        and len(new_text) < len(base_text) * MIN_RETAINED_FRACTION
    ):
        problems.append(
            f"replacement is {len(new_text)} chars against a {len(base_text)}-char base: more "
            "than half the document is gone; a truncated or wholesale rewrite cannot be "
            "reviewed as a targeted fix"
        )
    # The transcript the model saw was redacted and a base skill holding a
    # credential was refused before the model ran, so a credential shape in the
    # output can only have come from the model. It must not become a candidate,
    # a backup or a judge prompt. The shape is named; the value is never copied
    # into a receipt.
    introduced = credential_labels(new_text)
    if introduced:
        problems.append(
            f"replacement contains a credential ({', '.join(introduced)}); a model-written "
            "token or key never becomes a reviewable candidate"
        )
    return problems


def read_canonical_text(
    read_text: Callable[[Path], str] | None, path: Path, *, skill_id: str
) -> str:
    """Read the canonical document a proposal is built against.

    The store's read raises its own named refusals (a link, a fifo, bad UTF-8)
    and leaves a plain OSError for the caller to name. Any OSError this read
    meets (permissions changed, the file removed mid-run, a filesystem fault)
    is one skill's defect for this run, raised as a RecordDefect the loop
    quarantines, not an exception that ends the run with no receipt. A
    quarantine entry does not blacklist the skill, so a condition that clears
    by itself is simply retried by the next run.
    """
    try:
        if read_text is not None:
            return read_text(path)
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecordDefect(
            f"skill {skill_id!r}: its canonical document {path} could not be read "
            f"({exc.strerror or exc}); nothing is proposed against a document this run "
            "cannot see",
            record_id=skill_id,
        ) from exc


def propose_candidate(
    group: SkillGroup,
    cases: Sequence[FailureCase],
    runner: ModelRunner,
    *,
    conversations: Mapping[str, NormalizedConversation],
    cfg: PromptConfig,
    seed: str,
    candidate_id: str,
    created_at: str,
    protected_skill_ids: Sequence[str] = (),
    max_cases: int = 6,
    read_text: Callable[[Path], str] | None = None,
) -> Candidate:
    """Ask the improver for a replacement and wrap it in a reviewable candidate.

    ``read_text`` is the skill store's read when the caller has one, so the
    base the improver sees is read with the same refusals (a link, a fifo) as
    the base ``review`` and ``apply`` check against later.
    """
    if group.canonical is None:
        raise RecordDefect(
            f"skill {group.skill_id!r} has no canonical editable copy ({group.canonical_reason}); "
            "refusing to propose an edit rather than pointing one at a mirror",
            record_id=group.skill_id,
        )
    if group.skill_id in protected_skill_ids:
        raise SelfMutationRefused(
            f"skill {group.skill_id!r} is listed in [guard].protected_skill_ids and is never "
            "a proposal target"
        )
    canonical = group.canonical
    base_text = read_canonical_text(read_text, Path(canonical.path), skill_id=group.skill_id)
    found = credential_labels(base_text)
    if found:
        # The transcript is redacted before any model sees it; the skill text
        # cannot be, because the improver rewrites it whole and a placeholder
        # would come back as the file's new content. A credential-bearing skill
        # is therefore refused outright, before any role is called.
        raise CredentialInSkill(
            f"skill {group.skill_id!r} contains a credential ({', '.join(found)}) at "
            f"{canonical.path}; refusing to show it to any model role. Remove the "
            "credential from the skill file and the next pass will propose normally."
        )
    if 0 < cfg.max_skill_text_chars < len(base_text):
        # The improver rewrites the document whole from the text it is shown,
        # and the prompt clips that text at the cap. Measured before this check:
        # a document past the cap was clipped in the prompt, the improver
        # returned a replacement built from the clipped view, and the candidate
        # dropped every line past the cap, passing the structural checks.
        raise RecordDefect(
            f"skill {group.skill_id!r} is {len(base_text)} chars, above "
            f"[prompt].max_skill_text_chars = {cfg.max_skill_text_chars}: the improver would see "
            f"the first {cfg.max_skill_text_chars} chars and rewrite the document from that view, "
            "dropping the rest. Raise the cap to cover the document, or split the skill; nothing "
            "is proposed against it this run, and neither the improver nor the judge was called "
            "for it.",
            record_id=group.skill_id,
        )
    base_sha = sha256_text(base_text)
    if base_sha != canonical.sha256:
        raise RecordDefect(
            f"skill {group.skill_id!r} changed on disk while this run was in flight "
            f"(snapshot {canonical.sha256[:12]}, now {base_sha[:12]}); deferring the proposal",
            record_id=group.skill_id,
        )

    ordered = sorted(cases, key=lambda c: (-c.weight, c.criterion_id, c.conversation_id))[
        :max_cases
    ]
    user = prompts.improver_user_prompt(
        skill_id=group.skill_id,
        skill_path=str(canonical.path),
        base_text=base_text,
        cases=ordered,
        conversations=conversations,
        cfg=cfg,
        seed=seed,
    )
    payload = runner.run(
        role=ROLE_IMPROVER,
        system=prompts.improver_system_prompt(),
        user=user,
        response_schema="asi.proposal.v1",
    )
    try:
        proposal = parse_proposal(payload)
    except ValueError as exc:
        raise MalformedModelOutput(
            f"improver output violated the proposal contract: {exc}", record_id=group.skill_id
        ) from exc
    if proposal.target_skill_id != group.skill_id:
        raise RecordDefect(
            f"improver targeted {proposal.target_skill_id!r} but was asked about "
            f"{group.skill_id!r}; refusing to redirect a patch to a different skill",
            record_id=group.skill_id,
        )
    new_text = proposal.new_skill_text
    text_problems = skill_text_defects(new_text, base_text=base_text, skill_id=group.skill_id)
    if text_problems:
        raise RecordDefect(
            "proposed replacement text is unsafe: " + "; ".join(text_problems),
            record_id=group.skill_id,
        )
    if 0 < cfg.max_skill_text_chars < len(new_text):
        # The replay scores the candidate arm with this text and the judge
        # reads it as one blind option, both clipped at the same cap. Measured
        # before this check: a 1,120-char replacement under a 500-char cap was
        # scored and judged from its first 500 chars and the gate passed with
        # no note, so the approval token would have covered text no model read.
        raise RecordDefect(
            f"improver returned a {len(new_text)}-char replacement for skill "
            f"{group.skill_id!r}, above [prompt].max_skill_text_chars = "
            f"{cfg.max_skill_text_chars}: the scorer, and the judge when the gate requires one, "
            "would see it clipped at the cap, and a gate pass would certify text no model read "
            "to the end. Nothing is "
            "proposed against it this run and no replay was run; raise the cap to cover a "
            "replacement this size, or expect the improver to add less.",
            record_id=group.skill_id,
        )

    patch = make_patch(base_text, new_text, path_label=str(canonical.path))
    from ..domain.candidate import TargetRef

    target = TargetRef(
        skill_id=group.skill_id,
        root_id=canonical.root_id,
        canonical_path=str(canonical.path),
        base_sha256=base_sha,
        mirror_paths=tuple(str(m.path) for m in group.mirrors),
    )
    notes: list[str] = []
    if canonical.payload_files:
        notes.append(
            "ELEVATED REVIEW RISK: this skill's directory contains "
            f"{len(canonical.payload_files)} executable payload file(s) "
            f"({', '.join(canonical.payload_files[:5])}"
            + (", ..." if len(canonical.payload_files) > 5 else "")
            + "). This candidate replaces the SKILL.md document only, and the write path "
            "refuses any other filename -- but a reviewer should read the change knowing the "
            "surrounding directory is executable code, not documentation."
        )
    if canonical.unlistable_dirs:
        notes.append(
            "ELEVATED REVIEW RISK: "
            f"{len(canonical.unlistable_dirs)} director"
            f"{'y' if len(canonical.unlistable_dirs) == 1 else 'ies'} in this skill's "
            "directory could not be listed ("
            + ", ".join(f"{d} ({e})" for d, e in canonical.unlistable_dirs[:5])
            + (", ..." if len(canonical.unlistable_dirs) > 5 else "")
            + "), so whether executable payloads live there is unknown and the payload "
            "count above is a floor, not an inventory. The write path still refuses every "
            "filename but SKILL.md; a reviewer should know the directory was not fully seen."
        )
    if group.divergent_mirrors:
        notes.append(
            "mirror copies already differ from canonical ("
            + ", ".join(m.root_id for m in group.divergent_mirrors)
            + "): applying this candidate updates the canonical file only, and the mirrors will "
            "still hold their own version until they are synced by their own tooling"
        )
    candidate = Candidate(
        id=candidate_id,
        created_at=created_at,
        hypothesis=proposal.hypothesis,
        intended_change=proposal.intended_change,
        risks=proposal.risks,
        rollback=proposal.rollback,
        target=target,
        rubric_failures=failure_refs_from_cases(ordered),
        sources=sources_from_cases(ordered),
        patch=patch,
        approval_challenge=challenge_for(candidate_id, base_sha, patch.new_text_sha256),
        status=STATUS_PROPOSED,
        improver_command_id=runner.command_id(ROLE_IMPROVER),
        notes=tuple(notes),
    )
    defects = candidate_defects(candidate, base_text=base_text)
    if defects:
        raise RecordDefect(
            "candidate is not reviewable: " + "; ".join(defects), record_id=group.skill_id
        )
    return candidate
