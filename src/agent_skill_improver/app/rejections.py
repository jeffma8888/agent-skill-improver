"""Bounded fingerprint memory of fully measured replacement losses, not gate truth.

No historical candidate is scanned or trusted at lookup. Only a just-written,
verified candidate can teach; a stable whole-tree check must precede publication.
Optional-memory damage costs replay, never stops today's proposal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..config import Config
from ..domain.candidate import STATUS_GATE_FAILED, Candidate, one_line_label, validate_candidate_id
from ..domain.integrity import verify_full_integrity
from ..domain.rubric import Rubric, Verdict
from ..domain.validation import (
    ARM_CANDIDATE,
    CASE_HOLDOUT,
    CASE_TARGET,
    ValidationRecord,
    evaluate_gate,
    runs_the_same_command,
)
from ..errors import ReadOnlyViolation, SafetyRefusal
from ..hashing import sha256_text
from ..jsonio import dumps
from ..ports import ModelRunner
from .receipt import RunReceipt

if TYPE_CHECKING:
    from .state import StateStore

REJECTION_MEMORY_SCHEMA = "asi.rejection-memory.v1"
REJECTION_MEMORY_NAME = "rejected-replacements.json"
MAX_REJECTION_MEMORY_ENTRIES = 64
MAX_REJECTION_MEMORY_BYTES = 131072
MAX_REJECTION_ENTRY_BYTES = 2048
_ENTRY_KEYS = frozenset(
    {
        "skill_id",
        "base_sha256",
        "replacement_sha256",
        "cases_sha256",
        "evaluation_context_sha256",
        "candidate_id",
        "rejected_at",
    }
)
_HASH_KEYS = (
    "base_sha256",
    "replacement_sha256",
    "cases_sha256",
    "evaluation_context_sha256",
)


@dataclass(frozen=True)
class RejectionEntry:
    skill_id: str
    base_sha256: str
    replacement_sha256: str
    cases_sha256: str
    evaluation_context_sha256: str
    candidate_id: str
    rejected_at: str

    @property
    def key(self) -> tuple[str, ...]:
        """One shared five-part identity for insertion and lookup."""
        return (
            self.skill_id,
            self.base_sha256,
            self.replacement_sha256,
            self.cases_sha256,
            self.evaluation_context_sha256,
        )

    def to_dict(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in sorted(_ENTRY_KEYS)}


@dataclass(frozen=True)
class RejectionMemoryRead:
    entries: tuple[RejectionEntry, ...] = ()
    problem: str = ""
    writable: bool = True


def _parse_entry(raw: Any) -> RejectionEntry:
    if not isinstance(raw, dict) or set(raw) != _ENTRY_KEYS:
        raise ValueError("invalid entry")
    if any(not isinstance(value, str) for value in raw.values()):
        raise ValueError("invalid entry")
    skill_id = raw["skill_id"]
    if not skill_id.strip() or len(skill_id.encode("utf-8")) > 256:
        raise ValueError("invalid entry")
    if any(re.fullmatch(r"[0-9a-f]{64}", raw[key]) is None for key in _HASH_KEYS):
        raise ValueError("invalid entry")
    validate_candidate_id(raw["candidate_id"])
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", raw["rejected_at"]) is None:
        raise ValueError("invalid entry")
    datetime.strptime(raw["rejected_at"], "%Y-%m-%dT%H:%M:%SZ")
    if len(dumps(raw).encode("utf-8")) > MAX_REJECTION_ENTRY_BYTES:
        raise ValueError("invalid entry")
    return RejectionEntry(**raw)


def parse_rejection_memory(raw: Any) -> RejectionMemoryRead:
    """Closed parsing. Never salvage selected entries from a damaged file."""
    if not isinstance(raw, dict) or raw.get("schema") != REJECTION_MEMORY_SCHEMA:
        return RejectionMemoryRead(problem="unsupported schema", writable=False)
    if set(raw) != {"schema", "entries"} or not isinstance(raw["entries"], list):
        return RejectionMemoryRead(problem="invalid entry", writable=False)
    if len(raw["entries"]) > MAX_REJECTION_MEMORY_ENTRIES:
        return RejectionMemoryRead(problem="over count limit", writable=False)
    try:
        entries = tuple(_parse_entry(entry) for entry in raw["entries"])
        if len({entry.key for entry in entries}) != len(entries):
            raise ValueError("duplicate identity")
        if len(dumps(raw).encode("utf-8")) > MAX_REJECTION_MEMORY_BYTES:
            return RejectionMemoryRead(problem="over byte limit", writable=False)
    except (ValueError, UnicodeError, OverflowError):
        return RejectionMemoryRead(problem="invalid entry", writable=False)
    return RejectionMemoryRead(entries=entries)


def cases_sha256(candidate: Candidate) -> str:
    return sha256_text(dumps(sorted(ref.failure_case_id for ref in candidate.rubric_failures)))


def evaluation_context_sha256(
    config: Config, rubrics: Mapping[str, Rubric], runner: ModelRunner
) -> str:
    judge = runner.command_id("judge") if config.gate.require_judge else ""
    independent = bool(judge) and not runs_the_same_command(judge, runner.command_id("improver"))
    return sha256_text(
        dumps(
            {
                "contract": "c242-rejected-replacement-v1",
                "rubrics": [
                    {
                        "rubric_id": rubric.id,
                        "version": rubric.version,
                        "source_sha256": rubric.source_sha256,
                    }
                    for rubric in sorted(rubrics.values(), key=lambda r: r.id)
                ],
                "gate": config.gate.to_dict(),
                "prompt": config.prompt.to_dict(),
                "caps": config.caps.to_dict(),
                "commands": {"scorer": runner.command_id("scorer"), "judge": judge},
                "judge_independent": independent,
            }
        )
    )


def qualifying_rejection(record: ValidationRecord) -> bool:
    """Complete decided evidence with a replacement loss, not a coverage/error loss.

    Complete does not mean correct: a decided wrong score or all-tie judge can
    teach. There is no age expiry; clearing the optional file permits a retest.
    """
    independent = bool(record.judge_command_id) and not runs_the_same_command(
        record.judge_command_id, record.improver_command_id
    )
    gate = evaluate_gate(
        record.outcomes,
        record.judgements,
        record.policy,
        judge_independent=independent,
        target_refs=record.target_refs,
    )
    metrics, policy = gate.metrics, record.policy
    if gate.passed:
        return False
    if not (metrics["target_refs_replayed"] == metrics["target_refs_claimed"] > 0):
        return False
    targets = [o for o in record.outcomes if o.case.kind == CASE_TARGET]
    holdouts = [o for o in record.outcomes if o.case.kind == CASE_HOLDOUT]
    if not all(o.baseline == Verdict.FAIL for o in targets):
        return False
    if not (
        all(o.candidate.is_decided for o in targets)
        and all(o.baseline.is_decided and o.candidate.is_decided for o in holdouts)
    ):
        return False
    # c240 remains report-only, but incomplete unseen evidence cannot teach this
    # optional cost policy either (nor can an unseen replay error, through notes).
    if record.unseen is not None and not all(
        o.baseline.is_decided and o.candidate.is_decided for o in record.unseen.outcomes
    ):
        return False
    if policy.require_judge and metrics["judgements"] != metrics["target_cases"]:
        return False
    if record.notes:
        return False
    if metrics["holdout_cases"] < policy.min_holdout_cases:
        return False
    if policy.require_judge and policy.require_independent_judge and not independent:
        return False
    decided = [j for j in record.judgements if j.preferred_arm != "tie"]
    preferred = [j for j in decided if j.preferred_arm == ARM_CANDIDATE]
    return bool(
        metrics["fixes"] < policy.min_fixes
        or metrics["regressions"] > policy.max_regressions
        or (
            policy.require_judge
            and (not decided or len(preferred) / len(decided) < policy.min_judge_preference_ratio)
        )
    )


def rejection_entry_from_candidate(
    candidate: Candidate,
    *,
    base_text: str,
    rubrics: Mapping[str, Rubric],
    evaluation_context: str,
) -> RejectionEntry | None:
    if candidate.status != STATUS_GATE_FAILED or candidate.validation is None:
        return None
    verify_full_integrity(
        candidate, base_text=base_text, rubrics=rubrics, action="remember rejection"
    )
    record = ValidationRecord.parse(candidate.validation)
    if not qualifying_rejection(record):
        return None
    return RejectionEntry(
        skill_id=candidate.target.skill_id,
        base_sha256=candidate.target.base_sha256,
        replacement_sha256=candidate.recomputed_patch_sha256,
        cases_sha256=cases_sha256(candidate),
        evaluation_context_sha256=evaluation_context,
        candidate_id=candidate.id,
        rejected_at=candidate.created_at,
    )


def find_repeated_replacement(
    entries: Sequence[RejectionEntry], candidate: Candidate, *, evaluation_context: str
) -> RejectionEntry | None:
    identity = RejectionEntry(
        candidate.target.skill_id,
        candidate.target.base_sha256,
        candidate.recomputed_patch_sha256,
        cases_sha256(candidate),
        evaluation_context,
        candidate.id,
        candidate.created_at,
    ).key
    return next((entry for entry in entries if entry.key == identity), None)


def remember_rejections(
    existing: Sequence[RejectionEntry], new: Sequence[RejectionEntry]
) -> tuple[tuple[RejectionEntry, ...], int, int]:
    entries = list(existing)
    keys = {entry.key for entry in entries}
    evicted = skipped = 0
    for entry in new:
        if entry.key in keys:
            continue  # insertion-only FIFO: a repeat never renews the original
        try:
            _parse_entry(entry.to_dict())
        except (ValueError, UnicodeError, OverflowError):
            skipped += 1
            continue
        entries.append(entry)
        keys.add(entry.key)
        while (
            len(entries) > MAX_REJECTION_MEMORY_ENTRIES
            or len(
                dumps(
                    {"schema": REJECTION_MEMORY_SCHEMA, "entries": [e.to_dict() for e in entries]}
                ).encode("utf-8")
            )
            > MAX_REJECTION_MEMORY_BYTES
        ):
            keys.remove(entries.pop(0).key)
            evicted += 1
    return tuple(entries), evicted, skipped


def repeat_refusal_reason(entry: RejectionEntry) -> str:
    return (
        f"Replacement repeats fully measured gate-failed candidate {one_line_label(entry.candidate_id)} "
        "on the same skill, base, failure cases and examiner context. Replay and judge were not "
        "called; the improver was called once. This is a repeat refusal, not a new gate result."
    )


class RepeatedReplacement(SafetyRefusal):
    def __init__(self, entry: RejectionEntry) -> None:
        self.of_candidate_id = entry.candidate_id
        self.base_sha256 = entry.base_sha256
        self.replacement_sha256 = entry.replacement_sha256
        self.cases_sha256 = entry.cases_sha256
        self.evaluation_context_sha256 = entry.evaluation_context_sha256
        super().__init__(repeat_refusal_reason(entry))


class RejectionMemoryRun:
    """One bounded snapshot and pending list, used only while daily holds its lock."""

    def __init__(self, snapshot: RejectionMemoryRead, context: str) -> None:
        self.snapshot = snapshot
        self.context = context
        self.pending: list[RejectionEntry] = []

    @classmethod
    def load(
        cls,
        state: StateStore,
        *,
        config: Config,
        rubrics: Mapping[str, Rubric],
        runner: ModelRunner,
        receipt: RunReceipt,
    ) -> RejectionMemoryRun:
        snapshot = state.read_rejection_memory()
        for count in (
            "replacements_refused_as_repeats",
            "rejection_memory_ignored",
            "rejection_memory_write_failed",
        ):
            receipt.counts[count] = 0
        if snapshot.problem:
            receipt.counts["rejection_memory_ignored"] = 1
            receipt.note(
                f"Rejection memory `{REJECTION_MEMORY_NAME}` ignored: {snapshot.problem}. "
                "No entries were used; repeats may be replayed again. Proposal and validation "
                "continue without this memory. The file was left untouched. Inspect it, then "
                "delete this optional file to start empty."
            )
        return cls(snapshot, evaluation_context_sha256(config, rubrics, runner))

    def refuse_if_repeated(self, candidate: Candidate, *, base_text: str) -> None:
        # A raced base is not a match. Let validate's existing verifier defer it;
        # lookup must not turn that validation refusal into a proposal refusal.
        if sha256_text(base_text) != candidate.target.base_sha256:
            return
        entry = find_repeated_replacement(
            self.snapshot.entries, candidate, evaluation_context=self.context
        )
        if entry is not None:
            raise RepeatedReplacement(entry)

    def record_repeat(self, receipt: RunReceipt, skill_id: str, exc: RepeatedReplacement) -> None:
        receipt.counts["replacements_refused_as_repeats"] += 1
        receipt.skills[skill_id]["repeat"] = {
            "of_candidate_id": exc.of_candidate_id,
            "base_sha256": exc.base_sha256,
            "replacement_sha256": exc.replacement_sha256,
            "cases_sha256": exc.cases_sha256,
            "evaluation_context_sha256": exc.evaluation_context_sha256,
            "equality": "lf-normalized-sha256-and-same-cases",
            "validation_ran": False,
            "improver_calls": 1,
            "replay_scorer_calls": 0,
            "judge_calls": 0,
        }

    def stage_candidate(
        self,
        candidate: Candidate,
        *,
        base_text: str,
        rubrics: Mapping[str, Rubric],
        receipt: RunReceipt,
    ) -> None:
        if not self.snapshot.writable:
            return
        try:
            entry = rejection_entry_from_candidate(
                candidate,
                base_text=base_text,
                rubrics=rubrics,
                evaluation_context=self.context,
            )
        except (SafetyRefusal, ValueError):
            receipt.note("Rejection memory entry skipped: candidate integrity was not verified.")
            return
        if entry is not None:
            self.pending.append(entry)

    def publish_if_stable(
        self, state: StateStore, *, receipt: RunReceipt, mutation_detected: bool
    ) -> None:
        if mutation_detected or not self.snapshot.writable or not self.pending:
            self.pending.clear()
            return
        entries, evicted, skipped = remember_rejections(self.snapshot.entries, self.pending)
        if skipped:
            receipt.note(
                "Rejection memory entry skipped: identity exceeds the closed format bounds."
            )
        if entries == self.snapshot.entries:
            return
        # Merge, validation and byte accounting above are deliberately outside
        # this catch. Only optional leaf recheck/publication failures are ignored.
        try:
            state.write_rejection_memory(entries)
        except ReadOnlyViolation:
            raise
        except (OSError, SafetyRefusal):
            receipt.counts["rejection_memory_write_failed"] = 1
            receipt.note(
                "Rejection memory was not updated: write failed. Repeats may be replayed again."
            )
        else:
            if evicted:
                receipt.note(
                    f"Rejection memory evicted {evicted} oldest fingerprint(s) on insertion."
                )
