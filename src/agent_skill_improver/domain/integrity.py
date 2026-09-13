"""Recomputation of every candidate field that claims integrity.

A candidate record is a plain JSON file under the state tree, so every field in
it is *asserted* rather than proven. Some of those fields are derived: the patch
hash is a function of the patch text, the approval challenge is a function of the
candidate id and the two hashes, the diff and its counts are a function of the
base text and the patch text, and the status is a function of the recorded gate.

Storing a derived value next to its input creates a way for the two to disagree,
and the disagreement is silent unless somebody recomputes. That is not
hypothetical: the approval token is derived from the *stored* patch hash, so
editing ``patch.new_text`` while leaving ``patch.new_text_sha256`` alone left the
token matching and the reviewed bytes replaced. The token then authorised text no
human had seen, which is the one property the whole human gate exists to provide.

So every derived field is recomputed from its inputs before a candidate is
reviewed, applied or validated, and any disagreement is a refusal rather than a
warning. This module is pure: it takes the candidate and, where the caller can
supply it, the live base text, and returns reasons. The refusal wrapper is here
too so every call site reports the same thing in the same words.
"""

from __future__ import annotations

from pathlib import PurePath

from ..errors import CandidateTampered
from ..hashing import sha256_text, short
from .candidate import (
    STATUS_APPLIED,
    STATUS_GATE_FAILED,
    STATUS_PROPOSED,
    STATUS_ROLLED_BACK,
    STATUS_VALIDATED,
    Candidate,
    TargetRef,
    challenge_for,
    make_patch,
)
from .skills import SKILL_FILENAME

#: Statuses that may only exist once validation ran and recorded a gate.
_STATUSES_REQUIRING_VALIDATION = (
    STATUS_VALIDATED,
    STATUS_GATE_FAILED,
    STATUS_APPLIED,
    STATUS_ROLLED_BACK,
)

#: Statuses that assert the gate passed.
_STATUSES_ASSERTING_A_PASS = (STATUS_VALIDATED, STATUS_APPLIED, STATUS_ROLLED_BACK)

_SHA256_LEN = 64


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LEN and all(c in "0123456789abcdef" for c in value)


def target_identity_problems(target: TargetRef) -> list[str]:
    """Every way the target no longer identifies one canonical skill document.

    The write path resolves and re-checks the path it is handed, but it is handed
    ``canonical_path`` and told ``skill_id`` separately: a record whose skill id
    and path disagree describes two different skills, and the receipt, the backup
    manifest and the reviewer's memory would each name a different one.
    """
    problems: list[str] = []
    if not target.skill_id.strip():
        problems.append("target.skill_id is empty")
    if not target.root_id.strip():
        problems.append("target.root_id is empty")
    if not target.canonical_path.strip():
        problems.append("target.canonical_path is empty")
        return problems
    path = PurePath(target.canonical_path)
    if path.name != SKILL_FILENAME:
        problems.append(
            f"target.canonical_path names {path.name!r}, but only {SKILL_FILENAME} is ever written"
        )
    parent = path.parent.name
    if target.skill_id.strip() and parent and parent != target.skill_id:
        problems.append(
            f"target.skill_id is {target.skill_id!r} but target.canonical_path sits in "
            f"{parent!r}: the record names two different skills"
        )
    if not _is_sha256(target.base_sha256):
        problems.append("target.base_sha256 is not a 64-character lowercase sha256 digest")
    if target.canonical_path in target.mirror_paths:
        problems.append(
            "target.canonical_path is also listed as a mirror path, so the record cannot say "
            "which copy is canonical"
        )
    return problems


def gate_status_problems(candidate: Candidate) -> list[str]:
    """Disagreements between the recorded status and the recorded gate decision."""
    problems: list[str] = []
    validation = candidate.validation
    gate = (validation or {}).get("gate") or {}
    status = candidate.status
    if status in _STATUSES_REQUIRING_VALIDATION and not gate:
        problems.append(
            f"status is {status!r} but no gate decision is recorded, so the status is unsupported"
        )
        return problems
    if status in _STATUSES_ASSERTING_A_PASS and not gate.get("passed"):
        problems.append(f"status is {status!r} but the recorded gate did not pass")
    if status == STATUS_GATE_FAILED and gate.get("passed"):
        problems.append("status is 'gate-failed' but the recorded gate passed")
    if status == STATUS_PROPOSED and gate:
        problems.append("status is 'proposed' but a gate decision is already recorded")
    return problems


def derived_field_mismatches(candidate: Candidate, *, base_text: str | None = None) -> list[str]:
    """Every derived field that does not equal a fresh recomputation of itself.

    ``base_text`` is the *live* text of the canonical file. Pass it wherever the
    caller can read it: the diff, its counts and the recorded base hash can only
    be checked against the base they claim to describe. Without it the hash of the
    patch text, the approval challenge, the target identity and the status/gate
    agreement are still checked, because none of those need the base.
    """
    problems: list[str] = []
    patch = candidate.patch
    recomputed = sha256_text(patch.new_text)
    if patch.new_text_sha256 != recomputed:
        problems.append(
            f"patch.new_text_sha256 records {short(patch.new_text_sha256)} but new_text hashes "
            f"{short(recomputed)}: the reviewed text and its recorded hash disagree"
        )
    expected_challenge = challenge_for(candidate.id, candidate.target.base_sha256, recomputed)
    if candidate.approval_challenge != expected_challenge:
        problems.append(
            "approval_challenge is not the challenge derived from this candidate's id, its base "
            "hash and the hash of its actual new_text"
        )
    problems.extend(target_identity_problems(candidate.target))
    problems.extend(gate_status_problems(candidate))
    if base_text is None:
        return problems
    live = sha256_text(base_text)
    if candidate.target.base_sha256 != live:
        problems.append(
            f"target.base_sha256 records {short(candidate.target.base_sha256)} but the live base "
            f"text hashes {short(live)}"
        )
        return problems
    expected_patch = make_patch(
        base_text, patch.new_text, path_label=candidate.target.canonical_path
    )
    if patch.unified_diff != expected_patch.unified_diff:
        problems.append(
            "patch.unified_diff is not the diff between the live base text and new_text: the "
            "diff a human reviewed does not describe this change"
        )
    if patch.added_lines != expected_patch.added_lines:
        problems.append(
            f"patch.added_lines records {patch.added_lines} but the recomputed diff adds "
            f"{expected_patch.added_lines}"
        )
    if patch.removed_lines != expected_patch.removed_lines:
        problems.append(
            f"patch.removed_lines records {patch.removed_lines} but the recomputed diff removes "
            f"{expected_patch.removed_lines}"
        )
    return problems


def verify_candidate_integrity(
    candidate: Candidate, *, base_text: str | None = None, action: str
) -> None:
    """Refuse ``action`` unless every derived field recomputes to what is stored."""
    problems = derived_field_mismatches(candidate, base_text=base_text)
    if not problems:
        return
    raise CandidateTampered(
        f"refusing to {action} candidate {candidate.id}: {len(problems)} derived field(s) do not "
        "match a fresh recomputation, so the record has changed since it was written.\n"
        + "\n".join(f"  - {p}" for p in problems)
        + "\n  An approval token is derived from the candidate's content, so a record edited "
        "after review must never be authorised by the token printed before it. Delete this "
        "candidate and re-run `asi daily --commit` to propose and validate the change again; "
        "nothing has been written to any skill file."
    )
