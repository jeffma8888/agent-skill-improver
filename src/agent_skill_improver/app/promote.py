"""Promotion: the only path that writes a live skill file, and it needs a human.

Deliberately isolated in its own module so that the structural guarantee is
checkable: :mod:`agent_skill_improver.app.daily` does not import this module, and
a test asserts that by reading the source. Every other interlock here is a belt
on top of that.

The interlocks, in the order they fire:

1. the automation marker must be absent (the daily driver always sets it);
2. the candidate must have passed validation, with the gate decision recorded;
3. every derived field, and the persisted validation itself, must recompute to
   the value the record stores --
   :func:`~agent_skill_improver.domain.integrity.verify_full_integrity`, run
   before any backup or write, because the record is a plain file and a field
   derived from another field (or a validation derived from its own outcomes)
   can be edited to disagree with what it is derived from;
4. the on-disk hash must equal the base hash the candidate was written against,
   and the diff and its counts must recompute against that same live base;
5. the approval token must match, and it is derived from the candidate id, the
   base hash and the hash of the patch text *as it stands* -- so it stops matching
   if any of them changed;
6. the target must be the canonical editable copy, never a mirror, never inside
   this tool's own tree;
7. a verified backup must exist before the write, and the write is re-read and
   re-hashed afterwards, restoring the original if it did not land as intended.

Interlock 3 is load-bearing rather than defensive. The token used to be derived
from the *stored* patch hash, so editing ``patch.new_text`` and leaving that hash
alone kept the token valid and put unreviewed bytes into a skill file.
"""

from __future__ import annotations

import shlex
from typing import Any

from pathlib import Path

from ..clock import Clock, iso
from ..domain.approval import HumanApproval, automation_active
from ..domain.candidate import STATUS_APPLIED, STATUS_ROLLED_BACK, STATUS_VALIDATED, Candidate
from ..domain.integrity import verify_full_integrity
from ..errors import ApprovalRequired, AutomationRefused, BaseHashMismatch, GateFailed
from ..hashing import sha256_text, short
from .state import StateStore

APPLY_ENV_HINT = (
    "Promotion is a human action. Run it from an interactive shell: "
    "asi review <candidate-id> then asi apply <candidate-id> --approve <token> --i-reviewed-this"
)


def review_report(
    candidate: Candidate,
    *,
    base_text: str,
    rubrics: Any = None,
) -> dict[str, Any]:
    """Everything a human needs in order to decide, plus the exact apply command.

    Refuses before it prints anything in three cases:

    * a derived field on the candidate itself does not recompute (the diff or
      its counts, checked against ``base_text`` -- the *live* canonical text,
      never a value the record asserts about itself -- so changing only the
      persisted diff or its line counts cannot make review show a token for a
      diff that does not describe the patch);
    * the persisted validation does not recompute byte-exact from its own
      outcomes, judgements and policy (see
      :func:`~agent_skill_improver.domain.integrity.validation_mismatches`);
    * the status is anything other than exactly ``validated``. ``applied``,
      ``rolled-back``, ``gate-failed`` and ``blocked-run`` are all closed: none of them may ever
      expose a token or an apply command again, no matter what the recorded
      gate says.
    """
    if candidate.status != STATUS_VALIDATED:
        raise GateFailed(
            f"refusing to review candidate {candidate.id}: status is {candidate.status!r}, not "
            f"{STATUS_VALIDATED!r}. Only a validated candidate may expose a review token or an "
            "apply command; a closed candidate (applied, rolled-back or gate-failed) never does, "
            "regardless of what its recorded gate says."
        )
    verify_full_integrity(candidate, base_text=base_text, rubrics=rubrics, action="review")
    gate = (candidate.validation or {}).get("gate") or {}
    token = candidate.required_approval_token
    return {
        "candidate_id": candidate.id,
        "status": candidate.status,
        "skill_id": candidate.target.skill_id,
        "canonical_path": candidate.target.canonical_path,
        "base_sha256": candidate.target.base_sha256,
        "patch_sha256": candidate.patch.new_text_sha256,
        "mirror_paths": list(candidate.target.mirror_paths),
        "hypothesis": candidate.hypothesis,
        "intended_change": candidate.intended_change,
        "risks": list(candidate.risks),
        "rollback": candidate.rollback,
        "rubric_failures": [r.to_dict() for r in candidate.rubric_failures],
        "sources": [s.to_dict() for s in candidate.sources],
        "diff": candidate.patch.unified_diff,
        "added_lines": candidate.patch.added_lines,
        "removed_lines": candidate.patch.removed_lines,
        "gate": gate,
        "notes": list(candidate.notes),
        "approval_token": token,
        "apply_command": shlex.join(
            ["asi", "apply", candidate.id, "--approve", token, "--i-reviewed-this"]
        ),
    }


def apply_candidate(
    candidate: Candidate,
    *,
    approval: HumanApproval,
    skill_store: Any,
    state: StateStore,
    clock: Clock,
    operator: str = "",
    rubrics: Any = None,
) -> dict[str, Any]:
    """Write the candidate's text to the canonical skill file. Human-gated."""
    if automation_active():
        raise AutomationRefused("promotion refused: automation marker is set. " + APPLY_ENV_HINT)
    if candidate.status != STATUS_VALIDATED:
        raise GateFailed(
            f"candidate {candidate.id} has status {candidate.status!r}; only a "
            f"{STATUS_VALIDATED!r} candidate may be applied"
        )
    gate = (candidate.validation or {}).get("gate") or {}
    if not gate.get("passed"):
        reasons = "; ".join(gate.get("reasons", [])) or "no gate decision recorded"
        raise GateFailed(f"candidate {candidate.id} did not pass the gate: {reasons}")
    # Everything below happens before the backup and before the write. The order
    # is deliberate: the record is checked against itself first, then against the
    # live file, and only then is the operator's token compared -- so a tampered
    # record is named as tampered rather than reported as a token mismatch. Both
    # calls run the *combined* verifier: the candidate's own derived fields and
    # its persisted validation, recomputed from the validation's own outcomes,
    # judgements and policy rather than trusted off the stored gate.
    verify_full_integrity(candidate, rubrics=rubrics, action="apply")
    base_text = read_live_base_text(skill_store, candidate)
    live_sha = sha256_text(base_text)
    if live_sha != candidate.target.base_sha256:
        raise BaseHashMismatch(
            f"refusing to apply {candidate.id}: {candidate.target.canonical_path} now hashes "
            f"{short(live_sha)} but the candidate was written against "
            f"{short(candidate.target.base_sha256)}. The skill changed since the proposal was "
            "made -- re-run `asi daily --commit` to validate against the new base."
        )
    verify_full_integrity(candidate, base_text=base_text, rubrics=rubrics, action="apply")
    approval.verify(expected_token=candidate.required_approval_token)
    if getattr(skill_store, "read_only", True):
        raise ApprovalRequired(
            "internal error: apply was handed a read-only skill store; refusing to continue"
        )

    stamp = iso(clock.now()).replace(":", "").replace("-", "")
    backup_dir = state.backup_dir_for(candidate.id, stamp)
    receipt = skill_store.write_canonical(
        candidate.target.canonical_path,
        candidate.patch.new_text,
        expected_sha256=candidate.target.base_sha256,
        backup_dir=backup_dir,
        skill_id=candidate.target.skill_id,
        actor=f"human:{operator or 'unspecified'}",
    )
    applied = Candidate(
        id=candidate.id,
        created_at=candidate.created_at,
        hypothesis=candidate.hypothesis,
        intended_change=candidate.intended_change,
        risks=candidate.risks,
        rollback=candidate.rollback,
        target=candidate.target,
        rubric_failures=candidate.rubric_failures,
        sources=candidate.sources,
        patch=candidate.patch,
        approval_challenge=candidate.approval_challenge,
        status=STATUS_APPLIED,
        improver_command_id=candidate.improver_command_id,
        validation=candidate.validation,
        notes=candidate.notes
        + (
            f"applied {iso(clock.now())} by human:{operator or 'unspecified'}; "
            f"rollback: {receipt.rollback_command}",
        ),
    )
    state.write_candidate(applied)
    out = {
        "schema": "asi.apply_receipt.v1",
        "candidate_id": candidate.id,
        "applied_at": iso(clock.now()),
        "operator": operator,
        "approval": approval.to_dict(),
        "write": receipt.to_dict(),
        "gate": gate,
        "mirror_paths_not_written": list(candidate.target.mirror_paths),
    }
    state.write_action_receipt("apply", candidate.id, out)
    return out


def read_live_base_text(skill_store: Any, candidate: Candidate) -> str:
    """Read the canonical file the candidate claims to patch.

    A read failure here is a base mismatch, not an internal error: the candidate
    describes a file this store cannot read, so nothing about the recorded diff
    can be checked and the write must not proceed.
    """
    path = Path(candidate.target.canonical_path)
    try:
        return str(skill_store.read_text(path))
    except OSError as exc:
        raise BaseHashMismatch(
            f"refusing to apply {candidate.id}: its canonical target {path} could not be read "
            f"({exc}), so the recorded base hash and diff cannot be verified."
        ) from exc


def rollback_candidate(
    candidate: Candidate,
    *,
    skill_store: Any,
    state: StateStore,
    clock: Clock,
) -> dict[str, Any]:
    """Restore the file this candidate overwrote, from its verified backup."""
    manifest = state.latest_backup_manifest(candidate.id)
    if manifest is None:
        raise GateFailed(
            f"no backup manifest found for {candidate.id}; nothing to roll back. "
            "A candidate that was never applied has nothing to restore."
        )
    result = skill_store.rollback(manifest)
    rolled = Candidate(
        id=candidate.id,
        created_at=candidate.created_at,
        hypothesis=candidate.hypothesis,
        intended_change=candidate.intended_change,
        risks=candidate.risks,
        rollback=candidate.rollback,
        target=candidate.target,
        rubric_failures=candidate.rubric_failures,
        sources=candidate.sources,
        patch=candidate.patch,
        approval_challenge=candidate.approval_challenge,
        status=STATUS_ROLLED_BACK,
        improver_command_id=candidate.improver_command_id,
        validation=candidate.validation,
        notes=candidate.notes + (f"rolled back {iso(clock.now())} from {manifest}",),
    )
    state.write_candidate(rolled)
    out = {
        "schema": "asi.rollback_receipt.v1",
        "candidate_id": candidate.id,
        "rolled_back_at": iso(clock.now()),
        "manifest": str(manifest),
        "result": result,
    }
    state.write_action_receipt("rollback", candidate.id, out)
    return out
