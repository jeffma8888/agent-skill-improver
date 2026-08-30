"""Promotion: the only path that writes a live skill file, and it needs a human.

Deliberately isolated in its own module so that the structural guarantee is
checkable: :mod:`agent_skill_improver.app.daily` does not import this module, and
a test asserts that by reading the source. Every other interlock here is a belt
on top of that.

The interlocks, in the order they fire:

1. the automation marker must be absent (the daily driver always sets it);
2. the candidate must have passed validation, with the gate decision recorded;
3. the approval token must match, and it is derived from the candidate id, the
   base hash and the patch hash -- so it stops matching if any of them changed;
4. the target must be the canonical editable copy, never a mirror, never inside
   this tool's own tree;
5. the on-disk hash must equal the base hash the candidate was written against;
6. a verified backup must exist before the write, and the write is re-read and
   re-hashed afterwards, restoring the original if it did not land as intended.
"""

from __future__ import annotations

from typing import Any

from ..clock import Clock, iso
from ..domain.approval import HumanApproval, automation_active
from ..domain.candidate import STATUS_APPLIED, STATUS_ROLLED_BACK, STATUS_VALIDATED, Candidate
from ..errors import ApprovalRequired, AutomationRefused, GateFailed
from ..jsonio import write_json
from .state import StateStore

APPLY_ENV_HINT = (
    "Promotion is a human action. Run it from an interactive shell: "
    "asi review <candidate-id> then asi apply <candidate-id> --approve <token> --i-reviewed-this"
)


def review_report(candidate: Candidate) -> dict[str, Any]:
    """Everything a human needs in order to decide, plus the exact apply command."""
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
        "apply_command": (f"asi apply {candidate.id} --approve {token} --i-reviewed-this"),
    }


def apply_candidate(
    candidate: Candidate,
    *,
    approval: HumanApproval,
    skill_store: Any,
    state: StateStore,
    clock: Clock,
    operator: str = "",
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
    write_json(state.dir("receipts") / f"apply--{candidate.id}.json", out)
    return out


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
    write_json(state.dir("receipts") / f"rollback--{candidate.id}.json", out)
    return out
