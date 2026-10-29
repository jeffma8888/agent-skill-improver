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
   this tool's own tree, never reached through a link and never a skill listed
   in ``[guard].protected_skill_ids``, judged before the backup directory is
   made (``review`` asks the same questions before it shows a token);
7. a verified backup must exist before the write, and the write is re-read and
   re-hashed afterwards, restoring the original if it did not land as intended.

Interlock 3 is load-bearing rather than defensive. The token used to be derived
from the *stored* patch hash, so editing ``patch.new_text`` and leaving that hash
alone kept the token valid and put unreviewed bytes into a skill file.
"""

from __future__ import annotations

import os
import shlex
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from pathlib import Path

from ..clock import Clock, iso
from ..domain.approval import HumanApproval, automation_active
from ..domain.candidate import STATUS_APPLIED, STATUS_ROLLED_BACK, STATUS_VALIDATED, Candidate
from ..domain.integrity import verify_full_integrity
from ..domain.redaction import named_path, quoted_value, quoted_values, why_unreadable
from ..domain.validation import ValidationRecord
from ..errors import (
    AsiError,
    LockHeld,
    ApprovalRequired,
    AutomationRefused,
    BaseHashMismatch,
    GateFailed,
    SafetyRefusal,
)
from ..hashing import sha256_text, short
from .lock import RunLock
from .state import StateStore

#: The note an apply that records an earlier run's write adds to the candidate.
FINISHING_NOTE = (
    "recorded by a second run of this apply: an earlier run wrote the skill and did "
    "not record it; "
)

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

    Refuses before it prints anything in four cases, checked in this order so
    that each is named for what it is:

    * a derived field on the candidate itself does not recompute from the
      record alone (``CandidateTampered``);
    * the live canonical text no longer hashes to the base the candidate was
      written against (``BaseHashMismatch``). This is the ordinary case of a
      skill edited since the proposal, and it is reported as that rather than as
      a tampered record, exactly as ``apply`` reports it;
    * a derived field does not recompute against ``base_text`` -- the *live*
      canonical text, never a value the record asserts about itself -- so
      changing only the persisted diff or its line counts cannot make review
      show a token for a diff that does not describe the patch
      (``CandidateTampered``);
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
            f"refusing to review candidate {candidate.id}: status is "
            f"{quoted_value(candidate.status)}, not "
            f"{STATUS_VALIDATED!r}. Only a validated candidate may expose a review token or an "
            "apply command; a closed candidate (applied, rolled-back, gate-failed or blocked-run) "
            "never does, "
            "regardless of what its recorded gate says."
        )
    verify_full_integrity(candidate, rubrics=rubrics, action="review")
    live_sha = sha256_text(base_text)
    if live_sha != candidate.target.base_sha256:
        raise BaseHashMismatch(
            f"refusing to review {candidate.id}: {named_path(candidate.target.canonical_path)} "
            "now hashes "
            f"{short(live_sha)} but the candidate was written against "
            f"{short(candidate.target.base_sha256)}. The skill changed since the proposal was "
            "made, so the diff and the token would describe a file that no longer exists -- "
            "re-run `asi daily --commit` to propose and validate against the new base."
        )
    verify_full_integrity(candidate, base_text=base_text, rubrics=rubrics, action="review")
    gate = (candidate.validation or {}).get("gate") or {}
    # The validation's per-case evidence, read through the same strict parser
    # the integrity check just applied. The gate line says how many cases were
    # fixed or regressed; these say which, and what the blind judge said.
    validation = ValidationRecord.parse(candidate.validation)
    token = candidate.required_approval_token
    return {
        "candidate_id": candidate.id,
        "status": candidate.status,
        "skill_id": candidate.target.skill_id,
        "canonical_path": candidate.target.canonical_path,
        "base_sha256": candidate.target.base_sha256,
        "patch_sha256": candidate.patch.new_text_sha256,
        # No ``mirror_paths`` from the record: neither the token nor the
        # integrity check binds the list it carries, so ``cmd_review`` adds the
        # live inventory's, from the store, once it has judged the target.
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
        "outcomes": [
            {
                "kind": o.case.kind,
                "rubric_id": o.rubric_id,
                "conversation_id": o.case.conversation_id,
                "criterion_id": o.case.criterion_id,
                "baseline": o.baseline.value,
                "candidate": o.candidate.value,
                "change": o.change,
            }
            for o in validation.outcomes
        ],
        "judgements": [
            {
                "conversation_id": j.case.conversation_id,
                "criterion_id": j.case.criterion_id,
                "preferred_arm": j.preferred_arm,
                "rationale": j.rationale,
            }
            for j in validation.judgements
        ],
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
            f"candidate {candidate.id} has status {quoted_value(candidate.status)}; only a "
            f"{STATUS_VALIDATED!r} candidate may be applied"
        )
    # The record is checked against itself before anything in it is believed:
    # a validated candidate whose stored gate did not pass is a record edited
    # after its gate, and is named as tampered (as ``review`` names it) rather
    # than refused with the reasons it carries. Measured before, with the gate
    # read first: a stored reason holding a token and a line break was printed
    # as two lines with the token, ten thousand of them made a refusal of a
    # megabyte, and a reason that was not a string made ``apply`` fall with a
    # traceback.
    verify_full_integrity(candidate, rubrics=rubrics, action="apply")
    gate = (candidate.validation or {}).get("gate") or {}
    if not gate.get("passed"):
        reasons = quoted_values(gate.get("reasons") or []) or "no gate decision recorded"
        raise GateFailed(f"candidate {candidate.id} did not pass the gate: {reasons}")
    # Everything below happens before the backup and before the write. The order
    # is deliberate: the record is checked against itself first, then against the
    # live file, and only then is the operator's token compared -- so a tampered
    # record is named as tampered rather than reported as a token mismatch. Both
    # calls run the *combined* verifier: the candidate's own derived fields and
    # its persisted validation, recomputed from the validation's own outcomes,
    # judgements and policy rather than trusted off the stored gate. The first
    # of the two ran above, before the gate was read.
    # From the live read to the receipt, under the run lock ``daily`` holds: a
    # promotion while a run reads the tree would change what the run is
    # reading, and the run would record that as a skill tree mutation and
    # block the schedule over a write the tool made itself.
    with _holding_the_run_lock(state, "apply", candidate):
        base_text = read_live_base_text(skill_store, candidate)
        live_sha = sha256_text(base_text)
        if live_sha != candidate.target.base_sha256:
            earlier = _earlier_write_of(candidate, live_sha, skill_store, state)
            if earlier is not None:
                # An earlier run of this apply wrote the skill and stopped before
                # it recorded the write. The human gate holds as for a first run;
                # the base the diff is checked against is no longer the live
                # text, and that run checked it before it wrote.
                approval.verify(expected_token=candidate.required_approval_token)
                _refuse_unwritable_receipt(state, "apply", candidate)
                return _record_apply(
                    candidate,
                    earlier,
                    approval,
                    gate,
                    state,
                    clock,
                    operator,
                    mirror_paths=skill_store.mirror_paths_of(Path(candidate.target.canonical_path)),
                    finishing=True,
                )
            raise BaseHashMismatch(
                f"refusing to apply {candidate.id}: "
                f"{named_path(candidate.target.canonical_path)} now hashes "
                f"{short(live_sha)} but the candidate was written against "
                f"{short(candidate.target.base_sha256)}. The skill changed since the proposal "
                "was made -- re-run `asi daily --commit` to validate against the new base."
            )
        verify_full_integrity(candidate, base_text=base_text, rubrics=rubrics, action="apply")
        approval.verify(expected_token=candidate.required_approval_token)
        if getattr(skill_store, "read_only", True):
            # A wiring defect, not a token problem: name it as the residual
            # refusal so the operator is not sent hunting for a new token.
            raise SafetyRefusal(
                "internal error: apply was handed a read-only skill store; refusing to continue"
            )

        # The target is judged as the write judges it (a link on its path, a
        # file that is not the skill document, a protected path, a mirror or
        # unrecognised copy, a skill listed in ``[guard].protected_skill_ids``)
        # before the backup directory is made, so an apply refused for its target
        # leaves nothing in ``backups/``. Measured before: an apply refused for a
        # link on the skill's path was refused by the write, after the backup
        # directory was made, and left that directory empty.
        skill_store.assert_write_target(
            Path(candidate.target.canonical_path),
            action="write",
            skill_id=candidate.target.skill_id,
        )
        # The copies the receipt records as not written are the live
        # inventory's, not the list the record carries, which neither the token
        # nor the integrity check binds.
        mirror_paths = skill_store.mirror_paths_of(Path(candidate.target.canonical_path))
        # The receipt is written last, since it records the write; its path is
        # judged first, before the backup directory and the skill write, since
        # an apply whose receipt cannot be written would still have changed the
        # skill. Measured before: a directory at the receipt path let the apply
        # complete, mark the candidate applied and then exit 70 from the receipt
        # write, with no account of the apply on disk.
        _refuse_unwritable_receipt(state, "apply", candidate)
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
        return _record_apply(
            candidate, receipt, approval, gate, state, clock, operator, mirror_paths=mirror_paths
        )


def _record_apply(
    candidate: Candidate,
    receipt: Any,
    approval: HumanApproval,
    gate: dict[str, Any],
    state: StateStore,
    clock: Clock,
    operator: str,
    *,
    mirror_paths: Sequence[str],
    finishing: bool = False,
) -> dict[str, Any]:
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
            + (FINISHING_NOTE if finishing else "")
            + f"rollback: {receipt.rollback_command}",
        ),
    )
    out = {
        "schema": "asi.apply_receipt.v1",
        "candidate_id": candidate.id,
        "applied_at": iso(clock.now()),
        "operator": operator,
        "approval": approval.to_dict(),
        "write": receipt.to_dict(),
        "gate": gate,
        "mirror_paths_not_written": list(mirror_paths),
    }
    if finishing:
        out["finishes_an_earlier_run"] = True
    # The receipt before the candidate's status: a stop between the two leaves
    # the candidate ``validated`` over a skill that holds its text, which the
    # next apply records. The other order left a candidate marked applied with
    # no receipt, which nothing writes later.
    state.write_action_receipt("apply", candidate.id, out)
    state.write_candidate(applied)
    return out


def _earlier_write_of(
    candidate: Candidate, live_sha: str, skill_store: Any, state: StateStore
) -> Any:
    """The unrecorded write of this candidate the live file holds, else ``None``.

    It is one only when the live text is the candidate's own new text and the
    newest backup manifest of this candidate records that write, from the
    candidate's base to that text, for the same document.
    """
    if live_sha != sha256_text(candidate.patch.new_text):
        return None
    manifest = state.latest_backup_manifest(candidate.id)
    if manifest is None:
        return None
    return skill_store.earlier_write(
        manifest,
        path=Path(candidate.target.canonical_path),
        previous_sha256=candidate.target.base_sha256,
        new_sha256=live_sha,
    )


@contextmanager
def _holding_the_run_lock(state: StateStore, action: str, candidate: Candidate) -> Iterator[None]:
    """Hold the run lock for the part of a promotion that reads or writes the tree.

    The same lock ``daily`` takes, so a run and a promotion never overlap: the
    owner line names the promotion, so a run refused meanwhile says who holds
    it. Measured before: an apply during a scheduled run went through, the run
    finished ``blocked`` with the applied document in ``skill_tree.changes``,
    its candidates were closed as blocked-run, and the next morning's run was
    refused until the operator reviewed a mutation the tool had made itself.
    """
    lock = RunLock(state.lock_path(), owner=f"{action} {candidate.id} pid={os.getpid()}")
    try:
        lock.__enter__()
    except LockHeld as exc:
        raise LockHeld(
            f"refusing to {action} {candidate.id}: {exc} A promotion while a daily run reads "
            "the tree would change what the run is reading, and the run would record that as a "
            "skill tree mutation and block the schedule. Wait for the holder named above to "
            "finish (a daily run, or another apply or rollback); nothing was written."
        ) from exc
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def _refuse_unwritable_receipt(state: StateStore, action: str, candidate: Candidate) -> None:
    """Refuse before the skill write when the action's receipt could not be written after it."""
    problem = state.artifact_path_problem(
        state.action_receipt_path(action, candidate.id), f"the {action} receipt"
    )
    if problem:
        raise SafetyRefusal(
            f"refusing to {action} {candidate.id}: {problem}, where the receipt of this "
            f"{action} would be written; nothing was written. Remove it if you did not put "
            "it there"
        )


def read_live_base_text(skill_store: Any, candidate: Candidate, *, action: str = "apply") -> str:
    """Read the canonical file the candidate claims to patch.

    A read failure here is a base mismatch, not an internal error: the candidate
    describes a file this store cannot read, so nothing about the recorded diff
    can be checked and the write must not proceed. ``action`` names the command
    refusing, so ``review`` does not report itself as ``apply``.
    """
    path = Path(candidate.target.canonical_path)
    if "\x00" in str(path):
        # No file name holds a NUL, and a read of one falls with a ``ValueError``
        # rather than an ``OSError``: measured before, ``review`` and ``apply`` of
        # a record whose path held one exited 70 with a traceback.
        raise BaseHashMismatch(
            f"refusing to {action} {candidate.id}: its canonical target {named_path(path)} "
            "holds a NUL character, which no file name can, so it cannot be read and the "
            "recorded base hash and diff cannot be verified; re-run `asi daily --commit` to "
            "propose against the configured tree."
        )
    try:
        return str(skill_store.read_text(path))
    except OSError as exc:
        raise BaseHashMismatch(
            f"refusing to {action} {candidate.id}: its canonical target {named_path(path)} could "
            f"not be read ({why_unreadable(exc)}), so the recorded base hash and diff cannot be "
            "verified. If the skill root "
            "was moved or re-pointed, the candidate describes a file at its old path; re-run "
            "`asi daily --commit` to propose against the configured tree."
        ) from exc


def rollback_candidate(
    candidate: Candidate,
    *,
    skill_store: Any,
    state: StateStore,
    clock: Clock,
) -> dict[str, Any]:
    """Restore the file this candidate overwrote, from its verified backup. Human-gated."""
    if automation_active():
        raise AutomationRefused("rollback refused: automation marker is set. " + APPLY_ENV_HINT)
    # The manifest is chosen under the lock too: chosen before it, a rollback
    # waiting on an apply would restore from the backup that apply then made
    # obsolete.
    with _holding_the_run_lock(state, "rollback", candidate):
        manifest = state.latest_backup_manifest(candidate.id)
        if manifest is None:
            raise GateFailed(
                f"no backup manifest found for {candidate.id}; nothing to roll back. "
                "A candidate that was never applied has nothing to restore."
            )
        if candidate.status == STATUS_VALIDATED and _still_at_its_base(skill_store, candidate):
            # The manifest is written ahead of the skill, so an apply stopped
            # before its skill write leaves one for a write that never landed.
            # Restoring "from" it would close a candidate never applied as
            # rolled back, and the apply that would finish it is then refused.
            raise GateFailed(
                f"{candidate.id} was never applied: its skill still holds the text it was "
                "proposed against, so there is nothing to roll back. An apply of it stopped "
                "before its skill write leaves this; run that apply again to apply it."
            )
        _refuse_unwritable_receipt(state, "rollback", candidate)
        # The manifest must record this candidate's own write: its document,
        # from its base to its replacement. Measured before: a manifest edited
        # to name another skill, a planted backup or a later edit's hash was
        # restored from, and the candidate recorded rolled back.
        result = skill_store.rollback(
            manifest,
            path=Path(candidate.target.canonical_path),
            previous_sha256=candidate.target.base_sha256,
            new_sha256=sha256_text(candidate.patch.new_text),
        )
        return _record_rollback(candidate, manifest, result, state, clock)


def _still_at_its_base(skill_store: Any, candidate: Candidate) -> bool:
    """Whether the candidate's document still holds the text it was proposed against.

    A document this cannot read is not judged here; the rollback's own checks
    name what is wrong with it.
    """
    try:
        text = read_live_base_text(skill_store, candidate, action="roll back")
    except AsiError:
        return False
    return sha256_text(text) == candidate.target.base_sha256


def _record_rollback(
    candidate: Candidate, manifest: Path, result: Any, state: StateStore, clock: Clock
) -> dict[str, Any]:
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
