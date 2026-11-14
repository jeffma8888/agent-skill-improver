"""Promotion: the skill writer, intended for a person after review.

Deliberately isolated in its own module so that the structural guarantee is
checkable: :mod:`agent_skill_improver.app.daily` does not import this module, and
a test asserts that by reading the source. Every other interlock here is a belt
on top of that. Apply does not authenticate its caller: a separate unmarked
process can still apply with a token and acknowledgement. Do not schedule apply.

The interlocks, in the order they fire:

1. the automation marker must be inactive (daily sets it only in-process;
   scheduler templates set it for their whole job command);
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
from dataclasses import replace
from typing import Any, Iterator, Sequence

from pathlib import Path

from ..clock import Clock, iso
from ..domain.approval import HumanApproval, automation_active
from ..domain.candidate import (
    MAX_DECLINE_OPERATOR_CHARS,
    MAX_DECLINE_REASON_CHARS,
    STATUS_APPLIED,
    STATUS_DECLINED,
    STATUS_PROPOSED,
    STATUS_ROLLED_BACK,
    STATUS_VALIDATED,
    Candidate,
    Decline,
    decline_reason_problem,
    is_repair_input,
    one_line_label,
    settled_mask,
)
from ..domain.integrity import repropose_advice, verify_full_integrity
from ..domain.redaction import named_path, quoted_value, quoted_values, why_unreadable
from .. import SCHEMA_VALIDATION_V3
from ..domain.validation import UNSEEN_SCOPE, ReplayOutcome, ValidationRecord
from ..errors import (
    AsiError,
    LockHeld,
    ApprovalRequired,
    AutomationRefused,
    BaseHashMismatch,
    CandidateTampered,
    GateFailed,
    LinkedTargetRefused,
    SafetyRefusal,
    StateUnreadable,
    UndecodableDocument,
)
from ..hashing import sha256_text, short
from .lock import RunLock
from .state import CLOSED_STATUSES, StateStore

#: The note an apply that records an earlier run's write adds to the candidate.
FINISHING_NOTE = (
    "recorded by a second run of this apply: an earlier run wrote the skill and did "
    "not record it; "
)

APPLY_ENV_HINT = (
    "After review, run apply from an unmarked interactive shell; do not schedule apply. "
    "This marker is an interlock, not caller authentication."
)
ROLLBACK_ENV_HINT = (
    "Run rollback from an unmarked interactive shell; do not schedule rollback. "
    "This marker is an interlock, not caller authentication."
)
DECLINE_ENV_HINT = (
    "Run decline from an unmarked interactive shell; do not schedule decline. "
    "This marker is an interlock, not caller authentication."
)


def refuse_a_closed_review(candidate: Candidate) -> None:
    """Refuse to review a candidate whose status is anything but exactly ``validated``.

    ``review`` asks this before it reads the candidate's file, as ``apply`` asks
    its status first. Measured before, with the file read first: an ``applied``
    or ``rolled-back`` candidate whose skill folder had moved was refused as a
    base mismatch rather than as closed, told to propose again; with the record
    advice of ``repropose_advice`` that refusal would say to delete the record,
    and an applied candidate's record is what ``rollback`` restores from.
    ``review_report`` asks it too, for a caller that reads first.
    """
    if candidate.status != STATUS_VALIDATED:
        raise GateFailed(
            f"refusing to review candidate {candidate.id}: status is "
            f"{quoted_value(candidate.status)}, not "
            f"{STATUS_VALIDATED!r}. Only a validated candidate may expose a review token or an "
            "apply command; a closed candidate (applied, rolled-back, gate-failed, blocked-run "
            "or declined) never does, "
            "regardless of what its recorded gate says."
        )


def _outcome_row(outcome: ReplayOutcome) -> dict[str, Any]:
    return {
        "kind": outcome.case.kind,
        "rubric_id": outcome.rubric_id,
        "conversation_id": outcome.case.conversation_id,
        "criterion_id": outcome.case.criterion_id,
        "baseline": outcome.baseline.value,
        "candidate": outcome.candidate.value,
        "change": outcome.change,
    }


def _unseen_report(validation: ValidationRecord) -> dict[str, Any]:
    """The record's unseen block for review, or what a v3 record cannot say."""
    unseen = validation.unseen
    if unseen is None:
        return {"status": "not-recorded", "schema": SCHEMA_VALIDATION_V3}
    return {
        "status": unseen.status,
        "min_failures": unseen.min_failures,
        "withhold": unseen.withhold,
        "eligible": [r.to_dict() for r in unseen.eligible],
        "withheld": [r.to_dict() for r in unseen.withheld],
        "replayed": unseen.replayed,
        "fixed": unseen.fixed,
        "outcomes": [_outcome_row(o) for o in unseen.outcomes],
        "scope": UNSEEN_SCOPE,
    }


def review_report(
    candidate: Candidate,
    *,
    base_text: str,
    rubrics: Any = None,
    unseen_withhold: int = 0,
) -> dict[str, Any]:
    """Everything a human needs in order to decide, plus the exact apply command.

    ``unseen_withhold`` is the configured ``[caps].unseen_withhold``. It decides
    only whether a v3 record's report carries ``unseen``: with withholding off
    (the default) a v3 report has no such key, so it is byte for byte what it
    was before the unseen block existed; with it on, ``unseen`` says the record
    cannot tell (``not-recorded``). A v4 record's ``unseen`` is always reported.

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
    refuse_a_closed_review(candidate)
    verify_full_integrity(candidate, rubrics=rubrics, action="review")
    live_sha = sha256_text(base_text)
    if live_sha != candidate.target.base_sha256:
        raise BaseHashMismatch(
            f"refusing to review {candidate.id}: {named_path(candidate.target.canonical_path)} "
            "now hashes "
            f"{short(live_sha)} but the candidate was written against "
            f"{short(candidate.target.base_sha256)}. The skill changed since the proposal was "
            "made, so the diff and the token would describe a file that no longer exists. To "
            f"propose and validate against the new base, {repropose_advice(candidate.id)}."
        )
    verify_full_integrity(candidate, base_text=base_text, rubrics=rubrics, action="review")
    gate = (candidate.validation or {}).get("gate") or {}
    # The validation's per-case evidence, read through the same strict parser
    # the integrity check just applied. The gate line says how many cases were
    # fixed or regressed; these say which, and what the blind judge said.
    validation = ValidationRecord.parse(candidate.validation)
    token = candidate.required_approval_token
    report: dict[str, Any] = {
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
        # The gate's cases only. The unseen cases are in ``unseen``, never here.
        "outcomes": [_outcome_row(o) for o in validation.outcomes],
        "judgements": [
            {
                "conversation_id": j.case.conversation_id,
                "criterion_id": j.case.criterion_id,
                "preferred_arm": j.preferred_arm,
                "rationale": j.rationale,
            }
            for j in validation.judgements
        ],
    }
    # Report-only: the gate above never read it. Left out for a v3 record
    # while withholding is off, so the default report is the one before.
    if validation.unseen is not None or unseen_withhold > 0:
        report["unseen"] = _unseen_report(validation)
    report["notes"] = list(candidate.notes)
    report["approval_token"] = token
    report["apply_command"] = shlex.join(
        ["asi", "apply", candidate.id, "--approve", token, "--i-reviewed-this"]
    )
    return report


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
    """Write the candidate's text to the canonical skill file. Review-gated."""
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
    # From the record's re-read to the receipt, under the run lock ``daily``
    # holds: a promotion while a run reads the tree would change what the run
    # is reading, and the run would record that as a skill tree mutation and
    # block the schedule over a write the tool made itself.
    with _holding_the_run_lock(state, "apply", candidate):
        # The record as it is now, read once the lock is held: the object this
        # apply was handed was read before it waited for the lock, and another
        # apply or a rollback of the same candidate may have finished since.
        # Measured before (3.11 and 3.13), acting on the handed read: an apply
        # holding a read made before another apply finished recorded that
        # write again as its own and replaced that apply's receipt, and one
        # holding a read made before a rollback wrote the skill again over the
        # restored text. Only the status is compared: the handed object is
        # what the integrity check and the token were checked against, and the
        # write and the record still come from it. A record deleted meanwhile
        # raises ``FileNotFoundError`` here and one made unreadable
        # ``StateUnreadable``, before the skill is read and before anything is
        # written. An apply stopped after its skill write but before the
        # candidate is recorded ``applied`` leaves it ``validated``, so the
        # rerun that finishes it passes here; one stopped after that has
        # nothing left to finish, and its rerun is refused as ``applied``.
        current = state.read_candidate(candidate.id)
        if current.status != STATUS_VALIDATED:
            raise GateFailed(
                f"refusing to apply {candidate.id}: its record now says "
                f"{quoted_value(current.status)}, not {STATUS_VALIDATED!r}: it changed after "
                "this apply read it. Nothing was written."
            )
        base_text = read_live_base_text(skill_store, candidate)
        live_sha = sha256_text(base_text)
        if live_sha != candidate.target.base_sha256:
            earlier = _earlier_write_of(candidate, live_sha, skill_store, state)
            if earlier is not None:
                # An earlier run of this apply wrote the skill and stopped before
                # it recorded the write. The review interlocks hold as for a first run;
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
                "was made. To propose and validate against the new base, "
                f"{repropose_advice(candidate.id)}."
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
            candidate_id=candidate.id,
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
        # Judged here, where it was made before, so each refusal of it comes
        # where it came; the write makes it only as it takes its backup, after
        # its own target check, its read and the hash check, so a refusal of
        # the making itself (the mkdir failing, or the directory found not a
        # real one after it) comes there now, after any refusal of those.
        # Measured before, with it made here: each refusal at the write left
        # it behind, empty.
        # Earlier versions of this change made it here and removed it when the
        # write refused; each, by its independent review, removed or made a
        # directory outside the state tree in some window of that removal.
        state.backup_dir_for(candidate.id, stamp, make=False)
        receipt = skill_store.write_canonical(
            candidate.target.canonical_path,
            candidate.patch.new_text,
            expected_sha256=candidate.target.base_sha256,
            backup_dir=lambda: state.backup_dir_for(candidate.id, stamp),
            skill_id=candidate.target.skill_id,
            actor=f"operator:{operator or 'unspecified'}",
            candidate_id=candidate.id,
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
            f"applied {iso(clock.now())} by operator:{operator or 'unspecified'}; "
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


#: Why a promotion waits for the run lock, keyed by action (``apply`` and
#: ``rollback`` share the first).
_PROMOTION_WHY = (
    "A promotion while a daily run reads the tree would change what the run is reading, and "
    "the run would record that as a skill tree mutation and block the schedule. Wait for the "
    "holder named above to finish (a daily run, or another apply, rollback or decline); "
    "nothing was written."
)
_LOCK_HELD_WHY = {
    "decline": (
        "A decline while a daily run counts the open queue would change the count under it. "
        "Wait for the holder named above to finish (a daily run, or an apply, rollback or "
        "decline); nothing was written."
    ),
}


@contextmanager
def _holding_the_run_lock(
    state: StateStore, action: str, candidate: Candidate | str
) -> Iterator[None]:
    """Hold the run lock for the part of a promotion that reads or writes the tree.

    The same lock ``daily`` takes, so a run and a promotion never overlap: the
    owner line names the promotion, so a run refused meanwhile says who holds
    it. Measured before: an apply during a scheduled run went through, the run
    finished ``blocked`` with the applied document in ``skill_tree.changes``,
    its candidates were closed as blocked-run, and the next morning's run was
    refused until the operator reviewed a mutation the tool had made itself.

    Apply and rollback both read the candidate record again once they hold it,
    and act on what it says then: the record they were handed was read before
    they waited, and another apply or rollback of the same candidate may have
    finished in between. Apply refuses unless the record still says
    ``validated``; rollback decides and records from the record it re-read.
    A decline takes it too, by candidate id, and reads the record only once it
    holds it: a daily run counts the open queue twice and must not see a decline
    change the count between, and a decline must not cross an apply, a rollback
    or another decline of the same candidate.
    """
    candidate_id = candidate if isinstance(candidate, str) else candidate.id
    lock = RunLock(state.lock_path(), owner=f"{action} {candidate_id} pid={os.getpid()}")
    try:
        lock.__enter__()
    except LockHeld as exc:
        raise LockHeld(
            f"refusing to {action} {candidate_id}: {exc} "
            + _LOCK_HELD_WHY.get(action, _PROMOTION_WHY)
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


def no_document_to_read(candidate: Candidate, path: Path, action: str, exc: AsiError) -> str:
    """The refusal of a candidate whose target is there but is no document to read.

    The store's ``read_text`` refuses a target that is a symbolic link, one that
    is not a regular file (a fifo, a device) and one whose bytes are not UTF-8.
    Measured before, ``review`` and ``apply`` gave that refusal as it was: it
    named no candidate and said nothing of what to do, and the one for bytes
    that are not UTF-8 said "the next pass will handle it", which is what
    ``daily`` does with such a skill. A regular UTF-8 file holding the text the
    candidate was written against was reviewed and applied as before; with the
    record deleted, ``daily`` quarantined the skill while the target kept its
    shape, and proposed again on another day once a regular file was back.
    """
    if isinstance(exc, LinkedTargetRefused):
        shape = "is a symbolic link, which apply refuses to replace"
    elif isinstance(exc, UndecodableDocument):
        cause = exc.__cause__
        at = (
            f" at byte {cause.start} ({cause.reason})"
            if isinstance(cause, UnicodeDecodeError)
            else ""
        )
        shape = f"is not valid UTF-8{at}"
    else:
        shape = "is not a regular file"
    return (
        f"refusing to {action} {candidate.id}: its canonical target {named_path(path)} {shape}, "
        "so there is no base text to show or to verify against. The skill and the candidate "
        "are unchanged, and nothing was backed up: with a regular UTF-8 file holding the "
        "text it was written against back at that path, it is reviewed and applied as "
        "before. To propose against other text instead, put a regular UTF-8 file there "
        "first, then "
        f"{repropose_advice(candidate.id)}."
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
            "recorded base hash and diff cannot be verified. To propose against the configured "
            f"tree, {repropose_advice(candidate.id)}."
        )
    try:
        return str(skill_store.read_text(path))
    except (BaseHashMismatch, LinkedTargetRefused, UndecodableDocument) as exc:
        raise type(exc)(no_document_to_read(candidate, path, action, exc)) from exc
    except OSError as exc:
        raise BaseHashMismatch(
            f"refusing to {action} {candidate.id}: its canonical target {named_path(path)} could "
            f"not be read ({why_unreadable(exc)}), so the recorded base hash and diff cannot be "
            "verified. If the skill root "
            "was moved or re-pointed, the candidate describes a file at its old path. To propose "
            f"against the configured tree, {repropose_advice(candidate.id)}."
        ) from exc


def rollback_candidate(
    candidate: Candidate,
    *,
    skill_store: Any,
    state: StateStore,
    clock: Clock,
) -> dict[str, Any]:
    """Restore the file this candidate overwrote, from its verified backup.

    Refused under the automation marker; requires no approval token or acknowledgement.
    Acts on the candidate record as it is once the run lock is held, not on the
    object it was handed. That record is checked against itself, then against
    the object it was handed: the two must name the same candidate, document,
    skill, base and replacement. Either refusal is ``CandidateTampered``, with
    advice that keeps the record and its backup, and comes before any other
    decision and any write. Its status and notes, and every write, come from the
    record as it is.
    """
    if automation_active():
        raise AutomationRefused("rollback refused: automation marker is set. " + ROLLBACK_ENV_HINT)
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
        # The record as it is now, read once the lock is held and after the
        # manifest is chosen (so a candidate never applied is still answered
        # "no backup manifest found"). Every decision and write below is made
        # from it, not from the object this rollback was handed, which was read
        # before it waited for the lock. Measured before (3.11 and 3.13): a
        # rollback holding a read made before the apply finished, or before a
        # first rollback, wrote its record from that read and dropped the note
        # the apply or the first rollback had added. A record deleted meanwhile
        # raises ``FileNotFoundError`` here and one made unreadable
        # ``StateUnreadable``, before anything is restored or written.
        current = state.read_candidate(candidate.id)
        # Checked against itself before any decision is made from it, as apply
        # checks the record it was handed: the restore's document and hashes
        # come from this one, and the backup manifest is checked against them.
        # Measured without this check (3.11 and 3.13), with the record and the
        # manifest edited together after the command read the record and
        # before it held the lock: one edited to name another canonical skill
        # restored that skill from a planted backup, and one edited to name a
        # later edit of the skill as the apply's write erased that edit; each
        # exited 0 and recorded the candidate rolled back. Its validation is
        # judged against itself only (``rubrics=None``): a rubric changed since
        # the apply does not stand in the way of undoing it.
        verify_full_integrity(
            current,
            rubrics=None,
            action="roll back",
            advice=_rollback_of_a_changed_record_advice(candidate.id),
        )
        # Bound to the record this rollback was handed, which the command read
        # before it waited for the lock: what is restored, from what to what,
        # must be what that read named. A record checked against itself still
        # passes when it is edited so that it recomputes. Measured with the
        # check above and without this one (3.11 and 3.13), the record and its
        # manifest edited as the rollback reached for the lock to name another
        # canonical skill holding the same applied text, its hashes, challenge
        # and validation left as they were: the rollback exited 0, overwrote
        # that skill, erasing its edit, and recorded the candidate rolled back.
        # This holds the two reads to each other; it does not make either one
        # trustworthy, since both are reads of the same editable file, and an
        # edit made before the command started is in both. Status and notes
        # are not compared: a rollback that waited while the apply or a first
        # rollback finished takes them from the record as it is.
        changed = _restore_identity_changes(candidate, current)
        if changed:
            raise CandidateTampered(
                f"refusing to roll back candidate {candidate.id}: its record changed after this "
                f"rollback read it: {', '.join(changed)} "
                f"{'differs' if len(changed) == 1 else 'differ'} from that read, so it is not "
                "the record this rollback was started on.\n  "
                + _rollback_of_a_changed_record_advice(candidate.id)
            )
        # A declined record is refused here: after the record is checked against
        # itself and against the read this rollback was handed (so a declined
        # record that was edited gets their advice, which keeps it and its
        # backup), and before anything is restored. This version declines only a
        # candidate no apply of it started, so a manifest beside a declined record
        # was made by an older asi or by hand, and what to restore is not known
        # from the state files.
        if current.status == STATUS_DECLINED:
            raise GateFailed(
                f"refusing to roll back {candidate.id}: its record says 'declined' "
                f"({_decided(current)}); a candidate this version declined was never applied "
                f"by it. A backup manifest exists for it ({one_line_label(str(manifest))}), so an "
                "older asi or a hand edit applied it: see 'A declined candidate with a backup "
                "manifest' in OPERATIONS. Nothing was written."
            )
        if current.status == STATUS_VALIDATED and _still_at_its_base(skill_store, current):
            # The manifest is written ahead of the skill, so an apply stopped
            # before its skill write leaves one for a write that never landed.
            # Restoring "from" it would close a candidate never applied as
            # rolled back, and the apply that would finish it is then refused.
            raise GateFailed(
                f"{candidate.id} was never applied: its skill still holds the text it was "
                "proposed against, so there is nothing to roll back. An apply of it stopped "
                "before its skill write leaves this; run that apply again to apply it."
            )
        _refuse_unwritable_receipt(state, "rollback", current)
        # The manifest must record this candidate's own write: its document,
        # from its base to its replacement. Measured before: a manifest edited
        # to name another skill, a planted backup or a later edit's hash was
        # restored from, and the candidate recorded rolled back.
        result = skill_store.rollback(
            manifest,
            path=Path(current.target.canonical_path),
            previous_sha256=current.target.base_sha256,
            new_sha256=sha256_text(current.patch.new_text),
        )
        return _record_rollback(current, manifest, result, state, clock)


def _restore_identity_changes(handed: Candidate, current: Candidate) -> list[str]:
    """The fields naming what a rollback restores that differ between two reads.

    The candidate, its document and skill, the base the backup must hold and
    the hash of the replacement the skill must hold: the restore is made from
    these. The id cannot differ once ``read_candidate`` returns, which refuses
    a record that does not claim the id it was read by; it is compared anyway.
    """
    pairs = (
        ("id", handed.id, current.id),
        ("target.canonical_path", handed.target.canonical_path, current.target.canonical_path),
        ("target.skill_id", handed.target.skill_id, current.target.skill_id),
        ("target.base_sha256", handed.target.base_sha256, current.target.base_sha256),
        ("patch.new_text", sha256_text(handed.patch.new_text), sha256_text(current.patch.new_text)),
    )
    return [name for name, was, now in pairs if was != now]


def _rollback_of_a_changed_record_advice(candidate_id: str) -> str:
    """What a rollback refused for its record, as it is under the lock, says to do.

    Not ``repropose_advice``: the record of an applied candidate, with its
    backup, is what a rollback restores from, so the advice keeps both. It
    names nothing from the state files to act on: each of them is an editable
    file, and their agreeing with each other proves nothing about any of them.
    """
    return (
        "Nothing was restored or written: the skill, this record and its backup are as "
        "they were. Keep the record and the backup. Never run the backup manifest's "
        "rollback_command, or any other command taken from the state files: they are "
        "editable, and their agreeing with each other establishes no trust. Put the record "
        "back only if you know its original contents from outside the state directory, "
        f"then run `asi rollback {candidate_id}` again; otherwise restore the skill as "
        'docs/OPERATIONS.md, Recovery, "Manual restoration" says.'
    )


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


def _decided(candidate: Candidate) -> str:
    """When and under which label a declined record says it was declined."""
    decision = candidate.decline
    if decision is None:
        return "no decision recorded"
    return f"at {decision.declined_at} by operator:{decision.operator or 'unspecified'}"


def _decline_beside_a_manifest_advice(candidate_id: str, manifest: Path) -> str:
    """What a decline refused as tampered says when an apply of the candidate started.

    Not ``repropose_advice``, whose "delete its record" would leave a skill the
    apply may have written with nothing to restore it from, and not rollback's,
    whose "run `asi rollback` again" a declined record is refused by. It names no
    command and no receipt from the state files: they are editable.
    """
    return (
        "Nothing was written: the skill, this record and its backup are as they were. An "
        f"apply of this candidate started (backup manifest {one_line_label(str(manifest))}), so "
        "it may have written the skill. Keep this record and the backup. Restore the skill as "
        'docs/OPERATIONS.md, Recovery, "Manual restoration" says.'
    )


def _an_apply_started_refusal(candidate: Candidate, manifest: Path) -> str:
    """The keep-the-record refusal of a record an apply of it started."""
    cid = candidate.id
    if candidate.status == STATUS_VALIDATED:
        way_out = (
            "Run that apply again (the same `asi apply` command) to finish it, then "
            f"`asi rollback {cid}` to restore the skill. If that apply can no longer run, see"
        )
    else:
        way_out = (
            f"Its record says {quoted_value(candidate.status)}, which apply refuses, so that "
            "apply cannot run again: see"
        )
    return (
        f"refusing to decline {cid}: an apply of {cid} started (backup manifest "
        f"{one_line_label(str(manifest))}), and it may have written the skill; decline does not "
        "close a candidate an apply may have written. Keep this record. "
        + way_out
        + " 'An apply that started cannot be declined' in docs/OPERATIONS.md before you move "
        "the record: deleting only the record does not restore the skill and leaves "
        "`asi rollback` nothing to act on. Nothing was written."
    )


def _not_declinable(candidate: Candidate) -> str:
    """Why a record no apply started, whose status is not declinable, is refused."""
    cid = candidate.id
    status = candidate.status
    if status == STATUS_PROPOSED:
        return (
            f"refusing to decline {cid}: it was never validated, so there is no gate to decline "
            "against; the next run validates or replaces it. To close it now, delete its "
            "record: no apply of a proposed candidate starts"
        )
    if status == STATUS_APPLIED:
        return f"refusing to decline {cid}: it is applied; `asi rollback {cid}` restores the skill"
    if status in CLOSED_STATUSES:
        return (
            f"refusing to decline {cid}: already closed ({status}); a decline would change nothing"
        )
    return (
        f"refusing to decline {cid}: it is not validated (its status is "
        f"{quoted_value(status)}); only a validated candidate can be declined"
    )


DECLINE_RECEIPT_SCHEMA = "asi.decline_receipt.v1"


def _decline_receipt(candidate: Candidate, decision: Decline, *, finishing: bool) -> dict[str, Any]:
    """The decline receipt: the decision as the record holds it, and what was declined."""
    validation = candidate.validation or {}
    gate = validation.get("gate") or {}
    return {
        "schema": DECLINE_RECEIPT_SCHEMA,
        "candidate_id": candidate.id,
        "declined_at": decision.declined_at,
        "operator": decision.operator,
        "reason": decision.reason,
        "status_before": STATUS_VALIDATED,
        "skill_id": candidate.target.skill_id,
        "canonical_path": candidate.target.canonical_path,
        "base_sha256": candidate.target.base_sha256,
        "patch_sha256": candidate.recomputed_patch_sha256,
        "validation_schema": validation.get("schema"),
        "gate": {
            "passed": gate.get("passed"),
            "reasons": gate.get("reasons"),
            "metrics": gate.get("metrics"),
        },
        "improver_command_id": candidate.improver_command_id,
        "finishes_an_earlier_run": finishing,
    }


#: Every field of a decline receipt this version writes, each with the types it
#: writes there: a receipt without one of them, or holding any other, is not one
#: it wrote. Types are matched exactly, so a ``true`` is not a number.
_DECLINE_RECEIPT_TYPES: dict[str, tuple[type, ...]] = {
    "schema": (str,),
    "candidate_id": (str,),
    "declined_at": (str,),
    "operator": (str,),
    "reason": (str,),
    "status_before": (str,),
    "skill_id": (str,),
    "canonical_path": (str,),
    "base_sha256": (str,),
    "patch_sha256": (str,),
    "validation_schema": (str, type(None)),
    "gate": (dict,),
    "improver_command_id": (str, type(None)),
    "finishes_an_earlier_run": (bool,),
}

#: The receipt fields that must be what the declined record read under the
#: lock gives, as JSON (``same_json``: by type and value, nested fields
#: included): the candidate id and the decision first (the bound five), then
#: what was declined. The schema and ``finishes_an_earlier_run`` are judged
#: above them: the record does not say which run wrote the receipt.
_DECLINE_RECEIPT_IDENTITY = (
    "candidate_id",
    "declined_at",
    "operator",
    "reason",
    "status_before",
    "skill_id",
    "canonical_path",
    "base_sha256",
    "patch_sha256",
    "validation_schema",
    "gate",
    "improver_command_id",
)


def _fields(names: Sequence[str]) -> str:
    return ", ".join(names)


def same_json(left: Any, right: Any) -> bool:
    """Whether two JSON-loaded values are the same JSON value: by type and value, nested.

    Python's ``==`` is not JSON's: ``True == 1`` and ``1 == 1.0`` hold, while
    ``true``, ``1`` and ``1.0`` are three values. Here each pair must have
    exactly the same type (``type(x) is type(y)``, not ``isinstance``, which
    takes a ``bool`` for an ``int``); an object must have the same key set and
    the same value at each key, an array the same length and the same item at
    each place, in order; a string, a number of one type or a null is equal by
    value. Walked with a stack of its own rather than by recursion, so however
    deep the two values nest the answer is a bool, never a ``RecursionError``.
    Measured before (FIX 2, audit 2): with ``==``, a decline receipt edited from
    ``gate.passed: true`` to ``1``, or from a gate metric ``0`` to ``false``,
    agreed with its record and a rerun exited 0, on 3.13 and 3.11.
    """
    pending = [(left, right)]
    while pending:
        one, other = pending.pop()
        if type(one) is not type(other):
            return False
        if isinstance(one, dict):
            if one.keys() != other.keys():
                return False
            pending.extend((one[key], other[key]) for key in one)
        elif isinstance(one, list):
            if len(one) != len(other):
                return False
            pending.extend(zip(one, other))
        elif one != other:
            return False
    return True


def _decline_receipt_damage(receipt: dict[str, Any], current: Candidate) -> tuple[str, str] | None:
    """Why a decline receipt is not the one this version wrote for ``current``, or None.

    ``("shape", why)`` for a receipt this version does not write (its schema, a
    field missing or one it never writes, a field of another type), and
    ``("differs", why)`` for one that disagrees with the record. ``why`` names
    fields, never what they hold: the receipt is a state file, editable by
    anyone who can write ``state/``, so its content is neither printed nor
    trusted.
    """
    decision = current.decline
    if decision is None:  # pragma: no cover - the integrity check refuses it first
        return ("differs", "the record holds no decision")
    if receipt.get("schema") != DECLINE_RECEIPT_SCHEMA:
        return ("shape", f"its schema is not {DECLINE_RECEIPT_SCHEMA}")
    missing = [name for name in _DECLINE_RECEIPT_TYPES if name not in receipt]
    if missing:
        return ("shape", f"{_fields(missing)} missing")
    extra = len(set(receipt) - set(_DECLINE_RECEIPT_TYPES))
    if extra:
        return ("shape", f"{extra} field(s) this version does not write")
    wrong = [
        name for name, kinds in _DECLINE_RECEIPT_TYPES.items() if type(receipt[name]) not in kinds
    ]
    if wrong:
        return ("shape", f"{_fields(wrong)} of the wrong type")
    expected = _decline_receipt(current, decision, finishing=receipt["finishes_an_earlier_run"])
    differ = [
        name for name in _DECLINE_RECEIPT_IDENTITY if not same_json(receipt[name], expected[name])
    ]
    if differ:
        return ("differs", f"{_fields(differ)} {'differs' if len(differ) == 1 else 'differ'}")
    return None


def _refuse_a_damaged_decline_receipt(
    receipt: dict[str, Any], current: Candidate, receipt_path: Path, record_path: Path
) -> None:
    """A ``StateUnreadable`` naming the receipt, quoting none of it, unless it is this record's."""
    damage = _decline_receipt_damage(receipt, current)
    if damage is None:
        return
    kind, why = damage
    cid = current.id
    if kind == "shape":
        raise StateUnreadable(
            f"refusing to decline {cid}: the decline receipt {one_line_label(str(receipt_path))} "
            f"is not one this version writes ({why}), so it was changed by hand or written by "
            "another program; none of it is shown or used. It is left as it is. Move it aside "
            "and run decline again to write it from the record. Nothing was written."
        )
    raise StateUnreadable(
        f"refusing to decline {cid}: the decline receipt "
        f"{one_line_label(str(receipt_path))} does not agree with the record "
        f"{one_line_label(str(record_path))} ({why}); this version writes the "
        "receipt from the record, so one of them was changed by hand. Both are left as "
        "they are; nothing was written."
    )


#: What ``decline`` prints, text and ``--json`` alike, and nothing else.
DECLINE_RESULT_FIELDS = (
    "candidate_id",
    "status",
    "outcome",
    "declined_at",
    "operator",
    "reason",
    "skill_id",
    "receipt_path",
    "masked",
)


def _decline_result(
    candidate: Candidate, decision: Decline, outcome: str, receipt_path: Path, masked: int
) -> dict[str, Any]:
    """What a decline reports: fixed strings and what the checked record holds, never a receipt.

    The candidate id and the skill id are the record's, read under the lock and
    checked against itself; the time, the operator label and the reason are its
    decision, shape-checked and masked; the receipt path is the state store's own.
    A receipt's fields, even a verified receipt's, are not copied in.
    """
    return {
        "candidate_id": candidate.id,
        "status": STATUS_DECLINED,
        "outcome": outcome,
        "declined_at": decision.declined_at,
        "operator": decision.operator,
        "reason": decision.reason,
        "skill_id": candidate.target.skill_id,
        "receipt_path": str(receipt_path),
        "masked": masked,
    }


def _checked_decline_input(
    reason: str, operator: str, *, repair: bool
) -> tuple[str, str, int, bool]:
    """The reason and the operator label as they are stored, or a refusal naming the rule.

    The raw rules of ``decline_reason_problem`` on the text as typed, then the
    masking fixed point. Repair input (a stored reason over 500 given back) is
    carried as it is, flagged, and judged under the lock against the record.
    """
    text = reason.strip()
    if repair and is_repair_input(text):
        stored, masks = text, 0
    else:
        repair = False
        problem = decline_reason_problem(text, limit=MAX_DECLINE_REASON_CHARS, what="reason")
        if problem:
            raise SafetyRefusal(f"decline refused: {problem}. Nothing was written.")
        stored, masks = settled_mask(text)
    label = operator.strip()
    if label:
        problem = decline_reason_problem(
            label, limit=MAX_DECLINE_OPERATOR_CHARS, what="operator label"
        )
        if problem:
            raise SafetyRefusal(f"decline refused: {problem}. Nothing was written.")
        label, _ = settled_mask(label)
    return stored, label, masks, repair


def decline_candidate(
    candidate_id: str,
    *,
    reason: str,
    operator: str,
    state: StateStore,
    clock: Clock,
    repair: bool = False,
) -> dict[str, Any]:
    """Close a validated candidate on the record, with a reason. Writes no skill file.

    Refused under the automation marker. The reason and the operator label are
    checked and masked first; then, holding the run lock, every read before any
    write: the record; the backup manifest, whatever the status (a record an
    apply started is kept, never declined); the status; the record checked
    against itself (``rubrics=None``: a rubric changed since validation does
    not stand in the way of saying no); the decline receipt. A declined record
    is a rerun: the same reason writes a missing receipt from the record, or
    nothing; another is refused, the first kept. Otherwise the record is
    written first (``declined``, its decision and a note), then the receipt,
    which is never written over a leaf: a stop between the two leaves a closed
    record that apply and rollback refuse and a rerun finishes.

    An existing decline receipt beside a declined record is checked before it
    is relied on: its schema, its exact fields and their types, and its
    agreement with the record read under the lock, compared as JSON by type
    and value, nested fields included (``same_json``); any damage is a
    ``StateUnreadable`` that names the receipt and quotes none of it.

    Returns the ``DECLINE_RESULT_FIELDS`` (``_decline_result``), built from the
    record and fixed strings, the outcome one of ``declined``,
    ``receipt-written`` or ``already-declined``.
    """
    if automation_active():
        raise AutomationRefused("decline refused: automation marker is set. " + DECLINE_ENV_HINT)
    stored, label, masks, repair = _checked_decline_input(reason, operator, repair=repair)
    with _holding_the_run_lock(state, "decline", candidate_id):
        current = state.read_candidate(candidate_id)
        record_path = state.candidate_path(candidate_id)
        # Before any status refusal and before the integrity check: the refusals
        # of a proposed record and of a tampered one say to delete the record,
        # which is never the advice for a candidate an apply may have written.
        manifest = state.latest_backup_manifest(candidate_id)
        if (
            manifest is not None
            and current.status != STATUS_DECLINED
            and current.status not in CLOSED_STATUSES
        ):
            raise GateFailed(_an_apply_started_refusal(current, manifest))
        if current.status not in (STATUS_VALIDATED, STATUS_DECLINED):
            raise GateFailed(_not_declinable(current))
        verify_full_integrity(
            current,
            rubrics=None,
            action="decline",
            advice=(
                _decline_beside_a_manifest_advice(candidate_id, manifest)
                if manifest is not None
                else None
            ),
        )
        receipt = state.read_action_receipt("decline", candidate_id)
        receipt_path = state.action_receipt_path("decline", candidate_id)
        if current.status == STATUS_DECLINED:
            return _decline_again(current, stored, receipt, record_path, receipt_path, state, masks)
        if repair:
            raise SafetyRefusal(
                f"refusing to decline {candidate_id}: the reason is {len(stored)} characters; a "
                f"new decline takes at most {MAX_DECLINE_REASON_CHARS}. Nothing was written."
            )
        if receipt is not None:
            raise StateUnreadable(
                f"refusing to decline {candidate_id}: a decline receipt "
                f"{one_line_label(str(receipt_path))} exists but the record says validated; this "
                "version writes the record before the receipt, so that receipt is not one it "
                "wrote. Move it aside and run decline again. Nothing was written."
            )
        gate = (current.validation or {}).get("gate") or {}
        if not gate.get("passed"):
            reasons = quoted_values(gate.get("reasons") or []) or "no gate decision recorded"
            raise GateFailed(f"candidate {candidate_id} did not pass the gate: {reasons}")
        _refuse_unwritable_receipt(state, "decline", current)
        declined_at = iso(clock.now())
        decision = Decline(declined_at=declined_at, operator=label, reason=stored)
        # The record first: what every other command must see to stop is its
        # status. The reason is not in the note, which says who and when.
        state.write_candidate(
            replace(
                current,
                status=STATUS_DECLINED,
                decline=decision,
                notes=current.notes
                + (f"declined {declined_at} by operator:{label or 'unspecified'}",),
            )
        )
        payload = _decline_receipt(current, decision, finishing=False)
        path = state.write_action_receipt("decline", candidate_id, payload, replace=False)
        return _decline_result(current, decision, "declined", path, masks)


def _decline_again(
    current: Candidate,
    stored: str,
    receipt: dict[str, Any] | None,
    record_path: Path,
    receipt_path: Path,
    state: StateStore,
    masks: int,
) -> dict[str, Any]:
    """A decline of a record already declined: finish its receipt, or say nothing changed.

    The reason is matched byte for byte against the record read under the lock,
    never against an earlier read. The first decision is never replaced: the
    record is not rewritten, and a receipt is written only where none stands.
    """
    cid = current.id
    decision = current.decline
    if decision is None:  # pragma: no cover - the integrity check refuses it first
        raise StateUnreadable(f"refusing to decline {cid}: its declined record holds no decision")
    if receipt is not None:
        _refuse_a_damaged_decline_receipt(receipt, current, receipt_path, record_path)
    if stored != decision.reason:
        tail = (
            " Nothing was written."
            if receipt is not None
            else (
                " To write its missing receipt, run decline again with the recorded reason "
                f"exactly: copy `decline.reason` from {one_line_label(str(record_path))} (it is "
                "already masked, and may be longer than 500 characters) and give it on standard "
                "input with `--reason -`; 'Declining a candidate' in docs/OPERATIONS.md gives a "
                "pipe that decodes it from the record. Nothing was written."
            )
        )
        raise GateFailed(
            f"refusing to decline {cid}: already declined {_decided(current)} with another "
            "reason; the first is kept." + tail
        )
    if receipt is not None:
        return _decline_result(current, decision, "already-declined", receipt_path, 0)
    _refuse_unwritable_receipt(state, "decline", current)
    payload = _decline_receipt(current, decision, finishing=True)
    path = state.write_action_receipt("decline", cid, payload, replace=False)
    return _decline_result(current, decision, "receipt-written", path, 0)
