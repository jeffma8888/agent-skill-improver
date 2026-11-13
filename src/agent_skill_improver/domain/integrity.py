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
from typing import Mapping

from ..errors import CandidateTampered
from ..hashing import sha256_text, short
from .candidate import (
    STATUS_APPLIED,
    STATUS_BLOCKED_RUN,
    STATUS_GATE_FAILED,
    STATUS_PROPOSED,
    STATUS_ROLLED_BACK,
    STATUS_VALIDATED,
    Candidate,
    TargetRef,
    challenge_for,
    make_patch,
)
from .redaction import quoted_value, quoted_values
from .rubric import Rubric
from .skills import SKILL_FILENAME
from .validation import CASE_TARGET, TargetCaseRef, ValidationRecord

#: Statuses that may only exist once validation ran and recorded a gate.
_STATUSES_REQUIRING_VALIDATION = (
    STATUS_VALIDATED,
    STATUS_GATE_FAILED,
    STATUS_APPLIED,
    STATUS_ROLLED_BACK,
    STATUS_BLOCKED_RUN,
)

#: Statuses that assert the gate passed.
_STATUSES_ASSERTING_A_PASS = (STATUS_VALIDATED, STATUS_APPLIED, STATUS_ROLLED_BACK)

_SHA256_LEN = 64


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LEN and all(c in "0123456789abcdef" for c in value)


def _recorded_digest(value: str) -> str:
    """A hash read from the record as a refusal shows it.

    Its 12-character prefix, as before, when it is a digest; quoted otherwise
    (``quoted_value``), since a prefix of anything else is a raw slice of the
    record: a line break first and 11 characters of a forged line after it.
    """
    return short(value) if _is_sha256(value) else quoted_value(value)


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
            f"target.canonical_path names {quoted_value(path.name)}, but only {SKILL_FILENAME} "
            "is ever written"
        )
    parent = path.parent.name
    if target.skill_id.strip() and parent and parent != target.skill_id:
        problems.append(
            f"target.skill_id is {quoted_value(target.skill_id)} but target.canonical_path "
            f"sits in {quoted_value(parent)}: the record names two different skills"
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
    # STATUS_BLOCKED_RUN is deliberately consistent with either gate verdict: the
    # run that computed the verdict saw the skill tree change under it, so the
    # verdict is not what closed the candidate.
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
            f"patch.new_text_sha256 records {_recorded_digest(patch.new_text_sha256)} but "
            f"new_text hashes {short(recomputed)}: the reviewed text and its recorded hash "
            "disagree"
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
            f"target.base_sha256 records {_recorded_digest(candidate.target.base_sha256)} but "
            f"the live base text hashes {short(live)}"
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


def repropose_advice(candidate_id: str) -> str:
    """What proposing a refused candidate's change again takes.

    Every ``review`` or ``apply`` refusal that sends the operator back to
    ``daily`` says it in these words. "Re-run ``asi daily --commit``" alone was
    untrue twice over. Both commands refuse a candidate that is not
    ``validated`` before any check that gives this advice, so the record refused
    is still open and counts against its skill's
    ``caps.max_open_candidates_per_skill``: a run for another day refuses that
    skill ("open candidate(s) already target"), and review refuses the record again with the
    same advice. And a day never re-scores a conversation it has already scored,
    so a run for the same day does not look at the ones behind the candidate.
    Deleting the record is how a candidate review refuses is closed. A candidate
    ``daily`` refuses at validate has no record to delete: ``set_aside_advice``
    is what that refusal says.
    """
    return (
        f"delete its record ({candidate_id}.json in the state directory's candidates/), "
        "which counts against its skill's open-candidate cap while it exists, then run "
        "`asi daily --commit` for another day: a run for the same day skips every "
        "conversation that day already scored, the ones behind this candidate included"
    )


def set_aside_advice() -> str:
    """What proposing again takes after ``daily`` refuses a candidate at validate.

    ``validate_candidate`` checks a candidate ``daily`` has just built, before
    ``daily`` writes its record, and ``daily`` sets the skill aside (deferred,
    or quarantined once deferral is exhausted) when the check refuses. There is
    no record to delete or to review, and nothing counts against the skill's
    open-candidate cap: measured, a run for the same day stood down and a run
    for the next day proposed the change again. The record advice of
    ``repropose_advice`` named a file that was never written. A dry run marks
    no conversation as scored, so after one a committed run for the same day
    would propose again too; another day still does.
    """
    return (
        "No record of it was written, since `daily` writes one only after validating it, so "
        "there is nothing to delete or to review. This run sets its skill aside; to propose "
        "again, run `asi daily --commit` for another day, because a run for the same day skips "
        "every conversation that day already scored, the ones behind this candidate included"
    )


def verify_candidate_integrity(
    candidate: Candidate, *, base_text: str | None = None, action: str
) -> None:
    """Refuse ``action`` unless every derived field recomputes to what is stored.

    ``daily`` calls it with ``action="validate"``, on a candidate whose record it
    has not written yet; that refusal says so rather than describe a record.
    """
    problems = derived_field_mismatches(candidate, base_text=base_text)
    if not problems:
        return
    if action == "validate":
        raise CandidateTampered(
            f"refusing to validate candidate {candidate.id}: {len(problems)} derived field(s) "
            "do not match a fresh recomputation.\n"
            + "\n".join(f"  - {p}" for p in problems)
            + f"\n  {set_aside_advice()}. Nothing has been written to any skill file."
        )
    raise CandidateTampered(
        f"refusing to {action} candidate {candidate.id}: {len(problems)} derived field(s) do not "
        "match a fresh recomputation, so the record has changed since it was written.\n"
        + "\n".join(f"  - {p}" for p in problems)
        + "\n  An approval token is derived from the candidate's content, so a record edited "
        "after review must never be authorised by the token printed before it. To propose and "
        f"validate the change again, {repropose_advice(candidate.id)}; nothing has been "
        "written to any skill file."
    )


def validation_mismatches(
    candidate: Candidate, *, rubrics: Mapping[str, Rubric] | None = None
) -> list[str]:
    """Every disagreement between the persisted validation and a fresh recomputation.

    ``candidate.validation`` is a plain JSON blob like every other field here, so
    it is parsed strictly (``ValidationRecord.parse``): an unknown key, a missing
    key, a wrong type, a duplicate case or a value that contradicts another
    field is a mismatch, not a best-effort read. Once it parses, the gate is
    recomputed from the record's own outcomes, judgements and policy --
    ``judge_independent`` derived from the recorded judge and improver command
    ids, never from a stored boolean -- and compared byte-exact to the persisted
    decision. When ``rubrics`` (the currently configured rubrics, keyed by id)
    is supplied, every rubric binding the record claims to have scored against
    is also checked for exact version and source-hash agreement, and every
    outcome is checked against the live rubric it names actually grading the
    criterion it reports.

    The binding between record and candidate is the exact
    ``(rubric_id, conversation_id, criterion_id)`` triple, in both directions:

    * the record's claim set must equal the candidate's own ``rubric_failures``,
      so a failure ref cannot be added to or removed from either side alone;
    * no target outcome may name a triple the candidate does not claim, so a
      rubric id cannot be swapped for another rubric that happens to be live
      with a matching version and source hash;
    * for a promotion-eligible status (one that asserts the gate passed) every
      claimed triple must have a persisted target outcome, so a candidate can
      never be offered for apply on partial evidence.
    """
    validation = candidate.validation
    if candidate.status not in _STATUSES_REQUIRING_VALIDATION:
        return []
    if not validation:
        # No gate decision is recorded at all; gate_status_problems already
        # reports that disagreement against the status.
        return []
    try:
        record = ValidationRecord.parse(validation)
    except (ValueError, TypeError) as exc:
        return [f"validation record does not parse as {candidate.id}'s validation: {exc}"]

    problems: list[str] = []
    if record.candidate_id != candidate.id:
        problems.append(
            f"validation.candidate_id is {quoted_value(record.candidate_id)} but this is candidate "
            f"{candidate.id!r}: the validation record describes a different candidate"
        )
    if record.base_sha256 != candidate.target.base_sha256:
        problems.append(
            "validation.base_sha256 does not match target.base_sha256: the gate was decided "
            "against a different base than this record now claims"
        )
    if record.patch_sha256 != candidate.recomputed_patch_sha256:
        problems.append(
            "validation.patch_sha256 does not match the hash of the current new_text: the gate "
            "was decided about different patch text than this record now carries"
        )
    if record.improver_command_id != candidate.improver_command_id:
        problems.append(
            "validation.improver_command_id does not match the candidate's own "
            "improver_command_id: judge independence cannot be derived from a command id "
            "that disagrees with the candidate it is supposed to describe"
        )
    claimed_refs = {
        TargetCaseRef(
            rubric_id=ref.rubric_id,
            conversation_id=ref.conversation_id,
            criterion_id=ref.criterion_id,
        )
        for ref in candidate.rubric_failures
    }
    bound_refs = set(record.target_refs)
    if claimed_refs != bound_refs:
        detail = []
        omitted = sorted(r.key for r in claimed_refs - bound_refs)
        added = sorted(r.key for r in bound_refs - claimed_refs)
        if omitted:
            detail.append(
                "claimed by the candidate but not bound by the validation: "
                f"[{quoted_values(omitted)}]"
            )
        if added:
            detail.append(
                "bound by the validation but not claimed by the candidate: "
                f"[{quoted_values(added)}]"
            )
        problems.append(
            "validation.target_refs is not exactly this candidate's own rubric_failures as "
            "(rubric_id, conversation_id, criterion_id) triples: "
            + "; ".join(detail)
            + ". Adding or dropping a claimed failure on either side changes what the gate was "
            "asked to prove, so neither may move without the other"
        )
    replayed_refs = {o.ref for o in record.outcomes if o.case.kind == CASE_TARGET}
    invented = sorted(r.key for r in replayed_refs - claimed_refs)
    if invented:
        problems.append(
            f"validation.outcomes has target case(s) [{quoted_values(invented)}] that are not "
            "among this "
            "candidate's own rubric_failures: a target outcome must be about a failure this "
            "candidate actually claims to fix, named by rubric as well as by conversation and "
            "criterion, not an invented, unrelated or re-attributed case"
        )
    if candidate.status in _STATUSES_ASSERTING_A_PASS:
        unreplayed = sorted(r.key for r in claimed_refs - replayed_refs)
        if unreplayed:
            problems.append(
                f"status is {candidate.status!r}, which asserts the gate passed, but claimed "
                f"target case(s) [{quoted_values(unreplayed)}] have no persisted replay outcome: a "
                "promotion-eligible candidate must carry a measured target replay for every "
                "failure it claims to fix, and a partial replay is not the evidence the status "
                "advertises"
            )
    bound_versions = {binding.rubric_id: binding.version for binding in record.rubrics}
    for ref in candidate.rubric_failures:
        bound_version = bound_versions.get(ref.rubric_id)
        if bound_version is not None and bound_version != ref.rubric_version:
            problems.append(
                f"rubric_failures names {quoted_value(ref.rubric_id)} at version "
                f"{quoted_value(ref.rubric_version)} but "
                f"validation.rubrics binds that rubric at version {bound_version}: the failure "
                "was measured by a different version of the examiner than the gate was"
            )

    problems.extend(record.recompute_gate_mismatches())

    if rubrics is not None:
        for binding in record.rubrics:
            live = rubrics.get(binding.rubric_id)
            if live is None:
                problems.append(
                    f"validation.rubrics references {quoted_value(binding.rubric_id)}, which is "
                    "not among "
                    "the currently configured rubrics"
                )
                continue
            if live.version != binding.version:
                problems.append(
                    f"validation.rubrics[{quoted_value(binding.rubric_id)}].version is "
                    f"{binding.version} but "
                    f"the currently configured rubric is version {live.version}"
                )
            if live.source_sha256 != binding.source_sha256:
                problems.append(
                    f"validation.rubrics[{quoted_value(binding.rubric_id)}].source_sha256 does "
                    "not match the "
                    "currently configured rubric's source: the criteria that decided this gate "
                    "are not the criteria configured now"
                )
        # The unseen outcomes are checked the same way: they are never a gate
        # input, but an outcome naming an examiner that cannot have produced it
        # is invalid evidence wherever it sits in the record.
        scored = [("validation.outcomes", o) for o in record.outcomes]
        if record.unseen is not None:
            scored += [("validation.unseen.outcomes", o) for o in record.unseen.outcomes]
        for where, outcome in scored:
            live_rubric = rubrics.get(outcome.rubric_id)
            if live_rubric is None:
                # Already reported once by the binding check above; saying it
                # again per outcome would bury the specific problem in noise.
                continue
            if live_rubric.criterion(outcome.case.criterion_id) is None:
                problems.append(
                    f"{where} case {quoted_value(outcome.case.key)} says it was "
                    f"scored by {quoted_value(outcome.rubric_id)}, but that rubric has no "
                    f"criterion {quoted_value(outcome.case.criterion_id)}: the outcome names an "
                    "examiner that cannot "
                    "have produced it"
                )
    return problems


#: How many problems a tampered-record refusal lists before it counts the rest.
#: Each is one line, its quoted values bounded; a record edited to differ in ten
#: thousand places is refused in a screenful, with the full count in its first line.
PROBLEMS_SHOWN = 20


def verify_full_integrity(
    candidate: Candidate,
    *,
    base_text: str | None = None,
    rubrics: Mapping[str, Rubric] | None = None,
    action: str,
) -> None:
    """The one fail-closed verifier review and apply must call before anything else.

    Combines every derived-field check on the candidate itself with a strict
    recomputation of its persisted validation, and refuses ``action`` before a
    token, backup, receipt or write is produced if either disagrees with a
    fresh recomputation.
    """
    problems = derived_field_mismatches(candidate, base_text=base_text)
    problems += validation_mismatches(candidate, rubrics=rubrics)
    if not problems:
        return
    listed = [f"  - {p}" for p in problems[:PROBLEMS_SHOWN]]
    if len(problems) > PROBLEMS_SHOWN:
        listed.append(f"  - and {len(problems) - PROBLEMS_SHOWN} more not shown")
    raise CandidateTampered(
        f"refusing to {action} candidate {candidate.id}: {len(problems)} derived field(s) do not "
        "match a fresh recomputation, so the record has changed since it was written.\n"
        + "\n".join(listed)
        + "\n  An approval token is derived from the candidate's content, so a record edited "
        "after review must never be authorised by the token printed before it. To propose and "
        f"validate the change again, {repropose_advice(candidate.id)}; nothing has been "
        "written to any skill file."
    )
