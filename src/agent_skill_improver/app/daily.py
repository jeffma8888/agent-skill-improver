"""The daily driver.

What it is allowed to do: ingest, score, isolate, propose, validate, and write
artifacts under the gitignored state directory.

What it can never do: change a live skill file, or promote anything. That is not
enforced by care. It is enforced by three separate mechanisms:

1. this module does not import :mod:`agent_skill_improver.app.promote`, and a
   test asserts it by reading this file's source;
2. it refuses to start unless the skill store it was handed is read-only;
3. it hashes the whole skill tree before and after the run and records both in
   the receipt, so a mutation from any source at all is visible afterwards.

What it can never do *first*: read a private conversation. The isolation
decision is the first statement in the function, before the lock, before the
skill tree is hashed and before discovery -- because a check that runs after the
export has happened is an audit, not a gate. A private source whose model roles
are not confined to a verified OS no-egress sandbox raises
:class:`~agent_skill_improver.errors.IsolationRequired` and nothing is fetched.

The other properties worth naming, because each one is a failure mode this
package has seen elsewhere:

* **per-record quarantine and defer.** A defective record is quarantined alone; an
  unreachable one is deferred alone. One bad record can never veto a batch,
  because a whole-batch veto plus a non-zero failure rate is a ratchet: the queue
  grows for ever and nothing is ever promoted again.
* **bounded fair selection.** The batch is bounded by round-robin across a
  fairness dimension, never by truncating a sorted list. Sorted truncation is a
  permanent filter whenever the sort key correlates with the content, which for
  ids, filenames and timestamps it almost always does.
* **idempotence.** Records that reached a terminal state for the day are skipped
  on a re-run. Deferred records are deliberately retried.
* **a backlog guard.** When the promotable queue is already full the run stands
  down instead of adding to it, and says so.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

from ..clock import Clock, day_key, iso
from ..config import Config
from ..domain.approval import AutomationMarker
from ..domain.authorship import (
    CLASS_UNKNOWN,
    DISCOVERY_CLASSES,
    class_histogram,
    classify_conversation,
    classify_thread,
)
from ..domain.candidate import STATUS_GATE_FAILED, STATUS_VALIDATED
from ..domain.conversation import NormalizedConversation
from ..domain.discovery import error_page, merge_pages
from ..domain.failure import FailureCase
from ..domain.isolation import decide as decide_isolation
from ..domain.rubric import Rubric
from ..domain.selection import bucket_coverage, select_fair
from ..domain.skills import SkillGroup, group_for
from ..errors import AsiError, RecordDefect, SafetyRefusal, TransientFailure
from ..ports import ModelRunner
from .lock import RunLock
from .pipeline import propose_candidate, score_and_isolate
from .receipt import STATUS_STOOD_DOWN, RunReceipt
from .state import StateStore, safe_name
from .validate import validate_candidate

STAGE_INGEST = "ingest"
STAGE_SCORE = "score"
STAGE_PROPOSE = "propose"
STAGE_VALIDATE = "validate"


def _runner_isolation(runner: Any) -> Mapping[str, Any]:
    """Read the runner's confinement, treating an unanswerable runner as none.

    A runner that cannot describe its own isolation is reported as ``mode:
    "unknown"`` rather than skipped. Defaulting to "no requirement" when the
    question cannot be answered is the fail-open version of this check, and it
    is the version that lets a fake or a future adapter walk past the gate.
    """
    getter = getattr(runner, "isolation", None)
    if callable(getter):
        value = getter()
        if isinstance(value, Mapping):
            return value
    return {"mode": "unknown", "verified": False, "env_allowlist": []}


def _bucket_of(dimension: str, *, agent_title_markers: Sequence[str] = ()) -> Callable[[Any], str]:
    """The fairness dimension, computed from discovery metadata only.

    Selection has to bound the batch before anything is exported, so a dimension
    that needs the fetched conversation is not available here.
    """
    if dimension == "date":
        return lambda t: getattr(t, "updated_date", "") or "unknown-date"
    if dimension == "workspace":
        return lambda t: getattr(t, "workspace", "") or "unknown-workspace"
    return lambda t: classify_thread(t, agent_title_markers=tuple(agent_title_markers))


def run_daily(
    config: Config,
    *,
    source: Any,
    runner: ModelRunner,
    skill_store: Any,
    state: StateStore,
    clock: Clock,
    day: str | None = None,
    commit: bool | None = None,
    seed: str | None = None,
) -> RunReceipt:
    """Run one bounded, checkpointed, non-mutating daily pass."""
    # FIRST, before the lock, before any state is written and before a single
    # byte of conversation is read: is this configuration allowed to handle this
    # source at all? Ordering is the whole point. Everything below this line
    # either exports private content or writes something derived from it.
    isolation = decide_isolation(
        source_kind=config.source.kind,
        runner_isolation=_runner_isolation(runner),
        roles=sorted(config.model_roles),
    )
    isolation.require()

    if not getattr(skill_store, "read_only", False):
        raise SafetyRefusal(
            "the daily driver refuses a writable skill store. Automation may propose and "
            "validate; only `asi apply` may write, and only with a human approval token."
        )
    started = clock.now()
    run_day = day or day_key(started)
    run_seed = seed or config.seed or run_day
    dry_run = config.dry_run if commit is None else not commit

    state.ensure()
    receipt = RunReceipt(
        run_id=f"{run_day}:{iso(started)}",
        day=run_day,
        started_at=iso(started),
        dry_run=dry_run,
        config_path=str(config.config_path),
        seed=run_seed,
        state_dir=str(state.root),
    )
    receipt.source = config.source.to_dict()
    receipt.caps = config.caps.to_dict()
    receipt.isolation = isolation.to_dict()

    with RunLock(state.dir("locks") / "daily.lock"), AutomationMarker():
        digest_before = skill_store.tree_digest()
        rubrics: dict[str, Rubric] = {r.id: r for r in config.rubrics}
        known_skill_ids = skill_store.known_skill_ids()
        groups = skill_store.groups()
        receipt.skill_tree = {
            "digest_before": digest_before,
            "roots_present": [r.id for r in skill_store.existing_roots()],
            "roots_missing": [r.id for r in skill_store.missing_roots()],
            "skills_discovered": len(known_skill_ids),
            "mirrored_skills": sum(1 for g in groups if g.is_mirrored),
            "skills_with_divergent_mirrors": sum(1 for g in groups if g.divergent_mirrors),
            "skills_that_do_not_load": [s.skill_id for s in skill_store.snapshots() if not s.loads],
            "skills_with_executable_payloads": sorted(
                {s.skill_id for s in skill_store.snapshots() if s.payload_files}
            ),
            "executable_payload_files": sum(len(s.payload_files) for s in skill_store.snapshots()),
        }
        receipt.evaluator_independence = config.evaluator_independence()
        if not receipt.evaluator_independence.get("judge_independent"):
            receipt.note(
                "judge and improver are not distinct commands: every gate this run reports will "
                "carry a correlated-evaluator failure reason, because a self-graded preference is "
                "not independent evidence"
            )

        open_candidates = state.open_candidates()
        if len(open_candidates) >= config.caps.max_open_candidates:
            receipt.stand_down(
                f"backlog guard: {len(open_candidates)} candidate(s) already awaiting review, "
                f"cap is {config.caps.max_open_candidates}. Nothing was discarded; review or "
                "close the queue and the next run resumes."
            )
            return _finish(
                receipt, state, skill_store, digest_before, clock, dry_run, run_day, {}, []
            )

        threads = _discover(config, source, receipt)
        if not threads:
            receipt.stand_down(
                "discovery returned no conversations for the configured window "
                f"(since={config.source.since!r})"
            )
            return _finish(
                receipt, state, skill_store, digest_before, clock, dry_run, run_day, {}, []
            )

        already = state.processed_record_ids(run_day)
        deferred_ids = set(state.deferred_ids())
        fresh = [t for t in threads if t.thread_id not in already]
        receipt.discovery["already_processed_today"] = len(threads) - len(fresh)

        bucket_of = _bucket_of(
            config.selection.fairness_dimension,
            agent_title_markers=config.selection.agent_title_markers,
        )
        selected, report = select_fair(
            fresh,
            limit=config.selection.max_conversations_per_run,
            bucket_of=bucket_of,
            rank_of=lambda t: (0 if t.thread_id in deferred_ids else 1, t.thread_id),
        )
        receipt.selection = {
            **report.to_dict(),
            "fairness_dimension": config.selection.fairness_dimension,
            "bucket_coverage": round(bucket_coverage(report), 4),
            "method": "round-robin across the fairness dimension, best-first inside a bucket; "
            "never a sorted prefix",
            "retry_first_deferred": sorted(deferred_ids & {t.thread_id for t in selected}),
            "class_histogram_selected": (
                class_histogram((bucket_of(t) for t in selected), declared=DISCOVERY_CLASSES)
                if config.selection.fairness_dimension == "class"
                else {}
            ),
        }
        if not selected:
            receipt.stand_down("every discovered conversation was already processed for this day")
            return _finish(
                receipt, state, skill_store, digest_before, clock, dry_run, run_day, {}, []
            )

        conversations: dict[str, NormalizedConversation] = {}
        failures: list[FailureCase] = []
        holdout_pairs: list[tuple[str, str]] = []

        for thread in selected:
            record_id = thread.thread_id
            try:
                conversation = source.fetch(record_id)
            except RecordDefect as exc:
                state.quarantine(record_id, stage=STAGE_INGEST, reason=str(exc))
                receipt.record(record_id, status="quarantined", stage=STAGE_INGEST, reason=str(exc))
                continue
            except TransientFailure as exc:
                state.defer(record_id, stage=STAGE_INGEST, reason=str(exc))
                receipt.record(record_id, status="deferred", stage=STAGE_INGEST, reason=str(exc))
                continue
            state.write_conversation(record_id, conversation.to_dict())
            conversations[record_id] = conversation
            authored, class_reasons = classify_conversation(
                conversation, agent_title_markers=config.selection.agent_title_markers
            )
            receipt.classes[record_id] = {
                "authorship_class": authored,
                "reasons": list(class_reasons),
            }
            if authored == CLASS_UNKNOWN:
                receipt.note(
                    f"conversation {record_id} could not be classified as human or "
                    "orchestrated; it was kept and scored, and its class is recorded as unknown "
                    "rather than guessed"
                )

            record_failed = False
            for rubric in config.rubrics:
                try:
                    result = score_and_isolate(
                        conversation,
                        rubric,
                        runner,
                        cfg=config.prompt,
                        seed=run_seed,
                        known_skill_ids=known_skill_ids,
                    )
                except RecordDefect as exc:
                    state.quarantine(record_id, stage=f"{STAGE_SCORE}:{rubric.id}", reason=str(exc))
                    receipt.record(
                        record_id,
                        status="quarantined",
                        stage=f"{STAGE_SCORE}:{rubric.id}",
                        reason=str(exc),
                    )
                    record_failed = True
                    break
                except TransientFailure as exc:
                    state.defer(record_id, stage=f"{STAGE_SCORE}:{rubric.id}", reason=str(exc))
                    receipt.record(
                        record_id,
                        status="deferred",
                        stage=f"{STAGE_SCORE}:{rubric.id}",
                        reason=str(exc),
                    )
                    record_failed = True
                    break
                state.write_scorecard(record_id, rubric.id, result.scorecard.to_dict())
                state.write_failures(record_id, rubric.id, [f.to_dict() for f in result.failures])
                failures.extend(result.failures)
                holdout_pairs.extend(result.passes)
            if record_failed:
                continue
            state.clear_deferred(record_id)
            receipt.record(record_id, status="scored", stage=STAGE_SCORE)

        candidates = _propose_and_validate(
            config,
            runner=runner,
            state=state,
            receipt=receipt,
            groups=groups,
            failures=failures,
            conversations=conversations,
            rubrics=rubrics,
            holdout_pairs=holdout_pairs,
            known_skill_ids=known_skill_ids,
            skill_store=skill_store,
            run_day=run_day,
            run_seed=run_seed,
            dry_run=dry_run,
        )
        return _finish(
            receipt,
            state,
            skill_store,
            digest_before,
            clock,
            dry_run,
            run_day,
            conversations,
            candidates,
        )


def _discover(config: Config, source: Any, receipt: RunReceipt) -> list[Any]:
    """Discover one bounded page per configured workspace, and stay honest about it.

    A page that came back exactly full is not a total, and the receipt says so
    rather than printing a count that reads like coverage.
    """
    targets: Sequence[str | None] = config.source.workspaces or (None,)
    limit = config.source.discovery_limit
    pages = []
    for workspace in targets:
        try:
            if hasattr(source, "discover_page"):
                pages.append(
                    source.discover_page(
                        since=config.source.since,
                        limit=limit,
                        workspace=workspace,
                        query=config.source.query or None,
                    )
                )
            else:  # pragma: no cover - legacy source without paging
                from ..domain.discovery import page_from_threads

                pages.append(
                    page_from_threads(
                        source.discover(
                            since=config.source.since,
                            limit=limit,
                            workspace=workspace,
                            query=config.source.query or None,
                        ),
                        limit=limit,
                        since=config.source.since,
                        workspace=workspace or "",
                        query=config.source.query or "",
                    )
                )
        except TransientFailure as exc:
            pages.append(
                error_page(
                    limit=limit,
                    since=config.source.since,
                    workspace=workspace or "",
                    query=config.source.query or "",
                    error=str(exc),
                )
            )
    found, summary = merge_pages(pages)
    classes = [
        classify_thread(t, agent_title_markers=config.selection.agent_title_markers) for t in found
    ]
    receipt.discovery = {
        "workspaces_queried": [w or "<all>" for w in targets],
        "threads_returned": len(found),
        "is_total": summary["is_total"],
        "any_page_truncated_by_limit": summary["any_page_truncated_by_limit"],
        "pages": summary["pages"],
        "errors": summary["errors"],
        "since": config.source.since,
        "limit": limit,
        "class_histogram_discovered": class_histogram(classes, declared=DISCOVERY_CLASSES),
        "class_label_meaning": (
            "discovery can only split agent-titled from not-agent-titled: a title cannot prove "
            "human authorship. The three-way human/orchestrated/unknown class is decided after "
            "the transcript is fetched and is reported under class_counts_scored"
        ),
        "count_meaning": (
            "threads_returned is a PAGE COUNT, not a population size"
            if not summary["is_total"]
            else "threads_returned is the whole configured window"
        ),
    }
    for page in summary["pages"]:
        if page["truncated_by_limit"] or page["error"]:
            receipt.note(page["honesty_note"])
    for error in summary["errors"]:
        receipt.note(f"discovery error: {error}")
    return found


def _propose_and_validate(
    config: Config,
    *,
    runner: ModelRunner,
    state: StateStore,
    receipt: RunReceipt,
    groups: Sequence[SkillGroup],
    failures: Sequence[FailureCase],
    conversations: Mapping[str, NormalizedConversation],
    rubrics: Mapping[str, Rubric],
    holdout_pairs: Sequence[tuple[str, str]],
    known_skill_ids: Sequence[str],
    skill_store: Any,
    run_day: str,
    run_seed: str,
    dry_run: bool,
) -> list[Any]:
    by_skill: dict[str, list[FailureCase]] = {}
    unattributed = 0
    for case in failures:
        skill_id = case.primary_skill_id
        if not skill_id:
            unattributed += 1
            continue
        by_skill.setdefault(skill_id, []).append(case)
    receipt.counts["failure_cases"] = len(failures)
    receipt.counts["failure_cases_unattributed"] = unattributed
    receipt.counts["skills_implicated"] = len(by_skill)
    if unattributed:
        receipt.note(
            f"{unattributed} failure case(s) could not be attributed to a skill and were left "
            "alone rather than blamed on the nearest plausible one"
        )
    if not by_skill:
        return []

    items = sorted(by_skill.items())
    selected, report = select_fair(
        items,
        limit=min(config.caps.max_skills_per_run, config.caps.max_candidates_per_run),
        bucket_of=lambda item: sorted({c.rubric_id for c in item[1]})[0],
        rank_of=lambda item: (-sum(c.weight for c in item[1]), item[0]),
    )
    receipt.counts["skills_selected_for_proposal"] = len(selected)
    receipt.notes.append(
        "skill selection for proposals is round-robin across rubrics, heaviest failure weight "
        f"first inside a rubric: {report.to_dict()}"
    )

    produced: list[dict[str, Any]] = []
    seq = 0
    for skill_id, cases in selected:
        group = group_for(groups, skill_id)
        if group is None:
            state.quarantine(
                skill_id,
                stage=STAGE_PROPOSE,
                reason=f"skill {skill_id!r} was implicated but is not present in any configured root",
            )
            receipt.record(
                skill_id, status="quarantined", stage=STAGE_PROPOSE, reason="skill not found"
            )
            continue
        seq += 1
        candidate_id = f"CAND-{run_day}-{seq:02d}-{safe_name(skill_id)}"
        while state.candidate_path(candidate_id).exists():
            seq += 1
            candidate_id = f"CAND-{run_day}-{seq:02d}-{safe_name(skill_id)}"
        try:
            candidate = propose_candidate(
                group,
                cases,
                runner,
                conversations=conversations,
                cfg=config.prompt,
                seed=run_seed,
                candidate_id=candidate_id,
                created_at=iso(_now(state)),
                protected_skill_ids=config.protected_skill_ids,
                max_cases=config.caps.max_failures_per_candidate,
            )
        except SafetyRefusal as exc:
            # A deliberate policy refusal is NOT a defect, and it must not land in
            # the quarantine count. A counter that sums "this record is broken" with
            # "the guard did its job" is high in both the healthy and the broken
            # case, so it cannot inform any decision.
            receipt.note(f"proposal for {skill_id!r} refused on purpose: {exc}")
            state.refuse(skill_id, stage=STAGE_PROPOSE, reason=str(exc))
            receipt.record(skill_id, status="refused", stage=STAGE_PROPOSE, reason=str(exc))
            continue
        except RecordDefect as exc:
            state.quarantine(skill_id, stage=STAGE_PROPOSE, reason=str(exc))
            receipt.record(skill_id, status="quarantined", stage=STAGE_PROPOSE, reason=str(exc))
            continue
        except TransientFailure as exc:
            state.defer(skill_id, stage=STAGE_PROPOSE, reason=str(exc))
            receipt.record(skill_id, status="deferred", stage=STAGE_PROPOSE, reason=str(exc))
            continue

        base_text = skill_store.read_text(candidate.target.canonical_path)
        targets = {(r.conversation_id, r.criterion_id) for r in candidate.rubric_failures}
        holdouts = [p for p in dict.fromkeys(holdout_pairs) if p not in targets]
        try:
            validation = validate_candidate(
                candidate,
                runner,
                rubrics=rubrics,
                conversations=conversations,
                base_text=base_text,
                cfg=config.prompt,
                policy=config.gate,
                seed=f"{run_seed}:{candidate.id}",
                holdout_pairs=holdouts,
                max_extra_holdout_conversations=config.caps.max_extra_holdout_conversations,
                known_skill_ids=known_skill_ids,
                judge_independent=config.judge_is_independent(),
            )
        except AsiError as exc:
            state.defer(skill_id, stage=STAGE_VALIDATE, reason=str(exc))
            receipt.record(skill_id, status="deferred", stage=STAGE_VALIDATE, reason=str(exc))
            continue

        candidate = replace(
            candidate,
            status=STATUS_VALIDATED if validation.decision.passed else STATUS_GATE_FAILED,
            validation=validation.to_dict(),
        )
        entry = {
            "candidate_id": candidate.id,
            "skill_id": candidate.target.skill_id,
            "canonical_path": candidate.target.canonical_path,
            "base_sha256": candidate.target.base_sha256,
            "status": candidate.status,
            "gate_passed": validation.decision.passed,
            "gate_reasons": list(validation.decision.reasons),
            "gate_metrics": validation.decision.metrics,
            "added_lines": candidate.patch.added_lines,
            "removed_lines": candidate.patch.removed_lines,
            "addressed_failure_case_ids": [r.failure_case_id for r in candidate.rubric_failures],
            "validation_notes": list(validation.notes),
            "persisted": not dry_run,
        }
        if dry_run:
            entry["diff_preview"] = candidate.patch.unified_diff[:4000]
            receipt.note(
                f"dry run: candidate {candidate.id} was NOT written to the candidate queue, so it "
                "cannot be reviewed or applied. Re-run with --commit to persist it."
            )
        else:
            state.write_candidate(candidate)
        produced.append(entry)
    return produced


def _now(state: StateStore) -> datetime:
    from ..clock import SystemClock

    clock = getattr(state, "_clock", None)
    return (clock or SystemClock()).now()


def _finish(
    receipt: RunReceipt,
    state: StateStore,
    skill_store: Any,
    digest_before: str,
    clock: Clock,
    dry_run: bool,
    run_day: str,
    conversations: Mapping[str, NormalizedConversation],
    candidates: Sequence[dict[str, Any]],
) -> RunReceipt:
    digest_after = skill_store.tree_digest()
    receipt.candidates = list(candidates)
    receipt.skill_tree["digest_after"] = digest_after
    receipt.skill_tree["mutation_detected"] = digest_after != digest_before
    receipt.counts["conversations_ingested"] = len(conversations)
    receipt.counts["state_quarantine_files"] = state.count("quarantine")
    receipt.counts["state_refusal_files"] = state.count("refusals")
    receipt.counts["state_deferred_files"] = state.count("deferred")
    receipt.counts["open_candidates_after"] = len(state.open_candidates())
    receipt.finalize(finished_at=iso(clock.now()), mutation_detected=digest_after != digest_before)

    if not dry_run:
        checkpoint = state.read_checkpoint(run_day)
        checkpoint["runs"] = int(checkpoint.get("runs", 0)) + 1
        records = dict(checkpoint.get("records", {}))
        records.update(receipt.records)
        checkpoint["records"] = records
        checkpoint["candidates"] = sorted(
            set(checkpoint.get("candidates", [])) | {c["candidate_id"] for c in candidates}
        )
        checkpoint["completed"] = receipt.status != STATUS_STOOD_DOWN
        checkpoint["last_receipt"] = str(state.receipt_path(run_day, dry_run=False))
        state.write_checkpoint(run_day, checkpoint)
    state.write_receipt(run_day, receipt.to_dict(), dry_run=dry_run)
    return receipt
