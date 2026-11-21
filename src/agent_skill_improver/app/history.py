"""One skill's candidate records read back: ``asi history SKILL_ID``.

Pure reading. Every state reader here is called with ``create=False``
(``StateStore._looked_in``), so an absent tree stays absent and no mode is
reset; the run lock is not taken (taking it opens ``locks/daily.lock`` with
``O_CREAT``), and the payload says so. Nothing here imports a model adapter or
the promotion code: the receipt schemas are spelled out below rather than
imported from ``app/promote.py``.

What a row says about the skill file is what the copy read during this
command's discovery held (``HISTORY_LIVE_BASIS``): the canonical snapshot's
``sha256`` of the bytes discovery read through the pinned walk, used only when
they decoded as strict UTF-8. Nothing is read a second time, and no path a
record names is ever opened, so a change to the file after discovery is not
seen, and the wording never claims the file's present content.

Every value a record or receipt holds is untrusted: nested values pass a type
guard (``_mapping``, ``_text``, ``_flag``, ``_number``), a miss is named in the
row's ``invalid_fields`` and never raises, and text is shown through
``listed_value`` (masked, cut).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..domain.candidate import Candidate
from ..domain.integrity import derived_field_mismatches, validation_mismatches
from ..domain.redaction import LISTED_PATH_CHARS, LISTED_SKILL_ID_CHARS, listed_value
from ..domain.skills import SkillGroup, group_for
from ..errors import StateUnreadable
from .state import (
    CLOSED_STATUSES,
    OPEN_STATUSES,
    SetAsideScan,
    StateStore,
    UnreadableRecord,
    safe_name,
)

HISTORY_SCHEMA = "asi.history.v1"
#: How long a prose value (change, note, reason, problem) may be when shown.
HISTORY_PROSE_CHARS = 200
#: How long an operator label may be when shown.
HISTORY_OPERATOR_CHARS = 200
#: Gate reasons shown per row; the rest are counted.
HISTORY_REASONS_SHOWN = 5
#: Rows shown when ``--limit`` is not given, and the most it may ask for.
HISTORY_DEFAULT_LIMIT = 50
HISTORY_MAX_LIMIT = 1000
#: Integrity problems and invalid field names shown per row; the rest counted.
HISTORY_PROBLEMS_SHOWN = 5
HISTORY_INVALID_SHOWN = 5
#: The fields a decline record and its receipt are compared on (c247b).
HISTORY_DECLINE_FIELDS = ("candidate_id", "operator", "declined_at", "reason", "status_before")
#: What every live hash and live match describes: never the file afterwards.
HISTORY_LIVE_BASIS = "the copy read during this command's discovery"

#: The receipt schemas this view reads, spelled here so it imports no promotion code.
APPLY_RECEIPT_SCHEMA = "asi.apply_receipt.v1"
ROLLBACK_RECEIPT_SCHEMA = "asi.rollback_receipt.v1"
DECLINE_RECEIPT_SCHEMA = "asi.decline_receipt.v1"
#: The only status c247b's ``decline`` declines from: the record side of ``status_before``.
_DECLINED_FROM = "validated"
#: The stages a skill-level set-aside record carries (``app/daily.py``); a
#: conversation's are ``ingest`` and ``score:<rubric>``.
_SKILL_LEVEL_STAGES = ("propose", "validate")
_METRICS = ("fixes", "regressions", "holdout_cases", "judge_candidate_preference_ratio")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _flag(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _guarded(source: Mapping[str, Any], key: str, guard: Any, name: str, invalid: list[str]) -> Any:
    """``guard(source[key])``; a miss (absent, or the wrong type) is named in ``invalid``."""
    value = guard(source.get(key))
    if value is None:
        invalid.append(name)
    return value


def _prose(value: str | None) -> str | None:
    return None if value is None else listed_value(value, limit=HISTORY_PROSE_CHARS)


def _operator(value: str | None) -> str | None:
    return None if value is None else listed_value(value, limit=HISTORY_OPERATOR_CHARS)


def _path(value: str | None) -> str | None:
    return None if value is None else listed_value(value, limit=LISTED_PATH_CHARS)


def _when(value: str | None) -> str | None:
    return None if value is None else listed_value(value)


def _digest_or_invalid(value: object) -> tuple[str | None, str | None]:
    """``(digest, None)`` for 64 lowercase hex characters, else ``(None, masked value)``."""
    if isinstance(value, str) and _DIGEST.fullmatch(value):
        return value, None
    return None, listed_value(value, limit=HISTORY_PROSE_CHARS)


def _unreadable_row(record: UnreadableRecord) -> dict[str, str]:
    """The masked projection of an unreadable record: never its raw path or name."""
    return {
        "path": listed_value(str(record.path), limit=LISTED_PATH_CHARS),
        "name": listed_value(record.path.name, limit=LISTED_PATH_CHARS),
        "reason": listed_value(record.reason, limit=HISTORY_PROSE_CHARS),
    }


def _gate(validation: object, invalid: list[str]) -> dict[str, Any]:
    """The gate verdict and its four scores, each through its guard."""
    out: dict[str, Any] = {
        "ran": False,
        "passed": None,
        **{name: None for name in _METRICS},
        "reasons": [],
        "reasons_not_shown": 0,
    }
    if validation is None:
        return out
    record = _mapping(validation)
    if record is None:
        invalid.append("validation")
        return out
    if "gate" not in record:
        return out
    gate = _mapping(record.get("gate"))
    if gate is None:
        invalid.append("validation.gate")
        return out
    out["ran"] = True
    out["passed"] = _guarded(gate, "passed", _flag, "gate.passed", invalid)
    reasons = gate.get("reasons", [])
    if isinstance(reasons, list):
        texts = [r for r in reasons if isinstance(r, str)]
        if len(texts) != len(reasons):
            invalid.append("gate.reasons")
        out["reasons"] = [_prose(r) for r in texts[:HISTORY_REASONS_SHOWN]]
        out["reasons_not_shown"] = max(0, len(texts) - HISTORY_REASONS_SHOWN)
    else:
        invalid.append("gate.reasons")
    metrics = _mapping(gate.get("metrics", {}))
    if metrics is None:
        invalid.append("gate.metrics")
        return out
    for name in _METRICS:
        if name in metrics:
            out[name] = _guarded(metrics, name, _number, f"gate.metrics.{name}", invalid)
    return out


def _integrity(candidate: Candidate) -> dict[str, Any]:
    """Whether the record recomputes, judged as rollback judges one (rubrics=None)."""
    problems = list(derived_field_mismatches(candidate)) + list(
        validation_mismatches(candidate, rubrics=None)
    )
    return {
        "recomputes": not problems,
        "problems": [_prose(p) for p in problems[:HISTORY_PROBLEMS_SHOWN]],
        "problems_not_shown": max(0, len(problems) - HISTORY_PROBLEMS_SHOWN),
    }


def _read_receipt(state: StateStore, action: str, candidate_id: str) -> tuple[Any, str | None]:
    """``(receipt or None, None)``, or ``(None, masked reason)`` when it does not read."""
    try:
        return state.action_receipt_if_present(action, candidate_id), None
    except StateUnreadable as exc:
        return None, listed_value(str(exc), limit=HISTORY_PROSE_CHARS)


def _schema_problem(
    receipt: Mapping[str, Any], action: str, schema: str, invalid: list[str]
) -> str | None:
    stored = _guarded(receipt, "schema", _text, f"{action}.schema", invalid)
    if stored is not None and stored != schema:
        return f"holds schema {listed_value(stored)}, not {schema}"
    return None


def _receipt(
    state: StateStore, action: str, candidate_id: str, schema: str, invalid: list[str]
) -> dict[str, Any] | None:
    """The apply or rollback receipt of exactly this candidate, every field guarded."""
    receipt, unreadable = _read_receipt(state, action, candidate_id)
    if unreadable is not None:
        return {"unreadable": unreadable}
    if receipt is None:
        return None
    named = _text(receipt.get("candidate_id"))
    if named != candidate_id:
        # Bound to its file name by the reader only; a receipt edited to name
        # another candidate is not shown as this one's.
        return {"unreadable": "names another candidate"}
    wrong = _schema_problem(receipt, action, schema, invalid)
    if wrong is not None:
        return {"unreadable": wrong}
    if action == "apply":
        approval = _mapping(receipt.get("approval"))
        if approval is None:
            invalid.append("apply.approval")
        write = _mapping(receipt.get("write"))
        if write is None:
            invalid.append("apply.write")
        finishes: bool | None = False
        if "finishes_an_earlier_run" in receipt:
            finishes = _guarded(
                receipt, "finishes_an_earlier_run", _flag, "apply.finishes_an_earlier_run", invalid
            )
        note = None
        if approval is not None and "note" in approval:
            note = _guarded(approval, "note", _text, "apply.approval.note", invalid)
        manifest = None
        if write is not None:
            manifest = _guarded(
                write, "backup_manifest_path", _text, "apply.write.backup_manifest_path", invalid
            )
        return {
            "applied_at": _when(
                _guarded(receipt, "applied_at", _text, "apply.applied_at", invalid)
            ),
            "operator": _operator(_guarded(receipt, "operator", _text, "apply.operator", invalid)),
            "operator_verified": False,
            "note": _prose(note),
            "backup_manifest_path": _path(manifest),
            "finishes_an_earlier_run": finishes,
        }
    return {
        "rolled_back_at": _when(
            _guarded(receipt, "rolled_back_at", _text, "rollback.rolled_back_at", invalid)
        ),
        "manifest": _path(_guarded(receipt, "manifest", _text, "rollback.manifest", invalid)),
        # A rollback receipt holds no operator and ``rollback`` takes none.
        "operator": None,
    }


def _shown_decline_value(field: str, value: str | None) -> str | None:
    if field == "operator":
        return _operator(value)
    if field == "declined_at":
        return _when(value)
    return _prose(value)


def _decline(
    record_decline: object, receipt: Any, candidate_id: str, invalid: list[str]
) -> dict[str, Any] | None:
    """The record's decline beside its receipt: the record is the source; each
    of the five agreement fields that differs is named, never resolved."""
    record: dict[str, str | None] | None = None
    if record_decline is not None:
        record = {
            "candidate_id": candidate_id,
            "operator": _text(getattr(record_decline, "operator", None)),
            "declined_at": _text(getattr(record_decline, "declined_at", None)),
            "reason": _text(getattr(record_decline, "reason", None)),
            "status_before": _DECLINED_FROM,
        }
        for name in ("operator", "declined_at", "reason"):
            if record[name] is None:
                invalid.append(f"decline.{name}")
    stored: dict[str, str | None] | None = None
    incompatible: str | None = None
    if receipt is not None:
        # The schema first, as for apply and rollback: a receipt of another
        # schema is never read field by field, so none of its values is shown.
        incompatible = _schema_problem(receipt, "decline_receipt", DECLINE_RECEIPT_SCHEMA, invalid)
    if receipt is not None and incompatible is None and record is None:
        # With no record decline to compare it with, the id it holds is all that
        # ties a receipt to this candidate: one that names none, or another, is
        # not shown as this candidate's decline.
        named = _text(receipt.get("candidate_id"))
        if named is None:
            incompatible = "holds no candidate id"
        elif named != candidate_id:
            incompatible = "names another candidate"
    if incompatible is not None and record is None:
        return {"unreadable": incompatible}
    if receipt is not None and incompatible is None:
        stored = {
            name: _guarded(receipt, name, _text, f"decline_receipt.{name}", invalid)
            for name in HISTORY_DECLINE_FIELDS
        }
    if record is None and stored is None:
        return None
    source: dict[str, str | None] | None
    if record is not None and stored is not None:
        source, origin = record, "record and receipt"
    elif record is not None:
        source, origin = record, "record"
    else:
        source, origin = stored, "receipt"
    disagreements = []
    if record is not None and stored is not None:
        for name in HISTORY_DECLINE_FIELDS:
            if record[name] != stored[name]:
                disagreements.append(
                    {
                        "field": name,
                        "record": _shown_decline_value(name, record[name]),
                        "receipt": _shown_decline_value(name, stored[name]),
                    }
                )
    assert source is not None
    return {
        "declined_at": _when(source["declined_at"]),
        "operator": _operator(source["operator"]),
        "operator_verified": False,
        "reason": _prose(source["reason"]),
        "from": origin,
        "disagreements": disagreements,
        # A receipt beside the record that this asi does not read (another
        # schema): named, never compared and never shown.
        "receipt_unreadable": incompatible,
    }


def _live(groups: Sequence[SkillGroup], skill_id: str) -> dict[str, Any]:
    """What this command's discovery found for the skill. The keys starting
    ``_`` are for ``_live_match`` only and are dropped from the payload."""
    group = group_for(groups, skill_id)
    live: dict[str, Any] = {
        "state": "not-in-roots",
        "canonical_path": None,
        "sha256": None,
        "reason": None,
        "basis": HISTORY_LIVE_BASIS,
        "_path": None,
        "_not_compared": "the skill is not in the configured skill roots now",
    }
    if group is None:
        return live
    canonical = group.canonical
    if canonical is None:
        reason = listed_value(group.canonical_reason, limit=HISTORY_PROSE_CHARS)
        live.update(
            state="no-canonical",
            reason=reason,
            _not_compared=f"the skill has no canonical copy now ({group.canonical_reason})",
        )
        return live
    live.update(
        state="canonical",
        canonical_path=listed_value(str(canonical.path), limit=LISTED_PATH_CHARS),
        _path=str(canonical.path),
        _not_compared=None,
    )
    if canonical.strict_utf8:
        live["sha256"] = canonical.sha256
    else:
        warning = canonical.warnings[0] if canonical.warnings else "the bytes did not decode"
        live["reason"] = listed_value(warning, limit=HISTORY_PROSE_CHARS)
        live["_not_compared"] = (
            f"the canonical copy is not valid UTF-8 ({listed_value(warning, limit=HISTORY_PROSE_CHARS)})"
        )
    return live


def _live_match(
    candidate: Candidate, live: Mapping[str, Any], recomputes: bool
) -> tuple[str, str | None]:
    """Whether the copy read during this command's discovery held this record's
    base or its replacement; ``not-compared`` with the reason otherwise."""
    if not recomputes:
        return "not-compared", "the record does not recompute"
    base, _ = _digest_or_invalid(candidate.target.base_sha256)
    if base is None:
        return "not-compared", "the record's base is not a sha256 digest"
    if live["_path"] is None:
        return "not-compared", listed_value(live["_not_compared"], limit=HISTORY_PROSE_CHARS)
    if candidate.target.canonical_path != live["_path"]:
        # The record's own path is never opened: only the copy the store
        # enumerated, and read once, is compared.
        named = listed_value(candidate.target.canonical_path, limit=HISTORY_PROSE_CHARS)
        return "not-compared", listed_value(
            f"the record names {named}, which is not this skill's canonical copy now",
            limit=HISTORY_PROSE_CHARS,
        )
    if live["sha256"] is None:
        return "not-compared", listed_value(live["_not_compared"], limit=HISTORY_PROSE_CHARS)
    held = live["sha256"]
    replacement = candidate.recomputed_patch_sha256
    if held == base:
        return "base", None
    if held == replacement:
        return "replacement", None
    return "neither", None


def _row(candidate: Candidate, state: StateStore, live: Mapping[str, Any]) -> dict[str, Any]:
    invalid: list[str] = []
    base, base_invalid = _digest_or_invalid(candidate.target.base_sha256)
    gate = _gate(candidate.validation, invalid)
    integrity = _integrity(candidate)
    apply = _receipt(state, "apply", candidate.id, APPLY_RECEIPT_SCHEMA, invalid)
    rollback = _receipt(state, "rollback", candidate.id, ROLLBACK_RECEIPT_SCHEMA, invalid)
    decline_receipt, decline_unreadable = _read_receipt(state, "decline", candidate.id)
    if decline_unreadable is not None:
        decline: dict[str, Any] | None = {"unreadable": decline_unreadable}
    else:
        decline = _decline(candidate.decline, decline_receipt, candidate.id, invalid)
    manifest = state.latest_backup_manifest(candidate.id, create=False)
    match, reason = _live_match(candidate, live, integrity["recomputes"])
    names = list(dict.fromkeys(invalid))
    return {
        "id": candidate.id,
        "created_at": listed_value(candidate.created_at),
        "status": listed_value(candidate.status),
        "status_known": candidate.status in OPEN_STATUSES + CLOSED_STATUSES,
        "open": candidate.status in OPEN_STATUSES,
        "canonical_path": listed_value(candidate.target.canonical_path, limit=LISTED_PATH_CHARS),
        "base_sha256": base,
        "base_sha256_invalid": base_invalid,
        "replacement_sha256": candidate.recomputed_patch_sha256,
        "added_lines": candidate.patch.added_lines,
        "removed_lines": candidate.patch.removed_lines,
        "intended_change": listed_value(candidate.intended_change, limit=HISTORY_PROSE_CHARS),
        "gate": gate,
        "integrity": integrity,
        "invalid_fields": names[:HISTORY_INVALID_SHOWN],
        "invalid_not_shown": max(0, len(names) - HISTORY_INVALID_SHOWN),
        "apply": apply,
        "rollback": rollback,
        "decline": decline,
        "backup_manifest": (
            None if manifest is None else listed_value(str(manifest), limit=LISTED_PATH_CHARS)
        ),
        "live_match": match,
        "live_match_reason": reason,
        "live_match_basis": HISTORY_LIVE_BASIS,
    }


def _skill_level_set_aside(scan: SetAsideScan, skill_id: str) -> list[dict[str, Any]]:
    """The skill's own set-aside records (propose, validate), newest first as scanned."""
    rows = []
    for record in scan.readable:
        if record.record_id != skill_id:
            continue
        if (
            not isinstance(record.stage, str)
            or record.stage.split(":")[0] not in _SKILL_LEVEL_STAGES
        ):
            continue
        rows.append(
            {
                "kind": record.kind,
                "stage": listed_value(record.stage, limit=HISTORY_PROSE_CHARS),
                "reason": listed_value(record.reason, limit=HISTORY_PROSE_CHARS),
                "at": listed_value(record.at),
                "path": listed_value(str(record.path), limit=LISTED_PATH_CHARS),
            }
        )
    return rows


def skill_history(
    state: StateStore,
    groups: Sequence[SkillGroup],
    skill_id: str,
    *,
    read_at: str,
    limit: int,
) -> dict[str, Any]:
    """The ``asi.history.v1`` payload for one skill; reads, never writes."""
    scan = state.scan_candidates(create=False)
    set_aside_scan = state.scan_set_aside(create=False)
    live = _live(groups, skill_id)
    mine = sorted(
        (c for c in scan.readable if c.target.skill_id == skill_id),
        key=lambda c: (c.created_at, c.id),
    )
    rows = [_row(c, state, live) for c in mine]
    shown = rows[-limit:] if limit > 0 else []
    suffix = f"-{safe_name(skill_id)}.json"
    named = sorted(
        (u for u in scan.unreadable if u.path.name.endswith(suffix)), key=lambda u: u.path.name
    )
    unreadable = [_unreadable_row(u) for u in named[-limit:]]
    set_aside = _skill_level_set_aside(set_aside_scan, skill_id)
    trusted = [r for r in rows if r["integrity"]["recomputes"]]
    counts = {
        "candidates": len(rows),
        "open": sum(1 for r in trusted if r["open"]),
        "closed": sum(1 for r in trusted if r["status_known"] and not r["open"]),
        "unknown_status": sum(1 for r in trusted if not r["status_known"]),
        "untrusted": len(rows) - len(trusted),
        "unreadable": len(named),
        "set_aside": len(set_aside),
    }
    return {
        "schema": HISTORY_SCHEMA,
        "skill_id": listed_value(skill_id, limit=LISTED_SKILL_ID_CHARS),
        "read_at": read_at,
        "read_without_run_lock": True,
        "state_dir": "present" if state.root.is_dir() else "absent",
        "live": {k: v for k, v in live.items() if not k.startswith("_")},
        "limit": limit,
        "earlier_not_shown": len(rows) - len(shown),
        "candidates": shown,
        "unreadable": unreadable,
        "unreadable_not_shown": len(named) - len(unreadable),
        "unattributed_unreadable": len(scan.unreadable) - len(named),
        "set_aside": set_aside[:limit],
        "set_aside_not_shown": max(0, len(set_aside) - limit),
        "counts": counts,
    }
