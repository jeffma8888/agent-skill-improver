"""Run receipt: the honest record of what one run actually did.

Three rules the receipt has to keep, because they are the ones that go wrong.

* Report the status the run *earned*, not the status it hoped for. A run that
  deferred half its records is ``partial``, and a run that stood down is
  ``stood-down`` -- not ``ok`` with a note buried at the end.
* Counters carry their sample moment, because a later reader who differences two
  receipts otherwise computes a false rate.
* Report the skill-tree digest before and after. "The daily loop never mutates a
  live skill" is then a measured property of every run, not a claim in a README.
* Report the source classification and the isolation mode the model roles ran
  under. Without it, a receipt cannot distinguish a synthetic-fixture run from a
  private run, which is the difference that decides whether the run was allowed
  to happen at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import SCHEMA_RECEIPT

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_STOOD_DOWN = "stood-down"
STATUS_BLOCKED = "blocked"


@dataclass
class RunReceipt:
    run_id: str
    day: str
    started_at: str
    dry_run: bool
    config_path: str
    seed: str
    finished_at: str = ""
    status: str = STATUS_OK
    stand_down_reason: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    discovery: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    skill_tree: dict[str, Any] = field(default_factory=dict)
    caps: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    state_dir: str = ""
    #: Per-record authorship class with the reasons it was decided.
    classes: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Which configured model roles are actually distinct commands.
    evaluator_independence: dict[str, Any] = field(default_factory=dict)
    #: The source classification and the confinement the model roles ran under.
    #: Recorded on every run, including runs that stood down, because "was this
    #: private data handled inside a no-egress sandbox?" is not answerable from
    #: anything else in the receipt.
    isolation: dict[str, Any] = field(default_factory=dict)

    # -- derived ---------------------------------------------------------
    def note(self, text: str) -> None:
        self.notes.append(text)

    def record(self, record_id: str, *, status: str, stage: str, reason: str = "") -> None:
        self.records[record_id] = {"status": status, "stage": stage, "reason": reason}

    def count_of(self, status: str) -> int:
        return sum(1 for info in self.records.values() if info.get("status") == status)

    def class_counts(self) -> dict[str, int]:
        from ..domain.authorship import class_histogram

        return class_histogram(
            str(info.get("authorship_class", "")) for info in self.classes.values()
        )

    def finalize(self, *, finished_at: str, mutation_detected: bool) -> "RunReceipt":
        self.finished_at = finished_at
        quarantined = self.count_of("quarantined")
        deferred = self.count_of("deferred")
        scored = self.count_of("scored")
        self.counts.update(
            {
                "records_scored": scored,
                "records_quarantined": quarantined,
                "records_deferred": deferred,
                "proposals_refused_by_policy": self.count_of("refused"),
                "candidates_proposed": len(self.candidates),
                "candidates_gate_passed": sum(1 for c in self.candidates if c.get("gate_passed")),
                "candidates_gate_failed": sum(
                    1 for c in self.candidates if c.get("gate_passed") is False
                ),
            }
        )
        if mutation_detected:
            self.status = STATUS_BLOCKED
            self.note(
                "LIVE SKILL MUTATION DETECTED: the skill tree digest changed during this run. "
                "The daily loop never writes skills, so either another process edited a skill "
                "file while the run was in flight, or an invariant of this tool is broken. "
                "Investigate before trusting any candidate from this run."
            )
        elif self.status == STATUS_STOOD_DOWN:
            pass
        elif quarantined or deferred:
            self.status = STATUS_PARTIAL
        elif scored == 0:
            self.status = STATUS_STOOD_DOWN
            if not self.stand_down_reason:
                self.stand_down_reason = "no records were scored"
        else:
            self.status = STATUS_OK
        return self

    def stand_down(self, reason: str) -> "RunReceipt":
        self.status = STATUS_STOOD_DOWN
        self.stand_down_reason = reason
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_RECEIPT,
            "run_id": self.run_id,
            "day": self.day,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "sampled_at": self.finished_at or self.started_at,
            "status": self.status,
            "stand_down_reason": self.stand_down_reason,
            "dry_run": self.dry_run,
            "config_path": self.config_path,
            "state_dir": self.state_dir,
            "seed": self.seed,
            "source": dict(self.source),
            "discovery": dict(self.discovery),
            "selection": dict(self.selection),
            "records": dict(self.records),
            "counts": dict(self.counts),
            "candidates": list(self.candidates),
            "skill_tree": dict(self.skill_tree),
            "caps": dict(self.caps),
            "classes": dict(self.classes),
            "class_counts_scored": self.class_counts(),
            "evaluator_independence": dict(self.evaluator_independence),
            "isolation": dict(self.isolation),
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        bits = [
            f"status={self.status}",
            f"day={self.day}",
            ("mode=dry-run" if self.dry_run else "mode=commit"),
            f"scored={self.counts.get('records_scored', 0)}",
            f"quarantined={self.counts.get('records_quarantined', 0)}",
            f"deferred={self.counts.get('records_deferred', 0)}",
            f"candidates={self.counts.get('candidates_proposed', 0)}",
            f"gate_passed={self.counts.get('candidates_gate_passed', 0)}",
            f"live_mutation={self.skill_tree.get('mutation_detected')}",
            f"isolation={self.isolation.get('mode', 'unrecorded')}",
        ]
        if self.stand_down_reason:
            bits.append(f"reason={self.stand_down_reason!r}")
        return " ".join(bits)
