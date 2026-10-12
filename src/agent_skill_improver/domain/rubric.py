"""Rubric definitions and the three-way verdict.

PASS / FAIL / INSUFFICIENT are kept strictly distinct and never collapsed:

* PASS -- the criterion is met, on evidence.
* FAIL -- the criterion is violated, on evidence.
* INSUFFICIENT -- the conversation does not contain enough information to
  decide, or the scorer failed to cite resolvable evidence.

INSUFFICIENT is *not* a soft PASS and *not* a soft FAIL. It never produces a
failure case, never counts toward an improvement delta, and never gates a
promotion. Collapsing it in either direction is how a loop starts inventing
work or hiding real defects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .redaction import quoted_value


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT = "INSUFFICIENT"

    @staticmethod
    def parse(value: Any) -> "Verdict":
        if isinstance(value, Verdict):
            return value
        if not isinstance(value, str):
            raise ValueError(f"verdict must be a string, got {type(value).__name__}")
        norm = value.strip().upper()
        for member in Verdict:
            if member.value == norm:
                return member
        raise ValueError(
            f"unknown verdict {quoted_value(value)}; expected PASS, FAIL or INSUFFICIENT"
        )

    @property
    def is_decided(self) -> bool:
        return self in (Verdict.PASS, Verdict.FAIL)


@dataclass(frozen=True)
class Criterion:
    """One scoreable question about a conversation."""

    id: str
    title: str
    question: str
    weight: int = 1
    requires_evidence: bool = True
    insufficient_when: str = ""
    # Skill ids this criterion is *expected* to implicate when it fails. Used
    # by attribution as a prior, never as the sole reason to blame a skill.
    implicates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "question": self.question,
            "weight": self.weight,
            "requires_evidence": self.requires_evidence,
            "insufficient_when": self.insufficient_when,
            "implicates": list(self.implicates),
        }


@dataclass(frozen=True)
class Rubric:
    """A named, versioned set of criteria."""

    id: str
    version: int
    title: str
    criteria: tuple[Criterion, ...]
    min_evidence_refs: int = 1
    # Hash of the rubric's own source text, recorded on every scorecard so a
    # rubric edit can never be mistaken for a behaviour change.
    source_sha256: str = ""

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for c in self.criteria:
            if c.id in seen:
                raise ValueError(f"duplicate criterion id in rubric {self.id}: {c.id}")
            seen.add(c.id)

    @property
    def criterion_ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.criteria)

    def criterion(self, criterion_id: str) -> Criterion | None:
        for c in self.criteria:
            if c.id == criterion_id:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "title": self.title,
            "min_evidence_refs": self.min_evidence_refs,
            "source_sha256": self.source_sha256,
            "criteria": [c.to_dict() for c in self.criteria],
        }

    @staticmethod
    def from_mapping(raw: Mapping[str, Any], *, source_sha256: str = "") -> "Rubric":
        """Build a rubric from parsed TOML.

        Raises ``ValueError`` with a specific reason. Callers translate to
        ``ConfigError``; the domain never imports the app error module.
        """
        meta = raw.get("rubric")
        if not isinstance(meta, Mapping):
            raise ValueError("rubric file needs a [rubric] table")
        rid = meta.get("id")
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError("[rubric].id must be a non-empty string")
        version = meta.get("version", 1)
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError("[rubric].version must be an integer")
        raw_criteria = raw.get("criterion")
        if not isinstance(raw_criteria, list) or not raw_criteria:
            raise ValueError("rubric file needs at least one [[criterion]]")
        criteria: list[Criterion] = []
        for idx, entry in enumerate(raw_criteria):
            if not isinstance(entry, Mapping):
                raise ValueError(f"[[criterion]] #{idx} is not a table")
            cid = entry.get("id")
            question = entry.get("question")
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError(f"[[criterion]] #{idx} needs a non-empty id")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"criterion {cid} needs a non-empty question")
            weight = entry.get("weight", 1)
            if not isinstance(weight, int) or isinstance(weight, bool) or weight < 1:
                raise ValueError(f"criterion {cid} weight must be an integer >= 1")
            implicates = entry.get("implicates", [])
            if not isinstance(implicates, list) or any(not isinstance(x, str) for x in implicates):
                raise ValueError(f"criterion {cid} implicates must be a list of strings")
            criteria.append(
                Criterion(
                    id=cid,
                    title=str(entry.get("title", cid)),
                    question=question,
                    weight=weight,
                    requires_evidence=bool(entry.get("requires_evidence", True)),
                    insufficient_when=str(entry.get("insufficient_when", "")),
                    implicates=tuple(implicates),
                )
            )
        min_refs = meta.get("min_evidence_refs", 1)
        if not isinstance(min_refs, int) or isinstance(min_refs, bool) or min_refs < 0:
            raise ValueError("[rubric].min_evidence_refs must be an integer >= 0")
        return Rubric(
            id=rid,
            version=version,
            title=str(meta.get("title", rid)),
            criteria=tuple(criteria),
            min_evidence_refs=min_refs,
            source_sha256=source_sha256,
        )
