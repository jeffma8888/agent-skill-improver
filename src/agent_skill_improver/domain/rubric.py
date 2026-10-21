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

import difflib
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
        _refuse_unknown_keys(raw, RUBRIC_FILE_KEYS, where="the rubric file")
        meta = raw.get("rubric")
        if not isinstance(meta, Mapping):
            raise ValueError("rubric file needs a [rubric] table")
        _refuse_unknown_keys(meta, RUBRIC_TABLE_KEYS, where="[rubric]")
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
            _refuse_unknown_keys(entry, CRITERION_KEYS, where=f"[[criterion]] #{idx}")
            cid = entry.get("id")
            question = entry.get("question")
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError(f"[[criterion]] #{idx} needs a non-empty id")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"criterion {cid} needs a non-empty question")
            weight = entry.get("weight", 1)
            if not isinstance(weight, int) or isinstance(weight, bool) or weight < 1:
                raise ValueError(f"criterion {cid} weight must be an integer >= 1")
            requires_evidence = entry.get("requires_evidence", True)
            if not isinstance(requires_evidence, bool):
                raise ValueError(
                    f"criterion {cid} requires_evidence must be true or false, "
                    f"got {type(requires_evidence).__name__}"
                )
            implicates = entry.get("implicates", [])
            if not isinstance(implicates, list) or any(not isinstance(x, str) for x in implicates):
                raise ValueError(f"criterion {cid} implicates must be a list of strings")
            if any(not skill_id.strip() for skill_id in implicates):
                raise ValueError(f"criterion {cid} implicates holds an empty skill id")
            repeated = sorted(
                {skill_id for skill_id in implicates if implicates.count(skill_id) > 1}
            )
            if repeated:
                # Measured before: a repeated entry attributed one failure to the
                # same skill twice, both persisted and shown to the improver.
                raise ValueError(
                    f"criterion {cid} implicates lists "
                    f"{', '.join(repr(skill_id) for skill_id in repeated)} more than once"
                )
            criteria.append(
                Criterion(
                    id=cid,
                    title=_text_field(entry, "title", cid, where=f"criterion {cid} "),
                    question=question,
                    weight=weight,
                    requires_evidence=requires_evidence,
                    insufficient_when=_text_field(
                        entry, "insufficient_when", "", where=f"criterion {cid} "
                    ),
                    implicates=tuple(implicates),
                )
            )
        min_refs = meta.get("min_evidence_refs", 1)
        if not isinstance(min_refs, int) or isinstance(min_refs, bool) or min_refs < 0:
            raise ValueError("[rubric].min_evidence_refs must be an integer >= 0")
        return Rubric(
            id=rid,
            version=version,
            title=_text_field(meta, "title", rid, where="[rubric]."),
            criteria=tuple(criteria),
            min_evidence_refs=min_refs,
            source_sha256=source_sha256,
        )


#: The keys a rubric file may hold: at the top, in ``[rubric]`` and in each
#: ``[[criterion]]``. Any other key is refused rather than ignored, naming the
#: nearest one when one is close and listing them all. Measured before: a misspelt ``insufficient_when`` loaded and
#: was absent from every scorer prompt, and a misspelt ``requires_evidence``
#: left the default in force, each without a word.
RUBRIC_FILE_KEYS = frozenset({"rubric", "criterion"})
RUBRIC_TABLE_KEYS = frozenset({"id", "version", "title", "min_evidence_refs"})
CRITERION_KEYS = frozenset(
    {"id", "title", "question", "weight", "requires_evidence", "insufficient_when", "implicates"}
)


def _refuse_unknown_keys(table: Mapping[str, Any], accepted: frozenset[str], *, where: str) -> None:
    """Refuse a key the rubric loader would not read, naming the nearest key it would.

    The nearest key is named only when one is close (``difflib`` at 0.6); every
    accepted key is listed either way. Each refused key is quoted through
    ``quoted_value``, so a trailing space shows and a key as long as the file
    is cut rather than printed whole (measured before: a 100,000-character key
    made 100,345 bytes of ``doctor`` output).
    """
    unknown = sorted(str(key) for key in table if key not in accepted)
    if not unknown:
        return
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(accepted), n=1, cutoff=0.6)
        shown = quoted_value(key)
        hints.append(f"{shown} (did you mean {close[0]}?)" if close else shown)
    raise ValueError(
        f"{where} has unknown key(s): {', '.join(hints)}. Accepted keys are "
        f"{', '.join(sorted(accepted))}. An unknown key is refused rather than ignored, "
        "because a misspelt field that loads silently leaves its default in force."
    )


def _text_field(table: Mapping[str, Any], key: str, default: str, *, where: str) -> str:
    """``table[key]`` when it is a string, ``default`` when absent, else refused.

    Measured before: any value was passed through ``str()``, so a list or a
    table became its Python spelling in the scorer prompt.
    """
    value = table.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{where}{key} must be a string, got {type(value).__name__}")
    return value
