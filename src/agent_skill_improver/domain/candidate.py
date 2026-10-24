"""Candidate proposals: the only artifact the automated loop is allowed to emit.

A candidate is a *complete argument for a change*, not a diff. It must carry:

* a hypothesis (why this change would fix the observed failures);
* the exact rubric failures it claims to address;
* the source conversation and message references behind those failures;
* the target skill, its canonical path, and the base hash it was written against;
* the intended change in words;
* risks and a rollback plan.

Any of those missing means the candidate is refused at construction time. The
reason is unglamorous and load-bearing: six weeks later the only thing that
makes a proposal reviewable is the argument, and a diff with no hypothesis
cannot be evaluated by anyone, human or machine.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .. import SCHEMA_CANDIDATE, SCHEMA_PROPOSAL
from ..hashing import sha256_text
from .conversation import EvidenceRef
from .failure import FailureCase
from .redaction import kept_prose, quoted_value

STATUS_PROPOSED = "proposed"
STATUS_VALIDATED = "validated"
STATUS_GATE_FAILED = "gate-failed"
STATUS_APPLIED = "applied"
STATUS_ROLLED_BACK = "rolled-back"
#: Closed by the daily driver because the skill tree moved during the run that
#: produced the candidate. Independent of the gate verdict, which was computed
#: against a tree that then changed.
STATUS_BLOCKED_RUN = "blocked-run"

CHALLENGE_DOMAIN = "asi-challenge:v1"
APPROVAL_DOMAIN = "asi-approve:v1"
CHALLENGE_LEN = 24
TOKEN_LEN = 16
CANDIDATE_ID_RE = re.compile(r"CAND-[A-Za-z0-9][A-Za-z0-9._-]{0,198}\Z")


def validate_candidate_id(candidate_id: str) -> str:
    """Require a lossless path component that is also safe in displayed argv."""
    if not CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise ValueError(
            "candidate id must start with 'CAND-' and contain only ASCII letters, digits, "
            "dot, underscore or hyphen (maximum 204 characters)"
        )
    return candidate_id


#: The run day sits in the middle of every candidate id (``CAND-{day}-{seq}-{skill}``),
#: so it is held to the id's own alphabet and bounded. Whether a day and a skill
#: name fit the id TOGETHER is checked by the daily pass against its real skill tree.
RUN_DAY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
RUN_DAY_MAX_CHARS = 64


def validate_run_day(day: str) -> str:
    """Require a run day that can name a candidate and a checkpoint.

    Checked before a pass reads anything, because the day is only used at the
    end. Measured before this check: ``--day 2026/01/02`` was accepted on the
    command line, every conversation was ingested and scored (the model calls
    paid for, the conversations and scorecards written under ``--commit``),
    and the pass died building its first candidate id with ``StateUnreadable:
    invalid candidate id``, a message about the wrong thing. No receipt or
    checkpoint was written, so the next run with the same day repeated all of
    it.
    """
    if not RUN_DAY_RE.fullmatch(day):
        hint = ""
        if "/" in day and RUN_DAY_RE.fullmatch(day.replace("/", "-")):
            hint = f"; a date is spelled with hyphens, {quoted_value(day.replace('/', '-'))}"
        raise ValueError(
            f"run day {quoted_value(day)} cannot name a candidate: a day must start with an "
            "ASCII letter or digit and contain only ASCII letters, digits, dot, underscore or "
            f"hyphen (maximum {RUN_DAY_MAX_CHARS} characters){hint}"
        )
    return day


@dataclass(frozen=True)
class RubricFailureRef:
    failure_case_id: str
    rubric_id: str
    rubric_version: int
    criterion_id: str
    conversation_id: str
    evidence: tuple[EvidenceRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_case_id": self.failure_case_id,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "criterion_id": self.criterion_id,
            "conversation_id": self.conversation_id,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass(frozen=True)
class SourceRef:
    thread_id: str
    element_id: str
    position: int
    role: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "element_id": self.element_id,
            "position": self.position,
            "role": self.role,
        }


@dataclass(frozen=True)
class TargetRef:
    skill_id: str
    root_id: str
    canonical_path: str
    base_sha256: str
    mirror_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "root_id": self.root_id,
            "canonical_path": self.canonical_path,
            "base_sha256": self.base_sha256,
            "mirror_paths": list(self.mirror_paths),
        }


@dataclass(frozen=True)
class Patch:
    """A full-text replacement plus the diff we computed ourselves.

    The model returns the whole intended file, never a diff. A model-authored
    diff is a derived artifact we would have to trust; a full text we can hash,
    diff locally, and re-verify byte-for-byte at apply time.
    """

    new_text: str
    new_text_sha256: str
    unified_diff: str
    added_lines: int
    removed_lines: int

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": "full_text",
            "new_text_sha256": self.new_text_sha256,
            "unified_diff": self.unified_diff,
            "added_lines": self.added_lines,
            "removed_lines": self.removed_lines,
        }
        if include_text:
            out["new_text"] = self.new_text
        return out


def one_line_label(text: str) -> str:
    r"""Untrusted text that must occupy exactly one line: a line break inside
    it is spelled (``\n``, ``\r``), not obeyed.

    Used for the ``---``/``+++`` labels of a unified diff, which difflib copies
    verbatim: every content line of a diff carries a prefix character, but the
    header labels are whatever path they were given, and a skill directory
    named with a newline would otherwise put its text at column 0 of the diff
    a reviewer reads. The CLI's listing lines use the same spelling.
    """
    return text.replace("\r", "\\r").replace("\n", "\\n")


NO_NEWLINE_MARKER = "\\ No newline at end of file"


def _terminated(diff_line: str) -> str:
    """A diff line that ends where its line ends.

    ``difflib`` keeps each text's own line ending, so when the last line of
    the base or of the new text has none, the diff line made from it has none
    either and the next diff line is joined onto it: ``-last line+last line
    changed`` read as one removed line, and an added line could disappear into
    the removed line before it. The marker is the one ``diff`` and ``git``
    print for the same case, on its own line, so every changed line is its own
    line and the missing newline is itself visible.
    """
    if diff_line.endswith("\n"):
        return diff_line
    return f"{diff_line}\n{NO_NEWLINE_MARKER}\n"


def make_patch(base_text: str, new_text: str, *, path_label: str) -> Patch:
    label = one_line_label(path_label)
    base_lines = base_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            base_lines,
            new_lines,
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            n=3,
        )
    )
    # The two file headers are the first two lines of a non-empty unified diff
    # and nothing else is one. Judged by prefix they were mistaken for a changed
    # line that happens to begin with two dashes or two pluses: a removed
    # frontmatter delimiter renders as ``----`` and was not counted. Measured
    # before: a replacement that dropped both delimiters reported 0 removed.
    body = diff[2:]
    added = sum(1 for line in body if line.startswith("+"))
    removed = sum(1 for line in body if line.startswith("-"))
    return Patch(
        new_text=new_text,
        new_text_sha256=sha256_text(new_text),
        unified_diff="".join(_terminated(line) for line in diff),
        added_lines=added,
        removed_lines=removed,
    )


def challenge_for(candidate_id: str, base_sha256: str, patch_sha256: str) -> str:
    seed = f"{CHALLENGE_DOMAIN}:{candidate_id}:{base_sha256}:{patch_sha256}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:CHALLENGE_LEN]


def approval_token_for(challenge: str, base_sha256: str, patch_sha256: str) -> str:
    seed = f"{APPROVAL_DOMAIN}:{challenge}:{base_sha256}:{patch_sha256}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:TOKEN_LEN]


@dataclass(frozen=True)
class Candidate:
    id: str
    created_at: str
    hypothesis: str
    intended_change: str
    risks: tuple[str, ...]
    rollback: str
    target: TargetRef
    rubric_failures: tuple[RubricFailureRef, ...]
    sources: tuple[SourceRef, ...]
    patch: Patch
    approval_challenge: str
    status: str = STATUS_PROPOSED
    improver_command_id: str = ""
    validation: dict[str, Any] | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_candidate_id(self.id)

    @property
    def recomputed_patch_sha256(self) -> str:
        """The hash of the patch text as it stands, never the stored field.

        Derived on every read so a record whose ``new_text`` was edited after
        review cannot present the hash of the text that *was* reviewed.
        """
        return sha256_text(self.patch.new_text)

    @property
    def required_approval_token(self) -> str:
        """The token that authorises this candidate, bound to its actual bytes.

        Derived from the recomputed patch hash rather than the stored one. That is
        the difference between a token bound to the text a human reviewed and a
        token bound to a field sitting beside it: with the stored hash, editing
        ``new_text`` alone left this token unchanged and the edit authorised.
        """
        return approval_token_for(
            self.approval_challenge, self.target.base_sha256, self.recomputed_patch_sha256
        )

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        return {
            "schema": SCHEMA_CANDIDATE,
            "id": self.id,
            "created_at": self.created_at,
            "status": self.status,
            "hypothesis": self.hypothesis,
            "intended_change": self.intended_change,
            "risks": list(self.risks),
            "rollback": self.rollback,
            "target": self.target.to_dict(),
            "rubric_failures": [r.to_dict() for r in self.rubric_failures],
            "sources": [s.to_dict() for s in self.sources],
            "patch": self.patch.to_dict(include_text=include_text),
            "approval_challenge": self.approval_challenge,
            "improver_command_id": self.improver_command_id,
            "validation": self.validation,
            "notes": list(self.notes),
        }

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "Candidate":
        if raw.get("schema") != SCHEMA_CANDIDATE:
            raise ValueError(f"unexpected candidate schema {quoted_value(raw.get('schema'))}")
        patch_raw = _object(raw["patch"], "patch")
        patch = Patch(
            new_text=str(patch_raw.get("new_text", "")),
            new_text_sha256=str(patch_raw["new_text_sha256"]),
            unified_diff=str(patch_raw.get("unified_diff", "")),
            added_lines=_integer(patch_raw.get("added_lines", 0), "patch.added_lines"),
            removed_lines=_integer(patch_raw.get("removed_lines", 0), "patch.removed_lines"),
        )
        target_raw = _object(raw["target"], "target")
        target = TargetRef(
            skill_id=str(target_raw["skill_id"]),
            root_id=str(target_raw["root_id"]),
            canonical_path=str(target_raw["canonical_path"]),
            base_sha256=str(target_raw["base_sha256"]),
            mirror_paths=_strings(target_raw.get("mirror_paths", []), "target.mirror_paths"),
        )
        failures = tuple(
            RubricFailureRef(
                failure_case_id=str(f["failure_case_id"]),
                rubric_id=str(f["rubric_id"]),
                rubric_version=_integer(
                    f.get("rubric_version", 1), f"rubric_failures[{index}].rubric_version"
                ),
                criterion_id=str(f["criterion_id"]),
                conversation_id=str(f["conversation_id"]),
                evidence=tuple(EvidenceRef.from_dict(e) for e in f.get("evidence", [])),
            )
            for index, f in enumerate(_objects(raw.get("rubric_failures", []), "rubric_failures"))
        )
        sources = tuple(
            SourceRef(
                thread_id=str(s["thread_id"]),
                element_id=str(s["element_id"]),
                position=_integer(s["position"], f"sources[{index}].position"),
                role=str(s.get("role", "")),
            )
            for index, s in enumerate(_objects(raw.get("sources", []), "sources"))
        )
        validation = raw.get("validation")
        if validation is not None:
            validation = dict(_object(validation, "validation"))
            if validation.get("gate") is not None:
                _object(validation["gate"], "validation.gate")
        # The improver's prose is masked when its proposal is parsed
        # (``parse_proposal``); it, and the notes, are masked again here, as the
        # judgement reader masks its rationale, so a record edited on disk cannot
        # make ``review`` print a credential in them, nor a later write of the
        # record keep one. ``str()`` reads what it read before. Masking prose the
        # tool masked when it wrote it leaves it as written, except that two
        # masks a URL parameter's shape joins read back as one; a note this tool
        # wrote reads back changed only where it holds a credential's shape.
        return Candidate(
            id=str(raw["id"]),
            created_at=str(raw["created_at"]),
            hypothesis=kept_prose(str(raw.get("hypothesis", ""))),
            intended_change=kept_prose(str(raw.get("intended_change", ""))),
            risks=tuple(kept_prose(r) for r in _strings(raw.get("risks", []), "risks")),
            rollback=kept_prose(str(raw.get("rollback", ""))),
            target=target,
            rubric_failures=failures,
            sources=sources,
            patch=patch,
            approval_challenge=str(raw.get("approval_challenge", "")),
            status=str(raw.get("status", STATUS_PROPOSED)),
            improver_command_id=str(raw.get("improver_command_id", "")),
            validation=validation,
            notes=tuple(kept_prose(n) for n in _strings(raw.get("notes", []), "notes")),
        )


def _object(value: Any, what: str) -> Mapping[str, Any]:
    """``value`` as the JSON object the record's shape says ``what`` is.

    ``patch``, ``target`` and ``validation`` used to be read with ``.get``
    straight off whatever the file held, so a record whose ``patch`` was a
    string, a list or null escaped the reader as a bare ``AttributeError``,
    which no caller classified: measured before, one such hand-edited record
    took ``candidates``, ``review``, ``apply`` and ``doctor`` down together with
    exit 70 and a traceback, where the contract is that an unreadable record is
    listed as unreadable, by name. The array positions (``sources``,
    ``rubric_failures``) were read by subscript and already failed as a
    classified ``TypeError``; here they gain the field's name in the refusal.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be an object, got {type(value).__name__}")
    return value


def _objects(value: Any, what: str) -> list[Mapping[str, Any]]:
    """``value`` as a list of objects, each refused by position if it is not one."""
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"{what} must be an array, got {type(value).__name__}")
    return [_object(item, f"{what}[{index}]") for index, item in enumerate(value)]


def _integer(value: Any, what: str) -> int:
    """``value`` read with ``int()``, as before, refused by name and type, never by value.

    ``patch.added_lines``, ``patch.removed_lines``, a rubric failure's
    ``rubric_version`` and a source's ``position`` were read with ``int()``,
    whose refusal quotes the text it could not read: measured before, a field
    holding a ``ghp_`` token was printed whole by ``review``, ``apply``,
    ``candidates`` and ``doctor``, text and JSON. What ``int()`` accepted it
    still accepts.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be an integer, got {type(value).__name__}") from None


def _strings(value: Any, what: str) -> tuple[str, ...]:
    """``value`` as the array of strings the record's shape says ``what`` is.

    ``target.mirror_paths``, ``risks`` and ``notes`` used to be read with
    ``str()`` of each item of whatever the field held, so an edited record
    whose field was one string was read a character at a time (a path of eight
    characters became eight mirror paths of one character each) and one whose
    items were objects or numbers was read as their ``str``. Neither is a
    record this tool wrote, so the loader refuses both, the field named and an
    item by its position; the type is named, never the value.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{what} must be an array of strings, got {type(value).__name__}")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{what}[{index}] must be a string, got {type(item).__name__}")
    return tuple(value)


MIN_HYPOTHESIS_CHARS = 40
MIN_INTENDED_CHANGE_CHARS = 40
MIN_ROLLBACK_CHARS = 20

#: The control characters that are ordinary text: line and paragraph structure.
#: Every other ``Cc`` code point (ESC, NUL, DEL, BEL, the C1 block) is a
#: terminal instruction, not prose.
REVIEWABLE_CONTROLS = frozenset("\n\r\t")
#: Bidirectional embedding, override and isolate controls. They change the
#: order text is DISPLAYED in without changing the order it is stored and
#: applied in, so a diff can read one way to the reviewer and another to the
#: agent that loads the skill. Marks (U+200E/F) and joiners are left alone:
#: they do not reorder anything.
BIDI_CONTROLS = frozenset("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")
#: Invisible carriers that are not ``Cc``: the Unicode tag block (U+E0000 to
#: U+E007F), which spells out ASCII in characters no terminal renders, and the
#: line and paragraph separators (U+2028, U+2029), which break a line without
#: being a newline. Zero-width spaces, joiners, soft hyphens and the BOM are
#: left alone: they occur in ordinary prose and hide nothing on their own.
HIDDEN_CONTROLS = frozenset({chr(cp) for cp in range(0xE0000, 0xE0080)} | {"\u2028", "\u2029"})


def unreviewable_characters(text: str, *, tolerated: str = "") -> list[str]:
    """Code points in ``text`` a reviewer cannot see for what they are.

    A terminal escape erases or recolours the lines above it; a bidi override
    reorders them. ``asi review`` prints a candidate's prose and diff raw, so
    text carrying either is not reviewable and never becomes a candidate.
    Returned as ``U+XXXX`` names, each once in order of first appearance, so the
    refusal that quotes them is itself safe to print. Characters that already
    occur in ``tolerated`` (the live skill text) are not reported: the rule is
    about what a proposal introduces, not about a file the operator owns. The
    tolerance is per code point, not per occurrence or position: one legacy
    override in the live file admits that same override anywhere in the
    replacement, which is the price of never making a skill unimprovable.
    """
    allowed = REVIEWABLE_CONTROLS | set(tolerated)
    found: list[str] = []
    seen: set[str] = set()
    for ch in text:
        if ch in allowed or ch in seen:
            continue
        if ch in BIDI_CONTROLS or ch in HIDDEN_CONTROLS or unicodedata.category(ch) == "Cc":
            seen.add(ch)
            found.append(f"U+{ord(ch):04X}")
    return found


def candidate_defects(candidate: Candidate, *, base_text: str) -> list[str]:
    """Every reason this candidate is not reviewable. Empty list means usable."""
    problems: list[str] = []
    prose = {
        "hypothesis": candidate.hypothesis,
        "intended_change": candidate.intended_change,
        "risks": "\n".join(candidate.risks),
        "rollback": candidate.rollback,
    }
    for name, text in prose.items():
        bad = unreviewable_characters(text)
        if bad:
            problems.append(
                f"{name} contains control or bidi characters a reviewer cannot see "
                f"({', '.join(bad)}); a terminal escape is never reviewable"
            )
    for name, text in (
        ("patch new_text", candidate.patch.new_text),
        ("patch unified_diff", candidate.patch.unified_diff),
    ):
        bad = unreviewable_characters(text, tolerated=base_text)
        if bad:
            problems.append(
                f"{name} introduces control or bidi characters absent from the base text "
                f"({', '.join(bad)}); text that renders in one order and applies in another "
                "is never reviewable"
            )
    if len(candidate.hypothesis.strip()) < MIN_HYPOTHESIS_CHARS:
        problems.append(f"hypothesis shorter than {MIN_HYPOTHESIS_CHARS} chars")
    if len(candidate.intended_change.strip()) < MIN_INTENDED_CHANGE_CHARS:
        problems.append(f"intended_change shorter than {MIN_INTENDED_CHANGE_CHARS} chars")
    if not candidate.risks or all(not r.strip() for r in candidate.risks):
        problems.append("no risks stated")
    if len(candidate.rollback.strip()) < MIN_ROLLBACK_CHARS:
        problems.append(f"rollback shorter than {MIN_ROLLBACK_CHARS} chars")
    if not candidate.rubric_failures:
        problems.append("no rubric failures referenced")
    if not candidate.sources:
        problems.append("no source conversation/message references")
    if not candidate.target.base_sha256:
        problems.append("target has no base_sha256")
    if not candidate.target.canonical_path:
        problems.append("target has no canonical_path")
    if not candidate.patch.new_text.strip():
        problems.append("patch new_text is empty")
    normalized_base = base_text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_new = candidate.patch.new_text.replace("\r\n", "\n").replace("\r", "\n")
    if normalized_base == normalized_new:
        problems.append("patch is identical to the base text (no-op change)")
    if not candidate.patch.unified_diff.strip():
        problems.append("patch has no unified diff")
    if candidate.patch.new_text_sha256 != candidate.recomputed_patch_sha256:
        problems.append("patch new_text_sha256 does not match the hash of new_text")
    expected = challenge_for(
        candidate.id, candidate.target.base_sha256, candidate.recomputed_patch_sha256
    )
    if candidate.approval_challenge != expected:
        problems.append("approval_challenge does not match candidate identity")
    return problems


@dataclass(frozen=True)
class Proposal:
    """The improver's raw output, before we turn it into a candidate."""

    target_skill_id: str
    hypothesis: str
    intended_change: str
    risks: tuple[str, ...]
    rollback: str
    new_skill_text: str
    addressed_criterion_ids: tuple[str, ...] = ()


def parse_proposal(payload: Any) -> Proposal:
    """Parse the improver contract. Raises ``ValueError`` on any violation."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"proposal must be a JSON object, got {type(payload).__name__}")
    if payload.get("schema") != SCHEMA_PROPOSAL:
        raise ValueError(
            f"proposal schema must be {SCHEMA_PROPOSAL!r}, got {quoted_value(payload.get('schema'))}"
        )
    required = ("target_skill_id", "hypothesis", "intended_change", "rollback", "new_skill_text")
    for key in required:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"proposal field {key!r} must be a non-empty string")
    risks = payload.get("risks")
    if (
        not isinstance(risks, list)
        or not risks
        or any(not isinstance(r, str) or not r.strip() for r in risks)
    ):
        raise ValueError("proposal field 'risks' must be a non-empty list of non-empty strings")
    addressed = payload.get("addressed_criterion_ids", [])
    if not isinstance(addressed, list) or any(not isinstance(x, str) for x in addressed):
        raise ValueError("proposal field 'addressed_criterion_ids' must be a list of strings")
    # The prose fields are kept masked (kept_prose); new_skill_text is not,
    # because a masked replacement would be a silently different document.
    # A replacement that introduces a credential is refused in the pipeline.
    return Proposal(
        target_skill_id=str(payload["target_skill_id"]).strip(),
        hypothesis=kept_prose(payload["hypothesis"]).strip(),
        intended_change=kept_prose(payload["intended_change"]).strip(),
        risks=tuple(kept_prose(r).strip() for r in risks),
        rollback=kept_prose(payload["rollback"]).strip(),
        new_skill_text=str(payload["new_skill_text"]),
        addressed_criterion_ids=tuple(kept_prose(x) for x in addressed),
    )


def failure_refs_from_cases(cases: Sequence[FailureCase]) -> tuple[RubricFailureRef, ...]:
    return tuple(
        RubricFailureRef(
            failure_case_id=c.id,
            rubric_id=c.rubric_id,
            rubric_version=c.rubric_version,
            criterion_id=c.criterion_id,
            conversation_id=c.conversation_id,
            evidence=c.evidence,
        )
        for c in cases
    )


def sources_from_cases(cases: Sequence[FailureCase]) -> tuple[SourceRef, ...]:
    seen: set[tuple[str, str, int]] = set()
    out: list[SourceRef] = []
    for case in cases:
        for ref in case.evidence:
            key = (case.conversation_id, ref.element_id, ref.position)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                SourceRef(
                    thread_id=case.conversation_id,
                    element_id=ref.element_id,
                    position=ref.position,
                    role=ref.kind,
                )
            )
    return tuple(out)
