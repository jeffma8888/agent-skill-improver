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
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .. import SCHEMA_CANDIDATE, SCHEMA_PROPOSAL
from ..hashing import sha256_text
from .conversation import EvidenceRef
from .failure import FailureCase

STATUS_PROPOSED = "proposed"
STATUS_VALIDATED = "validated"
STATUS_GATE_FAILED = "gate-failed"
STATUS_APPLIED = "applied"
STATUS_ROLLED_BACK = "rolled-back"

CHALLENGE_DOMAIN = "asi-challenge:v1"
APPROVAL_DOMAIN = "asi-approve:v1"
CHALLENGE_LEN = 24
TOKEN_LEN = 16


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


def make_patch(base_text: str, new_text: str, *, path_label: str) -> Patch:
    base_lines = base_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            base_lines,
            new_lines,
            fromfile=f"a/{path_label}",
            tofile=f"b/{path_label}",
            n=3,
        )
    )
    added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    return Patch(
        new_text=new_text,
        new_text_sha256=sha256_text(new_text),
        unified_diff="".join(diff),
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
            raise ValueError(f"unexpected candidate schema {raw.get('schema')!r}")
        patch_raw = raw["patch"]
        patch = Patch(
            new_text=str(patch_raw.get("new_text", "")),
            new_text_sha256=str(patch_raw["new_text_sha256"]),
            unified_diff=str(patch_raw.get("unified_diff", "")),
            added_lines=int(patch_raw.get("added_lines", 0)),
            removed_lines=int(patch_raw.get("removed_lines", 0)),
        )
        target_raw = raw["target"]
        target = TargetRef(
            skill_id=str(target_raw["skill_id"]),
            root_id=str(target_raw["root_id"]),
            canonical_path=str(target_raw["canonical_path"]),
            base_sha256=str(target_raw["base_sha256"]),
            mirror_paths=tuple(str(p) for p in target_raw.get("mirror_paths", [])),
        )
        failures = tuple(
            RubricFailureRef(
                failure_case_id=str(f["failure_case_id"]),
                rubric_id=str(f["rubric_id"]),
                rubric_version=int(f.get("rubric_version", 1)),
                criterion_id=str(f["criterion_id"]),
                conversation_id=str(f["conversation_id"]),
                evidence=tuple(EvidenceRef.from_dict(e) for e in f.get("evidence", [])),
            )
            for f in raw.get("rubric_failures", [])
        )
        sources = tuple(
            SourceRef(
                thread_id=str(s["thread_id"]),
                element_id=str(s["element_id"]),
                position=int(s["position"]),
                role=str(s.get("role", "")),
            )
            for s in raw.get("sources", [])
        )
        return Candidate(
            id=str(raw["id"]),
            created_at=str(raw["created_at"]),
            hypothesis=str(raw.get("hypothesis", "")),
            intended_change=str(raw.get("intended_change", "")),
            risks=tuple(str(x) for x in raw.get("risks", [])),
            rollback=str(raw.get("rollback", "")),
            target=target,
            rubric_failures=failures,
            sources=sources,
            patch=patch,
            approval_challenge=str(raw.get("approval_challenge", "")),
            status=str(raw.get("status", STATUS_PROPOSED)),
            improver_command_id=str(raw.get("improver_command_id", "")),
            validation=raw.get("validation"),
            notes=tuple(str(x) for x in raw.get("notes", [])),
        )


MIN_HYPOTHESIS_CHARS = 40
MIN_INTENDED_CHANGE_CHARS = 40
MIN_ROLLBACK_CHARS = 20


def candidate_defects(candidate: Candidate, *, base_text: str) -> list[str]:
    """Every reason this candidate is not reviewable. Empty list means usable."""
    problems: list[str] = []
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
            f"proposal schema must be {SCHEMA_PROPOSAL!r}, got {payload.get('schema')!r}"
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
    return Proposal(
        target_skill_id=str(payload["target_skill_id"]).strip(),
        hypothesis=str(payload["hypothesis"]).strip(),
        intended_change=str(payload["intended_change"]).strip(),
        risks=tuple(str(r).strip() for r in risks),
        rollback=str(payload["rollback"]).strip(),
        new_skill_text=str(payload["new_skill_text"]),
        addressed_criterion_ids=tuple(str(x) for x in addressed),
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
