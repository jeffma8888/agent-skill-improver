"""Skill identity: snapshots, canonical source resolution, mirror avoidance.

The same logical skill is frequently present in several roots at once (a per-user
preference copy, a vendor-neutral steering repo, and one mirror per agent tool
that loads it). Editing the wrong copy is the classic silent failure: the edit
"lands", nothing errors, and the agent that actually loads the skill never sees
it -- or worse, the next mirror sync overwrites it.

So this module answers exactly two questions and refuses to guess:

* Which physical file is the *canonical editable source* for a logical skill?
* Which files are mirrors that must never be written to directly?

There is a third thing it *reports* and never touches. A skill is not a document
in the general case: it is a directory, and that directory frequently contains
executable payloads -- hooks, loop launchers, MCP launchers. Write access to a
skill tree is therefore arbitrary code execution on the host, and it can include
write access to the very guardrails that are supposed to contain the loop. So
every snapshot enumerates the executable payload files sitting beside it, the
count reaches the run receipt and the candidate, and the write path refuses any
target that is not the ``SKILL.md`` document itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SKILL_FILENAME = "SKILL.md"

#: File extensions treated as executable payloads regardless of the exec bit.
PAYLOAD_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".rb",
        ".pl",
        ".php",
        ".command",
        ".applescript",
        ".osascript",
        ".ps1",
        ".bat",
        ".cmd",
        ".exe",
    }
)


def is_payload_name(name: str) -> bool:
    """True when a filename looks like an executable payload.

    Matched on the suffix only, and deliberately without shelling out to
    anything. Real skill trees contain filenames with literal backslashes and
    spaces in them; any check that word-splits a shell command mangles those, and
    if the exit status is discarded the mangling reads as "no findings".
    """
    lowered = name.lower()
    for ext in PAYLOAD_EXTENSIONS:
        if lowered.endswith(ext):
            return True
    return False


@dataclass(frozen=True)
class SkillRoot:
    """A directory that contains skill folders."""

    id: str
    path: Path
    # Precedence orders candidate canonical sources; lower wins.
    precedence: int = 100
    # ``canonical`` marks a root the operator has declared as the place edits
    # belong. ``editable`` is a separate axis: a root can be canonical-by-policy
    # yet marked non-editable while an operator investigates.
    canonical: bool = False
    editable: bool = True
    kind: str = "builtin"  # builtin | custom | mirror
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path),
            "precedence": self.precedence,
            "canonical": self.canonical,
            "editable": self.editable,
            "kind": self.kind,
            "note": self.note,
        }


@dataclass(frozen=True)
class SkillSnapshot:
    """One physical SKILL.md, hashed."""

    skill_id: str
    root_id: str
    path: Path
    sha256: str
    size_bytes: int
    frontmatter_name: str = ""
    frontmatter_description: str = ""
    has_opening_delimiter: bool = True
    disable_model_invocation: bool = False
    warnings: tuple[str, ...] = ()
    #: Executable files that live in this skill's directory. Reported, never
    #: written: a "skill improvement" that could touch one of these would be a
    #: privileged change, not a docs change.
    payload_files: tuple[str, ...] = ()

    @property
    def payload_count(self) -> int:
        return len(self.payload_files)

    @property
    def has_executable_payload(self) -> bool:
        return bool(self.payload_files)

    @property
    def loads(self) -> bool:
        """False when the file cannot be parsed by the skill loader at all.

        A missing opening ``---`` makes the frontmatter unparseable, and the
        skill then never loads -- silently, with no error anywhere. Reporting
        that is more useful than any rubric score about the skill's content.
        """
        return self.has_opening_delimiter and bool(self.frontmatter_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "root_id": self.root_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "frontmatter_name": self.frontmatter_name,
            "frontmatter_description": self.frontmatter_description,
            "has_opening_delimiter": self.has_opening_delimiter,
            "disable_model_invocation": self.disable_model_invocation,
            "loads": self.loads,
            "payload_files": list(self.payload_files),
            "payload_count": self.payload_count,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SkillGroup:
    """All physical copies of one logical skill, with the canonical one named."""

    skill_id: str
    canonical: SkillSnapshot | None
    mirrors: tuple[SkillSnapshot, ...]
    canonical_reason: str = ""

    @property
    def all_copies(self) -> tuple[SkillSnapshot, ...]:
        if self.canonical is None:
            return self.mirrors
        return (self.canonical,) + self.mirrors

    @property
    def is_mirrored(self) -> bool:
        return len(self.all_copies) > 1

    @property
    def divergent_mirrors(self) -> tuple[SkillSnapshot, ...]:
        """Mirrors whose content already differs from canonical.

        A divergent mirror is a warning, not an error: it means the two copies
        have drifted, so a canonical edit will not reach whatever loads the
        mirror. We report it and refuse to "fix" it automatically.
        """
        if self.canonical is None:
            return ()
        return tuple(m for m in self.mirrors if m.sha256 != self.canonical.sha256)

    @property
    def payload_files(self) -> tuple[str, ...]:
        """Executable payloads across every copy of this skill, de-duplicated."""
        out: list[str] = []
        for copy in self.all_copies:
            for name in copy.payload_files:
                label = f"{copy.root_id}:{name}"
                if label not in out:
                    out.append(label)
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "canonical": self.canonical.to_dict() if self.canonical else None,
            "payload_files": list(self.payload_files),
            "canonical_reason": self.canonical_reason,
            "mirrors": [m.to_dict() for m in self.mirrors],
            "is_mirrored": self.is_mirrored,
            "divergent_mirror_root_ids": [m.root_id for m in self.divergent_mirrors],
        }


def resolve_groups(
    snapshots: Iterable[SkillSnapshot],
    roots: Sequence[SkillRoot],
    *,
    passed_over: Iterable[tuple[str, str, str]] = (),
) -> list[SkillGroup]:
    """Group snapshots by logical skill id and pick the canonical copy.

    Selection order, applied deterministically:

    1. roots flagged ``canonical`` and ``editable``, lowest precedence first;
    2. roots that are merely ``editable``, lowest precedence first;
    3. otherwise no canonical copy exists and proposals against the skill are
       refused rather than pointed at a mirror.

    ``passed_over`` is ``(root_id, skill_id, reason)`` for every document
    discovery saw and did not take. A copy passed over in an editable root is
    a copy this run cannot see, not an absent one: the skill then has no
    canonical copy and the reason says which copy and why, so no proposal is
    pointed at a lower-precedence copy while the real one is unreadable.
    Measured before: a canonical ``SKILL.md`` at mode 000 made the group read
    exactly like a mirror-only skill, and ``daily`` quarantined it with "no
    editable root holds this skill", which was false.
    """
    by_root = {r.id: r for r in roots}
    buckets: dict[str, list[SkillSnapshot]] = {}
    for snap in snapshots:
        buckets.setdefault(snap.skill_id, []).append(snap)
    withheld: dict[str, list[tuple[str, str]]] = {}
    for root_id, skill_id, why in passed_over:
        root = by_root.get(root_id)
        if root and root.editable:
            withheld.setdefault(skill_id, []).append((root_id, why))

    groups: list[SkillGroup] = []
    for skill_id in sorted(buckets):
        copies = buckets[skill_id]

        def sort_key(s: SkillSnapshot) -> tuple[int, str]:
            root = by_root.get(s.root_id)
            return (root.precedence if root else 10_000, s.root_id)

        ordered = sorted(copies, key=sort_key)
        canonical: SkillSnapshot | None = None
        reason = ""
        if skill_id in withheld:
            root_id, why = withheld[skill_id][0]
            groups.append(
                SkillGroup(
                    skill_id,
                    None,
                    tuple(ordered),
                    f"the copy in editable root {root_id!r} was passed over by discovery "
                    f"({why}); proposals are refused until it reads, rather than pointed "
                    "at another copy",
                )
            )
            continue
        for snap in ordered:
            root = by_root.get(snap.root_id)
            if root and root.canonical and root.editable:
                canonical, reason = snap, f"root {root.id!r} is canonical and editable"
                break
        if canonical is None:
            for snap in ordered:
                root = by_root.get(snap.root_id)
                if root and root.editable:
                    canonical, reason = (
                        snap,
                        f"root {root.id!r} is the highest-precedence editable root",
                    )
                    break
        if canonical is None:
            reason = "no editable root holds this skill; proposals will be refused"
        mirrors = tuple(s for s in ordered if s is not canonical)
        groups.append(SkillGroup(skill_id, canonical, mirrors, reason))
    return groups


def group_for(groups: Sequence[SkillGroup], skill_id: str) -> SkillGroup | None:
    for g in groups:
        if g.skill_id == skill_id:
            return g
    return None


def _same_path(a: Path, b: Path) -> bool:
    """Pure path comparison: normalized text only, no filesystem access.

    Symlink resolution is deliberately NOT done here, because this module does no
    I/O. The caller is responsible for handing in a fully resolved path -- see the
    contract note on :func:`classify_target`.
    """
    return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(os.path.normpath(str(b)))


def classify_target(groups: Sequence[SkillGroup], path: Path) -> tuple[str, SkillGroup | None]:
    """Return ``("canonical" | "mirror" | "unknown", group)`` for a path.

    CONTRACT: ``path`` must already be resolved, and the snapshots in ``groups``
    carry resolved paths. Comparing an unresolved path against a resolved one
    returned ``"unknown"``, and because a caller treated ``"unknown"`` as "not a
    mirror", the mirror interlock was bypassable by spelling the same file
    differently -- a relative path, a ``..`` segment, or a symlinked parent such as
    the platform temp directory. The failure direction was open, which is the wrong
    default for an interlock: an unrecognised target is now refused by the write
    path rather than permitted.
    """
    for g in groups:
        if g.canonical is not None and _same_path(g.canonical.path, path):
            return "canonical", g
        for m in g.mirrors:
            if _same_path(m.path, path):
                return "mirror", g
    return "unknown", None


def parse_frontmatter(text: str) -> tuple[dict[str, str], bool, list[str]]:
    """Parse the leading ``---`` block of a SKILL.md.

    Returns ``(fields, has_opening_delimiter, warnings)``. Intentionally a
    minimal scalar parser: skills use flat ``key: value`` frontmatter, and
    pulling in a YAML dependency to read four keys is not worth it.
    """
    warnings: list[str] = []
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        warnings.append(
            "no opening '---' delimiter: frontmatter is unparseable and the skill "
            "will never load (fails silently, with no error anywhere)"
        )
        return {}, False, warnings
    fields: dict[str, str] = {}
    closed = False
    for raw in lines[1:]:
        if raw.strip() == "---":
            closed = True
            break
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        fields[key.strip()] = value.strip().strip('"').strip("'")
    if not closed:
        warnings.append("frontmatter block is never closed with '---'")
    if not fields.get("name"):
        warnings.append("frontmatter has no 'name' field")
    if not fields.get("description"):
        warnings.append("frontmatter has no 'description' field")
    return fields, True, warnings
