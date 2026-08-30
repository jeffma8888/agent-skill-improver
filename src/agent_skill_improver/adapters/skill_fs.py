"""Filesystem skill store: discovery, snapshots, and the only write path.

Safety properties enforced here rather than by convention:

* the store is **read-only by default**, and the daily driver only ever
  constructs it read-only, so an accidental write raises instead of landing;
* a write verifies the on-disk hash against the candidate's recorded base hash
  first, so a skill edited by a human since the proposal was made cannot be
  silently clobbered;
* a write always takes a timestamped backup with a machine-readable manifest,
  so rollback is a single command and does not depend on the operator having
  git in that directory;
* writes to protected paths -- this package's own source, config and rubrics --
  are refused outright, which is what stops the loop from promoting its own
  rubric;
* writes to a mirror copy are refused; only the canonical source is writable.
"""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..domain.skills import (
    SKILL_FILENAME,
    SkillGroup,
    SkillRoot,
    SkillSnapshot,
    classify_target,
    is_payload_name,
    parse_frontmatter,
    resolve_groups,
)
from ..errors import (
    BaseHashMismatch,
    MirrorEditRefused,
    NonDocumentWriteRefused,
    ReadOnlyViolation,
    SelfMutationRefused,
)
from ..hashing import normalize_text, sha256_file, sha256_text, tree_digest
from ..jsonio import write_json, write_text_atomic

#: Backups live under the owner-only state tree, so they are owner-only too.
BACKUP_FILE_MODE = 0o600

DEFAULT_MAX_DEPTH = 3


@dataclass(frozen=True)
class WriteReceipt:
    path: str
    skill_id: str
    previous_sha256: str
    new_sha256: str
    backup_path: str
    backup_manifest_path: str
    rollback_command: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "skill_id": self.skill_id,
            "previous_sha256": self.previous_sha256,
            "new_sha256": self.new_sha256,
            "backup_path": self.backup_path,
            "backup_manifest_path": self.backup_manifest_path,
            "rollback_command": self.rollback_command,
        }


class FilesystemSkillStore:
    """Discovers and (only when explicitly writable) edits skill files."""

    def __init__(
        self,
        roots: Sequence[SkillRoot],
        *,
        read_only: bool = True,
        protected_paths: Iterable[Path] = (),
        max_depth: int = DEFAULT_MAX_DEPTH,
        follow_symlinks: bool = False,
    ) -> None:
        self._roots = tuple(roots)
        self.read_only = read_only
        self._protected = tuple(Path(p).resolve() for p in protected_paths)
        self._max_depth = max_depth
        self._follow_symlinks = follow_symlinks
        self._snapshot_cache: list[SkillSnapshot] | None = None

    # -- discovery -------------------------------------------------------
    def roots(self) -> Sequence[SkillRoot]:
        return self._roots

    def existing_roots(self) -> tuple[SkillRoot, ...]:
        return tuple(r for r in self._roots if r.path.is_dir())

    def missing_roots(self) -> tuple[SkillRoot, ...]:
        return tuple(r for r in self._roots if not r.path.is_dir())

    def snapshots(self, *, refresh: bool = False) -> list[SkillSnapshot]:
        if self._snapshot_cache is not None and not refresh:
            return list(self._snapshot_cache)
        found: list[SkillSnapshot] = []
        for root in self._roots:
            if not root.path.is_dir():
                continue
            base = root.path.resolve()
            for skill_file in self._iter_skill_files(base):
                found.append(self._snapshot(root, base, skill_file))
        found.sort(key=lambda s: (s.skill_id, s.root_id))
        self._snapshot_cache = found
        return list(found)

    def groups(self, *, refresh: bool = False) -> list[SkillGroup]:
        return resolve_groups(self.snapshots(refresh=refresh), self._roots)

    def known_skill_ids(self) -> list[str]:
        return sorted({s.skill_id for s in self.snapshots()})

    def tree_digest(self) -> str:
        """A single hash over every discovered skill file.

        The daily driver records this before and after its run and asserts they
        are equal. That turns "we never mutate live skills" from a claim in a
        README into a measured property of every run receipt.

        Keyed on ``root_id`` plus the path RELATIVE to that root, never the
        absolute path. Two reasons, and the first is a correctness bug rather than
        a nicety:

        * Relocating a skill root would otherwise change the digest while no skill
          content changed at all -- a false positive on the single most important
          field in the receipt, and false positives on a mutation alarm are how an
          alarm stops being read.
        * It makes a receipt comparable across machines and checkouts, which is
          what the determinism guarantee is worth having.

        A file appearing, disappearing, changing content, or moving between roots
        all still change the digest, because content and ``root_id`` are both in
        the key.
        """
        entries: dict[str, str] = {}
        for snapshot in self.snapshots(refresh=True):
            entries[f"{snapshot.root_id}:{self._relative_key(snapshot)}"] = snapshot.sha256
        return tree_digest(entries)

    def _relative_key(self, snapshot: SkillSnapshot) -> str:
        """The snapshot's path relative to its own root, as posix text."""
        for root in self._roots:
            if root.id != snapshot.root_id:
                continue
            try:
                return snapshot.path.relative_to(Path(root.path).resolve()).as_posix()
            except ValueError:
                try:
                    return snapshot.path.relative_to(Path(root.path)).as_posix()
                except ValueError:
                    break
        # Unresolvable against any root: fall back to the skill id plus filename,
        # which is stable, rather than to the absolute path, which is not.
        return f"{snapshot.skill_id}/{snapshot.path.name}"

    def read_text(self, path: Path) -> str:
        return Path(path).read_text(encoding="utf-8")

    # -- write path ------------------------------------------------------
    def write_canonical(
        self,
        path: Path,
        text: str,
        *,
        expected_sha256: str,
        backup_dir: Path,
        skill_id: str = "",
        actor: str = "human",
    ) -> WriteReceipt:
        """The single method in this package that modifies a skill file."""
        # Resolved once, so every later step (hash, backup, manifest, verify)
        # describes the same file even when the caller spelled the path
        # differently. Two spellings of one path is how a check and the write it
        # guards end up looking at different things.
        target = Path(path).resolve() if Path(path).exists() else Path(path)
        if self.read_only:
            raise ReadOnlyViolation(
                f"refusing to write {target}: this skill store is read-only. "
                "The daily driver always opens the store read-only; only `asi apply` "
                "opens it writable."
            )
        self._assert_document(target)
        self._assert_not_protected(target)
        self._assert_canonical(target)
        if not target.exists():
            raise BaseHashMismatch(f"refusing to write {target}: file does not exist")
        current = sha256_file(target)
        if current != expected_sha256:
            raise BaseHashMismatch(
                f"refusing to write {target}: on-disk sha256 is {current[:12]} but the "
                f"candidate was written against {expected_sha256[:12]}. The skill changed "
                "since the proposal was made -- re-run validation against the new base."
            )
        # The target's own mode, captured BEFORE anything is written. An atomic
        # write replaces the inode, so without this the new file would carry the
        # writer's default rather than the mode the skill had -- silently
        # tightening or loosening somebody else's file. Recorded in the manifest
        # so rollback can restore it too.
        target_mode = stat.S_IMODE(target.stat().st_mode)
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(backup_dir, 0o700)
        backup_path = backup_dir / f"{target.name}.bak"
        shutil.copy2(target, backup_path)
        # The backup lives under the state tree, which is owner-only. copy2
        # preserves the source mode, so a world-readable skill would otherwise
        # put a world-readable file inside a directory that promises 0600.
        os.chmod(backup_path, BACKUP_FILE_MODE)
        backup_sha = sha256_file(backup_path)
        if backup_sha != current:
            raise BaseHashMismatch(
                f"backup verification failed for {target}: copy hashes {backup_sha[:12]} "
                f"but source hashes {current[:12]}; refusing to proceed"
            )
        new_sha = sha256_text(text)
        write_text_atomic(target, normalize_text(text), mode=target_mode)
        written_sha = sha256_file(target)
        if written_sha != new_sha:
            # Restore immediately: a write that did not land as intended is a
            # rollback trigger, not something to report and leave in place.
            shutil.copy2(backup_path, target)
            os.chmod(target, target_mode)
            raise BaseHashMismatch(
                f"post-write verification failed for {target} "
                f"(expected {new_sha[:12]}, found {written_sha[:12]}); original restored"
            )
        manifest_path = backup_dir / "backup.json"
        rollback = f"cp {backup_path} {target}"
        write_json(
            manifest_path,
            {
                "schema": "asi.backup.v1",
                "target_path": str(target),
                "skill_id": skill_id,
                "actor": actor,
                "previous_sha256": current,
                "new_sha256": new_sha,
                "backup_path": str(backup_path),
                "target_mode": oct(target_mode),
                "rollback_command": rollback,
            },
        )
        self._snapshot_cache = None
        return WriteReceipt(
            path=str(target),
            skill_id=skill_id,
            previous_sha256=current,
            new_sha256=new_sha,
            backup_path=str(backup_path),
            backup_manifest_path=str(manifest_path),
            rollback_command=rollback,
        )

    def rollback(self, manifest_path: Path) -> dict[str, Any]:
        """Restore a file from a backup manifest and verify the restored hash."""
        from ..jsonio import read_json

        manifest = read_json(Path(manifest_path))
        target = Path(manifest["target_path"])
        backup = Path(manifest["backup_path"])
        if self.read_only:
            raise ReadOnlyViolation(f"refusing to roll back {target}: store is read-only")
        self._assert_document(target)
        self._assert_not_protected(target)
        if not backup.exists():
            raise BaseHashMismatch(f"backup {backup} is missing; cannot roll back")
        shutil.copy2(backup, target)
        # The backup was tightened to 0600 when it was taken, so copying it back
        # would leave the skill owner-only unless the recorded mode is restored.
        # Manifests written before this field existed have no recorded mode; in
        # that case leave whatever the copy produced rather than guessing, and
        # say so here so a reader is not surprised by the asymmetry.
        recorded_mode = str(manifest.get("target_mode", ""))
        if recorded_mode:
            try:
                os.chmod(target, int(recorded_mode, 8))
            except (OSError, ValueError):
                pass
        restored = sha256_file(target)
        expected = str(manifest["previous_sha256"])
        if restored != expected:
            raise BaseHashMismatch(
                f"rollback verification failed: restored sha {restored[:12]} != {expected[:12]}"
            )
        self._snapshot_cache = None
        return {
            "rolled_back": str(target),
            "restored_sha256": restored,
            "from_backup": str(backup),
        }

    # -- internals -------------------------------------------------------
    def _assert_document(self, target: Path) -> None:
        """Only the skill document is ever writable.

        A skill directory routinely contains hooks and launchers, so a write path
        that accepted an arbitrary filename inside it would be a code-execution
        primitive wearing a docs-change costume.
        """
        if target.name != SKILL_FILENAME:
            raise NonDocumentWriteRefused(
                f"refusing to write {target}: this tool only ever replaces {SKILL_FILENAME}. "
                "Executable payloads in a skill directory (hooks, launchers, MCP wrappers) are "
                "privileged code and are never edited by an automated proposal."
            )

    def _assert_not_protected(self, target: Path) -> None:
        resolved = target.resolve()
        for guard in self._protected:
            if resolved == guard or guard in resolved.parents:
                raise SelfMutationRefused(
                    f"refusing to write {target}: it is inside a protected path ({guard}). "
                    "This tool never edits its own source, config, or rubrics -- a loop that "
                    "can rewrite its own scoring criteria has no gate at all."
                )

    def _assert_canonical(self, target: Path) -> None:
        """Refuse a mirror copy, and refuse anything we cannot identify at all.

        Resolution happens HERE because this is the adapter layer; the domain
        comparison is pure text. Both sides must be resolved or the compare is
        meaningless: snapshots carry resolved paths, so an unresolved caller path
        classified as ``unknown`` and slipped straight through.

        ``unknown`` is refused rather than allowed. It is a genuinely surprising
        state -- every legitimate apply targets a path that came from a snapshot --
        and treating "I do not recognise this file" as permission is how an
        interlock becomes decorative.
        """
        resolved = Path(target).resolve()
        kind, group = classify_target(self.groups(refresh=True), resolved)
        if kind == "mirror":
            canonical = group.canonical.path if group and group.canonical else "<unknown>"
            raise MirrorEditRefused(
                f"refusing to write {target}: it is a mirror copy. The canonical source is "
                f"{canonical}. Editing a mirror either gets overwritten by the next sync or "
                "never reaches the agent that loads the canonical file."
            )
        if kind == "unknown":
            raise MirrorEditRefused(
                f"refusing to write {target}: it is not a skill document in any configured "
                "root, so this tool cannot tell whether it is canonical. An unrecognised "
                "target is refused rather than assumed safe."
            )

    def _iter_skill_files(self, base: Path) -> list[Path]:
        found: list[Path] = []
        for candidate in sorted(base.rglob(SKILL_FILENAME)):
            try:
                rel = candidate.resolve().relative_to(base)
            except ValueError:
                # Symlink escaping the root. Skipped unless explicitly allowed.
                if not self._follow_symlinks:
                    continue
                rel = Path(candidate.name)
            if len(rel.parts) > self._max_depth:
                continue
            if candidate.is_symlink() and not self._follow_symlinks:
                continue
            found.append(candidate)
        return found

    def _payload_files(self, skill_dir: Path) -> tuple[str, ...]:
        """Enumerate executable payloads beside a skill document.

        Pure ``pathlib``: filenames containing literal backslashes, spaces or
        quotes are common in real skill trees and any shell-based scan mangles
        them silently.
        """
        found: list[str] = []
        try:
            entries = sorted(skill_dir.rglob("*"))
        except OSError:
            return ()
        for entry in entries:
            try:
                if entry.is_dir() or entry.is_symlink():
                    continue
                if entry.name == SKILL_FILENAME:
                    continue
                rel = entry.relative_to(skill_dir).as_posix()
                if is_payload_name(entry.name):
                    found.append(rel)
                    continue
                if entry.stat().st_mode & 0o111:
                    found.append(rel)
            except (OSError, ValueError):
                continue
        return tuple(found)

    def _snapshot(self, root: SkillRoot, base: Path, skill_file: Path) -> SkillSnapshot:
        text = skill_file.read_text(encoding="utf-8", errors="replace")
        fields, has_open, warnings = parse_frontmatter(text)
        rel = skill_file.resolve().relative_to(base)
        skill_id = rel.parts[-2] if len(rel.parts) >= 2 else skill_file.parent.name
        declared = fields.get("name", "")
        if declared and declared != skill_id:
            warnings = warnings + [
                f"frontmatter name {declared!r} differs from directory name {skill_id!r}"
            ]
        disable = fields.get("disable-model-invocation", "").strip().lower() in ("true", "yes", "1")
        if disable:
            warnings = warnings + [
                "disable-model-invocation is true: the skill is hidden from the "
                "auto-loaded listing and can only be invoked by name"
            ]
        payloads = self._payload_files(skill_file.parent)
        if payloads:
            warnings = warnings + [
                f"{len(payloads)} executable payload file(s) live in this skill directory "
                f"({', '.join(payloads[:5])}"
                + (", ..." if len(payloads) > 5 else "")
                + "): changes here are privileged, and this tool never writes them"
            ]
        return SkillSnapshot(
            skill_id=skill_id,
            root_id=root.id,
            path=skill_file,
            sha256=sha256_text(text),
            size_bytes=len(text.encode("utf-8")),
            frontmatter_name=declared,
            frontmatter_description=fields.get("description", ""),
            has_opening_delimiter=has_open,
            disable_model_invocation=disable,
            payload_files=payloads,
            warnings=tuple(warnings),
        )
