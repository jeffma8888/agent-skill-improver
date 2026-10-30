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

import errno
import os
import re
import secrets
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..domain.integrity import repropose_advice
from ..domain.redaction import named_path, quoted_value, why_unreadable
from ..domain.skills import (
    SKILL_FILENAME,
    UNREADABLE_ENTRY,
    SkillGroup,
    SkillRoot,
    SkillSnapshot,
    classify_target,
    is_payload_name,
    parse_frontmatter,
    resolve_groups,
    tree_digest_of,
)
from ..errors import (
    BackupUnusable,
    BaseHashMismatch,
    LinkedTargetRefused,
    MirrorEditRefused,
    NonDocumentWriteRefused,
    ReadOnlyViolation,
    SelfMutationRefused,
    SkillWriteFailed,
    UndecodableDocument,
)
from ..hashing import normalize_text, sha256_file, sha256_text
from ..jsonio import write_bytes_atomic, write_json

#: Backups live under the owner-only state tree, so they are owner-only too.
BACKUP_FILE_MODE = 0o600

#: A skill document is read, replaced and re-read through a directory handle
#: opened component by component below its root, each with ``O_NOFOLLOW``; the
#: handle pins the directory, so a link planted on the way after the walk is
#: not followed. POSIX only, like the owner-only modes the state tree relies on.
PINNED_IO_SUPPORTED = (
    os.open in os.supports_dir_fd
    and os.rename in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
)

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


def _write_pinned(parent_fd: int, leaf: str, data: bytes, *, mode: int) -> None:
    """Atomic replace of ``leaf`` through a pinned directory handle.

    The temp file is created with ``O_CREAT | O_EXCL | O_NOFOLLOW`` relative to
    the handle, synced, given ``mode``, then renamed over ``leaf`` relative to
    the same handle: the directory entry is swapped, so a link at ``leaf`` is
    replaced rather than written through, and the directory itself is the one
    the handle was opened on, whatever the path now says.
    """
    tmp = f".tmp-{leaf}-{os.getpid()}-{secrets.token_hex(4)}.part"
    try:
        # Opened inside the ``try``, so a stop that lands as the open returns,
        # before the descriptor is held, still has the file removed. A name
        # already taken is another writer's, and is left alone.
        try:
            fd = os.open(
                tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd
            )
        except FileExistsError:
            tmp = ""
            raise
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), mode)
        os.rename(tmp, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    except BaseException:
        if tmp:
            try:
                os.unlink(tmp, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        raise


def _remove_manifest(path: Path) -> None:
    """Remove the manifest of a write that left the skill holding its original text.

    Best effort: a manifest left behind names a write that never landed, and
    ``rollback`` refuses a candidate still at its base as never applied, so a
    failure here must not hide the refusal the caller is about to raise.
    """
    try:
        os.unlink(path)
    except OSError:
        pass


#: How ``write_canonical`` spells the mode it records: ``oct()`` of the
#: permission bits, which never runs past four octal digits.
_RECORDED_MODE = re.compile(r"0o[0-7]{1,4}")


def _restorable_mode(recorded: Any, live_mode: int) -> int:
    """The mode a rollback writes the restored text at.

    Never a permission the live file lacks, and always one its owner can read,
    because the restore is verified by reading it back. In the flow this tool
    runs the two modes agree: ``write_canonical`` records the mode it writes
    the applied text at (the skill's own, plus the owner read bit). They differ
    after a chmod since the apply or a hand edit of the manifest, and then the
    recorded mode is honoured only in the shape ``write_canonical`` records,
    narrowed to the live file's bits, and only if its owner read bit survives
    that; otherwise the live file's mode is kept. Either way the owner read bit
    is added: the restore is a new file this process owns, and a live file read
    through its group or other bits would otherwise be restored unreadable to
    its new owner.

    Measured before: the field was parsed as any octal number, so a hand-edited
    ``"0"`` or ``"0o200"`` restored the skill at a mode its owner cannot read,
    the verification read then failed and the command reported a refusal with
    the file already rewritten and unreadable, and ``"-1"``, ``"0o7777"`` or
    ``"0o6755"`` restored it with setuid, setgid or world write and reported
    success.
    """
    mode = live_mode
    if isinstance(recorded, str) and _RECORDED_MODE.fullmatch(recorded):
        narrowed = int(recorded, 8) & live_mode
        if narrowed & stat.S_IRUSR:
            mode = narrowed
    return mode | stat.S_IRUSR


def _read_regular_file(path: Path) -> bytes | None:
    """``path``'s bytes when it is a regular file, else ``None``.

    Opened without following a link and without blocking, then judged by
    ``fstat`` on the open handle, so what is judged is what is read: a link
    or a fifo put in place after the checks before this is never read through
    or waited on. A link raises ``OSError`` (``ELOOP``) from the open. The raw
    handle is judged and read, and closed whatever happens: measured before,
    ``os.fdopen`` on a directory raised without closing the handle it was given.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _entry_kind(path: Path) -> str:
    """What a non-regular entry is, by stat, for a report line."""
    try:
        mode = path.stat().st_mode
    except OSError:
        return "unreadable"
    if stat.S_ISDIR(mode):
        return "a directory"
    if stat.S_ISFIFO(mode):
        return "a fifo"
    if stat.S_ISSOCK(mode):
        return "a socket"
    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        return "a device"
    return "neither a file nor a directory"


@dataclass(frozen=True)
class PassedOverDocument:
    """A path discovery saw and did not take, with the reason.

    Discovery steps over a link (unless the store follows links), a link that
    resolves outside its root, a document deeper than ``max_depth``, anything
    that is not a regular file (a dangling link, a fifo, a device), a directory
    it could not list, and a configured root whose path holds something other
    than a directory (recorded at the root's own path, with no skill id).
    Each is the right refusal, and each used to be silent: ``doctor`` said
    ``ready`` and ``skills`` listed nothing for a tree whose only ``SKILL.md``
    was a dangling link, so the operator could not tell a skipped skill from
    an absent one.
    """

    root_id: str
    path: Path
    reason: str
    #: The skill this entry would have been a copy of, had discovery taken it:
    #: set only for a ``SKILL.md`` at a depth discovery accepts, by the same
    #: rule ``_snapshot`` uses (the parent directory's name). None for an
    #: unlistable directory, a document too deep, a root-level document or the
    #: root path itself, none of which is a copy of anything, so none of them
    #: withholds a skill's slot.
    skill_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "path": str(self.path),
            "reason": self.reason,
            "skill_id": self.skill_id,
        }


def _root_is_directory(path: Path) -> bool:
    """Whether a configured root's path reaches a directory, without raising.

    ``Path.is_dir`` swallows only not-found; a root under a parent this process
    cannot search raised ``PermissionError`` out of it. Measured before: that
    one root ended ``doctor`` and ``daily`` as exit 70 with a traceback from the
    presence check, before anything was discovered.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def root_presence(path: Path) -> tuple[bool, bool]:
    """``(exists, is_directory)`` for a configured root, without raising.

    ``exists`` is what is on disk at the path, a dangling link included;
    ``is_directory`` is what discovery can use.
    """
    try:
        exists = path.exists() or path.is_symlink()
    except OSError:
        # Something is there that this process cannot examine: it exists.
        exists = True
    return exists, _root_is_directory(path)


def _root_shape_problem(path: Path) -> str | None:
    """What sits at a configured root's path when it is not a directory.

    ``None`` when the path is a directory (through a link or not) or when there
    is nothing there at all: absence is the ``missing_roots`` case and is
    reported as such. Measured with ``lstat`` first so a dangling link is named
    as a link rather than as absence; the final fallback names a fifo, socket
    or device the way the document walk does.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"a path this process cannot examine ({exc.strerror or exc})"
    if stat.S_ISLNK(st.st_mode):
        if _root_is_directory(path):
            return None
        # The link's text is a path, named as a refusal names one: a refusal
        # of a target below this root gives this reason whole.
        try:
            target = named_path(os.readlink(path))
        except OSError as exc:
            target = f"(unreadable: {exc.strerror or exc})"
        try:
            reachable = path.exists()
        except OSError:
            reachable = False
        if not reachable:
            return f"a dangling symbolic link to {target}, not a directory"
        return f"a symbolic link to {target}, which is not a directory"
    if stat.S_ISDIR(st.st_mode):
        return None
    if stat.S_ISREG(st.st_mode):
        return "a regular file, not a directory"
    return f"{_entry_kind(path)}, not a directory"


def changed_before_the_write(target: Path, current: str, expected: str, candidate_id: str) -> str:
    """The refusal of a write whose file no longer hashes what the caller checked.

    ``apply`` (``app/promote.py``) checks the live file against the candidate's
    base before it calls the write, so a file this check refuses changed while
    ``apply`` ran: after its check and before this read. Measured before, the
    refusal said "re-run validation against the new base", which no command
    does, and named no candidate. With the candidate it now names it and gives
    the advice ``apply``'s own check gives; without one it gives no advice.
    """
    if not candidate_id:
        return (
            f"refusing to write {named_path(target)}: on-disk sha256 is {current[:12]} but the "
            f"write was asked against {expected[:12]}, so the file changed after that hash "
            "was taken. Nothing was written."
        )
    return (
        f"refusing to apply {candidate_id}: {named_path(target)} now hashes {current[:12]} "
        f"but the candidate was written against {expected[:12]}. The skill changed while "
        "this apply ran, after it checked the file and before it wrote, and the skill was "
        "not written. To propose and validate against the new base, "
        f"{repropose_advice(candidate_id)}."
    )


def listed_as_protected(target: Path, named: str, candidate_id: str, verb: str) -> str:
    """The refusal of a skill listed in ``[guard].protected_skill_ids``.

    ``review`` and ``apply`` read the list again, so a candidate validated
    before its skill was listed is refused here. Measured before (a committed
    run with one validated candidate, then its skill listed): both refused it
    with "refusing to review <path>" and "refusing to write <path>", named no
    candidate, and said "nothing was touched", while ``apply`` had written the
    run lock's owner line. The record stayed ``validated``, so it stayed open:
    with ``[caps].max_open_candidates`` at 1, the next day stood down on it,
    and with it deleted the next day ran. With the skill taken off the list,
    ``review`` and ``apply`` with the same token took it. With the candidate,
    the refusal now names it, for the command that asked, and says how it is
    closed; without one it is worded as before. It says ``apply`` does not
    write the skill, not that the candidate is never applied: an ``apply``
    that finishes an earlier run of itself, stopped after it wrote the skill,
    records that write before this check is asked, and writes nothing.
    """
    if not candidate_id:
        return (
            f"refusing to {verb} {named_path(target)}: skill {quoted_value(named)} is "
            "listed in [guard].protected_skill_ids, so it is never a proposal target "
            "and never applied. A candidate validated before the skill was listed, or "
            "a record edited to name it, is refused here; nothing was touched."
        )
    command = "review" if verb == "review" else "apply"
    return (
        f"refusing to {command} {candidate_id}: its skill {quoted_value(named)} at "
        f"{named_path(target)} is listed in [guard].protected_skill_ids, so `apply` does "
        "not write it while it is listed. The skill and the candidate are unchanged. The "
        "candidate stays open, and counts against [caps].max_open_candidates, until it is "
        f"closed: to close it, delete its record ({candidate_id}.json in the state "
        "directory's candidates/). If the skill is not meant to stay listed, take it off "
        "the list, and `review` and `apply` take the candidate as before."
    )


def reached_through_a_link(spelled: Path, cursor: Path, candidate_id: str, verb: str) -> str:
    """The refusal of a target reached through ``cursor``, a symbolic link below its root.

    ``review`` and ``apply`` ask it through the target check, so a candidate
    whose skill folder became a link after it was validated is refused here.
    Measured before (3.11 and 3.13, the skill folder replaced by a link to an
    identical copy, beside it in the root or outside every root): ``review``
    and ``apply`` each exited 3 with ``LinkedTargetRefused`` "refusing to review
    <path>" and "refusing to replace <path>", which named no candidate and no
    way to close it. The record stayed ``validated``, so it stayed open: with
    ``[caps].max_open_candidates`` at 1 the next day stood down on it, and with
    it deleted the next day ran. With the folder put back in place of the link,
    ``review`` and ``apply`` with the same token took it. With the candidate,
    the refusal now names it, for the command that asked, and says how it is
    closed; without one (``rollback``) it is worded as before. It says
    ``apply`` does not write through the link: an ``apply`` that finishes an
    earlier run of itself records that write before this check is asked, and
    writes nothing.
    """
    if not candidate_id:
        return (
            f"refusing to {verb} {named_path(spelled)}: {named_path(cursor)} is a symbolic link, "
            "so the replacement would land wherever it points rather than on the skill the "
            "candidate was reviewed against. Neither the link nor its target was touched."
        )
    command = "review" if verb == "review" else "apply"
    itself = "the file itself" if cursor == spelled else "the directory itself"
    return (
        f"refusing to {command} {candidate_id}: its canonical target {named_path(spelled)} is "
        f"reached through {named_path(cursor)}, a symbolic link, so `apply` does not write "
        "through it: the replacement would land wherever the link points rather than on the "
        "skill the candidate was reviewed against. The link, what it points to and the "
        "candidate are unchanged. The candidate stays open, and counts against "
        "[caps].max_open_candidates, until it is closed: to close it, delete its record "
        f"({candidate_id}.json in the state directory's candidates/). If the link is not "
        f"meant to be there, put {itself} in its place: while the skill file there holds the "
        "text the candidate was written against, `review` and `apply` take the candidate as "
        "before."
    )


def canonical_target_refused(target: Path, why: str, candidate_id: str, action: str) -> str:
    """The refusal of a target that is not the canonical copy of its skill, for ``why``.

    ``review`` and ``apply`` ask it through the target check, so a candidate
    whose skill gained a second copy, whose root became a mirror, or whose
    root was re-pointed after it was validated is refused here. Measured
    before (3.11 and 3.13): ``review`` and ``apply`` each exited 3 with
    ``MirrorEditRefused`` "refusing to review <path>: ..." and "refusing to
    write <path>: ...", which named no candidate and no way to close it. The
    record stayed ``validated``, so it stayed open: with
    ``[caps].max_open_candidates`` at 1 the next day stood down on it, and with
    it deleted the next day ran. With the change undone, ``review`` and
    ``apply`` with the same token took it. With the candidate, the refusal now
    names it, for the command that asked, and says how it is closed; without
    one (``rollback``, and the write's own check) it is worded as before.
    """
    if not candidate_id:
        return f"refusing to {action} {named_path(target)}: {why}"
    command = "review" if action == "review" else "apply"
    return (
        f"refusing to {command} {candidate_id} at {named_path(target)}: {why} The file and "
        "the candidate are unchanged. The candidate stays open, and counts against "
        "[caps].max_open_candidates, until it is closed: to close it, delete its record "
        f"({candidate_id}.json in the state directory's candidates/). If that file is meant "
        "to be the skill's canonical copy, make it that again: while it holds the text the "
        "candidate was written against, `review` and `apply` take the candidate as before."
    )


class FilesystemSkillStore:
    """Discovers and (only when explicitly writable) edits skill files."""

    def __init__(
        self,
        roots: Sequence[SkillRoot],
        *,
        read_only: bool = True,
        protected_paths: Iterable[Path] = (),
        protected_skill_ids: Iterable[str] = (),
        max_depth: int = DEFAULT_MAX_DEPTH,
        follow_symlinks: bool = False,
    ) -> None:
        self._roots = tuple(roots)
        self.read_only = read_only
        self._protected = tuple(Path(p).resolve() for p in protected_paths)
        self._protected_skill_ids = frozenset(str(s) for s in protected_skill_ids)
        self._max_depth = max_depth
        self._follow_symlinks = follow_symlinks
        self._snapshot_cache: list[SkillSnapshot] | None = None
        self._passed_over: list[PassedOverDocument] = []

    # -- discovery -------------------------------------------------------
    def roots(self) -> Sequence[SkillRoot]:
        return self._roots

    def existing_roots(self) -> tuple[SkillRoot, ...]:
        return tuple(r for r in self._roots if _root_is_directory(r.path))

    def missing_roots(self) -> tuple[SkillRoot, ...]:
        """Configured roots with nothing at their path.

        A root whose path holds something that is not a directory is not
        missing; it is misplaced, and reported as a passed-over entry with what
        was found there. Measured before this distinction: a canonical root
        that was a regular file read as "not present on this machine", so the
        operator was sent to check the spelling of a path that was right.
        """
        return tuple(
            r
            for r in self._roots
            if not _root_is_directory(r.path) and _root_shape_problem(r.path) is None
        )

    def misplaced_roots(self) -> tuple[tuple[SkillRoot, str], ...]:
        """Configured roots whose path holds something other than a directory."""
        found: list[tuple[SkillRoot, str]] = []
        for root in self._roots:
            if _root_is_directory(root.path):
                continue
            problem = _root_shape_problem(root.path)
            if problem is not None:
                found.append((root, problem))
        return tuple(found)

    def snapshots(self, *, refresh: bool = False) -> list[SkillSnapshot]:
        if self._snapshot_cache is not None and not refresh:
            return list(self._snapshot_cache)
        found: list[SkillSnapshot] = []
        passed_over: list[PassedOverDocument] = []
        for root in self._roots:
            if not _root_is_directory(root.path):
                # Something at the root's path that is not a directory is a
                # finding at the root itself, with no skill id: nothing below
                # it can be discovered, and "not present" would be untrue.
                problem = _root_shape_problem(root.path)
                if problem is not None:
                    passed_over.append(
                        PassedOverDocument(
                            root.id,
                            root.path,
                            f"the configured root itself and is {problem}",
                            skill_id=None,
                        )
                    )
                continue
            base = root.path.resolve()
            for skill_file in self._iter_skill_files(
                base, root_id=root.id, passed_over=passed_over
            ):
                # The bytes are taken here so that a document the walk saw but
                # this process cannot read (mode 000, or gone between the
                # listing and the read) is passed over with its reason, like a
                # fifo or a dangling link. This covers the discovery read; the
                # propose stage reads the canonical copy again later. Measured
                # before: one unreadable SKILL.md ended `doctor` and `daily`
                # alike with exit 70 and a PermissionError traceback from here.
                try:
                    raw = skill_file.read_bytes()
                except OSError as exc:
                    passed_over.append(
                        PassedOverDocument(
                            root.id,
                            skill_file,
                            f"a document this process cannot read ({exc.strerror or exc})",
                            skill_id=self._skill_id_at(base, skill_file),
                        )
                    )
                    continue
                found.append(self._snapshot(root, base, skill_file, raw))
        found.sort(key=lambda s: (s.skill_id, s.root_id))
        self._snapshot_cache = found
        self._passed_over = passed_over
        return list(found)

    def passed_over(self, *, refresh: bool = False) -> list[PassedOverDocument]:
        """Every ``SKILL.md`` the last discovery saw and did not take."""
        if self._snapshot_cache is None or refresh:
            self.snapshots(refresh=refresh)
        return list(self._passed_over)

    def groups(self, *, refresh: bool = False) -> list[SkillGroup]:
        snapshots = self.snapshots(refresh=refresh)
        return resolve_groups(
            snapshots,
            self._roots,
            passed_over=[
                (d.root_id, d.skill_id, d.reason) for d in self._passed_over if d.skill_id
            ],
        )

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
        the key. The digest is over readable documents only; ``tree_entries``
        is the mapping the daily driver compares, and it also carries the copies
        discovery saw and could not read.
        """
        return tree_digest_of(self.tree_entries())

    def tree_entries(self) -> dict[str, str]:
        """Every skill document discovery saw, keyed like the digest, re-read now.

        A readable document maps to its content hash. A copy of a skill that
        discovery passed over because it could not read it maps to
        ``UNREADABLE_ENTRY``, so that a document another process makes
        unreadable mid-run is a named change on the receipt ("became
        unreadable: ...") rather than an entry that silently vanished from the
        digest and was then reported as an edit. Passed-over entries that are
        not a copy of any skill (a document too deep, an unlistable directory)
        are not in the tree and not here.
        """
        entries: dict[str, str] = {}
        for snapshot in self.snapshots(refresh=True):
            key = self._relative_key(snapshot.root_id, snapshot.path, snapshot.skill_id)
            entries[f"{snapshot.root_id}:{key}"] = snapshot.sha256
        for document in self._passed_over:
            if document.skill_id is None:
                continue
            key = self._relative_key(document.root_id, document.path, document.skill_id)
            entries.setdefault(f"{document.root_id}:{key}", UNREADABLE_ENTRY)
        return entries

    def _relative_key(self, root_id: str, path: Path, skill_id: str) -> str:
        """The document's path relative to its own root, as posix text."""
        for root in self._roots:
            if root.id != root_id:
                continue
            try:
                return path.relative_to(Path(root.path).resolve()).as_posix()
            except ValueError:
                try:
                    return path.relative_to(Path(root.path)).as_posix()
                except ValueError:
                    break
        # Unresolvable against any root: fall back to the skill id plus filename,
        # which is stable, rather than to the absolute path, which is not.
        return f"{skill_id}/{path.name}"

    def read_text(self, path: Path) -> str:
        """The document's text, judged the way the write path judges it.

        This read decides whether ``review`` shows a token and whether ``apply``
        accepts the base, so it must not accept what the write will refuse.
        Measured before this method opened the file itself: a target replaced by
        a link to an identical copy made ``review`` print a token for an apply
        that then refused the link, and a fifo at the target parked ``review``
        for good. The leaf is opened ``O_NOFOLLOW`` (a link is a refusal, not a
        read of its target) and ``O_NONBLOCK`` (a fifo opens instead of
        waiting for a writer), then judged on the open handle: anything but a
        regular file is refused. A file that is simply missing stays an
        ``OSError`` for the caller to name; bytes that are not UTF-8 are an
        ``UndecodableDocument`` (a record defect, so ``daily`` quarantines the
        one skill rather than ending the run). Line endings are folded to ``\n``
        as the text-mode read before it did, so a CRLF document still reviews
        as the lines that changed rather than as every line.
        """
        target = Path(path)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if not nofollow and target.is_symlink():
            raise LinkedTargetRefused(
                f"refusing to read {named_path(target)}: the document is a symbolic link, which the write "
                f"path would refuse to replace, so nothing may be shown or approved against it."
            )
        flags = os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(str(target), flags)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise LinkedTargetRefused(
                    f"refusing to read {named_path(target)}: the document is a symbolic link, which the "
                    f"write path would refuse to replace, so nothing may be shown or approved "
                    f"against it."
                ) from exc
            raise
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise BaseHashMismatch(
                    f"refusing to read {named_path(target)}: the document is not a regular file, so there "
                    f"is no base text to show or to verify against. Nothing was touched."
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(fd)
        try:
            text = b"".join(chunks).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UndecodableDocument(
                f"skill document {named_path(target)} is not valid UTF-8 at byte {exc.start} ({exc.reason}); "
                f"nothing is proposed, shown or applied against text this tool cannot read "
                f"exactly. The file is left as it is: fix the encoding and the next pass will "
                f"handle it.",
                record_id=target.parent.name,
            ) from exc
        return normalize_text(text)

    @staticmethod
    def _document_sha256(data: bytes, target: Path) -> str:
        """The hash of a document's bytes under the identity every other hash here uses.

        ``sha256_text`` folds line endings before hashing, and every snapshot,
        candidate and manifest records that value. Measured before this helper:
        apply hashed the raw bytes, so a CRLF document, proposed and validated
        from its folded text, was refused at apply as "changed since the
        proposal was made" when nothing had changed, every time.
        """
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UndecodableDocument(
                f"skill document {named_path(target)} is not valid UTF-8 at byte {exc.start} ({exc.reason}); "
                "nothing is written over text this tool cannot read exactly. The file is left "
                "as it is: fix the encoding and run the command again.",
                record_id=target.parent.name,
            ) from exc
        return sha256_text(text)

    @staticmethod
    def _in_document_convention(text: str, data: bytes) -> bytes:
        """``text`` encoded with the line endings the document on disk uses.

        The tool's text is LF throughout; a document written as CRLF throughout
        stays CRLF, so apply changes the lines the diff showed and no others.
        Anything else (LF, mixed, or lone carriage returns) is written as LF.
        """
        payload = normalize_text(text)
        if b"\r\n" in data and data.count(b"\n") == data.count(b"\r\n"):
            payload = payload.replace("\n", "\r\n")
        return payload.encode("utf-8")

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
        candidate_id: str = "",
    ) -> WriteReceipt:
        """The single method in this package that modifies a skill file.

        ``candidate_id`` names the candidate ``apply`` is writing, so a file that
        changed after ``apply`` checked it is refused with that candidate and the
        advice ``apply``'s own check gives (see ``changed_before_the_write``).
        """
        if self.read_only:
            raise ReadOnlyViolation(
                f"refusing to write {named_path(path)}: this skill store is read-only. "
                "The daily driver always opens the store read-only; only `asi apply` "
                "opens it writable."
            )
        # Every judgement of the target, made where ``review`` makes it too.
        target = self.assert_write_target(Path(path), action="write", skill_id=skill_id)
        if not target.exists():
            raise BaseHashMismatch(f"refusing to write {named_path(target)}: file does not exist")
        # From here on the document is addressed through a handle on its own
        # directory, taken with O_NOFOLLOW at every component below the root, so
        # the hash that is checked, the bytes that are backed up, the write and
        # the re-read all concern one directory: a link swapped in after the
        # walk above cannot redirect any of them.
        parent_fd, leaf = self._open_parent_pinned(Path(path), verb="write")
        try:
            data, target_mode = self._read_pinned(parent_fd, leaf, target, verb="write")
            current = self._document_sha256(data, target)
            if current != expected_sha256:
                raise BaseHashMismatch(
                    changed_before_the_write(target, current, expected_sha256, candidate_id)
                )
            # The target's own mode, captured BEFORE anything is written. An
            # atomic write replaces the inode, so without this the new file would
            # carry the writer's default rather than the mode the skill had --
            # silently tightening or loosening somebody else's file. Recorded in
            # the manifest so rollback can restore it too.
            #
            # Plus the owner read bit. The replacement is a new file this process
            # owns, and the write is verified by reading it back, so a skill this
            # process reads through its group or other bits would otherwise be
            # written at a mode its new owner cannot read. Measured before: the
            # verification read then failed and apply reported "Nothing was
            # touched" with the text already changed and no manifest written.
            written_mode = target_mode | stat.S_IRUSR
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(backup_dir, 0o700)
            backup_path = backup_dir / f"{target.name}.bak"
            # Written as a fresh regular file at 0600, atomically. The backup
            # lives under the owner-only state tree, so it must not inherit a
            # world-readable skill's mode; and ``os.replace`` swaps the directory
            # entry, so a link planted at this path is replaced rather than
            # written through to wherever it pointed.
            write_bytes_atomic(backup_path, data, mode=BACKUP_FILE_MODE)
            backup_sha = sha256_file(backup_path)
            if backup_sha != current:
                raise BaseHashMismatch(
                    f"backup verification failed for {named_path(target)}: copy hashes {backup_sha[:12]} "
                    f"but source hashes {current[:12]}; refusing to proceed"
                )
            new_sha = sha256_text(text)
            manifest_path = backup_dir / "backup.json"
            rollback = shlex.join(["cp", str(backup_path), str(target)])
            # Written before the skill is, so an apply stopped or crashed at any
            # point after the skill write leaves the manifest ``rollback``
            # restores from and a second ``apply`` records the write by
            # (``earlier_write``). Where the skill is known to hold its original
            # text again, it is removed below. Measured before, with it written
            # after: an apply stopped as the write returned left the skill
            # changed, the candidate ``validated`` and no manifest, and
            # ``rollback`` answered that the candidate was never applied.
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
                    "target_mode": oct(written_mode),
                    "rollback_command": rollback,
                },
            )
            try:
                _write_pinned(
                    parent_fd, leaf, self._in_document_convention(text, data), mode=written_mode
                )
            except OSError as exc:
                # Measured before: a skill directory at 0555 made apply fall
                # with a raw PermissionError on the temp file, exit 70.
                _remove_manifest(manifest_path)
                raise SkillWriteFailed(
                    f"refusing to apply: {named_path(target)} could not be written "
                    f"({exc.strerror or exc}). The skill is unchanged; the copy taken "
                    f"before the write is at {named_path(backup_path)}."
                ) from exc
            written_sha, found = self._read_back(parent_fd, leaf, target)
            if written_sha != new_sha:
                # Restore immediately: a write that did not land as intended, or
                # that cannot be read back, is a rollback trigger, not something
                # to report and leave in place. The bytes are the ones read and
                # backed up above, not a second read of the backup's path. A
                # restore the platform refuses (a directory put at the name, which
                # no rename puts a file over, or a skill directory that no longer
                # takes a new file) is named with where the copy is.
                try:
                    _write_pinned(parent_fd, leaf, data, mode=written_mode)
                except OSError as exc:
                    raise BaseHashMismatch(
                        f"post-write verification failed for {named_path(target)} (expected "
                        f"{new_sha[:12]}, found {found}), and the original could not be put "
                        f"back ({exc.strerror or exc}). The copy taken before the write is at "
                        f"{named_path(backup_path)}: restore it by hand once the name can be written."
                    ) from exc
                _remove_manifest(manifest_path)
                raise BaseHashMismatch(
                    f"post-write verification failed for {named_path(target)} "
                    f"(expected {new_sha[:12]}, found {found}); original restored"
                )
        finally:
            os.close(parent_fd)
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

    def earlier_write(
        self, manifest_path: Path, *, path: Path, previous_sha256: str, new_sha256: str
    ) -> WriteReceipt | None:
        """The receipt of the write ``manifest_path`` records, if it is this one.

        An ``apply`` stopped or crashed after it wrote the skill, and before it
        recorded the candidate applied, leaves the skill holding the candidate's
        text and the manifest ``write_canonical`` wrote before the write. This
        answers whether ``manifest_path`` records that write: the same document
        (both resolved, as the write resolved it), from ``previous_sha256`` to
        ``new_sha256``. A manifest that cannot be read, or records anything
        else, is not that write and the answer is ``None``. Nothing is written;
        the manual command is rebuilt from the recorded paths rather than read.
        """
        try:
            manifest = self._read_backup_manifest(manifest_path)
            same_document = Path(manifest["target_path"]).resolve() == Path(path).resolve()
        except (BackupUnusable, OSError, ValueError, RuntimeError):
            return None
        if not same_document or (manifest["previous_sha256"], manifest["new_sha256"]) != (
            previous_sha256,
            new_sha256,
        ):
            return None
        skill_id = manifest.get("skill_id")
        return WriteReceipt(
            path=manifest["target_path"],
            skill_id=skill_id if isinstance(skill_id, str) else "",
            previous_sha256=previous_sha256,
            new_sha256=new_sha256,
            backup_path=manifest["backup_path"],
            backup_manifest_path=str(manifest_path),
            rollback_command=shlex.join(["cp", manifest["backup_path"], manifest["target_path"]]),
        )

    def rollback(
        self,
        manifest_path: Path,
        *,
        path: Path | None = None,
        previous_sha256: str | None = None,
        new_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Restore a file from a backup manifest, verified before anything is written.

        Every check that can be made without touching the live file is made
        first, because this is the recovery command: if it overwrites the skill
        and *then* discovers the backup was truncated, unreadable or edited, the
        operator is left with a corrupt file, a raised exception, and no second
        copy to try again from. So the manifest shape and the backup's own hash
        are verified while the live file is still intact, and a refusal here
        leaves the tree exactly as it was.

        The post-copy hash check stays. It is now a check on the copy itself
        rather than the only check there is.

        The live file is checked too. Rollback undoes ONE apply, so the only
        text it may overwrite is the text that apply wrote (``new_sha256``). A
        file that already holds the previous text is reported as restored
        without a write; anything else is a later edit, by a human or by another
        candidate's apply, and overwriting it would make the recovery command
        the thing that destroyed work. ``apply`` has refused a changed base
        since the start; rollback holding the same line is the symmetric half.

        Given the write a candidate's apply made (``path``, from
        ``previous_sha256`` to ``new_sha256``), the manifest must record that
        write, or nothing is touched (``_assert_records_the_write``).
        """
        manifest = self._read_backup_manifest(manifest_path)
        target = Path(manifest["target_path"])
        backup = Path(manifest["backup_path"])
        expected = manifest["previous_sha256"]
        applied = manifest["new_sha256"]
        if self.read_only:
            raise ReadOnlyViolation(
                f"refusing to roll back {named_path(target)}: store is read-only"
            )
        # The manifest's paths are text anyone with the state tree can edit, so
        # judging them can fail in the platform's hands as well as ours. A NUL
        # byte or a name past the filesystem's limit raised ValueError or
        # OSError from the first stat, and on 3.11 a symlink loop on the way
        # to the target raised RuntimeError from ``resolve``; each made the
        # command fall as an internal error. Nothing is written before this
        # returns, so a refusal says so.
        try:
            if path is not None or previous_sha256 is not None or new_sha256 is not None:
                self._assert_records_the_write(
                    manifest,
                    Path(manifest_path),
                    path=path,
                    previous_sha256=previous_sha256,
                    new_sha256=new_sha256,
                )
            parent_fd, leaf, restored_bytes = self._judge_rollback(
                target, backup, expected, manifest_path=Path(manifest_path)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
            raise BackupUnusable(
                f"refusing to roll back: what backup manifest {named_path(manifest_path)} names could not "
                f"be checked ({reason}). No file was touched."
            ) from exc
        try:
            return self._rollback_pinned(
                parent_fd,
                leaf,
                target,
                backup=backup,
                data=restored_bytes,
                expected=expected,
                applied=applied,
                manifest=manifest,
            )
        finally:
            os.close(parent_fd)

    @staticmethod
    def _assert_records_the_write(
        manifest: dict[str, Any],
        manifest_path: Path,
        *,
        path: Path | None,
        previous_sha256: str | None,
        new_sha256: str | None,
    ) -> None:
        """Refuse a manifest that is not the record of the write being undone.

        A manifest is a plain file under the state tree, and every field of it
        was taken as written: the document, the backup and both hashes. Each
        check after this one is made against the manifest's own values, so a
        manifest naming another skill's document, a backup planted beside it
        with its hash, or a ``new_sha256`` changed to the hash of a later edit
        passed them all. Measured before (3.11 and 3.13): ``rollback`` wrote
        the planted text into another skill, or into the candidate's own
        document in place of its original text, or over a later edit, and each
        time exited 0 and recorded the candidate rolled back. The document is
        compared resolved on both sides, as ``earlier_write`` compares it.
        """
        target = Path(manifest["target_path"])
        same_document = path is not None and target.resolve() == Path(path).resolve()
        recorded = (manifest["previous_sha256"], manifest["new_sha256"])
        same_write = recorded == (previous_sha256, new_sha256)
        if same_document and same_write:
            return
        # Each clause is built inside the raise, so the rule that every value a
        # skill store refusal interpolates is named holds over them too.
        raise BackupUnusable(
            f"refusing to roll back: backup manifest {named_path(manifest_path)} is not the "
            "record of this candidate's apply: "
            + "; ".join(
                clause
                for differs, clause in (
                    (
                        not same_document,
                        f"it names {named_path(target)}, not this candidate's "
                        f"document {named_path(path)}",
                    ),
                    (
                        not same_write,
                        "it records the text going from "
                        f"{quoted_value(manifest['previous_sha256'])} to "
                        f"{quoted_value(manifest['new_sha256'])}, not from this "
                        f"candidate's base {quoted_value(previous_sha256)} to its "
                        f"replacement {quoted_value(new_sha256)}",
                    ),
                )
                if differs
            )
            + ". Rollback restores only what this candidate's apply wrote, so no file was "
            "touched."
        )

    def _judge_rollback(
        self, target: Path, backup: Path, expected: str, *, manifest_path: Path
    ) -> tuple[int, str, bytes]:
        """Every check :meth:`rollback` makes before the live file is read.

        Returns the pinned parent handle and leaf the restore goes through, and
        the backup's bytes: read once, so the bytes whose hash is checked here
        are the bytes written. Any refusal here leaves the tree exactly as it was.
        """
        self._assert_document(target)
        self._assert_not_protected(target)
        if backup.is_symlink() or not backup.is_file():
            raise BackupUnusable(
                f"refusing to roll back {named_path(target)}: backup {named_path(backup)} is missing, a symbolic "
                f"link or not a regular file. {named_path(target)} is unchanged."
            )
        # Apply writes the backup and its manifest into one directory under the
        # state tree. Measured before: a manifest naming a backup anywhere else
        # (a protected file, a copy outside the state tree) whose hash matched
        # was read and written into the skill.
        if backup.parent.resolve() != manifest_path.parent.resolve():
            raise BackupUnusable(
                f"refusing to roll back {named_path(target)}: backup {named_path(backup)} is not in the directory of "
                f"its manifest {named_path(manifest_path)}, where apply writes it, so it is not the copy "
                f"this apply took. {named_path(target)} is unchanged."
            )
        data = _read_regular_file(backup)
        if data is None:
            raise BackupUnusable(
                f"refusing to roll back {named_path(target)}: backup {named_path(backup)} is not a regular file. "
                f"{named_path(target)} is unchanged."
            )
        # Apply backs up only a document it read as UTF-8, so a backup that is
        # not is not that copy. Measured before: read with its bad bytes
        # replaced, such a backup hashed as the text it stood in for, was
        # written into the skill, and the read after the write then said
        # nothing was written.
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BackupUnusable(
                f"refusing to roll back {named_path(target)}: backup {named_path(backup)} is not valid UTF-8 at byte "
                f"{exc.start} ({exc.reason}), and apply backs up only a document it read as "
                f"UTF-8, so it is not the copy this apply took. {named_path(target)} is unchanged."
            ) from exc
        stored = sha256_text(text)
        if stored != expected:
            raise BackupUnusable(
                f"refusing to roll back {named_path(target)}: backup {named_path(backup)} hashes {stored[:12]} but the "
                f"manifest records {expected[:12]}. The backup is not the text this apply "
                f"replaced, so restoring it would overwrite the live skill with unverified "
                f"content. {named_path(target)} is unchanged."
            )
        # The manifest recorded a resolved regular file. A symlink now sitting at
        # that path is not that file, and a copy through it would write wherever
        # the link points. Checked with lstat semantics before any hash is read,
        # because hashing follows the link and would report the decoy's bytes.
        if target.is_symlink():
            raise BackupUnusable(
                f"refusing to roll back {named_path(target)}: the recorded target is now a symbolic link, "
                f"not the regular file this apply wrote. Neither the link nor what it points "
                f"to was touched."
            )
        # The leaf is judged above; a directory on the way to it that became a
        # link redirects the restore just as surely, and a same-text twin at the
        # far end would pass the applied-hash check below.
        self._assert_no_link_below_root(target)
        # The recovery command writes where apply writes and nowhere else. Asked
        # after the link checks, which name a link more exactly than this would.
        # Measured before: a manifest edited to name the mirror's copy, with
        # ``new_sha256`` set to that copy's hash, overwrote the mirror and
        # reported it rolled back; apply refuses that target as a mirror.
        self.assert_canonical_target(target, action="roll back")
        if not target.is_file():
            raise BaseHashMismatch(
                f"refusing to roll back {named_path(target)}: the recorded target is not a regular file "
                f"any more, so nothing here can be verified as the text this apply wrote."
            )
        parent_fd, leaf = self._open_parent_pinned(target, verb="roll back")
        return parent_fd, leaf, data

    def _rollback_pinned(
        self,
        parent_fd: int,
        leaf: str,
        target: Path,
        *,
        backup: Path,
        data: bytes,
        expected: str,
        applied: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """The verified half of :meth:`rollback`, through the pinned handle."""
        live_bytes, live_mode = self._read_pinned(parent_fd, leaf, target, verb="roll back")
        live = self._document_sha256(live_bytes, target)
        if live == expected:
            # Already holds the previous text: a second rollback, or a manual
            # `cp` of the backup. Nothing to write; say so rather than rewrite.
            self._snapshot_cache = None
            return {
                "rolled_back": str(target),
                "restored_sha256": live,
                "from_backup": str(backup),
                "already_restored": True,
            }
        if live != applied:
            raise BaseHashMismatch(
                f"refusing to roll back {named_path(target)}: the live file hashes {live[:12]}, but this "
                f"apply wrote {applied[:12]}. The skill was edited after the apply, so the "
                f"backup would overwrite work this candidate never touched. Restore by hand "
                f"from {named_path(backup)} if that is what you want. {named_path(target)} is unchanged."
            )
        # The backup was tightened to 0600 when it was taken, so restoring its
        # bytes must carry the skill's recorded mode or the skill ends up
        # owner-only. A manifest written before the field existed has none, and
        # one edited into a mode the restore cannot honestly use (see
        # _restorable_mode) is treated the same: the live file's mode.
        mode = _restorable_mode(manifest.get("target_mode"), live_mode)
        try:
            _write_pinned(parent_fd, leaf, data, mode=mode)
        except OSError as exc:
            # Measured before: a skill directory at 0555 made rollback fall
            # with a raw PermissionError on the temp file, exit 70.
            raise SkillWriteFailed(
                f"refusing to roll back: {named_path(target)} could not be written ({exc.strerror or exc}). "
                f"It still holds the text this apply wrote, and the backup {named_path(backup)} is "
                "unchanged."
            ) from exc
        restored, found = self._read_back(parent_fd, leaf, target)
        if restored != expected:
            # The backup has been written over the skill by now, so a refusal
            # saying nothing was touched would be false: say what is there.
            raise BaseHashMismatch(
                f"rollback verification failed: {named_path(target)} was rewritten from {named_path(backup)}, but "
                f"reading it back found {found}, not {expected[:12]}. Compare the two by hand."
            )
        self._snapshot_cache = None
        return {
            "rolled_back": str(target),
            "restored_sha256": restored,
            "from_backup": str(backup),
            "already_restored": False,
        }

    # -- internals -------------------------------------------------------
    def _read_backup_manifest(self, manifest_path: Path) -> dict[str, Any]:
        """Load a backup manifest, refusing anything it cannot be read from.

        A manifest is a plain JSON file under the state tree, so it can be
        truncated by a crash mid-write or edited by hand. Every field the restore
        depends on is required here, and a missing one is a named refusal rather
        than a ``KeyError`` from the middle of the restore.
        """
        from ..jsonio import read_json

        path = Path(manifest_path)
        # Judged before it is opened: a symlink would be read through, and a
        # fifo would hold the process until something wrote to it.
        if path.is_symlink() or not path.is_file():
            raise BackupUnusable(
                f"refusing to roll back: backup manifest {named_path(path)} is not a regular file "
                "(a symlink, a fifo or a missing entry is never read). No file was touched."
            )
        try:
            manifest = read_json(path)
        except (OSError, ValueError) as exc:
            raise BackupUnusable(
                f"refusing to roll back: backup manifest {named_path(path)} could not be read "
                f"({why_unreadable(exc)}). No file was touched."
            ) from exc
        if not isinstance(manifest, dict):
            raise BackupUnusable(
                f"refusing to roll back: backup manifest {named_path(path)} is not a JSON object. "
                "No file was touched."
            )
        # Each is text in a manifest this tool wrote. A number or an object
        # there used to be coerced with str() and refused later under another
        # name (a hash mismatch, a non-document write of "7"), never this one.
        missing = [
            field
            for field in ("target_path", "backup_path", "previous_sha256", "new_sha256")
            if not isinstance(manifest.get(field), str) or not manifest[field].strip()
        ]
        if missing:
            raise BackupUnusable(
                f"refusing to roll back: backup manifest {named_path(path)} is missing "
                f"{', '.join(missing)}, or holds something other than text there. "
                "No file was touched."
            )
        return manifest

    def assert_write_target(
        self, path: Path, *, action: str, skill_id: str = "", candidate_id: str = ""
    ) -> Path:
        """Every judgement the write makes of its target before it opens it, in its order.

        ``write_canonical`` makes them here, ``apply`` asks them before it makes
        the backup directory, and ``review`` asks the same with
        ``action="review"`` before it shows a token, so none of the three can
        ask different questions. Measured before: a target spelled through a
        symbolic link (a skill folder that is a link below its root, a folder
        outside every root that links into one, or an alias of the root) made
        ``review`` print a token and an apply command that ``apply`` then
        refused, leaving an empty backup directory; and a skill listed in
        ``[guard].protected_skill_ids`` after its candidate was validated was
        reviewed and written, the list being read only when a proposal is
        made. ``skill_id`` is the skill the caller's
        record names, and ``candidate_id`` the candidate ``review`` or ``apply``
        is judging, which the refusal of a listed skill (``listed_as_protected``),
        of a link below the root (``reached_through_a_link``) and of a target
        that is not the canonical copy (``canonical_target_refused``) name.
        Returns the target as the write addresses it: resolved when it exists.
        """
        replace = "replace" if action == "write" else action
        write = "write" if action == "write" else action
        # Judged on the path as spelled, with lstat semantics, before anything
        # resolves it: a link planted below the root between proposal and apply
        # would otherwise be followed to a twin with the same text, and the
        # hash check would pass while the write landed on the wrong skill.
        self._assert_no_link_below_root(Path(path), verb=replace, candidate_id=candidate_id)
        # Resolved once, so every later step (hash, backup, manifest, verify)
        # describes the same file even when the caller spelled the path
        # differently. Two spellings of one path is how a check and the write it
        # guards end up looking at different things.
        target = Path(path).resolve() if Path(path).exists() else Path(path)
        self._assert_document(target, verb=write)
        self._assert_not_protected(target, verb=write)
        group = self.assert_canonical_target(target, action=action, candidate_id=candidate_id)
        self._assert_not_a_protected_skill(
            target, group, skill_id, verb=write, candidate_id=candidate_id
        )
        return target

    def mirror_paths_of(self, path: Path) -> tuple[str, ...]:
        """The mirror copies the live inventory holds of the skill whose canonical copy is ``path``.

        ``review`` lists these as the copies ``apply`` does not write, and
        ``apply`` records them in its receipt as not written, in place of the
        list the candidate record carries, which neither the approval token nor
        the integrity check binds. Measured before: a record's list edited after
        validation was printed by ``review`` as written (a token in a path
        whole, a path of a megabyte, ten thousand paths), an emptied list hid
        the mirror the tree holds, and ``apply`` recorded the edited list in its
        receipt. Empty when ``path`` is not the canonical copy of a skill, which
        ``review`` and a first run of ``apply`` have refused by then; the run
        that finishes an earlier apply judges no target, and asks this before it
        records the write.
        """
        kind, group = classify_target(self.groups(refresh=True), Path(path).resolve())
        if kind != "canonical" or group is None:
            return ()
        return tuple(str(m.path) for m in group.mirrors)

    def _assert_not_a_protected_skill(
        self,
        target: Path,
        group: SkillGroup | None,
        skill_id: str,
        *,
        verb: str,
        candidate_id: str = "",
    ) -> None:
        """Refuse a skill listed in ``[guard].protected_skill_ids``.

        The skill the inventory finds at the target and the one the caller's
        record names are both judged, so a record naming another skill does not
        carry a write into a listed one, nor the other way round.
        """
        for named in (group.skill_id if group is not None else "", skill_id):
            if named and named in self._protected_skill_ids:
                raise SelfMutationRefused(listed_as_protected(target, named, candidate_id, verb))

    def _assert_document(self, target: Path, *, verb: str = "write") -> None:
        """Only the skill document is ever writable.

        A skill directory routinely contains hooks and launchers, so a write path
        that accepted an arbitrary filename inside it would be a code-execution
        primitive wearing a docs-change costume.
        """
        if target.name != SKILL_FILENAME:
            raise NonDocumentWriteRefused(
                f"refusing to {verb} {named_path(target)}: this tool only ever replaces {SKILL_FILENAME}. "
                "Executable payloads in a skill directory (hooks, launchers, MCP wrappers) are "
                "privileged code and are never edited by an automated proposal."
            )

    def _assert_not_protected(self, target: Path, *, verb: str = "write") -> None:
        resolved = target.resolve()
        for guard in self._protected:
            if resolved == guard or guard in resolved.parents:
                raise SelfMutationRefused(
                    f"refusing to {verb} {named_path(target)}: it is inside a protected path ({named_path(guard)}). "
                    "This tool never edits its own source, config, or rubrics -- a loop that "
                    "can rewrite its own scoring criteria has no gate at all."
                )

    def _assert_no_link_below_root(
        self, spelled: Path, *, verb: str = "replace", candidate_id: str = ""
    ) -> None:
        """Refuse a target reached through a symbolic link below its root.

        The root itself may legitimately be a link (a skills directory pointing
        at a checkout), so only the components beneath the configured root are
        judged, each with ``lstat`` semantics. Shared by the write and the
        rollback, which are the two places a skill file is ever replaced.
        """
        spelled = spelled if spelled.is_absolute() else spelled.absolute()
        for root in self._roots:
            for root_form in (Path(root.path), Path(root.path).resolve()):
                try:
                    below = spelled.relative_to(root_form)
                except ValueError:
                    continue
                cursor = root_form
                for part in below.parts:
                    cursor = cursor / part
                    if cursor.is_symlink():
                        raise LinkedTargetRefused(
                            reached_through_a_link(spelled, cursor, candidate_id, verb)
                        )
                return
        # No root form is a prefix of the spelling, yet it may still resolve into
        # a canonical root: the spelling then reaches the skill through a link
        # above or outside the root, which the walk above cannot see. Default
        # deny. A spelling that resolves nowhere canonical is left to
        # ``_assert_canonical``, which refuses it as a mirror or as unknown.
        resolved = spelled.resolve()
        for root in self._roots:
            # Every editable root, not only one marked canonical: a skill
            # present only in an unmarked editable root is written there, and
            # when none is marked canonical the highest-precedence editable
            # root is the canonical one for every skill.
            if not root.editable:
                continue
            root_resolved = Path(root.path).resolve()
            if resolved == root_resolved or root_resolved in resolved.parents:
                raise LinkedTargetRefused(
                    f"refusing to {verb} {named_path(spelled)}: it is not spelled under any configured skill "
                    f"root yet resolves to {named_path(resolved)} inside root {root.id}, so it reaches the "
                    f"skill through a symbolic link. Spell the target as the root does. Nothing "
                    f"was touched."
                )

    def _open_parent_pinned(self, spelled: Path, *, verb: str) -> tuple[int, str]:
        """A handle on the document's directory, plus the document's name.

        The root is opened as it is (it may legitimately be a link); every
        component below it is opened relative to the previous handle with
        ``O_DIRECTORY | O_NOFOLLOW``, so a link on the way is ``ELOOP`` rather
        than a redirection, and a directory swapped for a link after this
        returns is simply not what the handle refers to. The caller closes the
        handle. The spelled path is required to lie under a configured root,
        which :meth:`_assert_no_link_below_root` has already established.
        """
        if not PINNED_IO_SUPPORTED:
            raise LinkedTargetRefused(
                f"refusing to {verb} {named_path(spelled)}: this platform cannot address a file through "
                f"a directory handle, so the write could not be pinned to the directory that "
                f"was checked. Nothing was touched."
            )
        spelled = spelled if spelled.is_absolute() else spelled.absolute()
        for root in self._roots:
            for root_form in (Path(root.path), Path(root.path).resolve()):
                try:
                    below = spelled.relative_to(root_form)
                except ValueError:
                    continue
                if not below.parts:
                    raise NonDocumentWriteRefused(
                        f"refusing to {verb} {named_path(spelled)}: it is the skill root {root.id} itself, "
                        f"not a document in it. Nothing was touched."
                    )
                fd = os.open(str(root_form), os.O_RDONLY | os.O_DIRECTORY)
                # The directory each refusal names is the one being opened, the
                # root joined to every component so far. Measured before: it was
                # the root joined to the last component alone, so a copy at
                # ``<root>/x/tdd`` named ``<root>/tdd``, a directory that need
                # not exist.
                walked = root_form
                try:
                    for part in below.parts[:-1]:
                        walked = walked / part
                        try:
                            nxt = os.open(
                                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                            )
                        except OSError as exc:
                            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                                raise LinkedTargetRefused(
                                    f"refusing to {verb} {named_path(spelled)}: {named_path(walked)} is a "
                                    f"symbolic link or not a directory on the way to the "
                                    f"document, judged on the open handle. Nothing was touched."
                                ) from exc
                            raise BaseHashMismatch(
                                f"refusing to {verb} {named_path(spelled)}: {named_path(walked)} could not be "
                                f"opened ({exc.strerror}). Nothing was touched."
                            ) from exc
                        os.close(fd)
                        fd = nxt
                except BaseException:
                    os.close(fd)
                    raise
                return fd, below.parts[-1]
        raise LinkedTargetRefused(
            f"refusing to {verb} {named_path(spelled)}: it lies under no configured skill root, so there "
            f"is no directory handle to pin it to. Nothing was touched."
        )

    def _read_back(self, parent_fd: int, leaf: str, target: Path) -> tuple[str | None, str]:
        """The hash of what a write just left at ``leaf``, and how to name it.

        The verification read comes after the write, so it must not raise the
        refusals the reads before a write raise: each says nothing was touched,
        which is false by then. A document that cannot be opened, is not a
        regular file or is not UTF-8 comes back as ``None`` with what was found,
        for the caller to undo or report; otherwise its hash and the short form.
        """
        try:
            data, _mode = self._read_pinned(parent_fd, leaf, target, verb="verify")
            sha = self._document_sha256(data, target)
        except (BaseHashMismatch, LinkedTargetRefused, UndecodableDocument) as exc:
            cause = exc.__cause__
            if isinstance(cause, OSError) and cause.strerror:
                why = cause.strerror
            elif isinstance(cause, UnicodeDecodeError):
                why = "not valid UTF-8"
            else:
                why = "not a regular file"
            return None, f"an unreadable document ({why})"
        return sha, sha[:12]

    def _read_pinned(
        self, parent_fd: int, leaf: str, target: Path, *, verb: str
    ) -> tuple[bytes, int]:
        """The document's bytes and mode, read through the pinned handle.

        ``O_NOFOLLOW`` on the leaf, then ``fstat`` on what was opened: a link
        or anything but a regular file at the name is a refusal, not a read of
        whatever it points to.
        """
        try:
            fd = os.open(
                leaf, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0), dir_fd=parent_fd
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise LinkedTargetRefused(
                    f"refusing to {verb} {named_path(target)}: the document is a symbolic link, judged on "
                    f"the open handle. Neither the link nor its target was touched."
                ) from exc
            raise BaseHashMismatch(
                f"refusing to {verb} {named_path(target)}: the document could not be opened "
                f"({exc.strerror}). Nothing was touched."
            ) from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise BaseHashMismatch(
                    f"refusing to {verb} {named_path(target)}: the document is not a regular file. "
                    f"Nothing was touched."
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(fd)
        return b"".join(chunks), stat.S_IMODE(info.st_mode)

    def assert_canonical_target(
        self, target: Path, *, action: str, candidate_id: str = ""
    ) -> SkillGroup | None:
        """Refuse a mirror copy, and refuse anything we cannot identify at all.

        Resolution happens HERE because this is the adapter layer; the domain
        comparison is pure text. Both sides must be resolved or the compare is
        meaningless: snapshots carry resolved paths, so an unresolved caller path
        classified as ``unknown`` and slipped straight through.

        ``unknown`` is refused rather than allowed. It is a genuinely surprising
        state -- every legitimate apply targets a path that came from a snapshot --
        and treating "I do not recognise this file" as permission is how an
        interlock becomes decorative.

        ``review`` asks the same question with ``action="review"`` so that it
        never shows a token for a target the write would refuse. Measured
        before: with the canonical root re-pointed at a copy of the tree, the
        old file still on disk outside every root, ``review`` printed a token
        and an apply command, and ``apply`` refused the unrecognised target.

        ``candidate_id`` is the candidate ``review`` or ``apply`` is judging,
        which ``canonical_target_refused`` names in each refusal here;
        ``rollback`` and the write's own check pass none, and keep the words
        that name no candidate.
        """
        resolved = Path(target).resolve()
        kind, group = classify_target(self.groups(refresh=True), resolved)
        if kind == "mirror" and group is not None and group.canonical is None:
            # Every copy of a skill with no canonical copy classifies as a
            # mirror, but calling it one, of an unknown source, misdescribes
            # it: the resolver's own reason says what is unsettled.
            raise MirrorEditRefused(
                canonical_target_refused(
                    target,
                    f"skill {quoted_value(group.skill_id)} has no canonical copy "
                    f"({group.canonical_reason}); that must be settled before anything is "
                    "written.",
                    candidate_id,
                    action,
                )
            )
        if kind == "mirror":
            canonical = group.canonical.path if group and group.canonical else "<unknown>"
            raise MirrorEditRefused(
                canonical_target_refused(
                    target,
                    f"it is a mirror copy. The canonical source is {named_path(canonical)}. "
                    "Editing a mirror either gets overwritten by the next sync or never reaches "
                    "the agent that loads the canonical file.",
                    candidate_id,
                    action,
                )
            )
        if kind == "unknown":
            # Discovery says why it did not take a copy it saw; the refusal
            # says so too, rather than leave the operator to find it in
            # ``doctor``. Measured before: a copy whose directory could not be
            # listed, or that sat below another skill's directory, was refused
            # by ``review``, ``apply`` and ``rollback`` alike as "not a skill
            # document in any configured root", with no word of either.
            passed = self._passed_over_on_the_way(target, resolved)
            if passed is not None:
                raise MirrorEditRefused(
                    canonical_target_refused(
                        target,
                        f"discovery passed over {named_path(passed.path)}, which is "
                        f"{passed.reason}, so it is not a skill document in any configured root "
                        "and this tool cannot tell whether it is canonical. An unrecognised "
                        "target is refused rather than assumed safe.",
                        candidate_id,
                        action,
                    )
                )
            raise MirrorEditRefused(
                canonical_target_refused(
                    target,
                    "it is not a skill document in any configured root, so this tool cannot "
                    "tell whether it is canonical. An unrecognised target is refused rather than "
                    "assumed safe.",
                    candidate_id,
                    action,
                )
            )
        return group

    def _passed_over_on_the_way(self, target: Path, resolved: Path) -> PassedOverDocument | None:
        """The entry discovery passed over at ``target`` or above it, if any.

        Read after a refresh, so it is what this run's discovery saw. The
        first entry that applies is returned; discovery does not descend below
        a directory it passes over, so within one root only one can. The
        target is matched as spelled and as resolved: entries are recorded
        under the root's resolved path, and a link at the document's own place
        is recorded as the link.
        """
        forms = (Path(os.path.abspath(target)), resolved)
        for document in self._passed_over:
            if any(document.path == form or document.path in form.parents for form in forms):
                return document
        return None

    @staticmethod
    def _entries_named_like_a_document(
        base: Path, *, unlistable: list[tuple[Path, str]] | None = None
    ) -> list[Path]:
        """Every entry below ``base`` named ``SKILL.md``, whatever it is.

        ``os.walk`` rather than ``rglob``: on Python 3.11 a glob with a literal
        last component drops any entry whose ``exists()`` is false, so a
        dangling link named ``SKILL.md`` was never seen, and what is never seen
        cannot be reported. Links to directories are not descended (the same
        as ``rglob``), but a link or a directory that carries the document's
        name is still an entry here, so it can be judged and named. A
        directory the walk cannot list is appended to ``unlistable`` with the
        error, instead of being stepped over in silence one level up.
        """
        found: list[Path] = []

        def could_not_list(error: OSError) -> None:
            if unlistable is not None:
                where = Path(error.filename) if error.filename else base
                unlistable.append((where, error.strerror or type(error).__name__))

        for dirpath, dirnames, filenames in os.walk(
            base, followlinks=False, onerror=could_not_list
        ):
            for name in (*dirnames, *filenames):
                if name == SKILL_FILENAME:
                    found.append(Path(dirpath) / name)
        return sorted(found)

    def _iter_skill_files(
        self,
        base: Path,
        *,
        root_id: str,
        passed_over: list[PassedOverDocument],
    ) -> list[Path]:
        found: list[Path] = []

        def skip(candidate: Path, reason: str, *, copy_of_a_skill: bool = True) -> None:
            # An entry at a document's place is a copy this run cannot see and
            # carries the skill id it would have had; a directory or an entry at
            # a depth discovery never takes is not a copy of anything. Measured
            # before: an unlistable directory inside ``tdd/`` was recorded with
            # ``tdd`` as its skill and withheld the canonical slot of a readable
            # document, dropping a true divergence and refusing an approved apply.
            skill_id = self._skill_id_at(base, candidate) if copy_of_a_skill else None
            passed_over.append(
                PassedOverDocument(
                    root_id=root_id, path=candidate, reason=reason, skill_id=skill_id
                )
            )

        unlistable: list[tuple[Path, str]] = []
        entries = self._entries_named_like_a_document(base, unlistable=unlistable)
        # Every directory that carries an entry named like a document, whatever
        # the entry turns out to be: a document below one of these is inside
        # another skill's directory, and that holds whether or not the owning
        # document reads.
        skill_dirs = {entry.parent for entry in entries}
        for directory, error in unlistable:
            skip(
                directory,
                f"a directory that could not be listed ({error})",
                copy_of_a_skill=False,
            )
        for candidate in entries:
            # A path the system will not take is passed over by its length, as a
            # copy of the skill it would be: the listing that found it holds
            # names only, and this ``lstat`` is the first call on the whole path.
            # Any other refusal of it is raised as before.
            try:
                is_link = candidate.is_symlink()
            except OSError as exc:
                if exc.errno != errno.ENAMETOOLONG:
                    raise
                skip(
                    candidate,
                    f"a path of {len(os.fsencode(str(candidate)))} bytes, longer than this "
                    f"system opens ({exc.strerror})",
                )
                continue
            if is_link and not candidate.exists():
                skip(candidate, "a symlink whose target does not exist or cannot be reached")
                continue
            try:
                rel = candidate.resolve().relative_to(base)
            except ValueError:
                # Symlink escaping the root. Skipped unless explicitly allowed.
                if not self._follow_symlinks:
                    skip(candidate, "a symlink that resolves outside its root")
                    continue
                rel = Path(candidate.name)
            if len(rel.parts) > self._max_depth:
                skip(
                    candidate,
                    f"deeper than max_depth={self._max_depth} below the root",
                    copy_of_a_skill=False,
                )
                continue
            # A skill is ``<root>/<name>/SKILL.md``: the name is the directory.
            # A document at the root itself has no directory of its own to be
            # named by. Measured before: it was inventoried under the root
            # directory's basename, so the skill's identity changed with the
            # folder's name, and a proposal against it would have written the
            # root's own ``SKILL.md``.
            if len(rel.parts) < 2:
                skip(
                    candidate,
                    "at the root itself, not in a skill directory; a skill is "
                    f"<root>/<name>/{SKILL_FILENAME}",
                    copy_of_a_skill=False,
                )
                continue
            # A document below another skill's directory is that skill's
            # material (a template, an example, a reference), not a skill of
            # its own. The owner is judged by the entry at its place, whatever
            # that entry is (a dangling link named like a document still marks
            # the directory as a skill's), so the owner it names need not have
            # been inventoried. Measured before: ``tdd/sub/SKILL.md`` was
            # inventoried as a skill named ``sub``, and a proposal against it
            # would have written inside ``tdd``.
            owner = self._enclosing_skill_dir(base, candidate, skill_dirs)
            if owner is not None:
                skip(
                    candidate,
                    f"below {named_path(owner / SKILL_FILENAME)}, the place of skill "
                    f"{quoted_value(owner.name)}; a "
                    f"document under another skill's directory is that skill's material, not a skill",
                    copy_of_a_skill=False,
                )
                continue
            if is_link and not self._follow_symlinks:
                skip(candidate, "a symlink; this store does not follow links")
                continue
            # A fifo or a device named like a document is not one: reading it
            # would wait forever or read nonsense. Judged by stat, not by open.
            if not candidate.is_file():
                skip(candidate, f"not a regular file ({_entry_kind(candidate)})")
                continue
            found.append(candidate)
        return found

    @staticmethod
    def _enclosing_skill_dir(base: Path, document: Path, skill_dirs: set[Path]) -> Path | None:
        """The nearest directory strictly above ``document``'s own, below
        ``base``, that carries an entry named like a document; None if there is
        none. Judged on the entry's own place, no link followed."""
        for ancestor in document.parent.parents:
            if ancestor == base or base not in ancestor.parents:
                return None
            if ancestor in skill_dirs:
                return ancestor
        return None

    def _skill_id_at(self, base: Path, document: Path) -> str | None:
        """The skill id discovery would give a ``SKILL.md`` at this place, or None.

        Judged on the entry's own place under the root (no link is followed):
        ``<root>/<skill>/SKILL.md`` or deeper up to ``max_depth``, named by the
        parent directory as ``_snapshot`` names it. A deeper one is never
        taken, so it is not a copy of a skill, and neither is a document at the
        root itself, which discovery passes over: the root has no skill
        directory to be named by.
        """
        if document.name != SKILL_FILENAME:
            return None
        try:
            rel = document.relative_to(base)
        except ValueError:
            return None
        if len(rel.parts) < 2 or len(rel.parts) > self._max_depth:
            return None
        return rel.parts[-2]

    def _payload_files(
        self, skill_dir: Path
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        """Enumerate executable payloads beside a skill document, and the
        directories below it that could not be listed.

        No shell: filenames containing literal backslashes, spaces or quotes
        are common in real skill trees and any shell-based scan mangles them
        silently. ``os.walk`` with ``onerror`` rather than ``rglob``: ``rglob``
        steps over a directory it cannot open without a word, and measured
        before this, a ``scripts/`` at mode 000 or 111 (its files still run by
        name) made the hook inside it vanish from the inventory, so the
        candidate lost its elevated-risk note and the receipt counted no
        payload. What such a directory holds is unknown; it is named instead.
        """
        found: list[str] = []
        unlistable: list[tuple[str, str]] = []

        def could_not_list(error: OSError) -> None:
            where = Path(error.filename) if error.filename else skill_dir
            try:
                rel = where.relative_to(skill_dir).as_posix()
            except ValueError:
                rel = str(where)
            unlistable.append((rel or ".", error.strerror or type(error).__name__))

        entries: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(
            skill_dir, followlinks=False, onerror=could_not_list
        ):
            entries.extend(Path(dirpath) / name for name in (*dirnames, *filenames))
        entries.sort()
        for entry in entries:
            try:
                if entry.name == SKILL_FILENAME:
                    continue
                # A symlink is judged on its own name and on the mode of what it
                # points to. Skipping links outright made a symlinked launcher
                # invisible to the inventory, which is the cheapest way to put a
                # binary beside a skill. A link to a directory is still skipped:
                # rglob does not descend into it, and the directory is not itself
                # an executable.
                is_link = entry.is_symlink()
                if entry.is_dir() and not is_link:
                    continue
                rel = entry.relative_to(skill_dir).as_posix()
                if is_payload_name(entry.name):
                    found.append(rel)
                    continue
                if is_link and entry.is_dir():
                    continue
                # stat() follows the link; a dangling link raises and is skipped
                # here (a dangling non-payload name is noise, not a payload).
                if entry.stat().st_mode & 0o111:
                    found.append(rel)
            except (OSError, ValueError):
                continue
        return tuple(found), tuple(unlistable)

    def _snapshot(self, root: SkillRoot, base: Path, skill_file: Path, raw: bytes) -> SkillSnapshot:
        undecodable: list[str] = []
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            # Inventoried so the operator sees it, with the bytes replaced so the
            # rest of the document still reads; the strict read on the write
            # path refuses it, and the warning says so here rather than there.
            text = raw.decode("utf-8", errors="replace")
            undecodable = [
                f"not valid UTF-8 at byte {exc.start} ({exc.reason}): inventoried with the bytes "
                f"replaced; proposals against it are refused until the encoding is fixed"
            ]
        fields, has_open, warnings = parse_frontmatter(text)
        warnings = undecodable + warnings
        rel = skill_file.resolve().relative_to(base)
        # ``_iter_skill_files`` passes a root-level document over, so the
        # parent here is always a directory below the root.
        skill_id = rel.parts[-2]
        declared = fields.get("name", "")
        if declared and declared != skill_id:
            # The name is the document's own text, quoted masked and bounded
            # (``quoted_value``). Measured before: with ``!r`` a name of
            # 2,000,000 characters made one ``skills`` line of 2,000,070 and a
            # token in it printed whole, in the text and in ``--json``.
            warnings = warnings + [
                f"frontmatter name {quoted_value(declared)} differs from directory name "
                f"{skill_id!r}"
            ]
        disable = fields.get("disable-model-invocation", "").strip().lower() in ("true", "yes", "1")
        if disable:
            warnings = warnings + [
                "disable-model-invocation is true: the skill is hidden from the "
                "auto-loaded listing and can only be invoked by name"
            ]
        payloads, unlistable = self._payload_files(skill_file.parent)
        if payloads:
            warnings = warnings + [
                f"{len(payloads)} executable payload file(s) live in this skill directory "
                f"({', '.join(payloads[:5])}"
                + (", ..." if len(payloads) > 5 else "")
                + "): changes here are privileged, and this tool never writes them"
            ]
        if unlistable:
            warnings = warnings + [
                f"{len(unlistable)} director{'y' if len(unlistable) == 1 else 'ies'} in this "
                "skill directory could not be listed ("
                + ", ".join(f"{d} ({e})" for d, e in unlistable[:5])
                + (", ..." if len(unlistable) > 5 else "")
                + "): whether executable payloads live there is unknown, so the payload "
                "inventory above is incomplete"
            ]
        return SkillSnapshot(
            skill_id=skill_id,
            root_id=root.id,
            path=skill_file,
            sha256=sha256_text(text),
            size_bytes=len(raw),
            frontmatter_name=declared,
            frontmatter_description=fields.get("description", ""),
            has_opening_delimiter=has_open,
            disable_model_invocation=disable,
            payload_files=payloads,
            unlistable_dirs=unlistable,
            warnings=tuple(warnings),
        )
