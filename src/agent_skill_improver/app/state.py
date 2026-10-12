"""On-disk state: private data, checkpoints, per-record quarantine and defer.

Everything under the state directory is local-only and gitignored. That is not
a stylistic choice: raw exports are private conversation content, and the whole
point of this tool is that it can read them without ever publishing them.

Gitignoring it is only half of "private", and the weaker half. The state root and
every directory under it are **0700**, and every artifact in it is **0600**, set
explicitly rather than left to the umask -- a permissive umask is precisely the
condition under which a directory created with ``mkdir`` lands 0755 and the
transcripts inside it become readable by every account on the machine. This is a
multi-user concern on any shared or managed host, and it is not hypothetical: a
second local account is the normal state of a corporate laptop.

Existing state is **repaired, not tolerated**. ``ensure`` walks the tree, chmods
anything wider than the policy, then *re-reads* the modes and raises
``SafetyRefusal`` if any is still wrong. Verifying after repairing matters
because the repair can fail silently -- on a filesystem that does not carry POSIX
modes, ``chmod`` is accepted and does nothing, and a tool that only called chmod
would report a private state tree while writing world-readable transcripts.

Repairing modes is only meaningful if the tree being repaired is the tree that
was configured. ``chmod``, ``mkdir(exist_ok=True)`` and every later write follow
a symlink, so a state root -- or a mandatory subdirectory such as ``raw`` --
precreated as a link to somewhere else silently redirects the whole private tree:
the transcripts land outside the configured directory and the repair walk chmods a
directory this tool does not own. The layout is therefore **verified before it is
followed**: the root and every declared subdirectory must be a real directory,
measured with ``lstat``, or the store refuses. Dynamically named directories such
as a backup stamp are proven to be direct children of an already-verified
mandatory directory before they are created or chmodded. An arbitrary *leaf*
symlink inside the tree is still skipped rather than followed, which leaves its
target untouched.

The per-record split is the other important part. A defective record is
*quarantined* on its own and a record we simply could not reach is *deferred* on
its own. Neither is allowed to veto the batch. A verification pass that returns
one verdict for a whole inbox becomes a ratchet the moment its failure rate is
non-zero: every later record inherits the veto, the queue grows without bound,
and nothing is ever promoted again.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..clock import Clock, iso
from ..domain.candidate import (
    STATUS_APPLIED,
    STATUS_BLOCKED_RUN,
    STATUS_GATE_FAILED,
    STATUS_PROPOSED,
    STATUS_ROLLED_BACK,
    STATUS_VALIDATED,
    Candidate,
    validate_candidate_id,
)
from ..errors import ConfigError, SafetyRefusal, StateUnreadable
from ..jsonio import FILE_MODE, loads_untrusted, write_json, write_text_atomic

#: Owner-only on the state root and every directory beneath it.
DIR_MODE = 0o700

#: Owner-only on every artifact. Same value as :data:`jsonio.FILE_MODE`, named
#: here as well because this module also writes non-JSON (the README).
STATE_FILE_MODE = FILE_MODE

SUBDIRS = (
    "raw",
    "conversations",
    "scorecards",
    "failures",
    "candidates",
    "quarantine",
    "refusals",
    "deferred",
    "receipts",
    "backups",
    "checkpoints",
    "locks",
    # Per-call scratch for sandboxed model roles. Lives under the state tree
    # rather than the system temp directory so private scratch inherits the same
    # 0700 guarantee as everything else derived from a transcript.
    "sandbox",
)

README_TEXT = (
    "This directory holds LOCAL-ONLY data: raw conversation exports,\n"
    "normalized transcripts, scorecards, candidate patches, backups and\n"
    "run receipts. It is gitignored on purpose and must never be committed,\n"
    "synced, or pasted anywhere. Delete it freely; the loop rebuilds it.\n"
    "\n"
    "Modes are enforced: this directory and every subdirectory are 0700, and\n"
    "every file is 0600. `asi doctor` re-checks and repairs them.\n"
)

OPEN_STATUSES = (STATUS_PROPOSED, STATUS_VALIDATED)
CLOSED_STATUSES = (STATUS_APPLIED, STATUS_GATE_FAILED, STATUS_ROLLED_BACK, STATUS_BLOCKED_RUN)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(value: str) -> str:
    """A readable filesystem-safe stem. Use ``keyed_name`` for exact IDs."""
    cleaned = _UNSAFE.sub("_", value).strip("._-")
    return cleaned or "unnamed"


def keyed_name(value: str) -> str:
    """A readable, collision-resistant stem for an exact untrusted identifier."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{safe_name(value)[:96]}--{digest}"


@dataclass(frozen=True)
class UnreadableRecord:
    """A state record on disk that could not be read as one, with why.

    First kept for candidates, now also for day checkpoints. The listing used
    to drop these in silence. Measured: with one half-written
    record under ``candidates/``, ``asi candidates --all`` printed ``0
    candidate(s)``, ``doctor`` reported nothing, and the next ``daily`` run
    counted zero open candidates for that skill and proposed a competitor,
    which is the duplicate the per-skill cap exists to prevent.
    """

    path: Path
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "reason": self.reason}


@dataclass(frozen=True)
class CandidateScan:
    """Every record under ``candidates/``: the ones that read, and the ones that did not."""

    readable: tuple[Candidate, ...]
    unreadable: tuple[UnreadableRecord, ...]


#: The kinds of record a run sets aside, each in its own directory, in the
#: order the listing prints them: what to investigate, then what the guard
#: refused, then what is waiting for a retry.
SET_ASIDE_KINDS: tuple[tuple[str, str], ...] = (
    ("quarantined", "quarantine"),
    ("refused", "refusals"),
    ("deferred", "deferred"),
)


@dataclass(frozen=True)
class SetAsideRecord:
    """One record a run set aside: quarantined, refused or deferred, with why.

    The ``daily`` text summary names the records *that run* set aside; this is
    the same information read back from the state tree, for every run, so an
    operator reading yesterday's ``quarantined=3`` does not have to open the
    reason files (whose names flatten the stage) to learn what and why.
    """

    kind: str
    record_id: str
    stage: str
    reason: str
    at: str
    attempts: int | None
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "record_id": self.record_id,
            "stage": self.stage,
            "reason": self.reason,
            "at": self.at,
            "attempts": self.attempts,
            "path": str(self.path),
        }


@dataclass(frozen=True)
class LastRun:
    """What the newest daily receipt says, for the health questions the
    operations doc asks: is the newest receipt recent, did the run detect a
    skill tree mutation, and does it name a confinement mode."""

    day: str
    status: str
    finished_at: str
    dry_run: bool
    mutation_detected: bool | None
    isolation_mode: str
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "status": self.status,
            "finished_at": self.finished_at,
            "dry_run": self.dry_run,
            "mutation_detected": self.mutation_detected,
            "isolation_mode": self.isolation_mode,
            "path": str(self.path),
        }


@dataclass(frozen=True)
class ReceiptScan:
    """The newest daily receipt that reads, and every receipt file that did not."""

    last_run: LastRun | None
    unreadable: tuple[UnreadableRecord, ...]


@dataclass(frozen=True)
class SetAsideScan:
    """Every set-aside record on disk: the ones that read, and the ones that did not."""

    readable: tuple[SetAsideRecord, ...]
    unreadable: tuple[UnreadableRecord, ...]


@dataclass(frozen=True)
class RecordStatus:
    record_id: str
    status: str
    stage: str
    reason: str = ""
    at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "status": self.status,
            "stage": self.stage,
            "reason": self.reason,
            "at": self.at,
        }


#: The single run lock ``daily`` takes, under the ``locks`` subdirectory.
LOCK_NAME = "daily.lock"

# Field and expected type; a missing field defaults to the type's empty value.
# The shapes ``run_daily`` relies on when it merges a day's checkpoint; anything
# else is refused by name in ``read_checkpoint``.
CHECKPOINT_FIELDS: tuple[tuple[str, type], ...] = (
    ("runs", int),
    ("records", dict),
    ("skills", dict),
    ("candidates", list),
)


class StateStore:
    """Owns the state directory layout. Every write is atomic.

    With ``dry_run`` set, every artifact write is *withheld*: the method computes
    the path it would have written, records it in ``withheld_writes`` and returns
    without touching the disk. Reads are unaffected. The daily pass switches this
    on for a run without ``--commit``, so "a dry run persists nothing" is a
    property of the store rather than a discipline every call site has to keep.
    The layout itself (owner-only directories and the README) is still created,
    and the run lock is still taken, because neither carries private content and
    a dry run must not interleave with a committed one.
    """

    def __init__(self, root: Path, *, clock: Clock | None = None, dry_run: bool = False) -> None:
        self.root = Path(root)
        self._clock = clock
        self.dry_run = dry_run
        #: Paths (relative to the root) a committed run would have written.
        self.withheld_writes: list[str] = []
        #: Paths (relative to the root) whose mode the last ``ensure()`` reset to
        #: owner-only. Repair is never withheld: it is the one change to existing
        #: state a dry run makes, and it writes no content.
        self.repaired_paths: list[str] = []
        #: Resets made by ``dir()`` that no repair walk has reported yet. ``dir()``
        #: runs before ``ensure()`` in every command (the sandbox scratch root is
        #: taken while wiring), so without this hand-off a root it reset would be
        #: measured as already fine and never reported.
        self._pending_repairs: list[str] = []

    def _persist(self, path: Path, write: Callable[[], object]) -> Path:
        """Perform ``write`` unless writes are withheld; return the artifact path."""
        if self.dry_run:
            self.withheld_writes.append(str(path.relative_to(self.root)))
            return path
        write()
        return path

    # -- layout ----------------------------------------------------------
    def ensure(self) -> None:
        """Create the layout, repair its modes, then verify them.

        Repair-then-verify rather than repair-and-hope. ``chmod`` succeeding is
        not evidence that the mode changed.

        The layout is checked *before* the first ``mkdir`` or ``chmod``, because
        both of those follow a symlink and would carry the whole private tree
        outside the configured directory.
        """
        self.verify_layout()
        # A directory that already existed and had to be re-moded is a repair
        # worth reporting; one created just now is not, whatever the umask did.
        repaired: list[str] = []
        existed = self.root.exists()
        _mkdir_or_refuse(self.root, self.root, parents=True)
        if _chmod_to(self.root, DIR_MODE) and existed:
            repaired.append(".")
        for name in SUBDIRS:
            path = self.root / name
            _refuse_unless_real_dir(path, self.root)
            existed = path.exists()
            _mkdir_or_refuse(path, self.root)
            if _chmod_to(path, DIR_MODE) and existed:
                repaired.append(name)
        readme = self.root / "README.txt"
        if not readme.exists():
            write_text_atomic(readme, README_TEXT)
        repaired.extend(p for p in self.repair_permissions() if p not in repaired)
        self.repaired_paths = repaired
        self.verify_permissions()

    def layout_problems(self) -> list[str]:
        """Every mandatory state directory that is not a real directory.

        Measured with ``lstat`` so the check never follows the entry it judges.
        A symlink here is not a permission problem and must not be reported as
        one: it redirects the tree, so a mode read through it describes some
        other directory, and repairing that mode would mutate it.

        An absent directory is not a problem -- ``ensure`` creates those. A root
        that is a link is reported alone, because every path below it would be
        resolved through the link.
        """
        reason = _ancestor_layout_reason(self.root)
        if reason:
            return [reason]
        reason = _layout_reason(self.root, self.root)
        if reason:
            return [reason]
        problems: list[str] = []
        for name in SUBDIRS:
            reason = _layout_reason(self.root / name, self.root)
            if reason:
                problems.append(reason)
        return problems

    def verify_layout(self) -> None:
        """Fail closed when the layout is not confined to the configured root."""
        problems = self.layout_problems()
        if not problems:
            return
        raise SafetyRefusal(
            "refusing to use the state directory: it holds private conversation content and "
            f"{len(problems)} mandatory path(s) are not a real directory under {self.root}.\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n  A symlink is followed by mkdir, chmod and every later write, so a state "
            "directory that is a link would put verbatim transcripts outside the configured "
            "directory and would chmod a directory this tool does not own. Replace the entry "
            "with a real directory, or point [run].state_dir at the location you actually want."
        )

    def repair_permissions(self) -> list[str]:
        """Tighten anything in the existing tree that is wider than policy.

        Returns the paths it changed, relative to the root, so ``doctor`` can say
        what it did instead of claiming the tree was always fine.
        """
        # Resets ``dir()`` already made are reported here, once, even when the
        # walk itself is declined: they happened.
        changed: list[str] = list(self._pending_repairs)
        self._pending_repairs.clear()
        if self.layout_problems():
            return changed
        if self.root.exists() and _chmod_to(self.root, DIR_MODE) and "." not in changed:
            changed.append(".")
        for path in sorted(self.root.rglob("*")):
            if _foreign_reason(path):
                continue
            want = DIR_MODE if path.is_dir() else STATE_FILE_MODE
            rel = str(path.relative_to(self.root))
            if _chmod_to(path, want) and rel not in changed:
                changed.append(rel)
        return changed

    def permission_problems(self) -> list[str]:
        """Every path whose actual mode is still wrong. Measured, not assumed."""
        problems: list[str] = []
        if not self.root.exists() or self.layout_problems():
            return problems
        if not _mode_ok(self.root, DIR_MODE):
            problems.append(f". is {_mode_of(self.root)}, want {oct(DIR_MODE)}")
        for path in sorted(self.root.rglob("*")):
            if _foreign_reason(path):
                continue
            want = DIR_MODE if path.is_dir() else STATE_FILE_MODE
            if not _mode_ok(path, want):
                rel = path.relative_to(self.root)
                problems.append(f"{rel} is {_mode_of(path)}, want {oct(want)}")
        return problems

    def foreign_entries(self) -> list[str]:
        """Every entry under the root that the mode walk steps over, relative to it.

        This tool only ever creates regular files and directories, so anything
        else in the tree is somebody else's. A symlink's mode is not its own,
        and a chmod through it would mutate whatever it points at; a fifo or a
        socket carries no content whose mode could leak anything. So
        ``repair_permissions`` and ``permission_problems`` skip every such
        entry, by the same predicate used here. Skipping is safe: every state
        record read, the backup manifest read and the run lock refuse a leaf
        that is not a regular file, by name and without blocking on it, and a
        write replaces the entry rather than following it. Saying nothing
        about them is not safe, which is why ``doctor``
        names them. Nothing here removes one; deleting an entry this tool did
        not create is the operator's call.
        """
        if not self.root.exists() or self.layout_problems():
            return []
        found: list[str] = []
        for path in sorted(self.root.rglob("*")):
            reason = _foreign_reason(path)
            if reason:
                found.append(f"{path.relative_to(self.root)} {reason}")
        return found

    def verify_permissions(self) -> None:
        """Fail closed when the tree is not private after repair."""
        self.verify_layout()
        problems = self.permission_problems()
        if not problems:
            return
        shown = problems[:10]
        more = "" if len(problems) <= 10 else f"\n  ... and {len(problems) - 10} more"
        raise SafetyRefusal(
            "refusing to use the state directory: it holds private conversation content and "
            f"{len(problems)} path(s) are not owner-only after an attempted repair.\n"
            + "\n".join(f"  - {p}" for p in shown)
            + more
            + f"\n  Expected {oct(DIR_MODE)} on directories and {oct(STATE_FILE_MODE)} on files "
            f"under {self.root}. A chmod that is accepted and does nothing (a filesystem with "
            "no POSIX modes, for instance) looks identical to a private tree unless the modes "
            "are re-read, which is why this is checked rather than assumed."
        )

    def dir(self, name: str) -> Path:
        if name not in SUBDIRS:
            raise KeyError(f"unknown state subdirectory {name!r}")
        # The same three checks ``ensure()`` makes, because this is the path every
        # command that never calls ``ensure()`` takes into the tree: an aliased
        # ancestor would redirect the mkdir and the chmod below just as surely as
        # an aliased root.
        reason = _ancestor_layout_reason(self.root)
        if reason:
            raise SafetyRefusal(
                f"refusing to use {self.root} as the state directory: {reason}. Every state "
                "directory must be reachable without following a symlink, because mkdir, "
                "chmod and every later write would carry private conversation content "
                "outside the configured state directory. Point [run].state_dir at the "
                "real location you actually want."
            )
        _refuse_unless_real_dir(self.root, self.root)
        path = self.root / name
        _refuse_unless_real_dir(path, self.root)
        root_existed = self.root.is_dir()
        existed = path.is_dir()
        if not existed:
            _mkdir_or_refuse(self.root, self.root, parents=True)
            _mkdir_or_refuse(path, self.root)
        # Both levels, every time: the root used to be re-moded only when the
        # subdirectory was being created, so a command that read an existing
        # subdirectory left a world-readable root as it found it. A directory
        # created just now is not a repair, whatever the umask did to it.
        if _chmod_to(self.root, DIR_MODE) and root_existed:
            self._note_repair(".")
        if _chmod_to(path, DIR_MODE) and existed:
            self._note_repair(name)
        return path

    def _note_repair(self, rel: str) -> None:
        if rel not in self._pending_repairs:
            self._pending_repairs.append(rel)

    def _now(self) -> str:
        if self._clock is None:
            from ..clock import SystemClock

            return iso(SystemClock().now())
        return iso(self._clock.now())

    # -- conversations / scorecards / failures ---------------------------
    def write_conversation(self, conversation_id: str, payload: dict[str, Any]) -> Path:
        path = self.dir("conversations") / f"{keyed_name(conversation_id)}.json"
        return self._persist(path, lambda: write_json(path, payload))

    def read_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        path = self.dir("conversations") / f"{keyed_name(conversation_id)}.json"
        return _read_state_object(path, what="conversation") if path.is_file() else None

    def write_scorecard(
        self, conversation_id: str, rubric_id: str, payload: dict[str, Any]
    ) -> Path:
        name = f"{keyed_name(conversation_id)}--{keyed_name(rubric_id)}.json"
        path = self.dir("scorecards") / name
        return self._persist(path, lambda: write_json(path, payload))

    def write_failures(
        self, conversation_id: str, rubric_id: str, payload: list[dict[str, Any]]
    ) -> Path:
        name = f"{keyed_name(conversation_id)}--{keyed_name(rubric_id)}.json"
        path = self.dir("failures") / name
        return self._persist(
            path, lambda: write_json(path, {"schema": "asi.failures.v1", "cases": payload})
        )

    def write_raw(self, conversation_id: str, text: str) -> Path:
        """Persist a raw export. The most sensitive artifact this tool writes."""
        path = self.dir("raw") / f"{keyed_name(conversation_id)}.json"
        return self._persist(path, lambda: write_text_atomic(path, text))

    # -- candidates ------------------------------------------------------
    def candidate_path(self, candidate_id: str) -> Path:
        try:
            validated = validate_candidate_id(candidate_id)
        except ValueError as exc:
            raise StateUnreadable(f"invalid candidate id {candidate_id!r}: {exc}") from exc
        return self.dir("candidates") / f"{validated}.json"

    def write_candidate(self, candidate: Candidate) -> Path:
        path = self.candidate_path(candidate.id)
        return self._persist(path, lambda: write_json(path, candidate.to_dict()))

    def read_candidate(self, candidate_id: str) -> Candidate:
        """Load one candidate by id, failing closed on anything unexpected.

        The id is the only thing the operator typed, and the record is a plain
        file: the file the id resolves to must therefore also *claim* that id, or
        a promotion would be reviewed under one name and applied to another. The
        path is ``<id>.json`` exactly (``candidate_path`` validates the id and
        applies no lossy renaming), so a record whose declared id matches the
        requested one is by construction the file its own name derives; the
        listing applies the same binding from the other side, in
        ``scan_candidates``.

        A record that cannot be parsed is a refusal with a reason rather than a
        ``KeyError`` or a ``JSONDecodeError`` from the middle of a promotion.
        """
        path = self.candidate_path(candidate_id)
        if not path.is_file():
            raise FileNotFoundError(f"no candidate {candidate_id!r} at {path}")
        candidate = _candidate_from_file(path)
        if candidate.id != candidate_id:
            raise StateUnreadable(
                f"refusing to use {path} as candidate {candidate_id!r}: the record it holds "
                f"declares id {candidate.id!r}. A candidate is reviewed and approved by id, so "
                "a record that does not claim the id it was looked up by cannot be acted on."
            )
        return candidate

    def scan_candidates(self) -> CandidateScan:
        """Every candidate record, readable or not, each unreadable one with its reason.

        A record that does not parse is never guessed at and never acted on
        (``read_candidate`` refuses it by name), but it is not nothing either:
        it may be an open candidate whose skill the queue caps are counting, so
        the caller decides what an unreadable record means for it. ``daily``
        treats one as a queue it cannot measure and stands down; the listings
        name it.
        """
        readable: list[Candidate] = []
        unreadable: list[UnreadableRecord] = []
        for path in sorted(self.dir("candidates").glob("*.json")):
            # A link or any other non-file is not a record at all: this tool never
            # writes one, it is already named by ``foreign_entries`` and left
            # alone, and ``read_candidate`` refuses it by name. An unreadable
            # record is a regular file that does not parse.
            if path.is_symlink() or not path.is_file():
                continue
            try:
                candidate = _candidate_from_file(path)
            except StateUnreadable as exc:
                unreadable.append(UnreadableRecord(path=path, reason=str(exc)))
                continue
            # The same binding ``read_candidate`` enforces: a record is looked up
            # by the id it declares, so one whose filename is not derived from
            # that id can be listed but never opened. Measured before: such a
            # record was listed as a validated candidate, ``review`` of the
            # listed id answered "no candidate", ``review`` of the filename was
            # refused, ``doctor`` was clean, and the skill stayed capped by an
            # open candidate nobody could reach.
            misnamed = _name_binding_problem(path, candidate)
            if misnamed:
                unreadable.append(
                    UnreadableRecord(
                        path=path,
                        reason=f"refusing to count the candidate record {path}: {misnamed}",
                    )
                )
                continue
            readable.append(candidate)
        return CandidateScan(readable=tuple(readable), unreadable=tuple(unreadable))

    def list_candidates(self) -> list[Candidate]:
        """Every readable candidate record. See ``scan_candidates`` for the rest."""
        return list(self.scan_candidates().readable)

    def unreadable_candidates(self) -> list[UnreadableRecord]:
        return list(self.scan_candidates().unreadable)

    def open_candidates(self) -> list[Candidate]:
        return [c for c in self.list_candidates() if c.status in OPEN_STATUSES]

    # -- quarantine / defer ---------------------------------------------
    def quarantine(self, record_id: str, *, stage: str, reason: str, detail: Any = None) -> Path:
        path = self.dir("quarantine") / f"{keyed_name(record_id)}--{keyed_name(stage)}.json"
        payload = {
            "schema": "asi.quarantine.v1",
            "record_id": record_id,
            "stage": stage,
            "reason": reason,
            "detail": detail,
            "at": self._now(),
        }
        return self._persist(path, lambda: write_json(path, payload))

    def refuse(self, record_id: str, *, stage: str, reason: str) -> Path:
        """Record a deliberate policy refusal, kept separate from quarantine.

        Quarantine means "this record is defective". A refusal means "the guard
        worked as designed". They demand opposite responses -- investigate the
        first, do nothing about the second -- so a single counter covering both is
        uninformative in exactly the situation you would want to consult it.
        """
        path = self.dir("refusals") / f"{keyed_name(record_id)}--{keyed_name(stage)}.json"
        payload = {
            "schema": "asi.refusal.v1",
            "record_id": record_id,
            "stage": stage,
            "reason": reason,
            "expected": True,
            "at": self._now(),
        }
        return self._persist(path, lambda: write_json(path, payload))

    def _deferred_path(self, record_id: str) -> Path:
        return self.dir("deferred") / f"{keyed_name(record_id)}.json"

    @staticmethod
    def _read_deferred_attempts(path: Path) -> int:
        """The attempt count a deferral file holds; 0 for anything that is not ours to read.

        A link, a non-file or a record that does not parse is read as no
        streak at all. Measured before: a link planted at the path made
        ``defer`` raise ``StateUnreadable`` from inside the ingest handler,
        which ended the whole run with no receipt; and a fresh deferral then
        wrote through ``_persist``, which replaces the link rather than
        following it, so the streak restarts in a file of our own.
        """
        if path.is_symlink() or not path.is_file():
            return 0
        try:
            raw = _read_state_json(path)
        except (ValueError, StateUnreadable):
            return 0
        if not isinstance(raw, dict):
            # Valid JSON that is not an object (``[]``, ``"x"``, ``5``,
            # ``null``) has no attempts to read; it is rewritten, not followed.
            return 0
        try:
            return int(raw.get("attempts", 0))
        except (ValueError, TypeError):
            return 0

    def defer(self, record_id: str, *, stage: str, reason: str) -> Path:
        path = self._deferred_path(record_id)
        attempts = self._read_deferred_attempts(path) + 1
        payload = {
            "schema": "asi.deferred.v1",
            "record_id": record_id,
            "stage": stage,
            "reason": reason,
            "attempts": attempts,
            "at": self._now(),
        }
        return self._persist(path, lambda: write_json(path, payload))

    def deferred_attempts(self, record_id: str) -> int:
        """How many consecutive runs have deferred this record; 0 when none has."""
        return self._read_deferred_attempts(self._deferred_path(record_id))

    def clear_deferred(self, record_id: str) -> None:
        path = self._deferred_path(record_id)
        # Only a regular file of our own is removed. A link at this path is a
        # foreign entry: ``doctor`` names it and leaves it alone, and so does
        # this (``is_file`` follows a link, so it is asked second).
        if not path.is_symlink() and path.is_file():
            self._persist(path, path.unlink)

    def scan_set_aside(self) -> SetAsideScan:
        """Every quarantine, refusal and deferral record, newest first.

        Same rules as ``scan_candidates``: a link or other non-file is not a
        record and is skipped (``foreign_entries`` names it); a regular file
        that does not read as a record of the expected shape is listed as
        unreadable with the reason, never dropped. Newest first because the
        question this answers is "what went wrong lately"; ties keep the kind
        order of ``SET_ASIDE_KINDS``, then path order within a kind.
        """
        readable: list[SetAsideRecord] = []
        unreadable: list[UnreadableRecord] = []
        for kind, directory in SET_ASIDE_KINDS:
            for path in sorted(self.dir(directory).glob("*.json")):
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    raw = _read_state_json(path)
                except (ValueError, StateUnreadable) as exc:
                    unreadable.append(UnreadableRecord(path=path, reason=str(exc)))
                    continue
                problem = _set_aside_shape_problem(raw)
                if problem:
                    unreadable.append(
                        UnreadableRecord(
                            path=path, reason=f"refusing to list the record {path}: {problem}"
                        )
                    )
                    continue
                attempts = raw.get("attempts")
                readable.append(
                    SetAsideRecord(
                        kind=kind,
                        record_id=raw["record_id"],
                        stage=raw["stage"],
                        reason=str(raw.get("reason", "")),
                        at=str(raw.get("at", "")),
                        attempts=(
                            attempts
                            if isinstance(attempts, int) and not isinstance(attempts, bool)
                            else None
                        ),
                        path=path,
                    )
                )
        readable.sort(key=lambda r: r.at, reverse=True)
        return SetAsideScan(readable=tuple(readable), unreadable=tuple(unreadable))

    def deferred_ids(self) -> list[str]:
        out: list[str] = []
        for path in sorted(self.dir("deferred").glob("*.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                raw = _read_state_json(path)
            except (ValueError, StateUnreadable):
                continue
            if not isinstance(raw, dict) or "record_id" not in raw:
                continue
            out.append(str(raw["record_id"]))
        return out

    def count(self, name: str) -> int:
        # Only what a read would return: a symlinked leaf is refused by every
        # reader, so counting it would promise a record no command can show.
        return sum(1 for p in self.dir(name).glob("*.json") if p.is_file() and not p.is_symlink())

    # -- run lock --------------------------------------------------------
    def lock_path(self) -> Path:
        return self.dir("locks") / LOCK_NAME

    def lock_entry_problem(self) -> str | None:
        """Why ``daily`` would refuse what sits at the lock path, or None.

        None for an absent path or a regular file (the lock is taken with
        ``flock``, so a file left by a finished run is not a problem). Measured
        before: a link or a directory at the path left ``doctor`` at ``ready``
        (a directory is not even a foreign entry) while the next run refused.
        """
        path = self.lock_path()
        if path.is_symlink():
            return f"the run lock path {path} is a symlink"
        if not path.exists():
            return None
        if path.is_dir():
            return f"the run lock path {path} is a directory"
        if not path.is_file():
            return f"the run lock path {path} is not a regular file"
        return None

    # -- checkpoints -----------------------------------------------------
    def checkpoint_path(self, day: str) -> Path:
        return self.dir("checkpoints") / f"{keyed_name(day)}.json"

    def read_checkpoint(self, day: str) -> dict[str, Any]:
        path = self.checkpoint_path(day)
        if not path.is_file():
            return {
                "schema": "asi.checkpoint.v1",
                "day": day,
                "runs": 0,
                "records": {},
                "skills": {},
                "candidates": [],
                "completed": False,
            }
        return _load_checkpoint(path)

    def unreadable_checkpoints(self) -> list[UnreadableRecord]:
        """Every checkpoint file ``read_checkpoint`` would refuse, with its reason.

        Measured before: with today's checkpoint truncated, ``doctor`` said
        ``verdict: ready`` and the next ``daily`` refused by name. The refusal
        is right; a preflight that cannot see it coming is not. Links and other
        non-files are not records (``foreign_entries`` names them) and are not
        listed here.
        """
        out: list[UnreadableRecord] = []
        for path in sorted(self.dir("checkpoints").glob("*.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                _load_checkpoint(path)
            except StateUnreadable as exc:
                out.append(UnreadableRecord(path=path, reason=str(exc)))
        return out

    def write_checkpoint(self, day: str, payload: dict[str, Any]) -> Path:
        path = self.checkpoint_path(day)
        return self._persist(path, lambda: write_json(path, payload))

    def processed_record_ids(self, day: str) -> set[str]:
        """Records this day already reached a terminal per-record state for.

        Deferred records are deliberately *not* included: a deferral means we
        never got an answer, so a later run must retry it.
        """
        records = self.read_checkpoint(day).get("records", {})
        return {
            rid
            for rid, info in records.items()
            if isinstance(info, dict) and info.get("status") in ("scored", "quarantined")
        }

    # -- receipts --------------------------------------------------------
    def receipt_path(self, day: str, *, dry_run: bool) -> Path:
        stem = f"{keyed_name(day)}{'-dry-run' if dry_run else ''}.json"
        return self.dir("receipts") / stem

    def scan_receipts(self) -> ReceiptScan:
        """The newest daily receipt, by its own ``finished_at`` (then ``started_at``).

        Only daily receipts (``SCHEMA_RECEIPT``) count as a run; apply and
        rollback receipts share the directory under their own schemas and are
        passed over. A link or other non-file is not a record and is skipped;
        a regular file that does not read as a receipt is listed as unreadable,
        never dropped, and never chosen. Read without creating the directory
        (so not through ``dir``, whose guards are for writers): this scan adds
        nothing to the state tree, and a symlinked directory reads as absent,
        which ``doctor`` names separately as a layout problem.
        """
        from .. import SCHEMA_RECEIPT

        directory = self.root / "receipts"
        newest: LastRun | None = None
        newest_key = ""
        unreadable: list[UnreadableRecord] = []
        if not directory.is_dir() or directory.is_symlink():
            return ReceiptScan(last_run=None, unreadable=())
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                raw = _read_state_json(path)
            except (ValueError, StateUnreadable) as exc:
                unreadable.append(UnreadableRecord(path=path, reason=str(exc)))
                continue
            if not isinstance(raw, dict):
                unreadable.append(
                    UnreadableRecord(
                        path=path, reason=f"the record is {type(raw).__name__}, not an object"
                    )
                )
                continue
            if raw.get("schema") != SCHEMA_RECEIPT:
                continue
            problem = _receipt_shape_problem(raw)
            if problem:
                unreadable.append(
                    UnreadableRecord(
                        path=path, reason=f"refusing to read the receipt {path}: {problem}"
                    )
                )
                continue
            at = str(raw.get("finished_at") or raw.get("started_at") or "")
            skill_tree: dict[str, Any] = (
                raw["skill_tree"] if isinstance(raw.get("skill_tree"), dict) else {}
            )
            isolation: dict[str, Any] = (
                raw["isolation"] if isinstance(raw.get("isolation"), dict) else {}
            )
            detected = skill_tree.get("mutation_detected")
            run = LastRun(
                day=raw["day"],
                status=raw["status"],
                finished_at=at,
                dry_run=bool(raw.get("dry_run")),
                mutation_detected=detected if isinstance(detected, bool) else None,
                isolation_mode=str(isolation.get("mode", "")),
                path=path,
            )
            # Ties keep the first in path order; a later file with the same
            # stamp does not displace it, so the answer is stable across scans.
            if newest is None or at > newest_key:
                newest, newest_key = run, at
        return ReceiptScan(last_run=newest, unreadable=tuple(unreadable))

    def write_receipt(self, day: str, payload: dict[str, Any], *, dry_run: bool) -> Path:
        path = self.receipt_path(day, dry_run=dry_run)
        return self._persist(path, lambda: write_json(path, payload))

    def _refuse_promotion_when_withholding(self, what: str) -> None:
        # Promotion is never a preview. A store that withholds writes would keep
        # the skill edit and drop the receipt and backup that account for it.
        if self.dry_run:
            raise SafetyRefusal(
                f"refusing to write a {what} through a store that withholds writes (dry run); "
                "promotion needs a committed state store"
            )

    def write_action_receipt(self, action: str, candidate_id: str, payload: dict[str, Any]) -> Path:
        self._refuse_promotion_when_withholding("promotion receipt")
        if action not in ("apply", "rollback"):
            raise SafetyRefusal(f"unknown candidate receipt action {action!r}")
        path = self.dir("receipts") / f"{action}--{validate_candidate_id(candidate_id)}.json"
        write_json(path, payload)
        return path

    def backup_dir_for(self, candidate_id: str, stamp: str) -> Path:
        """A per-apply backup directory, created only as a direct child.

        The name is dynamic, so an attacker who can write into ``backups/`` can
        guess it and precreate it as a symlink. ``mkdir(exist_ok=True)`` accepts
        such a link without complaint, and the chmod and the backup copy then
        land outside the state tree, so the entry is proven to be a real direct
        child before either happens.
        """
        self._refuse_promotion_when_withholding("backup")
        stem = safe_name(stamp)
        if "--" in stem:
            # The owner of a backup directory is read off its name by splitting
            # at the last ``--`` (see ``latest_backup_manifest``); a stamp that
            # carried one would make the directory look like another candidate's.
            raise ValueError(f"backup stamp {stamp!r} must not contain '--'")
        return _ensure_child_dir(
            self.dir("backups"), f"{validate_candidate_id(candidate_id)}--{stem}", self.root
        )

    def latest_backup_manifest(self, candidate_id: str) -> Path | None:
        """The newest ``backup.json`` written by an apply of exactly this candidate.

        A backup directory is named ``<candidate id>--<stamp>``, and a candidate
        id may itself contain ``--``, so the owner is read off the directory name
        by splitting at the LAST ``--`` (the stamp never contains one), never by
        string prefix: a prefix match would hand ``CAND-a`` the backup of
        ``CAND-a--b`` and let a rollback restore the wrong candidate's file.
        """
        wanted = validate_candidate_id(candidate_id)
        dirs = sorted(
            p
            for p in self.dir("backups").glob(f"{wanted}--*")
            if p.is_dir() and not p.is_symlink() and p.name.rsplit("--", 1)[0] == wanted
        )
        for path in reversed(dirs):
            manifest = path / "backup.json"
            if manifest.is_file() and not manifest.is_symlink():
                return manifest
        return None

    def summary(self) -> dict[str, int]:
        return {
            name: self.count(name)
            for name in ("candidates", "quarantine", "refusals", "deferred", "conversations")
        }


def _read_state_json(path: Path) -> Any:
    """Read one state leaf without following a symlink or blocking on a fifo.

    ``O_NONBLOCK`` is set only for the ``open``: a regular file ignores it, and a
    fifo precreated at the path would otherwise park the process until something
    writes to it. The descriptor is then checked with ``fstat`` so that whatever
    the entry really is, only a regular file is ever read.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if not getattr(os, "O_NOFOLLOW", 0) and path.is_symlink():
        raise StateUnreadable(f"refusing to read state artifact {path}: the leaf is a symlink")
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if path.is_symlink():
            raise StateUnreadable(
                f"refusing to read state artifact {path}: the leaf is a symlink"
            ) from exc
        raise
    try:
        regular = stat.S_ISREG(os.fstat(fd).st_mode)
    except OSError:
        os.close(fd)
        raise
    if not regular:
        os.close(fd)
        raise StateUnreadable(
            f"refusing to read state artifact {path}: the leaf is not a regular file"
        )
    # Ownership of the descriptor passes to the file object exactly once: if
    # ``fdopen`` itself fails the raw descriptor is closed here, and from then
    # on the ``with`` block is the only thing that closes it.
    try:
        handle = os.fdopen(fd, "r", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise
    with handle:
        return loads_untrusted(handle.read())


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Parse one checkpoint file into the shape ``run_daily`` merges, or refuse by name.

    A parsed object whose fields have the wrong shape is refused like a
    truncated one. Measured before: ``"records": []`` escaped as
    ``AttributeError`` from ``processed_record_ids`` at the top of the pass,
    ``"skills": 5`` as ``TypeError`` from the checkpoint update at its end; a
    traceback either way, and no file named.
    """
    data = _read_state_object(path, what="checkpoint")
    for key, kind in CHECKPOINT_FIELDS:
        value = data.setdefault(key, kind())
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise _checkpoint_shape_refusal(
                path, key, f"holds {type(value).__name__}, not {kind.__name__}"
            )
    # The candidate ids are sorted and set-merged at the end of the pass, after
    # the receipt is on disk; an entry that is not a string failed there as a
    # bare ``TypeError``. Checked here, with the container, before anything runs.
    odd = [c for c in data["candidates"] if not isinstance(c, str)]
    if odd:
        raise _checkpoint_shape_refusal(
            path,
            "candidates",
            (
                f"holds 1 entry that is not a string ({type(odd[0]).__name__})"
                if len(odd) == 1
                else f"holds {len(odd)} entries that are not strings "
                f"(first: {type(odd[0]).__name__})"
            ),
        )
    return data


def _receipt_shape_problem(raw: Mapping[str, Any]) -> str:
    """Why ``raw`` is not a daily receipt the health line can read, or an empty string."""
    for key in ("day", "status"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            return f"field {key!r} is missing or not a non-empty string"
    return ""


def _set_aside_shape_problem(raw: Any) -> str:
    """Why ``raw`` is not a set-aside record, or an empty string when it is one."""
    if not isinstance(raw, dict):
        return f"the record is {type(raw).__name__}, not an object"
    for key in ("record_id", "stage"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            return f"field {key!r} is missing or not a non-empty string"
    if "reason" in raw and not isinstance(raw["reason"], str):
        return "field 'reason' is not a string"
    return ""


def _checkpoint_shape_refusal(path: Path, key: str, what: str) -> StateUnreadable:
    return StateUnreadable(
        f"refusing to use the checkpoint record {path}: field {key!r} {what}. A "
        "hand-edited state file is never guessed at; move it aside to start the day "
        "from a fresh checkpoint."
    )


def _read_state_object(path: Path, *, what: str) -> dict[str, Any]:
    """Read one state leaf that must be a JSON object, or refuse by name.

    A truncated or hand-edited leaf used to escape as a bare ``JSONDecodeError``
    from the middle of a pass: a traceback, exit code 1, and no file named. Every
    state read is a refusal with the path and the remedy instead. The corrupt
    file is left in place; nothing here guesses at what it was meant to say.
    Deep nesting is folded into ``ValueError`` by the reader, so one ``except``
    covers a truncated leaf and a pathological one alike.
    """
    try:
        raw = _read_state_json(path)
    except (OSError, ValueError) as exc:
        raise StateUnreadable(
            f"refusing to use the {what} record {path}: it could not be read ({exc}). "
            "A truncated or hand-edited state file is never guessed at; move it aside "
            "to continue without it."
        ) from exc
    if not isinstance(raw, dict):
        raise StateUnreadable(f"refusing to use the {what} record {path}: it is not a JSON object.")
    return raw


def _name_binding_problem(path: Path, candidate: Candidate) -> str | None:
    """Why ``path`` may not stand for ``candidate``, or None when its name derives from the id.

    The listing's side of the binding ``read_candidate`` enforces from the id:
    there, the path is built from the requested id and the record must claim
    it; here, the record is found first and its name must be the one that id
    would build.
    """
    if path.name == f"{candidate.id}.json":
        return None
    return (
        f"the record declares id {candidate.id!r}, which does not derive this filename. A "
        "candidate is reviewed, counted and approved by id, so a record that cannot be found "
        "by the id it claims is neither reviewable nor countable; rename it to "
        f"{candidate.id}.json if it is genuine, or move it out of the state tree."
    )


def _candidate_from_file(path: Path) -> Candidate:
    """Parse a candidate record, turning every read failure into one refusal."""
    try:
        raw = _read_state_json(path)
    except (OSError, ValueError) as exc:
        raise StateUnreadable(
            f"refusing to use the candidate record {path}: it could not be read ({exc}). "
            "A truncated or hand-edited record is never guessed at."
        ) from exc
    if not isinstance(raw, dict):
        raise StateUnreadable(
            f"refusing to use the candidate record {path}: it is not a JSON object."
        )
    try:
        return Candidate.from_dict(raw)
    except (ValueError, KeyError, TypeError) as exc:
        raise StateUnreadable(
            f"refusing to use the candidate record {path}: it does not have the shape of a "
            f"candidate ({type(exc).__name__}: {exc})."
        ) from exc


def _label(path: Path, root: Path) -> str:
    """A path named relative to the state root, so a message stays readable."""
    try:
        return str(path.relative_to(root)) if path != root else "."
    except ValueError:  # pragma: no cover - callers only pass paths under root
        return str(path)


def _ancestor_layout_reason(root: Path) -> str | None:
    """Why an existing ancestor would redirect ``root``, without following it.

    The path is walked as written, never after a lexical collapse of ``..``:
    ``os.path.abspath`` turns ``/a/link/../state`` into ``/a/state`` and would
    never look at ``/a/link``, while the kernel resolves the link first and
    lands under its target. A ``..`` steps back only once every component before
    it has been measured to be a real directory or to be absent, so the lexical
    parent is also the physical one.
    """
    absolute = root.absolute()
    ancestors = absolute.parts[1:-1]
    current = Path(absolute.anchor)
    for index, component in enumerate(ancestors):
        if component == ".":  # pragma: no cover - pathlib drops these at construction
            continue
        if component == "..":
            current = current.parent
            continue
        current /= component
        try:
            entry = os.lstat(current)
        except FileNotFoundError:
            # An absent component ends the chain only when nothing steps back
            # over it: a later ``..`` lands on entries that do exist, and one
            # of those may be the link this walk is for.
            if ".." in ancestors[index + 1 :]:
                continue
            return None
        except OSError as exc:
            return f"ancestor {current} cannot be inspected ({exc.strerror})"
        if stat.S_ISLNK(entry.st_mode):
            try:
                target = os.readlink(current)
            except OSError:  # pragma: no cover - readlink of a link we just saw
                target = "<unreadable>"
            return f"ancestor {current} is a symlink to {target!r}"
        if not stat.S_ISDIR(entry.st_mode):
            return f"ancestor {current} exists and is not a directory"
    return None


def _layout_reason(path: Path, root: Path) -> str | None:
    """Why ``path`` may not be used as a state directory, or None if it may.

    Absent is fine: the caller creates it. A symlink or a non-directory is not,
    and neither verdict follows the entry.
    """
    label = _label(path, root)
    if path.is_symlink():
        try:
            target = os.readlink(path)
        except OSError:  # pragma: no cover - readlink of a link we just saw
            target = "<unreadable>"
        return f"{label} is a symlink to {target!r}, not a real directory"
    try:
        entry = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"{label} cannot be inspected ({exc.strerror})"
    if not stat.S_ISDIR(entry.st_mode):
        return f"{label} exists and is not a directory"
    return None


def _mkdir_or_refuse(path: Path, root: Path, *, parents: bool = False) -> None:
    """Create ``path`` under the state root, or say why it cannot be created.

    The failure is named as a configuration problem because that is where the
    fix lives: ``[run].state_dir`` points somewhere this process cannot create
    (a read-only parent, a mount that refuses); a parent that is a file
    or a link is a layout refusal, judged before this runs. Measured
    before this helper: a state directory under a mode 555 parent ended
    ``doctor``, ``daily`` and ``candidates`` as internal errors with a
    traceback from the first ``mkdir``.
    """
    try:
        path.mkdir(parents=parents, exist_ok=True)
    except OSError as exc:
        where = (
            f"the state directory {root}"
            if path == root
            else f"{path} under the state directory {root}"
        )
        raise ConfigError(
            f"{where} cannot be created ({exc.strerror or exc}); "
            "check [run].state_dir and the permissions of its parent"
        ) from exc


def _refuse_unless_real_dir(path: Path, root: Path) -> None:
    """Refuse before following: mkdir and chmod both chase a symlink."""
    reason = _layout_reason(path, root)
    if reason is None:
        return
    raise SafetyRefusal(
        f"refusing to use {path} as a state directory: {reason}. Every state directory must be "
        f"a real directory under {root}, because mkdir, chmod and every later write follow a "
        "symlink and would carry private conversation content outside the configured state "
        "directory."
    )


def _ensure_child_dir(parent: Path, name: str, root: Path) -> Path:
    """Create ``parent/name``, having proven it is a real direct child.

    ``parent`` is a mandatory state directory the caller has already verified.
    ``name`` must be one path component, the entry must not be a symlink, and
    the created directory's parent must still be ``parent`` afterwards.
    """
    if name in ("", ".", "..") or name != Path(name).name:
        raise SafetyRefusal(
            f"refusing to create the state directory {name!r} under {parent}: a dynamically "
            "named state directory must be a single path component, not a path"
        )
    path = parent / name
    _refuse_unless_real_dir(path, root)
    _mkdir_or_refuse(path, root)
    _chmod_to(path, DIR_MODE)
    if path.is_symlink() or not path.is_dir() or path.parent.resolve() != parent.resolve():
        raise SafetyRefusal(
            f"refusing to use {path} as a state directory: it is not a real directory directly "
            f"beneath {parent} after creation"
        )
    return path


def _mode_of(path: Path) -> str:
    return oct(stat.S_IMODE(path.stat().st_mode))


def _mode_ok(path: Path, want: int) -> bool:
    try:
        return stat.S_IMODE(path.stat().st_mode) == want
    except OSError:
        return False


def _foreign_reason(path: Path) -> str | None:
    """Why an entry is not this tool's, or ``None`` for a regular file or directory.

    Measured with ``lstat`` first so a symlink is judged as a symlink, never as
    whatever it points at.
    """
    if path.is_symlink():
        return "is a symlink"
    if path.is_dir() or path.is_file():
        return None
    return "is not a regular file or directory"


def _chmod_to(path: Path, want: int) -> bool:
    """Set ``path`` to ``want``. Returns True when it actually changed.

    Errors are swallowed on purpose: this runs over a whole tree, and one
    unreadable entry must not abort a repair that is fixing everything else. The
    verification pass afterwards is what turns a failed repair into a refusal --
    so a swallowed error here is reported, not lost.
    """
    try:
        current = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return False
    if current == want:
        return False
    try:
        os.chmod(path, want)
    except OSError:
        return False
    return True


def status_map(entries: Iterable[RecordStatus]) -> dict[str, dict[str, Any]]:
    return {e.record_id: e.to_dict() for e in entries}
