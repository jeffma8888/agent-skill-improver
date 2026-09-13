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

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..clock import Clock, iso
from ..domain.candidate import (
    STATUS_APPLIED,
    STATUS_GATE_FAILED,
    STATUS_PROPOSED,
    STATUS_ROLLED_BACK,
    STATUS_VALIDATED,
    Candidate,
)
from ..errors import SafetyRefusal
from ..jsonio import FILE_MODE, read_json, write_json, write_text_atomic

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
CLOSED_STATUSES = (STATUS_APPLIED, STATUS_GATE_FAILED, STATUS_ROLLED_BACK)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(value: str) -> str:
    """A filesystem-safe stem. Never empty, never traverses."""
    cleaned = _UNSAFE.sub("_", value).strip("._-")
    return cleaned or "unnamed"


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


class StateStore:
    """Owns the state directory layout. Every write is atomic."""

    def __init__(self, root: Path, *, clock: Clock | None = None) -> None:
        self.root = Path(root)
        self._clock = clock

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
        self.root.mkdir(parents=True, exist_ok=True)
        _chmod_to(self.root, DIR_MODE)
        for name in SUBDIRS:
            path = self.root / name
            _refuse_unless_real_dir(path, self.root)
            path.mkdir(exist_ok=True)
            _chmod_to(path, DIR_MODE)
        readme = self.root / "README.txt"
        if not readme.exists():
            write_text_atomic(readme, README_TEXT)
        self.repair_permissions()
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
        if self.layout_problems():
            return []
        changed: list[str] = []
        if self.root.exists() and _chmod_to(self.root, DIR_MODE):
            changed.append(".")
        for path in sorted(self.root.rglob("*")):
            if path.is_symlink():
                continue
            want = DIR_MODE if path.is_dir() else STATE_FILE_MODE
            if _chmod_to(path, want):
                changed.append(str(path.relative_to(self.root)))
        return changed

    def permission_problems(self) -> list[str]:
        """Every path whose actual mode is still wrong. Measured, not assumed."""
        problems: list[str] = []
        if not self.root.exists() or self.layout_problems():
            return problems
        if not _mode_ok(self.root, DIR_MODE):
            problems.append(f". is {_mode_of(self.root)}, want {oct(DIR_MODE)}")
        for path in sorted(self.root.rglob("*")):
            if path.is_symlink():
                continue
            want = DIR_MODE if path.is_dir() else STATE_FILE_MODE
            if not _mode_ok(path, want):
                rel = path.relative_to(self.root)
                problems.append(f"{rel} is {_mode_of(path)}, want {oct(want)}")
        return problems

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
        _refuse_unless_real_dir(self.root, self.root)
        path = self.root / name
        _refuse_unless_real_dir(path, self.root)
        if not path.is_dir():
            self.root.mkdir(parents=True, exist_ok=True)
            _chmod_to(self.root, DIR_MODE)
            path.mkdir(exist_ok=True)
        _chmod_to(path, DIR_MODE)
        return path

    def _now(self) -> str:
        if self._clock is None:
            from ..clock import SystemClock

            return iso(SystemClock().now())
        return iso(self._clock.now())

    # -- conversations / scorecards / failures ---------------------------
    def write_conversation(self, conversation_id: str, payload: dict[str, Any]) -> Path:
        path = self.dir("conversations") / f"{safe_name(conversation_id)}.json"
        write_json(path, payload)
        return path

    def read_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        path = self.dir("conversations") / f"{safe_name(conversation_id)}.json"
        return read_json(path) if path.is_file() else None

    def write_scorecard(
        self, conversation_id: str, rubric_id: str, payload: dict[str, Any]
    ) -> Path:
        name = f"{safe_name(conversation_id)}--{safe_name(rubric_id)}.json"
        path = self.dir("scorecards") / name
        write_json(path, payload)
        return path

    def write_failures(
        self, conversation_id: str, rubric_id: str, payload: list[dict[str, Any]]
    ) -> Path:
        name = f"{safe_name(conversation_id)}--{safe_name(rubric_id)}.json"
        path = self.dir("failures") / name
        write_json(path, {"schema": "asi.failures.v1", "cases": payload})
        return path

    def write_raw(self, conversation_id: str, text: str) -> Path:
        """Persist a raw export. The most sensitive artifact this tool writes."""
        path = self.dir("raw") / f"{safe_name(conversation_id)}.json"
        write_text_atomic(path, text)
        return path

    # -- candidates ------------------------------------------------------
    def candidate_path(self, candidate_id: str) -> Path:
        return self.dir("candidates") / f"{safe_name(candidate_id)}.json"

    def write_candidate(self, candidate: Candidate) -> Path:
        path = self.candidate_path(candidate.id)
        write_json(path, candidate.to_dict())
        return path

    def read_candidate(self, candidate_id: str) -> Candidate:
        path = self.candidate_path(candidate_id)
        if not path.is_file():
            raise FileNotFoundError(f"no candidate {candidate_id!r} at {path}")
        return Candidate.from_dict(read_json(path))

    def list_candidates(self) -> list[Candidate]:
        out: list[Candidate] = []
        for path in sorted(self.dir("candidates").glob("*.json")):
            try:
                out.append(Candidate.from_dict(read_json(path)))
            except (ValueError, KeyError):
                continue
        return out

    def open_candidates(self) -> list[Candidate]:
        return [c for c in self.list_candidates() if c.status in OPEN_STATUSES]

    # -- quarantine / defer ---------------------------------------------
    def quarantine(self, record_id: str, *, stage: str, reason: str, detail: Any = None) -> Path:
        path = self.dir("quarantine") / f"{safe_name(record_id)}--{safe_name(stage)}.json"
        write_json(
            path,
            {
                "schema": "asi.quarantine.v1",
                "record_id": record_id,
                "stage": stage,
                "reason": reason,
                "detail": detail,
                "at": self._now(),
            },
        )
        return path

    def refuse(self, record_id: str, *, stage: str, reason: str) -> Path:
        """Record a deliberate policy refusal, kept separate from quarantine.

        Quarantine means "this record is defective". A refusal means "the guard
        worked as designed". They demand opposite responses -- investigate the
        first, do nothing about the second -- so a single counter covering both is
        uninformative in exactly the situation you would want to consult it.
        """
        path = self.dir("refusals") / f"{safe_name(record_id)}--{safe_name(stage)}.json"
        write_json(
            path,
            {
                "schema": "asi.refusal.v1",
                "record_id": record_id,
                "stage": stage,
                "reason": reason,
                "expected": True,
                "at": self._now(),
            },
        )
        return path

    def defer(self, record_id: str, *, stage: str, reason: str) -> Path:
        path = self.dir("deferred") / f"{safe_name(record_id)}.json"
        attempts = 1
        if path.is_file():
            try:
                attempts = int(read_json(path).get("attempts", 0)) + 1
            except (ValueError, KeyError, TypeError):
                attempts = 1
        write_json(
            path,
            {
                "schema": "asi.deferred.v1",
                "record_id": record_id,
                "stage": stage,
                "reason": reason,
                "attempts": attempts,
                "at": self._now(),
            },
        )
        return path

    def clear_deferred(self, record_id: str) -> None:
        path = self.dir("deferred") / f"{safe_name(record_id)}.json"
        if path.is_file():
            path.unlink()

    def deferred_ids(self) -> list[str]:
        out: list[str] = []
        for path in sorted(self.dir("deferred").glob("*.json")):
            try:
                out.append(str(read_json(path)["record_id"]))
            except (ValueError, KeyError):
                continue
        return out

    def count(self, name: str) -> int:
        return len(list(self.dir(name).glob("*.json")))

    # -- checkpoints -----------------------------------------------------
    def checkpoint_path(self, day: str) -> Path:
        return self.dir("checkpoints") / f"{safe_name(day)}.json"

    def read_checkpoint(self, day: str) -> dict[str, Any]:
        path = self.checkpoint_path(day)
        if not path.is_file():
            return {
                "schema": "asi.checkpoint.v1",
                "day": day,
                "runs": 0,
                "records": {},
                "candidates": [],
                "completed": False,
            }
        data = read_json(path)
        data.setdefault("records", {})
        data.setdefault("candidates", [])
        data.setdefault("runs", 0)
        return data

    def write_checkpoint(self, day: str, payload: dict[str, Any]) -> Path:
        path = self.checkpoint_path(day)
        write_json(path, payload)
        return path

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
        stem = f"{safe_name(day)}{'-dry-run' if dry_run else ''}.json"
        return self.dir("receipts") / stem

    def write_receipt(self, day: str, payload: dict[str, Any], *, dry_run: bool) -> Path:
        path = self.receipt_path(day, dry_run=dry_run)
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
        return _ensure_child_dir(
            self.dir("backups"), f"{safe_name(candidate_id)}--{safe_name(stamp)}", self.root
        )

    def latest_backup_manifest(self, candidate_id: str) -> Path | None:
        prefix = f"{safe_name(candidate_id)}--"
        dirs = sorted(
            p for p in self.dir("backups").glob(prefix + "*") if p.is_dir() and not p.is_symlink()
        )
        for path in reversed(dirs):
            manifest = path / "backup.json"
            if manifest.is_file():
                return manifest
        return None

    def summary(self) -> dict[str, int]:
        return {
            name: self.count(name)
            for name in ("candidates", "quarantine", "refusals", "deferred", "conversations")
        }


def _label(path: Path, root: Path) -> str:
    """A path named relative to the state root, so a message stays readable."""
    try:
        return str(path.relative_to(root)) if path != root else "."
    except ValueError:  # pragma: no cover - callers only pass paths under root
        return str(path)


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
    path.mkdir(exist_ok=True)
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
