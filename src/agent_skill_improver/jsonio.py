"""Deterministic JSON I/O with atomic writes, owner-only by default.

Determinism matters: candidate records and receipts are compared by hash, so
key order and separators must not drift between runs.

Permissions matter for a different reason. Everything written through here is
derived from private conversation content -- raw exports, scorecards, candidate
records, receipts, backups. ``tempfile.mkstemp`` already creates 0600, but
``os.replace`` preserves the temp file's mode rather than the destination's, so
the mode is set explicitly with ``os.fchmod`` before the rename. Explicitly,
because the alternative is inheriting whatever the operator's umask happens to
be, and a permissive umask is exactly the condition under which private
artifacts become world-readable.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

#: Owner read/write only. Every artifact this module writes is derived from
#: private content, so this is the default rather than an option.
FILE_MODE = 0o600


def dumps(obj: Any) -> str:
    """Canonical JSON: sorted keys, stable separators, trailing newline."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def loads(text: str) -> Any:
    return json.loads(text)


def write_json(path: Path, obj: Any, *, mode: int = FILE_MODE) -> str:
    """Atomically write ``obj``; return the exact text written."""
    text = dumps(obj)
    write_text_atomic(path, text, mode=mode)
    return text


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def loads_untrusted(text: str) -> Any:
    """Parse JSON that came from outside this process.

    ``json.loads`` raises ``RecursionError`` rather than ``JSONDecodeError`` on
    deep nesting, and every caller that caught only the latter let one
    pathological export or model reply abort the whole pass: a single record
    vetoing the batch. Here the recursion failure is folded into ``ValueError``
    (which ``JSONDecodeError`` already is) so a caller's one ``except ValueError``
    covers both, and classifies the record rather than crashing the run.
    """
    try:
        return json.loads(text)
    except RecursionError as exc:
        raise ValueError("JSON is nested too deeply to be a conversation or a verdict") from exc


def write_text_atomic(path: Path, text: str, *, mode: int = FILE_MODE) -> None:
    """Write via a temp file in the same directory, then ``os.replace``.

    A crash mid-write must never leave a half-written checkpoint that a later
    run would read as complete.

    ``mode`` is applied to the temp file *before* the rename, so the destination
    never exists at a wider mode even momentarily. Callers replacing a file
    outside the state tree (a skill document, say) pass its existing mode, since
    tightening somebody else's file is a surprise, not a safety improvement.
    """
    write_bytes_atomic(path, text.encode("utf-8"), mode=mode)


def write_bytes_atomic(path: Path, data: bytes, *, mode: int = FILE_MODE) -> None:
    """Byte-exact form of :func:`write_text_atomic`.

    Used where the content is a verified copy of an existing file (a rollback
    from backup) and must land byte-for-byte. ``os.replace`` swaps the directory
    entry, so a symlink sitting at ``path`` is replaced rather than followed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
