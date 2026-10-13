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
import math
import os
import tempfile
from pathlib import Path
from typing import Any

#: Owner read/write only. Every artifact this module writes is derived from
#: private content, so this is the default rather than an option.
FILE_MODE = 0o600


def dumps(obj: Any) -> str:
    """Canonical JSON: sorted keys, stable separators, trailing newline.

    ``allow_nan=False``: a non-finite float is a ``ValueError`` here, the mirror
    of :func:`loads_untrusted`, which refuses to read one. Without it a writer
    handed ``float("nan")`` would emit the token ``NaN`` into a state leaf and
    the failure would surface later, on a read, as an unreadable record.
    """
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_json(path: Path, obj: Any, *, mode: int = FILE_MODE) -> str:
    """Atomically write ``obj``; return the exact text written."""
    text = dumps(obj)
    write_text_atomic(path, text, mode=mode)
    return text


def read_json(path: Path) -> Any:
    """Read a JSON file a hand could have edited; deep nesting is a ``ValueError``."""
    return loads_untrusted(path.read_text(encoding="utf-8"))


def loads_untrusted(text: str) -> Any:
    """Parse JSON that came from outside this process.

    ``json.loads`` raises ``RecursionError`` rather than ``JSONDecodeError`` on
    deep nesting, and every caller that caught only the latter let one
    pathological export or model reply abort the whole pass: a single record
    vetoing the batch. Here the recursion failure is folded into ``ValueError``
    (which ``JSONDecodeError`` already is) so a caller's one ``except ValueError``
    covers both, and classifies the record rather than crashing the run. State
    leaves and backup manifests take the same route: a file under ``state/`` is
    something a hand can edit, so it is read as untrusted too.

    Numbers are held to the same standard. ``json.loads`` accepts the
    non-standard tokens ``NaN``, ``Infinity`` and ``-Infinity`` by default, and
    a legal literal that does not round to a finite double (``1e400``) parses to
    ``inf`` without a word. Measured before this guard: one export whose element
    ``position`` was ``1e400`` reached ``int()`` inside the normalizer and
    aborted a committed daily run as an internal error, with no receipt and the
    healthy export beside it unprocessed. Every float literal read here rounds
    to a finite double and the three tokens are refused, or the text is a
    ``ValueError``, so the record is classified like any other unreadable one.
    Integer literals are untouched (Python's own digit limit already bounds
    them), and the writer side matches: :func:`dumps` refuses a non-finite
    float, so a writer bug fails loudly at write time instead of leaving a leaf
    this reader can never read.
    """
    try:
        return json.loads(text, parse_float=_finite_float, parse_constant=_refuse_constant)
    except RecursionError as exc:
        raise ValueError("JSON is nested too deeply to be anything this tool reads") from exc


def _finite_float(token: str) -> float:
    number = float(token)
    if not math.isfinite(number):
        raise ValueError(
            f"JSON number {token} does not round to a finite double; nothing this tool reads "
            "can carry it"
        )
    return number


def _refuse_constant(token: str) -> Any:
    raise ValueError(f"{token} is not a JSON number; nothing this tool reads can carry it")


def write_text_atomic(path: Path, text: str, *, mode: int = FILE_MODE) -> None:
    """Write via a temp file in the same directory, then ``os.replace``.

    A crash mid-write must never leave a half-written checkpoint that a later
    run would read as complete.

    ``mode`` is applied to the temp file *before* the rename, so the destination
    never exists at a wider mode even momentarily. This is the writer for the
    state tree; a skill document is replaced through a pinned directory handle
    instead (``skill_fs._write_pinned``), which carries the document's own mode.
    """
    write_bytes_atomic(path, text.encode("utf-8"), mode=mode)


def write_bytes_atomic(path: Path, data: bytes, *, mode: int = FILE_MODE) -> None:
    """Byte-exact form of :func:`write_text_atomic`.

    Used where the content is a verified copy of an existing file (the pre-apply
    backup of a skill) and must land byte-for-byte. ``os.replace`` swaps the
    directory entry, so a symlink sitting at ``path`` is replaced rather than
    followed.
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
