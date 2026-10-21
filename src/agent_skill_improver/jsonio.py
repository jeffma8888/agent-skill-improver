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
import re
import sys
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

    A lone surrogate is written as its JSON escape (``\\udcff``). It is how a
    file name that is not UTF-8 reaches Python (``os.fsdecode``, and so
    ``os.scandir`` and ``os.walk``, give each such byte as one), and with
    ``ensure_ascii`` off the encoder writes it raw, which no UTF-8 encoder
    accepts. Measured before: a fixtures entry with such a name, passed over
    because it was not an export, made a committed daily run fall with
    ``UnicodeEncodeError`` from the receipt write, with no receipt. A surrogate
    can only sit inside a JSON string, where the escape is the same string to
    any parser, with one exception: a high half directly followed by a low half
    is read as the one character the pair spells. A directory listing gives
    only low halves (U+DC80 to U+DCFF), so a file name never meets it.
    :func:`loads_untrusted` reads a lone one back as U+FFFD.
    """
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    if _LONE_SURROGATE.search(text):
        text = _LONE_SURROGATE.sub(_json_escape, text)
    return text + "\n"


def _json_escape(match: re.Match[str]) -> str:
    return f"\\u{ord(match.group()):04x}"


def write_json(path: Path, obj: Any, *, mode: int = FILE_MODE) -> str:
    """Atomically write ``obj``; return the exact text written."""
    text = dumps(obj)
    write_text_atomic(path, text, mode=mode)
    return text


def read_json(path: Path) -> Any:
    """Read a JSON file a hand could have edited; deep nesting is a ``ValueError``."""
    return loads_untrusted(path.read_text(encoding="utf-8"))


#: A JSON escape for one UTF-16 surrogate half (U+D800 to U+DFFF). Two of them
#: in order decode to one character; one on its own decodes to a lone surrogate.
_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F][0-9a-fA-F]{2}")
#: A lone surrogate as ``json.loads`` leaves one in a decoded string.
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")
#: What a lone surrogate reads as: the Unicode replacement character.
REPLACEMENT_CHARACTER = "\ufffd"


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

    Strings are held to what a UTF-8 writer can encode. JSON can spell one half
    of a UTF-16 surrogate pair on its own (``"\\ud83d"``), which is what a tool
    that cuts an emoji between its halves writes, and ``json.loads`` decodes it
    to a lone surrogate that no UTF-8 encoder accepts. Measured before: one
    conversation export holding one such escape made a committed daily run fall
    with ``UnicodeEncodeError`` from the conversation write, with no receipt,
    and a scorer rationale or an improver's text holding one fell the same way
    from the scorecard write and the patch hash. Each lone surrogate, in a
    string or a key, now reads as U+FFFD: the record is kept (one cut emoji is
    not a reason to lose a conversation) and the mark shows where text was
    lost. A well-formed pair is one character by then and is untouched.
    """
    try:
        value = json.loads(
            text,
            parse_float=_finite_float,
            parse_int=_double_sized_int,
            parse_constant=_refuse_constant,
        )
    except RecursionError as exc:
        raise ValueError("JSON is nested too deeply to be anything this tool reads") from exc
    if _SURROGATE_ESCAPE.search(text) or _LONE_SURROGATE.search(text):
        value = _without_lone_surrogates(value)
    return value


def _without_lone_surrogates(value: Any) -> Any:
    """``value`` with each lone surrogate in its strings and keys made U+FFFD.

    Iterative rather than recursive, so a document as deep as the decoder
    accepted is walked without spending the stack the decoder just used.
    """

    def fix(text: str) -> str:
        return _LONE_SURROGATE.sub(REPLACEMENT_CHARACTER, text)

    if isinstance(value, str):
        return fix(value)
    pending: list[Any] = [value]
    while pending:
        node = pending.pop()
        if isinstance(node, list):
            for index, item in enumerate(node):
                if isinstance(item, str):
                    node[index] = fix(item)
                elif isinstance(item, (list, dict)):
                    pending.append(item)
        elif isinstance(node, dict):
            entries = list(node.items())
            node.clear()
            for key, item in entries:
                if isinstance(item, str):
                    item = fix(item)
                elif isinstance(item, (list, dict)):
                    pending.append(item)
                node[fix(key)] = item
    return value


#: The largest integer a stored record holds: no double is larger (about
#: 1.8e308), and ``loads_untrusted`` refuses an integer past it. The config and
#: rubric loaders hold every integer they read to it, so a number a run writes
#: into a receipt or a candidate is one the next command reads back.
MAX_STORED_INT = int(sys.float_info.max)
#: Its digits (309).
_DOUBLE_INT_DIGITS = len(str(MAX_STORED_INT))


def _double_sized_int(token: str) -> int:
    """An integer literal, refused when no double could hold it, as ``1e400`` is.

    Measured before: a stored validation's ``min_judge_preference_ratio`` of
    401 digits parsed as a Python int and made ``review`` fall with
    ``OverflowError`` from ``float()``, exit 70. JSON has no leading zeros, so
    the digits are the magnitude, and more of them than Python reads from text
    are refused here rather than by ``int`` with advice no reader can take.
    """
    digits = len(token.lstrip("-"))
    if digits <= _DOUBLE_INT_DIGITS:
        number = int(token)
        if abs(number) <= MAX_STORED_INT:
            return number
    raise ValueError(
        f"JSON integer of {digits} digits is larger than any double (the largest is about "
        "1.8e308); nothing this tool reads can carry it"
    )


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
