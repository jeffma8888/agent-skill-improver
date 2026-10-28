"""Offline conversation source over a directory of export-shaped JSON files.

This exists for two reasons, and the second one matters more.

1. The whole test suite runs with no network, no model and no external binary.
2. An operator can point the loop at a directory of files and get the *same code
   path* as production -- same normalizer, same scorer, same gate. A test harness
   that bypasses the real pipeline proves very little about the real pipeline.

Files are read in sorted order for reproducibility, but that order is never used
to *bound* a batch: bounding happens in the selection stage, which is fair by
construction. Truncating a sorted list is a permanent filter, not a sample.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from ..jsonio import loads_untrusted
from ..domain.candidate import one_line_label
from ..domain.conversation import NormalizedConversation
from ..domain.discovery import (
    MAX_THREAD_ID_LENGTH,
    DiscoveredThread,
    DiscoveryPage,
    page_from_threads,
)
from ..domain.redaction import quoted_value, quoted_values
from ..errors import RecordDefect, SourceUnavailable
from .transcript_export import MAX_EXPORT_BYTES, normalize_export

#: What an entry that is not a regular file is called in its passed-over reason.
_NOT_A_FILE = (
    (stat.S_ISDIR, "a directory"),
    (stat.S_ISFIFO, "a fifo"),
    (stat.S_ISCHR, "a character device"),
    (stat.S_ISBLK, "a block device"),
    (stat.S_ISSOCK, "a socket"),
)


class _NotAnExport(Exception):
    """An entry :func:`_read_export` refused without reading it; the message is why."""


def _refuse_what_is_not_a_file(mode: int) -> None:
    """Raise :class:`_NotAnExport` naming the kind, unless ``mode`` is a regular file's."""
    if not stat.S_ISREG(mode):
        kind = next((name for test, name in _NOT_A_FILE if test(mode)), "not a file")
        raise _NotAnExport(f"is not a regular file ({kind}), so it was not read")


def _read_export(path: Path, limit: int) -> bytes:
    """The bytes of one export, read only if it is a regular file of at most ``limit`` bytes.

    Judged by what the path names before it is opened, so a fifo, a device, a
    socket or a directory is refused by kind without being opened at all
    (opening a device can act on it: a tty opened by a scheduler-run session
    leader becomes its controlling terminal), whether it sits in the directory
    or a link there points at it. Judged again on the open handle, which
    decides, because the entry can be swapped between the two: it is opened
    ``O_NONBLOCK`` and ``O_NOCTTY``, so a fifo swapped in opens without waiting
    for a writer and is refused, and a tty is never taken as the terminal.
    Links are still followed, as the plain read before this followed them.
    Measured before: one fifo named ``stuck.json`` beside a
    good export parked ``doctor`` and ``daily`` for good, and the read had no
    bound at all, where the command-line source stops at ``MAX_EXPORT_BYTES``.
    An ``OSError`` (a missing or unreadable entry) is the caller's to name.
    """
    _refuse_what_is_not_a_file(os.stat(path).st_mode)
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOCTTY", 0)
    fd = os.open(path, flags)
    try:
        _refuse_what_is_not_a_file(os.fstat(fd).st_mode)
        chunks: list[bytes] = []
        size = 0
        while size <= limit:
            chunk = os.read(fd, min(1024 * 1024, limit + 1 - size))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            size += len(chunk)
        raise _NotAnExport(f"is larger than {limit} bytes, so it was not read past that")
    finally:
        os.close(fd)


class FixtureConversationSource:
    """Reads export-shaped JSON files from a directory.

    The shape is the element-graph contract documented in
    :mod:`agent_skill_improver.adapters.transcript_export`.
    """

    def __init__(
        self,
        directory: Path,
        *,
        output_excerpt_chars: int = 2000,
        max_messages: int | None = None,
        max_export_bytes: int = MAX_EXPORT_BYTES,
    ) -> None:
        self.directory = Path(directory)
        self.max_export_bytes = max_export_bytes
        self.output_excerpt_chars = output_excerpt_chars
        self.max_messages = max_messages

    def preflight(self) -> None:
        if not self.directory.is_dir():
            raise SourceUnavailable(
                f"fixtures_dir {one_line_label(str(self.directory))} is not a directory"
            )
        # Listing it is the question the run will ask; asked here so ``doctor``
        # and the daily preflight refuse before anything is spent.
        self._entries()

    def _entries(self) -> list[Path]:
        """Every ``*.json`` entry of the directory, or the source is unavailable.

        Measured before: ``Path.glob`` swallows the error of a directory this
        process cannot list (mode 000), so a run over one reported an empty
        window with ``is_total`` true, zero passed over and a clean ``doctor``,
        indistinguishable from a directory with nothing in it.
        """
        try:
            with os.scandir(self.directory) as it:
                names = [entry.name for entry in it if entry.name.endswith(".json")]
        except OSError as exc:
            raise SourceUnavailable(
                f"fixtures_dir {one_line_label(str(self.directory))} "
                f"cannot be listed ({exc.strerror or exc}); "
                "nothing in it can be discovered, so this is not an empty window"
            ) from exc
        return sorted(self.directory / name for name in names)

    def _scan(self) -> tuple[list[tuple[Path, dict[str, Any]]], list[tuple[str, str]]]:
        """Every export in the directory that reads as a thread, and every entry
        that does not, with the reason.

        Measured before the second list existed: an export at mode 000, one
        that was not JSON and one that was a JSON array sat beside five good
        ones, and the run reported five ingested with ``is_total`` true and no
        word about the three. Stepping over them is right (one bad file must
        not end the run); doing it silently was not.
        """
        accepted: list[tuple[Path, dict[str, Any]]] = []
        passed_over: list[tuple[str, str]] = []
        digests: dict[Path, str] = {}
        for path in self._entries():
            try:
                data = _read_export(path, self.max_export_bytes)
            except _NotAnExport as exc:
                passed_over.append((str(path), str(exc)))
                continue
            except OSError as exc:
                passed_over.append((str(path), f"cannot be read ({exc.strerror or exc})"))
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                passed_over.append((str(path), f"is not UTF-8 text ({exc})"))
                continue
            try:
                payload = loads_untrusted(text)
            except ValueError as exc:
                passed_over.append((str(path), f"is not valid JSON ({exc})"))
                continue
            if not isinstance(payload, dict):
                passed_over.append(
                    (str(path), f"is JSON but not an object ({type(payload).__name__})")
                )
                continue
            thread = payload.get("thread")
            if not isinstance(thread, dict):
                passed_over.append((str(path), "has no thread object"))
                continue
            if not isinstance(thread.get("id"), str) or not thread.get("id"):
                passed_over.append((str(path), "has a thread without an id"))
                continue
            thread_id = thread["id"]
            if len(thread_id) > MAX_THREAD_ID_LENGTH:
                # An id is its record's key in the receipt, the checkpoint and
                # each scorecard and failure case, so it is held to the length
                # a listed id is held to (the command-line source refuses a
                # longer one at discovery). Measured before: an export claiming
                # a 2,000,000-character id was taken, and the receipt and
                # ``daily --json`` grew to 4.0 MB (10 KB without it), the
                # checkpoint and the scorecard to 2.0 MB each.
                passed_over.append(
                    (
                        str(path),
                        f"claims thread {quoted_value(thread_id)}, an id that is over-long "
                        f"({len(thread_id)} characters; the limit is {MAX_THREAD_ID_LENGTH}, "
                        "as for an id a conversation command lists), so it was not taken "
                        "as a thread",
                    )
                )
                continue
            accepted.append((path, payload))
            digests[path] = hashlib.sha256(data).hexdigest()
        return self._one_export_per_thread(accepted, passed_over, digests), passed_over

    @staticmethod
    def _one_export_per_thread(
        accepted: list[tuple[Path, dict[str, Any]]],
        passed_over: list[tuple[str, str]],
        digests: dict[Path, str],
    ) -> list[tuple[Path, dict[str, Any]]]:
        """``accepted`` with each thread claimed by one export at most.

        Exports that claim one thread and differ in any byte are an ambiguity,
        not a choice: none of them is taken and each is passed over naming the
        others. A repeat of an export, the same bytes again, is taken once,
        from the first in directory order, and the rest are passed over as
        repeats. Bytes, not the parsed value: ``true == 1`` in Python, and the
        normalizer reads the two differently. Measured before: an export claiming a healthy
        thread's id, sorted ahead of it, was fetched in its place, and the
        run proposed nothing from that thread, with nothing passed over and
        ``doctor`` clean; one sorted after it was dropped without a word.
        """
        claims: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
        for path, payload in accepted:
            claims.setdefault(payload["thread"]["id"], []).append((path, payload))
        kept: list[tuple[Path, dict[str, Any]]] = []
        for path, payload in accepted:
            thread_id = payload["thread"]["id"]
            group = claims[thread_id]
            if len(group) == 1:
                kept.append((path, payload))
                continue
            first = group[0][0]
            if all(digests[other] == digests[first] for other, _ in group):
                if path == first:
                    kept.append((path, payload))
                else:
                    passed_over.append(
                        (
                            str(path),
                            f"repeats, byte for byte, the export of thread "
                            f"{quoted_value(thread_id)} in "
                            f"{quoted_value(first.name)}, which is taken once",
                        )
                    )
                continue
            others = [other.name for other, _ in group if other != path]
            passed_over.append(
                (
                    str(path),
                    f"claims thread {quoted_value(thread_id)}, as {quoted_values(others)} "
                    f"also {'does' if len(others) == 1 else 'do'} with other bytes; this "
                    "source cannot tell which is the conversation, so it takes none of them",
                )
            )
        return kept

    def _payloads(self) -> list[tuple[Path, dict[str, Any]]]:
        return self._scan()[0]

    def passed_over(self) -> list[tuple[str, str]]:
        """Entries in the directory this source cannot take as a thread, with reasons."""
        return self._scan()[1]

    def discover_page(
        self,
        *,
        since: str,
        limit: int,
        workspace: str | None = None,
        query: str | None = None,
    ) -> DiscoveryPage:
        """A page over the fixture directory, bounded exactly like the real source.

        The bound is applied here and the truncation flag is derived, so a fixture
        run exercises the same limit-honesty path as production instead of
        pretending the corpus is always small enough.
        """
        accepted, passed_over = self._scan()
        return page_from_threads(
            self._threads_of(accepted, workspace=workspace),
            limit=limit,
            since=since,
            workspace=workspace or "",
            query=query or "",
            passed_over=passed_over,
        )

    def discover(
        self,
        *,
        since: str,
        limit: int,
        workspace: str | None = None,
        query: str | None = None,
    ) -> list[DiscoveredThread]:
        return list(
            self.discover_page(since=since, limit=limit, workspace=workspace, query=query).threads
        )

    def _threads_of(
        self, accepted: list[tuple[Path, dict[str, Any]]], *, workspace: str | None = None
    ) -> list[DiscoveredThread]:
        found: list[DiscoveredThread] = []
        for _, payload in accepted:
            thread = payload["thread"]  # _scan accepted only payloads with one
            thread_id = str(thread["id"])
            ws = str(thread.get("workspacePath", ""))
            if workspace and ws != workspace:
                continue
            found.append(
                DiscoveredThread(
                    thread_id=thread_id,
                    title=str(thread.get("title", "")),
                    workspace=ws,
                    updated_date="",
                )
            )
        return found

    def fetch(self, thread_id: str) -> NormalizedConversation:
        accepted, passed_over = self._scan()
        for path, payload in accepted:
            thread = payload.get("thread")
            if isinstance(thread, dict) and thread.get("id") == thread_id:
                return normalize_export(
                    payload,
                    output_excerpt_chars=self.output_excerpt_chars,
                    max_messages=self.max_messages,
                )
        if not passed_over:
            raise RecordDefect(
                f"no fixture for thread {thread_id!r} in {one_line_label(str(self.directory))}",
                record_id=thread_id,
            )
        # Measured before: an export discovered a moment ago and then made
        # unreadable (mode 000, truncated mid-write) was reported as "no
        # fixture", as if the thread had never been there. The directory holds
        # entries this scan could not take as a thread; one of them may be it.
        shown = "; ".join(one_line_label(f"{entry} {reason}") for entry, reason in passed_over[:5])
        more = len(passed_over) - 5
        if more > 0:
            shown += f"; +{more} more"
        raise RecordDefect(
            f"no readable fixture for thread {thread_id!r} "
            f"in {one_line_label(str(self.directory))}; "
            f"{len(passed_over)} entr{'y' if len(passed_over) == 1 else 'ies'} there could not "
            f"be taken as a thread and may be it: {shown}",
            record_id=thread_id,
        )
