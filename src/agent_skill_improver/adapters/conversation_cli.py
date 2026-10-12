"""Conversation source backed by any command-line conversation tool.

Deliberately vendor-neutral. The adapter knows nothing about which tool it is
driving; it knows two **documented contracts** and a set of argv templates, and
the operator supplies the executable and arguments in their own gitignored
config. That is what makes this repository publishable: a tool name in an
adapter is a fact about somebody's environment, and it stays in their config.

Two contracts, both of which real conversation CLIs already satisfy.

**Discovery -- a line grammar over human-readable text.** Templates render to a
search command, and the output is parsed for three line shapes::

    1. "SOME TITLE" (2026-01-31)      <- numbered header: title and date
       Workspace: /path/to/workspace  <- optional, applies to the header above
       Thread: <thread-id>            <- the id; trailing text is ignored

The ``Thread:`` line is what commits a record, so a tool that prints the id
alongside a URL or any other suffix still parses. Discovery parses only what it
needs, so a tool that changes its human output degrades to "found nothing" --
which the daily driver reports as a stood-down run rather than as success. The
failure is visible instead of silently producing an empty batch that looks like
a clean day.

**Export -- the element-graph JSON contract** documented in
:mod:`agent_skill_improver.adapters.transcript_export`.

Argv templates use ``{placeholder}`` tokens: ``{since}`` and ``{limit}`` in
``discovery_argv`` (both required, because a discovery call with an implicit
page size samples whatever the tool most recently touched), ``{query}`` in
``query_argv``, ``{workspace}`` in ``workspace_argv``, and ``{thread_id}`` in
``export_argv``. ``{query}`` and ``{workspace}`` may instead appear inline in
``discovery_argv`` when the tool needs them in a specific position -- a
positional query, typically. An inline token whose value is empty removes the
whole argument rather than passing an empty string.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Sequence

from ..adapters.transcript_export import normalize_export
from ..domain.conversation import NormalizedConversation
from ..domain.discovery import (
    MAX_THREAD_ID_LENGTH,
    DiscoveredThread,
    DiscoveryPage,
    is_safe_thread_id,
    page_from_threads,
)
from ..domain.redaction import redact
from ..errors import RecordDefect, SourceUnavailable
from ..jsonio import loads_untrusted
from .bounded_process import (
    QUOTED_TAIL_CHARS,
    OutputOverflow,
    RunTimedOut,
    quotable,
    run_bounded,
)

__all__ = [
    "ConversationCliSource",
    "DiscoveredThread",
    "DiscoveryPage",
    "parse_discovery",
    "parse_discovery_output",
    "render_argv",
]

#: Defaults spell a plausible generic interface rather than any product's. They
#: exist so the schema has something to validate and the docs have something to
#: show; a real deployment overrides them in a gitignored config.
DEFAULT_DISCOVERY_ARGV: tuple[str, ...] = (
    "conversation-cli",
    "search",
    "--after",
    "{since}",
    "-n",
    "{limit}",
)
DEFAULT_QUERY_ARGV: tuple[str, ...] = ("{query}",)
DEFAULT_WORKSPACE_ARGV: tuple[str, ...] = ("-w", "{workspace}")
DEFAULT_EXPORT_ARGV: tuple[str, ...] = (
    "conversation-cli",
    "export",
    "--thread",
    "{thread_id}",
)

_THREAD_LINE = re.compile(r"^\s*Thread:\s*(?P<id>[^\s(]+)")
_HEADER_LINE = re.compile(
    r'^\s*(?P<number>\d+)\.\s+"(?P<title>.*)"\s*\((?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})\)'
)
_WORKSPACE_LINE = re.compile(r"^\s*Workspace:\s*(?P<path>.+?)\s*$")


def render_argv(template: Sequence[str], values: dict[str, str]) -> list[str]:
    """Substitute ``{placeholder}`` tokens, dropping args that render empty.

    An argument that is *exactly* one placeholder whose value is empty is
    removed. Substituting an empty string instead would pass a bare ``""`` to
    the tool, which is a different request from not passing the argument at all
    -- and for a positional query it usually means "match nothing".
    """
    out: list[str] = []
    for part in template:
        rendered = part
        for name, value in values.items():
            rendered = rendered.replace("{" + name + "}", value)
        if rendered == "" and part != "":
            continue
        out.append(rendered)
    return out


def parse_discovery_output(text: str) -> list[DiscoveredThread]:
    """The threads in a discovery listing; see :func:`parse_discovery`."""
    return parse_discovery(text)[0]


def _short(text: str, width: int) -> str:
    """A repr of untrusted listing text, cut to ``width`` for a receipt line.

    A cut repr keeps its closing quote, so the boundary between a title and
    an id stays readable on the line.
    """
    shown = repr(text)
    if len(shown) <= width:
        return shown
    return shown[: width - 4] + "..." + shown[-1]


def _unsafe_id_reason(raw_id: str) -> str:
    """Which rule of ``is_safe_thread_id`` the id failed, for the operator."""
    if not raw_id:
        return "is empty"
    if len(raw_id) > MAX_THREAD_ID_LENGTH:
        return f"is over-long ({len(raw_id)} characters; the limit is {MAX_THREAD_ID_LENGTH})"
    if raw_id[0] == "-":
        return "is option-shaped (it starts with a dash)"
    return "carries whitespace or control characters"


def _unsafe_id_entry(ordinal: str, raw_id: str, title: str) -> tuple[str, str]:
    """A passed-over entry for a listed id that cannot be a thread.

    The id is shown as a repr, cut short: it may carry the very control
    characters or option shape that made it unsafe, and the receipt is read
    in terminals and logs.
    """
    label = f"listing entry {ordinal} ({_short(title, 40)}) with id {_short(raw_id, 60)}"
    return (
        label,
        f"names an id that {_unsafe_id_reason(raw_id)}, so it was not treated as a thread "
        "and nothing was run for it",
    )


def parse_discovery(text: str) -> tuple[list[DiscoveredThread], list[tuple[str, str]]]:
    """Parse discovery output into thread records, per the documented grammar.

    Returns the threads and, beside them, every listed id that was NOT taken
    as a thread, with the reason. Measured before the second list existed: a
    forged ``Thread: --output=...`` line was dropped (rightly) and the receipt
    said nothing, so a listing under injection read exactly like a clean one
    that was simply shorter.

    Written as a pure function so it is tested against captured fixtures with
    no subprocess involved.
    """
    found: list[DiscoveredThread] = []
    passed_over: list[tuple[str, str]] = []
    title = ""
    date = ""
    workspace = ""
    number = ""  # the listing's own printed number, the one an operator greps for
    header_open = False  # a header has been read and no Thread: line has spent it
    seen_thread_lines = 0

    def spend_header() -> None:
        nonlocal title, date, number, workspace, header_open
        title = ""
        date = ""
        number = ""
        workspace = ""
        header_open = False

    def header_without_thread() -> None:
        # A listed conversation whose entry carries no Thread: line this parser
        # can read. Measured before: it vanished from the page with nothing
        # recorded, which is the same silence the unsafe-id case had.
        passed_over.append(
            (
                f"listing entry {number} ({_short(title, 40)})",
                "has no Thread: line this parser could read, so it was not treated as a "
                "thread and nothing was run for it",
            )
        )
        spend_header()

    for line in text.splitlines():
        header = _HEADER_LINE.match(line)
        if header:
            if header_open:
                header_without_thread()
            title = header.group("title")
            date = header.group("date")
            number = header.group("number")
            workspace = ""
            header_open = True
            continue
        ws = _WORKSPACE_LINE.match(line)
        if ws and not line.strip().startswith("Thread:"):
            workspace = ws.group("path")
            continue
        thread = _THREAD_LINE.match(line)
        if thread:
            seen_thread_lines += 1
            if not is_safe_thread_id(thread.group("id")):
                # An option-shaped or unprintable id is not a thread; it is text
                # that reached the listing. Dropped here so it never becomes an
                # argv element, and refused again in fetch() for ids that
                # arrive by any other route. Named, so the receipt shows it,
                # under the listing's printed number when a header gave one.
                ordinal = number or f"#{seen_thread_lines}"
                passed_over.append(_unsafe_id_entry(ordinal, thread.group("id"), title))
                spend_header()
                continue
            found.append(
                DiscoveredThread(
                    thread_id=thread.group("id"),
                    title=title,
                    workspace=workspace,
                    updated_date=date,
                )
            )
            spend_header()
            continue
        if line.strip().startswith("Thread:"):
            # A Thread: line the pattern could not read: blank after the colon,
            # or an id beginning with a parenthesis. Not a thread, and named.
            seen_thread_lines += 1
            rest = line.strip()[len("Thread:") :].strip()
            ordinal = number or f"#{seen_thread_lines}"
            passed_over.append(
                (
                    f"listing entry {ordinal} ({_short(title, 40)}) with id {_short(rest, 60)}",
                    "has a Thread: line this parser could not read as an id, so it was not "
                    "treated as a thread and nothing was run for it",
                )
            )
            spend_header()
    if header_open:
        header_without_thread()
    # De-duplicate while preserving discovery order.
    seen: set[str] = set()
    unique: list[DiscoveredThread] = []
    for item in found:
        if item.thread_id in seen:
            continue
        seen.add(item.thread_id)
        unique.append(item)
    return unique, passed_over


class ConversationCliSource:
    """Real source. Every subprocess call is funnelled through :meth:`_run`."""

    def __init__(
        self,
        *,
        discovery_argv: Sequence[str] = DEFAULT_DISCOVERY_ARGV,
        export_argv: Sequence[str] = DEFAULT_EXPORT_ARGV,
        query_argv: Sequence[str] = DEFAULT_QUERY_ARGV,
        workspace_argv: Sequence[str] = DEFAULT_WORKSPACE_ARGV,
        timeout_seconds: int = 120,
        raw_store: Any | None = None,
        max_export_bytes: int = 32 * 1024 * 1024,
        output_excerpt_chars: int = 2000,
        max_messages: int | None = None,
    ) -> None:
        self.discovery_argv = tuple(discovery_argv)
        self.export_argv = tuple(export_argv)
        self.query_argv = tuple(query_argv)
        self.workspace_argv = tuple(workspace_argv)
        self.timeout_seconds = timeout_seconds
        self.raw_store = raw_store
        self.max_export_bytes = max_export_bytes
        self.output_excerpt_chars = output_excerpt_chars
        self.max_messages = max_messages

    @property
    def executable(self) -> str:
        """The program discovery runs. Used by :meth:`preflight`."""
        return self.discovery_argv[0] if self.discovery_argv else ""

    # -- discovery -------------------------------------------------------
    def build_discovery_argv(
        self, *, since: str, limit: int, workspace: str | None = None, query: str | None = None
    ) -> list[str]:
        """Render the discovery command. Separated out so a test can read it."""
        values = {
            "since": since,
            "limit": str(limit),
            "query": query or "",
            "workspace": workspace or "",
        }
        argv = render_argv(self.discovery_argv, values)
        template_text = " ".join(self.discovery_argv)
        if query and "{query}" not in template_text:
            argv += render_argv(self.query_argv, values)
        if workspace and "{workspace}" not in template_text:
            argv += render_argv(self.workspace_argv, values)
        return argv

    def discover_page(
        self,
        *,
        since: str,
        limit: int,
        workspace: str | None = None,
        query: str | None = None,
    ) -> DiscoveryPage:
        """One bounded page, honest about whether it is a page or a total.

        The limit is always rendered explicitly, which is why ``{limit}`` is a
        required placeholder. A documented default page size is typically small,
        and a harvester that relies on it samples the most recently touched
        sliver of a corpus -- which is disproportionately orchestrator traffic
        when a loop is running continuously.
        """
        argv = self.build_discovery_argv(since=since, limit=limit, workspace=workspace, query=query)
        threads, passed_over = parse_discovery(self._run(argv))
        return page_from_threads(
            threads,
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

    # -- fetch -----------------------------------------------------------
    def fetch(self, thread_id: str) -> NormalizedConversation:
        if not is_safe_thread_id(thread_id):
            raise RecordDefect(
                f"refusing to export {thread_id!r}: a thread id is a command-line VALUE and "
                "must not start with a dash or contain whitespace or control characters; "
                "this one would have been read as an option by the conversation command",
                record_id=thread_id,
            )
        argv = render_argv(self.export_argv, {"thread_id": thread_id})
        # One oversized or undecodable export is a defect of THAT record. The
        # cap is enforced while the command's output is read, so the export is
        # never buffered past it.
        raw = self._run(argv, record_id=thread_id)
        try:
            payload = loads_untrusted(raw)
        except ValueError as exc:
            raise RecordDefect(f"export for {thread_id} is not JSON: {exc}", record_id=thread_id)
        if self.raw_store is not None:
            # The discovered id is untrusted. StateStore owns both confinement
            # and collision-safe leaf naming for this private artifact.
            self.raw_store.write_raw(thread_id, raw)
        return normalize_export(
            payload,
            output_excerpt_chars=self.output_excerpt_chars,
            max_messages=self.max_messages,
        )

    # -- plumbing --------------------------------------------------------
    def preflight(self) -> None:
        exe = self.executable
        if not exe:
            raise SourceUnavailable(
                "[source].discovery_argv is empty, so there is no command to run"
            )
        if shutil.which(exe) is None and not Path(exe).exists():
            raise SourceUnavailable(
                f"conversation command {exe!r} not found on PATH; set [source].discovery_argv "
                "and [source].export_argv in your config"
            )

    def _run(self, argv: Sequence[str], *, record_id: str | None = None) -> str:
        """Run the conversation command with its output bounded while read.

        ``max_export_bytes`` applies to every call, discovery included: a
        listing that never ends is as capable of exhausting the parent as an
        export is. With ``record_id`` set, an oversized or undecodable result is
        a :class:`RecordDefect` of that one record (quarantine it, keep the
        batch going); without it, it is the source being unusable. A timeout is
        always the source: the command, not the record, failed to answer.
        """
        # The command line is quoted into every failure reason below, and a
        # configured argv can carry a key as an option value just as stderr can.
        shown = redact(" ".join(argv)).text
        try:
            result = run_bounded(
                argv,
                timeout_seconds=self.timeout_seconds,
                max_output_bytes=self.max_export_bytes,
            )
        except FileNotFoundError as exc:
            raise SourceUnavailable(f"cannot execute {argv[0]!r}: {exc}") from exc
        except RunTimedOut as exc:
            raise SourceUnavailable(f"{shown} timed out after {self.timeout_seconds}s") from exc
        except OutputOverflow as exc:
            reason = (
                f"{shown} exceeds max_export_bytes ({self.max_export_bytes}); "
                "the command was stopped there"
            )
            if record_id is not None:
                raise RecordDefect(reason, record_id=record_id) from exc
            raise SourceUnavailable(reason) from exc
        if result.returncode != 0:
            # The end of stderr is where a CLI puts its reason; stdout stands in
            # only when stderr said nothing, blank lines included.
            stderr = result.stderr_tail.decode("utf-8", errors="replace").strip()
            stdout = result.stdout.decode("utf-8", errors="replace").strip()
            tail = quotable(stderr or stdout, limit=QUOTED_TAIL_CHARS, from_end=True)
            raise SourceUnavailable(f"{shown} exited {result.returncode}: {tail or '<no output>'}")
        try:
            return result.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            reason = f"{shown} produced output that is not UTF-8 ({exc.reason} at byte {exc.start})"
            if record_id is not None:
                raise RecordDefect(reason, record_id=record_id) from exc
            raise SourceUnavailable(reason) from exc
