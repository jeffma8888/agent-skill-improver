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
from pathlib import Path
from typing import Any, Sequence

from ..adapters.transcript_export import MAX_EXPORT_BYTES, normalize_export
from ..domain.conversation import NormalizedConversation
from ..domain.discovery import (
    MAX_THREAD_ID_LENGTH,
    DiscoveredThread,
    DiscoveryPage,
    is_safe_thread_id,
    page_from_threads,
)
from ..domain.redaction import NAMED_PATH_CHARS, quoted_value, redact
from ..errors import RecordDefect, SourceUnavailable
from ..jsonio import loads_untrusted
from .bounded_process import (
    QUOTED_TAIL_CHARS,
    OutputOverflow,
    program_problem,
    RunTimedOut,
    quotable,
    run_bounded,
    why_not_started,
)

__all__ = [
    "ConversationCliSource",
    "DiscoveredThread",
    "DiscoveryPage",
    "parse_discovery",
    "parse_discovery_output",
    "program_problem",
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
#: The path ends at its last character that is not space, so on one line, as
#: ``splitlines`` gives it, the space after the path is tried once, where
#: ``.+?\s*$`` tried it again from each place in it and took time in the square
#: of the run. A line of nothing but space after the word still reads as its
#: last space character, as it did.
_WORKSPACE_LINE = re.compile(r"^\s*Workspace:\s*(?P<path>.*?\S|[^\S\n])\s*$")


def render_argv(template: Sequence[str], values: dict[str, str]) -> list[str]:
    """Substitute ``{placeholder}`` tokens, dropping args that render empty.

    An argument that is *exactly* one placeholder whose value is empty is
    removed. Substituting an empty string instead would pass a bare ``""`` to
    the tool, which is a different request from not passing the argument at all
    -- and for a positional query it usually means "match nothing".

    A literal flag (an argument beginning with ``-`` and carrying no
    placeholder) immediately before a dropped argument is dropped with it.
    Measured before: ``--after {since}`` with an empty window rendered as
    ``--after -n 40``, so the flag took the limit's own flag as its value and
    the limit became a positional, a request nobody configured.

    The renderer cannot tell a flag that takes a value from one that does not,
    so a valueless flag written directly before a placeholder (``--json
    {query}``) goes with the empty value too; write such flags elsewhere in
    the template. ``--`` (end of options), a bare ``-`` and a negative number
    are literals that never introduce a value and are kept.

    Each placeholder is filled once, in one pass over the template argument,
    so a value is sent as written even when it holds a placeholder's name.
    Measured before (3.11 and 3.13, the same on both): the values were filled
    one after another, so a value naming a placeholder filled later had that
    one's value put in its place. Of 52 cases (a window, a query or a
    workspace naming another placeholder, alone or in other text, with that
    other value set or empty, for the default templates and one with the
    query inline), 28 sent other text than the value, and 6 dropped an
    argument: a window naming an empty query or workspace rendered empty and
    went with ``--after``, so discovery ran with no window, and a query naming
    an empty workspace was not sent, so the search was not narrowed by it.
    """
    out: list[str] = []
    sources: list[str] = []  # the template part each emitted argument came from
    fill = re.compile("|".join(re.escape("{" + name + "}") for name in values))
    for part in template:
        rendered = fill.sub(lambda found: values[found.group()[1:-1]], part) if values else part
        if rendered == "" and part != "":
            if sources and _is_literal_flag(sources[-1]):
                out.pop()
                sources.pop()
            continue
        out.append(rendered)
        sources.append(part)
    return out


def _is_literal_flag(template_part: str) -> bool:
    """A template argument that is a flag as written, with no placeholder in it.

    Judged on the template, not the rendered value: a rendered value that
    happens to begin with ``-`` came from the operator's config, not the
    template, and is not the flag introducing the next argument. ``--``, a
    bare ``-`` and a negative number begin with a dash and introduce nothing.
    """
    if "{" in template_part or not template_part.startswith("-"):
        return False
    if template_part in ("-", "--"):
        return False
    return not _is_number(template_part)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


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
        max_export_bytes: int = MAX_EXPORT_BYTES,
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
        if workspace:
            # A listing is numbered per query. "listing entry 1" in one workspace
            # and in the next are two conversations; merge_pages de-duplicates
            # passed-over items by their text, so the workspace has to be in it
            # or two become one on the receipt. The workspace goes in as written;
            # the entry is masked where it is recorded (``shown_passed_over``),
            # after merge_pages has told them apart.
            passed_over = [
                (f"{entry} in workspace {workspace}", reason) for entry, reason in passed_over
            ]
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
        conversation = normalize_export(
            payload,
            output_excerpt_chars=self.output_excerpt_chars,
            max_messages=self.max_messages,
        )
        if conversation.thread_id != thread_id:
            # Measured before: the run scored whatever conversation the command
            # answered with, under this record, and filed its failure cases
            # under the id the export named. A failing thread answered with a
            # passing one proposed nothing; a passing thread answered with a
            # failing one counted that failure twice; every record read as
            # scored. Ids are compared exactly, as the command line passed one.
            raise RecordDefect(
                f"export for {thread_id} names thread {quoted_value(conversation.thread_id)}; "
                "the conversation command answered for another conversation, so this "
                "export was not taken as either one",
                record_id=thread_id,
            )
        return conversation

    # -- plumbing --------------------------------------------------------
    def preflight(self) -> None:
        exe = self.executable
        if not exe:
            raise SourceUnavailable(
                "[source].discovery_argv is empty, so there is no command to run"
            )
        if not self.export_argv:
            raise SourceUnavailable("[source].export_argv is empty, so there is no command to run")
        # Both programs are checked: the export command may be a different
        # program from the discovery one, and a run that discovers and then
        # cannot export has already spent the discovery.
        for program in dict.fromkeys((exe, self.export_argv[0])):
            problem = program_problem(program)
            if problem:
                # Masked and cut as the start refusal names it (``_run``):
                # measured before, a key-shaped directory in the path was
                # printed raw, twice by ``doctor``, and a path of 874
                # characters made a ``doctor`` line of 1,076.
                named = quoted_value(program, limit=NAMED_PATH_CHARS)
                raise SourceUnavailable(
                    f"conversation command {named} {problem}; set [source].discovery_argv "
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
        except OSError as exc:
            # Not found, not executable, a directory, a wrong binary format: the
            # program could not be started, which is the source, not a record,
            # and not a bug in this tool. Measured before: a configured command
            # at mode 644 passed preflight (the file exists) and then ended the
            # run as an internal error with a traceback. The program is named
            # once, masked and cut: the error's own text named it again, raw.
            program = quoted_value(argv[0], limit=NAMED_PATH_CHARS)
            raise SourceUnavailable(f"cannot execute {program}: {why_not_started(exc)}") from exc
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
