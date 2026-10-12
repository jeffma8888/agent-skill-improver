"""A page of results is not a total, and the difference decides every rate.

The concrete hazard: a conversation search has a small default page size. A
harvester that accepts the default reads the most recently touched sliver of the
corpus, which in an orchestrated environment is disproportionately machine
traffic. Worse, nothing errors, so every downstream percentage is computed over a
population nobody chose.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import json
import shutil

from agent_skill_improver.adapters.conversation_cli import (
    ConversationCliSource,
    parse_discovery,
    parse_discovery_output,
    program_problem,
)
from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.errors import RecordDefect, SourceUnavailable
from agent_skill_improver.domain.discovery import (
    DiscoveredThread,
    DiscoveryPage,
    error_page,
    is_safe_thread_id,
    merge_pages,
    page_from_threads,
    passed_over_note,
)


def threads(n: int, prefix: str = "t") -> list[DiscoveredThread]:
    return [DiscoveredThread(thread_id=f"{prefix}{i}", title=f"title {i}") for i in range(n)]


class PageHonestyTests(unittest.TestCase):
    def test_a_short_page_is_a_total(self):
        page = page_from_threads(threads(3), limit=10)
        self.assertFalse(page.truncated_by_limit)
        self.assertTrue(page.is_total)

    def test_an_exactly_full_page_is_never_reported_as_a_total(self):
        page = page_from_threads(threads(10), limit=10)
        self.assertTrue(page.truncated_by_limit)
        self.assertFalse(
            page.is_total,
            "a page that came back exactly full is the ambiguous case: the source "
            "declined to say whether more exist, so claiming a total is a guess",
        )

    def test_an_overfull_response_is_clipped_and_flagged(self):
        page = page_from_threads(threads(25), limit=10)
        self.assertEqual(page.returned, 10)
        self.assertTrue(page.truncated_by_limit)

    def test_the_truncation_flag_is_derived_not_supplied(self):
        # Deriving it means a caller cannot forget to set it, which is the silent
        # failure this guards.
        page = page_from_threads(threads(10), limit=10)
        self.assertTrue(page.truncated_by_limit)
        blind = DiscoveryPage(threads=tuple(threads(10)), limit=10)
        self.assertFalse(
            blind.truncated_by_limit,
            "constructing the dataclass directly is exactly the mistake "
            "page_from_threads exists to prevent",
        )

    def test_a_full_page_carries_a_human_readable_honesty_note(self):
        note = page_from_threads(threads(10), limit=10).honesty_note
        self.assertTrue(note)
        self.assertIn("10", note)

    def test_a_short_page_note_does_not_claim_truncation(self):
        note = page_from_threads(threads(2), limit=10).honesty_note.lower()
        self.assertNotIn("more", note.replace("no more", ""))

    def test_an_unlimited_page_keeps_everything(self):
        page = page_from_threads(threads(7), limit=0)
        self.assertEqual(page.returned, 7)
        self.assertFalse(page.truncated_by_limit)

    def test_an_error_page_is_not_a_total_and_carries_no_threads(self):
        page = error_page(limit=10, error="source unreachable")
        self.assertEqual(page.returned, 0)
        self.assertFalse(page.is_total)
        self.assertEqual(page.error, "source unreachable")

    def test_an_error_page_is_distinct_from_an_empty_successful_page(self):
        empty_ok = page_from_threads([], limit=10)
        self.assertTrue(empty_ok.is_total, "genuinely zero results is a total")
        self.assertFalse(error_page(limit=10, error="boom").is_total)


class MergeTests(unittest.TestCase):
    def test_duplicates_across_pages_are_removed_once(self):
        a = page_from_threads(threads(3, "x"), limit=10)
        b = page_from_threads(threads(3, "x"), limit=10)
        merged, summary = merge_pages([a, b])
        self.assertEqual(len(merged), 3)
        self.assertEqual(summary["threads_returned"], 3)

    def test_one_truncated_page_makes_the_whole_merge_non_total(self):
        clean = page_from_threads(threads(2, "a"), limit=10)
        full = page_from_threads(threads(10, "b"), limit=10)
        _, summary = merge_pages([clean, full])
        self.assertFalse(summary["is_total"])
        self.assertTrue(summary["any_page_truncated_by_limit"])

    def test_one_failed_page_makes_the_whole_merge_non_total(self):
        clean = page_from_threads(threads(2, "a"), limit=10)
        broken = error_page(limit=10, error="workspace unreadable")
        _, summary = merge_pages([clean, broken])
        self.assertFalse(
            summary["is_total"],
            "a partial harvest reported as complete is how a wrong denominator "
            "reaches every downstream number",
        )

    def test_all_pages_clean_yields_a_total(self):
        _, summary = merge_pages(
            [
                page_from_threads(threads(2, "a"), limit=10),
                page_from_threads(threads(1, "b"), limit=10),
            ]
        )
        self.assertTrue(summary["is_total"])

    def test_every_page_is_preserved_in_the_summary_for_audit(self):
        pages = [page_from_threads(threads(1, "a"), limit=5), error_page(limit=5, error="e")]
        _, summary = merge_pages(pages)
        self.assertEqual(len(summary["pages"]), 2)

    def test_merging_nothing_is_a_total_of_nothing(self):
        merged, summary = merge_pages([])
        self.assertEqual(merged, [])
        self.assertTrue(summary["is_total"])

    def test_what_a_page_passed_over_reaches_the_summary_named_once(self):
        # The same directory is scanned once per workspace, so the same bad
        # export is on every page; the summary names it once.
        skipped = (("/x/a.json", "is not valid JSON"), ("/x/b.json", "has no thread object"))
        a = page_from_threads(threads(2, "a"), limit=10, workspace="w1", passed_over=skipped)
        b = page_from_threads(threads(1, "b"), limit=10, workspace="w2", passed_over=skipped[:1])
        _, summary = merge_pages([a, b])
        self.assertEqual(
            summary["passed_over"],
            [
                {"entry": "/x/a.json", "reason": "is not valid JSON"},
                {"entry": "/x/b.json", "reason": "has no thread object"},
            ],
        )
        self.assertEqual(a.to_dict()["passed_over"], summary["passed_over"])
        self.assertTrue(summary["is_total"], "the window was read whole; some of it did not parse")
        self.assertEqual(page_from_threads(threads(1), limit=10).passed_over, ())
        self.assertEqual(merge_pages([])[1]["passed_over"], [])

    def test_the_note_names_at_most_five_entries_and_counts_the_rest(self):
        entries = [{"entry": f"/x/{i}.json", "reason": "is not valid JSON"} for i in range(7)]
        note = passed_over_note(entries)
        self.assertTrue(note.startswith("discovery passed over 7 source entries that did not"))
        for i in range(5):
            self.assertIn(f"/x/{i}.json is not valid JSON", note)
        self.assertNotIn("/x/5.json", note)
        self.assertTrue(note.endswith("; +2 more"))
        one = passed_over_note(entries[:1])
        self.assertIn("1 source entry that", one)
        self.assertNotIn("more", one)
        self.assertEqual(passed_over_note([]), "", "a clean receipt gets no note")


class FixtureSourcePassedOverTests(unittest.TestCase):
    """Measured before: five good exports beside an export at mode 000, one
    that was not JSON and one that was a JSON array gave a receipt of five
    ingested, ``is_total`` true, and no word about the three; ``doctor`` said
    the source was ok. Stepping over them is right; silently was not.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.exports = Path(self._tmp.name) / "exports"
        shutil.copytree(Path(__file__).parent / "fixtures" / "exports", self.exports)
        self.good = sorted(p.name for p in self.exports.glob("*.json"))

    def source(self) -> FixtureConversationSource:
        return FixtureConversationSource(self.exports)

    def test_each_way_an_export_fails_to_read_as_a_thread_is_named(self):
        (self.exports / "0-garbage.json").write_text("{not json", encoding="utf-8")
        (self.exports / "0-list.json").write_text("[1, 2]", encoding="utf-8")
        (self.exports / "0-nothread.json").write_text('{"x": 1}', encoding="utf-8")
        (self.exports / "0-noid.json").write_text(
            json.dumps({"thread": {"title": "t"}}), encoding="utf-8"
        )
        (self.exports / "0-emptyid.json").write_text(
            json.dumps({"thread": {"id": ""}}), encoding="utf-8"
        )
        (self.exports / "0-latin1.json").write_bytes(b'{"thread": {"id": "\xff"}}')
        passed = dict(self.source().passed_over())
        by_name = {Path(k).name: v for k, v in passed.items()}
        self.assertTrue(by_name["0-garbage.json"].startswith("is not valid JSON ("))
        self.assertEqual(by_name["0-list.json"], "is JSON but not an object (list)")
        self.assertEqual(by_name["0-nothread.json"], "has no thread object")
        self.assertEqual(by_name["0-noid.json"], "has a thread without an id")
        self.assertEqual(by_name["0-emptyid.json"], "has a thread without an id")
        self.assertTrue(by_name["0-latin1.json"].startswith("is not UTF-8 text ("))
        self.assertEqual(len(by_name), 6, "the good exports are not in the list")
        page = self.source().discover_page(since="", limit=50)
        self.assertEqual(len(page.threads), 5, "discovery is unchanged by what it steps over")
        self.assertEqual(dict(page.passed_over), passed, "the page carries the same list")
        self.assertFalse(page.truncated_by_limit)
        self.assertTrue(page.is_total)

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file")
    def test_an_export_this_process_cannot_read_is_passed_over_not_fatal(self):
        locked = self.exports / "0-locked.json"
        locked.write_text("{}", encoding="utf-8")
        os.chmod(locked, 0o000)
        try:
            page = self.source().discover_page(since="", limit=50)
            self.assertEqual(len(page.threads), 5)
            self.assertEqual(
                page.passed_over, ((str(locked), "cannot be read (Permission denied)"),)
            )
            # fetch reads the same scan and still finds a good thread
            thread_id = page.threads[0].thread_id
            self.assertEqual(self.source().fetch(thread_id).thread_id, thread_id)
        finally:
            os.chmod(locked, 0o644)

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file")
    def test_a_discovered_export_made_unreadable_before_fetch_is_not_called_absent(self):
        # Measured before: "no fixture for thread X", as if it had never been
        # listed, when the file was there and this process could not read it.
        page = self.source().discover_page(since="", limit=50)
        thread_id = page.threads[0].thread_id
        victim = next(
            path
            for path, payload in self.source()._payloads()
            if payload["thread"]["id"] == thread_id
        )
        os.chmod(victim, 0o000)
        try:
            with self.assertRaises(RecordDefect) as caught:
                self.source().fetch(thread_id)
        finally:
            os.chmod(victim, 0o644)
        message = str(caught.exception)
        self.assertIn(f"no readable fixture for thread {thread_id!r}", message)
        self.assertIn("1 entry there did not read as a thread and may be it", message)
        self.assertIn(f"{victim} cannot be read (Permission denied)", message)
        self.assertEqual(caught.exception.record_id, thread_id)

    def test_a_thread_never_in_a_clean_directory_is_still_simply_absent(self):
        with self.assertRaises(RecordDefect) as caught:
            self.source().fetch("1700000000999-9")
        self.assertTrue(str(caught.exception).startswith("no fixture for thread '1700000000999-9'"))
        self.assertNotIn("may be it", str(caught.exception))

    def test_fetch_names_at_most_five_passed_over_entries_and_counts_the_rest(self):
        for n in range(7):
            (self.exports / f"0-bad{n}.json").write_text("[]", encoding="utf-8")
        with self.assertRaises(RecordDefect) as caught:
            self.source().fetch("1700000000999-9")
        message = str(caught.exception)
        self.assertIn("7 entries there did not read as a thread", message)
        self.assertEqual(message.count("is JSON but not an object (list)"), 5)
        self.assertTrue(message.endswith("; +2 more"))

    def test_a_clean_directory_passes_nothing_over(self):
        self.assertEqual(self.source().passed_over(), [])
        self.assertEqual(self.source().discover_page(since="", limit=50).passed_over, ())


class RawExportConfinementTests(unittest.TestCase):
    class _Export(ConversationCliSource):
        def __init__(self, raw: str, **kwargs):
            super().__init__(**kwargs)
            self.raw = raw

        def _run(self, argv, **kwargs):  # type: ignore[override]
            return self.raw

    def test_a_discovered_thread_id_cannot_escape_the_state_raw_directory(self):
        fixture = Path(__file__).parent / "fixtures" / "exports" / "compliant.json"
        raw = fixture.read_text(encoding="utf-8")
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            state = StateStore(root / "state")
            state.ensure()
            source = self._Export(raw, raw_store=state)

            source.fetch("x/../../../outside")

            self.assertFalse((root / "outside.json").exists())
            written = list((root / "state" / "raw").glob("*.json"))
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0].read_text(encoding="utf-8"), raw)


class SourceOutputBoundsTests(unittest.TestCase):
    """The conversation command is a subprocess too, and it gets the same bounds.

    ``max_export_bytes`` used to be compared against the export after the whole
    of stdout had been captured, and discovery had no cap at all. A command that
    streamed without end (a wrapper stuck in a loop, a corpus far larger than
    expected) therefore cost the daily run that much memory before anything
    refused. The same cap now stops the read, and one oversized or undecodable
    export is a record defect rather than a crash of the pass.
    """

    FLOOD = (
        "import sys\n"
        "chunk = b'x' * (1 << 20)\n"
        "for _ in range({chunks}):\n"
        "    sys.stdout.buffer.write(chunk)\n"
        "    sys.stdout.buffer.flush()\n"
        "open(sys.argv[1], 'w').write('finished')\n"
    )

    def _source(self, program: str, *extra: str, **kwargs) -> ConversationCliSource:
        argv = (sys.executable, "-c", program, *extra, "{thread_id}")
        disc = (sys.executable, "-c", program, *extra, "{since}", "{limit}")
        return ConversationCliSource(export_argv=argv, discovery_argv=disc, **kwargs)

    def test_a_flooding_export_is_a_record_defect_stopped_at_the_cap(self):
        with TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "finished"
            source = self._source(
                self.FLOOD.format(chunks=256),
                str(sentinel),
                max_export_bytes=64 * 1024,
                timeout_seconds=120,
            )
            with self.assertRaises(RecordDefect) as ctx:
                source.fetch("thread-1")
            self.assertIn("max_export_bytes", str(ctx.exception))
            self.assertEqual(ctx.exception.record_id, "thread-1")
            self.assertFalse(
                sentinel.exists(),
                "the child ran to completion: every byte was buffered before the cap applied",
            )

    def test_a_flooding_discovery_is_stopped_at_the_cap_too(self):
        with TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "finished"
            source = self._source(
                self.FLOOD.format(chunks=256),
                str(sentinel),
                max_export_bytes=64 * 1024,
                timeout_seconds=120,
            )
            with self.assertRaises(SourceUnavailable):
                source.discover_page(since="1d", limit=5)
            self.assertFalse(sentinel.exists())

    def test_an_export_that_is_not_utf8_is_a_record_defect_not_a_crash(self):
        source = self._source('import sys;sys.stdout.buffer.write(b\'{"a": "\\xff"}\')')
        with self.assertRaises(RecordDefect) as ctx:
            source.fetch("thread-2")
        self.assertEqual(ctx.exception.record_id, "thread-2")

    def test_a_hanging_command_is_unavailable_and_the_child_is_reaped(self):
        program = (
            "import sys,os,time\n"
            "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
            "sys.stdout.flush()\n"
            "time.sleep(600)\n"
        )
        with TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "pid"
            source = self._source(program, str(pidfile), timeout_seconds=1)
            with self.assertRaises(SourceUnavailable) as ctx:
                source.fetch("thread-3")
            self.assertIn("timed out", str(ctx.exception))
            pid = int(pidfile.read_text())
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive = False
            else:
                alive = True
            self.assertFalse(alive, "a timed-out conversation command must be killed")

    def test_a_failing_export_names_the_exit_code_and_quotes_a_bounded_stderr_tail(self):
        """The command, not the record, failed to answer, so this is the source
        being unavailable. The operator gets the exit code and the END of stderr
        (where a CLI puts its reason), at most 500 characters of it, and never
        the partial stdout when stderr said anything at all."""
        program = (
            "import sys\n"
            "sys.stdout.write('partial stdout')\n"
            "sys.stderr.write('E' * 5000 + ' THE-REASON')\n"
            "sys.exit(7)\n"
        )
        with self.assertRaises(SourceUnavailable) as ctx:
            self._source(program).fetch("thread-5")
        message = str(ctx.exception)
        self.assertIn("exited 7: ", message)
        quoted = message.split("exited 7: ", 1)[1]
        self.assertTrue(quoted.endswith("THE-REASON"), quoted[-40:])
        self.assertEqual(len(quoted), 500, "the documented bound is exactly 500 characters")
        self.assertNotIn("partial stdout", quoted)

    def test_a_failing_export_that_prints_a_credential_has_it_redacted_in_the_reason(self):
        """The quoted tail becomes a deferral reason, then a receipt line, a
        checkpoint entry and a scheduler-log line. A CLI that dies printing the
        key it was refused with must not seed four copies of that key."""
        planted = "".join(("AKIA", "ABCDEFGHIJKLMNOP"))
        # Assembled inside the child so the key is not on the command line too.
        program = (
            "import sys\n"
            "sys.stderr.write('401 unauthorized for key ' + 'AKIA' + 'ABCDEFGHIJKLMNOP')\n"
            "sys.exit(1)\n"
        )
        with self.assertRaises(SourceUnavailable) as ctx:
            self._source(program).fetch("thread-8")
        message = str(ctx.exception)
        self.assertIn("exited 1: 401 unauthorized for key [REDACTED:aws-access-key-id]", message)
        self.assertNotIn(planted, message)

    def test_a_credential_in_the_configured_command_line_is_redacted_from_the_reason(self):
        """The failure reason quotes the command that failed. A configured argv
        carrying a key as an option value would put that key in the same four
        places the stderr tail reaches."""
        planted = "".join(("AKIA", "ABCDEFGHIJKLMNOP"))
        with self.assertRaises(SourceUnavailable) as ctx:
            self._source("import sys; sys.exit(2)", "--key", planted).fetch("thread-9")
        message = str(ctx.exception)
        self.assertIn("--key [REDACTED:aws-access-key-id]", message)
        self.assertNotIn(planted, message)

    def test_a_failing_export_that_said_nothing_says_so(self):
        with self.assertRaises(SourceUnavailable) as ctx:
            self._source("import sys; sys.exit(2)").fetch("thread-6")
        self.assertIn("exited 2: <no output>", str(ctx.exception))

    def test_a_failing_export_with_only_blank_stderr_is_quoted_from_stdout(self):
        program = "import sys\nsys.stdout.write('reason on stdout')\nsys.stderr.write('  \\n')\nsys.exit(3)\n"
        with self.assertRaises(SourceUnavailable) as ctx:
            self._source(program).fetch("thread-7")
        self.assertIn("exited 3: reason on stdout", str(ctx.exception))

    def test_the_control_export_under_the_cap_is_normalized(self):
        fixture = Path(__file__).parent / "fixtures" / "exports" / "compliant.json"
        program = "import sys;sys.stdout.write(open(sys.argv[1]).read())"
        source = self._source(program, str(fixture))
        conversation = source.fetch("thread-4")
        self.assertTrue(conversation.messages, "the fixture export must normalize to messages")


class AProgramThatCannotStartIsTheSourceNotABugTests(unittest.TestCase):
    """Measured before: a configured conversation command at mode 644, or a
    directory given by mistake, passed ``preflight`` (the path existed) and the
    run then ended with "internal error: this is a bug in asi" and a traceback,
    exit 70, for what is a configuration problem. ``preflight`` now asks what
    the subprocess start will ask, for both programs, and ``_run`` turns any
    failure to start into the source being unavailable (exit 3, named).
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.not_executable = root / "conversation-cli"
        self.not_executable.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        os.chmod(self.not_executable, 0o644)
        self.a_directory = root / "conversation-cli.d"
        self.a_directory.mkdir()

    @staticmethod
    def _source(discovery_program: str, export_program: str) -> ConversationCliSource:
        return ConversationCliSource(
            discovery_argv=(discovery_program, "list", "{since}", "{limit}"),
            export_argv=(export_program, "export", "{thread_id}"),
        )

    @unittest.skipIf(os.geteuid() == 0, "root may execute a mode-644 file")
    def test_a_program_that_is_not_executable_is_refused_at_preflight_with_its_mode(self):
        with self.assertRaises(SourceUnavailable) as ctx:
            self._source(str(self.not_executable), sys.executable).preflight()
        message = str(ctx.exception)
        self.assertIn(f"conversation command {str(self.not_executable)!r}", message)
        self.assertIn("exists but is not executable (mode 644); chmod +x it", message)
        self.assertIn("[source].discovery_argv", message)

    def test_a_directory_given_as_the_program_is_refused_at_preflight(self):
        with self.assertRaises(SourceUnavailable) as ctx:
            self._source(str(self.a_directory), sys.executable).preflight()
        self.assertIn("is a directory, not a program", str(ctx.exception))

    def test_the_export_program_is_checked_too_not_only_discovery(self):
        with self.assertRaises(SourceUnavailable) as ctx:
            self._source(sys.executable, str(self.a_directory)).preflight()
        message = str(ctx.exception)
        self.assertIn(f"conversation command {str(self.a_directory)!r}", message)
        self.assertIn("is a directory, not a program", message)
        with self.assertRaises(SourceUnavailable) as ctx:
            ConversationCliSource(
                discovery_argv=(sys.executable, "{since}", "{limit}"), export_argv=()
            ).preflight()
        self.assertIn("[source].export_argv is empty", str(ctx.exception))

    def test_a_bare_name_is_looked_up_on_path_and_nowhere_else(self):
        self.assertEqual(program_problem("asi-no-such-program-cycle-81"), "not found on PATH")
        self.assertEqual(program_problem(sys.executable), "")
        self.assertEqual(program_problem(str(self.a_directory.parent / "missing")), "not found")
        with self.assertRaises(SourceUnavailable) as ctx:
            self._source("asi-no-such-program-cycle-81", sys.executable).preflight()
        self.assertIn("'asi-no-such-program-cycle-81' not found on PATH", str(ctx.exception))

    @unittest.skipIf(os.geteuid() == 0, "root may execute a mode-644 file")
    def test_a_start_that_fails_after_preflight_is_the_source_unavailable_not_a_crash(self):
        # preflight skipped on purpose: the file may lose its bit between the
        # check and the run, and the result must still be a named refusal.
        source = self._source(str(self.not_executable), str(self.not_executable))
        with self.assertRaises(SourceUnavailable) as ctx:
            source.discover_page(since="1d", limit=5)
        self.assertIn(f"cannot execute {str(self.not_executable)!r}", str(ctx.exception))
        self.assertIn("Permission denied", str(ctx.exception))
        with self.assertRaises(SourceUnavailable):
            source.fetch("1700000000101-1")
        with self.assertRaises(SourceUnavailable) as ctx:
            self._source(str(self.a_directory), sys.executable).discover_page(since="1d", limit=5)
        self.assertIn("cannot execute", str(ctx.exception))


class ThreadIdIsAnArgumentNotAnOptionTests(unittest.TestCase):
    """A discovered id is untrusted text that ends up on a command line.

    Discovery output is parsed from a tool whose titles are derived from
    conversation content, so a conversation can carry a newline followed by a
    forged ``Thread:`` line. The id on that line was substituted into the export
    argv unchecked: ``--output=/somewhere`` or ``-rf`` became an OPTION to the
    conversation CLI rather than the thread it was asked for. The id is a value,
    and a value that starts with a dash or carries control characters is refused
    at the boundary, before any command runs.
    """

    class _Recorded(ConversationCliSource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.argvs: list[list[str]] = []

        def _run(self, argv, **kwargs):  # type: ignore[override]
            self.argvs.append(list(argv))
            return "{}"

    FORGED = (
        '1. "a title" (2026-01-02)\n'
        "   Thread: 1700000000101-1 (conversation://thread/1700000000101-1)\n"
        '2. "injected" (2026-01-02)\n'
        "   Thread: --output=/tmp/owned\n"
        '3. "injected too" (2026-01-02)\n'
        "   Thread: -rf\n"
        '4. "fine" (2026-01-02)\n'
        "   Thread: 1700000000102-1\n"
    )

    def test_an_option_shaped_id_is_dropped_at_parse_time(self):
        ids = [t.thread_id for t in parse_discovery_output(self.FORGED)]
        self.assertEqual(ids, ["1700000000101-1", "1700000000102-1"])

    def test_a_dropped_id_is_named_beside_the_threads_not_only_dropped(self):
        # Measured before: the forged lines were dropped and the receipt said
        # nothing, so a listing under injection read like a clean, shorter one.
        threads, passed_over = parse_discovery(self.FORGED)
        self.assertEqual([t.thread_id for t in threads], ["1700000000101-1", "1700000000102-1"])
        self.assertEqual(
            [entry for entry, _ in passed_over],
            [
                "listing entry 2 ('injected') with id '--output=/tmp/owned'",
                "listing entry 3 ('injected too') with id '-rf'",
            ],
        )
        for _, reason in passed_over:
            self.assertIn("option-shaped", reason)
            self.assertIn("nothing was run for it", reason)
        self.assertEqual(parse_discovery("")[1], [])
        # A duplicate listing is de-duplicated, which is routine, not passed over.
        threads, passed_over = parse_discovery(self.FORGED + self.FORGED)
        self.assertEqual(len(threads), 2)
        self.assertEqual(len(passed_over), 4, "the forged lines appear twice and are named twice")

    def test_the_reason_names_the_rule_the_id_failed(self):
        def reason_for(raw: str) -> str:
            listing = f'7. "t" (2026-01-02)\n   Thread: {raw}\n'
            _, passed_over = parse_discovery(listing)
            self.assertEqual(len(passed_over), 1, raw)
            self.assertTrue(passed_over[0][0].startswith("listing entry 7 ("), passed_over[0][0])
            return passed_over[0][1]

        self.assertIn("is option-shaped (it starts with a dash)", reason_for("--output=/tmp/x"))
        self.assertIn("is over-long (300 characters; the limit is 256)", reason_for("x" * 300))
        self.assertIn("carries whitespace or control characters", reason_for("a\x07b"))
        # The empty and the whitespace case cannot come through the regex, which
        # requires a non-blank token; they are refused in fetch instead.

    def test_a_forged_header_does_not_lend_its_title_to_the_next_real_thread(self):
        listing = (
            '1. "forged banner" (2026-01-02)\n'
            "   Thread: --output=/tmp/x\n"
            "   Thread: 1700000000999-1\n"
            "   Thread: 1700000000998-1\n"
        )
        threads, passed_over = parse_discovery(listing)
        self.assertEqual(
            [(t.thread_id, t.title, t.updated_date) for t in threads],
            [("1700000000999-1", "", ""), ("1700000000998-1", "", "")],
            "a header is spent by the first Thread: line under it, forged or not",
        )
        self.assertEqual(len(passed_over), 1)
        self.assertTrue(passed_over[0][0].startswith("listing entry 1 ('forged banner')"))
        # Without a header above it, the entry falls back to its position among
        # Thread: lines, marked so it is not mistaken for a printed number.
        _, orphan = parse_discovery("   Thread: -rf\n")
        self.assertTrue(orphan[0][0].startswith("listing entry #1 ('')"), orphan[0][0])

    def test_a_dropped_id_is_shown_as_a_repr_cut_short_so_it_cannot_steer_a_terminal(self):
        hostile = "a\x1b[31m\x07" + "b" * 100
        listing = f'1. "{"t" * 80}" (2026-01-02)\n   Thread: {hostile}\n'
        _, passed_over = parse_discovery(listing)
        self.assertEqual(len(passed_over), 1)
        entry = passed_over[0][0]
        self.assertNotIn("\x1b", entry, "escaped, not raw")
        self.assertIn("\\x1b[31m\\x07", entry)
        self.assertTrue(entry.endswith("...'"), "cut, and the closing quote kept: " + entry)
        self.assertIn("...') with id '", entry, "the title is cut the same way")
        self.assertLess(len(entry), 140)

    def test_the_page_from_the_real_source_carries_what_it_passed_over(self):
        forged = self.FORGED

        class ListsForged(ConversationCliSource):
            def _run(self, argv, **kwargs):  # type: ignore[override]
                return forged

        page = ListsForged().discover_page(since="1d", limit=10)
        self.assertEqual(
            [t.thread_id for t in page.threads], ["1700000000101-1", "1700000000102-1"]
        )
        self.assertEqual(len(page.passed_over), 2)
        self.assertIn("'--output=/tmp/owned'", page.passed_over[0][0])
        self.assertEqual(len(page.to_dict()["passed_over"]), 2)

    def test_fetch_refuses_an_option_shaped_id_before_running_anything(self):
        source = self._Recorded()
        for bad in ("--output=/tmp/owned", "-rf", "-", "a\nb", "a\x00b", "", " ", "x" * 300):
            with self.subTest(bad=bad):
                with self.assertRaises(RecordDefect):
                    source.fetch(bad)
        self.assertEqual(source.argvs, [], "no command may run for a refused id")

    def test_the_control_id_still_reaches_the_command_line(self):
        source = self._Recorded()
        with self.assertRaises(RecordDefect):
            # "{}" is not a valid export; the point is that the command RAN.
            source.fetch("1700000000101-1")
        self.assertEqual(len(source.argvs), 1)
        self.assertIn("1700000000101-1", source.argvs[0])

    def test_the_predicate_names_the_rule(self):
        self.assertTrue(is_safe_thread_id("thr_abc.123:4/5"))
        self.assertFalse(is_safe_thread_id("--x"))
        self.assertFalse(is_safe_thread_id("a b"))
        self.assertFalse(is_safe_thread_id("a\tb"))


class AListedEntryWithoutAReadableThreadLineIsNamedTests(unittest.TestCase):
    """Measured before: a header followed by no Thread: line, a Thread: line
    blank after the colon and one whose id began with a parenthesis each made
    the listed conversation vanish from the page with ``passed_over`` empty,
    the same silence cycle 77 removed for unsafe ids.
    """

    LISTING = (
        '1. "headless" (2026-01-02)\n'
        '2. "blank id" (2026-01-02)\n'
        "   Thread:\n"
        '3. "paren id" (2026-01-02)\n'
        "   Thread: (odd)\n"
        '4. "fine" (2026-01-02)\n'
        "   Workspace: /w\n"
        "   Thread: 1700000000102-1\n"
        '5. "last and headless" (2026-01-02)\n'
    )

    def test_each_entry_the_parser_could_not_take_is_named_under_its_number(self):
        threads, passed_over = parse_discovery(self.LISTING)
        self.assertEqual(
            [(t.thread_id, t.title, t.workspace) for t in threads],
            [("1700000000102-1", "fine", "/w")],
            "the readable entry is unchanged, title and workspace intact",
        )
        self.assertEqual(
            [entry for entry, _ in passed_over],
            [
                "listing entry 1 ('headless')",
                "listing entry 2 ('blank id') with id ''",
                "listing entry 3 ('paren id') with id '(odd)'",
                "listing entry 5 ('last and headless')",
            ],
        )
        reasons = [reason for _, reason in passed_over]
        self.assertTrue(reasons[0].startswith("has no Thread: line this parser could read"))
        self.assertTrue(reasons[1].startswith("has a Thread: line this parser could not read"))
        self.assertTrue(reasons[2].startswith("has a Thread: line this parser could not read"))
        self.assertTrue(reasons[3].startswith("has no Thread: line this parser could read"))
        for reason in reasons:
            self.assertIn("nothing was run for it", reason)

    def test_a_clean_listing_passes_nothing_over(self):
        clean = (
            '1. "a" (2026-01-02)\n'
            "   Thread: 1700000000101-1\n"
            '2. "b" (2026-01-02)\n'
            "   Workspace: /w\n"
            "   Thread: 1700000000102-1 (conversation://thread/1700000000102-1)\n"
        )
        threads, passed_over = parse_discovery(clean)
        self.assertEqual(len(threads), 2)
        self.assertEqual(passed_over, [])
        self.assertEqual(parse_discovery("")[1], [])
        self.assertEqual(parse_discovery("no listing at all\n")[1], [])

    def test_a_thread_line_with_no_header_is_still_a_thread(self):
        # Pre-existing behaviour, kept: the id is what matters; a missing header
        # costs the title, not the thread.
        threads, passed_over = parse_discovery("   Thread: 1700000000103-1\n")
        self.assertEqual([(t.thread_id, t.title) for t in threads], [("1700000000103-1", "")])
        self.assertEqual(passed_over, [])

    def test_a_workspace_line_does_not_stand_in_for_a_thread_line(self):
        # A header followed only by Workspace: is still headless; the next
        # header must find it unspent and name it.
        listing = (
            '1. "x" (2026-01-02)\n'
            "   Workspace: /w\n"
            '2. "y" (2026-01-02)\n'
            "   Thread: 1700000000102-1\n"
        )
        threads, passed_over = parse_discovery(listing)
        self.assertEqual([(t.thread_id, t.title) for t in threads], [("1700000000102-1", "y")])
        self.assertEqual([entry for entry, _ in passed_over], ["listing entry 1 ('x')"])
        self.assertTrue(passed_over[0][1].startswith("has no Thread: line this parser could read"))

    def test_an_unreadable_thread_line_still_counts_toward_the_hash_position(self):
        # With no header to number them, consecutive unreadable Thread lines are
        # numbered by position among Thread lines, so the second is #2 not #1.
        threads, passed_over = parse_discovery("   Thread: (odd)\n   Thread: -rf\n")
        self.assertEqual(threads, [])
        self.assertEqual(
            [entry for entry, _ in passed_over],
            ["listing entry #1 ('') with id '(odd)'", "listing entry #2 ('') with id '-rf'"],
        )

    def test_the_same_listing_number_in_two_workspaces_is_two_entries_not_one(self):
        # Measured before: both pages named "listing entry 1 ('x')" with one
        # reason, merge_pages kept one, and the receipt said one conversation
        # was passed over when two were.
        listing = '1. "x" (2026-01-02)\n'

        class Headless(ConversationCliSource):
            def _run(self, argv, **kwargs):  # type: ignore[override]
                return listing

        source = Headless()
        pages = [source.discover_page(since="1d", limit=5, workspace=w) for w in ("/a", "/b")]
        self.assertEqual(pages[0].passed_over[0][0], "listing entry 1 ('x') in workspace /a")
        _, summary = merge_pages(pages)
        self.assertEqual(
            [item["entry"] for item in summary["passed_over"]],
            ["listing entry 1 ('x') in workspace /a", "listing entry 1 ('x') in workspace /b"],
        )
        self.assertIn("passed over 2 source entries", passed_over_note(summary["passed_over"]))
        # No workspace queried: nothing to append, the entry reads as before.
        self.assertEqual(
            source.discover_page(since="1d", limit=5).passed_over[0][0],
            "listing entry 1 ('x')",
        )


class ExplicitLimitTests(unittest.TestCase):
    """The limit is always sent. Relying on an interface default is the bug."""

    ARGV = ("probe-cli", "search", "--after", "{since}", "-n", "{limit}")

    class _Recorded(ConversationCliSource):
        """Records the argv instead of running it. No subprocess in this test."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.argvs: list[list[str]] = []

        def _run(self, argv, **kwargs):  # type: ignore[override]
            self.argvs.append(list(argv))
            return ""

    def test_the_source_always_passes_its_limit_explicitly(self):
        source = self._Recorded(discovery_argv=self.ARGV)
        source.discover_page(since="1d", limit=40)
        self.assertEqual(len(source.argvs), 1, "the source must actually shell out")
        argv = source.argvs[0]
        self.assertIn(
            "-n",
            argv,
            "an omitted limit means the interface's own small default applies, and "
            "the harvest silently becomes a recency sliver",
        )
        self.assertEqual(argv[argv.index("-n") + 1], "40")

    def test_a_limit_of_one_is_still_sent_rather_than_treated_as_falsey(self):
        source = self._Recorded(discovery_argv=self.ARGV)
        source.discover_page(since="1d", limit=1)
        argv = source.argvs[0]
        self.assertEqual(argv[argv.index("-n") + 1], "1")

    def test_the_window_and_workspace_reach_the_command_line(self):
        source = self._Recorded(discovery_argv=self.ARGV)
        source.discover_page(since="7d", limit=5, workspace="/w", query="gate")
        argv = source.argvs[0]
        self.assertEqual(argv[:2], ["probe-cli", "search"])
        self.assertIn("gate", argv)
        self.assertEqual(argv[argv.index("--after") + 1], "7d")
        self.assertEqual(argv[argv.index("-w") + 1], "/w")

    def test_an_empty_result_is_a_total_not_a_truncation(self):
        source = self._Recorded(discovery_argv=self.ARGV)
        page = source.discover_page(since="1d", limit=40)
        self.assertEqual(page.returned, 0)
        self.assertTrue(page.is_total)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
