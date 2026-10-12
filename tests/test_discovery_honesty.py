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
    parse_discovery_output,
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
