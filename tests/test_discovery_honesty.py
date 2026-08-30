"""A page of results is not a total, and the difference decides every rate.

The concrete hazard: a conversation search has a small default page size. A
harvester that accepts the default reads the most recently touched sliver of the
corpus, which in an orchestrated environment is disproportionately machine
traffic. Worse, nothing errors, so every downstream percentage is computed over a
population nobody chose.
"""

from __future__ import annotations

import unittest

from agent_skill_improver.adapters.conversation_cli import ConversationCliSource
from agent_skill_improver.domain.discovery import (
    DiscoveredThread,
    DiscoveryPage,
    error_page,
    merge_pages,
    page_from_threads,
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


class ExplicitLimitTests(unittest.TestCase):
    """The limit is always sent. Relying on an interface default is the bug."""

    ARGV = ("probe-cli", "search", "--after", "{since}", "-n", "{limit}")

    class _Recorded(ConversationCliSource):
        """Records the argv instead of running it. No subprocess in this test."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.argvs: list[list[str]] = []

        def _run(self, argv):  # type: ignore[override]
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
