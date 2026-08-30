"""Who wrote a conversation, and how a bounded run stays representative.

Two requirements meet here. Machine-generated threads can be the overwhelming
majority of a corpus when an orchestrator runs continuously, and a bounded run
that samples badly will grade the orchestrator instead of the skills a person
uses. So the classifier has to be honest about what it cannot know, and the
selector has to be fair rather than merely bounded.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from agent_skill_improver.domain.authorship import (
    CLASS_HUMAN,
    CLASS_NOT_AGENT_TITLED,
    CLASS_UNKNOWN,
    CLASS_ORCHESTRATED,
    CLASSES,
    DISCOVERY_CLASSES,
    class_histogram,
    classify_conversation,
    classify_thread,
    classify_title,
)
from agent_skill_improver.domain.discovery import DiscoveredThread
from agent_skill_improver.domain.selection import (
    UNCLASSIFIED_BUCKET,
    bucket_coverage,
    rank_by,
    select_fair,
)


@dataclass
class FakeMessage:
    text: str = ""
    origin: str = "user"


@dataclass
class FakeConversation:
    title: str = ""
    agent_name: str = ""
    user_messages: tuple[FakeMessage, ...] = ()


class DiscoveryClassVocabularyTests(unittest.TestCase):
    """A title cannot prove human authorship, and the labels must not pretend."""

    def test_the_pre_fetch_vocabulary_does_not_contain_human(self):
        self.assertIn(CLASS_ORCHESTRATED, DISCOVERY_CLASSES)
        self.assertIn(CLASS_NOT_AGENT_TITLED, DISCOVERY_CLASSES)
        self.assertNotIn(
            CLASS_HUMAN,
            DISCOVERY_CLASSES,
            "a thread whose title lacks an orchestrator marker is 'not agent titled', "
            "which is a statement about the title. Calling it 'human' would be a "
            "claim about authorship that discovery metadata cannot support.",
        )

    def test_the_post_fetch_vocabulary_keeps_unknown_separate_from_human(self):
        self.assertEqual(set(CLASSES), {CLASS_HUMAN, CLASS_ORCHESTRATED, CLASS_UNKNOWN})

    def test_an_unmarked_title_is_not_agent_titled(self):
        self.assertEqual(classify_title("Debug the retry storm"), CLASS_NOT_AGENT_TITLED)

    def test_a_marked_title_is_orchestrated(self):
        self.assertEqual(classify_title("Orchestrator Run :: iteration 41"), CLASS_ORCHESTRATED)

    def test_markers_are_configurable_because_one_orchestrator_is_not_all_of_them(self):
        self.assertEqual(
            classify_title("nightly harness run", agent_title_markers=["nightly harness"]),
            CLASS_ORCHESTRATED,
        )
        self.assertEqual(
            classify_title("nightly harness run", agent_title_markers=["some other tool"]),
            CLASS_NOT_AGENT_TITLED,
        )

    def test_marker_matching_respects_word_boundaries(self):
        # 'dispatcher' must not fire on a word that merely contains it.
        self.assertEqual(
            classify_title("redispatchered log", agent_title_markers=["dispatcher"]),
            CLASS_NOT_AGENT_TITLED,
        )
        self.assertEqual(
            classify_title("the dispatcher stalled", agent_title_markers=["dispatcher"]),
            CLASS_ORCHESTRATED,
        )

    def test_a_discovery_record_classifies_from_its_title(self):
        thread = DiscoveredThread(thread_id="t1", title="Orchestrator Run 41")
        self.assertEqual(classify_thread(thread), CLASS_ORCHESTRATED)


class ConversationClassTests(unittest.TestCase):
    def test_a_plain_human_thread_is_human_with_a_stated_reason(self):
        conv = FakeConversation(
            title="why is the gate refusing",
            user_messages=(FakeMessage(text="the gate refuses and I cannot tell why"),),
        )
        klass, reasons = classify_conversation(conv)
        self.assertEqual(klass, CLASS_HUMAN)
        self.assertTrue(reasons, "a classification with no stated reason is not reviewable")

    def test_a_role_prompt_opening_is_orchestrated_even_with_a_plain_title(self):
        conv = FakeConversation(
            title="fix the failing test",
            user_messages=(FakeMessage(text="You are the engineer. Implement story 3."),),
        )
        klass, reasons = classify_conversation(conv)
        self.assertEqual(klass, CLASS_ORCHESTRATED)
        self.assertTrue(any("role prompt" in r for r in reasons))

    def test_a_non_human_origin_marker_is_enough_on_its_own(self):
        conv = FakeConversation(
            title="ordinary title",
            user_messages=(FakeMessage(text="do the thing", origin="orchestrator"),),
        )
        klass, _ = classify_conversation(conv)
        self.assertEqual(klass, CLASS_ORCHESTRATED)

    def test_no_user_turn_is_unknown_and_never_folded_into_human(self):
        conv = FakeConversation(title="empty", user_messages=())
        klass, reasons = classify_conversation(conv)
        self.assertEqual(klass, CLASS_UNKNOWN)
        self.assertTrue(any("cannot be decided" in r for r in reasons))


class HistogramTests(unittest.TestCase):
    def test_zero_counts_are_present_because_an_absent_key_is_ambiguous(self):
        hist = class_histogram([CLASS_ORCHESTRATED, CLASS_ORCHESTRATED])
        self.assertEqual(hist[CLASS_HUMAN], 0)
        self.assertEqual(hist[CLASS_ORCHESTRATED], 2)
        self.assertEqual(
            hist[CLASS_UNKNOWN],
            0,
            "'human: 0' is a finding; a missing key is indistinguishable from a "
            "category nobody thought to count",
        )

    def test_the_declared_vocabulary_is_choosable_so_a_pre_fetch_histogram_is_honest(self):
        hist = class_histogram(
            [CLASS_NOT_AGENT_TITLED, CLASS_ORCHESTRATED], declared=DISCOVERY_CLASSES
        )
        self.assertEqual(hist, {CLASS_NOT_AGENT_TITLED: 1, CLASS_ORCHESTRATED: 1})
        self.assertNotIn(CLASS_HUMAN, hist)

    def test_an_undeclared_label_is_counted_under_its_own_name_not_re_bucketed(self):
        hist = class_histogram(["something-else"], declared=DISCOVERY_CLASSES)
        self.assertEqual(hist["something-else"], 1)


class FairSelectionTests(unittest.TestCase):
    """The anti-pattern under test is `sorted(items)[:limit]`.

    Whenever the sort key correlates with the property being selected, a prefix is
    a permanent filter and not a sample: the excluded categories are structurally
    unreachable, and every downstream artifact reads as 'under-represented'
    instead of 'never looked at'.
    """

    def _corpus(self):
        # Names are prefixed by class on purpose. Sorting by name therefore sorts
        # by class, which is exactly the correlation that makes a prefix lethal.
        agent = [(f"agent-{i:03d}", CLASS_ORCHESTRATED) for i in range(96)]
        human = [(f"human-{i:03d}", CLASS_HUMAN) for i in range(4)]
        return agent + human

    def test_a_sorted_prefix_would_exclude_the_minority_class_entirely(self):
        corpus = self._corpus()
        prefix = sorted(corpus)[:8]
        self.assertEqual(
            {klass for _, klass in prefix},
            {CLASS_ORCHESTRATED},
            "this is the failure being designed against, asserted so the test "
            "documents why round-robin exists",
        )

    def test_round_robin_reaches_every_class_within_the_same_budget(self):
        corpus = self._corpus()
        selected, report = select_fair(
            corpus, limit=8, bucket_of=lambda item: item[1], rank_of=rank_by(lambda i: i[0])
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual({klass for _, klass in selected}, {CLASS_ORCHESTRATED, CLASS_HUMAN})
        self.assertEqual(bucket_coverage(report), 1.0)

    def test_a_thin_bucket_contributes_all_it_has_and_no_budget_is_wasted(self):
        corpus = self._corpus()
        selected, report = select_fair(
            corpus, limit=20, bucket_of=lambda item: item[1], rank_of=rank_by(lambda i: i[0])
        )
        self.assertEqual(len(selected), 20, "the budget is spent, not reserved and dropped")
        self.assertEqual(report.selected_per_bucket[CLASS_HUMAN], 4)
        self.assertEqual(report.selected_per_bucket[CLASS_ORCHESTRATED], 16)
        self.assertIn(CLASS_HUMAN, report.exhausted_buckets)

    def test_an_unclassifiable_item_gets_its_own_bucket_rather_than_being_dropped(self):
        corpus = [("a", CLASS_HUMAN), ("b", ""), ("c", CLASS_ORCHESTRATED)]
        selected, report = select_fair(corpus, limit=3, bucket_of=lambda item: item[1])
        self.assertEqual(len(selected), 3)
        self.assertIn(UNCLASSIFIED_BUCKET, report.buckets)
        self.assertEqual(
            report.selected_per_bucket[UNCLASSIFIED_BUCKET],
            1,
            "an unreadable record is exactly the one a human needs to see, so it "
            "must not be filtered out before the gate",
        )

    def test_selection_is_deterministic_for_a_fixed_ranking(self):
        corpus = self._corpus()
        first, _ = select_fair(
            corpus, limit=9, bucket_of=lambda i: i[1], rank_of=rank_by(lambda i: i[0])
        )
        second, _ = select_fair(
            list(reversed(corpus)),
            limit=9,
            bucket_of=lambda i: i[1],
            rank_of=rank_by(lambda i: i[0]),
        )
        self.assertEqual(first, second, "input order must not change the outcome")

    def test_best_first_inside_a_bucket_is_honoured(self):
        corpus = [("z", "b1"), ("a", "b1"), ("m", "b1")]
        selected, _ = select_fair(
            corpus, limit=2, bucket_of=lambda i: i[1], rank_of=rank_by(lambda i: i[0])
        )
        self.assertEqual([name for name, _ in selected], ["a", "m"])

    def test_a_zero_limit_selects_nothing_and_says_it_truncated(self):
        selected, report = select_fair([("a", "x")], limit=0, bucket_of=lambda i: i[1])
        self.assertEqual(selected, [])
        self.assertTrue(report.truncated)

    def test_bucket_coverage_reports_a_narrow_run_as_narrow(self):
        corpus = [("a", "one"), ("b", "two"), ("c", "three")]
        _, report = select_fair(corpus, limit=1, bucket_of=lambda i: i[1])
        self.assertAlmostEqual(bucket_coverage(report), 1 / 3)

    def test_coverage_of_an_empty_corpus_is_not_a_division_by_zero(self):
        _, report = select_fair([], limit=5, bucket_of=lambda i: "x")
        self.assertEqual(bucket_coverage(report), 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
