"""The normalizer, against export-shaped fixtures and against broken input.

The fixtures are synthetic but their SHAPE is the documented export
output. Each assertion below corresponds to a quirk that a plausible guess gets
wrong, which is the reason these tests exist at all.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent_skill_improver.adapters.transcript_export import (
    normalize_export,
    normalize_export_text,
)
from agent_skill_improver.adapters.conversation_cli import parse_discovery_output
from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.domain.conversation import (
    NormalizedConversation,
    EvidenceRef,
    TOOL_STATUS_ERROR,
    TOOL_STATUS_RUNNING,
)
from agent_skill_improver.errors import RecordDefect

from tests import support


def load(name: str) -> NormalizedConversation:
    payload = json.loads((support.FIXTURE_EXPORTS / name).read_text(encoding="utf-8"))
    return normalize_export(payload)


class NormalizerShapeTests(unittest.TestCase):
    def test_message_text_comes_from_child_text_elements(self) -> None:
        """A `message` element carries NO content; the words are in its children."""
        conversation = load("compliant.json")
        self.assertEqual(conversation.thread_id, "1700000000001-1")
        first = conversation.messages[0]
        self.assertEqual(first.role, "user")
        self.assertIn("Add a module docstring", first.text)
        self.assertTrue(first.text_element_ids, "text element ids must be retained for citation")

    def test_ids_and_positions_are_both_retained(self) -> None:
        conversation = load("compliant.json")
        for message in conversation.messages:
            self.assertEqual(conversation.element_index[message.element_id], message.position)
        # Every element, not just messages, stays citable.
        self.assertEqual(len(conversation.element_index), conversation.total_elements)

    def test_thinking_is_excluded_from_scored_text_but_stays_citable(self) -> None:
        conversation = load("compliant.json")
        joined = "\n".join(m.text for m in conversation.messages)
        self.assertNotIn("Read the file before editing it", joined)
        kinds = set(conversation.element_kinds.values())
        self.assertIn("thinking", kinds)

    def test_tool_events_retain_name_status_and_output(self) -> None:
        conversation = load("failing.json")
        names = [t.name for t in conversation.tool_events]
        self.assertIn("Bash", names)
        errors = conversation.failed_tool_events
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(t.status == TOOL_STATUS_ERROR for t in errors))
        self.assertTrue(all(t.failed for t in errors))
        self.assertTrue(any(t.output_bytes > 0 for t in conversation.tool_events))

    def test_running_tool_and_bare_string_input_do_not_break_parsing(self) -> None:
        """Both are real quirks of live output."""
        conversation = load("edge-shapes.json")
        tool = conversation.tool_events[0]
        self.assertEqual(tool.status, TOOL_STATUS_RUNNING)
        self.assertEqual(tool.input_text, "tail -f build.log")

        failing = load("failing.json")
        string_input = [t for t in failing.tool_events if t.name == "view"]
        self.assertEqual(string_input[0].input_text, "skills/tdd/SKILL.md")

    def test_parentless_non_message_elements_are_not_treated_as_messages(self) -> None:
        conversation = load("failing.json")
        self.assertTrue(all(m.role in ("user", "assistant") for m in conversation.messages))
        self.assertIn("context", set(conversation.element_kinds.values()))

    def test_max_messages_truncates_and_says_so(self) -> None:
        payload = json.loads((support.FIXTURE_EXPORTS / "failing.json").read_text(encoding="utf-8"))
        conversation = normalize_export(payload, max_messages=2)
        self.assertEqual(len(conversation.messages), 2)
        self.assertTrue(conversation.truncated)
        self.assertFalse(load("failing.json").truncated)

    def test_round_trip_through_dict_is_lossless_for_scored_fields(self) -> None:
        original = load("failing.json")
        restored = NormalizedConversation.from_dict(original.to_dict())
        self.assertEqual(restored.to_dict(), original.to_dict())

    def test_from_dict_rejects_an_unknown_schema(self) -> None:
        payload = load("compliant.json").to_dict()
        payload["schema"] = "asi.conversation.v99"
        with self.assertRaises(ValueError):
            NormalizedConversation.from_dict(payload)


class EvidenceResolutionTests(unittest.TestCase):
    """Both the id AND the position must agree. This is the anti-fabrication rule."""

    def setUp(self) -> None:
        self.conversation = load("compliant.json")
        self.message = self.conversation.messages[0]

    def test_correct_reference_resolves(self) -> None:
        ref = EvidenceRef(self.message.element_id, self.message.position, "message")
        self.assertTrue(self.conversation.resolves(ref))

    def test_right_id_wrong_position_does_not_resolve(self) -> None:
        ref = EvidenceRef(self.message.element_id, self.message.position + 1, "message")
        self.assertFalse(self.conversation.resolves(ref))

    def test_invented_id_does_not_resolve(self) -> None:
        self.assertFalse(self.conversation.resolves(EvidenceRef("made-up-id", 0)))

    def test_wrong_kind_does_not_resolve(self) -> None:
        ref = EvidenceRef(self.message.element_id, self.message.position, "tool")
        self.assertFalse(self.conversation.resolves(ref))

    def test_unresolved_returns_only_the_bad_refs(self) -> None:
        good = EvidenceRef(self.message.element_id, self.message.position)
        bad = EvidenceRef("nope", 5)
        self.assertEqual(self.conversation.unresolved([good, bad]), [bad])

    def test_evidence_ref_parsing_rejects_malformed_input(self) -> None:
        for payload in (
            {},
            {"element_id": ""},
            {"element_id": "x"},
            {"element_id": "x", "position": "3"},
            "string",
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    EvidenceRef.from_dict(payload)


class MalformedExportTests(unittest.TestCase):
    """A broken export is a RecordDefect: quarantine that record, keep the batch."""

    def test_each_malformed_fixture_raises_record_defect(self) -> None:
        seen = 0
        for path in sorted(support.FIXTURE_MALFORMED.glob("*.json")):
            with self.subTest(fixture=path.name):
                text = path.read_text(encoding="utf-8")
                with self.assertRaises(RecordDefect):
                    normalize_export_text(text)
                seen += 1
        self.assertEqual(seen, 5, "all five malformed shapes must be covered")

    def test_non_mapping_payload_is_refused(self) -> None:
        for payload in ([], "text", 7, None):
            with self.subTest(payload=payload):
                with self.assertRaises(RecordDefect):
                    normalize_export(payload)

    def test_elements_without_ids_are_skipped_not_fatal(self) -> None:
        payload = {
            "thread": {"id": "1700000000099-1", "title": "t"},
            "elements": [
                {"kind": "message", "role": "user", "position": 0},
                {"id": "1700000000099-2", "kind": "message", "role": "user", "position": 1},
            ],
        }
        conversation = normalize_export(payload)
        self.assertEqual(conversation.total_elements, 1)


class FixtureSourceTests(unittest.TestCase):
    def test_discovery_reports_every_fixture_thread(self) -> None:
        source = FixtureConversationSource(support.FIXTURE_EXPORTS)
        source.preflight()
        found = source.discover(since="7d", limit=50)
        self.assertEqual(len(found), 5)
        self.assertEqual(len({t.thread_id for t in found}), 5)

    def test_discovery_can_filter_by_workspace(self) -> None:
        source = FixtureConversationSource(support.FIXTURE_EXPORTS)
        found = source.discover(since="7d", limit=50, workspace="/tmp/asi-fixture-workspace-d")
        self.assertEqual({t.thread_id for t in found}, {"1700000000004-1", "1700000000005-1"})

    def test_fetching_an_unknown_thread_is_a_record_defect(self) -> None:
        source = FixtureConversationSource(support.FIXTURE_EXPORTS)
        with self.assertRaises(RecordDefect):
            source.fetch("no-such-thread")

    def test_unreadable_directory_is_transient_not_a_defect(self) -> None:
        source = FixtureConversationSource(Path("/nonexistent/fixtures"))
        with self.assertRaises(Exception) as ctx:
            source.preflight()
        self.assertEqual(type(ctx.exception).__name__, "SourceUnavailable")


class DiscoveryParserTests(unittest.TestCase):
    """Parses the documented discovery line grammar.

    The sample exercises every shape the grammar has to survive: a numbered
    title line, an indented Workspace line that applies to the header above it,
    an indented Thread line whose id is followed by a URL suffix the parser must
    ignore, and a record with no Workspace line at all.
    """

    SAMPLE = """Found 3 conversation(s)

1. "Refactor the ingest path" (2026-08-29)
   Workspace: /tmp/one
   Thread: 1700000000101-1 (conversation://thread/1700000000101-1)

2. "Fix the flaky test" (2026-08-28)
   Workspace: /tmp/two
   Thread: 1700000000102-1 (conversation://thread/1700000000102-1)

3. "Untitled" (2026-08-27)
   Thread: 1700000000103-1 (conversation://thread/1700000000103-1)
"""

    def test_parses_id_title_workspace_and_date(self) -> None:
        found = parse_discovery_output(self.SAMPLE)
        self.assertEqual(len(found), 3)
        self.assertEqual(found[0].thread_id, "1700000000101-1")
        self.assertEqual(found[0].title, "Refactor the ingest path")
        self.assertEqual(found[0].workspace, "/tmp/one")
        self.assertEqual(found[0].updated_date, "2026-08-29")

    def test_a_thread_without_a_workspace_line_still_parses(self) -> None:
        found = parse_discovery_output(self.SAMPLE)
        self.assertEqual(found[2].workspace, "")
        self.assertEqual(found[2].updated_date, "2026-08-27")

    def test_duplicate_thread_ids_are_collapsed(self) -> None:
        found = parse_discovery_output(self.SAMPLE + self.SAMPLE)
        self.assertEqual(len(found), 3)

    def test_unrecognised_output_yields_nothing_rather_than_guessing(self) -> None:
        self.assertEqual(parse_discovery_output("some other format entirely"), [])
        self.assertEqual(parse_discovery_output(""), [])


if __name__ == "__main__":
    unittest.main()
