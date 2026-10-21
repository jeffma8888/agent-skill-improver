"""A lone surrogate in untrusted JSON reads as U+FFFD, never an aborted run.

JSON can spell one half of a UTF-16 surrogate pair on its own (``"\\ud83d"``),
which is what a tool that cuts an emoji between its halves writes, and
``json.loads`` decodes it to a string no UTF-8 writer can encode. Measured
before (3.11 and 3.13): one conversation export holding one such escape made the
committed daily run fall with ``UnicodeEncodeError`` from the conversation
write, with no receipt; a scorer rationale holding one fell the same way from
the scorecard write, and an improver's replacement text from the patch hash.
"""

from __future__ import annotations

import copy
import json
import shutil
import unittest

from agent_skill_improver.adapters.model_command import extract_json_object
from agent_skill_improver.jsonio import REPLACEMENT_CHARACTER, dumps, loads_untrusted

from tests.support import FIXTURE_EXPORTS, InProcessRunner
from tests.test_daily_isolation import DailyHarness

CUT = "\\ud83d"  # the escape, as JSON text spells it
LONE = "\ud83d"  # what json.loads makes of it


class DecodingTests(unittest.TestCase):
    def test_a_lone_surrogate_in_a_string_or_a_key_reads_as_the_replacement_character(self):
        doc = loads_untrusted('{"a": "x\\ud83d", "\\udc00k": ["\\ud800", {"b": ["\\udfff"]}]}')
        self.assertEqual(doc, {"a": "x\ufffd", "\ufffdk": ["\ufffd", {"b": ["\ufffd"]}]})
        dumps(doc).encode("utf-8")

    def test_a_top_level_string_reads_the_same_way(self):
        self.assertEqual(loads_untrusted('"\\udbff"'), REPLACEMENT_CHARACTER)

    def test_a_lone_surrogate_already_in_the_text_reads_the_same_way(self):
        # Every production reader decodes bytes strictly or with errors="replace",
        # so this is a caller handing over a str that already holds one.
        self.assertEqual(loads_untrusted('{"a": "x' + LONE + '"}'), {"a": "x\ufffd"})

    def test_a_surrogate_deep_in_nested_arrays_is_still_replaced(self):
        depth = 500
        doc = loads_untrusted("[" * depth + '"\\ud800"' + "]" * depth)
        for _ in range(depth):
            doc = doc[0]
        self.assertEqual(doc, REPLACEMENT_CHARACTER)

    def test_an_escape_spelled_in_uppercase_hex_reads_the_same_way(self):
        """Some JSON writers spell escapes in uppercase hex; the cut half must be
        found in that spelling too, in a value and in a key."""
        self.assertEqual(
            loads_untrusted('{"a": "x\\uD83D", "\\uDC00k": 1}'), {"a": "x\ufffd", "\ufffdk": 1}
        )
        self.assertEqual(loads_untrusted('"\\uD83D\\uDE00"'), "\U0001f600")

    def test_the_control_a_well_formed_pair_is_the_character_it_spells(self):
        self.assertEqual(loads_untrusted('"\\ud83d\\ude00"'), "\U0001f600")

    def test_the_control_an_escaped_backslash_is_text_not_a_surrogate(self):
        self.assertEqual(loads_untrusted('"\\\\ud800"'), "\\ud800")

    def test_the_control_a_document_without_surrogates_is_unchanged(self):
        text = json.dumps({"a": ["\u00e9", {"b": "\u2603"}], "n": 1, "k\u00e9": None})
        self.assertEqual(loads_untrusted(text), json.loads(text))


class ThroughTheReplyDecoder(InProcessRunner):
    """The in-process model, with its reply sent through the production decoder."""

    def __init__(self, role: str, spoil) -> None:
        super().__init__()
        self.target = role
        self.spoil = spoil

    def run(self, *, role, system, user, response_schema):
        payload = super().run(role=role, system=system, user=user, response_schema=response_schema)
        if role == self.target:
            payload = copy.deepcopy(payload)
            self.spoil(payload)
        return extract_json_object(json.dumps(payload), role=role)


class OneLoneSurrogateDoesNotAbortTheRunTests(DailyHarness):
    def committed(self, config, runner=None):
        receipt, state, _store = self.run_pass(config, runner=runner)
        payload = receipt.to_dict()
        self.assertEqual(payload["status"], "ok", payload["status"])
        self.assertEqual(payload["counts"]["records_scored"], 5, "every fixture is scored")
        self.assertEqual(payload["counts"]["records_quarantined"], 0)
        return payload, state

    @staticmethod
    def texts(state, name):
        return [p.read_text(encoding="utf-8") for p in sorted(state.dir(name).glob("*.json"))]

    def test_an_export_with_a_cut_emoji_is_scored_with_the_rest(self):
        exports = self.base / "exports"
        shutil.copytree(FIXTURE_EXPORTS, exports)
        spoiled = exports / "compliant.json"
        doc = json.loads(spoiled.read_text(encoding="utf-8"))
        element = next(
            e for e in doc["elements"] if isinstance(e.get("content"), str) and e["content"]
        )
        element["content"] += " cut emoji " + LONE
        spoiled.write_text(json.dumps(doc), encoding="utf-8")
        self.assertIn(CUT, spoiled.read_text(encoding="utf-8"), "the export spells the escape")
        _payload, state = self.committed(self.env(fixtures_dir=exports))
        self.assertTrue(
            any("cut emoji \ufffd" in t for t in self.texts(state, "conversations")),
            "the conversation is kept, with the mark where text was lost",
        )

    def test_a_scorer_rationale_with_a_cut_emoji_is_recorded(self):
        def spoil(payload):
            for item in payload.get("criteria", []):
                item["rationale"] = "cut " + LONE

        _payload, state = self.committed(self.env(), runner=ThroughTheReplyDecoder("scorer", spoil))
        self.assertTrue(any("cut \ufffd" in t for t in self.texts(state, "scorecards")))

    def test_an_improver_text_with_a_cut_emoji_becomes_a_reviewable_candidate(self):
        def spoil(payload):
            payload["new_skill_text"] += "\n- cut " + LONE + "\n"

        payload, state = self.committed(
            self.env(), runner=ThroughTheReplyDecoder("improver", spoil)
        )
        self.assertEqual(payload["counts"]["candidates_proposed"], 1)
        self.assertTrue(any("cut \ufffd" in t for t in self.texts(state, "candidates")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
