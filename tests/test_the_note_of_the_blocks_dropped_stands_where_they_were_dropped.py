"""The note of the blocks dropped stands where they were dropped.

``render_transcript`` in ``app/prompts.py`` keeps the first ``(m + 1) // 2`` and
the last ``m // 2`` of the blocks when ``m`` fit its budget, and says how many it
dropped in a note among the blocks kept. Measured before (3.11 and 3.13): the
note went in after the first ``m // 2``, so for every odd ``m`` it stood a block
early and the block after it, the last of the opening ones, read as the first
past the gap: 109 of 218 renders of 20 messages over budgets of 900 to 8,999
characters, every one that kept an odd number.
"""

from __future__ import annotations

import re
import unittest
from typing import Any

from agent_skill_improver.adapters.transcript_export import normalize_export
from agent_skill_improver.app.prompts import ELISION_MARKER, render_transcript
from agent_skill_improver.config import PromptConfig

N = 20


def conversation(n: int = N, size: int = 400) -> Any:
    els: list = []
    for i in range(n):
        els += [
            {"id": f"m{i}", "position": 2 * i, "kind": "message", "role": "user"},
            {"id": f"x{i}", "position": 2 * i + 1, "kind": "text", "parentId": f"m{i}"}
            | {"content": "w" * size},
        ]
    return normalize_export({"thread": {"id": "thr"}, "elements": els})


def around_the_note(text: str, elided: int) -> tuple[list[int], list[int]]:
    """The message numbers before and after the note, which appears exactly once."""
    note = ELISION_MARKER.format(n=elided)
    before, after = text.split(note)
    head = r"\[message element_id=m(\d+) "
    return [int(x) for x in re.findall(head, before)], [int(x) for x in re.findall(head, after)]


class TheNoteOfTheBlocksDroppedStandsWhereTheyWereDroppedTests(unittest.TestCase):
    def test_the_note_stands_between_the_opening_blocks_kept_and_the_closing_ones(self):
        conv = conversation()
        kept_counts = set()
        for budget in range(900, 9000, 101):
            render = render_transcript(conv, cfg=PromptConfig(max_transcript_chars=budget))
            if not render.blocks_elided:
                continue
            with self.subTest(budget=budget):
                first, last = around_the_note(render.text, render.blocks_elided)
                kept = len(first) + len(last)
                kept_counts.add(kept)
                self.assertEqual(kept, N - render.blocks_elided)
                self.assertEqual(first, list(range((kept + 1) // 2)))
                self.assertEqual(last, list(range(N - kept // 2, N)))
        self.assertTrue({3, 4, 5} <= kept_counts, kept_counts)

    def test_the_note_stands_at_the_gap_in_an_excerpt_too(self):
        only = [2 * i for i in (0, 2, 3, 5, 8, 11, 13, 17, 19)]
        render = render_transcript(
            conversation(), cfg=PromptConfig(max_transcript_chars=1500), only_positions=only
        )
        self.assertEqual((render.blocks_total, render.blocks_elided), (9, 6))
        self.assertEqual(around_the_note(render.text, 6), ([0, 2], [19]))

    def test_the_control_two_blocks_kept_have_the_note_between_them(self):
        render = render_transcript(conversation(), cfg=PromptConfig(max_transcript_chars=900))
        self.assertEqual(render.blocks_elided, N - 2)
        self.assertEqual(around_the_note(render.text, N - 2), ([0], [N - 1]))
        whole = render_transcript(conversation(3), cfg=PromptConfig())
        self.assertEqual(whole.blocks_elided, 0)
        self.assertNotIn("elided by the local prompt budget", whole.text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
