"""The transcript a role reads is held to its budget.

``render_transcript`` in ``app/prompts.py`` renders a conversation for the
scorer, and each excerpt a failure cites for the improver, inside
``max_transcript_chars``. Measured before (3.11 and 3.13), with the default
budget of 24,000: one reply with 500 tool calls put 1,460,401 characters into
the transcript and two messages with 20 tool calls 66,515, since dropping the
middle blocks stopped at two however large those two were; a tool's name or
status or a message's role of 2 MiB put 2.1 million into it, and a message id of
2 MiB 4.2 million into the scorer's prompt, once in the transcript and once in
the list of elements it may cite. Masking ran after the budget, and twelve
messages of assignments came to 54,194 characters once masked, 2.26 times it. A
key block opened in one message and closed in a later one masked every message
between. And dropping the middle summed what was left after every drop: 40,000
short messages took 15 s to render, four times as long per doubling.
"""

from __future__ import annotations

import random
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_skill_improver.adapters.transcript_export import normalize_export
from agent_skill_improver.app.prompts import (
    ELISION_MARKER,
    MAX_HEAD_FIELD_CHARS,
    _kept_indices,
    improver_user_prompt,
    render_transcript,
    scorer_user_prompt,
)
from agent_skill_improver.config import PromptConfig, load_config
from agent_skill_improver.domain.conversation import EvidenceRef
from agent_skill_improver.domain.failure import FailureCase

from tests.support import build_env
from tests.test_skill_identity_and_payloads import deadline

BUDGET = PromptConfig().max_transcript_chars
#: Room for what the budget does not count: a clip's own note on each of the two
#: blocks that may share it, and the note saying how many blocks were dropped.
MARKERS = 200
BIG = 2 * 1024 * 1024
CLIP_NOTE_AT_END = re.compile(r" \.\.\.\[clipped \d+ chars\]\Z")
KEY_OPEN = "-----BEGIN " + "PRIVATE KEY-----"
KEY_CLOSE = "-----END " + "PRIVATE KEY-----"
#: The letter each planted field is made of, and what its head line shows next.
LETTERS = {"message id": "I", "role": "R", "tool id": "T", "tool name": "N", "tool status": "S"}
AFTER = {
    "message id": " position=",
    "role": "]",
    "tool id": " position=",
    "tool name": " status=",
    "tool status": "]",
}


def message(i: int, position: int, *, role: str = "user", text: str = "hi", mid: str = "") -> list:
    mid = mid or f"m{i}"
    return [
        {"id": mid, "position": position, "kind": "message", "role": role},
        {"id": f"x{i}", "position": position + 1, "kind": "text", "parentId": mid, "content": text},
    ]


def tool(i: int, position: int, parent: str, **fields: Any) -> dict:
    raw = {"id": f"t{i}", "position": position, "kind": "tool", "parentId": parent}
    raw.update({"name": "Bash", "status": "done", "input": "a" * 900, "output": "b" * 2100})
    raw.update(fields)
    return raw


def conversation(elements: list) -> Any:
    return normalize_export({"thread": {"id": "thr"}, "elements": elements})


def busy_reply(calls: int, *, text: str = "hi") -> list:
    elements = message(1, 0, text=text) + message(2, 2, role="assistant")
    return elements + [tool(i, 10 + i, "m2") for i in range(calls)]


def old_kept(sizes: list[int], budget: int) -> list[int]:
    """What dropping the middle block until the rest fit kept, as indices."""
    kept = list(range(len(sizes)))
    while kept and sum(sizes[i] for i in kept) > budget and len(kept) > 2:
        kept.pop(len(kept) // 2)
    return kept


class TheTranscriptARoleReadsIsHeldToItsBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.config = load_config(build_env(Path(cls._tmp.name), dry_run=True))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def scorer(self, elements: list) -> str:
        text, _ = scorer_user_prompt(
            conversation(elements), self.config.rubrics[0], cfg=PromptConfig(), seed="c193"
        )
        return text

    def test_a_reply_with_many_tool_calls_is_held_to_the_budget(self):
        render = render_transcript(conversation(busy_reply(500)), cfg=PromptConfig())
        self.assertLessEqual(render.chars, BUDGET + MARKERS)
        self.assertIn("[message element_id=m1 position=0 role=user]\nhi", render.text)
        self.assertIn("[message element_id=m2 position=2 role=assistant]", render.text)
        self.assertRegex(render.text, r" \.\.\.\[clipped \d+ chars\]$")

    def test_two_blocks_over_the_budget_share_it_the_smaller_kept_whole(self):
        render = render_transcript(conversation(busy_reply(20)), cfg=PromptConfig())
        self.assertLessEqual(render.chars, BUDGET + MARKERS)
        self.assertTrue(
            render.text.startswith("[message element_id=m1 position=0 role=user]\nhi\n\n")
        )
        self.assertGreater(render.chars, BUDGET - MARKERS, "the larger takes all that is left")
        # Exactly: the budget counts each block's blank line, so what is shown
        # of the two and the one blank line between them is two short of it.
        self.assertEqual(len(CLIP_NOTE_AT_END.sub("", render.text)), BUDGET - 2)
        both = message(1, 0, text="q" * 3900) + [tool(100 + i, 2 + i, "m1") for i in range(8)]
        both += message(2, 20, role="assistant") + [tool(i, 30 + i, "m2") for i in range(20)]
        render = render_transcript(conversation(both), cfg=PromptConfig())
        self.assertLessEqual(render.chars, BUDGET + MARKERS)
        first, second = render.text.split("\n\n[message element_id=m2", 1)
        for block in (first, second):
            self.assertRegex(block, r" \.\.\.\[clipped \d+ chars\]$")
        shown = [CLIP_NOTE_AT_END.sub("", block) for block in (first, second)]
        self.assertEqual(sum(map(len, shown)), BUDGET - 4 - len("[message element_id=m2"))
        self.assertLess(abs(len(first) - len(second)), MARKERS, "two large blocks halve it")

    def test_a_long_id_role_name_or_status_is_clipped_in_its_head_line_and_the_list(self):
        planted = {
            "message id": message(1, 0, mid="I" * BIG) + message(2, 2, role="assistant"),
            "role": message(1, 0, role="R" * BIG) + message(2, 2, role="assistant"),
            "tool id": message(1, 0)
            + message(2, 2, role="assistant")
            + [tool(1, 9, "m2", id="T" * BIG)],
            "tool name": message(1, 0)
            + message(2, 2, role="assistant")
            + [tool(1, 9, "m2", name="N" * BIG)],
            "tool status": message(1, 0)
            + message(2, 2, role="assistant")
            + [tool(1, 9, "m2", status="S" * BIG)],
        }
        for field, elements in planted.items():
            with self.subTest(field=field):
                text = self.scorer(elements)
                self.assertLess(len(text), 3 * BUDGET)
                shown = (
                    LETTERS[field] * MAX_HEAD_FIELD_CHARS
                    + f" ...[clipped {BIG - MAX_HEAD_FIELD_CHARS} chars]"
                )
                self.assertIn(shown + AFTER[field], text)
                if field.endswith("id"):
                    self.assertIn(
                        '"element_id": "' + shown + '"', text, "the list shows it as the head"
                    )

    def test_a_masked_transcript_is_held_to_the_budget(self):
        elements: list = []
        for i in range(12):
            # 3,900 characters each, within the message clip, so each is masked
            # with its block and grows there (a text past the clip is masked
            # before the cut: the row on a prompt's clip).
            elements += message(i, 2 * i, text=("tok" + "en=abcdef ") * 300)
        render = render_transcript(conversation(elements), cfg=PromptConfig())
        self.assertLessEqual(render.chars, BUDGET + MARKERS)
        self.assertGreater(render.redaction_counts.get("secret-assignment", 0), 0)
        self.assertNotIn("abcdef", render.text)
        kept = render.text.split("\n\n")
        kept.remove(ELISION_MARKER.format(n=render.blocks_elided))
        self.assertEqual(render.blocks_elided, render.blocks_total - len(kept))
        self.assertEqual(len(kept), 2)
        for block in kept:
            # The middle ones are dropped on their masked lengths, so what is
            # kept fits whole: none is clipped.
            self.assertNotIn("[clipped", block)

    def test_a_key_block_opened_in_one_message_and_closed_in_a_later_one_hides_none_between(self):
        elements = message(1, 0, text=f"see {KEY_OPEN}\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC")
        elements += message(2, 2, role="assistant", text="AGENT-SENT-THE-KEY-ONWARD")
        elements += message(3, 4, text=f"tail {KEY_CLOSE} done")
        render = render_transcript(conversation(elements), cfg=PromptConfig())
        self.assertNotIn("MIIEvQIBADANBgkqhkiG9w0BAQEFAASC", render.text)
        self.assertIn("AGENT-SENT-THE-KEY-ONWARD", render.text)
        self.assertIn("[message element_id=m2 position=2 role=assistant]", render.text)

    def test_an_excerpt_the_improver_reads_is_held_to_the_budget(self):
        failure = FailureCase(
            id="FC:thr:r:c1",
            conversation_id="thr",
            rubric_id="r",
            rubric_version=1,
            criterion_id="c1",
            criterion_title="Checks first",
            weight=1,
            rationale="it skipped the check",
            evidence=(EvidenceRef("m2", 2, "message"),),
            signals={},
            attributions=(),
        )
        text = improver_user_prompt(
            skill_id="s1",
            skill_path="skills/s1/SKILL.md",
            base_text="---\nname: s1\n---\nbody\n",
            cases=[failure],
            conversations={"thr": conversation(busy_reply(500))},
            cfg=PromptConfig(),
            seed="c193",
        )
        excerpt = text.split("## CITED CONVERSATION EXCERPTS", 1)[1].split("## OUTPUT CONTRACT")[0]
        self.assertLessEqual(len(excerpt), BUDGET + 2 * MARKERS)
        self.assertIn("[message element_id=m2 position=2 role=assistant]", excerpt)

    def test_many_short_messages_are_rendered_in_linear_time(self):
        elements: list = []
        for i in range(200_000):
            elements += message(i, 2 * i)
        many = conversation(elements)
        with deadline(5.0):
            render = render_transcript(many, cfg=PromptConfig())
        self.assertLessEqual(render.chars, BUDGET + MARKERS)
        self.assertTrue(render.text.startswith("[message element_id=m0 position=0 role=user]"))
        self.assertTrue(
            render.text.endswith("[message element_id=m199999 position=399998 role=user]\nhi")
        )
        self.assertIn(ELISION_MARKER.format(n=render.blocks_elided), render.text)
        self.assertEqual(render.blocks_total, 200_000)
        self.assertEqual(
            render.blocks_elided, render.blocks_total - render.text.count("[message element_id=")
        )

    def test_the_control_dropping_the_middle_keeps_the_blocks_the_old_loop_kept(self):
        rng = random.Random(193)
        for _ in range(5_000):
            budget = rng.randint(1, 400)
            sizes = [rng.randint(1, 120) for _ in range(rng.randint(0, 30))]
            self.assertEqual(_kept_indices(sizes, budget), old_kept(sizes, budget), (sizes, budget))

    def test_the_control_a_transcript_inside_the_budget_is_rendered_as_before(self):
        elements = message(1, 0, text="please check") + message(2, 2, role="assistant", text="done")
        elements.append(tool(1, 3, "m2", input={"cmd": "ls"}, output="a.txt"))
        render = render_transcript(conversation(elements), cfg=PromptConfig())
        self.assertEqual(
            render.text,
            "[message element_id=m1 position=0 role=user]\nplease check\n\n"
            "[message element_id=m2 position=2 role=assistant]\ndone\n"
            "[tool element_id=t1 position=3 name=Bash status=done]\n"
            'input: {"cmd": "ls"}\noutput: a.txt',
        )
        self.assertEqual((render.blocks_total, render.blocks_elided), (2, 0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
