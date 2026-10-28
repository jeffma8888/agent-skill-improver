"""A prompt's clip masks before it cuts.

``render_transcript`` in ``app/prompts.py`` cuts each head field (an element's
id, a message's role, a tool's name and status) at 256 characters, a message's
text at ``max_message_chars`` and a tool's input and output at the prompt's own
bounds, and the citable list cuts an id at 256. Measured before (3.11 and
3.13): every one of these cut first and masked after, so a token the cut split
kept a prefix the masking shapes no longer knew. With a ``ghp_`` token of 40
characters after 200 to 269 characters of padding, 16 of the 70 cuts in each of
the five head fields showed 8 or more of its characters raw, 23 at most, and
the same 16 of 50 cuts of a message's text. A cut text is masked once: masked
again with its block, as this change's second form did, a URL parameter's
placeholder was read as the parameter's value, so its mask counted twice and a
GitHub token it held went uncounted.
"""

from __future__ import annotations

import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from agent_skill_improver.adapters.transcript_export import normalize_export
from agent_skill_improver.app import prompts
from agent_skill_improver.app.prompts import (
    MASK_PAST_CUT,
    MAX_HEAD_FIELD_CHARS,
    render_transcript,
    scorer_user_prompt,
)
from agent_skill_improver.config import PromptConfig, load_config
from agent_skill_improver.domain.redaction import PLACEHOLDER, redact, redact_placed

from tests.support import build_env

TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
MASK = "[REDACTED:github-token]"
KEY_OPEN = "-----BEGIN " + "PRIVATE KEY-----"
KEY_CLOSE = "-----END " + "PRIVATE KEY-----"
FIELDS = ("message id", "role", "tool id", "tool name", "tool status")
URL_KEY = "tok" + "en"


def elements(*, field: str = "", value: str = "", text: str = "hi", **tool: Any) -> list:
    m1: dict = {"id": "m1", "position": 0, "kind": "message", "role": "user"}
    m2: dict = {"id": "m2", "position": 2, "kind": "message", "role": "assistant"}
    t1: dict = {"id": "t1", "position": 3, "kind": "tool", "parentId": "m2"}
    t1.update({"name": "Bash", "status": "done", "input": "ls", "output": "ok"}, **tool)
    if field:
        target = t1 if field.startswith("tool") else m1
        target[{"message id": "id", "tool id": "id"}.get(field, field.split()[-1])] = value
    x1 = {"id": "x1", "position": 1, "kind": "text", "parentId": m1["id"], "content": text}
    return [m1, x1, m2, t1]


def conversation(els: list) -> Any:
    return normalize_export({"thread": {"id": "thr"}, "elements": els})


def shown_of_token(text: str) -> int:
    """How many of the token's leading characters appear raw in ``text``."""
    return max([k for k in range(5, len(TOKEN) + 1) if TOKEN[:k] in text] or [0])


class APromptsClipMasksBeforeItCutsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.config = load_config(build_env(Path(cls._tmp.name), dry_run=True))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def scorer(self, els: list, cfg: PromptConfig | None = None) -> str:
        text, _ = scorer_user_prompt(
            conversation(els), self.config.rubrics[0], cfg=cfg or PromptConfig(), seed="c194"
        )
        return text

    def test_a_token_a_head_fields_cut_would_split_is_masked_whole(self):
        for field in FIELDS:
            for pad in range(200, 270):
                with self.subTest(field=field, pad=pad):
                    text = self.scorer(elements(field=field, value="x" * pad + " " + TOKEN))
                    self.assertLess(shown_of_token(text), 8)
            with self.subTest(field=field, pad=236):
                text = self.scorer(elements(field=field, value="x" * 236 + " " + TOKEN))
                shown = "x" * 236 + " " + MASK[:19] + " ...[clipped 4 chars]"
                self.assertIn(("=" if field.endswith("id") else "") + shown, text)
                if field.endswith("id"):
                    self.assertIn('"element_id": "' + shown + '"', text, "the list cuts it so")

    def test_a_token_a_messages_or_a_tools_cut_would_split_is_masked_whole(self):
        small = PromptConfig(max_message_chars=300)
        cases = {
            "message text": (PromptConfig(), 4000, lambda v: elements(text=v)),
            "tool input": (small, 300, lambda v: elements(input=v)),
            "tool output": (small, 300, lambda v: elements(output=v)),
        }
        for where, (cfg, limit, build) in cases.items():
            for pad in range(limit - 45, limit + 5):
                with self.subTest(where=where, pad=pad):
                    render = render_transcript(
                        conversation(build("x" * pad + " " + TOKEN)), cfg=cfg
                    )
                    self.assertLess(shown_of_token(render.text), 8)

    def test_the_masks_counted_are_the_ones_the_cut_shows(self):
        one = "x" * 3990 + " " + TOKEN + " " + "y" * 100 + " " + TOKEN
        render = render_transcript(conversation(elements(text=one)), cfg=PromptConfig())
        self.assertIn(" " + MASK[:9] + " ...[clipped", render.text)
        self.assertEqual(render.redaction_counts, {"github-token": 1}, "the cut one, not the next")
        far = "x" * 4000 + " " + "y" * (2 * MASK_PAST_CUT) + " " + TOKEN
        render = render_transcript(conversation(elements(text=far)), cfg=PromptConfig())
        self.assertEqual(render.redaction_counts, {}, "a token past the cut is not shown")

    def test_the_masks_a_dropped_block_took_are_not_counted(self):
        els: list = []
        for i in range(12):
            role = "x" * 236 + " " + TOKEN
            text = ("tok" + "en=abcdef ") * 300
            els += [
                {"id": f"m{i}", "position": 2 * i, "kind": "message", "role": role},
                {"id": f"x{i}", "position": 2 * i + 1, "kind": "text", "parentId": f"m{i}"}
                | {"content": text},
            ]
        render = render_transcript(conversation(els), cfg=PromptConfig())
        kept = render.blocks_total - render.blocks_elided
        self.assertEqual(kept, 2)
        self.assertEqual(render.redaction_counts.get("github-token"), kept)
        self.assertNotIn(TOKEN[:8], render.text)

    def test_the_control_a_key_whose_end_is_past_what_the_mask_reads_is_masked(self):
        # The header before the cut, the body across it, the end far past what
        # the mask reads: a header with no end is masked to the end of the text.
        body = "x" * 3900 + "\n" + KEY_OPEN + "\n" + "MIIE" * 5000 + "\n" + KEY_CLOSE
        render = render_transcript(conversation(elements(text=body)), cfg=PromptConfig())
        self.assertNotIn("MIIEMIIE", render.text)
        self.assertNotIn("PRIVATE KEY", render.text)
        self.assertIn("x\n[REDACTED:private-key-block]", render.text)

    def test_only_what_a_kept_block_shows_is_masked(self):
        els: list = []
        for i in range(2000):
            els += [
                {"id": f"m{i}", "position": 2 * i, "kind": "message", "role": "user"},
                {"id": f"x{i}", "position": 2 * i + 1, "kind": "text", "parentId": f"m{i}"}
                | {"content": "word " * 4000},
            ]
        els[0]["role"] = "R" * (2 * 1024 * 1024)
        els[-2]["id"] = els[-1]["parentId"] = "I" * (2 * 1024 * 1024)
        with (
            mock.patch.object(prompts, "redact", wraps=redact) as spy,
            mock.patch.object(prompts, "redact_placed", wraps=redact_placed) as placed,
        ):
            text = self.scorer(els)
        calls = spy.call_args_list + placed.call_args_list
        self.assertTrue(placed.call_args_list, "a cut text is masked through redact_placed")
        masked = sum(len(call.args[0]) for call in calls)
        self.assertIn("[message element_id=" + "I" * MAX_HEAD_FIELD_CHARS + " ...[clipped", text)
        # The kept blocks, each cut text up to MASK_PAST_CUT past its cut, and
        # the citable list's ids, a long one as far: never a dropped block, nor
        # the far part of a long text (the two of 2 MiB, 40 MB of messages).
        self.assertLess(masked, 200_000, masked)

    def test_with_masking_turned_off_a_cut_is_made_as_written(self):
        off = PromptConfig(redact_secrets=False)
        text = self.scorer(elements(field="tool name", value="x" * 236 + " " + TOKEN), off)
        self.assertIn("x" * 236 + " " + TOKEN[:19] + " ...[clipped 21 chars]", text)
        render = render_transcript(conversation(elements(text="x" * 3990 + " " + TOKEN)), cfg=off)
        self.assertIn(" " + TOKEN[:9] + " ...[clipped 31 chars]", render.text)
        self.assertEqual(render.redaction_counts, {})

    def test_the_control_a_text_within_its_cut_is_masked_with_its_block(self):
        render = render_transcript(
            conversation(elements(field="tool name", value="n " + TOKEN, text="see " + TOKEN)),
            cfg=PromptConfig(),
        )
        self.assertIn("name=n " + MASK + " status=done]", render.text)
        self.assertIn("\nsee " + MASK + "\n", render.text)
        self.assertEqual(render.redaction_counts, {"github-token": 2})

    def test_a_url_credential_past_the_cut_is_counted_once_with_what_it_holds(self):
        plain = "https://h/p?" + URL_KEY + "=" + "Z" * 16 + "&a=1"
        holding = "https://h/p?" + URL_KEY + "=" + TOKEN
        cases = (
            (plain, {"url-credential": 1}),
            (holding, {"github-token": 1, "url-credential": 1}),
        )
        for url, want in cases:
            for pad, where in ((10, "within its cut"), (5000, "past it")):
                with self.subTest(url=url[:16], where=where):
                    text = url + " " + "x" * pad
                    render = render_transcript(
                        conversation(elements(text=text)), cfg=PromptConfig()
                    )
                    self.assertEqual(render.redaction_counts, want)
                    self.assertNotIn("Z" * 8, render.text)
                    self.assertNotIn(TOKEN[:8], render.text)

    def test_a_placeholder_the_cut_splits_is_shown_as_the_cut_left_it(self):
        text = "x" * 3970 + " ?" + URL_KEY + "=" + "Z" * 40 + " tail" * 10
        masked = redact(text).text
        render = render_transcript(conversation(elements(text=text)), cfg=PromptConfig())
        shown = masked[:4000] + f" ...[clipped {len(masked) - 4000} chars]"
        self.assertTrue(shown.startswith("x" * 3970 + " ?" + URL_KEY + "=[REDACTED:url-cred"))
        self.assertIn("\n" + shown + "\n", render.text)
        self.assertEqual(render.redaction_counts, {"url-credential": 1})

    def test_a_jwt_the_cut_splits_is_read_whole_as_far_as_the_mask_reads(self):
        # Its first part alone runs 4,000 characters past the cut.
        jwt = "ey" + "J" + "a" * 4000 + "." + "b" * 20 + "." + "c" * 20
        text = "x" * 3990 + " " + jwt
        render = render_transcript(conversation(elements(text=text)), cfg=PromptConfig())
        self.assertNotIn("ey" + "Jaaaaaa", render.text)
        self.assertIn("x " + "[REDACTED ...[clipped 5 chars]", render.text)
        self.assertEqual(render.redaction_counts, {"jwt": 1})

    def test_a_mask_counts_when_its_placeholder_starts_before_the_cut(self):
        for pad, want in ((3998, {"github-token": 1}), (3999, {})):
            with self.subTest(placeholder_at=pad + 1):
                text = "x" * pad + " " + TOKEN + " " + "y" * 100
                render = render_transcript(conversation(elements(text=text)), cfg=PromptConfig())
                self.assertEqual(render.redaction_counts, want)
                self.assertNotIn(TOKEN[:5], render.text)

    def test_a_cut_text_a_key_block_hides_is_counted_as_a_text_within_its_cut_is(self):
        renders = []
        for name in ("n " + TOKEN, "x" * 236 + " " + TOKEN):
            els = [
                {"id": "m1", "position": 0, "kind": "message", "role": "assistant"},
                {"id": "x1", "position": 1, "kind": "text", "parentId": "m1"}
                | {"content": "see\n" + KEY_OPEN + "\nMIIE"},
                {"id": "t1", "position": 2, "kind": "tool", "parentId": "m1", "name": name}
                | {"status": "done", "input": "ls", "output": "ok"},
            ]
            renders.append(render_transcript(conversation(els), cfg=PromptConfig()))
        within, past = renders
        self.assertTrue(within.text.endswith("\nsee\n[REDACTED:private-key-block]"), within.text)
        self.assertEqual(past.text, within.text)
        self.assertEqual(within.redaction_counts, {"github-token": 1, "private-key-block": 1})
        self.assertEqual(past.redaction_counts, within.redaction_counts)

    def test_each_placeholder_is_placed_with_the_masks_it_holds(self):
        rng = random.Random(194)
        pieces = [
            TOKEN,
            "?" + URL_KEY + "=" + TOKEN,
            "&sig=" + "Q" * 12,
            "pass" + "word=hunter22",
            KEY_OPEN,
            KEY_CLOSE,
            "AK" + "IA" + "A" * 16,
            "ey" + "J" + "a" * 12 + "." + "b" * 12 + ".c",
            "Bearer " + "t" * 20,
            " ",
            "\n",
            "x" * 7,
            "[",
            "]",
        ]
        for _ in range(2000):
            text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
            plain = redact(text)
            placed, places = redact_placed(text)
            self.assertEqual(placed, plain, text)
            total: dict[str, int] = {}
            for _start, held in places:
                for label, n in held.items():
                    total[label] = total.get(label, 0) + n
            self.assertEqual(total, plain.counts, text)
            starts = [start for start, _ in places]
            self.assertEqual(starts, sorted(set(starts)), text)
            for start, held in places:
                labels = [PLACEHOLDER.format(label=label) for label in held]
                self.assertTrue(any(plain.text.startswith(ph, start) for ph in labels), text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
