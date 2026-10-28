"""A tool's excerpt is cut before a credential it would split.

A tool call's input and output are cut when the export is read (800 and 2,000
characters by default, ``_clip`` in ``adapters/transcript_export.py``), before
the prompt masks anything, and a credential cut to a prefix is one the shapes no
longer know. Measured before (3.11 and 3.13, the same on both): of the 645 cut
points inside one credential of each of the fifteen shapes ``redact`` knows, 332
in a tool's output and 332 in its input showed more of the credential raw in
the scorer's transcript than the credential uncut shows, up to 35 of the 36
characters after an npm token's ``npm_``. The cut now moves back to where such a
credential begins; a text it splits none of is cut exactly as before.
"""

from __future__ import annotations

import random
import re
import unittest
from typing import Any
from unittest import mock

from agent_skill_improver.adapters import transcript_export
from agent_skill_improver.adapters.transcript_export import _clip, normalize_export
from agent_skill_improver.app.prompts import render_transcript
from agent_skill_improver.config import PromptConfig
from agent_skill_improver.domain.redaction import credential_starts, redact

from tests.test_the_candidate_listing_masks_and_bounds_what_it_reads import SHAPED

#: The export's two cuts, and how far past a cut it reads, written out so a
#: change in the code fails here.
LIMITS = {"output": 2000, "input": 800}
READS_PAST = 4096
TOKEN = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
WORDS = ("alpha", "beta", "path/to/file.py", "ERROR", "42", "=", "--flag", '{"k": 1}', "gh", "pm_")


def _as_before(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f" ...[clipped {len(text) - limit} chars]"


def _export(field: str, text: str) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "id": "k1",
        "kind": "tool",
        "position": 3,
        "parentId": "m1",
        "name": "bash",
        "status": "done",
        "input": {"cmd": "ls"},
        "output": "ok",
    }
    tool[field] = text
    return {
        "thread": {"id": "conv-1"},
        "elements": [
            {"id": "m1", "kind": "message", "position": 1, "role": "assistant"},
            {"id": "t1", "kind": "text", "position": 2, "parentId": "m1", "content": "hi"},
            tool,
        ],
    }


def _transcript(field: str, text: str) -> str:
    return render_transcript(normalize_export(_export(field, text)), cfg=PromptConfig()).text


def _raw_prefix(secret: str, shown: str) -> int:
    return max((k for k in range(1, len(secret) + 1) if secret[:k] in shown), default=0)


_LAST_PLACEHOLDER = re.compile(r"\[REDACTED:[a-z-]+\]\Z")


def _shows_none_raw(text: str, cut: int, limit: int) -> bool:
    """The head, masked, shows only what the text masked past the cut shows, but
    for one placeholder it may end in (a credential the cut made, which hides)."""
    head = redact(text[:cut]).text
    window = redact(text[: limit + READS_PAST]).text
    last = _LAST_PLACEHOLDER.search(head)
    return window.startswith(head) or (last is not None and window.startswith(head[: last.start()]))


def _kept(excerpt: str) -> str:
    return excerpt[: excerpt.rindex(" ...[clipped ")]


class NoCutLeavesAPrefixRawTests(unittest.TestCase):
    def test_no_cut_inside_any_shape_shows_more_of_it_than_the_credential_whole(self) -> None:
        for field, limit in LIMITS.items():
            for secret in SHAPED:
                with self.subTest(field=field, secret=secret[:12]):
                    whole = _raw_prefix(
                        secret, _transcript(field, "x" * 10 + " " + secret + " tail")
                    )
                    for k in range(1, len(secret)):
                        text = "x" * (limit - k - 1) + " " + secret + " tail"
                        shown = _raw_prefix(secret, _transcript(field, text))
                        self.assertLessEqual(shown, whole, f"cut {k} characters in")

    def test_the_cut_moves_back_to_where_the_credential_begins(self) -> None:
        # Cut 23 characters in, a GitHub token keeps 19 after its prefix, too
        # few for the shape (20); 24 in, the head is masked whole as it is. A
        # second token after the cut is not a place to cut, being past the limit.
        for limit in LIMITS.values():
            for k, moved in ((1, True), (4, True), (23, True), (24, False), (39, False)):
                with self.subTest(limit=limit, k=k):
                    head = "x" * (limit - k - 1) + " "
                    text = head + TOKEN + " " + TOKEN + " tail"
                    cut = len(head) if moved else limit
                    self.assertEqual(
                        _clip(text, limit), f"{text[:cut]} ...[clipped {len(text) - cut} chars]"
                    )

    def test_a_shape_that_keeps_its_name_is_cut_after_the_name(self) -> None:
        limit = LIMITS["output"]
        # Five characters of the value are too few for the shape (six), so the
        # cut moves back to the end of the name; six are masked whole as they are.
        for kept, cut_at_name in ((5, True), (6, False)):
            with self.subTest(kept=kept):
                head = "x" * (limit - kept - 11) + " pass" + "word: "
                text = head + "correct-horse-battery-staple tail"
                self.assertEqual(len(head), limit - kept)
                cut = len(head) if cut_at_name else limit
                self.assertEqual(
                    _clip(text, limit), f"{text[:cut]} ...[clipped {len(text) - cut} chars]"
                )


class TheCutIsAsBeforeWhereItSplitsNoneTests(unittest.TestCase):
    def test_a_credential_wholly_before_or_after_the_cut_is_cut_at_the_limit(self) -> None:
        for limit in LIMITS.values():
            for secret in SHAPED:
                with self.subTest(limit=limit, secret=secret[:12]):
                    before = "x" * (limit - len(secret) - 1) + " " + secret + " " + "y" * 50
                    after = "x" * (limit - 1) + " " + secret + " tail"
                    self.assertEqual(_clip(before, limit), _as_before(before, limit))
                    self.assertEqual(_clip(after, limit), _as_before(after, limit))

    def test_a_cut_inside_a_private_key_block_stays_at_the_limit(self) -> None:
        # The head masks a key block from its header on, its end line cut off or
        # not, so a cut inside one shows none of it and is made where it falls.
        limit = LIMITS["output"]
        begin = "-----BEGIN " + "RSA PRIVATE KEY-----\n"
        end = "\n-----END " + "RSA PRIVATE KEY-----\n"
        text = "p" * 99 + " " + begin + ("MIIEow" + "A" * 58 + "\n") * 40 + end + "q" * 3000
        self.assertLess(text.index(begin), limit)
        self.assertGreater(text.index(end), limit)
        self.assertEqual(_clip(text, limit), _as_before(text, limit))
        self.assertNotIn("MIIEow", _transcript("output", text))

    def test_a_text_with_no_credential_is_cut_exactly_as_before(self) -> None:
        rng = random.Random(198)
        for _ in range(300):
            text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 900)))
            limit = rng.randint(1, 2500)
            with self.subTest(text=text[:30], limit=limit):
                self.assertEqual(_clip(text, limit), _as_before(text, limit))

    def test_a_text_within_its_limit_or_with_no_limit_is_kept_whole(self) -> None:
        text = "x " + TOKEN
        self.assertEqual(_clip(text, len(text)), text)
        self.assertEqual(_clip(text, 0), text)

    def test_a_window_the_mask_finds_nothing_in_costs_one_pass(self) -> None:
        calls: list[int] = []
        real = transcript_export.redact

        def counted(text: str) -> Any:
            calls.append(len(text))
            return real(text)

        with mock.patch.object(transcript_export, "redact", counted):
            _clip("plain " * 2000, 2000)
        self.assertEqual(calls, [2000 + READS_PAST])


class TheCutIsTheLatestThatSplitsNoneTests(unittest.TestCase):
    def test_on_text_dense_with_credentials_the_cut_is_the_latest_place_that_shows_none_raw(
        self,
    ) -> None:
        rng = random.Random(1980)
        glue = (" ", "", "\n", "=", "?to" + "ken=", "/", "Bear" + "er ")
        for _ in range(150):
            parts: list[str] = []
            while sum(map(len, parts)) < 400:
                parts.append(rng.choice(SHAPED) if rng.random() < 0.6 else rng.choice(WORDS))
                parts.append(rng.choice(glue))
            text = "".join(parts)
            limit = rng.randint(20, 300)
            excerpt = _clip(text, limit)
            cut = len(_kept(excerpt))
            with self.subTest(text=text[:40], limit=limit):
                self.assertEqual(excerpt, f"{text[:cut]} ...[clipped {len(text) - cut} chars]")
                self.assertLessEqual(cut, limit)
                self.assertTrue(_shows_none_raw(text, cut, limit))
                places = {limit, *credential_starts(text[: limit + READS_PAST])}
                later = [c for c in places if cut < c <= limit and _shows_none_raw(text, c, limit)]
                self.assertEqual(later, [], "a later place shows none raw")

    def test_a_cut_that_makes_a_credential_stands_at_the_limit(self) -> None:
        # Cut at the limit, a name of 40 characters after ``npm_`` keeps 36, an
        # npm token's length, which the text as a whole does not hold: the head
        # masks it, hiding characters the text shows rather than showing any, so
        # the cut stands, and a token wholly before it does not draw it back.
        limit = LIMITS["output"]
        name = "np" + "m_" + "a" * 40
        real = transcript_export.redact
        for before in ("", "y" * 300 + " " + TOKEN + " "):
            with self.subTest(token_before=bool(before)):
                text = (
                    before + "x" * (limit - 41 - len(before)) + " " + name + " " + TOKEN + " tail"
                )
                self.assertEqual(text.index(name), limit - 40)
                self.assertIn("[REDACTED:npm-token]", redact(text[:limit]).text)
                self.assertNotIn("[REDACTED:npm-token]", redact(text).text)
                calls: list[int] = []

                def counted(part: str) -> Any:
                    calls.append(len(part))
                    return real(part)

                with mock.patch.object(transcript_export, "redact", counted):
                    excerpt = _clip(text, limit)
                self.assertEqual(excerpt, _as_before(text, limit))
                self.assertEqual(calls, [len(text), limit], "the text read, then the limit")
                self.assertLess(len(text), limit + READS_PAST)

    def test_a_credential_the_cut_splits_is_cut_where_it_begins_when_that_cut_makes_one(
        self,
    ) -> None:
        # ``1m`` (the end of a terminal colour code) lets a JWT begin right after
        # 34 characters glued to ``npm_``: cut where the JWT begins, the head
        # ends in an npm token's length, which it masks, and shows none raw.
        limit = LIMITS["output"]
        jwt = "e" + "yJ" + "a" * 300 + "." + "b" * 20 + "." + "c" * 10
        begins = limit - 100
        text = "q" * (begins - 41) + " " + "np" + "m_" + "a" * 34 + "1m" + jwt + " tail"
        self.assertEqual(text.index("e" + "yJ"), begins)
        self.assertIn("[REDACTED:npm-token]", redact(text[:begins]).text)
        self.assertNotIn("[REDACTED:npm-token]", redact(text).text)
        self.assertIn("[REDACTED:jwt]", redact(text).text)
        self.assertEqual(len(_kept(_clip(text, limit))), begins)
        self.assertNotIn("a" * 20, _transcript("output", text))

    def test_with_no_place_that_shows_none_raw_the_excerpt_keeps_nothing(self) -> None:
        # No text measured reaches it, but with no place below the limit to try
        # and a limit that splits a token, the cut falls to the empty head, which
        # shows nothing, rather than failing for want of a place.
        limit = LIMITS["output"]
        text = "x" * (limit - 5) + " " + TOKEN + " tail"
        self.assertEqual(len(_kept(_clip(text, limit))), limit - 4, "the token's start is tried")
        with mock.patch.object(transcript_export, "credential_starts", return_value=[]):
            self.assertEqual(_clip(text, limit), f" ...[clipped {len(text)} chars]")

    def test_each_shape_is_found_where_it_begins_and_a_kept_name_where_it_ends(self) -> None:
        for secret in SHAPED:
            with self.subTest(secret=secret[:12]):
                text = "x " + secret + " tail"
                starts = credential_starts(text)
                self.assertEqual(starts, sorted(set(starts)))
                masked = redact(text).text
                begins = masked.index("[REDACTED:")
                self.assertIn(begins, starts, "the mask begins at a place to try")
        self.assertEqual(credential_starts("x " + "pass" + "word: hunter22"), [2, 12])
        self.assertEqual(credential_starts("plain words only"), [])

    def test_a_credential_that_ends_past_what_the_cut_reads_is_cut_at_the_limit(self) -> None:
        # A JWT is known by its two dots; with the second one past what the cut
        # reads it is not seen whole, so the cut stands at the limit.
        limit = LIMITS["output"]
        head = "x" * (limit - 11) + " "
        for second_dot, moved in ((limit + READS_PAST - 1, True), (limit + READS_PAST, False)):
            with self.subTest(second_dot=second_dot):
                body = "e" + "yJ" + "a" * (second_dot - len(head) - 3 - 12) + "." + "b" * 11
                text = head + body + "." + "c" * 20
                self.assertEqual(text.index(".", len(head) + len(body) - 1), second_dot)
                expected = len(head) if moved else limit
                self.assertEqual(len(_kept(_clip(text, limit))), expected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
