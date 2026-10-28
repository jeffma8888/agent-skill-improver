"""A forged delimiter is neutralized in time in proportion to the text.

Measured before (3.11 and 3.13): ``wrap_untrusted`` searched for a forged
delimiter with ``<{2,}\\s*/?\\s*``, which tried a run of ``<`` from every one of
its characters and split the space around the optional ``/`` every way there
is. A run of 24,000 ``<``, the default skill text budget, took 10.4 to 11.6 s to
fence, four times as long per doubling, and ``<<`` and then 24,000 spaces 4.7
to 6.2 s. Since then a delimiter is replaced by its own characters only, and not
the text after it (``test_a_forged_delimiter_hides_no_text_but_its_own``).
"""

from __future__ import annotations

import random
import re
import unittest

from agent_skill_improver.app.prompts import FENCE_WORD, NEUTRALIZED, _FENCE_RE, wrap_untrusted

from tests.test_skill_identity_and_payloads import deadline

#: The pattern this change replaced, kept to show each of its matches still
#: begins one, at the same place.
OLD = re.compile(r"<{2,}\s*/?\s*(?:END-)?UNTRUSTED-DATA[^>]*>{0,3}", re.IGNORECASE)
LONG = 262_144  # 256 KiB


def _body(text: str) -> str:
    """What ``wrap_untrusted`` put between its own delimiters."""
    out = wrap_untrusted(text, label="t", nonce="9f2a1c7b")
    return out.split("\n", 1)[1].rsplit("\n", 1)[0]


class AForgedDelimiterIsNeutralizedInLinearTimeTests(unittest.TestCase):
    def test_a_long_run_of_angle_brackets_is_fenced_in_linear_time(self):
        with deadline(5.0):
            self.assertEqual(_body("<" * LONG), "<" * LONG)
            body = _body("<" * LONG + f"{FENCE_WORD} x>>> after")
        self.assertEqual(body, NEUTRALIZED + " x>>> after")

    def test_a_long_run_of_space_after_two_angle_brackets_is_fenced_in_linear_time(self):
        spaces = " " * LONG
        with deadline(5.0):
            self.assertEqual(_body("<<" + spaces + "x"), "<<" + spaces + "x")
            body = _body("<<" + spaces + "/" + spaces + f"END-{FENCE_WORD} x>>> after")
        self.assertEqual(body, NEUTRALIZED + " x>>> after")

    def test_the_control_a_whole_run_of_angle_brackets_is_neutralized_with_the_word(self):
        for text, expected in (
            (f"a <<<<<<<{FENCE_WORD} x>>> b", f"a {NEUTRALIZED} x>>> b"),
            (f"<<{FENCE_WORD}>><<< / end-{FENCE_WORD.lower()} y>>>>", NEUTRALIZED * 2 + " y>>>>"),
            (f"<<{FENCE_WORD}>>>>", NEUTRALIZED + ">"),
            (f"< {FENCE_WORD}> <<\t/\tEND-{FENCE_WORD}", f"< {FENCE_WORD}> {NEUTRALIZED}"),
        ):
            with self.subTest(text=text):
                self.assertEqual(_body(text), expected)

    def test_the_control_each_match_of_the_old_pattern_begins_with_one_of_this_one(self):
        pieces = ["<", "<", "<", ">", "/", " ", "\t", "\n", "END-", "end-", FENCE_WORD]
        pieces += [FENCE_WORD.lower(), "x", "-", "\u00a0", "\u2028", "\u3000", "\x1c", "\x85"]
        rng = random.Random(146)
        for _ in range(20_000):
            text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 16)))
            old = [m.span() for m in OLD.finditer(text)]
            new = {m.start(): m.end() for m in _FENCE_RE.finditer(text)}
            for start, end in old:
                self.assertIn(start, new, repr(text))
                self.assertLessEqual(new[start], end, repr(text))
            for start in new:
                self.assertTrue(any(s <= start < e for s, e in old), repr(text))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
