"""The scanner's hostname and jwt rules, and line numbers, take time in proportion to text.

Measured before: the hostname rule began with a run of host characters and no
anchor, so every position inside a long run of letters, digits and dashes was
a start of its own, each scanned to the end of the run; the jwt rule's first
segment may hold ``-``, so each ``-eyJ`` inside a run was a start of its own;
and each finding's line was counted from the start of the text. Each doubling
of such text took four times as long. A tracked file may be up to 2 MB, and
the history patch is scanned as one text.
"""

from __future__ import annotations

import dataclasses
import random
import re
import unittest

from tests.test_public_safety import leakscan
from tests.test_skill_identity_and_payloads import deadline


def _j(*parts: str) -> str:
    return "".join(parts)


RULES = {rule.id: rule for rule in leakscan.RULES}
HOST_PART = _j(".", "co", "rp", ".")
HEADER = _j("e", "yJ")
#: The lookbehind the hostname rule starts with; without it, the rule as it was.
RUN_START = "(?<![A-Za-z0-9-])"


def _scan(text: str, *rule_ids: str) -> list:
    rules = [RULES[rule_id] for rule_id in rule_ids] or list(leakscan.RULES)
    return leakscan.scan_text(text, scope="test", location="t", rules=rules)


class LeakScanTimeTests(unittest.TestCase):
    def test_a_long_run_of_host_characters_is_scanned_in_linear_time(self):
        text = "b-" * 131_072  # 256 KiB: still running at three minutes before
        with deadline(5.0):
            self.assertEqual(_scan(text), [])
            found = _scan(text + HOST_PART + "example.net", "corp-hostname")
        self.assertEqual(len(found), 1)

    def test_text_full_of_jwt_starts_is_scanned_in_linear_time(self):
        text = (HEADER + "-") * 65_536  # 256 KiB: 12 to 17 s before
        token = HEADER + "a" * 12 + "." + "b" * 12 + ".sig"
        with deadline(5.0):
            self.assertEqual(_scan(text), [])
            found = _scan(text + " " + token, "jwt")
        self.assertEqual(len(found), 1)
        self.assertIn(token[:20], found[0].excerpt)

    def test_many_findings_are_numbered_in_linear_time(self):
        sample = RULES["google-api-key"].positive
        text = (sample + "\n") * 80_000  # about 3 MiB: 36 to 40 s before
        with deadline(5.0):
            found = _scan(text, "google-api-key")
        self.assertEqual(len(found), 80_000)
        self.assertEqual(found[0].location, "t:1")
        self.assertEqual(found[-1].location, "t:80000")

    def test_the_control_each_rule_finds_what_its_old_pattern_found(self):
        host = RULES["corp-hostname"]
        jwt = RULES["jwt"]
        self.assertTrue(host.pattern.startswith(RUN_START))
        old_host = re.compile(host.pattern[len(RUN_START) :], host.flags)
        cases = (
            (
                host,
                old_host,
                ["b", "B", "-", ".", "9", " ", HOST_PART, "x.", "\n", "\u017f", "\u212a"],
            ),
            (jwt, jwt.compiled(), [HEADER, "a", "-", "_", ".", " ", "Z", HEADER + "-", "a" * 10]),
        )
        rng = random.Random(139)
        for rule, old, alphabet in cases:
            matched = 0
            for _ in range(4000):
                text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 40)))
                want = [m.span() for m in old.finditer(text)]
                got = [m.span() for m in leakscan._matches(rule, text)]
                self.assertEqual(got, want, (rule.id, text))
                matched += bool(want)
            with self.subTest(rule=rule.id):
                self.assertGreater(matched, 50, "the generated texts must hold matches")

    def test_the_control_a_finding_is_on_the_line_counted_from_the_start(self):
        sample = RULES["google-api-key"].positive
        for text in (
            "\n\nx " + sample + "\n" + "y\n" * 5 + sample + "\n" + sample,
            sample + "\n" + sample + "\n\n",
            "\n" + sample,
        ):
            with self.subTest(text=text[:8]):
                starts = [m.start() for m in RULES["google-api-key"].compiled().finditer(text)]
                want = [f"t:{text.count(chr(10), 0, start) + 1}" for start in starts]
                self.assertEqual([f.location for f in _scan(text, "google-api-key")], want)
        self.assertEqual(want, ["t:2"])
        # A match may span lines (its ``\s`` takes a line break): its line is the one it starts on.
        spanning = "x\n" + _j("api_", "key") + " =\n  " + '"abcdefgh12345678"'
        found = _scan(spanning, "assigned-secret-literal")
        self.assertEqual([f.location for f in found], ["t:2"])
        # No rule today starts a match on a line break; one that did would be
        # on the line the break ends, as counting from the start says.
        on_break = dataclasses.replace(
            RULES["jwt"], id="on-break", pattern=r"\nb", flags=0, start=None, run=None
        )
        found = leakscan.scan_text("a\nb\nc\nb", scope="test", location="t", rules=[on_break])
        self.assertEqual([f.location for f in found], ["t:1", "t:3"])

    def test_selftest_refuses_a_start_without_a_run_and_a_start_that_skips_its_sample(self):
        jwt = RULES["jwt"]
        broken = (
            dataclasses.replace(jwt, id="no-run", run=None),
            dataclasses.replace(jwt, id="bad-start", start="zzz"),
        )
        original = leakscan.RULES
        leakscan.RULES = original + broken
        try:
            problems = leakscan.selftest()
        finally:
            leakscan.RULES = original
        self.assertIn("no-run: names a start without a run, or a run without a start", problems)
        self.assertIn("bad-start: its start and run skip its own positive sample", problems)
        self.assertEqual(leakscan.selftest(), [])


if __name__ == "__main__":
    unittest.main()
