"""The redactor must know at least every credential shape the repo scanner bans.

Two pattern lists lived in this repository: ``tools/leakscan.py``, which keeps
credential shapes out of the public tree, and ``domain/redaction.py``, which
keeps them out of the text forwarded to a model. The scanner knew model-provider
keys, Google keys, fine-grained GitHub tokens and JWTs; the redactor did not. A
transcript that pasted any of those went to the model verbatim while the
scanner would have refused the same bytes in a fixture.

Every sample below is assembled at run time from fragments, so no credential
shape ever appears literally in tracked text.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from agent_skill_improver.domain.redaction import (
    _SHAPES,
    HIGH_CONFIDENCE_LABELS,
    PLACEHOLDER,
    _matches,
    credential_labels,
    kept_prose,
    quoted_value,
    redact,
)

from tests.test_skill_identity_and_payloads import deadline

REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("leakscan", REPO / "tools" / "leakscan.py")
assert _SPEC is not None and _SPEC.loader is not None
leakscan = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("leakscan", leakscan)
_SPEC.loader.exec_module(leakscan)

#: Scanner rules whose POSITIVE sample is a credential the redactor must catch.
#: Identity, path and product rules are about the repository, not transcripts.
CREDENTIAL_RULE_IDS = {
    "aws-access-key-id",
    "aws-secret-key-field",
    "github-token",
    "github-fine-grained-token",
    "slack-token",
    "openai-style-key",
    "google-api-key",
    "jwt",
    "assigned-secret-literal",
    "bearer-literal",
}


def _j(*parts: str) -> str:
    return "".join(parts)


class RedactorCoversTheScannerTests(unittest.TestCase):
    def test_the_rule_set_still_contains_every_shape_this_test_expects(self):
        ids = {rule.id for rule in leakscan.RULES}
        self.assertTrue(CREDENTIAL_RULE_IDS <= ids, CREDENTIAL_RULE_IDS - ids)

    def test_every_credential_shape_the_scanner_bans_is_redacted(self):
        for rule in leakscan.RULES:
            if rule.id not in CREDENTIAL_RULE_IDS:
                continue
            with self.subTest(rule=rule.id):
                sample = f"note: {rule.positive} end"
                result = redact(sample)
                self.assertGreater(
                    result.total,
                    0,
                    f"the scanner refuses {rule.id} in the repo but the redactor forwards it",
                )
                self.assertNotIn(rule.positive, result.text)

    def test_the_scanner_negatives_are_left_alone_where_they_are_benign(self):
        for rule in leakscan.RULES:
            if rule.id not in CREDENTIAL_RULE_IDS:
                continue
            with self.subTest(rule=rule.id):
                # The scanner's own negative is text that must NOT be reported.
                # A redactor that rewrites it is a redactor that mangles prose.
                if rule.id == "assigned-secret-literal":
                    continue  # its negative is a config line the redactor may touch
                self.assertEqual(redact(rule.negative).total, 0, rule.negative)


class SpecificShapesTests(unittest.TestCase):
    def test_a_private_key_cut_off_by_excerpt_clipping_is_still_redacted(self):
        """Excerpts are clipped to ``output_excerpt_chars``. A key block whose
        END line fell past the clip has no closing delimiter, and a pattern that
        requires one left the whole key body in the prompt."""
        head = _j("-----BE", "GIN RSA PRIV", "ATE KEY-----")
        body = "\n".join(["MIIEowIBAAKCAQEA" + "q" * 48] * 5)
        clipped = f"pasted:\n{head}\n{body}\nMIIE"
        result = redact(clipped)
        self.assertNotIn("MIIEowIBAAKCAQEA", result.text)
        self.assertIn("private-key-block", result.counts)

    def test_a_complete_private_key_block_is_redacted_as_one_unit(self):
        head = _j("-----BE", "GIN OPENSSH PRIV", "ATE KEY-----")
        tail = _j("-----E", "ND OPENSSH PRIV", "ATE KEY-----")
        text = f"{head}\nabc\n{tail}\nafter"
        result = redact(text)
        self.assertEqual(result.counts.get("private-key-block"), 1)
        self.assertTrue(result.text.endswith("after"))

    def test_a_jwt_is_redacted(self):
        token = _j("e", "yJ") + "a" * 20 + "." + _j("e", "yJ") + "b" * 20 + "." + "c" * 30
        self.assertNotIn(token, redact(f"Authorization: {token}").text)

    def test_a_secret_in_a_url_query_is_redacted(self):
        url = "https://example.invalid/cb?" + _j("access_", "token") + "=" + "t" * 32 + "&x=1"
        result = redact(url)
        self.assertNotIn("t" * 32, result.text)
        self.assertIn("&x=1", result.text, "only the credential goes, not the rest of the URL")

    def test_ordinary_prose_with_the_word_token_is_not_redacted(self):
        text = "The token budget is 4096 and the password policy page is short."
        self.assertEqual(redact(text).total, 0)

    def test_a_git_sha_and_a_uuid_are_not_mistaken_for_credentials(self):
        text = "commit 3a87c3d51fae0a8fab8419392d75c3e9cf8c6d25 id 123e4567-e89b-12d3-a456-426614174000"
        self.assertEqual(redact(text).total, 0)


class HighConfidenceDetectorTests(unittest.TestCase):
    """``credential_labels`` drives a REFUSAL (a skill is never shown to a model),
    so it must fire on vendor-prefixed tokens and key blocks and stay silent on
    the prose shapes a skill legitimately contains."""

    def test_a_vendor_token_in_a_skill_is_detected_by_label(self):
        text = "Example:\n  export GH_TOKEN=" + "g" + "hp_" + "A" * 36 + "\n"
        self.assertEqual(credential_labels(text), ("github-token",))

    def test_a_clipped_key_block_is_detected(self):
        text = _j("-----BE", "GIN EC PRIV", "ATE KEY-----") + "\nMHcCAQEE" + "k" * 40
        self.assertIn("private-key-block", credential_labels(text))

    def test_placeholder_prose_does_not_trigger_a_refusal(self):
        text = (
            "Set `password: <your-password>` in the config.\n"
            "Pass `--token $MY_TOKEN` on the command line.\n"
            "The key= parameter accepts api_key=YOUR_KEY_HERE_12345.\n"
        )
        self.assertEqual(credential_labels(text), ())
        self.assertGreater(redact(text).total, 0, "the same prose IS still redacted in transcripts")

    def test_the_high_confidence_set_is_a_subset_of_the_known_shapes(self):
        from agent_skill_improver.domain import redaction

        labels = {shape.label for shape in redaction._SHAPES}
        self.assertTrue(HIGH_CONFIDENCE_LABELS <= labels, HIGH_CONFIDENCE_LABELS - labels)
        self.assertNotIn("secret-assignment", HIGH_CONFIDENCE_LABELS)
        self.assertNotIn("url-credential", HIGH_CONFIDENCE_LABELS)


class MaskingTimeTests(unittest.TestCase):
    """Masking takes time in proportion to the text, whatever the text holds.

    Measured before: a JWT segment may hold ``-``, so each ``-eyJ`` inside a
    run of token characters was a start of its own, and each was scanned to
    the end of the run; 64 KiB of such text took about 0.7 s on 3.11 and 1.1 s
    on 3.13, and each doubling four times as long. Model output is masked
    before it is cut, up to the 4 MiB a role may print, so one role's reply
    could hold a daily run for most of an hour or more.
    """

    HEADER = _j("e", "yJ")
    #: 256 KiB of starts with no token after any of them: about 18 s before.
    STARTS = (HEADER + "-") * 65_536

    def token(self, fill: str = "a") -> str:
        return self.HEADER + fill * 20 + "." + self.HEADER + "b" * 20 + "." + "c" * 30

    def test_text_full_of_token_starts_and_no_token_is_masked_in_linear_time(self):
        with deadline(5.0):
            result = redact(self.STARTS)
            labels = credential_labels(self.STARTS)
        self.assertEqual(result.text, self.STARTS, "there is no token here, so nothing is masked")
        self.assertEqual(labels, ())

    def test_every_path_that_quotes_model_output_masks_such_text_in_linear_time(self):
        from agent_skill_improver.adapters.bounded_process import quotable

        with deadline(5.0):
            quoted = quotable(self.STARTS, limit=400, from_end=False)
            kept = kept_prose(self.STARTS, limit=2000)
            shown = quoted_value(self.STARTS)
        self.assertEqual(quoted, self.STARTS[:400])
        self.assertTrue(kept.startswith(self.STARTS[:2000]))
        self.assertIn("more chars not quoted", shown)

    def test_a_token_right_after_a_run_with_no_token_is_still_masked(self):
        token = self.token()
        text = (self.HEADER + "-") * 1000 + " " + token + " tail"
        result = redact(text)
        self.assertNotIn(token, result.text)
        self.assertEqual(result.counts.get("jwt"), 1)
        self.assertTrue(result.text.endswith(PLACEHOLDER.format(label="jwt") + " tail"))
        self.assertEqual(credential_labels(text), ("jwt",))

    def test_a_token_right_after_a_short_false_start_is_still_masked(self):
        """The false start's run ends at the dot; the token starts right after it."""
        token = self.token()
        text = self.HEADER + "-aaa." + token
        self.assertEqual(redact(text).text, self.HEADER + "-aaa." + PLACEHOLDER.format(label="jwt"))

    def test_a_token_after_a_word_character_and_a_dash_is_still_masked(self):
        """The ``eyJ`` right after an ASCII letter is no start (``_LEAD``); after
        ``_`` or a non-ASCII letter it is one, and its match runs on through
        the ``-`` to the token's end. Either way the token is masked once: a
        start found where no shape may begin would skip to the end of the run
        and past the token."""
        token = self.token()
        for before in ("a", "_", "\u00e9", "\u0663"):
            with self.subTest(before=before):
                text = before + self.HEADER + "-" + token
                result = redact(text)
                self.assertEqual(result.counts.get("jwt"), 1)
                self.assertNotIn(token, result.text)
                self.assertEqual(credential_labels(text), ("jwt",))

    def test_a_token_whose_later_segments_hold_further_starts_is_masked_once(self):
        """The scan resumes where a token ends, never inside it."""
        segment = self.HEADER + "c" * 10
        text = self.HEADER + "a" * 10 + "." + segment + "." + segment + "." + segment
        result = redact(text)
        self.assertEqual(result.counts, {"jwt": 1})
        self.assertEqual(result.text, PLACEHOLDER.format(label="jwt") + "." + segment)

    def _samples(self) -> list[str]:
        import random

        rng = random.Random(133)
        pieces = [
            self.HEADER,
            self.HEADER + "a" * 10,
            "." + self.HEADER + "b" * 10,
            "-",
            ".",
            "_",
            " ",
            "a" * 5,
            "b" * 10,
            "x",
            "=",
            "&",
            "?",
            _j("tok", "en="),
            _j("s", "k-") + "q" * 20,
            _j("x", "oxb-") + "1" * 10,
            "\\n",
            "%2F",
            "\x1b[1m",
            "\u00e9",
        ]
        return ["".join(rng.choice(pieces) for _ in range(rng.randint(0, 14))) for _ in range(4000)]

    def test_the_control_the_scan_finds_what_each_pattern_finds(self):
        for text in self._samples():
            for shape in _SHAPES:
                expected = [(m.start(), m.end()) for m in shape.pattern.finditer(text)]
                found = [(m.start(), m.end()) for m in _matches(shape, text)]
                self.assertEqual(found, expected, (shape.label, text))

    def test_the_control_redaction_leaves_what_the_patterns_alone_leave(self):
        for text in self._samples():
            out, counts = text, {}
            for shape in _SHAPES:
                placeholder = PLACEHOLDER.format(label=shape.label)
                repl = r"\g<1>" + placeholder if shape.keep_prefix else placeholder
                out, n = shape.pattern.subn(repl, out)
                if n:
                    counts[shape.label] = n
            result = redact(text)
            self.assertEqual((result.text, result.counts), (out, counts), text)


if __name__ == "__main__":
    unittest.main()


class KeptProseTests(unittest.TestCase):
    """``kept_prose`` is the one way a record keeps a model's own words."""

    def test_harmless_prose_is_kept_as_it_was_and_a_non_string_as_nothing(self) -> None:
        from agent_skill_improver.domain.redaction import kept_prose

        self.assertEqual(
            kept_prose("the skill never says when to stop"), "the skill never says when to stop"
        )
        self.assertEqual(kept_prose(None), "")
        self.assertEqual(kept_prose(["a"]), "")
        self.assertEqual(kept_prose("x" * 50, limit=50), "x" * 50, "fits: kept whole")

    def test_a_credential_is_masked_whatever_the_flag_says(self) -> None:
        from agent_skill_improver.domain.redaction import kept_prose

        key = "gh" + "p_" + "F" * 24
        text = kept_prose(f"I saw {key} in the transcript")
        self.assertNotIn(key, text)
        self.assertEqual(text, "I saw [REDACTED:github-token] in the transcript")

    def test_masking_runs_before_the_clip(self) -> None:
        from agent_skill_improver.domain.redaction import kept_prose

        key = "AK" + "IA" + "G" * 16
        # The cut at 17 falls inside the key (offsets 7 to 27): clipping first
        # would keep an unrecognisable fragment of it; masking first leaves a
        # (cut) placeholder and no fragment.
        text = kept_prose("prefix " + key + " and a long tail " + "t" * 100, limit=17)
        self.assertNotIn(key[:8], text, "no fragment of the key survives the cut")
        self.assertEqual(text, "prefix [REDACTED: ...[clipped]")
        whole = kept_prose("prefix " + key + " and a long tail " + "t" * 100, limit=40)
        self.assertEqual(whole, "prefix [REDACTED:aws-access-key-id] and  ...[clipped]")


class QuotedValueTests(unittest.TestCase):
    """``quoted_value`` is the one way a message quotes what a model sent."""

    def test_a_short_value_is_its_repr(self) -> None:
        from agent_skill_improver.domain.redaction import quoted_value

        self.assertEqual(quoted_value("abc"), "'abc'")
        self.assertEqual(quoted_value(None), "None")
        self.assertEqual(quoted_value(["a", 1]), "['a', 1]")

    def test_a_long_value_is_cut_and_the_cut_is_counted(self) -> None:
        from agent_skill_improver.domain.redaction import QUOTED_VALUE_CHARS, quoted_value

        text = quoted_value("v" * 1_000, limit=10)
        self.assertEqual(text, "'vvvvvvvvv... [992 more chars not quoted]")
        exact = quoted_value("w" * (QUOTED_VALUE_CHARS - 2))
        self.assertEqual(exact, repr("w" * (QUOTED_VALUE_CHARS - 2)), "fits: quoted whole")

    def test_a_credential_is_redacted_whole_before_the_cut(self) -> None:
        from agent_skill_improver.domain.redaction import quoted_value

        key = "AK" + "IA" + "C" * 16
        text = quoted_value("prefix " + key + " suffix", limit=36)
        self.assertNotIn(key, text)
        self.assertNotIn(key[:8], text, "no fragment of the key survives the cut either")
        self.assertIn("aws-access-key-id", text)
        self.assertIn("more chars not quoted", text, "the cut fell inside the suffix")
