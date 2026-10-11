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

from agent_skill_improver.domain.redaction import redact

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


if __name__ == "__main__":
    unittest.main()
