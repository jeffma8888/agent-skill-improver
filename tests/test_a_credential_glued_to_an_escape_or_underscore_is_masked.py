"""A credential glued to an escape, an underscore or a non-ASCII letter is masked.

Measured before (3.11 and 3.13): every shape began at ``\\b``, which needs a
non-word character before a letter, so a token right after one went through
whole. ``quoted_value`` quotes a ``repr``, where a line break before a key is
``\\n`` and the ``n`` is a word character: a model value or a typed candidate
id holding one put the key in the refusal, the quarantine reason and the
scheduler log. The same held for JSON text (``\\n``, ``\\u00e9``), a URL
(``%20``), a terminal colour code (``ESC[32m``), ``MY_TOKEN_`` before a token
and ``DB_`` before a password assignment; 67 of 72 such samples were left as
they were, among them some tokens followed by ``_``.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.cli import RC_REFUSED
from agent_skill_improver.domain.redaction import (
    PLACEHOLDER,
    credential_labels,
    quoted_value,
    redact,
)
from agent_skill_improver.domain.rubric import Verdict
from tests.support import build_env, run_cli
from tests.test_skill_identity_and_payloads import deadline


def _j(*parts: str) -> str:
    return "".join(parts)


BODY = "aB3dE5gH7jK9mN1pQ3sT5vX7zA9cE1gI3kM5"

#: One token for every way a high-confidence shape can begin, by label.
TOKENS: tuple[tuple[str, str], ...] = (
    ("aws-access-key-id", _j("AK", "IA") + "ABCDEFGHIJKLMNOP"),
    ("aws-access-key-id", _j("AS", "IA") + "QRSTUVWXYZ234567"),
    ("github-token", _j("gh", "p_") + BODY),
    ("github-token", _j("gh", "o_") + BODY),
    ("github-token", _j("gh", "u_") + BODY),
    ("github-token", _j("gh", "s_") + BODY),
    ("github-token", _j("gh", "r_") + BODY),
    ("github-fine-grained-token", _j("github_", "pat_") + BODY),
    ("slack-token", _j("xo", "xb-") + "1234567890-abcdefghij"),
    ("slack-token", _j("xo", "xp-") + "1234567890-abcdefghij"),
    ("model-provider-key", _j("s", "k-") + BODY),
    ("google-api-key", _j("AI", "za") + BODY),
    ("stripe-secret-key", _j("s", "k_", "live_") + BODY),
    ("stripe-secret-key", _j("r", "k_", "test_") + BODY),
    ("npm-token", _j("n", "pm_") + BODY),
    ("huggingface-token", _j("h", "f_") + "aBcDeFgHiJkLmNoPqRsTuVwXyZaBcDeFgH"),
    ("jwt", _j("ey", "J") + "hbGciOiJIUzI1NiJ9." + _j("ey", "J") + "zdWIiOiIxMjM0In0.abcdefghij"),
    ("bearer-token", _j("Bea", "rer ") + BODY),
    ("bearer-token", _j("bea", "rer ") + BODY),
    ("bearer-token", _j("BEA", "RER ") + BODY),
)

#: Text that ends right where a credential begins, as it reaches the redactor.
GLUE: tuple[tuple[str, str], ...] = (
    ("a repr line break", "'line one\\n"),
    ("a repr tab", "'a\\t"),
    ("a carriage return escape", "x\\r"),
    ("a JSON backspace or form feed", "x\\b"),
    ("a form feed escape", "x\\f"),
    ("a vertical tab escape", "x\\v"),
    ("a repr hex escape", "'\\x1b"),
    ("a JSON unicode escape", '"\\u00e9'),
    ("a repr wide escape", "'\\U0001f600"),
    ("a URL percent-encoding", "q=a%20"),
    ("a terminal colour code", "\x1b[32m"),
    ("a terminal colour code with two codes", "\x1b[1;32m"),
    ("a colour code in JSON text", '"\\u001b[0m'),
    ("an underscore", "MY_TOKEN_"),
    ("a non-ASCII letter", "\u00e9"),
)

#: Every key name an assignment shape knows, as each may begin.
ASSIGNED_NAMES: tuple[str, ...] = (
    _j("pass", "word"),
    _j("pass", "wd"),
    "secret",
    "api_key",
    "api-key",
    "apikey",
    "access_token",
    "auth_token",
    "client_secret",
    "private_token",
    "token",
)


class AGluedCredentialIsMaskedTests(unittest.TestCase):
    def assertNoPieceOf(self, token: str, text: str) -> None:
        """Masked whole: no eight characters of the token's secret part are left."""
        secret = token.split(" ")[-1]
        for i in range(len(secret) - 7):
            self.assertNotIn(secret[i : i + 8], text)

    def test_a_token_glued_to_an_escape_an_underscore_or_a_letter_is_masked(self):
        for label, token in TOKENS:
            for how, before in GLUE:
                with self.subTest(label=label, token=token[:7], before=how):
                    text = f"{before}{token} end"
                    result = redact(text)
                    self.assertNoPieceOf(token, result.text)
                    self.assertEqual(result.counts.get(label), 1, result.text)
                    self.assertIn(before + PLACEHOLDER.format(label=label), result.text)
                    self.assertIn(label, credential_labels(text))

    def test_a_token_followed_by_an_underscore_or_a_non_ascii_letter_is_masked(self):
        for label, token in TOKENS:
            for after in ("_prod", "\u00e9", "\\n"):
                with self.subTest(label=label, token=token[:7], after=after):
                    result = redact(f"x {token}{after}")
                    self.assertNoPieceOf(token, result.text)
                    self.assertEqual(result.counts.get(label), 1, result.text)

    def test_an_assignment_whose_name_ends_a_longer_name_or_follows_an_escape_is_masked(self):
        value = "hunter2-x9q"
        for name in ASSIGNED_NAMES:
            for before in ("DB_", "export MY_", "x\\n", '"\\u00e9', "q=a%20"):
                for spelled in (name, name.upper()):
                    with self.subTest(name=spelled, before=before):
                        text = f"{before}{spelled}={value} next"
                        result = redact(text)
                        self.assertNotIn(value, result.text)
                        self.assertEqual(
                            result.text,
                            f"{before}{spelled}="
                            + PLACEHOLDER.format(label="secret-assignment")
                            + " next",
                        )
        for before in ("MY_", "x\\n"):
            for name in (_j("aws_secr", "et_access_key"), _j("secr", "et_access_key")):
                with self.subTest(name=name, before=before):
                    result = redact(f"{before}{name.upper()}: {value}")
                    self.assertNotIn(value, result.text)
                    self.assertIn("aws-secret-assignment", result.counts)

    def test_a_snake_case_name_ending_in_a_key_name_is_masked_as_the_key_name_is(self):
        """The cost of ``DB_`` before a password key: any name that ends in a key
        name the assignment shape knows now counts as that key name, so code
        that assigns one loses its value, as a bare ``token = ...`` already
        did. The assignment shape replaces and never refuses."""
        for name in ("csrf_token", "next_page_token", "self._api_key", "max_token"):
            with self.subTest(name=name):
                bare = name.rsplit("_", 1)[-1] if name != "self._api_key" else "api_key"
                for spelled in (name, bare):
                    text = f"{spelled} = request.form.get('csrf')"
                    self.assertEqual(
                        redact(text).text,
                        f"{spelled} = " + PLACEHOLDER.format(label="secret-assignment"),
                    )
                self.assertEqual(credential_labels(f"{name} = request.form.get('csrf')"), ())

    def test_a_url_parameter_after_an_escaped_ampersand_is_masked(self):
        """An ``&`` written ``\\u0026`` in JSON or ``%26`` in a URL begins a URL
        parameter as ``&`` does (``_URL_PARAM``), so its value is masked as a
        URL credential's, up to the whitespace after it."""
        value = "Zq8vLm3pXr7tWk2s"
        names = (
            "token",
            "access_token",
            "auth_token",
            "api_key",
            "apikey",
            _j("pass", "word"),
            "secret",
            "client_secret",
        )
        for name in names:
            for amp in ("\\u0026", "%26"):
                with self.subTest(name=name, amp=amp):
                    url = f"https://h.example/cb?a=1{amp}{name}={value}"
                    result = redact(url + " next")
                    self.assertNotIn(value, result.text)
                    self.assertEqual(
                        result.text,
                        f"https://h.example/cb?a=1{amp}{name}="
                        + PLACEHOLDER.format(label="url-credential")
                        + " next",
                    )

    def test_json_text_holding_a_key_after_a_line_break_is_masked(self):
        """A tool's JSON output pasted into a conversation keeps its escapes."""
        token = _j("gh", "p_") + BODY
        text = json.dumps({"stdout": f"created\n{token}\n", "env": f"\u00e9{token}"})
        result = redact(text)
        self.assertNotIn(token, result.text)
        self.assertEqual(result.counts.get("github-token"), 2)
        self.assertEqual(json.loads(result.text)["stdout"], "created\n[REDACTED:github-token]\n")

    def test_text_full_of_underscore_and_escape_jwt_starts_is_masked_in_linear_time(self):
        """``_eyJ`` and ``\\neyJ`` are starts now; a start with no token after it
        still fails where its run ends, so the scan resumes past the run."""
        header = _j("e", "yJ")
        for glue in ("_", "\\n", "%2F"):
            with self.subTest(glue=glue):
                text = (glue + header) * 65_536
                with deadline(5.0):
                    result = redact(text)
                    labels = credential_labels(text)
                self.assertEqual(result.text, text, "there is no token here, so nothing is masked")
                self.assertEqual(labels, ())


class AQuotedValueMasksAKeyAfterALineBreakTests(unittest.TestCase):
    def test_a_model_value_with_a_line_break_before_a_key_is_quoted_masked(self):
        for label, token in TOKENS:
            with self.subTest(label=label, token=token[:7]):
                shown = quoted_value(f"see below\n{token}")
                self.assertNotIn(token.split(" ")[-1], shown)
                self.assertIn("\\n" + PLACEHOLDER.format(label=label), shown)

    def test_an_unknown_verdict_holding_a_key_after_a_line_break_is_refused_masked(self):
        token = _j("AK", "IA") + "ABCDEFGHIJKLMNOP"
        with self.assertRaises(ValueError) as ctx:
            Verdict.parse(f"PASS\n{token}")
        message = str(ctx.exception)
        self.assertNotIn(token, message)
        self.assertIn("'PASS\\n[REDACTED:aws-access-key-id]'", message)

    def test_a_candidate_id_typed_with_a_line_break_before_a_key_is_masked(self):
        token = _j("gh", "p_") + "C" * 36
        with TemporaryDirectory() as tmp:
            cfg = ["--config", str(build_env(Path(tmp).resolve()))]
            proc = run_cli([*cfg, "review", f"x\n{token}"])
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
        self.assertNotIn(token, proc.stderr)
        self.assertIn("invalid candidate id 'x\\n[REDACTED:github-token]'", proc.stderr)


class TheControlGluedToALetterOrDigitIsNotATokenTests(unittest.TestCase):
    def test_the_control_a_shape_glued_to_an_ascii_letter_or_digit_is_left_alone(self):
        for label, token in TOKENS:
            # The last letter or digit of an escape, without the rest of it.
            for before in ("x", "9", "Z", "m", "x1b", "u00e9", "%2"):
                with self.subTest(label=label, token=token[:7], before=before):
                    text = f"{before}{token} end"
                    if label == "bearer-token":
                        text = f"{before}{token}"
                    result = redact(text)
                    self.assertNotIn(label, result.counts, result.text)
                    self.assertNotIn(label, credential_labels(text))

    def test_the_control_prose_and_longer_names_are_left_alone(self):
        for text in (
            "The token budget is 4096 and the password policy page is short.",
            "risk-assessment-for-the-quarterly-planning-cycle",
            "task-runner-abcdefghijklmnopqrstuvwxyz",
            "max_tokens=409600 input_tokens: 123456",
            _j("pass", "word_hash=abcdefgh1234 mytoken=abcdefgh1234"),
            "the 5m mark, 100% done, a \\n in the text, ESC[0m reset",
            _j("AK", "IA") + "ABCDEFGHIJKLMNOPqrs is too long to be a key id",
        ):
            with self.subTest(text=text):
                self.assertEqual(redact(text).total, 0, redact(text).text)
                self.assertEqual(credential_labels(text), ())


if __name__ == "__main__":
    unittest.main()
