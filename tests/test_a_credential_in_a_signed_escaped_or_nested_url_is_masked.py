"""A credential in a signed, escaped or nested URL is masked, and only its value.

Measured before (3.11 and 3.13): the URL parameter shape began at ``?`` or
``&`` right before a name it knows and ended its value at ``&``, ``#`` or
whitespace. So a signed cloud-store URL kept its signature (``X-Amz-`` and
``X-Goog-`` prefix the name), a parameter after ``&`` written ``\\u0026`` in
JSON text or ``%26`` in a URL nested in another was kept whenever its name was
``sig``, ``signature`` or ``key`` or its ``=`` was ``%3D`` too, and a value that
was masked took the rest of its string with it: the closing quote, the JSON
after it, every later parameter written with an escaped ``&``. 9 of 11 such
URLs kept their secret, and 3 of 5 lost text after the value that was not one.

A value now ends only where the URL does. A character or an escape a URL can
hold (``\\/``, ``\\u002B``, the quote in ``it's``) stays part of it: the
assignment shape cannot mask past a placeholder, so a value cut short there
would leave the rest of the secret in the text.
"""

from __future__ import annotations

import json
import unittest

from agent_skill_improver.domain.redaction import PLACEHOLDER, credential_labels, redact

URL = PLACEHOLDER.format(label="url-credential")
VALUE = "Zq8vLm3pXr7tWk2sQ1"
SIGNATURE = "3b1f" * 16
FIRST, REST = "Qm4tRz8wKp", "Yh2nVc6xLs"


def _j(*parts: str) -> str:
    return "".join(parts)


class ASignedUrlTests(unittest.TestCase):
    def test_a_signed_cloud_store_url_has_its_signature_and_session_token_masked(self):
        for name in (
            "X-Amz-Signature",
            "x-amz-signature",
            "X-Amz-Security-Token",
            "X-Goog-Signature",
            "X-GOOG-SIGNATURE",
        ):
            with self.subTest(name=name):
                text = f"https://b.store.example/k?X-Amz-Date=20261023T000000Z&{name}={SIGNATURE}"
                self.assertEqual(
                    redact(text).text,
                    f"https://b.store.example/k?X-Amz-Date=20261023T000000Z&{name}={URL}",
                )

    def test_the_parameters_beside_the_signature_are_kept(self):
        text = (
            "https://b.store.example/k?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            f"&X-Amz-Signature={SIGNATURE}&X-Amz-Expires=3600&X-Amz-SignedHeaders=host"
        )
        self.assertEqual(
            redact(text).text,
            "https://b.store.example/k?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            f"&X-Amz-Signature={URL}&X-Amz-Expires=3600&X-Amz-SignedHeaders=host",
        )


class AnEscapedOrNestedUrlTests(unittest.TestCase):
    def test_a_parameter_after_an_ampersand_written_as_a_json_escape_is_masked(self):
        for name in ("sig", "signature", "key", "X-Amz-Signature", "token"):
            with self.subTest(name=name):
                url = f"https://a.example/c?sv=1&{name}={VALUE}&page=2"
                text = json.dumps({"url": url}).replace("&", "\\u0026")
                result = redact(text)
                self.assertNotIn(VALUE, result.text)
                self.assertEqual(
                    json.loads(result.text)["url"],
                    f"https://a.example/c?sv=1&{name}={URL}&page=2",
                )

    def test_a_parameter_of_a_url_nested_in_another_is_masked_and_its_neighbours_kept(self):
        for name in ("token", "sig", "access_token", _j("pass", "word")):
            for first in (True, False):
                with self.subTest(name=name, first=first):
                    inner = (
                        f"%3F{name}%3D{VALUE}%26page%3D2"
                        if first
                        else (f"%3Fx%3D1%26{name}%3D{VALUE}%26page%3D2")
                    )
                    text = f"https://login.example/a?next=https%3A%2F%2Fapp.example%2Fcb{inner}"
                    result = redact(text)
                    self.assertNotIn(VALUE, result.text)
                    self.assertEqual(
                        result.text, text.replace(VALUE, URL), "only the value is replaced"
                    )

    def test_a_value_ends_where_its_string_or_line_ends(self):
        url = "https://h.example/cb?sig=" + VALUE
        cases = (
            '{"u":"https://h.example/cb?token=' + VALUE + '","n":"keep"}',
            "for u in ['https://h.example/cb?token=" + VALUE + "']: keep",
            "'https://h.example/cb?token=" + VALUE + "' keep",
            "see 'https://h.example/cb?token=" + VALUE + "'",
            "<a href='https://h.example/cb?token=" + VALUE + "'>keep</a>",
            '{"r":"\'https://h.example/cb?token=' + VALUE + '\'","n":"keep"}',
            '"https://h.example/cb?sig=' + VALUE + '\\" keep',
            json.dumps({"x": json.dumps({"m": json.dumps({"u": url, "n": "keep"})})}),
            "'https://h.example/cb?token=" + VALUE + "\\nnext line'",
            '"https://h.example/cb?key=' + VALUE + '\\u0022 keep"',
            '"https://h.example/cb?key=' + VALUE + '\\u000Anext line"',
        )
        for i, text in enumerate(cases):
            with self.subTest(case=i):
                self.assertEqual(redact(text).text, text.replace(VALUE, URL))


class AValueIsNotCutShortTests(unittest.TestCase):
    def test_an_escape_a_url_can_hold_is_part_of_the_value(self):
        # PHP's json_encode writes every / as \/, so a base64 session token or
        # signature carries them; .NET's encoder writes + as \u002B.
        token = "IQoJb3JpZ2luX2VjEJr" + "\\/" * 5 + "wEaCXVzLWVhc3QtMSJH"
        text = '{"u":"https:\\/\\/b.example\\/k?X-Amz-Security-Token=' + token + '"}'
        self.assertEqual(
            redact(text).text,
            '{"u":"https:\\/\\/b.example\\/k?X-Amz-Security-Token=' + URL + '"}',
        )
        mids = ("\\/", "\\\\\\/", "\\u002B", "\\u002b", "\\\\u002B", "\\u002F", "\\u0027")
        mids += ("\\'", "\\U0001F600", "\\x", "\\\\d", "\\u005c", "\\T")
        for mid in mids:
            for name in ("token", "sig", "X-Amz-Security-Token", "X-Goog-Signature"):
                with self.subTest(mid=mid, name=name):
                    head = '{"u":"https:\\/\\/h.example\\/cb?' + name + "="
                    text = head + FIRST + mid + REST + '\\u0026page=2"}'
                    self.assertEqual(redact(text).text, head + URL + '\\u0026page=2"}')

    def test_a_quote_or_an_encoded_ampersand_a_url_can_hold_is_part_of_the_value(self):
        for mid in ("'", "'s", "')", "',", "'\\''", "%26", "%26x", "%26%26", "%26-"):
            for name in ("sig", "key", "token"):
                with self.subTest(mid=mid, name=name):
                    text = f"https://h.example/cb?{name}={FIRST}{mid}{REST}&page=2"
                    self.assertEqual(redact(text).text, f"https://h.example/cb?{name}={URL}&page=2")
        text = "open('https://h.example/cb?token=" + VALUE + "') then keep"
        self.assertEqual(
            redact(text).text,
            "open('https://h.example/cb?token=" + URL + " then keep",
            "a quote a URL could hold goes with the value, and the ) after it",
        )


class TheControlOrdinaryUrlsAreLeftAloneTests(unittest.TestCase):
    def test_the_control_ordinary_query_parameters_are_left_alone(self):
        for text in (
            "see https://example.com/path?page=2&sort=desc for more",
            "https://h.example/x?keyboard=qwertyuiop&signatures=abcdefghij",
            "https://h.example/x?X-Amz-Date=20261023T000000Z&X-Amz-Expires=3600",
            "https://l.example/a?n=https%3A%2F%2Fa%2Fcb%3Fpage%3D2%26sort%3Ddescending",
            '{"u":"https://a.example/c?sv=1\\u0026page=2\\u0026sort=descending"}',
            "a 50% signature rate, the key=value form, and ?q=search terms",
        ):
            with self.subTest(text=text[:40]):
                self.assertEqual(redact(text).total, 0, redact(text).text)

    def test_the_control_a_short_value_is_left_alone_and_none_drives_a_refusal(self):
        for text in (
            "https://h.example/cb?token=short",
            "https://h.example/cb?x=1%26sig%3Dshort",
            '{"u":"https://h.example/cb?x=1\\u0026key=short"}',
        ):
            with self.subTest(text=text[:40]):
                self.assertEqual(redact(text).total, 0)
        signed = f"https://b.example/k?X-Amz-Signature={SIGNATURE}"
        self.assertEqual(credential_labels(signed), ())


if __name__ == "__main__":
    unittest.main()
