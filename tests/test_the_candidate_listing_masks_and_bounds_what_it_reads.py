"""The candidate listing masks what it reads from a record and cuts what no real one could be.

Measured before (3.11 and 3.13, the same on both), with a real validated
candidate whose record was then edited: a ``ghp_`` token in the skill id was
printed whole by ``candidates`` and by ``candidates --json``, one in the status
by both under ``--all``, and one in the canonical path or the creation time by
``candidates --json``; a megabyte in any of the four made a listing of about a
megabyte (1,000,098 bytes for the skill id). ``candidates`` lists records no
integrity check has read, and showed these four as they were written.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_skill_improver.domain import redaction
from agent_skill_improver.domain.redaction import (
    LISTED_PATH_CHARS,
    LISTED_SKILL_ID_CHARS,
    QUOTED_VALUE_CHARS,
    listed_value,
    redact,
)
from tests.support import build_env, run_cli

TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
MASK = "[REDACTED:github-token]"
FORGED = "config error: forged"
SEED = "listing-masks-and-bounds"
LIMITS = (QUOTED_VALUE_CHARS, LISTED_SKILL_ID_CHARS, LISTED_PATH_CHARS)


def _j(*parts: str) -> str:
    return "".join(parts)


def _fits_once_masked(limit: int, *, head: str = "x", tail: str = "") -> str:
    """Text longer than ``limit`` as written and exactly ``limit`` long once masked."""
    room = limit - len(" [REDACTED:github-token]") - len(tail)
    return head * room + " " + TOKEN + tail


#: One credential of each shape ``redact`` knows, each built from pieces so no
#: scanner reads one in this file; the private key and the bearer token span a
#: line break. ``test_the_listing_text_set_holds_every_shape_redact_knows`` keeps
#: this list in step with ``domain/redaction.py``.
SHAPED = (
    _j("key ", "AKIA", "ABCDEFGHIJKLMNOP"),
    TOKEN,
    _j("github_", "pat_", "11ABCDEFG0123456789abcdefgh"),
    _j("xox", "b-", "123456789012-abcdefghijkl"),
    _j("sk-", "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6"),
    _j("AI", "za", "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"),
    _j("ey", "JhbGciOiJIUzI1NiJ9", ".", "eyJzdWIiOiIxMjM0NTY3ODkwIn0", ".", "c2lnbmF0dXJlc2ln"),
    _j("Authorization: Bear", "er\n", "abcdefghijklmnop0123456789"),
    _j(
        "-----BEGIN ",
        "PRIVATE KEY-----\n",
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n",
        "-----END ",
        "PRIVATE KEY-----",
    ),
    _j("see https://example.test/x?", "token=", "abcdefgh12345678"),
    _j("aws_secret_", "access_key = ", "wJalrXUtnFEMI/K7MDENG/", "bPxRfiCYEXAMPLEKEY"),
    _j("pass", "word: ", "correct-horse-battery-staple"),
)
#: The shortest text each of six shapes masks.
SHORTEST = (
    _j("tok", "en=", "abcdef"),
    _j("&k", "ey=", "abcdefgh"),
    _j("?s", "ig=", "abcdefgh"),
    _j("xox", "b-", "abcdefghij"),
    _j("secret_", "access_key=", "a"),
    _j("AKIA", "ABCDEFGHIJKLMNOP"),
)
#: Text of the kinds a listed field can hold: empty, blank, plain, a real skill
#: id with spaces and non-ASCII letters, each shape above alone and inside a
#: path, a credential on a later line, after a carriage return or a lone one,
#: among control characters, decomposed or astral letters, beside a mask, beside
#: a second shape, and at length.
VARIED = (
    "",
    "   ",
    "tdd",
    "my skill \u00fc-\u65e5\u672c",
    *SHAPED,
    *SHORTEST,
    *(f"/skills/{text}/SKILL.md" for text in SHAPED),
    f"first line\nsecond line {TOKEN}\nthird line",
    f"first line\r\nsecond line {TOKEN}\r\n",
    f"a lone\rcarriage return {TOKEN}",
    f"\x00\x07\t\x1b[2K\x7f\x85\u200b\u202e\u2028{TOKEN}\x1b[0m",
    f"cafe\u0301 \U0001f600 \u00a0{TOKEN}",
    f"{MASK} {TOKEN}",
    f"{TOKEN} and {SHAPED[4]}",
    *(_fits_once_masked(limit) for limit in LIMITS),
    "x" * (QUOTED_VALUE_CHARS - 11) + " " + TOKEN,
    "x" * (LISTED_SKILL_ID_CHARS - 11) + " " + TOKEN,
    "x" * (LISTED_PATH_CHARS - 11) + " " + TOKEN,
    "word " * 2_000 + TOKEN,
)


def _expected(text: str, limit: int) -> str:
    """What the listing must show of ``text``: ``redact``'s own mask, then the cut."""
    masked = redact(text).text
    if len(masked) <= limit:
        return masked
    return masked[:limit] + f"... [{len(masked) - limit} more chars not shown]"


def _validated(base: Path) -> tuple[list[str], str, Path]:
    cfg = ["--config", str(build_env(base, dry_run=False, seed=SEED))]
    proc = run_cli([*cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
    cid = next(
        c["candidate_id"]
        for c in json.loads(proc.stdout)["candidates"]
        if c.get("status") == "validated"
    )
    return cfg, cid, base / "state" / "candidates" / f"{cid}.json"


def _edit(record: Path, **fields: str) -> dict[str, Any]:
    """Write ``fields`` into the record (``skill_id`` and ``canonical_path`` under
    ``target``) and return the record as it now stands."""
    raw = json.loads(record.read_text(encoding="utf-8"))
    for name, value in fields.items():
        if name in ("skill_id", "canonical_path"):
            raw["target"][name] = value
        else:
            raw[name] = value
    record.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return raw


def _listing(cfg: list[str], *flags: str) -> tuple[str, dict[str, Any]]:
    text = run_cli([*cfg, "candidates", *flags])
    as_json = run_cli([*cfg, "--json", "candidates", *flags])
    assert text.returncode == 0 and as_json.returncode == 0, (text.stderr, as_json.stderr)
    return text.stdout, json.loads(as_json.stdout)


def _line(cid: str, raw: dict[str, Any], status: str, skill_id: str) -> str:
    closed = " (closed)" if raw["status"] != "validated" else ""
    patch = raw["patch"]
    return (
        f"{cid}  {status:<12} {skill_id:<28} gate=pass{closed} "
        f"+{patch['added_lines']}/-{patch['removed_lines']}"
    )


def _row(cid: str, raw: dict[str, Any], **shown: str) -> dict[str, Any]:
    row = {
        "id": cid,
        "status": raw["status"],
        "skill_id": raw["target"]["skill_id"],
        "canonical_path": raw["target"]["canonical_path"],
        "gate_passed": True,
        "added_lines": raw["patch"]["added_lines"],
        "removed_lines": raw["patch"]["removed_lines"],
        "created_at": raw["created_at"],
    }
    row.update(shown)
    return row


class ListedValueTests(unittest.TestCase):
    """``listed_value``: ``redact``'s mask, then a cut that says what it left out."""

    def test_the_listing_text_set_holds_every_shape_redact_knows(self):
        found = {label for text in VARIED for label in redact(text).counts}
        self.assertEqual(found, {shape.label for shape in redaction._SHAPES})
        for text in (*SHAPED, *SHORTEST):
            with self.subTest(text=text[:20]):
                self.assertEqual(redact(text).total, 1)

    def test_each_kind_of_text_is_shown_as_redact_masks_it_then_cut(self):
        for limit in LIMITS:
            for text in VARIED:
                with self.subTest(limit=limit, text=text[:30]):
                    self.assertEqual(listed_value(text, limit=limit), _expected(text, limit))
        for text in VARIED:
            with self.subTest(default=True, text=text[:30]):
                self.assertEqual(listed_value(text), _expected(text, QUOTED_VALUE_CHARS))

    def test_no_shape_survives_and_a_key_the_cut_would_split_is_masked_whole(self):
        for text in SHAPED:
            with self.subTest(text=text[:20]):
                self.assertNotEqual(listed_value(text), text)
                self.assertIn("[REDACTED:", listed_value(text))
        for limit in LIMITS:
            with self.subTest(limit=limit):
                # A space before the token: redaction does not mask one glued
                # to a letter, so there would be nothing to keep whole.
                shown = listed_value("x" * (limit - 11) + " " + TOKEN, limit=limit)
                self.assertNotIn("ghp_", shown)
                self.assertEqual(
                    shown,
                    "x" * (limit - 11)
                    + " "
                    + MASK[:10]
                    + f"... [{len(MASK) - 10} more chars not shown]",
                )

    def test_a_value_longer_than_its_limit_only_until_masked_is_shown_whole(self):
        for limit in LIMITS:
            with self.subTest(limit=limit):
                text = _fits_once_masked(limit)
                self.assertGreater(len(text), limit)
                self.assertEqual(len(redact(text).text), limit)
                self.assertEqual(listed_value(text, limit=limit), redact(text).text)

    def test_a_value_at_the_limit_is_whole_and_one_past_it_is_cut_with_its_count(self):
        for limit in LIMITS:
            with self.subTest(limit=limit):
                self.assertEqual(listed_value("a" * limit, limit=limit), "a" * limit)
                self.assertEqual(
                    listed_value("a" * (limit + 1), limit=limit),
                    "a" * limit + "... [1 more chars not shown]",
                )
                self.assertEqual(
                    listed_value("a" * (limit + 1000), limit=limit),
                    "a" * limit + "... [1000 more chars not shown]",
                )
        # The limit and the count are in characters, not UTF-8 bytes: a letter
        # outside ASCII counts once.
        for limit in LIMITS:
            for letter in ("\u00fc", "\u65e5", "\U0001f600"):
                with self.subTest(limit=limit, letter=letter):
                    self.assertEqual(listed_value(letter * limit, limit=limit), letter * limit)
                    self.assertEqual(
                        listed_value(letter * (limit + 1), limit=limit),
                        letter * limit + "... [1 more chars not shown]",
                    )
        self.assertEqual(listed_value("a" * QUOTED_VALUE_CHARS), "a" * QUOTED_VALUE_CHARS)
        self.assertEqual(
            listed_value("a" * (QUOTED_VALUE_CHARS + 1)),
            "a" * QUOTED_VALUE_CHARS + "... [1 more chars not shown]",
        )

    def test_a_plain_value_is_shown_exactly_as_written_with_no_quotes(self):
        for text in (
            "tdd",
            "validated",
            "2026-02-01T00:00:00Z",
            "my skill \u00fc-\u65e5\u672c",
            '/skills/it\'s a "skill"/SKILL.md',
            "",
            "line\nbreak",
        ):
            with self.subTest(text=text):
                self.assertEqual(listed_value(text), text)
                self.assertEqual(listed_value(text, limit=LISTED_PATH_CHARS), text)

    def test_the_limits_reach_what_this_platform_allows_a_name_and_a_path(self):
        with TemporaryDirectory() as d:
            self.assertGreaterEqual(LISTED_SKILL_ID_CHARS, os.pathconf(d, "PC_NAME_MAX"))
            self.assertGreaterEqual(LISTED_PATH_CHARS, os.pathconf(d, "PC_PATH_MAX"))
        self.assertGreaterEqual(LISTED_SKILL_ID_CHARS, 255)
        self.assertGreaterEqual(LISTED_PATH_CHARS, 4096)


class TheListingTests(unittest.TestCase):
    """``candidates`` and ``candidates --json`` on a real record, edited per test."""

    def test_a_credential_in_the_skill_id_path_or_creation_time_is_masked_when_listed(self):
        planted = {
            "skill_id": f"tdd {TOKEN}",
            "canonical_path": f"/skills/{SHAPED[4]}/SKILL.md",
            "created_at": f"2026-02-01T00:00:00Z {SHAPED[0]}",
        }
        with TemporaryDirectory() as d:
            cfg, cid, record = _validated(Path(d).resolve())
            raw = _edit(record, **planted)
            text, as_json = _listing(cfg)
        shown = {
            "skill_id": _expected(planted["skill_id"], LISTED_SKILL_ID_CHARS),
            "canonical_path": _expected(planted["canonical_path"], LISTED_PATH_CHARS),
            "created_at": _expected(planted["created_at"], QUOTED_VALUE_CHARS),
        }
        self.assertEqual(shown["skill_id"], f"tdd {MASK}")
        self.assertEqual(as_json["candidates"], [_row(cid, raw, **shown)])
        self.assertEqual(text.splitlines()[0], _line(cid, raw, raw["status"], shown["skill_id"]))
        for secret in (TOKEN, SHAPED[4], SHAPED[0][4:]):
            self.assertNotIn(secret, text + json.dumps(as_json))

    def test_a_credential_in_the_status_is_masked_when_all_records_are_listed(self):
        planted = f"{SHAPED[3]} {TOKEN}"
        with TemporaryDirectory() as d:
            cfg, cid, record = _validated(Path(d).resolve())
            raw = _edit(record, status=planted)
            text, as_json = _listing(cfg, "--all")
            open_text, open_json = _listing(cfg)
        shown = _expected(planted, QUOTED_VALUE_CHARS)
        self.assertNotIn("ghp_", shown)
        self.assertEqual(as_json["candidates"], [_row(cid, raw, status=shown)])
        self.assertEqual(text.splitlines()[0], _line(cid, raw, shown, raw["target"]["skill_id"]))
        for secret in (TOKEN, SHAPED[3]):
            self.assertNotIn(secret, text + json.dumps(as_json) + open_text)
        self.assertEqual(open_json["candidates"], [])
        self.assertEqual(open_json["closed_not_listed"], 1)

    def test_a_megabyte_in_each_listed_field_is_cut(self):
        planted = {
            "skill_id": "s" * 1_000_000,
            "canonical_path": "/" + "p" * 1_000_000 + "/SKILL.md",
            "status": "t" * 1_000_000,
            "created_at": "c" * 1_000_000,
        }
        with TemporaryDirectory() as d:
            cfg, cid, record = _validated(Path(d).resolve())
            raw = _edit(record, **planted)
            text, as_json = _listing(cfg, "--all")
        shown = {
            "skill_id": _expected(planted["skill_id"], LISTED_SKILL_ID_CHARS),
            "canonical_path": _expected(planted["canonical_path"], LISTED_PATH_CHARS),
            "status": _expected(planted["status"], QUOTED_VALUE_CHARS),
            "created_at": _expected(planted["created_at"], QUOTED_VALUE_CHARS),
        }
        self.assertTrue(shown["skill_id"].endswith("... [999745 more chars not shown]"))
        self.assertEqual(as_json["candidates"], [_row(cid, raw, **shown)])
        self.assertEqual(text.splitlines()[0], _line(cid, raw, shown["status"], shown["skill_id"]))
        self.assertLess(len(text.encode()), 1_000)
        self.assertLess(len(json.dumps(as_json).encode()), 6_000)

    def test_each_field_is_whole_at_its_limit_and_cut_one_past_it(self):
        sizes = {
            "skill_id": LISTED_SKILL_ID_CHARS,
            "canonical_path": LISTED_PATH_CHARS,
            "status": QUOTED_VALUE_CHARS,
            "created_at": QUOTED_VALUE_CHARS,
        }

        def value(field: str, length: int, letter: str) -> str:
            if field == "canonical_path":
                return "/" + (letter or "p") * (length - len("//SKILL.md")) + "/SKILL.md"
            return (letter or field[0]) * length

        with TemporaryDirectory() as d:
            cfg, cid, record = _validated(Path(d).resolve())
            for letter, past in ((l, p) for l in ("", "\u65e5") for p in (0, 1)):
                planted = {
                    field: value(field, size + past, letter) for field, size in sizes.items()
                }
                for field, size in sizes.items():
                    self.assertEqual(len(planted[field]), size + past)
                raw = _edit(record, **planted)
                text, as_json = _listing(cfg, "--all")
                with self.subTest(letter=letter, past=past):
                    shown = {
                        field: (
                            planted[field]
                            if not past
                            else planted[field][:size] + "... [1 more chars not shown]"
                        )
                        for field, size in sizes.items()
                    }
                    self.assertEqual(as_json["candidates"], [_row(cid, raw, **shown)])
                    self.assertEqual(
                        text.splitlines()[0],
                        _line(cid, raw, shown["status"], shown["skill_id"]),
                    )

    def test_each_field_too_long_only_until_masked_is_listed_whole(self):
        planted = {
            "skill_id": _fits_once_masked(LISTED_SKILL_ID_CHARS, head="s"),
            "canonical_path": "/"
            + _fits_once_masked(LISTED_PATH_CHARS - 1, head="p", tail="/SKILL.md"),
            "status": _fits_once_masked(QUOTED_VALUE_CHARS, head="t"),
            "created_at": _fits_once_masked(QUOTED_VALUE_CHARS, head="c"),
        }
        limits = {
            "skill_id": LISTED_SKILL_ID_CHARS,
            "canonical_path": LISTED_PATH_CHARS,
            "status": QUOTED_VALUE_CHARS,
            "created_at": QUOTED_VALUE_CHARS,
        }
        shown = {field: redact(value).text for field, value in planted.items()}
        for field, value in planted.items():
            self.assertGreater(len(value), limits[field])
            self.assertEqual(len(shown[field]), limits[field])
        with TemporaryDirectory() as d:
            cfg, cid, record = _validated(Path(d).resolve())
            raw = _edit(record, **planted)
            text, as_json = _listing(cfg, "--all")
        self.assertEqual(as_json["candidates"], [_row(cid, raw, **shown)])
        self.assertEqual(text.splitlines()[0], _line(cid, raw, shown["status"], shown["skill_id"]))
        self.assertNotIn("more chars not shown", text + json.dumps(as_json))

    def test_a_line_break_in_a_listed_field_stays_on_its_line(self):
        planted = {"skill_id": f"tdd\n{FORGED}", "status": f"validated\n{FORGED}"}
        with TemporaryDirectory() as d:
            cfg, cid, record = _validated(Path(d).resolve())
            raw = _edit(record, **planted)
            text, as_json = _listing(cfg, "--all")
        self.assertFalse([line for line in text.splitlines() if line.startswith(FORGED)])
        self.assertEqual(as_json["candidates"], [_row(cid, raw)])
        self.assertIn("tdd\\n" + FORGED, text.splitlines()[0])

    def test_the_control_an_unedited_record_and_a_real_odd_name_list_as_written(self):
        with TemporaryDirectory() as d:
            cfg, cid, record = _validated(Path(d).resolve())
            raw = json.loads(record.read_text(encoding="utf-8"))
            text, as_json = _listing(cfg)
            self.assertEqual(as_json["candidates"], [_row(cid, raw)])
            self.assertEqual(
                text.splitlines(),
                [
                    _line(cid, raw, "validated", "tdd"),
                    "1 open candidate(s); 0 unreadable record(s)",
                ],
            )
            odd = _edit(record, skill_id="my skill \u00fc-\u65e5\u672c")
            text, as_json = _listing(cfg)
        self.assertEqual(as_json["candidates"], [_row(cid, odd)])
        self.assertEqual(
            text.splitlines()[0], _line(cid, odd, "validated", "my skill \u00fc-\u65e5\u672c")
        )


if __name__ == "__main__":
    unittest.main()
