"""The candidate loader keeps the improver's prose and the notes masked on read.

Measured before (3.11 and 3.13, the same on both), with a real validated
candidate whose record was then edited: a ``ghp_`` token appended to the
hypothesis, the intended change or the rollback note, or added as a risk or a
note, was printed whole by ``review`` and by ``review --json``, each exiting 0,
and ``apply`` wrote it back into the record it closed. The proposal parser
masks the first four when the tool writes them; the loader read all five with
``str()``.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_skill_improver.domain.candidate import (
    STATUS_APPLIED,
    STATUS_BLOCKED_RUN,
    STATUS_GATE_FAILED,
    STATUS_PROPOSED,
    STATUS_ROLLED_BACK,
    STATUS_VALIDATED,
    Candidate,
)
from agent_skill_improver.app.daily import BLOCKED_RUN_NOTE
from agent_skill_improver.app.promote import FINISHING_NOTE
from agent_skill_improver.domain import redaction
from agent_skill_improver.domain.redaction import redact
from tests.support import build_env, run_cli

TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
MASK = "[REDACTED:github-token]"
SEED = "loader-keeps-prose-masked"
PROSE = ("hypothesis", "intended_change", "risks", "rollback", "notes")


def _j(*parts: str) -> str:
    return "".join(parts)


#: One credential of each shape ``redact`` knows, each built from pieces so no
#: scanner reads one in this file; the private key and the bearer token span a
#: line break. ``test_the_varied_text_holds_every_shape_redact_knows`` keeps
#: this list in step with ``domain/redaction.py``.
SHAPED = (
    _j("key ", "AKIA", "ABCDEFGHIJKLMNOP"),
    TOKEN,
    _j("github_", "pat_", "11ABCDEFG0123456789abcdefgh"),
    _j("xox", "b-", "123456789012-abcdefghijkl"),
    _j("sk-", "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6"),
    _j("AI", "za", "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"),
    _j("sk_", "live_", "Ab3dEf6hIj9kLm2nOp5qRs8t"),
    _j("npm_", "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8"),
    _j("hf_", "AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh"),
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
#: The shortest text each of six shapes masks, as short as a shape allows: an
#: assignment's value is six characters at least, a URL parameter's eight.
SHORTEST = (
    _j("tok", "en=", "abcdef"),
    _j("&k", "ey=", "abcdefgh"),
    _j("?s", "ig=", "abcdefgh"),
    _j("xox", "b-", "abcdefghij"),
    _j("secret_", "access_key=", "a"),
    _j("AKIA", "ABCDEFGHIJKLMNOP"),
)
#: The notes the tool writes, as ``apply`` (a finishing run's among them),
#: ``rollback`` and a blocked run write them, two holding a credential where a
#: path or an operator's name goes.
TOOL_NOTES = (
    f"applied 2026-02-02T00:00:00Z by human:{TOKEN}; rollback: asi rollback CAND-1",
    "applied 2026-02-02T00:00:00Z by human:op; " + FINISHING_NOTE + "rollback: asi rollback CAND-1",
    f"rolled back 2026-02-03T00:00:00Z from /state/backups/{TOKEN}/manifest.json",
    BLOCKED_RUN_NOTE,
)
#: Masked twice, this one reads as one mask fewer: the value of a URL
#: parameter is a second parameter, escaped as JSON escapes ``&``.
MERGING = _j("see ?", "token=", "\\u0026", "secret=", "x" * 20)
#: Text of the kinds a field can hold, each read in each field: empty, blank,
#: plain, each shape above alone, a credential on a later line, after a
#: carriage return, among non-ASCII letters, decomposed, astral or a lone
#: surrogate, among control characters, beside a mask, beside a second shape,
#: where masking twice merges two masks, at length, and a plain item repeated.
VARIED = (
    "",
    "   ",
    "plain words, nothing to mask",
    *SHAPED,
    *SHORTEST,
    *TOOL_NOTES,
    f"first line\nsecond line {TOKEN}\nthird line",
    f"\t{TOKEN}\r\nafter a carriage return {TOKEN}",
    f"caf\u00e9 {TOKEN} na\u00efve \u65e5\u672c\u8a9e",
    f"cafe\u0301 \U0001f600 \ud800 \u00a0{TOKEN}",
    f"\x00\x07\x1b[2K\x7f\x85\u200b\u202e\u2028{TOKEN}\x1b[0m",
    f"{MASK} {TOKEN}",
    MERGING,
    f"{TOKEN} and {SHAPED[4]}",
    "word " * 20_000 + TOKEN,
    "plain words, nothing to mask",
)


def _validated(base: Path) -> tuple[list[str], str, Path]:
    cfg = ["--config", str(build_env(base, dry_run=False, seed=SEED))]
    proc = run_cli([*cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
    cid = next(
        c["candidate_id"]
        for c in json.loads(proc.stdout)["candidates"]
        if c.get("status") == "validated"
    )
    return cfg, cid, base / "state" / "candidates" / f"{cid}.json"


def _plant(raw: dict[str, Any], field: str) -> None:
    if field in ("risks", "notes"):
        raw[field] = [*raw[field], f"leaks {TOKEN}"]
    else:
        raw[field] = raw[field] + f" see {TOKEN}"


def _edit(record: Path, *fields: str) -> None:
    raw = json.loads(record.read_text(encoding="utf-8"))
    for field in fields:
        _plant(raw, field)
    record.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class TheLoaderKeepsTheProseMaskedTests(unittest.TestCase):
    """``Candidate.from_dict`` on a real record, edited per case."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        _, _, record = _validated(Path(cls._tmp.name).resolve())
        cls.raw = json.loads(record.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def copy(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.raw))

    def assertKept(self, got: Any, want: Any) -> None:
        """Equal, or a failure naming where the two part. A diff of the long
        text is not printed: difflib recurses past Python's limit on it."""
        if got != want:
            g, w = json.dumps(got), json.dumps(want)
            at = next((i for i, (a, b) in enumerate(zip(g, w)) if a != b), min(len(g), len(w)))
            self.fail(f"differs at character {at}: {g[at:at + 60]!r} != {w[at:at + 60]!r}")

    def test_each_prose_field_an_edited_record_carries_is_read_masked(self):
        for field in PROSE:
            with self.subTest(field=field):
                raw = self.copy()
                _plant(raw, field)
                kept = json.dumps(Candidate.from_dict(raw).to_dict())
                self.assertNotIn(TOKEN, kept)
                self.assertEqual(kept.count(MASK), 1)

    def test_a_credential_in_any_item_of_a_list_is_masked(self):
        raw = self.copy()
        items = [f"first {TOKEN}", "middle", f"last {TOKEN}"]
        raw["risks"], raw["notes"] = list(items), list(items)
        candidate = Candidate.from_dict(raw)
        masked = (f"first {MASK}", "middle", f"last {MASK}")
        self.assertEqual(candidate.risks, masked)
        self.assertEqual(candidate.notes, masked)

    def test_masking_is_the_only_change_the_loader_makes(self):
        raw = self.copy()
        raw["hypothesis"] = f"  line one\n\tline two {TOKEN}  \n"
        raw["intended_change"] = f"\n{TOKEN}"
        raw["rollback"] = f"restore it ({TOKEN}) "
        raw["risks"] = [f" {TOKEN}\nsecond line "]
        raw["notes"] = [f"{TOKEN} "]
        candidate = Candidate.from_dict(raw)
        self.assertEqual(candidate.hypothesis, f"  line one\n\tline two {MASK}  \n")
        self.assertEqual(candidate.intended_change, f"\n{MASK}")
        self.assertEqual(candidate.rollback, f"restore it ({MASK}) ")
        self.assertEqual(candidate.risks, (f" {MASK}\nsecond line ",))
        self.assertEqual(candidate.notes, (f"{MASK} ",))

    def test_a_credential_beside_a_mask_and_a_second_credential_are_masked(self):
        second = "sk-" + "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6"
        text = f"was {MASK} then {TOKEN} and {TOKEN} and {second}"
        masked = f"was {MASK} then {MASK} and {MASK} and [REDACTED:model-provider-key]"
        for field in PROSE:
            with self.subTest(field=field):
                listed = field in ("risks", "notes")
                raw = self.copy()
                raw[field] = [text] if listed else text
                kept = Candidate.from_dict(raw).to_dict()[field]
                self.assertEqual(kept, [masked] if listed else masked)

    def test_long_prose_and_repeated_items_are_kept_whole(self):
        long, masked = "word " * 1000 + TOKEN, "word " * 1000 + MASK
        for field in PROSE:
            with self.subTest(field=field):
                listed = field in ("risks", "notes")
                raw = self.copy()
                raw[field] = ["same", long, "same"] if listed else long
                kept = Candidate.from_dict(raw).to_dict()[field]
                self.assertEqual(kept, ["same", masked, "same"] if listed else masked)

    def test_the_varied_text_holds_every_shape_redact_knows(self):
        found = {label for text in VARIED for label in redact(text).counts}
        self.assertEqual(min(len(text) for text in VARIED if redact(text).total), 12)
        for text in SHORTEST:
            with self.subTest(text=text):
                self.assertEqual(redact(text).total, 1)
        once = redact(MERGING).text
        self.assertNotEqual(redact(once).text, once)
        self.assertEqual(found, {shape.label for shape in redaction._SHAPES})
        for text in SHAPED:
            with self.subTest(text=text[:20]):
                self.assertEqual(redact(text).total, 1)

    def test_each_field_keeps_what_masking_alone_makes_of_each_kind_of_text(self):
        expected = [redact(text).text for text in VARIED]
        self.assertTrue(all(TOKEN not in text for text in expected))
        for field in ("hypothesis", "intended_change", "rollback"):
            for text, masked in zip(VARIED, expected):
                with self.subTest(field=field, text=text[:40]):
                    raw = self.copy()
                    raw[field] = text
                    self.assertKept(Candidate.from_dict(raw).to_dict()[field], masked)
        for field in ("risks", "notes"):
            with self.subTest(field=field):
                raw = self.copy()
                raw[field] = list(VARIED)
                self.assertKept(Candidate.from_dict(raw).to_dict()[field], expected)

    def test_a_record_in_each_status_the_tool_writes_is_read_masked(self):
        for status in (
            STATUS_PROPOSED,
            STATUS_VALIDATED,
            STATUS_GATE_FAILED,
            STATUS_APPLIED,
            STATUS_ROLLED_BACK,
            STATUS_BLOCKED_RUN,
        ):
            with self.subTest(status=status):
                raw = self.copy()
                raw["status"] = status
                for field in PROSE:
                    _plant(raw, field)
                kept = json.dumps(Candidate.from_dict(raw).to_dict())
                self.assertNotIn(TOKEN, kept)
                self.assertEqual(kept.count(MASK), len(PROSE))

    def test_a_record_this_tool_wrote_reads_back_unchanged(self):
        self.assertEqual(Candidate.from_dict(self.copy()).to_dict(), self.raw)
        raw = self.copy()
        raw["hypothesis"] += f" see {MASK}"
        raw["notes"] = [*raw["notes"], f"note {MASK}"]
        self.assertEqual(Candidate.from_dict(json.loads(json.dumps(raw))).to_dict(), raw)

    def test_a_prose_field_is_still_read_with_str(self):
        raw = self.copy()
        raw["hypothesis"] = 12345
        raw["rollback"] = ["restore", TOKEN]
        del raw["intended_change"]
        candidate = Candidate.from_dict(raw)
        self.assertEqual(candidate.hypothesis, "12345")
        self.assertEqual(candidate.rollback, f"['restore', '{MASK}']")
        self.assertEqual(candidate.intended_change, "")

    def test_a_list_that_is_not_an_array_of_strings_is_still_refused(self):
        for field, value, message in (
            ("risks", f"one {TOKEN}", "risks must be an array of strings, got str"),
            ("risks", ["fine", 7], "risks[1] must be a string, got int"),
            ("notes", {"n": TOKEN}, "notes must be an array of strings, got dict"),
            ("notes", [None], "notes[0] must be a string, got NoneType"),
        ):
            with self.subTest(field=field, value=value):
                raw = self.copy()
                raw[field] = value
                with self.assertRaises(ValueError) as caught:
                    Candidate.from_dict(raw)
                self.assertEqual(str(caught.exception), message)


class ReviewAndApplyKeepTheProseMaskedTests(unittest.TestCase):
    """The commands that read a record, each on its own environment."""

    def test_review_prints_no_planted_credential_in_any_prose_field(self):
        for field in PROSE:
            with self.subTest(field=field), TemporaryDirectory() as tmp:
                cfg, cid, record = _validated(Path(tmp).resolve())
                _edit(record, field)
                text = run_cli([*cfg, "review", cid])
                self.assertEqual(text.returncode, 0, text.stderr)
                self.assertNotIn(TOKEN, text.stdout + text.stderr)
                self.assertEqual(text.stdout.count(MASK), 1)
                as_json = run_cli([*cfg, "--json", "review", cid])
                self.assertEqual(as_json.returncode, 0, as_json.stderr)
                self.assertNotIn(TOKEN, as_json.stdout + as_json.stderr)
                shown = json.loads(as_json.stdout)[field]
                self.assertIn(MASK, shown[-1] if field in ("risks", "notes") else shown)

    def test_a_clean_record_is_reviewed_as_written(self):
        with TemporaryDirectory() as tmp:
            cfg, cid, record = _validated(Path(tmp).resolve())
            raw = json.loads(record.read_text(encoding="utf-8"))
            proc = run_cli([*cfg, "--json", "review", cid])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            shown = json.loads(proc.stdout)
            for field in PROSE:
                with self.subTest(field=field):
                    self.assertEqual(shown[field], raw[field])
            self.assertNotIn("[REDACTED:", run_cli([*cfg, "review", cid]).stdout)

    def test_apply_writes_the_record_back_masked(self):
        with TemporaryDirectory() as tmp:
            cfg, cid, record = _validated(Path(tmp).resolve())
            _edit(record, "hypothesis", "notes")
            token = json.loads(run_cli([*cfg, "--json", "review", cid]).stdout)["approval_token"]
            proc = run_cli([*cfg, "apply", cid, "--approve", token, "--i-reviewed-this"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertNotIn(TOKEN, proc.stdout + proc.stderr)
            after = record.read_text(encoding="utf-8")
            self.assertEqual(json.loads(after)["status"], "applied")
            self.assertNotIn(TOKEN, after)
            self.assertEqual(after.count(MASK), 2)


if __name__ == "__main__":
    unittest.main()
