"""A number refused for not rounding to a finite double is named by its length
when it is long, as an integer too large is named by its digits.

``_finite_float`` in ``jsonio.py`` refuses a literal past the double range for
every untrusted parse, and its refusal is the reason a conversation export is
passed over, which ``doctor`` prints, ``daily`` prints and the run's receipt
keeps. Measured before (3.11 and 3.13, the same on both): one export beside the
healthy fixtures holding the 1,000,003-character literal ``1`` followed by a
million zeros and ``e9`` made ``doctor`` print 1,000,941 bytes with a
1,000,164-character line, ``doctor --json`` 2,004,703 bytes, ``daily --json``
3,011,408 bytes, and ``daily --commit`` keep a 3,010,702-byte receipt, the
literal spelled whole in each. A literal of up to 32 characters is still
spelled as it is: ``1e400`` reads as before.
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.jsonio import loads_untrusted
from tests.support import FIXTURE_EXPORTS, build_env, run_cli

TAIL = " does not round to a finite double; nothing this tool reads can carry it"
#: A literal of this many zeros after the ``1``, then ``e9``.
ZEROS = 100_000
LONG = "1" + "0" * ZEROS + "e9"
DAY = "2026-02-01"


def refusal(text: str) -> str:
    try:
        loads_untrusted(text)
    except ValueError as exc:
        return str(exc)
    raise AssertionError(f"{text[:40]!r} was read")


class ALongLiteralIsNamedByItsLengthTests(unittest.TestCase):
    def test_a_long_literal_is_named_by_its_length(self) -> None:
        self.assertEqual(
            refusal('{"n": ' + LONG + "}"), f"JSON number of {len(LONG)} characters{TAIL}"
        )

    def test_a_literal_at_the_bound_is_spelled_and_one_past_it_is_named_by_its_length(
        self,
    ) -> None:
        at = "1" + "0" * 27 + "e999"
        past = "1" + "0" * 28 + "e999"
        self.assertEqual((len(at), len(past)), (32, 33))
        self.assertEqual(refusal(at), f"JSON number {at}{TAIL}")
        self.assertEqual(refusal(past), f"JSON number of 33 characters{TAIL}")
        # The sign is one of the characters counted, at the bound as in the length.
        negative_at = "-1" + "0" * 26 + "e999"
        negative_past = "-1" + "0" * 27 + "e999"
        self.assertEqual((len(negative_at), len(negative_past)), (32, 33))
        self.assertEqual(refusal(negative_at), f"JSON number {negative_at}{TAIL}")
        self.assertEqual(refusal(negative_past), f"JSON number of 33 characters{TAIL}")

    def test_a_long_negative_literal_and_a_long_fraction_are_named_by_their_length(
        self,
    ) -> None:
        for literal in ("-" + LONG, "1." + "0" * ZEROS + "e400"):
            with self.subTest(chars=len(literal)):
                self.assertEqual(
                    refusal("[" + literal + "]"),
                    f"JSON number of {len(literal)} characters{TAIL}",
                )

    def test_the_control_a_short_literal_reads_as_before(self) -> None:
        for literal in ("1e400", "-1e400", "1.5e999"):
            with self.subTest(literal=literal):
                self.assertEqual(refusal(literal), f"JSON number {literal}{TAIL}")

    def test_the_control_a_long_literal_that_rounds_to_a_double_is_read(self) -> None:
        self.assertEqual(loads_untrusted("[0." + "0" * ZEROS + "1]"), [0.0])


class ACommandNamesTheExportByTheLiteralsLengthTests(unittest.TestCase):
    """One export holding the long literal, beside the healthy fixtures."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        base = Path(cls._tmp.name).resolve()
        exports = base / "exports"
        shutil.copytree(FIXTURE_EXPORTS, exports)
        (exports / "huge.json").write_text(
            '{"thread": {"id": "t-huge", "n": ' + LONG + "}}", encoding="utf-8"
        )
        cls.env = base / "env"
        cls.cfg = ["--config", str(build_env(cls.env, fixtures_dir=exports, dry_run=False))]
        cls.reason = f"is not valid JSON (JSON number of {len(LONG)} characters{TAIL})"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_doctor_names_the_export_on_one_short_line(self) -> None:
        proc = run_cli([*self.cfg, "doctor"])
        [line] = [line for line in proc.stdout.splitlines() if "huge.json" in line]
        self.assertTrue(line.endswith(f"huge.json {self.reason}"), line[:300])
        self.assertLess(len(proc.stdout), 4096)

    def test_the_receipt_keeps_the_reason_by_the_literal_s_length(self) -> None:
        proc = run_cli([*self.cfg, "daily", "--commit", "--day", DAY])
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        self.assertLess(len(proc.stdout) + len(proc.stderr), 4096)
        [receipt] = sorted(self.env.rglob("receipts/*.json"))
        self.assertNotIn("0" * 200, receipt.read_text(encoding="utf-8"))
        [entry] = [
            e
            for e in json.loads(receipt.read_text(encoding="utf-8"))["discovery"]["passed_over"]
            if e["entry"].endswith("huge.json")
        ]
        self.assertEqual(entry["reason"], self.reason)


if __name__ == "__main__":
    unittest.main()
