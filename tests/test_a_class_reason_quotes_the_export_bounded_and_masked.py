"""A class reason quotes what the export says, bounded and masked.

``classify_conversation`` (``domain/authorship.py``) records why it decided a
conversation's class, quoting the title, the agent name or the opening turn's
origin, and ``daily`` puts those reasons in the receipt's ``classes`` table and
the run's JSON output. Each was quoted whole with ``repr``. Measured before (3.11
and 3.13, the same on both), one fixture export changed at a time: a title of
"dispatcher " and 200,000 more characters, an agent name or an origin of
200,000 characters each made a receipt of about 210 KB where the healthy
fixtures make one of 10 KB; a GitHub token in the title was copied into the
receipt and the JSON output unmasked, and one in the agent name lowercased but
otherwise whole. They are now quoted
through ``quoted_value`` (``repr``, masked, cut at 120 characters), the agent
name and origin as written rather than as lowercased for matching.
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from agent_skill_improver.domain.authorship import classify_conversation
from tests.support import FIXTURE_EXPORTS, build_env, run_cli

GITHUB_TOKEN = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
AWS_KEY_ID = "AK" + "IA" + "ABCDEFGHIJKLMNOP"
LONG = 200_000


def reasons(*, title: str = "plain", agent_name: str = "", origin: str = "") -> list[str]:
    opening = SimpleNamespace(text="please fix the test", origin=origin)
    conversation = SimpleNamespace(title=title, agent_name=agent_name, user_messages=(opening,))
    return list(classify_conversation(conversation)[1])


def the_reason(found: list[str], start: str) -> str:
    matching = [r for r in found if r.startswith(start)]
    if len(matching) != 1:
        raise AssertionError(f"want one reason starting {start!r}, got {found!r}")
    return matching[0]


TITLE = "title matches an orchestrator marker: "
AGENT = "thread carries an agent_name ("
ORIGIN = "opening user turn has a non-human origin marker ("


class EachQuotedValueIsBoundedTests(unittest.TestCase):
    def test_a_long_title_makes_a_bounded_reason(self) -> None:
        reason = the_reason(reasons(title="dispatcher " + "t" * LONG), TITLE)
        self.assertLess(len(reason), 300)
        self.assertIn("more chars not quoted", reason)
        self.assertTrue(reason.startswith(TITLE + "'dispatcher ttt"))

    def test_a_long_agent_name_makes_a_bounded_reason(self) -> None:
        reason = the_reason(reasons(agent_name="a" * LONG), AGENT)
        self.assertLess(len(reason), 300)
        self.assertIn("more chars not quoted", reason)

    def test_a_long_origin_makes_a_bounded_reason(self) -> None:
        reason = the_reason(reasons(origin="bot-" + "o" * LONG), ORIGIN)
        self.assertLess(len(reason), 300)
        self.assertIn("more chars not quoted", reason)


class EachQuotedValueIsMaskedTests(unittest.TestCase):
    def assert_masked(self, reason: str, secret: str) -> None:
        self.assertNotIn(secret, reason)
        self.assertNotIn(secret.lower(), reason)
        self.assertIn("[REDACTED:", reason)

    def test_a_credential_in_the_title_is_masked(self) -> None:
        reason = the_reason(reasons(title=f"dispatcher {GITHUB_TOKEN}"), TITLE)
        self.assert_masked(reason, GITHUB_TOKEN)

    def test_a_credential_in_the_agent_name_is_masked_as_written(self) -> None:
        # Lowercased for matching, an AWS key id no longer reads as one to the
        # masking shapes, so the name is quoted as the export wrote it.
        for secret in (GITHUB_TOKEN, AWS_KEY_ID):
            with self.subTest(secret=secret[:4]):
                self.assert_masked(the_reason(reasons(agent_name=secret), AGENT), secret)

    def test_a_credential_in_the_origin_is_masked_as_written(self) -> None:
        reason = the_reason(reasons(origin=f"relay {AWS_KEY_ID}"), ORIGIN)
        self.assert_masked(reason, AWS_KEY_ID)

    def test_a_line_break_in_the_agent_name_stays_on_one_line(self) -> None:
        reason = the_reason(reasons(agent_name="x\nWARN forged line"), AGENT)
        self.assertEqual(reason, AGENT + "'x\\nWARN forged line')")

    def test_the_control_short_values_are_quoted_as_before(self) -> None:
        found = reasons(title="Orchestrator Run 41", agent_name="builder", origin="orchestrator")
        self.assertEqual(
            found,
            [
                TITLE + "'Orchestrator Run 41'",
                AGENT + "'builder')",
                ORIGIN + "'orchestrator')",
            ],
        )


class TheClassIsStillDecidedOnTheMatchedFormTests(unittest.TestCase):
    """Only the quote is as written; the decision reads the lowercased, trimmed form."""

    def test_a_human_origin_in_another_case_or_spacing_is_still_human(self) -> None:
        for origin in ("Human", "  USER ", "Interactive\n"):
            with self.subTest(origin=origin):
                opening = SimpleNamespace(text="please fix the test", origin=origin)
                conversation = SimpleNamespace(
                    title="plain", agent_name="", user_messages=(opening,)
                )
                klass, found = classify_conversation(conversation)
                self.assertEqual(klass, "human")
                self.assertFalse([r for r in found if r.startswith(ORIGIN)])

    def test_an_agent_name_of_only_spaces_is_no_agent_name(self) -> None:
        opening = SimpleNamespace(text="please fix the test", origin="user")
        conversation = SimpleNamespace(title="plain", agent_name="  \t ", user_messages=(opening,))
        klass, found = classify_conversation(conversation)
        self.assertEqual(klass, "human")
        self.assertFalse([r for r in found if r.startswith(AGENT)])


class TheReceiptStaysSmallAndMaskedTests(unittest.TestCase):
    def daily(self, title: str | None) -> tuple[str, str]:
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            exports = base / "exports"
            shutil.copytree(FIXTURE_EXPORTS, exports)
            if title is not None:
                path = exports / "compliant.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["thread"]["title"] = title
                path.write_text(json.dumps(payload), encoding="utf-8")
            config = build_env(base / "env", fixtures_dir=exports, seed="c176")
            proc = run_cli(
                ["--config", str(config), "--json", "daily", "--commit", "--day", "2026-02-01"]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            (receipt,) = (base / "env" / "state" / "receipts").glob("2026-02-01*.json")
            return proc.stdout, receipt.read_text(encoding="utf-8")

    def test_a_long_title_bearing_a_credential_makes_a_small_masked_receipt(self) -> None:
        stdout, receipt = self.daily(f"dispatcher {GITHUB_TOKEN} " + "t" * LONG)
        for text in (stdout, receipt):
            self.assertLess(len(text), 20_000)
            self.assertNotIn(GITHUB_TOKEN, text)
        classes = json.loads(receipt)["classes"]["1700000000001-1"]
        self.assertEqual(classes["authorship_class"], "orchestrated")
        self.assertIn("[REDACTED:github-token]", the_reason(classes["reasons"], TITLE))

    def test_the_control_the_healthy_fixtures_make_a_receipt_of_the_same_order(self) -> None:
        stdout, receipt = self.daily(None)
        self.assertLess(len(receipt), 20_000)
        self.assertEqual(stdout, receipt)


if __name__ == "__main__":
    unittest.main()
