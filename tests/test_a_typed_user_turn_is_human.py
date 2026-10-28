"""A typed user turn is a person's, so its conversation is human.

``classify_conversation`` (``domain/authorship.py``) marks a conversation
orchestrated when its opening user turn carries an origin outside a short list
of human ones, and the list was ``user``, ``human`` and ``interactive``. The
export contract gives a typed user turn the origin ``user-input``: every user
message the fixture builder writes carries it. Measured before (3.11 and 3.13,
the same on both): ``daily --commit`` over the five healthy fixture exports,
none agent-titled and none naming an agent, classed all five orchestrated, each
for "opening user turn has a non-human origin marker ('user-input')", and the
receipt's ``class_counts_scored`` read ``human: 0, orchestrated: 5``.
``user-input`` is now a human origin; any other origin still marks the turn
generated.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from agent_skill_improver.adapters.transcript_export import normalize_export
from agent_skill_improver.domain.authorship import classify_conversation
from tests.support import FIXTURE_EXPORTS, build_env, run_cli

HUMAN_REASON = "a human user turn opens the thread and no orchestrator marker matched"
HEALTHY = ("compliant", "failing", "injection", "second-workspace", "edge-shapes")


def class_of_origin(origin: str | None) -> tuple[str, tuple[str, ...]]:
    opening = SimpleNamespace(text="please fix the test", origin=origin)
    conversation = SimpleNamespace(title="plain", agent_name="", user_messages=(opening,))
    return classify_conversation(conversation)


class TheHealthyFixturesAreHumanTests(unittest.TestCase):
    def test_each_healthy_fixture_conversation_is_human(self) -> None:
        for name in HEALTHY:
            with self.subTest(fixture=name):
                payload = json.loads((FIXTURE_EXPORTS / f"{name}.json").read_text(encoding="utf-8"))
                conversation = normalize_export(payload)
                self.assertEqual(conversation.user_messages[0].origin, "user-input")
                self.assertEqual(classify_conversation(conversation), ("human", (HUMAN_REASON,)))

    def test_the_receipt_counts_the_healthy_fixtures_as_human(self) -> None:
        with TemporaryDirectory() as tmp:
            config = build_env(Path(tmp).resolve() / "env", seed="c177")
            proc = run_cli(
                ["--config", str(config), "--json", "daily", "--commit", "--day", "2026-02-01"]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            receipt = json.loads(proc.stdout)
        self.assertEqual(
            receipt["class_counts_scored"], {"human": 5, "orchestrated": 0, "unknown": 0}
        )
        self.assertEqual(
            {info["authorship_class"] for info in receipt["classes"].values()}, {"human"}
        )


class TheOriginListTests(unittest.TestCase):
    def test_user_input_in_another_case_or_spacing_is_human(self) -> None:
        for origin in ("user-input", "User-Input", " USER-INPUT\n"):
            with self.subTest(origin=origin):
                self.assertEqual(class_of_origin(origin), ("human", (HUMAN_REASON,)))

    def test_a_user_turn_with_no_origin_is_human(self) -> None:
        for origin in ("", None, "   "):
            with self.subTest(origin=origin):
                self.assertEqual(class_of_origin(origin), ("human", (HUMAN_REASON,)))

    def test_the_control_the_origins_listed_before_are_still_human(self) -> None:
        for origin in ("user", "human", "interactive"):
            with self.subTest(origin=origin):
                self.assertEqual(class_of_origin(origin)[0], "human")

    def test_the_control_an_origin_that_only_contains_a_human_one_is_generated(self) -> None:
        for origin in ("orchestrator", "user-input-relay", "relayed-user-input", "users", "agent"):
            with self.subTest(origin=origin):
                klass, reasons = class_of_origin(origin)
                self.assertEqual(klass, "orchestrated")
                self.assertEqual(
                    reasons, (f"opening user turn has a non-human origin marker ({origin!r})",)
                )


if __name__ == "__main__":
    unittest.main()
