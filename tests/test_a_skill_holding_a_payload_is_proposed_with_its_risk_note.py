"""A skill holding an executable payload is proposed against with its risk note.

``propose_candidate`` in ``app/pipeline.py`` refuses a skill listed in
``[guard].protected_skill_ids`` (``SelfMutationRefused``, written to
``state/refusals/``), and proposes against a skill whose directory holds an
executable payload, noting ``ELEVATED REVIEW RISK`` with the payload count and
the first five names. ``docs/ARCHITECTURE.md`` said such a skill was refused as
a target, as a protected one is; measured (3.11 and 3.13, ``daily --commit``
over the fixtures with an executable ``scripts/hook.sh`` beside the failing
skill): one candidate, no refusal, the note on the candidate and in ``review``.
Nothing tested a payload the inventory could see; only a directory it could
not list.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.support import build_env, run_cli

NOTE = "ELEVATED REVIEW RISK: this skill's directory contains"


def run_case(base: Path, *, hooks: tuple[str, ...], protected: bool) -> dict:
    config = build_env(base / "env", seed="c182", protected_skill_ids=("tdd",) if protected else ())
    scripts = base / "env" / "skills-canonical" / "tdd" / "scripts"
    for name in hooks:
        scripts.mkdir(exist_ok=True)
        (scripts / name).write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        os.chmod(scripts / name, 0o755)
    argv = ["--config", str(config)]
    daily = run_cli([*argv, "--json", "daily", "--commit", "--day", "2026-01-02"])
    state = base / "env" / "state"
    candidates = sorted((state / "candidates").rglob("*.json"))
    refusals = (
        sorted(f for f in (state / "refusals").rglob("*.json"))
        if (state / "refusals").exists()
        else []
    )
    records = [json.loads(c.read_text(encoding="utf-8")) for c in candidates]
    review = run_cli([*argv, "review", records[0]["id"]]) if records else None
    return {
        "daily_rc": daily.returncode,
        "stderr": daily.stderr,
        "counts": json.loads(daily.stdout)["counts"],
        "records": records,
        "refusals": [json.loads(r.read_text(encoding="utf-8")) for r in refusals],
        "review_rc": review.returncode if review else None,
        "review": review.stdout if review else "",
    }


class APayloadSkillIsProposedAgainstTests(unittest.TestCase):
    cases: dict[str, dict] = {}

    @classmethod
    def setUpClass(cls) -> None:
        plans = {
            "control": ((), False),
            "hook": (("hook.sh",), False),
            "six": (("hook.sh", "tool0.sh", "tool1.sh", "tool2.sh", "tool3.sh", "tool4.sh"), False),
            "protected": ((), True),
            "protected_hook": (("hook.sh",), True),
        }
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name).resolve()
        for name, (hooks, protected) in plans.items():
            cls.cases[name] = run_case(root / name, hooks=hooks, protected=protected)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def notes(self, case: str) -> list[str]:
        (record,) = self.cases[case]["records"]
        return [n for n in record["notes"] if n.startswith("ELEVATED REVIEW RISK")]

    def test_a_skill_holding_a_hook_is_proposed_against_and_not_refused(self) -> None:
        case = self.cases["hook"]
        self.assertEqual(case["daily_rc"], 0, case["stderr"])
        self.assertEqual(len(case["records"]), 1)
        self.assertEqual(case["counts"]["candidates_proposed"], 1)
        self.assertEqual(case["counts"]["proposals_refused_by_policy"], 0)
        self.assertEqual(case["refusals"], [])

    def test_its_candidate_and_review_carry_the_note_naming_the_hook(self) -> None:
        (note,) = self.notes("hook")
        self.assertTrue(
            note.startswith(f"{NOTE} 1 executable payload file(s) (scripts/hook.sh). "), note
        )
        self.assertEqual(self.cases["hook"]["review_rc"], 0)
        self.assertIn(f"note: {note}", self.cases["hook"]["review"])

    def test_past_five_payload_files_the_note_counts_them_all_and_names_five(self) -> None:
        (note,) = self.notes("six")
        named = ", ".join(
            f"scripts/{n}" for n in ("hook.sh", "tool0.sh", "tool1.sh", "tool2.sh", "tool3.sh")
        )
        self.assertTrue(
            note.startswith(f"{NOTE} 6 executable payload file(s) ({named}, ...). "), note
        )

    def test_a_skill_listed_as_protected_is_refused_with_or_without_a_hook(self) -> None:
        for name in ("protected", "protected_hook"):
            with self.subTest(case=name):
                case = self.cases[name]
                self.assertEqual(case["daily_rc"], 0, case["stderr"])
                self.assertEqual(case["records"], [])
                self.assertEqual(case["counts"]["proposals_refused_by_policy"], 1)
                (refusal,) = case["refusals"]
                self.assertEqual((refusal["stage"], refusal["record_id"]), ("propose", "tdd"))
                self.assertIn("[guard].protected_skill_ids", refusal["reason"])

    def test_the_control_a_skill_without_payloads_has_no_risk_note(self) -> None:
        case = self.cases["control"]
        self.assertEqual(len(case["records"]), 1)
        self.assertEqual(self.notes("control"), [])
        self.assertEqual(case["refusals"], [])
        self.assertNotIn("ELEVATED", case["review"])


if __name__ == "__main__":
    unittest.main()
