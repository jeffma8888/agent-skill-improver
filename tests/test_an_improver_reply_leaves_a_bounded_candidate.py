"""An improver reply leaves a bounded candidate.

``parse_proposal`` keeps the improver's hypothesis, intended change, risks and
rollback note masked (``kept_prose``); they are persisted in the candidate and
printed by ``review``, and the reply is bounded only by the role's
``max_output_bytes``. Measured before (3.11 and 3.13, through ``daily --commit``
with an improver that pads the fake model's reply): a hypothesis of 100,000
lines made a candidate of about 2 MB and a review of 100,052 lines; 200,000
risks one of about 1.8 MB and a review of 200,052 lines; a 1,000,000-character
intended change, rollback note or risk a candidate and a review of about 1 MB
each; the healthy candidate is about 14 KB and its review 52 lines. Each prose
field is now clipped (masked first) at ``MAX_PROPOSAL_PROSE_CHARS``, each risk at
``MAX_RISK_CHARS``, and the first ``MAX_RISKS_KEPT`` risks are kept, a candidate
note counting the ones that were not.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver import SCHEMA_PROPOSAL
from agent_skill_improver.domain.candidate import (
    MAX_PROPOSAL_PROSE_CHARS,
    MAX_RISK_CHARS,
    MAX_RISKS_KEPT,
    parse_proposal,
)
from tests.support import REPO_ROOT, build_env, run_cli

GITHUB_TOKEN = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
MASK = "[REDACTED:github-token]"
CLIPPED = " ...[clipped]"
PROSE = ("hypothesis", "intended_change", "rollback")


def proposal(**fields):
    payload = {
        "schema": SCHEMA_PROPOSAL,
        "target_skill_id": "tdd",
        "hypothesis": "h",
        "intended_change": "i",
        "risks": ["r"],
        "rollback": "b",
        "new_skill_text": "---\nname: tdd\n---\nbody\n",
    }
    payload.update(fields)
    return parse_proposal(payload)


class TheLimitsTests(unittest.TestCase):
    def test_the_limits_are_the_ones_the_docs_state(self) -> None:
        # The tests below read the constants, so a changed value would move
        # their expectations with it; the documented numbers are pinned here.
        self.assertEqual(
            (MAX_PROPOSAL_PROSE_CHARS, MAX_RISK_CHARS, MAX_RISKS_KEPT), (4_000, 2_000, 20)
        )


class EachProseFieldIsClippedTests(unittest.TestCase):
    def test_a_prose_field_up_to_the_limit_is_kept_whole(self) -> None:
        for name in PROSE:
            with self.subTest(field=name):
                text = "p" * MAX_PROPOSAL_PROSE_CHARS
                self.assertEqual(getattr(proposal(**{name: text}), name), text)

    def test_a_longer_prose_field_is_clipped_and_says_so(self) -> None:
        for name in PROSE:
            with self.subTest(field=name):
                kept = getattr(proposal(**{name: "p" * (MAX_PROPOSAL_PROSE_CHARS + 1)}), name)
                self.assertEqual(kept, "p" * MAX_PROPOSAL_PROSE_CHARS + CLIPPED)

    def test_a_credential_at_the_cut_is_masked_whole_before_the_clip(self) -> None:
        for name in PROSE:
            with self.subTest(field=name):
                # The raw token straddles the cut; its mask ends before it, and the
                # masked text is still longer than the limit.
                text = "p" * (MAX_PROPOSAL_PROSE_CHARS - 30) + f" {GITHUB_TOKEN} " + "t" * 100
                kept = getattr(proposal(**{name: text}), name)
                self.assertNotIn(GITHUB_TOKEN[:12], kept)
                self.assertIn(f" {MASK} ", kept)
                self.assertTrue(kept.endswith(CLIPPED))


class RisksAreBoundedTests(unittest.TestCase):
    def test_a_risk_up_to_the_limit_is_kept_whole(self) -> None:
        risk = "r" * MAX_RISK_CHARS
        self.assertEqual(proposal(risks=[risk]).risks, (risk,))

    def test_a_longer_risk_is_clipped_and_says_so(self) -> None:
        kept = proposal(risks=["r" * (MAX_RISK_CHARS + 1)]).risks
        self.assertEqual(kept, ("r" * MAX_RISK_CHARS + CLIPPED,))

    def test_up_to_the_limit_every_risk_is_kept(self) -> None:
        risks = [f"risk {i}" for i in range(MAX_RISKS_KEPT)]
        result = proposal(risks=risks)
        self.assertEqual(result.risks, tuple(risks))
        self.assertEqual(result.risks_not_kept, 0)

    def test_past_the_limit_the_first_risks_are_kept_and_the_rest_counted(self) -> None:
        risks = [f"risk {i}" for i in range(200_000)]
        result = proposal(risks=risks)
        self.assertEqual(result.risks, tuple(risks[:MAX_RISKS_KEPT]))
        self.assertEqual(result.risks_not_kept, 200_000 - MAX_RISKS_KEPT)

    def test_a_credential_in_a_risk_is_masked(self) -> None:
        self.assertEqual(proposal(risks=[f"leaks {GITHUB_TOKEN}"]).risks, (f"leaks {MASK}",))

    def test_a_credential_at_a_risk_cut_is_masked_whole_before_the_clip(self) -> None:
        # As for the prose fields: the raw token straddles the risk's cut, and a
        # risk cut before it is masked would keep a fragment no shape matches.
        risk = "r" * (MAX_RISK_CHARS - 30) + f" {GITHUB_TOKEN} " + "t" * 100
        (kept,) = proposal(risks=[risk]).risks
        self.assertNotIn(GITHUB_TOKEN[:12], kept)
        self.assertIn(f" {MASK} ", kept)
        self.assertTrue(kept.endswith(CLIPPED))


class TheCandidateAndReviewTests(unittest.TestCase):
    """Through ``daily --commit`` with an improver that adds risks to the fake reply."""

    def run_daily(self, extra_risks: int) -> tuple[dict, str]:
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            improver = base / "improver.py"
            improver.write_text(
                "import json, subprocess, sys\n"
                f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r})\n"
                "from agent_skill_improver.adapters.model_command import extract_json_object\n"
                f"fake = {str(REPO_ROOT / 'tests' / 'fake_model.py')!r}\n"
                "raw = sys.stdin.read()\n"
                "out = subprocess.run([sys.executable, fake, *sys.argv[1:]], input=raw,"
                " capture_output=True, text=True).stdout\n"
                "if json.loads(raw).get('role') == 'improver':\n"
                "    obj = extract_json_object(out)\n"
                f"    obj['risks'] = obj['risks'] + ['extra risk'] * {extra_risks}\n"
                "    out = json.dumps(obj)\n"
                "sys.stdout.write(out)\n",
                encoding="utf-8",
            )
            config = build_env(base / "env", seed="c181")
            text = config.read_text(encoding="utf-8")
            text, n = re.subn(
                r'(\[model\.improver\]\nargv = \[)"([^"]+)", "([^"]+)"',
                lambda m: f'{m.group(1)}"{sys.executable}", "{improver}"',
                text,
            )
            self.assertEqual(n, 1, "the improver's argv was not found")
            config.write_text(text, encoding="utf-8")
            argv = ["--config", str(config)]
            daily = run_cli([*argv, "--json", "daily", "--commit", "--day", "2026-01-02"])
            self.assertEqual(daily.returncode, 0, daily.stderr)
            (path,) = (base / "env" / "state" / "candidates").rglob("*.json")
            record = json.loads(path.read_text(encoding="utf-8"))
            review = run_cli([*argv, "review", record["id"]])
            self.assertEqual(review.returncode, 0, review.stderr)
            return record | {"_size": path.stat().st_size}, review.stdout

    def test_many_risks_make_a_small_candidate_whose_note_review_prints(self) -> None:
        record, review = self.run_daily(200_000)
        note = f"the improver listed {200_000 + 1} risks; the first {MAX_RISKS_KEPT} are kept"
        self.assertEqual(len(record["risks"]), MAX_RISKS_KEPT)
        self.assertIn(note, record["notes"])
        self.assertIn(note, review)
        self.assertLess(record["_size"], 30_000)
        self.assertLess(review.count("\n"), 200)

    def test_the_control_a_healthy_reply_has_no_risks_note(self) -> None:
        record, review = self.run_daily(0)
        self.assertEqual(len(record["risks"]), 1)
        self.assertEqual([n for n in record["notes"] if "risks;" in n], [])
        self.assertNotIn("are kept", review)


if __name__ == "__main__":
    unittest.main()
