"""Automation refusals name the attempted action, not a human-authentication guarantee."""

from __future__ import annotations

import datetime
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agent_skill_improver.app.promote import apply_candidate
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.domain.approval import HumanApproval
from agent_skill_improver.errors import AutomationRefused
from tests.support import build_env, run_cli

MARKER = "ASI_AUTOMATED_RUN"
DAY = "2026-02-01"


class AMarkedRefusalNamesApplyOrRollbackWithoutClaimingAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        clean = mock.patch.dict(os.environ)
        clean.start()
        self.addCleanup(clean.stop)
        os.environ.pop(MARKER, None)
        self.cfg = ["--config", str(build_env(self.base, dry_run=False, seed="c243"))]
        daily = run_cli([*self.cfg, "daily", "--commit", "--day", DAY], env={MARKER: "0"})
        self.assertEqual(daily.returncode, 0, daily.stderr)
        candidates = run_cli([*self.cfg, "--json", "candidates"], env={MARKER: "0"})
        self.assertEqual(candidates.returncode, 0, candidates.stderr)
        self.cid = sorted(c["id"] for c in json.loads(candidates.stdout)["candidates"])[0]
        review = run_cli([*self.cfg, "--json", "review", self.cid], env={MARKER: "0"})
        self.assertEqual(review.returncode, 0, review.stderr)
        self.token = json.loads(review.stdout)["approval_token"]
        self.record = self.base / "state" / "candidates" / f"{self.cid}.json"
        target = json.loads(self.record.read_text())["target"]
        self.target = Path(target["canonical_path"])
        self.before = self.target.read_bytes()
        self.apply = [*self.cfg, "apply", self.cid, "--approve", self.token, "--i-reviewed-this"]

    def test_a_marked_cli_apply_refuses_with_the_review_interlock_limit(self) -> None:
        refused = run_cli(self.apply, env={MARKER: "1"})
        self.assertEqual(refused.returncode, 3, refused.stderr)
        self.assertIn("the automation marker (ASI_AUTOMATED_RUN) is set", refused.stderr)
        self.assertIn("run apply from an unmarked interactive shell", refused.stderr)
        self.assertIn("do not schedule apply", refused.stderr)
        self.assertIn("not caller authentication", refused.stderr)
        self.assertEqual(self.target.read_bytes(), self.before)
        self.assertEqual(json.loads(self.record.read_text())["status"], "validated")

    def test_an_apply_belt_refusal_names_apply_and_not_caller_authentication(self) -> None:
        clock = FrozenClock(datetime.datetime(2026, 2, 2, tzinfo=datetime.timezone.utc))
        state = StateStore(self.base / "state", clock=clock)
        candidate = state.read_candidate(self.cid)
        approval = HumanApproval.from_cli(token=self.token, acknowledged=True)
        with mock.patch.dict(os.environ, {MARKER: "1"}):
            with self.assertRaises(AutomationRefused) as caught:
                apply_candidate(
                    candidate,
                    approval=approval,
                    skill_store=object(),
                    state=state,
                    clock=clock,
                )
        text = str(caught.exception)
        self.assertIn("promotion refused: automation marker is set", text)
        self.assertIn("run apply from an unmarked interactive shell", text)
        self.assertIn("do not schedule apply", text)
        self.assertIn("not caller authentication", text)
        self.assertNotIn("run rollback", text)
        self.assertEqual(self.target.read_bytes(), self.before)

    def test_a_marked_rollback_names_rollback_and_never_advises_apply(self) -> None:
        applied = run_cli(self.apply, env={MARKER: "0"})
        self.assertEqual(applied.returncode, 0, applied.stderr)
        changed = self.target.read_bytes()
        self.assertNotEqual(changed, self.before)
        refused = run_cli([*self.cfg, "rollback", self.cid], env={MARKER: "1"})
        self.assertEqual(refused.returncode, 3, refused.stderr)
        self.assertIn("rollback refused: automation marker is set", refused.stderr)
        self.assertIn("Run rollback from an unmarked interactive shell", refused.stderr)
        self.assertIn("do not schedule rollback", refused.stderr)
        self.assertIn("not caller authentication", refused.stderr)
        self.assertNotIn("run apply", refused.stderr)
        self.assertNotIn("asi apply", refused.stderr)
        self.assertEqual(self.target.read_bytes(), changed)
        self.assertEqual(json.loads(self.record.read_text())["status"], "applied")
        self.assertFalse((self.base / "state" / "receipts" / f"rollback--{self.cid}.json").exists())
        restored = run_cli([*self.cfg, "rollback", self.cid], env={MARKER: "0"})
        self.assertEqual(restored.returncode, 0, restored.stderr)
        self.assertEqual(self.target.read_bytes(), self.before)

    def test_apply_help_calls_the_command_review_gated_not_human_gated(self) -> None:
        general = run_cli(["--help"], env={MARKER: "0"})
        self.assertEqual(general.returncode, 0, general.stderr)
        self.assertNotRegex(general.stdout, r"(?i)human-\s*gated")
        self.assertRegex(general.stdout, r"review-\s*gated")


if __name__ == "__main__":
    unittest.main()
