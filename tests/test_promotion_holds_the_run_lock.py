"""A promotion and a daily run never overlap: apply and rollback hold the run lock.

Measured before: ``apply`` took no lock. An apply during a scheduled run went
through, the run finished ``blocked`` with the applied document in
``skill_tree.changes``, its candidates were closed as blocked-run, and the next
morning's run was refused until the operator reviewed a mutation the tool had
made itself. Now the live read, the skill write and the records of a promotion
happen under the lock ``daily`` takes, the owner line names the promotion, and a
promotion that finds the lock held refuses by name with nothing written.
"""

from __future__ import annotations

import datetime
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.app.lock import RunLock
from agent_skill_improver.app.promote import _holding_the_run_lock, rollback_candidate
from agent_skill_improver.domain.approval import AutomationMarker
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.cli import RC_OK, RC_REFUSED
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.errors import AutomationRefused, GateFailed, LockHeld

from tests.support import build_env, run_cli


def _clock() -> FrozenClock:
    return FrozenClock(datetime.datetime(2026, 2, 2, 3, 4, 5, tzinfo=datetime.timezone.utc))


class PromotionUnderTheRunLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name).resolve()
        cls.cfg = [
            "--config",
            str(
                build_env(
                    cls.base,
                    rubric_ids=("instruction-compliance",),
                    use_subprocess_model=True,
                    dry_run=False,
                    seed="run-lock",
                )
            ),
        ]
        daily = run_cli([*cls.cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
        assert daily.returncode == RC_OK, daily.stderr
        listed = json.loads(run_cli([*cls.cfg, "--json", "candidates"]).stdout)["candidates"]
        cls.candidate_id = sorted(c["id"] for c in listed)[0]
        review = json.loads(run_cli([*cls.cfg, "--json", "review", cls.candidate_id]).stdout)
        cls.token = review["approval_token"]
        cls.state = StateStore(cls.base / "state", clock=_clock())

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _skill_text(self) -> str:
        record = json.loads(self.state.candidate_path(self.candidate_id).read_text("utf-8"))
        return Path(record["target"]["canonical_path"]).read_text(encoding="utf-8")

    def _status(self) -> str:
        return json.loads(self.state.candidate_path(self.candidate_id).read_text("utf-8"))["status"]

    def test_apply_and_rollback_refuse_while_a_run_holds_the_lock_and_write_nothing(self) -> None:
        before = self._skill_text()
        apply = [
            *self.cfg,
            "apply",
            self.candidate_id,
            "--approve",
            self.token,
            "--i-reviewed-this",
        ]
        with RunLock(self.state.lock_path(), owner="pid=4242 the morning run"):
            refused = run_cli(apply)
        self.assertEqual(refused.returncode, RC_REFUSED, refused.stderr)
        self.assertIn(
            f"LockHeld: refusing to apply {self.candidate_id}: another asi process holds "
            f"{self.state.lock_path()} (pid=4242 the morning run)",
            refused.stderr,
        )
        self.assertIn("would record that as a skill tree mutation", refused.stderr)
        self.assertIn("Wait for the holder named above to finish", refused.stderr)
        self.assertIn("nothing was written", refused.stderr)
        self.assertNotIn("Traceback", refused.stderr)
        self.assertEqual(self._skill_text(), before, "the skill was not touched")
        self.assertIsNone(self.state.latest_backup_manifest(self.candidate_id), "no backup made")
        self.assertEqual(self._status(), "validated")
        self.assertFalse(self.state.action_receipt_path("apply", self.candidate_id).exists())

        applied = run_cli(apply)
        self.assertEqual(applied.returncode, RC_OK, applied.stderr)
        changed = self._skill_text()
        self.assertNotEqual(changed, before)
        self.assertEqual(self._status(), "applied")

        with RunLock(self.state.lock_path(), owner="pid=4242 the morning run"):
            refused = run_cli([*self.cfg, "rollback", self.candidate_id])
        self.assertEqual(refused.returncode, RC_REFUSED, refused.stderr)
        self.assertIn(
            f"LockHeld: refusing to rollback {self.candidate_id}: another asi process holds",
            refused.stderr,
        )
        self.assertEqual(self._skill_text(), changed, "the rollback did not happen")
        self.assertEqual(self._status(), "applied")
        self.assertFalse(self.state.action_receipt_path("rollback", self.candidate_id).exists())

        rolled = run_cli([*self.cfg, "rollback", self.candidate_id])
        self.assertEqual(rolled.returncode, RC_OK, rolled.stderr)
        self.assertEqual(self._skill_text(), before)
        self.assertEqual(self._status(), "rolled-back")

    def test_the_lock_names_the_promotion_and_a_run_meanwhile_is_refused_naming_it(self) -> None:
        candidate = self.state.read_candidate(self.candidate_id)
        with _holding_the_run_lock(self.state, "apply", candidate):
            owner = self.state.lock_path().read_text(encoding="utf-8").strip()
            self.assertEqual(owner, f"apply {self.candidate_id} pid={os.getpid()}")
            with self.assertRaises(LockHeld) as caught:
                with RunLock(self.state.lock_path()):
                    pass
            self.assertIn(f"(apply {self.candidate_id} pid={os.getpid()})", str(caught.exception))
            # A dry run takes the lock too, and keeps the blocked-receipt gate
            # (which runs before the lock) out of this assertion's path.
            run = run_cli([*self.cfg, "daily", "--day", "2026-02-03"])
            self.assertEqual(run.returncode, RC_REFUSED, run.stderr)
            self.assertIn("LockHeld: another asi process holds", run.stderr)
            self.assertIn(f"apply {self.candidate_id} pid=", run.stderr)
        # Released on exit: the next holder takes it.
        with RunLock(self.state.lock_path()):
            pass

    def test_rollback_inside_an_automated_run_is_refused_as_automation_before_the_lock(
        self,
    ) -> None:
        candidate = self.state.read_candidate(self.candidate_id)
        leaf = self.state.lock_path()
        before = leaf.read_bytes() if leaf.exists() else None
        with AutomationMarker():
            with self.assertRaises(AutomationRefused) as caught:
                rollback_candidate(
                    candidate, skill_store=object(), state=self.state, clock=_clock()
                )
        self.assertIn("rollback refused: automation marker is set", str(caught.exception))
        # Refused before the lock was reached: taking it would have rewritten
        # the owner line.
        self.assertEqual(leaf.read_bytes() if leaf.exists() else None, before)

    def test_the_backup_manifest_is_chosen_under_the_lock_not_before_it(self) -> None:
        # A candidate never applied has no manifest; while a run holds the lock
        # the answer is LockHeld, not GateFailed, because the manifest is looked
        # up only once the lock is held (chosen before it, a rollback waiting on
        # an apply would restore from the backup that apply made obsolete).
        never_applied = replace(
            self.state.read_candidate(self.candidate_id), id="CAND-2026-02-01-77-x"
        )
        with RunLock(self.state.lock_path(), owner="pid=5 run"):
            with self.assertRaises(LockHeld):
                rollback_candidate(
                    never_applied, skill_store=object(), state=self.state, clock=_clock()
                )
        with self.assertRaises(GateFailed) as caught:
            rollback_candidate(
                never_applied, skill_store=object(), state=self.state, clock=_clock()
            )
        self.assertIn("no backup manifest found", str(caught.exception))

    def test_a_lock_held_refusal_is_raised_with_the_action_and_the_lock_released(self) -> None:
        candidate = self.state.read_candidate(self.candidate_id)
        # The owner line is read back from a file, so a break in it is spelled.
        with RunLock(self.state.lock_path(), owner="pid=7 held\nPROBLEM forged"):
            with self.assertRaises(LockHeld) as caught:
                with _holding_the_run_lock(self.state, "rollback", candidate):
                    self.fail("the body ran while the lock was held")
        message = str(caught.exception)
        self.assertTrue(message.startswith(f"refusing to rollback {self.candidate_id}: "), message)
        self.assertIn("(pid=7 held\\nPROBLEM forged)", message)
        self.assertNotIn("\n", message)
        # The failed attempt left nothing open: the lock is free again.
        with RunLock(self.state.lock_path(), owner="pid=8 next"):
            self.assertEqual(
                self.state.lock_path().read_text(encoding="utf-8").strip(), "pid=8 next"
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
