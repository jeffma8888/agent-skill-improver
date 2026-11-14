"""Automation refusals name the attempted action, not a human-authentication guarantee."""

from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app.promote import apply_candidate
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.domain.approval import HumanApproval
from agent_skill_improver.domain.integrity import verify_candidate_integrity
from agent_skill_improver.domain.skills import SkillRoot
from agent_skill_improver.errors import AutomationRefused, CandidateTampered
from agent_skill_improver.hashing import sha256_text
from tests.support import InProcessRunner, build_env, run_cli
from tests.test_candidate_integrity import BASE_TEXT, NEW_TEXT, build_candidate, frozen_clock

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


class OutputNamesReviewInterlocksNotAuthenticatedHumansTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        clean = mock.patch.dict(os.environ, {"ASI_AUTOMATED_RUN": "0"})
        clean.start()
        self.addCleanup(clean.stop)

    def test_selftest_names_a_separate_review_interlocked_path(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cli.main(["--json", "selftest"])
        self.assertEqual(rc, 0, out.getvalue())
        check = next(
            c
            for c in json.loads(out.getvalue())["checks"]
            if c["name"] == "daily does not import the promotion path"
        )
        self.assertEqual(
            check["requirement"],
            "app/daily.py must not import app.promote; "
            "promotion is a separate review-interlocked path",
        )
        self.assertNotIn("human-only", check["requirement"])

    def test_daily_refusal_prints_review_interlocks_not_human_authentication(self) -> None:
        config = build_env(self.base)
        err = io.StringIO()
        with (
            mock.patch.object(cli, "build_runner", return_value=InProcessRunner()),
            mock.patch.object(
                cli, "build_skill_store", return_value=SimpleNamespace(read_only=False)
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(err),
        ):
            rc = cli.main(["--config", str(config), "daily", "--commit", "--day", "2026-02-01"])
        self.assertEqual(rc, 3, err.getvalue())
        self.assertIn(
            "the daily driver refuses a writable skill store. Automation may propose and "
            "validate; only `asi apply` may write, with an approval token and acknowledgement, "
            "a matching base hash and no active automation marker. These are review interlocks, "
            "not caller authentication.",
            err.getvalue(),
        )
        self.assertNotIn("human approval token", err.getvalue())
        self.assertNotIn("human", err.getvalue().lower())

    def _store(self) -> tuple[FilesystemSkillStore, Path]:
        root = self.base / "skills"
        target = root / "tdd" / "SKILL.md"
        target.parent.mkdir(parents=True)
        target.write_text(BASE_TEXT, encoding="utf-8")
        return (
            FilesystemSkillStore(
                [SkillRoot(id="canonical", path=root, canonical=True, kind="custom")],
                read_only=False,
            ),
            target,
        )

    def _apply(self) -> tuple[StateStore, dict]:
        store, target = self._store()
        candidate = build_candidate(canonical_path=str(target), mirror_paths=())
        clock = frozen_clock()
        state = StateStore(self.base / "state", clock=clock)
        state.ensure()
        state.write_candidate(candidate)
        approval = HumanApproval.from_cli(
            token=candidate.required_approval_token, acknowledged=True, operator="batch-job"
        )
        receipt = apply_candidate(
            candidate,
            approval=approval,
            skill_store=store,
            state=state,
            clock=clock,
            operator="batch-job",
        )
        self.assertEqual(target.read_text(encoding="utf-8"), NEW_TEXT)
        self.assertEqual(receipt["approval"]["origin"], "cli")
        self.assertTrue(receipt["approval"]["acknowledged"])
        return state, receipt

    def test_apply_manifest_names_the_operator_without_claiming_a_human(self) -> None:
        _, receipt = self._apply()
        manifest = json.loads(Path(receipt["write"]["backup_manifest_path"]).read_text())
        self.assertEqual(manifest["actor"], "operator:batch-job")
        self.assertNotIn("human:", manifest["actor"])

    def test_apply_note_names_the_operator_without_claiming_a_human(self) -> None:
        state, receipt = self._apply()
        note = state.read_candidate(receipt["candidate_id"]).notes[-1]
        self.assertEqual(
            note,
            "applied 2026-02-01T00:00:00Z by operator:batch-job; "
            + f"rollback: {receipt['write']['rollback_command']}",
        )
        self.assertNotIn("by human:", note)

    def test_a_direct_write_records_an_unspecified_actor_not_an_assumed_human(self) -> None:
        store, target = self._store()
        receipt = store.write_canonical(
            target,
            NEW_TEXT,
            expected_sha256=sha256_text(BASE_TEXT),
            backup_dir=self.base / "backups",
            skill_id="tdd",
        )
        manifest = json.loads(Path(receipt.backup_manifest_path).read_text())
        self.assertEqual(manifest["actor"], "unspecified")
        self.assertEqual(target.read_text(encoding="utf-8"), NEW_TEXT)

    def test_a_diff_refusal_names_shown_bytes_not_a_known_human_reviewer(self) -> None:
        candidate = build_candidate()
        tampered = replace(candidate, patch=replace(candidate.patch, unified_diff="not the diff"))
        with self.assertRaises(CandidateTampered) as caught:
            verify_candidate_integrity(tampered, base_text=BASE_TEXT, action="review")
        self.assertIn("diff shown for review does not describe this change", str(caught.exception))
        self.assertNotIn("diff a human reviewed", str(caught.exception))
        self.assertNotIn("human", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main()
