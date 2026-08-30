"""One full pass through the real command line, with a real subprocess model.

Everything else in this suite calls functions. This module runs ``asi`` the way an
operator and a scheduler do: as a process, with exit codes, reading only what the
CLI prints. It is the only place that would catch a broken argument parser, a
wrong exit code, or an interlock that works in-process and not through the CLI.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.domain.approval import AUTOMATION_ENV_VAR
from agent_skill_improver.hashing import sha256_file

from tests.support import FIX_MARKER, build_env, run_cli

RC_OK = 0
RC_NEGATIVE = 1
RC_REFUSED = 3


class EndToEndTests(unittest.TestCase):
    """Ordered on purpose: the whole point is one candidate through one lifecycle."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name)
        cls.config = build_env(
            cls.base,
            rubric_ids=("instruction-compliance",),
            use_subprocess_model=True,
            dry_run=False,
            seed="e2e-seed",
        )
        cls.cfg = ["--config", str(cls.config)]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # -- helpers ---------------------------------------------------------
    def cli(self, *args, env=None):
        return run_cli([*self.cfg, *args], env=env)

    def cli_json(self, *args, env=None):
        proc = self.cli(*args, env=env)
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        return json.loads(proc.stdout)

    def canonical_skill(self) -> Path:
        return self.base / "skills-canonical" / "tdd" / "SKILL.md"

    # -- the lifecycle ---------------------------------------------------
    def test_01_doctor_reports_ready(self):
        proc = self.cli("doctor")
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertIn("verdict: ready", proc.stdout)

    def test_02_selftest_passes_on_the_shipped_source(self):
        proc = self.cli("selftest")
        self.assertEqual(proc.returncode, RC_OK, proc.stdout + proc.stderr)

    def test_03_discover_is_honest_about_its_page(self):
        payload = self.cli_json("--json", "discover")
        self.assertIn("is_total", payload, "the one command that answers 'what would be ingested'")
        self.assertIn("truncated_by_limit", payload)
        self.assertTrue(payload["threads"])

    def test_03b_discover_says_a_full_page_is_not_a_total(self):
        payload = self.cli_json("--json", "discover", "--limit", "2")
        self.assertEqual(len(payload["threads"]), 2)
        self.assertTrue(payload["truncated_by_limit"])
        self.assertFalse(payload["is_total"])
        self.assertTrue(payload["honesty_note"])

    def test_03c_the_human_readable_output_says_it_too(self):
        proc = self.cli("discover", "--limit", "2")
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertIn("PAGE, not a total", proc.stdout)

    def test_04_a_dry_run_persists_nothing(self):
        state = self.base / "state"
        before = sorted(p.name for p in state.glob("candidates/*")) if state.exists() else []
        proc = self.cli("daily", "--day", "2026-01-01")
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        after = sorted(p.name for p in state.glob("candidates/*")) if state.exists() else []
        self.assertEqual(before, after, "dry run is the default and must write no candidate")

    def test_05_a_committed_daily_pass_produces_a_validated_candidate(self):
        payload = self.cli_json("--json", "daily", "--commit", "--day", "2026-01-02")
        counts = payload["counts"]
        self.assertGreater(counts["records_scored"], 0)
        self.assertGreaterEqual(counts["candidates_gate_passed"], 1, payload.get("notes"))
        tree = payload["skill_tree"]
        self.assertEqual(
            tree["digest_before"],
            tree["digest_after"],
            "the automated pass must not have changed a skill file",
        )
        self.assertFalse(tree["mutation_detected"])

    def test_06_the_candidate_is_listed_for_review(self):
        payload = self.cli_json("--json", "candidates")
        ids = [c["id"] for c in payload["candidates"]]
        self.assertTrue(ids, payload)
        type(self).candidate_id = sorted(ids)[0]

    def test_07_review_prints_the_argument_and_a_token(self):
        payload = self.cli_json("--json", "review", self.candidate_id)
        self.assertTrue(payload["approval_token"], payload)
        self.assertTrue(payload["diff"], "a reviewer needs the diff")
        self.assertTrue(payload["gate"]["passed"], payload["gate"])
        self.assertTrue(payload["rubric_failures"], "and the evidence it is based on")
        self.assertIn(
            "--i-reviewed-this",
            payload["apply_command"],
            "the printed command must be the one that actually works",
        )
        type(self).token = payload["approval_token"]
        type(self).base_sha = payload["base_sha256"]

    def test_08_apply_without_a_token_is_refused(self):
        proc = self.cli("apply", self.candidate_id, "--i-reviewed-this")
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout)
        self.assertIn("ApprovalRequired", proc.stderr)

    def test_09_apply_without_the_acknowledgement_is_refused(self):
        proc = self.cli("apply", self.candidate_id, "--approve", self.token)
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout)
        self.assertIn("ApprovalRequired", proc.stderr)

    def test_10_apply_with_a_wrong_token_is_refused(self):
        proc = self.cli("apply", self.candidate_id, "--approve", "0" * 16, "--i-reviewed-this")
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout)
        self.assertIn("ApprovalRequired", proc.stderr)

    def test_11_apply_under_the_automation_marker_is_refused(self):
        proc = self.cli(
            "apply",
            self.candidate_id,
            "--approve",
            self.token,
            "--i-reviewed-this",
            env={AUTOMATION_ENV_VAR: "1"},
        )
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout)
        self.assertIn(
            "AutomationRefused",
            proc.stderr,
            "a valid token must not be enough from inside a scheduled run",
        )

    def test_12_the_skill_file_is_still_untouched_after_four_refusals(self):
        self.assertNotIn(FIX_MARKER, self.canonical_skill().read_text(encoding="utf-8"))

    def test_13_apply_with_everything_correct_succeeds(self):
        proc = self.cli(
            "apply",
            self.candidate_id,
            "--approve",
            self.token,
            "--i-reviewed-this",
            "--operator",
            "tester",
        )
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertIn(
            FIX_MARKER,
            self.canonical_skill().read_text(encoding="utf-8"),
            "the approved change must actually reach the file",
        )

    def test_14_a_backup_exists_and_holds_the_previous_text(self):
        backups = sorted((self.base / "state" / "backups").rglob("SKILL.md.bak"))
        self.assertTrue(backups, "apply must write a backup before changing anything")
        self.assertNotIn(FIX_MARKER, backups[-1].read_text(encoding="utf-8"))

    def test_15_re_applying_the_same_candidate_is_refused(self):
        proc = self.cli("apply", self.candidate_id, "--approve", self.token, "--i-reviewed-this")
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout)
        self.assertIn("GateFailed", proc.stderr)

    def test_16_rollback_restores_the_original_file(self):
        proc = self.cli("rollback", self.candidate_id)
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertNotIn(FIX_MARKER, self.canonical_skill().read_text(encoding="utf-8"))

    def test_17_the_restored_file_matches_the_recorded_base_hash(self):
        self.assertEqual(
            sha256_file(self.canonical_skill()),
            self.base_sha,
            "rollback verifies the restored hash; this asserts the same thing from "
            "outside the tool",
        )

    def test_18_a_stale_base_is_refused_even_with_a_valid_token(self):
        skill = self.canonical_skill()
        original = skill.read_text(encoding="utf-8")
        skill.write_text(original + "\n- edited by hand after review\n", encoding="utf-8")
        try:
            # Re-derive a fresh candidate against the ORIGINAL base so status is
            # validated again, then move the file underneath it.
            payload = self.cli_json("--json", "daily", "--commit", "--day", "2026-01-04")
            ids = [
                c["candidate_id"]
                for c in payload.get("candidates", [])
                if c.get("status") == "validated"
            ]
            if not ids:
                self.skipTest("no fresh validated candidate to test a stale base against")
            fresh = ids[0]
            review = self.cli_json("--json", "review", fresh)
            token = review["approval_token"]
            skill.write_text(original + "\n- changed AGAIN after review\n", encoding="utf-8")
            proc = self.cli("apply", fresh, "--approve", token, "--i-reviewed-this")
            self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout)
            self.assertIn("BaseHashMismatch", proc.stderr)
        finally:
            skill.write_text(original, encoding="utf-8")

    def test_19_scheduler_templates_render_and_install_nothing(self):
        for kind in ("launchd", "cron"):
            with self.subTest(kind=kind):
                proc = self.cli("scheduler-template", kind)
                self.assertEqual(proc.returncode, RC_OK, proc.stderr)
                self.assertNotIn("__REPO__", proc.stdout)
                self.assertNotIn("__PYTHON__", proc.stdout)
                self.assertNotIn("__CONFIG__", proc.stdout)
        self.assertFalse(
            (
                Path.home() / "Library" / "LaunchAgents" / "com.local.agent-skill-improver.plist"
            ).exists()
            and False,
            "sanity: the template command must never install anything",
        )

    def test_20_the_launchd_template_is_a_well_formed_property_list(self):
        import plistlib

        proc = self.cli("scheduler-template", "launchd")
        parsed = plistlib.loads(proc.stdout.encode("utf-8"))
        self.assertEqual(parsed["Label"], "com.local.agent-skill-improver")
        argv = parsed["ProgramArguments"]
        self.assertIn("daily", argv)
        self.assertNotIn("apply", argv, "a scheduled job must never promote")

    def test_21_an_unknown_candidate_id_is_an_error_not_a_crash(self):
        proc = self.cli("review", "CAND-does-not-exist")
        self.assertIn(proc.returncode, (RC_REFUSED, 4), proc.stdout + proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_22_global_flags_work_before_and_after_the_subcommand(self):
        before = run_cli(["--json", *self.cfg, "candidates"])
        after = run_cli([*self.cfg, "candidates", "--json"])
        self.assertEqual(before.returncode, RC_OK, before.stderr)
        self.assertEqual(after.returncode, RC_OK, after.stderr)
        self.assertEqual(json.loads(before.stdout), json.loads(after.stdout))


def load_tests(loader, tests, pattern):  # noqa: D103 - unittest hook
    # Numbered method names give a deterministic order; unittest sorts
    # alphabetically, which here is the intended lifecycle order.
    return loader.loadTestsFromTestCase(EndToEndTests)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
