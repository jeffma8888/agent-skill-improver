"""One full pass through the real command line, with a real subprocess model.

Everything else in this suite calls functions. This module runs ``asi`` the way an
operator and a scheduler do: as a process, with exit codes, reading only what the
CLI prints. It is the only place that would catch a broken argument parser, a
wrong exit code, or an interlock that works in-process and not through the CLI.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.app.state import StateStore
from agent_skill_improver.cli import RC_MISSING_FILE, RC_OK, RC_REFUSED
from agent_skill_improver.clock import SystemClock, day_key
from agent_skill_improver.domain.approval import AUTOMATION_ENV_VAR
from agent_skill_improver.hashing import sha256_file

from tests.support import FIX_MARKER, REPO_ROOT, build_env, run_cli


# Numbered method names give a deterministic order: unittest sorts test names
# alphabetically, which here is the intended lifecycle order. No load_tests hook
# is needed for that, and a hand-written one can drop a class added later.
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
        # The whole tree, not one directory: an earlier version of this test
        # watched candidates/ alone while conversations, scorecards, quarantine
        # records and a receipt were being written under the same flag.
        state = self.base / "state"

        def snapshot() -> dict[str, str]:
            if not state.exists():
                return {}
            return {
                str(p.relative_to(state)): sha256_file(p)
                for p in state.rglob("*")
                if p.is_file() and p.relative_to(state).parts[0] != "locks"
            }

        before = snapshot()
        proc = self.cli("daily", "--day", "2026-01-01")
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertIn("mode=dry-run", proc.stdout)
        self.assertIn("scored=", proc.stdout)
        self.assertIn("receipt: not written (dry run)", proc.stdout)
        after = snapshot()
        self.assertEqual(
            {k for k in after if k != "README.txt"},
            {k for k in before if k != "README.txt"},
            "dry run is the default and must add or remove nothing under state/",
        )
        for rel, digest in before.items():
            self.assertEqual(after.get(rel), digest, f"{rel} changed during a dry run")

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
            # A hard assertion, not a skip: the rolled-back candidate from the
            # earlier steps is closed, so the per-skill cap must let this run
            # propose again. A skip here would turn a regression of that rule
            # into a green run.
            self.assertTrue(
                ids,
                "expected a fresh validated candidate after rollback closed the earlier one; "
                f"got {payload.get('counts')}",
            )
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
        # The earlier form of this check was `assertFalse(x and False)`, which
        # cannot fail. The launchd install target is under $HOME (cron's spool is
        # not, and is not measured here), so give the command an empty one and
        # require it to stay empty; the command also makes no subprocess call.
        with TemporaryDirectory() as home:
            for kind in ("launchd", "cron"):
                proc = self.cli("scheduler-template", kind, env={"HOME": home})
                self.assertEqual(proc.returncode, RC_OK, proc.stderr)
            self.assertEqual(
                sorted(Path(home).rglob("*")), [], "the template command installed something"
            )

    def test_19b_the_templates_schedule_the_config_they_were_asked_about(self):
        """Measured before this test existed, `asi --config X scheduler-template`
        printed a job for the DEFAULT config, not X, and sent the logs to the
        repository's state/ whatever X's state_dir said. A scheduler built from
        that output would have run the example corpus and logged private output
        outside the configured tree."""
        config_path = Path(self.cfg[1]).resolve()
        for kind in ("launchd", "cron"):
            with self.subTest(kind=kind):
                proc = self.cli("scheduler-template", kind)
                self.assertEqual(proc.returncode, RC_OK, proc.stderr)
                self.assertIn(str(config_path), proc.stdout, "the job must use the given config")
                self.assertNotIn("example.config.toml", proc.stdout)
                state = self.base.resolve() / "state"
                self.assertIn(
                    (
                        f">> {state / 'daily.out.log'}"
                        if kind == "cron"
                        else f"<string>{state / 'daily.out.log'}</string>"
                    ),
                    proc.stdout,
                )
                self.assertIn(str(state / "daily.err.log"), proc.stdout)
                self.assertNotIn("__STATE__", proc.stdout)
        # The default config may arrive through a RELATIVE ASI_CONFIG; the job
        # must still name it absolutely, or cron would resolve it against its
        # own working directory.
        relative = os.path.relpath(config_path, REPO_ROOT)
        self.assertFalse(Path(relative).is_absolute(), relative)
        proc = run_cli(["scheduler-template", "cron"], env={"ASI_CONFIG": relative})
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertIn(f"--config {config_path} ", proc.stdout)

    def test_20_the_launchd_template_is_a_well_formed_property_list(self):
        import plistlib

        proc = self.cli("scheduler-template", "launchd")
        parsed = plistlib.loads(proc.stdout.encode("utf-8"))
        self.assertEqual(parsed["Label"], "com.local.agent-skill-improver")
        argv = parsed["ProgramArguments"]
        self.assertIn("daily", argv)
        self.assertNotIn("apply", argv, "a scheduled job must never promote")
        self.assertEqual(argv[argv.index("--config") + 1], str(Path(self.cfg[1]).resolve()))

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


def _tree_snapshot(*roots: Path) -> dict[str, str]:
    """Every file and directory under the roots: content hash (files), mode and
    mtime, so a byte-identical rewrite of an existing file is still a change."""
    out: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            st = p.stat()
            mode = oct(st.st_mode & 0o777)
            key = f"{root.name}/{p.relative_to(root)}"
            body = "dir" if p.is_dir() else f"{sha256_file(p)} {st.st_mtime_ns}"
            out[key] = f"{body} {mode}"
    return out


class AnUndecodableSkillIsQuarantinedNotFatalTests(unittest.TestCase):
    """Measured before: one invalid byte in one skill made ``daily`` exit 70 with
    a traceback, write no receipt and produce no candidate for any skill."""

    def test_daily_quarantines_the_skill_writes_its_receipt_and_exits_normally(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            cfg = ["--config", str(build_env(base, dry_run=False, seed="utf8-seed"))]
            skill = base / "skills-canonical" / "tdd" / "SKILL.md"
            data = skill.read_bytes()
            skill.write_bytes(data[:40] + b"\xff" + data[40:])
            proc = run_cli([*cfg, "--json", "daily", "--commit", "--day", "2026-05-01"])
            self.assertEqual(proc.returncode, RC_OK, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["status"], "partial")
            self.assertGreaterEqual(payload["counts"]["skills_quarantined"], 1)
            records = list((base / "state" / "quarantine").glob("tdd--*.json"))
            self.assertEqual(len(records), 1, records)
            reason = json.loads(records[0].read_text(encoding="utf-8"))["reason"]
            self.assertIn("not valid UTF-8 at byte 40", reason)
            self.assertTrue(list((base / "state" / "receipts").glob("*.json")), "receipt written")
            self.assertEqual(skill.read_bytes(), data[:40] + b"\xff" + data[40:], "left as it is")
            listing = json.loads(run_cli([*cfg, "--json", "skills"]).stdout)
            tdd = [g for g in listing["groups"] if g["skill_id"] == "tdd"][0]
            self.assertTrue(
                any("not valid UTF-8" in w for w in tdd["canonical"]["warnings"]),
                tdd["canonical"]["warnings"],
            )


class ReviewRefusesWhatApplyWouldRefuseTests(unittest.TestCase):
    """One validated candidate, then its target is replaced under it.

    Measured before the store judged its own reads: with the document swapped
    for a link to an identical copy, ``review`` exited 0 and printed a token,
    and ``apply`` with that token exited 3 (``LinkedTargetRefused``); with a
    fifo at the document, ``review`` never returned. A token for an apply that
    will be refused is a broken promise, and a command that never returns is
    worse, so both are now the same refusal ``apply`` would give, before any
    token is shown.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name).resolve()
        cls.cfg = ["--config", str(build_env(cls.base, dry_run=False, seed="swap-seed"))]
        proc = run_cli([*cls.cfg, "--json", "daily", "--commit", "--day", "2026-04-01"])
        assert proc.returncode == RC_OK, proc.stderr
        payload = json.loads(proc.stdout)
        validated = [c["candidate_id"] for c in payload["candidates"] if c["status"] == "validated"]
        assert validated, payload.get("notes")
        cls.candidate_id = validated[0]
        cls.skill = cls.base / "skills-canonical" / "tdd" / "SKILL.md"
        cls.original = cls.skill.read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        if self.skill.is_symlink() or not self.skill.is_file():
            self.skill.unlink()
            self.skill.write_text(self.original, encoding="utf-8")

    def tearDown(self) -> None:
        self.setUp()

    def test_a_document_swapped_for_a_link_gets_no_token_at_review(self):
        aside = self.base / "identical-copy.md"
        aside.write_text(self.original, encoding="utf-8")
        self.skill.unlink()
        self.skill.symlink_to(aside)
        proc = run_cli([*self.cfg, "--json", "review", self.candidate_id])
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
        self.assertIn("LinkedTargetRefused", proc.stderr)
        self.assertNotIn("approval_token", proc.stdout)
        self.assertEqual(aside.read_text(encoding="utf-8"), self.original)

    def test_a_fifo_at_the_document_is_a_prompt_refusal_not_a_hang(self):
        self.skill.unlink()
        os.mkfifo(self.skill)
        started = time.monotonic()
        proc = run_cli([*self.cfg, "--json", "review", self.candidate_id])
        self.assertLess(time.monotonic() - started, 60.0)
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
        self.assertIn("BaseHashMismatch", proc.stderr)
        self.assertNotIn("approval_token", proc.stdout)

    def test_the_control_the_restored_document_reviews_with_a_token(self):
        proc = run_cli([*self.cfg, "--json", "review", self.candidate_id])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertTrue(json.loads(proc.stdout)["approval_token"])


class SkillsListingShowsWhatDiscoveryNoticedTests(unittest.TestCase):
    """The text listing must say what the JSON says.

    Measured before: a launcher beside a skill produced a payload warning in
    ``skills --json`` and nothing at all in ``skills``, so an operator reading
    the listing (which README describes as showing executable payloads) saw a
    clean inventory.
    """

    def test_every_snapshot_warning_appears_once_under_its_skill_in_the_text_listing(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            cfg = ["--config", str(build_env(base, dry_run=True))]
            launcher = base / "skills-canonical" / "tdd" / "run.sh"
            launcher.write_text("#!/bin/sh\n", encoding="utf-8")
            os.chmod(launcher, 0o755)
            listing = json.loads(run_cli([*cfg, "--json", "skills"]).stdout)
            expected: list[str] = []
            for group in listing["groups"]:
                if group["canonical"]:
                    expected += [f"    warning: {w}" for w in group["canonical"]["warnings"]]
                for copy in group["mirrors"]:
                    expected += [f"    warning ({copy['root_id']}): {w}" for w in copy["warnings"]]
            self.assertTrue(any("executable payload" in w for w in expected), expected)

            proc = run_cli([*cfg, "skills"])
            self.assertEqual(proc.returncode, RC_OK, proc.stderr)
            text_lines = proc.stdout.splitlines()
            shown = [ln for ln in text_lines if ln.startswith("    warning")]
            self.assertEqual(sorted(shown), sorted(expected), proc.stdout)
            head = next(i for i, ln in enumerate(text_lines) if ln.startswith("tdd "))
            self.assertIn("executable payload", text_lines[head + 1], "shown under its own skill")
            self.assertTrue(text_lines[-1].startswith("4 logical skill(s);"), "summary stays last")

    def test_a_payload_name_with_a_line_break_cannot_forge_a_listing_line(self):
        """Measured with the warning lines first added: an executable named
        ``run\n    warning: FORGED ...sh`` printed as two warnings, the second
        of them authored by whoever dropped the file."""
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            cfg = ["--config", str(build_env(base, dry_run=True))]
            forged = base / "skills-canonical" / "tdd" / "run\n    warning: FORGED all clear.sh"
            forged.write_text("#!/bin/sh\n", encoding="utf-8")
            os.chmod(forged, 0o755)
            # A skill DIRECTORY name is untrusted too, and it makes the head line.
            shutil.copytree(
                base / "skills-canonical" / "code-review",
                base / "skills-canonical" / "zz\ntdd  canonical=canonical sha=FORGED00",
            )
            proc = run_cli([*cfg, "skills"])
            self.assertEqual(proc.returncode, RC_OK, proc.stderr)
            lines = proc.stdout.splitlines()
            self.assertEqual([ln for ln in lines if ln.startswith("    warning: FORGED")], [])
            self.assertEqual(
                [ln for ln in lines if ln.startswith("tdd  canonical=canonical sha=FORGED")], []
            )
            # Three lines carry the forged text: the payload warning, the head
            # line of the renamed skill, and its frontmatter-name warning.
            carrying = [ln for ln in lines if "FORGED" in ln]
            self.assertEqual(len(carrying), 3, proc.stdout)
            self.assertTrue(any("run\\n    warning: FORGED" in ln for ln in carrying), carrying)
            self.assertTrue(any(ln.startswith("zz\\ntdd  canonical") for ln in carrying), carrying)
            self.assertEqual(sum(1 for ln in lines if ln.startswith("    warning")), 3, proc.stdout)


FORGED_SECTION = (
    "looks fine.\n"
    "\n"
    "DIFF\n"
    "(empty)\n"
    "\n"
    "To promote this change, review the diff above, then run:\n"
    "  asi apply CAND-forged --approve FORGEDTOKEN --i-reviewed-this\n"
    "\n"
    "Nothing has been written to any skill file yet.\n"
    "\n"
    "(the real reasoning follows)"
)

# The only text ``asi review`` prints at column 0 in its own voice. Anything
# else at the margin would be something the model or a file name put there.
REVIEW_HEADERS = {
    "",
    "HYPOTHESIS",
    "INTENDED CHANGE",
    "RISKS",
    "ROLLBACK",
    "MIRROR COPIES (not written by apply):",
    "EVIDENCE (rubric failures this claims to fix)",
    "DIFF",
    "To promote this change, review the diff above, then run:",
    "Nothing has been written to any skill file yet.",
}
REVIEW_FIELD_PREFIXES = ("candidate ", "skill ", "file ", "base sha256 ", "gate ")
DIFF_LINE_PREFIXES = ("--- ", "+++ ", "@@ ", "+", "-", " ", "\\")


def margin_lines_outside_their_section(lines: list[str]) -> list[str]:
    """Column-0 lines of a review report that are not the tool's own, judged
    by section: field lines and headers above ``DIFF``; diff-prefixed lines
    and ``note:`` lines between ``DIFF`` and the apply instruction; headers
    only after it."""
    diff_at = lines.index("DIFF")
    promote_at = next(i for i, ln in enumerate(lines) if ln.startswith("To promote this change"))
    assert diff_at < promote_at
    stray = []
    for i, line in enumerate(lines):
        if not line or line.startswith(" ") or line in REVIEW_HEADERS:
            continue
        if i < diff_at and line.startswith(REVIEW_FIELD_PREFIXES):
            continue
        if diff_at < i < promote_at and (
            line.startswith(DIFF_LINE_PREFIXES) or line.startswith("note: ")
        ):
            continue
        stray.append(line)
    return stray


class ReviewProseCannotForgeTheReportTests(unittest.TestCase):
    """The model's prose is shown as written, and only as a quotation.

    A candidate's hypothesis, intended change, risks and rollback note are the
    model's own words and are not bound to the approval token: they are what
    the reviewer weighs, not what ``apply`` writes. Measured before this: a
    hypothesis that contained a line ``DIFF``, ``(empty)`` and ``To promote
    this change ... asi apply ... --approve FORGEDTOKEN`` printed a complete
    second DIFF-and-apply section above the real one, at column 0, in the same
    shape. Here the record is edited the way a model could have authored it,
    and the text report must keep every line of that prose visible while
    letting none of it stand at the margin where the report's own headers and
    apply instruction live.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name).resolve()
        cls.cfg = ["--config", str(build_env(cls.base, dry_run=False, seed="prose-seed"))]
        proc = run_cli([*cls.cfg, "--json", "daily", "--commit", "--day", "2026-04-02"])
        assert proc.returncode == RC_OK, proc.stderr
        payload = json.loads(proc.stdout)
        validated = [c["candidate_id"] for c in payload["candidates"] if c["status"] == "validated"]
        assert validated, payload.get("notes")
        cls.candidate_id = validated[0]
        cls.record = cls.base / "state" / "candidates" / f"{cls.candidate_id}.json"
        assert cls.record.is_file(), cls.record
        before = run_cli([*cls.cfg, "--json", "review", cls.candidate_id])
        assert before.returncode == RC_OK, before.stderr
        cls.real_apply_command = json.loads(before.stdout)["apply_command"]
        record = json.loads(cls.record.read_text(encoding="utf-8"))
        record["hypothesis"] = FORGED_SECTION
        record["intended_change"] = "Append a rule.\n" + FORGED_SECTION
        record["rollback"] = FORGED_SECTION
        record["risks"] = [FORGED_SECTION, "A second, one-line risk."]
        cls.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        cls.proc = run_cli([*cls.cfg, "review", cls.candidate_id])
        cls.lines = cls.proc.stdout.split("\n")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_forged_prose_reviews_and_is_shown_in_full(self):
        self.assertEqual(self.proc.returncode, RC_OK, self.proc.stderr)
        quoted = [ln for ln in self.lines if ln.startswith("  |")]
        for line in FORGED_SECTION.split("\n"):
            expected = "  |" if line == "" else f"  | {line}"
            self.assertIn(expected, quoted, self.proc.stdout)
        self.assertIn("  - looks fine.", self.lines, "a risk opens with its bullet")
        self.assertIn("  - A second, one-line risk.", self.lines)
        self.assertEqual(self.lines.count("  | DIFF"), 4, "four forged fields, each quoted")

    def test_the_report_has_one_diff_one_apply_instruction_and_the_real_token(self):
        self.assertEqual(self.lines.count("DIFF"), 1, self.proc.stdout)
        self.assertEqual(
            sum(1 for ln in self.lines if ln.startswith("To promote this change")),
            1,
            self.proc.stdout,
        )
        self.assertIn(f"  {self.real_apply_command}", self.lines)
        self.assertNotIn("FORGEDTOKEN", self.real_apply_command)

    def test_nothing_the_model_wrote_stands_at_column_zero(self):
        forged = [ln for ln in self.lines if "FORGEDTOKEN" in ln]
        self.assertEqual(len(forged), 4, self.proc.stdout)
        for line in forged:
            self.assertTrue(line.startswith(" "), repr(line))
        self.assertEqual(margin_lines_outside_their_section(self.lines), [], self.proc.stdout)
        after_diff = self.lines[self.lines.index("DIFF") + 1 :]
        self.assertTrue(after_diff[0].startswith("--- "), after_diff[:2])
        self.assertNotIn(
            "(empty)", after_diff, "the real diff is not empty; the forged one said so"
        )


class ReviewDiffHeaderCannotBeForgedByAPathTests(unittest.TestCase):
    """The other author of review text is the file system.

    ``skill`` and ``file`` lines are kept to one line, but the diff's own
    ``---``/``+++`` labels are copied from the path by ``difflib``. Measured
    before ``make_patch`` spelled its labels: a skill directory named with
    line breaks made ``review`` print three ``DIFF`` headers and three apply
    instructions, two of them above the real one, and ``review_report``
    accepted the record (the id, path and diff all agreed with each other).
    The record here is edited to name such a directory consistently, the way
    a proposal against that directory would have been written.
    """

    FORGED_DIR = (
        "tdd\nDIFF\n(empty)\n\nTo promote this change, review the diff above, then run:\n"
        "  asi apply CAND-forged --approve FORGEDTOKEN --i-reviewed-this\n\n"
        "Nothing has been written to any skill file yet.\n\nzz"
    )

    def test_a_skill_directory_named_with_line_breaks_cannot_author_a_diff_header(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            cfg = ["--config", str(build_env(base, dry_run=False, seed="path-seed"))]
            proc = run_cli([*cfg, "--json", "daily", "--commit", "--day", "2026-04-03"])
            self.assertEqual(proc.returncode, RC_OK, proc.stderr)
            validated = [
                c["candidate_id"]
                for c in json.loads(proc.stdout)["candidates"]
                if c["status"] == "validated" and c["skill_id"] == "tdd"
            ]
            self.assertTrue(validated, proc.stdout)
            record_path = base / "state" / "candidates" / f"{validated[0]}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            old_dir = base / "skills-canonical" / "tdd"
            new_dir = base / "skills-canonical" / self.FORGED_DIR
            old_dir.rename(new_dir)
            old_path, new_path = record["target"]["canonical_path"], str(new_dir / "SKILL.md")
            record["target"]["canonical_path"] = new_path
            record["target"]["skill_id"] = self.FORGED_DIR
            record["target"]["mirror_paths"] = []
            record["patch"]["unified_diff"] = record["patch"]["unified_diff"].replace(
                old_path, new_path.replace("\n", "\\n")
            )
            record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            proc = run_cli([*cfg, "review", validated[0]])
            self.assertEqual(proc.returncode, RC_OK, proc.stderr)
            lines = proc.stdout.split("\n")
            self.assertEqual(lines.count("DIFF"), 1, proc.stdout)
            self.assertEqual(
                sum(1 for ln in lines if ln.startswith("To promote this change")), 1, proc.stdout
            )
            self.assertEqual(margin_lines_outside_their_section(lines), [], proc.stdout)
            carrying = [ln for ln in lines if "FORGEDTOKEN" in ln]
            self.assertEqual(len(carrying), 4, proc.stdout)
            self.assertTrue(all("\\nDIFF\\n" in ln for ln in carrying), carrying)
            self.assertTrue(any(ln.startswith("--- a/") for ln in carrying), carrying)
            self.assertTrue(any(ln.startswith("+++ b/") for ln in carrying), carrying)


class PassedOverDocumentsReachTheOperatorTests(unittest.TestCase):
    """``doctor`` and ``skills`` name every ``SKILL.md`` discovery stepped over.

    Measured before: with a dangling link, a fifo, an unfollowed link and a
    too-deep document all named ``SKILL.md`` in the canonical root, ``doctor``
    printed ``skills 4 discovered`` and ``verdict: ready``, its JSON had no
    field for them, and ``skills`` listed the same four skills as a clean tree.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name).resolve()
        cls.cfg = ["--config", str(build_env(cls.base, dry_run=True, seed="passed-over"))]
        root = cls.base / "skills-canonical"
        (root / "dangling").mkdir()
        (root / "dangling" / "SKILL.md").symlink_to(root / "nowhere.md")
        (root / "pipe").mkdir()
        os.mkfifo(root / "pipe" / "SKILL.md")
        (root / "alias").mkdir()
        (root / "alias" / "SKILL.md").symlink_to(root / "tdd" / "SKILL.md")
        deep = root / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "SKILL.md").write_text("---\nname: deep\n---\n# deep\n", encoding="utf-8")
        (root / "SKILL.md").write_text("---\nname: atroot\n---\n# at the root\n", encoding="utf-8")
        (root / "tdd" / "examples").mkdir()
        (root / "tdd" / "examples" / "SKILL.md").write_text(
            "---\nname: examples\n---\n# an example shipped inside tdd\n", encoding="utf-8"
        )
        cls.expected = {
            str(
                root / "a" / "b" / "c" / "d" / "SKILL.md"
            ): "deeper than max_depth=3 below the root",
            str(root / "SKILL.md"): (
                "at the root itself, not in a skill directory; a skill is <root>/<name>/SKILL.md"
            ),
            str(root / "tdd" / "examples" / "SKILL.md"): (
                f"below {root / 'tdd' / 'SKILL.md'}, the place of skill 'tdd'; a document "
                "under another skill's directory is that skill's material, not a skill"
            ),
            str(
                root / "dangling" / "SKILL.md"
            ): "a symlink whose target does not exist or cannot be reached",
            str(root / "pipe" / "SKILL.md"): "not a regular file (a fifo)",
            str(root / "alias" / "SKILL.md"): "a symlink; this store does not follow links",
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_doctor_names_each_passed_over_document_in_json_and_text(self):
        started = time.monotonic()
        as_json = run_cli([*self.cfg, "--json", "doctor"])
        as_text = run_cli([*self.cfg, "doctor"])
        self.assertLess(time.monotonic() - started, 120.0, "a fifo must not be opened")
        self.assertEqual(as_json.returncode, RC_OK, as_json.stderr)
        payload = json.loads(as_json.stdout)
        reported = {d["path"]: d["reason"] for d in payload["skill_documents_passed_over"]}
        self.assertEqual(reported, self.expected)
        self.assertTrue(
            all(d["root_id"] == "canonical" for d in payload["skill_documents_passed_over"])
        )
        self.assertEqual(payload["skills_discovered"], 4, "the same four skills as a clean tree")
        lines = as_text.stdout.split("\n")
        warned = [ln for ln in lines if ln.startswith("WARN  skill document passed over")]
        self.assertEqual(len(warned), 6, as_text.stdout)
        for path, reason in self.expected.items():
            self.assertIn(
                f"WARN  skill document passed over in root canonical: {path} is {reason}", lines
            )
        self.assertTrue(
            any(ln.startswith("skills ") and ln.endswith("6 passed over") for ln in lines),
            as_text.stdout,
        )

    def test_skills_lists_each_passed_over_document_in_json_and_text(self):
        as_json = run_cli([*self.cfg, "--json", "skills"])
        as_text = run_cli([*self.cfg, "skills"])
        self.assertEqual(as_json.returncode, RC_OK, as_json.stderr)
        payload = json.loads(as_json.stdout)
        self.assertEqual({d["path"]: d["reason"] for d in payload["passed_over"]}, self.expected)
        ids = {g["skill_id"] for g in payload["groups"]}
        self.assertEqual(ids, {"broken-frontmatter", "code-review", "mirror-only", "tdd"})
        self.assertNotIn(self.base.name, ids, "no skill named after the root folder")
        self.assertNotIn("examples", ids, "the example inside tdd is not a skill")
        lines = as_text.stdout.split("\n")
        for path, reason in self.expected.items():
            self.assertIn(f"passed over (canonical): {path} is {reason}", lines)
        self.assertEqual(sum(1 for ln in lines if ln.startswith("passed over (")), 6)
        self.assertTrue(lines[-2].endswith("6 document(s) passed over"), as_text.stdout)
        self.assertNotIn("alias", [ln.split("  ")[0] for ln in lines], "the link was not taken")


@unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file")
class AnUnreadableSkillDocumentDoesNotEndTheRunTests(unittest.TestCase):
    """Measured before: one ``SKILL.md`` at mode 000 ended ``doctor`` and
    ``daily --commit`` alike with exit 70 and a ``PermissionError`` traceback
    from inside discovery. It is passed over with its reason now, like a fifo.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = ["--config", str(build_env(self.base, seed="unreadable-skill"))]
        self.target = self.base / "skills-canonical" / "code-review" / "SKILL.md"
        os.chmod(self.target, 0o000)
        self.addCleanup(os.chmod, self.target, 0o644)  # runs before the directory goes

    def test_doctor_warns_and_daily_completes(self):
        as_json = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(as_json.returncode, RC_OK, as_json.stderr)
        payload = json.loads(as_json.stdout)
        self.assertEqual(
            {d["path"]: d["reason"] for d in payload["skill_documents_passed_over"]},
            {str(self.target): "a document this process cannot read (Permission denied)"},
        )
        self.assertEqual(payload["skills_discovered"], 3)
        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        self.assertNotIn("Traceback", daily.stderr)
        receipt = json.loads(daily.stdout)
        self.assertIn(receipt["status"], ("ok", "partial"))
        self.assertEqual(os.stat(self.target).st_mode & 0o777, 0, "untouched")
        self.assertEqual(
            {d["path"]: d["reason"] for d in receipt["skill_tree"]["skill_documents_passed_over"]},
            {str(self.target): "a document this process cannot read (Permission denied)"},
            "the receipt is the unattended surface; it names the document too",
        )

    def test_the_skill_is_not_described_as_mirror_only_and_no_proposal_goes_elsewhere(self):
        # The canonical root holds tdd; make that copy unreadable while its mirror
        # reads. Measured before: the group read exactly like mirror-only, and daily
        # quarantined tdd with "no editable root holds this skill", which was false.
        os.chmod(self.target, 0o644)
        tdd = self.base / "skills-canonical" / "tdd" / "SKILL.md"
        os.chmod(tdd, 0o000)
        self.addCleanup(os.chmod, tdd, 0o644)
        skills = run_cli([*self.cfg, "--json", "skills"])
        self.assertEqual(skills.returncode, RC_OK, skills.stderr)
        group = next(g for g in json.loads(skills.stdout)["groups"] if g["skill_id"] == "tdd")
        self.assertIsNone(group["canonical"])
        self.assertIn("passed over by discovery", group["canonical_reason"])
        self.assertIn("'canonical'", group["canonical_reason"])
        self.assertNotIn("no editable root holds", group["canonical_reason"])
        as_text = run_cli([*self.cfg, "skills"])
        self.assertTrue(
            any(
                ln.startswith("    why: the copy in editable root 'canonical' was passed over")
                for ln in as_text.stdout.split("\n")
            ),
            as_text.stdout,
        )
        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        receipt = json.loads(daily.stdout)
        self.assertEqual([c for c in receipt["candidates"] if c["skill_id"] == "tdd"], [])
        text = json.dumps(receipt)
        self.assertNotIn("no editable root holds this skill", text)
        if "'tdd' has no canonical editable copy" in text:
            self.assertIn("passed over by discovery", text)
        self.assertEqual(os.stat(tdd).st_mode & 0o777, 0, "untouched")


class AnUnreadableCandidateRecordIsNamedNotDroppedTests(unittest.TestCase):
    """One validated candidate, then its record is truncated on disk.

    Measured before: ``candidates --all`` printed ``0 candidate(s)``, ``doctor``
    reported nothing, and the next ``daily --commit`` counted zero open
    candidates for ``tdd`` and proposed ``CAND-...-02-tdd`` beside the broken
    record, the duplicate the per-skill cap exists to prevent. Now the record
    is named by the listings, is a ``PROBLEM`` for ``doctor``, and stands
    ``daily`` down with the queue uncounted; the file itself is not touched.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name).resolve()
        cls.cfg = ["--config", str(build_env(cls.base, dry_run=False, seed="unreadable"))]
        proc = run_cli([*cls.cfg, "--json", "daily", "--commit", "--day", "2026-05-01"])
        assert proc.returncode == RC_OK, proc.stderr
        validated = [c for c in json.loads(proc.stdout)["candidates"] if c["status"] == "validated"]
        assert validated and validated[0]["skill_id"] == "tdd", proc.stdout
        cls.record = cls.base / "state" / "candidates" / f"{validated[0]['candidate_id']}.json"
        cls.intact = cls.record.read_bytes()
        cls.record.write_bytes(cls.intact[: len(cls.intact) // 2])
        cls.truncated_sha = sha256_file(cls.record)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_candidates_names_the_record_as_unreadable_in_json_and_text(self):
        as_json = run_cli([*self.cfg, "--json", "candidates", "--all"])
        self.assertEqual(as_json.returncode, RC_OK, as_json.stderr)
        payload = json.loads(as_json.stdout)
        self.assertEqual(payload["candidates"], [])
        self.assertEqual([u["path"] for u in payload["unreadable"]], [str(self.record)])
        self.assertIn("could not be read", payload["unreadable"][0]["reason"])
        as_text = run_cli([*self.cfg, "candidates", "--all"])
        lines = as_text.stdout.split("\n")
        self.assertTrue(
            any(ln.startswith(f"UNREADABLE  {self.record.name}  ") for ln in lines), as_text.stdout
        )
        self.assertIn("0 candidate(s); 1 unreadable record(s)", lines)

    def test_doctor_reports_it_as_a_problem_and_exits_nonzero(self):
        as_json = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(as_json.returncode, 1, as_json.stderr)
        payload = json.loads(as_json.stdout)
        self.assertEqual(
            [u["path"] for u in payload["candidate_records_unreadable"]], [str(self.record)]
        )
        self.assertTrue(any("candidate record cannot be read" in p for p in payload["problems"]))
        as_text = run_cli([*self.cfg, "doctor"])
        self.assertIn("PROBLEM  candidate record cannot be read", as_text.stdout)
        self.assertIn("verdict: problems found", as_text.stdout)

    def test_daily_stands_down_proposes_nothing_and_leaves_the_record_as_it_is(self):
        proc = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-05-02"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        receipt = json.loads(proc.stdout)
        self.assertEqual(receipt["status"], "stood-down", receipt)
        self.assertIn("queue guard", receipt["stand_down_reason"])
        self.assertIn(self.record.name, receipt["stand_down_reason"])
        self.assertEqual(receipt["candidates"], [])
        self.assertEqual(receipt["counts"]["candidate_records_unreadable"], 1)
        self.assertEqual(
            sorted(p.name for p in self.record.parent.glob("*.json")), [self.record.name]
        )
        self.assertEqual(sha256_file(self.record), self.truncated_sha, "left exactly as found")

    def test_zz_the_control_the_restored_record_counts_toward_the_per_skill_cap(self):
        self.record.write_bytes(self.intact)
        proc = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-05-03"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        receipt = json.loads(proc.stdout)
        self.assertEqual(receipt["counts"]["candidate_records_unreadable"], 0)
        self.assertEqual(receipt["counts"]["skills_with_open_candidate"], 1, receipt["counts"])
        self.assertEqual(receipt["candidates"], [], "no competitor for tdd")


class AMisnamedCandidateRecordIsUnreachableAndSaysSoTests(unittest.TestCase):
    """A validated record copied under a filename its id does not derive.

    Measured before: ``candidates`` listed it as ``validated gate=pass`` under
    the id it declares, ``review`` of that id exited 4 (``no candidate``),
    ``review`` of the filename's id was refused, ``doctor`` was clean, and
    every later ``daily`` counted it as the one open ``tdd`` candidate, so the
    skill was capped by a record nobody could open. The listing now applies the
    rule ``review`` applies, and the record is named with the remedy.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name).resolve()
        cls.cfg = ["--config", str(build_env(cls.base, dry_run=False, seed="misnamed"))]
        proc = run_cli([*cls.cfg, "--json", "daily", "--commit", "--day", "2026-06-01"])
        assert proc.returncode == RC_OK, proc.stderr
        validated = [c for c in json.loads(proc.stdout)["candidates"] if c["status"] == "validated"]
        assert validated and validated[0]["skill_id"] == "tdd", proc.stdout
        cls.declared_id = validated[0]["candidate_id"]
        genuine = cls.base / "state" / "candidates" / f"{cls.declared_id}.json"
        cls.misnamed = genuine.with_name("CAND-2026-06-01-09-tdd.json")
        genuine.rename(cls.misnamed)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_listing_names_the_record_with_both_names_and_lists_no_candidate(self):
        payload = json.loads(run_cli([*self.cfg, "--json", "candidates", "--all"]).stdout)
        self.assertEqual(payload["candidates"], [])
        self.assertEqual([u["path"] for u in payload["unreadable"]], [str(self.misnamed)])
        reason = payload["unreadable"][0]["reason"]
        self.assertIn(f"declares id '{self.declared_id}'", reason)
        self.assertIn(f"rename it to {self.declared_id}.json", reason)
        text = run_cli([*self.cfg, "candidates", "--all"]).stdout
        self.assertIn(f"UNREADABLE  {self.misnamed.name}  ", text)
        self.assertNotIn(f"{self.declared_id}  validated", text)

    def test_review_by_either_name_is_a_refusal_the_listing_already_predicted(self):
        by_declared = run_cli([*self.cfg, "review", self.declared_id])
        self.assertEqual(by_declared.returncode, RC_MISSING_FILE, by_declared.stderr)
        by_filename = run_cli([*self.cfg, "review", "CAND-2026-06-01-09-tdd"])
        self.assertEqual(by_filename.returncode, RC_REFUSED, by_filename.stderr)
        self.assertIn(f"declares id '{self.declared_id}'", by_filename.stderr)
        self.assertNotIn("approval_token", by_declared.stdout + by_filename.stdout)

    def test_doctor_and_daily_treat_it_as_the_unreadable_record_it_is(self):
        doctor = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(doctor.returncode, 1, doctor.stderr)
        self.assertEqual(
            [u["path"] for u in json.loads(doctor.stdout)["candidate_records_unreadable"]],
            [str(self.misnamed)],
        )
        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-06-02"])
        receipt = json.loads(daily.stdout)
        self.assertEqual(receipt["status"], "stood-down", receipt)
        self.assertIn(self.misnamed.name, receipt["stand_down_reason"])
        self.assertEqual(receipt["counts"]["candidate_records_unreadable"], 1)
        self.assertTrue(self.misnamed.is_file(), "left where it was")

    def test_zz_the_control_renamed_back_it_is_the_one_open_candidate_again(self):
        genuine = self.misnamed.with_name(f"{self.declared_id}.json")
        self.misnamed.rename(genuine)
        payload = json.loads(run_cli([*self.cfg, "--json", "candidates"]).stdout)
        self.assertEqual([c["id"] for c in payload["candidates"]], [self.declared_id])
        self.assertEqual(payload["unreadable"], [])
        review = run_cli([*self.cfg, "--json", "review", self.declared_id])
        self.assertEqual(review.returncode, RC_OK, review.stderr)


class ACrlfDocumentIsAppliedAndRolledBackInItsOwnConventionTests(unittest.TestCase):
    """Measured before: a CRLF ``tdd`` was proposed and validated from its folded
    text, ``review`` printed the token, and ``apply`` refused every time with
    "The skill changed since the proposal was made": the on-disk hash was taken
    over the raw bytes while every other hash folds line endings. Nothing had
    changed. Now apply compares under one identity, writes the content change
    in the document's own line endings, and rollback restores the bytes."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = ["--config", str(build_env(self.base, dry_run=False, seed="crlf"))]
        self.skill = self.base / "skills-canonical" / "tdd" / "SKILL.md"
        for path in (self.skill, self.base / "skills-mirror" / "tdd" / "SKILL.md"):
            if path.exists():
                text = path.read_text(encoding="utf-8")
                path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
        self.original = self.skill.read_bytes()
        assert b"\r\n" in self.original and self.original.count(b"\n") == self.original.count(
            b"\r\n"
        )

    def _validated_candidate_id(self) -> str:
        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        rows = [c for c in json.loads(daily.stdout)["candidates"] if c["skill_id"] == "tdd"]
        self.assertEqual([c["status"] for c in rows], ["validated"], rows)
        return rows[0]["candidate_id"]

    def test_apply_succeeds_keeps_crlf_and_rollback_restores_the_bytes(self):
        candidate_id = self._validated_candidate_id()
        review = run_cli([*self.cfg, "--json", "review", candidate_id])
        self.assertEqual(review.returncode, RC_OK, review.stderr)
        report = json.loads(review.stdout)
        self.assertEqual(
            sha256_file(self.skill), report["base_sha256"], "one identity for the document"
        )
        apply = run_cli(
            [
                *self.cfg,
                "apply",
                candidate_id,
                "--approve",
                report["approval_token"],
                "--i-reviewed-this",
                "--operator",
                "test",
            ]
        )
        self.assertEqual(apply.returncode, RC_OK, apply.stderr)
        written = self.skill.read_bytes()
        self.assertIn(FIX_MARKER.encode("utf-8"), written)
        self.assertEqual(written.count(b"\n"), written.count(b"\r\n"), "CRLF throughout")
        self.assertTrue(
            written.replace(b"\r\n", b"\n").startswith(
                self.original.replace(b"\r\n", b"\n").rstrip(b"\n")
            ),
            "the document's own lines come first, unchanged",
        )
        rollback = run_cli([*self.cfg, "rollback", candidate_id])
        self.assertEqual(rollback.returncode, RC_OK, rollback.stderr)
        self.assertEqual(self.skill.read_bytes(), self.original, "byte-for-byte")

    def test_the_control_an_lf_document_is_still_written_as_lf(self):
        lf = self.original.replace(b"\r\n", b"\n")
        for path in (self.skill, self.base / "skills-mirror" / "tdd" / "SKILL.md"):
            if path.exists():
                path.write_bytes(lf)
        candidate_id = self._validated_candidate_id()
        report = json.loads(run_cli([*self.cfg, "--json", "review", candidate_id]).stdout)
        apply = run_cli(
            [
                *self.cfg,
                "apply",
                candidate_id,
                "--approve",
                report["approval_token"],
                "--i-reviewed-this",
            ]
        )
        self.assertEqual(apply.returncode, RC_OK, apply.stderr)
        self.assertNotIn(b"\r", self.skill.read_bytes())


class ARootHoldingSeveralMirrorCopiesNamesEachByPathTests(unittest.TestCase):
    """Measured before: two copies of ``tdd`` in the one mirror root rendered as
    ``mirrors=mirror,mirror  DIVERGENT=mirror`` in ``asi skills``: the root id
    twice, and no way to tell which of the two copies had diverged. The JSON
    carried both paths; the text named neither. A root is now one name with a
    count, and each copy under such a root is listed by path with its own
    divergence tag.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = ["--config", str(build_env(self.base, seed="several-mirrors"))]
        self.first = self.base / "skills-mirror" / "tdd" / "SKILL.md"
        self.second = self.base / "skills-mirror" / "nested" / "tdd" / "SKILL.md"

    def _add_second_copy(self, *, divergent: bool) -> None:
        self.second.parent.mkdir(parents=True)
        text = self.first.read_text(encoding="utf-8")
        self.second.write_text(text + ("\nlocal note\n" if divergent else ""), encoding="utf-8")

    def _tdd_lines(self) -> list[str]:
        proc = run_cli([*self.cfg, "skills"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        lines = proc.stdout.splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.startswith("tdd  "))
        end = next(
            (i for i in range(start + 1, len(lines)) if not lines[i].startswith("    ")),
            len(lines),
        )
        return lines[start:end]

    def test_the_root_is_named_once_with_a_count_and_the_divergent_copy_by_path(self):
        self._add_second_copy(divergent=True)
        head, *rest = self._tdd_lines()
        self.assertTrue(head.endswith("  mirrors=mirror x2  DIVERGENT=mirror"), head)
        self.assertNotIn("mirror,mirror", head)
        self.assertEqual(
            rest,
            [
                f"    copy (mirror): {self.second}  DIVERGENT",
                f"    copy (mirror): {self.first}",
            ],
        )
        as_json = run_cli([*self.cfg, "--json", "skills"])
        group = next(g for g in json.loads(as_json.stdout)["groups"] if g["skill_id"] == "tdd")
        self.assertEqual(
            sorted(m["path"] for m in group["mirrors"]), sorted([str(self.first), str(self.second)])
        )

    def test_two_identical_copies_are_counted_and_neither_is_tagged(self):
        self._add_second_copy(divergent=False)
        head, *rest = self._tdd_lines()
        self.assertTrue(head.endswith("  mirrors=mirror x2"), head)
        self.assertNotIn("DIVERGENT", head)
        self.assertEqual(len(rest), 2, rest)
        self.assertTrue(all(ln.startswith("    copy (mirror): ") for ln in rest), rest)
        self.assertFalse(any("DIVERGENT" in ln for ln in rest), rest)

    def test_the_control_one_copy_per_root_renders_as_before_with_no_copy_lines(self):
        head, *rest = self._tdd_lines()
        self.assertTrue(head.endswith("  mirrors=mirror"), head)
        self.assertEqual(rest, [])

    def test_the_reviewer_note_names_each_divergent_copy_by_path(self):
        self._add_second_copy(divergent=True)
        self.first.write_text(
            self.first.read_text(encoding="utf-8") + "\nanother local note\n", encoding="utf-8"
        )
        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        receipt = json.loads(daily.stdout)
        cid = next(c["candidate_id"] for c in receipt["candidates"] if c["skill_id"] == "tdd")
        review = run_cli([*self.cfg, "--json", "review", cid])
        self.assertEqual(review.returncode, RC_OK, review.stderr)
        notes = [n for n in json.loads(review.stdout)["notes"] if n.startswith("mirror copies")]
        self.assertEqual(len(notes), 1, notes)
        self.assertIn(f"mirror: {self.first}", notes[0])
        self.assertIn(f"mirror: {self.second}", notes[0])
        self.assertNotIn("(mirror, mirror)", notes[0])

    def test_root_labels_keep_first_seen_order_and_count_only_past_one(self):
        from types import SimpleNamespace

        from agent_skill_improver.cli import _root_labels

        copies = [SimpleNamespace(root_id=r) for r in ("zeta", "alpha", "zeta", "zeta", "mid")]
        self.assertEqual(_root_labels(copies), "zeta x3,alpha,mid")
        self.assertEqual(_root_labels([]), "")


class ReviewAsksTheQuestionTheWriteWillAskTests(unittest.TestCase):
    """Measured before: with the canonical root re-pointed at a copy of the
    tree and the old directory left on disk outside every root, ``review`` of
    a candidate recorded against the old path printed a diff, a token and an
    apply command; ``apply`` then refused the unrecognised target. Review and
    apply disagreed, and a reviewer was handed a command that would not work.
    With the root merely moved, the read failure was reported as "refusing to
    apply" by the review command. ``review`` now runs the store's canonical
    check (mirror copy, copy with no canonical, no skill document in any root)
    before it shows anything, and names itself in a read failure.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.config_path = Path(build_env(self.base, seed="review-asks"))
        self.cfg = ["--config", str(self.config_path)]
        self.text = self.config_path.read_text(encoding="utf-8")
        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        receipt = json.loads(daily.stdout)
        self.cid = next(c["candidate_id"] for c in receipt["candidates"] if c["skill_id"] == "tdd")
        self.old_file = self.base / "skills-canonical" / "tdd" / "SKILL.md"
        self.before = sha256_file(self.old_file)

    def _rewrite_config(self, old: str, new: str) -> None:
        assert self.text.count(old) >= 1, old
        self.config_path.write_text(self.text.replace(old, new), encoding="utf-8")

    def _review(self, *extra: str):
        return run_cli([*self.cfg, *extra, "review", self.cid])

    def _real_token(self) -> str:
        proc = self._review("--json")
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        return str(json.loads(proc.stdout)["approval_token"])

    def test_a_target_outside_every_configured_root_gets_no_token_in_text_or_json(self):
        token = self._real_token()
        shutil.copytree(self.base / "skills-canonical", self.base / "skills-main")
        self._rewrite_config("skills-canonical", "skills-main")
        for extra in ((), ("--json",)):
            with self.subTest(json=bool(extra)):
                proc = self._review(*extra)
                self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout + proc.stderr)
                self.assertIn(f"refusing to review {self.old_file}", proc.stderr)
                self.assertIn("not a skill document in any configured root", proc.stderr)
                self.assertNotIn("--approve", proc.stdout + proc.stderr)
                self.assertNotIn("approval_token", proc.stdout)
        # The token is the real one, so apply reaches the same question and
        # answers it in its own voice: the write refuses, by the same rule.
        applied = run_cli([*self.cfg, "apply", self.cid, "--approve", token, "--i-reviewed-this"])
        self.assertEqual(applied.returncode, RC_REFUSED, applied.stderr)
        self.assertIn(f"refusing to write {self.old_file}", applied.stderr)
        self.assertIn("not a skill document in any configured root", applied.stderr)
        self.assertEqual(sha256_file(self.old_file), self.before, "nothing written anywhere")
        self.assertEqual(
            sha256_file(self.base / "skills-main" / "tdd" / "SKILL.md"), self.before, "nor here"
        )

    def test_a_skill_that_lost_its_canonical_copy_gets_no_token(self):
        # A second document claiming the skill in the editable root: the skill
        # now has no canonical copy, and review says so in its own voice.
        newcomer = self.base / "skills-canonical" / "group" / "tdd" / "SKILL.md"
        newcomer.parent.mkdir(parents=True)
        shutil.copy(self.old_file, newcomer)
        proc = self._review()
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout + proc.stderr)
        self.assertIn(
            f"refusing to review {self.old_file}: skill 'tdd' has no canonical copy", proc.stderr
        )
        self.assertIn("holds 2 copies of this skill", proc.stderr)
        self.assertIn("that must be settled before anything is written", proc.stderr)
        self.assertNotIn("--approve", proc.stdout)

    def test_a_closed_candidate_is_named_as_closed_before_its_target_is_judged(self):
        # The same order apply asks its questions in: the record first, the
        # tree after. An applied candidate whose root was then re-pointed is a
        # closed candidate, not an unrecognised target.
        token = self._real_token()
        applied = run_cli([*self.cfg, "apply", self.cid, "--approve", token, "--i-reviewed-this"])
        self.assertEqual(applied.returncode, RC_OK, applied.stderr)
        shutil.copytree(self.base / "skills-canonical", self.base / "skills-main")
        self._rewrite_config("skills-canonical", "skills-main")
        proc = self._review()
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout + proc.stderr)
        self.assertIn("status is 'applied'", proc.stderr)
        self.assertNotIn("not a skill document", proc.stderr)
        self.assertNotIn("--approve", proc.stdout)

    def test_a_target_that_became_a_mirror_copy_gets_no_token(self):
        # The two roots swap roles: the recorded target is now a mirror copy.
        swapped = (
            self.text.replace("precedence = 10\ncanonical = true\neditable = true", "@A@")
            .replace("precedence = 90\ncanonical = false\neditable = false", "@B@")
            .replace("@A@", "precedence = 90\ncanonical = false\neditable = false")
            .replace("@B@", "precedence = 10\ncanonical = true\neditable = true")
            .replace('kind = "custom"', "@K@")
            .replace('kind = "mirror"', 'kind = "custom"')
            .replace("@K@", 'kind = "mirror"')
        )
        self.assertNotEqual(swapped, self.text)
        self.config_path.write_text(swapped, encoding="utf-8")
        proc = self._review()
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout + proc.stderr)
        self.assertIn(f"refusing to review {self.old_file}: it is a mirror copy", proc.stderr)
        self.assertIn(str(self.base / "skills-mirror" / "tdd" / "SKILL.md"), proc.stderr)
        self.assertNotIn("--approve", proc.stdout)

    def test_a_moved_root_is_reported_by_the_reviewing_command_with_the_remedy(self):
        shutil.move(self.base / "skills-canonical", self.base / "skills-main")
        self._rewrite_config("skills-canonical", "skills-main")
        proc = self._review()
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout + proc.stderr)
        self.assertIn(f"refusing to review {self.cid}", proc.stderr)
        self.assertNotIn("refusing to apply", proc.stderr)
        self.assertIn("could not be read", proc.stderr)
        self.assertIn("moved or re-pointed", proc.stderr)
        applied = run_cli(
            [*self.cfg, "apply", self.cid, "--approve", "0123456789abcdef", "--i-reviewed-this"]
        )
        self.assertEqual(applied.returncode, RC_REFUSED, applied.stderr)
        self.assertIn(f"refusing to apply {self.cid}", applied.stderr)

    def test_the_control_an_unchanged_tree_still_reviews_with_a_token(self):
        proc = self._review()
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertIn("--approve", proc.stdout)
        self.assertIn("--i-reviewed-this", proc.stdout)


class TwoRootsTiedOnPrecedenceAreRefusedNotPickedByIdTests(unittest.TestCase):
    """Measured before: the harness mirror root renamed ``alpha`` and made
    canonical and editable at the canonical root's precedence had its ``tdd``
    proposed against, the real copy (root ``zeta``) was listed as its mirror,
    and ``doctor`` had no warning: the pick was alphabetical and silent."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        config_path = build_env(self.base, seed="tied-roots")
        text = config_path.read_text(encoding="utf-8")
        for old, new in (
            ('id = "canonical"\n', 'id = "zeta"\n'),
            ('id = "mirror"\n', 'id = "alpha"\n'),
            (
                'kind = "mirror"\nprecedence = 90\ncanonical = false\neditable = false\n',
                'kind = "custom"\nprecedence = 10\ncanonical = true\neditable = true\n',
            ),
        ):
            assert text.count(old) == 1, (old, text)
            text = text.replace(old, new)
        config_path.write_text(text, encoding="utf-8")
        self.cfg = ["--config", str(config_path)]
        self.copies = [
            self.base / "skills-canonical" / "tdd" / "SKILL.md",
            self.base / "skills-mirror" / "tdd" / "SKILL.md",
        ]
        self.before = {p: sha256_file(p) for p in self.copies}

    def test_skills_and_doctor_name_both_roots_and_daily_proposes_against_neither(self):
        skills = run_cli([*self.cfg, "--json", "skills"])
        self.assertEqual(skills.returncode, RC_OK, skills.stderr)
        group = next(g for g in json.loads(skills.stdout)["groups"] if g["skill_id"] == "tdd")
        self.assertIsNone(group["canonical"])
        self.assertEqual(sorted(group["ambiguous_copy_paths"]), sorted(map(str, self.copies)))
        text = run_cli([*self.cfg, "skills"])
        self.assertIn(
            "tdd  [NO CANONICAL EDITABLE COPY -- proposals refused]  ambiguous=alpha,zeta",
            text.stdout,
        )
        self.assertIn("why: roots 'alpha', 'zeta' each hold a copy of this skill", text.stdout)

        doctor = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(doctor.returncode, RC_OK, doctor.stderr)
        warnings = [w for w in json.loads(doctor.stdout)["warnings"] if "'tdd'" in w]
        self.assertEqual(len(warnings), 1, json.loads(doctor.stdout)["warnings"])
        self.assertIn(
            "has a copy in each of roots 'alpha', 'zeta', which tie on precedence", warnings[0]
        )
        self.assertIn("distinct precedences", warnings[0])

        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        receipt = json.loads(daily.stdout)
        self.assertEqual(receipt["candidates"], [], "no proposal at either copy")
        row = receipt["skills"]["tdd"]
        self.assertEqual((row["status"], row["stage"]), ("quarantined", "propose"))
        self.assertIn("decided by root id alone", row["reason"])
        self.assertEqual({p: sha256_file(p) for p in self.copies}, self.before, "untouched")

    def test_the_control_giving_one_root_a_lower_precedence_restores_the_skill(self):
        config_path = Path(self.cfg[1])
        text = config_path.read_text(encoding="utf-8")
        assert text.count("precedence = 10\n") == 2, text
        config_path.write_text(text.replace("precedence = 10\n", "precedence = 5\n", 1))
        skills = run_cli([*self.cfg, "--json", "skills"])
        self.assertEqual(skills.returncode, RC_OK, skills.stderr)
        group = next(g for g in json.loads(skills.stdout)["groups"] if g["skill_id"] == "tdd")
        self.assertEqual(group["canonical"]["root_id"], "zeta", "the first table is 'zeta'")
        self.assertEqual(group["ambiguous_copy_paths"], [])
        doctor = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual([w for w in json.loads(doctor.stdout)["warnings"] if "'tdd'" in w], [])


class ADuplicateCopyInTheEditableRootIsRefusedNotPickedTests(unittest.TestCase):
    """Measured before: a second ``SKILL.md`` for ``tdd`` nested elsewhere in
    the canonical root became the canonical copy by walk order, ``skills``
    listed the real document as a mirror in its own root, ``doctor`` said
    ready, and ``daily --commit`` proposed against the newcomer with exit 0.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = ["--config", str(build_env(self.base, seed="duplicate-copy"))]
        self.real = self.base / "skills-canonical" / "tdd" / "SKILL.md"
        self.newcomer = self.base / "skills-canonical" / "group" / "tdd" / "SKILL.md"
        self.newcomer.parent.mkdir(parents=True)
        self.newcomer.write_text(
            self.real.read_text(encoding="utf-8").replace("description:", "description: COPY"),
            encoding="utf-8",
        )
        self.before = {p: sha256_file(p) for p in (self.real, self.newcomer)}

    def test_skills_and_doctor_name_both_copies_and_daily_proposes_against_neither(self):
        skills = run_cli([*self.cfg, "--json", "skills"])
        self.assertEqual(skills.returncode, RC_OK, skills.stderr)
        group = next(g for g in json.loads(skills.stdout)["groups"] if g["skill_id"] == "tdd")
        self.assertIsNone(group["canonical"])
        self.assertEqual(
            sorted(group["ambiguous_copy_paths"]), sorted([str(self.real), str(self.newcomer)])
        )
        text = run_cli([*self.cfg, "skills"])
        self.assertIn(
            "tdd  [NO CANONICAL EDITABLE COPY -- proposals refused]  ambiguous=canonical x2"
            "  mirrors=mirror",
            text.stdout,
        )
        self.assertIn("why: editable root 'canonical' holds 2 copies of this skill", text.stdout)
        # The ambiguous copies are on the why line by path; they are not mirror
        # copies, so no copy lines are printed for the editable root.
        self.assertNotIn("copy (canonical):", text.stdout)

        doctor = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(doctor.returncode, RC_OK, doctor.stderr)
        warnings = [w for w in json.loads(doctor.stdout)["warnings"] if "'tdd'" in w]
        self.assertEqual(len(warnings), 1, json.loads(doctor.stdout)["warnings"])
        self.assertIn("has 2 copies in editable root 'canonical'", warnings[0])
        self.assertIn(str(self.newcomer), warnings[0])
        self.assertIn("until one copy remains", warnings[0])

        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        receipt = json.loads(daily.stdout)
        self.assertEqual(receipt["status"], "partial")
        self.assertEqual(receipt["candidates"], [], "no proposal at either document")
        row = receipt["skills"]["tdd"]
        self.assertEqual((row["status"], row["stage"]), ("quarantined", "propose"))
        self.assertIn("ambiguous", row["reason"])
        self.assertEqual({p: sha256_file(p) for p in self.before}, self.before, "untouched")

    def test_removing_the_newcomer_restores_the_skill_without_any_other_change(self):
        self.newcomer.unlink()
        skills = run_cli([*self.cfg, "--json", "skills"])
        group = next(g for g in json.loads(skills.stdout)["groups"] if g["skill_id"] == "tdd")
        self.assertEqual(group["canonical"]["path"], str(self.real))
        self.assertEqual(group["ambiguous_copy_paths"], [])
        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        receipt = json.loads(daily.stdout)
        self.assertEqual([c["canonical_path"] for c in receipt["candidates"]], [str(self.real)])


class ARoleCommandThatCannotStartIsAConfigProblemTests(unittest.TestCase):
    """Measured before: with ``[model.scorer].argv`` pointing at a mode-644
    file, ``doctor`` exited 0 with no problem and ``daily`` exited 70 with a
    PermissionError traceback.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.exports = self.base / "exports"
        shutil.copytree(REPO_ROOT / "tests" / "fixtures" / "exports", self.exports)
        config_path = build_env(self.base, seed="unstartable-scorer", fixtures_dir=self.exports)
        self.scorer = self.base / "scorer-644"
        self.scorer.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        os.chmod(self.scorer, 0o644)
        text = config_path.read_text(encoding="utf-8")
        head, _, rest = text.partition("[model.scorer]\nargv = ")
        _, _, after = rest.partition("\n")
        config_path.write_text(
            f"{head}[model.scorer]\nargv = [{json.dumps(str(self.scorer))}]\n{after}",
            encoding="utf-8",
        )
        self.cfg = ["--config", str(config_path)]

    @unittest.skipIf(os.geteuid() == 0, "root may execute a mode-644 file")
    def test_doctor_reports_the_problem_and_daily_refuses_before_any_model_call(self):
        doctor = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(doctor.returncode, 1, doctor.stdout)
        problems = [p for p in json.loads(doctor.stdout)["problems"] if p.startswith("model role")]
        self.assertEqual(len(problems), 1, json.loads(doctor.stdout)["problems"])
        self.assertIn(
            f"model role 'scorer' command {str(self.scorer)!r} exists but is not executable "
            "(mode 644); chmod +x it",
            problems[0],
        )
        # The source is made unusable too: the refusal must name the role, not
        # the source, because the model check is answered before the source is
        # touched (no subprocess, no file read).
        os.chmod(self.exports, 0o000)
        try:
            daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        finally:
            os.chmod(self.exports, 0o755)
        self.assertEqual(daily.returncode, RC_REFUSED, daily.stderr)
        self.assertIn("ConfigError: model role 'scorer' command", daily.stderr)
        self.assertNotIn("cannot be listed", daily.stderr, "the source was never asked")
        self.assertNotIn("Traceback", daily.stderr)
        self.assertFalse(
            list((self.base / "state" / "receipts").glob("*.json")),
            "refused before the run started, so no receipt",
        )
        os.chmod(self.scorer, 0o755)
        doctor = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(
            [p for p in json.loads(doctor.stdout)["problems"] if p.startswith("model role")], []
        )


class DoctorDescribesWhichRootIsCanonicalTests(unittest.TestCase):
    """Measured before: with the editable root not marked canonical, ``doctor``
    warned "proposals will be refused" and ``daily`` then proposed a candidate
    from that very root, because ``resolve_groups`` treats the highest-precedence
    editable root as canonical when none is marked.
    """

    def _doctor_warnings(self, base: Path, replace: tuple[str, str]) -> list[str]:
        config_path = build_env(base, seed="which-root")
        text = config_path.read_text(encoding="utf-8")
        assert text.count(replace[0]) == 1, replace
        config_path.write_text(text.replace(*replace), encoding="utf-8")
        proc = run_cli(["--config", str(config_path), "--json", "doctor"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        return json.loads(proc.stdout)["warnings"]

    def test_an_editable_root_not_marked_canonical_is_named_as_the_one_treated_as_canonical(self):
        with TemporaryDirectory() as tmp:
            warnings = self._doctor_warnings(
                Path(tmp).resolve(), ("canonical = true", "canonical = false")
            )
            self.assertIn(
                "no editable skill root is marked canonical; the highest-precedence editable "
                "root 'canonical' is treated as the canonical one",
                warnings,
            )
            self.assertFalse(any("proposals will be refused" in w for w in warnings), warnings)

    def test_no_editable_root_is_the_case_where_proposals_are_refused(self):
        with TemporaryDirectory() as tmp:
            warnings = self._doctor_warnings(
                Path(tmp).resolve(), ("editable = true", "editable = false")
            )
            self.assertIn("no skill root is editable; proposals will be refused", warnings)


class ARunWithNoSkillToImproveReadsNoConversationTests(unittest.TestCase):
    """Measured before: with both skill roots gone, ``daily`` ingested and
    scored five conversations, marked them processed for the day and wrote a
    receipt whose status was ``ok``. Nothing could have been proposed.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = ["--config", str(build_env(self.base, seed="no-skills"))]
        self.canonical = self.base / "skills-canonical"
        self.mirror = self.base / "skills-mirror"

    def _run_expecting_a_stand_down_before_discovery(self) -> dict:
        proc = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        receipt = json.loads(proc.stdout)
        self.assertEqual(receipt["status"], "stood-down", receipt["stand_down_reason"])
        # ``discovery`` is filled in by the discovery stage; empty means the
        # source was never asked for a conversation.
        self.assertEqual(receipt["discovery"], {}, "discovery never ran")
        self.assertEqual(receipt["counts"]["conversations_ingested"], 0)
        self.assertEqual(receipt["counts"]["records_scored"], 0)
        self.assertEqual(receipt["records"], {})
        self.assertIn("no conversation was read or scored", receipt["stand_down_reason"])
        return receipt

    def test_no_root_that_is_a_directory_stands_the_run_down_before_discovery(self):
        shutil.rmtree(self.canonical)
        shutil.rmtree(self.mirror)
        self.mirror.write_text("not a directory\n", encoding="utf-8")
        receipt = self._run_expecting_a_stand_down_before_discovery()
        self.assertTrue(
            receipt["stand_down_reason"].startswith(
                "no configured skill root is a directory (canonical has nothing at its path; "
                "mirror is a regular file, not a directory); nothing could be proposed"
            ),
            receipt["stand_down_reason"],
        )

    def test_present_roots_with_no_skill_document_stand_the_run_down_too(self):
        shutil.rmtree(self.canonical)
        shutil.rmtree(self.mirror)
        self.canonical.mkdir()
        self.mirror.mkdir()
        (self.mirror / "tdd").mkdir()
        (self.mirror / "tdd" / "SKILL.md").symlink_to(self.base / "nowhere.md")
        receipt = self._run_expecting_a_stand_down_before_discovery()
        self.assertTrue(
            receipt["stand_down_reason"].startswith(
                "no skill document was discovered under the present root(s) canonical, mirror; "
                "1 path(s) were passed over, see skill_tree; nothing could be proposed"
            ),
            receipt["stand_down_reason"],
        )
        self.assertEqual(len(receipt["skill_tree"]["skill_documents_passed_over"]), 1)
        # `doctor` says the same thing, so the operator is not surprised by
        # the stand-down: before, a tree with no document read as ready.
        doctor = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(doctor.returncode, 1, doctor.stdout)
        self.assertIn(
            "no skill document was discovered under the present root(s) canonical, mirror; "
            "`daily` will stand down with nothing to propose",
            json.loads(doctor.stdout)["problems"],
        )

    def test_the_control_a_tree_with_one_skill_still_scores(self):
        proc = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        receipt = json.loads(proc.stdout)
        self.assertNotEqual(receipt["status"], "stood-down", receipt["stand_down_reason"])
        self.assertGreater(receipt["counts"]["records_scored"], 0)


class ARootThatIsARegularFileIsNotReportedAsAbsentTests(unittest.TestCase):
    """Measured before: with ``skills-canonical`` a regular file, ``doctor``
    warned "skill root(s) not present on this machine: canonical" with
    ``exists`` false, and the receipt listed it under ``roots_missing``.
    """

    def test_doctor_and_the_receipt_name_what_sits_at_the_root(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            cfg = ["--config", str(build_env(base, seed="file-root"))]
            root = base / "skills-canonical"
            shutil.rmtree(root)
            root.write_text("not a directory\n", encoding="utf-8")
            doctor = run_cli([*cfg, "--json", "doctor"])
            report = json.loads(doctor.stdout)
            self.assertNotIn(
                "skill root(s) not present on this machine: canonical", report["warnings"]
            )
            self.assertIn(
                f"skill document passed over in root canonical: {root} is the configured root "
                "itself and is a regular file, not a directory",
                report["warnings"],
            )
            canonical = [r for r in report["skill_roots"] if r["id"] == "canonical"][0]
            self.assertEqual((canonical["exists"], canonical["is_directory"]), (True, False))
            daily = run_cli([*cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
            self.assertEqual(daily.returncode, RC_OK, daily.stderr)
            tree = json.loads(daily.stdout)["skill_tree"]
            self.assertEqual(tree["roots_missing"], [])
            self.assertEqual(
                tree["roots_misplaced"],
                [
                    {
                        "id": "canonical",
                        "path": str(root),
                        "found": "a regular file, not a directory",
                    }
                ],
            )
            self.assertEqual(
                [d["skill_id"] for d in tree["skill_documents_passed_over"]],
                [None],
                "the finding is at the root, not pinned on a skill",
            )
            # The control: with nothing at the path, it is missing again.
            root.unlink()
            report = json.loads(run_cli([*cfg, "--json", "doctor"]).stdout)
            self.assertIn(
                "skill root(s) not present on this machine: canonical", report["warnings"]
            )
            canonical = [r for r in report["skill_roots"] if r["id"] == "canonical"][0]
            self.assertEqual((canonical["exists"], canonical["is_directory"]), (False, False))


class AStateDirectoryThatCannotBeCreatedIsAConfigProblemTests(unittest.TestCase):
    """Measured before: with ``[run].state_dir`` under a mode-555 parent,
    ``doctor``, ``daily`` and ``candidates`` all exited 70 with a
    PermissionError traceback from the first ``mkdir``.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        config_path = build_env(self.base, seed="uncreatable-state")
        self.locked = self.base / "locked"
        self.locked.mkdir()
        self.state_dir = self.locked / "state"
        text = config_path.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.startswith("state_dir = ")]
        assert len(lines) == 1, lines
        config_path.write_text(
            text.replace(lines[0], f"state_dir = {json.dumps(str(self.state_dir))}"),
            encoding="utf-8",
        )
        self.cfg = ["--config", str(config_path)]
        os.chmod(self.locked, 0o555)
        self.addCleanup(os.chmod, self.locked, 0o755)
        self.expected = (
            f"the state directory {self.state_dir} cannot be created (Permission denied); "
            "check [run].state_dir and the permissions of its parent"
        )

    @unittest.skipIf(os.geteuid() == 0, "root creates directories under a mode-555 parent")
    def test_doctor_reports_the_problem_and_the_other_commands_refuse_by_name(self):
        doctor = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(doctor.returncode, 1, doctor.stdout)
        report = json.loads(doctor.stdout)
        self.assertIn(self.expected, report["problems"])
        # The rest of the report was still produced: the finding did not
        # pre-empt the verdict.
        self.assertIn("skill_roots", report)
        self.assertNotIn("Traceback", doctor.stderr)
        for command in (["daily", "--commit", "--day", "2026-01-01"], ["candidates"]):
            with self.subTest(command=command[0]):
                proc = run_cli([*self.cfg, "--json", *command])
                self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
                self.assertIn(f"ConfigError: {self.expected}", proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(self.state_dir.exists(), "nothing was created")
        os.chmod(self.locked, 0o755)
        doctor = run_cli([*self.cfg, "--json", "doctor"])
        self.assertNotIn(self.expected, json.loads(doctor.stdout)["problems"])


class ADirectoryThePayloadInventoryCannotListIsReportedTests(unittest.TestCase):
    """Measured before: ``scripts/`` at mode 000 beside ``tdd/SKILL.md`` gave
    ``skills`` a document with no payload warning and ``doctor`` a report of
    zero payloads, the same as a documentation-only directory.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = ["--config", str(build_env(self.base, seed="unlistable-scripts"))]
        self.hidden = self.base / "skills-canonical" / "tdd" / "scripts"
        self.hidden.mkdir()
        (self.hidden / "hook.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    @unittest.skipIf(os.geteuid() == 0, "root lists a mode-000 directory")
    def test_skills_and_doctor_both_say_the_inventory_is_a_floor(self):
        os.chmod(self.hidden, 0o000)
        try:
            skills = run_cli([*self.cfg, "--json", "skills"])
            self.assertEqual(skills.returncode, RC_OK, skills.stderr)
            group = next(g for g in json.loads(skills.stdout)["groups"] if g["skill_id"] == "tdd")
            self.assertEqual(group["unlistable_dirs"], ["canonical:scripts"])
            self.assertEqual(group["payload_files"], [])
            self.assertFalse(group["canonical"]["payload_inventory_complete"])
            text = run_cli([*self.cfg, "skills"])
            self.assertIn(
                "1 directory in this skill directory could not be listed "
                "(scripts (Permission denied))",
                text.stdout,
            )
            doctor = run_cli([*self.cfg, "--json", "doctor"])
            self.assertEqual(doctor.returncode, RC_OK, doctor.stderr)
            payload = json.loads(doctor.stdout)
            self.assertEqual(
                payload["skills_with_unlistable_directories"],
                [
                    {
                        "skill_id": "tdd",
                        "root_id": "canonical",
                        "unlistable_dirs": [{"path": "scripts", "error": "Permission denied"}],
                    }
                ],
            )
            self.assertEqual(payload["skills_with_executable_payloads"], [])
            warned = [w for w in payload["warnings"] if "payload inventory could not list" in w]
            self.assertEqual(len(warned), 1, payload["warnings"])
            self.assertIn("skill 'tdd' in root 'canonical' has 1 directory", warned[0])
            self.assertIn("a floor, not an inventory", warned[0])
        finally:
            os.chmod(self.hidden, 0o755)
        doctor = run_cli([*self.cfg, "--json", "doctor"])
        payload = json.loads(doctor.stdout)
        self.assertEqual(payload["skills_with_unlistable_directories"], [])
        self.assertEqual(
            payload["skills_with_executable_payloads"],
            [{"skill_id": "tdd", "payload_files": ["scripts/hook.sh"]}],
            "listable again, the hook is inventoried",
        )


class AnUnlistableFixturesDirectoryIsRefusedNotAnEmptyWindowTests(unittest.TestCase):
    """Measured before: with the exports directory at mode 000, ``doctor``
    reported the source ok and ``daily`` stood down with zero ingested,
    ``is_total`` true, no error and no note, exit 0: a directory the process
    could not open was indistinguishable from an empty one.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.exports = self.base / "exports"
        shutil.copytree(REPO_ROOT / "tests" / "fixtures" / "exports", self.exports)
        self.cfg = [
            "--config",
            str(build_env(self.base, seed="unlistable-exports", fixtures_dir=self.exports)),
        ]

    @unittest.skipIf(os.geteuid() == 0, "root lists a mode-000 directory")
    def test_doctor_finds_the_problem_and_daily_refuses_instead_of_standing_down(self):
        os.chmod(self.exports, 0o000)
        try:
            doctor = run_cli([*self.cfg, "--json", "doctor"])
            self.assertEqual(doctor.returncode, 1, doctor.stdout)
            payload = json.loads(doctor.stdout)
            self.assertNotEqual(payload["source"]["status"], "ok")
            self.assertTrue(
                any("cannot be listed (Permission denied)" in p for p in payload["problems"]),
                payload["problems"],
            )
            daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
            self.assertEqual(daily.returncode, RC_REFUSED, daily.stdout)
            self.assertIn("SourceUnavailable", daily.stderr)
            self.assertIn("cannot be listed", daily.stderr)
            self.assertIn("not an empty window", daily.stderr)
        finally:
            os.chmod(self.exports, 0o755)


class AnExportThatDoesNotReadAsAThreadIsNamedTests(unittest.TestCase):
    """Measured before: a non-JSON export and a JSON array beside the five good
    ones gave ``doctor`` a clean source and ``daily`` a receipt of five ingested
    with ``is_total`` true and no word about the two.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.exports = self.base / "exports"
        shutil.copytree(REPO_ROOT / "tests" / "fixtures" / "exports", self.exports)
        (self.exports / "0-garbage.json").write_text("{not json", encoding="utf-8")
        (self.exports / "0-list.json").write_text("[1, 2]", encoding="utf-8")
        self.cfg = [
            "--config",
            str(build_env(self.base, seed="passed-over-export", fixtures_dir=self.exports)),
        ]

    def test_doctor_warns_and_the_receipt_names_each_entry_in_no_count(self):
        doctor = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(doctor.returncode, RC_OK, doctor.stderr)
        payload = json.loads(doctor.stdout)
        self.assertEqual(payload["source"]["status"], "ok")
        self.assertEqual(
            [Path(e["entry"]).name for e in payload["source"]["passed_over"]],
            ["0-garbage.json", "0-list.json"],
        )
        warned = [w for w in payload["warnings"] if w.startswith("source entry passed over: ")]
        self.assertEqual(len(warned), 2, payload["warnings"])
        self.assertIn("is JSON but not an object (list)", warned[1])
        text = run_cli([*self.cfg, "doctor"])
        self.assertIn("0-garbage.json is not valid JSON", text.stdout)

        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        receipt = json.loads(daily.stdout)
        self.assertEqual(receipt["counts"]["conversations_ingested"], 5, "the good five")
        self.assertTrue(receipt["discovery"]["is_total"])
        self.assertEqual(
            [Path(e["entry"]).name for e in receipt["discovery"]["passed_over"]],
            ["0-garbage.json", "0-list.json"],
        )
        notes = [n for n in receipt["notes"] if n.startswith("discovery passed over 2 source")]
        self.assertEqual(len(notes), 1, receipt["notes"])
        self.assertIn("0-garbage.json is not valid JSON", notes[0])
        self.assertIn("0-list.json is JSON but not an object (list)", notes[0])
        self.assertIn("in no count here", notes[0])


class ACheckpointDailyWouldRefuseIsNamedByDoctorTests(unittest.TestCase):
    """Measured before: with today's checkpoint truncated, ``doctor`` printed
    ``verdict: ready`` and exited 0; the next ``daily`` then refused by name
    with exit 3. The refusal is right; a preflight that cannot see it coming is
    not. Today's is a PROBLEM (the next run refuses), a past day's a WARN.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.cfg = ["--config", str(build_env(self.base, seed="checkpoint-doctor"))]
        ready = run_cli([*self.cfg, "--json", "doctor"])
        assert ready.returncode == RC_OK, ready.stderr
        self.state = StateStore(self.base / "state")
        self.today = day_key(SystemClock().now())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_todays_truncated_checkpoint_is_a_problem_and_daily_confirms_the_refusal(self):
        bad = self.state.checkpoint_path(self.today)
        bad.write_text('{"schema": "asi.checkpoint.v1", "records": {"a": ', encoding="utf-8")
        as_json = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(as_json.returncode, 1, as_json.stderr)
        payload = json.loads(as_json.stdout)
        self.assertEqual([u["path"] for u in payload["checkpoint_records_unreadable"]], [str(bad)])
        self.assertTrue(any("today's checkpoint cannot be read" in p for p in payload["problems"]))
        as_text = run_cli([*self.cfg, "doctor"])
        self.assertIn("PROBLEM  today's checkpoint cannot be read", as_text.stdout)
        self.assertIn("verdict: problems found", as_text.stdout)
        daily = run_cli([*self.cfg, "daily", "--commit"])
        self.assertEqual(daily.returncode, RC_REFUSED, "the preflight predicted this")
        self.assertIn(str(bad), daily.stderr)

    def test_a_past_days_wrong_shaped_checkpoint_is_a_warning_not_a_problem(self):
        bad = self.state.checkpoint_path("2026-01-01")
        bad.write_text(json.dumps({"records": []}), encoding="utf-8")
        as_json = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(as_json.returncode, RC_OK, as_json.stderr)
        payload = json.loads(as_json.stdout)
        self.assertEqual([u["path"] for u in payload["checkpoint_records_unreadable"]], [str(bad)])
        self.assertIn("'records' holds list, not dict", payload["warnings"][-1])
        self.assertFalse(any("checkpoint" in p for p in payload["problems"]))
        as_text = run_cli([*self.cfg, "doctor"])
        self.assertIn("WARN  another day's checkpoint cannot be read", as_text.stdout)
        self.assertIn("verdict: ready", as_text.stdout)
        self.assertEqual(bad.read_text(encoding="utf-8"), json.dumps({"records": []}), "untouched")

    def test_a_link_at_todays_checkpoint_path_is_a_problem_not_only_a_left_alone_warning(self):
        # Measured by the cycle-69 audit: a link here left the verdict at
        # ready while the next daily exited 3 refusing to read the leaf.
        outside = self.base / "keep.json"
        outside.write_text(json.dumps({"runs": 1, "records": {}}), encoding="utf-8")
        link = self.state.checkpoint_path(self.today)
        link.symlink_to(outside)
        as_json = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(as_json.returncode, 1, as_json.stderr)
        payload = json.loads(as_json.stdout)
        self.assertEqual(payload["checkpoint_records_unreadable"], [], "a link is not a record")
        self.assertTrue(
            any("today's checkpoint path" in p and "symlink" in p for p in payload["problems"])
        )
        self.assertTrue(any(link.name in w for w in payload["state_permissions"]["left_alone"]))
        daily = run_cli([*self.cfg, "daily", "--commit"])
        self.assertEqual(daily.returncode, RC_REFUSED, "the preflight predicted this")
        self.assertTrue(link.is_symlink(), "neither command touched the link")
        self.assertEqual(json.loads(outside.read_text(encoding="utf-8"))["runs"], 1)

    def test_the_control_a_genuine_checkpoint_is_not_reported(self):
        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        self.assertTrue(self.state.checkpoint_path("2026-01-01").is_file())
        as_json = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(as_json.returncode, RC_OK, as_json.stderr)
        payload = json.loads(as_json.stdout)
        self.assertEqual(payload["checkpoint_records_unreadable"], [])
        self.assertFalse(any("checkpoint" in w for w in payload["warnings"] + payload["problems"]))


class AForeignEntryAtTheLockPathIsNamedByDoctorTests(unittest.TestCase):
    """The run lock is the first thing ``daily`` touches. Measured before: a
    link at ``state/locks/daily.lock`` left ``doctor`` at ``ready`` (one
    left-alone WARN) and a directory there was not even a warning, while the
    next ``daily --commit`` exited 3 for the link and 70, with a traceback,
    for the directory.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.cfg = ["--config", str(build_env(self.base, seed="lock-doctor"))]
        ready = run_cli([*self.cfg, "--json", "doctor"])
        assert ready.returncode == RC_OK, ready.stderr
        self.lock = self.base / "state" / "locks" / "daily.lock"
        self.lock.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def assert_problem_then_refusal(self, expect: str) -> None:
        as_json = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(as_json.returncode, 1, as_json.stderr)
        payload = json.loads(as_json.stdout)
        self.assertTrue(
            any(f"the run lock path {self.lock} is {expect}" in p for p in payload["problems"]),
            payload["problems"],
        )
        as_text = run_cli([*self.cfg, "doctor"])
        self.assertIn(f"PROBLEM  the run lock path {self.lock} is {expect}", as_text.stdout)
        daily = run_cli([*self.cfg, "daily", "--commit"])
        self.assertEqual(daily.returncode, RC_REFUSED, daily.stderr)
        self.assertIn(str(self.lock), daily.stderr)
        self.assertNotIn("Traceback", daily.stderr)

    def test_a_link_at_the_lock_path(self):
        outside = self.base / "keep"
        outside.write_text("", encoding="utf-8")
        self.lock.symlink_to(outside)
        self.assert_problem_then_refusal("a symlink")
        self.assertTrue(self.lock.is_symlink(), "neither command touched it")

    def test_a_directory_at_the_lock_path(self):
        self.lock.mkdir()
        self.assert_problem_then_refusal("a directory")
        self.assertTrue(self.lock.is_dir(), "neither command touched it")

    def test_the_control_the_lock_file_a_finished_run_leaves_is_not_reported(self):
        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        self.assertTrue(self.lock.is_file())
        as_json = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(as_json.returncode, RC_OK, as_json.stderr)
        payload = json.loads(as_json.stdout)
        self.assertFalse(any("run lock" in p for p in payload["problems"] + payload["warnings"]))


class ReadOnlyCommandsTests(unittest.TestCase):
    """README's command table says `skills`, `rubrics`, `discover`, `candidates`,
    `review` and `scheduler-template` write nothing. Measured through the real
    CLI against a fresh environment whose state tree was populated by exactly one
    committed `daily` pass and nothing else, so a write these commands make on
    their own cannot already be on disk when the snapshot is taken, and a rewrite
    of something `daily` wrote shows up through its mtime. Files and directories,
    content, mode and mtime, over the environment and the shared fixture
    corpus."""

    READ_ONLY = (
        ("skills",),
        ("rubrics",),
        ("--json", "discover"),
        ("candidates",),
        ("--json", "review", "{candidate}"),
        ("scheduler-template", "launchd"),
        ("scheduler-template", "cron"),
    )

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.cfg = ["--config", str(build_env(self.base, dry_run=False, seed="ro-seed"))]
        self.fixtures = Path(__file__).resolve().parent / "fixtures" / "exports"

    def commands(self, candidate: str) -> list[tuple[str, ...]]:
        return [tuple(a.format(candidate=candidate) for a in args) for args in self.READ_ONLY]

    def test_against_a_populated_state_tree_nothing_changes(self):
        proc = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        candidates = sorted((self.base / "state" / "candidates").glob("*.json"))
        self.assertTrue(candidates, "the committed pass must leave a candidate to review")
        candidate = json.loads(candidates[0].read_text(encoding="utf-8"))["id"]
        before = _tree_snapshot(self.base, self.fixtures)
        self.assertTrue(any(k.startswith("exports/") for k in before), "fixtures are in scope")
        for args in self.commands(candidate):
            with self.subTest(command=args):
                proc = run_cli([*self.cfg, *args])
                self.assertEqual(proc.returncode, RC_OK, proc.stderr)
                after = _tree_snapshot(self.base, self.fixtures)
                changed = sorted(
                    k for k in set(before) | set(after) if before.get(k) != after.get(k)
                )
                self.assertEqual(changed, [], f"{args} changed these paths")

    def test_against_an_absent_state_tree_only_owner_only_directories_may_appear(self):
        """Without a state tree, the commands that read one create the empty
        owner-only directories they look in, and nothing else; the commands that
        never touch state leave it absent. A directory they look in that already
        exists with a wider mode is reset to owner-only, which is the "modes only"
        the README admits to."""
        state = self.base / "state"
        self.assertFalse(state.exists())
        for args in (("skills",), ("rubrics",), ("scheduler-template", "cron")):
            with self.subTest(command=args):
                proc = run_cli([*self.cfg, *args])
                self.assertEqual(proc.returncode, RC_OK, proc.stderr)
                self.assertFalse(state.exists(), f"{args} created a state tree")
        for args in (("--json", "discover"), ("candidates",)):
            with self.subTest(command=args):
                proc = run_cli([*self.cfg, *args])
                self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        # `review` of a candidate that does not exist is the only way to run it
        # without a state tree; it must still create no file on the way out.
        proc = run_cli([*self.cfg, "review", "CAND-2026-01-01-01-none"])
        self.assertEqual(proc.returncode, RC_MISSING_FILE, proc.stderr)
        self.assertTrue(state.is_dir(), "discover and candidates create the tree they read")
        files = [p for p in state.rglob("*") if not p.is_dir()]
        self.assertEqual(files, [], "a read-only command wrote a file")
        wide = [p for p in [state, *state.rglob("*")] if p.stat().st_mode & 0o077]
        self.assertEqual(wide, [], "a read-only command left a path readable beyond its owner")
        state.chmod(0o755)
        (state / "candidates").chmod(0o755)
        proc = run_cli([*self.cfg, "candidates"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertEqual((state / "candidates").stat().st_mode & 0o777, 0o700, "modes are reset")
        self.assertEqual(state.stat().st_mode & 0o777, 0o700, "the root it looked through too")
        self.assertEqual([p for p in state.rglob("*") if not p.is_dir()], [], "still no file")


class AValueThatCannotBeUsedIsAUsageErrorBeforeTheRunTests(unittest.TestCase):
    """``--day`` names every candidate of the run and ``--limit`` is a page
    size, so a value that cannot serve is refused by argparse (exit 2), before
    the config is read or a conversation ingested. Measured before: ``--day
    2026/01/02`` ran the whole pass and died on the first candidate id;
    ``--limit -1`` printed ``5 of at most -1`` and called it a whole window."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.cfg = ["--config", str(build_env(self.base, dry_run=False, seed="s"))]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_slash_date_is_refused_by_argparse_with_the_spelling_it_wanted(self):
        proc = run_cli([*self.cfg, "daily", "--commit", "--day", "2026/01/02"])
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("argument --day: run day '2026/01/02' cannot name a candidate", proc.stderr)
        self.assertIn("'2026-01-02'", proc.stderr)
        self.assertNotIn("invalid candidate id", proc.stderr)
        self.assertFalse((self.base / "state" / "conversations").exists(), "nothing was ingested")

    def test_a_whitespace_day_is_refused_but_an_empty_one_still_means_today(self):
        proc = run_cli([*self.cfg, "daily", "--day", "  "])
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("argument --day", proc.stderr)
        proc = run_cli([*self.cfg, "--json", "daily", "--day", ""])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertRegex(json.loads(proc.stdout)["day"], r"^\d{4}-\d{2}-\d{2}$")

    def test_a_page_size_below_one_is_refused_and_one_is_honoured(self):
        for bad in ("0", "-1", "two"):
            with self.subTest(limit=bad):
                proc = run_cli([*self.cfg, "--json", "discover", "--limit", bad])
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn("argument --limit", proc.stderr)
        proc = run_cli([*self.cfg, "--json", "discover", "--limit", "1"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["page"]["limit"], 1)
        self.assertEqual(len(payload["threads"]), 1)


class ADocumentWithAByteOrderMarkIsImprovableTests(unittest.TestCase):
    """Measured before: with the canonical and mirror ``tdd`` both beginning
    with a UTF-8 byte-order mark, ``skills`` reported ``tdd`` as a document that
    "will never load", ``daily --commit`` scored every conversation and then
    quarantined the proposal as having no opening delimiter, and no candidate
    was ever produced for the skill."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = [
            "--config",
            str(build_env(self.base, use_subprocess_model=True, dry_run=False, seed="bom")),
        ]
        self.skill = self.base / "skills-canonical" / "tdd" / "SKILL.md"
        marked = b"\xef\xbb\xbf" + self.skill.read_bytes()
        self.skill.write_bytes(marked)
        (self.base / "skills-mirror" / "tdd" / "SKILL.md").write_bytes(marked)

    def test_the_skill_loads_is_proposed_against_and_keeps_its_mark_through_apply(self):
        skills = run_cli([*self.cfg, "--json", "skills"])
        self.assertEqual(skills.returncode, RC_OK, skills.stderr)
        group = next(g for g in json.loads(skills.stdout)["groups"] if g["skill_id"] == "tdd")
        self.assertTrue(group["canonical"]["loads"])
        self.assertTrue(any("U+FEFF" in w for w in group["canonical"]["warnings"]))
        text = run_cli([*self.cfg, "skills"])
        tdd_lines = [ln for ln in text.stdout.split("\n") if ln.startswith("tdd ")]
        self.assertEqual(len(tdd_lines), 1, text.stdout)
        self.assertNotIn("DOES NOT LOAD", tdd_lines[0])
        self.assertIn("byte-order mark (U+FEFF)", text.stdout)

        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-02"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        payload = json.loads(daily.stdout)
        validated = [c for c in payload["candidates"] if c["status"] == "validated"]
        self.assertEqual(len(validated), 1, payload.get("skills"))
        self.assertEqual(payload["skills"], {}, "nothing quarantined")

        review = json.loads(
            run_cli([*self.cfg, "--json", "review", validated[0]["candidate_id"]]).stdout
        )
        applied = run_cli(
            [
                *self.cfg,
                "apply",
                validated[0]["candidate_id"],
                "--approve",
                review["approval_token"],
                "--i-reviewed-this",
            ]
        )
        self.assertEqual(applied.returncode, RC_OK, applied.stderr)
        after = self.skill.read_bytes()
        self.assertTrue(after.startswith(b"\xef\xbb\xbf---\n"), after[:16])
        self.assertIn(FIX_MARKER.encode(), after)


class SchedulerTemplatesQuoteTheirPathsTests(unittest.TestCase):
    """The templates drop four paths into two foreign syntaxes: a crontab line,
    which the shell splits on spaces and cron turns ``%`` into a newline on, and
    a property list, which is XML. Measured before this class existed, a
    configured tree under a directory such as ``My Projects & co 100%`` rendered
    a cron job that ran ``cd /.../My`` and a plist that did not parse. A path with
    a space or an ampersand is ordinary on macOS (``Application Support``), so the
    substitution must speak each syntax: shell quoting plus ``\\%`` for cron,
    entity escaping for XML.
    """

    AWKWARD = "My Projects & co 100%"

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve() / self.AWKWARD
        self.base.mkdir()
        self.config_path = Path(build_env(self.base, dry_run=False)).resolve()
        self.state = self.base / "state"

    def _job_line(self, text: str) -> str:
        jobs = [
            line
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#") and "daily --commit" in line
        ]
        self.assertEqual(len(jobs), 1, text)
        return jobs[0]

    def test_the_cron_job_survives_the_shell_and_cron_percent_handling(self):
        import shlex

        proc = run_cli(["--config", str(self.config_path), "scheduler-template", "cron"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        line = self._job_line(proc.stdout)
        command = line.split(None, 5)[5]
        # Every percent sign cron would see must be escaped for it.
        self.assertNotIn("%", command.replace("\\%", ""))
        # What cron hands the shell after consuming the backslashes.
        shell_sees = command.replace("\\%", "%")
        tokens = shlex.split(shell_sees)
        self.assertEqual(tokens[tokens.index("cd") + 1], str(REPO_ROOT))
        self.assertEqual(tokens[tokens.index("--config") + 1], str(self.config_path))
        self.assertEqual(tokens[tokens.index(">>") + 1], str(self.state / "daily.out.log"))
        self.assertEqual(tokens[tokens.index("2>>") + 1], str(self.state / "daily.err.log"))
        self.assertIn("daily", tokens)
        self.assertIn("--commit", tokens)

    def test_the_launchd_plist_still_parses_and_names_the_exact_paths(self):
        import plistlib

        proc = run_cli(["--config", str(self.config_path), "scheduler-template", "launchd"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        parsed = plistlib.loads(proc.stdout.encode("utf-8"))
        argv = parsed["ProgramArguments"]
        self.assertEqual(argv[argv.index("--config") + 1], str(self.config_path))
        self.assertEqual(parsed["WorkingDirectory"], str(REPO_ROOT))
        self.assertEqual(parsed["StandardOutPath"], str(self.state / "daily.out.log"))
        self.assertEqual(parsed["StandardErrorPath"], str(self.state / "daily.err.log"))

    def test_cron_quoting_round_trips_through_crons_own_rewrite(self):
        """cron's command rewrite (Vixie ``do_command.c``, shared by cronie and
        macOS): a backslash escapes the next character and is consumed, a bare
        ``%`` ends the command. Modelled here on the quoted text, then handed
        to a real shell, the original path must come back whole."""
        import shlex
        import subprocess

        from agent_skill_improver.cli import _cron_quote

        def cron_rewrites(command: str) -> str:
            out, escaped = [], False
            for ch in command:
                if escaped:
                    out.append(ch if ch in "%\\" else "\\" + ch)
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "%":
                    raise AssertionError(f"cron would cut the command at a bare %: {command!r}")
                else:
                    out.append(ch)
            return "".join(out)

        for value in (
            "/plain/path",
            "/My Projects & co 100%/state",
            "/a/100%",
            "/a/b\\%c",
            "/a/b\\\\c",
            "/it's/here",
        ):
            with self.subTest(value=value):
                shell_sees = cron_rewrites(_cron_quote(value))
                self.assertEqual(shlex.split(shell_sees), [value])
                proc = subprocess.run(
                    ["/bin/sh", "-c", f"printf %s {shell_sees}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.stdout, value, proc.stderr)

    def test_the_control_a_plain_path_renders_byte_for_byte_as_before(self):
        """Quoting must be invisible when nothing needs it: an operator who has
        pasted this line for months must see the same text."""
        with TemporaryDirectory() as plain_tmp:
            plain = Path(plain_tmp).resolve() / "plain"
            plain.mkdir()
            cfg = Path(build_env(plain, dry_run=False)).resolve()
            proc = run_cli(["--config", str(cfg), "scheduler-template", "cron"])
            self.assertEqual(proc.returncode, RC_OK, proc.stderr)
            line = self._job_line(proc.stdout)
            # Only the two paths this test controls are held to "unquoted": the
            # repository and interpreter paths are the machine's, and a checkout
            # under a directory with a space is entitled to see them quoted.
            self.assertIn(f"--config {cfg} daily --commit >> {plain / 'state'}/daily.out.log", line)
            self.assertNotIn(f"'{cfg}'", line)
            self.assertNotIn(f"'{plain / 'state'}'", line)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
