"""One full pass through the real command line, with a real subprocess model.

Everything else in this suite calls functions. This module runs ``asi`` the way an
operator and a scheduler do: as a process, with exit codes, reading only what the
CLI prints. It is the only place that would catch a broken argument parser, a
wrong exit code, or an interlock that works in-process and not through the CLI.
"""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.cli import RC_MISSING_FILE, RC_OK, RC_REFUSED
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
