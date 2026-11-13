"""An apply step added to a marked scheduler job is refused, not authenticated.

Measured before on 3.11 and 3.13: neither CLI-rendered template set a marker;
daily exited 0, then a separate noninteractive apply exited 0 and changed the
fixture skill. Daily's marker is only process-local. The same unmarked apply
still succeeds: a token and acknowledgement are not caller authentication.
"""

from __future__ import annotations

import json
import os
import plistlib
import re
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tests.support import build_env, run_cli

MARKER = "ASI_AUTOMATED_RUN"
DAY = "2026-02-01"
AUTOMATION_MESSAGE = "the automation marker (ASI_AUTOMATED_RUN) is set"


def cron_environment_before_job(text: str) -> dict[str, str]:
    """Read environment assignments in order, stopping at the first job.

    This models cronie, Vixie and OpenBSD, not BusyBox's limited variable list.
    A marker after the job cannot mark that job.
    """
    environment: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assignment = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)=(.*)", line)
        if assignment is None:
            break
        environment[assignment[1]] = assignment[2]
    return environment


class AnApplyStepAddedToASchedulerTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.cfg = ["--config", str(build_env(self.base, dry_run=False, seed="c239"))]
        # The unmarked control must have no variable at all, not just a falsey
        # value. Restore the caller's environment when this test ends.
        unmarked = mock.patch.dict(os.environ)
        unmarked.start()
        self.addCleanup(unmarked.stop)
        os.environ.pop(MARKER, None)
        self.clean: dict[str, str] = {}
        daily = run_cli([*self.cfg, "daily", "--commit", "--day", DAY], env=self.clean)
        self.assertEqual(daily.returncode, 0, daily.stderr)
        listed = run_cli([*self.cfg, "--json", "candidates"], env=self.clean)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.cid = sorted(c["id"] for c in json.loads(listed.stdout)["candidates"])[0]
        reviewed = run_cli([*self.cfg, "--json", "review", self.cid], env=self.clean)
        self.assertEqual(reviewed.returncode, 0, reviewed.stderr)
        self.token = json.loads(reviewed.stdout)["approval_token"]
        self.record = self.base / "state" / "candidates" / f"{self.cid}.json"
        target = json.loads(self.record.read_text())["target"]
        self.target = Path(target["canonical_path"])
        self.before = self.target.read_bytes()
        self.apply = [*self.cfg, "apply", self.cid, "--approve", self.token, "--i-reviewed-this"]

    def render(self, flavor: str) -> str:
        rendered = run_cli([*self.cfg, "scheduler-template", flavor], env=self.clean)
        self.assertEqual(rendered.returncode, 0, rendered.stderr)
        return rendered.stdout

    def assert_refused(self, applied: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(applied.returncode, 3, applied.stdout + applied.stderr)
        self.assertIn("AutomationRefused", applied.stderr)
        self.assertIn(AUTOMATION_MESSAGE, applied.stderr)
        self.assertEqual(self.target.read_bytes(), self.before)
        self.assertEqual(json.loads(self.record.read_text())["status"], "validated")
        self.assertFalse(list((self.base / "state" / "backups").rglob("backup.json")))
        self.assertFalse((self.base / "state" / "receipts" / f"apply--{self.cid}.json").exists())

    def shell(self, command: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment.pop(MARKER, None)
        environment["PYTHONPATH"] = os.pathsep.join(sys.path)
        return subprocess.run(
            ["/bin/sh", "-c", command],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def test_an_apply_on_a_later_cron_line_inherits_the_marker_and_is_refused(self) -> None:
        environment = cron_environment_before_job(self.render("cron"))
        self.assert_refused(run_cli(self.apply, env={**self.clean, **environment}))
        self.assertEqual(environment, {MARKER: "1"})

    def test_an_apply_added_to_the_launchd_job_inherits_the_marker_and_is_refused(self) -> None:
        job = plistlib.loads(self.render("launchd").encode("utf-8"))
        environment = job.get("EnvironmentVariables", {})
        self.assert_refused(run_cli(self.apply, env={**self.clean, **environment}))
        self.assertEqual(environment, {MARKER: "1"})

    def test_an_apply_appended_to_the_cron_command_is_refused_even_on_busybox(self) -> None:
        # BusyBox ignores the environment line; execute the template's command
        # with no inherited marker, including daily then apply in the same shell.
        jobs = [
            line for line in self.render("cron").splitlines() if line.lstrip().startswith("17 ")
        ]
        self.assertEqual(len(jobs), 1)
        command = jobs[0].split(None, 5)[5]
        command = command.replace("daily --commit", f"daily --commit --day {DAY}")
        apply = shlex.join([sys.executable, "-m", "agent_skill_improver", *self.apply])
        self.assert_refused(self.shell(command + "; " + apply))

    def test_candidates_and_doctor_accept_each_templates_marker(self) -> None:
        for flavor in ("cron", "launchd"):
            rendered = self.render(flavor)
            environment = (
                cron_environment_before_job(rendered)
                if flavor == "cron"
                else plistlib.loads(rendered.encode())["EnvironmentVariables"]
            )
            for command in ("candidates", "doctor"):
                with self.subTest(flavor=flavor, command=command):
                    proc = run_cli([*self.cfg, command], env={**self.clean, **environment})
                    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                    self.assertNotIn("AutomationRefused", proc.stderr)

    def test_the_weekly_cron_commands_export_the_marker_and_still_work(self) -> None:
        for name in ("candidates", "doctor"):
            with self.subTest(command=name):
                lines = [
                    line.removeprefix("#").strip()
                    for line in self.render("cron").splitlines()
                    if line.startswith("#  ") and line.endswith(" " + name)
                ]
                self.assertEqual(len(lines), 1)
                command = lines[0].split(None, 5)[5]
                # A child sees only exported variables, not shell-local ones.
                child = shlex.join(["/bin/sh", "-c", 'printf "\nmarker=%s\n" "$ASI_AUTOMATED_RUN"'])
                proc = self.shell(command + " && " + child)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertTrue(proc.stdout.endswith("\nmarker=1\n"), proc.stdout)

    def test_a_separate_unmarked_apply_after_daily_still_succeeds_without_a_terminal(self) -> None:
        # run_cli uses pipes, not a tty. This is the old template-job behaviour,
        # and remains possible in a separately authored unmarked scheduler job.
        applied = run_cli(self.apply, env=self.clean)
        self.assertEqual(applied.returncode, 0, applied.stderr)
        self.assertNotEqual(self.target.read_bytes(), self.before)
        self.assertEqual(json.loads(self.record.read_text())["status"], "applied")


if __name__ == "__main__":
    unittest.main()
