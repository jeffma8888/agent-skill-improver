"""``doctor`` names SIGCHLD ignored in the process it runs in as a problem.

``asi`` puts SIGCHLD back to its default as it starts; a program that calls
``cli.main`` keeps the disposition it has, and while that is an ignore no role
or source command is started (``ExitUnreadable``), so a run scores nothing.
Measured before (3.11 and 3.13, the same on both): ``main(["doctor"])`` with
SIGCHLD ignored said ``verdict: ready`` with exit 0, and ``main(["daily",
"--commit"])`` called the same way then scored none of 5 conversations and
deferred all 5. ``doctor`` now asks the question ``run_bounded`` asks
(``children_reaped_by_the_kernel``) and reports it as a problem, exit 1.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver import cli
from tests.support import REPO_ROOT, build_env

SIGCHLD = getattr(signal, "SIGCHLD", None)
NAMED = "SIGCHLD is ignored in this process, so no role or source command would be started"


@unittest.skipIf(SIGCHLD is None, "this platform has no SIGCHLD")
class DoctorNamesAnIgnoredSigchldTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.config = str(build_env(Path(tmp.name), seed="c208"))
        self.addCleanup(signal.signal, SIGCHLD, signal.getsignal(SIGCHLD))

    def doctor(self, *extra: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = cli.main(["--config", self.config, *extra, "doctor"])
        return rc, out.getvalue()

    def test_a_caller_of_main_that_ignores_sigchld_is_told_it_is_not_ready(self):
        signal.signal(SIGCHLD, signal.SIG_IGN)
        rc, text = self.doctor()
        self.assertEqual(rc, 1)
        self.assertEqual(
            [line for line in text.splitlines() if NAMED in line][:1],
            [
                "PROBLEM  "
                + NAMED
                + " (a child's exit status could not be read) and the run would "
                "score nothing; `asi` puts SIGCHLD back to its default as it starts, and a program "
                "that calls `main` must do the same first"
            ],
        )
        self.assertIn("verdict: problems found", text.splitlines())

    def test_the_json_report_lists_it_among_the_problems(self):
        signal.signal(SIGCHLD, signal.SIG_IGN)
        rc, text = self.doctor("--json")
        self.assertEqual(rc, 1)
        problems = json.loads(text)["problems"]
        self.assertEqual(sum(1 for p in problems if p.startswith(NAMED)), 1)

    def test_at_the_default_it_is_not_named(self):
        signal.signal(SIGCHLD, signal.SIG_DFL)
        rc, text = self.doctor()
        self.assertEqual(rc, 0, text)
        self.assertNotIn("SIGCHLD", text)
        self.assertIn("verdict: ready", text.splitlines())

    def test_asi_started_with_sigchld_ignored_puts_it_back_and_is_ready(self):
        proc = subprocess.run(
            [sys.executable, "-m", "agent_skill_improver", "--config", self.config, "doctor"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=_env(),
            timeout=300,
            preexec_fn=lambda: signal.signal(SIGCHLD, signal.SIG_IGN),
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn("SIGCHLD", proc.stdout + proc.stderr)


def _env() -> dict[str, str]:
    env = dict(os.environ, PYTHONPATH=f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}")
    env.pop("ASI_CONFIG", None)
    return env


if __name__ == "__main__":
    unittest.main()
