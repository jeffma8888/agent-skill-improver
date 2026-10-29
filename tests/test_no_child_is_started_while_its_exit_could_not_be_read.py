"""No child is started while its exit could not be read.

With SIGCHLD ignored in the run the kernel reaps each child itself as it
exits, and ``subprocess`` then reads every exit as 0. Measured before, on both
Pythons: a role's answer that exited 3 was taken, an export that exited 1 was
taken as the export, the kill at the output cap went to an id the kernel had
already freed, and what a role started ran on after the call, 5 calls of 5
each. A process started with SIGCHLD ignored that called the library rather
than ``asi`` (which puts SIGCHLD back as it starts) left what a role started
running on macOS too, 5 of 5. ``run_bounded`` now starts nothing while SIGCHLD
is ignored and raises ``ExitUnreadable``, an ``OSError``, which each caller
reports as a command that cannot be started.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_skill_improver.adapters import bounded_process
from agent_skill_improver.adapters.bounded_process import ExitUnreadable, run_bounded
from agent_skill_improver.adapters.conversation_cli import ConversationCliSource
from agent_skill_improver.adapters.model_command import CommandModelRunner, RoleCommand
from agent_skill_improver.adapters.sandbox import NoSandbox
from agent_skill_improver.errors import AsiError, SourceUnavailable, TransientFailure

from tests.test_a_role_leaves_nothing_running_in_its_group_when_it_exits import (
    QUIET,
    REPLY,
    _end,
    _still_running,
    _token,
)

REFUSAL = "SIGCHLD is ignored in this process"

#: Run as ``python -c`` in a process its parent started with SIGCHLD ignored:
#: a role called through the library, not through ``asi``.
LIBRARY_CALL = """import json, signal, sys
from pathlib import Path
from agent_skill_improver.adapters.model_command import CommandModelRunner, RoleCommand
from agent_skill_improver.adapters.sandbox import NoSandbox
command = RoleCommand(role="scorer", argv=("/bin/sh", "-c", sys.argv[2]), timeout_seconds=20)
runner = CommandModelRunner(
    {"scorer": command},
    isolation=NoSandbox(),
    workdir_root=Path(sys.argv[1]),
    parent_env={"PATH": "/usr/bin:/bin"},
)
found = {"started_ignored": signal.getsignal(signal.SIGCHLD) is signal.SIG_IGN}
try:
    found["answer"] = runner.run(role="scorer", system="s", user="u", response_schema="x")
except Exception as exc:
    found["failure"] = str(exc)
print(json.dumps(found))
"""


class _Case(unittest.TestCase):
    ignored = True

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.tmp)])
        previous = signal.getsignal(signal.SIGCHLD)
        self.addCleanup(signal.signal, signal.SIGCHLD, previous)
        signal.signal(signal.SIGCHLD, signal.SIG_IGN if self.ignored else signal.SIG_DFL)
        self.sent: list[int] = []

        def killpg(pid: int, sig: int) -> None:
            self.sent.append(pid)
            raise ProcessLookupError(pid)

        patcher = mock.patch.object(bounded_process.os, "killpg", side_effect=killpg)
        patcher.start()
        self.addCleanup(patcher.stop)

    def runner(self, script: str, **limits: int) -> CommandModelRunner:
        root = Path(tempfile.mkdtemp(dir=self.tmp))
        return CommandModelRunner(
            {
                "scorer": RoleCommand(
                    role="scorer", argv=("/bin/sh", "-c", script), timeout_seconds=20, **limits
                )
            },
            isolation=NoSandbox(),
            workdir_root=root,
            parent_env={"PATH": "/usr/bin:/bin"},
        )

    def started(self, call):  # type: ignore[no-untyped-def]
        """Run ``call`` and return how many children it started, and what it raised."""
        with mock.patch.object(
            bounded_process.subprocess, "Popen", wraps=subprocess.Popen
        ) as popen:
            try:
                call()
            except Exception as exc:  # noqa: BLE001 - the test reads what was raised
                return popen.call_count, exc
        return popen.call_count, None


class WithSigchldIgnoredInTheRunTests(_Case):
    def test_with_sigchld_ignored_a_role_is_not_started_and_its_answer_is_not_taken(self):
        runner = self.runner(f"{REPLY}; exit 3")
        count, raised = self.started(
            lambda: runner.run(role="scorer", system="s", user="u", response_schema="x")
        )
        self.assertIsInstance(raised, TransientFailure, "an answer from a role that exited 3")
        self.assertIn("cannot be started", str(raised))
        self.assertIn(REFUSAL, str(raised))
        self.assertEqual(count, 0, "a child was started")

    def test_with_sigchld_ignored_an_export_is_not_started_nor_taken(self):
        source = ConversationCliSource(timeout_seconds=20)
        count, raised = self.started(
            lambda: source._run(["/bin/sh", "-c", "printf %s partial; exit 1"])
        )
        self.assertIsInstance(raised, SourceUnavailable, "an export that exited 1 was taken")
        self.assertIn(REFUSAL, str(raised))
        self.assertEqual(count, 0, "a child was started")

    def test_with_sigchld_ignored_a_role_past_its_cap_signals_no_id(self):
        runner = self.runner(
            "printf %s 0123456789012345678901234567890123456789", max_output_bytes=8
        )
        _count, raised = self.started(
            lambda: runner.run(role="scorer", system="s", user="u", response_schema="x")
        )
        self.assertIsInstance(raised, AsiError)
        self.assertIn(REFUSAL, str(raised))
        self.assertEqual(self.sent, [], "an id the kernel may have freed was signalled")

    def test_with_sigchld_ignored_nothing_a_role_would_start_runs(self):
        token = _token()
        self.addCleanup(_end, token)
        runner = self.runner(f"(sleep {token}) {QUIET} &\n{REPLY}")
        _count, raised = self.started(
            lambda: runner.run(role="scorer", system="s", user="u", response_schema="x")
        )
        self.assertIn(REFUSAL, str(raised))
        self.assertEqual(_still_running(token), [], "what the role started outlived the call")

    def test_run_bounded_refuses_before_it_starts_anything(self):
        count, raised = self.started(
            lambda: run_bounded(["/bin/sh", "-c", "exit 0"], timeout_seconds=5, max_output_bytes=8)
        )
        self.assertIsInstance(raised, ExitUnreadable)
        self.assertIsInstance(raised, OSError)
        self.assertEqual(count, 0)


class AtItsDefaultTests(_Case):
    ignored = False

    def test_the_control_with_sigchld_at_its_default_a_child_runs_as_before(self):
        failed = self.runner(f"{REPLY}; exit 3")
        count, raised = self.started(
            lambda: failed.run(role="scorer", system="s", user="u", response_schema="x")
        )
        self.assertEqual(count, 1)
        self.assertIsInstance(raised, TransientFailure)
        self.assertIn("exited 3", str(raised))
        answered = self.runner(REPLY)
        reply = answered.run(role="scorer", system="s", user="u", response_schema="x")
        self.assertEqual(reply, {"x": 1})
        source = ConversationCliSource(timeout_seconds=20)
        _count, raised = self.started(
            lambda: source._run(["/bin/sh", "-c", "printf %s partial; exit 1"])
        )
        self.assertIsInstance(raised, SourceUnavailable)
        self.assertIn("exited 1", str(raised))


class ALibraryCallStartedWithSigchldIgnoredTests(unittest.TestCase):
    def test_a_library_call_started_with_sigchld_ignored_is_refused(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(root)])
        token = _token()
        self.addCleanup(_end, token)
        here = Path(__file__).resolve()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env["PYTHONPATH"] = os.pathsep.join([str(here.parents[1] / "src"), str(here.parents[1])])
        proc = subprocess.run(
            [sys.executable, "-c", LIBRARY_CALL, str(root), f"(sleep {token}) {QUIET} &\n{REPLY}"],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            preexec_fn=lambda: signal.signal(signal.SIGCHLD, signal.SIG_IGN),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        found = json.loads(proc.stdout.splitlines()[-1])
        self.assertTrue(found["started_ignored"], "the call was not started with SIGCHLD ignored")
        self.assertNotIn("answer", found)
        self.assertIn(REFUSAL, found["failure"])
        self.assertEqual(_still_running(token), [], "what the role started outlived the call")


if __name__ == "__main__":
    unittest.main()
