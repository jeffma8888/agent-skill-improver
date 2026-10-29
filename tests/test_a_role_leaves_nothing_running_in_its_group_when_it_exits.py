"""Nothing a model role started in its process group outlives the call.

Each role runs as the leader of a process group of its own, and the bounded
reader killed that group only while the role itself still ran: at the deadline,
at the output cap, or on a stop signal. A role that exited on its own left
whatever it had started running. Measured before this change, with no sandbox
and under ``sandbox-exec``, on both Pythons: a background ``sleep``, the same
under ``nohup``, a subshell ignoring the hang-up and termination signals, a
double fork, a loop of 200 background jobs and a background job of a role that
then exited 3 all ran on after the call returned, and a job that made the
role's scratch again after its removal and wrote there left it in the state
tree. The group is now ended once the role has exited and before it is reaped,
so its id cannot have passed to another process.
"""

from __future__ import annotations

import contextlib
import errno
import io
import itertools
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from agent_skill_improver.adapters import bounded_process
from agent_skill_improver.adapters.conversation_cli import ConversationCliSource
from agent_skill_improver.adapters.model_command import CommandModelRunner, RoleCommand
from agent_skill_improver.adapters.sandbox import NoSandbox, OsSandbox, detect
from agent_skill_improver.errors import AsiError, TransientFailure

REPLY = 'printf %s "{\\"x\\":1}"'
QUIET = "</dev/null >/dev/null 2>&1"
_SERIAL = itertools.count(1)


def _token() -> str:
    """A ``sleep`` length no other process on the machine is running."""
    return f"600.{os.getpid()}{next(_SERIAL):03d}"


def _running(token: str) -> list[int]:
    """The live processes whose command line holds ``token`` (zombies are not)."""
    listing = subprocess.run(
        ["ps", "-A", "-o", "pid=,stat=,args="], capture_output=True, text=True
    ).stdout
    found = []
    for line in listing.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and token in parts[2] and not parts[1].startswith("Z"):
            if int(parts[0]) != os.getpid():
                found.append(int(parts[0]))
    return found


def _still_running(token: str, *, within: float = 3.0) -> list[int]:
    """What still runs with ``token`` once a SIGKILL has had time to land."""
    end = time.monotonic() + within
    while True:
        left = _running(token)
        if not left or time.monotonic() > end:
            return left
        time.sleep(0.05)


def _end(token: str) -> None:
    for pid in _running(token):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


class _RoleCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.tmp)])

    def _call(self, script: str, *, isolation=None, timeout: float = 20, token: str = ""):
        if token:
            self.addCleanup(_end, token)
        root = self.tmp / f"state-{next(_SERIAL)}"
        root.mkdir()
        runner = CommandModelRunner(
            {
                "scorer": RoleCommand(
                    role="scorer", argv=("/bin/sh", "-c", script), timeout_seconds=timeout
                )
            },
            isolation=isolation or NoSandbox(),
            workdir_root=root,
            parent_env={"PATH": "/usr/bin:/bin"},
        )
        return root, runner


class WhatARoleLeavesInItsGroupIsEndedTests(_RoleCase):
    def _assert_nothing_left(self, script: str, token: str, *, isolation=None) -> None:
        _root, runner = self._call(script, isolation=isolation, token=token)
        self.assertEqual(
            runner.run(role="scorer", system="s", user="u", response_schema="x"), {"x": 1}
        )
        self.assertEqual(_still_running(token), [], "a process the role left is still running")

    def test_what_a_role_leaves_running_in_its_group_is_ended_when_it_exits(self):
        for kind, form in (
            ("a background job", "(sleep {t}) {q} &"),
            ("one under nohup", "nohup sleep {t} {q} &"),
            (
                "a subshell ignoring hang-up and termination",
                "(trap '' HUP TERM INT; sleep {t}) {q} &",
            ),
            ("a double fork", "( (sleep {t} {q} &) ) {q} &"),
            (
                "a loop of 50 jobs",
                "(i=0; while [ $i -lt 50 ]; do sleep {t} & i=$((i+1)); done) {q} &",
            ),
        ):
            with self.subTest(kind):
                token = _token()
                self._assert_nothing_left(form.format(t=token, q=QUIET) + "\n" + REPLY, token)

    def test_a_child_forked_as_the_group_is_signalled_is_reached_too(self):
        """One signal misses a child forked as it arrives (measured on macOS:
        most runs of this case); the rounds reach it."""
        for run in range(6):
            with self.subTest(run=run):
                token = _token()
                script = f"(trap '' HUP TERM INT; sleep {token}) {QUIET} &\n{REPLY}"
                self._assert_nothing_left(script, token)

    def test_a_role_that_exits_non_zero_leaves_nothing_running_either(self):
        token = _token()
        _root, runner = self._call(f"(sleep {token}) {QUIET} &\nexit 3", token=token)
        with self.assertRaises(TransientFailure) as ctx:
            runner.run(role="scorer", system="s", user="u", response_schema="x")
        self.assertIn("exited 3", str(ctx.exception))
        self.assertEqual(_still_running(token), [])

    @unittest.skipIf(detect() is None, "no OS sandbox on this machine")
    def test_a_confined_role_leaves_nothing_running_and_no_scratch(self):
        mechanism = detect()
        token = _token()
        self._assert_nothing_left(
            f"(trap '' HUP TERM INT; sleep {token}) {QUIET} &\n{REPLY}",
            token,
            isolation=OsSandbox(mechanism),
        )
        late = f'(sleep 1; mkdir -p "$HOME"; echo late > "$HOME/late") {QUIET} &\n{REPLY}'
        root, runner = self._call(late, isolation=OsSandbox(mechanism))
        runner.run(role="scorer", system="s", user="u", response_schema="x")
        time.sleep(2)
        self.assertEqual(sorted(p.name for p in root.iterdir()), [], "the scratch came back")

    def test_a_role_that_keeps_working_after_closing_its_output_is_not_cut_short(self):
        """The group is ended once the role has exited, not when its output closes."""
        done = self.tmp / "done"
        script = f'{REPLY}\nexec >&- 2>&-\nsleep 0.5\n: > "{done}"\nexit 0'
        _root, runner = self._call(script)
        started = time.monotonic()
        self.assertEqual(
            runner.run(role="scorer", system="s", user="u", response_schema="x"), {"x": 1}
        )
        self.assertGreaterEqual(time.monotonic() - started, 0.5)
        self.assertTrue(done.exists(), "the role was killed before it finished")

    def test_a_role_still_running_at_the_deadline_after_closing_its_output_times_out(self):
        token = _token()
        script = f"{REPLY}\nexec >&- 2>&-\nsleep {token}"
        _root, runner = self._call(script, timeout=1, token=token)
        with self.assertRaises(TransientFailure) as ctx:
            runner.run(role="scorer", system="s", user="u", response_schema="x")
        self.assertIn("timed out", str(ctx.exception))
        self.assertEqual(_still_running(token), [])

    def test_the_group_is_ended_before_the_role_is_reaped(self):
        """While the role is unreaped its id cannot pass to another process, so
        the signal to its group reaches that group and no other."""
        states = []
        real = bounded_process._end_group

        def watched(pid: int) -> bool:
            states.append(
                subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
                ).stdout.strip()
            )
            return real(pid)

        token = _token()
        _root, runner = self._call(f"(sleep {token}) {QUIET} &\n{REPLY}", token=token)
        with mock.patch.object(bounded_process, "_end_group", watched):
            runner.run(role="scorer", system="s", user="u", response_schema="x")
        self.assertEqual(len(states), 1)
        self.assertTrue(states[0].startswith("Z"), f"the role was not a zombie: {states!r}")
        self.assertEqual(_still_running(token), [])

    def test_a_call_that_leaves_nothing_signals_its_group_once(self):
        calls = []
        real = os.killpg

        def counted(pid: int, sig: int) -> None:
            calls.append(sig)
            real(pid, sig)

        _root, runner = self._call(REPLY)
        with mock.patch.object(bounded_process.os, "killpg", counted):
            runner.run(role="scorer", system="s", user="u", response_schema="x")
        self.assertEqual(calls, [signal.SIGKILL])

    def test_where_the_exit_cannot_be_waited_for_unreaped_nothing_is_signalled(self):
        """Then the group is left as before: once the role is reaped its id may
        be another process's, and before it exits the signal would end the role."""
        done = self.tmp / "done"
        token = _token()
        script = f'(sleep {token}) {QUIET} &\n{REPLY}\nexec >&- 2>&-\nsleep 0.3\n: > "{done}"'
        _root, runner = self._call(script, token=token)
        with mock.patch.object(bounded_process, "_exited_unreaped", return_value=None):
            self.assertEqual(
                runner.run(role="scorer", system="s", user="u", response_schema="x"), {"x": 1}
            )
        self.assertTrue(done.exists())
        self.assertNotEqual(_running(token), [])

    def test_the_limit_a_process_that_moved_to_a_group_of_its_own_runs_on(self):
        token = _token()
        ready = self.tmp / "ready"
        mover = f"import os,time;os.setsid();open({str(ready)!r},'w').close();time.sleep({token})"
        script = (
            f'"{sys.executable}" -c "{mover}" {QUIET} &\n'
            f'while [ ! -e "{ready}" ]; do sleep 0.05; done\n{REPLY}'
        )
        _root, runner = self._call(script, token=token)
        runner.run(role="scorer", system="s", user="u", response_schema="x")
        time.sleep(0.3)
        self.assertNotEqual(_running(token), [], "a process in a session of its own was reached")


class AnIgnoredSigchldTests(_RoleCase):
    """With SIGCHLD ignored in the run, the kernel reaps the role itself as
    it exits, so its id is free at once and another process may take it.
    ``run_bounded`` starts no role then (the module of the row "No child is
    started while its exit could not be read"); an ignore taken up while a
    role runs is still not taken for an exit not yet reaped. Found by this
    row's first audit: on 3.11 (macOS waits with a kqueue there) the exit was
    taken for one not yet reaped and the freed id was signalled."""

    def setUp(self) -> None:
        super().setUp()
        previous = signal.getsignal(signal.SIGCHLD)
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        self.addCleanup(signal.signal, signal.SIGCHLD, previous)

    def test_an_ignore_taken_up_while_a_role_runs_leaves_its_group_unsignalled(self):
        # The check before the start is made to miss the ignore, as one a signal
        # handler took up after the role started would be missed.
        checks = iter([False])
        real = bounded_process.children_reaped_by_the_kernel
        sent = []
        token = _token()
        _root, runner = self._call(f"(sleep {token}) {QUIET} &\n{REPLY}", token=token)
        with mock.patch.object(
            bounded_process, "children_reaped_by_the_kernel", lambda: next(checks, real())
        ):
            with mock.patch.object(
                bounded_process.os, "killpg", side_effect=lambda pid, sig: sent.append(pid)
            ):
                reply = runner.run(role="scorer", system="s", user="u", response_schema="x")
        self.assertEqual(reply, {"x": 1})
        self.assertEqual(sent, [], "an id the kernel had already freed was signalled")

    def test_with_sigchld_ignored_no_exit_is_taken_for_one_not_yet_reaped(self):
        with mock.patch.object(bounded_process, "_waitid_exited", return_value=True):
            with mock.patch.object(bounded_process, "_kqueue_exited", return_value=True):
                self.assertIsNone(bounded_process._exited_unreaped(123, 1))
                signal.signal(signal.SIGCHLD, signal.SIG_DFL)
                self.assertIs(bounded_process._exited_unreaped(123, 1), True)


def _started_ignored(case: str, root: str, token: str) -> None:
    """Run in a process its parent started with SIGCHLD ignored: ``case`` as
    the process's own command (``cli.entry``) or, for ``library``, as a call of
    ``cli.main``; then one line of JSON on stdout."""
    from agent_skill_improver import cli

    started = signal.getsignal(signal.SIGCHLD) is signal.SIG_IGN

    def runner(script: str) -> CommandModelRunner:
        where = Path(root) / f"state-{next(_SERIAL)}"
        where.mkdir()
        return CommandModelRunner(
            {
                "scorer": RoleCommand(
                    role="scorer", argv=("/bin/sh", "-c", script), timeout_seconds=20
                )
            },
            isolation=NoSandbox(),
            workdir_root=where,
            parent_env={"PATH": "/usr/bin:/bin"},
        )

    def report(**found: object) -> None:
        print(json.dumps({"started_ignored": started, **found}), flush=True)

    if case == "library":
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cli.main(["scheduler-template", "cron"])
        report(rc=rc, ignored_after=signal.getsignal(signal.SIGCHLD) is signal.SIG_IGN)
        return

    def work(argv: object, *, exiting: bool) -> int:
        found: dict[str, object] = {
            "ignored_in_the_command": signal.getsignal(signal.SIGCHLD) is signal.SIG_IGN
        }
        if case == "left":
            checked: dict[int, str] = {}
            real = os.killpg

            def killpg(pid: int, sig: int) -> None:
                # A real signal only to a group whose leader is this process's
                # child, exited and not yet reaped; anything else is refused.
                if pid not in checked:
                    checked[pid] = subprocess.run(
                        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
                    ).stdout.strip()
                if not checked[pid].startswith("Z"):
                    raise ProcessLookupError(pid)
                real(pid, sig)

            answers = 0
            with mock.patch.object(bounded_process.os, "killpg", side_effect=killpg):
                for _ in range(3):
                    reply = runner(f"(sleep {token}) {QUIET} &\n{REPLY}").run(
                        role="scorer", system="s", user="u", response_schema="x"
                    )
                    answers += reply == {"x": 1}
            found["answers"] = answers
            found["groups_ended"] = sum(stat.startswith("Z") for stat in checked.values())
            found["refused"] = sum(not stat.startswith("Z") for stat in checked.values())
        if case == "exit":
            try:
                runner(f"{REPLY}; exit 3").run(
                    role="scorer", system="s", user="u", response_schema="x"
                )
                found["failure"] = ""
            except TransientFailure as exc:
                found["failure"] = str(exc)
        report(**found)
        return 0

    with mock.patch.object(cli, "_main", work):
        cli.entry()


class ARunStartedWithSigchldIgnoredTests(unittest.TestCase):
    """A parent can leave SIGCHLD ignored through ``exec``, and Python does not
    reset it. macOS does not reap the children of a process started so, yet
    ``getsignal`` reports the ignore, so the runner took the role for one the
    kernel had reaped and left what it started running (measured: 5 calls of
    5, both Pythons). Linux does reap them then, and each child's exit reads
    as 0 (its kernel/signal.c; this module's ``exit`` case passed on macOS
    before too). The process's own command puts SIGCHLD back first."""

    def _start(self, case: str, token: str = "") -> dict:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(root)])
        if token:
            self.addCleanup(_end, token)
        here = Path(__file__).resolve()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env["PYTHONPATH"] = os.pathsep.join([str(here.parents[1] / "src"), str(here.parents[1])])
        code = f"import sys\nfrom tests.{here.stem} import _started_ignored\n"
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                code + "_started_ignored(*sys.argv[1:])",
                case,
                str(root),
                token,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            preexec_fn=lambda: signal.signal(signal.SIGCHLD, signal.SIG_IGN),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        found = json.loads(proc.stdout.splitlines()[-1])
        self.assertTrue(found["started_ignored"], "the run was not started with SIGCHLD ignored")
        return found

    def test_a_run_started_with_sigchld_ignored_puts_it_back_to_the_default(self):
        self.assertFalse(self._start("disposition")["ignored_in_the_command"])

    def test_a_run_started_with_sigchld_ignored_leaves_nothing_running_in_the_role_group(self):
        token = _token()
        found = self._start("left", token)
        self.assertEqual(found["answers"], 3)
        self.assertEqual(_still_running(token), [], "what the role started outlived the call")
        self.assertEqual((found["groups_ended"], found["refused"]), (3, 0))

    def test_a_run_started_with_sigchld_ignored_reads_a_role_exit_as_it_is(self):
        self.assertIn("exited 3", self._start("exit")["failure"])

    def test_a_call_of_main_keeps_the_sigchld_its_caller_left(self):
        found = self._start("library")
        self.assertEqual(found["rc"], 0)
        self.assertTrue(found["ignored_after"], "a caller of main had its SIGCHLD changed")


class TheConversationCommandKeepsWhatItStartsTests(unittest.TestCase):
    def test_the_control_a_conversation_command_keeps_what_it_starts(self):
        """The operator's own tool, not an untrusted role: a helper it starts is
        left running as before."""
        token = _token()
        self.addCleanup(_end, token)
        program = (
            "import subprocess, sys\n"
            f"subprocess.Popen(['sleep', '{token}'], stdin=subprocess.DEVNULL,"
            " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "print('{}')\n"
        )
        source = ConversationCliSource(
            export_argv=(sys.executable, "-c", program, "{thread_id}"),
            discovery_argv=(sys.executable, "-c", program, "{since}", "{limit}"),
            timeout_seconds=20,
        )
        try:
            source.fetch("thread-1")
        except AsiError:
            pass
        time.sleep(0.3)
        self.assertNotEqual(_running(token), [])


class WaitingForTheExitWithoutReapingTests(unittest.TestCase):
    def _forms(self):
        forms = []
        if getattr(os, "waitid", None) is not None and getattr(os, "WNOWAIT", None) is not None:
            forms.append(("waitid", bounded_process._waitid_exited))
        if getattr(select, "kqueue", None) is not None:
            forms.append(("kqueue", bounded_process._kqueue_exited))
        if not forms:
            self.skipTest("neither waitid nor kqueue here")
        return forms

    def test_an_exited_child_is_reported_and_left_unreaped(self):
        for name, wait in self._forms():
            with self.subTest(name):
                child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
                self.assertIs(wait(child.pid, 10), True)
                state = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(child.pid)], capture_output=True, text=True
                ).stdout.strip()
                self.assertTrue(state.startswith("Z"), state)
                self.assertIs(wait(child.pid, 10), True)
                self.assertEqual(child.wait(timeout=5), 7, "the exit status was consumed")

    def test_a_running_child_is_waited_for_until_the_timeout(self):
        for name, wait in self._forms():
            with self.subTest(name):
                child = subprocess.Popen(["sleep", "30"])
                try:
                    started = time.monotonic()
                    self.assertIs(wait(child.pid, 0.3), False)
                    self.assertGreaterEqual(time.monotonic() - started, 0.25)
                finally:
                    child.kill()
                    child.wait()

    def test_the_form_follows_what_the_platform_has(self):
        with mock.patch.object(bounded_process.os, "waitid", None, create=True):
            with mock.patch.object(bounded_process.select, "kqueue", None, create=True):
                self.assertIsNone(bounded_process._exited_unreaped(os.getpid(), 0))
        with mock.patch.object(bounded_process.os, "waitid", None, create=True):
            with mock.patch.object(bounded_process, "_kqueue_exited", return_value=True) as kq:
                with mock.patch.object(bounded_process.select, "kqueue", object(), create=True):
                    self.assertIs(bounded_process._exited_unreaped(123, 1), True)
        kq.assert_called_once_with(123, 1)
        with mock.patch.object(bounded_process.os, "waitid", lambda *a: None, create=True):
            with mock.patch.object(bounded_process.os, "WNOWAIT", 0x1000000, create=True):
                with mock.patch.object(bounded_process, "_waitid_exited", return_value=False) as wi:
                    self.assertIs(bounded_process._exited_unreaped(321, 2), False)
        wi.assert_called_once_with(321, 2)

    def test_a_failed_wait_is_not_taken_for_an_exit(self):
        with mock.patch.object(
            bounded_process.os,
            "waitid",
            side_effect=ChildProcessError(errno.ECHILD, "no"),
            create=True,
        ):
            with mock.patch.object(bounded_process.os, "WEXITED", 4, create=True):
                with mock.patch.object(bounded_process.os, "WNOWAIT", 0x1000000, create=True):
                    with mock.patch.object(bounded_process.os, "P_PID", 1, create=True):
                        self.assertIsNone(bounded_process._waitid_exited(123, 1))

    @unittest.skipIf(getattr(select, "kqueue", None) is None, "no kqueue here")
    def test_a_kqueue_error_other_than_no_such_process_is_not_taken_for_an_exit(self):
        class Event:
            def __init__(self, flags, data, fflags=0):
                self.flags, self.data, self.fflags = flags, data, fflags

        class Queue:
            def __init__(self, result):
                self.result = result

            def control(self, changes, count, timeout):
                if isinstance(self.result, BaseException):
                    raise self.result
                return self.result

            def close(self):
                pass

        for result, expected in (
            ([Event(select.KQ_EV_ERROR, errno.ESRCH)], True),
            ([Event(select.KQ_EV_ERROR, errno.EINVAL)], None),
            (OSError(errno.ESRCH, "gone"), True),
            (OSError(errno.EBADF, "bad"), None),
            ([Event(0, 0, select.KQ_NOTE_EXIT)], True),
            ([], False),
        ):
            with self.subTest(result=repr(result)):
                with mock.patch.object(bounded_process.select, "kqueue", lambda: Queue(result)):
                    self.assertIs(bounded_process._kqueue_exited(123, 1), expected)


class EndingAGroupTests(unittest.TestCase):
    def test_the_group_is_signalled_again_until_nothing_in_it_runs(self):
        effects = [None, None, None, PermissionError(errno.EPERM, "only the dead")]
        with mock.patch.object(bounded_process, "ONE_ROUND_ENDS_A_GROUP", False):
            with mock.patch.object(bounded_process.os, "killpg", side_effect=effects) as killpg:
                self.assertIs(bounded_process._end_group(4242), True)
        self.assertEqual(killpg.call_count, 4)
        killpg.assert_called_with(4242, signal.SIGKILL)

    def test_on_linux_one_round_is_all(self):
        """Linux reports a signal to an unreaped member as sent, so rounds there
        would run out their bound on every call."""
        with mock.patch.object(bounded_process, "ONE_ROUND_ENDS_A_GROUP", True):
            with mock.patch.object(bounded_process.os, "killpg", return_value=None) as killpg:
                started = time.monotonic()
                self.assertIs(bounded_process._end_group(4242), True)
        self.assertEqual(killpg.call_count, 1)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(bounded_process.ONE_ROUND_ENDS_A_GROUP, sys.platform.startswith("linux"))

    def test_the_rounds_are_bounded(self):
        with mock.patch.object(bounded_process, "ONE_ROUND_ENDS_A_GROUP", False):
            with mock.patch.object(bounded_process, "GROUP_END_SECONDS", 0.2):
                with mock.patch.object(bounded_process.os, "killpg", return_value=None) as killpg:
                    started = time.monotonic()
                    self.assertIs(bounded_process._end_group(4242), True)
        took = time.monotonic() - started
        self.assertGreaterEqual(took, 0.2)
        self.assertLess(took, 1.0)
        self.assertGreater(killpg.call_count, 2)
        self.assertEqual(bounded_process.GROUP_END_SECONDS, 1.0)

    def test_a_group_that_cannot_be_signalled_falls_back_to_the_child(self):
        with mock.patch.object(bounded_process.os, "killpg", side_effect=ProcessLookupError()):
            self.assertIs(bounded_process._end_group(4242), False)
        # No real signal leaves this test, whatever ``_kill`` does.
        refused = ProcessLookupError()
        for ended, killed in ((False, True), (True, False)):
            with self.subTest(ended=ended):
                proc = mock.Mock(pid=4242)
                with mock.patch.object(bounded_process.os, "killpg", side_effect=refused):
                    with mock.patch.object(bounded_process, "_end_group", return_value=ended):
                        bounded_process._kill(proc)
                self.assertEqual(proc.kill.called, killed)
                proc.wait.assert_called_once_with(timeout=5)


if __name__ == "__main__":
    unittest.main()
