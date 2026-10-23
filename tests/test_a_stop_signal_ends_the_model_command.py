"""A stop signal ends the model command with the run, never leaves it running.

A model or source command runs in a session of its own (``start_new_session``),
so the signal a terminal or a scheduler sends to the run does not reach it, and
``run_bounded`` ends it in a ``finally``. Ctrl-C raised ``KeyboardInterrupt``
and so reached that ``finally``; SIGTERM, which a scheduler sends to stop a job,
and SIGHUP, which a closed terminal sends, killed the run where it stood.
Measured before (3.11 and 3.13): either signal during a model call ended the run
with no message (exit -15, -1), and the model command went on running,
re-parented to process 1 and past its own timeout, until it wrote its answer
to an output nobody read.
"""

from __future__ import annotations

import contextlib
import io
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters import bounded_process

from tests.support import FAKE_MODEL, REPO_ROOT, build_env

DAY = "2026-01-02"

#: The first model call writes its pid, then waits this long before it answers;
#: every later call answers at once.
WRAPPER = """import os, runpy, sys, time
from pathlib import Path
here = Path({here!r})
first = not (here / "model.pid").exists()
if first:
    (here / "model.pid.tmp").write_text(str(os.getpid()))
    os.replace(here / "model.pid.tmp", here / "model.pid")
    time.sleep({wait})
sys.argv = [{fake!r}] + sys.argv[1:]
runpy.run_path({fake!r}, run_name="__main__")
"""


#: A doctor that turns any error into its own answer. No command does this
#: today; one that did must still not turn a stop into an answer.
STAND_IN = """import os, signal, sys, time
from agent_skill_improver import cli
def swallows_errors(args):
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(30)
    except Exception:
        print("swallowed", flush=True)
    return 0
cli.cmd_doctor = swallows_errors
sys.exit(cli.main(sys.argv[1:]))
"""

#: A doctor that stops itself, run once through ``main`` and once as the
#: process's own command, printing the handlers each leaves behind.
STOPS_ITSELF = """import os, signal, sys, time
from agent_skill_improver import cli
def stops(args):
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(30)
    return 0
cli.cmd_doctor = stops
def left():
    return [str(signal.getsignal(s)) for s in (signal.SIGTERM, signal.SIGHUP)] + [
        str(signal.getsignal(signal.SIGINT) is signal.default_int_handler)]
print("main", cli.main(sys.argv[1:]), *left(), flush=True)
print("exiting", cli._main(sys.argv[1:], exiting=True), *left(), flush=True)
"""

#: A doctor that is refused, and a stop that arrives while the refusal is
#: being written out.
STOPS_WHILE_REPORTING = """import os, signal, sys, time
from agent_skill_improver import cli
from agent_skill_improver.errors import AsiError
def refuses(args):
    raise AsiError("refused for the test")
real_err = cli._err
def err(text):
    if "refused for the test" in text:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(30)
    real_err(text)
cli.cmd_doctor = refuses
cli._err = err
cli.entry()
"""


def _open_then_stop(real_open):  # type: ignore[no-untyped-def]
    """An ``os.open`` that creates a temporary file and is then cut short, as a
    stop landing just as the call returns would: the file is there and the
    descriptor is lost."""

    def opened(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        fd = real_open(path, flags, *args, **kwargs)
        if os.path.basename(os.fsdecode(path)).startswith(".tmp-"):
            os.close(fd)
            raise KeyboardInterrupt
        return fd

    return opened


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class AStopSignalEndsTheModelCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base, seed="stop-signals"))
        self.pid_file = self.base / "model.pid"
        self.cfg = ["--config", str(self.config_path)]
        self.model_pids: list[int] = []
        self.addCleanup(self._kill_models)

    def _slow_first_call(self, wait: float) -> None:
        wrapper = self.base / "slow_model.py"
        wrapper.write_text(
            WRAPPER.format(here=str(self.base), wait=wait, fake=str(FAKE_MODEL)), encoding="utf-8"
        )
        text = self.config_path.read_text(encoding="utf-8")
        self.assertIn(str(FAKE_MODEL), text)
        self.config_path.write_text(text.replace(str(FAKE_MODEL), str(wrapper)), encoding="utf-8")

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + str(REPO_ROOT)
        env.pop("ASI_CONFIG", None)
        return env

    def _start(self, *args: str, ignore_hup: bool = False) -> subprocess.Popen[str]:
        env = self._env()

        def ignore() -> None:  # what nohup does before it starts the command
            signal.signal(signal.SIGHUP, signal.SIG_IGN)

        proc = subprocess.Popen(
            [sys.executable, "-m", "agent_skill_improver", *self.cfg, *args],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=ignore if ignore_hup else None,
        )
        self.addCleanup(self._reap, proc)
        return proc

    def _reap(self, proc: subprocess.Popen[str]) -> None:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    def _model_pid(self) -> int:
        end = time.monotonic() + 60
        while not self.pid_file.exists():
            self.assertLess(time.monotonic(), end, "the first model call never started")
            time.sleep(0.05)
        pid = int(self.pid_file.read_text())
        self.model_pids.append(pid)
        return pid

    def _kill_models(self) -> None:
        # Against the code before the change the model outlives its run; a
        # failing test must not leave it behind too.
        for pid in self.model_pids:
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)

    def _assert_gone(self, pid: int) -> None:
        end = time.monotonic() + 10
        while _alive(pid):
            self.assertLess(time.monotonic(), end, f"model command {pid} outlived the run")
            time.sleep(0.05)

    def test_sigterm_or_sighup_mid_model_call_ends_the_model_command_with_the_run(self):
        self._slow_first_call(wait=120)
        for signum in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=signum.name):
                self.pid_file.unlink(missing_ok=True)
                proc = self._start("daily", "--commit", "--day", DAY)
                pid = self._model_pid()
                proc.send_signal(signum)
                _, err = proc.communicate(timeout=60)
                self.assertEqual(proc.returncode, 128 + signum, err)
                named = {
                    signal.SIGTERM: cli.RC_STOPPED_BY_SIGTERM,
                    signal.SIGHUP: cli.RC_STOPPED_BY_SIGHUP,
                }
                self.assertEqual(proc.returncode, named[signum])
                self.assertIn(f"stopped by {signum.name}", err)
                self.assertNotIn("Traceback", err)
                self._assert_gone(pid)

    def test_after_a_stop_the_next_daily_runs_and_no_partial_file_is_left(self):
        self._slow_first_call(wait=120)
        proc = self._start("daily", "--commit", "--day", DAY)
        pid = self._model_pid()
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=60)
        self._assert_gone(pid)
        self.assertEqual(sorted(p.name for p in self.base.rglob("*.part")), [])
        again = self._start("daily", "--commit", "--day", DAY)
        out, err = again.communicate(timeout=300)
        self.assertEqual(again.returncode, 0, err)
        self.assertIn("receipt:", out + err)

    def test_the_control_a_hangup_nohup_ignores_still_does_not_stop_the_run(self):
        self._slow_first_call(wait=2)
        proc = self._start("daily", "--commit", "--day", DAY, ignore_hup=True)
        self._model_pid()
        proc.send_signal(signal.SIGHUP)
        out, err = proc.communicate(timeout=300)
        self.assertEqual(proc.returncode, 0, err)
        self.assertNotIn("stopped by", err)
        self.assertIn("receipt:", out + err)

    def test_the_control_ctrl_c_still_ends_the_model_command_with_the_run(self):
        self._slow_first_call(wait=120)
        proc = self._start("daily", "--commit", "--day", DAY)
        pid = self._model_pid()
        proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=60)
        self.assertNotEqual(proc.returncode, 0)
        self._assert_gone(pid)

    @unittest.skipUnless(
        signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, "the runner owns SIGTERM"
    )
    def test_a_second_signal_while_the_first_unwinds_does_not_cut_the_cleanup_short(self):
        cleaned: list[bool] = []
        with self.assertRaises(cli.Stopped) as ctx:
            with cli._stop_signals_unwind():
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(5)
                finally:
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(0.2)
                    cleaned.append(True)
        self.assertEqual(ctx.exception.signum, signal.SIGTERM)
        self.assertEqual(cleaned, [True])
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)

    def test_a_command_that_catches_every_error_still_does_not_catch_a_stop(self):
        proc = subprocess.run(
            [sys.executable, "-c", STAND_IN, *self.cfg, "doctor"],
            cwd=str(REPO_ROOT),
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 128 + signal.SIGTERM, proc.stderr)
        self.assertNotIn("swallowed", proc.stdout)
        self.assertIn("stopped by SIGTERM before the command finished", proc.stderr)

    def test_main_restores_the_handlers_it_found_and_runs_off_the_main_thread(self):
        before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main([*self.cfg, "doctor"]), 0)
        self.assertEqual({s: signal.getsignal(s) for s in before}, before)
        result: list[object] = []
        worker = threading.Thread(target=lambda: result.append(cli.main([*self.cfg, "doctor"])))
        with contextlib.redirect_stdout(io.StringIO()):
            worker.start()
            worker.join(timeout=120)
        self.assertEqual(result, [0])

    def test_a_burst_of_stop_signals_still_exits_as_the_stop(self):
        # Measured before: the handlers were put back as the command ended,
        # and this burst printed the stop, then ended the run by the signal.
        self._slow_first_call(wait=120)
        proc = self._start("daily", "--commit", "--day", DAY)
        pid = self._model_pid()
        for _ in range(40):
            proc.send_signal(signal.SIGTERM)
            time.sleep(0.002)
        _, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, cli.RC_STOPPED_BY_SIGTERM, err)
        self.assertEqual(err.count("stopped by SIGTERM"), 1, err)
        self.assertIn("any model or source command it had started was ended with it", err)
        self._assert_gone(pid)

    def test_the_process_entry_leaves_the_stops_ignored_and_main_puts_them_back(self):
        proc = subprocess.run(
            [sys.executable, "-c", STOPS_ITSELF, *self.cfg, "doctor"],
            cwd=str(REPO_ROOT),
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertEqual(
            lines,
            [
                f"main 143 {signal.SIG_DFL} {signal.SIG_DFL} True",
                f"exiting 143 {signal.SIG_IGN} {signal.SIG_IGN} True",
            ],
            proc.stderr,
        )

    @unittest.skipUnless(
        signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
        and signal.getsignal(signal.SIGINT) is signal.default_int_handler,
        "the runner owns SIGTERM or SIGINT",
    )
    def test_a_stop_signal_while_ctrl_c_unwinds_is_let_pass(self):
        cleaned: list[bool] = []
        with self.assertRaises(KeyboardInterrupt):
            with cli._stop_signals_unwind():
                try:
                    os.kill(os.getpid(), signal.SIGINT)
                    time.sleep(5)
                finally:
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(0.2)
                    cleaned.append(True)
        self.assertEqual(cleaned, [True])
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)
        self.assertIs(signal.getsignal(signal.SIGINT), signal.default_int_handler)

    @unittest.skipUnless(
        signal.getsignal(signal.SIGINT) is signal.default_int_handler, "the runner owns SIGINT"
    )
    def test_the_control_each_ctrl_c_still_raises_as_pythons_own_does(self):
        presses: list[str] = []
        with self.assertRaises(KeyboardInterrupt):
            with cli._stop_signals_unwind():
                try:
                    os.kill(os.getpid(), signal.SIGINT)
                    time.sleep(5)
                finally:
                    presses.append("first")
                    os.kill(os.getpid(), signal.SIGINT)
                    time.sleep(5)
                    presses.append("not reached")
        self.assertEqual(presses, ["first"])

    def test_a_stop_before_the_read_loop_still_ends_the_child(self):
        real_popen = subprocess.Popen
        started: list[subprocess.Popen[bytes]] = []

        def spy(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
            child = real_popen(*args, **kwargs)  # type: ignore[call-overload]
            started.append(child)
            return child

        with (
            mock.patch.object(bounded_process.subprocess, "Popen", spy),
            mock.patch.object(threading.Thread, "start", side_effect=KeyboardInterrupt),
        ):
            with self.assertRaises(KeyboardInterrupt):
                bounded_process.run_bounded(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    timeout_seconds=60,
                    max_output_bytes=1000,
                )
        self.assertEqual(len(started), 1)
        child = started[0]
        self.addCleanup(lambda: child.poll() is None and child.kill())
        self.assertIsNotNone(child.poll(), "the child outlived the call that started it")
        for stream in (child.stdin, child.stdout, child.stderr):
            if stream is not None:
                stream.close()

    def test_a_stop_while_a_failure_is_reported_is_a_stop_too(self):
        proc = subprocess.run(
            [sys.executable, "-c", STOPS_WHILE_REPORTING, *self.cfg, "doctor"],
            cwd=str(REPO_ROOT),
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, cli.RC_STOPPED_BY_SIGTERM, proc.stderr)
        self.assertIn("stopped by SIGTERM before the command finished", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_a_stop_as_a_temporary_file_is_created_leaves_no_temporary_file(self):
        from agent_skill_improver import jsonio
        from agent_skill_improver.adapters import skill_fs

        records = self.base / "records"
        records.mkdir()
        with mock.patch.object(os, "open", _open_then_stop(os.open)):
            with self.assertRaises(KeyboardInterrupt):
                jsonio.write_text_atomic(records / "record.json", "{}")
        self.assertEqual(sorted(p.name for p in records.iterdir()), [])

        pinned = self.base / "pinned"
        pinned.mkdir()
        handle = os.open(pinned, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, handle)
        with mock.patch.object(os, "open", _open_then_stop(os.open)):
            with self.assertRaises(KeyboardInterrupt):
                skill_fs._write_pinned(handle, "SKILL.md", b"text", mode=0o644)
        self.assertEqual(sorted(p.name for p in pinned.iterdir()), [])

    def test_the_control_a_temporary_name_already_taken_is_left_alone(self):
        from agent_skill_improver import jsonio
        from agent_skill_improver.adapters import skill_fs

        records = self.base / "records"
        records.mkdir()
        other = records / ".tmp-000000000000.part"
        other.write_bytes(b"another writer's")
        names = iter(["000000000000", "111111111111"])
        with mock.patch.object(jsonio.secrets, "token_hex", side_effect=lambda n: next(names)):
            jsonio.write_text_atomic(records / "record.json", "{}")
        self.assertEqual(other.read_bytes(), b"another writer's")
        self.assertEqual((records / "record.json").read_text(), "{}")
        self.assertEqual(sorted(p.name for p in records.iterdir()), [other.name, "record.json"])

        pinned = self.base / "pinned"
        pinned.mkdir()
        handle = os.open(pinned, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, handle)
        taken = pinned / f".tmp-SKILL.md-{os.getpid()}-00000000.part"
        taken.write_bytes(b"another writer's")
        with mock.patch.object(skill_fs.secrets, "token_hex", return_value="00000000"):
            with self.assertRaises(FileExistsError):
                skill_fs._write_pinned(handle, "SKILL.md", b"text", mode=0o644)
        self.assertEqual(taken.read_bytes(), b"another writer's")

    def test_a_temporary_file_is_owner_only_from_the_moment_it_is_created(self):
        # Private content is written into it before its final mode is set.
        from agent_skill_improver import jsonio
        from agent_skill_improver.adapters import skill_fs

        real_open = os.open
        modes: list[int] = []

        def recording(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
            fd = real_open(path, flags, *args, **kwargs)
            if os.path.basename(os.fsdecode(path)).startswith(".tmp-"):
                modes.append(os.fstat(fd).st_mode & 0o777)
            return fd

        records = self.base / "records"
        records.mkdir()
        pinned = self.base / "pinned"
        pinned.mkdir()
        handle = os.open(pinned, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, handle)
        previous = os.umask(0)
        try:
            with mock.patch.object(os, "open", recording):
                jsonio.write_text_atomic(records / "record.json", "{}", mode=0o644)
                skill_fs._write_pinned(handle, "SKILL.md", b"text", mode=0o644)
        finally:
            os.umask(previous)
        self.assertEqual(modes, [0o600, 0o600])
        self.assertEqual((records / "record.json").stat().st_mode & 0o777, 0o644)
        self.assertEqual((pinned / "SKILL.md").stat().st_mode & 0o777, 0o644)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
