"""A command that cannot be started is named once, a key in its path masked.

``str(exc)`` of a failed start ends with the program it was given. The
conversation source quoted it raw after naming the program itself, and the
model runner quoted it raw as its only naming of the role command. Measured
before (3.11 and 3.13, the same on both): the source named the program twice,
1,788 characters for a path of 859 (950 for a role); a key-shaped directory in
the path was printed raw, twice by the source and once by a role; and a role
command that passed the preflight and failed to start (a wrong binary format)
under such a directory put the key raw in 5 lines of a daily run's output and
5 times under each of ``receipts``, ``deferred`` and ``checkpoints``, with
``redact_secrets`` on. Each is now named once, masked and cut
(``why_not_started``, ``not_started_reason`` in ``adapters/bounded_process.py``).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_skill_improver import cli
from agent_skill_improver.adapters.conversation_cli import ConversationCliSource
from agent_skill_improver.adapters.model_command import CommandModelRunner, RoleCommand
from agent_skill_improver.adapters.sandbox import NoSandbox
from agent_skill_improver.domain.redaction import NAMED_PATH_CHARS
from agent_skill_improver.errors import SourceUnavailable, TransientFailure

from tests.support import build_env

#: Shaped like a provider key, so redaction masks it; split so no scanner
#: reads this file as holding one.
KEY = "sk-ant-" + "api03-" + "A1b2C3d4E5f6G7h8" * 4
#: Not a program and no ``#!``: ``execve`` refuses it with ENOEXEC, after the
#: preflight (which checks only that the file exists and may be executed).
NOT_A_PROGRAM = b"\x00\x01 not a program\n"
SIGCHLD_TEXT = (
    "SIGCHLD is ignored in this process, so the child's exit status could not be "
    "read; nothing was started (asi puts SIGCHLD back to its default as it starts)"
)


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.tmp)])

    def program(self, *parts: str, body: bytes = NOT_A_PROGRAM, mode: int = 0o755) -> str:
        path = self.tmp.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        os.chmod(path, mode)
        return str(path)

    def source_refusal(self, program: str) -> str:
        source = ConversationCliSource(
            discovery_argv=(program, "{since}", "{limit}"), export_argv=(program, "{thread_id}")
        )
        with self.assertRaises(SourceUnavailable) as ctx:
            source.discover_page(since="1d", limit=5)
        return str(ctx.exception)

    def role_refusal(self, *argv: str) -> str:
        (self.tmp / "work").mkdir(exist_ok=True)
        runner = CommandModelRunner(
            {"scorer": RoleCommand(role="scorer", argv=tuple(argv), timeout_seconds=20)},
            isolation=NoSandbox(),
            workdir_root=self.tmp / "work",
            parent_env={"PATH": "/usr/bin:/bin"},
        )
        with self.assertRaises(TransientFailure) as ctx:
            runner.run(role="scorer", system="s", user="u", response_schema="x")
        return str(ctx.exception)

    def long_program(self) -> str:
        return self.program(*["d" * 100 + str(i) for i in range(8)], "p")

    def assert_cut_at_the_limit(self, message: str, program: str) -> None:
        """The program is quoted whole up to ``NAMED_PATH_CHARS``, and cut there once."""
        quoted = repr(program)
        self.assertGreater(len(quoted), NAMED_PATH_CHARS)
        self.assertNotIn(program, message)
        kept = quoted[:NAMED_PATH_CHARS]
        cut = f"{kept}... [{len(quoted) - NAMED_PATH_CHARS} more chars not quoted]"
        self.assertEqual(message.count(cut), 1, message)
        self.assertEqual(message.count("more chars not quoted]"), 1, message)


class ASourceCommandIsNamedOnceTests(_Case):
    def test_a_source_command_that_cannot_be_started_is_named_once(self):
        program = self.program("not-a-program")
        self.assertEqual(
            self.source_refusal(program), f"cannot execute {program!r}: [Errno 8] Exec format error"
        )

    def test_a_key_in_a_source_command_path_is_masked(self):
        program = self.program(KEY, "export")
        message = self.source_refusal(program)
        self.assertNotIn(KEY, message)
        self.assertEqual(message.count("[REDACTED:"), 1, message)
        self.assertTrue(message.endswith("/export': [Errno 8] Exec format error"), message)

    def test_a_long_source_command_path_is_cut(self):
        program = self.long_program()
        message = self.source_refusal(program)
        self.assert_cut_at_the_limit(message, program)
        self.assertTrue(message.startswith("cannot execute '"), message)
        self.assertTrue(message.endswith(": [Errno 8] Exec format error"), message)


class ARoleCommandIsMaskedTests(_Case):
    def test_the_control_a_plain_role_command_path_reads_as_before(self):
        program = self.program("not-a-program")
        self.assertEqual(
            self.role_refusal(program),
            f"role 'scorer' command cannot be started: [Errno 8] Exec format error: {program!r}",
        )

    def test_a_key_in_a_role_command_path_is_masked(self):
        program = self.program(KEY, "scorer")
        message = self.role_refusal(program)
        self.assertNotIn(KEY, message)
        self.assertEqual(message.count("[REDACTED:"), 1, message)
        self.assertTrue(message.endswith("/scorer'"), message)

    def test_a_long_role_command_path_is_cut(self):
        program = self.long_program()
        message = self.role_refusal(program)
        self.assert_cut_at_the_limit(message, program)
        self.assertTrue(
            message.startswith(
                "role 'scorer' command cannot be started: [Errno 8] Exec format error: '"
            ),
            message,
        )


@unittest.skipIf(getattr(signal, "SIGCHLD", None) is None, "this platform has no SIGCHLD")
class TheSigchldRefusalReadsAsBeforeTests(_Case):
    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(signal.signal, signal.SIGCHLD, signal.getsignal(signal.SIGCHLD))
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    def test_with_sigchld_ignored_each_refusal_is_its_own_text_whole(self):
        self.assertEqual(
            self.source_refusal("/bin/sh"), f"cannot execute '/bin/sh': {SIGCHLD_TEXT}"
        )
        self.assertEqual(
            self.role_refusal("/bin/sh", "-c", "true"),
            f"role 'scorer' command cannot be started: {SIGCHLD_TEXT}",
        )


class ADailyRunKeepsNoKeyFromARoleCommandPathTests(_Case):
    def test_a_daily_run_keeps_no_key_from_a_role_command_it_cannot_start(self):
        program = self.program(KEY, "scorer")
        base = self.tmp / "env"
        base.mkdir()
        config = build_env(base, seed="c210")
        text = config.read_text(encoding="utf-8")
        start = text.index("[model.scorer]\nargv = ")
        end = text.index("\n", start + len("[model.scorer]\n"))
        config.write_text(
            text[:start] + "[model.scorer]\nargv = " + json.dumps([program]) + text[end:],
            encoding="utf-8",
        )
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = cli.main(["--config", str(config), "daily", "--commit", "--day", "2026-01-02"])
        shown = out.getvalue()
        self.assertEqual(rc, 0, shown)
        self.assertIn(" scored=0 ", shown)
        self.assertIn(" deferred=5 ", shown)
        refusals = [line for line in shown.splitlines() if "cannot be started" in line]
        self.assertEqual(len(refusals), 5, shown)
        self.assertTrue(all("[Errno 8] Exec format error" in line for line in refusals), shown)
        self.assertNotIn(KEY, shown)
        state = base / "state"
        written = [p for p in state.rglob("*") if p.is_file()]
        self.assertTrue(any(p.relative_to(state).parts[0] == "deferred" for p in written))
        carrying = [p for p in written if KEY.encode() in p.read_bytes()]
        self.assertEqual(carrying, [])


if __name__ == "__main__":
    unittest.main()
