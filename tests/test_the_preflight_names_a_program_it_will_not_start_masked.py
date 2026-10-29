"""The preflight names a program it can see will not start masked and cut.

A role or conversation command whose program is missing, a directory or not
executable is refused before anything runs: ``doctor`` lists it as a problem
and ``daily`` refuses with exit 3. Both refusals quoted the program with
``repr``. Measured before (3.11 and 3.13, the same on both), for a program at
mode 644: a key-shaped directory in its path was printed raw, once by
``daily`` and once (a role) or twice (the source: its ``source`` line and its
``PROBLEM``) by ``doctor``; and a path of 872 or 874 characters made lines of
1,004 to 1,076. They are now quoted as the start refusal quotes them
(``quoted_value`` at ``NAMED_PATH_CHARS``), so a plain path reads as before.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent_skill_improver import cli
from agent_skill_improver.adapters.conversation_cli import ConversationCliSource
from agent_skill_improver.adapters.model_command import CommandModelRunner, RoleCommand
from agent_skill_improver.adapters.sandbox import NoSandbox
from agent_skill_improver.domain.redaction import NAMED_PATH_CHARS
from agent_skill_improver.errors import SourceUnavailable

from tests.support import build_env

#: Shaped like a provider key, so redaction masks it; split so no scanner
#: reads this file as holding one.
KEY = "sk-ant-" + "api03-" + "A1b2C3d4E5f6G7h8" * 4
NOT_EXECUTABLE = "exists but is not executable (mode 644); chmod +x it"
SET_ARGV = "; set [source].discovery_argv and [source].export_argv in your config"


@unittest.skipIf(os.geteuid() == 0, "root may execute a mode-644 file")
class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.tmp)])

    def program(self, *parts: str) -> str:
        path = self.tmp.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\necho []\n", encoding="utf-8")
        os.chmod(path, 0o644)
        return str(path)

    def long_program(self) -> str:
        return self.program(*["d" * 100 + str(i) for i in range(8)], "prog")

    def role_problems(self, program: str) -> list[str]:
        runner = CommandModelRunner(
            {
                "scorer": RoleCommand(role="scorer", argv=(program,)),
                "judge": RoleCommand(role="judge", argv=(sys.executable,)),
            },
            isolation=NoSandbox(),
            parent_env=dict(os.environ),
        )
        return runner.program_problems()

    def source_refusal(self, program: str) -> str:
        source = ConversationCliSource(
            discovery_argv=(program, "{since}", "{limit}"), export_argv=(program, "{thread_id}")
        )
        with self.assertRaises(SourceUnavailable) as ctx:
            source.preflight()
        return str(ctx.exception)

    def assert_cut_at_the_limit(self, message: str, program: str) -> None:
        """The program is quoted whole up to ``NAMED_PATH_CHARS``, and cut there once."""
        quoted = repr(program)
        self.assertGreater(len(quoted), NAMED_PATH_CHARS)
        self.assertNotIn(program, message)
        cut = f"{quoted[:NAMED_PATH_CHARS]}... [{len(quoted) - NAMED_PATH_CHARS} more chars not quoted]"
        self.assertEqual(message.count(cut), 1, message)
        self.assertEqual(message.count("more chars not quoted]"), 1, message)


class ARoleProgramTests(_Case):
    def test_the_control_a_plain_role_program_reads_as_before(self):
        program = self.program("prog")
        self.assertEqual(
            self.role_problems(program), [f"role 'scorer' command {program!r} {NOT_EXECUTABLE}"]
        )

    def test_a_key_in_a_role_program_path_is_masked(self):
        problems = self.role_problems(self.program(KEY, "prog"))
        self.assertEqual(len(problems), 1, problems)
        self.assertNotIn(KEY, problems[0])
        self.assertEqual(problems[0].count("[REDACTED:"), 1, problems)
        self.assertTrue(problems[0].startswith("role 'scorer' command '"), problems)
        self.assertTrue(problems[0].endswith(f"/prog' {NOT_EXECUTABLE}"), problems)

    def test_a_long_role_program_path_is_cut_at_the_limit(self):
        program = self.long_program()
        problems = self.role_problems(program)
        self.assertEqual(len(problems), 1, problems)
        self.assert_cut_at_the_limit(problems[0], program)
        self.assertTrue(problems[0].endswith(f"more chars not quoted] {NOT_EXECUTABLE}"), problems)


class ASourceProgramTests(_Case):
    def test_the_control_a_plain_source_program_reads_as_before(self):
        program = self.program("prog")
        self.assertEqual(
            self.source_refusal(program),
            f"conversation command {program!r} {NOT_EXECUTABLE}{SET_ARGV}",
        )

    def test_a_key_in_a_source_program_path_is_masked(self):
        message = self.source_refusal(self.program(KEY, "prog"))
        self.assertNotIn(KEY, message)
        self.assertEqual(message.count("[REDACTED:"), 1, message)
        self.assertTrue(message.startswith("conversation command '"), message)
        self.assertTrue(message.endswith(f"/prog' {NOT_EXECUTABLE}{SET_ARGV}"), message)

    def test_a_long_source_program_path_is_cut_at_the_limit(self):
        program = self.long_program()
        message = self.source_refusal(program)
        self.assert_cut_at_the_limit(message, program)
        self.assertTrue(message.endswith(f"more chars not quoted] {NOT_EXECUTABLE}{SET_ARGV}"))


class DoctorAndDailyTests(_Case):
    def config_with(self, kind: str, program: str) -> Path:
        base = self.tmp / f"env-{kind}"
        base.mkdir()
        config = build_env(base, seed="c211")
        text = config.read_text(encoding="utf-8")
        if kind == "role":
            start = text.index("[model.scorer]\nargv = ")
            end = text.index("\n", start + len("[model.scorer]\n"))
            text = text[:start] + "[model.scorer]\nargv = " + json.dumps([program]) + text[end:]
        else:
            start = text.index('kind = "fixtures"')
            end = text.index("\n", text.index("fixtures_dir", start))
            text = (
                text[:start]
                + 'kind = "conversation-cli"\ndiscovery_argv = '
                + json.dumps([program, "{since}", "{limit}"])
                + "\nexport_argv = "
                + json.dumps([program, "{thread_id}"])
                + text[end:]
            )
        config.write_text(text, encoding="utf-8")
        return config

    def run_cli(self, config: Path, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = cli.main(["--config", str(config), *argv])
        return rc, out.getvalue()

    def test_doctor_and_daily_print_no_key_from_a_program_they_will_not_start(self):
        for kind in ("role", "source"):
            with self.subTest(kind=kind):
                config = self.config_with(kind, self.program(kind, KEY, "prog"))
                rc, shown = self.run_cli(config, "doctor")
                self.assertEqual(rc, 1, shown)
                self.assertTrue(any(NOT_EXECUTABLE in line for line in shown.splitlines()), shown)
                self.assertNotIn(KEY, shown)
                rc, shown = self.run_cli(config, "daily", "--commit", "--day", "2026-01-02")
                self.assertEqual(rc, 3, shown)
                self.assertIn(NOT_EXECUTABLE, shown)
                self.assertNotIn(KEY, shown)


if __name__ == "__main__":
    unittest.main()
