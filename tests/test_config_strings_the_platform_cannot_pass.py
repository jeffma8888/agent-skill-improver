"""A NUL in a config string, or an env name no environment holds, is refused at load.

Every string a config holds is a path, a word of a command line, an
environment entry, a value rendered into one of those, or a label printed
beside them, and the platform takes each as a NUL-terminated string. TOML can
spell a NUL (``"\\u0000"``), and Python refuses one only at the call that
uses it, with a ``ValueError`` (``embedded null byte``; for a path on 3.13,
``lstat: embedded null character in path``).

Measured before (3.11 and 3.13): a NUL in ``[run].state_dir``,
``[run].base_dir``, ``[rubrics].dir`` or a ``[[skill_root]]`` path made
``doctor`` and ``daily`` fall at that call, exit 70 with a traceback; a NUL
in a word of a ``[model.*]`` or ``[source]`` command, or in a
``[model.*].env`` name or value, and an env name holding ``=``, passed
``doctor`` and made ``daily`` fall at its first model or source call, exit 70
with a traceback.

Not covered here: a path longer than the platform takes.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.config import load_config
from agent_skill_improver.errors import ConfigError

from tests.support import build_env, run_cli
from tests.test_non_finite_numbers import _set_field

DAY = "2026-01-02"


class AConfigStringThePlatformCannotPassTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base, seed="platform-strings"))

    def refusal(self) -> str:
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        return str(ctx.exception)

    def test_a_nul_in_any_config_string_is_a_config_error_naming_the_field(self):
        cases = [
            ("run", "state_dir", '"st\\u0000ate"', "run.state_dir"),
            ("run", "base_dir", '"ba\\u0000se"', "run.base_dir"),
            ("rubrics", "dir", '"rub\\u0000rics"', "rubrics.dir"),
            ("source", "since", '"1d\\u0000"', "source.since"),
            ("source", "query", '"q\\u0000"', "source.query"),
            ("source", "workspaces", '["main", "w\\u0000"]', "source.workspaces[1]"),
            (
                "source",
                "discovery_argv",
                '["conv", "list", "{since}", "{limit}", "a\\u0000b"]',
                "source.discovery_argv[4]",
            ),
            ("model.scorer", "command", '"fake --role a\\u0000b"', "model.scorer.command"),
            ("model.scorer", "env", '{ ASI_PROBE = "zq\\u0000zq" }', "model.scorer.env.ASI_PROBE"),
        ]
        original = self.config_path.read_text(encoding="utf-8")
        for table, key, value, label in cases:
            with self.subTest(field=label):
                self.config_path.write_text(original, encoding="utf-8")
                _set_field(self.config_path, table, key, value)
                message = self.refusal()
                self.assertIn(f"{label} holds a NUL character", message)
                self.assertIn(str(self.config_path), message)
                # The config's own path is a random temp name, which may hold "zq".
                shown = message.replace(str(self.config_path), "")
                self.assertNotIn("zq", shown, "the value is never shown")

    def test_a_nul_in_a_skill_root_path_or_an_argv_word_is_named_by_position(self):
        original = self.config_path.read_text(encoding="utf-8")
        with_root, count = re.subn(
            r'(\[\[skill_root\]\][^\[]*?path = "[^"]*)(")', r"\1\\u0000x\2", original, count=1
        )
        self.assertEqual(count, 1)
        self.config_path.write_text(with_root, encoding="utf-8")
        self.assertIn("skill_root[0].path holds a NUL character", self.refusal())
        with_word, count = re.subn(
            r"(\[model\.scorer\]\nargv = \[)(.*?)\]", r'\1\2, "x\\u0000y"]', original, count=1
        )
        self.assertEqual(count, 1)
        self.config_path.write_text(with_word, encoding="utf-8")
        self.assertIn("model.scorer.argv[4] holds a NUL character", self.refusal())

    def test_a_nul_in_a_key_is_named_by_its_table(self):
        original = self.config_path.read_text(encoding="utf-8")
        _set_field(self.config_path, "model.scorer", "env", '{ "A\\u0000B" = "x" }')
        self.assertIn("a key in model.scorer.env holds a NUL character", self.refusal())
        self.config_path.write_text('"t\\u0000p" = 1\n' + original, encoding="utf-8")
        self.assertIn("a key in the top level holds a NUL character", self.refusal())

    def test_a_credential_shaped_name_is_masked_in_the_refusal(self):
        """An env name is a key, and a pasted token can be one. Measured
        before: a NUL value, or an integer no record holds, under a ``ghp_`` or
        ``sk-ant-`` name printed the token whole, while the ``=`` refusal
        masked the same name."""
        github = "gh" + "p_" + "R" * 36
        provider = "s" + "k-ant-api03-" + "Q" * 40 + "AA"
        original = self.config_path.read_text(encoding="utf-8")
        for name, value, needle in (
            (github, '"a\\u0000b"', "holds a NUL character"),
            (provider, '"a\\u0000b"', "holds a NUL character"),
            (github, "1" + "0" * 400, "is an integer of 401 digits"),
        ):
            with self.subTest(name=name[:6], needle=needle[:12]):
                self.config_path.write_text(original, encoding="utf-8")
                _set_field(
                    self.config_path, "model.scorer", "env", "{ " + name + " = " + value + " }"
                )
                message = self.refusal()
                self.assertIn(needle, message)
                self.assertIn("'model.scorer.env.[REDACTED:", message)
                self.assertNotIn(name[4:24], message)
        self.config_path.write_text(original, encoding="utf-8")
        _set_field(self.config_path, "model.scorer", "env", '{ ASI_PROBE = "a\\u0000b" }')
        self.assertIn("model.scorer.env.ASI_PROBE holds", self.refusal())

    def test_an_env_name_that_is_empty_or_holds_an_equals_sign_is_a_config_error(self):
        original = self.config_path.read_text(encoding="utf-8")
        for name in ('""', '"A=B"', '"=A"'):
            with self.subTest(name=name):
                self.config_path.write_text(original, encoding="utf-8")
                _set_field(self.config_path, "model.scorer", "env", "{ " + name + ' = "x" }')
                message = self.refusal()
                self.assertIn("[model.scorer].env name", message)
                self.assertIn("is not one an environment can hold", message)

    def test_a_name_in_these_refusals_is_quoted_bounded(self):
        original = self.config_path.read_text(encoding="utf-8")
        red = "red\\u001b[31m"
        for table, key, value, needle in (
            (
                "model.scorer",
                "env",
                f'{{ "{red}" = "a\\u0000b" }}',
                "'model.scorer.env.red\\x1b[31m' holds",
            ),
            ("caps", f'"{red}"', '{ "A\\u0000" = 1 }', "a key in 'caps.red\\x1b[31m' holds"),
            (
                "model.scorer",
                "env",
                '{ "' + "E" * 5000 + '=" = "x" }',
                "more chars not quoted] is not",
            ),
        ):
            with self.subTest(needle=needle[:24]):
                self.config_path.write_text(original, encoding="utf-8")
                _set_field(self.config_path, table, key, value)
                message = self.refusal()
                self.assertIn(needle, message)
                self.assertNotIn("\x1b", message)
                self.assertLess(len(message), 1000)

    def test_doctor_and_daily_name_each_as_a_config_error_never_a_fall(self):
        """Through the real CLI: exit 3 with the field named, before anything is read."""
        original = self.config_path.read_text(encoding="utf-8")
        state = self.base / "state"
        cases = [
            ("run", "state_dir", '"st\\u0000ate"', "run.state_dir holds a NUL"),
            ("model.scorer", "env", '{ ASI_PROBE = "a\\u0000b" }', "model.scorer.env.ASI_PROBE"),
            ("model.scorer", "env", '{ "A=B" = "x" }', "is not one an environment can hold"),
        ]
        for table, key, value, needle in cases:
            with self.subTest(needle=needle):
                self.config_path.write_text(original, encoding="utf-8")
                _set_field(self.config_path, table, key, value)
                cfg = ["--config", str(self.config_path)]
                for proc in (run_cli([*cfg, "doctor"]), run_cli([*cfg, "daily", "--day", DAY])):
                    self.assertEqual(proc.returncode, 3, proc.stderr)
                    self.assertIn(needle, proc.stderr)
                    self.assertNotIn("Traceback", proc.stderr)
                self.assertFalse((state / "conversations").exists(), "nothing was read")

    def test_the_control_escapes_a_platform_can_pass_still_load_and_run(self):
        _set_field(self.config_path, "model.scorer", "env", '{ ASI_PROBE = "a=b\\tc\\u00e9" }')
        config = load_config(self.config_path)
        self.assertEqual(config.model_roles["scorer"].env["ASI_PROBE"], "a=b\tc\u00e9")
        proc = run_cli(["--config", str(self.config_path), "daily", "--day", DAY])
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
