"""A config refusal that names a path with a line break in it stays on one line.

Every config refusal is written as one line, but many name a value the config
supplied: a skill root's path, the config file's own path, the rubric file's.
A path may hold a line break, and it was printed raw, so the text after the
break stood at column 0 of stderr as a line of its own. Measured before (3.11
and 3.13): a ``mirror`` root nested inside ``canonical``, each path holding
``\\nconfig error: forged``, made ``doctor`` print one refusal as 3 lines, two
of them starting ``config error: forged``. :class:`ConfigError` now spells a
line break in its message (``\\n``, ``\\r``), so the fix holds at every site that
names a path and at any written later.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.config import load_config
from agent_skill_improver.errors import ConfigError, SafetyRefusal
from tests.support import build_env, run_cli

FORGED = "config error: forged"


class ConfigRefusalOnOneLineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base / "env", dry_run=False, seed="one-line"))
        self.config_text = self.config_path.read_text(encoding="utf-8")

    def edit(self, old: str, new: str) -> None:
        text = self.config_path.read_text(encoding="utf-8")
        self.assertEqual(text.count(old), 1, old)
        self.config_path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def assert_one_line(self, config: Path, *commands: str) -> list[str]:
        """Each command refuses with exit 3 on exactly one stderr line, which
        spells the line break; returns those lines."""
        shown = []
        for command in commands:
            with self.subTest(command=command):
                proc = run_cli(["--config", str(config), command])
                self.assertEqual(proc.returncode, 3, proc.stderr)
                lines = proc.stderr.splitlines()
                self.assertEqual(len(lines), 1, proc.stderr)
                self.assertIn(f"\\n{FORGED}", lines[0])
                self.assertFalse(lines[0].startswith(FORGED), proc.stderr)
                shown.append(lines[0])
        return shown

    def test_nested_skill_roots_whose_paths_hold_line_breaks_are_refused_on_one_line(self):
        canonical = str(self.base / "env" / "skills-canonical")
        self.edit(f'path = "{canonical}"', f'path = "{canonical}\\n{FORGED}"')
        self.edit(
            f'path = "{self.base / "env" / "skills-mirror"}"',
            f'path = "{canonical}\\n{FORGED}/inner"',
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        message = str(ctx.exception)
        self.assertNotIn("\n", message)
        self.assertIn(f"skills-canonical\\n{FORGED}/inner) lies inside skill_root", message)
        self.assert_one_line(self.config_path, "doctor", "skills")

    def test_a_rubric_directory_whose_path_holds_a_line_break_is_named_on_one_line(self):
        self.edit('dir = "../rubrics"', f'dir = "rubrics\\n{FORGED}"')
        for line in self.assert_one_line(self.config_path, "doctor", "skills"):
            self.assertIn("not found at ", line)

    def test_a_config_file_path_with_a_line_break_is_named_on_one_line(self):
        missing = self.base / f"absent\n{FORGED}.toml"
        for line in self.assert_one_line(missing, "doctor", "skills"):
            self.assertIn("config file not found: ", line)

    def test_a_config_file_that_is_not_utf8_at_such_a_path_is_named_on_one_line(self):
        undecodable = self.base / f"latin\n{FORGED}.toml"
        undecodable.write_bytes(self.config_text.encode("utf-8") + b"# caf\xe9\n")
        for line in self.assert_one_line(undecodable, "doctor", "skills"):
            self.assertIn("is not UTF-8 text", line)

    def test_only_a_config_refusal_is_flattened_and_its_text_survives_a_rebuild(self):
        error = ConfigError("a\nb\r\nc")
        self.assertEqual(str(error), "a\\nb\\r\\nc")
        self.assertEqual(str(ConfigError("a\rb")), "a\\rb")
        self.assertEqual(str(ConfigError(*error.args)), str(error))
        self.assertEqual(str(ConfigError("plain: one line")), "plain: one line")
        # The refusals that list their reasons one per line keep that layout.
        self.assertEqual(str(SafetyRefusal("refusing:\n  - one\n  - two")).count("\n"), 2)


if __name__ == "__main__":
    unittest.main()
