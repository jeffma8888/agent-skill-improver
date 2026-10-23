"""An unknown config key is named the way every other refused field is named.

A quoted TOML key may hold almost anything (a NUL is refused before the key is
checked): a line break, a terminal escape, a pasted credential, a megabyte.
``_refuse_unknown_keys`` in the config loader, and the refusal ``[source]``
words for itself, printed each refused key as written, where the rubric
loader's refusal and ``_named`` already quoted, masked and cut them. Measured
before (3.11 and 3.13), in both: a key holding a line break and then ``config
error: forged`` printed as two refusal lines in ``doctor``'s output, a
``ghp_`` token used as a key was printed whole, and a 1,000,000-character key
made a 1 MB refusal.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.config import load_config
from agent_skill_improver.domain.redaction import QUOTED_VALUE_CHARS
from agent_skill_improver.errors import ConfigError
from tests.support import build_env, run_cli

DAY = "2026-03-01"

#: Built in pieces so the leak scan does not read this file as holding one.
TOKEN = "g" + "hp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"

FORGED = "config error: forged"


class UnknownConfigKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base, seed="unknown-keys"))
        self.config_text = self.config_path.read_text(encoding="utf-8")

    def with_key_in(self, table: str, key: str) -> None:
        """Write the config with ``key = 1`` added to ``table``, a header the
        generated config has once; ``""`` adds a top-level table named ``key``."""
        if table:
            header = f"[{table}]\n"
            self.assertEqual(self.config_text.count(header), 1)
            text = self.config_text.replace(header, f"{header}{key} = 1\n", 1)
        else:
            text = self.config_text + f"\n[{key}]\nx = 1\n"
        self.config_path.write_text(text, encoding="utf-8")

    def refusal(self) -> str:
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        return str(ctx.exception)

    def test_a_line_break_in_an_unknown_key_cannot_start_a_line(self):
        cases = (
            ("caps", "a", "'a\\nconfig error: forged'"),
            ("source", "a", "'a\\nconfig error: forged'"),
            ("", "a", "'a\\nconfig error: forged'"),
            # Close enough to a setting to be offered as a misspelling of it.
            (
                "caps",
                "max_open_candidate",
                "'max_open_candidate\\nconfig error: forged' (did you mean max_open_candidates?)",
            ),
        )
        for table, head, shown in cases:
            with self.subTest(table=table or "top level", head=head):
                self.with_key_in(table, f'"{head}\\n{FORGED}"')
                self.assertNotIn("\n", self.refusal())
                for cmd in (["doctor"], ["daily", "--day", DAY]):
                    proc = run_cli(["--config", str(self.config_path), *cmd])
                    self.assertEqual(proc.returncode, 3, proc.stderr)
                    lines = proc.stderr.splitlines()
                    self.assertEqual(len(lines), 1, proc.stderr)
                    self.assertFalse(lines[0].startswith(FORGED), proc.stderr)
                    self.assertIn(f"has unknown key(s): {shown}", lines[0])

    def test_a_credential_shaped_unknown_key_is_masked(self):
        for table in ("caps", "source", "model.defaults", ""):
            with self.subTest(table=table or "top level"):
                self.with_key_in(table, f'"{TOKEN}"')
                message = self.refusal()
                self.assertNotIn(TOKEN[len("g" + "hp_") :], message)
                self.assertIn("[REDACTED:github-token]", message)
                self.assertIn("has unknown key(s): ", message)
                proc = run_cli(["--config", str(self.config_path), "doctor"])
                self.assertEqual(proc.returncode, 3, proc.stderr)
                self.assertNotIn(TOKEN[len("g" + "hp_") :], proc.stderr)

    def test_a_long_unknown_key_is_cut(self):
        for table in ("caps", "source"):
            with self.subTest(table=table):
                self.with_key_in(table, '"' + "k" * 1_000_000 + '"')
                message = self.refusal()
                self.assertIn("more chars not quoted]", message)
                self.assertLess(len(message), 2_000)
                proc = run_cli(["--config", str(self.config_path), "doctor"])
                self.assertEqual(proc.returncode, 3, proc.stderr[:500])
                self.assertLess(len(proc.stderr), 2_000)

    def test_each_of_several_unknown_keys_is_named_on_its_own(self):
        keys = ("aaa_plain", '"b\\nconfig error: forged"', "ccc_plain", f'"{TOKEN}"')
        shown = "aaa_plain, 'b\\nconfig error: forged', ccc_plain, '[REDACTED:github-token]'"
        for table in ("caps", "source"):
            with self.subTest(table=table):
                header = f"[{table}]\n"
                self.assertEqual(self.config_text.count(header), 1)
                added = "".join(f"{key} = 1\n" for key in keys)
                self.config_path.write_text(
                    self.config_text.replace(header, header + added, 1), encoding="utf-8"
                )
                self.assertIn(f"[{table}] has unknown key(s): {shown}. ", self.refusal())

    def test_the_control_a_plain_unknown_key_is_still_named_bare(self):
        self.with_key_in("caps", "max_open_candidate")
        self.assertTrue(
            self.refusal().startswith(
                "[caps] has unknown key(s): max_open_candidate (did you mean "
                "max_open_candidates?)."
            )
        )
        self.with_key_in("run", "k" * QUOTED_VALUE_CHARS)
        self.assertTrue(self.refusal().startswith(f"[run] has unknown key(s): {'k' * 120}. "))
        self.assertEqual(QUOTED_VALUE_CHARS, 120)
        self.with_key_in("source", "zz_plain")
        self.assertTrue(self.refusal().startswith("[source] has unknown key(s): zz_plain. "))


if __name__ == "__main__":
    unittest.main()
