"""A skill root id is refused at load unless it can be printed and stored as written.

The id is written into every candidate and receipt for a skill in its root,
labels the payload files beside its skills as ``root:name`` and is named in
refusals, and it was checked only for being a non-empty string. Measured before
(3.11 and 3.13): an id holding a line break and then ``config error: forged``
made a ``precedence`` or ``canonical`` refusal print as two lines; a ``ghp_``
token used as an id was printed whole by six refusals and by ``skills``, and
written into the receipt and the candidate of a committed ``daily``; a
1,000,000-character id made each refusal about 1 MB; ``can:onical`` loaded and
labelled a payload file beside its skill ``can:onical:run.sh`` in
``skills --json``. An id is now 1 to 64 letters, digits, ``.``, ``_`` or ``-``,
starting with a letter or digit (the shape of a run day), and is not something
redaction would mask.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.config import load_config
from agent_skill_improver.errors import ConfigError
from tests.support import build_env, run_cli

DAY = "2026-03-01"

#: Built in pieces so the leak scan does not read this file as holding one.
TOKEN = "g" + "hp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
TOKEN_BODY = TOKEN[len("g" + "hp_") :]

FORGED = "config error: forged"


class SkillRootIdTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base, dry_run=False, seed="root-ids"))
        self.config_text = self.config_path.read_text(encoding="utf-8")
        self.assertEqual(self.config_text.count('id = "canonical"'), 1)

    def with_root_id(self, literal: str, *, also: tuple[str, str] | None = None) -> None:
        """Give the canonical root the id ``literal`` (as TOML writes it), and
        make the one further edit ``also`` (old, new) if one is given."""
        text = self.config_text.replace('id = "canonical"', f"id = {literal}", 1)
        if also is not None:
            self.assertEqual(text.count(also[0]), 1)
            text = text.replace(also[0], also[1], 1)
        self.config_path.write_text(text, encoding="utf-8")

    def refusal(self) -> str:
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        return str(ctx.exception)

    def doctor_and_daily(self) -> list:
        cfg = ["--config", str(self.config_path)]
        return [run_cli([*cfg, "doctor"]), run_cli([*cfg, "daily", "--day", DAY, "--commit"])]

    def test_a_root_id_with_a_line_break_is_refused_on_one_line(self):
        # With a refusal of its own beside it, which used to name the id raw.
        for also in (None, ("precedence = 10\n", 'precedence = "x"\n')):
            with self.subTest(also=also):
                self.with_root_id(f'"a\\n{FORGED}"', also=also)
                message = self.refusal()
                self.assertTrue(
                    message.startswith(f"[[skill_root]] #0 id 'a\\n{FORGED}' must be "), message
                )
                for proc in self.doctor_and_daily():
                    self.assertEqual(proc.returncode, 3, proc.stderr)
                    lines = proc.stderr.splitlines()
                    self.assertEqual(len(lines), 1, proc.stderr)
                    self.assertFalse(lines[0].startswith(FORGED), proc.stderr)

    def test_a_credential_shaped_root_id_is_refused_masked_and_never_stored(self):
        self.with_root_id(f'"{TOKEN}"')
        message = self.refusal()
        self.assertIn("[[skill_root]] #0 id '[REDACTED:github-token]' must be ", message)
        self.assertNotIn(TOKEN_BODY, message)
        for proc in self.doctor_and_daily():
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertNotIn(TOKEN_BODY, proc.stdout + proc.stderr)
        stored = [p for p in self.base.rglob("*") if p.is_file() and p != self.config_path]
        for path in stored:
            self.assertNotIn(TOKEN_BODY, path.read_text(encoding="utf-8", errors="replace"), path)
        self.assertFalse((self.base / "state" / "receipts").exists())

    def test_a_root_id_longer_than_sixty_four_is_refused_and_cut(self):
        self.with_root_id('"' + "k" * 65 + '"')
        self.assertIn("must be 1 to 64 letters", self.refusal())
        self.with_root_id('"' + "k" * 1_000_000 + '"')
        message = self.refusal()
        self.assertIn("more chars not quoted]", message)
        self.assertLess(len(message), 2_000)
        proc = run_cli(["--config", str(self.config_path), "doctor"])
        self.assertEqual(proc.returncode, 3, proc.stderr[:500])
        self.assertLess(len(proc.stderr), 2_000)

    def test_a_root_id_with_a_colon_a_space_or_a_leading_mark_is_refused(self):
        for rid in ("can:onical", "my skills", ".hidden", "-x", "_x", "café", " canonical"):
            with self.subTest(rid=rid):
                self.with_root_id(f'"{rid}"')
                self.assertIn(f"[[skill_root]] #0 id {rid!r} must be 1 to 64 ", self.refusal())

    def test_the_control_an_id_of_that_shape_still_loads_and_runs(self):
        for rid in ("c", "k" * 64, "example-canonical", "a.b_c-1", "Canonical2"):
            with self.subTest(rid=rid[:12]):
                self.with_root_id(f'"{rid}"')
                config = load_config(self.config_path)
                self.assertIn(rid, [root.id for root in config.skill_roots])
                doctor = run_cli(["--config", str(self.config_path), "doctor"])
                self.assertEqual(doctor.returncode, 0, doctor.stderr)


if __name__ == "__main__":
    unittest.main()
