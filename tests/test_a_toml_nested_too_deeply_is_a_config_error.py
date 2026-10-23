"""A TOML value nested too deeply for the parser is a config error, not a fall.

``tomllib`` descends once per array or inline table in a value, so a few
hundred nested raise ``RecursionError``, which is not the ``ValueError`` the
loader turns into a config error. Measured before (3.11 and 3.13): ``doctor``
and ``daily`` first fell inside the parser, exit 70 with a traceback, at 493 or
494 nested arrays and at 328 or 329 nested inline tables, in a config value or
a rubric. At 400 nested arrays (3.13) the same config was refused for its
unknown table, exit 3.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agent_skill_improver import config as config_module
from agent_skill_improver.config import TOML_TOO_DEEP, load_config
from tests.support import build_env, run_cli

DAY = "2026-03-01"

#: Past the parser's reach under the default recursion limit (1,000 frames):
#: ``doctor`` first fell at 493 or 494 nested arrays and 328 or 329 inline
#: tables, so this holds unless something raises the limit past about 2,000.
DEEP = 1000

#: Well inside it, for the control.
SHALLOW = 20


def nested_arrays(depth: int) -> str:
    return "[" * depth + "]" * depth


def nested_tables(depth: int) -> str:
    return "{ b = " * depth + "1" + " }" * depth


SHAPES = {"arrays": nested_arrays, "inline tables": nested_tables}


class TomlNestedTooDeeplyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base, seed="toml-depth"))
        self.config_text = self.config_path.read_text(encoding="utf-8")
        rubrics = sorted((self.base / "rubrics").glob("*.toml"))
        self.assertTrue(rubrics)
        self.rubric_path = rubrics[0]
        self.rubric_text = self.rubric_path.read_text(encoding="utf-8")

    def doctor_and_daily(self) -> list:
        cfg = ["--config", str(self.config_path)]
        return [run_cli([*cfg, "doctor"]), run_cli([*cfg, "daily", "--day", DAY])]

    def assert_refused_as_too_deep(self, where: Path) -> None:
        for proc in self.doctor_and_daily():
            self.assertEqual(proc.returncode, 3, proc.stderr[-2000:])
            self.assertIn(f"{where}: invalid TOML: {TOML_TOO_DEEP}", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertNotIn("RecursionError", proc.stderr)

    def test_a_config_value_nested_too_deeply_is_a_config_error(self):
        self.assertIn("nested too deeply", TOML_TOO_DEEP)
        for shape, nested in SHAPES.items():
            with self.subTest(shape=shape):
                self.config_path.write_text(
                    self.config_text + "\n[extra]\nx = " + nested(DEEP) + "\n", encoding="utf-8"
                )
                self.assert_refused_as_too_deep(self.config_path)

    def test_a_rubric_value_nested_too_deeply_is_a_config_error(self):
        for shape, nested in SHAPES.items():
            with self.subTest(shape=shape):
                self.rubric_path.write_text(
                    "zz = " + nested(DEEP) + "\n" + self.rubric_text, encoding="utf-8"
                )
                self.assert_refused_as_too_deep(self.rubric_path)

    def test_the_control_a_shallow_nesting_is_read_and_refused_for_what_it_says(self):
        for shape, nested in SHAPES.items():
            for where in ("config", "rubric"):
                with self.subTest(shape=shape, where=where):
                    if where == "config":
                        self.config_path.write_text(
                            self.config_text + "\n[extra]\nx = " + nested(SHALLOW) + "\n",
                            encoding="utf-8",
                        )
                        said = "the config file has unknown key(s): extra"
                    else:
                        self.config_path.write_text(self.config_text, encoding="utf-8")
                        self.rubric_path.write_text(
                            "zz = " + nested(SHALLOW) + "\n" + self.rubric_text, encoding="utf-8"
                        )
                        said = f"{self.rubric_path}: the rubric file has unknown key(s): 'zz'"
                    for proc in self.doctor_and_daily():
                        self.assertEqual(proc.returncode, 3, proc.stderr[-2000:])
                        self.assertIn(said, proc.stderr)
                        self.assertNotIn(TOML_TOO_DEEP, proc.stderr)
                        self.assertNotIn("Traceback", proc.stderr)
                    self.rubric_path.write_text(self.rubric_text, encoding="utf-8")

    def test_a_recursion_error_from_reading_the_file_is_not_called_nesting(self):
        read = config_module._read_utf8_or_refuse

        def read_until(target: Path):
            def reader(path: Path, *, what: str) -> str:
                if path == target:
                    raise RecursionError("maximum recursion depth exceeded")
                return read(path, what=what)

            return reader

        for target in (self.config_path, self.rubric_path):
            with self.subTest(file=target.name):
                with mock.patch.object(
                    config_module, "_read_utf8_or_refuse", read_until(target.resolve())
                ):
                    with self.assertRaises(RecursionError):
                        load_config(self.config_path)


if __name__ == "__main__":
    unittest.main()
