"""A configured path the platform cannot hold is refused at load, by name.

A file name holds at most NAME_MAX bytes and a whole path at most PATH_MAX less
one for its terminator (255 and 1023 on macOS), and the system calls on a path
past either (``stat``, ``open``, ``mkdir``, ``listdir``) fail with ``File name
too long``. The state directory needs room below it too: a record is written up
to three names below it (a backup: the backups directory, its candidate's
directory, the file).

Measured before (3.11 and 3.13): a 300-byte component in ``[run].state_dir``,
``[rubrics].dir`` or ``[source].fixtures_dir``, or in the ``--config`` path,
made ``doctor`` and ``daily`` fall with ``OSError: [Errno 63] File name too
long``, exit 70; in ``[run].base_dir``, ``doctor`` fell the same way and
``daily`` refused with a ``SafetyRefusal``; a 300-byte ``[rubrics].enabled`` id
made ``doctor`` fall too; and a ``state_dir`` of 849 bytes passed ``doctor``,
then made a committed ``daily`` fall the same way mid-run, after it had written
records, with no receipt.

Each length below is sized from the limits read, so that a case meets its limit
where PATH_MAX is another number than macOS's, as Linux's 4096 is. Measured
before, on the Linux CI jobs (3.11, 3.12 and 3.13): three tests sized for 1024
failed there, the link under ``base_dir`` and the 1000-byte ``[rubrics].dir``
loading without a ``ConfigError``, and the 849-byte ``state_dir`` passing
``doctor`` (exit 0) and a committed ``daily`` writing below it.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver import config as config_module
from agent_skill_improver.app.state import SUBDIRS
from agent_skill_improver.config import NAME_MAX, PATH_MAX, STATE_PATH_RESERVE, load_config
from agent_skill_improver.errors import ConfigError

from tests.support import build_env, run_cli
from tests.test_non_finite_numbers import _set_field

DAY = "2026-01-02"
LONG_NAME = "c" * (NAME_MAX + 45)
#: A ``state_dir`` this long is within PATH_MAX but leaves less than the reserve
#: below it: 849 bytes on macOS, where it was measured.
PAST_THE_RESERVE = PATH_MAX - 175


def relative_path_of(total: int, base: Path, *, name: str = "s" * 200) -> str:
    """A relative path that makes ``base / it`` exactly ``total`` bytes, in copies of ``name``."""
    room = total - len(os.fsencode(str(base))) - 1
    step = len(os.fsencode(name)) + 1
    parts: list[str] = []
    while room > step:
        parts.append(name)
        room -= step
    parts.append("t" * room)
    rel = "/".join(parts)
    assert len(os.fsencode(str(base / rel))) == total
    return rel


class AConfigPathThePlatformCannotHoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base, seed="platform-paths"))
        self.original = self.config_path.read_text(encoding="utf-8")

    def refusal(self) -> str:
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        return str(ctx.exception)

    def test_a_name_past_name_max_in_any_configured_path_is_a_config_error(self):
        for table, key in (
            ("run", "state_dir"),
            ("run", "base_dir"),
            ("rubrics", "dir"),
            ("source", "fixtures_dir"),
        ):
            with self.subTest(field=f"{table}.{key}"):
                self.config_path.write_text(self.original, encoding="utf-8")
                _set_field(self.config_path, table, key, json.dumps(LONG_NAME))
                message = self.refusal()
                self.assertTrue(message.startswith(f"[{table}].{key} '"), message[:80])
                self.assertIn(f"has a name of {len(LONG_NAME)} bytes", message)
                self.assertIn(f"at most {NAME_MAX}", message)
                self.assertIn("more chars not quoted]", message, "the path is quoted bounded")

    def test_a_name_is_measured_in_bytes_not_characters(self):
        """150 characters of "\u00e9" are 300 bytes: past NAME_MAX, though shorter than it."""
        _set_field(self.config_path, "run", "state_dir", json.dumps("\u00e9" * 150))
        self.assertIn("has a name of 300 bytes", self.refusal())

    def with_extra_root(self, root_id: str, path: str) -> None:
        """The config with one more root, neither canonical nor editable."""
        self.config_path.write_text(
            self.original
            + "\n[[skill_root]]\n"
            + f"id = {json.dumps(root_id)}\npath = {json.dumps(path)}\n"
            + 'kind = "custom"\nprecedence = 50\ncanonical = false\neditable = false\n',
            encoding="utf-8",
        )

    def test_a_name_past_name_max_in_a_skill_root_path_is_a_config_error(self):
        """Every root, the canonical one, the mirror and an extra read-only one,
        though a run goes on without a mirror or a read-only root that is absent."""
        roots = tomllib.loads(self.original)["skill_root"]
        self.assertEqual([r["kind"] for r in roots], ["custom", "mirror"])
        blocks = list(re.finditer(r'(\[\[skill_root\]\][^\[]*?path = "[^"]*)(")', self.original))
        self.assertEqual(len(blocks), len(roots))
        for root, block in zip(roots, blocks):
            with self.subTest(root=root["id"]):
                self.config_path.write_text(
                    self.original[: block.end(1)] + "/" + LONG_NAME + self.original[block.end(1) :],
                    encoding="utf-8",
                )
                message = self.refusal()
                self.assertIn(f"has a name of {len(LONG_NAME)} bytes", message)
                self.assertTrue(
                    message.startswith(f"skill_root {root['id']!r} path '"), message[:80]
                )
        with self.subTest(root="extra"):
            self.with_extra_root("extra", str(self.base / LONG_NAME))
            self.assertTrue(self.refusal().startswith("skill_root 'extra' path '"))

    def test_a_root_or_rubric_id_in_the_refusal_is_quoted_bounded_and_masked(self):
        token = "gh" + "p_" + "R" * 36
        for root_id in ("i" * 400, token, "a\nb\x1b[31m"):
            with self.subTest(root=root_id[:8]):
                self.with_extra_root(root_id, str(self.base / LONG_NAME))
                message = self.refusal()
                # The id's shape is checked before its path, so each of these is
                # refused as an id; what holds is that none reaches it raw.
                self.assertTrue(message.startswith("[[skill_root]] #"), message[:40])
                self.assertIn(" id '", message)
                self.assertNotIn("i" * 200, message)
                self.assertNotIn("R" * 20, message)
                self.assertNotIn("\n", message)
                self.assertNotIn("\x1b", message)
        for rubric_id, needle in (
            (LONG_NAME, "more chars not quoted] in [rubrics].enabled"),
            (token + "c" * 260, "[REDACTED:github-token]"),
        ):
            with self.subTest(rubric=rubric_id[:8]):
                self.config_path.write_text(self.original, encoding="utf-8")
                _set_field(self.config_path, "rubrics", "enabled", json.dumps([rubric_id]))
                message = self.refusal()
                self.assertIn(needle, message)
                self.assertNotIn(LONG_NAME, message)
                self.assertNotIn("R" * 20, message)

    def test_a_name_of_exactly_name_max_bytes_loads_and_one_more_is_refused(self):
        _set_field(self.config_path, "run", "state_dir", json.dumps("n" * NAME_MAX))
        self.assertEqual(load_config(self.config_path).state_dir.name, "n" * NAME_MAX)
        _set_field(self.config_path, "run", "state_dir", json.dumps("n" * (NAME_MAX + 1)))
        self.assertIn(f"has a name of {NAME_MAX + 1} bytes", self.refusal())

    def test_a_rubric_id_too_long_for_a_file_name_is_a_config_error(self):
        """Measured before: ``is_file()`` on the rubric's file fell with
        ``OSError: [Errno 63] File name too long``, and ``doctor`` exited 70."""
        _set_field(self.config_path, "rubrics", "enabled", json.dumps([LONG_NAME]))
        message = self.refusal()
        self.assertIn("in [rubrics].enabled", message)
        self.assertIn(f"has a name of {len(LONG_NAME) + len('.toml')} bytes", message)
        proc = run_cli(["--config", str(self.config_path), "doctor"])
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_a_path_past_path_max_is_a_config_error(self):
        rel = relative_path_of(PATH_MAX, self.base)
        _set_field(self.config_path, "source", "fixtures_dir", json.dumps(rel))
        message = self.refusal()
        self.assertIn(f"is {PATH_MAX} bytes long", message)
        self.assertIn(f"at most {PATH_MAX - 1}", message)

    def test_a_path_length_is_measured_in_bytes_not_characters(self):
        """Names of 100 "\u00e9" are 200 bytes each: the path reaches PATH_MAX
        bytes in far fewer characters."""
        rel = relative_path_of(PATH_MAX, self.base, name="\u00e9" * 100)
        self.assertLess(len(str(self.base / rel)), PATH_MAX - 1)
        _set_field(self.config_path, "source", "fixtures_dir", json.dumps(rel))
        self.assertIn(f"is {PATH_MAX} bytes long", self.refusal())

    def test_a_long_path_is_quoted_bounded_and_masked_in_the_refusal(self):
        token = "gh" + "p_" + "R" * 36
        top = Path("/" + token)
        where = top / relative_path_of(PATH_MAX, top)
        _set_field(self.config_path, "source", "fixtures_dir", json.dumps(str(where)))
        message = self.refusal()
        self.assertIn(f"is {PATH_MAX} bytes long", message)
        self.assertIn("[REDACTED:github-token]", message)
        self.assertNotIn("R" * 20, message)
        self.assertIn("more chars not quoted]", message)
        self.assertLess(len(message), 1000)

    def test_a_state_dir_that_leaves_no_room_for_the_deepest_record_is_refused(self):
        widest = PATH_MAX - 1 - STATE_PATH_RESERVE
        # Past PATH_MAX as well, it is told the state directory's own limit
        # first, not PATH_MAX's and then, once shortened, the reserve's.
        for length in (widest + 1, PATH_MAX + 76):
            with self.subTest(length=length):
                self.config_path.write_text(self.original, encoding="utf-8")
                _set_field(
                    self.config_path,
                    "run",
                    "state_dir",
                    json.dumps(relative_path_of(length, self.base)),
                )
                message = self.refusal()
                self.assertIn(f"is {length} bytes long", message)
                self.assertIn(f"at most {widest} so that {STATE_PATH_RESERVE} bytes", message)
                self.assertIn("No record was written", message)

    def test_the_reserve_is_measured_with_the_ancestors_resolved(self):
        """The store works in the state directory with its ancestors resolved,
        so a short path through a link to a long directory is measured long."""
        widest = PATH_MAX - 1 - STATE_PATH_RESERVE
        target = self.base / relative_path_of(widest - 5, self.base)
        target.mkdir(parents=True)
        (self.base / "short").symlink_to(target)
        _set_field(
            self.config_path, "run", "state_dir", json.dumps(str(self.base / "short" / "state-x"))
        )
        self.assertLess(len(str(self.base / "short" / "state-x")), widest)
        message = self.refusal()
        self.assertTrue(message.startswith("[run].state_dir '"), message[:40])
        self.assertIn(f"is {widest - 5 + len('/state-x')} bytes long", message)

    def test_a_base_dir_that_resolves_too_long_is_named_as_itself(self):
        """A short base_dir through a link to a directory past PATH_MAX is
        refused as the base_dir, not later as whatever lies under it."""
        deep = relative_path_of(PATH_MAX + 1, self.base, name="d" * 200)
        here = os.getcwd()
        self.addCleanup(os.chdir, here)
        os.chdir(self.base)  # past PATH_MAX, the directories are made one level at a time
        for part in deep.split("/"):
            os.mkdir(part)
            os.chdir(part)
        os.chdir(here)
        (self.base / "baselink").symlink_to(deep)
        _set_field(self.config_path, "run", "base_dir", json.dumps(str(self.base / "baselink")))
        message = self.refusal()
        self.assertTrue(message.startswith("[run].base_dir '"), message[:60])
        self.assertIn("bytes long, and here it can be at most", message)

    def test_a_linked_rubrics_directory_that_resolves_too_long_is_a_config_error(self):
        """Measured before: a ``[rubrics].dir`` linked to a 1000-byte directory
        (PATH_MAX less 24 on macOS) made ``doctor`` fall at the rubric file's
        ``is_file`` with ``File name too long``, exit 70."""
        deep = self.base / relative_path_of(PATH_MAX - 24, self.base)
        deep.mkdir(parents=True)
        here = os.getcwd()
        self.addCleanup(os.chdir, here)
        os.chdir(deep)  # the rubric's whole path is past PATH_MAX: write it from inside
        for rubric in (self.base / "rubrics").glob("*.toml"):
            Path(rubric.name).write_bytes(rubric.read_bytes())
        os.chdir(here)
        (self.base / "rublink").symlink_to(deep)
        _set_field(self.config_path, "rubrics", "dir", json.dumps(str(self.base / "rublink")))
        message = self.refusal()
        self.assertIn("in [rubrics].enabled", message)
        self.assertIn("bytes long, and here it can be at most", message)
        proc = run_cli(["--config", str(self.config_path), "doctor"])
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_the_control_the_widest_state_dir_allowed_runs_a_committed_daily(self):
        widest = PATH_MAX - 1 - STATE_PATH_RESERVE
        rel = relative_path_of(widest, self.base)
        _set_field(self.config_path, "run", "state_dir", json.dumps(rel))
        self.assertEqual(len(os.fsencode(str(load_config(self.config_path).state_dir))), widest)
        proc = run_cli(["--config", str(self.config_path), "daily", "--day", DAY, "--commit"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        written = [p for p in (self.base / rel).rglob("*") if p.is_file()]
        self.assertTrue(any(p.parent.name == "receipts" for p in written), "a receipt was written")

    def test_doctor_and_daily_name_a_long_path_as_a_config_error_never_a_fall(self):
        cases = (
            ("run", "state_dir", json.dumps(LONG_NAME), "has a name of"),
            ("rubrics", "dir", json.dumps(LONG_NAME), "has a name of"),
            (
                "run",
                "state_dir",
                json.dumps(relative_path_of(PAST_THE_RESERVE, self.base)),
                f"is {PAST_THE_RESERVE} bytes",
            ),
        )
        for table, key, value, needle in cases:
            with self.subTest(field=f"{table}.{key}", needle=needle):
                self.config_path.write_text(self.original, encoding="utf-8")
                _set_field(self.config_path, table, key, value)
                cfg = ["--config", str(self.config_path)]
                runs = (
                    run_cli([*cfg, "doctor"]),
                    run_cli([*cfg, "daily", "--day", DAY, "--commit"]),
                )
                for proc in runs:
                    self.assertEqual(proc.returncode, 3, proc.stderr)
                    self.assertIn(needle, proc.stderr)
                    self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(any(self.base.rglob("conversations")), "nothing was read or written")

    def test_a_config_path_past_the_limits_is_a_config_error(self):
        for where in (
            self.base / LONG_NAME / "c.toml",
            self.base / relative_path_of(PATH_MAX, self.base),
        ):
            with self.subTest(length=len(str(where))):
                proc = run_cli(["--config", str(where), "doctor"])
                self.assertEqual(proc.returncode, 3, proc.stderr)
                self.assertIn("config file", proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)

    def test_the_reserve_formula_follows_the_state_subdirectories(self):
        """A record is at most a subdirectory, one directory and one file below
        the state directory (``backups/<candidate>--<stamp>/SKILL.md.bak``).
        This holds the formula to the store's subdirectory names; it does not
        walk the records the store writes."""
        self.assertEqual(config_module._LONGEST_STATE_SUBDIR, max(len(name) for name in SUBDIRS))
        self.assertEqual(
            STATE_PATH_RESERVE, 1 + max(len(name) for name in SUBDIRS) + 2 * (1 + NAME_MAX)
        )


if __name__ == "__main__":
    unittest.main()
