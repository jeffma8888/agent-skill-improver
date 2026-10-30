"""A skill document whose path is longer than the system opens is passed over.

Discovery lists a skill root by names, then takes each document by its whole
path, resolved below the root. A root reached through a link to a deep
directory can hold an entry whose whole path is past PATH_MAX less its
terminator, and the first call on it, ``lstat``, is refused with ``File name
too long``. Measured before (3.11 and 3.13): with the canonical root linked to
such a directory, an entry of 1,024 bytes made ``doctor`` and ``daily`` fall in
``_iter_skill_files`` with that ``OSError``, exit 70; at 1,023 bytes both ran.
"""

from __future__ import annotations

import errno
import os
import re
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.config import NAME_MAX, PATH_MAX, load_config

from tests.support import build_env, run_cli

DAY = "2026-01-02"
LONGEST = PATH_MAX - 1


def _deep_dir(start: Path, total: int) -> Path:
    """A new directory whose path is exactly ``total`` bytes, made a level at a time."""
    here = start
    while len(os.fsencode(str(here))) < total:
        room = total - len(os.fsencode(str(here))) - 1
        here = here / ("d" * min(NAME_MAX - 55, room))
        here.mkdir()
    assert len(os.fsencode(str(here))) == total
    return here


class ASkillPathLongerThanTheSystemOpensTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base, dry_run=False, seed="long-skill-path"))
        lines = [
            line for line in self.config_path.read_text().splitlines() if line.startswith("path = ")
        ]
        self.canonical = Path(lines[0].split("=", 1)[1].strip().strip('"'))
        self.assertTrue((self.canonical / "tdd" / "SKILL.md").is_file())
        self.cfg = ["--config", str(self.config_path)]

    def _link_canonical_so_tdd_is(self, size: int, *, skill: str = "tdd") -> Path:
        """Move the canonical root under a deep directory, link it back, and
        return where ``<skill>/SKILL.md`` now resolves, ``size`` bytes long."""
        leaf = f"/{skill}/SKILL.md"
        far = self.base / "far"
        far.mkdir()
        deep = _deep_dir(far, size - len(leaf))
        deep.rmdir()
        shutil.move(str(self.canonical), str(deep))
        self.canonical.symlink_to(deep)
        resolved = deep / skill / "SKILL.md"
        self.assertEqual(len(os.fsencode(str(resolved))), size)
        return resolved

    def test_an_entry_past_the_path_limit_is_passed_over_by_its_length(self):
        resolved = self._link_canonical_so_tdd_is(LONGEST + 1)
        doctor = run_cli([*self.cfg, "doctor"])
        self.assertEqual(doctor.returncode, 0, doctor.stderr)
        self.assertNotIn("Traceback", doctor.stderr)
        self.assertIn("verdict: ready", doctor.stdout)
        self.assertIn(
            f"{resolved} is a path of {LONGEST + 1} bytes, longer than this system opens "
            "(File name too long)",
            doctor.stdout,
        )

    def test_daily_runs_and_names_the_canonical_copy_it_passed_over(self):
        self._link_canonical_so_tdd_is(LONGEST + 7)
        daily = run_cli([*self.cfg, "daily", "--commit", "--day", DAY])
        self.assertEqual(daily.returncode, 0, daily.stderr)
        self.assertNotIn("Traceback", daily.stderr)
        report = daily.stdout + daily.stderr
        self.assertIn("receipt:", report)
        # The copy keeps its skill's canonical slot: tdd is held back, by name,
        # rather than proposed against another root's copy.
        self.assertRegex(
            report,
            re.compile(
                r"skill 'tdd' has no canonical editable copy \(the copy in editable root "
                rf"'canonical' was passed over by discovery \(a path of {LONGEST + 7} bytes"
            ),
        )

    def test_the_control_an_entry_at_the_path_limit_is_inventoried(self):
        longest_name = max((p.name for p in self.canonical.iterdir() if p.is_dir()), key=len)
        self._link_canonical_so_tdd_is(LONGEST, skill=longest_name)
        doctor = run_cli([*self.cfg, "doctor"])
        self.assertEqual(doctor.returncode, 0, doctor.stderr)
        self.assertIn(" 0 passed over", doctor.stdout)
        self.assertNotIn("longer than this system opens", doctor.stdout)

    def test_any_other_refusal_of_that_call_is_not_taken_for_a_length(self):
        # Only ``File name too long`` is a length. A document the system
        # refuses for another reason, here permission on the way to it, is not
        # passed over as a long one but in the system's words, as an entry
        # discovery cannot examine (test_an_entry_discovery_cannot_examine_is_
        # passed_over_with_why); before c226 it was raised.
        config = load_config(self.config_path)
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        real = Path.is_symlink
        refused: list[Path] = []

        def is_symlink(path: Path) -> bool:
            if path.name == "SKILL.md" and path.parent.name == "tdd":
                refused.append(path)
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
            return real(path)

        with mock.patch.object(Path, "is_symlink", is_symlink):
            known = store.known_skill_ids()
        # Once for each root's copy, at the call this change wraps: the walk
        # goes on past the first.
        self.assertEqual(len(refused), 2, refused)
        self.assertEqual(len(set(refused)), 2, refused)
        self.assertNotIn("tdd", known)
        self.assertEqual(
            sorted(
                (d.path, d.skill_id, d.reason) for d in store.passed_over() if d.path in refused
            ),
            sorted(
                (path, "tdd", "an entry this process cannot examine (Permission denied)")
                for path in refused
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
