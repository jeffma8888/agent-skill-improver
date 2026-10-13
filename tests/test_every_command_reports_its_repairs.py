"""Every command that opens the state tree names the modes it reset.

``dir`` resets a drifted mode on the way in and leaves the reset pending for
``ensure`` to report. Only ``daily`` calls ``ensure`` and only ``doctor``
reports its own walk. Measured before: with ``state/`` and ``state/candidates``
at 0755, ``candidates``, ``set-aside``, ``review``, ``discover``, ``apply`` and
``rollback`` each put the directory back to 0700 and none of them said so, on
success or on refusal. Now ``main`` names any reset no command output has
named, on stderr, before whatever ends the command.
"""

from __future__ import annotations

import contextlib
import io
import json
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver import cli
from agent_skill_improver.cli import RC_OK, RC_REFUSED

from tests.support import build_env, run_cli

NOTE = "had a mode other than owner-only and were reset to it while this command ran"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


class EveryCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name).resolve()
        cls.cfg = [
            "--config",
            str(build_env(cls.base, seed="rep-all", use_subprocess_model=True, dry_run=False)),
        ]
        cls.state_dir = cls.base / "state"
        first = run_cli([*cls.cfg, "daily", "--commit", "--day", "2026-01-01"])
        assert first.returncode == RC_OK, first.stderr
        listed = json.loads(run_cli([*cls.cfg, "--json", "candidates"]).stdout)["candidates"]
        cls.candidate_id = sorted(c["id"] for c in listed)[0]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _drift(self) -> None:
        self.state_dir.chmod(0o755)
        (self.state_dir / "candidates").chmod(0o755)

    def test_a_listing_a_review_and_a_refused_promotion_each_name_the_reset(self) -> None:
        cases = {
            "candidates": ([*self.cfg, "candidates"], RC_OK, None),
            "review": ([*self.cfg, "review", self.candidate_id], RC_OK, None),
            "apply": (
                [*self.cfg, "apply", self.candidate_id, "--approve", "bogus", "--i-reviewed-this"],
                RC_REFUSED,
                "ApprovalRequired: ",
            ),
            "rollback": ([*self.cfg, "rollback", self.candidate_id], RC_REFUSED, "GateFailed: "),
        }
        for name, (argv, rc, refusal) in cases.items():
            with self.subTest(command=name):
                self._drift()
                proc = run_cli(argv)
                self.assertEqual(proc.returncode, rc, proc.stderr)
                lines = proc.stderr.strip().splitlines()
                self.assertTrue(
                    lines[0].startswith(f"note: 2 path(s) under {self.state_dir} {NOTE} (")
                )
                self.assertIn("(., candidates)", lines[0])
                self.assertTrue(lines[0].endswith("the repair changed no content"), lines[0])
                if refusal is None:
                    self.assertEqual(len(lines), 1, proc.stderr)
                else:
                    self.assertEqual(len(lines), 2, proc.stderr)
                    self.assertTrue(lines[1].startswith(refusal), "the note comes first")
                self.assertEqual(_mode(self.state_dir), 0o700)
                self.assertEqual(_mode(self.state_dir / "candidates"), 0o700)
                if rc == RC_OK and name != "review":
                    self.assertTrue(proc.stdout.strip(), "stdout still carries the listing")

    def test_only_the_directories_a_command_opens_are_reset_and_named(self) -> None:
        # set-aside opens quarantine, refusals and deferred, not candidates: the
        # candidates mode is left as found and the note says what was reset.
        self._drift()
        proc = run_cli([*self.cfg, "set-aside"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        (note,) = proc.stderr.strip().splitlines()
        self.assertIn(f"1 path(s) under {self.state_dir} {NOTE} (.)", note)
        self.assertEqual(_mode(self.state_dir), 0o700)
        self.assertEqual(_mode(self.state_dir / "candidates"), 0o755, "not opened, not touched")
        (self.state_dir / "candidates").chmod(0o700)

    def test_a_clean_tree_adds_no_line_and_a_json_listing_stays_json(self) -> None:
        proc = run_cli([*self.cfg, "--json", "candidates"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertEqual(proc.stderr, "")
        json.loads(proc.stdout)
        self._drift()
        drifted = run_cli([*self.cfg, "--json", "candidates"])
        self.assertEqual(drifted.returncode, RC_OK, drifted.stderr)
        self.assertIn(NOTE, drifted.stderr, "the note goes to stderr")
        json.loads(drifted.stdout)

    def test_an_in_process_second_call_does_not_inherit_the_first_calls_store(self) -> None:
        # The registry is module-level; cleared on exit, a stale store cannot
        # attribute its reset to the next command in the same process.
        self._drift()
        first = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(first):
            self.assertEqual(cli.main([*self.cfg, "candidates"]), RC_OK)
        self.assertIn(NOTE, first.getvalue())
        self.assertEqual(cli._OPEN_STORES, [], "cleared on the way out")
        second = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(second):
            self.assertEqual(cli.main([*self.cfg, "skills"]), RC_OK)
        self.assertEqual(second.getvalue(), "", "skills opens no state tree and says nothing")

    def test_daily_and_doctor_report_once_in_their_own_output_not_twice(self) -> None:
        self._drift()
        doctor = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(doctor.returncode, RC_OK, doctor.stderr)
        self.assertNotIn(NOTE, doctor.stderr, "doctor's report already names them")
        payload = json.loads(doctor.stdout)
        self.assertEqual(sorted(payload["state_permissions"]["repaired"]), [".", "candidates"])
        self._drift()
        daily = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-02"])
        self.assertEqual(daily.returncode, RC_OK, daily.stderr)
        self.assertNotIn("mode other than owner-only", daily.stderr, "the receipt names them")
        self.assertEqual(json.loads(daily.stdout)["counts"]["state_modes_repaired"], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
