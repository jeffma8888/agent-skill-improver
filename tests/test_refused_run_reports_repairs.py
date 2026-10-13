"""A run that refuses still reports the modes it reset first.

The owner-only mode repair every ``daily`` run performs comes before every
refusal the run can make, and a refused run writes no receipt. Measured before:
a receipt left at 0644 and a directory at 0755 were reset by the run that then
refused as blocked (exit 5), nothing said so, and the next ``doctor`` reported
0 repaired. A drifted mode means something else touched the private tree; the
command now names the repair on stderr before the refusal so the signal is not
swallowed by the thing that stopped the run.
"""

from __future__ import annotations

import json
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.cli import RC_BLOCKED, RC_OK, RC_REFUSED

from tests.support import build_env, run_cli

NOTE = "had a mode other than owner-only and were reset to it before this run refused"


class ARefusedRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.cfg = ["--config", str(build_env(self.base, seed="rep", use_subprocess_model=True))]
        self.state_dir = self.base / "state"
        first = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(first.returncode, RC_OK, first.stderr)
        (self.receipt,) = (self.state_dir / "receipts").glob("*.json")

    def _block(self) -> None:
        payload = json.loads(self.receipt.read_text(encoding="utf-8"))
        payload["status"] = "blocked"
        payload["skill_tree"]["mutation_detected"] = True
        self.receipt.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _mode(path: Path) -> int:
        return stat.S_IMODE(path.lstat().st_mode)

    def test_the_reset_modes_are_named_on_stderr_before_the_refusal(self) -> None:
        self._block()
        self.receipt.chmod(0o644)
        (self.state_dir / "conversations").chmod(0o755)
        refused = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-02"])
        self.assertEqual(refused.returncode, RC_BLOCKED, refused.stderr)
        note, refusal = refused.stderr.strip().splitlines()
        self.assertTrue(note.startswith(f"note: 2 path(s) under {self.state_dir} {NOTE} ("), note)
        self.assertIn("conversations", note)
        self.assertIn(f"receipts/{self.receipt.name}", note)
        self.assertTrue(note.endswith("so no receipt reports the repair"), note)
        self.assertTrue(refusal.startswith("BlockedRunUnreviewed: "), refusal)
        self.assertEqual(self._mode(self.receipt), 0o600, "the repair itself still happens")
        self.assertEqual(self._mode(self.state_dir / "conversations"), 0o700)
        self.assertEqual(refused.stdout, "", "a refusal prints no receipt")

    def test_an_ordinary_refusal_reports_it_too_and_a_clean_tree_says_nothing(self) -> None:
        self._block()
        clean = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-02"])
        self.assertEqual(clean.returncode, RC_BLOCKED, clean.stderr)
        self.assertNotIn("note:", clean.stderr, "nothing was reset, so nothing is reported")
        self.assertEqual(len(clean.stderr.strip().splitlines()), 1, clean.stderr)
        # A different refusal (exit 3), the same report.
        self.receipt.write_text("{", encoding="utf-8")  # torn: StateUnreadable, exit 3
        self.receipt.chmod(0o600)
        self.receipt.parent.chmod(0o755)
        refused = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-02"])
        self.assertEqual(refused.returncode, RC_REFUSED, refused.stderr)
        note, refusal = refused.stderr.strip().splitlines()
        self.assertIn(f"1 path(s) under {self.state_dir} {NOTE} (receipts)", note)
        self.assertTrue(refusal.startswith("StateUnreadable: "), refusal)

    def test_a_refusal_before_the_run_itself_reports_the_sandbox_reset_wiring_made(self) -> None:
        # Wiring the runner opens state/sandbox and resets its mode; a model
        # command that cannot start then refuses before run_daily and ensure().
        (self.state_dir / "sandbox").chmod(0o755)
        config_path = Path(self.cfg[1])
        text = config_path.read_text(encoding="utf-8")
        with_missing = text.replace("argv = [", 'argv = ["/nonexistent/asi-model-binary", ', 1)
        self.assertNotEqual(text, with_missing)
        config_path.write_text(with_missing, encoding="utf-8")
        refused = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-02"])
        self.assertEqual(refused.returncode, RC_REFUSED, refused.stderr)
        note, refusal = refused.stderr.strip().splitlines()
        self.assertIn(f"1 path(s) under {self.state_dir} {NOTE} (sandbox)", note)
        self.assertTrue(refusal.startswith("ConfigError: model "), refusal)
        self.assertEqual(self._mode(self.state_dir / "sandbox"), 0o700)

    def test_a_run_that_proceeds_reports_the_repair_in_its_receipt_not_on_stderr(self) -> None:
        self.receipt.chmod(0o644)
        proceeded = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-01-02"])
        self.assertEqual(proceeded.returncode, RC_OK, proceeded.stderr)
        self.assertNotIn(NOTE, proceeded.stderr)
        payload = json.loads(proceeded.stdout)
        self.assertEqual(payload["counts"]["state_modes_repaired"], 1)
        self.assertTrue(any("had a mode other than owner-only" in n for n in payload["notes"]))

    def test_a_reset_path_with_a_line_break_in_its_name_is_spelled_onto_the_one_line(self) -> None:
        self._block()
        odd = self.state_dir / "conversations" / "first\nsecond.json"
        odd.write_text("{}", encoding="utf-8")
        odd.chmod(0o644)
        refused = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-02"])
        self.assertEqual(refused.returncode, RC_BLOCKED, refused.stderr)
        lines = refused.stderr.strip().splitlines()
        self.assertEqual(len(lines), 2, refused.stderr)
        self.assertIn("conversations/first", lines[0])
        self.assertIn("second.json", lines[0])
        self.assertNotIn(os.linesep + "second", refused.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
