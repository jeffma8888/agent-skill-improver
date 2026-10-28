"""``rollback`` prints what it restored as lines, as ``apply`` does.

Measured before (3.11 and 3.13): ``rollback`` printed the skill store's result
as a Python dict on the line that named the candidate, its keys, quotes and
``True``/``False`` included (``rolled back CAND-...: {'rolled_back': '...',
'restored_sha256': '...', 'from_backup': '...', 'already_restored': False}``).
It now prints the path it restored, the hash it restored and the backup it
restored from, one per line through ``_emit``, and says when nothing was
written because the file already held the backup's text. ``--json`` is as
before.
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.domain.candidate import make_patch
from agent_skill_improver.hashing import sha256_text
from tests.support import build_env, run_cli

FORGED = "config error: forged"


class _Applied(unittest.TestCase):
    """A fresh applied candidate per test, against the ``tdd`` skill."""

    name = "tdd"

    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.cfg = ["--config", str(build_env(base, dry_run=False, seed="c170-rollback"))]
        proc = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
        self.cid = next(
            c["candidate_id"]
            for c in json.loads(proc.stdout)["candidates"]
            if c.get("status") == "validated"
        )
        record = base / "state" / "candidates" / f"{self.cid}.json"
        self.target = base / "skills-canonical" / "tdd" / "SKILL.md"
        if self.name != "tdd":
            moved = self.target.parent.parent / self.name
            self.target.parent.rename(moved)
            self.target = moved / "SKILL.md"
            data = json.loads(record.read_text(encoding="utf-8"))
            data["target"]["skill_id"] = self.name
            data["target"]["canonical_path"] = str(self.target)
            data["patch"]["unified_diff"] = make_patch(
                self.target.read_text(encoding="utf-8"),
                data["patch"]["new_text"],
                path_label=str(self.target),
            ).unified_diff
            record.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.before = self.target.read_bytes()
        token = json.loads(run_cli([*self.cfg, "--json", "review", self.cid]).stdout)
        proc = run_cli(
            [
                *self.cfg,
                "apply",
                self.cid,
                "--approve",
                token["approval_token"],
                "--i-reviewed-this",
            ]
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        backups = sorted((base / "state").rglob("*.bak"))
        self.assertEqual(len(backups), 1, backups)
        self.backup = backups[0]

    def rollback(self, *flags: str):
        proc = run_cli([*self.cfg, *flags, "rollback", self.cid])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        return proc


class RollbackPrintsWhatItRestoredTests(_Applied):
    def test_rollback_prints_the_path_hash_and_backup_each_on_its_line(self) -> None:
        proc = self.rollback()
        self.assertEqual(self.target.read_bytes(), self.before)
        self.assertEqual(
            proc.stdout.split("\n"),
            [
                f"rolled back {self.cid}: {self.target}",
                f"  restored sha256 {sha256_text(self.before.decode('utf-8'))}",
                f"  from backup     {self.backup}",
                "",
            ],
        )

    def test_rollback_of_a_file_already_holding_the_backup_says_nothing_was_written(self) -> None:
        shutil.copyfile(self.backup, self.target)
        proc = self.rollback()
        lines = proc.stdout.split("\n")
        self.assertEqual(len(lines), 5, proc.stdout)
        self.assertEqual(lines[0], f"rolled back {self.cid}: {self.target}")
        self.assertEqual(
            lines[3],
            "  nothing written the file already held the backup's text (a second rollback, "
            "or the backup copied by hand)",
        )
        self.assertEqual(lines[4], "")

    def test_rollback_json_is_the_receipt_as_before(self) -> None:
        proc = self.rollback("--json")
        out = json.loads(proc.stdout)
        self.assertEqual(
            sorted(out), ["candidate_id", "manifest", "result", "rolled_back_at", "schema"]
        )
        self.assertEqual(out["schema"], "asi.rollback_receipt.v1")
        self.assertEqual(out["candidate_id"], self.cid)
        self.assertEqual(out["result"]["rolled_back"], str(self.target), out["result"])
        self.assertEqual(out["result"]["from_backup"], str(self.backup))
        self.assertIs(out["result"]["already_restored"], False)

    def test_the_text_holds_no_python_dict(self) -> None:
        proc = self.rollback()
        for piece in ("{", "}", "'rolled_back'", "already_restored", "False", "True"):
            with self.subTest(piece=piece):
                self.assertNotIn(piece, proc.stdout)


class ALineBreakInTheSkillPathStaysOnItsLineTests(_Applied):
    name = f"tdd\n{FORGED}"

    def test_a_line_break_in_the_restored_path_is_spelled_on_its_line(self) -> None:
        proc = self.rollback()
        lines = proc.stdout.split("\n")
        self.assertEqual(len(lines), 4, proc.stdout)
        self.assertFalse(any(line.startswith(FORGED) for line in lines), proc.stdout)
        self.assertIn("tdd\\nconfig error: forged", lines[0])
        self.assertEqual(self.target.read_bytes(), self.before)


if __name__ == "__main__":
    unittest.main()
