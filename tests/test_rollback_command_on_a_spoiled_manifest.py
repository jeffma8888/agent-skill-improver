"""The rollback command refuses a backup manifest it cannot judge, by name.

Measured before (3.11 and 3.13): a manifest whose target path held a NUL byte
made ``asi rollback`` fall with exit 70 and a traceback although nothing had
been written, and a recorded mode of ``"0o0"`` restored the skill at a mode its
owner cannot read, then reported a hash refusal with the file already
rewritten. The store-level cases are in ``tests/test_skill_identity_and_payloads.py``;
this drives the command a human runs.
"""

from __future__ import annotations

import json
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.app.state import StateStore
from agent_skill_improver.cli import RC_OK, RC_REFUSED
from agent_skill_improver.domain.skills import SKILL_FILENAME

from tests.support import build_env, run_cli


class TheRollbackCommandOnASpoiledManifestTests(unittest.TestCase):
    """Ordered: the refusal leaves the apply in place for the restore after it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name).resolve()
        cls.cfg = [
            "--config",
            str(
                build_env(
                    cls.base,
                    rubric_ids=("instruction-compliance",),
                    use_subprocess_model=True,
                    dry_run=False,
                    seed="rollback-manifest",
                )
            ),
        ]
        daily = run_cli([*cls.cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
        assert daily.returncode == RC_OK, daily.stderr
        listed = json.loads(run_cli([*cls.cfg, "--json", "candidates"]).stdout)["candidates"]
        cls.candidate_id = sorted(c["id"] for c in listed)[0]
        review = json.loads(run_cli([*cls.cfg, "--json", "review", cls.candidate_id]).stdout)
        applied = run_cli(
            [
                *cls.cfg,
                "apply",
                cls.candidate_id,
                "--approve",
                review["approval_token"],
                "--i-reviewed-this",
                "--operator",
                "tester",
            ]
        )
        assert applied.returncode == RC_OK, applied.stderr
        cls.state = StateStore(cls.base / "state")
        manifest = cls.state.latest_backup_manifest(cls.candidate_id)
        assert manifest is not None
        cls.manifest = manifest
        cls.original = json.loads(manifest.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _record(self) -> dict:
        return json.loads(self.state.candidate_path(self.candidate_id).read_text("utf-8"))

    def _skill(self) -> Path:
        return Path(self._record()["target"]["canonical_path"])

    def _spoil(self, field: str, value: object) -> None:
        raw = dict(self.original)
        raw[field] = value
        self.manifest.write_text(json.dumps(raw), encoding="utf-8")

    def test_1_an_unjudgeable_target_path_is_a_refusal_with_nothing_written(self):
        before = self._skill().read_text(encoding="utf-8")
        self._spoil("target_path", "a\x00b/" + SKILL_FILENAME)
        proc = run_cli([*self.cfg, "rollback", self.candidate_id])
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
        self.assertIn("BackupUnusable", proc.stderr)
        self.assertIn("No file was touched", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(self._skill().read_text(encoding="utf-8"), before)
        self.assertEqual(self._record()["status"], "applied")

    def test_2_a_recorded_mode_its_owner_cannot_read_is_not_obeyed(self):
        self._spoil("target_mode", "0o0")
        os.chmod(self._skill(), 0o640)
        proc = run_cli([*self.cfg, "rollback", self.candidate_id])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        backup_text = Path(self.original["backup_path"]).read_text(encoding="utf-8")
        self.assertEqual(self._skill().read_text(encoding="utf-8"), backup_text)
        self.assertEqual(oct(stat.S_IMODE(self._skill().stat().st_mode)), oct(0o640))
        self.assertEqual(self._record()["status"], "rolled-back")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
