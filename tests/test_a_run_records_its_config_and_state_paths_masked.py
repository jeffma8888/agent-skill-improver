"""A run records its config and state paths masked.

A committed daily run records where its config and its state tree are, and
which skill file each candidate would change, in its receipt, and the path of
that receipt in its checkpoint; nothing reads any of them back. Measured
before (3.11 and 3.13, a committed run with the fixtures source under a
directory named like a provider key, one candidate): the key was raw in the
receipt three times (as ``config_path``, as ``state_dir`` and as the
candidate's ``canonical_path``) and once in the checkpoint
(``last_receipt``). ``app/daily.py`` now masks each as it records it. The
candidate record keeps its skill file as written, because ``apply`` writes
there, so review and apply work as they did.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_skill_improver import cli

from tests.support import build_env

#: Shaped like a provider key, so redaction masks it; split so no scanner
#: reads this file as holding one.
KEY = "sk-ant-" + "api03-" + "A1b2C3d4E5f6G7h8" * 4
MASK = "[REDACTED:model-provider-key]"
DAY = "2026-02-01"


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


class _Run:
    """One committed daily run under ``base``, and what it recorded."""

    def __init__(self, base: Path) -> None:
        self.config = build_env(base / "env", seed="c216")
        self.state = base / "env" / "state"
        self.document = base / "env" / "skills-canonical" / "tdd" / "SKILL.md"
        self.rc, self.output = run(self.config, "daily", "--commit", "--day", DAY)
        (self.receipt_file,) = sorted(self.state.glob("receipts/*.json"))
        (self.checkpoint_file,) = sorted(self.state.glob("checkpoints/*.json"))
        (self.candidate_file,) = sorted(self.state.glob("candidates/*.json"))
        self.receipt = json.loads(self.receipt_file.read_text(encoding="utf-8"))
        self.checkpoint = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
        self.candidate = json.loads(self.candidate_file.read_text(encoding="utf-8"))


class ARunRecordsItsPathsMaskedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp()).resolve()
        cls.keyed = _Run(cls.tmp / "key" / KEY)
        cls.plain = _Run(cls.tmp / "plain" / "plain-dir")

    @classmethod
    def tearDownClass(cls) -> None:
        subprocess.run(["rm", "-rf", str(cls.tmp)])

    def masked(self, path: Path) -> str:
        """The path as written, with the key's directory name replaced by the
        mask (computed here, not by ``redact``)."""
        self.assertIn(KEY, str(path))
        return str(path).replace(KEY, MASK)

    def test_both_runs_are_healthy_and_propose_one_candidate(self) -> None:
        for each in (self.keyed, self.plain):
            self.assertEqual(each.rc, 0, each.output)
            self.assertIn(f"status=ok day={DAY} mode=commit", each.output)
            self.assertEqual(len(each.receipt["candidates"]), 1)

    def test_the_receipt_records_the_config_path_masked(self) -> None:
        self.assertEqual(self.keyed.receipt["config_path"], self.masked(self.keyed.config))

    def test_the_receipt_records_the_state_dir_masked(self) -> None:
        self.assertEqual(self.keyed.receipt["state_dir"], self.masked(self.keyed.state))

    def test_the_receipt_records_each_candidate_skill_file_masked(self) -> None:
        (entry,) = self.keyed.receipt["candidates"]
        self.assertEqual(entry["canonical_path"], self.masked(self.keyed.document))

    def test_the_checkpoint_records_the_receipt_path_masked(self) -> None:
        self.assertEqual(
            self.keyed.checkpoint["last_receipt"], self.masked(self.keyed.receipt_file)
        )

    def test_no_byte_of_the_receipt_or_the_checkpoint_holds_the_key(self) -> None:
        for path in (self.keyed.receipt_file, self.keyed.checkpoint_file):
            self.assertNotIn(KEY.encode(), path.read_bytes(), path.name)

    def test_the_candidate_record_keeps_its_skill_file_as_written(self) -> None:
        self.assertEqual(self.keyed.candidate["target"]["canonical_path"], str(self.keyed.document))

    def test_a_plain_directory_is_recorded_as_written(self) -> None:
        plain = self.plain
        self.assertEqual(plain.receipt["config_path"], str(plain.config))
        self.assertEqual(plain.receipt["state_dir"], str(plain.state))
        (entry,) = plain.receipt["candidates"]
        self.assertEqual(entry["canonical_path"], str(plain.document))
        self.assertEqual(plain.checkpoint["last_receipt"], str(plain.receipt_file))
        self.assertEqual(plain.candidate["target"]["canonical_path"], str(plain.document))


class ReviewAndApplyStillWriteTheSkillTests(unittest.TestCase):
    """The masked receipt changes nothing ``review`` and ``apply`` read."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.tmp)])

    def test_apply_writes_the_skill_under_the_key_shaped_directory(self) -> None:
        keyed = _Run(self.tmp / KEY)
        self.assertEqual(keyed.rc, 0, keyed.output)
        before = keyed.document.read_text(encoding="utf-8")
        cid = keyed.candidate["id"]
        rc, review = run(keyed.config, "review", cid)
        self.assertEqual(rc, 0, review)
        token = re.search(r"--approve (\S+)", review)
        self.assertIsNotNone(token, review)
        rc, applied = run(
            keyed.config, "apply", cid, "--approve", token.group(1), "--i-reviewed-this"
        )
        self.assertEqual(rc, 0, applied)
        after = keyed.document.read_text(encoding="utf-8")
        self.assertNotEqual(after, before)
        record = json.loads(keyed.candidate_file.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "applied")
        self.assertEqual(record["target"]["canonical_path"], str(keyed.document))


if __name__ == "__main__":
    unittest.main()
