"""Review, apply and rollback name what stands at a candidate's record path rather than calling it missing.

``read_candidate`` in ``app/state.py`` reads ``<id>.json`` under the candidates
directory for ``review``, ``apply`` and ``rollback``. This tool writes each record
as a regular file, so anything else at that name is somebody else's, and
``doctor`` already warns of it (a directory as one that sits where only records
are written, a link or a fifo as an entry left alone). Measured before (3.11 and
3.13, the same on both): a directory, a directory holding the record, a link to
nothing, a link to a directory and a fifo at a validated candidate's path each
made all three commands say "not found: no candidate", exit 4, as if nothing
were there (15 of 15); only a link to a regular file was refused by name. Each
is now refused by name for what it is, exit 3.
"""

from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.app.state import StateStore, StateUnreadable
from agent_skill_improver.cli import RC_MISSING_FILE, RC_OK, RC_REFUSED

from tests.support import build_env, run_cli

CID = "CAND-2026-02-01-01-tdd"
#: What stands at the record's name, and the words the refusal names it by.
ENTRIES = {
    "a directory": "it is a directory",
    "a directory holding the record": "it is a directory",
    "a link to nothing": "it is a symlink",
    "a link to a directory": "it is a symlink",
    "a fifo": "it is not a regular file or directory",
}


def _put(kind: str, record: Path, base: Path, body: bytes = b"{}") -> None:
    if record.is_symlink() or (record.exists() and not record.is_dir()):
        record.unlink()
    elif record.is_dir():
        shutil.rmtree(record)
    if kind == "a directory":
        record.mkdir()
    elif kind == "a directory holding the record":
        record.mkdir()
        (record / "inside.json").write_bytes(body)
    elif kind == "a link to nothing":
        os.symlink(base / "nowhere.json", record)
    elif kind == "a link to a directory":
        (base / "a-directory").mkdir(exist_ok=True)
        os.symlink(base / "a-directory", record)
    elif kind == "a fifo":
        os.mkfifo(record)
    elif kind == "the record":
        record.write_bytes(body)


class TheStoreNamesWhatStandsAtTheNameTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.store = StateStore(self.base / "state")
        self.store.ensure()
        self.record = self.store.candidate_path(CID)

    def test_each_entry_that_is_not_a_regular_file_is_refused_for_what_it_is(self) -> None:
        for kind, words in ENTRIES.items():
            with self.subTest(kind=kind):
                _put(kind, self.record, self.base)
                with self.assertRaises(StateUnreadable) as caught:
                    self.store.read_candidate(CID)
                self.assertEqual(
                    str(caught.exception),
                    f"refusing to use the candidate record {self.record}: {words}. This tool "
                    "writes each candidate record as a regular file, so this entry is not one "
                    "it wrote; it is left alone, and `doctor` warns of it.",
                )
                self.assertTrue(self.record.is_symlink() or self.record.exists(), "left alone")

    def test_a_state_directory_with_a_line_break_is_named_on_one_line(self) -> None:
        store = StateStore(self.base / "state\nconfig error: forged")
        store.ensure()
        record = store.candidate_path(CID)
        record.mkdir()
        with self.assertRaises(StateUnreadable) as caught:
            store.read_candidate(CID)
        message = str(caught.exception)
        self.assertNotIn("\n", message)
        self.assertIn(
            f" {str(record).replace(chr(10), chr(92) + 'n')}: it is a directory. ", message
        )

    def test_the_control_a_missing_record_is_still_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            self.store.read_candidate(CID)
        self.assertEqual(str(caught.exception), f"no candidate {CID!r} at {self.record}")

    def test_the_control_a_link_to_a_regular_file_is_refused_as_before(self) -> None:
        target = self.base / "aside.json"
        target.write_text("{}", encoding="utf-8")
        os.symlink(target, self.record)
        with self.assertRaises(StateUnreadable) as caught:
            self.store.read_candidate(CID)
        self.assertEqual(
            str(caught.exception),
            f"refusing to read state artifact {self.record}: the leaf is a symlink",
        )


class TheCommandsSayWhatStandsThereTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name).resolve()
        cls.cfg = ["--config", str(build_env(cls.base, dry_run=False, seed="what-stands-there"))]
        out = run_cli([*cls.cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
        cls.cid = next(
            c["candidate_id"]
            for c in json.loads(out.stdout)["candidates"]
            if c.get("status") == "validated"
        )
        review = run_cli([*cls.cfg, "--json", "review", cls.cid])
        cls.token = json.loads(review.stdout)["approval_token"]
        cls.record = cls.base / "state" / "candidates" / f"{cls.cid}.json"
        cls.body = cls.record.read_bytes()
        cls.skill = Path(json.loads(cls.body)["target"]["canonical_path"])
        cls.skill_body = cls.skill.read_bytes()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _commands(self) -> dict[str, list[str]]:
        return {
            "review": ["review", self.cid],
            "apply": ["apply", self.cid, "--approve", self.token, "--i-reviewed-this"],
            "rollback": ["rollback", self.cid],
        }

    def test_review_apply_and_rollback_refuse_each_entry_by_name(self) -> None:
        for kind, words in ENTRIES.items():
            _put(kind, self.record, self.base, self.body)
            for name, argv in self._commands().items():
                with self.subTest(kind=kind, command=name):
                    proc = run_cli([*self.cfg, *argv], timeout=60)
                    self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
                    self.assertEqual(
                        proc.stderr.splitlines()[0],
                        f"StateUnreadable: refusing to use the candidate record {self.record}: "
                        f"{words}. This tool writes each candidate record as a regular file, so "
                        "this entry is not one it wrote; it is left alone, and `doctor` warns "
                        "of it.",
                    )
        _put("the record", self.record, self.base, self.body)
        self.assertEqual(self.record.read_bytes(), self.body)
        self.assertEqual(self.skill.read_bytes(), self.skill_body, "nothing was applied")

    def test_doctor_warns_of_each_entry_the_refusal_says_it_does(self) -> None:
        for kind in ENTRIES:
            with self.subTest(kind=kind):
                _put(kind, self.record, self.base, self.body)
                proc = run_cli([*self.cfg, "doctor"], timeout=60)
                warned = [
                    line
                    for line in (proc.stdout + proc.stderr).splitlines()
                    if line.startswith("WARN") and f"candidates/{self.record.name}" in line
                ]
                self.assertEqual(len(warned), 1, proc.stdout + proc.stderr)
        _put("the record", self.record, self.base, self.body)

    def test_the_control_a_missing_record_is_exit_4_and_the_record_reads(self) -> None:
        self.record.unlink()
        try:
            for name, argv in self._commands().items():
                with self.subTest(command=name):
                    proc = run_cli([*self.cfg, *argv], timeout=60)
                    self.assertEqual(proc.returncode, RC_MISSING_FILE, proc.stderr)
                    self.assertEqual(
                        proc.stderr.splitlines()[0],
                        f"not found: no candidate {self.cid!r} at {self.record}",
                    )
        finally:
            _put("the record", self.record, self.base, self.body)
        proc = run_cli([*self.cfg, "review", self.cid])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
