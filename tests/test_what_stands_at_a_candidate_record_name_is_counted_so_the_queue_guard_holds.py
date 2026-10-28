"""What stands at a candidate record's name is counted, so the queue guard holds.

``scan_candidates`` in ``app/state.py`` is how ``candidates``, ``doctor`` and
``daily`` read the candidates directory. A record that does not parse is listed
as unreadable and stands ``daily`` down, since it may be an open candidate the
queue caps would count. Measured before (3.11 and 3.13, the same on both):
anything but a regular file at a record's name (a directory, empty or holding
the record, a link to the record moved elsewhere, a link to nothing, a link to
a directory, a fifo) was passed over instead, so ``candidates --all`` listed 0
candidates and 0 unreadable, ``doctor`` stayed ready with its warning, and the
next day's committed run proposed a second candidate for the same skill, where
with the record in place it proposed none. Each is now listed as unreadable,
judged with ``lstat`` and never opened or followed.
"""

from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.app.state import StateStore, UnreadableRecord
from agent_skill_improver.cli import RC_NEGATIVE, RC_OK

from tests.support import build_env, run_cli

CID = "CAND-2026-02-01-01-tdd"
#: The link Emacs makes beside a file it is editing: ``*.json`` matches it, no candidate id does.
EDITOR_LOCK = f".#{CID}.json"
#: What stands at the record's name, and the words the listing names it by.
ENTRIES = {
    "a directory": "it is a directory",
    "a directory holding the record": "it is a directory",
    "a link to the record moved elsewhere": "it is a symlink",
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
    elif kind == "a link to the record moved elsewhere":
        moved = base / "moved" / record.name
        moved.parent.mkdir(exist_ok=True)
        moved.write_bytes(body)
        os.symlink(moved, record)
    elif kind == "a link to nothing":
        os.symlink(base / "nowhere.json", record)
    elif kind == "a link to a directory":
        (base / "a-directory").mkdir(exist_ok=True)
        os.symlink(base / "a-directory", record)
    elif kind == "a fifo":
        os.mkfifo(record)
    elif kind == "the record":
        record.write_bytes(body)


def _reason(record: Path, words: str) -> str:
    return (
        f"refusing to count the candidate record {record}: {words}. This tool writes each "
        "candidate record as a regular file, so this entry is not one it wrote and may stand "
        "where an open candidate's record was; it is left alone."
    )


def _still_there(kind: str, record: Path) -> bool:
    if kind.startswith("a link"):
        return record.is_symlink()
    if kind.startswith("a directory"):
        return record.is_dir() and not record.is_symlink()
    return record.exists() and not record.is_file() and not record.is_symlink()


class TheScanCountsWhatStandsAtARecordNameTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.store = StateStore(self.base / "state")
        self.store.ensure()
        self.record = self.store.candidate_path(CID)

    def test_each_entry_at_a_record_name_is_listed_as_unreadable_for_what_it_is(self) -> None:
        for kind, words in ENTRIES.items():
            with self.subTest(kind=kind):
                _put(kind, self.record, self.base)
                scan = self.store.scan_candidates()
                self.assertEqual(scan.readable, ())
                self.assertEqual(
                    scan.unreadable,
                    (UnreadableRecord(path=self.record, reason=_reason(self.record, words)),),
                )
                self.assertTrue(_still_there(kind, self.record), "left as found")

    def test_the_listing_names_a_state_directory_with_a_line_break_on_one_line(self) -> None:
        store = StateStore(self.base / "state\nconfig error: forged")
        store.ensure()
        record = store.candidate_path(CID)
        record.mkdir()
        (found,) = store.scan_candidates().unreadable
        self.assertNotIn("\n", found.reason)
        self.assertEqual(
            found.reason,
            _reason(Path(str(record).replace(chr(10), chr(92) + "n")), "it is a directory"),
        )

    def test_the_control_an_entry_at_a_name_no_record_can_have_is_not_counted(self) -> None:
        folder = self.store.dir("candidates")
        os.symlink("someone@host.1234:1700000000", folder / EDITOR_LOCK)
        (folder / "notes.json").mkdir()
        os.mkfifo(folder / "CAND-not an id.json")
        scan = self.store.scan_candidates()
        self.assertEqual((scan.readable, scan.unreadable), ((), ()))
        self.assertEqual(
            self.store.foreign_entries(),
            [
                f"candidates/{EDITOR_LOCK} is a symlink",
                "candidates/CAND-not an id.json is not a regular file or directory",
            ],
        )

    def test_the_control_a_name_without_the_json_suffix_is_still_not_a_record(self) -> None:
        folder = self.store.dir("candidates")
        (folder / "notes").mkdir()
        (folder / "notes.txt").write_text("not a record", encoding="utf-8")
        os.symlink(self.base / "nowhere", folder / "link")
        scan = self.store.scan_candidates()
        self.assertEqual((scan.readable, scan.unreadable), ((), ()))

    def test_the_control_a_record_that_does_not_parse_keeps_its_own_reason(self) -> None:
        self.record.write_text("{", encoding="utf-8")
        (found,) = self.store.scan_candidates().unreadable
        self.assertEqual(found.path, self.record)
        self.assertNotIn("refusing to count the candidate record", found.reason)
        self.assertNotIn("may stand where an open candidate's record was", found.reason)


class TheCommandsCountWhatStandsAtARecordNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name).resolve()
        cls.cfg = ["--config", str(build_env(cls.base, dry_run=False, seed="dir-record"))]
        out = run_cli([*cls.cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
        assert out.returncode == RC_OK, out.stderr
        cls.cid = next(
            c["candidate_id"]
            for c in json.loads(out.stdout)["candidates"]
            if c.get("status") == "validated"
        )
        cls.record = cls.base / "state" / "candidates" / f"{cls.cid}.json"
        cls.body = cls.record.read_bytes()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def tearDown(self) -> None:
        _put("the record", self.record, self.base, self.body)

    def _record_files(self) -> list[str]:
        folder = self.base / "state" / "candidates"
        return sorted(p.name for p in folder.glob("*.json") if p.is_file() and not p.is_symlink())

    def test_the_control_a_regular_file_at_a_name_no_record_can_have_is_still_counted(
        self,
    ) -> None:
        # The name rule is for entries that are not regular files: a regular
        # file under any ``*.json`` name is read, and one that does not bind is
        # unreadable, as before. A Finder duplicate of an open record is such a
        # file, and dropping it would hide an open candidate from the caps.
        folder = self.record.parent
        copy = folder / f"{self.cid} copy.json"
        notes = folder / "notes.json"
        copy.write_bytes(self.body)
        notes.write_text("{}", encoding="utf-8")
        self.addCleanup(copy.unlink)
        self.addCleanup(notes.unlink)
        scan = StateStore(self.base / "state").scan_candidates()
        self.assertEqual([c.id for c in scan.readable], [self.cid])
        self.assertEqual(
            sorted(u.path.name for u in scan.unreadable), sorted([copy.name, notes.name])
        )

    def test_candidates_lists_each_entry_as_unreadable_in_text_and_json(self) -> None:
        for kind, words in ENTRIES.items():
            with self.subTest(kind=kind):
                _put(kind, self.record, self.base, self.body)
                text = run_cli([*self.cfg, "candidates", "--all"], timeout=60)
                as_json = run_cli([*self.cfg, "--json", "candidates", "--all"], timeout=60)
                self.assertEqual(text.returncode, RC_OK, text.stderr)
                reason = _reason(self.record, words)
                self.assertIn(f"UNREADABLE  {self.record.name}  {reason}", text.stdout.splitlines())
                self.assertTrue(
                    text.stdout.splitlines()[-1].endswith("; 1 unreadable record(s)"), text.stdout
                )
                payload = json.loads(as_json.stdout)
                self.assertEqual(
                    payload["unreadable"], [{"path": str(self.record), "reason": reason}]
                )
                self.assertNotIn(self.cid, [c["id"] for c in payload["candidates"]])
                self.assertTrue(_still_there(kind, self.record), "left as found")

    def test_doctor_makes_each_entry_a_problem_and_still_warns_of_it(self) -> None:
        for kind, words in ENTRIES.items():
            with self.subTest(kind=kind):
                _put(kind, self.record, self.base, self.body)
                proc = run_cli([*self.cfg, "--json", "doctor"], timeout=60)
                self.assertEqual(proc.returncode, RC_NEGATIVE, proc.stdout + proc.stderr)
                payload = json.loads(proc.stdout)
                self.assertEqual(
                    payload["problems"],
                    [
                        "candidate record cannot be read and `daily` will stand down until it "
                        f"is moved or repaired: {_reason(self.record, words)}"
                    ],
                )
                warned = [w for w in payload["warnings"] if f"candidates/{self.record.name}" in w]
                self.assertEqual(len(warned), 1, payload["warnings"])

    def test_the_next_daily_stands_down_and_proposes_nothing(self) -> None:
        for number, (kind, words) in enumerate(ENTRIES.items(), start=2):
            with self.subTest(kind=kind):
                _put(kind, self.record, self.base, self.body)
                before = self._record_files()
                proc = run_cli(
                    [*self.cfg, "--json", "daily", "--commit", "--day", f"2026-02-{number:02d}"],
                    timeout=120,
                )
                payload = json.loads(proc.stdout)
                self.assertEqual(payload["status"], "stood-down", proc.stdout + proc.stderr)
                self.assertEqual(payload["candidates"], [])
                reason = payload["stand_down_reason"]
                self.assertTrue(
                    reason.startswith("queue guard: 1 candidate record(s) under "), reason
                )
                self.assertIn(f"First: {self.record.name}: {_reason(self.record, words)} ", reason)
                self.assertEqual(self._record_files(), before, "no record was written")
                self.assertTrue(_still_there(kind, self.record), "left as found")


class TheControlTheRecordInPlaceHoldsTheSkillTests(unittest.TestCase):
    def test_with_the_record_in_place_the_next_daily_proposes_none_for_the_skill(self) -> None:
        self._next_day_proposes_none(beside=None)

    def test_an_editors_lock_link_beside_the_record_does_not_stand_the_run_down(self) -> None:
        self._next_day_proposes_none(beside=EDITOR_LOCK)

    def _next_day_proposes_none(self, *, beside: str | None) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            cfg = ["--config", str(build_env(base, dry_run=False, seed="dir-record"))]
            first = run_cli([*cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
            self.assertEqual(first.returncode, RC_OK, first.stderr)
            self.assertEqual(
                [c["candidate_id"] for c in json.loads(first.stdout)["candidates"]], [CID]
            )
            if beside is not None:
                os.symlink("someone@host.1234:1700000000", base / "state" / "candidates" / beside)
            listed = json.loads(run_cli([*cfg, "--json", "candidates", "--all"]).stdout)
            self.assertEqual(listed["unreadable"], [])
            self.assertEqual([c["id"] for c in listed["candidates"]], [CID])
            second = run_cli([*cfg, "--json", "daily", "--commit", "--day", "2026-02-02"])
            payload = json.loads(second.stdout)
            self.assertEqual(payload["status"], "ok", second.stdout + second.stderr)
            self.assertEqual(payload["candidates"], [])


if __name__ == "__main__":
    unittest.main()
