"""``doctor`` answers the health questions the operations doc asks.

The doc says there is no dashboard and names the questions that matter: is the
newest receipt recent, did the last run detect a skill tree mutation, does the
receipt name a confinement mode. Measured before: nothing answered them short
of listing the receipts directory; ``doctor`` said ``verdict: ready`` with the
schedule silent for days and with a mutation recorded in the newest receipt.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver import SCHEMA_RECEIPT
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.cli import RC_NEGATIVE, RC_OK
from agent_skill_improver.clock import FrozenClock

from tests.support import build_env, run_cli


def _receipt(day: str, finished_at: str, **extra) -> dict:
    payload = {
        "schema": SCHEMA_RECEIPT,
        "day": day,
        "status": "ok",
        "started_at": finished_at,
        "finished_at": finished_at,
        "dry_run": False,
        "skill_tree": {"mutation_detected": False},
        "isolation": {"mode": "none"},
    }
    payload.update(extra)
    return payload


class TheStoreFindsTheNewestDailyReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve() / "state"
        clock = FrozenClock(datetime.datetime(2026, 1, 5, 0, 0, 0, tzinfo=datetime.timezone.utc))
        self.store = StateStore(self.root, clock=clock)

    def _write(self, name: str, payload) -> Path:
        path = self.root / "receipts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_without_a_receipts_directory_nothing_is_created(self) -> None:
        self.root.mkdir()
        scan = self.store.scan_receipts()
        self.assertIsNone(scan.last_run)
        self.assertEqual(scan.unreadable, ())
        self.assertFalse((self.root / "receipts").exists(), "doctor creates nothing")

    def test_the_newest_by_its_own_stamp_wins_and_other_schemas_are_passed_over(self) -> None:
        self._write("b.json", _receipt("2026-01-02", "2026-01-02T05:00:00Z"))
        self._write("a.json", _receipt("2026-01-04", "2026-01-04T05:00:00Z", status="partial"))
        self._write("c.json", _receipt("2026-01-03", "2026-01-03T05:00:00Z"))
        self._write(
            "apply--CAND-x.json",
            {
                "schema": "asi.apply_receipt.v1",
                "day": "2026-01-09",
                "finished_at": "2026-01-09T00:00:00Z",
            },
        )
        scan = self.store.scan_receipts()
        self.assertEqual(scan.unreadable, ())
        assert scan.last_run is not None
        self.assertEqual((scan.last_run.day, scan.last_run.status), ("2026-01-04", "partial"))
        self.assertEqual(scan.last_run.finished_at, "2026-01-04T05:00:00Z")
        self.assertIs(scan.last_run.mutation_detected, False)
        self.assertEqual(scan.last_run.isolation_mode, "none")
        self.assertEqual(scan.last_run.path.name, "a.json")

    def test_a_tie_keeps_path_order_and_a_missing_stamp_falls_back(self) -> None:
        self._write("z.json", _receipt("2026-01-02", "2026-01-02T05:00:00Z"))
        self._write("y.json", _receipt("2026-01-02", "2026-01-02T05:00:00Z"))
        first = self.store.scan_receipts().last_run
        assert first is not None
        self.assertEqual(first.path.name, "y.json")
        self.assertEqual(self.store.scan_receipts().last_run, first, "stable across scans")
        started_only = _receipt("2026-01-06", "2026-01-06T00:00:00Z")
        started_only["finished_at"] = None
        self._write("s.json", started_only)
        latest = self.store.scan_receipts().last_run
        assert latest is not None
        self.assertEqual((latest.day, latest.finished_at), ("2026-01-06", "2026-01-06T00:00:00Z"))

    def test_what_does_not_read_is_named_and_a_link_is_skipped(self) -> None:
        good = self._write("ok.json", _receipt("2026-01-02", "2026-01-02T05:00:00Z"))
        self._write("half.json", {"schema": SCHEMA_RECEIPT, "day": "2026-01-09"})
        (self.root / "receipts" / "junk.json").write_text("{", encoding="utf-8")
        (self.root / "receipts" / "list.json").write_text("[]", encoding="utf-8")
        os.symlink(good, self.root / "receipts" / "link.json")
        scan = self.store.scan_receipts()
        assert scan.last_run is not None
        self.assertEqual(scan.last_run.path, good)
        self.assertEqual(
            sorted(u.path.name for u in scan.unreadable), ["half.json", "junk.json", "list.json"]
        )
        reasons = {u.path.name: u.reason for u in scan.unreadable}
        self.assertIn("field 'status' is missing", reasons["half.json"])
        self.assertIn("list, not an object", reasons["list.json"])

    def test_a_linked_directory_reads_as_absent_and_a_directory_entry_is_skipped(self) -> None:
        elsewhere = self.root.parent / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "ok.json").write_text(
            json.dumps(_receipt("2026-01-02", "2026-01-02T05:00:00Z")), encoding="utf-8"
        )
        self.root.mkdir()
        os.symlink(elsewhere, self.root / "receipts")
        scan = self.store.scan_receipts()
        self.assertIsNone(scan.last_run, "a linked receipts directory is not read through")
        self.assertEqual(scan.unreadable, ())

        (self.root / "receipts").unlink()
        good = self._write("ok.json", _receipt("2026-01-02", "2026-01-02T05:00:00Z"))
        (self.root / "receipts" / "newer.json").mkdir()
        (self.root / "receipts" / "newer.json" / "inner.json").write_text(
            json.dumps(_receipt("2026-01-09", "2026-01-09T05:00:00Z")), encoding="utf-8"
        )
        scan = self.store.scan_receipts()
        assert scan.last_run is not None
        self.assertEqual(scan.last_run.path, good, "a directory named like a record is skipped")
        self.assertEqual(scan.unreadable, ())

    def test_a_receipt_that_does_not_say_about_the_tree_reads_as_unknown(self) -> None:
        payload = _receipt("2026-01-02", "2026-01-02T05:00:00Z")
        del payload["skill_tree"]
        del payload["isolation"]
        self._write("old.json", payload)
        run = self.store.scan_receipts().last_run
        assert run is not None
        self.assertIsNone(run.mutation_detected)
        self.assertEqual(run.isolation_mode, "")


class DoctorAnswersTheHealthQuestionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = ["--config", str(build_env(self.base, seed="health", use_subprocess_model=True))]

    def _doctor(self):
        text = run_cli([*self.cfg, "doctor"])
        as_json = run_cli([*self.cfg, "--json", "doctor"])
        self.assertEqual(text.returncode, as_json.returncode, text.stdout + as_json.stderr)
        line = next(ln for ln in text.stdout.splitlines() if ln.startswith("last run"))
        return text, json.loads(as_json.stdout), line

    def _receipts(self) -> list[Path]:
        return sorted((self.base / "state" / "receipts").glob("*.json"))

    def test_before_any_committed_run_there_is_no_last_run(self) -> None:
        text, payload, line = self._doctor()
        self.assertEqual(text.returncode, RC_OK, text.stdout)
        self.assertEqual(line, "last run     none recorded (no daily receipt yet)")
        self.assertIsNone(payload["last_run"])
        self.assertEqual(payload["receipt_records_unreadable"], [])
        self.assertFalse(
            (self.base / "state" / "receipts").exists(), "doctor created the directory"
        )
        proc = run_cli([*self.cfg, "daily", "--day", "2026-01-01"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        _, payload, line = self._doctor()
        self.assertIsNone(payload["last_run"], "a dry run withholds its receipt")
        self.assertTrue(line.endswith("none recorded (no daily receipt yet)"), line)

    def test_after_a_committed_run_the_line_answers_all_three_questions(self) -> None:
        proc = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        text, payload, line = self._doctor()
        self.assertEqual(text.returncode, RC_OK, text.stdout)
        self.assertRegex(
            line,
            r"^last run     2026-01-01 ok at \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \(0 day\(s\) ago\); "
            r"skill tree mutation=False; isolation=none$",
        )
        last = payload["last_run"]
        self.assertEqual(
            set(last),
            {
                "day",
                "status",
                "finished_at",
                "dry_run",
                "mutation_detected",
                "isolation_mode",
                "path",
                "age_days",
            },
        )
        self.assertEqual(
            (last["day"], last["status"], last["dry_run"]), ("2026-01-01", "ok", False)
        )
        self.assertIs(last["mutation_detected"], False)
        self.assertEqual(last["age_days"], 0)
        self.assertEqual(Path(last["path"]), self._receipts()[0])

    def test_an_old_receipt_a_mutation_and_an_unreadable_one_are_each_said(self) -> None:
        proc = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        (receipt,) = self._receipts()
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        three_days = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=3, hours=1
        )
        payload["finished_at"] = three_days.strftime("%Y-%m-%dT%H:%M:%SZ")
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        _, doc, line = self._doctor()
        self.assertIn("(3 day(s) ago)", line)
        self.assertEqual(doc["last_run"]["age_days"], 3)
        self.assertTrue(
            any(w.startswith("the last run finished 3 day(s) ago; ") for w in doc["warnings"]),
            doc["warnings"],
        )

        payload["skill_tree"]["mutation_detected"] = True
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        text, doc, line = self._doctor()
        self.assertEqual(text.returncode, RC_NEGATIVE, text.stdout)
        self.assertIn("skill tree mutation=True", line)
        self.assertTrue(
            any(
                p.startswith("the last run (2026-01-01) detected a skill tree mutation")
                for p in doc["problems"]
            ),
            doc["problems"],
        )
        self.assertIn("verdict: problems found", text.stdout)

        del payload["skill_tree"]
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        text, doc, line = self._doctor()
        self.assertEqual(text.returncode, RC_OK, text.stdout)
        self.assertIn("skill tree mutation=unknown", line)
        self.assertTrue(
            any("does not say whether the skill tree" in w for w in doc["warnings"]),
            doc["warnings"],
        )

        payload["skill_tree"] = {"mutation_detected": False}
        ahead = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=400)
        payload["finished_at"] = ahead.strftime("%Y-%m-%dT%H:%M:%SZ")
        payload["dry_run"] = True
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        text, doc, line = self._doctor()
        self.assertEqual(text.returncode, RC_OK, text.stdout)
        self.assertIn(" ok (dry run) at ", line)
        self.assertRegex(line, r"\((399|400) day\(s\) in the future\)")
        self.assertLess(doc["last_run"]["age_days"], 0)
        self.assertTrue(
            any("says it finished in the future" in w for w in doc["warnings"]), doc["warnings"]
        )

        payload["finished_at"] = "yesterday afternoon"
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        text, doc, line = self._doctor()
        self.assertEqual(text.returncode, RC_OK, text.stdout)
        self.assertIn("at yesterday afternoon (time does not parse);", line)
        self.assertIsNone(doc["last_run"]["age_days"])
        self.assertTrue(
            any(
                "carries a finish time that does not parse: 'yesterday afternoon'" in w
                for w in doc["warnings"]
            ),
            doc["warnings"],
        )

        payload["finished_at"] = three_days.strftime("%Y-%m-%dT%H:%M:%SZ")
        payload["dry_run"] = False
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        (receipt.parent / "torn.json").write_text("{", encoding="utf-8")
        text, doc, line = self._doctor()
        self.assertEqual(text.returncode, RC_OK, text.stdout)
        self.assertIn("2026-01-01", line, "the readable receipt still answers")
        self.assertEqual(
            [Path(u["path"]).name for u in doc["receipt_records_unreadable"]], ["torn.json"]
        )
        self.assertTrue(
            any(w.startswith("a receipt cannot be read: ") for w in doc["warnings"]),
            doc["warnings"],
        )
        self.assertTrue(
            any(
                re.match(r"^WARN  a receipt cannot be read: ", ln)
                for ln in text.stdout.splitlines()
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
