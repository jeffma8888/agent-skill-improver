"""What a run set aside can be read back: ``asi set-aside``.

Measured before: ``daily`` printed ``quarantined=3`` (and, since the count-is-
not-a-reason change, named that run's records), but a record set aside by an
earlier run could only be found by opening the files under the quarantine,
refusals and deferred directories, whose names flatten the stage. The listing
reads them all back, newest first, names the ones that do not read, and writes
nothing.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.app.state import StateStore
from agent_skill_improver.cli import RC_OK
from agent_skill_improver.clock import FrozenClock

from tests.support import build_env, run_cli


def _clock() -> FrozenClock:
    return FrozenClock(datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc))


class TheStoreReadsBackWhatItSetAsideTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve() / "state"
        self.clock = _clock()
        self.store = StateStore(self.root, clock=self.clock)
        self.store.ensure()

    def test_every_kind_is_read_back_newest_first_with_its_fields(self) -> None:
        self.store.quarantine("conv-1", stage="score:c1", reason="schema mismatch", detail={"x": 1})
        self.clock.advance(60)
        self.store.defer("tdd", stage="propose", reason="timed out")
        self.clock.advance(60)
        self.store.defer("tdd", stage="propose", reason="timed out again")
        self.clock.advance(60)
        self.store.refuse("conv-2", stage="ingest", reason="outside the window")
        scan = self.store.scan_set_aside()
        self.assertEqual(scan.unreadable, ())
        self.assertEqual(
            [(r.kind, r.record_id, r.stage, r.reason, r.attempts) for r in scan.readable],
            [
                ("refused", "conv-2", "ingest", "outside the window", None),
                ("deferred", "tdd", "propose", "timed out again", 2),
                ("quarantined", "conv-1", "score:c1", "schema mismatch", None),
            ],
        )
        self.assertEqual(
            [r.at for r in scan.readable], sorted((r.at for r in scan.readable), reverse=True)
        )
        self.assertTrue(all(r.path.is_file() for r in scan.readable))

    def test_a_record_that_does_not_read_is_named_and_a_link_is_skipped(self) -> None:
        self.store.quarantine("conv-1", stage="score", reason="r")
        junk = self.root / "quarantine" / "half-written.json"
        junk.write_text('{"schema": "asi.quarantine.v1", "record_id": ', encoding="utf-8")
        shapeless = self.root / "refusals" / "shapeless.json"
        shapeless.write_text(
            json.dumps({"schema": "asi.refusal.v1", "reason": "no id"}), encoding="utf-8"
        )
        nested = self.root / "quarantine" / "nested-reason.json"
        nested.write_text(
            json.dumps({"record_id": "conv-9", "stage": "score", "reason": {"nested": 1}}),
            encoding="utf-8",
        )
        odd = self.root / "deferred" / "odd-attempts.json"
        odd.write_text(
            json.dumps({"record_id": "conv-8", "stage": "score", "reason": "r", "attempts": True}),
            encoding="utf-8",
        )
        link = self.root / "deferred" / "planted.json"
        os.symlink(junk, link)
        scan = self.store.scan_set_aside()
        self.assertEqual(sorted(r.record_id for r in scan.readable), ["conv-1", "conv-8"])
        odd_record = next(r for r in scan.readable if r.record_id == "conv-8")
        self.assertIsNone(odd_record.attempts, "a boolean is not an attempt count")
        self.assertEqual(
            sorted(u.path.name for u in scan.unreadable),
            ["half-written.json", "nested-reason.json", "shapeless.json"],
        )
        reasons = {u.path.name: u.reason for u in scan.unreadable}
        self.assertIn("field 'record_id' is missing", reasons["shapeless.json"])
        self.assertIn("field 'reason' is not a string", reasons["nested-reason.json"])
        self.assertTrue(link.is_symlink(), "the planted link is left alone")

    @unittest.skipIf(os.geteuid() == 0, "root opens anything")
    def test_a_record_that_cannot_be_opened_is_listed_not_raised(self) -> None:
        self.store.quarantine("conv-1", stage="score", reason="r")
        self.store.defer("conv-2", stage="ingest", reason="later")
        self.store.defer("conv-2", stage="ingest", reason="later")
        locked = self.store.quarantine("conv-3", stage="score", reason="r")
        locked.chmod(0)
        self.addCleanup(locked.chmod, 0o600)
        deferral = self.store.defer("conv-4", stage="ingest", reason="later")
        deferral.chmod(0)
        self.addCleanup(lambda: deferral.exists() and deferral.chmod(0o600))
        scan = self.store.scan_set_aside()
        self.assertEqual(sorted(r.record_id for r in scan.readable), ["conv-1", "conv-2"])
        self.assertEqual(
            sorted(u.path.name for u in scan.unreadable), sorted([locked.name, deferral.name])
        )
        for u in scan.unreadable:
            self.assertTrue(u.reason.startswith("cannot be opened: "), u.reason)
        # The daily pass reads deferrals too: one it cannot open is no streak,
        # not a crash; a fresh deferral replaces it and the streak restarts.
        self.assertEqual(self.store.deferred_ids(), ["conv-2"])
        self.assertEqual(self.store.deferred_attempts("conv-4"), 0)
        self.assertEqual(self.store.deferred_attempts("conv-2"), 2)
        replaced = self.store.defer("conv-4", stage="ingest", reason="later")
        self.assertEqual(replaced, deferral)
        self.assertEqual(self.store.deferred_attempts("conv-4"), 1)
        self.assertEqual(sorted(self.store.deferred_ids()), ["conv-2", "conv-4"])

    def test_an_empty_tree_is_an_empty_scan(self) -> None:
        scan = self.store.scan_set_aside()
        self.assertEqual(scan, type(scan)(readable=(), unreadable=()))


class TheListingSaysWhatTheRunsSetAsideTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.config_path = Path(build_env(self.base, seed="set-aside", use_subprocess_model=True))
        self.cfg = ["--config", str(self.config_path)]

    def _replace_role(self, role: str, program: str) -> None:
        text = self.config_path.read_text(encoding="utf-8")
        pattern = re.compile(rf"(\[model\.{role}\]\n)argv = .*\n")
        self.assertEqual(len(pattern.findall(text)), 1, text)
        argv = f'argv = ["{sys.executable}", "-c", {json.dumps(program)}]\n'
        self.config_path.write_text(pattern.sub(lambda m: m.group(1) + argv, text, count=1))

    def _restore_roles(self) -> None:
        fresh = Path(build_env(self.base, seed="set-aside", use_subprocess_model=True))
        self.assertEqual(fresh, self.config_path)

    def test_before_any_run_the_listing_is_empty_and_writes_nothing(self) -> None:
        proc = run_cli([*self.cfg, "set-aside"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertEqual(
            proc.stdout.splitlines(),
            [
                "0 set-aside record(s) (0 quarantined, 0 refused, 0 deferred); 0 unreadable record(s)"
            ],
        )
        files = [p for p in (self.base / "state").rglob("*") if not p.is_dir()]
        self.assertEqual(files, [], "a listing wrote a file")

    def test_records_from_earlier_runs_are_listed_with_stage_reason_and_attempt(self) -> None:
        self._replace_role("improver", "import sys; sys.stderr.write('bbb\\n'); sys.exit(7)")
        for day in ("2026-01-01", "2026-01-02"):
            proc = run_cli([*self.cfg, "daily", "--commit", "--day", day])
            self.assertEqual(proc.returncode, RC_OK, proc.stderr)
            self.assertIn("skills_deferred=1", proc.stdout.splitlines()[0])
        self._restore_roles()
        self._replace_role("scorer", 'import sys; sys.stdout.write(\'{"schema": "wrong.v9"}\')')
        proc = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-03"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        self.assertIn("quarantined=5", proc.stdout.splitlines()[0])

        proc = run_cli([*self.cfg, "set-aside"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertEqual(
            lines[-1],
            "6 set-aside record(s) (5 quarantined, 0 refused, 1 deferred); 0 unreadable record(s)",
        )
        body = lines[:-1]
        self.assertEqual(len(body), 6)
        self.assertEqual([line.split()[1] for line in body], ["quarantined"] * 5 + ["deferred"])
        self.assertTrue(all(" at score:" in line and "schema" in line for line in body[:5]), body)
        self.assertRegex(
            body[5],
            r"^\S+ +deferred +'tdd' at propose \(attempt 2\): role 'improver' exited 7: bbb$",
        )

        proc = run_cli([*self.cfg, "set-aside", "--kind", "deferred"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(
            lines[-1],
            "1 deferred record(s), 5 of other kinds not listed "
            "(5 quarantined, 0 refused, 1 deferred); 0 unreadable record(s)",
        )

        proc = run_cli([*self.cfg, "--json", "set-aside", "--kind", "quarantined"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["listing"], "quarantined")
        self.assertEqual(payload["not_listed"], 1)
        self.assertEqual(payload["counts"], {"quarantined": 5, "refused": 0, "deferred": 1})
        self.assertEqual(len(payload["records"]), 5)
        record = payload["records"][0]
        self.assertEqual(
            set(record), {"kind", "record_id", "stage", "reason", "at", "attempts", "path"}
        )
        self.assertEqual(record["kind"], "quarantined")
        self.assertIsNone(record["attempts"])
        self.assertTrue(Path(record["path"]).is_file())

    def test_a_record_that_does_not_read_is_named_in_both_forms(self) -> None:
        run_cli([*self.cfg, "set-aside"])  # creates the directories it reads
        junk = self.base / "state" / "quarantine" / "half-written.json"
        junk.write_text("{", encoding="utf-8")
        proc = run_cli([*self.cfg, "set-aside"])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("UNREADABLE  half-written.json  "), lines[0])
        self.assertTrue(lines[-1].endswith("; 1 unreadable record(s)"), lines[-1])
        payload = json.loads(run_cli([*self.cfg, "--json", "set-aside"]).stdout)
        self.assertEqual(
            [Path(u["path"]).name for u in payload["unreadable"]], ["half-written.json"]
        )
        if os.geteuid() != 0:
            # Measured before: exit 70 with a PermissionError traceback.
            junk.chmod(0)
            self.addCleanup(junk.chmod, 0o600)
            proc = run_cli([*self.cfg, "set-aside"])
            self.assertEqual(proc.returncode, RC_OK, proc.stderr)
            self.assertIn("cannot be opened: ", proc.stdout)
            self.assertTrue(proc.stdout.startswith("UNREADABLE  half-written.json  "), proc.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
