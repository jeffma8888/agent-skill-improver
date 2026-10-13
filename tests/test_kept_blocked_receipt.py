"""The run that says a mutation was reviewed must not destroy the record of it.

The receipt path is per day. A run blocked in the morning is reviewed, and the
operator re-runs the same day with ``--commit --mutation-reviewed`` naming it:
that run's receipt lands on the blocked one. Measured before: it overwrote it,
and ``skill_tree.changes``, the only record of what wrote to the tree, was gone,
destroyed by the run that said it was reviewed. Now whatever committed run comes
to write at a path holding a receipt that recorded a mutation keeps that receipt
under another name first, inside the lock and before anything else is written;
the new receipt says where, nothing already at that name is replaced, a kept
receipt is never read as a run again (so it cannot re-block the schedule), and
``doctor`` counts them.
"""

from __future__ import annotations

import datetime
import json
import unittest
from pathlib import Path
from unittest import mock
from tempfile import TemporaryDirectory

from agent_skill_improver import SCHEMA_RECEIPT
from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app.daily import run_daily
from agent_skill_improver.app.lock import RunLock
from agent_skill_improver.app.state import StateStore, is_kept_receipt_name, keyed_name
from agent_skill_improver.cli import RC_OK
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.config import load_config
from agent_skill_improver.errors import BlockedRunUnreviewed, SafetyRefusal, StateUnreadable

from tests.support import InProcessRunner, build_env, run_cli

DAY = "2026-01-01"
FINISHED = f"{DAY}T09:00:30Z"
CHANGES = [{"key": "canonical:tdd/SKILL.md", "kind": "content-changed"}]


def _clock() -> FrozenClock:
    return FrozenClock(datetime.datetime(2026, 1, 1, 15, 4, 5, tzinfo=datetime.timezone.utc))


def _blocked_payload(day: str = DAY) -> dict:
    return {
        "schema": SCHEMA_RECEIPT,
        "day": day,
        "status": "blocked",
        "started_at": f"{day}T09:00:00Z",
        "finished_at": FINISHED,
        "dry_run": False,
        "skill_tree": {"mutation_detected": True, "changes": CHANGES},
        "isolation": {"mode": "none"},
    }


def _kept_name(day: str = DAY, suffix: str = "") -> str:
    return f"_blocked--{keyed_name(day)}--{FINISHED.replace(':', '_')}{suffix}.json"


class _DiesOnFirstRead:
    """A skill store whose first tree read fails: the run dies after the keep."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def tree_entries(self):
        raise RuntimeError("the tree went away")

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class TheSameDayReviewedRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config = load_config(
            build_env(
                self.base,
                dry_run=False,
                rubric_ids=("instruction-compliance",),
                use_subprocess_model=False,
            )
        )
        self.state = StateStore(self.config.state_dir, clock=_clock())
        self.state.ensure()
        self.store = FilesystemSkillStore(
            self.config.skill_roots, read_only=True, protected_paths=self.config.protected_paths
        )
        self.receipts = self.config.state_dir / "receipts"

    def _plant(self, path: Path, day: str = DAY) -> bytes:
        raw = json.dumps(_blocked_payload(day)).encode("utf-8")
        path.write_bytes(raw)
        path.chmod(0o600)
        return raw

    def _run(self, *, day: str = DAY, commit: bool = True, reviewed: str | None = DAY, store=None):
        return run_daily(
            self.config,
            source=FixtureConversationSource(self.config.source.fixtures_dir),
            runner=InProcessRunner(),
            skill_store=store or self.store,
            state=self.state,
            clock=_clock(),
            day=day,
            commit=commit,
            mutation_reviewed=reviewed,
        )

    def _kept_notes(self, receipt) -> list[str]:
        return [n for n in receipt.notes if n.startswith("the blocked receipt of the")]

    def test_the_blocked_receipt_is_kept_byte_for_byte_and_the_new_receipt_says_where(self) -> None:
        day_path = self.state.receipt_path(DAY, dry_run=False)
        raw = self._plant(day_path)
        receipt = self._run()
        self.assertEqual(receipt.status, "ok", receipt.notes)
        kept = self.receipts / _kept_name()
        self.assertEqual(kept.read_bytes(), raw, "the blocked receipt is the same bytes, elsewhere")
        written = json.loads(day_path.read_text(encoding="utf-8"))
        self.assertEqual(written["status"], "ok", "the day's path now holds the reviewed run")
        self.assertFalse(written["skill_tree"]["mutation_detected"])
        (note,) = self._kept_notes(receipt)
        self.assertIn(f"of the {DAY} run", note)
        self.assertIn(str(kept), note, "the receipt names where the blocked one went")
        self.assertIn("read skill_tree.changes there", note)
        self.assertEqual(
            sorted(p.name for p in self.receipts.glob("*.json")), sorted([day_path.name, kept.name])
        )
        # The reviewed run, not the kept receipt, is the newest: the morning
        # after runs without the flag.
        newest = self.state.scan_receipts().last_run
        assert newest is not None
        self.assertEqual(newest.path, day_path)
        self.assertFalse(newest.mutation_detected)
        after = self._run(day="2026-01-02", reviewed=None)
        self.assertEqual(after.status, "ok")
        self.assertEqual(self._kept_notes(after), [])

    def test_a_blocked_receipt_the_run_would_not_overwrite_is_left_where_it_is(self) -> None:
        # Another day's receipt: the reviewed run writes its own day's path.
        other_path = self.state.receipt_path(DAY, dry_run=False)
        raw = self._plant(other_path)
        receipt = self._run(day="2026-01-02")
        self.assertEqual(receipt.status, "ok", receipt.notes)
        self.assertEqual(other_path.read_bytes(), raw)
        self.assertEqual(self._kept_notes(receipt), [])
        self.assertEqual([p.name for p in self.receipts.glob("_blocked--*")], [])
        # The same day under a name that is not the day's receipt path (moved
        # by hand): nothing lands on it, so it is not moved again.
        for p in self.receipts.glob("*.json"):
            p.unlink()
        aside = self.receipts / f"{DAY}.json"
        raw = self._plant(aside)
        receipt = self._run()
        self.assertEqual(receipt.status, "ok", receipt.notes)
        self.assertEqual(aside.read_bytes(), raw)
        self.assertEqual(self._kept_notes(receipt), [])
        self.assertEqual([p.name for p in self.receipts.glob("_blocked--*")], [])

    def test_an_older_blocked_receipt_at_the_run_day_is_kept_whichever_run_is_newest(self) -> None:
        # Audit finding: keyed on the newest receipt, a plain re-run of an older
        # blocked day (its block long cleared by a newer run) destroyed it.
        day_path = self.state.receipt_path(DAY, dry_run=False)
        raw = self._plant(day_path)
        later = _blocked_payload("2026-01-02")
        later["status"] = "ok"
        later["finished_at"] = "2026-01-02T09:00:30Z"
        later["skill_tree"] = {"mutation_detected": False, "changes": []}
        later_path = self.state.receipt_path("2026-01-02", dry_run=False)
        later_path.write_text(json.dumps(later), encoding="utf-8")
        later_path.chmod(0o600)
        assert self.state.scan_receipts().last_run.path == later_path
        receipt = self._run(reviewed=None)
        self.assertEqual(receipt.status, "ok", receipt.notes)
        self.assertEqual((self.receipts / _kept_name()).read_bytes(), raw)
        self.assertFalse(json.loads(day_path.read_text())["skill_tree"]["mutation_detected"])
        (note,) = self._kept_notes(receipt)
        self.assertIn(f"of the {DAY} run", note)

    def test_the_same_entry_under_another_spelling_is_kept_not_clobbered(self) -> None:
        # Audit finding: on a case-insensitive filesystem the blocked receipt
        # renamed with upper-case hex was the same directory entry the run
        # wrote to, and string equality with the newest receipt's path missed it.
        day_path = self.state.receipt_path(DAY, dry_run=False)
        stem, _, hexpart = day_path.stem.rpartition("--")
        spelled = day_path.with_name(f"{stem}--{hexpart.upper()}.json")
        raw = self._plant(spelled)
        if not day_path.exists():
            self.skipTest("a case-sensitive filesystem: the spelling is another entry")
        receipt = self._run()
        self.assertEqual(receipt.status, "ok", receipt.notes)
        self.assertEqual((self.receipts / _kept_name()).read_bytes(), raw)
        names = sorted(p.name for p in self.receipts.glob("*.json"))
        self.assertEqual(names, sorted([day_path.name, _kept_name()]))
        self.assertFalse(json.loads(day_path.read_text())["skill_tree"]["mutation_detected"])

    def test_a_kept_receipt_is_never_read_as_a_run_again(self) -> None:
        # Audit finding: a blocked receipt stamped ahead of the clock (skew, a
        # hand edit) stayed the newest after the keep and blocked every later
        # run; so did one whose day sorted after the kept name on a full tie.
        day_path = self.state.receipt_path(DAY, dry_run=False)
        payload = _blocked_payload()
        payload["finished_at"] = f"{DAY}T20:00:00Z"
        day_path.write_text(json.dumps(payload), encoding="utf-8")
        day_path.chmod(0o600)
        receipt = self._run()
        self.assertEqual(receipt.status, "ok", receipt.notes)
        scan = self.state.scan_receipts()
        (kept,) = scan.kept
        self.assertTrue(is_kept_receipt_name(kept.name))
        self.assertEqual(json.loads(kept.read_text())["finished_at"], f"{DAY}T20:00:00Z")
        assert scan.last_run is not None
        self.assertEqual(scan.last_run.path, day_path, "the kept receipt is not the newest run")
        after = self._run(day="2026-01-02", reviewed=None)
        self.assertEqual(after.status, "ok", "the kept receipt does not block the next run")
        # A hand-made file under the kept name is an archive by that name too.
        hand = self.receipts / "_blocked--by-hand.json"
        hand.write_text(json.dumps(_blocked_payload("2026-01-03")), encoding="utf-8")
        scan = self.state.scan_receipts()
        self.assertEqual(sorted(scan.kept), sorted([kept, hand]))
        self.assertEqual(scan.unreadable, ())
        self.assertEqual(scan.last_run.day, "2026-01-02")

    def test_a_day_named_blocked_still_blocks_its_receipt_is_not_a_kept_one(self) -> None:
        # ``--day blocked`` is a valid run day and keys to ``blocked--HEX``; a
        # kept name that began with the bare word would swallow its receipt.
        for day in ("blocked", "blocked--x", "_blocked"):
            self.assertFalse(is_kept_receipt_name(f"{keyed_name(day)}.json"), day)
        self.assertTrue(is_kept_receipt_name(_kept_name()))
        day_path = self.state.receipt_path("blocked", dry_run=False)
        self._plant(day_path, day="blocked")
        scan = self.state.scan_receipts()
        self.assertEqual(scan.kept, ())
        assert scan.last_run is not None
        self.assertEqual((scan.last_run.day, scan.last_run.mutation_detected), ("blocked", True))
        with self.assertRaises(BlockedRunUnreviewed):
            self._run(day="2026-01-02", reviewed=None)

    def test_a_run_that_dies_after_the_keep_leaves_exactly_one_copy(self) -> None:
        # Audit finding: a copy instead of a move survived every test; a copy
        # would leave the evidence at the day path to be overwritten anyway.
        day_path = self.state.receipt_path(DAY, dry_run=False)
        raw = self._plant(day_path)
        with self.assertRaises(RuntimeError):
            self._run(store=_DiesOnFirstRead(self.store))
        self.assertFalse(day_path.exists(), "the day path no longer holds the blocked receipt")
        holders = [p for p in self.receipts.glob("*.json") if p.read_bytes() == raw]
        self.assertEqual(holders, [self.receipts / _kept_name()])
        # The run that died had passed the gate on the operator's statement, so
        # the next plain run of the day proceeds and writes the day's receipt.
        receipt = self._run(reviewed=None)
        self.assertEqual(receipt.status, "ok", receipt.notes)
        self.assertTrue(day_path.exists())
        self.assertEqual(self._kept_notes(receipt), [])

    def test_the_store_reads_what_sits_at_the_days_path_and_refuses_what_does_not_read(
        self,
    ) -> None:
        self.assertIsNone(self.state.committed_receipt_at(DAY), "nothing there")
        day_path = self.state.receipt_path(DAY, dry_run=False)
        day_path.write_text(json.dumps({"schema": "asi.other.v1"}), encoding="utf-8")
        self.assertIsNone(self.state.committed_receipt_at(DAY), "another schema is not a run")
        self._plant(day_path)
        found = self.state.committed_receipt_at(DAY)
        assert found is not None
        self.assertEqual((found.day, found.mutation_detected, found.path), (DAY, True, day_path))
        day_path.write_text("{ torn", encoding="utf-8")
        with self.assertRaises(StateUnreadable) as caught:
            self.state.committed_receipt_at(DAY)
        self.assertIn(f"the receipt {day_path} cannot be read", str(caught.exception))
        self.assertIn("this run would write there", str(caught.exception))
        self.assertIn("move it out of the receipts directory", str(caught.exception))

    def test_an_ordinary_same_day_re_run_still_overwrites_and_keeps_nothing(self) -> None:
        # Only a blocked receipt is evidence; the day's receipt otherwise stays
        # per day, as it always was, and nothing is kept.
        day_path = self.state.receipt_path(DAY, dry_run=False)
        payload = _blocked_payload()
        payload["status"] = "ok"
        payload["skill_tree"] = {"mutation_detected": False, "changes": []}
        raw = json.dumps(payload).encode("utf-8")
        day_path.write_bytes(raw)
        day_path.chmod(0o600)
        receipt = self._run(reviewed=None)
        self.assertEqual(receipt.status, "ok", receipt.notes)
        self.assertNotEqual(day_path.read_bytes(), raw, "the day's receipt was overwritten")
        self.assertEqual(self._kept_notes(receipt), [])
        self.assertEqual(sorted(p.name for p in self.receipts.glob("*.json")), [day_path.name])

    def test_a_dry_run_of_the_same_day_moves_nothing(self) -> None:
        day_path = self.state.receipt_path(DAY, dry_run=False)
        raw = self._plant(day_path)
        receipt = self._run(commit=False)
        self.assertEqual(receipt.status, "ok", receipt.notes)
        self.assertEqual(day_path.read_bytes(), raw, "a dry run changes no content under state/")
        self.assertEqual(self._kept_notes(receipt), [])
        self.assertEqual(sorted(p.name for p in self.receipts.glob("*.json")), [day_path.name])

    def test_a_kept_name_already_taken_is_not_replaced(self) -> None:
        day_path = self.state.receipt_path(DAY, dry_run=False)
        raw = self._plant(day_path)
        first = self.receipts / _kept_name()
        # Readable, so the fail-closed gate does not refuse the run over it, and
        # not a daily receipt, so it is passed over.
        earlier = json.dumps({"schema": "asi.test.earlier.v1", "note": "an earlier kept receipt"})
        first.write_text(earlier, encoding="utf-8")
        second = self.receipts / _kept_name(suffix="-2")
        second.mkdir(mode=0o700)
        receipt = self._run()
        self.assertEqual(receipt.status, "ok", receipt.notes)
        third = self.receipts / _kept_name(suffix="-3")
        self.assertEqual(third.read_bytes(), raw)
        self.assertEqual(first.read_text(encoding="utf-8"), earlier)
        self.assertTrue(second.is_dir())
        (note,) = self._kept_notes(receipt)
        self.assertIn(str(third), note)

    def test_the_store_refuses_to_move_a_receipt_while_withholding_writes(self) -> None:
        day_path = self.state.receipt_path(DAY, dry_run=False)
        raw = self._plant(day_path)
        withholding = StateStore(self.config.state_dir, clock=_clock(), dry_run=True)
        run = withholding.scan_receipts().last_run
        assert run is not None and run.path == day_path
        with self.assertRaises(SafetyRefusal) as caught:
            withholding.keep_blocked_receipt(run)
        self.assertIn("withholds writes (dry run)", str(caught.exception))
        self.assertEqual(day_path.read_bytes(), raw)
        self.assertEqual(sorted(p.name for p in self.receipts.glob("*.json")), [day_path.name])

    def test_a_move_the_tree_refuses_is_a_named_refusal_and_the_receipt_stays(self) -> None:
        day_path = self.state.receipt_path(DAY, dry_run=False)
        raw = self._plant(day_path)
        before = sorted(p.name for p in self.receipts.glob("*.json"))
        with mock.patch("agent_skill_improver.app.state.os.replace") as replace:
            replace.side_effect = PermissionError(13, "Permission denied")
            with self.assertRaises(SafetyRefusal) as caught:
                self._run()
        message = str(caught.exception)
        self.assertIn(f"the blocked receipt {day_path} could not be kept at", message)
        self.assertIn("Permission denied", message)
        self.assertIn("nothing was written", message)
        self.assertEqual(day_path.read_bytes(), raw)
        self.assertEqual(sorted(p.name for p in self.receipts.glob("*.json")), before)
        # The lock leaf stays on disk by design; released means the next run takes it.
        with RunLock(self.state.lock_path()):
            pass

    def test_a_receipt_without_a_finish_time_is_kept_as_unfinished(self) -> None:
        day_path = self.state.receipt_path(DAY, dry_run=False)
        payload = _blocked_payload()
        payload["finished_at"] = ""
        payload["started_at"] = ""
        day_path.write_text(json.dumps(payload), encoding="utf-8")
        day_path.chmod(0o600)
        run = self.state.scan_receipts().last_run
        assert run is not None
        self.assertEqual(
            self.state.kept_blocked_receipt_path(run).name,
            f"_blocked--{keyed_name(DAY)}--unfinished.json",
        )


class ThroughTheCommandTests(unittest.TestCase):
    def test_the_reviewed_run_of_the_blocked_day_keeps_its_receipt_and_doctor_counts_it(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            cfg = ["--config", str(build_env(base, seed="kept", use_subprocess_model=True))]
            proc = run_cli([*cfg, "daily", "--commit", "--day", DAY])
            self.assertEqual(proc.returncode, RC_OK, proc.stderr)
            receipts = base / "state" / "receipts"
            (day_path,) = receipts.glob("*.json")
            payload = json.loads(day_path.read_text(encoding="utf-8"))
            payload["status"] = "blocked"
            payload["skill_tree"]["mutation_detected"] = True
            payload["skill_tree"]["changes"] = CHANGES
            raw = json.dumps(payload).encode("utf-8")
            day_path.write_bytes(raw)

            reviewed = run_cli(
                [*cfg, "daily", "--commit", "--day", DAY, "--mutation-reviewed", DAY]
            )
            self.assertEqual(reviewed.returncode, RC_OK, reviewed.stderr)
            kept = [p for p in receipts.glob("*.json") if p.name.startswith("_blocked--")]
            self.assertEqual(len(kept), 1, sorted(p.name for p in receipts.glob("*.json")))
            self.assertEqual(kept[0].read_bytes(), raw)
            self.assertIn(str(kept[0]), reviewed.stdout, "the command's report names the kept path")
            # The day's records were already ingested by the morning run, so the
            # re-run stands down; what matters is that it is not the blocked one.
            now = json.loads(day_path.read_text(encoding="utf-8"))
            self.assertNotEqual(now["status"], "blocked")
            self.assertFalse(now["skill_tree"]["mutation_detected"])
            self.assertTrue(any(str(kept[0]) in n for n in now["notes"]))
            doctor = run_cli([*cfg, "doctor"])
            self.assertEqual(doctor.returncode, RC_OK, doctor.stdout + doctor.stderr)
            self.assertIn("WARN  1 kept blocked receipt(s) under", doctor.stdout)
            self.assertIn(str(kept[0]), doctor.stdout)
            self.assertIn("read skill_tree.changes in each", doctor.stdout)
            as_json = json.loads(run_cli([*cfg, "doctor", "--json"]).stdout)
            self.assertEqual(as_json["kept_blocked_receipts"], [str(kept[0])])
            self.assertEqual(as_json["last_run"]["path"], str(day_path))
            self.assertFalse(as_json["last_run"]["mutation_detected"])
            # Without the keep, nothing is said.
            kept[0].unlink()
            quiet = run_cli([*cfg, "doctor"])
            self.assertEqual(quiet.returncode, RC_OK)
            self.assertNotIn("kept blocked receipt", quiet.stdout)
            self.assertEqual(
                json.loads(run_cli([*cfg, "doctor", "--json"]).stdout)["kept_blocked_receipts"], []
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
