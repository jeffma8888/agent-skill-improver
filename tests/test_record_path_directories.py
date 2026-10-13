"""A directory where a record goes is refused by name, and doctor says it is there.

Every state record write ends in ``os.replace``, which swaps a link or a fifo
out of the way but cannot swap a directory out. Measured before: a directory
at one conversation record's path escaped as ``IsADirectoryError`` from the
middle of the pass (exit 70, the day's receipt never written) while ``doctor``
said ``verdict: ready``, since a directory is not a foreign entry and no
record read lists one. Now ``StateStore._persist`` refuses it by name before
the write, and ``doctor`` warns about every directory sitting where only
records are written.
"""

from __future__ import annotations

import datetime
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver import SCHEMA_RECEIPT
from agent_skill_improver.app.state import RECORD_SUBDIRS, SUBDIRS, StateStore, keyed_name
from agent_skill_improver.cli import RC_OK, RC_REFUSED
from agent_skill_improver.clock import FrozenClock, day_key
from agent_skill_improver.errors import SafetyRefusal

from tests.support import build_env, run_cli


def _clock() -> FrozenClock:
    return FrozenClock(datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc))


class ARecordWriteOverADirectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve() / "state"
        self.state = StateStore(self.root, clock=_clock())
        self.state.ensure()
        self.path = self.root / "conversations" / f"{keyed_name('conv-1')}.json"

    def test_a_directory_at_the_record_path_is_refused_by_name_and_left_as_found(self) -> None:
        self.path.mkdir(mode=0o700)
        (self.path / "keep.txt").write_text("not ours", encoding="utf-8")
        with self.assertRaises(SafetyRefusal) as caught:
            self.state.write_conversation("conv-1", {"schema": "x"})
        message = str(caught.exception)
        self.assertEqual(
            message,
            f"refusing to write the state record {self.path}: a directory sits at that path, "
            "which a write cannot replace; remove it if you did not put it there",
        )
        self.assertTrue(self.path.is_dir())
        self.assertEqual((self.path / "keep.txt").read_text(encoding="utf-8"), "not ours")
        self.assertEqual(
            [p.name for p in self.root.joinpath("conversations").iterdir()],
            [self.path.name],
            "no temporary file was left beside it",
        )

    def test_every_record_kind_is_refused_the_same_way(self) -> None:
        cases = {
            "scorecards": lambda: self.state.write_scorecard("conv-1", "tdd", {"schema": "x"}),
            "quarantine": lambda: self.state.quarantine("conv-1", stage="score", reason="r"),
            "refusals": lambda: self.state.refuse("tdd", stage="propose", reason="r"),
            "deferred": lambda: self.state.defer("tdd", stage="propose", reason="r"),
            "raw": lambda: self.state.write_raw("conv-1", "text"),
        }
        for name, write in cases.items():
            with self.subTest(name=name):
                try:
                    write()
                except TypeError as exc:  # pragma: no cover - a signature drift
                    self.fail(f"{name}: {exc}")
                (record,) = [p for p in self.root.joinpath(name).iterdir()]
                record.unlink()
                record.mkdir(mode=0o700)
                with self.assertRaises(SafetyRefusal) as caught:
                    write()
                self.assertIn(f"refusing to write the state record {record}", str(caught.exception))
                self.assertTrue(record.is_dir())
                record.rmdir()

    def test_the_apply_and_rollback_receipts_are_records_too(self) -> None:
        for action in ("apply", "rollback"):
            with self.subTest(action=action):
                path = self.state.action_receipt_path(action, "CAND-2026-01-02-01-tdd")
                self.assertEqual(path.parent, self.root / "receipts")
                path.mkdir(mode=0o700)
                with self.assertRaises(SafetyRefusal) as caught:
                    self.state.write_action_receipt(action, "CAND-2026-01-02-01-tdd", {"a": 1})
                self.assertIn(f"refusing to write the state record {path}", str(caught.exception))
                self.assertTrue(path.is_dir())
                path.rmdir()
                self.state.write_action_receipt(action, "CAND-2026-01-02-01-tdd", {"a": 1})
                self.assertTrue(path.is_file())

    def test_a_dry_run_withholds_the_write_and_does_not_ask(self) -> None:
        self.path.mkdir(mode=0o700)
        self.state.dry_run = True
        self.assertEqual(self.state.write_conversation("conv-1", {"schema": "x"}), self.path)
        self.assertEqual(self.state.withheld_writes, [f"conversations/{self.path.name}"])
        self.assertTrue(self.path.is_dir())

    def test_a_link_or_a_fifo_at_the_record_path_is_replaced_not_followed(self) -> None:
        # The documented contract for every other non-file: ``os.replace``
        # swaps the entry, and nothing is written through it.
        outside = self.root.parent / "outside.json"
        outside.write_text("untouched", encoding="utf-8")
        self.path.symlink_to(outside)
        self.state.write_conversation("conv-1", {"schema": "x"})
        self.assertFalse(self.path.is_symlink())
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"schema": "x"})
        self.assertEqual(outside.read_text(encoding="utf-8"), "untouched")
        self.path.unlink()
        os.mkfifo(self.path)
        self.state.write_conversation("conv-1", {"schema": "y"})
        self.assertTrue(self.path.is_file())
        # A link to a directory is still a link: swapped out, the directory intact.
        self.path.unlink()
        elsewhere = self.root.parent / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "keep").write_text("kept", encoding="utf-8")
        self.path.symlink_to(elsewhere)
        self.state.write_conversation("conv-1", {"schema": "z"})
        self.assertTrue(self.path.is_file() and not self.path.is_symlink())
        self.assertEqual((elsewhere / "keep").read_text(encoding="utf-8"), "kept")

    def test_a_directory_at_a_deferral_path_is_not_a_record_to_clear(self) -> None:
        path = self.root / "deferred" / f"{keyed_name('tdd')}.json"
        path.mkdir(mode=0o700)
        self.state.clear_deferred("tdd")
        self.assertTrue(path.is_dir(), "left alone, as every entry this tool did not write")


class MisplacedDirectoriesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve() / "state"
        self.state = StateStore(self.root, clock=_clock())
        self.state.ensure()

    def test_the_record_directories_are_every_subdir_but_backups_sandbox_and_locks(self) -> None:
        # backups keeps a directory per candidate, sandbox per-call scratch, and
        # locks the one lock leaf, which lock_entry_problem judges on its own.
        self.assertEqual(set(SUBDIRS) - set(RECORD_SUBDIRS), {"backups", "sandbox", "locks"})
        self.assertEqual(self.state.misplaced_directories(), [])
        (self.root / "locks" / "stray").mkdir()
        self.assertEqual(self.state.misplaced_directories(), [])

    @unittest.skipIf(os.geteuid() == 0, "root lists any directory")
    def test_a_record_directory_this_process_cannot_list_is_passed_over(self) -> None:
        (self.root / "candidates" / "a").mkdir()
        locked = self.root / "conversations"
        (locked / "b.json").mkdir()
        locked.chmod(0o000)
        self.addCleanup(locked.chmod, 0o700)
        self.assertEqual(self.state.misplaced_directories(), ["candidates/a"])

    def test_a_directory_under_a_record_directory_is_listed_and_others_are_not(self) -> None:
        (self.root / "conversations" / "b.json").mkdir()
        (self.root / "candidates" / "a").mkdir()
        (self.root / "backups" / "CAND-x--1").mkdir()
        (self.root / "sandbox" / "call-1").mkdir()
        (self.root / "conversations" / "c.json").write_text("{}", encoding="utf-8")
        (self.root / "refusals" / "link").symlink_to(self.root / "candidates" / "a")
        self.assertEqual(
            self.state.misplaced_directories(),
            ["conversations/b.json", "candidates/a"],
            "in layout order; a link to a directory is a foreign entry, not a misplaced directory",
        )
        self.assertIn("refusals/link is a symlink", self.state.foreign_entries())

    def test_nothing_is_listed_through_a_broken_layout_or_an_absent_root(self) -> None:
        (self.root / "conversations" / "b.json").mkdir()
        self.assertEqual(self.state.misplaced_directories(), ["conversations/b.json"])
        (self.root / "candidates").rmdir()
        (self.root / "candidates").symlink_to(self.root / "conversations")
        self.assertTrue(self.state.layout_problems())
        self.assertEqual(self.state.misplaced_directories(), [])
        self.assertEqual(
            StateStore(self.root / "nowhere", clock=_clock()).misplaced_directories(), []
        )


class ThroughTheCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.cfg = ["--config", str(build_env(self.base, seed="dirs", use_subprocess_model=True))]
        self.state_dir = self.base / "state"

    def _warnings(self, proc) -> list[str]:
        return [line for line in proc.stdout.splitlines() if line.startswith("WARN")]

    def _receipt_days(self) -> list[str]:
        out = []
        for path in sorted((self.state_dir / "receipts").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") == SCHEMA_RECEIPT:
                out.append(payload["day"])
        return out

    def test_doctor_warns_and_the_run_refuses_by_name_instead_of_exiting_seventy(self) -> None:
        first = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(first.returncode, RC_OK, first.stderr)
        records = sorted((self.state_dir / "conversations").iterdir())
        self.assertGreater(len(records), 1)
        victim = records[0]
        victim.unlink()
        victim.mkdir()

        doctor = run_cli([*self.cfg, "doctor"])
        self.assertEqual(doctor.returncode, RC_OK, "a warning, until a run needs that record")
        (warning,) = [w for w in self._warnings(doctor) if "only records are written" in w]
        self.assertIn("1 directory sits where only records are written", warning)
        self.assertIn(f"conversations/{victim.name}", warning)
        self.assertIn(
            "refuses by name rather than replacing it; remove it if you did not put it there",
            warning,
        )
        as_json = json.loads(run_cli([*self.cfg, "--json", "doctor"]).stdout)
        self.assertEqual(
            as_json["state_permissions"]["misplaced_directories"],
            [f"conversations/{victim.name}"],
        )
        self.assertEqual(as_json["state_permissions"]["left_alone"], [])

        refused = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-02"])
        self.assertEqual(refused.returncode, RC_REFUSED, refused.stderr)
        self.assertIn(f"SafetyRefusal: refusing to write the state record {victim}", refused.stderr)
        self.assertIn("a directory sits at that path", refused.stderr)
        self.assertNotIn("Traceback", refused.stderr)
        self.assertEqual(self._receipt_days(), ["2026-01-01"], "the refused run left no receipt")
        self.assertTrue(victim.is_dir(), "left as found")

        (self.state_dir / "candidates" / "another").mkdir()
        plural = run_cli([*self.cfg, "doctor"])
        (warning,) = [w for w in self._warnings(plural) if "only records are written" in w]
        self.assertIn("2 directories sit where only records are written", warning)
        self.assertIn("remove them if you did not put them there", warning)

    def test_todays_receipt_path_is_a_problem_once_not_a_problem_and_a_warning(self) -> None:
        first = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-01"])
        self.assertEqual(first.returncode, RC_OK, first.stderr)
        today = day_key(datetime.datetime.now(datetime.timezone.utc))
        state = StateStore(self.state_dir, clock=_clock())
        state.receipt_path(today, dry_run=False).mkdir()
        doctor = run_cli([*self.cfg, "--json", "doctor"])
        payload = json.loads(doctor.stdout)
        self.assertEqual(len(payload["problems"]), 1, payload["problems"])
        self.assertIn("today's receipt path", payload["problems"][0])
        self.assertEqual(
            [w for w in payload["warnings"] if "only records are written" in w],
            [],
            "named once, as the problem it is",
        )
        self.assertEqual(
            payload["state_permissions"]["misplaced_directories"],
            [f"receipts/{state.receipt_path(today, dry_run=False).name}"],
            "the JSON list is the scan, unfiltered",
        )


class ThePromotionReceiptPathTests(unittest.TestCase):
    """An apply or rollback whose receipt could not be written must not change the skill.

    Measured before: a directory at the apply receipt path let the apply
    complete, mark the candidate applied and then exit 70 from the receipt
    write, so the skill was changed and no receipt said so.
    """

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
                    seed="receipt-path",
                )
            ),
        ]
        daily = run_cli([*cls.cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
        assert daily.returncode == RC_OK, daily.stderr
        listed = json.loads(run_cli([*cls.cfg, "--json", "candidates"]).stdout)["candidates"]
        cls.candidate_id = sorted(c["id"] for c in listed)[0]
        review = json.loads(run_cli([*cls.cfg, "--json", "review", cls.candidate_id]).stdout)
        cls.token = review["approval_token"]
        cls.state = StateStore(cls.base / "state", clock=_clock())

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _skill_text(self) -> str:
        record = json.loads(self.state.candidate_path(self.candidate_id).read_text("utf-8"))
        return Path(record["target"]["canonical_path"]).read_text(encoding="utf-8")

    def test_apply_then_rollback_each_refuse_before_the_skill_write(self) -> None:
        before = self._skill_text()
        receipt = self.state.action_receipt_path("apply", self.candidate_id)
        receipt.mkdir()
        refused = run_cli(
            [*self.cfg, "apply", self.candidate_id, "--approve", self.token, "--i-reviewed-this"]
        )
        self.assertEqual(refused.returncode, RC_REFUSED, refused.stderr)
        self.assertIn(
            f"SafetyRefusal: refusing to apply {self.candidate_id}: the apply receipt path "
            f"{receipt} is a directory, where the receipt of this apply would be written; "
            "nothing was written.",
            refused.stderr,
        )
        self.assertNotIn("Traceback", refused.stderr)
        self.assertEqual(self._skill_text(), before, "the skill was not touched")
        self.assertEqual(list((self.base / "state" / "backups").iterdir()), [], "no backup made")
        record = json.loads(self.state.candidate_path(self.candidate_id).read_text("utf-8"))
        self.assertEqual(record["status"], "validated")
        receipt.rmdir()

        applied = run_cli(
            [*self.cfg, "apply", self.candidate_id, "--approve", self.token, "--i-reviewed-this"]
        )
        self.assertEqual(applied.returncode, RC_OK, applied.stderr)
        self.assertTrue(receipt.is_file(), "the receipt of the apply is where it belongs")
        changed = self._skill_text()
        self.assertNotEqual(changed, before)

        rollback_receipt = self.state.action_receipt_path("rollback", self.candidate_id)
        os.mkfifo(rollback_receipt)
        refused = run_cli([*self.cfg, "rollback", self.candidate_id])
        self.assertEqual(refused.returncode, RC_REFUSED, refused.stderr)
        self.assertIn(
            f"refusing to rollback {self.candidate_id}: the rollback receipt path "
            f"{rollback_receipt} is not a regular file",
            refused.stderr,
        )
        self.assertEqual(self._skill_text(), changed, "the rollback did not happen")
        rollback_receipt.unlink()
        rolled = run_cli([*self.cfg, "rollback", self.candidate_id])
        self.assertEqual(rolled.returncode, RC_OK, rolled.stderr)
        self.assertEqual(self._skill_text(), before)
        self.assertTrue(rollback_receipt.is_file())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
