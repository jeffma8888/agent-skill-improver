"""A blocked run blocks the next one until somebody says they looked.

The operations doc says of a run that detected a skill tree mutation: do not
re-run, find out what wrote to the tree. A scheduler that fires every morning
re-runs regardless. Measured before: the morning after a ``blocked`` receipt,
``daily`` started, took the lock and read the source as if nothing had
happened, and exit 5 was seen exactly once. Now the next run refuses before the
lock, before any write and before the source is touched, with the same exit
code, until the operator names the blocked run's day with
``--mutation-reviewed``; the run that does so records the statement.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver import SCHEMA_RECEIPT
from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app.daily import run_daily
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.cli import RC_BLOCKED, RC_NEGATIVE, RC_OK
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.config import load_config
from agent_skill_improver.errors import BlockedRunUnreviewed, ConfigError, StateUnreadable

from tests.support import InProcessRunner, build_env, run_cli

BLOCKED_DAY = "2026-01-01"


def _clock() -> FrozenClock:
    return FrozenClock(datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc))


def _blocked_receipt(day: str = BLOCKED_DAY) -> dict:
    return {
        "schema": SCHEMA_RECEIPT,
        "day": day,
        "status": "blocked",
        "started_at": f"{day}T09:00:00Z",
        "finished_at": f"{day}T09:00:30Z",
        "dry_run": False,
        "skill_tree": {
            "mutation_detected": True,
            "changes": [{"kind": "content-changed", "document": "canonical:tdd/SKILL.md"}],
        },
        "isolation": {"mode": "none"},
    }


def _tree_fingerprint(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_symlink():
            out[rel] = "link"
        elif path.is_dir():
            out[rel] = "dir"
        else:
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


class _Untouched:
    """A source the refused run must never reach; any use is the failure."""

    def __getattr__(self, name: str):
        raise AssertionError(f"the source was touched ({name}) by a run that should have refused")


class TheRunAfterABlockedOneTests(unittest.TestCase):
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

    def _write_blocked(self, day: str = BLOCKED_DAY, **edits) -> Path:
        path = self.config.state_dir / "receipts" / f"{day}.json"
        payload = _blocked_receipt(day)
        payload.update(edits)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _run(self, *, source=None, day="2026-01-02", commit=True, reviewed=None):
        return run_daily(
            self.config,
            source=source or FixtureConversationSource(self.config.source.fixtures_dir),
            runner=InProcessRunner(),
            skill_store=self.store,
            state=self.state,
            clock=_clock(),
            day=day,
            commit=commit,
            mutation_reviewed=reviewed,
        )

    def test_it_is_refused_before_the_lock_the_source_and_any_write(self) -> None:
        path = self._write_blocked()
        before = _tree_fingerprint(self.config.state_dir)
        with self.assertRaises(BlockedRunUnreviewed) as caught:
            self._run(source=_Untouched())
        message = str(caught.exception)
        self.assertIn(f"the {BLOCKED_DAY} run recorded a skill tree mutation", message)
        self.assertIn(str(path), message, "the receipt to read is named")
        self.assertIn(f"--mutation-reviewed {BLOCKED_DAY}", message, "the way out is spelled")
        self.assertEqual(_tree_fingerprint(self.config.state_dir), before, "nothing was written")
        self.assertFalse(self.state.lock_path().exists(), "the lock was never taken")

    def test_naming_the_blocked_day_lets_it_proceed_and_the_receipt_says_so(self) -> None:
        self._write_blocked()
        receipt = self._run(reviewed=BLOCKED_DAY)
        self.assertEqual(receipt.status, "ok", receipt.to_dict()["notes"])
        statement = [n for n in receipt.notes if n.startswith("the operator stated")]
        self.assertEqual(len(statement), 1, receipt.notes)
        self.assertIn(f"recorded by the {BLOCKED_DAY} run was reviewed", statement[0])
        self.assertIn(f"--mutation-reviewed {BLOCKED_DAY}", statement[0])
        self.assertNotIn("state it again", statement[0])
        # The reviewed run's receipt is now the newest, so the morning after
        # runs without the flag, and says nothing about a review.
        receipt = self._run(day="2026-01-03")
        self.assertEqual(receipt.status, "ok")
        self.assertFalse(any(n.startswith("the operator stated") for n in receipt.notes))

    def test_the_wrong_day_and_a_flag_with_nothing_blocked_are_each_refused(self) -> None:
        self._write_blocked()
        with self.assertRaises(BlockedRunUnreviewed) as caught:
            self._run(source=_Untouched(), reviewed="2026-01-03")
        self.assertIn("the flag named 2026-01-03, which is not that run", str(caught.exception))
        self.assertIn(f"--mutation-reviewed {BLOCKED_DAY}", str(caught.exception))
        with self.assertRaises(ConfigError) as bad:
            self._run(source=_Untouched(), reviewed="not a day!")
        self.assertTrue(str(bad.exception).startswith("--mutation-reviewed: "), str(bad.exception))

        (self.config.state_dir / "receipts" / f"{BLOCKED_DAY}.json").unlink()
        with self.assertRaises(ConfigError) as nothing:
            self._run(source=_Untouched(), reviewed=BLOCKED_DAY)
        self.assertIn("names no blocked run: there is no daily receipt", str(nothing.exception))
        ok = self._run()
        self.assertEqual(ok.status, "ok")
        with self.assertRaises(ConfigError) as unblocked:
            self._run(source=_Untouched(), day="2026-01-03", reviewed=BLOCKED_DAY)
        self.assertIn("(2026-01-02, ok) recorded no mutation", str(unblocked.exception))

    def test_the_field_is_the_signal_not_the_status(self) -> None:
        # A status is a word anyone can edit; the field is what the run measured.
        self._write_blocked(status="ok")
        with self.assertRaises(BlockedRunUnreviewed):
            self._run(source=_Untouched())
        # Without the field (an older build, a hand edit) nothing blocks, and the
        # flag has nothing to name; ``doctor`` is what warns about such a receipt.
        self._write_blocked(skill_tree={"changes": []})
        receipt = self._run()
        self.assertEqual(receipt.status, "ok")
        self.assertFalse(any(n.startswith("the operator stated") for n in receipt.notes))

    def test_a_day_with_a_line_break_is_spelled_not_obeyed(self) -> None:
        self._write_blocked(day=BLOCKED_DAY + "\nverdict: ready")
        with self.assertRaises(BlockedRunUnreviewed) as caught:
            self._run(source=_Untouched())
        self.assertNotIn("\n", str(caught.exception))
        self.assertIn("2026-01-01\\nverdict: ready run recorded", str(caught.exception))

    def test_an_unreadable_receipt_fails_closed_rather_than_clearing_the_block(self) -> None:
        path = self._write_blocked()
        path.write_text("{", encoding="utf-8")
        before = _tree_fingerprint(self.config.state_dir)
        with self.assertRaises(StateUnreadable) as caught:
            self._run(source=_Untouched())
        self.assertIn(f"the receipt {path} cannot be read", str(caught.exception))
        self.assertIn("move it aside", str(caught.exception))
        self.assertEqual(_tree_fingerprint(self.config.state_dir), before)
        if os.geteuid() != 0:
            path.write_text(json.dumps(_blocked_receipt()), encoding="utf-8")
            path.chmod(0)
            self.addCleanup(path.chmod, 0o600)
            with self.assertRaises(StateUnreadable) as denied:
                self._run(source=_Untouched())
            self.assertIn("cannot be opened: ", str(denied.exception))
            self.assertEqual(
                [u.path for u in self.state.scan_receipts().unreadable],
                [path],
                "the scan lists it instead of raising through doctor as a 70",
            )

    def test_a_dry_run_is_refused_too_and_its_statement_does_not_persist(self) -> None:
        self._write_blocked()
        with self.assertRaises(BlockedRunUnreviewed):
            self._run(source=_Untouched(), commit=False)
        receipt = self._run(commit=False, reviewed=BLOCKED_DAY)
        self.assertEqual(receipt.status, "ok")
        statement = next(n for n in receipt.notes if n.startswith("the operator stated"))
        self.assertIn("a dry run writes no receipt, so the next run must state it again", statement)
        with self.assertRaises(BlockedRunUnreviewed):
            self._run(source=_Untouched(), day="2026-01-03")


class ThroughTheCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.cfg = ["--config", str(build_env(self.base, seed="gate", use_subprocess_model=True))]

    def _receipts(self) -> dict[str, dict]:
        out = {}
        for path in sorted((self.base / "state" / "receipts").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") == SCHEMA_RECEIPT:
                out[payload["day"]] = payload
        return out

    def test_the_exit_code_stays_five_until_the_run_is_reviewed(self) -> None:
        proc = run_cli([*self.cfg, "daily", "--commit", "--day", BLOCKED_DAY])
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        (receipt_path,) = (self.base / "state" / "receipts").glob("*.json")
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["day"], BLOCKED_DAY)
        payload["status"] = "blocked"
        payload["skill_tree"]["mutation_detected"] = True
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")

        refused = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-02"])
        self.assertEqual(refused.returncode, RC_BLOCKED, refused.stderr)
        self.assertIn("BlockedRunUnreviewed: ", refused.stderr)
        self.assertIn(f"--mutation-reviewed {BLOCKED_DAY}", refused.stderr)
        self.assertEqual(sorted(self._receipts()), [BLOCKED_DAY], "the refused run left no receipt")
        doctor = run_cli([*self.cfg, "doctor"])
        self.assertEqual(doctor.returncode, RC_NEGATIVE, "doctor still names the mutation")

        bad = run_cli([*self.cfg, "daily", "--commit", "--mutation-reviewed", "not a day!"])
        self.assertEqual(bad.returncode, 2, "argparse refuses the value before any config is read")

        reviewed = run_cli(
            [
                *self.cfg,
                "daily",
                "--commit",
                "--day",
                "2026-01-02",
                "--mutation-reviewed",
                BLOCKED_DAY,
            ]
        )
        self.assertEqual(reviewed.returncode, RC_OK, reviewed.stderr)
        self.assertIn("the operator stated", reviewed.stdout)
        self.assertTrue(
            any(
                n.startswith("the operator stated") for n in self._receipts()["2026-01-02"]["notes"]
            )
        )
        after = run_cli([*self.cfg, "daily", "--commit", "--day", "2026-01-03"])
        self.assertEqual(after.returncode, RC_OK, after.stderr)
        self.assertEqual(sorted(self._receipts()), [BLOCKED_DAY, "2026-01-02", "2026-01-03"])
        self.assertEqual(run_cli([*self.cfg, "doctor"]).returncode, RC_OK)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
