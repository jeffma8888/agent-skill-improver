"""``asi history SKILL_ID``: one skill's candidates, oldest first, through the CLI.

The setup is the measured-before scenario of c248 (DESIGN 4.2): one candidate
applied with ``--operator alice --note "probe note"`` and rolled back, one that
failed a real gate (``min_fixes=50``), one open and validated. Before c248b no
command answered "what happened to this skill": ``asi history`` was an invalid
choice (exit 2), and the readers it would have reused created ``state/``.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from agent_skill_improver.app.state import StateStore
from agent_skill_improver.clock import SystemClock
from agent_skill_improver.domain.candidate import challenge_for
from agent_skill_improver.hashing import sha256_file, sha256_text
from tests.support import build_env, run_cli

A = "CAND-2026-01-01-01-tdd"
B = "CAND-2026-01-02-01-tdd"
C = "CAND-2026-01-03-01-tdd"
TOKEN = "ghp_" + "Z9" * 18
#: ``app/history.HISTORY_LIVE_BASIS``, spelled here so this module imports on a
#: tree without ``app/history.py`` (and fails there by assertion); module 5's
#: import-check test pins that the two agree.
HISTORY_LIVE_BASIS = "the copy read during this command's discovery"


def _cli(cfg: list[str], *args: str, timeout: float = 300):
    return run_cli([*cfg, *args], timeout=timeout)


def _json(cfg: list[str], *args: str) -> dict:
    proc = _cli(cfg, "--json", *args)
    assert proc.returncode == 0, (args, proc.returncode, proc.stderr[-2000:])
    return json.loads(proc.stdout)


def _env(base: Path, **kw) -> list[str]:
    return ["--config", str(build_env(base, dry_run=False, seed="probe", **kw))]


def _daily(cfg: list[str], day: str) -> None:
    proc = _cli(cfg, "daily", "--commit", "--day", day)
    assert proc.returncode == 0, proc.stderr[-2000:]


def _apply(cfg: list[str], cid: str, *extra: str) -> None:
    token = _json(cfg, "review", cid)["approval_token"]
    proc = _cli(cfg, "apply", cid, "--approve", token, "--i-reviewed-this", *extra)
    assert proc.returncode == 0, proc.stderr[-2000:]


def _snapshot(*roots: Path) -> dict[str, str]:
    """Every path under the roots with its kind, mode, mtime and content hash."""
    out: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            st = p.lstat()
            body = "dir" if p.is_dir() else f"{sha256_file(p)} {st.st_mtime_ns}"
            out[f"{root.name}/{p.relative_to(root)}"] = f"{body} {oct(st.st_mode)}"
    return out


def fabricate(template: dict, cid: str, created_at: str, **changes) -> dict:
    """A copy of a real record under another id, its challenge re-derived so it
    still recomputes; ``changes`` set top-level keys (``status`` ...)."""
    record = json.loads(json.dumps(template))
    record["id"] = cid
    record["created_at"] = created_at
    record.update(changes)
    if isinstance(record.get("validation"), dict):
        record["validation"]["candidate_id"] = cid
    new_text = record["patch"]["new_text"]
    record["approval_challenge"] = challenge_for(
        cid, record["target"]["base_sha256"], sha256_text(new_text)
    )
    return record


def write_record(state: Path, record: dict) -> Path:
    directory = state / "candidates"
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(state, 0o700)
    os.chmod(directory, 0o700)
    path = directory / f"{record['id']}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


class _Scenario(unittest.TestCase):
    """The 4.2 scenario, built once; tests that change state build their own."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls._tmp.name)
        cls.cfg = _env(cls.base)
        _daily(cls.cfg, "2026-01-01")
        _apply(cls.cfg, A, "--operator", "alice", "--note", "probe note")
        proc = _cli(cls.cfg, "rollback", A)
        assert proc.returncode == 0, proc.stderr[-2000:]
        _daily(_env(cls.base, min_fixes=50), "2026-01-02")
        cls.cfg = _env(cls.base)
        _daily(cls.cfg, "2026-01-03")
        cls.state = cls.base / "state"
        cls.template = json.loads((cls.state / "candidates" / f"{C}.json").read_text())
        # A record of another skill whose id still ends in ``-tdd``:
        # ``safe_name("tdd.")`` is ``tdd``, so only ``target.skill_id`` tells them apart.
        other = fabricate(cls.template, "CAND-2026-01-05-01-tdd", "2026-11-01T00:00:00Z")
        other["target"]["skill_id"] = "tdd."
        write_record(cls.state, other)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def history(self, *args: str) -> dict:
        return _json(self.cfg, "history", "tdd", *args)

    def text(self, *args: str) -> str:
        proc = _cli(self.cfg, "history", "tdd", *args)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout


class HistoryOfTheMeasuredScenarioTests(_Scenario):
    def test_every_candidate_for_the_skill_is_listed_oldest_first_with_its_status_and_gate(self):
        payload = self.history()
        rows = payload["candidates"]
        self.assertEqual([r["id"] for r in rows], [A, B, C])
        self.assertEqual([r["status"] for r in rows], ["rolled-back", "gate-failed", "validated"])
        self.assertEqual([r["gate"]["passed"] for r in rows], [True, False, True])
        self.assertEqual([r["open"] for r in rows], [False, False, True])
        self.assertTrue(all(r["integrity"]["recomputes"] for r in rows))
        self.assertTrue(rows[1]["gate"]["reasons"], rows[1]["gate"])
        self.assertEqual(payload["counts"]["candidates"], 3)
        self.assertEqual((payload["counts"]["open"], payload["counts"]["closed"]), (1, 2))
        text = self.text()
        lines = text.splitlines()
        heads = [line.split("  ")[0] for line in lines if line.startswith("CAND-")]
        self.assertEqual(heads, [A, B, C])
        self.assertIn("rolled-back  gate=pass", text)
        self.assertIn("gate-failed  gate=fail", text)
        self.assertIn(f"validated  gate=pass", text)
        self.assertIn(f"(open: asi review {C})", text)
        self.assertIn("  gate said  ", text)
        self.assertTrue(
            lines[0].startswith("history of tdd: 3 candidate(s) recorded, 3 of 3 shown")
        )
        self.assertIn("without the run lock", lines[0])
        self.assertTrue(
            lines[-1].startswith("3 candidate(s): 1 open, 2 closed, 0 of unknown status")
        )
        self.assertIn("this is what state/ says, not proof of what happened", lines[-1])

    def test_only_records_whose_target_names_the_skill_are_listed(self):
        ids = [r["id"] for r in self.history()["candidates"]]
        self.assertNotIn("CAND-2026-01-05-01-tdd", ids)
        self.assertNotIn("CAND-2026-01-05-01-tdd", self.text())
        other = _json(self.cfg, "history", "tdd.")
        self.assertEqual([r["id"] for r in other["candidates"]], ["CAND-2026-01-05-01-tdd"])

    def test_the_apply_and_rollback_receipts_are_shown_with_an_unauthenticated_label_and_no_rollback_operator(
        self,
    ):
        row = self.history()["candidates"][0]
        self.assertEqual(row["apply"]["operator"], "alice")
        self.assertIs(row["apply"]["operator_verified"], False)
        self.assertEqual(row["apply"]["note"], "probe note")
        self.assertTrue(row["apply"]["backup_manifest_path"].endswith("backup.json"))
        self.assertIsNone(row["rollback"]["operator"])
        self.assertTrue(row["rollback"]["manifest"].endswith("backup.json"))
        self.assertTrue(row["rollback"]["rolled_back_at"])
        self.assertIsNotNone(row["backup_manifest"])
        text = self.text()
        applied = [line for line in text.splitlines() if line.startswith("  applied    ")]
        self.assertEqual(len(applied), 1, text)
        self.assertIn('operator label "alice" (unauthenticated)', applied[0])
        self.assertIn('note "probe note"', applied[0])
        rolled = [line for line in text.splitlines() if line.startswith("  rolled back ")]
        self.assertEqual(len(rolled), 1, text)
        self.assertIn("(a rollback records no operator)", rolled[0])
        self.assertNotIn("alice", rolled[0])
        # The backup rollback would use, as state/ names it (masked, cut).
        self.assertIn(f"  backup     {row['backup_manifest']}", text.splitlines())

    def test_a_populated_state_tree_is_left_exactly_as_it_was(self):
        roots = [self.state, self.base / "skills-canonical", self.base / "skills-mirror"]
        before = _snapshot(*roots)
        lock = self.state / "locks" / "daily.lock"
        lock_before = lock.exists()
        for args in ((), ("--limit", "2")):
            with self.subTest(args=args):
                self.text(*args)
                self.history(*args)
                self.assertEqual(_snapshot(*roots), before)
                self.assertEqual(lock.exists(), lock_before)

    def test_the_json_has_the_documented_keys_and_operator_verified_is_false(self):
        payload = self.history()
        self.assertEqual(
            set(payload),
            {
                "schema",
                "skill_id",
                "read_at",
                "read_without_run_lock",
                "state_dir",
                "live",
                "limit",
                "earlier_not_shown",
                "candidates",
                "unreadable",
                "unreadable_not_shown",
                "unattributed_unreadable",
                "set_aside",
                "set_aside_not_shown",
                "counts",
            },
        )
        self.assertEqual(payload["schema"], "asi.history.v1")
        self.assertIs(payload["read_without_run_lock"], True)
        self.assertEqual(payload["state_dir"], "present")
        self.assertEqual(
            set(payload["live"]), {"state", "canonical_path", "sha256", "reason", "basis"}
        )
        self.assertEqual(payload["live"]["basis"], HISTORY_LIVE_BASIS)
        self.assertEqual(
            set(payload["counts"]),
            {
                "candidates",
                "open",
                "closed",
                "unknown_status",
                "untrusted",
                "unreadable",
                "set_aside",
            },
        )
        row_keys = {
            "id",
            "created_at",
            "status",
            "status_known",
            "open",
            "canonical_path",
            "base_sha256",
            "base_sha256_invalid",
            "replacement_sha256",
            "added_lines",
            "removed_lines",
            "intended_change",
            "gate",
            "integrity",
            "invalid_fields",
            "invalid_not_shown",
            "apply",
            "rollback",
            "decline",
            "backup_manifest",
            "live_match",
            "live_match_reason",
            "live_match_basis",
        }
        for row in payload["candidates"]:
            self.assertEqual(set(row), row_keys)
            self.assertEqual(row["live_match_basis"], HISTORY_LIVE_BASIS)
            self.assertEqual(
                set(row["gate"]),
                {
                    "ran",
                    "passed",
                    "fixes",
                    "regressions",
                    "holdout_cases",
                    "judge_candidate_preference_ratio",
                    "reasons",
                    "reasons_not_shown",
                },
            )
            self.assertEqual(
                set(row["integrity"]), {"recomputes", "problems", "problems_not_shown"}
            )
        apply = payload["candidates"][0]["apply"]
        self.assertEqual(
            set(apply),
            {
                "applied_at",
                "operator",
                "operator_verified",
                "note",
                "backup_manifest_path",
                "finishes_an_earlier_run",
            },
        )
        self.assertIs(apply["operator_verified"], False)
        self.assertEqual(
            set(payload["candidates"][0]["rollback"]), {"rolled_back_at", "manifest", "operator"}
        )

    def test_a_limit_outside_one_to_a_thousand_exits_2(self):
        for value in ("0", "-1", "1001", "x", "9" * 5000):
            with self.subTest(value=value[:8]):
                proc = _cli(self.cfg, "history", "tdd", f"--limit={value}")
                self.assertEqual(proc.returncode, 2, proc.stderr[-500:])
                self.assertIn("a whole number from 1 to 1000", proc.stderr)
                self.assertLess(len(proc.stderr), 2000)
        for value in ("1", "1000"):
            with self.subTest(value=value):
                self.assertEqual(self.history("--limit", value)["limit"], int(value))

    def test_an_unknown_skill_id_with_a_newline_escape_or_token_is_one_masked_line_in_both_modes(
        self,
    ):
        for skill in ("tdd\nFORGED found", "\x1b[2J", TOKEN):
            for mode in ((), ("--json",)):
                with self.subTest(skill=skill[:6], mode=mode):
                    proc = _cli(self.cfg, *mode, "history", skill)
                    self.assertEqual(proc.returncode, 4, proc.stderr)
                    self.assertEqual(len(proc.stderr.splitlines()), 1, repr(proc.stderr))
                    self.assertTrue(proc.stderr.startswith("not found: no skill "), proc.stderr)
                    self.assertNotIn(TOKEN, proc.stderr)
                    self.assertNotIn("\x1b", proc.stderr)
                    self.assertEqual(proc.stdout, "")

    def test_a_read_beside_a_run_holding_the_lock_waits_for_nothing_and_changes_nothing(self):
        lock = self.state / "locks" / "daily.lock"
        lock.parent.mkdir(exist_ok=True)
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            roots = [self.state, self.base / "skills-canonical"]
            before = _snapshot(*roots)
            started = time.monotonic()
            proc = _cli(self.cfg, "history", "tdd", timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertLess(time.monotonic() - started, 60)
            self.assertIn("without the run lock", proc.stdout)
            self.assertEqual(_snapshot(*roots), before)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


class HistoryOfAFreshTreeTests(unittest.TestCase):
    """Each test builds its own tree: these change state or need it absent."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.cfg = _env(self.base)
        self.state = self.base / "state"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_the_live_match_follows_the_file_on_disk_not_the_status(self):
        _daily(self.cfg, "2026-01-01")
        _apply(self.cfg, A)
        skill = self.base / "skills-canonical" / "tdd" / "SKILL.md"
        self.assertEqual(
            _json(self.cfg, "history", "tdd")["candidates"][0]["live_match"], "replacement"
        )
        applied = skill.read_bytes()
        skill.write_bytes(applied + b"\nA hand edit.\n")
        row = _json(self.cfg, "history", "tdd")["candidates"][0]
        self.assertEqual((row["status"], row["live_match"]), ("applied", "neither"))
        skill.write_bytes(applied)
        proc = _cli(self.cfg, "rollback", A)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = _json(self.cfg, "history", "tdd")["candidates"][0]
        self.assertEqual((row["status"], row["live_match"]), ("rolled-back", "base"))

    def test_an_absent_state_tree_stays_absent_and_the_live_skill_is_still_shown(self):
        payload = _json(self.cfg, "history", "tdd")
        self.assertFalse(self.state.exists())
        self.assertEqual(payload["state_dir"], "absent")
        self.assertEqual(payload["live"]["state"], "canonical")
        self.assertRegex(payload["live"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(payload["counts"]["candidates"], 0)
        proc = _cli(self.cfg, "history", "tdd")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("0 candidate(s) recorded for this skill", proc.stdout)
        self.assertFalse(self.state.exists())
        with self.subTest("state/ a symlink"):
            elsewhere = self.base / "elsewhere"
            elsewhere.mkdir()
            self.state.symlink_to(elsewhere)
            proc = _cli(self.cfg, "history", "tdd")
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertEqual(list(elsewhere.iterdir()), [])

    def test_an_unknown_skill_exits_4_and_a_removed_skill_still_shows_its_records(self):
        proc = _cli(self.cfg, "history", "nope")
        self.assertEqual(proc.returncode, 4, proc.stderr)
        self.assertEqual(
            proc.stderr,
            "not found: no skill nope in the configured roots and no candidate record names it "
            "(asi skills lists the ids)\n",
        )
        _daily(self.cfg, "2026-01-01")
        for root in ("skills-canonical", "skills-mirror"):
            if (self.base / root / "tdd").exists():
                (self.base / root / "tdd").rename(self.base / f"removed-{root}")
        payload = _json(self.cfg, "history", "tdd")
        self.assertEqual(payload["live"]["state"], "not-in-roots")
        self.assertEqual([r["id"] for r in payload["candidates"]], [A])
        row = payload["candidates"][0]
        self.assertEqual(row["live_match"], "not-compared")
        self.assertEqual(
            row["live_match_reason"], "the skill is not in the configured skill roots now"
        )
        proc = _cli(self.cfg, "history", "tdd")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("live: not in the configured skill roots now", proc.stdout)

    def test_a_declined_record_shows_its_time_operator_label_and_reason(self):
        _daily(self.cfg, "2026-01-01")
        proc = _cli(self.cfg, "decline", A, "--reason", "too broad", "--operator", "bob")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = _json(self.cfg, "history", "tdd")["candidates"][0]
        self.assertEqual(row["status"], "declined")
        decline = row["decline"]
        self.assertEqual((decline["operator"], decline["reason"]), ("bob", "too broad"))
        self.assertTrue(decline["declined_at"])
        self.assertIs(decline["operator_verified"], False)
        self.assertEqual((decline["from"], decline["disagreements"]), ("record and receipt", []))
        text = _cli(self.cfg, "history", "tdd").stdout
        line = [x for x in text.splitlines() if x.startswith("  declined   ")]
        self.assertEqual(len(line), 1, text)
        self.assertIn('operator label "bob" (unauthenticated)', line[0])
        self.assertIn('reason "too broad"', line[0])
        self.assertNotIn("decline disagrees", text)

    def test_skill_level_set_aside_records_are_listed_and_a_conversations_are_not(self):
        state = StateStore(self.base.resolve() / "state", clock=SystemClock())
        state.defer("tdd", stage="propose", reason="skill-level deferral")
        state.refuse("tdd", stage="validate", reason="skill-level refusal")
        state.quarantine("tdd", stage="ingest", reason="a conversation named tdd")
        state.quarantine("tdd", stage="score:instruction-compliance", reason="a conversation score")
        state.quarantine("other", stage="propose", reason="another skill")
        payload = _json(self.cfg, "history", "tdd")
        reasons = sorted(r["reason"] for r in payload["set_aside"])
        self.assertEqual(reasons, ["skill-level deferral", "skill-level refusal"])
        self.assertEqual(payload["counts"]["set_aside"], 2)
        text = _cli(self.cfg, "history", "tdd").stdout
        self.assertEqual(len([x for x in text.splitlines() if x.startswith("set aside   ")]), 2)
        self.assertIn("deferred at propose", text)
        self.assertNotIn("a conversation", text)
        self.assertNotIn("another skill", text)

    def _sixty(self) -> list[dict]:
        """60 recomputing records, created_at running against id order; the
        oldest is the only open one."""
        _daily(self.cfg, "2026-01-01")
        template = json.loads((self.state / "candidates" / f"{A}.json").read_text())
        (self.state / "candidates" / f"{A}.json").unlink()
        records = []
        for i in range(60):
            status = "validated" if i == 59 else "rolled-back"
            record = fabricate(
                template,
                f"CAND-2026-02-{1 + i // 20:02d}-{1 + i % 20:02d}-tdd",
                f"2026-11-01T00:{59 - i:02d}:00Z",
                status=status,
            )
            write_record(self.state, record)
            records.append(record)
        return sorted(records, key=lambda r: (r["created_at"], r["id"]))

    def test_the_default_shows_the_fifty_most_recent_oldest_first_and_names_how_many_earlier(self):
        ordered = self._sixty()
        payload = _json(self.cfg, "history", "tdd")
        self.assertEqual(payload["limit"], 50)
        self.assertEqual([r["id"] for r in payload["candidates"]], [r["id"] for r in ordered[10:]])
        self.assertEqual(payload["earlier_not_shown"], 10)
        text = _cli(self.cfg, "history", "tdd").stdout
        self.assertIn("10 earlier candidate record(s) for this skill not shown", text)
        heads = [x.split("  ")[0] for x in text.splitlines() if x.startswith("CAND-")]
        self.assertEqual(heads, [r["id"] for r in ordered[10:]])

    def test_limit_n_shows_n_and_counts_cover_every_record(self):
        ordered = self._sixty()
        payload = _json(self.cfg, "history", "tdd", "--limit", "3")
        self.assertEqual([r["id"] for r in payload["candidates"]], [r["id"] for r in ordered[-3:]])
        self.assertEqual(payload["earlier_not_shown"], 57)
        counts = payload["counts"]
        self.assertEqual(counts["candidates"], 60)
        self.assertEqual((counts["open"], counts["closed"], counts["untrusted"]), (1, 59, 0))
        text = _cli(self.cfg, "history", "tdd", "--limit", "3").stdout
        self.assertIn("60 candidate(s): 1 open, 59 closed", text)
        self.assertIn("3 of 60 shown (--limit 3)", text)

    def test_an_absent_partial_or_wide_state_tree_is_read_as_found_in_text_and_json(self):
        with self.subTest("no state/"):
            for mode in ((), ("--json",)):
                self.assertEqual(_cli(self.cfg, *mode, "history", "tdd").returncode, 0)
            self.assertFalse(self.state.exists())
        with self.subTest("state/ with candidates/ only"):
            _daily(self.cfg, "2026-01-01")
            for name in ("receipts", "backups", "locks", "deferred", "quarantine", "refusals"):
                if (self.state / name).exists():
                    shutil.rmtree(self.state / name)
            kept = sorted(p.name for p in self.state.iterdir())
            for mode in ((), ("--json",)):
                proc = _cli(self.cfg, *mode, "history", "tdd")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn(A, proc.stdout)
            self.assertEqual(sorted(p.name for p in self.state.iterdir()), kept)
        with self.subTest("state/ at 0o755"):
            os.chmod(self.state, 0o755)
            for mode in ((), ("--json",)):
                self.assertEqual(_cli(self.cfg, *mode, "history", "tdd").returncode, 0)
            self.assertEqual(self.state.stat().st_mode & 0o777, 0o755)

    def test_a_decline_whose_receipt_disagrees_on_any_field_names_each_field(self):
        _daily(self.cfg, "2026-01-01")
        proc = _cli(self.cfg, "decline", A, "--reason", "too broad", "--operator", "alice")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        receipt_path = self.state / "receipts" / f"decline--{A}.json"
        record_path = self.state / "candidates" / f"{A}.json"
        receipt = json.loads(receipt_path.read_text())
        record = json.loads(record_path.read_text())
        everything = {
            "operator": "mallory",
            "status_before": "applied",
            "declined_at": "2020-01-01T00:00:00Z",
            "candidate_id": "CAND-2026-01-09-01-tdd",
            "reason": "another reason",
        }
        cases = [({k: v}, [k]) for k, v in everything.items()]
        cases.append(
            (everything, ["candidate_id", "operator", "declined_at", "reason", "status_before"])
        )
        for edits, fields in cases:
            with self.subTest(fields=fields):
                receipt_path.write_text(json.dumps({**receipt, **edits}))
                row = _json(self.cfg, "history", "tdd")["candidates"][0]
                decline = row["decline"]
                self.assertNotIn("unreadable", decline)
                self.assertEqual([d["field"] for d in decline["disagreements"]], fields)
                self.assertEqual((decline["operator"], decline["reason"]), ("alice", "too broad"))
                self.assertEqual(decline["declined_at"], record["decline"]["declined_at"])
                self.assertIs(decline["operator_verified"], False)
                self.assertEqual(decline["from"], "record and receipt")
                text = _cli(self.cfg, "history", "tdd").stdout
                said = [x for x in text.splitlines() if x.startswith("  decline disagrees  ")]
                self.assertEqual([x.split("  ")[2].split(":")[0] for x in said], fields)
                if "operator" in fields:
                    self.assertIn('operator: record "alice", receipt "mallory"', text)
        with self.subTest("record only"):
            receipt_path.rename(receipt_path.with_name("kept-aside.json"))
            row = _json(self.cfg, "history", "tdd")["candidates"][0]
            self.assertEqual(
                (row["decline"]["from"], row["decline"]["operator"]), ("record", "alice")
            )
            self.assertIn("no decline receipt found", _cli(self.cfg, "history", "tdd").stdout)
            receipt_path.with_name("kept-aside.json").rename(receipt_path)
        with self.subTest("receipt only"):
            receipt_path.write_text(json.dumps({**receipt, "operator": "mallory"}))
            bare = {k: v for k, v in record.items() if k != "decline"}
            bare["status"] = "validated"
            record_path.write_text(json.dumps(bare))
            row = _json(self.cfg, "history", "tdd")["candidates"][0]
            self.assertEqual(row["decline"]["from"], "receipt")
            self.assertEqual(row["decline"]["operator"], "mallory")
            self.assertIs(row["decline"]["operator_verified"], False)
            self.assertIn(
                "decline receipt present but the record holds no decline",
                _cli(self.cfg, "history", "tdd").stdout,
            )
        # With no record decline, only the id a receipt holds ties it to this
        # candidate: one naming another, or none, is never shown as its decline.
        for label, why in (
            ("naming another candidate", "names another candidate"),
            ("naming no candidate", "holds no candidate id"),
        ):
            with self.subTest(f"receipt only, {label}"):
                forged = {**receipt, "operator": "mallory", "reason": "receipt reason"}
                if why == "names another candidate":
                    forged["candidate_id"] = "CAND-2026-01-09-01-tdd"
                else:
                    del forged["candidate_id"]
                receipt_path.write_text(json.dumps(forged))
                row = _json(self.cfg, "history", "tdd")["candidates"][0]
                self.assertEqual(row["decline"], {"unreadable": why})
                self.assertNotIn("mallory", json.dumps(row))
                self.assertNotIn("receipt reason", json.dumps(row))
                text = _cli(self.cfg, "history", "tdd").stdout
                self.assertIn(f"  decline    receipt unreadable: {why}", text.splitlines())
                self.assertNotIn("  declined   ", text)
                self.assertNotIn("mallory", text)
                self.assertNotIn("receipt reason", text)

    def test_a_decline_receipt_of_another_schema_is_unreadable_and_none_of_its_values_shown(self):
        _daily(self.cfg, "2026-01-01")
        proc = _cli(self.cfg, "decline", A, "--reason", "too broad", "--operator", "alice")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        receipt_path = self.state / "receipts" / f"decline--{A}.json"
        record_path = self.state / "candidates" / f"{A}.json"
        receipt = json.loads(receipt_path.read_text())
        record = json.loads(record_path.read_text())
        newer = {
            **receipt,
            "schema": "asi.decline_receipt.v999",
            "operator": "mallory",
            "reason": "receipt reason",
        }
        receipt_path.write_text(json.dumps(newer))
        why = "holds schema asi.decline_receipt.v999, not asi.decline_receipt.v1"
        with self.subTest("beside the record's decline"):
            row = _json(self.cfg, "history", "tdd")["candidates"][0]
            decline = row["decline"]
            self.assertEqual(
                set(decline),
                {
                    "declined_at",
                    "operator",
                    "operator_verified",
                    "reason",
                    "from",
                    "disagreements",
                    "receipt_unreadable",
                },
            )
            self.assertEqual(
                (decline["from"], decline["operator"], decline["reason"]),
                ("record", "alice", "too broad"),
            )
            self.assertEqual(decline["declined_at"], record["decline"]["declined_at"])
            self.assertEqual(decline["disagreements"], [])
            self.assertEqual(decline["receipt_unreadable"], why)
            self.assertNotIn("mallory", json.dumps(row))
            text = _cli(self.cfg, "history", "tdd").stdout
            self.assertIn(f"  decline    receipt unreadable: {why}", text.splitlines())
            self.assertIn('operator label "alice" (unauthenticated)  reason "too broad"', text)
            for absent in ("mallory", "receipt reason", "decline disagrees", "no decline receipt"):
                self.assertNotIn(absent, text)
        with self.subTest("receipt only"):
            bare = {k: v for k, v in record.items() if k != "decline"}
            bare["status"] = "validated"
            record_path.write_text(json.dumps(bare))
            row = _json(self.cfg, "history", "tdd")["candidates"][0]
            self.assertEqual(row["decline"], {"unreadable": why})
            self.assertNotIn("mallory", json.dumps(row))
            text = _cli(self.cfg, "history", "tdd").stdout
            self.assertIn(f"  decline    receipt unreadable: {why}", text.splitlines())
            for absent in ("mallory", "receipt reason", "  declined   ", "receipt present but"):
                self.assertNotIn(absent, text)


if __name__ == "__main__":
    unittest.main()
