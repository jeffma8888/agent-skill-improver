"""An export's thread id is held to the limit a listed id is held to.

A thread id is its record's key in the receipt, the checkpoint and each
scorecard and failure case. The command-line source already refuses a listed
id longer than ``MAX_THREAD_ID_LENGTH`` at discovery; ``_scan`` in
``adapters/fixture_source.py`` now passes over an export claiming one, with
its reason, and reads the exports beside it as before. Measured before (3.11
and 3.13, ``daily --commit`` over the healthy fixtures and one export more): a
2,000,000-character id was taken, and the receipt and ``daily --json`` grew to
about 4.0 MB each (about 10 KB without it), the checkpoint, the scorecard and
the stored conversation to about 2.0 MB each; a 341-character id beginning
with a GitHub token put the token in the receipt and on stdout as written.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.adapters.conversation_cli import _unsafe_id_reason
from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.domain.discovery import MAX_THREAD_ID_LENGTH
from tests.support import REPO_ROOT, build_env, run_cli

TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
EXPORTS = REPO_ROOT / "tests" / "fixtures" / "exports"
IDS = {
    "control": None,
    "long": "A" * 2_000_000,
    "at_limit": "B" * 256,
    "wide_at_limit": "\u00e9" * 256,
    "past_limit": "C" * 257,
    "token": TOKEN + "/" + "x" * 300,
}


def run_case(base: Path, thread_id: str | None) -> dict:
    exports = base / "exports"
    exports.mkdir(parents=True)
    for export in sorted(EXPORTS.glob("*.json")):
        (exports / export.name).write_bytes(export.read_bytes())
    if thread_id is not None:
        export = {
            "thread": {"id": thread_id, "workspacePath": "/w/h", "title": "h"},
            "elements": [{"id": "m1", "kind": "message", "position": 1, "role": "user"}],
        }
        (exports / "zz-long.json").write_text(json.dumps(export), encoding="utf-8")
    config = build_env(base / "env", fixtures_dir=exports, dry_run=False)
    argv = ["--config", str(config)]
    daily = run_cli([*argv, "--json", "daily", "--commit", "--day", "2026-01-02"])
    doctor = run_cli([*argv, "doctor"])
    state = sorted(p for p in (base / "env" / "state").rglob("*") if p.is_file())
    return {
        "daily": daily,
        "doctor": doctor,
        "summary": json.loads(daily.stdout) if daily.stdout.strip() else {},
        "state": {str(p.relative_to(base / "env" / "state")): p.read_bytes() for p in state},
    }


class AnExportsThreadIdIsHeldToTheListedIdLimitTests(unittest.TestCase):
    cases: dict[str, dict] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name).resolve()
        for case, thread_id in IDS.items():
            cls.cases[case] = run_case(root / case, thread_id)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def summary(self, case: str) -> dict:
        result = self.cases[case]
        self.assertEqual(result["daily"].returncode, 0, result["daily"].stderr)
        return result["summary"]

    def counts(self, case: str) -> tuple[int, int, int, int]:
        c = self.summary(case)["counts"]
        return (
            c["conversations_ingested"],
            c["records_scored"],
            c["failure_cases"],
            c["candidates_proposed"],
        )

    def passed_over(self, case: str) -> list[dict]:
        return self.summary(case)["discovery"]["passed_over"]

    def only_reason(self, case: str) -> str:
        (entry,) = self.passed_over(case)
        self.assertTrue(entry["entry"].endswith("zz-long.json"), entry)
        return entry["reason"]

    def test_the_control_the_healthy_fixtures_pass_nothing_over(self) -> None:
        self.assertEqual(self.counts("control"), (5, 5, 2, 1))
        self.assertEqual(self.passed_over("control"), [])

    def test_an_over_long_id_is_passed_over_and_the_exports_beside_it_are_read(self) -> None:
        self.assertEqual(self.counts("long"), self.counts("control"))
        reason = self.only_reason("long")
        self.assertIn(
            "an id that is over-long (2000000 characters; the limit is 256, as for an id a "
            "conversation command lists), so it was not taken as a thread",
            reason,
        )
        self.assertLess(len(reason), 400)

    def test_no_output_or_state_file_carries_the_over_long_id(self) -> None:
        result = self.cases["long"]
        run = ("A" * 200).encode()
        self.assertNotIn(run, result["daily"].stdout.encode())
        self.assertNotIn(run, result["doctor"].stdout.encode())
        holders = sorted(name for name, data in result["state"].items() if run in data)
        self.assertEqual(holders, [])
        control_stdout = len(self.cases["control"]["daily"].stdout.encode())
        self.assertLess(len(result["daily"].stdout.encode()), control_stdout + 4_000)
        for name, data in result["state"].items():
            with self.subTest(file=name):
                self.assertLess(len(data), 50_000)

    def test_the_limit_is_the_one_a_listed_id_is_held_to(self) -> None:
        self.assertEqual(MAX_THREAD_ID_LENGTH, 256)
        for case in ("at_limit", "wide_at_limit"):
            with self.subTest(case=case):
                self.assertEqual(self.counts(case), (6, 6, 2, 1))
                self.assertEqual(self.passed_over(case), [])
        self.assertEqual(self.counts("past_limit"), (5, 5, 2, 1))
        reason = self.only_reason("past_limit")
        self.assertIn("is over-long (257 characters; the limit is 256", reason)
        listed = _unsafe_id_reason("C" * 257)
        self.assertEqual(listed, "is over-long (257 characters; the limit is 256)")
        self.assertIn(listed.removesuffix(")"), reason)

    def test_the_length_is_counted_as_written(self) -> None:
        # Padding is part of the id: it is the record's key as written.
        with TemporaryDirectory() as tmp:
            exports = Path(tmp).resolve()
            ids = {"padded.json": "D" * 256 + " ", "exact.json": "E" * 256}
            for name, thread_id in ids.items():
                export = {"thread": {"id": thread_id}, "elements": []}
                (exports / name).write_text(json.dumps(export), encoding="utf-8")
            source = FixtureConversationSource(exports)
            threads = [t.thread_id for t in source.discover_page(since="", limit=10).threads]
            passed = {Path(entry).name: reason for entry, reason in source.passed_over()}
        self.assertEqual(threads, ["E" * 256])
        self.assertEqual(list(passed), ["padded.json"])
        self.assertIn("is over-long (257 characters; the limit is 256", passed["padded.json"])

    def test_the_reason_quotes_the_id_masked_and_cut(self) -> None:
        result = self.cases["token"]
        self.assertEqual(self.counts("token"), (5, 5, 2, 1))
        reason = self.only_reason("token")
        self.assertIn("claims thread '[REDACTED:github-token]/xxx", reason)
        self.assertIn("more chars not quoted], an id that is over-long (341 characters", reason)
        self.assertNotIn(TOKEN, result["daily"].stdout)
        self.assertNotIn(TOKEN, result["doctor"].stdout)
        holders = sorted(n for n, data in result["state"].items() if TOKEN.encode() in data)
        self.assertEqual(holders, [])

    def test_doctor_warns_for_the_export_it_passes_over(self) -> None:
        doctor = self.cases["long"]["doctor"]
        self.assertEqual(doctor.returncode, 0, doctor.stderr)
        warns = [line for line in doctor.stdout.splitlines() if "zz-long.json" in line]
        self.assertEqual(len(warns), 1, doctor.stdout)
        self.assertIn("WARN", warns[0])
        self.assertIn("is over-long (2000000 characters; the limit is 256", warns[0])
        self.assertLess(len(doctor.stdout.encode()), 5_000)
        control = self.cases["control"]["doctor"].stdout
        self.assertNotIn("zz-long.json", control)


if __name__ == "__main__":
    unittest.main()
