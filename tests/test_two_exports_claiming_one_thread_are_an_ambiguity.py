"""Two exports claiming one thread are an ambiguity, not a choice.

The fixtures source (``adapters/fixture_source.py``) discovers a thread by the
id in each export, and ``fetch`` returned the first export in directory order
whose id matched. Measured before (3.11 and 3.13, the same on both), with an
export ``aa.json`` claiming the id of the healthy ``failing.json`` and holding
no elements: ``daily --commit`` ingested 5, found 0 failure cases and proposed
nothing, where the healthy fixtures alone give 2 failure cases and 1 proposal;
nothing was passed over and ``doctor`` printed no warning. With the same export
sorted after it (``zz.json``) the run was as if it were not there, again
without a word. Exports claiming one thread that differ in any byte are now
each passed over naming the others, and none of them is taken; a repeat of an
export, byte for byte, is taken once, the rest passed over as repeats.
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.domain.discovery import passed_over_note
from agent_skill_improver.errors import RecordDefect
from tests.support import FIXTURE_EXPORTS, build_env, run_cli

HEALTHY = json.loads((FIXTURE_EXPORTS / "failing.json").read_text(encoding="utf-8"))
THREAD = HEALTHY["thread"]["id"]
DECOY = {"thread": {"id": THREAD, "title": "decoy"}, "elements": []}
NONE_TAKEN = "this source cannot tell which is the conversation, so it takes none of them"


def ambiguous(thread_id: str, others: str, verb: str = "does") -> str:
    return f"claims thread {thread_id!r}, as {others} also {verb} with other bytes; {NONE_TAKEN}"


class TheSourceTakesNoneOfTheClaimsTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.exports = Path(tmp.name).resolve() / "exports"
        shutil.copytree(FIXTURE_EXPORTS, self.exports)

    def write(self, name: str, payload: object) -> Path:
        path = self.exports / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def copy_of_the_healthy_export(self, name: str) -> Path:
        path = self.exports / name
        shutil.copyfile(self.exports / "failing.json", path)
        return path

    def scan(self) -> tuple[list[str], dict[str, str]]:
        source = FixtureConversationSource(self.exports)
        page = source.discover_page(since="", limit=100)
        return [t.thread_id for t in page.threads], {
            Path(entry).name: reason for entry, reason in source.passed_over()
        }

    def test_a_claim_sorted_ahead_of_the_healthy_export_is_not_fetched_in_its_place(
        self,
    ) -> None:
        self.write("aa.json", DECOY)
        threads, passed = self.scan()
        self.assertNotIn(THREAD, threads)
        self.assertEqual(
            passed,
            {
                "aa.json": ambiguous(THREAD, "'failing.json'"),
                "failing.json": ambiguous(THREAD, "'aa.json'"),
            },
        )
        with self.assertRaises(RecordDefect) as caught:
            FixtureConversationSource(self.exports).fetch(THREAD)
        self.assertIn(
            "2 entries there could not be taken as a thread and may be it: ", str(caught.exception)
        )

    def test_a_claim_sorted_after_the_healthy_export_is_named_too(self) -> None:
        self.write("zz.json", DECOY)
        threads, passed = self.scan()
        self.assertNotIn(THREAD, threads)
        self.assertEqual(
            passed,
            {
                "failing.json": ambiguous(THREAD, "'zz.json'"),
                "zz.json": ambiguous(THREAD, "'failing.json'"),
            },
        )

    def test_three_claims_each_name_the_other_two(self) -> None:
        self.write("aa.json", DECOY)
        self.write("zz.json", {**DECOY, "elements": [{}]})
        _, passed = self.scan()
        self.assertEqual(passed["aa.json"], ambiguous(THREAD, "'failing.json', 'zz.json'", "do"))
        self.assertEqual(passed["zz.json"], ambiguous(THREAD, "'aa.json', 'failing.json'", "do"))

    def test_a_repeat_of_an_export_is_taken_once_and_named(self) -> None:
        self.copy_of_the_healthy_export("copy.json")
        threads, passed = self.scan()
        self.assertEqual(threads.count(THREAD), 1)
        self.assertEqual(
            passed,
            {
                "failing.json": f"repeats, byte for byte, the export of thread {THREAD!r} in "
                "'copy.json', which is taken once"
            },
        )
        conversation = FixtureConversationSource(self.exports).fetch(THREAD)
        self.assertEqual(conversation.thread_id, THREAD)

    def test_a_twin_python_calls_equal_is_an_ambiguity_not_a_repeat(self) -> None:
        # ``true == 1`` and ``1 == 1.0`` in Python; the normalizer reads a
        # ``true`` position as 0. A twin with every 0 and 1 spelled false and
        # true is equal to the export as parsed and is another conversation.
        def booleans(value: object) -> object:
            if isinstance(value, dict):
                return {k: booleans(v) for k, v in value.items()}
            if isinstance(value, list):
                return [booleans(v) for v in value]
            if type(value) is int and value in (0, 1):
                return bool(value)
            return value

        twin = booleans(HEALTHY)
        self.assertEqual(twin, HEALTHY)
        self.write("aa.json", twin)
        threads, passed = self.scan()
        self.assertNotIn(THREAD, threads)
        self.assertEqual(passed["aa.json"], ambiguous(THREAD, "'failing.json'"))

    def test_a_claim_of_the_same_length_with_one_byte_changed_is_an_ambiguity(self) -> None:
        healthy = (self.exports / "failing.json").read_bytes()
        changed = bytearray(healthy)
        at = healthy.index(b'"title": "') + len(b'"title": "')
        changed[at] = ord("X") if changed[at] != ord("X") else ord("Y")
        self.assertEqual(len(changed), len(healthy))
        self.assertNotEqual(bytes(changed), healthy)
        (self.exports / "aa.json").write_bytes(bytes(changed))
        threads, passed = self.scan()
        self.assertNotIn(THREAD, threads)
        self.assertEqual(passed["aa.json"], ambiguous(THREAD, "'failing.json'"))

    def test_a_copy_that_differs_only_in_its_title_is_an_ambiguity(self) -> None:
        retitled = json.loads(json.dumps(HEALTHY))
        retitled["thread"]["title"] = "decoy"
        self.write("zz.json", retitled)
        _, passed = self.scan()
        self.assertEqual(passed["zz.json"], ambiguous(THREAD, "'failing.json'"))

    def test_a_repeat_beside_a_claim_with_other_bytes_is_an_ambiguity(self) -> None:
        self.copy_of_the_healthy_export("copy.json")
        self.write("zz.json", DECOY)
        threads, passed = self.scan()
        self.assertNotIn(THREAD, threads)
        self.assertEqual(set(passed), {"copy.json", "failing.json", "zz.json"})
        self.assertEqual(passed["zz.json"], ambiguous(THREAD, "'copy.json', 'failing.json'", "do"))

    def test_a_long_thread_id_and_many_claims_make_a_bounded_reason(self) -> None:
        thread_id = "t" * 10_000
        for n in range(8):
            self.write(f"c{n}.json", {"thread": {"id": thread_id}, "elements": [{}] * n})
        _, passed = self.scan()
        reason = passed["c0.json"]
        self.assertLess(len(reason), 600)
        self.assertIn("... [", reason)
        self.assertIn("'c1.json', 'c2.json', 'c3.json', 'c4.json', 'c5.json', and 2 more", reason)

    def test_a_credential_in_a_claimed_thread_id_is_masked(self) -> None:
        token = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
        self.write("a.json", {"thread": {"id": token}, "elements": []})
        self.write("b.json", {"thread": {"id": token}, "elements": [{}]})
        _, passed = self.scan()
        self.assertNotIn(token, passed["a.json"])
        self.assertIn("[REDACTED:", passed["a.json"])

    def test_the_control_threads_with_their_own_ids_pass_nothing_over(self) -> None:
        self.write("other.json", {**DECOY, "thread": {"id": THREAD + "-other"}})
        threads, passed = self.scan()
        self.assertIn(THREAD, threads)
        self.assertIn(THREAD + "-other", threads)
        self.assertEqual(passed, {})

    def test_the_note_says_the_source_could_not_take_them_as_a_thread(self) -> None:
        note = passed_over_note([{"entry": "aa.json", "reason": "claims thread 't'"}])
        self.assertEqual(
            note,
            "discovery passed over 1 source entry it could not take as a thread, so they are "
            "in no count here: aa.json claims thread 't'",
        )


class ARunNamesEachClaimTests(unittest.TestCase):
    """``daily --commit`` and ``doctor`` with a claim sorted ahead of the healthy export."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        base = Path(cls._tmp.name).resolve()
        cls.exports = base / "exports"
        shutil.copytree(FIXTURE_EXPORTS, cls.exports)
        (cls.exports / "aa.json").write_text(json.dumps(DECOY), encoding="utf-8")
        cls.cfg = [
            "--config",
            str(build_env(base / "env", fixtures_dir=cls.exports, dry_run=False)),
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_receipt_names_both_claims_and_proposes_nothing_from_either(self) -> None:
        proc = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        receipt = json.loads(proc.stdout)
        entries = {Path(e["entry"]).name: e["reason"] for e in receipt["discovery"]["passed_over"]}
        self.assertEqual(
            entries,
            {
                "aa.json": ambiguous(THREAD, "'failing.json'"),
                "failing.json": ambiguous(THREAD, "'aa.json'"),
            },
        )
        self.assertEqual(receipt["counts"]["conversations_ingested"], 4)
        self.assertTrue(
            any("2 source entries it could not take as a thread" in n for n in receipt["notes"]),
            receipt["notes"],
        )

    def test_doctor_warns_of_each_claim(self) -> None:
        proc = run_cli([*self.cfg, "doctor"])
        warns = [line for line in proc.stdout.splitlines() if "source entry passed over" in line]
        self.assertEqual(len(warns), 2, proc.stdout)
        self.assertTrue(warns[0].endswith(f"aa.json {ambiguous(THREAD, repr('failing.json'))}"))


if __name__ == "__main__":
    unittest.main()
