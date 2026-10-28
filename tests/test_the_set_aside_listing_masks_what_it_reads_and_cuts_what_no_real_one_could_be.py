"""The set-aside listing masks what it reads from a record and cuts what no real one could be.

``asi set-aside`` lists the quarantine, refusal and deferral records a run
wrote, and no integrity check has read them, so what it shows of one is
whatever was last written there. Measured before (3.11 and 3.13, the same on
both): a ``ghp_`` token written into a record's id, stage, reason or time, in
any of the three kinds, was printed whole in the text and in ``--json`` (24 of
24), and a mebibyte written into any of them made a listing of a mebibyte.
The four are now shown as ``listed_value`` shows a candidate's fields: masked,
then cut past what a real one can be, the cut saying how much it left out. A
record the tool wrote lists exactly as before.
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.app.state import SetAsideRecord, SetAsideScan, StateStore
from agent_skill_improver.cli import LISTED_RECORD_ID_CHARS, RC_OK
from agent_skill_improver.domain.discovery import MAX_THREAD_ID_LENGTH
from agent_skill_improver.domain.redaction import LISTED_SKILL_ID_CHARS

from tests.support import build_env, run_cli

TOKEN = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"
MASK = "[REDACTED:github-token]"
KINDS = {
    "quarantined": ("quarantine", "asi.quarantine.v1"),
    "refused": ("refusals", "asi.refusal.v1"),
    "deferred": ("deferred", "asi.deferred.v1"),
}
REAL = {
    "record_id": "conv-1",
    "stage": "score:instruction-compliance",
    "reason": "role 'scorer' exited 7: bbb",
    "at": "2026-01-02T03:04:05Z",
}
#: Written out rather than imported, so a limit changed in the code fails here.
LIMITS = {"record_id": 256, "stage": 4096, "reason": 4096, "at": 120}


def _cut(text: str, limit: int) -> str:
    return f"{text[:limit]}... [{len(text) - limit} more chars not shown]"


class _Listing(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.cfg = ["--config", str(build_env(self.base, seed="set-aside-bounds"))]
        self.assertEqual(run_cli([*self.cfg, "set-aside"]).returncode, RC_OK)

    def _plant(self, kind: str, **fields: str) -> None:
        directory, schema = KINDS[kind]
        record: dict[str, object] = {"schema": schema, **REAL, **fields}
        if kind == "deferred":
            record["attempts"] = 2
        path = self.base / "state" / directory / "planted.json"
        path.write_text(json.dumps(record), encoding="utf-8")

    def _clear(self) -> None:
        for directory, _ in KINDS.values():
            for path in (self.base / "state" / directory).glob("*.json"):
                path.unlink()

    def _list(self) -> tuple[list[str], list[dict[str, object]], str, str]:
        text = run_cli([*self.cfg, "set-aside"])
        data = run_cli([*self.cfg, "--json", "set-aside"])
        self.assertEqual(text.returncode, RC_OK, text.stderr)
        self.assertEqual(data.returncode, RC_OK, data.stderr)
        return (
            text.stdout.splitlines(),
            json.loads(data.stdout)["records"],
            text.stdout,
            data.stdout,
        )


class AValueReadFromARecordIsMaskedTests(_Listing):
    def test_a_credential_in_any_field_of_any_kind_is_masked_in_both_forms(self) -> None:
        for kind in KINDS:
            for field in REAL:
                with self.subTest(kind=kind, field=field):
                    self._clear()
                    self._plant(kind, **{field: f"a {TOKEN} b"})
                    lines, records, text, data = self._list()
                    self.assertNotIn(TOKEN, text)
                    self.assertNotIn(TOKEN, data)
                    self.assertEqual(records[0][field], f"a {MASK} b")
                    self.assertIn(f"a {MASK} b", lines[0])

    def test_a_token_the_cut_would_split_is_masked_whole(self) -> None:
        for kind in KINDS:
            for field, limit in LIMITS.items():
                with self.subTest(kind=kind, field=field):
                    self._clear()
                    # The space keeps the token a word of its own: one glued
                    # to a letter is taken for part of a longer word.
                    value = "a" * (limit - 17) + " " + TOKEN
                    self._plant(kind, **{field: value})
                    lines, records, text, data = self._list()
                    shown = "a" * (limit - 17) + " " + MASK
                    self.assertEqual(records[0][field], _cut(shown, limit))
                    self.assertNotIn("gh" + "p_", text)
                    self.assertNotIn("gh" + "p_", data)


class AValueNoRealRecordHoldsIsCutTests(_Listing):
    def test_a_mebibyte_in_any_field_of_any_kind_makes_a_short_listing(self) -> None:
        big = "x" * (1 << 20)
        for kind in KINDS:
            for field, limit in LIMITS.items():
                with self.subTest(kind=kind, field=field):
                    self._clear()
                    self._plant(kind, **{field: big})
                    lines, records, text, data = self._list()
                    self.assertEqual(records[0][field], _cut(big, limit))
                    self.assertIn(_cut(big, limit), lines[0])
                    self.assertLess(len(text.encode()), 5_000)
                    self.assertLess(len(data.encode()), 5_000)

    def test_a_value_as_long_as_its_limit_is_whole_and_one_longer_is_cut(self) -> None:
        for field, limit in LIMITS.items():
            with self.subTest(field=field):
                for length in (limit, limit + 1):
                    self._clear()
                    value = "y" * length
                    self._plant("quarantined", **{field: value})
                    _, records, _, _ = self._list()
                    expected = value if length == limit else _cut(value, limit)
                    self.assertEqual(records[0][field], expected)

    def test_the_record_ids_limit_is_the_longest_a_real_id_can_be(self) -> None:
        # A conversation's id is held to MAX_THREAD_ID_LENGTH where it is
        # discovered and a skill's is a directory name, so neither is cut.
        self.assertEqual(LISTED_RECORD_ID_CHARS, MAX_THREAD_ID_LENGTH)
        self.assertGreaterEqual(LISTED_RECORD_ID_CHARS, LISTED_SKILL_ID_CHARS)
        store = StateStore(self.base / "state")
        thread_id = "t" * MAX_THREAD_ID_LENGTH
        store.quarantine(thread_id, stage="ingest", reason="r")
        lines, records, _, _ = self._list()
        self.assertEqual(records[0]["record_id"], thread_id)
        self.assertIn(f" {thread_id!r} at ingest: r", lines[0])

    def test_a_token_in_an_id_the_store_files_is_masked_in_the_path_too(self) -> None:
        # The file name keeps the first 96 characters of the id it is keyed on.
        store = StateStore(self.base / "state")
        written = store.quarantine(TOKEN, stage="ingest", reason="r")
        self.assertIn(TOKEN, written.name)
        lines, records, text, data = self._list()
        self.assertNotIn(TOKEN, text)
        self.assertNotIn(TOKEN, data)
        self.assertEqual(records[0]["record_id"], MASK)
        self.assertIn(MASK, str(records[0]["path"]))
        self.assertEqual(
            str(records[0]["path"]).replace(MASK, TOKEN), str(written), "only the token is masked"
        )


class ThePathIsCutPastWhatASystemCallTakesTests(_Listing):
    def test_a_path_as_long_as_its_limit_is_whole_and_one_longer_is_cut(self) -> None:
        # No file the store can open has a path this long, so the listing is
        # handed one: the scan is patched and the command runs in process.
        limit = 4096
        for length in (limit, limit + 1):
            with self.subTest(length=length):
                path = Path("/" + "p" * (length - 1))
                self.assertEqual(len(str(path)), length)
                record = SetAsideRecord(
                    kind="quarantined",
                    record_id="conv-1",
                    stage="ingest",
                    reason="r",
                    at="2026-01-02T03:04:05Z",
                    attempts=None,
                    path=path,
                )
                out = io.StringIO()
                scan = SetAsideScan(readable=(record,), unreadable=())
                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.object(StateStore, "scan_set_aside", return_value=scan)
                    )
                    stack.enter_context(contextlib.redirect_stdout(out))
                    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    rc = cli.main([*self.cfg, "--json", "set-aside"])
                self.assertEqual(rc, RC_OK)
                listed = json.loads(out.getvalue())["records"][0]["path"]
                expected = str(path) if length == limit else _cut(str(path), limit)
                self.assertEqual(listed, expected)


class TheControlARealRecordListsAsBeforeTests(_Listing):
    def test_the_control_records_the_tool_wrote_list_byte_for_byte_as_before(self) -> None:
        store = StateStore(self.base / "state")
        store.quarantine(
            "conv-1",
            stage="score:instruction-compliance",
            reason="the scorecard has schema 'wrong.v9'",
        )
        store.refuse("tdd", stage="propose", reason="skill 'tdd' already has an open candidate")
        store.defer("conv-2", stage="ingest", reason="role 'scorer' exited 7: bbb")
        store.defer("line\nbreak", stage="ingest", reason="a\nb")
        lines, records, _, _ = self._list()
        self.assertEqual(
            sorted(line[21:] for line in lines[:-1]),
            [
                "deferred    'conv-2' at ingest (attempt 1): role 'scorer' exited 7: bbb",
                "deferred    'line\\nbreak' at ingest (attempt 1): a\\nb",
                "quarantined 'conv-1' at score:instruction-compliance: "
                "the scorecard has schema 'wrong.v9'",
                "refused     'tdd' at propose: skill 'tdd' already has an open candidate",
            ],
        )
        self.assertTrue(all(len(line[:20].strip()) == 20 for line in lines[:-1]), lines)
        written = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for directory, _ in KINDS.values()
            for path in (self.base / "state" / directory).glob("*.json")
        }
        for record in records:
            raw = written[Path(str(record["path"])).name]
            self.assertEqual(
                set(record), {"kind", "record_id", "stage", "reason", "at", "attempts", "path"}
            )
            for field in REAL:
                self.assertEqual(record[field], raw[field])
            self.assertEqual(record["attempts"], raw.get("attempts"))
            self.assertTrue(Path(str(record["path"])).is_file(), record["path"])

    def test_the_control_a_record_with_no_time_lists_a_question_mark(self) -> None:
        directory, schema = KINDS["refused"]
        record = {"schema": schema, "record_id": "conv-1", "stage": "ingest", "reason": "r"}
        path = self.base / "state" / directory / "no-time.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        lines, records, _, _ = self._list()
        self.assertEqual(lines[0], "?" + " " * 20 + "refused     'conv-1' at ingest: r")
        self.assertEqual(records[0]["at"], "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
