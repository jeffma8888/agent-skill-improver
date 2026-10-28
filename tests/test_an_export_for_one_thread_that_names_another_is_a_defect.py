"""An export for one thread that names another is a defect, not that thread.

The conversation command source (``adapters/conversation_cli.py``) runs the
export command with the id it listed and normalized whatever came back, and the
normalizer takes the thread id from the export. Nothing compared the two.
Measured before (3.11 and 3.13, the same on both), with a conversation command
over the five fixture exports that answers one thread's export with another's:
asked for the failing thread and answered with the compliant one, ``daily
--commit`` ingested 5, found 0 failure cases and proposed nothing, where the
same command answering each thread with its own export gives 2 and 1; asked for
the compliant thread and answered with the failing one, it found 4 failure
cases and proposed one candidate addressing the failing thread's two failures
twice each. Every record read as scored either way. ``fetch`` now refuses an
export that names a thread other than the one it was asked for, as a record
defect naming both, so the record is quarantined at ingest.
"""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.adapters.conversation_cli import ConversationCliSource
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.errors import RecordDefect
from tests.support import FIXTURE_EXPORTS, build_env, run_cli

COMPLIANT = "1700000000001-1"
FAILING = "1700000000002-1"
OTHER = (
    "the conversation command answered for another conversation, "
    "so this export was not taken as either one"
)


def refusal(asked: str, named: str) -> str:
    return f"export for {asked} names thread {named!r}; {OTHER}"


def export_naming(thread_id: str) -> str:
    payload = json.loads((FIXTURE_EXPORTS / "compliant.json").read_text(encoding="utf-8"))
    payload["thread"]["id"] = thread_id
    return json.dumps(payload)


class _Answers(ConversationCliSource):
    """A conversation command that answers every export with ``raw``."""

    def __init__(self, raw: str, **kwargs):
        super().__init__(**kwargs)
        self.raw = raw

    def _run(self, argv, **kwargs):  # type: ignore[override]
        return self.raw


class FetchTakesAnExportOnlyAsTheThreadAskedForTests(unittest.TestCase):
    def refused(self, asked: str, named: str) -> RecordDefect:
        with self.assertRaises(RecordDefect) as ctx:
            _Answers(export_naming(named)).fetch(asked)
        return ctx.exception

    def test_an_export_naming_another_thread_is_a_defect_naming_both(self) -> None:
        exc = self.refused(FAILING, COMPLIANT)
        self.assertEqual(str(exc), refusal(FAILING, COMPLIANT))
        self.assertEqual(exc.record_id, FAILING)

    def test_a_near_miss_id_is_another_thread(self) -> None:
        for named in (
            "THREAD-A",
            "thread-a0",
            "thread-",
            " thread-a",
            "thread-a ",
            "thread-a\u200b",
        ):
            with self.subTest(named=named):
                self.assertEqual(str(self.refused("thread-a", named)), refusal("thread-a", named))

    def test_a_line_break_in_the_named_id_stays_on_one_line(self) -> None:
        message = str(self.refused(FAILING, "b\nWARN forged line"))
        self.assertNotIn("\n", message)
        self.assertIn("'b\\nWARN forged line'", message)

    def test_a_long_named_id_makes_a_bounded_reason(self) -> None:
        message = str(self.refused(FAILING, "b" * 1_000_000))
        self.assertLess(len(message), 400)
        self.assertIn("more chars not quoted", message)
        self.assertTrue(message.startswith(f"export for {FAILING} names thread 'bbb"))

    def test_a_credential_in_the_named_id_is_masked(self) -> None:
        token = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
        message = str(self.refused(FAILING, token))
        self.assertNotIn(token, message)
        self.assertIn("[REDACTED:", message)

    def test_the_raw_export_is_still_kept_under_the_id_asked_for(self) -> None:
        raw = export_naming(COMPLIANT)
        with TemporaryDirectory() as tmp:
            state = StateStore(Path(tmp).resolve() / "state")
            state.ensure()
            with self.assertRaises(RecordDefect):
                _Answers(raw, raw_store=state).fetch(FAILING)
            written = list((Path(tmp).resolve() / "state" / "raw").glob("*.json"))
            self.assertEqual(len(written), 1)
            self.assertTrue(written[0].name.startswith(FAILING))
            self.assertEqual(written[0].read_text(encoding="utf-8"), raw)

    def test_the_control_an_export_naming_the_thread_asked_for_is_taken(self) -> None:
        conversation = _Answers(export_naming(FAILING)).fetch(FAILING)
        self.assertEqual(conversation.thread_id, FAILING)
        self.assertTrue(conversation.messages)


#: A conversation command over a directory of exports. ``swap.json`` maps an id
#: asked for to the export file it is answered with.
FAKE_COMMAND = """\
import json, os, sys
exports, swapfile, mode = sys.argv[1], sys.argv[2], sys.argv[3]
by_id = {}
for name in sorted(os.listdir(exports)):
    with open(os.path.join(exports, name), encoding="utf-8") as f:
        by_id[json.load(f)["thread"]["id"]] = name
with open(swapfile, encoding="utf-8") as f:
    swap = json.load(f)
if mode == "list":
    for n, (tid, name) in enumerate(sorted(by_id.items()), 1):
        print(f'{n}. "fixture {name}" (2026-01-30)')
        print(f"   Thread: {tid}")
else:
    name = swap.get(sys.argv[4], by_id[sys.argv[4]])
    with open(os.path.join(exports, name), encoding="utf-8") as f:
        sys.stdout.write(f.read())
"""


class TheDailyRunQuarantinesTheRecordTests(unittest.TestCase):
    def daily(self, swap: dict[str, str]) -> tuple[dict, list[dict]]:
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            exports = base / "exports"
            shutil.copytree(FIXTURE_EXPORTS, exports)
            command = base / "conversation_command.py"
            command.write_text(FAKE_COMMAND, encoding="utf-8")
            swapfile = base / "swap.json"
            swapfile.write_text(json.dumps(swap), encoding="utf-8")
            config = build_env(base / "env", fixtures_dir=exports, seed="c175")
            head = [sys.executable, str(command), str(exports), str(swapfile)]
            text = config.read_text(encoding="utf-8")
            fixtures_line = f"fixtures_dir = {json.dumps(str(exports))}\n"
            self.assertEqual(text.count('kind = "fixtures"\n'), 1)
            self.assertEqual(text.count(fixtures_line), 1)
            text = text.replace(fixtures_line, "").replace(
                'kind = "fixtures"\n',
                'kind = "conversation-cli"\n'
                f"discovery_argv = {json.dumps([*head, 'list', '{since}', '{limit}'])}\n"
                f"export_argv = {json.dumps([*head, 'export', '{thread_id}'])}\n",
            )
            config.write_text(text, encoding="utf-8")
            proc = run_cli(
                ["--config", str(config), "--json", "daily", "--commit", "--day", "2026-02-01"]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            candidates = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((base / "env").rglob("candidates/*.json"))
            ]
            return json.loads(proc.stdout), candidates

    def test_a_failing_thread_answered_with_a_passing_one_is_quarantined_not_scored_clean(
        self,
    ) -> None:
        receipt, candidates = self.daily({FAILING: "compliant.json"})
        record = receipt["records"][FAILING]
        self.assertEqual(
            (record["status"], record["stage"], record["reason"]),
            ("quarantined", "ingest", refusal(FAILING, COMPLIANT)),
        )
        self.assertEqual(receipt["records"][COMPLIANT]["status"], "scored")
        self.assertEqual(receipt["counts"]["conversations_ingested"], 4)
        self.assertEqual(candidates, [])

    def test_a_passing_thread_answered_with_a_failing_one_does_not_count_its_failures_twice(
        self,
    ) -> None:
        receipt, candidates = self.daily({COMPLIANT: "failing.json"})
        record = receipt["records"][COMPLIANT]
        self.assertEqual(
            (record["status"], record["stage"], record["reason"]),
            ("quarantined", "ingest", refusal(COMPLIANT, FAILING)),
        )
        self.assertEqual(receipt["counts"]["failure_cases"], 2)
        self.assertEqual(len(candidates), 1)
        addressed = [r["conversation_id"] for r in candidates[0]["rubric_failures"]]
        self.assertEqual(addressed, [FAILING, FAILING])
        self.assertEqual(len({r["failure_case_id"] for r in candidates[0]["rubric_failures"]}), 2)

    def test_the_control_a_command_answering_each_thread_with_its_own_export(self) -> None:
        receipt, candidates = self.daily({})
        self.assertEqual({r["status"] for r in receipt["records"].values()}, {"scored"})
        self.assertEqual(receipt["counts"]["conversations_ingested"], 5)
        self.assertEqual(receipt["counts"]["failure_cases"], 2)
        self.assertEqual(len(candidates), 1)


if __name__ == "__main__":
    unittest.main()
