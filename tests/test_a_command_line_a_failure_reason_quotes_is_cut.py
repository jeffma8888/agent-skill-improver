"""A command line a failure reason quotes is masked, then cut.

The conversation source quotes its command line into every failure reason (a
timeout, output over ``max_export_bytes``, a non-zero exit, output that is not
UTF-8), and a reason lands in each quarantine or deferral record, its
checkpoint, the receipt and the run's output, once per conversation when a
fetch fails. Measured before (3.11 and 3.13): with a 5,000-character option in
the command, a committed run over five conversations whose exports were not
UTF-8 made each quarantine reason 5,265 characters (5,306 on 3.11, whose
interpreter path is longer), and its receipt 44 KB where it was 9 KB without
the option. ``shown_command`` in ``adapters/conversation_cli.py`` now cuts
the masked line at ``SHOWN_COMMAND_CHARS`` and says how much was left.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent_skill_improver import cli
from agent_skill_improver.adapters import conversation_cli
from agent_skill_improver.adapters.conversation_cli import ConversationCliSource
from agent_skill_improver.domain.redaction import NAMED_PATH_CHARS
from agent_skill_improver.errors import RecordDefect, SourceUnavailable

from tests.support import FIXTURE_EXPORTS, build_env

#: Shaped like a provider key, so redaction masks it; split so no scanner
#: reads this file as holding one.
KEY = "sk-ant-" + "api03-" + "A1b2C3d4E5f6G7h8" * 4
LONG = "--note=" + "A" * 5000
THREAD = "1700000000001-1"

#: A conversation command: lists the fixture exports, and fails each export
#: the way its second argument says.
FAKE = """import json, os, sys, time
args = [a for a in sys.argv[1:] if not a.startswith("--")]
exports, how, mode = args[0], args[1], args[2]
if mode == "list" and how == "exit":
    sys.stderr.write("listing is down\\n"); sys.exit(2)
if mode == "list" and how == "sleep":
    time.sleep(5)
by_id = {}
for name in sorted(os.listdir(exports)):
    with open(os.path.join(exports, name), encoding="utf-8") as f:
        by_id[json.load(f)["thread"]["id"]] = name
if mode == "list":
    for n, (tid, name) in enumerate(sorted(by_id.items()), 1):
        print(f'{n}. "fixture {name}" (2026-01-30)')
        print(f"   Thread: {tid}")
elif how == "exit":
    sys.stderr.write("export is down\\n"); sys.exit(3)
elif how == "bad8":
    sys.stdout.buffer.write(b"\\xff\\xfe not utf-8")
elif how == "big":
    sys.stdout.write("x" * 4000)
else:
    with open(os.path.join(exports, by_id[args[3]]), encoding="utf-8") as f:
        sys.stdout.write(f.read())
"""


def cut_as_stated(line: str) -> str:
    """The cut the module states, computed here rather than by the helper."""
    return f"{line[:1024]}... [{len(line) - 1024} more chars not quoted]"


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.tmp)])
        self.exports = self.tmp / "exports"
        shutil.copytree(FIXTURE_EXPORTS, self.exports)
        self.script = self.tmp / "conv.py"
        self.script.write_text(FAKE, encoding="utf-8")

    def head(self, how: str, *extra: str) -> list[str]:
        return [sys.executable, str(self.script), str(self.exports), how, *extra]


class TheHelperTests(unittest.TestCase):
    def test_the_limit_is_room_for_a_named_path_and_as_much_again(self) -> None:
        self.assertEqual(conversation_cli.SHOWN_COMMAND_CHARS, 2 * NAMED_PATH_CHARS)
        self.assertEqual(conversation_cli.SHOWN_COMMAND_CHARS, 1024)

    def test_a_line_within_the_limit_is_quoted_as_written(self) -> None:
        self.assertEqual(
            conversation_cli.shown_command(["prog", "export", THREAD]), f"prog export {THREAD}"
        )
        at_limit = ["p" * 1000, "q" * 23]
        self.assertEqual(len(" ".join(at_limit)), 1024)
        self.assertEqual(conversation_cli.shown_command(at_limit), " ".join(at_limit))

    def test_a_line_over_the_limit_is_cut_saying_how_much_was_left(self) -> None:
        one_over = ["p" * 1000, "q" * 24]
        self.assertEqual(
            conversation_cli.shown_command(one_over),
            "p" * 1000 + " " + "q" * 23 + "... [1 more chars not quoted]",
        )
        argv = ["prog", LONG, "export", THREAD]
        self.assertEqual(conversation_cli.shown_command(argv), cut_as_stated(" ".join(argv)))
        self.assertEqual(len(" ".join(argv)), 5035)
        self.assertIn("... [4011 more chars not quoted]", conversation_cli.shown_command(argv))

    def test_a_key_across_the_cut_is_masked_whole_first(self) -> None:
        # A bare key with 15 characters before the cut: the key's first 15
        # characters on their own are too short for the shapes, so a cut made
        # before the masking would leave them raw. (An option such as
        # ``--token=`` would hide that: its value is masked whatever it holds.)
        argv = ["p" * 1008, KEY, LONG]
        self.assertEqual(" ".join(argv).index(KEY), 1009)
        self.assertGreater(1009 + len(KEY), 1024)
        shown = conversation_cli.shown_command(argv)
        self.assertNotIn("sk-ant", shown)
        self.assertTrue(shown.startswith("p" * 1008 + " [REDACTED:"), shown[1000:1040])
        early = conversation_cli.shown_command(["prog", "--token=" + KEY, LONG])
        self.assertIn("--token=[REDACTED:model-provider-key] --note=A", early)
        self.assertNotIn(KEY, early)


class TheSourceTests(_Case):
    def reason(self, how: str, call: str, *extra: str, **kwargs: int) -> tuple[str, type]:
        head = self.head(how, *extra)
        source = ConversationCliSource(
            discovery_argv=(*head, "list", "{since}", "{limit}"),
            export_argv=(*head, "export", "{thread_id}"),
            **kwargs,
        )
        with self.assertRaises((SourceUnavailable, RecordDefect)) as ctx:
            if call == "fetch":
                source.fetch(THREAD)
            else:
                source.discover_page(since="7d", limit=50)
        return str(ctx.exception), type(ctx.exception)

    CASES = (
        ("exit", "fetch", {}, SourceUnavailable, ["export", THREAD]),
        ("bad8", "fetch", {}, RecordDefect, ["export", THREAD]),
        ("big", "fetch", {"max_export_bytes": 100}, RecordDefect, ["export", THREAD]),
        ("exit", "discover", {}, SourceUnavailable, ["list", "7d", "50"]),
        ("sleep", "discover", {"timeout_seconds": 1}, SourceUnavailable, ["list", "7d", "50"]),
    )

    def test_every_failure_reason_quotes_a_long_command_line_cut(self) -> None:
        for how, call, kwargs, kind, tail in self.CASES:
            with self.subTest(how=how, call=call):
                reason, raised = self.reason(how, call, LONG, **kwargs)
                self.assertIs(raised, kind)
                line = " ".join([*self.head(how, LONG), *tail])
                self.assertTrue(reason.startswith(cut_as_stated(line) + " "), reason[:80])
                self.assertNotIn("A" * 4000, reason)
                self.assertLess(len(reason), 1024 + 200)

    def test_a_short_command_line_is_quoted_as_written(self) -> None:
        for how, call, kwargs, kind, tail in self.CASES:
            with self.subTest(how=how, call=call):
                reason, raised = self.reason(how, call, **kwargs)
                self.assertIs(raised, kind)
                line = " ".join([*self.head(how), *tail])
                self.assertLess(len(line), 1024)
                self.assertTrue(reason.startswith(line + " "), reason[:80])
                self.assertNotIn("not quoted", reason)


class TheRunTests(_Case):
    def run_daily(self, *extra: str) -> tuple[int, str, Path]:
        head = self.head("bad8", *extra)
        cfg = build_env(self.tmp / "env", fixtures_dir=self.exports, seed="c215")
        text = cfg.read_text(encoding="utf-8")
        fixtures_line = f"fixtures_dir = {json.dumps(str(self.exports))}\n"
        self.assertEqual(text.count('kind = "fixtures"\n'), 1)
        self.assertEqual(text.count(fixtures_line), 1)
        block = (
            'kind = "conversation-cli"\n'
            f"discovery_argv = {json.dumps([*head, 'list', '{since}', '{limit}'])}\n"
            f"export_argv = {json.dumps([*head, 'export', '{thread_id}'])}\n"
        )
        cfg.write_text(
            text.replace(fixtures_line, "").replace('kind = "fixtures"\n', block),
            encoding="utf-8",
        )
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = cli.main(["--config", str(cfg), "daily", "--commit", "--day", "2026-02-01"])
        return rc, out.getvalue(), self.tmp / "env" / "state"

    def test_a_run_over_failing_exports_keeps_each_reason_cut(self) -> None:
        rc, out, state = self.run_daily(LONG)
        self.assertEqual(rc, 0, out)
        self.assertIn("quarantined=5", out)
        self.assertNotIn("A" * 4000, out)
        self.assertEqual(out.count("more chars not quoted]"), 5)
        records = sorted(state.glob("quarantine/**/*.json"))
        self.assertEqual(len(records), 5)
        for path in records:
            reason = json.loads(path.read_text(encoding="utf-8"))["reason"]
            self.assertIn("... [", reason)
            self.assertLess(len(reason), 1024 + 200)
        # Not covered: the receipt's ``source`` block records the two configured
        # templates whole, as configured (masked, not cut), and nothing else does.
        receipts = sorted(state.glob("receipts/*.json"))
        self.assertEqual(len(receipts), 1)
        receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
        self.assertEqual(json.dumps(receipt).count("A" * 4000), 2)
        self.assertEqual(json.dumps(receipt["source"]).count("A" * 4000), 2)
        for path in state.rglob("*"):
            if path.is_file() and path not in receipts:
                self.assertNotIn(b"A" * 4000, path.read_bytes(), path)

    def test_the_control_a_run_with_a_short_command_quotes_it_whole(self) -> None:
        rc, out, state = self.run_daily()
        self.assertEqual(rc, 0, out)
        self.assertIn("quarantined=5", out)
        self.assertNotIn("not quoted", out)
        line = " ".join([*self.head("bad8"), "export"])
        for path in sorted(state.glob("quarantine/**/*.json")):
            self.assertIn(line + " ", json.loads(path.read_text(encoding="utf-8"))["reason"])


if __name__ == "__main__":
    unittest.main()
