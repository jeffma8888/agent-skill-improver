"""A configured query and workspace are masked wherever a run records or prints them.

``[source].query`` and each of ``[source].workspaces`` are put into the
conversation command's line, which a refusal quotes masked, and since cycle
c212 the receipt masks the command's templates; the values themselves were
recorded as written. Measured before (3.11 and 3.13, the same on both): a key
as the one workspace was raw 4 times in the receipt of a committed run that
went well (once in ``source``, 3 times in ``discovery``), 5 times when the page
was cut by the discovery limit or its discovery failed (the 5th in the note the
run prints and keeps), and printed by ``discover --workspace``; a key as the
query was raw twice in every receipt. An entry discovery passed over (a
listing header with no Thread: line) named the workspace raw, twice more in the
receipt's ``discovery`` block and once in its notes, and printed it; a token
as that header's title was raw the same. Both are now masked, each passed-over
entry whole, after entries are told apart; a plain workspace and query read as
before, and the command still gets both as written.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent_skill_improver import cli
from agent_skill_improver.adapters.conversation_cli import ConversationCliSource
from agent_skill_improver.config import SourceConfig
from agent_skill_improver.domain import discovery

from tests.support import FIXTURE_EXPORTS, build_env

#: Shaped like a provider key, so redaction masks it; split so no scanner
#: reads this file as holding one.
KEY = "sk-ant-" + "api03-" + "A1b2C3d4E5f6G7h8" * 4
#: Another of the same shape, so the two mask alike.
KEY2 = "sk-ant-" + "api03-" + "Z9y8X7w6V5u4T3s2" * 4
#: A GitHub token, short enough that a listing title holding it is not cut.
SHORT = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0"

#: A conversation command over the fixture exports that writes each argv it
#: was given to a file, and whose listing fails, or ends with a header that has
#: no Thread: line, when told to. Options are not positions:
#: ``exports seen ok|fail|orphan mode [thread]``.
FAKE_COMMAND = """\
import json, os, sys
with open(sys.argv[2], "a", encoding="utf-8") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
exports, fail, mode = args[0], args[2], args[3]
if fail == "fail" and mode == "list":
    sys.stderr.write("listing is down\\n")
    sys.exit(2)
by_id = {}
for name in sorted(os.listdir(exports)):
    with open(os.path.join(exports, name), encoding="utf-8") as f:
        by_id[json.load(f)["thread"]["id"]] = name
if mode == "list":
    for n, (tid, name) in enumerate(sorted(by_id.items()), 1):
        print(f'{n}. "fixture {name}" (2026-01-30)')
        print(f"   Thread: {tid}")
    if fail == "orphan":
        print('99. "orphan" (2026-01-30)')
else:
    with open(os.path.join(exports, by_id[args[4]]), encoding="utf-8") as f:
        sys.stdout.write(f.read())
"""


class TheRecordsTests(unittest.TestCase):
    def test_a_workspace_is_shown_masked_and_none_is_shown_as_all(self):
        shown = discovery.shown_workspace(KEY)
        self.assertNotIn(KEY, shown)
        self.assertTrue(shown.startswith("[REDACTED:"), shown)
        self.assertEqual(discovery.shown_workspace(""), "<all>")
        self.assertEqual(discovery.shown_workspace("team-a"), "team-a")

    def test_a_page_masks_a_key_as_its_workspace_or_query_in_each_note(self):
        pages = {
            "failed": discovery.error_page(limit=5, workspace=KEY, query=KEY, error="it broke"),
            "cut": discovery.DiscoveryPage(
                threads=(), limit=5, workspace=KEY, query=KEY, truncated_by_limit=True
            ),
            "whole": discovery.DiscoveryPage(threads=(), limit=5, workspace=KEY, query=KEY),
        }
        for kind, page in pages.items():
            with self.subTest(page=kind):
                record = page.to_dict()
                self.assertNotIn(KEY, json.dumps(record))
                self.assertNotIn(KEY, page.honesty_note)
                self.assertTrue(record["workspace"].startswith("[REDACTED:"), record)
                self.assertTrue(record["query"].startswith("[REDACTED:"), record)
                self.assertIn("workspace=", page.honesty_note)
                self.assertIn("[REDACTED:", page.honesty_note)

    def test_the_control_a_plain_page_reads_as_before(self):
        failed = discovery.error_page(
            limit=5, workspace="team-a", query="is:open", error="it broke"
        )
        self.assertEqual(failed.to_dict()["workspace"], "team-a")
        self.assertEqual(failed.to_dict()["query"], "is:open")
        self.assertTrue(failed.honesty_note.startswith("discovery for workspace='team-a' failed"))
        cut = discovery.DiscoveryPage(
            threads=(), limit=5, workspace="team-a", truncated_by_limit=True
        )
        self.assertIn("for workspace=team-a: this is a PAGE", cut.honesty_note)
        whole = discovery.DiscoveryPage(threads=(), limit=5)
        self.assertEqual(whole.to_dict()["workspace"], "<all>")
        self.assertIn("for workspace=<all>: fewer than the limit", whole.honesty_note)

    def test_the_source_record_masks_a_key_as_the_query_or_a_workspace(self):
        record = SourceConfig(query=KEY, workspaces=("team-a", KEY)).to_dict()
        self.assertNotIn(KEY, json.dumps(record))
        self.assertTrue(record["query"].startswith("[REDACTED:"), record)
        self.assertEqual(len(record["workspaces"]), 2, record)
        self.assertEqual(record["workspaces"][0], "team-a")
        self.assertTrue(record["workspaces"][1].startswith("[REDACTED:"), record)
        plain = SourceConfig(query="is:open label", workspaces=("team-a", "team b")).to_dict()
        self.assertEqual(plain["query"], "is:open label")
        self.assertEqual(plain["workspaces"], ["team-a", "team b"])

    def test_a_passed_over_entry_is_recorded_masked_and_two_keys_stay_two(self):
        listings = {}

        class Listing(ConversationCliSource):
            def _run(self, argv, **kwargs):  # type: ignore[override]
                return listings["now"]

        source = Listing()
        listings["now"] = '1. "x" (2026-01-02)\n'
        pages = [source.discover_page(since="1d", limit=5, workspace=w) for w in (KEY, KEY2)]
        for page in pages:
            entry = page.to_dict()["passed_over"][0]["entry"]
            self.assertEqual(
                entry, "listing entry 1 ('x') in workspace [REDACTED:model-provider-key]"
            )
        _, summary = discovery.merge_pages(pages)
        entries = [item["entry"] for item in summary["passed_over"]]
        self.assertEqual(len(entries), 2, entries)
        note = discovery.passed_over_note(summary["passed_over"])
        self.assertIn("passed over 2 source entries", note)
        for text in (json.dumps(summary), note):
            self.assertNotIn(KEY, text)
            self.assertNotIn(KEY2, text)
        # A token as the listed title is masked as well; a plain workspace is not.
        listings["now"] = f'1. "{SHORT}" (2026-01-02)\n'
        page = source.discover_page(since="1d", limit=5, workspace="team-a")
        self.assertEqual(
            page.to_dict()["passed_over"][0]["entry"],
            "listing entry 1 ('[REDACTED:github-token]') in workspace team-a",
        )
        _, summary = discovery.merge_pages([page])
        self.assertNotIn(SHORT, json.dumps(summary))


class ARunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.tmp)])

    def config(
        self, name: str, *, workspace: str, query: str, fail: bool = False, orphan: bool = False
    ) -> Path:
        base = self.tmp / name
        exports = base / "exports"
        shutil.copytree(FIXTURE_EXPORTS, exports)
        script = base / "conv.py"
        script.write_text(FAKE_COMMAND, encoding="utf-8")
        config = build_env(base / "env", fixtures_dir=exports, seed="c213")
        head = [sys.executable, str(script), str(exports), str(base / "seen.jsonl")]
        head.append("fail" if fail else "orphan" if orphan else "ok")
        text = config.read_text(encoding="utf-8")
        fixtures_line = f"fixtures_dir = {json.dumps(str(exports))}\n"
        self.assertEqual(text.count('kind = "fixtures"\n'), 1)
        self.assertEqual(text.count(fixtures_line), 1)
        self.assertEqual(len(re.findall(r"(?m)^discovery_limit = \d+$", text)), 1)
        text = text.replace(fixtures_line, "").replace(
            'kind = "fixtures"\n',
            'kind = "conversation-cli"\n'
            f"discovery_argv = {json.dumps([*head, 'list', '{since}', '{limit}'])}\n"
            f"export_argv = {json.dumps([*head, 'export', '{thread_id}'])}\n"
            f"workspace_argv = {json.dumps(['--w={workspace}'])}\n"
            f"workspaces = {json.dumps([workspace])}\n"
            f"query_argv = {json.dumps(['--q={query}'])}\n"
            f"query = {json.dumps(query)}\n",
        )
        text = re.sub(r"(?m)^discovery_limit = \d+$", "discovery_limit = 2", text)
        config.write_text(text, encoding="utf-8")
        return config

    def run_cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = cli.main(list(argv))
        return rc, out.getvalue()

    def receipt(self, config: Path) -> dict:
        receipts = sorted(config.parent.parent.rglob("receipts/*.json"))
        self.assertEqual(len(receipts), 1, receipts)
        return json.loads(receipts[0].read_text(encoding="utf-8"))

    def assert_no_state_holds_the_key(self, config: Path) -> None:
        state = [p for p in (config.parent.parent / "state").rglob("*") if p.is_file()]
        self.assertTrue(state)
        self.assertEqual([p for p in state if KEY.encode() in p.read_bytes()], [])

    def test_a_cut_page_keeps_no_key_from_the_workspace_or_the_query(self):
        config = self.config("cut", workspace=KEY, query=KEY)
        rc, shown = self.run_cli(
            "--config", str(config), "daily", "--commit", "--day", "2026-02-01"
        )
        self.assertEqual(rc, 0, shown)
        self.assertNotIn(KEY, shown)
        self.assertIn("for workspace=[REDACTED:", shown)
        self.assert_no_state_holds_the_key(config)
        receipt = self.receipt(config)
        self.assertTrue(receipt["discovery"]["workspaces_queried"][0].startswith("[REDACTED:"))
        self.assertTrue(any("for workspace=[REDACTED:" in note for note in receipt["notes"]))
        seen = config.parent.parent.parent / "seen.jsonl"
        calls = [json.loads(line) for line in seen.read_text(encoding="utf-8").splitlines()]
        listing = [call for call in calls if "list" in call]
        self.assertTrue(listing, calls)
        for call in listing:
            self.assertIn(f"--w={KEY}", call)
            self.assertIn(f"--q={KEY}", call)

    def test_a_failed_discovery_names_the_workspace_masked(self):
        config = self.config("failed", workspace=KEY, query=KEY, fail=True)
        rc, shown = self.run_cli(
            "--config", str(config), "daily", "--commit", "--day", "2026-02-01"
        )
        self.assertEqual(rc, 0, shown)
        self.assertNotIn(KEY, shown)
        self.assertIn("discovery for workspace='[REDACTED:", shown)
        self.assert_no_state_holds_the_key(config)
        self.assertEqual(self.receipt(config)["status"], "stood-down")

    def test_a_run_keeps_no_key_from_an_entry_discovery_passed_over(self):
        config = self.config("orphan", workspace=KEY, query="", orphan=True)
        rc, shown = self.run_cli(
            "--config", str(config), "daily", "--commit", "--day", "2026-02-01"
        )
        self.assertEqual(rc, 0, shown)
        self.assertNotIn(KEY, shown)
        self.assertIn("in workspace [REDACTED:model-provider-key]", shown)
        self.assert_no_state_holds_the_key(config)
        receipt = self.receipt(config)
        self.assertEqual(
            [item["entry"] for item in receipt["discovery"]["passed_over"]],
            ["listing entry 99 ('orphan') in workspace [REDACTED:model-provider-key]"],
        )
        rc, shown = self.run_cli("--config", str(config), "--json", "discover", "--workspace", KEY)
        self.assertEqual(rc, 0, shown)
        self.assertNotIn(KEY, shown)
        self.assertEqual(
            json.loads(shown)["page"]["passed_over"][0]["entry"],
            "listing entry 99 ('orphan') in workspace [REDACTED:model-provider-key]",
        )

    def test_discover_names_a_workspace_it_is_given_masked(self):
        config = self.config("discover", workspace="team-a", query="")
        for fmt in ((), ("--json",)):
            with self.subTest(fmt=fmt):
                args = ("--config", str(config), *fmt, "discover", "--workspace")
                rc, shown = self.run_cli(*args, KEY)
                self.assertEqual(rc, 0, shown)
                self.assertNotIn(KEY, shown)
                self.assertIn("workspace=[REDACTED:", shown)
                rc, shown = self.run_cli(*args, "team-a")
                self.assertEqual(rc, 0, shown)
                self.assertIn("workspace=team-a: this is a PAGE", shown)

    def test_the_control_a_plain_workspace_and_query_are_recorded_as_written(self):
        config = self.config("plain", workspace="team-a", query="is:open")
        rc, shown = self.run_cli(
            "--config", str(config), "daily", "--commit", "--day", "2026-02-01"
        )
        self.assertEqual(rc, 0, shown)
        self.assertIn("for workspace=team-a: this is a PAGE", shown)
        receipt = self.receipt(config)
        self.assertEqual(receipt["source"]["workspaces"], ["team-a"])
        self.assertEqual(receipt["source"]["query"], "is:open")
        self.assertEqual(receipt["discovery"]["workspaces_queried"], ["team-a"])
        self.assertEqual(
            [(p["workspace"], p["query"]) for p in receipt["discovery"]["pages"]],
            [("team-a", "is:open")],
        )


if __name__ == "__main__":
    unittest.main()
