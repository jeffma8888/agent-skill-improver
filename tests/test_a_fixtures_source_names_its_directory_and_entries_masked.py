"""A fixtures source names its directory, and what it passes over, masked.

The directory is the operator's ``[source].fixtures_dir``, as a conversation
command's program is the operator's, and since cycle c210 that program is
named masked. Measured before (3.11 and 3.13, the same on both), with a
provider-shaped key as a directory above it: ``doctor`` put it raw in its source
line and its problem when the directory was a file or could not be listed
(twice each, with ``--json`` too) and ``daily`` in its refusal; a fetch that
found no readable export named it, and the entries it passed over, in the line
and the record of each quarantined conversation, its checkpoint and the
receipt (30 times each in the output, the quarantine records and the
checkpoints for five conversations whose exports turned unreadable, and 31 in
the receipt); the receipt's ``source`` block recorded ``fixtures_dir`` raw; and
``doctor`` listed an export it passed over, under a key-shaped name or
directory, raw in its warning and in ``--json`` (the daily receipt has masked
those since cycle c213). Each is masked now; a plain directory reads as
before.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.config import SourceConfig
from agent_skill_improver.errors import RecordDefect, SourceUnavailable

from tests.support import FIXTURE_EXPORTS, build_env

#: Shaped like a provider key, so redaction masks it; split so no scanner
#: reads this file as holding one.
KEY = "sk-ant-" + "api03-" + "A1b2C3d4E5f6G7h8" * 4
NOT_JSON = "not json at all"


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.tmp)])
        self.addCleanup(subprocess.run, ["chmod", "-R", "u+rwx", str(self.tmp)])

    def exports(self, *parts: str) -> Path:
        exports = self.tmp.joinpath(*parts, "exports")
        shutil.copytree(FIXTURE_EXPORTS, exports)
        return exports

    def run_cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = cli.main(list(argv))
        return rc, out.getvalue()

    def state_files_holding_the_key(self, config: Path) -> list[Path]:
        state = config.parent.parent / "state"
        files = [p for p in state.rglob("*") if p.is_file()] if state.exists() else []
        return [p for p in files if KEY.encode() in p.read_bytes()]


class TheSourceTests(Base):
    def test_its_directory_is_named_masked_and_a_plain_one_as_written(self):
        keyed = FixtureConversationSource(self.tmp / KEY / "exports").named_directory()
        self.assertNotIn(KEY, keyed)
        self.assertEqual(keyed, f"{self.tmp}/[REDACTED:model-provider-key]/exports")
        plain = self.tmp / "plain" / "exports"
        self.assertEqual(FixtureConversationSource(plain).named_directory(), str(plain))
        broken = self.tmp / "two\nlines"
        self.assertEqual(
            FixtureConversationSource(broken).named_directory(), f"{self.tmp}/two\\nlines"
        )

    def test_a_directory_that_is_a_file_is_refused_masked(self):
        path = self.tmp / KEY / "exports"
        path.parent.mkdir()
        path.write_text("x", encoding="utf-8")
        with self.assertRaises(SourceUnavailable) as caught:
            FixtureConversationSource(path).preflight()
        self.assertNotIn(KEY, str(caught.exception))
        self.assertIn(
            "[REDACTED:model-provider-key]/exports is not a directory", str(caught.exception)
        )

    @unittest.skipIf(os.geteuid() == 0, "root lists a mode-000 directory")
    def test_a_directory_it_cannot_list_is_refused_masked(self):
        exports = self.exports(KEY)
        os.chmod(exports, 0)
        with self.assertRaises(SourceUnavailable) as caught:
            FixtureConversationSource(exports).preflight()
        self.assertNotIn(KEY, str(caught.exception))
        self.assertIn(
            "[REDACTED:model-provider-key]/exports cannot be listed", str(caught.exception)
        )

    def test_a_fetch_that_finds_no_export_names_the_directory_masked(self):
        exports = self.exports(KEY)
        for path in exports.iterdir():
            path.unlink()
        with self.assertRaises(RecordDefect) as caught:
            FixtureConversationSource(exports).fetch("1700000000001-1")
        self.assertNotIn(KEY, str(caught.exception))
        self.assertTrue(str(caught.exception).endswith("[REDACTED:model-provider-key]/exports"))

    def test_a_fetch_that_finds_no_readable_export_names_each_entry_masked(self):
        exports = self.exports("plain")
        named = exports / f"{KEY}.json"
        named.write_text(NOT_JSON, encoding="utf-8")
        for path in exports.iterdir():
            if path != named:
                path.unlink()
        with self.assertRaises(RecordDefect) as caught:
            FixtureConversationSource(exports).fetch("1700000000001-1")
        text = str(caught.exception)
        self.assertNotIn(KEY, text)
        self.assertIn(f"in {exports}; 1 entry there could not be taken", text)
        self.assertIn(f"may be it: {exports}/[REDACTED:model-provider-key].json is not", text)

    def test_the_source_record_masks_a_key_in_the_directory(self):
        record = SourceConfig(fixtures_dir=self.tmp / KEY / "exports").to_dict()
        self.assertEqual(
            record["fixtures_dir"], f"{self.tmp}/[REDACTED:model-provider-key]/exports"
        )
        plain = self.tmp / "plain" / "exports"
        self.assertEqual(SourceConfig(fixtures_dir=plain).to_dict()["fixtures_dir"], str(plain))
        self.assertEqual(SourceConfig().to_dict()["fixtures_dir"], "")


class TheCommandTests(Base):
    def test_doctor_and_daily_refuse_a_key_shaped_directory_masked(self):
        exports = self.exports(KEY)
        config = build_env(self.tmp / "env", fixtures_dir=exports, seed="c214")
        shutil.rmtree(exports)
        exports.write_text("x", encoding="utf-8")
        for argv, want in ((["doctor"], 1), (["--json", "doctor"], 1), (["daily", "--commit"], 3)):
            with self.subTest(argv=argv):
                rc, shown = self.run_cli("--config", str(config), *argv)
                self.assertEqual(rc, want, shown)
                self.assertNotIn(KEY, shown)
                self.assertIn("[REDACTED:model-provider-key]/exports is not a directory", shown)
        self.assertEqual(self.state_files_holding_the_key(config), [])

    def test_doctor_lists_an_export_it_passed_over_masked(self):
        cases = {
            "name": (self.exports("by-name"), f"{KEY}.json"),
            "directory": (self.exports("by-dir", KEY), "bad.json"),
        }
        for case, (exports, name) in cases.items():
            (exports / name).write_text(NOT_JSON, encoding="utf-8")
            config = build_env(self.tmp / f"env-{case}", fixtures_dir=exports, seed="c214")
            with self.subTest(case=case):
                rc, shown = self.run_cli("--config", str(config), "doctor")
                self.assertEqual(rc, 0, shown)
                self.assertNotIn(KEY, shown)
                warning = [
                    line for line in shown.splitlines() if "source entry passed over" in line
                ]
                self.assertEqual(len(warning), 1, shown)
                self.assertIn("[REDACTED:model-provider-key]", warning[0])
                rc, shown = self.run_cli("--config", str(config), "--json", "doctor")
                self.assertEqual(rc, 0, shown)
                self.assertNotIn(KEY, shown)
                listed = json.loads(shown)["source"]["passed_over"]
                self.assertEqual(len(listed), 1, listed)
                self.assertIn("[REDACTED:model-provider-key]", listed[0]["entry"])
                self.assertTrue(listed[0]["reason"].startswith("is not valid JSON"), listed)

    def test_the_control_doctor_lists_a_plain_export_as_written(self):
        exports = self.exports("plain")
        (exports / "bad.json").write_text(NOT_JSON, encoding="utf-8")
        config = build_env(self.tmp / "env", fixtures_dir=exports, seed="c214")
        rc, shown = self.run_cli("--config", str(config), "--json", "doctor")
        self.assertEqual(rc, 0, shown)
        listed = json.loads(shown)["source"]["passed_over"]
        self.assertEqual(
            listed,
            [
                {
                    "entry": str(exports / "bad.json"),
                    "reason": "is not valid JSON (Expecting value: line 1 column 1 (char 0))",
                }
            ],
        )
        rc, shown = self.run_cli("--config", str(config), "doctor")
        self.assertIn(f"source entry passed over: {exports / 'bad.json'} is not valid JSON", shown)

    def test_a_run_whose_exports_turn_unreadable_keeps_no_key(self):
        exports = self.exports(KEY)
        config = build_env(self.tmp / "env", fixtures_dir=exports, seed="c214")
        fetch = FixtureConversationSource.fetch

        def fetch_after_they_turn_unreadable(source, thread_id):
            for path in exports.iterdir():
                os.chmod(path, 0)
            return fetch(source, thread_id)

        with mock.patch.object(
            FixtureConversationSource, "fetch", fetch_after_they_turn_unreadable
        ):
            rc, shown = self.run_cli(
                "--config", str(config), "daily", "--commit", "--day", "2026-02-01"
            )
        for path in exports.iterdir():
            os.chmod(path, 0o644)
        self.assertEqual(rc, 0, shown)
        self.assertNotIn(KEY, shown)
        self.assertIn("quarantined=5", shown)
        quarantined = [line for line in shown.splitlines() if line.startswith("quarantined ")]
        self.assertEqual(len(quarantined), 5, shown)
        for line in quarantined:
            self.assertIn("no readable fixture", line)
            self.assertIn("[REDACTED:model-provider-key]/exports;", line)
        self.assertEqual(self.state_files_holding_the_key(config), [])


if __name__ == "__main__":
    unittest.main()
