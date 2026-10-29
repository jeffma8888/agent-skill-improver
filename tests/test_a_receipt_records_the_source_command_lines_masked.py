"""A receipt records the conversation command's templates with a key masked.

Every daily receipt carries a ``source`` block, and the four argv templates of
``[source]`` were copied into it as written. Measured before (3.11 and 3.13,
the same on both), a committed run through a working conversation command: a
key as an option value (``--api-key=...``), as the element after ``--api-key``,
or as the directory the command's script sits under was raw in the receipt once
per template that held it (twice, for ``discovery_argv`` and ``export_argv``;
once in ``query_argv``), while the run printed none of it and no other state
file held it. Each element is now masked as ``RoleCommand.to_dict`` masks a
model role's argv, a short token on its own too; a plain template is recorded
as written, and the command still runs with the template as written.
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
from agent_skill_improver.config import SourceConfig

from tests.support import FIXTURE_EXPORTS, build_env

#: Shaped like a provider key, so redaction masks it; split so no scanner
#: reads this file as holding one.
KEY = "sk-ant-" + "api03-" + "A1b2C3d4E5f6G7h8" * 4
#: The shortest shape the redactor knows, 24 characters, likewise split.
SHORT = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0"
FIELDS = ("discovery_argv", "query_argv", "workspace_argv", "export_argv")

#: A conversation command over the fixture exports that writes each argv it
#: was given to a file, so a test can see what it ran with. Options are not
#: positions: ``exports seen mode [thread]``.
FAKE_COMMAND = """\
import json, os, sys
with open(sys.argv[2], "a", encoding="utf-8") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
exports, mode = args[0], args[2]
by_id = {}
for name in sorted(os.listdir(exports)):
    with open(os.path.join(exports, name), encoding="utf-8") as f:
        by_id[json.load(f)["thread"]["id"]] = name
if mode == "list":
    for n, (tid, name) in enumerate(sorted(by_id.items()), 1):
        print(f'{n}. "fixture {name}" (2026-01-30)')
        print(f"   Thread: {tid}")
else:
    with open(os.path.join(exports, by_id[args[3]]), encoding="utf-8") as f:
        sys.stdout.write(f.read())
"""


class TheSourceRecordTests(unittest.TestCase):
    def test_a_key_as_an_option_value_is_masked_in_each_template(self):
        for field in FIELDS:
            with self.subTest(field=field):
                record = SourceConfig(**{field: ("prog", f"--api-key={KEY}", "{x}")}).to_dict()
                self.assertNotIn(KEY, json.dumps(record))
                self.assertEqual(len(record[field]), 3, record[field])
                self.assertEqual(record[field][0], "prog")
                self.assertTrue(record[field][1].startswith("--api-key=[REDACTED:"), record[field])
                self.assertTrue(record[field][1].endswith("]"), record[field])
                self.assertEqual(record[field][2], "{x}")
                for other in FIELDS:
                    if other != field:
                        self.assertEqual(record[other], list(getattr(SourceConfig(), other)))

    def test_a_key_as_its_own_element_or_as_a_directory_is_masked(self):
        for field in FIELDS:
            with self.subTest(field=field):
                template = (f"/opt/{KEY}/conv", "--api-key", KEY, "{thread_id}")
                recorded = SourceConfig(**{field: template}).to_dict()[field]
                self.assertNotIn(KEY, json.dumps(recorded))
                self.assertEqual(len(recorded), 4, recorded)
                self.assertTrue(recorded[0].startswith("/opt/[REDACTED:"), recorded)
                self.assertTrue(recorded[0].endswith("]/conv"), recorded)
                self.assertEqual(recorded[1], "--api-key")
                self.assertTrue(recorded[2].startswith("[REDACTED:"), recorded)
                self.assertEqual(recorded[3], "{thread_id}")

    def test_a_short_token_on_its_own_is_masked(self):
        self.assertEqual(len(SHORT), 24)
        for field in FIELDS:
            with self.subTest(field=field):
                recorded = SourceConfig(**{field: ("prog", SHORT, "{x}")}).to_dict()[field]
                self.assertNotIn(SHORT, json.dumps(recorded))
                self.assertEqual(len(recorded), 3, recorded)
                self.assertEqual(recorded[0], "prog")
                self.assertTrue(recorded[1].startswith("[REDACTED:"), recorded)
                self.assertEqual(recorded[2], "{x}")

    def test_the_control_a_plain_template_is_recorded_as_written(self):
        plain = ("/usr/bin/conv tool", "--since", "{since}", "--limit={limit}", "{query}", "über")
        for field in FIELDS:
            with self.subTest(field=field):
                recorded = SourceConfig(**{field: plain}).to_dict()[field]
                self.assertEqual(recorded, list(plain))
        defaults = SourceConfig()
        record = defaults.to_dict()
        for field in FIELDS:
            with self.subTest(default=field):
                self.assertEqual(record[field], list(getattr(defaults, field)))


class ACommittedDailyRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.tmp)])

    def daily(self, name: str, *, keyed: bool) -> tuple[int, str, Path, list[str], list[str]]:
        base = self.tmp / name
        exports = base / "exports"
        shutil.copytree(FIXTURE_EXPORTS, exports)
        script = base / (KEY if keyed else "cmd") / "conv.py"
        script.parent.mkdir(parents=True)
        script.write_text(FAKE_COMMAND, encoding="utf-8")
        seen = base / "seen.jsonl"
        config = build_env(base / "env", fixtures_dir=exports, seed="c212")
        head = [sys.executable, str(script), str(exports), str(seen)]
        extra = [f"--api-key={KEY}"] if keyed else []
        discovery = [*head, *extra, "list", "{since}", "{limit}"]
        export = [*head, *extra, "export", "{thread_id}"]
        text = config.read_text(encoding="utf-8")
        fixtures_line = f"fixtures_dir = {json.dumps(str(exports))}\n"
        self.assertEqual(text.count('kind = "fixtures"\n'), 1)
        self.assertEqual(text.count(fixtures_line), 1)
        text = text.replace(fixtures_line, "").replace(
            'kind = "fixtures"\n',
            'kind = "conversation-cli"\n'
            f"discovery_argv = {json.dumps(discovery)}\n"
            f"export_argv = {json.dumps(export)}\n",
        )
        config.write_text(text, encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = cli.main(["--config", str(config), "daily", "--commit", "--day", "2026-02-01"])
        self.assertTrue(seen.exists(), out.getvalue())
        return rc, out.getvalue(), base / "env", discovery, export

    def receipt(self, env: Path) -> tuple[bytes, dict]:
        receipts = sorted(env.rglob("receipts/*.json"))
        self.assertEqual(len(receipts), 1, receipts)
        raw = receipts[0].read_bytes()
        return raw, json.loads(raw)

    def test_a_committed_receipt_holds_no_key_from_the_source_templates(self):
        rc, shown, env, discovery, export = self.daily("keyed", keyed=True)
        self.assertEqual(rc, 0, shown)
        self.assertNotIn(KEY, shown)
        raw, receipt = self.receipt(env)
        self.assertEqual(receipt["status"], "ok", receipt["status"])
        self.assertNotIn(KEY.encode(), raw)
        for field, template in (("discovery_argv", discovery), ("export_argv", export)):
            with self.subTest(field=field):
                recorded = receipt["source"][field]
                self.assertEqual(len(recorded), len(template), recorded)
                self.assertIn("[REDACTED:", recorded[1])
                self.assertTrue(recorded[1].endswith("/conv.py"), recorded)
                self.assertTrue(recorded[4].startswith("--api-key=[REDACTED:"), recorded)
                unkeyed = [i for i, part in enumerate(template) if KEY not in part]
                self.assertEqual([recorded[i] for i in unkeyed], [template[i] for i in unkeyed])
        state = [p for p in (env / "state").rglob("*") if p.is_file()]
        self.assertTrue(state)
        self.assertEqual([p for p in state if KEY.encode() in p.read_bytes()], [])

    def test_the_command_still_runs_with_the_template_as_written(self):
        rc, shown, env, _, _ = self.daily("keyed", keyed=True)
        self.assertEqual(rc, 0, shown)
        lines = (env.parent / "seen.jsonl").read_text(encoding="utf-8").splitlines()
        calls = [json.loads(line) for line in lines]
        self.assertGreaterEqual(len(calls), 2, calls)
        for call in calls:
            self.assertIn(f"--api-key={KEY}", call)
        after_the_key = {call[call.index(f"--api-key={KEY}") + 1] for call in calls}
        self.assertEqual(after_the_key, {"list", "export"})

    def test_the_control_a_plain_template_is_in_the_receipt_as_written(self):
        rc, shown, env, discovery, export = self.daily("plain", keyed=False)
        self.assertEqual(rc, 0, shown)
        _, receipt = self.receipt(env)
        self.assertEqual(receipt["source"]["discovery_argv"], discovery)
        self.assertEqual(receipt["source"]["export_argv"], export)


if __name__ == "__main__":
    unittest.main()
