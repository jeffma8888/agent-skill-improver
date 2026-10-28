"""A replacement is diffed with the line endings apply writes.

The store reads a skill document with its line endings folded to LF, and
``write_canonical`` writes the replacement folded the same way, in the
document's own convention (CRLF throughout stays CRLF). ``propose_candidate``
in ``app/pipeline.py`` now folds the improver's replacement before it is
checked, diffed and replayed. Measured before (3.11 and 3.13, ``daily --commit``
with an improver that changes the fake model's line endings): a CRLF reply,
against a CRLF or an LF document, made a diff of 12 added and 10 removed lines
for a change of two added lines, and a lone carriage return put a
"\\ No newline at end of file" marker in the middle of the diff, after the
line it ended; apply wrote the folded text in each case, so for an LF
document, and for the lone carriage return, the bytes on disk differed from
the replacement the reviewer approved; and a reply whose every line ended in a
lone carriage return was quarantined at propose as having no opening '---'
delimiter, because the frontmatter check read it unfolded.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.support import REPO_ROOT, build_env, run_cli

MARKER = "\\ No newline at end of file"


def improver_script(case: str) -> str:
    return (
        "import json, subprocess, sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r})\n"
        "from agent_skill_improver.adapters.model_command import extract_json_object\n"
        f"fake = {str(REPO_ROOT / 'tests' / 'fake_model.py')!r}\n"
        f"case = {case!r}\n"
        "raw = sys.stdin.read()\n"
        "out = subprocess.run([sys.executable, fake, *sys.argv[1:]], input=raw,"
        " capture_output=True, text=True).stdout\n"
        "if json.loads(raw).get('role') == 'improver' and case != 'control':\n"
        "    obj = extract_json_object(out)\n"
        "    text = obj['new_skill_text'].replace('\\r\\n', '\\n')\n"
        "    if case == 'bare_cr':\n"
        "        text = text.replace('- Make it pass', '- Keep it simple.\\r- Make it pass', 1)\n"
        "    if case == 'crlf_noop':\n"
        "        text = open(BASE, encoding='utf-8').read()\n"
        "    sep = '\\r' if case == 'lone_cr_all' else '\\r\\n'\n"
        "    obj['new_skill_text'] = text.replace('\\n', sep)\n"
        "    out = json.dumps(obj)\n"
        "sys.stdout.write(out)\n"
    )


def run_case(base: Path, case: str, *, crlf_document: bool) -> dict:
    config = build_env(base / "env", seed="c183")
    document = base / "env" / "skills-canonical" / "tdd" / "SKILL.md"
    lf_base = document.read_text(encoding="utf-8")
    if crlf_document:
        document.write_bytes(document.read_bytes().replace(b"\n", b"\r\n"))
    improver = base / "improver.py"
    improver.write_text(
        f"BASE = {str(base / 'lf_base.md')!r}\n" + improver_script(case), encoding="utf-8"
    )
    (base / "lf_base.md").write_text(lf_base, encoding="utf-8")
    text, n = re.subn(
        r'(\[model\.improver\]\nargv = \[)"([^"]+)", "([^"]+)"',
        lambda m: f'{m.group(1)}"{sys.executable}", "{improver}"',
        config.read_text(encoding="utf-8"),
    )
    assert n == 1, "the improver's argv was not found"
    config.write_text(text, encoding="utf-8")
    argv = ["--config", str(config)]
    daily = run_cli([*argv, "--json", "daily", "--commit", "--day", "2026-01-02"])
    result: dict = {"daily_rc": daily.returncode, "stderr": daily.stderr}
    result["skill"] = json.loads(daily.stdout)["skills"].get("tdd", {})
    records = sorted((base / "env" / "state" / "candidates").rglob("*.json"))
    result["records"] = [json.loads(r.read_text(encoding="utf-8")) for r in records]
    if result["records"]:
        record = result["records"][0]
        review = run_cli([*argv, "review", record["id"]])
        result["review_rc"] = review.returncode
        body = review.stdout[review.stdout.index("DIFF") : review.stdout.index("To promote")]
        result["diff_body"] = body
        token = re.search(r"--approve (\S+)", review.stdout)
        assert token, review.stdout
        applied = run_cli(
            [*argv, "apply", record["id"], "--approve", token.group(1), "--i-reviewed-this"]
        )
        result["apply_rc"] = applied.returncode
        result["disk"] = document.read_bytes()
    return result


class TheDiffShowsTheLinesApplyWritesTests(unittest.TestCase):
    cases: dict[str, dict] = {}

    @classmethod
    def setUpClass(cls) -> None:
        plans = {
            "control": False,
            "crlf_both": True,
            "lf_base_crlf_reply": False,
            "bare_cr": True,
            "crlf_noop": False,
            "lone_cr_all": False,
        }
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name).resolve()
        for case, crlf_document in plans.items():
            reply_case = "crlf" if case in ("crlf_both", "lf_base_crlf_reply") else case
            cls.cases[case] = run_case(root / case, reply_case, crlf_document=crlf_document)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def record(self, case: str) -> dict:
        result = self.cases[case]
        self.assertEqual(result["daily_rc"], 0, result["stderr"])
        (record,) = result["records"]
        return record

    def changed_lines(self, case: str) -> list[str]:
        lines = self.cases[case]["diff_body"].splitlines()
        return [l for l in lines if l[:1] in ("+", "-") and not l.startswith(("+++ ", "--- "))]

    def test_the_control_an_lf_reply_is_two_added_lines(self) -> None:
        record = self.record("control")
        self.assertEqual((record["patch"]["added_lines"], record["patch"]["removed_lines"]), (2, 0))
        self.assertEqual(len(self.changed_lines("control")), 2)
        self.assertEqual(self.cases["control"]["disk"], record["patch"]["new_text"].encode())

    def test_a_crlf_reply_is_the_same_replacement_and_the_same_diff(self) -> None:
        control = self.record("control")
        for case in ("crlf_both", "lf_base_crlf_reply"):
            with self.subTest(case=case):
                record = self.record(case)
                self.assertEqual(record["patch"]["new_text"], control["patch"]["new_text"])
                self.assertEqual(
                    record["patch"]["new_text_sha256"], control["patch"]["new_text_sha256"]
                )
                self.assertEqual(
                    (record["patch"]["added_lines"], record["patch"]["removed_lines"]), (2, 0)
                )
                self.assertEqual(self.changed_lines(case), self.changed_lines("control"))

    def test_apply_writes_the_replacement_in_the_documents_convention(self) -> None:
        new_text = self.record("control")["patch"]["new_text"]
        self.assertEqual(self.cases["crlf_both"]["apply_rc"], 0)
        self.assertEqual(self.cases["crlf_both"]["disk"], new_text.replace("\n", "\r\n").encode())
        self.assertEqual(self.cases["lf_base_crlf_reply"]["apply_rc"], 0)
        self.assertEqual(self.cases["lf_base_crlf_reply"]["disk"], new_text.encode())

    def test_a_lone_carriage_return_is_the_line_break_apply_writes(self) -> None:
        record = self.record("bare_cr")
        self.assertNotIn("\r", record["patch"]["new_text"])
        self.assertEqual((record["patch"]["added_lines"], record["patch"]["removed_lines"]), (3, 0))
        changed = self.changed_lines("bare_cr")
        self.assertIn("+- Keep it simple.", changed)
        self.assertEqual(len(changed), 3)
        self.assertNotIn(MARKER, self.cases["bare_cr"]["diff_body"])
        self.assertEqual(
            self.cases["bare_cr"]["disk"],
            record["patch"]["new_text"].replace("\n", "\r\n").encode(),
        )

    def test_a_reply_that_changes_only_line_endings_is_a_no_op(self) -> None:
        result = self.cases["crlf_noop"]
        self.assertEqual(result["daily_rc"], 0, result["stderr"])
        self.assertEqual(result["records"], [])
        self.assertEqual(result["skill"].get("status"), "quarantined")
        self.assertIn(
            "patch is identical to the base text (no-op change)", result["skill"]["reason"]
        )

    def test_the_checks_read_the_replacement_with_its_line_endings_folded(self) -> None:
        # Every line of this reply ends in a lone carriage return. Unfolded it
        # has no '---' line of its own, and the frontmatter check would refuse
        # it; folded, it is the control's replacement.
        result = self.cases["lone_cr_all"]
        self.assertEqual(result["daily_rc"], 0, result["stderr"])
        self.assertEqual(len(result["records"]), 1, result["skill"])
        record = result["records"][0]
        control = self.record("control")
        self.assertEqual(record["patch"]["new_text"], control["patch"]["new_text"])
        self.assertEqual((record["patch"]["added_lines"], record["patch"]["removed_lines"]), (2, 0))
        self.assertEqual(self.changed_lines("lone_cr_all"), self.changed_lines("control"))
        self.assertEqual(result["apply_rc"], 0)
        self.assertEqual(result["disk"], control["patch"]["new_text"].encode())


if __name__ == "__main__":
    unittest.main()
