"""A skill document's own frontmatter name is quoted masked and bounded.

The name a SKILL.md declares is the document's own text, and two messages
quoted it with ``!r``: the discovery warning that it differs from the directory
name (``_snapshot`` in ``adapters/skill_fs.py``; ``skills`` prints it, and
``skills --json`` carries it), and the refusal of a replacement that renames the
skill (``skill_text_defects`` in ``app/pipeline.py``), whose reason becomes the
quarantine record, the checkpoint, the receipt and the daily summary. Both now
quote it with ``quoted_value``: the ``repr``, masked, then cut at
``QUOTED_VALUE_CHARS`` with the cut counted. Measured before (3.11 and 3.13,
the same on both): a name of 2,000,000 characters made one ``skills`` line of
2,000,070 characters (stdout 2,000,671 bytes); a token in the name printed
whole in the text and in the JSON warning; and a base name of 20,000
characters, under the default ``[prompt].max_skill_text_chars``, made a daily
summary of 21,028 bytes, a quarantine record of 20,459, a checkpoint of 21,241
and a receipt of 29,053.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

from agent_skill_improver.app.pipeline import skill_text_defects
from tests.support import FAKE_MODEL, build_env, run_cli

TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
LONG_NAME = "n" * 2_000_000
BASE_NAME = "n" * 20_000
PLANTED = {"short-name": "other-name", "long-name": LONG_NAME, "token-name": TOKEN}
#: The improver line of ``tests/fake_model.py`` this module rewrites so the
#: replacement renames the skill; asserted present once, so a change to the
#: fake fails here rather than silently testing nothing.
FAKE_LINE = '    new_text = "\\n".join(lines) + "\\n"\n'
RENAMING_LINE = (
    '    new_text = __import__("re").sub(r"(?m)^name:.*$", "name: renamed-skill", '
    '"\\n".join(lines), count=1) + "\\n"\n'
)


def snapshots(value: Any) -> Iterator[dict]:
    if isinstance(value, dict):
        if "frontmatter_name" in value:
            yield value
        for child in value.values():
            yield from snapshots(child)
    elif isinstance(value, list):
        for child in value:
            yield from snapshots(child)


def warning_line(text: str, skill_id: str) -> str:
    (line,) = [
        line
        for line in text.splitlines()
        if "frontmatter name" in line and line.endswith(f"directory name '{skill_id}'")
    ]
    return line


def rename_case(base: Path, name: str) -> dict[str, str]:
    config = build_env(base, dry_run=False, seed="c186")
    skill = base / "skills-canonical" / "tdd" / "SKILL.md"
    text = skill.read_text(encoding="utf-8")
    skill.write_text(re.sub(r"(?m)^name:.*$", f"name: {name}", text, count=1), encoding="utf-8")
    source = FAKE_MODEL.read_text(encoding="utf-8")
    assert source.count(FAKE_LINE) == 1, "tests/fake_model.py changed: update FAKE_LINE"
    renamer = base / "fake_renaming_model.py"
    renamer.write_text(source.replace(FAKE_LINE, RENAMING_LINE), encoding="utf-8")
    config_text = config.read_text(encoding="utf-8")
    assert str(FAKE_MODEL) in config_text
    config.write_text(config_text.replace(str(FAKE_MODEL), str(renamer)), encoding="utf-8")
    daily = run_cli(["--config", str(config), "daily", "--commit", "--day", "2026-01-02"])
    assert daily.returncode == 0, daily.stderr
    state = base / "state"
    (quarantine,) = sorted((state / "quarantine").glob("tdd--*--propose--*.json"))
    (receipt,) = sorted((state / "receipts").glob("2026-01-02--*.json"))
    (checkpoint,) = sorted((state / "checkpoints").glob("2026-01-02--*.json"))
    return {
        "daily summary": daily.stdout,
        "quarantine record": quarantine.read_text(encoding="utf-8"),
        "receipt": receipt.read_text(encoding="utf-8"),
        "checkpoint": checkpoint.read_text(encoding="utf-8"),
    }


class TheListingQuotesTheNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        base = Path(cls._tmp.name).resolve()
        config = build_env(base, dry_run=False, seed="c186")
        for skill_id, name in PLANTED.items():
            directory = base / "skills-canonical" / skill_id
            directory.mkdir()
            (directory / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: planted\n---\nBody.\n", encoding="utf-8"
            )
        text = run_cli(["--config", str(config), "skills"])
        listed = run_cli(["--config", str(config), "--json", "skills"])
        assert text.returncode == 0 and listed.returncode == 0, (text.stderr, listed.stderr)
        cls.text = text.stdout
        cls.snapshots = {s["skill_id"]: s for s in snapshots(json.loads(listed.stdout))}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_control_a_short_differing_name_is_warned_as_before(self) -> None:
        self.assertEqual(
            warning_line(self.text, "short-name"),
            "    warning: frontmatter name 'other-name' differs from directory name 'short-name'",
        )
        self.assertIn(
            "frontmatter name 'other-name' differs from directory name 'short-name'",
            self.snapshots["short-name"]["warnings"],
        )

    def test_a_long_name_is_cut_on_its_line_and_the_cut_counted(self) -> None:
        line = warning_line(self.text, "long-name")
        self.assertEqual(
            line,
            "    warning: frontmatter name '" + "n" * 119 + "... [1999882 more chars not quoted]"
            " differs from directory name 'long-name'",
        )
        self.assertLess(len(self.text), 5_000)
        for warning in self.snapshots["long-name"]["warnings"]:
            self.assertLess(len(warning), 300)

    def test_a_token_in_the_name_is_masked_in_the_text_and_the_json_warning(self) -> None:
        self.assertNotIn(TOKEN, self.text)
        self.assertEqual(
            warning_line(self.text, "token-name"),
            "    warning: frontmatter name '[REDACTED:github-token]' differs from directory "
            "name 'token-name'",
        )
        self.assertFalse(any(TOKEN in w for w in self.snapshots["token-name"]["warnings"]))

    def test_the_json_snapshot_keeps_the_documents_own_name_whole(self) -> None:
        # Not covered by the bound, as the row says: the snapshot's field is the
        # document's own value, the way ``discover --json`` keeps a thread whole.
        self.assertEqual(self.snapshots["long-name"]["frontmatter_name"], LONG_NAME)
        self.assertEqual(self.snapshots["token-name"]["frontmatter_name"], TOKEN)


class TheRenameRefusalQuotesTheBaseNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name).resolve()
        cls.control = rename_case(root / "control", "tdd")
        cls.long = rename_case(root / "long", BASE_NAME)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_control_a_rename_is_refused_naming_both_names_as_before(self) -> None:
        reason = (
            "replacement changes the frontmatter name from 'tdd' to 'renamed-skill'; "
            "renaming a skill is not an in-place improvement"
        )
        for where, text in self.control.items():
            with self.subTest(where=where):
                self.assertIn(reason, text)

    def test_a_long_base_name_is_cut_in_every_record_it_reaches(self) -> None:
        quoted = "from '" + "n" * 119 + "... [19882 more chars not quoted] to 'renamed-skill'"
        for where, text in self.long.items():
            with self.subTest(where=where):
                self.assertIn(quoted, text)
                self.assertNotIn("n" * 200, text)
        self.assertLess(len(self.long["daily summary"]), 3_000)
        self.assertLess(len(self.long["quarantine record"]), 1_500)

    def test_the_refusal_masks_a_token_in_the_base_name(self) -> None:
        # A base document holding a credential is refused before any model
        # runs, so this is reached only through the function itself.
        base = f"---\nname: {TOKEN}\ndescription: d\n---\nBody.\n"
        new = "---\nname: renamed\ndescription: d\n---\nBody.\n"
        (problem,) = [
            p for p in skill_text_defects(new, base_text=base, skill_id="tdd") if "renam" in p
        ]
        self.assertNotIn(TOKEN, problem)
        self.assertIn("from '[REDACTED:github-token]' to 'renamed'", problem)


if __name__ == "__main__":
    unittest.main()
