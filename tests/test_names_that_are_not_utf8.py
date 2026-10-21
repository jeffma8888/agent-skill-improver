"""A file name that is not UTF-8 is written into state as its JSON escape.

``os.fsdecode`` (and so ``os.scandir`` and ``os.walk`` over a str path) gives
each byte of a name that is not UTF-8 as one lone surrogate, and ``json.dumps``
with ``ensure_ascii`` off writes that surrogate raw, which no UTF-8 encoder
accepts. Measured before: a fixtures entry with such a name, which the source
passed over, made the committed daily run fall with ``UnicodeEncodeError`` from
the receipt write, with no receipt. Linux allows such names and APFS refuses
them, so one test lists the name through a stand-in for the directory listing
and another makes a real one wherever the file system allows it.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

from agent_skill_improver.adapters import fixture_source
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.hashing import sha256_bytes, tree_digest
from agent_skill_improver.jsonio import REPLACEMENT_CHARACTER, dumps, loads_untrusted

from tests.support import FIXTURE_EXPORTS
from tests.test_daily_isolation import DailyHarness

NAME = "\udcff.json"  # how Python spells the file name b"\xff.json"


class WritingTests(unittest.TestCase):
    def test_a_lone_surrogate_is_written_as_its_json_escape(self):
        value = {"k\udcff": ["a\udcff", "\ud800"]}
        text = dumps(value)
        text.encode("utf-8")
        self.assertIn("a\\udcff", text)
        self.assertEqual(json.loads(text), value, "the escape is the same string to a parser")

    def test_what_is_written_reads_back_as_the_replacement_character(self):
        self.assertEqual(
            loads_untrusted(dumps({"path": NAME})), {"path": REPLACEMENT_CHARACTER + ".json"}
        )

    def test_the_control_other_text_is_written_as_it_is(self):
        self.assertIn("caf\u00e9 \u2603 \U0001f600", dumps({"a": "caf\u00e9 \u2603 \U0001f600"}))

    def test_a_name_that_is_not_utf8_is_digested_by_its_bytes(self):
        self.assertEqual(
            tree_digest({"\udcff/SKILL.md": "abc"}), sha256_bytes(b"\xff/SKILL.md\x00abc")
        )
        self.assertNotEqual(
            tree_digest({"\udcff/SKILL.md": "abc"}),
            tree_digest({"\ufffd/SKILL.md": "abc"}),
            "a name that is not UTF-8 is not folded into another name",
        )


class _Entry:
    def __init__(self, name: str) -> None:
        self.name = name


class NamesOutsideUtf8InADailyRunTests(DailyHarness):
    def exports(self) -> Path:
        exports = self.base / "exports"
        shutil.copytree(FIXTURE_EXPORTS, exports)
        return exports

    def committed(self, exports: Path):
        receipt, state, _store = self.run_pass(self.env(fixtures_dir=exports))
        payload = receipt.to_dict()
        self.assertEqual(payload["status"], "ok", payload["status"])
        self.assertEqual(payload["counts"]["records_scored"], 5, "every fixture is scored")
        receipts = sorted(state.dir("receipts").glob("*.json"))
        self.assertEqual(len(receipts), 1, "the run wrote its receipt")
        return receipts[0].read_text(encoding="utf-8")

    def test_a_listed_name_that_is_not_utf8_is_named_on_the_receipt(self):
        exports = self.exports()
        listing = os.scandir

        class WithTheName:
            def __init__(self, path) -> None:
                self.path = path

            def __enter__(self):
                with listing(self.path) as it:
                    names = [entry.name for entry in it]
                return iter([_Entry(name) for name in names + [NAME]])

            def __exit__(self, *exc) -> bool:
                return False

        def scandir(path):
            return WithTheName(path) if Path(path) == exports else listing(path)

        with mock.patch.object(fixture_source.os, "scandir", scandir):
            text = self.committed(exports)
        self.assertIn("\\udcff.json", text, "the entry is named, by its escape")

    def test_a_real_file_whose_name_is_not_utf8_is_named_on_the_receipt(self):
        exports = self.exports()
        try:
            (exports / NAME).write_text("not an export", encoding="utf-8")
        except OSError as exc:
            if exc.errno in (errno.EILSEQ, errno.EINVAL):
                self.skipTest("this file system refuses a name that is not UTF-8")
            raise
        text = self.committed(exports)
        self.assertIn("\\udcff.json", text)
        self.assertIn("is not valid JSON", text)

    def test_a_tree_entry_named_outside_utf8_is_digested_and_the_run_goes_on(self):
        """A skill directory whose name is not UTF-8 is a tree entry keyed by
        that name; the stand-in adds one to both readings of the tree."""
        listing = FilesystemSkillStore.tree_entries

        def with_the_name(store):
            entries = listing(store)
            entries["canonical:\udcff/SKILL.md"] = "0" * 64
            return entries

        with mock.patch.object(FilesystemSkillStore, "tree_entries", with_the_name):
            receipt, _state, _store = self.run_pass(self.env())
        payload = receipt.to_dict()
        self.assertEqual(payload["status"], "ok", payload["status"])
        self.assertEqual(payload["counts"]["records_scored"], 5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
