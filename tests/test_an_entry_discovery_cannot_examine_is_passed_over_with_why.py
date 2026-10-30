"""An entry named SKILL.md that discovery cannot examine is passed over with why.

Discovery (``_iter_skill_files`` in ``adapters/skill_fs.py``) judges each entry
named like a document with ``lstat``, then, for a link, whether what it names
exists. Measured before (3.11 and 3.13, a committed run with one validated
candidate, its token taken, then in the canonical root a folder ``bad`` holding
a ``SKILL.md`` that is a link to a file in a directory of mode 000, or a folder
``locked`` of mode 0o444, listable but not searchable, holding a ``SKILL.md``):
``doctor``, ``skills``, ``review``, ``apply`` and ``daily`` each exited 70,
"internal error: this is a bug in asi, not a negative answer", with a
``PermissionError`` traceback naming that entry. Such an entry is now passed
over, as a document this process cannot read is, with the system's words: "a
symlink whose target this process cannot reach (Permission denied)" and "an
entry this process cannot examine (Permission denied)". With the candidate's
own folder so made, or its document such a link, ``doctor``, ``skills`` and
``daily`` exited 70 the same way, and ``review`` and ``apply`` refused (exit
3); now the copy withholds the skill's canonical slot, ``daily`` quarantines
the skill naming it, and ``review`` and ``apply`` refuse as before.
"""

from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.config import load_config

from tests.support import build_env

DAY, NEXT_DAY = "2026-02-01", "2026-02-02"
UNREACHED = "a symlink whose target this process cannot reach (Permission denied)"
UNEXAMINED = "an entry this process cannot examine (Permission denied)"


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


class _Validated:
    """One committed run with one validated candidate, its token taken."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.config = build_env(base / "env", seed="c226")
        self.state = base / "env" / "state"
        rc, said = run(self.config, "daily", "--commit", "--day", DAY)
        assert rc == 0, said
        (self.record_file,) = sorted(self.state.glob("candidates/*.json"))
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self.cid = record["id"]
        self.skill = Path(record["target"]["canonical_path"])
        self.root = self.skill.parent.parent
        rc, shown = run(self.config, "review", self.cid)
        assert rc == 0, shown
        found = re.search(r"--approve (\S+)", shown)
        assert found, shown
        self.token = found.group(1)


@unittest.skipIf(os.geteuid() == 0, "root reaches what a mode of 000 or 0o444 bars")
class TheEntriesArePassedOverWithWhyTests(unittest.TestCase):
    """A link to a file in a directory of mode 000, and a folder of mode 0o444."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.case = case = _Validated(Path(cls._tmp.name))
        cls.hidden = case.base / "hidden"
        cls.hidden.mkdir()
        (cls.hidden / "notes").write_text("text", encoding="utf-8")
        cls.link = case.root / "bad" / "SKILL.md"
        cls.link.parent.mkdir()
        cls.link.symlink_to(cls.hidden / "notes")
        cls.locked = case.root / "locked" / "SKILL.md"
        cls.locked.parent.mkdir()
        cls.locked.write_text("---\nname: locked\ndescription: x\n---\nbody\n", encoding="utf-8")
        cls.hidden.chmod(0)
        cls.locked.parent.chmod(0o444)
        try:
            cls.said = {
                "doctor": run(case.config, "doctor"),
                "doctor --json": run(case.config, "--json", "doctor"),
                "skills": run(case.config, "skills"),
                "review": run(case.config, "review", case.cid),
                "apply": run(
                    case.config, "apply", case.cid, "--approve", case.token, "--i-reviewed-this"
                ),
                "daily": run(case.config, "--json", "daily", "--commit", "--day", NEXT_DAY),
            }
            cls.modes = (
                os.stat(cls.hidden).st_mode & 0o777,
                os.stat(cls.locked.parent).st_mode & 0o777,
                os.readlink(cls.link),
            )
        finally:
            cls.hidden.chmod(0o700)
            cls.locked.parent.chmod(0o700)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_doctor_names_each_entry_and_why_on_a_warn_line(self) -> None:
        rc, said = self.said["doctor"]
        self.assertEqual(rc, 0, said)
        lines = said.splitlines()
        for path, why in ((self.link, UNREACHED), (self.locked, UNEXAMINED)):
            with self.subTest(path=path.parent.name):
                self.assertIn(
                    f"WARN  skill document passed over in root canonical: {path} is {why}", lines
                )
        rc, said = self.said["doctor --json"]
        self.assertEqual(rc, 0, said)
        self.assertEqual(
            {d["path"]: d["reason"] for d in json.loads(said)["skill_documents_passed_over"]},
            {str(self.link): UNREACHED, str(self.locked): UNEXAMINED},
        )

    def test_skills_lists_each_entry_and_why(self) -> None:
        rc, said = self.said["skills"]
        self.assertEqual(rc, 0, said)
        lines = said.splitlines()
        self.assertIn(f"passed over (canonical): {self.link} is {UNREACHED}", lines)
        self.assertIn(f"passed over (canonical): {self.locked} is {UNEXAMINED}", lines)

    def test_review_and_apply_of_another_skill_take_its_candidate_past_them(self) -> None:
        rc, said = self.said["review"]
        self.assertEqual(rc, 0, said)
        self.assertIn(f"--approve {self.case.token}", said)
        rc, said = self.said["apply"]
        self.assertEqual(rc, 0, said)
        record = json.loads(self.case.record_file.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "applied")
        self.assertEqual(self.case.skill.read_text(encoding="utf-8"), record["patch"]["new_text"])

    def test_daily_runs_and_its_receipt_names_each_entry_and_why(self) -> None:
        rc, said = self.said["daily"]
        self.assertEqual(rc, 0, said)
        receipt = json.loads(said)
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(
            {d["path"]: d["reason"] for d in receipt["skill_tree"]["skill_documents_passed_over"]},
            {str(self.link): UNREACHED, str(self.locked): UNEXAMINED},
        )

    def test_the_entries_are_left_as_they_were(self) -> None:
        self.assertEqual(self.modes, (0, 0o444, str(self.hidden / "notes")))


@unittest.skipIf(os.geteuid() == 0, "root reaches what a mode of 000 or 0o444 bars")
class SuchAnEntryInPlaceOfASkillTests(unittest.TestCase):
    def test_such_an_entry_in_place_of_a_skill_withholds_its_canonical_slot(self) -> None:
        # The skill's own folder made listable but not searchable, and its
        # document made a link to a file in a directory of mode 000: like a
        # copy discovery cannot read, it withholds the skill's canonical slot,
        # and `daily` quarantines the skill, naming the copy and why, rather
        # than propose against the skill's other copy.
        for shape, why in (("folder", UNEXAMINED), ("link", UNREACHED)):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as tmp:
                case = _Validated(Path(tmp))
                case.record_file.unlink()
                if shape == "folder":
                    barred = case.skill.parent
                    mode = 0o444
                else:
                    barred = case.base / "hidden"
                    barred.mkdir()
                    (barred / "SKILL.md").write_bytes(case.skill.read_bytes())
                    case.skill.unlink()
                    case.skill.symlink_to(barred / "SKILL.md")
                    mode = 0
                barred.chmod(mode)
                try:
                    rc, said = run(case.config, "--json", "skills")
                    self.assertEqual(rc, 0, said)
                    (group,) = [g for g in json.loads(said)["groups"] if g["skill_id"] == "tdd"]
                    self.assertIsNone(group["canonical"])
                    self.assertIn("passed over by discovery", group["canonical_reason"])
                    rc, said = run(case.config, "--json", "daily", "--commit", "--day", NEXT_DAY)
                    self.assertEqual(rc, 0, said)
                    receipt = json.loads(said)
                    self.assertIn(
                        {"path": str(case.skill), "reason": why},
                        [
                            {"path": d["path"], "reason": d["reason"]}
                            for d in receipt["skill_tree"]["skill_documents_passed_over"]
                        ],
                    )
                    self.assertEqual(sorted(case.state.glob("candidates/*tdd*.json")), [])
                    self.assertEqual(receipt["counts"]["skills_quarantined"], 1, said)
                    self.assertIn(
                        "skill 'tdd' has no canonical editable copy (the copy in editable root "
                        f"'canonical' was passed over by discovery ({why})",
                        said,
                    )
                finally:
                    barred.chmod(0o700)


class TheWordsAreTheSystemsTests(unittest.TestCase):
    """Another error of either call, put on one document by a stand-in for the call."""

    def passed_over_for(self, call: str, error: OSError) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(build_env(Path(tmp) / "env", seed="c226"))
            store = FilesystemSkillStore(config.skill_roots, read_only=True)
            real_is_symlink, real_exists = Path.is_symlink, Path.exists
            aimed: list[Path] = []

            def at_it(path: Path) -> bool:
                return path.name == "SKILL.md" and path.parent.name == "code-review"

            def is_symlink(path: Path) -> bool:
                if at_it(path):
                    aimed.append(path)
                    if call == "is_symlink":
                        raise error
                    return True
                return real_is_symlink(path)

            def exists(path: Path, *args: object, **kwargs: object) -> bool:
                if call == "exists" and at_it(path):
                    raise error
                return real_exists(path, *args, **kwargs)  # type: ignore[arg-type]

            with mock.patch.object(Path, "is_symlink", is_symlink):
                with mock.patch.object(Path, "exists", exists):
                    store.snapshots(refresh=True)
            self.assertEqual(len(aimed), 1, aimed)
            (entry,) = [d for d in store.passed_over() if d.path in aimed]
            self.assertEqual(entry.skill_id, "code-review")
            self.assertNotIn("code-review", {s.skill_id for s in store.snapshots()})
            return entry.reason

    def test_another_error_of_either_call_is_passed_over_in_the_systems_words(self) -> None:
        eio = OSError(errno.EIO, os.strerror(errno.EIO), "SKILL.md")
        bare = OSError("refused without a number")
        for call, why in (
            ("is_symlink", "an entry this process cannot examine"),
            ("exists", "a symlink whose target this process cannot reach"),
        ):
            with self.subTest(call=call):
                self.assertEqual(
                    self.passed_over_for(call, eio), f"{why} ({os.strerror(errno.EIO)})"
                )
                # An error with no words of the system's is named as it is.
                self.assertEqual(
                    self.passed_over_for(call, bare), f"{why} (refused without a number)"
                )


if __name__ == "__main__":
    unittest.main()
