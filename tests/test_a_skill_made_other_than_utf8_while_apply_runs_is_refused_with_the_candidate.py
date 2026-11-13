"""A skill made other than UTF-8 while ``apply`` runs is refused with the candidate.

``apply`` reads its target and refuses bytes that are not UTF-8 with the
candidate and two ways on (``no_document_to_read`` in ``app/promote.py``); then
the write (``write_canonical`` in ``adapters/skill_fs.py``) reads the file again
on its own pinned handle before it takes its backup. Measured before (3.11 and
3.13, a committed run with one validated candidate, its token taken, then the
skill's bytes made other than UTF-8 at the step ``apply`` takes after its own
check): ``apply`` exited 3 with "UndecodableDocument: skill document <path> is
not valid UTF-8 at byte 0 (invalid start byte); nothing is written over text
this tool cannot read exactly. The file is left as it is: fix the encoding and
run the command again.", naming no candidate, the record ``validated`` and no
backup directory; with the text the candidate was written against put back,
``apply`` with the same token applied it.

Given the candidate, the write's read now words the refusal as ``apply``'s own
check does, for the window it is met in (``undecodable_before_the_write``);
without one, as before.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.config import load_config
from agent_skill_improver.domain.integrity import repropose_advice
from agent_skill_improver.errors import UndecodableDocument
from tests.support import build_env

DAY, NEXT_DAY = "2026-02-01", "2026-02-02"
REFUSED = 3
CID = f"CAND-{DAY}-01-tdd"


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


def first_line(said: str) -> str:
    return next(line for line in said.splitlines() if line.strip())


def ways_on(cid: str) -> str:
    return (
        "with a regular UTF-8 file holding the text it was written against back at that "
        "path, it is reviewed and applied as before. To propose against other text instead, "
        f"put a regular UTF-8 file there first, then {repropose_advice(cid)}."
    )


class Validated:
    """A committed run's one validated candidate, its token taken."""

    def __init__(self, base: Path) -> None:
        self.base = base.resolve()
        self.config = build_env(base / "env", seed="c230")
        self.state = self.base / "env" / "state"
        rc, said = run(self.config, "daily", "--commit", "--day", DAY)
        assert rc == 0, said
        (self.record,) = sorted(self.state.glob("candidates/*.json"))
        assert self.record.stem == CID, self.record.stem
        self.skill = Path(json.loads(self.record.read_text())["target"]["canonical_path"])
        rc, shown = run(self.config, "review", CID)
        found = re.search(r"--approve (\S+)", shown)
        assert rc == 0 and found, shown
        self.token = found.group(1)
        self.original = self.skill.read_bytes()
        self.bad = b"\xff" + self.original

    def apply_after(self, change: Callable[[], None]) -> tuple[int, str]:
        """``apply``, with ``change`` made at the step it takes after its own check."""
        real = FilesystemSkillStore.mirror_paths_of
        done: list[bool] = []

        def change_then_list(store: Any, path: Path) -> tuple[str, ...]:
            if not done:
                done.append(True)
                change()
            return real(store, path)

        with mock.patch.object(FilesystemSkillStore, "mirror_paths_of", change_then_list):
            return self.apply()

    def made_bad_while_apply_runs(self) -> tuple[int, str]:
        return self.apply_after(lambda: self.skill.write_bytes(self.bad))

    def apply(self) -> tuple[int, str]:
        return run(self.config, "apply", CID, "--approve", self.token, "--i-reviewed-this")

    def status(self) -> str:
        return str(json.loads(self.record.read_text())["status"])

    def backups(self) -> list[Path]:
        return sorted(self.state.glob("backups/*"))


class TheWritesReadNamesTheCandidateTests(unittest.TestCase):
    def test_bytes_made_other_than_utf8_after_applys_own_check_are_refused_with_the_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            rc, said = case.made_bad_while_apply_runs()
            self.assertEqual(rc, REFUSED, said)
            self.assertEqual(
                first_line(said),
                f"UndecodableDocument: refusing to apply {CID}: its canonical target "
                f"{case.skill} is not valid UTF-8 at byte 0 (invalid start byte), so there "
                "is no base text to verify against. The skill changed while this apply ran, "
                "after it checked the file and before it wrote, and the skill was not "
                f"written, nor anything backed up: {ways_on(CID)}",
            )
            self.assertEqual(case.status(), "validated")
            self.assertEqual(case.skill.read_bytes(), case.bad)
            self.assertEqual(case.backups(), [])

    def test_its_ways_on_are_the_ones_applys_own_check_gives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            case.skill.write_bytes(case.bad)
            rc, own = case.apply()
            self.assertEqual(rc, REFUSED, own)
            self.assertTrue(first_line(own).endswith(ways_on(CID)), own)
            opening = (
                f"UndecodableDocument: refusing to apply {CID}: its canonical target "
                f"{case.skill} is not valid UTF-8 at byte 0 (invalid start byte), so there is "
                "no base text to "
            )
            self.assertTrue(first_line(own).startswith(opening), own)
            case.skill.write_bytes(case.original)
            rc, said = case.made_bad_while_apply_runs()
            self.assertEqual(rc, REFUSED, said)
            self.assertTrue(first_line(said).startswith(opening), said)
            self.assertTrue(first_line(said).endswith(ways_on(CID)), said)

    def test_the_utf8_text_put_back_applies_with_the_same_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            rc, said = case.made_bad_while_apply_runs()
            self.assertEqual(rc, REFUSED, said)
            case.skill.write_bytes(case.original)
            rc, said = case.apply()
            self.assertEqual(rc, 0, said)
            self.assertEqual(case.status(), "applied")
            (backup,) = case.backups()
            self.assertEqual((backup / "SKILL.md.bak").read_bytes(), case.original)

    def test_other_utf8_text_and_the_record_deleted_propose_again_on_another_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Validated(Path(tmp))
            rc, said = case.made_bad_while_apply_runs()
            self.assertEqual(rc, REFUSED, said)
            case.skill.write_bytes(case.original + b"\n- a line someone added\n")
            case.record.unlink()
            rc, said = run(case.config, "daily", "--commit", "--day", NEXT_DAY)
            self.assertEqual(rc, 0, said)
            self.assertEqual(
                sorted(p.stem for p in case.state.glob("candidates/*.json")),
                [f"CAND-{NEXT_DAY}-01-tdd"],
            )


class WithoutTheCandidateTests(unittest.TestCase):
    def _store_and_skill(self, tmp: str) -> tuple[FilesystemSkillStore, Path]:
        config = build_env(Path(tmp) / "env", seed="c230")
        skill = (Path(tmp) / "env").resolve() / "skills-canonical" / "tdd" / "SKILL.md"
        return FilesystemSkillStore(load_config(config).skill_roots, read_only=False), skill

    def test_without_a_candidate_the_writes_read_is_worded_as_before(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, skill = self._store_and_skill(tmp)
            skill.write_bytes(b"\xff" + skill.read_bytes())
            made: list[bool] = []

            def backup_dir() -> Path:
                made.append(True)
                return Path(tmp) / "backup"

            with self.assertRaises(UndecodableDocument) as ctx:
                store.write_canonical(
                    skill,
                    "new text\n",
                    expected_sha256="0" * 64,
                    backup_dir=backup_dir,
                    skill_id="tdd",
                )
            self.assertEqual(
                str(ctx.exception),
                f"skill document {skill} is not valid UTF-8 at byte 0 (invalid start byte); "
                "nothing is written over text this tool cannot read exactly. The file is left "
                "as it is: fix the encoding and run the command again.",
            )
            self.assertEqual(made, [])

    def test_given_the_candidate_it_keeps_the_reads_record_id_and_cause(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, skill = self._store_and_skill(tmp)
            original = skill.read_bytes()
            # A multibyte sequence cut short: another reason than ``\xff``'s.
            skill.write_bytes(original + b"\xe2\x82")
            with self.assertRaises(UndecodableDocument) as ctx:
                store.write_canonical(
                    skill,
                    "new text\n",
                    expected_sha256="0" * 64,
                    backup_dir=Path(tmp) / "backup",
                    skill_id="tdd",
                    candidate_id=CID,
                )
            self.assertTrue(str(ctx.exception).startswith(f"refusing to apply {CID}: "))
            # The byte the read stopped at, here past the start, and its reason.
            self.assertIn(
                f"is not valid UTF-8 at byte {len(original)} (unexpected end of data)",
                str(ctx.exception),
            )
            self.assertEqual(ctx.exception.record_id, "tdd")
            read = ctx.exception.__cause__
            self.assertIsInstance(read, UndecodableDocument)
            self.assertIsInstance(read.__cause__, UnicodeDecodeError)
            self.assertFalse((Path(tmp) / "backup").exists())

    def test_one_that_is_no_decode_error_is_raised_as_it_is_even_with_a_candidate(self) -> None:
        for cause in (None, ValueError("not a decode error")):
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as tmp:
                store, skill = self._store_and_skill(tmp)
                raised = UndecodableDocument("words of their own", record_id="tdd")
                raised.__cause__ = cause

                def refuse(data: bytes, target: Path) -> str:
                    raise raised

                with mock.patch.object(
                    FilesystemSkillStore, "_document_sha256", staticmethod(refuse)
                ):
                    with self.assertRaises(UndecodableDocument) as ctx:
                        store.write_canonical(
                            skill,
                            "new text\n",
                            expected_sha256="0" * 64,
                            backup_dir=Path(tmp) / "backup",
                            skill_id="tdd",
                            candidate_id=CID,
                        )
                self.assertIs(ctx.exception, raised)
                self.assertIs(ctx.exception.__cause__, cause)


if __name__ == "__main__":
    unittest.main()
