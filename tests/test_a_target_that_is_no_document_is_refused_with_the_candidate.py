"""A target that is no document to read is refused with the candidate and what to do.

``review`` and ``apply`` read a candidate's skill file through the store's
``read_text``, which refuses a target that is a symbolic link
(``LinkedTargetRefused``), one that is not a regular file (``BaseHashMismatch``)
and one whose bytes are not UTF-8 (``UndecodableDocument``). Measured before
(3.11 and 3.13, a committed run with one validated candidate, then its skill
file replaced by a fifo, by a link to an identical copy, or given bytes that are
not UTF-8): ``review`` and ``apply`` each exited 3 with the store's refusal,
"refusing to read <path>: ...", which named no candidate and gave no remedy; the
one for bytes that are not UTF-8 said "the next pass will handle it", which is
what ``daily`` does. The record stayed ``validated``. With the base file put
back, ``review`` and ``apply`` with the same token went through; with the record
deleted, ``daily`` quarantined the skill while the target kept its shape, and
proposed again on another day once a regular file was back.
``no_document_to_read`` in ``app/promote.py`` now words the refusal: the
candidate, the shape, and both remedies, keeping the store's class and exit.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Callable

from agent_skill_improver import cli
from agent_skill_improver.app import promote
from agent_skill_improver.domain.integrity import repropose_advice
from agent_skill_improver.errors import (
    BaseHashMismatch,
    LinkedTargetRefused,
    UndecodableDocument,
)

from tests.support import build_env

DAY, NEXT_DAY, THIRD_DAY = "2026-02-01", "2026-02-02", "2026-02-03"
REFUSED = 3
DAILY_WORDS = "the next pass will handle it"


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


def fifo(path: Path) -> None:
    path.unlink()
    os.mkfifo(path)


def linked(path: Path) -> None:
    twin = path.with_name("twin.md")
    twin.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(twin)


def undecodable(path: Path) -> None:
    path.write_bytes(path.read_bytes() + b"\xff\xfe\n")


def remedies(cid: str) -> str:
    return (
        "The skill and the candidate are unchanged, and nothing was backed up: with a "
        "regular UTF-8 file holding the text it was written against back at that path, it "
        "is reviewed and applied as before. To propose against other text instead, put a "
        f"regular UTF-8 file there first, then {repropose_advice(cid)}."
    )


class _Shaped:
    """One validated candidate, its token taken, then its skill file given ``effect``."""

    def __init__(self, base: Path, effect: Callable[[Path], None]):
        self.config = build_env(base / "env", seed="c221")
        self.state = base / "env" / "state"
        rc, said = run(self.config, "daily", "--commit", "--day", DAY)
        assert rc == 0, said
        (self.record_file,) = sorted(self.state.glob("candidates/*.json"))
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self.cid = record["id"]
        self.skill = Path(record["target"]["canonical_path"])
        self.base = self.skill.read_bytes()
        rc, shown = run(self.config, "review", self.cid)
        assert rc == 0, shown
        found = re.search(r"--approve (\S+)", shown)
        assert found, shown
        self.token = found.group(1)
        effect(self.skill)

    def review(self) -> tuple[int, str]:
        return run(self.config, "review", self.cid)

    def apply(self) -> tuple[int, str]:
        return run(self.config, "apply", self.cid, "--approve", self.token, "--i-reviewed-this")

    def put_back(self, data: bytes) -> None:
        if self.skill.is_symlink() or self.skill.exists():
            self.skill.unlink()
        for twin in self.skill.parent.glob("twin.md"):
            twin.unlink()
        self.skill.write_bytes(data)

    def status(self) -> str:
        return str(json.loads(self.record_file.read_text(encoding="utf-8"))["status"])

    def left_as_it_is(self) -> tuple[object, bytes, list[Path]]:
        """What the skill path is, the record's bytes, and the backups made."""
        mode = os.lstat(self.skill).st_mode
        if stat.S_ISLNK(mode):
            shape: object = ("link", os.readlink(self.skill))
        elif stat.S_ISREG(mode):
            shape = ("file", self.skill.read_bytes())
        else:
            shape = ("other", stat.S_IFMT(mode))
        return shape, self.record_file.read_bytes(), sorted(self.state.glob("backups/*"))


class TheWordsTests(unittest.TestCase):
    """The refusal itself, reached through its module so on a tree without it only this errors."""

    def test_each_shape_is_named_with_the_candidate_and_both_remedies(self) -> None:
        cid = "CAND-2026-02-01-01-tdd"
        path = Path("/skills/tdd/SKILL.md")
        candidate = type("C", (), {"id": cid})()
        try:
            b"\xff".decode("utf-8")
        except UnicodeDecodeError as decode_error:
            undecodable_error = UndecodableDocument("store words")
            undecodable_error.__cause__ = decode_error
        for exc, shape in (
            (BaseHashMismatch("store words"), "is not a regular file"),
            (
                LinkedTargetRefused("store words"),
                "is a symbolic link, which apply refuses to replace",
            ),
            (undecodable_error, "is not valid UTF-8 at byte 0 (invalid start byte)"),
        ):
            with self.subTest(type(exc).__name__):
                said = promote.no_document_to_read(candidate, path, "review", exc)
                self.assertEqual(
                    said,
                    f"refusing to review {cid}: its canonical target {path} {shape}, so there "
                    f"is no base text to show or to verify against. {remedies(cid)}",
                )


class _RefusedShapeTests(unittest.TestCase):
    effect: Callable[[Path], None]
    kind: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.case = _Shaped(Path(cls._tmp.name), cls.effect)
        cls.left = cls.case.left_as_it_is()
        cls.refused = {"review": cls.case.review(), "apply": cls.case.apply()}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def assert_named_with_both_remedies(self) -> None:
        for action, (rc, said) in self.refused.items():
            with self.subTest(action):
                self.assertEqual(rc, REFUSED, said)
                self.assertTrue(
                    said.startswith(
                        f"{self.kind}: refusing to {action} {self.case.cid}: its canonical "
                        f"target {self.case.skill} "
                    ),
                    said,
                )
                self.assertIn(remedies(self.case.cid), said)
                self.assertNotIn(DAILY_WORDS, said)
                self.assertEqual(said.count("\n"), 1, said)
        self.assertEqual(self.case.status(), "validated")
        # What the refusal says is unchanged: the skill path, the record, no backup.
        self.assertEqual(self.case.left_as_it_is(), self.left)
        self.assertEqual(self.left[2], [])


class AFifoTests(_RefusedShapeTests):
    effect = staticmethod(fifo)
    kind = "BaseHashMismatch"

    def test_review_and_apply_of_a_fifo_name_the_candidate_and_what_to_do(self) -> None:
        self.assert_named_with_both_remedies()


class ALinkedFileTests(_RefusedShapeTests):
    effect = staticmethod(linked)
    kind = "LinkedTargetRefused"

    def test_review_and_apply_of_a_linked_file_name_the_candidate_and_what_to_do(self) -> None:
        self.assert_named_with_both_remedies()


class AnUndecodableFileTests(_RefusedShapeTests):
    effect = staticmethod(undecodable)
    kind = "UndecodableDocument"

    def test_review_and_apply_of_an_undecodable_file_name_the_candidate_and_what_to_do(
        self,
    ) -> None:
        self.assert_named_with_both_remedies()


class WhatTheRemediesDoTests(unittest.TestCase):
    """Each remedy the refusal gives, followed, for each shape."""

    SHAPES = (("fifo", fifo), ("link", linked), ("undecodable", undecodable))

    def test_the_base_text_put_back_reviews_and_applies_with_the_same_token(self) -> None:
        for name, effect in self.SHAPES:
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                case = _Shaped(Path(tmp), effect)
                self.assertEqual(case.review()[0], REFUSED)
                case.put_back(case.base)
                rc, said = case.review()
                self.assertEqual(rc, 0, said)
                rc, said = case.apply()
                self.assertEqual(rc, 0, said)
                self.assertEqual(case.status(), "applied")

    def test_with_the_record_deleted_another_day_proposes_once_a_regular_file_is_back(
        self,
    ) -> None:
        for name, effect in self.SHAPES:
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                case = _Shaped(Path(tmp), effect)
                case.record_file.unlink()
                rc, said = run(case.config, "daily", "--commit", "--day", NEXT_DAY)
                self.assertEqual(rc, 0, said)
                self.assertIn("skills_quarantined=1", said)
                self.assertEqual(list(case.state.glob("candidates/*.json")), [])
                case.put_back(case.base + b"\n- a line someone added\n")
                rc, said = run(case.config, "daily", "--commit", "--day", THIRD_DAY)
                self.assertEqual(rc, 0, said)
                self.assertEqual(
                    sorted(p.stem for p in case.state.glob("candidates/*.json")),
                    [f"CAND-{THIRD_DAY}-01-tdd"],
                )


if __name__ == "__main__":
    unittest.main()
