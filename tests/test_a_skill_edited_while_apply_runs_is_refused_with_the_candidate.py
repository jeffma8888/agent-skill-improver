"""A skill edited while ``apply`` runs is refused with the candidate and what proposing again takes.

``apply`` (``app/promote.py``) checks the live skill file against the
candidate's base before it writes, and refuses a file that changed with the
candidate's id and ``repropose_advice``. The write (``write_canonical`` in
``adapters/skill_fs.py``) checks the hash again on its own pinned read, so a
file edited after ``apply``'s check and before the write is refused there.
Measured before (3.11 and 3.13, a committed run with one validated candidate,
the skill file edited between the two checks): ``apply`` exited 3 with
"refusing to write <path>: on-disk sha256 is ... The skill changed since the
proposal was made -- re-run validation against the new base.", which named no
candidate and gave advice no command carries out (there is no validate
command). The record stayed ``validated``, the skill kept the other edit, and
``review`` then refused the candidate with ``repropose_advice``. ``apply``
now passes the candidate's id to the write, and the write's refusal says what
``apply``'s own check says; without an id it gives no advice.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Callable

from agent_skill_improver import cli
from agent_skill_improver.app import promote
from agent_skill_improver.domain.integrity import repropose_advice
from agent_skill_improver.domain.skills import SKILL_FILENAME
from agent_skill_improver.errors import BaseHashMismatch

from tests.support import build_env
from tests.test_skill_identity_and_payloads import GOOD, Harness

DAY, NEXT_DAY = "2026-02-01", "2026-02-02"
OLD_ADVICE = "re-run validation against the new base"
EDIT = "\nedited by someone else while apply ran\n"
REFUSED = 3
STALE = "0" * 64


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


class _Raced:
    """One validated candidate whose skill file is edited after ``apply``'s check, before its write.

    ``_refuse_unwritable_receipt`` is the last step ``apply`` takes before it
    calls the write, so the edit lands between the two hash checks.
    """

    def __init__(self, base: Path, effect: Callable[[Path], None]):
        self.config = build_env(base / "env", seed="c220")
        self.state = base / "env" / "state"
        rc, said = run(self.config, "daily", "--commit", "--day", DAY)
        assert rc == 0, said
        (self.record_file,) = sorted(self.state.glob("candidates/*.json"))
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self.cid = record["id"]
        self.skill = Path(record["target"]["canonical_path"])
        self.original = self.skill.read_bytes()
        rc, shown = run(self.config, "review", self.cid)
        assert rc == 0, shown
        found = re.search(r"--approve (\S+)", shown)
        assert found, shown
        self.token = found.group(1)
        checked = promote._refuse_unwritable_receipt

        def then_edit(state, action, candidate):  # type: ignore[no-untyped-def]
            checked(state, action, candidate)
            effect(Path(candidate.target.canonical_path))

        promote._refuse_unwritable_receipt = then_edit  # type: ignore[assignment]
        try:
            self.rc, self.said = self.apply()
        finally:
            promote._refuse_unwritable_receipt = checked  # type: ignore[assignment]

    def apply(self) -> tuple[int, str]:
        return run(self.config, "apply", self.cid, "--approve", self.token, "--i-reviewed-this")

    def status(self) -> str:
        return str(json.loads(self.record_file.read_text(encoding="utf-8"))["status"])


def edit(path: Path) -> None:
    path.write_text(path.read_text(encoding="utf-8") + EDIT, encoding="utf-8")


class TheStoreRefusalTests(Harness):
    """The write's own refusal, asked directly of the store."""

    def refusal(self, **kwargs: str) -> tuple[str, Path, Path]:
        spelled = self.canonical_dir / "shared" / SKILL_FILENAME
        # The store names the path it resolved: a temporary directory under a
        # link (macOS's /var and /tmp are links) is named by where it leads.
        target = spelled.resolve()
        backup = self.base / "bak"
        with self.assertRaises(BaseHashMismatch) as caught:
            self.store(read_only=False).write_canonical(
                spelled,
                GOOD + "\n- an approved addition\n",
                expected_sha256=STALE,
                backup_dir=backup,
                skill_id="shared",
                **kwargs,
            )
        return str(caught.exception), target, backup

    def test_the_store_refusal_names_the_candidate_it_is_given(self) -> None:
        cid = "CAND-2026-02-01-01-shared"
        said, target, backup = self.refusal(candidate_id=cid)
        self.assertTrue(said.startswith(f"refusing to apply {cid}: {target} now hashes "), said)
        self.assertIn(f"but the candidate was written against {STALE[:12]}.", said)
        self.assertIn(
            "The skill changed while this apply ran, after it checked the file and before it "
            "wrote, and the skill was not written. To propose and validate against the new "
            f"base, {repropose_advice(cid)}.",
            said,
        )
        self.assertNotIn(OLD_ADVICE, said)
        self.assertEqual(target.read_text(encoding="utf-8"), GOOD)
        self.assertFalse(backup.exists())

    def test_without_a_candidate_it_gives_no_advice_it_cannot_back(self) -> None:
        said, target, backup = self.refusal()
        self.assertTrue(said.startswith(f"refusing to write {target}: on-disk sha256 is "), said)
        self.assertTrue(
            said.endswith(
                f"but the write was asked against {STALE[:12]}, so the file changed after that "
                "hash was taken. Nothing was written."
            ),
            said,
        )
        self.assertNotIn(OLD_ADVICE, said)
        self.assertNotIn("candidate", said)
        self.assertEqual(target.read_text(encoding="utf-8"), GOOD)
        self.assertFalse(backup.exists())


class AnEditWhileApplyRunsTests(unittest.TestCase):
    """``apply`` of a candidate whose skill file is edited between its check and its write."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.case = _Raced(Path(cls._tmp.name), edit)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_apply_names_the_candidate(self) -> None:
        self.assertEqual(self.case.rc, REFUSED, self.case.said)
        self.assertIn(
            f"BaseHashMismatch: refusing to apply {self.case.cid}: {self.case.skill} now hashes ",
            self.case.said,
        )
        # The write's refusal, not the check ``apply`` makes before it, which
        # names the candidate in the same words up to here.
        self.assertIn(
            "The skill changed while this apply ran, after it checked the file and before it wrote",
            self.case.said,
        )

    def test_apply_gives_the_advice_review_then_gives(self) -> None:
        advice = "To propose and validate against the new base, " + repropose_advice(self.case.cid)
        self.assertIn(advice, self.case.said)
        self.assertNotIn(OLD_ADVICE, self.case.said)
        rc, reviewed = run(self.case.config, "review", self.case.cid)
        self.assertEqual(rc, REFUSED, reviewed)
        self.assertIn("BaseHashMismatch", reviewed)
        self.assertIn(advice, reviewed)


class WhatTheRefusalLeavesTests(unittest.TestCase):
    """The state the refusal leaves, and what its advice and its alternative then do."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.case = _Raced(Path(self._tmp.name), edit)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_the_skill_and_the_record_are_left_as_they_were(self) -> None:
        self.assertEqual(self.case.rc, REFUSED, self.case.said)
        self.assertEqual(self.case.skill.read_bytes(), self.case.original + EDIT.encode("utf-8"))
        self.assertEqual(self.case.status(), "validated")
        self.assertEqual(list(self.case.state.glob("backups/**/backup.json")), [])

    def test_following_the_advice_proposes_again_another_day(self) -> None:
        self.case.record_file.unlink()
        rc, said = run(self.case.config, "daily", "--commit", "--day", NEXT_DAY)
        self.assertEqual(rc, 0, said)
        self.assertEqual(
            sorted(p.stem for p in self.case.state.glob("candidates/*.json")),
            [f"CAND-{NEXT_DAY}-01-tdd"],
        )

    def test_the_text_put_back_applies_with_the_same_token(self) -> None:
        self.case.skill.write_bytes(self.case.original)
        rc, said = self.case.apply()
        self.assertEqual(rc, 0, said)
        self.assertEqual(self.case.status(), "applied")


if __name__ == "__main__":
    unittest.main()
