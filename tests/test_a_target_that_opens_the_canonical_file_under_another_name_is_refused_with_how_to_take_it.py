"""A target that opens the canonical file under another name is refused with how to take it.

The target check (``assert_canonical_target`` in ``adapters/skill_fs.py``)
judges a candidate's target by its name. One file can carry more than one: on a
case-insensitive volume ``skills-canonical/tdd/SKILL.md`` and
``Skills-Canonical/tdd/SKILL.md`` open the same file, as two hard links do.
Measured before (3.11 and 3.13, a committed run with one validated candidate,
its token taken, then the config's canonical root spelled ``Skills-Canonical``
or the skill's directory renamed ``TDD``, on a case-insensitive volume):
``review`` and ``apply`` each exited 3, refusing the target as "not a skill
document in any configured root", and asked that the file be made the skill's
canonical copy again, which it was; ``review`` changed nothing and ``apply``
only the run lock, the record stayed ``validated`` and no backup was made. With
the old name given back, both took the candidate. So did a hard link outside
every root (the canonical root re-pointed at a copy of the tree whose
``tdd/SKILL.md`` links the old file), with the root pointed back. The refusal is
kept; its words now name the copy whose file the target opens and say what lets
``review`` and ``apply`` take the candidate: when the two names differ only in
letter case, the spelling given back; when they differ by more, discovery taking
the file by the candidate's name again, as it did when the candidate was written.
"""

from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters import skill_fs
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.config import load_config
from agent_skill_improver.errors import MirrorEditRefused

from tests.support import build_env

DAY = "2026-02-01"
REFUSED = 3
CID = "CAND-2026-02-01-01-tdd"
WHY = (
    "it opens the file discovery takes as the canonical copy of skill {skill}, {copy}, under "
    "another name (a case-insensitive volume, or a hard link, gives one file more than one), and "
    "the target check judges a target by its name, so it is not a skill document in any "
    "configured root. An unrecognised target is refused rather than assumed safe."
)
HOLDS = (
    ": while it holds the text the candidate was written against, `review` and `apply` take the "
    "candidate as before."
)
IN_CASE = (
    "The name the candidate holds and the copy's differ only in letter case. To take it instead, "
    "give the file back the spelling the candidate holds, in the config's root or the names on "
    "disk" + HOLDS
)
OTHERWISE = (
    "The name the candidate holds and the copy's differ by more than letter case, and a second "
    "name does not make a file the canonical copy. To take it instead, have discovery take the "
    "file by the name the candidate holds again, as it did when the candidate was written (a "
    "root re-pointed since then, pointed back)" + HOLDS
)
PLAIN = (
    "it is not a skill document in any configured root, so this tool cannot tell whether it is "
    "canonical. An unrecognised target is refused rather than assumed safe."
)
MAKE_IT = "If that file is meant to be the skill's canonical copy, make it that again"


def _case_insensitive() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "Probe").touch()
        return (Path(tmp) / "probe").exists()


CASE_INSENSITIVE = _case_insensitive()


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            rc = cli.main(["--config", str(config), *argv])
        except SystemExit as exc:  # pragma: no cover - the CLI returns its code
            rc = int(exc.code or 0)
    return rc, out.getvalue()


class Validated:
    """A committed run's one validated candidate, its token taken."""

    def __init__(self, base: Path) -> None:
        self.config = build_env(base / "env", seed="c228")
        self.env = (base / "env").resolve()
        rc, _ = run(self.config, "daily", "--commit", "--day", DAY)
        assert rc == 0, rc
        (self.record,) = (base / "env" / "state").glob("candidates/*.json")
        assert self.record.stem == CID, self.record.stem
        self.target = Path(json.loads(self.record.read_text())["target"]["canonical_path"])
        rc, out = run(self.config, "review", CID)
        match = re.search(r"--approve (\S+)", out)
        assert rc == 0 and match, out
        self.token = match.group(1)
        self.skill_bytes = self.target.read_bytes()
        self.record_bytes = self.record.read_bytes()

    def apply(self) -> tuple[int, str]:
        return run(self.config, "apply", CID, "--approve", self.token, "--i-reviewed-this")

    def status(self) -> str:
        return str(json.loads(self.record.read_text())["status"])

    def backups(self) -> list[Path]:
        return sorted((self.env / "state").rglob("backups/**/*"))


def refusal(out: str) -> str:
    return next(line for line in out.splitlines() if line.strip())


@unittest.skipUnless(CASE_INSENSITIVE, "needs a case-insensitive volume")
class OnACaseInsensitiveVolumeTests(unittest.TestCase):
    def test_review_names_the_copy_and_how_to_take_it_when_the_root_is_spelled_otherwise(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_ = Validated(Path(tmp))
            text = run_.config.read_text(encoding="utf-8")
            run_.config.write_text(text.replace("skills-canonical", "Skills-Canonical"))
            rc, out = run(run_.config, "review", CID)
            copy = run_.env / "Skills-Canonical" / "tdd" / "SKILL.md"
            self.assertEqual(rc, REFUSED)
            self.assertEqual(
                refusal(out),
                f"MirrorEditRefused: refusing to review {CID} at {run_.target}: "
                + WHY.format(skill="'tdd'", copy=copy)
                + " The file and the candidate are unchanged. The candidate stays open, and "
                "counts against [caps].max_open_candidates, until it is closed: to close it, "
                f"delete its record ({CID}.json in the state directory's candidates/). " + IN_CASE,
            )
            self.assertNotIn("--approve", out)
            self.assertEqual(run_.target.read_bytes(), run_.skill_bytes)
            self.assertEqual(run_.record.read_bytes(), run_.record_bytes)

    def test_apply_is_refused_in_the_same_words_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_ = Validated(Path(tmp))
            text = run_.config.read_text(encoding="utf-8")
            run_.config.write_text(text.replace("skills-canonical", "Skills-Canonical"))
            rc, out = run_.apply()
            copy = run_.env / "Skills-Canonical" / "tdd" / "SKILL.md"
            self.assertEqual(rc, REFUSED)
            line = refusal(out)
            self.assertTrue(
                line.startswith(f"MirrorEditRefused: refusing to apply {CID} at {run_.target}: "),
                line,
            )
            self.assertIn(WHY.format(skill="'tdd'", copy=copy), line)
            self.assertTrue(line.endswith(IN_CASE), line)
            self.assertEqual(run_.target.read_bytes(), run_.skill_bytes)
            self.assertEqual(run_.status(), "validated")
            self.assertEqual(run_.backups(), [])

    def test_a_directory_renamed_in_case_names_the_skill_discovery_now_takes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_ = Validated(Path(tmp))
            directory = run_.target.parent
            directory.rename(directory.with_name("tdd-moving"))
            directory.with_name("tdd-moving").rename(directory.with_name("TDD"))
            rc, out = run(run_.config, "review", CID)
            copy = directory.with_name("TDD") / "SKILL.md"
            self.assertEqual(rc, REFUSED)
            self.assertIn(WHY.format(skill="'TDD'", copy=copy), refusal(out))
            self.assertTrue(refusal(out).endswith(IN_CASE))

    def test_the_name_given_back_review_and_apply_take_the_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_ = Validated(Path(tmp))
            text = run_.config.read_text(encoding="utf-8")
            run_.config.write_text(text.replace("skills-canonical", "Skills-Canonical"))
            self.assertEqual(run(run_.config, "review", CID)[0], REFUSED)
            run_.config.write_text(text)
            rc, out = run(run_.config, "review", CID)
            self.assertEqual(rc, 0, out)
            self.assertIn(f"--approve {run_.token}", out)
            rc, out = run_.apply()
            self.assertEqual(rc, 0, out)
            self.assertEqual(run_.status(), "applied")

    def test_the_control_a_document_renamed_skill_md_keeps_the_plain_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_ = Validated(Path(tmp))
            run_.target.rename(run_.target.with_name("moving.md"))
            run_.target.with_name("moving.md").rename(run_.target.with_name("skill.md"))
            rc, out = run(run_.config, "review", CID)
            self.assertEqual(rc, REFUSED)
            self.assertIn(PLAIN, refusal(out))
            self.assertIn(MAKE_IT, refusal(out))
            self.assertNotIn("under another name", out)


class OnAnyVolumeTests(unittest.TestCase):
    """A hard link is another name for one file on any volume that has them."""

    def store(self, config: Path) -> FilesystemSkillStore:
        return FilesystemSkillStore(load_config(config).skill_roots, read_only=True)

    def linked(self, base: Path, source: Path) -> Path:
        elsewhere = base / "elsewhere"
        elsewhere.mkdir()
        link = elsewhere.resolve() / "SKILL.md"
        try:
            os.link(source, link)
        except OSError as exc:  # pragma: no cover - a volume without hard links
            self.skipTest(f"no hard link here: {exc}")
        return link

    def test_a_hard_link_to_the_canonical_file_is_named_as_that_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_env(Path(tmp) / "env", seed="c228")
            copy = (Path(tmp) / "env").resolve() / "skills-canonical" / "tdd" / "SKILL.md"
            link = self.linked(Path(tmp), copy)
            with self.assertRaises(MirrorEditRefused) as ctx:
                self.store(config).assert_canonical_target(link, action="review", candidate_id=CID)
            words = str(ctx.exception)
            self.assertTrue(words.startswith(f"refusing to review {CID} at {link}: "), words)
            self.assertIn(WHY.format(skill="'tdd'", copy=copy), words)
            self.assertTrue(words.endswith(" candidates/). " + OTHERWISE), words)

    def test_a_root_re_pointed_at_a_hard_link_is_taken_with_the_root_pointed_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_ = Validated(Path(tmp))
            old = run_.target.parent.parent
            new = old.with_name("skills-new")
            shutil.copytree(old, new)
            (new / "tdd" / "SKILL.md").unlink()
            try:
                os.link(run_.target, new / "tdd" / "SKILL.md")
            except OSError as exc:  # pragma: no cover - a volume without hard links
                self.skipTest(f"no hard link here: {exc}")
            text = run_.config.read_text(encoding="utf-8")
            run_.config.write_text(text.replace("skills-canonical", "skills-new"))
            copy = new / "tdd" / "SKILL.md"
            for command, argv in (
                ("review", ("review", CID)),
                ("apply", ("apply", CID, "--approve", run_.token, "--i-reviewed-this")),
            ):
                with self.subTest(command=command):
                    rc, out = run(run_.config, *argv)
                    line = refusal(out)
                    self.assertEqual(rc, REFUSED, out)
                    self.assertTrue(
                        line.startswith(
                            f"MirrorEditRefused: refusing to {command} {CID} at {run_.target}: "
                        ),
                        line,
                    )
                    self.assertIn(WHY.format(skill="'tdd'", copy=copy), line)
                    self.assertTrue(line.endswith(" candidates/). " + OTHERWISE), line)
                    self.assertEqual(run_.target.read_bytes(), run_.skill_bytes)
                    self.assertEqual(run_.status(), "validated")
                    self.assertEqual(run_.backups(), [])
            run_.config.write_text(text)
            rc, out = run(run_.config, "review", CID)
            self.assertEqual(rc, 0, out)
            self.assertIn(f"--approve {run_.token}", out)
            rc, out = run_.apply()
            self.assertEqual(rc, 0, out)
            self.assertEqual(run_.status(), "applied")

    def test_two_names_are_one_spelling_but_for_letter_case_by_casefold(self) -> None:
        same = skill_fs._in_letter_case_only
        self.assertTrue(
            same(Path("/r/Skills-Canonical/TDD/SKILL.md"), Path("/r/skills-canonical/tdd/SKILL.md"))
        )
        self.assertTrue(same(Path("/r/Straße/SKILL.md"), Path("/r/STRASSE/SKILL.md")))
        self.assertFalse(
            same(Path("/r/skills-canonical/tdd/SKILL.md"), Path("/r/skills-new/tdd/SKILL.md"))
        )
        self.assertFalse(
            same(Path("/r/elsewhere/SKILL.md"), Path("/r/skills-canonical/tdd/SKILL.md"))
        )

    def test_without_a_candidate_the_words_name_no_way_to_take_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_env(Path(tmp) / "env", seed="c228")
            copy = (Path(tmp) / "env").resolve() / "skills-canonical" / "tdd" / "SKILL.md"
            link = self.linked(Path(tmp), copy)
            with self.assertRaises(MirrorEditRefused) as ctx:
                self.store(config).assert_canonical_target(link, action="roll back")
            self.assertEqual(
                str(ctx.exception),
                f"refusing to roll back {link}: " + WHY.format(skill="'tdd'", copy=copy),
            )

    def test_a_hard_link_to_a_mirror_copy_keeps_the_plain_words(self) -> None:
        # ``tdd`` has a canonical copy elsewhere; ``mirror-only`` has none, so
        # its mirror copy is the only copy there is: neither is the canonical one.
        for skill in ("tdd", "mirror-only"):
            with self.subTest(skill=skill), tempfile.TemporaryDirectory() as tmp:
                config = build_env(Path(tmp) / "env", seed="c228")
                mirror = (Path(tmp) / "env").resolve() / "skills-mirror" / skill / "SKILL.md"
                link = self.linked(Path(tmp), mirror)
                with self.assertRaises(MirrorEditRefused) as ctx:
                    self.store(config).assert_canonical_target(
                        link, action="review", candidate_id=CID
                    )
                self.assertIn(PLAIN, str(ctx.exception))
                self.assertIn(MAKE_IT, str(ctx.exception))

    def test_a_target_that_opens_no_file_keeps_the_plain_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_env(Path(tmp) / "env", seed="c228")
            gone = Path(tmp).resolve() / "elsewhere" / "SKILL.md"
            with self.assertRaises(MirrorEditRefused) as ctx:
                self.store(config).assert_canonical_target(gone, action="review", candidate_id=CID)
            self.assertIn(PLAIN, str(ctx.exception))
            self.assertIn(MAKE_IT, str(ctx.exception))

    def test_a_stat_that_fails_gives_no_copy_to_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = build_env(Path(tmp) / "env", seed="c228")
            copy = (Path(tmp) / "env").resolve() / "skills-canonical" / "tdd" / "SKILL.md"
            link = self.linked(Path(tmp), copy)
            groups = self.store(config).groups(refresh=True)
            real = os.stat

            def refusing(refused: Path):
                def stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
                    if Path(os.fspath(path)) == refused:  # type: ignore[arg-type]
                        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), path)
                    return real(path, *args, **kwargs)  # type: ignore[arg-type]

                return stat

            self.assertEqual(skill_fs._canonical_copy_opened_by(groups, link).path, copy)
            for refused in (link, copy):
                with self.subTest(refused=refused.parent.name):
                    with mock.patch.object(skill_fs.os, "stat", refusing(refused)):
                        self.assertIsNone(skill_fs._canonical_copy_opened_by(groups, link))


if __name__ == "__main__":
    unittest.main()
