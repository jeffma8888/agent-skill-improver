"""A refusal of a leaf this process cannot open or read names its path once.

The read of a state record (a candidate, a conversation, a checkpoint), of a
backup manifest and the run lock's open, stat and chmod each refuse by naming
the path, then said why with ``str()`` of the error, which for an ``OSError``
ends with the file name it was given: the path a second time, raw, where the
refusal had masked or cut it. The reason is now ``why_unreadable``: the error's
number and the system's words for it. Measured before (3.11 and 3.13, the same
on both), with each leaf at mode 000, or a directory at the lock's path, under a
state directory named with a ``ghp_`` token: a backup manifest's refusal masked
the token in its name and printed it whole in the error (315 characters, 203
after); a candidate record's printed the path twice (425, 292 after), a
conversation record's twice (559, 373 after) and the lock's twice (``daily``
361, 257 after); under a path of about 900 characters a conversation record's
refusal was 2,087 characters (1,137 after) and a backup manifest's 1,529, the
path cut at 512 in its name and whole in the error (653 after). A record that
does not parse keeps the parser's words, which name no file.
"""

from __future__ import annotations

import errno
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app.lock import RunLock
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.domain.redaction import NAMED_PATH_CHARS, why_unreadable
from agent_skill_improver.domain.skills import SkillRoot
from agent_skill_improver.errors import BackupUnusable, SafetyRefusal, StateUnreadable

TOKEN = "gh" + "p_" + "Zq8vLm3pXr7tWk2sQ1aB0123456789abcdef"
MASKED = "[REDACTED:github-token]"
DENIED = f"[Errno {errno.EACCES}] {os.strerror(errno.EACCES)}"
CANDIDATE_ID = "CAND-2026-01-02-01-tdd"
DAY = "2026-01-02"


class _Tree(unittest.TestCase):
    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()

    def unreadable(self, path: Path) -> None:
        """``path`` at mode 000, put back before the tree is removed."""
        if os.geteuid() == 0:
            self.skipTest("root reads a file at mode 000")
        os.chmod(path, 0)
        self.addCleanup(os.chmod, path, 0o600)


class ABackupManifestItCannotReadTests(_Tree):
    def refusal(self, folder: Path) -> str:
        folder.mkdir(parents=True)
        manifest = folder / "backup.json"
        manifest.write_text("{}", encoding="utf-8")
        self.unreadable(manifest)
        root = self.tmp / "root"
        root.mkdir()
        store = FilesystemSkillStore([SkillRoot(id="r", path=root, canonical=True)])
        with self.assertRaises(BackupUnusable) as ctx:
            store.rollback(manifest)
        return str(ctx.exception)

    def test_a_token_in_its_path_is_masked_and_not_printed_again(self):
        message = self.refusal(self.tmp / f"st-{TOKEN}")
        self.assertNotIn(TOKEN, message)
        self.assertEqual(message.count(MASKED), 1)
        self.assertEqual(message.count("backup.json"), 1)
        self.assertIn(f"could not be read ({DENIED}). No file was touched.", message)
        self.assertEqual(len(message.splitlines()), 1)

    def test_a_long_path_is_cut_once(self):
        folder = self.tmp.joinpath(*(["d" * 200] * 4))
        message = self.refusal(folder)
        self.assertNotIn(str(folder), message)
        self.assertEqual(message.count("more chars not quoted]"), 1)
        self.assertLess(len(message), NAMED_PATH_CHARS + 200)
        self.assertIn(f"could not be read ({DENIED}). No file was touched.", message)


class AStateRecordItCannotReadTests(_Tree):
    def store(self) -> StateStore:
        return StateStore(self.tmp / f"st-{TOKEN}")

    def test_a_candidate_record_names_its_path_once(self):
        store = self.store()
        path = store.candidate_path(CANDIDATE_ID)
        path.write_text("{}", encoding="utf-8")
        self.unreadable(path)
        with self.assertRaises(StateUnreadable) as ctx:
            store.read_candidate(CANDIDATE_ID)
        message = str(ctx.exception)
        self.assertEqual(
            message,
            f"refusing to use the candidate record {path}: it could not be read ({DENIED}). "
            "A truncated or hand-edited record is never guessed at.",
        )
        self.assertEqual(message.count(TOKEN), 1)

    def test_a_conversation_record_names_its_path_once(self):
        store = self.store()
        path = store.write_conversation("conv-1", {"x": 1})
        self.unreadable(path)
        with self.assertRaises(StateUnreadable) as ctx:
            store.read_conversation("conv-1")
        message = str(ctx.exception)
        self.assertIn(
            f"refusing to use the conversation record {path}: it could not be read "
            f"({DENIED}). ",
            message,
        )
        self.assertEqual(message.count(TOKEN), 1)

    def test_a_checkpoint_record_names_its_path_once(self):
        store = self.store()
        path = store.write_checkpoint(DAY, {"schema": "asi.checkpoint.v1", "day": DAY})
        self.unreadable(path)
        with self.assertRaises(StateUnreadable) as ctx:
            store.read_checkpoint(DAY)
        message = str(ctx.exception)
        self.assertIn(
            f"refusing to use the checkpoint record {path}: it could not be read ({DENIED}). ",
            message,
        )
        self.assertEqual(message.count(TOKEN), 1)

    def test_a_record_that_does_not_parse_keeps_the_parsers_words(self):
        store = self.store()
        path = store.write_conversation("conv-1", {"x": 1})
        path.write_text("{", encoding="utf-8")
        with self.assertRaises(StateUnreadable) as ctx:
            store.read_conversation("conv-1")
        try:
            json.loads("{")
        except ValueError as exc:
            words = str(exc)
        self.assertIn(f"{path}: it could not be read ({words}). ", str(ctx.exception))


class TheRunLockItCannotUseTests(_Tree):
    def setUp(self) -> None:
        super().setUp()
        self.path = self.tmp / f"st-{TOKEN}" / "locks" / "daily.lock"

    def refusal(self) -> str:
        with self.assertRaises(SafetyRefusal) as ctx:
            with RunLock(self.path):
                pass
        return str(ctx.exception)

    def test_a_directory_at_its_path_is_named_once(self):
        self.path.mkdir(parents=True)
        self.assertEqual(
            self.refusal(),
            f"refusing to use {self.path} as a run lock: it could not be opened "
            f"([Errno {errno.EISDIR}] {os.strerror(errno.EISDIR)}). Remove the entry at that "
            "path if you did not put it there.",
        )

    def failing(self, name: str, code: int) -> str:
        """The refusal when ``os.<name>`` fails with ``code`` on the open lock."""
        error = OSError(code, os.strerror(code), str(self.path))
        with mock.patch(f"agent_skill_improver.app.lock.os.{name}", side_effect=error):
            return self.refusal()

    def test_a_stat_that_fails_is_named_by_its_number(self):
        self.assertEqual(
            self.failing("fstat", errno.EIO),
            f"refusing to use {self.path} as a run lock: cannot stat it "
            f"([Errno {errno.EIO}] {os.strerror(errno.EIO)})",
        )

    def test_a_chmod_that_fails_is_named_by_its_number(self):
        self.assertEqual(
            self.failing("fchmod", errno.EPERM),
            f"refusing to use {self.path} as a run lock: cannot enforce mode 0600 "
            f"([Errno {errno.EPERM}] {os.strerror(errno.EPERM)})",
        )


class WhyUnreadableTests(unittest.TestCase):
    def test_an_error_that_is_not_an_os_error_is_its_own_text(self):
        error = ValueError("Expecting value: line 1 column 1 (char 0)")
        self.assertEqual(why_unreadable(error), "Expecting value: line 1 column 1 (char 0)")

    def test_an_os_error_of_any_kind_drops_its_file_name(self):
        error = PermissionError(errno.EACCES, os.strerror(errno.EACCES), f"/x/{TOKEN}")
        self.assertEqual(why_unreadable(error), DENIED)

    def test_an_os_error_with_no_words_is_quoted_whole_its_file_name_masked(self):
        error = OSError(errno.EIO, "", f"/x/{TOKEN}/backup.json")
        self.assertEqual(
            why_unreadable(error), f"\"[Errno {errno.EIO}] : '/x/{MASKED}/backup.json'\""
        )
        cut = why_unreadable(OSError(errno.EIO, "", "/x/" + "d" * 300))
        self.assertTrue(cut.endswith("... [199 more chars not quoted]"), cut)


if __name__ == "__main__":
    unittest.main()
