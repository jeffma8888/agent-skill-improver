"""A refusal from the pinned walk names the directory it was opening.

``FilesystemSkillStore._open_parent_pinned`` opens each directory between the
root and the document relative to the one before, refusing a link (``ELOOP``,
``ENOTDIR``) or a directory it cannot open. Measured before (3.11 and 3.13, the
same on both), calling it as the existing handle-walk tests do: for a document
at ``<root>/x/tdd/SKILL.md`` whose ``x/tdd`` was a link, or search-only (mode
0o311, so the document reads but the directory does not open), the refusal
named ``<root>/tdd``, the root joined to the last component alone, a directory
not on the way to the document; at ``<root>/a/b/tdd`` it named ``<root>/tdd``
too. A document right under the root was named correctly. Each refusal now
names the root joined to every component walked so far. Through the CLI the
walk is the second lock: the lstat walk and discovery refuse such a copy first,
so a refusal from here takes a change between the two checks.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.errors import BaseHashMismatch, LinkedTargetRefused
from agent_skill_improver.domain.skills import SKILL_FILENAME, SkillRoot

DOCUMENT = "---\nname: tdd\n---\n# tdd\n"
LINKED = "is a symbolic link or not a directory on the way to the document"


class APinnedWalkRefusalNamesTheWholeWayTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve() / "root"
        for rel in ("x/tdd", "a/b/tdd", "tdd"):
            (self.root / rel).mkdir(parents=True)
            (self.root / rel / SKILL_FILENAME).write_text(DOCUMENT, encoding="utf-8")
        self.store = FilesystemSkillStore(
            [SkillRoot(id="r", path=self.root, canonical=True)], read_only=False
        )

    def refusal(self, document: Path, kind: type[Exception]) -> str:
        with self.assertRaises(kind) as ctx:
            fd, _leaf = self.store._open_parent_pinned(document, verb="write")
            os.close(fd)
        return str(ctx.exception)

    def skip_as_root(self) -> None:
        """Only the search-only case: root opens such a directory, and still refuses a link."""
        if os.geteuid() == 0:
            self.skipTest("root opens a search-only directory")

    def search_only(self, directory: Path) -> None:
        """``directory`` made search-only: the document reads, the directory does not open."""
        os.chmod(directory, 0o311)
        self.addCleanup(os.chmod, directory, 0o755)

    def linked(self, directory: Path) -> None:
        """``directory`` replaced by a link to itself under another name, put back after."""
        real = directory.with_name(directory.name + ".real")
        directory.rename(real)
        try:
            os.symlink(real, directory)
        except (OSError, NotImplementedError):  # pragma: no cover - permissions
            real.rename(directory)
            self.skipTest("cannot create a symlink here")

        def put_back() -> None:
            directory.unlink()
            real.rename(directory)

        self.addCleanup(put_back)

    @unittest.skipIf(os.geteuid() == 0, "root opens a search-only directory")
    def test_an_unopenable_directory_below_another_is_named_whole(self) -> None:
        nested = self.root / "x" / "tdd"
        self.search_only(nested)
        document = nested / SKILL_FILENAME
        self.assertTrue(document.read_text(encoding="utf-8"), "the document still reads")
        self.assertEqual(
            self.refusal(document, BaseHashMismatch),
            f"refusing to write {document}: {nested} could not be opened (Permission denied). "
            "Nothing was touched.",
        )

    def test_a_linked_directory_below_another_is_named_whole(self) -> None:
        nested = self.root / "x" / "tdd"
        self.linked(nested)
        document = nested / SKILL_FILENAME
        self.assertEqual(
            self.refusal(document, LinkedTargetRefused),
            f"refusing to write {document}: {nested} {LINKED}, judged on the open handle. "
            "Nothing was touched.",
        )

    def test_a_directory_in_the_middle_of_the_way_is_named_whole(self) -> None:
        # Not the document's own directory: the one above it, so a refusal that
        # named the document's directory instead would be caught here.
        middle = self.root / "a" / "b"
        document = middle / "tdd" / SKILL_FILENAME
        with self.subTest(refusal="could not be opened"):
            self.skip_as_root()
            os.chmod(middle, 0o311)
            try:
                message = self.refusal(document, BaseHashMismatch)
            finally:
                os.chmod(middle, 0o755)
            self.assertIn(f": {middle} could not be opened", message)
        with self.subTest(refusal="linked"):
            self.linked(middle)
            self.assertIn(f": {middle} {LINKED}", self.refusal(document, LinkedTargetRefused))

    def test_two_directories_deep_each_refusal_names_the_whole_way(self) -> None:
        nested = self.root / "a" / "b" / "tdd"
        document = nested / SKILL_FILENAME
        with self.subTest(refusal="could not be opened"):
            self.skip_as_root()
            os.chmod(nested, 0o311)
            try:
                message = self.refusal(document, BaseHashMismatch)
            finally:
                os.chmod(nested, 0o755)
            self.assertIn(f": {nested} could not be opened", message)
        with self.subTest(refusal="linked"):
            self.linked(nested)
            self.assertIn(f": {nested} {LINKED}", self.refusal(document, LinkedTargetRefused))

    def test_the_control_a_directory_right_under_the_root_is_named_as_before(self) -> None:
        own = self.root / "tdd"
        self.linked(own)
        document = own / SKILL_FILENAME
        self.assertEqual(
            self.refusal(document, LinkedTargetRefused),
            f"refusing to write {document}: {own} {LINKED}, judged on the open handle. "
            "Nothing was touched.",
        )


if __name__ == "__main__":
    unittest.main()
