"""A directory the payload inventory can list but not search is an unknown, not an absence.

The payload inventory (``_payload_files`` in ``adapters/skill_fs.py``) walks a
skill's directory and judges each entry the listing names with ``lstat``,
then by its name and mode. A directory this process may list but not search
(mode 0o444) gives its names and nothing else, and every ``lstat`` of an entry
in it was refused and stepped over in silence. Measured before (3.11 and 3.13,
``tdd``'s ``scripts/`` at 0o444 holding ``hook.sh`` and an executable
``tool``): ``skills --json`` gave ``payload_files`` empty, ``unlistable_dirs``
empty and ``payload_inventory_complete`` true; ``doctor`` counted no skill
carrying payloads and warned of nothing; ``daily`` proposed against ``tdd``
with no risk note. At mode 000 the same directory was named as one that could
not be listed. Such a directory is now named too, once, as "listed, but an
entry in it could not be examined: <the system's words>", the inventory is
incomplete, and an entry whose name marks it a payload is counted by that
name when the listing itself gives it as a file or a link; where the listing
gives no type, it is not, so the count stays a floor. The words that name
these directories say "could not be listed or fully examined".
"""

from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.config import load_config

from tests.support import build_env

DAY = "2026-02-01"
UNSEARCHED = "listed, but an entry in it could not be examined: Permission denied"


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


def with_scripts(base: Path, *, subfolder: bool = False) -> tuple[Path, Path]:
    """A built env whose ``tdd`` holds ``scripts/hook.sh`` and an executable ``scripts/tool``."""
    config = build_env(base / "env", seed="c227")
    scripts = base / "env" / "skills-canonical" / "tdd" / "scripts"
    scripts.mkdir()
    for name in ("hook.sh", "tool"):
        (scripts / name).write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        os.chmod(scripts / name, 0o755)
    if subfolder:
        # A directory named like a payload: the walk lists it as a directory.
        (scripts / "bin.sh").mkdir()
    return config, scripts


def tdd_snapshot(config: Path):
    loaded = load_config(config)
    store = FilesystemSkillStore(loaded.skill_roots, read_only=True)
    (snap,) = [
        s for s in store.snapshots(refresh=True) if s.skill_id == "tdd" and s.root_id == "canonical"
    ]
    return snap


@unittest.skipIf(os.geteuid() == 0, "root searches a directory of mode 0o444")
class TheStoresInventoryTests(unittest.TestCase):
    def test_such_a_directory_is_named_once_and_a_payload_name_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config, scripts = with_scripts(Path(tmp), subfolder=True)
            scripts.chmod(0o444)
            try:
                snap = tdd_snapshot(config)
                mode = os.stat(scripts).st_mode & 0o777
            finally:
                scripts.chmod(0o755)
            self.assertEqual(mode, 0o444, "the inventory changes no mode")
            # ``tool`` is executable only by its mode, which cannot be read;
            # ``bin.sh`` is a directory, which the walk could not open.
            self.assertEqual(snap.payload_files, ("scripts/hook.sh",))
            self.assertEqual(
                snap.unlistable_dirs,
                (("scripts/bin.sh", "Permission denied"), ("scripts", UNSEARCHED)),
            )
            self.assertFalse(snap.payload_inventory_complete)
            (warning,) = [w for w in snap.warnings if "fully examined" in w]
            self.assertIn(
                "2 directories in this skill directory could not be listed or fully examined "
                f"(scripts/bin.sh (Permission denied), scripts ({UNSEARCHED})): whether "
                "executable payloads live there is unknown, so the payload inventory above is "
                "incomplete",
                warning,
            )

    def test_the_control_the_directory_searchable_again_is_a_complete_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config, scripts = with_scripts(Path(tmp))
            scripts.chmod(0o444)
            scripts.chmod(0o755)
            snap = tdd_snapshot(config)
            self.assertEqual(snap.payload_files, ("scripts/hook.sh", "scripts/tool"))
            self.assertEqual(snap.unlistable_dirs, ())
            self.assertTrue(snap.payload_inventory_complete)


class Untyped:
    """A listed entry as a file system without ``d_type`` gives it: its type needs a stat."""

    def __init__(self, entry: os.DirEntry[str]) -> None:
        self._entry = entry
        self.name = entry.name
        self.path = entry.path

    def _refused(self, *args: object, **kwargs: object) -> bool:
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), self.path)

    is_dir = is_file = is_symlink = _refused

    def __getattr__(self, name: str) -> object:
        return getattr(self._entry, name)


class UntypedListing:
    def __init__(self, listing: object) -> None:
        self._listing = listing

    def __enter__(self) -> UntypedListing:
        return self

    def __exit__(self, *exc: object) -> None:
        self._listing.close()  # type: ignore[attr-defined]

    def __iter__(self) -> UntypedListing:
        return self

    def __next__(self) -> Untyped:
        return Untyped(next(self._listing))  # type: ignore[call-overload]


@unittest.skipIf(os.geteuid() == 0, "root searches a directory of mode 0o444")
class TheTypeTheListingGivesTests(unittest.TestCase):
    """A name is counted only for an entry the listing itself gives as a file or a link."""

    def test_a_link_named_like_a_payload_is_counted_as_the_examined_path_counts_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config, scripts = with_scripts(Path(tmp))
            (scripts / "launch.sh").symlink_to("elsewhere")
            examined = tdd_snapshot(config).payload_files
            scripts.chmod(0o444)
            try:
                listed = tdd_snapshot(config).payload_files
            finally:
                scripts.chmod(0o755)
        self.assertEqual(examined, ("scripts/hook.sh", "scripts/launch.sh", "scripts/tool"))
        self.assertEqual(listed, ("scripts/hook.sh", "scripts/launch.sh"))

    def test_where_the_listing_gives_no_type_no_name_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config, scripts = with_scripts(Path(tmp), subfolder=True)
            shut = scripts.resolve()
            real = os.scandir

            def scandir(path: object) -> object:
                listing = real(path)  # type: ignore[call-overload]
                if Path(os.fspath(path)).resolve() == shut:  # type: ignore[arg-type]
                    return UntypedListing(listing)
                return listing

            scripts.chmod(0o444)
            try:
                with mock.patch("os.scandir", scandir):
                    snap = tdd_snapshot(config)
            finally:
                scripts.chmod(0o755)
        # ``os.walk`` files the directory ``bin.sh`` among the files, as it
        # does when the type cannot be had; neither it nor ``hook.sh`` counts.
        self.assertEqual(snap.payload_files, ())
        self.assertEqual(snap.unlistable_dirs, (("scripts", UNSEARCHED),))
        self.assertFalse(snap.payload_inventory_complete)

    def test_a_directory_shut_after_the_walk_listed_it_counts_no_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config, scripts = with_scripts(Path(tmp))
            skill_dir = scripts.parent.resolve()
            real = os.walk

            def walk(top: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
                for dirpath, dirnames, filenames in real(top, *args, **kwargs):
                    yield dirpath, dirnames, filenames
                if Path(os.fspath(top)).resolve() == skill_dir:  # type: ignore[arg-type]
                    # Shut once the walk has listed it: what is examined next is refused.
                    scripts.chmod(0o000)

            try:
                with mock.patch("os.walk", walk):
                    snap = tdd_snapshot(config)
            finally:
                scripts.chmod(0o755)
        self.assertEqual(snap.payload_files, ())
        self.assertEqual(snap.unlistable_dirs, (("scripts", UNSEARCHED),))


@unittest.skipIf(os.geteuid() == 0, "root searches a directory of mode 0o444")
class WhatTheCommandsSayTests(unittest.TestCase):
    """``tdd``'s ``scripts/`` at 0o444 holding ``hook.sh`` and an executable ``tool``."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.config, cls.scripts = with_scripts(Path(cls._tmp.name))
        cls.state = Path(cls._tmp.name) / "env" / "state"
        cls.scripts.chmod(0o444)
        try:
            cls.said = {
                "skills": run(cls.config, "--json", "skills"),
                "doctor": run(cls.config, "--json", "doctor"),
                "daily": run(cls.config, "--json", "daily", "--commit", "--day", DAY),
            }
            (record,) = sorted(cls.state.glob("candidates/*.json"))
            cls.record = json.loads(record.read_text(encoding="utf-8"))
            cls.said["review"] = run(cls.config, "review", cls.record["id"])
            cls.mode = os.stat(cls.scripts).st_mode & 0o777
        finally:
            cls.scripts.chmod(0o755)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_skills_gives_the_payload_by_name_and_the_inventory_as_incomplete(self) -> None:
        rc, said = self.said["skills"]
        self.assertEqual(rc, 0, said)
        (group,) = [g for g in json.loads(said)["groups"] if g["skill_id"] == "tdd"]
        self.assertEqual(group["payload_files"], ["canonical:scripts/hook.sh"])
        self.assertEqual(group["unlistable_dirs"], ["canonical:scripts"])
        self.assertEqual(
            group["canonical"]["unlistable_dirs"], [{"path": "scripts", "error": UNSEARCHED}]
        )
        self.assertFalse(group["canonical"]["payload_inventory_complete"])

    def test_doctor_counts_the_payload_and_warns_the_count_is_a_floor(self) -> None:
        rc, said = self.said["doctor"]
        self.assertEqual(rc, 0, said)
        payload = json.loads(said)
        self.assertEqual(
            payload["skills_with_executable_payloads"],
            [{"skill_id": "tdd", "payload_files": ["scripts/hook.sh"]}],
        )
        self.assertEqual(
            payload["skills_with_unlistable_directories"],
            [
                {
                    "skill_id": "tdd",
                    "root_id": "canonical",
                    "unlistable_dirs": [{"path": "scripts", "error": UNSEARCHED}],
                }
            ],
        )
        self.assertIn(
            "skill 'tdd' in root 'canonical' has 1 directory the payload inventory could not "
            f"list or fully examine (scripts ({UNSEARCHED})); its payload count is a floor, not "
            "an inventory",
            payload["warnings"],
        )

    def test_daily_proposes_with_both_risk_notes_and_review_shows_them(self) -> None:
        rc, said = self.said["daily"]
        self.assertEqual(rc, 0, said)
        tree = json.loads(said)["skill_tree"]
        self.assertEqual(tree["skills_with_unlistable_directories"], ["tdd"])
        self.assertEqual(tree["skills_with_executable_payloads"], ["tdd"])
        self.assertEqual(tree["executable_payload_files"], 1)
        risk = [n for n in self.record["notes"] if n.startswith("ELEVATED REVIEW RISK")]
        self.assertEqual(len(risk), 2, self.record["notes"])
        self.assertIn("contains 1 executable payload file(s) (scripts/hook.sh)", risk[0])
        self.assertIn(
            "1 directory in this skill's directory could not be listed or fully examined "
            f"(scripts ({UNSEARCHED})), so whether executable payloads live there is unknown",
            risk[1],
        )
        rc, shown = self.said["review"]
        self.assertEqual(rc, 0, shown)
        for note in risk:
            self.assertIn(note[:80], shown)

    def test_the_directory_is_left_as_it_was(self) -> None:
        self.assertEqual(self.mode, 0o444)


class TheWordsAreTheSystemsTests(unittest.TestCase):
    """Another error of the ``lstat``, put on chosen entries by a stand-in for the call."""

    def inventory_with(self, refused: dict[str, OSError]):
        with tempfile.TemporaryDirectory() as tmp:
            config, scripts = with_scripts(Path(tmp))
            (scripts.parent / "notes.sh").write_text("text\n", encoding="utf-8")
            skill_dir = scripts.parent.resolve()
            real = Path.lstat

            def lstat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
                try:
                    rel = path.relative_to(skill_dir).as_posix()
                except ValueError:
                    rel = ""
                if rel in refused:
                    raise refused[rel]
                return real(path, *args, **kwargs)  # type: ignore[arg-type]

            with mock.patch.object(Path, "lstat", lstat):
                snap = tdd_snapshot(config)
            return snap.payload_files, snap.unlistable_dirs

    def test_another_error_names_the_directory_once_in_the_systems_words(self) -> None:
        eio = OSError(errno.EIO, os.strerror(errno.EIO), "x")
        bare = OSError("refused without a number")
        found, unknown = self.inventory_with(
            {"scripts/hook.sh": eio, "scripts/tool": eio, "notes.sh": bare}
        )
        self.assertEqual(found, ("notes.sh", "scripts/hook.sh"))
        self.assertEqual(
            unknown,
            (
                (".", "listed, but an entry in it could not be examined: refused without a number"),
                (
                    "scripts",
                    "listed, but an entry in it could not be examined: "
                    f"{os.strerror(errno.EIO)}",
                ),
            ),
        )

    def test_of_two_errors_in_one_directory_the_first_in_order_is_kept(self) -> None:
        eio = OSError(errno.EIO, os.strerror(errno.EIO), "x")
        eacces = OSError(errno.EACCES, os.strerror(errno.EACCES), "x")
        found, unknown = self.inventory_with({"scripts/hook.sh": eio, "scripts/tool": eacces})
        self.assertEqual(found, ("notes.sh", "scripts/hook.sh"))
        self.assertEqual(
            unknown,
            (
                (
                    "scripts",
                    "listed, but an entry in it could not be examined: "
                    f"{os.strerror(errno.EIO)}",
                ),
            ),
        )

    def test_an_entry_gone_since_the_listing_is_nothing_there(self) -> None:
        gone = FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "x")
        found, unknown = self.inventory_with({"scripts/hook.sh": gone})
        self.assertEqual(found, ("notes.sh", "scripts/tool"))
        self.assertEqual(unknown, ())


if __name__ == "__main__":
    unittest.main()
