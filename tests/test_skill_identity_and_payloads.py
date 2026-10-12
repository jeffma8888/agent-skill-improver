"""Which file is 'the skill', and what else lives beside it.

Two failures live here and both are silent. Editing a mirror copy succeeds, is
reported as success, and never reaches whatever loads the canonical file. And a
skill directory is not just a document: it can hold scripts, hooks and launchers,
so a write path that accepts any filename inside it is a code-execution primitive
wearing a documentation-change costume.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import signal
import stat
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.domain.skills import (
    CHANGE_ADDED,
    CHANGE_ADDED_UNREADABLE,
    CHANGE_BECAME_READABLE,
    CHANGE_BECAME_UNREADABLE,
    CHANGE_CONTENT,
    CHANGE_REMOVED,
    PAYLOAD_EXTENSIONS,
    SKILL_FILENAME,
    UNREADABLE_ENTRY,
    SkillRoot,
    TreeChange,
    describe_tree_change,
    group_for,
    is_payload_name,
    parse_frontmatter,
    resolve_groups,
    summarize_tree_changes,
    tree_digest_of,
)
from agent_skill_improver.errors import (
    BackupUnusable,
    BaseHashMismatch,
    LinkedTargetRefused,
    MirrorEditRefused,
    NonDocumentWriteRefused,
    ReadOnlyViolation,
    SelfMutationRefused,
    UndecodableDocument,
)
from agent_skill_improver.hashing import sha256_file, sha256_text
from tests.support import build_env, run_cli

GOOD = """---
name: example
description: A skill that loads.
---

# Example

- Do the thing.
"""

NO_OPENING_DELIMITER = """name: broken
description: This file never loads and nothing errors.

# Broken
"""


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class Harness(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.canonical_dir = self.base / "canonical"
        self.mirror_dir = self.base / "mirror"
        write(self.canonical_dir / "shared" / SKILL_FILENAME, GOOD)
        write(self.mirror_dir / "shared" / SKILL_FILENAME, GOOD + "\n- mirror drifted\n")
        write(self.mirror_dir / "mirror-only" / SKILL_FILENAME, GOOD)
        write(self.canonical_dir / "broken" / SKILL_FILENAME, NO_OPENING_DELIMITER)
        self.roots = [
            SkillRoot(
                id="canonical",
                path=self.canonical_dir,
                precedence=10,
                canonical=True,
                editable=True,
                kind="custom",
            ),
            SkillRoot(
                id="mirror",
                path=self.mirror_dir,
                precedence=90,
                canonical=False,
                editable=False,
                kind="mirror",
            ),
        ]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def store(self, *, read_only: bool = True, protected=()) -> FilesystemSkillStore:
        return FilesystemSkillStore(self.roots, read_only=read_only, protected_paths=protected)

    def canonical_snapshots(self) -> dict[str, object]:
        """Snapshots from the canonical root only.

        Indexing every snapshot by ``skill_id`` silently drops one of them when a
        skill exists in two roots, which is the same shadowing problem the
        canonical/mirror design exists to make explicit.
        """
        return {
            s.skill_id: s for s in self.store().snapshots(refresh=True) if s.root_id == "canonical"
        }


class FrontmatterTests(unittest.TestCase):
    def test_a_well_formed_skill_parses(self):
        fields, has_open, warnings = parse_frontmatter(GOOD)
        self.assertTrue(has_open)
        self.assertEqual(fields.get("name"), "example")
        self.assertEqual(warnings, [])

    def test_a_leading_byte_order_mark_is_read_past_and_named_not_called_unloadable(self):
        # Measured before: the mark made the first line differ from '---', the
        # document was reported as one that "will never load", and (through the
        # same parser) every replacement that kept the mark was quarantined.
        fields, has_open, warnings = parse_frontmatter("\ufeff" + GOOD)
        self.assertTrue(has_open)
        self.assertEqual(fields.get("name"), "example")
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("byte-order mark (U+FEFF)", warnings[0])
        self.assertIn("shown as \\ufeff", warnings[0])
        self.assertNotIn("never load", warnings[0])

    def test_a_byte_order_mark_document_is_inventoried_as_one_that_loads(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            write(
                root / "marked" / SKILL_FILENAME,
                "\ufeff" + GOOD.replace("name: example", "name: marked"),
            )
            store = FilesystemSkillStore(
                [
                    SkillRoot(
                        id="r",
                        path=root,
                        precedence=1,
                        canonical=True,
                        editable=True,
                        kind="custom",
                    )
                ]
            )
            snap = next(s for s in store.snapshots() if s.skill_id == "marked")
            self.assertTrue(snap.loads, snap.warnings)
            self.assertEqual(snap.frontmatter_name, "marked")
            self.assertTrue(any("U+FEFF" in w for w in snap.warnings), snap.warnings)

    def test_a_missing_opening_delimiter_is_reported_not_ignored(self):
        fields, has_open, warnings = parse_frontmatter(NO_OPENING_DELIMITER)
        self.assertFalse(has_open)
        self.assertTrue(
            warnings,
            "this is the real-world case: the skill silently never loads and "
            "nothing anywhere errors, so the only defence is reporting it",
        )


class PayloadDetectionTests(Harness):
    def test_a_shell_script_beside_a_skill_is_a_payload(self):
        write(self.canonical_dir / "shared" / "scripts" / "hook.sh", "#!/bin/sh\necho hi\n")
        snaps = self.canonical_snapshots()
        self.assertIn("scripts/hook.sh", snaps["shared"].payload_files)
        self.assertTrue(snaps["shared"].has_executable_payload)
        self.assertEqual(snaps["shared"].payload_count, 1)

    def test_a_plain_document_has_no_payloads(self):
        snaps = self.canonical_snapshots()
        self.assertEqual(snaps["broken"].payload_files, ())
        self.assertFalse(snaps["broken"].has_executable_payload)

    def test_the_executable_bit_counts_even_with_an_unremarkable_name(self):
        target = write(self.canonical_dir / "shared" / "runme", "#!/bin/sh\n")
        os.chmod(target, 0o755)
        snaps = self.canonical_snapshots()
        self.assertIn("runme", snaps["shared"].payload_files)

    def test_a_filename_containing_a_backslash_is_handled(self):
        # These exist in real skill trees and break every shell-based scan.
        write(self.canonical_dir / "shared" / "weird\\name.sh", "#!/bin/sh\n")
        snaps = self.canonical_snapshots()
        self.assertIn(
            "weird\\name.sh",
            snaps["shared"].payload_files,
            "detection is pure pathlib on purpose: a shell pipeline mangles this "
            "filename silently",
        )

    def test_a_filename_with_spaces_and_quotes_is_handled(self):
        write(self.canonical_dir / "shared" / "a b 'c'.py", "print(1)\n")
        snaps = self.canonical_snapshots()
        self.assertIn("a b 'c'.py", snaps["shared"].payload_files)

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_a_symlinked_launcher_is_inventoried_not_skipped(self):
        """A symlink is the cheapest way to put a binary beside a skill without
        the inventory noticing: the scan skipped every symlink before looking
        at its name or mode, so ``hook.sh -> /usr/bin/something`` reported as a
        plain document. The link is a payload on its own evidence (its name, or
        the executable bit of what it points to); where it points is secondary.
        """
        outside = write(self.base / "elsewhere" / "real-tool", "#!/bin/sh\n")
        os.chmod(outside, 0o755)
        skill_dir = self.canonical_dir / "shared"
        try:
            os.symlink(outside, skill_dir / "hook.sh")
            os.symlink(outside, skill_dir / "tool")
            os.symlink(self.base / "elsewhere" / "missing", skill_dir / "dangling.py")
        except (OSError, NotImplementedError):  # pragma: no cover - permissions
            self.skipTest("cannot create a symlink here")
        snaps = self.canonical_snapshots()
        payloads = snaps["shared"].payload_files
        self.assertIn("hook.sh", payloads, "a payload-named link is a payload")
        self.assertIn("tool", payloads, "a link to an executable is a payload")
        self.assertIn("dangling.py", payloads, "a dangling payload-named link still counts")
        self.assertTrue(snaps["shared"].has_executable_payload)

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_a_symlink_to_a_plain_text_file_is_not_a_payload(self):
        note = write(self.base / "elsewhere" / "notes.txt", "hello\n")
        try:
            os.symlink(note, self.canonical_dir / "shared" / "notes.txt")
        except (OSError, NotImplementedError):  # pragma: no cover - permissions
            self.skipTest("cannot create a symlink here")
        snaps = self.canonical_snapshots()
        self.assertEqual(snaps["shared"].payload_files, ())

    def test_the_payload_extension_set_covers_the_obvious_executables(self):
        for name in ("x.sh", "x.py", "x.js"):
            self.assertTrue(is_payload_name(name), name)
        self.assertFalse(is_payload_name(SKILL_FILENAME))
        self.assertFalse(is_payload_name("notes.md"))
        self.assertTrue(PAYLOAD_EXTENSIONS, "the set must not be empty")

    @unittest.skipIf(os.geteuid() == 0, "root lists a mode-000 directory")
    def test_a_payload_in_a_directory_the_scan_cannot_list_is_an_unknown_not_an_absence(self):
        # Measured before: rglob stepped over the directory without a word, the
        # inventory read () and the document carried no warning, for a hook
        # that still runs by name when the directory is mode 111.
        hidden = self.canonical_dir / "shared" / "scripts"
        write(hidden / "hook.sh", "#!/bin/sh\n")
        for mode in (0o000, 0o111):
            os.chmod(hidden, mode)
            try:
                snap = self.canonical_snapshots()["shared"]
            finally:
                os.chmod(hidden, 0o755)
            with self.subTest(mode=oct(mode)):
                self.assertEqual(snap.payload_files, ())
                self.assertEqual(snap.unlistable_dirs, (("scripts", "Permission denied"),))
                self.assertFalse(snap.payload_inventory_complete)
                warnings = [w for w in snap.warnings if "could not be listed" in w]
                self.assertEqual(len(warnings), 1, snap.warnings)
                self.assertIn(
                    "1 directory in this skill directory could not be listed "
                    "(scripts (Permission denied))",
                    warnings[0],
                )
                self.assertIn("inventory above is incomplete", warnings[0])
                shown = snap.to_dict()
                self.assertEqual(
                    shown["unlistable_dirs"], [{"path": "scripts", "error": "Permission denied"}]
                )
                self.assertFalse(shown["payload_inventory_complete"])
        snap = self.canonical_snapshots()["shared"]
        self.assertEqual(snap.payload_files, ("scripts/hook.sh",))
        self.assertTrue(snap.payload_inventory_complete, "listable again, nothing unknown")

    def test_a_listable_skill_directory_reports_a_complete_inventory(self):
        write(self.canonical_dir / "shared" / "scripts" / "hook.sh", "#!/bin/sh\n")
        snap = self.canonical_snapshots()["shared"]
        self.assertEqual(snap.unlistable_dirs, ())
        self.assertTrue(snap.payload_inventory_complete)
        self.assertEqual(snap.to_dict()["unlistable_dirs"], [])
        self.assertFalse(any("could not be listed" in w for w in snap.warnings))

    @unittest.skipIf(os.geteuid() == 0, "root lists a mode-000 directory")
    def test_the_group_rolls_unlistable_directories_up_by_root(self):
        hidden = self.mirror_dir / "shared" / "bin"
        hidden.mkdir()
        os.chmod(hidden, 0o000)
        try:
            group = group_for(self.store().groups(refresh=True), "shared")
        finally:
            os.chmod(hidden, 0o755)
        assert group is not None
        self.assertEqual(group.unlistable_dirs, ("mirror:bin",))
        self.assertEqual(group.to_dict()["unlistable_dirs"], ["mirror:bin"])
        self.assertEqual(group.payload_files, ())

    def test_the_group_rolls_payloads_up_across_copies(self):
        write(self.canonical_dir / "shared" / "scripts" / "hook.sh", "#!/bin/sh\n")
        group = group_for(self.store().groups(refresh=True), "shared")
        assert group is not None
        self.assertIn(
            "canonical:scripts/hook.sh",
            group.payload_files,
            "the rollup names the root, so a reviewer can tell which copy carries "
            "the payload rather than only that one does",
        )


class WritePathRefusalTests(Harness):
    def test_a_read_only_store_refuses_every_write(self):
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        with self.assertRaises(ReadOnlyViolation):
            self.store().write_canonical(
                target, "x", expected_sha256=sha256_file(target), backup_dir=self.base / "bak"
            )

    def test_only_the_skill_document_is_writable(self):
        payload = write(self.canonical_dir / "shared" / "scripts" / "hook.sh", "#!/bin/sh\n")
        with self.assertRaises(NonDocumentWriteRefused):
            self.store(read_only=False).write_canonical(
                payload,
                "#!/bin/sh\nrm -rf /\n",
                expected_sha256=sha256_file(payload),
                backup_dir=self.base / "bak",
            )
        self.assertEqual(
            payload.read_text(encoding="utf-8"),
            "#!/bin/sh\n",
            "the refusal must happen before any write",
        )

    def test_the_document_check_runs_before_the_hash_check(self):
        # Ordering matters: a caller supplying a bogus hash for a payload file
        # must still be refused for the right reason.
        payload = write(self.canonical_dir / "shared" / "scripts" / "hook.sh", "#!/bin/sh\n")
        with self.assertRaises(NonDocumentWriteRefused):
            self.store(read_only=False).write_canonical(
                payload, "x", expected_sha256="0" * 64, backup_dir=self.base / "bak"
            )

    def _plant_twin(self) -> tuple[Path, Path, str]:
        """A second canonical skill holding the same text as ``shared``.

        Same text is the point: a link from one to the other passes the base
        hash check, so only an lstat judgement can tell the write it is about to
        land on the wrong skill. Measured before the check existed, the write
        went through the link and the manifest named the twin.
        """
        twin = write(self.canonical_dir / "twin" / SKILL_FILENAME, GOOD)
        shared = self.canonical_dir / "shared" / SKILL_FILENAME
        return shared, twin, sha256_file(shared)

    def _link(self, link: Path, to: Path) -> None:
        try:
            os.symlink(to, link)
        except (OSError, NotImplementedError):  # pragma: no cover - permissions
            self.skipTest("cannot create a symlink here")

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_a_skill_document_that_became_a_link_is_refused_and_the_twin_is_untouched(self):
        shared, twin, base = self._plant_twin()
        shared.unlink()
        self._link(shared, twin)
        with self.assertRaises(LinkedTargetRefused) as ctx:
            self.store(read_only=False).write_canonical(
                shared, GOOD + "\n- changed\n", expected_sha256=base, backup_dir=self.base / "bak"
            )
        self.assertIn(str(shared), str(ctx.exception))
        self.assertEqual(twin.read_text(encoding="utf-8"), GOOD, "the twin must not change")
        self.assertTrue(shared.is_symlink(), "the link itself is left in place")
        self.assertFalse((self.base / "bak").exists(), "refused before any backup was taken")

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_a_skill_directory_that_became_a_link_is_refused_too(self):
        shared, twin, base = self._plant_twin()
        shutil.rmtree(shared.parent)
        self._link(shared.parent, twin.parent)
        self.assertFalse(shared.is_symlink(), "the document itself is a regular file here")
        with self.assertRaises(LinkedTargetRefused) as ctx:
            self.store(read_only=False).write_canonical(
                shared, GOOD + "\n- changed\n", expected_sha256=base, backup_dir=self.base / "bak"
            )
        self.assertIn(str(shared.parent), str(ctx.exception))
        self.assertEqual(twin.read_text(encoding="utf-8"), GOOD)

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_a_spelling_that_reaches_the_root_through_an_outside_link_is_refused(self):
        """No root form prefixes this spelling, yet it resolves inside the
        canonical root. The walk cannot see a link above or outside the root,
        so the no-match case is a refusal rather than a pass-through to the
        resolved-path checks, which would accept it."""
        shared, twin, base = self._plant_twin()
        via = self.base / "elsewhere" / "via"
        via.parent.mkdir(parents=True)
        self._link(via, self.canonical_dir)
        spelled = via / "shared" / SKILL_FILENAME
        with self.assertRaises(LinkedTargetRefused) as ctx:
            self.store(read_only=False).write_canonical(
                spelled, GOOD + "\n- changed\n", expected_sha256=base, backup_dir=self.base / "bak"
            )
        self.assertIn("canonical", str(ctx.exception))
        self.assertEqual(shared.read_text(encoding="utf-8"), GOOD)
        self.assertEqual(twin.read_text(encoding="utf-8"), GOOD)

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_the_outside_link_refusal_covers_an_editable_root_not_marked_canonical(self):
        """``resolve_groups`` treats the highest-precedence editable root as the
        canonical one when no root is marked canonical, so the write path must
        guard that root the same way. Measured before: with ``canonical =
        false`` on the editable root, the spelling through the outside link was
        not refused by the link check and the resolved-path check accepted it.
        """
        self.roots = [
            dataclasses.replace(r, canonical=False) if r.id == "canonical" else r
            for r in self.roots
        ]
        shared, twin, base = self._plant_twin()
        via = self.base / "elsewhere" / "via"
        via.parent.mkdir(parents=True)
        self._link(via, self.canonical_dir)
        spelled = via / "shared" / SKILL_FILENAME
        with self.assertRaises(LinkedTargetRefused) as ctx:
            self.store(read_only=False).write_canonical(
                spelled, GOOD + "\n- changed\n", expected_sha256=base, backup_dir=self.base / "bak"
            )
        self.assertIn("through a symbolic link", str(ctx.exception))
        self.assertEqual(shared.read_text(encoding="utf-8"), GOOD)

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_the_control_a_root_that_is_itself_a_link_still_takes_the_write(self):
        """A skills directory that is a link to a checkout is an ordinary setup;
        only links BELOW the root redirect a reviewed write."""
        real_root = self.base / "real-canonical"
        shutil.move(str(self.canonical_dir), str(real_root))
        self._link(self.canonical_dir, real_root)
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        receipt = self.store(read_only=False).write_canonical(
            target,
            GOOD + "\n- changed\n",
            expected_sha256=sha256_file(target),
            backup_dir=self.base / "bak",
        )
        self.assertEqual(Path(receipt.path), (real_root / "shared" / SKILL_FILENAME).resolve())
        self.assertIn("- changed", target.read_text(encoding="utf-8"))

    def test_a_mirror_copy_is_refused_with_the_canonical_path_named(self):
        target = self.mirror_dir / "shared" / SKILL_FILENAME
        with self.assertRaises(MirrorEditRefused) as ctx:
            self.store(read_only=False).write_canonical(
                target, "x", expected_sha256=sha256_file(target), backup_dir=self.base / "bak"
            )
        self.assertIn("canonical", str(ctx.exception).lower())

    def test_a_protected_path_is_refused(self):
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        store = FilesystemSkillStore(
            self.roots, read_only=False, protected_paths=[self.canonical_dir]
        )
        with self.assertRaises(SelfMutationRefused):
            store.write_canonical(
                target, "x", expected_sha256=sha256_file(target), backup_dir=self.base / "bak"
            )

    def test_a_mirror_reached_through_a_redundant_path_segment_is_still_refused(self):
        """Regression: the mirror interlock was bypassable by re-spelling the path.

        Snapshots store resolved paths; ``classify_target`` compared raw strings.
        A caller passing the same file spelled differently -- a ``..`` segment, a
        relative path, or a symlinked parent such as the platform temp directory --
        classified as ``unknown``, and ``unknown`` was treated as "not a mirror",
        so the write proceeded. The failure direction was OPEN, which is the wrong
        default for an interlock.
        """
        indirect = self.mirror_dir / "shared" / ".." / "shared" / SKILL_FILENAME
        self.assertTrue(indirect.exists(), "the redundant spelling must name a real file")
        with self.assertRaises(MirrorEditRefused):
            self.store(read_only=False).write_canonical(
                indirect,
                "x",
                expected_sha256=sha256_file(indirect),
                backup_dir=self.base / "bak",
            )

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_a_mirror_reached_through_a_symlinked_root_is_still_refused(self):
        link = self.base / "linked-mirror"
        try:
            os.symlink(self.mirror_dir, link)
        except (OSError, NotImplementedError):  # pragma: no cover - permissions
            self.skipTest("cannot create a symlink here")
        target = link / "shared" / SKILL_FILENAME
        with self.assertRaises(MirrorEditRefused):
            self.store(read_only=False).write_canonical(
                target, "x", expected_sha256=sha256_file(target), backup_dir=self.base / "bak"
            )

    def test_an_unrecognised_skill_document_is_refused_not_assumed_safe(self):
        stray = write(self.base / "not-a-root" / "whatever" / SKILL_FILENAME, GOOD)
        with self.assertRaises(MirrorEditRefused) as ctx:
            self.store(read_only=False).write_canonical(
                stray, "x", expected_sha256=sha256_file(stray), backup_dir=self.base / "bak"
            )
        self.assertIn("not a skill document in any configured root", str(ctx.exception))

    def test_the_canonical_copy_is_writable_when_everything_lines_up(self):
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        receipt = self.store(read_only=False).write_canonical(
            target,
            GOOD + "\n- an approved addition\n",
            expected_sha256=sha256_file(target),
            backup_dir=self.base / "bak",
            skill_id="shared",
        )
        self.assertIn("an approved addition", target.read_text(encoding="utf-8"))
        payload = receipt.to_dict()
        self.assertTrue(payload["backup_path"], "a backup is written before the change")
        self.assertTrue(Path(payload["backup_path"]).is_file())
        self.assertEqual(
            Path(payload["backup_path"]).read_text(encoding="utf-8"),
            GOOD,
            "the backup must hold the PREVIOUS text, or rollback restores the change",
        )
        self.assertTrue(payload["rollback_command"], "the receipt must say how to undo itself")
        import shlex

        self.assertEqual(
            shlex.split(payload["rollback_command"]),
            ["cp", payload["backup_path"], payload["path"]],
            "the copy-paste rollback command must preserve paths with spaces or shell syntax as argv",
        )
        self.assertNotEqual(payload["previous_sha256"], payload["new_sha256"])

    def test_a_traversal_style_target_is_still_refused_on_its_filename(self):
        outside = self.base / "elsewhere" / "SKILL.md.bak"
        write(outside, "x")
        with self.assertRaises(NonDocumentWriteRefused):
            self.store(read_only=False).write_canonical(
                outside, "y", expected_sha256=sha256_file(outside), backup_dir=self.base / "bak"
            )


class WritesArePinnedToTheDirectoryHandleTests(Harness):
    """The lstat walk judges the path at one instant; the write came later, by
    path. Between the two, a directory on the way could become a link and the
    kernel would follow it at write time, with a same-text twin passing every
    hash. So the document is addressed through a handle on its directory, taken
    with ``O_NOFOLLOW`` per component, for the hash read, the write and the
    re-read alike. The swap is simulated here at the one point in
    ``write_canonical`` that sits between the read and the write.
    """

    def _twin_and_target(self) -> tuple[Path, Path]:
        twin = write(self.canonical_dir / "twin" / SKILL_FILENAME, GOOD)
        return self.canonical_dir / "shared" / SKILL_FILENAME, twin

    def _link(self, link: Path, to: Path) -> None:
        try:
            os.symlink(to, link)
        except (OSError, NotImplementedError):  # pragma: no cover - permissions
            self.skipTest("cannot create a symlink here")

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_a_directory_swapped_for_a_link_mid_write_still_receives_the_write(self):
        from unittest import mock

        from agent_skill_improver.adapters import skill_fs as skill_fs_mod

        target, twin = self._twin_and_target()
        moved_to = self.base / "real-shared"
        base_sha = sha256_file(target)
        real_normalize = skill_fs_mod.normalize_text

        def swap_then_normalize(text: str) -> str:
            # Runs after the hash read, before the write: the directory the
            # path names is now a link to the twin.
            shutil.move(str(target.parent), str(moved_to))
            self._link(target.parent, twin.parent)
            return real_normalize(text)

        with mock.patch.object(skill_fs_mod, "normalize_text", swap_then_normalize):
            receipt = self.store(read_only=False).write_canonical(
                target,
                GOOD + "\n- changed\n",
                expected_sha256=base_sha,
                backup_dir=self.base / "bak",
                skill_id="shared",
            )
        self.assertTrue(target.parent.is_symlink(), "the swap really happened mid-write")
        self.assertEqual(twin.read_text(encoding="utf-8"), GOOD, "the twin must not change")
        self.assertIn(
            "- changed",
            (moved_to / SKILL_FILENAME).read_text(encoding="utf-8"),
            "the write landed in the directory the handle was opened on",
        )
        self.assertEqual(receipt.new_sha256, sha256_file(moved_to / SKILL_FILENAME))
        self.assertFalse((moved_to / SKILL_FILENAME).is_symlink())

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_the_handle_walk_refuses_a_linked_directory_on_its_own(self):
        """Independent of the lstat walk: the handle walk is the second lock."""
        target, twin = self._twin_and_target()
        shutil.rmtree(target.parent)
        self._link(target.parent, twin.parent)
        with self.assertRaises(LinkedTargetRefused) as ctx:
            self.store(read_only=False)._open_parent_pinned(target, verb="write")
        self.assertIn("open handle", str(ctx.exception))

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_the_pinned_read_refuses_a_linked_document_on_its_own(self):
        target, twin = self._twin_and_target()
        store = self.store(read_only=False)
        parent_fd, leaf = store._open_parent_pinned(target, verb="write")
        try:
            target.unlink()
            self._link(target, twin)
            with self.assertRaises(LinkedTargetRefused):
                store._read_pinned(parent_fd, leaf, target, verb="write")
        finally:
            os.close(parent_fd)
        self.assertEqual(twin.read_text(encoding="utf-8"), GOOD)

    def test_a_host_without_directory_handles_is_refused_rather_than_degraded(self):
        """The gate is dead code on every supported host, so it is measured by
        flipping it: the refusal must happen before the hash read."""
        from unittest import mock

        from agent_skill_improver.adapters import skill_fs as skill_fs_mod

        target, _twin = self._twin_and_target()
        with mock.patch.object(skill_fs_mod, "PINNED_IO_SUPPORTED", False):
            with self.assertRaises(LinkedTargetRefused) as ctx:
                self.store(read_only=False).write_canonical(
                    target,
                    GOOD + "\n- changed\n",
                    expected_sha256=sha256_file(target),
                    backup_dir=self.base / "bak",
                )
        self.assertIn("directory handle", str(ctx.exception))
        self.assertEqual(target.read_text(encoding="utf-8"), GOOD)
        self.assertFalse((self.base / "bak").exists(), "refused before any backup was taken")

    def test_the_control_a_pinned_write_keeps_the_documents_mode_and_is_atomic(self):
        from agent_skill_improver.adapters.skill_fs import _write_pinned

        target, _twin = self._twin_and_target()
        os.chmod(target, 0o640)
        store = self.store(read_only=False)
        parent_fd, leaf = store._open_parent_pinned(target, verb="write")
        try:
            _write_pinned(parent_fd, leaf, b"fresh\n", mode=0o640)
            data, mode = store._read_pinned(parent_fd, leaf, target, verb="write")
        finally:
            os.close(parent_fd)
        self.assertEqual(data, b"fresh\n")
        self.assertEqual(oct(mode), oct(0o640))
        leftovers = [p.name for p in target.parent.iterdir() if p.name.startswith(".tmp-")]
        self.assertEqual(leftovers, [], "no temp file survives the rename")


@contextmanager
def deadline(seconds: float) -> Iterator[None]:
    """Fail, rather than hang the suite, if the body blocks in a syscall.

    A fifo with no writer blocks ``open`` for good, and no thread can interrupt
    that; an interval timer on the main thread can. Measured: without this
    guard the two fifo tests below hung the whole suite against the code they
    were written to catch.
    """
    if not hasattr(signal, "setitimer"):  # pragma: no cover - POSIX only
        yield
        return

    def expired(signum: int, frame: object) -> None:
        raise AssertionError(f"blocked for more than {seconds}s; a fifo is being waited on")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class BaseReadsAreJudgedLikeWritesTests(Harness):
    """``read_text`` is the read that decides whether ``review`` prints a token
    and whether ``apply`` accepts the base, so it must refuse what the write
    path refuses. Measured before it opened the file itself: a document replaced
    by a link to an identical copy made ``review`` print a token for an apply
    that then refused the link, and a fifo at the document parked ``review``
    indefinitely. Discovery is covered too: a fifo named like a document is
    not a document and is not inventoried, so ``daily`` and ``doctor`` do not
    wait on it either.
    """

    def test_a_linked_document_is_refused_on_read_as_it_would_be_on_write(self):
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        aside = self.base / "identical-copy.md"
        aside.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        target.unlink()
        target.symlink_to(aside)
        with self.assertRaises(LinkedTargetRefused) as ctx:
            self.store().read_text(target)
        self.assertIn("symbolic link", str(ctx.exception))
        self.assertEqual(aside.read_text(encoding="utf-8"), GOOD, "the target was not touched")

    def test_a_fifo_at_the_document_is_refused_on_read_rather_than_waited_on(self):
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        target.unlink()
        os.mkfifo(target)
        with deadline(5.0), self.assertRaises(BaseHashMismatch) as ctx:
            self.store().read_text(target)
        self.assertIn("not a regular file", str(ctx.exception))

    def test_a_fifo_named_like_a_document_is_not_discovered(self):
        fifo_dir = self.canonical_dir / "fifo-skill"
        fifo_dir.mkdir()
        os.mkfifo(fifo_dir / SKILL_FILENAME)
        with deadline(5.0):
            ids = self.store().known_skill_ids()
        self.assertNotIn("fifo-skill", ids)
        self.assertIn("shared", ids, "regular documents are still discovered")

    def test_a_crlf_document_still_reads_with_folded_line_endings(self):
        """The text-mode read this replaces folded ``\r\n``; so does this one,
        or a Windows-authored skill would review as every line changed."""
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        target.write_bytes(GOOD.replace("\n", "\r\n").encode("utf-8"))
        self.assertEqual(self.store().read_text(target), GOOD)

    def test_the_controls_a_regular_document_reads_whole_and_a_missing_one_is_an_os_error(self):
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        self.assertEqual(self.store().read_text(target), GOOD)
        with self.assertRaises(FileNotFoundError):
            self.store().read_text(self.canonical_dir / "absent" / SKILL_FILENAME)


class UndecodableDocumentsAreOneDefectNotAFatalityTests(Harness):
    """One stray byte in one skill must cost that skill its turn, not the run.

    Measured before this class existed: discovery decoded with replacement
    characters and the base read decoded strictly, so a skill with a single
    invalid byte was inventoried, selected, and then ended ``daily`` with an
    uncaught ``UnicodeDecodeError`` (exit 70, no receipt, no candidate for any
    other skill), while ``doctor`` and ``skills`` said nothing about it.
    """

    def _corrupt(self) -> Path:
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        data = target.read_bytes()
        target.write_bytes(data[:10] + b"\xff" + data[10:])
        return target

    def test_the_base_read_names_the_byte_and_is_a_record_defect(self):
        from agent_skill_improver.errors import RecordDefect

        target = self._corrupt()
        with self.assertRaises(UndecodableDocument) as ctx:
            self.store().read_text(target)
        self.assertIsInstance(ctx.exception, RecordDefect, "quarantine the skill, not the run")
        self.assertIn("byte 10", str(ctx.exception))
        self.assertEqual(ctx.exception.record_id, "shared")

    def test_discovery_still_inventories_it_with_a_warning_that_names_the_byte(self):
        target = self._corrupt()
        snapshot = self.canonical_snapshots()["shared"]
        self.assertTrue(
            any(w.startswith("not valid UTF-8 at byte 10") for w in snapshot.warnings),
            snapshot.warnings,
        )
        self.assertEqual(
            snapshot.sha256,
            sha256_text(target.read_bytes().decode("utf-8", errors="replace")),
            "the inventory hash is over the replaced text, as it was before",
        )

    def test_the_control_a_clean_document_carries_no_such_warning(self):
        snapshot = self.canonical_snapshots()["shared"]
        self.assertFalse(any("UTF-8" in w for w in snapshot.warnings), snapshot.warnings)
        self.assertEqual(self.store().read_text(snapshot.path), GOOD)


class ARootThatIsNotADirectoryIsMisplacedNotMissingTests(Harness):
    """Measured before: a canonical root whose path held a regular file read as
    "skill root(s) not present on this machine" in ``doctor`` and as
    ``roots_missing`` in the receipt, so the operator was sent to check the
    spelling of a path that was right. What sits there is now named, at the
    root itself, with no skill id.
    """

    def test_the_control_an_absent_root_is_missing_and_passes_nothing_over(self):
        shutil.rmtree(self.canonical_dir)
        store = self.store()
        self.assertEqual([r.id for r in store.missing_roots()], ["canonical"])
        self.assertEqual(store.misplaced_roots(), ())
        self.assertEqual(store.passed_over(), [])

    def test_a_regular_file_at_the_root_is_named_at_the_root_itself(self):
        shutil.rmtree(self.canonical_dir)
        self.canonical_dir.write_text("not a directory\n", encoding="utf-8")
        store = self.store()
        self.assertEqual(store.missing_roots(), ())
        self.assertEqual([r.id for r in store.existing_roots()], ["mirror"])
        self.assertEqual(
            [(r.id, p) for r, p in store.misplaced_roots()],
            [("canonical", "a regular file, not a directory")],
        )
        self.assertEqual(
            [(d.root_id, d.path, d.reason, d.skill_id) for d in store.passed_over()],
            [
                (
                    "canonical",
                    self.canonical_dir,
                    "the configured root itself and is a regular file, not a directory",
                    None,
                )
            ],
        )
        self.assertEqual(sorted(s.skill_id for s in store.snapshots()), ["mirror-only", "shared"])

    def test_a_dangling_link_at_the_root_names_its_target(self):
        shutil.rmtree(self.canonical_dir)
        self.canonical_dir.symlink_to(self.base / "moved-away")
        store = self.store()
        self.assertEqual(store.missing_roots(), ())
        self.assertEqual(
            [(r.id, p) for r, p in store.misplaced_roots()],
            [
                (
                    "canonical",
                    f"a dangling symbolic link to {self.base / 'moved-away'}, not a directory",
                )
            ],
        )

    def test_a_link_to_a_file_at_the_root_names_its_target(self):
        shutil.rmtree(self.canonical_dir)
        target = write(self.base / "a-file.md", GOOD)
        self.canonical_dir.symlink_to(target)
        store = self.store()
        self.assertEqual(
            [(r.id, p) for r, p in store.misplaced_roots()],
            [("canonical", f"a symbolic link to {target}, which is not a directory")],
        )

    def test_a_fifo_at_the_root_is_named_as_a_fifo(self):
        shutil.rmtree(self.canonical_dir)
        os.mkfifo(self.canonical_dir)
        store = self.store()
        self.assertEqual(
            [(r.id, p) for r, p in store.misplaced_roots()],
            [("canonical", "a fifo, not a directory")],
        )

    @unittest.skipIf(os.geteuid() == 0, "root searches a mode-000 directory")
    def test_a_root_under_a_parent_this_process_cannot_search_is_named_not_a_crash(self):
        # Measured before: ``Path.is_dir`` raised PermissionError out of the
        # presence check, so ``doctor`` and ``daily`` ended as exit 70 with a
        # traceback before anything was discovered.
        from agent_skill_improver.adapters.skill_fs import root_presence

        locked = self.base / "locked"
        locked.mkdir()
        shutil.move(self.canonical_dir, locked / "canonical")
        hidden = locked / "canonical"
        self.roots = [
            dataclasses.replace(r, path=hidden) if r.id == "canonical" else r for r in self.roots
        ]
        os.chmod(locked, 0o000)
        try:
            store = self.store()
            self.assertEqual(store.missing_roots(), ())
            self.assertEqual([r.id for r in store.existing_roots()], ["mirror"])
            self.assertEqual(
                [(r.id, p) for r, p in store.misplaced_roots()],
                [("canonical", "a path this process cannot examine (Permission denied)")],
            )
            self.assertEqual(
                [(d.path, d.skill_id) for d in store.passed_over()],
                [(hidden, None)],
            )
            self.assertEqual(root_presence(hidden), (True, False))
        finally:
            os.chmod(locked, 0o755)

    def test_a_link_to_a_directory_at_the_root_is_a_present_root(self):
        real = self.base / "real-canonical"
        shutil.move(self.canonical_dir, real)
        self.canonical_dir.symlink_to(real)
        store = self.store()
        self.assertEqual(store.misplaced_roots(), ())
        self.assertEqual(store.missing_roots(), ())
        self.assertEqual([r.id for r in store.existing_roots()], ["canonical", "mirror"])
        # The judgement itself, not only the callers that filter on is_dir first:
        # a link that reaches a directory has no shape problem.
        from agent_skill_improver.adapters.skill_fs import _root_shape_problem

        self.assertIsNone(_root_shape_problem(self.canonical_dir))
        self.assertIsNone(_root_shape_problem(real))
        self.assertIsNone(_root_shape_problem(self.base / "never-made"))


class PassedOverDocumentsAreReportedTests(Harness):
    """What discovery steps over is listed with its reason, never silent.

    Measured before: a tree holding a dangling ``SKILL.md`` link, a fifo named
    ``SKILL.md``, a link to a real document (links not followed) and a document
    below ``max_depth`` discovered exactly the same skills as the tree without
    them, ``doctor`` said ``ready`` and ``skills`` listed nothing about any of
    the four. A skipped skill read as an absent one.
    """

    def passed_over(self, store: FilesystemSkillStore) -> dict[str, str]:
        root = self.canonical_dir.resolve()
        return {d.path.relative_to(root).as_posix(): d.reason for d in store.passed_over()}

    def test_the_control_a_clean_tree_passes_nothing_over(self):
        store = self.store()
        self.assertEqual(store.passed_over(), [])
        self.assertEqual(
            sorted(s.skill_id for s in store.snapshots()),
            ["broken", "mirror-only", "shared", "shared"],
        )

    def test_a_dangling_link_is_named_as_one(self):
        (self.canonical_dir / "gone").mkdir()
        (self.canonical_dir / "gone" / SKILL_FILENAME).symlink_to(self.canonical_dir / "nowhere.md")
        store = self.store()
        self.assertEqual(
            self.passed_over(store),
            {"gone/SKILL.md": "a symlink whose target does not exist or cannot be reached"},
        )
        self.assertEqual([d.root_id for d in store.passed_over()], ["canonical"])
        self.assertNotIn("gone", {s.skill_id for s in store.snapshots()})

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file")
    def test_a_document_this_process_cannot_read_is_passed_over_not_fatal(self):
        # Measured before: PermissionError out of discovery, so `doctor` and
        # `daily` both ended with exit 70 and a traceback for one such file.
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        os.chmod(target, 0o000)
        try:
            store = self.store()
            self.assertEqual(
                self.passed_over(store),
                {"shared/SKILL.md": "a document this process cannot read (Permission denied)"},
            )
            self.assertEqual(
                sorted(s.skill_id for s in store.snapshots()),
                ["broken", "mirror-only", "shared"],
                "the mirror copy of shared is still inventoried",
            )
            self.assertEqual(os.stat(target).st_mode & 0o777, 0, "the mode is not ours to change")
        finally:
            os.chmod(target, 0o644)

    def test_a_dangling_link_is_enumerated_whatever_the_glob_implementation_does(self):
        """On Python 3.11 ``Path.rglob("SKILL.md")`` drops an entry whose
        ``exists()`` is false, so the dangling link was never seen at all; the
        store walks the tree itself so the entry reaches the judgement."""
        gone = self.canonical_dir / "gone" / SKILL_FILENAME
        gone.parent.mkdir()
        gone.symlink_to(self.canonical_dir / "nowhere.md")
        entries = FilesystemSkillStore._entries_named_like_a_document(self.canonical_dir.resolve())
        self.assertIn(self.canonical_dir.resolve() / "gone" / SKILL_FILENAME, entries)

    def test_a_directory_carrying_the_documents_name_is_named_as_one(self):
        (self.canonical_dir / "odd" / SKILL_FILENAME).mkdir(parents=True)
        self.assertEqual(
            self.passed_over(self.store()), {"odd/SKILL.md": "not a regular file (a directory)"}
        )

    def test_a_directory_that_cannot_be_listed_is_named_with_its_error(self):
        if os.geteuid() == 0:
            self.skipTest("root lists every directory")
        sealed = self.canonical_dir / "sealed"
        write(sealed / SKILL_FILENAME, GOOD)
        os.chmod(sealed, 0)
        try:
            reasons = self.passed_over(self.store())
        finally:
            os.chmod(sealed, 0o755)
        self.assertEqual(list(reasons), ["sealed"])
        self.assertTrue(
            reasons["sealed"].startswith("a directory that could not be listed ("), reasons
        )
        self.assertIn("ermission", reasons["sealed"])

    def test_a_fifo_is_named_without_being_opened(self):
        (self.canonical_dir / "pipe").mkdir()
        os.mkfifo(self.canonical_dir / "pipe" / SKILL_FILENAME)
        with deadline(20):
            store = self.store()
            reasons = self.passed_over(store)
        self.assertEqual(reasons, {"pipe/SKILL.md": "not a regular file (a fifo)"})

    def test_a_link_to_a_real_document_is_passed_over_only_while_links_are_not_followed(self):
        (self.canonical_dir / "alias").mkdir()
        (self.canonical_dir / "alias" / SKILL_FILENAME).symlink_to(
            self.canonical_dir / "shared" / SKILL_FILENAME
        )
        self.assertEqual(
            self.passed_over(self.store()),
            {"alias/SKILL.md": "a symlink; this store does not follow links"},
        )
        following = FilesystemSkillStore(self.roots, follow_symlinks=True)
        self.assertEqual(following.passed_over(), [])
        taken = [s.path for s in following.snapshots()]
        self.assertIn(self.canonical_dir.resolve() / "alias" / SKILL_FILENAME, taken)

    def test_a_document_at_the_root_itself_is_passed_over_not_named_after_the_folder(self):
        # Measured before: inventoried as a skill called 'skills-canonical', the
        # root directory's own basename, so the id moved with the folder's name
        # and a proposal against it would have written the root's SKILL.md.
        write(self.canonical_dir / SKILL_FILENAME, GOOD)
        store = self.store()
        self.assertEqual(
            self.passed_over(store),
            {
                "SKILL.md": "at the root itself, not in a skill directory; a skill is "
                "<root>/<name>/SKILL.md"
            },
        )
        self.assertEqual([d.skill_id for d in store.passed_over()], [None])
        self.assertEqual(
            sorted(s.skill_id for s in store.snapshots()),
            ["broken", "mirror-only", "shared", "shared"],
            "no skill is named after the root directory",
        )
        self.assertNotIn(self.canonical_dir.name, store.known_skill_ids())

    def test_a_document_below_another_skill_is_that_skills_material_not_a_skill(self):
        # Measured before: tdd/sub/SKILL.md was inventoried as a skill named
        # 'sub', so a template or example shipped inside a skill became a
        # proposal target, and the proposal would have written inside 'shared'.
        nested = self.canonical_dir / "shared" / "templates" / SKILL_FILENAME
        write(nested, GOOD.replace("name: shared", "name: templates"))
        store = self.store()
        owner = self.canonical_dir.resolve() / "shared"
        self.assertEqual(
            self.passed_over(store),
            {
                "shared/templates/SKILL.md": f"below {owner / SKILL_FILENAME}, the place of "
                f"skill 'shared'; a document under another skill's directory is that skill's "
                f"material, not a skill"
            },
        )
        self.assertEqual([d.skill_id for d in store.passed_over()], [None])
        self.assertEqual(
            sorted(s.skill_id for s in store.snapshots()),
            ["broken", "mirror-only", "shared", "shared"],
        )
        canonical = next(
            s for s in store.snapshots() if s.skill_id == "shared" and s.root_id == "canonical"
        )
        self.assertEqual(canonical.path, owner / SKILL_FILENAME, "the owner is untouched")

    def test_the_nesting_rule_holds_when_the_owning_document_itself_does_not_read(self):
        # The owner's SKILL.md may be a fifo, a dangling link or unreadable; the
        # directory still belongs to that skill, so a document below it is
        # still not a skill of its own.
        (self.canonical_dir / "pipe").mkdir()
        os.mkfifo(self.canonical_dir / "pipe" / SKILL_FILENAME)
        write(self.canonical_dir / "pipe" / "example" / SKILL_FILENAME, GOOD)
        with deadline(20):
            reasons = self.passed_over(self.store())
        self.assertEqual(reasons["pipe/SKILL.md"], "not a regular file (a fifo)")
        self.assertTrue(
            reasons["pipe/example/SKILL.md"].endswith(
                ", the place of skill 'pipe'; a document "
                "under another skill's directory is that skill's material, not a skill"
            ),
            reasons,
        )
        self.assertNotIn("example", self.store().known_skill_ids())
        # A dangling link named like a document marks its directory the same
        # way: the owner it names was never inventoried, and both entries say so.
        (self.canonical_dir / "group").mkdir()
        (self.canonical_dir / "group" / SKILL_FILENAME).symlink_to(
            self.canonical_dir / "nowhere.md"
        )
        write(self.canonical_dir / "group" / "real" / SKILL_FILENAME, GOOD)
        with deadline(20):
            store = self.store()
            reasons = self.passed_over(store)
        self.assertEqual(
            reasons["group/SKILL.md"], "a symlink whose target does not exist or cannot be reached"
        )
        self.assertIn("the place of skill 'group'", reasons["group/real/SKILL.md"])
        self.assertNotIn("real", store.known_skill_ids())
        self.assertNotIn("group", store.known_skill_ids())

    def test_the_control_a_grouping_directory_without_a_document_is_not_an_owner(self):
        # <root>/group/<name>/SKILL.md is a skill under a grouping folder, as
        # before: only a directory that carries its own SKILL.md owns what is
        # below it.
        write(
            self.canonical_dir / "group" / "grouped" / SKILL_FILENAME,
            GOOD.replace("name: shared", "name: grouped"),
        )
        store = self.store()
        self.assertEqual(store.passed_over(), [])
        self.assertIn("grouped", store.known_skill_ids())

    def test_a_link_that_leaves_its_root_is_named_as_one(self):
        outside = self.base / "elsewhere" / SKILL_FILENAME
        write(outside, GOOD)
        (self.canonical_dir / "escape").mkdir()
        (self.canonical_dir / "escape" / SKILL_FILENAME).symlink_to(outside)
        self.assertEqual(
            self.passed_over(self.store()),
            {"escape/SKILL.md": "a symlink that resolves outside its root"},
        )

    def test_a_document_below_max_depth_names_the_limit(self):
        write(self.canonical_dir / "a" / "b" / "c" / "d" / SKILL_FILENAME, GOOD)
        store = FilesystemSkillStore(self.roots, max_depth=4)
        self.assertEqual(
            self.passed_over(store), {"a/b/c/d/SKILL.md": "deeper than max_depth=4 below the root"}
        )
        self.assertEqual(FilesystemSkillStore(self.roots, max_depth=5).passed_over(), [])

    def test_a_refresh_forgets_a_document_that_was_repaired(self):
        link = self.canonical_dir / "gone" / SKILL_FILENAME
        link.parent.mkdir()
        link.symlink_to(self.canonical_dir / "nowhere.md")
        store = self.store()
        self.assertEqual(len(store.passed_over()), 1)
        link.unlink()
        write(link, GOOD)
        self.assertEqual(len(store.passed_over()), 1, "cached until asked to refresh")
        self.assertEqual(store.passed_over(refresh=True), [])
        self.assertIn("gone", {s.skill_id for s in store.snapshots()})


class BackupWriteSafetyTests(Harness):
    """The backup is the only copy of the original; it must land where it says."""

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_a_link_planted_at_the_backup_path_is_replaced_not_written_through(self):
        """``shutil.copy2`` follows a symlink at its destination. A link planted
        at ``<backup_dir>/SKILL.md.bak`` turned the pre-apply backup into a write
        to wherever the link pointed, and left the backup itself as a link to a
        file the rollback checks would later read through. The backup is
        written as a fresh regular file, atomically, and the link target is
        never touched."""
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        backup_dir = self.base / "bak"
        backup_dir.mkdir()
        decoy = write(self.base / "elsewhere" / "decoy.txt", "untouched\n")
        try:
            os.symlink(decoy, backup_dir / f"{SKILL_FILENAME}.bak")
        except (OSError, NotImplementedError):  # pragma: no cover - permissions
            self.skipTest("cannot create a symlink here")
        receipt = self.store(read_only=False).write_canonical(
            target,
            GOOD + "\n- an approved addition\n",
            expected_sha256=sha256_file(target),
            backup_dir=backup_dir,
            skill_id="shared",
        )
        backup = Path(receipt.backup_path)
        self.assertFalse(
            backup.is_symlink(), "the backup must be a regular file, not the planted link"
        )
        self.assertEqual(backup.read_text(encoding="utf-8"), GOOD)
        self.assertEqual(decoy.read_text(encoding="utf-8"), "untouched\n")
        self.assertEqual(stat.S_IMODE(backup.stat().st_mode), 0o600)


class RollbackSafetyTests(Harness):
    """Rollback is the recovery command, so it must not be able to lose the file.

    The failure it has to survive is a backup that is no longer the text the
    apply replaced: truncated by a crash mid-copy, edited by hand, or described
    by a manifest that itself got cut short. Copying first and hashing afterwards
    detects that only after the live skill has already been overwritten, which
    leaves the operator with a corrupt file and nothing left to restore from.
    """

    def apply_a_change(self) -> tuple[Path, Path, str]:
        """Perform a real write and return the target, its manifest and the old text."""
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        previous = target.read_text(encoding="utf-8")
        receipt = self.store(read_only=False).write_canonical(
            target,
            GOOD + "\n- an approved addition\n",
            expected_sha256=sha256_file(target),
            backup_dir=self.base / "bak",
            skill_id="shared",
        )
        return target, Path(receipt.backup_manifest_path), previous

    def test_a_manifest_without_a_recorded_mode_restores_at_the_live_files_mode(self):
        """Manifests written before ``target_mode`` existed carry none; the
        restore then keeps the mode the file has now rather than guessing."""
        target, manifest, previous = self.apply_a_change()
        os.chmod(target, 0o640)
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        del raw["target_mode"]
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        result = self.store(read_only=False).rollback(manifest)
        self.assertFalse(result["already_restored"])
        self.assertEqual(target.read_text(encoding="utf-8"), previous)
        self.assertEqual(oct(stat.S_IMODE(target.stat().st_mode)), oct(0o640))

    def test_the_control_rollback_restores_the_previous_text(self):
        target, manifest, previous = self.apply_a_change()
        result = self.store(read_only=False).rollback(manifest)
        self.assertEqual(target.read_text(encoding="utf-8"), previous)
        self.assertEqual(result["restored_sha256"], sha256_file(target))

    def test_a_backup_that_no_longer_matches_the_manifest_is_refused(self):
        target, manifest, _previous = self.apply_a_change()
        backup = Path(json.loads(manifest.read_text(encoding="utf-8"))["backup_path"])
        backup.write_text("truncated\n", encoding="utf-8")
        with self.assertRaises(BackupUnusable) as ctx:
            self.store(read_only=False).rollback(manifest)
        self.assertIn("is unchanged", str(ctx.exception))

    def test_the_live_file_is_untouched_after_a_refused_rollback(self):
        target, manifest, _previous = self.apply_a_change()
        applied = target.read_text(encoding="utf-8")
        backup = Path(json.loads(manifest.read_text(encoding="utf-8"))["backup_path"])
        backup.write_text("truncated\n", encoding="utf-8")
        with self.assertRaises(BackupUnusable):
            self.store(read_only=False).rollback(manifest)
        self.assertEqual(
            target.read_text(encoding="utf-8"),
            applied,
            "the check must run before the copy: a refusal that has already "
            "overwritten the file has destroyed the only copy of the original",
        )

    def test_a_missing_backup_is_refused_and_says_the_file_is_unchanged(self):
        target, manifest, _previous = self.apply_a_change()
        applied = target.read_text(encoding="utf-8")
        Path(json.loads(manifest.read_text(encoding="utf-8"))["backup_path"]).unlink()
        with self.assertRaises(BackupUnusable):
            self.store(read_only=False).rollback(manifest)
        self.assertEqual(target.read_text(encoding="utf-8"), applied)

    def test_a_manifest_missing_a_required_field_is_refused_by_name(self):
        _target, manifest, _previous = self.apply_a_change()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        del payload["previous_sha256"]
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(BackupUnusable) as ctx:
            self.store(read_only=False).rollback(manifest)
        self.assertIn("previous_sha256", str(ctx.exception))

    def test_an_unreadable_manifest_is_a_refusal_not_a_crash(self):
        _target, manifest, _previous = self.apply_a_change()
        manifest.write_text('{"target_path":', encoding="utf-8")
        with self.assertRaises(BackupUnusable):
            self.store(read_only=False).rollback(manifest)

    def test_a_pathologically_nested_manifest_is_a_refusal_not_a_crash(self):
        """Deep nesting raises ``RecursionError`` rather than ``JSONDecodeError``;
        the recovery command must still refuse by name and touch nothing.
        """
        target, manifest, _previous = self.apply_a_change()
        applied = target.read_text(encoding="utf-8")
        manifest.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
        with self.assertRaises(BackupUnusable) as ctx:
            self.store(read_only=False).rollback(manifest)
        self.assertIn("could not be read", str(ctx.exception))
        self.assertEqual(target.read_text(encoding="utf-8"), applied)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "fifos are POSIX")
    def test_a_manifest_that_is_a_fifo_is_refused_rather_than_waited_on(self):
        """``open`` on a fifo with no writer never returns. The manifest is judged
        before it is opened, so a planted one is a refusal, not a hung rollback."""
        import signal

        target, manifest, _previous = self.apply_a_change()
        applied = target.read_text(encoding="utf-8")
        manifest.unlink()
        os.mkfifo(manifest)

        def hung(signum, frame):  # pragma: no cover - only fires on regression
            raise AssertionError("opening the fifo manifest blocked instead of refusing")

        previous = signal.signal(signal.SIGALRM, hung)
        signal.alarm(5)
        try:
            with self.assertRaises(BackupUnusable) as ctx:
                self.store(read_only=False).rollback(manifest)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        self.assertIn("not a regular file", str(ctx.exception))
        self.assertEqual(target.read_text(encoding="utf-8"), applied, "no file was touched")

    def test_a_manifest_that_is_a_symlink_is_refused_before_it_is_read(self):
        target, manifest, _previous = self.apply_a_change()
        applied = target.read_text(encoding="utf-8")
        real = manifest.with_name("real-manifest.json")
        manifest.rename(real)
        manifest.symlink_to(real)
        with self.assertRaises(BackupUnusable) as ctx:
            self.store(read_only=False).rollback(manifest)
        self.assertIn("not a regular file", str(ctx.exception))
        self.assertEqual(target.read_text(encoding="utf-8"), applied, "no file was touched")

    def test_a_manifest_pointing_at_a_payload_file_is_refused_on_its_filename(self):
        _target, manifest, _previous = self.apply_a_change()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        script = write(self.canonical_dir / "shared" / "scripts" / "hook.sh", "#!/bin/sh\n")
        payload["target_path"] = str(script)
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(NonDocumentWriteRefused):
            self.store(read_only=False).rollback(manifest)
        self.assertEqual(script.read_text(encoding="utf-8"), "#!/bin/sh\n")

    def test_a_rollback_refuses_to_overwrite_edits_made_after_the_apply(self):
        """Rollback undoes ONE apply; it is not permission to overwrite the file.

        ``apply`` refuses when the live file is not the base the candidate was
        built against. Rollback had no such check: it copied the backup over
        whatever was on disk, so a human edit made after the apply (or a later
        apply of a different candidate) was silently destroyed by a command the
        docs describe as safe to run without diagnosing first.
        """
        target, manifest, _previous = self.apply_a_change()
        edited = target.read_text(encoding="utf-8") + "\n- a later human edit\n"
        target.write_text(edited, encoding="utf-8")
        with self.assertRaises(BaseHashMismatch) as ctx:
            self.store(read_only=False).rollback(manifest)
        self.assertIn("is unchanged", str(ctx.exception))
        self.assertEqual(
            target.read_text(encoding="utf-8"),
            edited,
            "a refused rollback must leave the later edit exactly where it was",
        )

    def test_rolling_back_twice_is_a_verified_no_op(self):
        target, manifest, previous = self.apply_a_change()
        store = self.store(read_only=False)
        store.rollback(manifest)
        result = store.rollback(manifest)
        self.assertEqual(target.read_text(encoding="utf-8"), previous)
        self.assertEqual(result["restored_sha256"], sha256_file(target))
        self.assertTrue(result.get("already_restored"), "the receipt must say nothing was written")

    def test_a_manifest_missing_the_applied_hash_is_refused_by_name(self):
        _target, manifest, _previous = self.apply_a_change()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        del payload["new_sha256"]
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(BackupUnusable) as ctx:
            self.store(read_only=False).rollback(manifest)
        self.assertIn("new_sha256", str(ctx.exception))

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_a_target_that_became_a_symlink_is_refused_and_the_link_target_untouched(self):
        """The manifest recorded a resolved regular file; a link in its place is not it.

        ``shutil.copy2`` follows a symlink at the destination, so a link planted
        at the recorded path between apply and rollback would turn the recovery
        command into a write to wherever the link points.
        """
        target, manifest, _previous = self.apply_a_change()
        applied = target.read_text(encoding="utf-8")
        decoy = write(self.base / "elsewhere" / "decoy.md", applied)
        target.unlink()
        try:
            os.symlink(decoy, target)
        except (OSError, NotImplementedError):  # pragma: no cover - permissions
            self.skipTest("cannot create a symlink here")
        with self.assertRaises(BackupUnusable):
            self.store(read_only=False).rollback(manifest)
        self.assertTrue(target.is_symlink(), "the link itself must not be replaced or removed")
        self.assertEqual(decoy.read_text(encoding="utf-8"), applied)

    @unittest.skipUnless(hasattr(os, "symlink"), "platform has no symlinks")
    def test_a_skill_directory_that_became_a_link_is_refused_on_rollback_too(self):
        """The leaf check above sees a linked document; a linked DIRECTORY on
        the recorded path leaves the leaf a regular file, hashes through to a
        twin holding the applied text, and would restore over the twin."""
        target, manifest, _previous = self.apply_a_change()
        applied = target.read_text(encoding="utf-8")
        twin = write(self.canonical_dir / "twin" / SKILL_FILENAME, applied)
        shutil.rmtree(target.parent)
        try:
            os.symlink(twin.parent, target.parent)
        except (OSError, NotImplementedError):  # pragma: no cover - permissions
            self.skipTest("cannot create a symlink here")
        self.assertFalse(target.is_symlink(), "the document itself is a regular file here")
        with self.assertRaises(LinkedTargetRefused):
            self.store(read_only=False).rollback(manifest)
        self.assertEqual(twin.read_text(encoding="utf-8"), applied, "the twin must not change")
        self.assertTrue(target.parent.is_symlink(), "the link itself is left in place")


class TreeChangeTests(unittest.TestCase):
    """Two readings of the tree, compared entry by entry and named by kind."""

    def test_an_unchanged_tree_has_no_changes(self):
        entries = {"canonical:a/SKILL.md": "aa", "canonical:b/SKILL.md": UNREADABLE_ENTRY}
        self.assertEqual(describe_tree_change(entries, dict(entries)), ())

    def test_each_kind_of_difference_is_named_and_only_a_moved_hash_is_an_edit(self):
        before = {
            "r:edited/SKILL.md": "aa",
            "r:removed/SKILL.md": "bb",
            "r:unreadable-now/SKILL.md": "cc",
            "r:readable-now/SKILL.md": UNREADABLE_ENTRY,
            "r:gone-unreadable/SKILL.md": UNREADABLE_ENTRY,
            "r:same/SKILL.md": "dd",
        }
        after = {
            "r:edited/SKILL.md": "a2",
            "r:unreadable-now/SKILL.md": UNREADABLE_ENTRY,
            "r:readable-now/SKILL.md": "ee",
            "r:same/SKILL.md": "dd",
            "r:new/SKILL.md": "ff",
            "r:new-unreadable/SKILL.md": UNREADABLE_ENTRY,
        }
        self.assertEqual(
            describe_tree_change(before, after),
            (
                TreeChange("r:edited/SKILL.md", CHANGE_CONTENT),
                TreeChange("r:gone-unreadable/SKILL.md", CHANGE_REMOVED),
                TreeChange("r:new-unreadable/SKILL.md", CHANGE_ADDED_UNREADABLE),
                TreeChange("r:new/SKILL.md", CHANGE_ADDED),
                TreeChange("r:readable-now/SKILL.md", CHANGE_BECAME_READABLE),
                TreeChange("r:removed/SKILL.md", CHANGE_REMOVED),
                TreeChange("r:unreadable-now/SKILL.md", CHANGE_BECAME_UNREADABLE),
            ),
        )

    def test_the_digest_is_over_readable_entries_only(self):
        readable = {"r:a/SKILL.md": "aa"}
        with_unreadable = {"r:a/SKILL.md": "aa", "r:b/SKILL.md": UNREADABLE_ENTRY}
        self.assertEqual(tree_digest_of(readable), tree_digest_of(with_unreadable))
        self.assertNotEqual(tree_digest_of(readable), tree_digest_of({"r:a/SKILL.md": "a2"}))

    def test_the_summary_names_documents_under_their_kind_and_caps_the_list(self):
        changes = [TreeChange(f"r:s{i}/SKILL.md", CHANGE_CONTENT) for i in range(7)]
        changes.append(TreeChange("r:x/SKILL.md", CHANGE_BECAME_UNREADABLE))
        summary = summarize_tree_changes(changes, limit=5)
        self.assertEqual(
            summary,
            "content-changed: r:s0/SKILL.md, r:s1/SKILL.md, r:s2/SKILL.md, r:s3/SKILL.md, "
            "r:s4/SKILL.md, +2 more; became-unreadable: r:x/SKILL.md",
        )


class CanonicalResolutionTests(Harness):
    def test_a_skill_in_both_roots_resolves_to_the_canonical_copy(self):
        group = group_for(self.store().groups(refresh=True), "shared")
        assert group is not None
        assert group.canonical is not None
        self.assertEqual(group.canonical.root_id, "canonical")

    def test_divergence_between_canonical_and_mirror_is_reported(self):
        group = group_for(self.store().groups(refresh=True), "shared")
        assert group is not None
        self.assertTrue(group.is_mirrored)
        self.assertTrue(
            group.divergent_mirrors,
            "divergence means somebody's edit already went to the wrong copy",
        )

    def test_a_mirror_only_skill_has_no_canonical_copy(self):
        group = group_for(self.store().groups(refresh=True), "mirror-only")
        assert group is not None
        self.assertIsNone(
            group.canonical,
            "there is no editable copy, so a proposal against it must be refused "
            "rather than written somewhere ineffective",
        )

    def test_a_passed_over_editable_copy_withholds_the_canonical_slot_and_says_why(self):
        # Pure domain: the store's passed-over list reaches resolve_groups as
        # (root_id, skill_id, reason). Measured before: with the canonical copy
        # of ``shared`` passed over, the group read exactly like ``mirror-only``.
        snapshots = [s for s in self.store().snapshots() if s.root_id == "mirror"]
        groups = resolve_groups(
            snapshots,
            self.roots,
            passed_over=[("canonical", "shared", "a document this process cannot read (x)")],
        )
        shared = group_for(groups, "shared")
        assert shared is not None
        self.assertIsNone(shared.canonical)
        self.assertEqual([m.root_id for m in shared.mirrors], ["mirror"])
        self.assertEqual(
            shared.canonical_reason,
            "the copy in editable root 'canonical' was passed over by discovery (a document "
            "this process cannot read (x)); proposals are refused until it reads, rather than "
            "pointed at another copy",
        )
        only = group_for(groups, "mirror-only")
        assert only is not None
        self.assertEqual(
            only.canonical_reason, "no editable root holds this skill; proposals will be refused"
        )

    def test_two_copies_in_the_selected_root_withhold_the_canonical_slot_and_name_both(self):
        # Measured before: the nested newcomer became canonical by walk order and
        # the real document was listed as a "mirror" in its own root.
        newcomer = self.canonical_dir / "group" / "shared" / SKILL_FILENAME
        write(newcomer, GOOD + "\n- a second document claiming the same skill\n")
        store = self.store()
        shared = group_for(store.groups(), "shared")
        assert shared is not None
        self.assertIsNone(shared.canonical)
        self.assertEqual(
            [c.path for c in shared.ambiguous_copies],
            [newcomer.resolve(), (self.canonical_dir / "shared" / SKILL_FILENAME).resolve()],
            "path-sorted within the root, not walk order",
        )
        self.assertIn("holds 2 copies of this skill", shared.canonical_reason)
        self.assertIn(str(newcomer.resolve()), shared.canonical_reason)
        self.assertIn("ambiguous", shared.canonical_reason)
        self.assertEqual(len(shared.all_copies), 3, "both canonical-root copies and the mirror")
        self.assertEqual(shared.divergent_mirrors, (), "nothing to diverge from")
        payload = shared.to_dict()
        self.assertIsNone(payload["canonical"])
        self.assertEqual(len(payload["ambiguous_copy_paths"]), 2)
        self.assertEqual([], [d for d in store.passed_over()], "both documents were readable")

    def test_two_copies_in_a_lower_precedence_editable_root_leave_the_selected_root_unambiguous(
        self,
    ):
        second = self.base / "second"
        write(second / "shared" / SKILL_FILENAME, GOOD)
        write(second / "group" / "shared" / SKILL_FILENAME, GOOD)
        roots = [
            self.roots[0],
            SkillRoot(
                id="second",
                path=second,
                precedence=50,
                canonical=False,
                editable=True,
                kind="custom",
            ),
            self.roots[1],
        ]
        store = FilesystemSkillStore(roots, read_only=True)
        shared = group_for(store.groups(), "shared")
        assert shared is not None and shared.canonical is not None
        self.assertEqual(shared.canonical.root_id, "canonical")
        self.assertEqual(shared.ambiguous_copies, ())
        self.assertEqual(sorted(m.root_id for m in shared.mirrors), ["mirror", "second", "second"])
        # ...but when that lower root IS the one selected, its two copies are
        # the ambiguity.
        only_second = FilesystemSkillStore([roots[1], roots[2]], read_only=True)
        shared = group_for(only_second.groups(), "shared")
        assert shared is not None
        self.assertIsNone(shared.canonical)
        self.assertEqual({c.root_id for c in shared.ambiguous_copies}, {"second"})
        self.assertIn("editable root 'second' holds 2 copies", shared.canonical_reason)

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file")
    def test_a_passed_over_copy_leads_the_reason_and_the_readable_duplicate_is_named_too(self):
        # Cycle 72 and this cycle at once: the unreadable copy is the actionable
        # fact and withholds the slot; the readable copy beside it is named so the
        # operator does not learn of the second problem only after fixing the first.
        newcomer = self.canonical_dir / "group" / "shared" / SKILL_FILENAME
        write(newcomer, GOOD)
        real = self.canonical_dir / "shared" / SKILL_FILENAME
        os.chmod(real, 0o000)
        try:
            shared = group_for(self.store().groups(), "shared")
        finally:
            os.chmod(real, 0o644)
        assert shared is not None
        self.assertIsNone(shared.canonical)
        self.assertEqual(shared.ambiguous_copies, (), "not stated as ambiguity while one is unread")
        self.assertTrue(shared.canonical_reason.startswith("the copy in editable root 'canonical'"))
        self.assertIn("Permission denied", shared.canonical_reason)
        self.assertIn(
            "also holds 1 readable document(s) claiming this skill", shared.canonical_reason
        )
        self.assertIn(str(newcomer.resolve()), shared.canonical_reason)

    def _root(self, root_id: str, path: Path, *, precedence: int, canonical: bool, editable=True):
        return SkillRoot(
            id=root_id,
            path=path,
            precedence=precedence,
            canonical=canonical,
            editable=editable,
            kind="custom" if editable else "mirror",
        )

    def test_two_canonical_roots_tied_on_precedence_withhold_the_slot_and_name_both_roots(self):
        # Measured before: with roots ``alpha`` and ``zeta`` both canonical and
        # editable at precedence 10, ``alpha`` was proposed against and ``zeta``
        # listed as its mirror, and nothing said the choice was alphabetical.
        alpha = self.base / "alpha"
        write(alpha / "shared" / SKILL_FILENAME, GOOD + "\n- alpha's own line\n")
        roots = [
            self._root("zeta", self.canonical_dir, precedence=10, canonical=True),
            self._root("alpha", alpha, precedence=10, canonical=True),
            self.roots[1],
        ]
        shared = group_for(FilesystemSkillStore(roots, read_only=True).groups(), "shared")
        assert shared is not None
        self.assertIsNone(shared.canonical)
        self.assertEqual(sorted(c.root_id for c in shared.ambiguous_copies), ["alpha", "zeta"])
        self.assertEqual(len(shared.all_copies), 3, "both tied copies and the mirror")
        self.assertIn(
            "roots 'alpha', 'zeta' each hold a copy of this skill", shared.canonical_reason
        )
        self.assertIn("all canonical and editable at precedence 10", shared.canonical_reason)
        self.assertIn("decided by root id alone", shared.canonical_reason)
        self.assertIn("distinct precedences (the lower number wins)", shared.canonical_reason)
        self.assertEqual(
            sorted(shared.to_dict()["ambiguous_copy_paths"]),
            sorted(
                [
                    str((alpha / "shared" / SKILL_FILENAME).resolve()),
                    str((self.canonical_dir / "shared" / SKILL_FILENAME).resolve()),
                ]
            ),
        )

    def test_two_merely_editable_roots_tied_on_precedence_are_the_same_ambiguity(self):
        alpha = self.base / "alpha"
        write(alpha / "shared" / SKILL_FILENAME, GOOD)
        roots = [
            self._root("zeta", self.canonical_dir, precedence=40, canonical=False),
            self._root("alpha", alpha, precedence=40, canonical=False),
            self.roots[1],
        ]
        shared = group_for(FilesystemSkillStore(roots, read_only=True).groups(), "shared")
        assert shared is not None
        self.assertIsNone(shared.canonical)
        self.assertIn(
            "all editable, and none of them is marked canonical at precedence 40",
            shared.canonical_reason,
        )
        self.assertIn(
            "one is marked canonical = true",
            shared.canonical_reason,
            "the remedy that works in this tier; canonical = false would change nothing",
        )
        self.assertNotIn("canonical = false", shared.canonical_reason)

    def test_a_duplicate_inside_a_tied_root_is_named_in_the_same_reason(self):
        alpha = self.base / "alpha"
        write(alpha / "shared" / SKILL_FILENAME, GOOD)
        nested = self.canonical_dir / "group" / "shared" / SKILL_FILENAME
        write(nested, GOOD)
        roots = [
            self._root("zeta", self.canonical_dir, precedence=10, canonical=True),
            self._root("alpha", alpha, precedence=10, canonical=True),
            self.roots[1],
        ]
        shared = group_for(FilesystemSkillStore(roots, read_only=True).groups(), "shared")
        assert shared is not None
        self.assertIsNone(shared.canonical)
        self.assertEqual(len(shared.ambiguous_copies), 3)
        self.assertTrue(shared.canonical_reason.startswith("roots 'alpha', 'zeta' each hold"))
        self.assertIn("root 'zeta' also holds 2 copies of this skill", shared.canonical_reason)
        self.assertIn(str(nested.resolve()), shared.canonical_reason)
        self.assertIn("an ambiguity of its own once the tie is settled", shared.canonical_reason)

    def test_a_tied_copy_is_refused_at_the_write_path_as_unsettled_not_as_a_mirror(self):
        alpha = self.base / "alpha"
        tied = write(alpha / "shared" / SKILL_FILENAME, GOOD)
        roots = [
            self._root("zeta", self.canonical_dir, precedence=10, canonical=True),
            self._root("alpha", alpha, precedence=10, canonical=True),
        ]
        store = FilesystemSkillStore(roots, read_only=False)
        for target in (tied, self.canonical_dir / "shared" / SKILL_FILENAME):
            with self.assertRaises(MirrorEditRefused) as ctx:
                store.write_canonical(
                    target, "x", expected_sha256=sha256_file(target), backup_dir=self.base / "bak"
                )
            message = str(ctx.exception)
            self.assertIn("skill 'shared' has no canonical copy (roots 'alpha', 'zeta'", message)
            self.assertNotIn("mirror copy", message)
            self.assertNotIn("unknown", message)
        self.assertEqual(sha256_file(tied), sha256_file(alpha / "shared" / SKILL_FILENAME))

    def test_the_control_distinct_precedences_pick_the_lower_number_whatever_the_ids(self):
        alpha = self.base / "alpha"
        write(alpha / "shared" / SKILL_FILENAME, GOOD)
        roots = [
            self._root("alpha", alpha, precedence=20, canonical=True),
            self._root("zeta", self.canonical_dir, precedence=10, canonical=True),
            self.roots[1],
        ]
        shared = group_for(FilesystemSkillStore(roots, read_only=True).groups(), "shared")
        assert shared is not None and shared.canonical is not None
        self.assertEqual(shared.canonical.root_id, "zeta", "precedence, not the alphabet")
        self.assertEqual(shared.ambiguous_copies, ())
        self.assertEqual(sorted(m.root_id for m in shared.mirrors), ["alpha", "mirror"])

    def test_the_control_a_marked_canonical_root_outranks_an_unmarked_one_at_the_same_precedence(
        self,
    ):
        alpha = self.base / "alpha"
        write(alpha / "shared" / SKILL_FILENAME, GOOD)
        roots = [
            self._root("alpha", alpha, precedence=10, canonical=False),
            self.roots[0],
            self.roots[1],
        ]
        shared = group_for(FilesystemSkillStore(roots, read_only=True).groups(), "shared")
        assert shared is not None and shared.canonical is not None
        self.assertEqual(shared.canonical.root_id, "canonical")
        self.assertEqual(shared.ambiguous_copies, (), "different tiers do not tie")

    def test_the_control_a_read_only_root_at_the_same_precedence_does_not_tie(self):
        # Marked canonical too: a root that cannot be written is in no tier,
        # whatever else it is marked, so it cannot tie with the writable one.
        roots = [
            self.roots[0],
            self._root("mirror", self.mirror_dir, precedence=10, canonical=True, editable=False),
        ]
        shared = group_for(FilesystemSkillStore(roots, read_only=True).groups(), "shared")
        assert shared is not None and shared.canonical is not None
        self.assertEqual(shared.canonical.root_id, "canonical")
        self.assertEqual(shared.ambiguous_copies, ())

    def test_two_copies_in_a_root_that_is_not_editable_change_nothing(self):
        write(self.mirror_dir / "group" / "shared" / SKILL_FILENAME, GOOD)
        shared = group_for(self.store().groups(), "shared")
        assert shared is not None and shared.canonical is not None
        self.assertEqual(shared.canonical.root_id, "canonical")
        self.assertEqual(shared.ambiguous_copies, ())
        self.assertEqual([m.root_id for m in shared.mirrors], ["mirror", "mirror"])

    def test_a_passed_over_copy_in_a_root_that_is_not_editable_changes_nothing(self):
        snapshots = self.store().snapshots()
        groups = resolve_groups(
            snapshots, self.roots, passed_over=[("mirror", "shared", "a symlink; not followed")]
        )
        shared = group_for(groups, "shared")
        assert shared is not None and shared.canonical is not None
        self.assertEqual(shared.canonical.root_id, "canonical")

    def test_a_lower_precedence_editable_copy_is_not_promoted_over_a_passed_over_one(self):
        # Two editable roots; the higher-precedence copy is passed over. Pointing
        # a proposal at the other copy would edit the wrong file.
        second = self.base / "second"
        write(second / "shared" / SKILL_FILENAME, GOOD)
        roots = [
            self.roots[0],
            SkillRoot(
                id="second",
                path=second,
                precedence=50,
                canonical=False,
                editable=True,
                kind="custom",
            ),
        ]
        snapshots = [s for s in FilesystemSkillStore(roots, read_only=True).snapshots()]
        snapshots = [
            s for s in snapshots if not (s.root_id == "canonical" and s.skill_id == "shared")
        ]
        groups = resolve_groups(snapshots, roots, passed_over=[("canonical", "shared", "mode 000")])
        shared = group_for(groups, "shared")
        assert shared is not None
        self.assertIsNone(shared.canonical, "not the second root's copy")
        self.assertIn("'canonical'", shared.canonical_reason)

    def test_an_entry_that_is_not_a_copy_of_the_skill_withholds_nothing(self):
        # Measured by the cycle-72 audit: an unlistable directory inside tdd/, a
        # link at nested/tdd/SKILL.md, a too-deep a/b/tdd/SKILL.md and a dangling
        # root-level SKILL.md all carried the skill id of a readable document and
        # withheld its canonical slot, dropping a true divergence and turning an
        # approved apply into a mirror-edit refusal.
        shared_dir = self.canonical_dir / "shared"
        (shared_dir / "sub").mkdir()
        (self.canonical_dir / "nested" / "shared").mkdir(parents=True)
        (self.canonical_dir / "nested" / "shared" / SKILL_FILENAME).symlink_to(
            shared_dir / SKILL_FILENAME
        )
        deep = self.canonical_dir / "a" / "b" / "c" / "shared"
        deep.mkdir(parents=True)
        write(deep / SKILL_FILENAME, GOOD)
        (self.canonical_dir / SKILL_FILENAME).symlink_to(self.canonical_dir / "nowhere.md")
        if os.geteuid() != 0:
            os.chmod(shared_dir / "sub", 0o000)
        try:
            store = self.store()
            passed = store.passed_over()
            self.assertGreaterEqual(len(passed), 3, [d.reason for d in passed])
            nested = (self.canonical_dir / "nested" / "shared" / SKILL_FILENAME).resolve()
            for document in passed:
                if document.path.resolve() == nested:
                    # A second copy of ``shared`` in the same root, at a place
                    # discovery accepts: that one IS a copy, and withholds.
                    self.assertEqual(document.skill_id, "shared")
                else:
                    self.assertIsNone(document.skill_id, document)
            # Without the nested link, nothing here is a copy of ``shared``.
            (self.canonical_dir / "nested" / "shared" / SKILL_FILENAME).unlink()
            store = self.store()
            self.assertEqual(
                [d.skill_id for d in store.passed_over()], [None] * len(store.passed_over())
            )
            group = group_for(store.groups(), "shared")
            assert group is not None and group.canonical is not None
            self.assertEqual(group.canonical.root_id, "canonical", "the readable document rules")
            self.assertTrue(group.divergent_mirrors, "the true divergence is still reported")
        finally:
            if os.geteuid() != 0:
                os.chmod(shared_dir / "sub", 0o755)

    def test_a_passed_over_link_at_a_nested_place_is_a_copy_of_that_skill_not_another(self):
        # nested/shared/SKILL.md names skill ``shared`` under discovery's own rule
        # (parent directory); the canonical shared/SKILL.md stays readable, so the
        # withhold is about a second copy of the same skill in the same editable
        # root, the one case where refusing is right: two copies, one unseen.
        (self.canonical_dir / "nested" / "other").mkdir(parents=True)
        (self.canonical_dir / "nested" / "other" / SKILL_FILENAME).symlink_to(
            self.canonical_dir / "shared" / SKILL_FILENAME
        )
        store = self.store()
        self.assertEqual([d.skill_id for d in store.passed_over()], ["other"])
        shared = group_for(store.groups(), "shared")
        assert shared is not None and shared.canonical is not None
        self.assertEqual(shared.canonical.root_id, "canonical")
        self.assertIsNone(group_for(store.groups(), "other"), "no readable copy, no group")

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file")
    def test_the_store_feeds_what_it_passed_over_into_its_groups(self):
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        os.chmod(target, 0o000)
        try:
            group = group_for(self.store().groups(refresh=True), "shared")
            assert group is not None
            self.assertIsNone(group.canonical)
            self.assertIn("passed over by discovery", group.canonical_reason)
            self.assertIn("Permission denied", group.canonical_reason)
            self.assertEqual(group.divergent_mirrors, (), "nothing to compare against")
        finally:
            os.chmod(target, 0o644)

    def test_the_tree_digest_changes_when_any_skill_changes(self):
        store = self.store()
        before = store.tree_digest()
        write(self.canonical_dir / "shared" / SKILL_FILENAME, GOOD + "\n- new line\n")
        after = self.store().tree_digest()
        self.assertNotEqual(before, after)

    def test_the_tree_digest_is_stable_for_an_unchanged_tree(self):
        self.assertEqual(self.store().tree_digest(), self.store().tree_digest())

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file")
    def test_tree_entries_name_an_unreadable_copy_and_the_digest_leaves_it_out(self):
        # Measured before: the digest was the only reading of the tree, and a copy
        # that could not be read was simply absent from it, so a document made
        # unreadable mid-run read exactly like one that had been edited away.
        readable = self.store().tree_entries()
        self.assertEqual(readable["canonical:shared/SKILL.md"], sha256_text(GOOD))
        self.assertNotIn(UNREADABLE_ENTRY, readable.values())
        target = self.canonical_dir / "shared" / SKILL_FILENAME
        os.chmod(target, 0o000)
        try:
            store = self.store()
            entries = store.tree_entries()
            self.assertEqual(entries["canonical:shared/SKILL.md"], UNREADABLE_ENTRY)
            self.assertEqual(
                {k: v for k, v in entries.items() if k != "canonical:shared/SKILL.md"},
                {k: v for k, v in readable.items() if k != "canonical:shared/SKILL.md"},
                "every other entry is untouched",
            )
            self.assertEqual(store.tree_digest(), tree_digest_of(entries))
            self.assertNotIn(
                "canonical:shared/SKILL.md",
                {k for k, v in entries.items() if v != UNREADABLE_ENTRY},
                "the digest is over readable documents only",
            )
            self.assertEqual(
                describe_tree_change(readable, entries),
                (TreeChange("canonical:shared/SKILL.md", CHANGE_BECAME_UNREADABLE),),
            )
        finally:
            os.chmod(target, 0o644)

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file")
    def test_a_nested_copy_keeps_one_key_across_readable_and_unreadable(self):
        # A nested skill is where the relative key and the skill-id fallback differ;
        # if the readable and the passed-over reading keyed the same document
        # differently, one chmod would read as removed plus added-unreadable.
        nested = self.canonical_dir / "group" / "nested"
        nested.mkdir(parents=True)
        target = nested / SKILL_FILENAME
        write(target, GOOD.replace("name: shared", "name: nested"))
        before = self.store().tree_entries()
        self.assertEqual(before["canonical:group/nested/SKILL.md"], sha256_file(target))
        os.chmod(target, 0o000)
        try:
            after = self.store().tree_entries()
            self.assertEqual(after["canonical:group/nested/SKILL.md"], UNREADABLE_ENTRY)
            self.assertEqual(
                describe_tree_change(before, after),
                (TreeChange("canonical:group/nested/SKILL.md", CHANGE_BECAME_UNREADABLE),),
            )
        finally:
            os.chmod(target, 0o644)

    def test_a_document_that_is_not_a_copy_of_a_skill_is_not_a_tree_entry(self):
        # Too deep for discovery, so not a copy of anything: it withholds no slot
        # and its appearance or disappearance is not a tree change either.
        deep = self.canonical_dir / "a" / "b" / "c" / "d" / "deep"
        deep.mkdir(parents=True)
        write(deep / SKILL_FILENAME, GOOD)
        store = self.store()
        self.assertTrue([d for d in store.passed_over() if d.skill_id is None])
        self.assertFalse([k for k in store.tree_entries() if k.endswith("deep/SKILL.md")])

    def test_relocating_a_root_does_not_look_like_a_skill_mutation(self):
        """The digest is about skill CONTENT, not about filesystem layout.

        It used to key on the absolute path, so moving a root changed the digest
        while nothing about any skill changed. That is a false positive on the one
        receipt field an operator is told to stop and investigate, and a mutation
        alarm that cries wolf is an alarm nobody reads.
        """
        before = self.store().tree_digest()
        moved_base = self.base / "relocated"
        moved_base.mkdir()
        moved_canonical = moved_base / "canonical"
        moved_mirror = moved_base / "mirror"
        shutil.copytree(self.canonical_dir, moved_canonical)
        shutil.copytree(self.mirror_dir, moved_mirror)
        relocated = FilesystemSkillStore(
            [
                SkillRoot(
                    id="canonical",
                    path=moved_canonical,
                    precedence=10,
                    canonical=True,
                    editable=True,
                    kind="custom",
                ),
                SkillRoot(
                    id="mirror",
                    path=moved_mirror,
                    precedence=90,
                    canonical=False,
                    editable=False,
                    kind="mirror",
                ),
            ],
            read_only=True,
        )
        self.assertEqual(before, relocated.tree_digest())

    def test_a_file_moving_between_roots_still_changes_the_digest(self):
        before = self.store().tree_digest()
        write(self.mirror_dir / "broken" / SKILL_FILENAME, NO_OPENING_DELIMITER)
        self.assertNotEqual(
            before,
            self.store().tree_digest(),
            "root_id is part of the key, so the same content in a different root is "
            "a different tree",
        )


class SkillRootsMustBeDisjointTests(unittest.TestCase):
    """Two roots that name one directory, or one root inside another, are a
    config error rather than a working configuration.

    Measured before this check existed: a ``mirror`` root spelled as the
    canonical directory loaded cleanly, the inventory found every skill twice
    (once per root id), each group listed the skill as a mirror of itself, and
    the root flagged ``editable = false`` was the exact directory
    ``write_canonical`` landed in. The mirror interlock judged the file
    canonical first, so the read-only flag protected nothing. The check compares
    the lexically folded spelling and the link-resolved one, so ``.``, ``..`` and
    a symbolic link to another root are all caught with no directory needing to
    exist; roots that do exist are also compared by identity, which catches a
    case variant on a filesystem that folds case. A sibling that merely shares
    a name prefix is not inside the other.
    """

    def _config_with_mirror_at(self, base: Path, spelling: str) -> Path:
        """The harness config with the mirror root's path replaced by ``spelling``."""
        config = Path(build_env(base, dry_run=True))
        text = config.read_text(encoding="utf-8")
        mirror = json.dumps(str(base / "skills-mirror"))
        assert text.count(mirror) == 1, text
        config.write_text(text.replace(mirror, json.dumps(spelling)), encoding="utf-8")
        return config

    def _refusal(self, config: Path) -> str:
        from agent_skill_improver.config import load_config
        from agent_skill_improver.errors import ConfigError

        with self.assertRaises(ConfigError) as ctx:
            load_config(config)
        return str(ctx.exception)

    def test_a_mirror_root_naming_the_canonical_directory_is_refused_by_both_ids(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            message = self._refusal(
                self._config_with_mirror_at(base, str(base / "skills-canonical"))
            )
            self.assertIn("'canonical'", message)
            self.assertIn("'mirror'", message)
            self.assertIn("same directory", message)

    def test_the_same_directory_spelled_with_redundant_segments_is_still_refused(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            spelled = str(base / "skills-mirror" / ".." / "skills-canonical" / ".")
            message = self._refusal(self._config_with_mirror_at(base, spelled))
            self.assertIn("same directory", message)
            self.assertIn(spelled, message, "the refusal quotes the spelling as configured")

    def test_a_root_reached_through_a_link_to_another_root_is_refused(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "alias").symlink_to(base / "skills-canonical")
            message = self._refusal(self._config_with_mirror_at(base, str(base / "alias")))
            self.assertIn("same directory", message)
            self.assertIn(str(base / "alias"), message)

    def test_a_root_inside_another_is_refused_whichever_is_listed_first(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            inner = str(base / "skills-canonical" / "nested")
            message = self._refusal(self._config_with_mirror_at(base, inner))
            self.assertIn(f"'mirror' ({inner}) lies inside skill_root 'canonical'", message)
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            outer = str(base)
            message = self._refusal(self._config_with_mirror_at(base, outer))
            self.assertIn("'canonical'", message)
            self.assertIn(f"lies inside skill_root 'mirror' ({outer})", message)

    def test_a_case_variant_of_an_existing_root_is_judged_by_identity_not_text(self):
        """On a filesystem that folds case (macOS APFS by default), ``SKILLS``
        and ``skills`` are one directory and the alias is refused. On one that
        does not (Linux ext4, so the CI runner), they are two directories and
        the config is legitimate. Both verdicts are asserted, each where it is
        true, so the test is a measurement on either host rather than a skip."""
        from agent_skill_improver.config import load_config
        from agent_skill_improver.errors import ConfigError

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            variant = base / "SKILLS-CANONICAL"
            config = self._config_with_mirror_at(base, str(variant))
            folds_case = variant.exists() and os.path.samefile(base / "skills-canonical", variant)
            if folds_case:
                with self.assertRaises(ConfigError) as ctx:
                    load_config(config)
                self.assertIn("same directory", str(ctx.exception))
                self.assertIn(str(variant), str(ctx.exception))
            else:
                self.assertFalse(variant.exists())
                loaded = load_config(config)
                self.assertEqual([r.id for r in loaded.skill_roots], ["canonical", "mirror"])

    def test_the_control_sibling_roots_load_even_when_one_does_not_exist_yet(self):
        from agent_skill_improver.config import load_config

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = self._config_with_mirror_at(base, str(base / "skills-canonical-to-be"))
            self.assertFalse((base / "skills-canonical-to-be").exists())
            loaded = load_config(config)
            self.assertEqual([r.id for r in loaded.skill_roots], ["canonical", "mirror"])

    def test_doctor_reports_overlapping_roots_as_a_config_error(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = self._config_with_mirror_at(base, str(base / "skills-canonical"))
            proc = run_cli(["--config", str(config), "doctor"])
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertIn("config error", proc.stderr)
            self.assertIn("same directory", proc.stderr)


class EditableRootsMustBeWritableTests(unittest.TestCase):
    """A canonical, editable root inside a protected path is a config error.

    The protected set (package, config directory, state directory, rubrics
    directory) is derived in code and every write under it is refused as
    self-mutation. Measured before this check existed: a canonical root at
    ``config/skills`` loaded, ``doctor`` reported nothing, and the first write
    under it raised ``SelfMutationRefused``, after the candidates it refused had
    been proposed, scored and reviewed. A read-only mirror in the same place is
    legitimate, and so is a root that contains the config or state directory.
    """

    def _config(self, base: Path, *, canonical: str | None = None, mirror: str | None = None):
        config = Path(build_env(base, dry_run=True))
        text = config.read_text(encoding="utf-8")
        for current, replacement in (
            (str(base / "skills-canonical"), canonical),
            (str(base / "skills-mirror"), mirror),
        ):
            if replacement is None:
                continue
            assert text.count(json.dumps(current)) == 1, text
            text = text.replace(json.dumps(current), json.dumps(replacement))
        config.write_text(text, encoding="utf-8")
        return config

    def _refusal(self, config: Path) -> str:
        from agent_skill_improver.config import load_config
        from agent_skill_improver.errors import ConfigError

        with self.assertRaises(ConfigError) as ctx:
            load_config(config)
        return str(ctx.exception)

    def test_a_canonical_root_inside_the_config_directory_is_refused_at_load(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            inside = str(base / "config" / "skills")
            message = self._refusal(self._config(base, canonical=inside))
            self.assertIn(f"'canonical' ({inside})", message)
            self.assertIn("the config directory", message)
            self.assertIn("self-mutation", message)

    def test_a_canonical_root_inside_or_at_the_state_directory_is_refused_at_load(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            message = self._refusal(self._config(base, canonical=str(base / "state" / "skills")))
            self.assertIn("the state directory", message)
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            message = self._refusal(self._config(base, canonical=str(base / "state")))
            self.assertIn("the state directory", message)

    def test_a_canonical_root_inside_the_rubrics_directory_is_refused_at_load(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            message = self._refusal(self._config(base, canonical=str(base / "rubrics" / "skills")))
            self.assertIn("the rubrics directory", message)

    def test_a_canonical_root_inside_the_tool_package_is_refused_at_load(self):
        from agent_skill_improver.config import package_dir

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            inside = str(package_dir() / "would-be-skills")
            self.assertFalse(Path(inside).exists(), "nothing is created in the package")
            message = self._refusal(self._config(base, canonical=inside))
            self.assertIn("this tool's own package", message)
            self.assertFalse(Path(inside).exists())

    def test_an_unmarked_editable_root_inside_a_protected_path_is_refused_at_load_too(self):
        """The resolver treats the highest-precedence editable root as the
        canonical one when none is marked, so an editable root that is not
        marked canonical is proposed against all the same and must be held to
        the same check. Measured before: with ``canonical = false`` the root
        loaded and the write was refused only at apply."""
        from agent_skill_improver.config import load_config

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            inside = str(base / "rubrics" / "skills")
            config = self._config(base, canonical=inside)
            text = config.read_text(encoding="utf-8")
            assert text.count("canonical = true\n") == 1, text
            config.write_text(
                text.replace("canonical = true\n", "canonical = false\n"), encoding="utf-8"
            )
            message = self._refusal(config)
            self.assertIn(f"'canonical' ({inside}) is editable but", message)
            self.assertIn("the rubrics directory", message)
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = self._config(base, canonical=str(base / "rubrics" / "skills"))
            text = config.read_text(encoding="utf-8")
            assert text.count("canonical = true\neditable = true\n") == 1, text
            config.write_text(
                text.replace(
                    "canonical = true\neditable = true\n", "canonical = false\neditable = false\n"
                ),
                encoding="utf-8",
            )
            loaded = load_config(config)
            self.assertEqual([r.id for r in loaded.skill_roots], ["canonical", "mirror"])

    def test_the_control_a_read_only_mirror_inside_the_config_directory_loads(self):
        from agent_skill_improver.config import load_config

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            loaded = load_config(self._config(base, mirror=str(base / "config" / "skills")))
            self.assertEqual([r.id for r in loaded.skill_roots], ["canonical", "mirror"])

    def test_the_control_a_root_that_contains_the_config_and_state_directories_loads(self):
        """The reverse containment is legitimate: skills elsewhere under that
        root are writable, and only the ones under the protected directories
        are refused, by the write path, with the protected path named."""
        from agent_skill_improver.config import load_config

        with TemporaryDirectory() as tmp:
            wide = Path(tmp) / "wide"
            config = self._config(wide, canonical=str(wide), mirror=str(Path(tmp) / "elsewhere"))
            loaded = load_config(config)
            self.assertEqual(loaded.skill_roots[0].path, wide)
            self.assertIn(loaded.config_path.parent.resolve(), loaded.protected_paths)

    def test_doctor_reports_a_canonical_root_in_a_protected_path_as_a_config_error(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = self._config(base, canonical=str(base / "config" / "skills"))
            proc = run_cli(["--config", str(config), "doctor"])
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertIn("config error", proc.stderr)
            self.assertIn("self-mutation", proc.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
