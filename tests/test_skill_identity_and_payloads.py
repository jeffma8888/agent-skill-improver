"""Which file is 'the skill', and what else lives beside it.

Two failures live here and both are silent. Editing a mirror copy succeeds, is
reported as success, and never reaches whatever loads the canonical file. And a
skill directory is not just a document: it can hold scripts, hooks and launchers,
so a write path that accepts any filename inside it is a code-execution primitive
wearing a documentation-change costume.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.domain.skills import (
    PAYLOAD_EXTENSIONS,
    SKILL_FILENAME,
    SkillRoot,
    group_for,
    is_payload_name,
    parse_frontmatter,
)
from agent_skill_improver.errors import (
    BackupUnusable,
    BaseHashMismatch,
    LinkedTargetRefused,
    MirrorEditRefused,
    NonDocumentWriteRefused,
    ReadOnlyViolation,
    SelfMutationRefused,
)
from agent_skill_improver.hashing import sha256_file
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

    def test_the_tree_digest_changes_when_any_skill_changes(self):
        store = self.store()
        before = store.tree_digest()
        write(self.canonical_dir / "shared" / SKILL_FILENAME, GOOD + "\n- new line\n")
        after = self.store().tree_digest()
        self.assertNotEqual(before, after)

    def test_the_tree_digest_is_stable_for_an_unchanged_tree(self):
        self.assertEqual(self.store().tree_digest(), self.store().tree_digest())

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
