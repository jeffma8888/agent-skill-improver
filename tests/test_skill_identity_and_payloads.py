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
    MirrorEditRefused,
    NonDocumentWriteRefused,
    ReadOnlyViolation,
    SelfMutationRefused,
)
from agent_skill_improver.hashing import sha256_file

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

    def test_a_manifest_pointing_at_a_payload_file_is_refused_on_its_filename(self):
        _target, manifest, _previous = self.apply_a_change()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        script = write(self.canonical_dir / "shared" / "scripts" / "hook.sh", "#!/bin/sh\n")
        payload["target_path"] = str(script)
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(NonDocumentWriteRefused):
            self.store(read_only=False).rollback(manifest)
        self.assertEqual(script.read_text(encoding="utf-8"), "#!/bin/sh\n")


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
