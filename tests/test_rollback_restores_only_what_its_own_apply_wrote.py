"""Rollback restores only what its own apply wrote.

``rollback`` read the document, the backup and both hashes from the backup
manifest, a plain file under the state tree, and checked each against the
manifest's own values, never against the candidate. ``rollback_candidate`` in
``app/promote.py`` now passes the candidate's write (its document, from its
base to its replacement) and ``FilesystemSkillStore.rollback`` refuses a
manifest that does not record that write (``_assert_records_the_write``) before
anything is read or written. Measured before (3.11 and 3.13, the same on both;
``daily --commit``, ``review``, ``apply``, then the manifest edited, then
``rollback``): a manifest naming another skill's document, with a backup
planted beside it and the hashes to match, wrote the planted text into that
skill; a backup planted for the candidate's own document was written in place
of its original text; and a ``new_sha256`` changed to the hash of a later edit
let rollback overwrite that edit. Each time rollback exited 0 and recorded the
candidate rolled back.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.support import build_env, run_cli

TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
PLANTED = "---\nname: {name}\ndescription: planted\n---\nPLANTED TEXT\n"
LATER = "\nA LATER EDIT BY HAND\n"
CASES = (
    "control",
    "another_skill",
    "planted_backup",
    "later_edit",
    "hostile_hash",
    "same_document_respelled",
    "another_skill_backup_missing",
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_case(base: Path, case: str) -> dict:
    config = ["--config", str(build_env(base, dry_run=False, seed="c185"))]
    daily = run_cli([*config, "--json", "daily", "--commit", "--day", "2026-01-02"])
    assert daily.returncode == 0, daily.stderr
    (record_path,) = sorted((base / "state" / "candidates").glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    cid = record["id"]
    own = Path(record["target"]["canonical_path"])
    other = base / "skills-canonical" / "code-review" / "SKILL.md"
    result: dict = {"own_base": own.read_bytes(), "other_before": other.read_bytes()}
    token = json.loads(run_cli([*config, "--json", "review", cid]).stdout)["approval_token"]
    applied = run_cli([*config, "apply", cid, "--approve", token, "--i-reviewed-this"])
    assert applied.returncode == 0, applied.stderr
    result["own_applied"] = own.read_bytes()
    (manifest_dir,) = sorted((base / "state" / "backups").glob(f"{cid}--*"))
    manifest_path = manifest_dir / "backup.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    if case == "another_skill":
        planted = PLANTED.format(name="code-review")
        (manifest_dir / "planted.bak").write_text(planted, encoding="utf-8")
        manifest.update(
            target_path=str(other.resolve()),
            backup_path=str(manifest_dir / "planted.bak"),
            previous_sha256=sha(planted),
            new_sha256=sha(result["other_before"].decode("utf-8")),
        )
    elif case == "planted_backup":
        planted = PLANTED.format(name="tdd")
        (manifest_dir / "planted.bak").write_text(planted, encoding="utf-8")
        manifest.update(backup_path=str(manifest_dir / "planted.bak"), previous_sha256=sha(planted))
    elif case == "later_edit":
        later = result["own_applied"].decode("utf-8") + LATER
        own.write_text(later, encoding="utf-8")
        result["own_later"] = own.read_bytes()
        manifest.update(new_sha256=sha(later))
    elif case == "hostile_hash":
        manifest.update(new_sha256=TOKEN + "x" * 200_000)
    elif case == "another_skill_backup_missing":
        # Checked first, the record refuses this manifest; with the backup
        # judged first the refusal would be the missing backup's instead.
        manifest.update(
            target_path=str(other.resolve()),
            backup_path=str(manifest_dir / "missing.bak"),
            previous_sha256=sha("not on disk"),
            new_sha256=sha(result["other_before"].decode("utf-8")),
        )
    elif case == "same_document_respelled":
        manifest.update(target_path=str(own.parent / ".." / own.parent.name / own.name))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rollback = run_cli([*config, "rollback", cid])
    result.update(
        rollback=rollback,
        own_after=own.read_bytes(),
        other_after=other.read_bytes(),
        status=json.loads(record_path.read_text(encoding="utf-8"))["status"],
        rollback_receipts=sorted(
            p.name for p in (base / "state").rglob("*") if p.is_file() and "rollback" in p.name
        ),
    )
    if case != "control":
        manifest_path.write_bytes(original)
        again = run_cli([*config, "rollback", cid])
        result.update(again=again, own_after_again=own.read_bytes())
    return result


class RollbackRestoresOnlyWhatItsOwnApplyWroteTests(unittest.TestCase):
    cases: dict[str, dict] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name).resolve()
        for case in CASES:
            cls.cases[case] = run_case(root / case, case)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def refused(self, case: str) -> str:
        result = self.cases[case]
        self.assertEqual(result["rollback"].returncode, 3, result["rollback"].stdout)
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["rollback_receipts"], [])
        self.assertEqual(result["other_after"], result["other_before"])
        message = result["rollback"].stderr
        self.assertIn("is not the record of this candidate's apply", message)
        self.assertIn("so no file was touched", message)
        return message

    def test_the_control_its_own_manifest_restores_the_base(self) -> None:
        result = self.cases["control"]
        self.assertEqual(result["rollback"].returncode, 0, result["rollback"].stderr)
        self.assertEqual(result["own_after"], result["own_base"])
        self.assertEqual(result["status"], "rolled-back")
        self.assertEqual(result["other_after"], result["other_before"])
        self.assertEqual(len(result["rollback_receipts"]), 1)

    def test_a_manifest_naming_another_skill_is_refused(self) -> None:
        message = self.refused("another_skill")
        result = self.cases["another_skill"]
        self.assertEqual(result["own_after"], result["own_applied"])
        self.assertIn("/skills-canonical/code-review/SKILL.md, not this candidate's", message)
        self.assertIn("document ", message)
        self.assertIn("/skills-canonical/tdd/SKILL.md", message)

    def test_a_backup_planted_for_its_own_document_is_refused(self) -> None:
        message = self.refused("planted_backup")
        result = self.cases["planted_backup"]
        self.assertEqual(result["own_after"], result["own_applied"])
        self.assertNotIn("it names", message)
        base = sha(result["own_base"].decode("utf-8"))
        self.assertIn(
            f"it records the text going from '{sha(PLANTED.format(name='tdd'))}'", message
        )
        self.assertIn(f"not from this candidate's base '{base}' to its replacement", message)

    def test_a_later_edit_is_not_overwritten_through_an_edited_manifest(self) -> None:
        self.refused("later_edit")
        result = self.cases["later_edit"]
        self.assertEqual(result["own_after"], result["own_later"])

    def test_the_refusal_quotes_the_manifest_masked_and_bounded(self) -> None:
        message = self.refused("hostile_hash")
        self.assertNotIn(TOKEN, message)
        self.assertIn("[REDACTED:github-token]", message)
        self.assertLess(len(message), 3_000)
        result = self.cases["hostile_hash"]
        self.assertEqual(result["own_after"], result["own_applied"])

    def test_the_record_is_checked_before_the_backup_is_read(self) -> None:
        message = self.refused("another_skill_backup_missing")
        result = self.cases["another_skill_backup_missing"]
        self.assertEqual(result["own_after"], result["own_applied"])
        self.assertNotIn("missing.bak", message)
        self.assertNotIn("could not be checked", message)

    def test_the_same_document_spelled_another_way_still_rolls_back(self) -> None:
        result = self.cases["same_document_respelled"]
        self.assertEqual(result["rollback"].returncode, 0, result["rollback"].stderr)
        self.assertEqual(result["own_after"], result["own_base"])
        self.assertEqual(result["status"], "rolled-back")

    def test_with_its_manifest_restored_the_candidate_rolls_back(self) -> None:
        for case in (
            "another_skill",
            "planted_backup",
            "hostile_hash",
            "another_skill_backup_missing",
        ):
            with self.subTest(case=case):
                result = self.cases[case]
                self.assertEqual(result["again"].returncode, 0, result["again"].stderr)
                self.assertEqual(result["own_after_again"], result["own_base"])


if __name__ == "__main__":
    unittest.main()
