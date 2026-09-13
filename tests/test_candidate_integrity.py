"""Derived candidate fields are recomputed, so an approval binds the reviewed bytes.

The defect this module exists for was measured, not imagined. A validated
candidate was reviewed, its token saved, and then exactly one field of the
persisted record was edited -- ``patch.new_text`` -- leaving ``new_text_sha256``,
``approval_challenge``, ``status`` and the recorded gate untouched. ``asi apply``
with the original token returned 0 and wrote the edited text into the live skill
file. The token was derived from the *stored* hash, so it was never bound to the
text a human read.

Two halves, both needed:

* :class:`TamperedApplyIsRefusedEndToEnd` runs the real CLI as a process, tampers
  with the record on disk exactly as the reproduction did, and asserts the skill
  file, the candidate's status, the backup directory and the receipts are all
  untouched afterwards -- with healthy controls on either side, because a suite
  where apply always refuses proves nothing about the interlock.
* the unit classes cover one derived field each, in both directions: the
  recomputation must report the mismatch, and must stay silent on a clean record.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_skill_improver.app.promote import apply_candidate, review_report
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.app.validate import validate_candidate
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.domain.approval import HumanApproval
from agent_skill_improver.domain.candidate import (
    STATUS_APPLIED,
    STATUS_GATE_FAILED,
    STATUS_PROPOSED,
    STATUS_VALIDATED,
    Candidate,
    RubricFailureRef,
    SourceRef,
    TargetRef,
    challenge_for,
    make_patch,
)
from agent_skill_improver.domain.integrity import (
    derived_field_mismatches,
    gate_status_problems,
    target_identity_problems,
    verify_candidate_integrity,
)
from agent_skill_improver.domain.validation import GatePolicy
from agent_skill_improver.errors import (
    BaseHashMismatch,
    CandidateTampered,
    StateUnreadable,
)
from agent_skill_improver.hashing import sha256_file, sha256_text

from tests.support import FIX_MARKER, build_env, run_cli

RC_OK = 0
RC_REFUSED = 3


def frozen_clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 2, 1, tzinfo=timezone.utc))


TAMPER_MARKER = "- inserted after the review, never seen by a human"


# -- the end-to-end regression -------------------------------------------
class TamperedApplyIsRefusedEndToEnd(unittest.TestCase):
    """Ordered: review, tamper, refuse, restore, apply. One real candidate."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.base = Path(cls._tmp.name)
        cls.config = build_env(cls.base, dry_run=False, seed="integrity-seed")
        cls.cfg = ["--config", str(cls.config)]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def cli(self, *args):
        return run_cli([*self.cfg, *args])

    def cli_json(self, *args):
        proc = self.cli(*args)
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        return json.loads(proc.stdout)

    def skill(self) -> Path:
        return self.base / "skills-canonical" / "tdd" / "SKILL.md"

    def record(self) -> Path:
        return self.base / "state" / "candidates" / f"{self.candidate_id}.json"

    def backups(self) -> list[str]:
        root = self.base / "state" / "backups"
        return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.is_dir() else []

    def receipts(self) -> list[str]:
        root = self.base / "state" / "receipts"
        return sorted(p.name for p in root.glob("*.json")) if root.is_dir() else []

    def test_01_a_daily_pass_produces_a_validated_candidate_and_a_token(self):
        payload = self.cli_json("--json", "daily", "--commit", "--day", "2026-02-01")
        validated = [
            c["candidate_id"]
            for c in payload.get("candidates", [])
            if c.get("status") == STATUS_VALIDATED
        ]
        self.assertTrue(validated, payload.get("notes"))
        type(self).candidate_id = validated[0]
        review = self.cli_json("--json", "review", self.candidate_id)
        self.assertTrue(review["approval_token"])
        type(self).token = review["approval_token"]
        type(self).clean_record = self.record().read_text(encoding="utf-8")
        type(self).skill_sha_before = sha256_file(self.skill())
        type(self).backups_before = self.backups()
        type(self).receipts_before = self.receipts()

    def test_02_tampering_changes_only_new_text_and_nothing_derived(self):
        raw = json.loads(self.clean_record)
        before = dict(raw["patch"])
        raw["patch"]["new_text"] = raw["patch"]["new_text"] + f"\n{TAMPER_MARKER}\n"
        self.record().write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        after = json.loads(self.record().read_text(encoding="utf-8"))
        self.assertEqual(after["patch"]["new_text_sha256"], before["new_text_sha256"])
        self.assertEqual(after["patch"]["unified_diff"], before["unified_diff"])
        self.assertEqual(after["patch"]["added_lines"], before["added_lines"])
        self.assertEqual(
            after["approval_challenge"], json.loads(self.clean_record)["approval_challenge"]
        )
        self.assertEqual(after["status"], STATUS_VALIDATED)
        self.assertTrue(after["validation"]["gate"]["passed"])

    def test_03_review_refuses_and_prints_no_token_for_a_tampered_record(self):
        proc = self.cli("--json", "review", self.candidate_id)
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout)
        self.assertIn("CandidateTampered", proc.stderr)
        self.assertNotIn(self.token, proc.stdout)
        self.assertNotIn("Traceback", proc.stderr)

    def test_04_apply_with_the_original_token_is_refused(self):
        proc = self.cli(
            "apply",
            self.candidate_id,
            "--approve",
            self.token,
            "--i-reviewed-this",
            "--operator",
            "tester",
        )
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stdout)
        self.assertIn("CandidateTampered", proc.stderr)
        self.assertIn("new_text_sha256", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_05_the_skill_file_is_byte_identical_after_the_refusal(self):
        text = self.skill().read_text(encoding="utf-8")
        self.assertNotIn(TAMPER_MARKER, text)
        self.assertNotIn(FIX_MARKER, text)
        self.assertEqual(sha256_file(self.skill()), self.skill_sha_before)

    def test_06_no_backup_and_no_receipt_were_written_by_the_refusal(self):
        self.assertEqual(
            self.backups(),
            self.backups_before,
            "a refusal must happen before the backup, or a failed apply leaves debris",
        )
        self.assertEqual(self.receipts(), self.receipts_before)
        self.assertFalse(
            (self.base / "state" / "receipts" / f"apply--{self.candidate_id}.json").exists()
        )

    def test_07_the_candidate_status_was_not_advanced_by_the_refusal(self):
        raw = json.loads(self.record().read_text(encoding="utf-8"))
        self.assertEqual(raw["status"], STATUS_VALIDATED)
        self.assertNotIn("applied", " ".join(raw["notes"]))

    def test_08_control_the_untampered_record_reviews_to_the_same_token(self):
        self.record().write_text(self.clean_record, encoding="utf-8")
        review = self.cli_json("--json", "review", self.candidate_id)
        self.assertEqual(review["approval_token"], self.token)

    def test_09_control_apply_still_succeeds_on_the_record_that_was_reviewed(self):
        proc = self.cli(
            "apply",
            self.candidate_id,
            "--approve",
            self.token,
            "--i-reviewed-this",
            "--operator",
            "tester",
        )
        self.assertEqual(proc.returncode, RC_OK, proc.stderr)
        text = self.skill().read_text(encoding="utf-8")
        self.assertIn(FIX_MARKER, text, "the interlock must not have broken the healthy path")
        self.assertNotIn(TAMPER_MARKER, text)

    def test_10_control_the_daily_path_is_still_proposal_only(self):
        payload = self.cli_json("--json", "daily", "--commit", "--day", "2026-02-02")
        tree = payload["skill_tree"]
        self.assertEqual(tree["digest_before"], tree["digest_after"])
        self.assertFalse(tree["mutation_detected"])


def load_tests(loader, tests, pattern):  # noqa: D103 - unittest hook
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TamperedApplyIsRefusedEndToEnd))
    for case in (
        DerivedPatchFieldTests,
        TargetIdentityTests,
        GateStatusTests,
        RefusalWrapperTests,
        CallSiteTests,
        StateReadTests,
    ):
        suite.addTests(loader.loadTestsFromTestCase(case))
    return suite


# -- the pure half -------------------------------------------------------
BASE_TEXT = """---
name: tdd
description: Test-driven development.
---

# Test-driven development

- Write the failing test before the fix.
"""

NEW_TEXT = BASE_TEXT + "- Run the test and watch it fail for the right reason.\n"

CANONICAL = "skills-canonical/tdd/SKILL.md"

GATE_PASSED = {"gate": {"passed": True, "reasons": [], "metrics": {}}}


def build_candidate(
    *,
    base_text: str = BASE_TEXT,
    new_text: str = NEW_TEXT,
    candidate_id: str = "CAND-2026-02-01-01-tdd",
    status: str = STATUS_VALIDATED,
    validation: dict[str, Any] | None = None,
    canonical_path: str = CANONICAL,
    skill_id: str = "tdd",
    mirror_paths: tuple[str, ...] = ("skills-mirror/tdd/SKILL.md",),
) -> Candidate:
    """A candidate whose every derived field really is derived."""
    patch = make_patch(base_text, new_text, path_label=canonical_path)
    base_sha = sha256_text(base_text)
    return Candidate(
        id=candidate_id,
        created_at="2026-02-01T00:00:00Z",
        hypothesis="The skill never tells the agent to watch the test fail for the right reason.",
        intended_change="Add a line requiring the failing test to be observed before the fix.",
        risks=("The added line could be read as requiring a manual run.",),
        rollback="Restore the previous SKILL.md from the backup this apply takes.",
        target=TargetRef(
            skill_id=skill_id,
            root_id="canonical",
            canonical_path=canonical_path,
            base_sha256=base_sha,
            mirror_paths=mirror_paths,
        ),
        rubric_failures=(
            RubricFailureRef(
                failure_case_id="FC-1",
                rubric_id="instruction-compliance",
                rubric_version=1,
                criterion_id="c1",
                conversation_id="t1",
            ),
        ),
        sources=(SourceRef(thread_id="t1", element_id="m1", position=0, role="user"),),
        patch=patch,
        approval_challenge=challenge_for(candidate_id, base_sha, patch.new_text_sha256),
        status=status,
        validation=GATE_PASSED if validation is None else validation,
    )


class DerivedPatchFieldTests(unittest.TestCase):
    """One derived patch field per test, each with the clean control beside it."""

    def setUp(self) -> None:
        self.candidate = build_candidate()

    def test_a_clean_candidate_reports_no_mismatch_with_the_live_base(self):
        self.assertEqual(derived_field_mismatches(self.candidate, base_text=BASE_TEXT), [])

    def test_a_clean_candidate_reports_no_mismatch_without_a_base_text(self):
        self.assertEqual(derived_field_mismatches(self.candidate), [])

    def test_editing_new_text_alone_is_detected_although_every_other_field_is_intact(self):
        tampered = replace(
            self.candidate,
            patch=replace(self.candidate.patch, new_text=NEW_TEXT + "- smuggled\n"),
        )
        problems = derived_field_mismatches(tampered, base_text=BASE_TEXT)
        self.assertTrue(any("new_text_sha256" in p for p in problems), problems)

    def test_the_token_of_a_record_whose_text_changed_is_not_the_reviewed_token(self):
        reviewed = self.candidate.required_approval_token
        tampered = replace(
            self.candidate,
            patch=replace(self.candidate.patch, new_text=NEW_TEXT + "- smuggled\n"),
        )
        self.assertNotEqual(
            reviewed,
            tampered.required_approval_token,
            "the token must be a function of the bytes, not of a field beside them",
        )

    def test_a_stored_patch_hash_that_is_not_the_hash_of_the_text_is_detected(self):
        tampered = replace(
            self.candidate, patch=replace(self.candidate.patch, new_text_sha256="0" * 64)
        )
        self.assertTrue(
            any("new_text_sha256" in p for p in derived_field_mismatches(tampered)),
        )

    def test_a_rewritten_approval_challenge_is_detected(self):
        tampered = replace(self.candidate, approval_challenge="f" * 24)
        self.assertTrue(
            any("approval_challenge" in p for p in derived_field_mismatches(tampered)),
        )

    def test_an_edited_unified_diff_is_detected_against_the_live_base(self):
        tampered = replace(
            self.candidate,
            patch=replace(self.candidate.patch, unified_diff="--- a/x\n+++ b/x\n+nothing\n"),
        )
        problems = derived_field_mismatches(tampered, base_text=BASE_TEXT)
        self.assertTrue(any("unified_diff" in p for p in problems), problems)

    def test_an_edited_diff_is_invisible_without_the_base_which_is_why_apply_reads_it(self):
        tampered = replace(
            self.candidate,
            patch=replace(self.candidate.patch, unified_diff="--- a/x\n+++ b/x\n+nothing\n"),
        )
        self.assertEqual(
            derived_field_mismatches(tampered),
            [],
            "the diff can only be checked against the base it claims to describe",
        )

    def test_a_wrong_added_line_count_is_detected(self):
        tampered = replace(self.candidate, patch=replace(self.candidate.patch, added_lines=99))
        problems = derived_field_mismatches(tampered, base_text=BASE_TEXT)
        self.assertTrue(any("added_lines" in p for p in problems), problems)

    def test_a_wrong_removed_line_count_is_detected(self):
        tampered = replace(self.candidate, patch=replace(self.candidate.patch, removed_lines=7))
        problems = derived_field_mismatches(tampered, base_text=BASE_TEXT)
        self.assertTrue(any("removed_lines" in p for p in problems), problems)

    def test_a_base_hash_that_does_not_describe_the_live_base_is_detected(self):
        problems = derived_field_mismatches(
            self.candidate, base_text=BASE_TEXT + "- edited by hand\n"
        )
        self.assertTrue(any("base_sha256" in p for p in problems), problems)


class TargetIdentityTests(unittest.TestCase):
    def test_a_healthy_target_reports_nothing(self):
        self.assertEqual(target_identity_problems(build_candidate().target), [])

    def test_a_skill_id_that_disagrees_with_the_path_is_reported(self):
        target = replace(build_candidate().target, skill_id="code-review")
        self.assertTrue(any("two different skills" in p for p in target_identity_problems(target)))

    def test_a_path_that_is_not_the_skill_document_is_reported(self):
        target = replace(build_candidate().target, canonical_path="skills-canonical/tdd/hook.sh")
        self.assertTrue(any("SKILL.md" in p for p in target_identity_problems(target)))

    def test_an_empty_canonical_path_is_reported_and_stops_further_path_claims(self):
        problems = target_identity_problems(replace(build_candidate().target, canonical_path=""))
        self.assertEqual(problems, ["target.canonical_path is empty"])

    def test_a_base_hash_that_is_not_a_sha256_digest_is_reported(self):
        target = replace(build_candidate().target, base_sha256="not-a-hash")
        self.assertTrue(any("sha256" in p for p in target_identity_problems(target)))

    def test_a_canonical_path_that_is_also_a_mirror_is_reported(self):
        target = replace(build_candidate().target, mirror_paths=(CANONICAL,))
        self.assertTrue(any("mirror" in p for p in target_identity_problems(target)))

    def test_an_empty_root_id_is_reported(self):
        target = replace(build_candidate().target, root_id="")
        self.assertIn("target.root_id is empty", target_identity_problems(target))

    def test_target_problems_reach_the_whole_recomputation(self):
        candidate = build_candidate(skill_id="code-review")
        self.assertTrue(
            any("two different skills" in p for p in derived_field_mismatches(candidate))
        )


class GateStatusTests(unittest.TestCase):
    def test_a_validated_candidate_with_a_passing_gate_is_consistent(self):
        self.assertEqual(gate_status_problems(build_candidate()), [])

    def test_a_proposed_candidate_with_no_validation_is_consistent(self):
        candidate = build_candidate(status=STATUS_PROPOSED, validation={})
        self.assertEqual(gate_status_problems(replace(candidate, validation=None)), [])

    def test_a_validated_status_over_a_failing_gate_is_reported(self):
        candidate = build_candidate(
            validation={"gate": {"passed": False, "reasons": ["no holdout"]}}
        )
        self.assertTrue(any("did not pass" in p for p in gate_status_problems(candidate)))

    def test_a_validated_status_with_no_gate_recorded_is_reported(self):
        candidate = replace(build_candidate(), validation=None)
        self.assertTrue(any("no gate decision" in p for p in gate_status_problems(candidate)))

    def test_a_gate_failed_status_over_a_passing_gate_is_reported(self):
        candidate = build_candidate(status=STATUS_GATE_FAILED)
        self.assertTrue(any("gate passed" in p for p in gate_status_problems(candidate)))

    def test_a_proposed_status_carrying_a_gate_decision_is_reported(self):
        candidate = build_candidate(status=STATUS_PROPOSED)
        self.assertTrue(any("already recorded" in p for p in gate_status_problems(candidate)))

    def test_an_applied_candidate_stays_consistent_so_rollback_is_not_blocked(self):
        self.assertEqual(gate_status_problems(build_candidate(status=STATUS_APPLIED)), [])


class RefusalWrapperTests(unittest.TestCase):
    def test_a_healthy_candidate_passes_verification_silently(self):
        verify_candidate_integrity(build_candidate(), base_text=BASE_TEXT, action="apply")

    def test_a_tampered_candidate_raises_and_names_the_field_and_the_action(self):
        tampered = replace(
            build_candidate(),
            patch=replace(build_candidate().patch, new_text=NEW_TEXT + "- smuggled\n"),
        )
        with self.assertRaises(CandidateTampered) as ctx:
            verify_candidate_integrity(tampered, action="apply")
        message = str(ctx.exception)
        self.assertIn("refusing to apply", message)
        self.assertIn("new_text_sha256", message)
        self.assertIn("asi daily --commit", message)


class RecordingStore:
    """A skill store that records whether the write path was ever reached."""

    read_only = False

    def __init__(self, base_text: str = BASE_TEXT) -> None:
        self.base_text = base_text
        self.writes: list[str] = []

    def read_text(self, path: Any) -> str:
        return self.base_text

    def write_canonical(self, path: Any, text: str, **kwargs: Any) -> Any:  # pragma: no cover
        self.writes.append(str(path))
        raise AssertionError("write_canonical must not be reached by a refused apply")


class CallSiteTests(unittest.TestCase):
    """Review, apply and validation each recompute before they act."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name) / "state"
        self.state = StateStore(self.root, clock=frozen_clock())
        self.state.ensure()
        self.clean = build_candidate()
        self.tampered = replace(
            self.clean, patch=replace(self.clean.patch, new_text=NEW_TEXT + "- smuggled\n")
        )
        self.approval = HumanApproval(
            token=self.clean.required_approval_token, acknowledged=True, origin="cli"
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_review_returns_a_report_for_a_clean_candidate(self):
        self.assertEqual(review_report(self.clean)["candidate_id"], self.clean.id)

    def test_review_refuses_a_tampered_candidate(self):
        with self.assertRaises(CandidateTampered):
            review_report(self.tampered)

    def test_apply_refuses_a_tampered_candidate_before_the_store_is_written(self):
        store = RecordingStore()
        with self.assertRaises(CandidateTampered):
            apply_candidate(
                self.tampered,
                approval=self.approval,
                skill_store=store,
                state=self.state,
                clock=frozen_clock(),
            )
        self.assertEqual(store.writes, [])

    def test_apply_creates_no_backup_directory_when_it_refuses(self):
        store = RecordingStore()
        with self.assertRaises(CandidateTampered):
            apply_candidate(
                self.tampered,
                approval=self.approval,
                skill_store=store,
                state=self.state,
                clock=frozen_clock(),
            )
        self.assertEqual(list((self.root / "backups").iterdir()), [])
        self.assertEqual(list((self.root / "receipts").glob("apply--*.json")), [])

    def test_apply_refuses_a_live_base_that_moved_under_the_candidate(self):
        store = RecordingStore(base_text=BASE_TEXT + "- edited by hand\n")
        with self.assertRaises(BaseHashMismatch):
            apply_candidate(
                self.clean,
                approval=self.approval,
                skill_store=store,
                state=self.state,
                clock=frozen_clock(),
            )
        self.assertEqual(store.writes, [])

    def test_apply_refuses_a_canonical_target_it_cannot_read(self):
        class Unreadable(RecordingStore):
            def read_text(self, path: Any) -> str:
                raise FileNotFoundError(str(path))

        store = Unreadable()
        with self.assertRaises(BaseHashMismatch):
            apply_candidate(
                self.clean,
                approval=self.approval,
                skill_store=store,
                state=self.state,
                clock=frozen_clock(),
            )
        self.assertEqual(store.writes, [])

    def test_validation_refuses_a_tampered_candidate_before_any_model_call(self):
        calls: list[str] = []

        class Runner:
            def command_id(self, role: str) -> str:  # pragma: no cover - never reached
                return role

            def run(self, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover
                calls.append("run")
                raise AssertionError("validation must refuse before calling a model")

        with self.assertRaises(CandidateTampered):
            validate_candidate(
                self.tampered,
                Runner(),
                rubrics={},
                conversations={},
                base_text=BASE_TEXT,
                cfg=None,
                policy=GatePolicy(),
                seed="s",
            )
        self.assertEqual(calls, [])


class StateReadTests(unittest.TestCase):
    """A state read fails closed with a reason, never with a traceback."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.state = StateStore(Path(self._tmp.name) / "state")
        self.state.ensure()
        self.candidate = build_candidate()
        self.path = self.state.write_candidate(self.candidate)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_healthy_record_reads_back_by_its_own_id(self):
        self.assertEqual(self.state.read_candidate(self.candidate.id).id, self.candidate.id)

    def test_a_record_that_claims_a_different_id_is_refused(self):
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw["id"] = "CAND-2026-02-01-99-other"
        self.path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(StateUnreadable) as ctx:
            self.state.read_candidate(self.candidate.id)
        self.assertIn("declares id", str(ctx.exception))

    def test_a_truncated_record_is_a_named_refusal_not_a_json_error(self):
        self.path.write_text('{"schema": "asi.candi', encoding="utf-8")
        with self.assertRaises(StateUnreadable) as ctx:
            self.state.read_candidate(self.candidate.id)
        self.assertIn("could not be read", str(ctx.exception))

    def test_a_record_of_the_wrong_schema_is_a_named_refusal(self):
        self.path.write_text('{"schema": "asi.receipt.v1"}', encoding="utf-8")
        with self.assertRaises(StateUnreadable):
            self.state.read_candidate(self.candidate.id)

    def test_a_record_that_is_not_an_object_is_a_named_refusal(self):
        self.path.write_text("[]", encoding="utf-8")
        with self.assertRaises(StateUnreadable) as ctx:
            self.state.read_candidate(self.candidate.id)
        self.assertIn("not a JSON object", str(ctx.exception))

    def test_a_record_missing_a_required_field_is_a_named_refusal(self):
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        del raw["target"]
        self.path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(StateUnreadable) as ctx:
            self.state.read_candidate(self.candidate.id)
        self.assertIn("shape of a candidate", str(ctx.exception))

    def test_an_absent_candidate_is_still_a_file_not_found_rather_than_a_refusal(self):
        with self.assertRaises(FileNotFoundError):
            self.state.read_candidate("CAND-does-not-exist")

    def test_listing_skips_an_unreadable_record_rather_than_failing_the_command(self):
        broken = self.state.dir("candidates") / "broken.json"
        broken.write_text("{", encoding="utf-8")
        ids = [c.id for c in self.state.list_candidates()]
        self.assertEqual(ids, [self.candidate.id])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
