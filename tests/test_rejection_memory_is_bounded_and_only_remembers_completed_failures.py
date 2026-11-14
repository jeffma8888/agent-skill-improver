"""Only stable decided replacement losses teach optional bounded fingerprints."""

from __future__ import annotations

import copy
import json
import os
import shutil
import stat
import unittest
from dataclasses import replace
from unittest.mock import patch

from agent_skill_improver.app import daily, rejections
from agent_skill_improver.app.state import SUBDIRS, StateStore
from agent_skill_improver.app.wiring import build_runner
from agent_skill_improver.config import load_config
from agent_skill_improver.domain.candidate import Candidate, challenge_for
from agent_skill_improver.domain.integrity import verify_full_integrity
from agent_skill_improver.domain.rubric import Verdict
from agent_skill_improver.domain.validation import (
    ARM_BASELINE,
    ARM_CANDIDATE,
    CASE_HOLDOUT,
    CASE_TARGET,
    ValidationRecord,
    blind_arms,
    evaluate_gate,
    evidence_call_count,
    runs_the_same_command,
)
from agent_skill_improver.errors import SafetyRefusal, TransientFailure
from agent_skill_improver.jsonio import dumps
from tests import fake_model
from tests.support import FIXTURE_EXPORTS
from tests.test_a_damaged_rejection_memory_does_not_block_a_proposal import (
    _MemoryHarness,
    memory_payload,
    sample_entry,
)
from tests.test_a_rejected_replacement_is_not_replayed_again import _RoleOverride, role_counts


def rebuilt(result, **changes):
    """Typed evidence injection, not forged metrics: every derived field recomputed."""
    result = replace(result, **changes)
    independent = bool(result.judge_command_id) and not runs_the_same_command(
        result.judge_command_id, result.improver_command_id
    )
    result = replace(
        result,
        decision=evaluate_gate(
            result.outcomes,
            result.judgements,
            result.decision.policy,
            judge_independent=independent,
            target_refs=result.target_refs,
        ),
        evidence_model_calls=evidence_call_count(
            result.outcomes + result.unseen_outcomes, result.judgements
        ),
    )
    record = ValidationRecord.parse(result.to_dict())
    assert not record.recompute_gate_mismatches()
    return result


def preferred(result, arms):
    judgements = []
    for judgement, arm in zip(result.judgements, arms):
        labels = blind_arms(judgement.case.key, result.seed)
        label = "tie" if arm == "tie" else next(k for k, v in labels.items() if v == arm)
        judgements.append(replace(judgement, preferred_label=label, preferred_arm=arm))
    return tuple(judgements)


class RejectionMemoryIsBoundedAndOnlyRemembersCompletedFailuresTests(_MemoryHarness):
    def test_an_incomplete_or_configuration_failure_is_not_remembered(self):
        self.seed()  # fully replayed FAIL->FAIL, all decided holdouts, all-tie positive
        positive = Candidate.from_dict(json.loads(self.candidates()[0].read_text()))
        positive_record = ValidationRecord.parse(positive.validation)
        self.assertEqual(positive_record.notes, ())
        cfg = load_config(self.config)
        rubrics = {r.id: r for r in cfg.rubrics}
        runner = build_runner(cfg)
        inactive_n = replace(
            cfg, caps=replace(cfg.caps, unseen_min_failures=cfg.caps.unseen_min_failures + 1)
        )
        self.assertEqual(cfg.caps.unseen_withhold, 0)
        self.assertNotIn("unseen_min_failures", cfg.caps.to_dict())
        self.assertNotIn("unseen_withhold", cfg.caps.to_dict())
        self.assertEqual(
            rejections.evaluation_context_sha256(cfg, rubrics, runner),
            rejections.evaluation_context_sha256(inactive_n, rubrics, runner),
        )
        enabled = replace(cfg, caps=replace(cfg.caps, unseen_withhold=1))
        self.assertIn("unseen_min_failures", enabled.caps.to_dict())
        self.assertIn("unseen_withhold", enabled.caps.to_dict())
        self.assertNotEqual(
            rejections.evaluation_context_sha256(enabled, rubrics, runner),
            rejections.evaluation_context_sha256(
                replace(enabled, caps=replace(enabled.caps, unseen_min_failures=5)), rubrics, runner
            ),
        )
        self.assertTrue(
            all(
                o.baseline is Verdict.FAIL and o.candidate is Verdict.FAIL
                for o in positive_record.outcomes
                if o.case.kind == CASE_TARGET
            )
        )
        self.assertTrue(
            all(o.baseline.is_decided and o.candidate.is_decided for o in positive_record.outcomes)
        )
        self.assertEqual(len(positive_record.judgements), 2)
        self.assertTrue(all(j.preferred_arm == "tie" for j in positive_record.judgements))
        modes = [
            ("target-timeout", "scorer", "replay-timeout-exit"),
            ("judge-exit69", "judge", "timeout_exit"),
            ("judge-malformed", "judge", "malformed_json"),
            ("extra-holdout-timeout", "scorer", "replay-holdout-timeout-exit"),
            ("baseline-target-partial", "scorer", "replay-baseline-target-partial"),
            ("baseline-partial", "scorer", "replay-baseline-partial"),
            ("candidate-partial", "scorer", "replay-candidate-partial"),
            ("candidate-target-bad-ref", "scorer", "replay-candidate-target-unresolvable-ref"),
            ("candidate-holdout-partial", "scorer", "replay-candidate-holdout-partial"),
        ]
        for name, role, mode in modes:
            with self.subTest(real_role_mode=name):
                self.fresh()
                self.mode(role, mode)
                receipt, calls = self.invoke()
                self.assertEqual(len(self.candidates()), 1)
                candidate = Candidate.from_dict(json.loads(self.candidates()[0].read_text()))
                record = ValidationRecord.parse(candidate.validation)
                self.assertEqual(candidate.status, "gate-failed")
                self.assertFalse(self.memory.exists(), receipt)
                targets = [o for o in record.outcomes if o.case.kind == CASE_TARGET]
                holdouts = [o for o in record.outcomes if o.case.kind == CASE_HOLDOUT]
                if "partial" in mode or "unresolvable" in mode:
                    self.assertEqual(record.notes, ())
                    self.assertEqual(len(targets), len(record.target_refs))
                    self.assertEqual(len(record.judgements), len(targets))
                    if mode == "replay-baseline-target-partial":
                        self.assertTrue(
                            all(
                                o.baseline is Verdict.INSUFFICIENT and o.candidate is Verdict.FAIL
                                for o in targets
                            )
                        )
                        self.assertTrue(
                            all(o.baseline.is_decided and o.candidate.is_decided for o in holdouts)
                        )
                    elif mode == "replay-candidate-target-unresolvable-ref":
                        self.assertTrue(
                            all(
                                o.baseline is Verdict.FAIL and o.candidate is Verdict.INSUFFICIENT
                                for o in targets
                            )
                        )
                        self.assertTrue(
                            all(o.baseline.is_decided and o.candidate.is_decided for o in holdouts)
                        )
                    elif mode == "replay-candidate-holdout-partial":
                        self.assertTrue(
                            all(
                                o.baseline is Verdict.FAIL and o.candidate is Verdict.FAIL
                                for o in targets
                            )
                        )
                        self.assertTrue(any(o.candidate is Verdict.INSUFFICIENT for o in holdouts))
                    else:
                        self.assertTrue(
                            any(
                                not o.baseline.is_decided or not o.candidate.is_decided
                                for o in record.outcomes
                            )
                        )
                else:
                    self.assertTrue(record.notes)
                if name == "extra-holdout-timeout":
                    self.assertEqual(len(targets), len(record.target_refs))
                    self.assertEqual(len(record.judgements), len(targets))
                    self.assertGreaterEqual(len(holdouts), record.policy.min_holdout_cases)
                    self.assertTrue(
                        all(
                            o.baseline.is_decided and o.candidate.is_decided
                            for o in record.outcomes
                        )
                    )
                self.assertGreater(role_counts(calls)["replay_scorer"], 0)
                self.mode(role, "repeat-no-fix")
                healthy, later = self.invoke("2026-02-02")
                self.assertEqual(healthy["counts"]["replacements_refused_as_repeats"], 0)
                self.assertGreater(role_counts(later)["replay_scorer"], 0)
                self.assertTrue(self.memory.is_file(), healthy)

        # Schema-valid JSON with no preferred verdict is not an all-tie
        # measurement: the real judgement parser records notes and no judges.
        self.fresh()

        def no_verdict(kwargs, payload):
            if kwargs["role"] == "judge":
                payload.pop("preferred")
            return payload

        receipt, _ = self.invoke(transform=no_verdict)
        missing_judge = Candidate.from_dict(json.loads(self.candidates()[0].read_text()))
        missing_record = ValidationRecord.parse(missing_judge.validation)
        self.assertEqual(missing_judge.status, "gate-failed")
        self.assertEqual(missing_record.judgements, ())
        self.assertTrue(missing_record.notes)
        self.assertFalse(self.memory.exists(), receipt)
        healthy, calls = self.invoke("2026-02-02")
        self.assertEqual(healthy["counts"]["replacements_refused_as_repeats"], 0)
        self.assertGreater(role_counts(calls)["replay_scorer"], 0)
        self.assertTrue(self.memory.is_file())

        real_validate = daily.validate_candidate
        scenarios = {
            "coverage-only": False,
            "holdout-size-only": False,
            "independence-only": False,
            "degradation-only": False,
            "mixed-size-and-loss": False,
            "mixed-independence-and-loss": False,
            "mixed-degradation-and-loss": False,
            "regression-only": True,
            "ratio-below-actual-boundary": True,
            "ratio-at-boundary-passes": False,
        }
        for name, eligible in scenarios.items():
            with self.subTest(parsed_evidence=name):
                self.fresh(min_holdout_cases=2 if "size" in name else 1)
                seen = []

                def inject(*args, **kwargs):
                    result = real_validate(*args, **kwargs)
                    outcomes = [
                        replace(o, candidate=Verdict.PASS) if o.case.kind == CASE_TARGET else o
                        for o in result.outcomes
                    ]
                    judges = preferred(result, [ARM_CANDIDATE] * len(result.judgements))
                    changes = {"outcomes": tuple(outcomes), "judgements": judges, "notes": ()}
                    if name.startswith("mixed-"):
                        changes = {"notes": ()}
                        outcomes = list(result.outcomes)
                    if "coverage" in name:
                        missing = next(o for o in outcomes if o.case.kind == CASE_TARGET)
                        outcomes.remove(missing)
                        changes["judgements"] = tuple(
                            j for j in judges if j.case.key != missing.case.key
                        )
                    if "size" in name:
                        outcomes = [o for o in outcomes if o.case.kind == CASE_TARGET]
                    if "independence" in name:
                        changes["judge_command_id"] = result.improver_command_id
                    if "degradation" in name or "regression" in name:
                        index = next(
                            i for i, o in enumerate(outcomes) if o.case.kind == CASE_HOLDOUT
                        )
                        outcomes[index] = replace(
                            outcomes[index],
                            candidate=(
                                Verdict.FAIL if "regression" in name else Verdict.INSUFFICIENT
                            ),
                        )
                    if "ratio" in name:
                        self.assertEqual(len(judges), 2)
                        changes["judgements"] = preferred(result, [ARM_CANDIDATE, ARM_BASELINE])
                        changes["decision"] = replace(
                            result.decision,
                            policy=replace(
                                result.decision.policy,
                                min_judge_preference_ratio=0.50001 if "below" in name else 0.5,
                            ),
                        )
                    changes["outcomes"] = tuple(outcomes)
                    altered = rebuilt(result, **changes)
                    seen.append(altered)
                    return altered

                def configure(active):
                    active.enter_context(
                        patch.object(daily, "validate_candidate", side_effect=inject)
                    )

                receipt, _ = self.invoke(stack=configure)
                self.assertEqual(self.memory.exists(), eligible, receipt)
                result = seen[0]
                candidate = Candidate.from_dict(json.loads(self.candidates()[0].read_text()))
                cfg = load_config(self.config)
                verify_full_integrity(
                    candidate,
                    base_text=self.skill.read_text(),
                    rubrics={r.id: r for r in cfg.rubrics},
                    action="test evidence",
                )
                self.assertEqual(result.notes, ())
                self.assertFalse(any("integrity was not verified" in n for n in receipt["notes"]))
                if name == "coverage-only":
                    self.assertEqual(result.decision.metrics["target_refs_replayed"], 1)
                    self.assertEqual(result.decision.metrics["fixes"], 1)
                    self.assertEqual(len(result.decision.reasons), 1)
                if name == "regression-only":
                    self.assertEqual(result.decision.metrics["regressions"], 1)
                    self.assertEqual(len(result.decision.reasons), 1)
                if "ratio" in name:
                    self.assertEqual(
                        result.decision.metrics["judge_candidate_preference_ratio"], 0.5
                    )
                    self.assertEqual(result.decision.passed, name == "ratio-at-boundary-passes")
                if not eligible and not result.decision.passed:
                    later, calls = self.invoke("2026-02-02")
                    self.assertEqual(later["counts"]["replacements_refused_as_repeats"], 0)
                    self.assertGreater(role_counts(calls)["replay_scorer"], 0)

        # A real measured c240 split with two failing conversations; only unseen
        # evidence is varied afterwards. It never changes the actual gate.
        exports = self.base / "unseen-exports"
        shutil.copytree(FIXTURE_EXPORTS, exports)
        extra = json.loads((exports / "failing.json").read_text())
        extra["thread"]["id"] = "1700000000006-1"
        (exports / "second-failing.json").write_text(json.dumps(extra))
        decided_gate = None
        for state in ("decided", "insufficient", "replay-error"):
            with self.subTest(unseen=state):
                self.fresh(fixtures_dir=exports)
                body = self.config.read_text()
                self.config.write_text(
                    body.replace(
                        "max_open_candidates = 12",
                        "max_open_candidates = 12\nunseen_withhold = 1\nunseen_min_failures = 3",
                    )
                )

                def extra_failure(kwargs, payload):
                    if (
                        kwargs["role"] == "scorer"
                        and fake_model._scored_thread(kwargs["user"]) == "1700000000006-1"
                    ):
                        for row in payload["criteria"]:
                            if row["criterion_id"] in {
                                "explicit-directive-followed",
                                "correction-not-repeated",
                            }:
                                row["verdict"] = "FAIL"
                    return payload

                seen = []

                def unseen_inject(candidate, runner, **kwargs):
                    if state == "replay-error":
                        withheld_threads = {
                            ref.conversation_id for ref in kwargs["unseen"].withheld
                        }

                        def fault(call, payload):
                            if (
                                call["role"] == "scorer"
                                and "## SKILL TEXT UNDER EVALUATION" in call["user"]
                                and fake_model._scored_thread(call["user"]) in withheld_threads
                            ):
                                raise TransientFailure("fixture unseen replay failure")
                            return payload

                        runner = _RoleOverride(runner, fault)
                    result = real_validate(candidate, runner, **kwargs)
                    if state == "insufficient":
                        self.assertTrue(result.unseen_outcomes)
                        result = rebuilt(
                            result,
                            unseen_outcomes=tuple(
                                replace(o, candidate=Verdict.INSUFFICIENT)
                                for o in result.unseen_outcomes
                            ),
                        )
                    seen.append(result)
                    return result

                def configure(active):
                    active.enter_context(
                        patch.object(daily, "validate_candidate", side_effect=unseen_inject)
                    )

                receipt, _ = self.invoke(stack=configure, transform=extra_failure)
                candidate = Candidate.from_dict(json.loads(self.candidates()[0].read_text()))
                record = ValidationRecord.parse(candidate.validation)
                self.assertEqual(record.unseen.status, "measured")
                self.assertGreater(len(record.unseen.withheld), 0)
                self.assertEqual(record.gate_raw["metrics"]["fixes"], 0)
                self.assertEqual(
                    record.gate_raw["metrics"]["target_refs_claimed"],
                    record.gate_raw["metrics"]["target_refs_replayed"],
                )
                self.assertEqual(self.memory.exists(), state == "decided", receipt)
                if state == "decided":
                    decided_gate = record.gate_raw
                    self.assertEqual(len(json.loads(self.memory.read_text())["entries"]), 1)
                else:
                    # The gate is unchanged even when the optional cost policy
                    # refuses to learn from incomplete unseen measurements.
                    self.assertEqual(record.gate_raw, decided_gate)
                if state == "replay-error":
                    self.assertTrue(any("unseen" in n and "failed" in n for n in record.notes))
                else:
                    self.assertEqual(record.notes, ())
                    self.assertTrue(
                        all(
                            o.baseline.is_decided and o.candidate.is_decided
                            for o in record.outcomes
                        )
                    )
                    self.assertTrue(
                        all(
                            o.candidate
                            is (Verdict.INSUFFICIENT if state == "insufficient" else Verdict.FAIL)
                            for o in record.unseen.outcomes
                        )
                    )

    def test_memory_keeps_the_newest_sixty_four_distinct_entries(self):
        entries = [replace(sample_entry(i), rejected_at="2000-01-01T00:00:00Z") for i in range(64)]
        self.write_memory(entries)
        # A real preexisting closed candidate is not deleted when its separate
        # fingerprint is evicted; memory and candidate retention are independent.
        original_env = self.env
        original_paths = (self.config, self.log, self.state, self.memory, self.skill)
        self.fresh()
        historical = self.seed()
        historical_bytes = self.candidates()[0].read_bytes()
        self.env = original_env
        self.config, self.log, self.state, self.memory, self.skill = original_paths
        old_path = self.state / "candidates" / f"{historical.id}.json"
        old_path.parent.mkdir(parents=True)
        old_raw = json.loads(historical_bytes)
        # The old closed record has a distinct valid id; recomputed provenance
        # remains in the retained source artifact, which daily never rewrites.
        old_raw["id"] = "CAND-2026-01-01-01-tdd"
        old_raw["approval_challenge"] = challenge_for(
            old_raw["id"], old_raw["target"]["base_sha256"], historical.recomputed_patch_sha256
        )
        old_raw["validation"]["candidate_id"] = old_raw["id"]
        old_path = old_path.with_name(old_raw["id"] + ".json")
        old_path.write_text(dumps(old_raw))
        historical_bytes = old_path.read_bytes()
        candidate = self.seed()
        self.assertEqual(old_path.read_bytes(), historical_bytes)
        raw = json.loads(self.memory.read_text())
        self.assertEqual(len(raw["entries"]), 64)
        self.assertEqual(raw["entries"][:-1], [e.to_dict() for e in entries[1:]])
        self.assertEqual(raw["entries"][-1]["candidate_id"], candidate.id)
        self.assertGreater(len({entry["skill_id"] for entry in raw["entries"]}), 1)
        current_path = next(p for p in self.candidates() if p.name == candidate.id + ".json")
        original = current_path.read_bytes()
        content, mtime = self.memory.read_bytes(), self.memory.stat().st_mtime_ns
        for day in ("2026-02-02", "2026-02-03"):
            receipt, calls = self.invoke(day)
            self.assertEqual(receipt["counts"]["replacements_refused_as_repeats"], 1)
            self.assertEqual(role_counts(calls)["replay_scorer"], 0)
            self.assertEqual(self.memory.read_bytes(), content)
            self.assertEqual(self.memory.stat().st_mtime_ns, mtime)
            self.assertEqual(current_path.read_bytes(), original)
            self.assertEqual(old_path.read_bytes(), historical_bytes)
        existing = (entries[0], entries[1], entries[2])
        merged, evicted, skipped = rejections.remember_rejections(
            existing, [replace(entries[0], rejected_at="2026-01-01T00:00:00Z")]
        )
        self.assertEqual(merged, existing, "duplicate insertion never renews position or timestamp")
        self.assertEqual((evicted, skipped), (0, 0))

    def test_serialized_bytes_are_bounded_without_truncating_identity(self):
        self.assertEqual(rejections.MAX_REJECTION_MEMORY_BYTES, 131072)
        entry = replace(sample_entry(), skill_id="é" * 128)
        self.assertEqual(len(entry.skill_id.encode()), 256)
        self.assertEqual(
            rejections.parse_rejection_memory(memory_payload([entry])).entries, (entry,)
        )
        too_long = replace(entry, skill_id="é" * 129)
        self.assertTrue(rejections.parse_rejection_memory(memory_payload([too_long])).problem)
        overflow = replace(sample_entry(), skill_id="\x01" * 256, candidate_id="CAND-" + "a" * 199)
        self.assertGreater(
            len(dumps(overflow.to_dict()).encode()), rejections.MAX_REJECTION_ENTRY_BYTES
        )
        self.assertTrue(rejections.parse_rejection_memory(memory_payload([overflow])).problem)
        kept, evicted, skipped = rejections.remember_rejections([entry], [overflow, too_long])
        self.assertEqual(kept, (entry,))
        self.assertEqual((evicted, skipped), (0, 2))
        large = memory_payload([sample_entry(i) for i in range(64)])
        self.assertLess(len(dumps(large).encode()), 131072)
        # Fixed compact shape normally reaches the count cap first. Lower only
        # the budget for the pure parser/FIFO byte edges; no fake CLI capacity.
        one = sample_entry(1)
        budget = len(dumps(memory_payload([entry, one])).encode()) - 1
        with patch.object(rejections, "MAX_REJECTION_MEMORY_BYTES", budget):
            self.assertEqual(
                rejections.parse_rejection_memory(memory_payload([entry, one])).problem,
                "over byte limit",
            )
            merged, evicted, skipped = rejections.remember_rejections([entry], [one])
        self.assertEqual(merged, (one,))
        self.assertEqual((evicted, skipped), (1, 0))
        self.assertEqual(merged[0].to_dict(), one.to_dict())

    def test_memory_uses_locally_recomputed_candidate_and_validation_fields(self):
        candidate = self.seed()
        cfg = load_config(self.config)
        kwargs = {
            "base_text": self.skill.read_text(),
            "rubrics": {r.id: r for r in cfg.rubrics},
            "evaluation_context": self.context(),
        }
        entry = rejections.rejection_entry_from_candidate(candidate, **kwargs)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.replacement_sha256, candidate.recomputed_patch_sha256)
        faults = {
            "patch-digest": replace(
                candidate, patch=replace(candidate.patch, new_text_sha256="0" * 64)
            ),
            "patch-diff": replace(
                candidate, patch=replace(candidate.patch, unified_diff="invented diff")
            ),
            "patch-count": replace(candidate, patch=replace(candidate.patch, added_lines=99)),
            "base": replace(candidate, target=replace(candidate.target, base_sha256="0" * 64)),
            "challenge": replace(candidate, approval_challenge="0" * 64),
        }
        for name in ("gate", "validation-patch", "outcome", "rubric"):
            raw = copy.deepcopy(candidate.validation)
            if name == "gate":
                raw["gate"]["metrics"]["fixes"] += 1
            elif name == "validation-patch":
                raw["patch_sha256"] = "0" * 64
            elif name == "outcome":
                raw["outcomes"][0]["candidate"] = "PASS"
            else:
                raw["rubrics"][0]["source_sha256"] = "0" * 64
            faults[name] = replace(candidate, validation=raw)
        for name, broken in faults.items():
            with self.subTest(unit_integrity_fault=name):
                with self.assertRaises(SafetyRefusal):
                    rejections.rejection_entry_from_candidate(broken, **kwargs)
        self.assertEqual(
            set(entry.to_dict()),
            {
                "skill_id",
                "base_sha256",
                "replacement_sha256",
                "cases_sha256",
                "evaluation_context_sha256",
                "candidate_id",
                "rejected_at",
            },
        )
        serialized = dumps(entry.to_dict())
        for private in (
            candidate.patch.new_text,
            candidate.patch.unified_diff,
            candidate.hypothesis,
            candidate.approval_challenge,
        ):
            self.assertNotIn(private, serialized)
        # This proves local construction, not authentication of a syntactically
        # valid manually poisoned memory; no historical candidate is consulted.

    def test_memory_is_private_and_default_absent_without_a_committed_qualifying_failure(self):
        self.fresh(mode="")
        receipt, _ = self.invoke()
        self.assertGreater(receipt["counts"]["candidates_gate_passed"], 0)
        self.assertFalse(self.memory.exists())
        self.assertEqual({p.name for p in self.state.iterdir()}, {"README.txt", *SUBDIRS})
        self.fresh(min_holdout_cases=100)
        receipt, _ = self.invoke()
        self.assertGreater(receipt["counts"]["candidates_gate_failed"], 0)
        self.assertFalse(self.memory.exists())
        self.assertEqual({p.name for p in self.state.iterdir()}, {"README.txt", *SUBDIRS})
        for mode, qualifying in (("", False), ("repeat-no-fix", True)):
            with self.subTest(dry_run=mode):
                self.fresh(mode=mode)
                receipt, _ = self.invoke(commit=False, stack=self.capture_store)
                self.assertFalse(self.memory.exists())
                self.assertEqual({p.name for p in self.state.iterdir()}, {"README.txt", *SUBDIRS})
                self.assertEqual(
                    [
                        p.relative_to(self.state).as_posix()
                        for p in self.state.rglob("*")
                        if p.is_file() and p.relative_to(self.state).parts[0] != "locks"
                    ],
                    ["README.txt"],
                )
                writes = self.last_store.withheld_writes
                kinds = {p.split("/", 1)[0] for p in writes}
                expected = {
                    "conversations",
                    "scorecards",
                    "failures",
                    "candidates",
                    "receipts",
                    "checkpoints",
                }
                if qualifying:
                    expected.add(rejections.REJECTION_MEMORY_NAME)
                self.assertEqual(kinds, expected)
                self.assertEqual(writes.count(rejections.REJECTION_MEMORY_NAME), int(qualifying))
                self.assertEqual(receipt["counts"]["artifacts_withheld_dry_run"] + 2, len(writes))
                self.assertEqual(writes[-2].split("/")[0], "receipts")
                self.assertEqual(writes[-1].split("/")[0], "checkpoints")
                if qualifying:
                    self.assertLess(writes.index(rejections.REJECTION_MEMORY_NAME), len(writes) - 2)
        # A dry-run qualifying loss also preserves existing nonmatching memory
        # and all content, while counting one prospective optional publication.
        self.fresh()
        self.write_memory([sample_entry(99)])
        StateStore(self.state).ensure()
        snapshot = {
            p.relative_to(self.state).as_posix(): p.read_bytes()
            for p in self.state.rglob("*")
            if p.is_file() and p.relative_to(self.state).parts[0] != "locks"
        }
        receipt, _ = self.invoke(commit=False, stack=self.capture_store)
        self.assertEqual(self.last_store.withheld_writes.count(rejections.REJECTION_MEMORY_NAME), 1)
        self.assertEqual(
            {
                p.relative_to(self.state).as_posix(): p.read_bytes()
                for p in self.state.rglob("*")
                if p.is_file() and p.relative_to(self.state).parts[0] != "locks"
            },
            snapshot,
        )
        self.fresh()
        umask = os.umask(0)
        try:
            self.seed()
        finally:
            os.umask(umask)
        self.assertEqual(stat.S_IMODE(self.state.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.memory.stat().st_mode), 0o600)
        before, mtime = self.memory.read_bytes(), self.memory.stat().st_mtime_ns
        receipt, _ = self.invoke("2026-02-02", commit=False, stack=self.capture_store)
        self.assertEqual(receipt["counts"]["replacements_refused_as_repeats"], 1)
        self.assertTrue(any(p.startswith("refusals/") for p in self.last_store.withheld_writes))
        self.assertNotIn(rejections.REJECTION_MEMORY_NAME, self.last_store.withheld_writes)
        self.assertEqual(self.memory.read_bytes(), before)
        self.assertEqual(self.memory.stat().st_mtime_ns, mtime)
        self.memory.chmod(0o644)
        receipt, _ = self.invoke("2026-02-03")
        self.assertEqual(stat.S_IMODE(self.memory.stat().st_mode), 0o600)
        self.assertGreaterEqual(receipt["counts"]["state_modes_repaired"], 1)
        self.assertEqual(self.memory.read_bytes(), before)
        other = self.env / "skills-canonical" / "code-review" / "SKILL.md"
        changed = []

        def mutation(kwargs, payload):
            if kwargs["role"] == "improver":
                payload["new_skill_text"] += "\n"
            if (
                kwargs["role"] == "scorer"
                and "## SKILL TEXT UNDER EVALUATION" in kwargs["user"]
                and not changed
            ):
                other.write_text(other.read_text() + "\nA different skill was externally edited.\n")
                changed.append(True)
            return payload

        pending, after_publish = [], []
        real_publish = rejections.RejectionMemoryRun.publish_if_stable

        def publish(run, state, **kwargs):
            pending.append(len(run.pending))
            result = real_publish(run, state, **kwargs)
            after_publish.append(len(run.pending))
            return result

        def configure(active):
            active.enter_context(
                patch.object(rejections.RejectionMemoryRun, "publish_if_stable", publish)
            )

        receipt, _ = self.invoke("2026-02-04", transform=mutation, stack=configure, rc=5)
        self.assertTrue(changed)
        self.assertEqual(receipt["status"], "blocked")
        self.assertEqual(pending, [1])
        self.assertEqual(after_publish, [0])
        self.assertEqual(self.memory.read_bytes(), before)
        self.assertTrue(all(c["status"] == "blocked-run" for c in receipt["candidates"]))
        self.fresh()
        real_persist = StateStore._persist

        def failing_candidate(store, path, write):
            if path.parent.name == "candidates":
                raise OSError("fixture candidate persist fault")
            return real_persist(store, path, write)

        with patch.object(rejections.RejectionMemoryRun, "stage_candidate", autospec=True) as stage:
            receipt, _ = self.invoke(
                stack=lambda active: active.enter_context(
                    patch.object(StateStore, "_persist", failing_candidate)
                ),
                rc=70,
            )
        stage.assert_not_called()
        self.assertIsNone(receipt)
        self.assertFalse(self.memory.exists())
        self.assertEqual(self.candidates(), [])


if __name__ == "__main__":
    unittest.main()
