"""A measured rejection saves replay, not the next proposal or a new gate verdict."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent_skill_improver import cli
from agent_skill_improver.app import daily, rejections
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.app.wiring import build_runner
from agent_skill_improver.config import load_config
from agent_skill_improver.domain.candidate import Candidate
from agent_skill_improver.hashing import sha256_text
from agent_skill_improver.jsonio import dumps
from tests import fake_model
from tests.support import FIXTURE_EXPORTS, build_env, run_cli

DAY1 = "2026-02-01"
DAY2 = "2026-02-02"
SEED = "c242-fixed-seed"
# Per-two-day call count measured before c242 on Python 3.13.0 and 3.11.15.
BEFORE_C242_TOTAL_CALLS = 32
BEFORE_C242_PER_DAY_CALLS = {
    "initial_scorer": 5,
    "replay_scorer": 8,
    "improver": 1,
    "judge": 2,
}


def wire_modes(config: Path, log: Path, *, mode: str = "repeat-no-fix") -> None:
    """Role env is explicit; changes here deliberately keep command ids fixed."""
    body = config.read_text(encoding="utf-8")
    for role in ("scorer", "improver", "judge"):
        heading = f"[model.{role}]\n"
        assert body.count(heading) == 1
        env = {"ASI_TEST_ROLE_CALL_LOG": str(log), "ASI_FAKE_MODE": mode}
        table = ", ".join(f"{key} = {json.dumps(value)}" for key, value in env.items())
        body = body.replace(heading, heading + "env = { " + table + " }\n")
    config.write_text(body, encoding="utf-8")


def role_counts(calls: list[dict]) -> dict[str, int]:
    return {
        "initial_scorer": sum(c["role"] == "scorer" and not c["replay"] for c in calls),
        "replay_scorer": sum(c["role"] == "scorer" and c["replay"] for c in calls),
        "improver": sum(c["role"] == "improver" for c in calls),
        "judge": sum(c["role"] == "judge" for c in calls),
    }


class _RoleOverride:
    """Real command runner with one focused response override, unchanged identity."""

    def __init__(self, delegate, transform):
        self.delegate = delegate
        self.transform = transform

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def run(self, **kwargs):
        payload = self.delegate.run(**kwargs)
        return self.transform(kwargs, payload)


class ARejectedReplacementIsNotReplayedAgainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="c242-repeat-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.config = build_env(self.base / "env", seed=SEED)
        self.log = self.base / "calls.jsonl"
        wire_modes(self.config, self.log)
        self.state = self.base / "env" / "state"
        self.memory = self.state / rejections.REJECTION_MEMORY_NAME
        self.skill = self.base / "env" / "skills-canonical" / "tdd" / "SKILL.md"

    def calls(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def invoke(self, day: str, *, transform=None, raced_read=False) -> tuple[dict, list[dict]]:
        offset = len(self.calls())
        args = [
            "--config",
            str(self.config),
            "--json",
            "daily",
            "--commit",
            "--day",
            day,
            "--seed",
            SEED,
        ]
        if transform is None and not raced_read:
            result = run_cli(args)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            receipt = json.loads(result.stdout)
        else:
            out = io.StringIO()
            original_read = daily.read_canonical_text
            with contextlib.ExitStack() as stack:
                if transform is not None:
                    stack.enter_context(
                        patch.object(
                            cli,
                            "build_runner",
                            side_effect=lambda cfg, state: _RoleOverride(
                                build_runner(cfg, state), transform
                            ),
                        )
                    )
                if raced_read:
                    stack.enter_context(
                        patch.object(
                            daily,
                            "read_canonical_text",
                            side_effect=lambda *a, **k: original_read(*a, **k) + "\nRaced edit.\n",
                        )
                    )
                stack.enter_context(contextlib.redirect_stdout(out))
                stack.enter_context(contextlib.redirect_stderr(out))
                rc = cli.main(args)
            self.assertEqual(rc, 0, out.getvalue())
            receipt = json.loads(out.getvalue())
        return receipt, self.calls()[offset:]

    def candidates(self) -> list[Path]:
        return sorted((self.state / "candidates").glob("*.json"))

    def seed_failure(self) -> tuple[dict, Candidate]:
        receipt, calls = self.invoke(DAY1)
        self.assertEqual(
            role_counts(calls),
            {
                "initial_scorer": 5,
                "replay_scorer": 8,
                "improver": 1,
                "judge": 2,
            },
        )
        candidate = Candidate.from_dict(json.loads(self.candidates()[0].read_text()))
        self.assertEqual(candidate.status, "gate-failed")
        self.assertTrue(self.memory.is_file(), receipt)
        self.assertEqual(len(json.loads(self.memory.read_text())["entries"]), 1)
        return receipt, candidate

    def assert_new_trial(self, receipt: dict, calls: list[dict]) -> None:
        self.assertEqual(receipt["counts"]["replacements_refused_as_repeats"], 0, receipt)
        self.assertGreater(role_counts(calls)["replay_scorer"], 0)
        self.assertGreater(role_counts(calls)["judge"], 0)
        self.assertEqual(len(receipt["candidates"]), 1)

    def test_the_next_day_refuses_the_exact_replacement_before_replay_or_judge(self) -> None:
        initial = self.skill.read_bytes()
        first, calls1 = self.invoke(DAY1)
        (path,) = self.candidates()
        original = path.read_bytes()
        candidate = json.loads(original)
        self.assertEqual(candidate["status"], "gate-failed")
        remembered = self.memory.read_bytes()
        second, calls2 = self.invoke(DAY2)
        expected = [
            {"initial_scorer": 5, "replay_scorer": 8, "improver": 1, "judge": 2},
            {"initial_scorer": 5, "replay_scorer": 0, "improver": 1, "judge": 0},
        ]
        self.assertEqual([role_counts(calls1), role_counts(calls2)], expected)
        self.assertEqual(second["status"], "ok")
        self.assertEqual(second["counts"]["records_scored"], 5)
        self.assertEqual(second["counts"]["proposals_refused_by_policy"], 1)
        self.assertEqual(second["counts"]["replacements_refused_as_repeats"], 1)
        self.assertEqual(second["counts"]["candidates_gate_failed"], 0)
        self.assertEqual(second["counts"]["candidates_proposed"], 0)
        self.assertEqual(second["counts"]["records_quarantined"], 0)
        self.assertEqual(second["counts"]["skills_deferred"], 0)
        self.assertEqual(len(self.candidates()), 1)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.skill.read_bytes(), initial)
        self.assertEqual(self.memory.read_bytes(), remembered)
        repeat = second["skills"]["tdd"]["repeat"]
        self.assertEqual(repeat["of_candidate_id"], candidate["id"])
        self.assertFalse(repeat["validation_ran"])
        self.assertEqual(
            (repeat["improver_calls"], repeat["replay_scorer_calls"], repeat["judge_calls"]),
            (1, 0, 0),
        )
        self.assertEqual(second["skills"]["tdd"]["status"], "refused")
        self.assertEqual(second["skills"]["tdd"]["stage"], "propose")
        (memory_entry,) = json.loads(remembered)["entries"]
        for key in (
            "base_sha256",
            "replacement_sha256",
            "cases_sha256",
            "evaluation_context_sha256",
        ):
            self.assertEqual(repeat[key], memory_entry[key])
        checkpoint = StateStore(self.state).read_checkpoint(DAY2)
        self.assertEqual(checkpoint["skills"]["tdd"]["repeat"], repeat)
        self.assertEqual(checkpoint["candidates"], [])
        hashes = [
            next(row for row in calls if row["role"] == "improver") for calls in (calls1, calls2)
        ]
        self.assertEqual(hashes[0], hashes[1], "No prompt bytes may change with history")
        (refusal,) = (self.state / "refusals").glob("*.json")
        self.assertEqual(json.loads(refusal.read_text())["record_id"], "tdd")
        listed = run_cli(["--config", str(self.config), "--json", "set-aside"])
        self.assertEqual(listed.returncode, 0, listed.stdout + listed.stderr)
        self.assertIn(candidate["id"], listed.stdout)
        listing = run_cli(["--config", str(self.config), "--json", "candidates", "--all"])
        self.assertEqual(listing.returncode, 0, listing.stdout + listing.stderr)
        self.assertIn(candidate["id"], listing.stdout)
        for command in ("review", "apply"):
            args = ["--config", str(self.config), command, candidate["id"]]
            if command == "apply":
                args += ["--approve", "not-a-token", "--i-reviewed-this"]
            result = run_cli(args)
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("gate-failed", result.stdout + result.stderr)
        nonexistent = run_cli(["--config", str(self.config), "review", "CAND-2026-02-02-01-tdd"])
        self.assertEqual(nonexistent.returncode, 4, nonexistent.stdout + nonexistent.stderr)
        self.assertEqual(2 * sum(BEFORE_C242_PER_DAY_CALLS.values()), BEFORE_C242_TOTAL_CALLS)
        after_total = sum(sum(role_counts(calls).values()) for calls in (calls1, calls2))
        saved_calls = BEFORE_C242_TOTAL_CALLS - after_total
        saved_fraction = saved_calls / BEFORE_C242_TOTAL_CALLS
        self.assertEqual(saved_calls, 10)
        self.assertEqual(saved_fraction, 0.3125)
        report = {
            "python": sys.version.split()[0],
            "days": [DAY1, DAY2],
            "per_day_calls": [role_counts(calls1), role_counts(calls2)],
            "before_per_day_calls": BEFORE_C242_PER_DAY_CALLS,
            "before_total": BEFORE_C242_TOTAL_CALLS,
            "after_total": after_total,
            "saved_calls": saved_calls,
            "saved_fraction": saved_fraction,
            "day2_exit": 0,
            "day2_status": second["status"],
            "repeat": repeat,
            "improver_prompt_hashes": hashes,
            "original_candidate_unchanged": True,
            "installed_skill_unchanged": True,
            "one_original_candidate_only": True,
        }
        if os.environ.get("ASI_TEST_MEASUREMENT_DIR"):
            destination = (
                Path(os.environ["ASI_TEST_MEASUREMENT_DIR"])
                / f"measured_after_{sys.version.split()[0]}.json"
            )
            destination.write_text(dumps(report), encoding="utf-8")
        print("C242_MEASURED_AFTER " + json.dumps(report, sort_keys=True))

    def test_a_crlf_repeat_has_the_same_lf_normalized_fingerprint(self) -> None:
        self.seed_failure()
        body = self.config.read_text()
        self.assertEqual(body.count('"repeat-no-fix"'), 3)
        self.config.write_text(body.replace('"repeat-no-fix"', '"repeat-no-fix-crlf"'))
        receipt, calls = self.invoke(DAY2)
        self.assertEqual(receipt["counts"]["replacements_refused_as_repeats"], 1)
        self.assertEqual(role_counts(calls)["replay_scorer"], 0)
        self.config.write_text(body.replace('"repeat-no-fix"', '"repeat-no-fix-blank-line"'))
        receipt, calls = self.invoke("2026-02-03")
        self.assert_new_trial(receipt, calls)
        for text in ("x\n", "\ufeffx", "é", "e\u0301"):
            with self.subTest(text=text):
                self.assertNotEqual(sha256_text(text), sha256_text("x"))
        self.assertNotEqual(sha256_text("é"), sha256_text("e\u0301"))
        self.assertEqual(sha256_text("x\r\ny\r"), sha256_text("x\ny\n"))

    def test_changed_skill_base_replacement_or_examiner_is_a_new_trial(self) -> None:
        _, candidate = self.seed_failure()
        memory_bytes = self.memory.read_bytes()
        original_config = self.config.read_text()
        rubric = self.base / "env" / "rubrics" / "instruction-compliance.toml"
        original_rubric = rubric.read_text()
        original_skill = self.skill.read_text()
        day_index = 3
        changes = {
            "other-skill": None,
            "changed-base-only": None,
            "replacement": ('"repeat-no-fix"', '"repeat-no-fix-blank-line"'),
            "rubric-bytes": None,
            "gate": ("min_fixes = 1", "min_fixes = 2"),
            "scorer-command": (
                '"--role", "scorer"]',
                '"--role", "scorer", "--different-examiner"]',
            ),
            "judge-command": ('"--role", "judge"]', '"--role", "judge", "--different-examiner"]'),
            "prompt-cap": ("max_message_chars = 4000", "max_message_chars = 4001"),
            "extra-holdout-cap": (
                "max_extra_holdout_conversations = 3",
                "max_extra_holdout_conversations = 2",
            ),
            "unseen-withhold": (
                "max_open_candidates = 12",
                "max_open_candidates = 12\nunseen_withhold = 1\nunseen_min_failures = 3",
            ),
            "unseen-min-failures": None,
        }
        for name, change in changes.items():
            with self.subTest(component=name):
                self.memory.write_bytes(memory_bytes)
                self.config.write_text(original_config)
                self.skill.write_text(original_skill)
                rubric.write_text(original_rubric)
                transform = None
                if name == "other-skill":
                    raw = json.loads(memory_bytes)
                    raw["entries"][0]["skill_id"] = "code-review"
                    self.memory.write_text(dumps(raw))
                elif name == "changed-base-only":
                    self.skill.write_text(original_skill + "\nBase changed.\n")

                    def transform(kwargs, payload):
                        if kwargs["role"] == "improver":
                            payload["new_skill_text"] = candidate.patch.new_text
                        return payload

                elif name == "unseen-min-failures":
                    # Shipped c240 omits N/k from to_dict() while k=0. Isolate
                    # an N-only examiner change with withholding enabled.
                    cfg_n3 = original_config.replace(
                        "max_open_candidates = 12",
                        "max_open_candidates = 12\nunseen_withhold = 1\nunseen_min_failures = 3",
                    )
                    self.config.write_text(cfg_n3)
                    current = load_config(self.config)
                    raw = json.loads(memory_bytes)
                    raw["entries"][0]["evaluation_context_sha256"] = (
                        rejections.evaluation_context_sha256(
                            current, {r.id: r for r in current.rubrics}, build_runner(current)
                        )
                    )
                    self.memory.write_text(dumps(raw))
                    self.config.write_text(
                        cfg_n3.replace("unseen_min_failures = 3", "unseen_min_failures = 4")
                    )
                elif name == "rubric-bytes":
                    rubric.write_text(original_rubric + "\n# Undeclared examiner byte edit.\n")
                elif change is not None:
                    old, new = change
                    self.assertGreater(original_config.count(old), 0)
                    self.config.write_text(original_config.replace(old, new))
                receipt, calls = self.invoke(f"2026-03-{day_index:02d}", transform=transform)
                self.assert_new_trial(receipt, calls)
                new_candidate = Candidate.from_dict(json.loads(self.candidates()[-1].read_text()))
                if name in ("other-skill", "changed-base-only"):
                    self.assertEqual(
                        new_candidate.recomputed_patch_sha256, candidate.recomputed_patch_sha256
                    )
                    self.assertEqual(
                        rejections.cases_sha256(new_candidate), rejections.cases_sha256(candidate)
                    )
                if name == "changed-base-only":
                    self.assertNotEqual(
                        new_candidate.target.base_sha256, candidate.target.base_sha256
                    )
                day_index += 1
        self.config.write_text(original_config)
        self.skill.write_text(original_skill)
        rubric.write_text(original_rubric)
        self.memory.write_bytes(memory_bytes)
        cfg = load_config(self.config)
        rubrics = {r.id: r for r in cfg.rubrics}
        runner = build_runner(cfg)
        context = rejections.evaluation_context_sha256(cfg, rubrics, runner)
        # Every cap is in the exact closed context object, not a hand-picked list.
        real_dumps = rejections.dumps
        captured = []

        def capture(obj):
            captured.append(obj)
            return real_dumps(obj)

        with patch.object(rejections, "dumps", side_effect=capture):
            self.assertEqual(
                rejections.evaluation_context_sha256(
                    cfg, dict(reversed(list(rubrics.items()))), runner
                ),
                context,
            )
        self.assertEqual(captured[0]["caps"], cfg.caps.to_dict())
        second_rubric = replace(cfg.rubrics[0], id="another-enabled-rubric")
        two = {cfg.rubrics[0].id: cfg.rubrics[0], second_rubric.id: second_rubric}
        self.assertEqual(
            rejections.evaluation_context_sha256(cfg, two, runner),
            rejections.evaluation_context_sha256(cfg, dict(reversed(list(two.items()))), runner),
        )
        changed_improver = _RoleOverride(runner, lambda k, p: p)
        changed_improver.command_id = lambda role: (
            runner.command_id(role)
            if role != "improver"
            else "improver:different:iso=none:another-improver"
        )
        self.assertEqual(
            rejections.evaluation_context_sha256(cfg, rubrics, changed_improver), context
        )
        changed_improver.command_id = lambda role: (
            runner.command_id("judge") if role == "improver" else runner.command_id(role)
        )
        self.assertNotEqual(
            rejections.evaluation_context_sha256(cfg, rubrics, changed_improver), context
        )
        # Matching old memory cannot intercept validate's raced-live-base refusal.
        prior = self.memory.read_bytes()
        receipt, calls = self.invoke("2026-04-01", raced_read=True)
        self.assertEqual(receipt["counts"]["skills_deferred"], 1)
        self.assertEqual(receipt["counts"]["candidates_proposed"], 0)
        self.assertEqual(receipt["counts"]["replacements_refused_as_repeats"], 0)
        self.assertEqual(receipt["counts"]["proposals_refused_by_policy"], 0)
        self.assertEqual(receipt["skills"]["tdd"]["stage"], "validate")
        self.assertEqual(receipt["skills"]["tdd"]["status"], "deferred")
        self.assertEqual(self.memory.read_bytes(), prior)
        self.assertEqual(role_counts(calls)["replay_scorer"], 0)
        listed = run_cli(["--config", str(self.config), "--json", "set-aside"])
        self.assertEqual(listed.returncode, 0, listed.stdout + listed.stderr)
        self.assertTrue(
            any(
                row["kind"] == "deferred"
                and row["record_id"] == "tdd"
                and row["stage"] == "validate"
                for row in json.loads(listed.stdout)["records"]
            )
        )

    def test_the_same_text_for_different_failures_is_replayed(self) -> None:
        _, candidate = self.seed_failure()

        def one_failure(kwargs, payload):
            if (
                kwargs["role"] == "scorer"
                and fake_model._scored_thread(kwargs["user"]) == fake_model.FAILING_THREAD
            ):
                for row in payload["criteria"]:
                    if row["criterion_id"] == "explicit-directive-followed":
                        row["verdict"] = "PASS"
            return payload

        receipt, calls = self.invoke(DAY2, transform=one_failure)
        self.assert_new_trial(receipt, calls)
        second = Candidate.from_dict(json.loads(self.candidates()[-1].read_text()))
        self.assertEqual(candidate.recomputed_patch_sha256, second.recomputed_patch_sha256)
        self.assertEqual(candidate.target.base_sha256, second.target.base_sha256)
        self.assertNotEqual(rejections.cases_sha256(candidate), rejections.cases_sha256(second))
        self.assertEqual(
            rejections.cases_sha256(candidate),
            rejections.cases_sha256(
                replace(candidate, rubric_failures=tuple(reversed(candidate.rubric_failures)))
            ),
        )
        entries = json.loads(self.memory.read_text())["entries"]
        self.assertEqual(
            entries[0]["evaluation_context_sha256"], entries[1]["evaluation_context_sha256"]
        )
        # Holdouts are absent from the five-part identity: advice is not truth on
        # changed corpora. Only claimed refs enter the case digest.
        self.assertEqual(
            rejections.cases_sha256(candidate),
            rejections.cases_sha256(replace(candidate, validation=None)),
        )

    def test_a_repeat_uses_one_slot_and_never_asks_for_another_candidate(self) -> None:
        exports = self.base / "exports"
        shutil.copytree(FIXTURE_EXPORTS, exports)
        body = self.config.read_text()
        old = f"fixtures_dir = {json.dumps(str(FIXTURE_EXPORTS))}"
        self.assertEqual(body.count(old), 1)
        self.config.write_text(body.replace(old, f"fixtures_dir = {json.dumps(str(exports))}"))
        self.seed_failure()
        clone = (
            (exports / "failing.json")
            .read_text()
            .replace(fake_model.FAILING_THREAD, "1700000000099-1")
            .replace("tdd", "code-review")
        )
        (exports / "other-skill-failing.json").write_text(clone)

        def other_skill_failures(kwargs, payload):
            if (
                kwargs["role"] == "scorer"
                and fake_model._scored_thread(kwargs["user"]) == "1700000000099-1"
            ):
                for row in payload["criteria"]:
                    if row["criterion_id"] in fake_model.SEEDED_FAILURES:
                        row["verdict"] = "FAIL"
            return payload

        receipt, calls = self.invoke(DAY2, transform=other_skill_failures)
        self.assertEqual(
            role_counts(calls), {"initial_scorer": 6, "replay_scorer": 8, "improver": 2, "judge": 2}
        )
        self.assertEqual(receipt["counts"]["skills_selected_for_proposal"], 2)
        self.assertEqual(receipt["counts"]["replacements_refused_as_repeats"], 1)
        self.assertEqual(receipt["counts"]["proposals_refused_by_policy"], 1)
        self.assertEqual(receipt["counts"]["records_scored"], 6)
        self.assertEqual(receipt["counts"]["records_quarantined"], 0)
        self.assertEqual(receipt["counts"]["skills_quarantined"], 0)
        self.assertEqual(receipt["counts"]["candidates_proposed"], 1)
        self.assertEqual(receipt["counts"]["candidates_gate_failed"], 1)
        self.assertEqual(receipt["skills"]["tdd"]["repeat"]["replay_scorer_calls"], 0)
        self.assertEqual(receipt["candidates"][0]["skill_id"], "code-review")
        self.assertEqual(len(self.candidates()), 2)


if __name__ == "__main__":
    unittest.main()
