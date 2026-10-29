"""A judge running the improver's command is not independent, whatever the ids say.

The gate refuses, by default, a candidate whose blind judge runs the same
command as its improver, and it derives that from the two command ids the
validation records (in ``app/validate.py``, and again when review and apply
recompute the gate from the record). Every id starts with its role, and its
label, so the two ids always differed and every configured judge counted as
independent. Measured before (3.11 and 3.13, a committed run with
``[model.judge]`` given the improver's byte-identical argv, the default
``require_independent_judge = true``): the gate passed, the candidate was
``validated`` with no reason and no note, and review offered an approval
token, while ``doctor`` and the run's receipt said the judge was not
independent and that the gate would refuse. ``runs_the_same_command`` in
``domain/validation.py`` now compares what follows each id's last ``:iso=``,
the isolation mode and the masked command line, and both places use it.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_skill_improver import cli
from agent_skill_improver.adapters.model_command import RoleCommand
from agent_skill_improver.domain import validation as gate_rules

from tests.support import build_env

DAY = "2026-02-01"
SAME_COMMAND = "the judge role runs the same command as the improver role"


def run(config: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(["--config", str(config), *argv])
    return rc, out.getvalue()


def argv_line(text: str, role: str) -> str:
    found = re.search(rf"\[model\.{role}\]\nargv = (.*)\n", text)
    assert found, f"no [model.{role}] argv"
    return found.group(1)


class _Run:
    """One committed daily run, the judge given the improver's argv if ``same``."""

    def __init__(
        self,
        base: Path,
        *,
        same: bool,
        label: str = "",
        independent: bool = True,
        secrets: bool = False,
    ):
        self.config = build_env(base / "env", seed="c217", require_independent_judge=independent)
        text = self.config.read_text(encoding="utf-8")
        if same:
            improver = argv_line(text, "improver")
            text, n = re.subn(
                r"(\[model\.judge\]\nargv = ).*\n", lambda m: m.group(1) + improver + "\n", text
            )
            assert n == 1
        if secrets:
            for role in ("judge", "improver"):
                line = argv_line(text, role)
                text = text.replace(
                    f"[model.{role}]\nargv = {line}",
                    f'[model.{role}]\nargv = {line[:-1]}, "--token={role}Secret123"]',
                )
        if label:
            for role in ("judge", "improver"):
                text = text.replace(f'model_label = "fake-{role}"', f'model_label = "{label}"')
        self.config.write_text(text, encoding="utf-8")
        self.same_argv = argv_line(text, "judge") == argv_line(text, "improver")
        self.doctor_rc, self.doctor = run(self.config, "doctor")
        self.rc, self.output = run(self.config, "daily", "--commit", "--day", DAY)
        state = base / "env" / "state"
        (self.record_file,) = sorted(state.glob("candidates/*.json"))
        self.record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self.validation = self.record["validation"]
        self.gate = self.validation["gate"]


class TheIdsNameTheCommandThatRunsTests(unittest.TestCase):
    """``runs_the_same_command`` over ids ``RoleCommand.command_id`` builds.

    Reached through its module, so on a tree without it only these tests error.
    """

    ARGV = ("python3", "model.py", "--fast")

    def cid(self, role: str, argv: tuple[str, ...], label: str = "") -> str:
        return RoleCommand(role=role, argv=argv, model_label=label).command_id("none")

    def test_one_argv_under_two_roles_and_labels_is_the_same_command(self) -> None:
        judge = self.cid("judge", self.ARGV, "judge-model")
        improver = self.cid("improver", self.ARGV, "improver-model")
        self.assertNotEqual(judge, improver)
        self.assertTrue(gate_rules.runs_the_same_command(judge, improver))

    def test_two_argvs_are_two_commands_whatever_the_labels(self) -> None:
        other = ("python3", "model.py", "--slow")
        for label in ("", "same-label"):
            with self.subTest(label=label):
                self.assertFalse(
                    gate_rules.runs_the_same_command(
                        self.cid("judge", other, label), self.cid("improver", self.ARGV, label)
                    )
                )

    def test_argvs_differing_only_in_a_masked_secret_are_one_command(self) -> None:
        judge = self.cid("judge", (*self.ARGV, "--token=judgeSecret123"))
        improver = self.cid("improver", (*self.ARGV, "--token=improverSecret456"))
        self.assertTrue(gate_rules.runs_the_same_command(judge, improver))

    def test_a_label_holding_the_marker_cannot_make_one_command_two(self) -> None:
        judge = self.cid("judge", self.ARGV, "x:iso=sandbox:y")
        self.assertTrue(
            gate_rules.runs_the_same_command(judge, self.cid("improver", self.ARGV, "plain"))
        )

    def test_two_isolation_modes_are_two_commands(self) -> None:
        confined = RoleCommand(role="judge", argv=self.ARGV).command_id("sandbox")
        self.assertFalse(
            gate_rules.runs_the_same_command(confined, self.cid("improver", self.ARGV))
        )

    def test_an_id_without_the_marker_is_compared_whole(self) -> None:
        self.assertTrue(gate_rules.runs_the_same_command("judge:old", "judge:old"))
        self.assertFalse(gate_rules.runs_the_same_command("judge:old", "improver:old"))


class AJudgeRunningTheImproversCommandFailsTheGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp()).resolve()
        cls.same = _Run(cls.tmp / "same", same=True)
        cls.same_label = _Run(cls.tmp / "same_label", same=True, label="one-model")
        cls.distinct_label = _Run(cls.tmp / "distinct_label", same=False, label="one-model")
        cls.opted_out = _Run(cls.tmp / "opted_out", same=True, independent=False)
        cls.secrets = _Run(cls.tmp / "secrets", same=True, secrets=True)

    @classmethod
    def tearDownClass(cls) -> None:
        subprocess.run(["rm", "-rf", str(cls.tmp)])

    def test_every_run_is_healthy_and_proposes_one_candidate(self) -> None:
        for each in (self.same, self.same_label, self.distinct_label, self.opted_out, self.secrets):
            self.assertEqual(each.rc, 0, each.output)
            self.assertIn(f"status=ok day={DAY} mode=commit", each.output)

    def test_the_same_argv_fails_the_gate_with_its_reason(self) -> None:
        self.assertTrue(self.same.same_argv)
        self.assertIn("candidates=1 gate_passed=0", self.same.output)
        self.assertEqual(self.same.record["status"], "gate-failed")
        self.assertFalse(self.same.gate["passed"])
        self.assertFalse(self.same.gate["metrics"]["judge_independent"])
        self.assertTrue(any(SAME_COMMAND in r for r in self.same.gate["reasons"]))
        self.assertTrue(any("same command" in n for n in self.same.validation["notes"]))

    def test_review_offers_no_token_for_it(self) -> None:
        rc, review = run(self.same.config, "review", self.same.record["id"])
        self.assertNotEqual(rc, 0, review)
        self.assertNotIn("--approve", review)

    def test_one_label_on_both_does_not_change_that(self) -> None:
        self.assertEqual(self.same_label.record["status"], "gate-failed")
        self.assertFalse(self.same_label.gate["metrics"]["judge_independent"])

    def test_the_control_distinct_argvs_under_one_label_pass(self) -> None:
        self.assertFalse(self.distinct_label.same_argv)
        self.assertIn("candidates=1 gate_passed=1", self.distinct_label.output)
        self.assertEqual(self.distinct_label.record["status"], "validated")
        self.assertTrue(self.distinct_label.gate["metrics"]["judge_independent"])

    def test_doctor_warns_of_the_same_argv_and_the_gate_agrees(self) -> None:
        self.assertIn("judge        NOT independent of the improver", self.same.doctor)
        self.assertIn("judge        independent of the improver", self.distinct_label.doctor)

    def test_argvs_differing_only_in_a_masked_secret_fail_where_doctor_passes_them(self) -> None:
        self.assertFalse(self.secrets.same_argv)
        self.assertIn("judge        independent of the improver", self.secrets.doctor)
        self.assertEqual(self.secrets.record["status"], "gate-failed")
        self.assertTrue(any(SAME_COMMAND in r for r in self.secrets.gate["reasons"]))

    def test_the_opt_out_passes_and_records_the_judge_as_not_independent(self) -> None:
        gate = self.opted_out.gate
        self.assertEqual(self.opted_out.record["status"], "validated")
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["metrics"]["judge_independent"])
        self.assertFalse(gate["policy"]["require_independent_judge"])
        self.assertTrue(any("correlated" in n for n in self.opted_out.validation["notes"]))


class ARecordWhoseIdsNameOneCommandIsRefusedTests(unittest.TestCase):
    """Review recomputes the gate from the record's own ids, by the same rule."""

    def test_a_passed_record_whose_judge_id_names_the_improvers_command_is_refused(self) -> None:
        tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(subprocess.run, ["rm", "-rf", str(tmp)])
        passed = _Run(tmp, same=False)
        self.assertTrue(passed.gate["passed"])
        record = json.loads(passed.record_file.read_text(encoding="utf-8"))
        improver_id = record["validation"]["improver_command_id"]
        record["validation"]["judge_command_id"] = "judge:another-label:iso=" + (
            improver_id.rpartition(":iso=")[2]
        )
        passed.record_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
        rc, review = run(passed.config, "review", record["id"])
        self.assertNotEqual(rc, 0, review)
        self.assertIn("validation.gate.metrics.judge_independent: False != True", review)
        self.assertNotIn("--approve", review)


if __name__ == "__main__":
    unittest.main()
