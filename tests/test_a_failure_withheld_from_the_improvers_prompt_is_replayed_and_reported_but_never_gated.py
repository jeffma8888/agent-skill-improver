"""A failure withheld from the improver's prompt is replayed and reported, never gated.

The CLI tests of c240 (DESIGN v3, 4.3, file 10). The fixture is a temporary copy of
the repo's ``config/``, ``examples/`` and ``rubrics/`` with the ``python3`` argv
replaced by this interpreter, ``conv-human-0001`` cloned as 0006, 0007 and 0008
(0006 with its two rules) and ``[selection].max_conversations_per_run = 11``. The
daily run is made in process with the shipped fake model; ``review`` and ``apply``
run as the command line. Every test asserts its precondition first.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_skill_improver.app.daily import run_daily
from agent_skill_improver.app.wiring import build_skill_store, build_source, build_state
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.config import load_config
from agent_skill_improver.domain.candidate import Candidate
from agent_skill_improver.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
CID = "CAND-2026-01-02-01-example-evidence-check"
SEED = f"example-seed:{CID}"
SCOPE = (
    "withheld from the improver's prompt as failures and excerpts only, not from the "
    "skill text that prompt shows or files its command can read; not used by the gate"
)
MARKER = "name the file you read"
FAILURE_CONVERSATIONS = ("conv-human-0001", "conv-human-0006", "conv-human-0007", "conv-human-0008")
W_KEYS = [
    "evidence-discipline::conv-human-0006::cites-checkable-evidence",
    "evidence-discipline::conv-human-0006::separates-measured-from-assumed",
]
SHOWN_KEYS = [f"evidence-discipline::{c}::cites-checkable-evidence" for c in FAILURE_CONVERSATIONS]
SHOWN_KEYS.remove(W_KEYS[0])

_TMP = tempfile.TemporaryDirectory(prefix="asi-c240-")
_COUNTER = [0]


def _fresh_dir() -> Path:
    _COUNTER[0] += 1
    root = Path(_TMP.name)
    root.mkdir(exist_ok=True)  # again after a tearDownModule (see there)
    path = root.resolve() / f"tree-{_COUNTER[0]:02d}"
    path.mkdir()
    return path


def _set(text: str, key: str, value: Any) -> str:
    pattern = re.compile(rf"^{re.escape(key)} = .*$", re.M)
    assert len(pattern.findall(text)) == 1, key
    return pattern.sub(f"{key} = {value}", text)


def make_tree(
    *,
    variant: str = "one",
    k: int | None = 0,
    n: int | None = 4,
    min_fixes: int | None = None,
    max_failures: int | None = None,
    extra_slots: int | None = None,
) -> Path:
    """The CLI fixture of 4.3. ``k``/``n`` None removes the key from the config."""
    tree = _fresh_dir()
    for name in ("config", "examples", "rubrics"):
        shutil.copytree(
            REPO_ROOT / name, tree / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    cfg = tree / "config" / "example.config.toml"
    text = cfg.read_text(encoding="utf-8").replace('"python3"', json.dumps(sys.executable))
    text = _set(text, "state_dir", json.dumps(str(tree / "state")))
    text = _set(text, "max_conversations_per_run", 11)
    for key, value in (("unseen_withhold", k), ("unseen_min_failures", n)):
        if value is None:
            text = re.sub(rf"^{key} = .*\n", "", text, flags=re.M)
        else:
            text = _set(text, key, value)
    if min_fixes is not None:
        text = _set(text, "min_fixes", min_fixes)
    if max_failures is not None:
        text = _set(text, "max_failures_per_candidate", max_failures)
    if extra_slots is not None:
        text = _set(text, "max_extra_holdout_conversations", extra_slots)
    cfg.write_text(text, encoding="utf-8")
    rules = {
        "one": ("name the file you read", "state the command you ran"),
        "zero": ("name the file you read", "name the file you read"),
        "two": ("state the command you ran", "state the command you ran"),
    }[variant]
    source = (tree / "examples/fixtures/conv-human-0001.json").read_text(encoding="utf-8")
    for suffix in ("0006", "0007", "0008"):
        clone = source.replace("conv-human-0001", f"conv-human-{suffix}")
        if suffix == "0006":
            clone = clone.replace(
                "cites-checkable-evidence|state the command you ran",
                f"cites-checkable-evidence|{rules[0]}",
            ).replace(
                "[[PASS:separates-measured-from-assumed]]",
                f"[[FIXABLE:separates-measured-from-assumed|{rules[1]}]]",
            )
        (tree / f"examples/fixtures/conv-human-{suffix}.json").write_text(clone, encoding="utf-8")
    return tree


class Runner:
    """The shipped fake model, in process, keeping every request it was sent."""

    isolation_mode = "in-process"

    def __init__(self) -> None:
        from examples import fake_model

        self.fake = fake_model
        self.requests: list[dict[str, str]] = []

    def run(self, *, role, system, user, response_schema):
        self.requests.append({"role": role, "system": system, "user": user})
        return self.fake.ROLES[role](self.requests[-1])

    def command_id(self, role):
        return f"{role}:fixture:iso=in-process:fake --role {role}"

    def program_problems(self):
        return []

    def isolation(self):
        return {"mode": "in-process", "verified": False}

    def users(self, role: str) -> list[str]:
        return [r["user"] for r in self.requests if r["role"] == role]

    def improver_request(self) -> str:
        """The improver request for the fixture's skill (the run proposes for two)."""
        (user,) = [u for u in self.users("improver") if "'example-evidence-check'" in u]
        return user


@dataclass
class Run:
    tree: Path
    cfg: Path
    receipt: Any
    runner: Runner
    state: Any

    @property
    def candidate(self) -> Candidate:
        return self.state.read_candidate(CID)

    @property
    def validation(self) -> dict[str, Any]:
        return self.candidate.validation

    @property
    def metrics(self) -> dict[str, Any]:
        return self.validation["gate"]["metrics"]

    @property
    def entry(self) -> dict[str, Any]:
        (entry,) = [e for e in self.receipt.candidates if CID in json.dumps(e)]
        return entry

    def notes(self, needle: str) -> list[str]:
        return [n for n in self.receipt.notes if needle in n]

    def cli(self, *args: str, cfg: Path | None = None) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env.update(
            PYTHONPATH=os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)]),
            PYTHONDONTWRITEBYTECODE="1",
        )
        env.pop("ASI_AUTOMATED_RUN", None)
        return subprocess.run(
            [sys.executable, "-m", "agent_skill_improver", "--config", str(cfg or self.cfg)]
            + list(args),
            cwd=self.tree,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )


_RUNS: dict[tuple, Run] = {}


def run_once(*, skill_suffix: str = "", **options: Any) -> Run:
    """A committed daily run on a fresh tree, for a test that edits what it wrote.

    ``skill_suffix`` is appended to the fixture skill's SKILL.md before the run.
    """
    tree = make_tree(**options)
    if skill_suffix:
        skill = tree / "examples" / "skills" / "example-evidence-check" / "SKILL.md"
        skill.write_text(skill.read_text(encoding="utf-8") + skill_suffix, encoding="utf-8")
    cfg = tree / "config" / "example.config.toml"
    config = load_config(cfg)
    state = build_state(config)
    clock = FrozenClock(datetime(2026, 1, 2, tzinfo=timezone.utc))
    state._clock = clock
    runner = Runner()
    receipt = run_daily(
        config,
        source=build_source(config, state),
        runner=runner,
        skill_store=build_skill_store(config),
        state=state,
        clock=clock,
        day="2026-01-02",
        commit=True,
    )
    return Run(tree=tree, cfg=cfg, receipt=receipt, runner=runner, state=state)


def daily(**options: Any) -> Run:
    """``run_once``, cached per options: the tests that share it only read it."""
    key = tuple(sorted(options.items()))
    if key not in _RUNS:
        _RUNS[key] = run_once(**options)
    return _RUNS[key]


def improver_failures(run: Run) -> list[dict[str, Any]]:
    user = run.runner.improver_request()
    section = user.split("## RUBRIC FAILURES TO ADDRESS", 1)[1]
    block = re.search(r"```json\n(.*?)\n```", section, re.S)
    assert block, "no fenced json list under the failures heading"
    return json.loads(block.group(1))


def triple(row: dict[str, Any]) -> str:
    return f"{row['rubric_id']}::{row['conversation_id']}::{row['criterion_id']}"


def ref_keys(refs: list[dict[str, Any]]) -> list[str]:
    return [triple(r) for r in refs]


def unseen_line(text: str) -> list[str]:
    return [line for line in text.splitlines() if "unseen failures fixed:" in line]


def skill_section(user: str) -> str:
    """An improver prompt's CURRENT SKILL TEXT section, up to the next section heading."""
    section = user.split("## CURRENT SKILL TEXT", 1)[1]
    for heading in ("## CITED CONVERSATION EXCERPTS", "## OUTPUT CONTRACT"):
        section = section.split(heading, 1)[0]
    return section


def candidate_file(run: Run) -> Path:
    return run.tree / "state" / "candidates" / f"{CID}.json"


def edit_unseen_verdict(run: Run) -> None:
    """Flip the first unseen verdict in the candidate file, as an editor would."""
    path = candidate_file(run)
    raw = json.loads(path.read_text(encoding="utf-8"))
    row = raw["validation"]["unseen"]["outcomes"][0]
    row["candidate"] = "PASS" if row["candidate"] != "PASS" else "FAIL"
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def support_run(base: Path, *, unseen: str = "") -> tuple[Path, dict[str, Any]]:
    """The support env (``tdd``, two failures on one conversation), committed."""
    from tests.support import build_env, run_cli

    cfg = build_env(base / "env", seed="c240")
    if unseen:
        text = cfg.read_text(encoding="utf-8")
        assert text.count("[caps]\n") == 1
        cfg.write_text(text.replace("[caps]\n", "[caps]\n" + unseen), encoding="utf-8")
    proc = run_cli(["--config", str(cfg), "--json", "daily", "--commit", "--day", "2026-01-02"])
    assert proc.returncode == 0, proc.stderr
    (record,) = sorted((base / "env" / "state" / "candidates").rglob("*.json"))
    return cfg, json.loads(record.read_text(encoding="utf-8"))


class _Base(unittest.TestCase):
    def assert_fixture(self, run: Run) -> None:
        """The precondition: 11 conversations scored, all four failure conversations in the pool."""
        self.assertEqual(len(run.receipt.records), 11, sorted(run.receipt.records))
        pool = {f.conversation_id for f in run.candidate.rubric_failures} | {
            r["conversation_id"] for r in run.validation.get("unseen", {}).get("withheld_refs", [])
        }
        self.assertEqual(pool, set(FAILURE_CONVERSATIONS))


class TurningWithholdingOnIsTheOnlyThingThatChangesBehaviour(_Base):
    def test_the_fixture_selects_all_four_failure_conversations(self):
        run = daily(k=0)
        self.assertEqual(len(run.receipt.records), 11, sorted(run.receipt.records))
        conversations = {f.conversation_id for f in run.candidate.rubric_failures}
        self.assertEqual(conversations, set(FAILURE_CONVERSATIONS))

    def test_with_withholding_off_the_run_is_todays(self):
        run = daily(k=0)
        self.assert_fixture(run)
        m = run.metrics
        self.assertEqual((m["target_cases"], m["fixes"], m["holdout_cases"]), (5, 4, 14))
        self.assertEqual(run.validation["evidence_model_calls"], 29)
        self.assertEqual(run.validation["schema"], "asi.validation.v3")
        self.assertNotIn("unseen", run.validation)
        self.assertNotIn("unseen_failures", run.entry)
        self.assertEqual(run.notes("unseen failures fixed"), [])
        shown = run.cli("review", CID)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertEqual(unseen_line(shown.stdout), [])
        as_json = run.cli("--json", "review", CID)
        self.assertEqual(as_json.returncode, 0, as_json.stderr)
        self.assertNotIn("unseen", json.loads(as_json.stdout))

    def test_a_k_0_receipt_has_no_unseen_key(self):
        """Off is byte-for-byte off in the receipt: no caps key, no candidate key, no note.

        Two-sided: the k = 1 receipt of the same fixture names both caps and the count,
        so a check that could never see an unseen key would fail here.
        """
        off, on = daily(k=0), daily(k=1)
        self.assert_fixture(off)
        written = off.state.receipt_path("2026-01-02", dry_run=False).read_text(encoding="utf-8")
        for text in (json.dumps(off.receipt.to_dict(), sort_keys=True), written):
            self.assertNotIn("unseen", text)
        self.assertEqual(
            sorted(off.receipt.caps),
            [
                "max_candidates_per_run",
                "max_defer_attempts",
                "max_extra_holdout_conversations",
                "max_failures_per_candidate",
                "max_open_candidates",
                "max_open_candidates_per_skill",
                "max_skills_per_run",
            ],
        )
        self.assert_fixture(on)
        self.assertEqual(
            (on.receipt.caps["unseen_withhold"], on.receipt.caps["unseen_min_failures"]), (1, 4)
        )
        self.assertIn("unseen_failures", on.entry)
        written_on = on.state.receipt_path("2026-01-02", dry_run=False).read_text(encoding="utf-8")
        self.assertIn("unseen_failures", written_on)

    def test_with_k_1_conversation_0006_and_both_its_failures_are_withheld(self):
        run = daily(k=1)
        self.assert_fixture(run)
        unseen = run.validation["unseen"]
        self.assertEqual(ref_keys(unseen["withheld_refs"]), W_KEYS)
        self.assertEqual(len(unseen["withheld_refs"]), 2)

    def test_the_improver_prompt_carries_no_withheld_instance(self):
        run = daily(k=1)
        self.assert_fixture(run)
        rows = improver_failures(run)
        self.assertEqual(sorted(triple(r) for r in rows), SHOWN_KEYS)
        for row in rows:
            self.assertNotEqual(row["conversation_id"], "conv-human-0006")
            self.assertFalse(row.get("failure_case_id", "").startswith("FC:conv-human-0006:"))
            for evidence in row.get("evidence", []):
                self.assertFalse(evidence.get("element_id", "").startswith("el-conv-human-0006-"))
        user = run.runner.improver_request()
        self.assertNotIn("conv-human-0006", user)
        self.assertNotIn(MARKER, user)
        self.assertIn(
            "A cited turn violates this criterion. required-rule: state the command you ran", user
        )

    def test_with_k_1_targets_fixes_holdouts_and_the_unseen_count_change_as_stated(self):
        run = daily(k=1)
        self.assert_fixture(run)
        m = run.metrics
        self.assertEqual((m["target_cases"], m["fixes"], m["holdout_cases"]), (3, 3, 13))
        holdout_conversations = {
            o["case"]["conversation_id"]
            for o in run.validation["outcomes"]
            if o["case"]["kind"] == "holdout"
        }
        self.assertNotIn("conv-human-0006", holdout_conversations)
        unseen = run.validation["unseen"]
        self.assertEqual((unseen["fixed"], len(unseen["withheld_refs"])), (1, 2))
        self.assertTrue(run.validation["gate"]["passed"])

    def test_only_the_withheld_pairs_are_excluded_from_the_holdouts(self):
        """Exclusion is per pair, not per conversation.

        The two withheld failure pairs are never holdouts. The withheld
        conversation's passing pair is not excluded: with no shown target left on
        it, the conversation competes for an extra holdout slot. At the default 3
        slots it loses (13 holdouts above); with 8 it wins one and the count stays 14.
        """
        run = daily(k=1, extra_slots=8)
        self.assert_fixture(run)
        holdouts = [
            "{conversation_id}::{criterion_id}".format(**o["case"])
            for o in run.validation["outcomes"]
            if o["case"]["kind"] == "holdout"
        ]
        self.assertEqual(ref_keys(run.validation["unseen"]["withheld_refs"]), W_KEYS)
        for key in W_KEYS:
            self.assertNotIn(key.split("::", 1)[1], holdouts)
        self.assertIn("conv-human-0006::honours-explicit-constraint", holdouts)
        self.assertEqual((run.metrics["holdout_cases"], len(holdouts)), (14, 14))

    def test_review_prints_the_unseen_line_and_each_unseen_case(self):
        run = daily(k=1)
        shown = run.cli("review", CID)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        lines = shown.stdout.splitlines()
        (at,) = [i for i, line in enumerate(lines) if line.strip().startswith("metrics:")]
        self.assertEqual(lines[at + 1], f"  unseen failures fixed: 1/2 ({SCOPE})")
        rows = [line for line in lines if "[unseen]" in line]
        self.assertEqual(len(rows), 2, rows)
        self.assertEqual(sum("FAIL -> FAIL" in r for r in rows), 1, rows)
        self.assertEqual(sum("FAIL -> PASS" in r for r in rows), 1, rows)
        report = json.loads(run.cli("--json", "review", CID).stdout)
        kinds = [o["case"]["kind"] if "case" in o else o["kind"] for o in report["outcomes"]]
        self.assertEqual(set(kinds), {"target", "holdout"})
        self.assertEqual(kinds.count("target"), run.metrics["target_cases"])
        self.assertEqual(kinds.count("holdout"), run.metrics["holdout_cases"])

    def test_the_record_the_receipt_the_note_and_the_json_agree(self):
        run = daily(k=1)
        unseen = run.validation["unseen"]
        expected = ("measured", 2, 2, 1)
        self.assertEqual(
            (unseen["status"], len(unseen["withheld_refs"]), unseen["replayed"], unseen["fixed"]),
            expected,
        )
        entry = run.entry["unseen_failures"]
        self.assertEqual(
            (entry["status"], entry["withheld"], entry["replayed"], entry["fixed"]), expected
        )
        (note,) = run.notes("unseen failures fixed")
        self.assertIn(f"candidate {CID}: unseen failures fixed: 1/2 (", note)
        report = json.loads(run.cli("--json", "review", CID).stdout)["unseen"]
        self.assertEqual(
            (report["status"], len(report["withheld"]), report["replayed"], report["fixed"]),
            expected,
        )

    def test_review_reads_the_unseen_count_from_the_record_not_the_receipt(self):
        run = run_once(k=1)
        path = run.state.receipt_path("2026-01-02", dry_run=False)
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = [e for e in raw["candidates"] if e.get("candidate_id") == CID]
        self.assertEqual([e["unseen_failures"]["fixed"] for e in entries], [1])
        entries[0]["unseen_failures"]["fixed"] = 777
        path.write_text(json.dumps(raw), encoding="utf-8")
        shown = run.cli("--json", "review", CID)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertEqual(json.loads(shown.stdout)["unseen"]["fixed"], 1)
        text = run.cli("review", CID)
        self.assertEqual(unseen_line(text.stdout), [f"  unseen failures fixed: 1/2 ({SCOPE})"])

    def test_evidence_model_calls_counts_each_distinct_pair_once(self):
        record = daily(k=1).validation
        rows = record["outcomes"] + record["unseen"]["outcomes"]
        pairs = {(o["case"]["conversation_id"], o["rubric_id"]) for o in rows}
        expected = 2 * len(pairs) + len(record["judgements"])
        self.assertEqual(expected, 25)
        self.assertEqual(record["evidence_model_calls"], expected)

    def test_the_split_is_made_before_the_cap(self):
        run = daily(k=1, max_failures=2)
        unseen = run.validation["unseen"]
        self.assertEqual(unseen["status"], "measured")
        self.assertEqual(ref_keys(unseen["withheld_refs"]), W_KEYS)
        claimed = ref_keys(run.validation["target_refs"])
        self.assertEqual(len(claimed), 2)
        self.assertTrue(set(claimed) <= set(SHOWN_KEYS), claimed)


class AGateFailedCandidateStillReportsItsUnseenCount(_Base):
    def test_min_fixes_4_passes_with_withholding_off_and_fails_with_k_1(self):
        off = daily(k=0, min_fixes=4)
        self.assertEqual(off.candidate.status, "validated")
        self.assertEqual(off.metrics["fixes"], 4)
        on = daily(k=1, min_fixes=4)
        self.assertEqual(on.candidate.status, "gate-failed")
        self.assertEqual(on.metrics["fixes"], 3)
        (note,) = on.notes("unseen failures fixed")
        self.assertIn("unseen failures fixed: 1/2 (", note)

    def test_a_gate_failed_candidate_gets_0_of_2_in_the_note_and_the_receipt(self):
        run = daily(variant="zero", k=1, min_fixes=4)
        self.assertEqual(run.candidate.status, "gate-failed")
        (note,) = run.notes("unseen failures fixed")
        self.assertEqual(note, f"candidate {CID}: unseen failures fixed: 0/2 ({SCOPE})")
        self.assertEqual(
            run.entry["unseen_failures"],
            {"status": "measured", "withheld": 2, "replayed": 2, "fixed": 0, "scope": SCOPE},
        )
        self.assertEqual(run.validation["unseen"]["fixed"], 0)

    def test_review_of_the_gate_failed_candidate_is_still_refused(self):
        run = daily(k=1, min_fixes=4)
        shown = run.cli("review", CID)
        self.assertNotEqual(shown.returncode, 0)
        self.assertIn("gate-failed", shown.stdout + shown.stderr)
        # The architecture check that keeps review validated-only (DESIGN 4.3
        # says doctor; it is selftest's check list, cli.py cmd_selftest) still
        # passes: no review route for a gate-failed candidate was added.
        import argparse
        import io
        from contextlib import redirect_stdout

        from agent_skill_improver import cli as cli_module

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli_module.cmd_selftest(argparse.Namespace(json=True))
        checks = {c["name"]: c["passed"] for c in json.loads(buffer.getvalue())["checks"]}
        self.assertIs(checks["review requires exact status validated"], True)


class TheUnseenCountNeverChangesWhetherApplyProceeds(_Base):
    def test_the_gate_is_identical_with_0_of_2_and_2_of_2(self):
        zero, two = daily(variant="zero", k=1), daily(variant="two", k=1)
        self.assertEqual(zero.validation["gate"], two.validation["gate"])
        m = zero.metrics
        self.assertEqual((m["target_cases"], m["fixes"], m["holdout_cases"]), (3, 3, 13))
        self.assertTrue(zero.validation["gate"]["passed"])
        self.assertEqual(zero.validation["unseen"]["fixed"], 0)
        self.assertEqual(two.validation["unseen"]["fixed"], 2)

    def _apply(self, variant: str) -> None:
        run = daily(variant=variant, k=1)
        candidate = run.candidate
        token = json.loads(run.cli("--json", "review", CID).stdout)["approval_token"]
        done = run.cli(
            "apply", CID, "--approve", token, "--i-reviewed-this", "--operator", "c240-test"
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        skill = Path(candidate.target.canonical_path).read_text(encoding="utf-8")
        self.assertEqual(skill, candidate.patch.new_text)

    def test_apply_is_refused_when_the_unseen_block_was_edited(self):
        run = daily(k=1, min_fixes=1)
        candidate = run.candidate
        token = json.loads(run.cli("--json", "review", CID).stdout)["approval_token"]
        skill = Path(candidate.target.canonical_path)
        before = skill.read_bytes()
        edit_unseen_verdict(run)
        for args in (
            ("review", CID),
            ("apply", CID, "--approve", token, "--i-reviewed-this", "--operator", "c240-test"),
        ):
            with self.subTest(command=args[0]):
                refused = run.cli(*args)
                self.assertNotEqual(refused.returncode, 0)
                self.assertIn("CandidateTampered", refused.stderr)
                self.assertIn("unseen.outcomes[0]", refused.stderr)
        self.assertEqual(skill.read_bytes(), before)

    def test_a_coherent_unseen_verdict_edit_is_accepted_and_an_incoherent_one_refused(self):
        """The disclosed limit, paired with its refusal: consistency is checked, not origin.

        The scorer's verdicts are not authenticated. Rewriting an unseen verdict
        together with what it derives (the outcome's flags, the block's fixed count)
        is a coherent record: review shows the edited count and apply proceeds, the
        gate untouched. The change word review prints is derived from the verdict
        pair and is not stored, so writing one into an outcome is refused as an
        unknown key. Rewriting the verdict alone is refused by both, and the skill
        file is left as it was.
        """
        from agent_skill_improver.domain.rubric import Verdict
        from agent_skill_improver.domain.validation import CASE_UNSEEN, ReplayCase, ReplayOutcome

        run = run_once(k=1)
        self.assert_fixture(run)
        path, skill = candidate_file(run), Path(run.candidate.target.canonical_path)
        good, before = path.read_text(encoding="utf-8"), skill.read_bytes()
        token = json.loads(run.cli("--json", "review", CID).stdout)["approval_token"]
        apply = ("apply", CID, "--approve", token, "--i-reviewed-this", "--operator", "c240-test")

        edit_unseen_verdict(run)
        for args in (("review", CID), apply):
            with self.subTest(edit="incoherent", command=args[0]):
                refused = run.cli(*args)
                self.assertNotEqual(refused.returncode, 0)
                self.assertIn("CandidateTampered", refused.stderr)
                self.assertIn("unseen.outcomes[0]", refused.stderr)
        self.assertEqual(skill.read_bytes(), before)

        raw = json.loads(good)
        gate, block = raw["validation"]["gate"], raw["validation"]["unseen"]
        recorded = block["fixed"]
        row = block["outcomes"][0]
        edited = ReplayOutcome(
            case=ReplayCase(
                row["case"]["conversation_id"], row["case"]["criterion_id"], CASE_UNSEEN
            ),
            baseline=Verdict(row["baseline"]),
            candidate=Verdict.PASS if row["candidate"] != "PASS" else Verdict.FAIL,
            rubric_id=row["rubric_id"],
        )
        row.update(edited.to_dict())
        block["fixed"] = sum(1 for o in block["outcomes"] if o["fixed"])
        self.assertNotEqual(block["fixed"], recorded)
        path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shown = run.cli("--json", "review", CID)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        report = json.loads(shown.stdout)
        self.assertEqual(report["unseen"]["fixed"], block["fixed"])
        self.assertEqual(report["gate"], gate)
        done = run.cli(*apply[:3], report["approval_token"], *apply[4:])
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(skill.read_text(encoding="utf-8"), run.candidate.patch.new_text)

    def test_apply_proceeds_with_0_of_2(self):
        self._apply("zero")

    def test_apply_proceeds_with_2_of_2(self):
        self._apply("two")


class TheWithholdingIsFromThePromptOnly(_Base):
    def test_the_improver_command_can_still_read_the_withheld_failure_from_the_state_directory(
        self,
    ):
        from agent_skill_improver.adapters.sandbox import OsSandbox

        tree = make_tree(k=1)
        cfg = tree / "config" / "example.config.toml"
        config = load_config(cfg)
        state = build_state(config)
        clock = FrozenClock(datetime(2026, 1, 2, tzinfo=timezone.utc))
        state._clock = clock

        class Reading(Runner):
            def run(self, *, role, system, user, response_schema):
                reply = super().run(
                    role=role, system=system, user=user, response_schema=response_schema
                )
                if role == "improver" and "'example-evidence-check'" in user:
                    found = [
                        f
                        for f in (tree / "state" / "failures").rglob("*")
                        if f.is_file() and "conv-human-0006" in f.read_text(encoding="utf-8")
                    ]
                    if found:
                        reply["hypothesis"] += " read from state: conv-human-0006"
                return reply

        runner = Reading()
        run_daily(
            config,
            source=build_source(config, state),
            runner=runner,
            skill_store=build_skill_store(config),
            state=state,
            clock=clock,
            day="2026-01-02",
            commit=True,
        )
        self.assertIn("read from state: conv-human-0006", state.read_candidate(CID).hypothesis)
        self.assertNotIn("conv-human-0006", runner.improver_request())
        profile = object.__new__(OsSandbox).profile_text(tree)  # reads only its argument
        self.assertIn("(version 1)", profile)
        self.assertNotIn("deny file-read", profile)

    def test_a_withheld_instance_an_earlier_edit_wrote_into_the_skill_reaches_the_prompt(self):
        """The visible limit: the prompt shows the current skill text whole.

        The withholding leaves out the withheld cases' failure rows and excerpts;
        a line an earlier edit put into SKILL.md naming the withheld conversation
        and its failure words is still in CURRENT SKILL TEXT. A fix that filters
        the skill text flips the last two assertions.
        """
        planted = f"Historical example: conv-human-0006 - {MARKER}."
        run = run_once(k=1, skill_suffix=f"\n{planted}\n")
        self.assert_fixture(run)
        unseen = run.validation["unseen"]
        self.assertEqual(unseen["status"], "measured")
        self.assertEqual(ref_keys(unseen["withheld_refs"]), W_KEYS)
        self.assertEqual(sorted(triple(r) for r in improver_failures(run)), SHOWN_KEYS)
        user = run.runner.improver_request()
        skill_text = skill_section(user)
        self.assertNotIn("conv-human-0006", user.replace(skill_text, ""))
        self.assertIn(planted, skill_text)

    def test_the_skill_section_check_sees_an_id_in_the_excerpts_and_not_one_in_the_skill_text(
        self,
    ):
        """The control of the limit test's section check, on the real prompt, both ways.

        An id injected at either end of CITED CONVERSATION EXCERPTS is found outside
        ``skill_section``, so a section that swallowed the excerpts fails here; the
        same id injected at the end of the skill text is inside it, so a section that
        stopped short fails here too.
        """
        user = daily(k=1).runner.improver_request()
        self.assertNotIn("conv-human-0006", user)
        heading = "## CITED CONVERSATION EXCERPTS"
        self.assertEqual((user.count(heading), user.count("## OUTPUT CONTRACT")), (1, 1))
        leak = "leaked withheld id conv-human-0006\n"
        start = user.index("\n", user.index(heading)) + 1  # after the heading's own line
        for name, injected in (
            ("excerpts-start", user[:start] + leak + user[start:]),
            ("excerpts-end", user.replace("## OUTPUT CONTRACT", f"{leak}## OUTPUT CONTRACT")),
        ):
            with self.subTest(name):
                self.assertEqual(injected.count(leak), 1)
                self.assertNotIn("conv-human-0006", skill_section(injected))
                self.assertIn("conv-human-0006", injected.replace(skill_section(injected), ""))
        in_skill = user.replace(heading, f"{leak}{heading}")
        self.assertIn("conv-human-0006", skill_section(in_skill))
        self.assertNotIn("conv-human-0006", in_skill.replace(skill_section(in_skill), ""))

    def test_the_scope_is_printed_next_to_every_number(self):
        run = daily(k=1)
        (note,) = run.notes("unseen failures fixed")
        self.assertTrue(note.endswith(f"({SCOPE})"), note)
        self.assertEqual(run.entry["unseen_failures"]["scope"], SCOPE)
        self.assertIn(f"({SCOPE})", "\n".join(unseen_line(run.cli("review", CID).stdout)))
        self.assertEqual(
            json.loads(run.cli("--json", "review", CID).stdout)["unseen"]["scope"], SCOPE
        )
        self.assertNotIn("scope", run.validation["unseen"])


class BelowTheThresholdNothingIsWithheld(_Base):
    def test_n_6_k_1_is_too_few_over_four_conversations_and_the_improver_sees_all_five(self):
        run = daily(k=1, n=6)
        first = min(
            FAILURE_CONVERSATIONS,
            key=lambda c: hashlib.sha256(f"{SEED}\x00unseen\x00{c}".encode()).hexdigest(),
        )
        self.assertEqual(first, "conv-human-0006")
        unseen = run.validation["unseen"]
        self.assertEqual(unseen["status"], "too-few")
        self.assertEqual(len(unseen["eligible_refs"]), 5)
        self.assertEqual(
            (unseen["withheld_refs"], unseen["outcomes"], unseen["replayed"]), ([], [], 0)
        )
        rows = improver_failures(run)
        self.assertEqual(len(rows), 5)
        self.assertIn(W_KEYS[0], [triple(r) for r in rows])
        self.assertIn(W_KEYS[1], ref_keys(run.validation["target_refs"]))
        m = run.metrics
        self.assertEqual((m["target_cases"], m["fixes"], m["holdout_cases"]), (5, 4, 14))
        self.assertEqual(run.validation["evidence_model_calls"], 29)
        self.assertEqual(
            unseen_line(run.cli("review", CID).stdout),
            [
                "  unseen failures fixed: not measured (the skill had 5 failure(s), fewer than "
                "[caps].unseen_min_failures = 6)"
            ],
        )

    def test_k_0_writes_v3_and_no_unseen_line(self):
        from tests.support import run_cli

        with tempfile.TemporaryDirectory() as tmp:
            cfg, record = support_run(Path(tmp))
            self.assertNotIn("unseen_", cfg.read_text(encoding="utf-8"))
            v = record["validation"]
            self.assertEqual(v["schema"], "asi.validation.v3")
            self.assertNotIn("unseen", v)
            self.assertEqual(
                (v["gate"]["metrics"]["fixes"], v["gate"]["metrics"]["holdout_cases"]), (2, 14)
            )
            self.assertEqual(v["evidence_model_calls"], 10)
            shown = run_cli(["--config", str(cfg), "review", record["id"]])
            self.assertEqual(shown.returncode, 0, shown.stderr)
            self.assertEqual(unseen_line(shown.stdout), [])

    def test_n_2_k_1_is_one_conversation(self):
        from tests.support import run_cli

        with tempfile.TemporaryDirectory() as tmp:
            cfg, record = support_run(
                Path(tmp), unseen="unseen_min_failures = 2\nunseen_withhold = 1\n"
            )
            unseen = record["validation"]["unseen"]
            self.assertEqual(unseen["status"], "one-conversation")
            self.assertEqual(unseen["withheld_refs"], [])
            self.assertEqual(len(unseen["eligible_refs"]), 2)
            self.assertEqual(
                ref_keys(record["validation"]["target_refs"]), ref_keys(unseen["eligible_refs"])
            )
            shown = run_cli(["--config", str(cfg), "review", record["id"]])
            self.assertEqual(shown.returncode, 0, shown.stderr)
            self.assertEqual(
                unseen_line(shown.stdout),
                [
                    "  unseen failures fixed: not measured (all 2 failure(s) are on one "
                    "conversation, so none can be withheld while one stays shown)"
                ],
            )


# The review keys of the build before withholding (b07045f), and the length and
# sha256 of its `--json review` of the frozen v3 fixture in a make_tree tree (its
# config without the two unseen keys, which that build refuses), the tree's path
# written {TREE}.
BEFORE_REVIEW_KEYS = [
    "added_lines",
    "apply_command",
    "approval_token",
    "base_sha256",
    "candidate_id",
    "canonical_path",
    "diff",
    "gate",
    "hypothesis",
    "intended_change",
    "judgements",
    "mirror_paths",
    "notes",
    "outcomes",
    "patch_sha256",
    "removed_lines",
    "risks",
    "rollback",
    "rubric_failures",
    "skill_id",
    "sources",
    "status",
]
BEFORE_REVIEW_JSON = (
    12108,
    "1485e771975a73352399aef99493199bdac8fc63e6e1dcb1bc51f9f458aff3bf",
)
NOT_RECORDED = {"status": "not-recorded", "schema": "asi.validation.v3"}


class TheFrozenV3Record(_Base):
    def frozen_run(self) -> tuple[Run, Candidate, Path]:
        """The frozen v3 fixture written into a k = 0 tree, with a k = 1 config beside it."""
        tree = make_tree(k=0)
        frozen = (REPO_ROOT / "tests/fixtures/validation-v3-record.json").read_text(
            encoding="utf-8"
        )
        candidate = Candidate.from_dict(json.loads(frozen.replace("{TREE}", str(tree))))
        base = Path(candidate.target.canonical_path)
        self.assertEqual(
            hashlib.sha256(base.read_bytes()).hexdigest(), candidate.target.base_sha256
        )
        self.assertTrue(candidate.target.base_sha256.startswith("f77de6c6"))
        cfg = tree / "config" / "example.config.toml"
        state = build_state(load_config(cfg))
        state.write_candidate(candidate)
        on = tree / "config" / "withhold-on.toml"
        on.write_text(_set(cfg.read_text(encoding="utf-8"), "unseen_withhold", 1), encoding="utf-8")
        return Run(tree=tree, cfg=cfg, receipt=None, runner=Runner(), state=state), candidate, on

    def test_a_frozen_v3_candidate_is_reviewed_and_applied_unchanged(self):
        run, candidate, on = self.frozen_run()
        base = Path(candidate.target.canonical_path)
        state = run.state
        shown = run.cli("--json", "review", CID)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        report = json.loads(shown.stdout)
        self.assertEqual(report["approval_token"], "d9a9f4534d53af0d")
        self.assertNotIn("unseen", report)
        self.assertEqual(unseen_line(run.cli("review", CID).stdout), [])
        shown_on = run.cli("--json", "review", CID, cfg=on)
        self.assertEqual(shown_on.returncode, 0, shown_on.stderr)
        self.assertEqual(json.loads(shown_on.stdout)["unseen"], NOT_RECORDED)
        text = run.cli("review", CID, cfg=on)
        self.assertEqual(text.returncode, 0, text.stderr)
        (line,) = unseen_line(text.stdout)
        self.assertIn("not recorded (an asi.validation.v3 record", line)
        self.assertIn("d9a9f4534d53af0d", text.stdout)
        done = run.cli(
            "apply",
            CID,
            "--approve",
            "d9a9f4534d53af0d",
            "--i-reviewed-this",
            "--operator",
            "fixture-review",
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(base.read_text(encoding="utf-8"), candidate.patch.new_text)
        self.assertEqual(state.read_candidate(CID).status, "applied")

    def test_a_k_0_review_json_is_byte_identical_to_the_build_before_withholding(self):
        """Off is byte-for-byte off in `review --json` too, pinned against b07045f.

        The length and digest are of the build before withholding reviewing this same
        record in this same tree. Two-sided: with k = 1 the report is the same one
        plus `unseen` (not-recorded), so a check that could not see the key fails there.
        """
        run, _, on = self.frozen_run()
        off = run.cli("--json", "review", CID)
        self.assertEqual(off.returncode, 0, off.stderr)
        report = json.loads(off.stdout)
        self.assertEqual(sorted(report), BEFORE_REVIEW_KEYS)
        pinned = off.stdout.replace(str(run.tree), "{TREE}").encode("utf-8")
        self.assertEqual(
            (len(pinned), hashlib.sha256(pinned).hexdigest()), BEFORE_REVIEW_JSON, off.stdout
        )
        shown_on = run.cli("--json", "review", CID, cfg=on)
        self.assertEqual(shown_on.returncode, 0, shown_on.stderr)
        report_on = json.loads(shown_on.stdout)
        self.assertEqual(sorted(report_on), sorted(BEFORE_REVIEW_KEYS + ["unseen"]))
        self.assertEqual(report_on.pop("unseen"), NOT_RECORDED)
        self.assertEqual(report_on, report)
        self.assertNotEqual(shown_on.stdout, off.stdout)


class TheDowngradeLimit(_Base):
    def test_a_v4_record_rewritten_as_v3_is_reviewable_and_says_not_recorded(self):
        run = daily(k=1, n=4)
        good = json.loads(candidate_file(run).read_text(encoding="utf-8"))
        edit_unseen_verdict(run)
        refused = run.cli("review", CID)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("unseen.outcomes[0]", refused.stderr)
        v = good["validation"]
        del v["unseen"]
        v["schema"] = "asi.validation.v3"
        pairs = {(o["case"]["conversation_id"], o["rubric_id"]) for o in v["outcomes"]}
        v["evidence_model_calls"] = 2 * len(pairs) + len(v["judgements"])
        candidate_file(run).write_text(
            json.dumps(good, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shown = run.cli("review", CID)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        (line,) = unseen_line(shown.stdout)
        self.assertIn("not recorded (an asi.validation.v3 record", line)
        token = json.loads(run.cli("--json", "review", CID).stdout)["approval_token"]
        self.assertIn(token, shown.stdout)
        done = run.cli(
            "apply", CID, "--approve", token, "--i-reviewed-this", "--operator", "c240-test"
        )
        self.assertEqual(done.returncode, 0, done.stderr)


class TheConfigBound(_Base):
    def test_the_withhold_must_be_smaller_than_the_minimum_when_on(self):
        absent = load_config(make_tree(k=None, n=None) / "config" / "example.config.toml")
        self.assertEqual((absent.caps.unseen_withhold, absent.caps.unseen_min_failures), (0, 4))
        for k, n in ((4, 4), (5, 4)):
            with self.subTest(k=k, n=n):
                cfg = make_tree(k=k, n=n) / "config" / "example.config.toml"
                with self.assertRaises(ConfigError) as caught:
                    load_config(cfg)
                self.assertIn(
                    "[caps].unseen_withhold (k) must be smaller than [caps].unseen_min_failures "
                    f"(N), got k={k} N={n}",
                    str(caught.exception),
                )
        loaded = load_config(make_tree(k=0, n=2) / "config" / "example.config.toml")
        self.assertEqual((loaded.caps.unseen_withhold, loaded.caps.unseen_min_failures), (0, 2))
        with self.assertRaises(ConfigError) as caught:
            load_config(make_tree(k=0, n=1) / "config" / "example.config.toml")
        self.assertIn("[caps].unseen_min_failures must be an integer >= 2", str(caught.exception))


class TheUnseenCaseIsNeverJudged(_Base):
    def test_the_judge_is_never_asked_about_an_unseen_case(self):
        run = daily(k=1)
        judged = run.runner.users("judge")
        self.assertTrue(judged)
        for user in judged:
            self.assertNotIn("conv-human-0006", user)
        self.assertEqual(len(run.validation["judgements"]), run.metrics["target_cases"])
        self.assertEqual(len(run.validation["judgements"]), 3)
        withheld = {tuple(r.values()) for r in run.validation["unseen"]["withheld_refs"]}
        for j in run.validation["judgements"]:
            self.assertNotIn(j["case"]["conversation_id"], {w[1] for w in withheld})


def tearDownModule() -> None:
    # unittest calls this each time a run leaves this module, which happens more
    # than once when test ids of two modules are interleaved on one command line:
    # the cached runs go with their trees, and _fresh_dir recreates the root.
    _RUNS.clear()
    _TMP.cleanup()


if __name__ == "__main__":
    unittest.main()
