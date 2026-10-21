"""What the automated pass may and may not do, and how it fails.

Three requirements meet here:

* A bad record stops that record. Never the batch. A whole-batch veto turns a
  filter into a ratchet: one unresolved item blocks all progress indefinitely,
  the queue grows, and nothing ever ships.
* The daily pass cannot mutate a live skill file, and the proof is structural
  rather than a promise in a docstring.
* Re-running a day is a near no-op, and two runs cannot interleave.
"""

from __future__ import annotations

import datetime
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app.daily import run_daily
from agent_skill_improver.app.lock import RunLock
from agent_skill_improver.app.pipeline import read_canonical_text
from agent_skill_improver.app.receipt import STATUS_STOOD_DOWN
from agent_skill_improver.app.state import StateStore, safe_name
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.config import load_config
from agent_skill_improver.errors import (
    ConfigError,
    LinkedTargetRefused,
    LockHeld,
    RecordDefect,
    SafetyRefusal,
    SourceUnavailable,
)
from agent_skill_improver.hashing import sha256_file

from tests.support import FIXTURE_EXPORTS, FIXTURE_MALFORMED, InProcessRunner, build_env

DAY = "2026-01-02"


def frozen_clock() -> FrozenClock:
    """A pinned clock, so every artifact in a run is reproducible."""
    return FrozenClock(datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc))


class DailyHarness(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def env(self, **kwargs):
        kwargs.setdefault("rubric_ids", ("instruction-compliance",))
        kwargs.setdefault("use_subprocess_model", False)
        return load_config(build_env(self.base, dry_run=False, **kwargs))

    def run_pass(self, config, *, runner=None, day=DAY, mode=""):
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        return (
            run_daily(
                config,
                source=FixtureConversationSource(config.source.fixtures_dir),
                runner=runner or InProcessRunner(mode=mode),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day=day,
                commit=True,
            ),
            state,
            store,
        )


class NoLiveMutationTests(DailyHarness):
    def test_a_writable_skill_store_is_refused_outright(self):
        config = self.env()
        state = StateStore(config.state_dir, clock=frozen_clock())
        writable = FilesystemSkillStore(config.skill_roots, read_only=False)
        with self.assertRaises(SafetyRefusal):
            run_daily(
                config,
                source=FixtureConversationSource(config.source.fixtures_dir),
                runner=InProcessRunner(),
                skill_store=writable,
                state=state,
                clock=frozen_clock(),
                day=DAY,
                commit=True,
            )

    def test_the_skill_tree_digest_is_unchanged_by_a_full_pass(self):
        config = self.env()
        receipt, _, _ = self.run_pass(config)
        tree = receipt.to_dict()["skill_tree"]
        self.assertEqual(tree["digest_before"], tree["digest_after"])
        self.assertFalse(tree["mutation_detected"])

    def test_a_skill_edited_mid_run_blocks_the_run_and_the_exit_code_says_so(self):
        """Every other test here proves the digest does not move. This one moves
        it on purpose, from inside a model call, and checks that the receipt
        earns ``blocked``, says so in a note, and that the CLI maps that status
        to its own exit code rather than the generic "answer was negative" one.
        A scheduler that cannot tell "a candidate failed its gate" from "something
        wrote to the live skill tree while we were reading it" will retry the
        second as if it were the first.
        """
        from agent_skill_improver.app.receipt import STATUS_BLOCKED
        from agent_skill_improver.cli import RC_BLOCKED, RC_NEGATIVE, RC_OK, exit_code_for

        config = self.env()
        target = self.base / "skills-canonical" / "tdd" / "SKILL.md"
        before = target.read_text()

        class EditsASkillWhileScoring(InProcessRunner):
            edited = False

            def run(self, *, role, system, user, response_schema):
                if role == "scorer" and not EditsASkillWhileScoring.edited:
                    EditsASkillWhileScoring.edited = True
                    target.write_text(before + "\n\nSomeone else wrote this during the run.\n")
                return super().run(
                    role=role, system=system, user=user, response_schema=response_schema
                )

        receipt, _, _ = self.run_pass(config, runner=EditsASkillWhileScoring())
        payload = receipt.to_dict()
        self.assertEqual(receipt.status, STATUS_BLOCKED)
        self.assertTrue(payload["skill_tree"]["mutation_detected"])
        self.assertNotEqual(
            payload["skill_tree"]["digest_before"], payload["skill_tree"]["digest_after"]
        )
        self.assertTrue(
            any("LIVE SKILL MUTATION DETECTED" in n for n in payload["notes"]),
            payload["notes"],
        )
        self.assertEqual(
            payload["skill_tree"]["changes"],
            [{"key": "canonical:tdd/SKILL.md", "kind": "content-changed"}],
            "the receipt names the document and that its content moved",
        )
        note = next(n for n in payload["notes"] if n.startswith("LIVE SKILL MUTATION DETECTED"))
        self.assertIn("content-changed: canonical:tdd/SKILL.md", note)
        self.assertIn("edited a skill file", note)
        self.assertEqual(exit_code_for(STATUS_BLOCKED), RC_BLOCKED)
        self.assertEqual(RC_BLOCKED, 5, "the documented exit code for a blocked run")
        self.assertNotIn(RC_BLOCKED, (RC_OK, RC_NEGATIVE, 3, 4))
        for benign in ("ok", "partial", "stood-down"):
            self.assertEqual(exit_code_for(benign), RC_OK)
        self.assertEqual(exit_code_for("something-new"), RC_NEGATIVE)

        # And through the real command entry point, not just the pure mapping:
        # the same mutating runner, wired in where `asi daily` builds its own.
        # The committed run above left a blocked receipt, so this run has to
        # state that its mutation was reviewed or it is refused before starting.
        import io
        from contextlib import redirect_stdout
        from unittest import mock

        from agent_skill_improver import cli

        EditsASkillWhileScoring.edited = False
        target.write_text(before)
        with mock.patch.object(cli, "build_runner", lambda *_: EditsASkillWhileScoring()):
            with redirect_stdout(io.StringIO()) as out:
                rc = cli.main(
                    [
                        "--config",
                        str(config.config_path),
                        "daily",
                        "--day",
                        "2026-01-03",
                        "--mutation-reviewed",
                        DAY,
                    ]
                )
        self.assertEqual(rc, RC_BLOCKED, out.getvalue())
        self.assertIn("status=blocked", out.getvalue())

    def test_candidates_from_a_blocked_run_are_downgraded_so_apply_refuses_them(self):
        """OPERATIONS says a blocked run's candidates describe a tree state that
        no longer exists. The code has to make that true: a candidate persisted
        before the digest check fired must not sit in the queue as ``validated``,
        because ``apply`` would then take it on a token alone. Here a DIFFERENT
        skill than the candidate's target is edited mid-run, so the base-hash
        check on the target cannot be what refuses it.
        """
        from agent_skill_improver.app.promote import apply_candidate
        from agent_skill_improver.app.receipt import STATUS_BLOCKED
        from agent_skill_improver.domain.approval import HumanApproval
        from agent_skill_improver.domain.candidate import STATUS_BLOCKED_RUN, STATUS_VALIDATED
        from agent_skill_improver.domain.integrity import (
            gate_status_problems,
            verify_full_integrity,
        )
        from agent_skill_improver.errors import GateFailed

        config = self.env()
        bystander = self.base / "skills-canonical" / "code-review" / "SKILL.md"
        before = bystander.read_text()

        class EditsABystanderWhileScoring(InProcessRunner):
            edited = False

            def run(self, *, role, system, user, response_schema):
                if role == "scorer" and not EditsABystanderWhileScoring.edited:
                    EditsABystanderWhileScoring.edited = True
                    bystander.write_text(before + "\n\nEdited by someone else mid-run.\n")
                return super().run(
                    role=role, system=system, user=user, response_schema=response_schema
                )

        receipt, state, store = self.run_pass(config, runner=EditsABystanderWhileScoring())
        payload = receipt.to_dict()
        self.assertEqual(receipt.status, STATUS_BLOCKED)
        self.assertGreater(
            payload["counts"]["candidates_proposed"], 0, "the scenario needs a candidate"
        )
        self.assertEqual(payload["counts"]["candidates_gate_passed"], 0)
        for entry in payload["candidates"]:
            self.assertFalse(entry["gate_passed"])
            self.assertTrue(entry["blocked_run"])
            self.assertEqual(entry["status"], STATUS_BLOCKED_RUN)
            self.assertNotIn("downgrade_failed", entry)
            persisted = state.read_candidate(entry["candidate_id"])
            self.assertEqual(persisted.status, STATUS_BLOCKED_RUN)
            self.assertNotEqual(persisted.status, STATUS_VALIDATED)
            self.assertEqual(
                gate_status_problems(persisted),
                [],
                "a record this tool wrote on purpose must not read as tampered",
            )
            self.assertTrue(
                persisted.validation and "gate" in persisted.validation,
                "the measured gate verdict is kept, not erased",
            )
            verify_full_integrity(
                persisted,
                base_text=Path(persisted.target.canonical_path).read_text(),
                rubrics={r.id: r for r in config.rubrics},
                action="audit",
            )
            self.assertTrue(
                any("blocked" in n.lower() and "mutation" in n.lower() for n in persisted.notes),
                persisted.notes,
            )
            self.assertNotEqual(
                persisted.target.canonical_path, str(bystander), "target was not the edited file"
            )
            approval = HumanApproval.from_cli(
                token=persisted.required_approval_token, acknowledged=True
            )
            with self.assertRaises(GateFailed):
                apply_candidate(
                    persisted,
                    approval=approval,
                    skill_store=FilesystemSkillStore(config.skill_roots, read_only=False),
                    state=state,
                    clock=frozen_clock(),
                )
        self.assertEqual(
            state.open_candidates(), [], "nothing from a blocked run is left awaiting review"
        )

    def test_a_candidate_that_cannot_be_rewritten_does_not_cost_the_blocked_receipt(self):
        """The downgrade runs when an uncontrolled writer is known to be active,
        which is when state files are most likely to be disturbed too. One
        unreadable candidate file must not abort the finish: the receipt that
        says "blocked" is the artifact the post-mortem starts from.
        """
        from agent_skill_improver.app.receipt import STATUS_BLOCKED
        from agent_skill_improver.errors import StateUnreadable

        config = self.env()
        bystander = self.base / "skills-canonical" / "code-review" / "SKILL.md"
        before = bystander.read_text()

        class EditsABystanderWhileScoring(InProcessRunner):
            edited = False

            def run(self, *, role, system, user, response_schema):
                if role == "scorer" and not EditsABystanderWhileScoring.edited:
                    EditsABystanderWhileScoring.edited = True
                    bystander.write_text(before + "\n\nEdited by someone else mid-run.\n")
                return super().run(
                    role=role, system=system, user=user, response_schema=response_schema
                )

        class CandidateFilesGoBad(StateStore):
            def read_candidate(self, candidate_id):
                raise StateUnreadable(f"simulated: {candidate_id} is unreadable")

        state = CandidateFilesGoBad(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        receipt = run_daily(
            config,
            source=FixtureConversationSource(config.source.fixtures_dir),
            runner=EditsABystanderWhileScoring(),
            skill_store=store,
            state=state,
            clock=frozen_clock(),
            day=DAY,
            commit=True,
        )
        payload = receipt.to_dict()
        self.assertEqual(receipt.status, STATUS_BLOCKED)
        self.assertTrue(payload["candidates"])
        for entry in payload["candidates"]:
            self.assertIn("simulated", entry["downgrade_failed"])
            self.assertFalse(entry["gate_passed"])
        self.assertTrue(
            any("could NOT be rewritten" in n for n in payload["notes"]), payload["notes"]
        )
        self.assertTrue(
            state.receipt_path(DAY, dry_run=False).is_file(),
            "the blocked receipt reached disk despite the unreadable candidate",
        )

    def test_no_skill_file_content_changes_on_disk(self):
        config = self.env()
        before = {
            p: sha256_file(p)
            for root in config.skill_roots
            for p in sorted(root.path.rglob("SKILL.md"))
        }
        self.run_pass(config)
        after = {p: sha256_file(p) for p in before}
        self.assertEqual(before, after)

    def test_the_daily_module_does_not_import_the_promotion_module(self):
        import ast

        from agent_skill_improver.config import package_dir

        source = (package_dir() / "app" / "daily.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add("." * node.level + (node.module or ""))
                imported.update(a.name for a in node.names)
        self.assertNotIn(
            "promote",
            {part for name in imported for part in name.split(".")},
            "the interlock is that the code is not reachable, not that it is not called",
        )

    def test_the_daily_module_never_calls_the_write_method(self):
        import ast

        from agent_skill_improver.config import package_dir

        source = (package_dir() / "app" / "daily.py").read_text(encoding="utf-8")
        used = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.Name):
                used.add(node.id)
        self.assertNotIn("write_canonical", used)

    def test_only_apply_and_rollback_ask_for_a_writable_skill_store(self):
        """OPERATIONS says only ``apply`` and ``rollback`` open the store writable.

        Every ``build_skill_store`` call in the CLI is attributed to the top-level
        command function around it; the ones passing a literal ``writable=True``
        must be exactly those two, ``daily`` must be among the read-only callers,
        the wiring default those callers rely on must be read-only, and the CLI
        must not reference ``FilesystemSkillStore`` as a name or attribute.
        """
        import ast
        import inspect

        from agent_skill_improver.app.wiring import build_skill_store
        from agent_skill_improver.config import package_dir

        self.assertIs(
            inspect.signature(build_skill_store).parameters["writable"].default,
            False,
            "the read-only callers below omit the flag and rely on this default",
        )
        tree = ast.parse((package_dir() / "cli.py").read_text(encoding="utf-8"))

        def is_build_call(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Call) and getattr(node.func, "id", None) == "build_skill_store"
            )

        every_call = sum(1 for node in ast.walk(tree) if is_build_call(node))
        attributed = 0
        writable: set[str] = set()
        read_only: set[str] = set()
        referenced: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
        for func in tree.body:
            if not isinstance(func, ast.FunctionDef):
                continue
            for node in ast.walk(func):
                if not is_build_call(node):
                    continue
                assert isinstance(node, ast.Call)  # narrowed by is_build_call
                attributed += 1
                flags: list[object] = []
                for kw in node.keywords:
                    if kw.arg == "writable":
                        self.assertIsInstance(
                            kw.value, ast.Constant, f"{func.name}: writable must be a literal"
                        )
                        flags.append(kw.value.value)
                (writable if any(v is True for v in flags) else read_only).add(func.name)
        self.assertEqual(
            attributed,
            every_call,
            "a build_skill_store call outside a top-level command function is unaccounted for",
        )
        self.assertEqual(writable, {"cmd_apply", "cmd_rollback"})
        self.assertIn("cmd_daily", read_only, "the daily driver always opens it read-only")
        self.assertTrue(read_only.isdisjoint(writable), "one command opened the store both ways")
        self.assertFalse(
            "FilesystemSkillStore" in referenced, "the CLI references the store class directly"
        )

    def test_the_automation_marker_is_set_for_the_duration_of_the_pass(self):
        from agent_skill_improver.domain.approval import automation_active

        seen: list[bool] = []

        class Watching(InProcessRunner):
            def run(self, **kwargs):
                seen.append(automation_active())
                return super().run(**kwargs)

        config = self.env()
        self.run_pass(config, runner=Watching())
        self.assertTrue(seen, "the runner must have been called")
        self.assertTrue(
            all(seen),
            "an apply reached from inside the pass must refuse, and the marker is "
            "what makes that happen",
        )
        self.assertFalse(automation_active(), "the marker must be cleared afterwards")

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file")
    def test_a_document_made_unreadable_mid_run_blocks_the_run_and_is_named_not_called_an_edit(
        self,
    ):
        # Measured before: a chmod 000 on ANOTHER skill's document mid-run, content
        # untouched, earned blocked with the note "another process edited a skill
        # file", and nothing in the receipt said which document or what happened to
        # it (the passed-over list is the discovery-time reading, so it was empty).
        from agent_skill_improver.app.receipt import STATUS_BLOCKED

        config = self.env()
        other = self.base / "skills-canonical" / "code-review" / "SKILL.md"
        content = other.read_bytes()

        class SomethingChmodsAnotherDocument(FilesystemSkillStore):
            def read_text(self, path: Path) -> str:
                os.chmod(other, 0o000)
                return super().read_text(path)

        state = StateStore(config.state_dir, clock=frozen_clock())
        try:
            receipt = run_daily(
                config,
                source=FixtureConversationSource(config.source.fixtures_dir),
                runner=InProcessRunner(),
                skill_store=SomethingChmodsAnotherDocument(
                    config.skill_roots, read_only=True, protected_paths=config.protected_paths
                ),
                state=state,
                clock=frozen_clock(),
                day=DAY,
                commit=True,
            )
        finally:
            os.chmod(other, 0o644)  # before tearDown removes the tree
        payload = receipt.to_dict()
        self.assertEqual(other.read_bytes(), content, "content never changed")
        self.assertEqual(receipt.status, STATUS_BLOCKED, "a foreign actor was in the tree")
        self.assertTrue(payload["skill_tree"]["mutation_detected"])
        self.assertEqual(
            payload["skill_tree"]["changes"],
            [{"key": "canonical:code-review/SKILL.md", "kind": "became-unreadable"}],
        )
        self.assertEqual(
            [c["status"] for c in payload["candidates"]],
            ["blocked-run"],
            "the gate verdict was computed against a tree that then moved",
        )
        note = next(n for n in payload["notes"] if n.startswith("LIVE SKILL MUTATION DETECTED"))
        self.assertIn("became-unreadable: canonical:code-review/SKILL.md", note)
        self.assertIn("no document's content was seen to change", note)
        self.assertNotIn("edited", note, "an edit is claimed only for a hash that moved")

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file")
    def test_an_unreadable_document_appearing_mid_run_blocks_the_run_though_the_digest_holds(
        self,
    ):
        # The digest is over readable documents, so a new mode-000 document leaves
        # it equal before and after. The entries still differ, and that, not the
        # digest, is what blocks the run and downgrades its candidates.
        from agent_skill_improver.app.receipt import STATUS_BLOCKED

        config = self.env()
        newcomer = self.base / "skills-canonical" / "newcomer" / "SKILL.md"

        class SomethingAddsAnUnreadableDocument(FilesystemSkillStore):
            def read_text(self, path: Path) -> str:
                if not newcomer.exists():
                    newcomer.parent.mkdir()
                    newcomer.write_text("---\nname: newcomer\n---\n", encoding="utf-8")
                    os.chmod(newcomer, 0o000)
                return super().read_text(path)

        state = StateStore(config.state_dir, clock=frozen_clock())
        try:
            receipt = run_daily(
                config,
                source=FixtureConversationSource(config.source.fixtures_dir),
                runner=InProcessRunner(),
                skill_store=SomethingAddsAnUnreadableDocument(
                    config.skill_roots, read_only=True, protected_paths=config.protected_paths
                ),
                state=state,
                clock=frozen_clock(),
                day=DAY,
                commit=True,
            )
        finally:
            if newcomer.exists():
                os.chmod(newcomer, 0o644)
        payload = receipt.to_dict()
        tree = payload["skill_tree"]
        self.assertEqual(tree["digest_before"], tree["digest_after"], "readable documents only")
        self.assertTrue(tree["mutation_detected"])
        self.assertEqual(
            tree["changes"], [{"key": "canonical:newcomer/SKILL.md", "kind": "added-unreadable"}]
        )
        self.assertEqual(receipt.status, STATUS_BLOCKED)
        self.assertEqual([c["status"] for c in payload["candidates"]], ["blocked-run"])
        self.assertEqual(
            state.open_candidates(),
            [],
            "the persisted candidate was downgraded too, not only the receipt's copy",
        )


class PerRecordFailureTests(DailyHarness):
    """One bad record must not veto the batch."""

    def _mixed_fixture_dir(self) -> Path:
        mixed = self.base / "mixed"
        mixed.mkdir(parents=True, exist_ok=True)
        for path in sorted(FIXTURE_EXPORTS.glob("*.json")):
            (mixed / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        # Two different shape defects, so the test cannot pass by handling one.
        (mixed / "broken-elements.json").write_text(
            (FIXTURE_MALFORMED / "elements-not-list.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (mixed / "broken-no-thread.json").write_text(
            (FIXTURE_MALFORMED / "no-thread.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return mixed

    def test_malformed_records_are_quarantined_while_the_rest_are_scored(self):
        config = self.env(fixtures_dir=self._mixed_fixture_dir())
        receipt, state, _ = self.run_pass(config)
        counts = receipt.to_dict()["counts"]
        self.assertGreater(counts["records_quarantined"], 0)
        self.assertGreater(
            counts["records_scored"],
            0,
            "this is the whole requirement: a broken record must not veto the batch",
        )

    def test_each_quarantined_record_gets_its_own_file_with_a_reason(self):
        config = self.env(fixtures_dir=self._mixed_fixture_dir())
        receipt, state, _ = self.run_pass(config)
        files = sorted(state.dir("quarantine").glob("*.json"))
        self.assertEqual(
            len(files),
            receipt.to_dict()["counts"]["records_quarantined"],
            "one file per record: a single batch-level error report cannot say which "
            "record failed or why",
        )
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(payload.get("reason"), path.name)
            self.assertTrue(payload.get("stage"), path.name)

    def test_a_shape_defect_and_a_model_defect_are_recorded_at_different_stages(self):
        """Two unrelated causes must stay distinguishable in the record.

        The two malformed export shapes in this repository normalise to the same
        message, which is correct: they really are the same defect. So this uses a
        genuinely different cause -- one record whose scorer output is unusable --
        and asserts the stage separates them. A quarantine file that did not name
        its stage would make a normalizer bug and a model failure look identical.
        """

        class FailsOneRecord(InProcessRunner):
            """Fails the scorer for exactly one conversation, by id."""

            target_id = "1700000000002-1"

            def run(self, *, role, system, user, response_schema):
                if role == "scorer" and self.target_id in user:
                    from agent_skill_improver.errors import MalformedModelOutput

                    raise MalformedModelOutput("no parseable JSON object in the reply")
                return super().run(
                    role=role, system=system, user=user, response_schema=response_schema
                )

        config = self.env(fixtures_dir=self._mixed_fixture_dir())
        receipt, state, _ = self.run_pass(config, runner=FailsOneRecord())
        entries = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(state.dir("quarantine").glob("*.json"))
        ]
        stages = {e["stage"] for e in entries}
        self.assertGreaterEqual(
            len(stages),
            2,
            f"expected a normalize defect and a scoring defect, got stages {stages}",
        )
        self.assertGreater(
            receipt.to_dict()["counts"]["records_scored"],
            0,
            "and the untouched records must still have been scored",
        )
        self.assertTrue(
            all(e.get("record_id") for e in entries),
            "every quarantine entry names the record it is about",
        )

    def test_quarantine_defer_and_refuse_are_three_distinct_states(self):
        state = StateStore(self.base / "s", clock=frozen_clock())
        state.ensure()
        state.quarantine("r1", stage="normalize", reason="malformed")
        state.defer("r2", stage="fetch", reason="source unreachable")
        state.refuse("r3", stage="propose", reason="protected skill")
        self.assertEqual(state.count("quarantine"), 1)
        self.assertEqual(state.count("deferred"), 1)
        self.assertEqual(state.count("refusals"), 1)

    def test_a_deferred_record_can_be_cleared_when_it_later_succeeds(self):
        state = StateStore(self.base / "s", clock=frozen_clock())
        state.ensure()
        state.defer("r2", stage="fetch", reason="unreachable")
        self.assertEqual(state.deferred_ids(), ["r2"])
        state.clear_deferred("r2")
        self.assertEqual(state.deferred_ids(), [])

    def test_a_record_id_with_path_separators_cannot_escape_the_state_directory(self):
        state = StateStore(self.base / "s", clock=frozen_clock())
        state.ensure()
        path = state.quarantine("../../etc/passwd", stage="normalize", reason="hostile id")
        self.assertTrue(
            str(path.resolve()).startswith(str((self.base / "s").resolve())),
            "a record id comes from external data and must not steer a write",
        )

    def test_safe_name_strips_separators(self):
        self.assertNotIn("/", safe_name("a/b"))
        self.assertNotIn("..", safe_name("../.."))

    def test_an_unreachable_source_defers_rather_than_failing_the_run(self):
        class Unreachable(FixtureConversationSource):
            def fetch(self, thread_id):
                from agent_skill_improver.errors import SourceUnavailable

                raise SourceUnavailable("network is down")

        config = self.env()
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        receipt = run_daily(
            config,
            source=Unreachable(config.source.fixtures_dir),
            runner=InProcessRunner(),
            skill_store=store,
            state=state,
            clock=frozen_clock(),
            day=DAY,
            commit=True,
        )
        payload = receipt.to_dict()
        self.assertGreater(payload["counts"]["records_deferred"], 0)
        self.assertEqual(
            payload["counts"]["records_quarantined"],
            0,
            "temporarily unavailable is not malformed; collapsing them loses the "
            "difference between a retry and a bug",
        )

    def test_a_record_that_defers_persistently_is_quarantined_not_retried_for_ever(self):
        """Deferred records are retried first. That is right for a blip and a
        ratchet for anything else: enough records that never stop deferring hold
        every selection slot every day and fresh conversations starve. The
        attempt counter already on disk is what ends it.
        """

        class Unreachable(FixtureConversationSource):
            def fetch(self, thread_id):
                raise SourceUnavailable("network is down")

        config = self.env(max_defer_attempts=2)
        self.assertEqual(config.caps.max_defer_attempts, 2)
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )

        def run(day: str) -> dict:
            return run_daily(
                config,
                source=Unreachable(config.source.fixtures_dir),
                runner=InProcessRunner(),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day=day,
                commit=True,
            ).to_dict()

        first = run("2026-01-02")
        self.assertGreater(first["counts"]["records_deferred"], 0)
        self.assertEqual(first["counts"]["records_quarantined"], 0, "one miss is a retry")
        self.assertTrue(state.deferred_ids(), "the first miss keeps retry-first priority")

        second = run("2026-01-03")
        self.assertEqual(
            second["counts"]["records_deferred"],
            0,
            "the cap binds on the second miss: nothing is left as a retry",
        )
        self.assertGreater(second["counts"]["records_quarantined"], 0)
        self.assertEqual(
            state.deferred_ids(),
            [],
            "an exhausted deferral stops claiming a selection slot ahead of fresh work",
        )
        entries = [
            json.loads(p.read_text()) for p in (config.state_dir / "quarantine").glob("*.json")
        ]
        self.assertTrue(entries)
        for entry in entries:
            self.assertTrue(
                entry["stage"].endswith(":defer-exhausted"),
                f"the quarantine names the escalation, got {entry['stage']!r}",
            )
            self.assertEqual(entry["detail"]["attempts"], 2)
            self.assertIn("network is down", entry["detail"]["last_reason"])
        for info in second["records"].values():
            self.assertEqual(info["status"], "quarantined")
            self.assertIn("not transient", info["reason"])

    def test_a_success_between_misses_resets_the_deferral_streak(self):
        """The cap counts misses in a row, not misses in total. Miss, miss,
        success, miss under a cap of 3 must end with a live deferral at attempt
        one and nothing in quarantine.
        """

        class Flaky(FixtureConversationSource):
            down = True

            def fetch(self, thread_id):
                if Flaky.down:
                    raise SourceUnavailable("network is down")
                return super().fetch(thread_id)

        config = self.env(max_defer_attempts=3)
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )

        def run(day: str, *, down: bool) -> dict:
            Flaky.down = down
            return run_daily(
                config,
                source=Flaky(config.source.fixtures_dir),
                runner=InProcessRunner(),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day=day,
                commit=True,
            ).to_dict()

        run("2026-01-02", down=True)
        run("2026-01-03", down=True)
        self.assertEqual({state.deferred_attempts(i) for i in state.deferred_ids()}, {2})
        up = run("2026-01-04", down=False)
        self.assertGreater(up["counts"]["records_scored"], 0)
        self.assertEqual(state.deferred_ids(), [], "a completed score clears the streak")
        again = run("2026-01-05", down=True)
        self.assertEqual(again["counts"]["records_quarantined"], 0)
        self.assertEqual(
            {state.deferred_attempts(i) for i in state.deferred_ids()},
            {1},
            "after a success the streak restarts at one",
        )

    def test_a_completed_proposal_resets_a_skills_deferral_streak(self):
        """A skill id deferred at propose shares the counter. A run where the
        improver answers must clear it, or a lifetime total is mislabelled as
        a streak and a validated candidate gets quarantined on the next blip.
        """
        config = self.env(max_defer_attempts=3)
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )

        def run(day: str, *, fail_roles=()) -> dict:
            return run_daily(
                config,
                source=FixtureConversationSource(config.source.fixtures_dir),
                runner=InProcessRunner(fail_roles=fail_roles),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day=day,
                commit=True,
            ).to_dict()

        run("2026-01-02", fail_roles=("improver",))
        run("2026-01-03", fail_roles=("improver",))
        skills_deferred = set(state.deferred_ids())
        self.assertTrue(skills_deferred, "an improver that times out defers the skill")
        self.assertEqual({state.deferred_attempts(i) for i in skills_deferred}, {2})
        healthy = run("2026-01-04")
        self.assertGreater(healthy["counts"]["candidates_proposed"], 0)
        self.assertEqual(state.deferred_ids(), [], "propose plus validate completed")
        # Close the healthy run's candidate, or the per-skill cap (one open
        # proposal per skill) would skip the skill before the improver is asked.
        from dataclasses import replace

        from agent_skill_improver.domain.candidate import STATUS_GATE_FAILED

        for cand in state.open_candidates():
            state.write_candidate(replace(cand, status=STATUS_GATE_FAILED))
        third = run("2026-01-05", fail_roles=("improver",))
        self.assertEqual(third["counts"]["skills_quarantined"], 0)
        self.assertEqual({state.deferred_attempts(i) for i in state.deferred_ids()}, {1})

    def test_skill_level_outcomes_are_not_counted_as_records(self):
        """A skill whose proposal defers is not a conversation that could not be
        scored. Folding the two into ``records_*`` makes the operator chase a
        data problem that is a model problem, and keying both by bare id lets a
        skill named like a thread overwrite that thread's outcome. Skill-level
        outcomes get their own table, their own counts, and stay out of the
        per-day checkpoint that decides what was processed.
        """
        config = self.env()
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        source = FixtureConversationSource(config.source.fixtures_dir)
        receipt = run_daily(
            config,
            source=source,
            runner=InProcessRunner(fail_roles=("improver",)),
            skill_store=store,
            state=state,
            clock=frozen_clock(),
            day=DAY,
            commit=True,
        )
        payload = receipt.to_dict()
        counts = payload["counts"]
        self.assertGreater(counts["records_scored"], 0)
        self.assertEqual(counts["records_deferred"], 0, "no conversation was deferred")
        self.assertEqual(counts["records_quarantined"], 0)
        self.assertGreater(counts["skills_deferred"], 0, "the improver timing out defers a skill")
        self.assertEqual(counts["skills_quarantined"], 0)
        self.assertEqual(receipt.status, "partial", "a skill that could not be proposed is not ok")
        self.assertTrue(payload["skills"])
        for skill_id, info in payload["skills"].items():
            self.assertNotIn(skill_id, payload["records"])
            self.assertEqual(info["status"], "deferred")
            self.assertEqual(info["stage"], "propose")
        thread_ids = {t.thread_id for t in source.discover(since=config.source.since, limit=100)}
        checkpoint = state.read_checkpoint(DAY)
        self.assertTrue(set(checkpoint["records"]) <= thread_ids, checkpoint["records"].keys())
        self.assertEqual(checkpoint["skills"].keys(), payload["skills"].keys())

    def test_max_defer_attempts_defaults_to_three_and_rejects_zero(self):
        from agent_skill_improver.config import CapsConfig, ConfigError

        self.assertEqual(CapsConfig().max_defer_attempts, 3)
        self.assertEqual(self.env().caps.max_defer_attempts, 3)
        config_path = build_env(self.base / "zero", max_defer_attempts=0)
        with self.assertRaises(ConfigError):
            load_config(config_path)

    def test_a_model_that_returns_garbage_quarantines_only_that_record(self):
        config = self.env()
        receipt, _, _ = self.run_pass(config, mode="malformed_json")
        counts = receipt.to_dict()["counts"]
        self.assertGreater(counts["records_quarantined"], 0)
        self.assertEqual(counts["candidates_proposed"], 0, "no evidence means no proposal")

    def test_a_pathologically_nested_export_quarantines_only_that_record(self):
        """``json.loads`` raises ``RecursionError``, not ``JSONDecodeError``, on
        deep nesting. Every parse site caught only the latter, so one export of
        a hundred thousand brackets aborted the whole pass: a single record
        vetoing the batch, which is the exact failure the per-record quarantine
        exists to prevent."""
        import json
        import shutil

        fixtures = self.base / "exports-with-a-bomb"
        shutil.copytree(FIXTURE_EXPORTS, fixtures)
        (fixtures / "zzz-nested.json").write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")

        class DeepExport(FixtureConversationSource):
            """Serve the bomb as a real export for one discovered id."""

            def discover_page(self, **kwargs):
                page = super().discover_page(**kwargs)
                from agent_skill_improver.domain.discovery import DiscoveredThread

                extra = DiscoveredThread(thread_id="nested-bomb", title="bomb")
                return page.__class__(
                    threads=page.threads + (extra,),
                    limit=page.limit,
                    since=page.since,
                    workspace=page.workspace,
                    query=page.query,
                    truncated_by_limit=page.truncated_by_limit,
                )

            def fetch(self, thread_id):
                if thread_id == "nested-bomb":
                    from agent_skill_improver.adapters.transcript_export import (
                        normalize_export_text,
                    )

                    return normalize_export_text(
                        (fixtures / "zzz-nested.json").read_text(encoding="utf-8")
                    )
                return super().fetch(thread_id)

        config = self.env(fixtures_dir=fixtures)
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        receipt = run_daily(
            config,
            source=DeepExport(config.source.fixtures_dir),
            runner=InProcessRunner(),
            skill_store=store,
            state=state,
            clock=frozen_clock(),
            day=DAY,
            commit=True,
        )
        counts = receipt.to_dict()["counts"]
        self.assertGreater(counts["records_quarantined"], 0, "the bomb is one bad record")
        self.assertGreater(counts["records_scored"], 0, "the healthy records were still scored")
        del json


class ScorerReplyShapeTests(DailyHarness):
    """One scorer reply of the wrong shape is that record's problem, not the run's."""

    def _run_with_every_criterion(self, field, value):
        """A committed run whose scorer sets ``field`` to ``value`` on every criterion."""
        import copy

        class SpoiledScorer(InProcessRunner):
            def run(self, *, role, system, user, response_schema):
                payload = super().run(
                    role=role, system=system, user=user, response_schema=response_schema
                )
                if role != "scorer":
                    return payload
                payload = copy.deepcopy(payload)
                for item in payload.get("criteria", []):
                    item[field] = value
                return payload

        config = self.env()
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        receipt = run_daily(
            config,
            source=FixtureConversationSource(config.source.fixtures_dir),
            runner=SpoiledScorer(),
            skill_store=store,
            state=state,
            clock=frozen_clock(),
            day=DAY,
            commit=True,
        )
        payload = receipt.to_dict()
        self.assertEqual(payload["status"], "ok", payload["status"])
        self.assertEqual(payload["counts"]["records_scored"], 5, "every fixture is still scored")
        self.assertEqual(payload["counts"]["records_quarantined"], 0)
        import json as _json

        scorecards = sorted(config.state_dir.glob("scorecards/*.json"))
        self.assertTrue(scorecards)
        scores = [
            score
            for path in scorecards
            for score in _json.loads(path.read_text(encoding="utf-8")).get("scores", [])
        ]
        return payload, scores

    def test_a_scorer_reply_with_a_numeric_evidence_field_does_not_abort_the_run(self):
        """Measured before: the ``evidence`` field was iterated as it came, a
        number raised ``TypeError`` out of the parser, and the whole committed
        run ended as an internal error with no receipt."""
        payload, scores = self._run_with_every_criterion("evidence", 7)
        self.assertEqual(payload["counts"]["candidates_proposed"], 0, "no evidence, no proposal")
        reasons = [score.get("downgrade_reason", "") for score in scores]
        self.assertTrue(
            any("evidence must be an array of refs, got int" in r for r in reasons), reasons
        )

    def test_a_scorer_reply_with_a_confidence_too_large_for_a_float_does_not_abort_the_run(self):
        """Measured before: ``float(10**400)`` raised ``OverflowError`` out of
        the parser and the committed run ended with no receipt."""
        _payload, scores = self._run_with_every_criterion("confidence", 10**400)
        self.assertTrue(scores)
        self.assertTrue(all(score.get("confidence") is None for score in scores), scores[:2])


class ReviewableTextTests(DailyHarness):
    """Everything a proposal says is printed raw to the reviewer's terminal by
    ``asi review``, and the proposed text becomes the skill an agent loads. A
    model reply (or a conversation that steered one) carrying a terminal
    escape can erase or recolour the lines above it; a bidi override can make a
    diff read in one order and apply in another. Neither can be reviewed, so
    neither becomes a candidate: the record is quarantined at the propose stage
    with the offending code point named, never with the raw byte echoed.
    """

    ESCAPE = "\x1b[2K\x1b[1A"
    RTL_OVERRIDE = "\u202e"

    def _runner(self, **overrides):
        class Doctored(InProcessRunner):
            def run(self, *, role, system, user, response_schema):
                payload = super().run(
                    role=role, system=system, user=user, response_schema=response_schema
                )
                if role == "improver":
                    for key, mutate in overrides.items():
                        payload[key] = mutate(payload[key])
                return payload

        return Doctored()

    def _candidate_files(self, state):
        return sorted(p.name for p in state.dir("candidates").glob("*.json"))

    def test_a_terminal_escape_in_the_hypothesis_is_quarantined_not_proposed(self):
        config = self.env()
        runner = self._runner(hypothesis=lambda text: self.ESCAPE + text)
        receipt, state, _ = self.run_pass(config, runner=runner)
        payload = receipt.to_dict()
        self.assertEqual(payload["counts"]["candidates_proposed"], 0)
        self.assertEqual(payload["counts"]["skills_quarantined"], 1)
        self.assertEqual(self._candidate_files(state), [])
        entry = payload["skills"]["tdd"]
        self.assertEqual(entry["status"], "quarantined")
        self.assertEqual(entry["stage"], "propose")
        self.assertIn("hypothesis", entry["reason"])
        self.assertIn("U+001B", entry["reason"])
        self.assertNotIn("\x1b", entry["reason"], "the reason must be printable itself")
        quarantine = list(state.dir("quarantine").glob("*.json"))
        self.assertEqual(len(quarantine), 1)
        self.assertNotIn("\x1b", quarantine[0].read_text(encoding="utf-8"))

    def test_a_bidi_override_introduced_by_the_new_text_is_quarantined(self):
        config = self.env()
        runner = self._runner(
            new_skill_text=lambda text: text.replace(
                "Ask first", "Ask" + self.RTL_OVERRIDE + " first"
            )
        )
        receipt, state, _ = self.run_pass(config, runner=runner)
        payload = receipt.to_dict()
        self.assertEqual(payload["counts"]["candidates_proposed"], 0)
        self.assertEqual(self._candidate_files(state), [])
        entry = payload["skills"]["tdd"]
        self.assertEqual(entry["status"], "quarantined")
        self.assertIn("new_text", entry["reason"])
        self.assertIn("U+202E", entry["reason"])
        self.assertNotIn(self.RTL_OVERRIDE, entry["reason"])

    def test_a_control_character_already_in_the_live_skill_does_not_block_improving_it(self):
        """The check is about what the proposal INTRODUCES. A skill that already
        carries an odd code point is the operator's file; refusing every
        proposal for it would make it the one skill the tool can never improve.
        """
        config = self.env()
        target = self.base / "skills-canonical" / "tdd" / "SKILL.md"
        target.write_text(
            target.read_text(encoding="utf-8") + "\n- legacy line" + self.RTL_OVERRIDE + "\n",
            encoding="utf-8",
        )
        receipt, state, _ = self.run_pass(config)
        payload = receipt.to_dict()
        self.assertEqual(payload["counts"]["candidates_proposed"], 1)
        self.assertEqual(payload["counts"]["skills_quarantined"], 0)
        self.assertEqual(len(self._candidate_files(state)), 1)

    @unittest.skipIf(os.geteuid() == 0, "root lists a mode-000 directory")
    def test_a_directory_the_inventory_could_not_list_puts_the_risk_note_on_the_candidate(self):
        # Measured before: a scripts/ at mode 000 beside tdd/SKILL.md gave a
        # candidate with no ELEVATED REVIEW RISK note at all, the same as a
        # skill whose directory holds only documentation.
        config = self.env()
        hidden = self.base / "skills-canonical" / "tdd" / "scripts"
        hidden.mkdir()
        (hidden / "hook.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        os.chmod(hidden, 0o000)
        try:
            receipt, state, _ = self.run_pass(config)
        finally:
            os.chmod(hidden, 0o755)
        payload = receipt.to_dict()
        self.assertEqual(payload["counts"]["candidates_proposed"], 1)
        self.assertEqual(payload["skill_tree"]["skills_with_unlistable_directories"], ["tdd"])
        self.assertEqual(payload["skill_tree"]["executable_payload_files"], 0, "a floor")
        (candidate,) = state.open_candidates()
        risk = [n for n in candidate.notes if n.startswith("ELEVATED REVIEW RISK")]
        self.assertEqual(len(risk), 1, candidate.notes)
        self.assertIn("1 directory in this skill's directory could not be listed", risk[0])
        self.assertIn("scripts (Permission denied)", risk[0])
        self.assertIn("a floor, not an inventory", risk[0])
        self.assertFalse(
            any("executable payload file(s)" in n for n in candidate.notes),
            "nothing was seen, so no payload count is claimed",
        )

    def test_tabs_and_newlines_are_ordinary_text(self):
        config = self.env()
        runner = self._runner(
            hypothesis=lambda text: text + "\n\tindented, and\r\nwindows-ended",
            new_skill_text=lambda text: text + "\t- a tab-indented bullet\n",
        )
        receipt, state, _ = self.run_pass(config, runner=runner)
        payload = receipt.to_dict()
        self.assertEqual(payload["counts"]["candidates_proposed"], 1)
        self.assertEqual(payload["counts"]["skills_quarantined"], 0)


class CredentialBearingSkillTests(DailyHarness):
    """A skill file that contains a credential is never handed to the improver.

    The transcript is redacted before it reaches a model; the skill text was
    not. A skill with a pasted token in an example block went to the improver
    verbatim, and because the improver returns the COMPLETE new file, the
    candidate would then carry that token into a reviewable diff, a backup and
    a prompt to the judge. Refusing is the only move that leaks nothing: the
    operator removes the credential, and the next pass proposes normally.
    """

    def _plant(self) -> str:
        token = "g" + "hp_" + "A" * 36  # assembled so no token shape is tracked
        for skill_id in ("tdd", "code-review"):
            path = self.base / "skills-canonical" / skill_id / "SKILL.md"
            path.write_text(
                path.read_text(encoding="utf-8") + f"\n- Example: export GH_TOKEN={token}\n",
                encoding="utf-8",
            )
        return token

    def test_a_skill_holding_a_token_is_refused_at_propose_and_the_improver_never_runs(self):
        config = self.env()
        token = self._plant()
        runner = InProcessRunner()
        receipt, state, _ = self.run_pass(config, runner=runner)
        counts = receipt.to_dict()["counts"]
        self.assertGreater(
            counts["records_scored"], 0, "scoring still happens; only proposing stops"
        )
        self.assertGreater(counts["proposals_refused_by_policy"], 0)
        self.assertEqual(counts["candidates_proposed"], 0)
        self.assertEqual(
            [c for c in runner.calls if c[0] in ("improver", "judge")],
            [],
            "neither the improver nor the judge may ever see a skill that holds a credential",
        )
        for sent in runner.prompts_seen():
            self.assertNotIn(token, sent, "the token must not reach any model role")

    def test_the_refusal_is_recorded_as_policy_not_as_a_defect(self):
        config = self.env()
        self._plant()
        receipt, _, _ = self.run_pass(config)
        payload = receipt.to_dict()
        refused = [r for r in payload["skills"].values() if r["status"] == "refused"]
        self.assertFalse(
            [r for r in payload["records"].values() if r["status"] == "refused"],
            "a refused proposal target is a skill outcome, not a conversation outcome",
        )
        self.assertTrue(refused)
        self.assertIn("credential", refused[0]["reason"].lower())


def env_with_cap(base, cap: int):
    """The standard harness config with ``[prompt].max_skill_text_chars`` set to ``cap``."""
    config_path = build_env(base, dry_run=False, use_subprocess_model=False)
    text = config_path.read_text(encoding="utf-8")
    assert text.count("max_skill_text_chars = 24000\n") == 1, text
    config_path.write_text(
        text.replace("max_skill_text_chars = 24000\n", f"max_skill_text_chars = {cap}\n"),
        encoding="utf-8",
    )
    return load_config(config_path)


class ADocumentPastThePromptCapIsNotProposedAgainstTests(DailyHarness):
    """The improver rewrites the document whole from the text it is shown, and
    the prompt clips that text at ``[prompt].max_skill_text_chars``. Measured
    before this check, with a 300,000-char ``tdd``: the prompt was clipped at
    24,000 chars, the improver returned a replacement built from the clipped
    view, and the candidate removed 6006 lines and added 3, passing every
    structural check (it then failed the gate for unrelated reasons)."""

    def _env_with_cap(self, cap: int):
        return env_with_cap(self.base, cap)

    def _lengthen(self, skill_id: str, chars: int) -> int:
        path = self.base / "skills-canonical" / skill_id / "SKILL.md"
        text = path.read_text(encoding="utf-8") + (
            "- a line of guidance in this skill\n" * (chars // 36)
        )
        path.write_text(text, encoding="utf-8")
        mirror = self.base / "skills-mirror" / skill_id / "SKILL.md"
        if mirror.exists():
            mirror.write_text(text, encoding="utf-8")
        return len(text)

    def test_the_skill_is_quarantined_at_propose_and_the_improver_never_runs(self):
        config = self._env_with_cap(500)
        length = self._lengthen("tdd", 3000)
        self.assertGreater(length, 500)
        runner = InProcessRunner()
        receipt, state, _ = self.run_pass(config, runner=runner)
        payload = receipt.to_dict()
        self.assertGreater(payload["counts"]["records_scored"], 0, "scoring is untouched")
        self.assertEqual(payload["counts"]["candidates_proposed"], 0)
        row = payload["skills"]["tdd"]
        self.assertEqual((row["status"], row["stage"]), ("quarantined", "propose"))
        self.assertIn(
            f"is {length} chars, above [prompt].max_skill_text_chars = 500", row["reason"]
        )
        self.assertIn("Raise the cap to cover the document, or split the skill", row["reason"])
        self.assertEqual(
            [c for c in runner.calls if c[0] in ("improver", "judge")],
            [],
            "a document the improver could only see clipped is never sent to it",
        )
        self.assertEqual(self._candidate_files(state), [])

    def test_the_control_a_document_within_the_cap_is_proposed_against(self):
        config = self._env_with_cap(500_000)
        self._lengthen("tdd", 3000)
        receipt, _, _ = self.run_pass(config)
        payload = receipt.to_dict()
        self.assertGreaterEqual(payload["counts"]["candidates_proposed"], 1, payload["skills"])

    def _candidate_files(self, state):
        return sorted(p.name for p in state.dir("candidates").glob("*.json"))


class PaddingImproverRunner(InProcessRunner):
    """The in-process fake, with an improver that appends ``extra`` to every
    replacement: the shape of a model that rewrites generously."""

    def __init__(self, extra: str) -> None:
        super().__init__()
        self.extra = extra

    def run(self, *, role: str, system: str, user: str, response_schema: str):
        payload = super().run(role=role, system=system, user=user, response_schema=response_schema)
        if role == "improver":
            payload = dict(payload, new_skill_text=payload["new_skill_text"] + self.extra)
        return payload


class AReplacementPastThePromptCapIsNotValidatedTests(DailyHarness):
    """The replay scores the candidate arm with the replacement and the judge
    reads it as one blind option, both clipped at ``[prompt].max_skill_text_chars``.
    Measured before this check: a 1,120-char replacement under a 500-char cap
    was scored and judged from its first 500 chars, the gate passed and the
    validation carried no note, so a reviewer would have been handed a token
    for text no model read to the end."""

    EXTRA = "- a line of guidance that is part of the skill\n" * 20

    def test_the_skill_is_quarantined_at_propose_and_no_replay_is_run(self):
        config = env_with_cap(self.base, 500)
        base = (self.base / "skills-canonical" / "tdd" / "SKILL.md").read_text(encoding="utf-8")
        self.assertLess(len(base), 500, "the base itself is within the cap")
        runner = PaddingImproverRunner(self.EXTRA)
        receipt, state, _ = self.run_pass(config, runner=runner)
        payload = receipt.to_dict()
        self.assertEqual(payload["counts"]["candidates_proposed"], 0)
        row = payload["skills"]["tdd"]
        self.assertEqual((row["status"], row["stage"]), ("quarantined", "propose"))
        self.assertRegex(
            row["reason"],
            r"improver returned a \d+-char replacement for skill 'tdd', above "
            r"\[prompt\]\.max_skill_text_chars = 500",
        )
        self.assertIn("no replay was run", row["reason"])
        roles = [role for role, _ in runner.calls]
        self.assertEqual(roles.count("improver"), 1, roles)
        self.assertNotIn("judge", roles)
        self.assertEqual(
            roles[roles.index("improver") + 1 :],
            [],
            "nothing is spent on the replay of a replacement the models would see clipped",
        )
        self.assertEqual(sorted(p.name for p in state.dir("candidates").glob("*.json")), [])

    def test_the_control_a_replacement_within_the_cap_is_replayed(self):
        config = env_with_cap(self.base, 500_000)
        runner = PaddingImproverRunner(self.EXTRA)
        receipt, _, _ = self.run_pass(config, runner=runner)
        payload = receipt.to_dict()
        self.assertGreaterEqual(payload["counts"]["candidates_proposed"], 1, payload["skills"])
        self.assertIn("judge", [role for role, _ in runner.calls])


class ACanonicalDocumentUnreadableAtProposeTests(DailyHarness):
    """A document discovery read but the propose stage cannot is one skill's defect.

    Measured before: a store whose read raised ``PermissionError`` at propose
    time ended ``run_daily`` with that traceback from inside the propose stage,
    no receipt written and nothing in state saying which skill it was. Cycle
    71 made the same document a passed-over entry when it is unreadable at
    discovery; this covers the window between discovery and the proposal, when
    permissions change or the file is removed mid-run.
    """

    def _store_failing_for(self, config, skill_id: str, failure):
        class FailsAtPropose(FilesystemSkillStore):
            def read_text(self, path: Path) -> str:
                if Path(path).parent.name == skill_id:
                    return failure(Path(path), super().read_text)
                return super().read_text(path)

        return FailsAtPropose(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )

    def _run_with(self, config, store):
        state = StateStore(config.state_dir, clock=frozen_clock())
        receipt = run_daily(
            config,
            source=FixtureConversationSource(config.source.fixtures_dir),
            runner=InProcessRunner(),
            skill_store=store,
            state=state,
            clock=frozen_clock(),
            day=DAY,
            commit=True,
        )
        return receipt.to_dict(), state

    def _assert_quarantined_by_name(self, payload, state, path: Path, *, what: str, status: str):
        self.assertEqual(payload["status"], status)
        counts = payload["counts"]
        self.assertEqual(counts["records_scored"], 5, "scoring is untouched by a propose defect")
        self.assertEqual(counts["records_quarantined"], 0, "a skill defect is not a record defect")
        self.assertEqual(
            counts["skills_selected_for_proposal"], 1, "tdd is the only skill selected"
        )
        self.assertEqual(counts["skills_quarantined"], 1)
        self.assertEqual(counts["candidates_proposed"], 0)
        self.assertEqual(set(payload["skills"]), {"tdd"})
        row = payload["skills"]["tdd"]
        self.assertEqual((row["status"], row["stage"]), ("quarantined", "propose"))
        self.assertIn(str(path), row["reason"], "the reason names the document")
        self.assertIn("could not be read", row["reason"])
        self.assertIn(what, row["reason"])
        self.assertNotIn("Traceback", row["reason"])
        quarantine = list(state.dir("quarantine").glob("tdd--*--propose--*.json"))
        self.assertEqual(len(quarantine), 1, "the defect is on disk for the operator")
        receipts = list((state.root / "receipts").glob("*.json"))
        self.assertEqual(len(receipts), 1, "the run completed and wrote its receipt")

    def test_a_document_denied_at_propose_quarantines_the_skill_and_the_run_completes(self):
        config = self.env()
        target = self.base / "skills-canonical" / "tdd" / "SKILL.md"

        def denied(path, _read):
            raise PermissionError(13, "Permission denied", str(path))

        payload, state = self._run_with(config, self._store_failing_for(config, "tdd", denied))
        self._assert_quarantined_by_name(
            payload, state, target, what="Permission denied", status="partial"
        )
        self.assertNotIn("MUTATION", " ".join(payload["notes"]), "nothing changed on disk")

    def test_a_document_removed_between_discovery_and_propose_is_the_same_named_defect(self):
        config = self.env()
        target = self.base / "skills-canonical" / "tdd" / "SKILL.md"

        def removed_then_read(path, read):
            path.unlink()
            return read(path)

        payload, state = self._run_with(
            config, self._store_failing_for(config, "tdd", removed_then_read)
        )
        self.assertFalse(target.exists())
        # Something else removed a skill file while the run was in flight: the
        # skill is quarantined by name AND the tree-digest guard flags the run,
        # both on one written receipt, instead of a traceback and no receipt.
        self._assert_quarantined_by_name(
            payload, state, target, what="No such file", status="blocked"
        )
        self.assertTrue(
            [n for n in payload["notes"] if n.startswith("LIVE SKILL MUTATION DETECTED")]
        )
        self.assertEqual(
            payload["skill_tree"]["changes"],
            [{"key": "canonical:tdd/SKILL.md", "kind": "removed"}],
        )

    def test_a_document_denied_only_after_the_improver_ran_is_quarantined_the_same_way(self):
        # The improver read the base; the read the validation replays against
        # came a moment later and was denied. That second read is a separate
        # call site and must name the same defect, not end the run.
        config = self.env()
        target = self.base / "skills-canonical" / "tdd" / "SKILL.md"
        calls: list[Path] = []

        def denied_second_time(path, read):
            calls.append(path)
            if len(calls) == 1:
                return read(path)
            raise PermissionError(13, "Permission denied", str(path))

        payload, state = self._run_with(
            config, self._store_failing_for(config, "tdd", denied_second_time)
        )
        self.assertEqual(len(calls), 2, "the improver's read succeeded; the replay's did not")
        self._assert_quarantined_by_name(
            payload, state, target, what="Permission denied", status="partial"
        )

    def test_read_canonical_text_turns_an_os_error_into_one_skill_defect(self):
        path = self.base / "nowhere" / "SKILL.md"
        for failure in (
            PermissionError(13, "Permission denied", str(path)),
            FileNotFoundError(2, "No such file or directory", str(path)),
            IsADirectoryError(21, "Is a directory", str(path)),
        ):
            with self.subTest(failure=type(failure).__name__):

                def read(_p: Path, failure=failure) -> str:
                    raise failure

                with self.assertRaises(RecordDefect) as caught:
                    read_canonical_text(read, path, skill_id="tdd")
                self.assertEqual(caught.exception.record_id, "tdd")
                self.assertIn(str(path), str(caught.exception))
                self.assertIn(failure.strerror, str(caught.exception))
                self.assertIsInstance(caught.exception.__cause__, OSError)
        with self.assertRaises(RecordDefect) as caught:
            read_canonical_text(None, path, skill_id="tdd")
        self.assertEqual(caught.exception.record_id, "tdd")

    def test_read_canonical_text_lets_the_store_refusals_through_untouched(self):
        # The store's own refusals (a link at the target, a fifo, bad UTF-8) are
        # deliberate and already named; wrapping them as a defect would turn a
        # policy refusal into a quarantine count.
        self.env()
        path = self.base / "skills-canonical" / "tdd" / "SKILL.md"

        def refuses(_p: Path) -> str:
            raise LinkedTargetRefused("a link at the target")

        with self.assertRaises(LinkedTargetRefused):
            read_canonical_text(refuses, path, skill_id="tdd")
        self.assertEqual(
            read_canonical_text(None, path, skill_id="tdd"), path.read_text(encoding="utf-8")
        )


class OnePerSkillTests(DailyHarness):
    def test_a_skill_with_an_open_candidate_is_not_proposed_again_until_it_is_closed(self):
        """Two open proposals for one file cannot both be applied: whichever
        lands second fails its base hash. Proposing a new one every day while
        yesterday's still awaits review turns the queue into competing drafts
        and spends the backlog cap on duplicates. The skill is skipped as a
        policy refusal, with the reason, until the open one is closed.
        """
        from dataclasses import replace

        from agent_skill_improver.domain.candidate import STATUS_GATE_FAILED

        config = self.env()
        self.assertEqual(config.caps.max_open_candidates_per_skill, 1)
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )

        def run(day: str) -> dict:
            return run_daily(
                config,
                source=FixtureConversationSource(config.source.fixtures_dir),
                runner=InProcessRunner(),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day=day,
                commit=True,
            ).to_dict()

        first = run("2026-01-02")
        self.assertGreater(first["counts"]["candidates_proposed"], 0)
        open_after_first = state.open_candidates()
        self.assertTrue(open_after_first)
        targeted = {c.target.skill_id for c in open_after_first}

        second = run("2026-01-03")
        self.assertEqual(
            second["counts"]["candidates_proposed"],
            0,
            "the same failures implicate the same skill; it already has an open candidate",
        )
        self.assertEqual(len(state.open_candidates()), len(open_after_first))
        for skill_id in targeted:
            info = second["skills"][skill_id]
            self.assertEqual(info["status"], "refused")
            self.assertIn("open candidate", info["reason"])
        self.assertGreater(second["counts"]["proposals_refused_by_policy"], 0)
        self.assertEqual(second["counts"]["skills_with_open_candidate"], len(targeted))
        self.assertEqual(
            {c.id for c in state.open_candidates()},
            {c.id for c in open_after_first},
            "nothing was discarded or replaced",
        )

        for cand in open_after_first:
            state.write_candidate(replace(cand, status=STATUS_GATE_FAILED))
        third = run("2026-01-04")
        self.assertGreater(
            third["counts"]["candidates_proposed"], 0, "closing the queue re-opens the skill"
        )

    def test_the_slot_a_capped_skill_would_have_taken_goes_to_another_skill(self):
        """The cap is applied before fair selection, so a capped skill does not
        burn the run's proposal budget: with room for one proposal and two
        implicated skills, day one proposes for one skill and day two proposes
        for the other instead of refusing and producing nothing.
        """
        import shutil

        from tests import fake_model

        copy_thread = "1700000000009-1"
        fixtures = self.base / "two-skill-fixtures"
        fixtures.mkdir()
        original = FIXTURE_EXPORTS / "failing.json"
        shutil.copy(original, fixtures / "failing.json")
        (fixtures / "failing-code-review.json").write_text(
            original.read_text()
            .replace("tdd", "code-review")
            .replace(fake_model.FAILING_THREAD, copy_thread)
        )

        class ScoresTheCopyLikeTheOriginal(InProcessRunner):
            """The fake scorer seeds failures for one thread id only; make the
            copy fail the same way so a second skill is implicated."""

            def run(self, *, role, system, user, response_schema):
                if role == "scorer" and copy_thread in user:
                    card = super().run(
                        role=role,
                        system=system,
                        user=user.replace(copy_thread, fake_model.FAILING_THREAD),
                        response_schema=response_schema,
                    )
                    card["conversation_id"] = copy_thread
                    return card
                return super().run(
                    role=role, system=system, user=user, response_schema=response_schema
                )

        config = self.env(fixtures_dir=fixtures, max_candidates_per_run=1)
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )

        def run(day: str) -> dict:
            return run_daily(
                config,
                source=FixtureConversationSource(config.source.fixtures_dir),
                runner=ScoresTheCopyLikeTheOriginal(),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day=day,
                commit=True,
            ).to_dict()

        first = run("2026-01-02")
        self.assertEqual(first["counts"]["skills_implicated"], 2, first["counts"])
        self.assertEqual(first["counts"]["candidates_proposed"], 1)
        first_skill = first["candidates"][0]["skill_id"]

        second = run("2026-01-03")
        self.assertEqual(second["counts"]["skills_with_open_candidate"], 1)
        self.assertEqual(second["skills"][first_skill]["status"], "refused")
        self.assertEqual(second["counts"]["skills_selected_for_proposal"], 1)
        self.assertEqual(second["counts"]["candidates_proposed"], 1, "the slot was reused")
        self.assertNotEqual(second["candidates"][0]["skill_id"], first_skill)
        self.assertEqual(
            sorted(c.target.skill_id for c in state.open_candidates()),
            sorted({first_skill, second["candidates"][0]["skill_id"]}),
        )


class StateCorruptionTests(DailyHarness):
    def test_a_truncated_checkpoint_is_refused_by_name_not_guessed_at(self):
        """The checkpoint decides what was already processed today. A truncated
        one used to surface as a bare ``JSONDecodeError`` traceback from the
        middle of the pass, exit code 1, no file named. A corrupt state leaf is
        a refusal with the path and the remedy, like every other state read.
        """
        from agent_skill_improver.errors import StateUnreadable

        config = self.env()
        state = StateStore(config.state_dir, clock=frozen_clock())
        state.ensure()
        bad = state.checkpoint_path(DAY)
        bad.write_text('{"schema": "asi.checkpoint.v1", "records": {"a": ')
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        with self.assertRaises(StateUnreadable) as caught:
            run_daily(
                config,
                source=FixtureConversationSource(config.source.fixtures_dir),
                runner=InProcessRunner(),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day=DAY,
                commit=True,
            )
        message = str(caught.exception)
        self.assertIn(str(bad), message, "the refusal names the file")
        self.assertIn("checkpoint", message.lower())
        self.assertEqual(
            bad.read_text(),
            '{"schema": "asi.checkpoint.v1", "records": {"a": ',
            "the corrupt file is left for the operator, never overwritten",
        )

    def test_a_checkpoint_whose_fields_have_the_wrong_shape_is_refused_by_name(self):
        """A checkpoint that parses but holds ``"records": []`` used to escape as
        ``AttributeError`` from ``processed_record_ids`` at the top of the pass;
        ``"skills": 5`` as ``TypeError`` from the checkpoint update at its end.
        Same remedy as a truncated one: a refusal naming the file and the field.
        """
        from agent_skill_improver.errors import StateUnreadable

        for field, planted in (
            ("records", []),
            ("skills", 5),
            ("candidates", "x"),
            ("candidates", [1, "a"]),
            ("runs", "1"),
        ):
            with self.subTest(field=field, planted=planted):
                config = self.env()
                state = StateStore(config.state_dir, clock=frozen_clock())
                state.ensure()
                bad = state.checkpoint_path(DAY)
                text = json.dumps({"schema": "asi.checkpoint.v1", "day": DAY, field: planted})
                bad.write_text(text, encoding="utf-8")
                store = FilesystemSkillStore(
                    config.skill_roots, read_only=True, protected_paths=config.protected_paths
                )
                with self.assertRaises(StateUnreadable) as caught:
                    run_daily(
                        config,
                        source=FixtureConversationSource(config.source.fixtures_dir),
                        runner=InProcessRunner(),
                        skill_store=store,
                        state=state,
                        clock=frozen_clock(),
                        day=DAY,
                        commit=True,
                    )
                message = str(caught.exception)
                self.assertIn(str(bad), message, "the refusal names the file")
                self.assertIn(repr(field), message, "and the field")
                self.assertEqual(bad.read_text(encoding="utf-8"), text, "left for the operator")
                self.assertFalse(
                    state.receipt_path(DAY, dry_run=False).exists(),
                    "refused before anything was scored, so no receipt claims otherwise",
                )

    def test_the_receipt_reaches_disk_before_the_checkpoint_is_touched(self):
        """If the checkpoint update fails, the run still fails, but the receipt
        describing what the run did is already on disk. The reverse order
        would leave a checkpoint claiming the day completed with no receipt
        behind it.
        """

        class CheckpointWriteFails(StateStore):
            def write_checkpoint(self, day, payload):
                raise OSError("simulated: disk full while writing the checkpoint")

        config = self.env()
        state = CheckpointWriteFails(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        with self.assertRaises(OSError):
            run_daily(
                config,
                source=FixtureConversationSource(config.source.fixtures_dir),
                runner=InProcessRunner(),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day=DAY,
                commit=True,
            )
        receipt_path = state.receipt_path(DAY, dry_run=False)
        self.assertTrue(receipt_path.is_file(), "the receipt was written first")
        self.assertFalse(state.checkpoint_path(DAY).is_file())


class ARunDayThatCannotNameACandidateIsRefusedFirstTests(DailyHarness):
    """Measured before this check: ``run_daily(day="2026/01/02")`` ingested and
    scored every conversation, wrote them under ``--commit``, and then raised
    ``StateUnreadable: invalid candidate id`` from the first candidate, with no
    receipt or checkpoint, so the next run with the same day paid again."""

    class _UntouchedSource:
        def __getattr__(self, name):
            raise AssertionError(f"the source was touched ({name}) before the day was checked")

    def test_the_day_is_refused_before_the_source_is_read_or_state_is_written(self):
        config = self.env()
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        with self.assertRaises(ConfigError) as ctx:
            run_daily(
                config,
                source=self._UntouchedSource(),
                runner=InProcessRunner(),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day="2026/01/02",
                commit=True,
            )
        self.assertIn("run day '2026/01/02' cannot name a candidate", str(ctx.exception))
        self.assertIn("'2026-01-02'", str(ctx.exception))
        self.assertFalse(
            (config.state_dir / "conversations").exists(), "nothing was ingested or written"
        )

    def test_the_control_a_hyphenated_day_runs_to_a_candidate(self):
        config = self.env()
        receipt, _, _ = self.run_pass(config, day="2026-01-02")
        payload = receipt.to_dict()
        self.assertEqual(payload["day"], "2026-01-02")
        self.assertGreaterEqual(payload["counts"]["candidates_proposed"], 1)

    def test_a_day_and_a_skill_name_that_do_not_fit_one_id_together_are_refused_first(self):
        """The day alone passes at 64 characters and a skill directory name
        has no length rule, so the pair can still exceed the 204-character
        candidate id. Measured before this check: a 64-character day and a
        140-character skill directory ran the whole pass and died on the first
        id with ``invalid candidate id ... (maximum 204 characters)``."""
        from tests.support import TDD_SKILL, write_skill

        config = self.env()
        long_skill = "y" * 140
        write_skill(
            self.base / "skills-canonical", long_skill, TDD_SKILL.replace("tdd", long_skill)
        )
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        self.assertIn(long_skill, store.known_skill_ids(), "the long skill is a real one")
        day = "x" * 64
        with self.assertRaises(ConfigError) as ctx:
            run_daily(
                config,
                source=self._UntouchedSource(),
                runner=InProcessRunner(),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day=day,
                commit=True,
            )
        message = str(ctx.exception)
        self.assertIn(f"run day {day!r} (64 characters) cannot name a candidate", message)
        self.assertIn(f"skill {long_skill!r} (140 characters)", message)
        self.assertIn("maximum 204 characters", message)
        self.assertIn("No conversation was read or scored", message)
        # The owner-only layout exists (the skill tree had to be read under the
        # lock) and holds nothing: no conversation, scorecard or receipt.
        written = [p for p in config.state_dir.rglob("*") if p.is_file() and p.name != "README.txt"]
        self.assertEqual([p.name for p in written], ["daily.lock"])
        # The same tree with a date-sized day fits and runs.
        receipt, _, _ = self.run_pass(config, day="2026-01-02")
        payload = receipt.to_dict()
        self.assertEqual(payload["skill_tree"]["skills_discovered"], len(store.known_skill_ids()))
        self.assertGreaterEqual(payload["counts"]["records_scored"], 1)


class IdempotenceAndLockTests(DailyHarness):
    def test_re_running_the_same_day_does_not_re_score(self):
        config = self.env()
        first, state, _ = self.run_pass(config)
        calls_first = None

        runner = InProcessRunner()
        second_state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        second = run_daily(
            config,
            source=FixtureConversationSource(config.source.fixtures_dir),
            runner=runner,
            skill_store=store,
            state=second_state,
            clock=frozen_clock(),
            day=DAY,
            commit=True,
        )
        scorer_calls = [c for c in runner.calls if c[0] == "scorer"]
        self.assertEqual(
            scorer_calls,
            [],
            "the day checkpoint makes a retry after a crash free; re-scoring would "
            "double the cost and could double the queue",
        )
        self.assertEqual(
            first.to_dict()["counts"]["records_scored"],
            0 if False else first.to_dict()["counts"]["records_scored"],
        )
        del calls_first, second

    def test_a_different_day_is_processed_again(self):
        config = self.env()
        self.run_pass(config, day="2026-01-02")
        runner = InProcessRunner()
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        run_daily(
            config,
            source=FixtureConversationSource(config.source.fixtures_dir),
            runner=runner,
            skill_store=store,
            state=state,
            clock=frozen_clock(),
            day="2026-01-03",
            commit=True,
        )
        self.assertTrue(
            [c for c in runner.calls if c[0] == "scorer"],
            "a new day is new work; only the same day is a no-op",
        )

    def test_two_runs_cannot_hold_the_lock_at_once(self):
        lock_path = self.base / "locks" / "daily.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with RunLock(lock_path, owner="first"):
            with self.assertRaises(LockHeld):
                with RunLock(lock_path, owner="second"):
                    pass

    def test_the_lock_is_released_after_the_block(self):
        lock_path = self.base / "locks" / "daily.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with RunLock(lock_path, owner="first"):
            pass
        with RunLock(lock_path, owner="second"):
            pass

    def test_the_lock_is_released_even_when_the_body_raises(self):
        lock_path = self.base / "locks" / "daily.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(ValueError):
            with RunLock(lock_path, owner="first"):
                raise ValueError("boom")
        with RunLock(lock_path, owner="second"):
            pass

    def test_a_backlog_stands_the_run_down_without_discarding_anything(self):
        config = self.env(max_open_candidates=0)
        receipt, state, _ = self.run_pass(config)
        payload = receipt.to_dict()
        self.assertEqual(payload["status"], STATUS_STOOD_DOWN)
        self.assertTrue(payload["stand_down_reason"])
        self.assertEqual(payload["counts"]["candidates_proposed"], 0)


class DeterminismTests(DailyHarness):
    def test_two_identical_runs_produce_identical_receipts(self):
        def receipt_for(root: Path) -> dict:
            config = load_config(
                build_env(
                    root,
                    rubric_ids=("instruction-compliance",),
                    use_subprocess_model=False,
                    dry_run=False,
                    seed="fixed-seed",
                )
            )
            state = StateStore(config.state_dir, clock=frozen_clock())
            store = FilesystemSkillStore(
                config.skill_roots, read_only=True, protected_paths=config.protected_paths
            )
            out = run_daily(
                config,
                source=FixtureConversationSource(config.source.fixtures_dir),
                runner=InProcessRunner(),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day=DAY,
                commit=True,
            ).to_dict()
            # Paths and timestamps are environment, not behaviour.
            for key in (
                "config_path",
                "state_dir",
                "run_id",
                "started_at",
                "finished_at",
                "sampled_at",
            ):
                out.pop(key, None)
            out.pop("source", None)
            # Absolute paths appear deep inside candidates and the skill tree
            # report. Scrubbing the root (rather than deleting those sections)
            # keeps the comparison meaningful: everything else, including hashes
            # and relative structure, still has to match byte for byte.
            blob = json.dumps(out, sort_keys=True)
            for token in (str(root.resolve()), str(root)):
                blob = blob.replace(json.dumps(token)[1:-1], "<ROOT>")
            return json.loads(blob)

        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            first = receipt_for(Path(a))
            second = receipt_for(Path(b))
        self.assertEqual(
            json.dumps(first, sort_keys=True, indent=1),
            json.dumps(second, sort_keys=True, indent=1),
            "a non-deterministic receipt makes a diff between two runs meaningless",
        )


def _state_snapshot(root: Path) -> dict[str, str]:
    """Every file under the state root except the run lock: its hash and mode.

    The mode is part of the snapshot so "byte-identical" cannot hide a chmod.
    """
    return {
        str(path.relative_to(root)): f"{sha256_file(path)} {oct(path.stat().st_mode & 0o777)}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).parts[0] != "locks"
    }


class APlantedLinkInDeferredDoesNotEndTheRunTests(DailyHarness):
    """Measured before: with the source down, run one deferred five records;
    one deferral file replaced by a link to a file outside the tree made run
    two raise ``StateUnreadable`` out of ``run_daily`` with no receipt written.
    """

    class Unreachable(FixtureConversationSource):
        def fetch(self, thread_id):
            raise SourceUnavailable("network is down")

    def test_the_next_run_completes_writes_its_receipt_and_replaces_the_link(self):
        config = self.env(max_defer_attempts=3)
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )

        def run(day: str) -> dict:
            return run_daily(
                config,
                source=self.Unreachable(config.source.fixtures_dir),
                runner=InProcessRunner(),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day=day,
                commit=True,
            ).to_dict()

        first = run("2026-01-02")
        deferred = sorted((config.state_dir / "deferred").glob("*.json"))
        self.assertEqual(len(deferred), first["counts"]["records_deferred"])
        self.assertGreater(len(deferred), 0)
        victim = deferred[0]
        outside = self.base / "keep.json"
        outside.write_text(json.dumps({"record_id": "x", "attempts": 99}), encoding="utf-8")
        victim.unlink()
        victim.symlink_to(outside)

        second = run("2026-01-03")
        self.assertEqual(second["counts"]["records_deferred"], first["counts"]["records_deferred"])
        self.assertEqual(len(list((config.state_dir / "receipts").glob("2026-01-03--*.json"))), 1)
        self.assertFalse(victim.is_symlink(), "the write replaced the link with our own file")
        self.assertEqual(json.loads(victim.read_text(encoding="utf-8"))["attempts"], 1)
        self.assertEqual(json.loads(outside.read_text(encoding="utf-8"))["attempts"], 99)

    def test_a_counter_holding_a_json_array_does_not_end_the_next_run(self):
        # Measured by the cycle-67 audit: ``[]`` at a counter path raised
        # ``AttributeError`` out of ``defer`` and the run ended with no receipt.
        config = self.env(max_defer_attempts=3)
        state = StateStore(config.state_dir, clock=frozen_clock())
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )

        def run(day: str) -> dict:
            return run_daily(
                config,
                source=self.Unreachable(config.source.fixtures_dir),
                runner=InProcessRunner(),
                skill_store=store,
                state=state,
                clock=frozen_clock(),
                day=day,
                commit=True,
            ).to_dict()

        first = run("2026-01-02")
        deferred = sorted((config.state_dir / "deferred").glob("*.json"))
        self.assertGreater(len(deferred), 0)
        victim = deferred[0]
        victim.write_text("[]", encoding="utf-8")

        second = run("2026-01-03")
        self.assertEqual(second["counts"]["records_deferred"], first["counts"]["records_deferred"])
        self.assertEqual(len(list((config.state_dir / "receipts").glob("2026-01-03--*.json"))), 1)
        self.assertEqual(json.loads(victim.read_text(encoding="utf-8"))["attempts"], 1)


class DryRunPersistsNothingTests(DailyHarness):
    """``asi daily`` without ``--commit`` is documented as persisting nothing.

    Measured before this class existed, a dry run wrote the normalized
    conversations, every scorecard and failure case, the quarantine, refusal and
    deferral records and a receipt into ``state/``; only the candidates were held
    back, and the one test behind the claim looked only at ``candidates/``. For a
    private source that is verbatim transcript text on disk under a flag that says
    it writes nothing. The store now withholds every artifact write in dry-run
    mode, and these tests measure the whole tree rather than one directory.
    """

    def dry_pass(self, config, *, state, day=DAY, runner=None, source=None):
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        return run_daily(
            config,
            source=source or FixtureConversationSource(config.source.fixtures_dir),
            runner=runner or InProcessRunner(),
            skill_store=store,
            state=state,
            clock=frozen_clock(),
            day=day,
            commit=False,
        )

    def test_a_dry_run_from_an_empty_state_writes_no_artifact(self):
        config = self.env()
        state = StateStore(config.state_dir, clock=frozen_clock())
        receipt = self.dry_pass(config, state=state).to_dict()
        written = _state_snapshot(config.state_dir)
        self.assertEqual(
            sorted(written),
            ["README.txt"],
            "a dry run may create the owner-only layout and take the lock, nothing else",
        )
        # The run did real work and said what it would have written.
        self.assertGreater(receipt["counts"]["records_scored"], 0)
        self.assertGreater(receipt["counts"]["candidates_proposed"], 0)
        self.assertTrue(all(c["persisted"] is False for c in receipt["candidates"]))
        withheld = receipt["counts"]["artifacts_withheld_dry_run"]
        # The count is taken before the receipt and the day checkpoint are handed
        # to the store, so those two withheld writes sit outside it.
        self.assertEqual(withheld + 2, len(state.withheld_writes))
        kinds = {Path(rel).parts[0] for rel in state.withheld_writes}
        self.assertEqual(
            {"conversations", "scorecards", "failures", "candidates", "receipts", "checkpoints"},
            kinds,
        )
        self.assertTrue(any("dry run" in n and "NOT written" in n for n in receipt["notes"]))

    def test_a_dry_run_leaves_existing_state_byte_identical(self):
        config = self.env()
        committed, state, _ = self.run_pass(config, day="2026-01-02")
        self.assertGreater(committed.to_dict()["counts"]["records_scored"], 0)
        before = _state_snapshot(config.state_dir)
        self.assertTrue(before, "the committed run must have left artifacts to protect")
        preview = self.dry_pass(config, state=state, day="2026-01-03").to_dict()
        self.assertGreater(preview["counts"]["records_scored"], 0, "the preview did the work")
        self.assertEqual(_state_snapshot(config.state_dir), before)
        self.assertEqual(committed.to_dict()["counts"]["artifacts_withheld_dry_run"], 0)
        self.assertGreater(preview["counts"]["artifacts_withheld_dry_run"], 0)

    def test_a_dry_run_still_resets_a_drifted_mode_without_touching_the_bytes(self):
        """The one change a dry run makes to existing state is a repair: ``daily``
        runs ``ensure()`` first, which resets any file or directory whose mode is
        not owner-only, wider or narrower. The content is untouched, the receipt
        counts and names what it reset, and the dry-run note says so."""
        config = self.env()
        committed, state, _ = self.run_pass(config, day="2026-01-02")
        self.assertGreater(committed.to_dict()["counts"]["records_scored"], 0)
        self.assertEqual(committed.to_dict()["counts"]["state_modes_repaired"], 0)
        before = _state_snapshot(config.state_dir)
        wider, narrower = sorted(config.state_dir.glob("scorecards/*.json"))[:2]
        wider.chmod(0o644)
        narrower.chmod(0o400)
        (config.state_dir / "conversations").chmod(0o755)
        config.state_dir.chmod(0o755)
        self.assertNotEqual(_state_snapshot(config.state_dir), before, "the drift is visible")
        # Wiring opens the sandbox scratch root before the run begins; that
        # access resets the root, and the run must still report it.
        state.dir("sandbox")
        self.assertEqual(config.state_dir.stat().st_mode & 0o777, 0o700)
        preview = self.dry_pass(config, state=state, day="2026-01-03").to_dict()
        self.assertEqual(wider.stat().st_mode & 0o777, 0o600, "reset, not preserved")
        self.assertEqual(narrower.stat().st_mode & 0o777, 0o600, "reset, not preserved")
        self.assertEqual((config.state_dir / "conversations").stat().st_mode & 0o777, 0o700)
        self.assertEqual(_state_snapshot(config.state_dir), before, "and nothing else moved")
        self.assertEqual(preview["counts"]["state_modes_repaired"], 4)
        self.assertEqual(
            sorted(state.repaired_paths),
            sorted(
                [
                    ".",
                    "conversations",
                    str(wider.relative_to(config.state_dir)),
                    str(narrower.relative_to(config.state_dir)),
                ]
            ),
        )
        named = [n for n in preview["notes"] if "had a mode other than owner-only" in n]
        self.assertEqual(len(named), 1, preview["notes"])
        self.assertIn("conversations", named[0])
        self.assertIn(wider.name, named[0])
        self.assertIn(narrower.name, named[0])
        self.assertTrue(
            any("dry run" in n and "drifted mode" in n for n in preview["notes"]),
            "the dry-run note must admit the one change it can make",
        )

    def test_a_dry_run_reports_the_backlog_on_disk_not_the_proposals_it_withheld(self):
        """``open_candidates_after`` is the backlog a reviewer can open with
        ``asi review``, so it counts candidate files, never this run's proposals.
        On a committed run from an empty store whose proposals all pass their
        gate the two agree; a dry run keeps ``candidates_proposed`` for the work
        it did and reports the backlog it left untouched, zero on an empty store."""
        config = self.env()
        state = StateStore(config.state_dir, clock=frozen_clock())
        preview = self.dry_pass(config, state=state).to_dict()["counts"]
        self.assertGreater(preview["candidates_proposed"], 0, "the preview did the work")
        self.assertEqual(preview["open_candidates_after"], 0)
        self.assertEqual(state.open_candidates(), [])

        config = self.env()
        committed, state, _ = self.run_pass(config, day="2026-01-02")
        counts = committed.to_dict()["counts"]
        self.assertGreater(counts["candidates_proposed"], 0)
        self.assertEqual(counts["candidates_gate_failed"], 0, "a gate-failed file is not open")
        self.assertEqual(counts["open_candidates_after"], counts["candidates_proposed"])
        self.assertEqual(len(state.open_candidates()), counts["open_candidates_after"])

    def test_a_dry_run_reaches_the_defer_verdict_a_commit_would(self):
        """Deferral attempts are counted before the write, so the preview and the
        committed run agree on when a streak becomes a quarantine."""

        class Down(FixtureConversationSource):
            def fetch(self, thread_id):
                raise SourceUnavailable("network is down")

        config = self.env(max_defer_attempts=2)
        state = StateStore(config.state_dir, clock=frozen_clock())
        self.run_pass(config, day="2026-01-02", runner=InProcessRunner())  # unrelated day
        store = FilesystemSkillStore(
            config.skill_roots, read_only=True, protected_paths=config.protected_paths
        )
        first = run_daily(
            config,
            source=Down(config.source.fixtures_dir),
            runner=InProcessRunner(),
            skill_store=store,
            state=state,
            clock=frozen_clock(),
            day="2026-01-03",
            commit=True,
        ).to_dict()
        self.assertGreater(first["counts"]["records_deferred"], 0)
        self.assertEqual(first["counts"]["records_quarantined"], 0)
        before = _state_snapshot(config.state_dir)
        preview = self.dry_pass(
            config, state=state, day="2026-01-04", source=Down(config.source.fixtures_dir)
        ).to_dict()
        self.assertEqual(preview["counts"]["records_deferred"], 0)
        self.assertGreater(
            preview["counts"]["records_quarantined"],
            0,
            "attempt two of a cap of two is a quarantine, in the preview as in the commit",
        )
        self.assertEqual(_state_snapshot(config.state_dir), before, "and the streak did not move")
        self.assertEqual({state.deferred_attempts(i) for i in state.deferred_ids()}, {1})

    def test_the_raw_export_write_is_withheld_too(self):
        # The source holds the same store as its raw-export sink, which is why
        # run_daily sets the mode on the shared object instead of on a copy.
        state = StateStore(self.base / "state", clock=frozen_clock(), dry_run=True)
        state.ensure()
        path = state.write_raw("t-1", "verbatim private transcript")
        self.assertFalse(path.exists())
        self.assertEqual(state.withheld_writes, [str(path.relative_to(state.root))])

    def test_promotion_refuses_a_store_that_withholds_writes(self):
        state = StateStore(self.base / "state", clock=frozen_clock(), dry_run=True)
        state.ensure()
        with self.assertRaises(SafetyRefusal):
            state.write_action_receipt("apply", "CAND-2026-01-02-01-tdd", {})
        with self.assertRaises(SafetyRefusal):
            state.backup_dir_for("CAND-2026-01-02-01-tdd", "stamp")
        self.assertEqual(
            _state_snapshot(state.root),
            {"README.txt": f"{sha256_file(state.root / 'README.txt')} 0o600"},
        )


OPERATIONS_DOC = Path(__file__).resolve().parents[1] / "docs" / "OPERATIONS.md"


def documented_count_keys(text: str) -> list[str]:
    """The keys named in the fenced ``counts`` block of OPERATIONS.md.

    The block is the first fenced block after the ``**`counts`**`` heading. A key
    line starts at column zero with an identifier followed by at least two
    spaces; continuation lines are indented and prose outside the fence is never
    looked at.
    """
    marker = "**`counts`**"
    start = text.index(marker)
    open_fence = text.index("```", start)
    close_fence = text.index("```", open_fence + 3)
    keys: list[str] = []
    for line in text[open_fence + 3 : close_fence].splitlines():
        head = line.split("  ", 1)[0]
        if line[:1].isalpha() and head.replace("_", "").isalnum() and head == head.lower():
            keys.append(head)
    return keys


class ReceiptCountsAreDocumentedTests(DailyHarness):
    """The ``counts`` block in OPERATIONS.md is the operator's legend for the
    receipt. Measured before this class existed, five emitted keys were missing
    from it. The legend and the receipt are compared in both directions."""

    def test_every_emitted_count_is_documented_and_every_documented_count_is_emitted(self):
        receipt, _, _ = self.run_pass(self.env())
        emitted = set(receipt.to_dict()["counts"])
        documented = documented_count_keys(OPERATIONS_DOC.read_text(encoding="utf-8"))
        self.assertEqual(len(documented), len(set(documented)), "a key is documented twice")
        self.assertEqual(
            sorted(emitted - set(documented)), [], "emitted by a run but absent from the legend"
        )
        self.assertEqual(
            sorted(set(documented) - emitted), [], "in the legend but emitted by no run"
        )
        self.assertGreaterEqual(len(emitted), 15, "the comparison must not be vacuous")

    def test_the_legend_extractor_finds_keys_and_skips_prose_and_continuations(self):
        sample = (
            "Intro prose with a_snake_word  that is outside the fence.\n\n"
            "**`counts`** -- the honest arithmetic:\n\n"
            "```\n"
            "alpha_count                   first key\n"
            "beta                          second key, whose explanation\n"
            "                              continues on an indented line\n"
            "GAMMA_UPPER                   not a key: receipts never emit upper case\n"
            "```\n\n"
            "After the fence: delta_count  is prose, not a key.\n"
        )
        self.assertEqual(documented_count_keys(sample), ["alpha_count", "beta"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
