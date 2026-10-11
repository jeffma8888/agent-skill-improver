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
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app.daily import run_daily
from agent_skill_improver.app.lock import RunLock
from agent_skill_improver.app.receipt import STATUS_STOOD_DOWN
from agent_skill_improver.app.state import StateStore, safe_name
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.config import load_config
from agent_skill_improver.errors import LockHeld, SafetyRefusal
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
        refused = [r for r in payload["records"].values() if r["status"] == "refused"]
        self.assertTrue(refused)
        self.assertIn("credential", refused[0]["reason"].lower())


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
