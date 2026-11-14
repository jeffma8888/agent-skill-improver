"""A declined candidate cannot be applied or rolled back.

``asi decline`` closes a validated candidate on the record: its status becomes
``declined``, its record holds the decision, and nothing leaves that status in
this version. ``review`` refuses it as closed and prints no token; ``apply``
refuses it before the lock on the status it was handed and again under the lock
on the record it re-reads; ``rollback`` refuses it after its re-read, its check
of the record against itself and its identity binding, before it restores
anything, and a declined record that does not recompute, or that changed after
the rollback read it, is refused as tampered with rollback's own advice, which
keeps the record and its backup. ``candidates`` lists it only with ``--all``.

The module also holds ``DeclineTreeCase``, the shared tree the other c247b
modules build on: one committed daily run with a validated candidate that
``review`` has shown, put back before each test.
"""

from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
import re
import shutil
import signal
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Sequence
from unittest import mock

import agent_skill_improver.adapters.skill_fs as skill_fs
from agent_skill_improver import cli
from agent_skill_improver.app.lock import RunLock
from agent_skill_improver.app.promote import apply_candidate, rollback_candidate
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.app.wiring import build_skill_store
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.config import load_config
from agent_skill_improver.domain.approval import AUTOMATION_ENV_VAR, HumanApproval
from agent_skill_improver.domain.candidate import Candidate
from agent_skill_improver.domain.integrity import verify_full_integrity
from agent_skill_improver.errors import CandidateTampered, GateFailed, LockHeld

from tests.support import build_env, run_cli

#: The lock as it is before any test patches it, for the probe of lock_is_held.
_REAL_ENTER = RunLock.__enter__
_REAL_EXIT = RunLock.__exit__

DAY = "2026-02-01"
T0 = datetime.datetime(2026, 2, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
#: A decision as an older asi or a hand edit would leave it on a record.
HAND_DECISION = {
    "declined_at": "2026-02-01T13:00:00Z",
    "operator": "elder",
    "reason": "declined elsewhere",
}
#: What a rollback refused for a changed record keeps saying (c247a).
KEEP_BOTH = "Keep the record and the backup."
NEVER_RUN = "Never run the backup manifest's rollback_command"
MANUAL = '"Manual restoration" says.'
DECLINED_AT = "'declined' (at"

Tree = dict[str, tuple[int, bytes | None]]


def clock(hours: int = 0) -> FrozenClock:
    return FrozenClock(T0 + datetime.timedelta(hours=hours))


class _Stdin:
    """A stand-in for ``sys.stdin``: bytes on ``buffer``, and whether it is a terminal."""

    def __init__(self, data: bytes, *, tty: bool = False) -> None:
        self.buffer = io.BytesIO(data)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class DeclineTreeCase(unittest.TestCase):
    """One committed daily run, its first candidate validated and reviewed, put back per test."""

    seed = "c247b-decline"
    env_options: dict[str, Any] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name).resolve()
        cls.base = root / "env"
        cls.config_path = build_env(cls.base, dry_run=False, seed=cls.seed, **cls.env_options)
        cls.cfg = ["--config", str(cls.config_path)]
        rc, out, err = cls._main([*cls.cfg, "--json", "daily", "--commit", "--day", DAY])
        assert rc == 0, err
        rc, out, err = cls._main([*cls.cfg, "--json", "candidates"])
        cls.cid = sorted(c["id"] for c in json.loads(out)["candidates"])[0]
        rc, out, err = cls._main([*cls.cfg, "--json", "review", cls.cid])
        assert rc == 0, err
        cls.token = json.loads(out)["approval_token"]
        cls.apply_argv = [*cls.cfg, "apply", cls.cid, "--approve", cls.token, "--i-reviewed-this"]
        cls.rollback_argv = [*cls.cfg, "rollback", cls.cid]
        cls.pristine = root / "pristine"
        shutil.copytree(cls.base, cls.pristine, symlinks=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        shutil.rmtree(self.base)
        shutil.copytree(self.pristine, self.base, symlinks=True)
        config = load_config(self.config_path)
        self.config = config
        self.state = StateStore(config.state_dir)
        self.record_path = self.state.candidate_path(self.cid)
        # Spelled out, not asked of the store: before c247b it refuses the action.
        self.receipt_path = self.base / "state" / "receipts" / f"decline--{self.cid}.json"
        self.read_before = self.state.read_candidate(self.cid)
        assert self.read_before.status == "validated", self.read_before.status
        self.skill = Path(self.read_before.target.canonical_path)
        self.original = self.skill.read_bytes()

    # -- running -------------------------------------------------------------

    @staticmethod
    def _main(
        argv: Sequence[str],
        *patches: Any,
        stdin: bytes | _Stdin | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(mock.patch.dict(os.environ, env or {}))
            if AUTOMATION_ENV_VAR not in (env or {}):
                os.environ.pop(AUTOMATION_ENV_VAR, None)
            if stdin is not None:
                feed = stdin if isinstance(stdin, _Stdin) else _Stdin(stdin)
                stack.enter_context(mock.patch.object(sys, "stdin", feed))
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            try:
                rc = cli.main(list(argv))
            except SystemExit as stop:  # argparse's usage error, as the shell sees it
                rc = stop.code if isinstance(stop.code, int) else 2
        return rc, out.getvalue(), err.getvalue()

    def cli(self, *argv: str, **kwargs: Any) -> tuple[int, str, str]:
        return self._main([*self.cfg, *argv], **kwargs)

    def decline(
        self, reason: str = "too broad", *extra: str, **kwargs: Any
    ) -> tuple[int, str, str]:
        return self.cli("decline", self.cid, "--reason", reason, *extra, **kwargs)

    # -- what is on disk -----------------------------------------------------

    def tree(self) -> Tree:
        lock = self.state.lock_path()
        out: Tree = {}
        for path in sorted(self.base.rglob("*")):
            if path == lock:
                continue
            info = path.lstat()
            data = path.read_bytes() if path.is_file() and not path.is_symlink() else None
            out[str(path.relative_to(self.base))] = (info.st_mode, data)
        return out

    def raw_record(self) -> dict[str, Any]:
        return json.loads(self.record_path.read_text(encoding="utf-8"))

    def edit_record(self, change: Callable[[dict[str, Any]], None]) -> None:
        raw = self.raw_record()
        change(raw)
        self.record_path.write_text(json.dumps(raw), encoding="utf-8")

    def hand_decline(self, **decision: Any) -> None:
        """The record set to ``declined`` by hand, as an older asi or an edit leaves it."""

        def declined(raw: dict[str, Any]) -> None:
            raw["status"] = "declined"
            raw["decline"] = {**HAND_DECISION, **decision}

        self.edit_record(declined)

    def tamper_patch(self) -> None:
        def edited(raw: dict[str, Any]) -> None:
            raw["patch"]["new_text"] += "\nan edit after review\n"

        self.edit_record(edited)

    def manifests(self) -> list[Path]:
        return sorted((self.base / "state" / "backups").rglob("backup.json"))

    def lock_is_held(self) -> bool:
        """Whether the run lock is held now, asked through the unpatched lock.

        Only ``LockHeld`` answers yes: any other error is the probe's own and
        is raised, so a broken probe can never read as a held lock.
        """
        probe = RunLock(self.state.lock_path(), owner="probe: is the run lock held")
        try:
            _REAL_ENTER(probe)
        except LockHeld:
            return True
        _REAL_EXIT(probe, None, None, None)
        return False

    # -- an apply stopped part way ------------------------------------------

    def stop_apply_after_its_skill_write(self) -> None:
        real = skill_fs._write_pinned

        def write_then_stop(*args: Any, **kwargs: Any) -> None:
            real(*args, **kwargs)
            raise cli.Stopped(signal.SIGTERM)

        self._main(self.apply_argv, mock.patch.object(skill_fs, "_write_pinned", write_then_stop))
        assert self.raw_record()["status"] == "validated"
        assert self.manifests(), "the apply wrote its manifest"
        assert self.skill.read_bytes() != self.original, "the apply wrote the skill"

    def stop_apply_before_its_skill_write(self) -> None:
        def stop(*_args: Any, **_kwargs: Any) -> None:
            raise cli.Stopped(signal.SIGTERM)

        self._main(self.apply_argv, mock.patch.object(skill_fs, "_write_pinned", stop))
        assert self.raw_record()["status"] == "validated"
        assert self.manifests(), "the apply wrote its manifest"
        assert self.skill.read_bytes() == self.original, "the skill was not written"


class ADeclinedCandidateCannotBeAppliedOrRolledBackTests(DeclineTreeCase):
    def test_review_refuses_a_declined_candidate_names_it_closed_and_prints_no_token(self) -> None:
        rc, out, err = self.decline()
        self.assertEqual(rc, 0, err)
        rc, out, err = self.cli("review", self.cid)
        self.assertEqual(rc, cli.RC_REFUSED, err)
        self.assertIn("status is 'declined', not 'validated'", err)
        self.assertIn("(applied, rolled-back, gate-failed, blocked-run or declined)", err)
        self.assertNotIn(self.token, out + err)
        self.assertNotIn("--approve", out + err)

    def test_apply_of_a_declined_candidate_is_refused_before_the_lock_and_writes_nothing(
        self,
    ) -> None:
        self.assertEqual(self.decline()[0], 0)
        before = self.tree()
        rc, out, err = self._main(self.apply_argv)
        self.assertEqual(rc, cli.RC_REFUSED, err)
        self.assertIn("has status 'declined'; only a 'validated' candidate may be applied", err)
        self.assertEqual(self.tree(), before)
        with RunLock(self.state.lock_path(), owner="daily 2026-02-02 pid=1"):
            rc, out, err = self._main(self.apply_argv)
        self.assertEqual(rc, cli.RC_REFUSED, err)
        self.assertIn("has status 'declined'", err)
        self.assertNotIn("LockHeld", err)
        self.assertEqual(self.tree(), before)

    def test_an_apply_holding_a_read_from_before_the_decline_is_refused_under_the_lock(
        self,
    ) -> None:
        self.assertEqual(self.decline()[0], 0)
        before = self.tree()
        store = build_skill_store(self.config, writable=True)
        with self.assertRaises(GateFailed) as caught:
            apply_candidate(
                self.read_before,
                approval=HumanApproval.from_cli(token=self.token, acknowledged=True, operator="x"),
                skill_store=store,
                state=self.state,
                clock=clock(),
                operator="x",
            )
        self.assertIn("its record now says 'declined'", str(caught.exception))
        self.assertEqual(self.tree(), before)
        self.assertIsNotNone(self.state.read_candidate(self.cid).decline)

    def test_rollback_of_a_declined_candidate_with_no_manifest_says_so_and_writes_nothing(
        self,
    ) -> None:
        self.assertEqual(self.decline()[0], 0)
        before = self.tree()
        rc, out, err = self._main(self.rollback_argv)
        self.assertEqual(rc, cli.RC_REFUSED, err)
        self.assertIn(f"no backup manifest found for {self.cid}", err)
        self.assertEqual(self.tree(), before)

    def _declined_beside_a_manifest(self) -> Candidate:
        """An apply stopped after its write, a read taken, then the record set declined by hand."""
        self.stop_apply_after_its_skill_write()
        stale = self.state.read_candidate(self.cid)
        self.hand_decline()
        return stale

    def test_a_rollback_holding_a_read_from_before_the_decline_is_refused_and_writes_nothing(
        self,
    ) -> None:
        stale = self._declined_beside_a_manifest()
        (manifest,) = self.manifests()
        before = self.tree()
        store = build_skill_store(self.config, writable=True)
        with self.assertRaises(GateFailed) as caught:
            rollback_candidate(stale, skill_store=store, state=self.state, clock=clock(1))
        said = str(caught.exception)
        self.assertIn(f"{DECLINED_AT} 2026-02-01T13:00:00Z by operator:elder)", said)
        self.assertIn(str(manifest), said)
        self.assertEqual(self.tree(), before)

    def test_cli_rollback_of_a_declined_record_with_a_manifest_names_it_and_the_recovery(
        self,
    ) -> None:
        self._declined_beside_a_manifest()
        (manifest,) = self.manifests()
        before = self.tree()
        rc, out, err = self._main(self.rollback_argv)
        self.assertEqual(rc, cli.RC_REFUSED, err)
        self.assertIn("GateFailed", err)
        self.assertIn(str(manifest), err)
        self.assertIn("'A declined candidate with a backup manifest' in OPERATIONS", err)
        self.assertEqual(self.tree(), before)

    def test_candidates_shows_a_declined_candidate_only_with_all_and_as_closed(self) -> None:
        rc, set_aside_before, err = self.cli("set-aside")
        self.assertEqual(self.decline()[0], 0)
        rc, out, err = self.cli("candidates")
        self.assertEqual(rc, 0, err)
        self.assertNotIn(self.cid, out)
        self.assertIn("closed not listed", out)
        rc, out, err = self.cli("candidates", "--all")
        self.assertEqual(rc, 0, err)
        (line,) = [ln for ln in out.splitlines() if ln.startswith(self.cid)]
        self.assertIn("declined", line)
        self.assertIn("(closed)", line)
        rc, out, err = self.cli("doctor")
        self.assertEqual(rc, 0, err)
        self.assertIn("ready", out)
        rc, set_aside_after, err = self.cli("set-aside")
        self.assertEqual(rc, 0, err)
        self.assertEqual(set_aside_after, set_aside_before)

    def test_a_tampered_declined_record_with_a_manifest_is_refused_as_tampered_with_the_rollback_advice(
        self,
    ) -> None:
        with self.subTest("tampered: it does not recompute"):
            self._declined_beside_a_manifest()
            self.tamper_patch()
            before = self.tree()
            rc, out, err = self._main(self.rollback_argv)
            self.assertEqual(rc, cli.RC_REFUSED, err)
            self.assertIn("CandidateTampered", err)
            self.assertNotIn("GateFailed", err)
            for kept in (KEEP_BOTH, NEVER_RUN, MANUAL):
                self.assertIn(kept, err)
            self.assertNotIn("delete its record", err)
            self.assertNotIn(DECLINED_AT, err)
            self.assertEqual(self.tree(), before)
        with self.subTest("recomputing: it names another skill holding the same text"):
            self.setUp()
            other = self.base / "skills-canonical" / "code-review" / "SKILL.md"
            other.write_bytes(self.original)
            self.stop_apply_after_its_skill_write()
            self.hand_decline()
            handed = self.state.read_candidate(self.cid)
            applied_text = self.skill.read_bytes()
            other.write_bytes(applied_text)

            def to_the_other_skill(raw: dict[str, Any]) -> None:
                raw["target"]["canonical_path"] = str(other)
                raw["target"]["skill_id"] = "code-review"

            self.edit_record(to_the_other_skill)
            (manifest,) = self.manifests()
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            raw.update(target_path=str(other), skill_id="code-review")
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            verify_full_integrity(self.state.read_candidate(self.cid), rubrics=None, action="x")
            before = self.tree()
            store = build_skill_store(self.config, writable=True)
            with self.assertRaises((CandidateTampered, GateFailed)) as caught:
                rollback_candidate(handed, skill_store=store, state=self.state, clock=clock(1))
            self.assertIsInstance(caught.exception, CandidateTampered, caught.exception)
            said = str(caught.exception)
            self.assertIn("its record changed after this rollback read it", said)
            for kept in (KEEP_BOTH, NEVER_RUN, MANUAL):
                self.assertIn(kept, said)
            self.assertNotIn(DECLINED_AT, said)
            self.assertEqual(self.tree(), before)
            self.assertEqual(other.read_bytes(), applied_text)
            self.assertEqual(self.skill.read_bytes(), applied_text)


if __name__ == "__main__":
    unittest.main()
