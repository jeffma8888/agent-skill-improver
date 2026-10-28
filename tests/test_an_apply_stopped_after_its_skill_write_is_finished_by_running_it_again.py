"""An apply stopped after it wrote the skill is finished by running it again.

An apply writes the skill, then records the write: its receipt, and the
candidate marked applied. Measured before (3.11 and 3.13): stopped by SIGTERM as
the skill write returned, it left the skill holding the candidate's text, the
candidate ``validated`` and no backup manifest, since the manifest was written
after the skill; ``rollback`` then answered that the candidate was never
applied, and the next ``apply`` was refused with ``BaseHashMismatch`` because
the skill had "changed since the proposal was made", each exit 3. Left after the
manifest and before the record, the next ``apply`` was refused the same way
while ``doctor`` said ready.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import signal
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence
from unittest import mock

import agent_skill_improver.adapters.skill_fs as skill_fs
from agent_skill_improver import cli
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.hashing import sha256_text

from tests.support import build_env, run_cli

DAY = "2026-02-01"
FINISHES = (
    "finishes        an earlier run of this apply, which wrote the skill and did not record it"
)


class AnApplyStoppedAfterItsSkillWriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name).resolve()
        cls.base = root / "env"
        cls.cfg = ["--config", str(build_env(cls.base, dry_run=False, seed="stopped-apply"))]
        daily = run_cli([*cls.cfg, "--json", "daily", "--commit", "--day", DAY])
        assert daily.returncode == 0, daily.stderr
        listed = json.loads(run_cli([*cls.cfg, "--json", "candidates"]).stdout)["candidates"]
        cls.cid = sorted(c["id"] for c in listed)[0]
        review = json.loads(run_cli([*cls.cfg, "--json", "review", cls.cid]).stdout)
        cls.token = review["approval_token"]
        cls.record = cls.base / "state" / "candidates" / f"{cls.cid}.json"
        cls.receipt = cls.base / "state" / "receipts" / f"apply--{cls.cid}.json"
        target = json.loads(cls.record.read_text())["target"]
        cls.target = Path(target["canonical_path"])
        cls.mirror = Path(target["mirror_paths"][0])
        cls.before = cls.target.read_text()
        cls.apply = [*cls.cfg, "apply", cls.cid, "--approve", cls.token, "--i-reviewed-this"]
        cls.rollback = [*cls.cfg, "rollback", cls.cid]
        cls.pristine = root / "pristine"
        shutil.copytree(cls.base, cls.pristine, symlinks=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self._fresh()

    def _fresh(self) -> None:
        """Put back the tree as it was with the candidate reviewed and not applied."""
        shutil.rmtree(self.base)
        shutil.copytree(self.pristine, self.base, symlinks=True)

    def _status(self) -> str:
        return str(json.loads(self.record.read_text())["status"])

    def _manifests(self) -> list[Path]:
        return sorted((self.base / "state" / "backups").rglob("backup.json"))

    def _in_process(self, argv: Sequence[str], *patches: Any) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stderr(err))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            rc = cli.main(list(argv))
        return rc, err.getvalue()

    def _stopped_as_the_skill_write_returns(self, argv: Sequence[str]) -> tuple[int, str]:
        real = skill_fs._write_pinned

        def write_then_stop(*args: Any, **kwargs: Any) -> None:
            real(*args, **kwargs)
            raise cli.Stopped(signal.SIGTERM)

        return self._in_process(argv, mock.patch.object(skill_fs, "_write_pinned", write_then_stop))

    def _stopped_at(self, method: str) -> tuple[int, str]:
        """The apply, stopped by SIGTERM as ``StateStore.<method>`` is called."""

        def stop(*_args: Any, **_kwargs: Any) -> None:
            raise cli.Stopped(signal.SIGTERM)

        return self._in_process(self.apply, mock.patch.object(StateStore, method, stop))

    def _assert_finished_by_running_it_again(self) -> None:
        again = run_cli(self.apply)
        self.assertEqual(again.returncode, cli.RC_OK, again.stderr)
        self.assertIn(f"applied {self.cid} to ", again.stdout)
        self.assertIn(FINISHES, again.stdout)
        self.assertEqual(self._status(), "applied")
        self.assertIs(json.loads(self.receipt.read_text())["finishes_an_earlier_run"], True)
        notes = json.loads(self.record.read_text())["notes"]
        self.assertIn("recorded by a second run of this apply", notes[-1])
        rolled = run_cli(self.rollback)
        self.assertEqual(rolled.returncode, cli.RC_OK, rolled.stderr)
        self.assertEqual(self.target.read_text(), self.before)
        self.assertEqual(self._status(), "rolled-back")

    def _torn(self) -> None:
        """The apply stopped after the skill write and before anything recorded it."""
        rc, err = self._stopped_as_the_skill_write_returns(self.apply)
        self.assertEqual(rc, cli.RC_STOPPED_BY_SIGTERM, err)
        self.assertNotEqual(self.target.read_text(), self.before, "the write landed")
        self.assertEqual(self._status(), "validated")
        self.assertFalse(self.receipt.exists())

    def _torn_after_the_manifest(self) -> None:
        """The state a stop between the manifest and the record leaves, made by
        applying and then putting back the record as it was before."""
        before = self.record.read_bytes()
        applied = run_cli(self.apply)
        self.assertEqual(applied.returncode, cli.RC_OK, applied.stderr)
        self.record.write_bytes(before)
        self.receipt.unlink()

    def _assert_refused_as_before(self, refused: Any) -> None:
        self.assertEqual(refused.returncode, cli.RC_REFUSED, refused.stderr)
        self.assertIn(f"BaseHashMismatch: refusing to apply {self.cid}", refused.stderr)
        self.assertIn("The skill changed since the proposal was made", refused.stderr)
        self.assertEqual(self._status(), "validated")
        self.assertFalse(self.receipt.exists())

    def test_an_apply_stopped_as_its_skill_write_returned_is_finished_by_running_it_again(
        self,
    ) -> None:
        rc, err = self._stopped_as_the_skill_write_returns(self.apply)
        self.assertEqual(rc, cli.RC_STOPPED_BY_SIGTERM, err)
        self.assertIn("stopped by SIGTERM before the command finished", err)
        self.assertIn("run the same apply again to finish it", err)
        self.assertNotEqual(self.target.read_text(), self.before, "the write landed")
        self.assertEqual(self._status(), "validated")
        self.assertEqual(len(self._manifests()), 1, "the manifest was written before the skill")
        review = run_cli([*self.cfg, "review", self.cid])
        self.assertEqual(review.returncode, cli.RC_REFUSED, review.stderr)
        self.assertIn(f"BaseHashMismatch: refusing to review {self.cid}", review.stderr)
        self._assert_finished_by_running_it_again()

    def test_an_apply_stopped_as_it_records_its_write_is_finished_too(self) -> None:
        for method in ("write_action_receipt", "write_candidate"):
            with self.subTest(stopped_at=method):
                self._fresh()
                rc, err = self._stopped_at(method)
                self.assertEqual(rc, cli.RC_STOPPED_BY_SIGTERM, err)
                self.assertNotEqual(self.target.read_text(), self.before, "the write landed")
                self.assertEqual(self._status(), "validated")
                self._assert_finished_by_running_it_again()

    def test_a_stop_after_the_apply_was_recorded_says_the_rerun_has_nothing_to_do(
        self,
    ) -> None:
        real = StateStore.write_candidate

        def record_then_stop(store: StateStore, candidate: Any) -> Path:
            real(store, candidate)
            raise cli.Stopped(signal.SIGTERM)

        rc, err = self._in_process(
            self.apply, mock.patch.object(StateStore, "write_candidate", record_then_stop)
        )
        self.assertEqual(rc, cli.RC_STOPPED_BY_SIGTERM, err)
        self.assertIn(
            "if this run had recorded it, the apply is refused as already `applied`, with "
            "nothing left to do",
            err,
        )
        self.assertEqual(self._status(), "applied")
        self.assertTrue(self.receipt.exists())
        again = run_cli(self.apply)
        self.assertEqual(again.returncode, cli.RC_REFUSED, again.stderr)
        self.assertIn("has status 'applied'", again.stderr)
        self.assertEqual(self._status(), "applied")

    def test_the_second_run_still_needs_the_candidates_token(self) -> None:
        self._torn()
        wrong = self.token[:-1] + ("0" if self.token[-1] != "0" else "1")
        refused = run_cli([*self.cfg, "apply", self.cid, "--approve", wrong, "--i-reviewed-this"])
        self.assertEqual(refused.returncode, cli.RC_REFUSED, refused.stderr)
        self.assertIn(
            "ApprovalRequired: approval token does not match this candidate", refused.stderr
        )
        self.assertEqual(self._status(), "validated")
        self.assertFalse(self.receipt.exists())

    def test_the_control_an_edit_after_the_stopped_apply_is_refused_as_before(self) -> None:
        self._torn()
        self.target.write_text(self.target.read_text() + "\nedited by hand after the stop\n")
        self._assert_refused_as_before(run_cli(self.apply))

    def test_an_edit_the_manifest_is_edited_to_record_is_refused_too(self) -> None:
        # What the skill holds must be the candidate's own text, whatever the
        # manifest, a file anyone with the state tree can edit, says was written.
        self._torn_after_the_manifest()
        self.target.write_text(self.target.read_text() + "\nedited by hand after the stop\n")
        (manifest,) = self._manifests()
        raw = json.loads(manifest.read_text())
        raw["new_sha256"] = sha256_text(self.target.read_text())
        manifest.write_text(json.dumps(raw))
        self._assert_refused_as_before(run_cli(self.apply))

    def test_a_manifest_that_does_not_record_this_write_is_not_taken_for_it(self) -> None:
        def edit(**fields: str) -> Any:
            def apply_edit(manifest: Path) -> None:
                raw = json.loads(manifest.read_text())
                raw.update(fields)
                manifest.write_text(json.dumps(raw))

            return apply_edit

        cases = {
            "another base": edit(previous_sha256="0" * 64),
            "another text": edit(new_sha256="0" * 64),
            "another document": edit(target_path=str(self.mirror.resolve())),
            "unreadable": lambda manifest: manifest.write_text("{"),
        }
        for name, spoil in cases.items():
            with self.subTest(manifest=name):
                self._fresh()
                self._torn_after_the_manifest()
                (manifest,) = self._manifests()
                spoil(manifest)
                self._assert_refused_as_before(run_cli(self.apply))

    @unittest.skipIf(os.geteuid() == 0, "root writes through a read-only directory")
    def test_the_control_a_write_the_platform_refuses_leaves_no_manifest(self) -> None:
        skill_dir = self.target.parent
        mode = skill_dir.stat().st_mode
        skill_dir.chmod(0o555)
        self.addCleanup(skill_dir.chmod, mode)
        refused = run_cli(self.apply)
        self.assertEqual(refused.returncode, cli.RC_REFUSED, refused.stderr)
        self.assertIn("SkillWriteFailed", refused.stderr)
        self.assertEqual(self.target.read_text(), self.before)
        self.assertEqual(self._manifests(), [])
        backups = list((self.base / "state" / "backups").rglob("*.bak"))
        self.assertEqual(len(backups), 1, "the copy taken before the write is kept")
        nothing = run_cli(self.rollback)
        self.assertEqual(nothing.returncode, cli.RC_REFUSED, nothing.stderr)
        self.assertIn("no backup manifest found", nothing.stderr)

    def test_the_control_a_write_that_does_not_verify_and_is_put_back_leaves_no_manifest(
        self,
    ) -> None:
        def misread(*_args: Any, **_kwargs: Any) -> tuple[str, str]:
            return "0" * 64, "0" * 12

        rc, err = self._in_process(
            self.apply, mock.patch.object(skill_fs.FilesystemSkillStore, "_read_back", misread)
        )
        self.assertEqual(rc, cli.RC_REFUSED, err)
        self.assertIn("original restored", err)
        self.assertEqual(self.target.read_text(), self.before)
        self.assertEqual(self._manifests(), [])
        self.assertEqual(self._status(), "validated")

    def test_a_rollback_after_an_apply_stopped_before_its_skill_write_leaves_it_to_apply(
        self,
    ) -> None:
        def stop(*_args: Any, **_kwargs: Any) -> None:
            raise cli.Stopped(signal.SIGTERM)

        rc, err = self._in_process(self.apply, mock.patch.object(skill_fs, "_write_pinned", stop))
        self.assertEqual(rc, cli.RC_STOPPED_BY_SIGTERM, err)
        self.assertEqual(self.target.read_text(), self.before, "nothing was written")
        self.assertEqual(len(self._manifests()), 1, "the manifest is written ahead of the skill")
        refused = run_cli(self.rollback)
        self.assertEqual(refused.returncode, cli.RC_REFUSED, refused.stderr)
        self.assertIn(f"GateFailed: {self.cid} was never applied", refused.stderr)
        self.assertIn("run that apply again to apply it", refused.stderr)
        self.assertEqual(self._status(), "validated")
        applied = run_cli(self.apply)
        self.assertEqual(applied.returncode, cli.RC_OK, applied.stderr)
        self.assertNotIn(FINISHES, applied.stdout, "an apply as usual: the skill was at its base")
        self.assertEqual(self._status(), "applied")

    def test_rollback_restores_an_apply_stopped_as_its_skill_write_returned(
        self,
    ) -> None:
        self._torn()
        rolled = run_cli(self.rollback)
        self.assertEqual(rolled.returncode, cli.RC_OK, rolled.stderr)
        self.assertEqual(self.target.read_text(), self.before)
        self.assertEqual(self._status(), "rolled-back")

    def test_a_stopped_rollback_says_to_run_it_again_and_running_it_again_finishes_it(
        self,
    ) -> None:
        applied = run_cli(self.apply)
        self.assertEqual(applied.returncode, cli.RC_OK, applied.stderr)
        self.assertNotIn(FINISHES, applied.stdout, "a first run finishes nothing")
        rc, err = self._stopped_as_the_skill_write_returns(self.rollback)
        self.assertEqual(rc, cli.RC_STOPPED_BY_SIGTERM, err)
        self.assertIn("run the same rollback again to finish it", err)
        self.assertEqual(self.target.read_text(), self.before, "the restore landed")
        self.assertEqual(self._status(), "applied")
        again = run_cli(self.rollback)
        self.assertEqual(again.returncode, cli.RC_OK, again.stderr)
        self.assertIn("\n  nothing written the file already held the backup's text", again.stdout)
        self.assertEqual(self._status(), "rolled-back")

    def test_the_control_a_stop_of_another_command_says_nothing_of_running_it_again(
        self,
    ) -> None:
        def stop(*_args: Any, **_kwargs: Any) -> list[str]:
            raise cli.Stopped(signal.SIGTERM)

        rc, err = self._in_process(
            [*self.cfg, "doctor"],
            mock.patch.object(skill_fs.FilesystemSkillStore, "known_skill_ids", stop),
        )
        self.assertEqual(rc, cli.RC_STOPPED_BY_SIGTERM, err)
        self.assertIn("stopped by SIGTERM before the command finished", err)
        self.assertNotIn("run the same", err)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
