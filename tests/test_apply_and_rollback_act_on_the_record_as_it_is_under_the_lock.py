"""Apply and rollback act on the candidate record as it is once they hold the run lock.

A promotion is handed a record read before it waited for the run lock, and
another apply or rollback of the same candidate may finish in between. Measured
before (3.11 and 3.13), each acted on the read it was handed: an apply holding a
read made before another apply finished took the finishing path, recorded that
write again as its own and replaced the first apply's receipt (operator
``first`` became ``second``); one holding a read made before a rollback wrote the
skill again over the restored text; a rollback holding a read made before the
apply, or before a first rollback, wrote its record from that read and dropped
the ``applied`` note or the first rollback's note. Now apply reads the record
again first under the lock and refuses anything but ``validated``, and rollback
reads it again under the lock after choosing its manifest and decides, restores
and records from what it read; a record deleted or made unreadable meanwhile is
refused with nothing written.

Rollback checks the record it read again against itself before it decides
anything from it (``verify_full_integrity`` with ``rubrics=None``). Measured
without that check (3.11 and 3.13), with the record and the backup manifest
edited together after the command read the record and before it held the lock:
an edit naming another canonical skill restored that skill from a planted
backup, and one naming a later edit of the skill as the apply's write erased
that edit, each exiting 0. Now each is refused as ``CandidateTampered``, with
advice that keeps the record and the backup, and nothing is written.

It then holds that record to the one it was handed: the candidate, document,
skill, base and replacement must be the same. Measured with the check of the
record against itself and without this (3.11 and 3.13), the record and its
manifest edited as the rollback reached for the lock to name another canonical
skill holding the same applied text, every hash left as it was, the record
recomputed and the rollback exited 0, overwrote that skill and erased its edit.
Now it is refused as ``CandidateTampered`` before the never-applied guard and
any write. The advice of either refusal never sends the operator to a command
taken from the state files, and points to OPERATIONS for restoring by hand.
"""

from __future__ import annotations

import contextlib
import datetime
import io
import json
import re
import shutil
import signal
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterator, Sequence
from unittest import mock

import agent_skill_improver.adapters.skill_fs as skill_fs
from agent_skill_improver import cli
from agent_skill_improver.app.lock import RunLock
from agent_skill_improver.app.promote import apply_candidate, rollback_candidate
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.app.wiring import build_skill_store
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.config import load_config
from agent_skill_improver.domain.approval import HumanApproval
from agent_skill_improver.domain.candidate import Candidate
from agent_skill_improver.domain.integrity import repropose_advice, verify_full_integrity
from agent_skill_improver.domain.validation import ValidationRecord
from agent_skill_improver.errors import AsiError, CandidateTampered, GateFailed, LockHeld
from agent_skill_improver.hashing import sha256_text

from tests.support import build_env, run_cli

DAY = "2026-02-01"
T0 = datetime.datetime(2026, 2, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
FINISHES = (
    "finishes        an earlier run of this apply, which wrote the skill and did not record it"
)

#: What each file and directory under the environment holds: its mode, and a
#: file's bytes. The run lock's own file is left out: taking the lock writes
#: its owner line, which is not a write of the promotion.
Tree = dict[str, tuple[int, bytes | None]]


#: What a rollback refused for its record says, in the order it says it.
KEEP = "Nothing was restored or written: the skill, this record and its backup are as they were."
NEVER_RUN = (
    "Never run the backup manifest's rollback_command, or any other command taken from the "
    "state files: they are editable, and their agreeing with each other establishes no trust."
)
PUT_BACK = (
    "Put the record back only if you know its original contents from outside the state "
    "directory, then run `asi rollback {cid}` again"
)
MANUAL = 'otherwise restore the skill as docs/OPERATIONS.md, Recovery, "Manual restoration" says.'
#: The one description of restoring by hand, which the refusals point to.
OPERATIONS = Path(__file__).resolve().parents[1] / "docs" / "OPERATIONS.md"
MANUAL_HEAD = "**Manual restoration.**"
#: A line that opens a block without a blank line before it: a quote, a list
#: item or a heading.
BLOCK_OPENER = re.compile(r"[ \t]*(?:>|[-+*][ \t]|\d+[.)][ \t]|#)")
#: The quote, list, task-list box and heading marks that open a block.
BLOCK_MARKS = re.compile(
    r"[ \t]*(?:(?:>|[-+*](?=[ \t])|\d+[.)](?=[ \t])|\[[ xX]\](?=[ \t])|#{1,6})[ \t]*)*"
)
#: The paragraph's name, in any letter case.
MANUAL_NAME = re.compile(r"(?i:manual restoration)")
#: What ends a label: the end of the line, a period, colon, dash or closing
#: hashes, or a run-in sentence (a capital letter). Other words after the name
#: make a mention, not a label.
LABEL_END = re.compile(r"[ \t]*(?:$|[.:#\u2014\u2013-])|[ \t]+[A-Z]")
#: A Setext heading's underline.
UNDERLINE = re.compile(r"[ \t]*(?:=+|-+)[ \t]*$")
#: An emphasised run at the start of a block.
EMPHASIS = re.compile(r"(\*\*|__|\*|_)(.+?)\1")
#: Emphasis and code marks, dropped before the name is looked for.
MARKUP = str.maketrans("", "", "*_`")


def _restoration_headings(text: str) -> int:
    """How many blocks of ``text`` are headed Manual restoration.

    A block opens at the start of the text, after a blank line, or at a quote,
    list item or heading line. It is headed by the name when, its emphasis and
    code marks dropped, it opens with the name and that opening is a Markdown
    heading, an emphasised run that ends in a period or colon or is followed by
    a label's end (either one with more words after the name or not), or the
    name and then a label's end (LABEL_END). Not a Markdown parser: it
    recognises the ways a contributor heads a block, not every way one could be
    disguised, and it errs toward counting: a block that only mentions the
    paragraph but opens with its name and then a colon or a capital letter, or
    is one bold sentence that starts with it, counts as a heading.
    """
    lines = text.split("\n")
    count = 0
    for i, line in enumerate(lines):
        if i and lines[i - 1].strip() and not BLOCK_OPENER.match(line):
            continue
        marks = BLOCK_MARKS.match(line)
        prefix = marks.group() if marks else ""
        opened = line[len(prefix) :].lstrip()
        rest = opened.translate(MARKUP).lstrip()
        name = MANUAL_NAME.match(rest)
        if name is None:
            continue
        heading = "#" in prefix or (
            i + 1 < len(lines) and UNDERLINE.match(lines[i + 1]) is not None
        )
        run = EMPHASIS.match(opened)
        label = (
            run is not None
            and MANUAL_NAME.match(run.group(2).translate(MARKUP)) is not None
            and (
                run.group(2).rstrip().endswith((".", ":"))
                or LABEL_END.match(opened[run.end() :].translate(MARKUP)) is not None
            )
        )
        if heading or label or LABEL_END.match(rest, name.end()) is not None:
            count += 1
    return count


#: How each of the two refusals begins, after ``refusing to roll back candidate <id>: ``.
NOT_RECOMPUTING = "derived field(s) do not match a fresh recomputation"
CHANGED_SINCE_READ = "its record changed after this rollback read it: "


def _clock(hours: int = 0) -> FrozenClock:
    return FrozenClock(T0 + datetime.timedelta(hours=hours))


def _refused_after_it_read(cid: str, status: str) -> str:
    return (
        f"refusing to apply {cid}: its record now says '{status}', not 'validated': it "
        "changed after this apply read it. Nothing was written."
    )


class ApplyAndRollbackActOnTheRecordAsItIsUnderTheLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name).resolve()
        cls.base = root / "env"
        cls.config_path = build_env(cls.base, dry_run=False, seed="re-read-under-the-lock")
        cls.cfg = ["--config", str(cls.config_path)]
        daily = run_cli([*cls.cfg, "--json", "daily", "--commit", "--day", DAY])
        assert daily.returncode == 0, daily.stderr
        listed = json.loads(run_cli([*cls.cfg, "--json", "candidates"]).stdout)["candidates"]
        cls.cid = sorted(c["id"] for c in listed)[0]
        review = json.loads(run_cli([*cls.cfg, "--json", "review", cls.cid]).stdout)
        cls.token = review["approval_token"]
        cls.apply_argv = [*cls.cfg, "apply", cls.cid, "--approve", cls.token, "--i-reviewed-this"]
        cls.rollback_argv = [*cls.cfg, "rollback", cls.cid]
        cls.pristine = root / "pristine"
        shutil.copytree(cls.base, cls.pristine, symlinks=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        """The tree as it was with the candidate validated and reviewed, and a read of it."""
        shutil.rmtree(self.base)
        shutil.copytree(self.pristine, self.base, symlinks=True)
        config = load_config(self.config_path)
        self.state = StateStore(config.state_dir)
        self.store = build_skill_store(config, writable=True)
        # Read now, before anything below happens: the read a promotion that
        # waited for the lock was handed.
        self.read_before = self.state.read_candidate(self.cid)
        self.assertEqual(self.read_before.status, "validated")
        self.skill = Path(self.read_before.target.canonical_path)
        self.original = self.skill.read_bytes()
        self.record_path = self.state.candidate_path(self.cid)
        self.apply_receipt = self.state.action_receipt_path("apply", self.cid)
        self.rollback_receipt = self.state.action_receipt_path("rollback", self.cid)

    # -- helpers -------------------------------------------------------------

    def _approval(self, operator: str) -> HumanApproval:
        return HumanApproval.from_cli(token=self.token, acknowledged=True, operator=operator)

    def _apply(self, candidate: Candidate, operator: str, hours: int = 0) -> dict[str, Any]:
        return apply_candidate(
            candidate,
            approval=self._approval(operator),
            skill_store=self.store,
            state=self.state,
            clock=_clock(hours),
            operator=operator,
        )

    def _rollback(self, candidate: Candidate, hours: int = 1) -> dict[str, Any]:
        return rollback_candidate(
            candidate, skill_store=self.store, state=self.state, clock=_clock(hours)
        )

    def _record(self) -> Candidate:
        return self.state.read_candidate(self.cid)

    def _tree(self) -> Tree:
        lock = self.state.lock_path()
        out: Tree = {}
        for path in sorted(self.base.rglob("*")):
            if path == lock:
                continue
            info = path.lstat()
            data = path.read_bytes() if path.is_file() and not path.is_symlink() else None
            out[str(path.relative_to(self.base))] = (info.st_mode, data)
        return out

    def _manifests(self) -> list[Path]:
        return sorted((self.base / "state" / "backups").rglob("backup.json"))

    def _lock_is_held(self) -> bool:
        """Whether some holder has the run lock now, asked of the lock itself."""
        probe = RunLock(self.state.lock_path(), owner="probe: is the run lock held")
        try:
            self._real_enter(probe)
        except LockHeld:
            return True
        self._real_exit(probe, None, None, None)
        return False

    @contextlib.contextmanager
    def _as_the_next_promotion_reaches_for_the_lock(
        self, competing: Callable[[], None]
    ) -> Iterator[list[tuple[str, bool]]]:
        """Run ``competing`` once, as the next promotion is about to take the run lock.

        Yields the record reads and manifest lookups made meanwhile, each with
        whether the run lock was held when it was made, asked of the lock.
        """
        self._real_enter = RunLock.__enter__
        self._real_exit = RunLock.__exit__
        real_read = StateStore.read_candidate
        real_manifest = StateStore.latest_backup_manifest
        events: list[tuple[str, bool]] = []
        pending = [competing]

        def entering(lock: RunLock) -> RunLock:
            if pending:
                pending.pop()()
            return self._real_enter(lock)

        def reading(store: StateStore, candidate_id: str) -> Candidate:
            events.append(("read", self._lock_is_held()))
            return real_read(store, candidate_id)

        def choosing(store: StateStore, candidate_id: str) -> Path | None:
            events.append(("manifest", self._lock_is_held()))
            return real_manifest(store, candidate_id)

        with (
            mock.patch.object(RunLock, "__enter__", entering),
            mock.patch.object(StateStore, "read_candidate", reading),
            mock.patch.object(StateStore, "latest_backup_manifest", choosing),
        ):
            yield events
        self.assertEqual(pending, [], "the competing promotion ran")

    def _in_process(self, argv: Sequence[str], *patches: Any) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stderr(err))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            rc = cli.main(list(argv))
        return rc, err.getvalue()

    def _cli_with_the_record_changed_at_the_lock(
        self, argv: Sequence[str], change: Callable[[], None]
    ) -> tuple[int, str]:
        """The command through ``cli.main``, ``change`` made as it reaches for the lock.

        The CLI reads the record itself before it calls apply or rollback, so a
        change made before the command starts is refused by that read; this one
        is made after it, where a competing process would make it.
        """
        real_enter = RunLock.__enter__
        pending = [change]

        def entering(lock: RunLock) -> RunLock:
            if pending:
                pending.pop()()
            return real_enter(lock)

        rc, err = self._in_process(argv, mock.patch.object(RunLock, "__enter__", entering))
        self.assertEqual(pending, [], "the change was made as the command reached for the lock")
        return rc, err

    def _without(self, tree: Tree, *paths: Path) -> Tree:
        names = {str(p.relative_to(self.base)) for p in paths}
        return {name: entry for name, entry in tree.items() if name not in names}

    # -- apply ---------------------------------------------------------------

    def test_a_second_apply_holding_a_read_from_before_the_first_is_refused_and_the_first_receipt_kept(
        self,
    ) -> None:
        self._apply(self.read_before, "first")
        after_first = self._tree()
        first_receipt = json.loads(self.apply_receipt.read_text())
        self.assertEqual(first_receipt["operator"], "first")
        with self.assertRaises(GateFailed) as caught:
            self._apply(self.read_before, "second", hours=1)
        self.assertEqual(str(caught.exception), _refused_after_it_read(self.cid, "applied"))
        self.assertEqual(self._tree(), after_first, "record, receipt, backups and skill kept")
        self.assertEqual(json.loads(self.apply_receipt.read_text())["operator"], "first")
        self.assertEqual(self._record().status, "applied")
        self.assertEqual(self.skill.read_text(), self.read_before.patch.new_text)

    def test_an_apply_holding_a_read_from_before_a_rollback_is_refused_and_the_skill_stays_restored(
        self,
    ) -> None:
        self._apply(self.read_before, "first")
        self._rollback(self._record())
        self.assertEqual(self.skill.read_bytes(), self.original)
        after_rollback = self._tree()
        with self.assertRaises(GateFailed) as caught:
            self._apply(self.read_before, "stale", hours=2)
        self.assertEqual(str(caught.exception), _refused_after_it_read(self.cid, "rolled-back"))
        self.assertEqual(self.skill.read_bytes(), self.original, "the skill stays restored")
        self.assertEqual(len(self._manifests()), 1, "no new backup manifest")
        self.assertEqual(self._tree(), after_rollback)
        self.assertEqual(self._record().status, "rolled-back")

    def test_an_apply_reads_its_record_while_it_holds_the_lock_so_one_finished_as_it_waited_is_honoured(
        self,
    ) -> None:
        # Another apply of the same candidate finishes in the moment this one
        # reaches for the lock: after this one was handed its read and passed
        # every check before the lock, before it takes the lock. A read of the
        # record made before the lock is taken still says ``validated``.
        competed: list[Tree] = []

        def another_apply_finishes() -> None:
            self._apply(self.read_before, "first")
            competed.append(self._tree())

        with self._as_the_next_promotion_reaches_for_the_lock(another_apply_finishes) as events:
            with self.assertRaises(GateFailed) as caught:
                self._apply(self.read_before, "second", hours=1)
        self.assertEqual(str(caught.exception), _refused_after_it_read(self.cid, "applied"))
        self.assertEqual(
            events,
            [("read", True), ("read", True)],
            "each apply read the record once, while the run lock was held",
        )
        self.assertEqual(self._tree(), competed[0], "the finished apply's record and receipt kept")
        self.assertEqual(json.loads(self.apply_receipt.read_text())["operator"], "first")
        self.assertNotIn("finishes_an_earlier_run", json.loads(self.apply_receipt.read_text()))

    def test_an_apply_stopped_after_its_write_still_finishes_when_run_again_with_the_re_read(
        self,
    ) -> None:
        real = skill_fs._write_pinned

        def write_then_stop(*args: Any, **kwargs: Any) -> None:
            real(*args, **kwargs)
            raise cli.Stopped(signal.SIGTERM)

        rc, err = self._in_process(
            self.apply_argv, mock.patch.object(skill_fs, "_write_pinned", write_then_stop)
        )
        self.assertEqual(rc, cli.RC_STOPPED_BY_SIGTERM, err)
        self.assertEqual(
            self.skill.read_text(), self.read_before.patch.new_text, "the write landed"
        )
        self.assertEqual(self._record().status, "validated")
        self.assertEqual(len(self._manifests()), 1)
        again = run_cli(self.apply_argv)
        self.assertEqual(again.returncode, cli.RC_OK, again.stderr)
        self.assertIn(f"applied {self.cid} to ", again.stdout)
        self.assertIn(FINISHES, again.stdout)
        self.assertEqual(self._record().status, "applied")
        self.assertIs(json.loads(self.apply_receipt.read_text())["finishes_an_earlier_run"], True)
        self.assertIn("recorded by a second run of this apply", self._record().notes[-1])

    # -- rollback ------------------------------------------------------------

    def test_a_rollback_holding_a_read_from_before_the_apply_keeps_the_applied_note(self) -> None:
        self._apply(self.read_before, "first")
        applied = self._record()
        self.assertEqual(len(applied.notes), len(self.read_before.notes) + 1)
        self.assertTrue(applied.notes[-1].startswith("applied "), applied.notes)
        self._rollback(self.read_before)
        after = self._record()
        self.assertEqual(after.status, "rolled-back")
        self.assertEqual(self.skill.read_bytes(), self.original)
        self.assertEqual(after.notes[:-1], applied.notes, "the applied note is kept")
        self.assertTrue(after.notes[-1].startswith("rolled back "), after.notes)
        self.assertEqual(len(after.notes), 2)

    def test_a_second_rollback_holding_an_old_read_keeps_the_first_rollbacks_note(self) -> None:
        self._apply(self.read_before, "first")
        applied = self._record()
        self._rollback(applied, hours=1)
        first = self._record()
        self.assertEqual(len(first.notes), 2)
        second = self._rollback(applied, hours=2)
        self.assertIs(second["result"]["already_restored"], True)
        after = self._record()
        self.assertEqual(after.status, "rolled-back")
        self.assertEqual(after.notes[:-1], first.notes, "the first rollback's note is kept")
        self.assertTrue(after.notes[-1].startswith("rolled back "), after.notes)
        self.assertEqual(len(after.notes), 3)
        self.assertEqual(self.skill.read_bytes(), self.original)

    def test_a_rollback_reads_its_record_while_it_holds_the_lock_after_choosing_its_manifest(
        self,
    ) -> None:
        # A first rollback finishes in the moment this second one reaches for
        # the lock, as it would if both were started together.
        self._apply(self.read_before, "first")
        applied = self._record()
        competed: list[Candidate] = []

        def another_rollback_finishes() -> None:
            self._rollback(applied, hours=1)
            competed.append(real_read(self.state, self.cid))

        real_read = StateStore.read_candidate
        with self._as_the_next_promotion_reaches_for_the_lock(another_rollback_finishes) as events:
            self._rollback(applied, hours=2)
        self.assertEqual(
            events,
            [("manifest", True), ("read", True)] * 2,
            "each rollback chose its manifest, then read the record, holding the run lock",
        )
        after = self._record()
        self.assertEqual(len(competed[0].notes), 2)
        self.assertEqual(after.notes[:-1], competed[0].notes, "the first rollback's note is kept")
        self.assertEqual(len(after.notes), 3)
        self.assertEqual(self.skill.read_bytes(), self.original)

    def test_a_rollback_holding_a_read_from_before_an_apply_and_its_rollback_is_judged_on_the_record_as_it_is(
        self,
    ) -> None:
        # The handed read says ``validated`` and the skill holds its base text
        # again, which is what the never-applied refusal looks for; the record
        # says ``rolled-back``, so this is a second rollback, as it is when
        # handed the record as it is.
        self._apply(self.read_before, "first")
        self._rollback(self._record(), hours=1)
        first = self._record()
        try:
            out = self._rollback(self.read_before, hours=2)
        except GateFailed as exc:
            self.fail(f"judged on the read it was handed, not the record: {exc}")
        self.assertIs(out["result"]["already_restored"], True)
        after = self._record()
        self.assertEqual(after.status, "rolled-back")
        self.assertEqual(after.notes[:-1], first.notes)
        self.assertEqual(len(after.notes), 3)

    def test_a_rollback_handed_a_read_that_differs_from_its_record_in_what_it_restores_is_refused(
        self,
    ) -> None:
        # The record on disk is untouched and recomputes; the object handed to
        # the rollback, the read the command made before it waited for the
        # lock, differs from it in one field the restore is made from. The id
        # cannot differ: the re-read is made by it, and ``read_candidate``
        # refuses a record that does not claim it.
        other = self.base / "skills-canonical" / "code-review" / "SKILL.md"
        cases: dict[str, Callable[[Candidate], Candidate]] = {
            "target.canonical_path": lambda c: replace(
                c, target=replace(c.target, canonical_path=str(other))
            ),
            "target.skill_id": lambda c: replace(
                c, target=replace(c.target, skill_id="code-review")
            ),
            "target.base_sha256": lambda c: replace(
                c, target=replace(c.target, base_sha256="0" * 64)
            ),
            "patch.new_text": lambda c: replace(
                c, patch=replace(c.patch, new_text=c.patch.new_text + "\nhanded\n")
            ),
        }
        for field, differ in cases.items():
            with self.subTest(field=field):
                self.setUp()
                self._apply(self.read_before, "first")
                applied = self._record()
                before = self._tree()
                with mock.patch.object(
                    self.store, "rollback", wraps=self.store.rollback
                ) as restore:
                    with self.assertRaises(AsiError) as caught:
                        self._rollback(differ(applied))
                self.assertIsInstance(caught.exception, CandidateTampered)
                self._assert_changed_since_read(str(caught.exception), field)
                self.assertEqual(restore.call_count, 0, "nothing restored")
                self.assertEqual(self._tree(), before, "nothing written")
                self.assertEqual(self._record(), applied)

    # -- a record deleted or made unreadable while the command waited ----------

    def test_an_apply_or_rollback_whose_record_was_deleted_under_it_writes_nothing(self) -> None:
        for name, argv in (("apply", self.apply_argv), ("rollback", self.rollback_argv)):
            with self.subTest(command=name):
                self.setUp()
                if name == "rollback":
                    self._apply(self.read_before, "first")
                before = self._tree()
                rc, err = self._cli_with_the_record_changed_at_the_lock(
                    argv, self.record_path.unlink
                )
                self.assertEqual(rc, cli.RC_MISSING_FILE, err)
                self.assertTrue(err.startswith(f"not found: no candidate '{self.cid}'"), err)
                self.assertEqual(self._tree(), self._without(before, self.record_path))
                self.assertFalse(self.record_path.exists(), "no record written back")

    def test_an_apply_whose_record_became_unreadable_under_it_writes_nothing(self) -> None:
        before = self._tree()
        rc, err = self._cli_with_the_record_changed_at_the_lock(
            self.apply_argv, lambda: self.record_path.write_text("{", encoding="utf-8")
        )
        self.assertEqual(rc, cli.RC_REFUSED, err)
        self.assertTrue(err.startswith("StateUnreadable: "), err)
        after = self._tree()
        self.assertEqual(after.pop(str(self.record_path.relative_to(self.base)))[1], b"{")
        self.assertEqual(after, self._without(before, self.record_path))
        self.assertEqual(self.skill.read_bytes(), self.original)
        self.assertEqual(self._manifests(), [])

    def test_a_rollback_whose_record_became_unreadable_under_it_writes_nothing(self) -> None:
        self._apply(self.read_before, "first")
        before = self._tree()
        rc, err = self._cli_with_the_record_changed_at_the_lock(
            self.rollback_argv, lambda: self.record_path.write_text("{", encoding="utf-8")
        )
        self.assertEqual(rc, cli.RC_REFUSED, err)
        self.assertTrue(err.startswith("StateUnreadable: "), err)
        after = self._tree()
        self.assertEqual(after.pop(str(self.record_path.relative_to(self.base)))[1], b"{")
        self.assertEqual(after, self._without(before, self.record_path))
        self.assertEqual(self.skill.read_text(), self.read_before.patch.new_text, "not restored")
        self.assertFalse(self.rollback_receipt.exists())

    # -- a rollback whose record does not recompute once the lock is held ----

    def _edit_the_record(self, change: Callable[[dict[str, Any]], None]) -> None:
        raw = json.loads(self.record_path.read_text(encoding="utf-8"))
        change(raw)
        self.record_path.write_text(json.dumps(raw), encoding="utf-8")

    def _edit_the_manifest(self, **fields: str) -> Path:
        manifest = self.state.latest_backup_manifest(self.cid)
        assert manifest is not None
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        raw.update(fields)
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        return manifest

    def _status_on_disk(self) -> str:
        return str(json.loads(self.record_path.read_text(encoding="utf-8"))["status"])

    def _assert_rollback_advice(self, text: str) -> None:
        """The advice either refusal gives, after what it refused."""
        # It keeps the record and the backup a rollback restores from, sends
        # the operator to no command taken from the state files, and points to
        # the one description of restoring by hand.
        said = (
            KEEP,
            "Keep the record and the backup.",
            NEVER_RUN,
            PUT_BACK.format(cid=self.cid),
            MANUAL,
        )
        at = [text.find(sentence) for sentence in said]
        self.assertNotIn(-1, at, text)
        self.assertEqual(at, sorted(at), text)
        self.assertTrue(text.endswith(MANUAL), text)
        self.assertNotIn(repropose_advice(self.cid), text)
        self.assertNotIn("delete its record", text)
        self.assertNotIn("An approval token is derived", text)
        self.assertNotIn("the copy to make", text)
        self.assertNotIn("check that they agree", text)

    def _assert_changed_since_read(self, text: str, *fields: str) -> None:
        verb = "differs" if len(fields) == 1 else "differ"
        self.assertTrue(
            text.startswith(
                f"refusing to roll back candidate {self.cid}: {CHANGED_SINCE_READ}"
                f"{', '.join(fields)} {verb} from that read, so it is not the record this "
                "rollback was started on.\n  "
            ),
            text,
        )
        self._assert_rollback_advice(text)

    def _assert_refused_as_a_changed_record(
        self, rc: int, err: str, *, changed: Sequence[str] = ()
    ) -> None:
        """Refused as ``CandidateTampered``: for not recomputing, or, given
        ``changed``, for those fields changing after the command read the record."""
        self.assertEqual(rc, cli.RC_REFUSED, err)
        self.assertTrue(err.startswith("CandidateTampered: "), err)
        text = err.removeprefix("CandidateTampered: ").rstrip("\n")
        if changed:
            self._assert_changed_since_read(text, *changed)
            return
        self.assertTrue(text.startswith(f"refusing to roll back candidate {self.cid}: "), text)
        self.assertIn(NOT_RECOMPUTING, text.splitlines()[0])
        self.assertNotIn(CHANGED_SINCE_READ, text)
        self._assert_rollback_advice(text)

    def _rollback_with(self, change: Callable[[], None], when: str) -> tuple[int, str]:
        """The CLI rollback, ``change`` made before it starts or as it reaches for the lock."""
        if when == "before it started":
            change()
            return self._in_process(self.rollback_argv)
        return self._cli_with_the_record_changed_at_the_lock(self.rollback_argv, change)

    def test_a_rollback_whose_record_and_manifest_were_edited_to_name_another_skill_writes_nothing(
        self,
    ) -> None:
        # Edited before the command started, both reads hold the edit, so only
        # the check of the record against itself refuses it; edited as it
        # reached for the lock, that check still comes first.
        for when in ("before it started", "as it reached for the lock"):
            with self.subTest(when=when):
                self.setUp()
                self._apply(self.read_before, "first")
                applied_text = self.skill.read_bytes()
                other = self.base / "skills-canonical" / "code-review" / "SKILL.md"
                other_text = other.read_text(encoding="utf-8")
                planted = "---\nname: code-review\ndescription: planted\n---\nPLANTED RESTORE\n"
                edited: list[Tree] = []

                def name_another_skill_in_both() -> None:
                    # The record and the manifest agree on a write of the other
                    # skill, from a planted text to its live text, with a backup
                    # of the plant.
                    def to_the_other_skill(raw: dict[str, Any]) -> None:
                        raw["target"]["canonical_path"] = str(other)
                        raw["target"]["base_sha256"] = sha256_text(planted)
                        raw["patch"]["new_text"] = other_text

                    self._edit_the_record(to_the_other_skill)
                    manifest = self.state.latest_backup_manifest(self.cid)
                    assert manifest is not None
                    backup = manifest.parent / "planted.bak"
                    backup.write_text(planted, encoding="utf-8")
                    self._edit_the_manifest(
                        target_path=str(other),
                        backup_path=str(backup),
                        previous_sha256=sha256_text(planted),
                        new_sha256=sha256_text(other_text),
                    )
                    edited.append(self._tree())

                rc, err = self._rollback_with(name_another_skill_in_both, when)
                self._assert_refused_as_a_changed_record(rc, err)
                self.assertEqual(
                    other.read_text(encoding="utf-8"), other_text, "the other skill kept"
                )
                self.assertEqual(
                    self.skill.read_bytes(), applied_text, "its own skill not restored"
                )
                self.assertFalse(self.rollback_receipt.exists(), "no rollback receipt")
                self.assertEqual(self._status_on_disk(), "applied")
                self.assertEqual(self._tree(), edited[0], "nothing written after the edit")

    def test_a_rollback_whose_record_and_manifest_were_edited_to_a_later_edit_keeps_that_edit(
        self,
    ) -> None:
        for when in ("before it started", "as it reached for the lock"):
            with self.subTest(when=when):
                self.setUp()
                self._apply(self.read_before, "first")
                later = (
                    self.skill.read_text(encoding="utf-8") + "\nA LATER EDIT THAT MUST SURVIVE\n"
                )
                self.skill.write_text(later, encoding="utf-8")
                edited: list[Tree] = []

                def name_the_later_edit_in_both() -> None:
                    # The record and the manifest agree that the apply wrote the
                    # later edit, so the restore would overwrite it.
                    def to_the_later_edit(raw: dict[str, Any]) -> None:
                        raw["patch"]["new_text"] = later

                    self._edit_the_record(to_the_later_edit)
                    self._edit_the_manifest(new_sha256=sha256_text(later))
                    edited.append(self._tree())

                rc, err = self._rollback_with(name_the_later_edit_in_both, when)
                self._assert_refused_as_a_changed_record(rc, err)
                self.assertEqual(
                    self.skill.read_text(encoding="utf-8"), later, "the later edit kept"
                )
                self.assertFalse(self.rollback_receipt.exists(), "no rollback receipt")
                self.assertEqual(self._status_on_disk(), "applied")
                self.assertEqual(self._tree(), edited[0], "nothing written after the edit")

    def test_a_rollback_whose_record_does_not_recompute_is_refused_before_it_decides_anything(
        self,
    ) -> None:
        # Each case would be refused otherwise by a check made from the record:
        # the never-applied guard, for an apply stopped before its skill write,
        # and the receipt check, for a directory where the receipt goes.
        def stop(*_args: Any, **_kwargs: Any) -> None:
            raise cli.Stopped(signal.SIGTERM)

        cases = [
            (case, when)
            for case in ("never applied", "receipt blocked")
            for when in ("before it started", "as it reached for the lock")
        ]
        for case, when in cases:
            with self.subTest(case=case, when=when):
                self.setUp()
                if case == "never applied":
                    rc, err = self._in_process(
                        self.apply_argv, mock.patch.object(skill_fs, "_write_pinned", stop)
                    )
                    self.assertEqual(rc, cli.RC_STOPPED_BY_SIGTERM, err)
                    self.assertEqual(self.skill.read_bytes(), self.original)
                    self.assertEqual(len(self._manifests()), 1)
                else:
                    self._apply(self.read_before, "first")
                    self.rollback_receipt.mkdir()
                status = self._status_on_disk()
                skill_text = self.skill.read_bytes()
                edited: list[Tree] = []

                def change_the_new_text() -> None:
                    def edit(raw: dict[str, Any]) -> None:
                        raw["patch"]["new_text"] += "\nedited\n"

                    self._edit_the_record(edit)
                    edited.append(self._tree())

                rc, err = self._rollback_with(change_the_new_text, when)
                self._assert_refused_as_a_changed_record(rc, err)
                self.assertNotIn("was never applied", err)
                self.assertNotIn("where the receipt of this rollback would be written", err)
                self.assertEqual(self.skill.read_bytes(), skill_text)
                self.assertEqual(self._status_on_disk(), status)
                self.assertEqual(self._tree(), edited[0], "nothing written after the edit")

    def _name_the_other_skill_at_the_lock(self, other: Path) -> tuple[int, str, Tree]:
        """The CLI rollback, its record and manifest edited as it reaches for the
        lock to name ``other``, every hash, the challenge and the validation left
        as they were; returns the tree as the edit left it."""
        edited: list[Tree] = []

        def name_the_other_skill() -> None:
            def to_the_other_skill(raw: dict[str, Any]) -> None:
                raw["target"]["canonical_path"] = str(other)
                raw["target"]["skill_id"] = "code-review"

            self._edit_the_record(to_the_other_skill)
            self._edit_the_manifest(target_path=str(other), skill_id="code-review")
            # The edit recomputes: the record checked against itself passes.
            verify_full_integrity(self._record(), rubrics=None, action="roll back")
            edited.append(self._tree())

        rc, err = self._cli_with_the_record_changed_at_the_lock(
            self.rollback_argv, name_the_other_skill
        )
        return rc, err, edited[0]

    def test_a_rollback_whose_record_was_edited_to_name_another_skill_holding_the_same_text_writes_nothing(
        self,
    ) -> None:
        # The auditor's case: another canonical skill holds the base text, the
        # apply is made, and an edit of that other skill installs the same
        # text the apply wrote. Restoring it from this apply's backup would
        # erase that edit.
        other = self.base / "skills-canonical" / "code-review" / "SKILL.md"
        other.write_bytes(self.original)
        self._apply(self.read_before, "first")
        applied_text = self.skill.read_bytes()
        other.write_bytes(applied_text)
        rc, err, edited = self._name_the_other_skill_at_the_lock(other)
        self._assert_refused_as_a_changed_record(
            rc, err, changed=("target.canonical_path", "target.skill_id")
        )
        self.assertEqual(other.read_bytes(), applied_text, "the other skill and its edit kept")
        self.assertEqual(self.skill.read_bytes(), applied_text, "its own skill unchanged")
        self.assertFalse(self.rollback_receipt.exists(), "no rollback receipt")
        self.assertEqual(self._status_on_disk(), "applied")
        self.assertEqual(self._tree(), edited, "nothing written after the edit")

    def test_a_rollback_whose_record_changed_after_it_read_it_is_refused_before_the_never_applied_guard(
        self,
    ) -> None:
        # An apply stopped before its skill write leaves a manifest and the
        # skill at its base; the other skill holds the same base text, so the
        # never-applied guard, asked of the edited record, would refuse too.
        def stop(*_args: Any, **_kwargs: Any) -> None:
            raise cli.Stopped(signal.SIGTERM)

        other = self.base / "skills-canonical" / "code-review" / "SKILL.md"
        other.write_bytes(self.original)
        rc, err = self._in_process(
            self.apply_argv, mock.patch.object(skill_fs, "_write_pinned", stop)
        )
        self.assertEqual(rc, cli.RC_STOPPED_BY_SIGTERM, err)
        self.assertEqual(self._status_on_disk(), "validated")
        rc, err, edited = self._name_the_other_skill_at_the_lock(other)
        self._assert_refused_as_a_changed_record(
            rc, err, changed=("target.canonical_path", "target.skill_id")
        )
        self.assertNotIn("was never applied", err)
        self.assertEqual(other.read_bytes(), self.original)
        self.assertEqual(self.skill.read_bytes(), self.original)
        self.assertEqual(self._status_on_disk(), "validated")
        self.assertEqual(self._tree(), edited, "nothing written after the edit")

    def test_a_rollback_after_its_rubric_changed_still_restores_the_skill(self) -> None:
        # The record is checked against itself, not against the rubrics
        # configured now: a rubric edited since the apply does not stop it.
        self._apply(self.read_before, "first")
        applied = self._record()
        rubric = self.base / "rubrics" / "instruction-compliance.toml"
        rubric.write_text(
            rubric.read_text(encoding="utf-8") + "\n# edited after the apply\n", encoding="utf-8"
        )
        live = {r.id: r for r in load_config(self.config_path).rubrics}
        bound = ValidationRecord.parse(applied.validation).rubrics
        self.assertTrue(bound)
        self.assertTrue(
            all(live[b.rubric_id].source_sha256 != b.source_sha256 for b in bound),
            "the rubric the gate was decided by has changed",
        )
        with self.assertRaises(CandidateTampered):
            verify_full_integrity(applied, rubrics=live, action="roll back")
        done = run_cli(self.rollback_argv)
        self.assertEqual(done.returncode, cli.RC_OK, done.stderr)
        self.assertEqual(self.skill.read_bytes(), self.original)
        self.assertEqual(self._record().status, "rolled-back")
        self.assertTrue(self.rollback_receipt.is_file())

    # -- the one description of restoring by hand ------------------------------

    def _the_one_manual_restoration(self, text: str) -> str:
        """The Manual restoration paragraph of ``text``, whitespace-normalised.

        Asserts that one block in the whole document is headed Manual
        restoration (_restoration_headings: a quote, list item, colon or dash
        label, run-in or Markdown heading included, and a heading or emphasised
        label that goes on past the name), so no second set of
        instructions can sit beside it, while a mention inside a sentence is
        not a heading; and that the paragraph's whole text, not
        only its heading, is inside Recovery. It opens no subTest, so a copy of
        the document that breaks either rule raises out of it.
        """
        self.assertEqual(_restoration_headings(text), 1, "one restoration heading anywhere")
        paragraphs = [" ".join(p.split()) for p in text.split("\n\n")]
        manual = [p for p in paragraphs if p.startswith(MANUAL_HEAD)]
        self.assertEqual(len(manual), 1, "one paragraph describes restoring by hand")
        recovery = text.split("\n## Recovery\n", 1)[1].split("\n## ", 1)[0]
        self.assertIn(manual[0], " ".join(recovery.split()), "the instructions are in Recovery")
        return manual[0]

    def test_operations_gives_manual_restoration_once_in_recovery_and_the_refusals_point_to_it(
        self,
    ) -> None:
        text = OPERATIONS.read_text(encoding="utf-8")
        paragraphs = [" ".join(p.split()) for p in text.split("\n\n")]
        said = self._the_one_manual_restoration(text)
        self.assertIn('Recovery, "Manual restoration"', MANUAL)
        # Copies of the document that break the rule are refused by the same
        # checks. Measured in audit 3 (3.11 and 3.13): the checks before this
        # passed the first two, which counted only paragraphs that start with
        # the heading and found only the heading in Recovery.
        raw = next(p for p in text.split("\n\n") if p.startswith(MANUAL_HEAD))
        elsewhere = "\n\n## Alternate recovery\n\n" + raw + "\n"
        quoted = "> **Manual restoration.** Run the manifest rollback_command without checking it."
        # Measured in audit 4 (3.11 and 3.13): a count of the exact source
        # heading passed this one, its period outside the bold.
        emphasis = (
            "\n\n## Emergency recovery\n\n**Manual restoration**. Stop every asi run and owner "
            "edit. Run the manifest `rollback_command` without inspecting its paths or checking "
            "any hashes. Archive the record after the command returns.\n"
        )
        for name, doc, why in (
            (
                "a second, blockquoted Manual restoration elsewhere",
                text + "\n\n" + quoted + "\n",
                "one restoration heading anywhere",
            ),
            (
                "the instructions moved out of Recovery, a reference left there",
                text.replace(raw, "See **Manual restoration.** below for recovery.") + elsewhere,
                "the instructions are in Recovery",
            ),
            (
                "a second Manual restoration elsewhere, its period outside the bold",
                text + emphasis,
                "one restoration heading anywhere",
            ),
            (
                "the instructions moved out of Recovery, nothing left there",
                text.replace("\n\n" + raw, "") + elsewhere,
                "the instructions are in Recovery",
            ),
        ):
            with self.subTest(refused=name):
                self.assertEqual(text.count(raw), 1)
                with self.assertRaisesRegex(AssertionError, why):
                    self._the_one_manual_restoration(doc)
        # Measured in audit 5 (3.11 and 3.13): a count of the name with a period
        # passed the first four of these; the next two are the same forms in
        # title case and as a list item. Measured in audit 6: a count of line
        # labels passed the next four. Measured before audit 7 (3.11 and 3.13):
        # a count of labels that end at the name passed the next four. Measured
        # in audit 7: marks without a task-list box passed the last three.
        harmful = (
            " Stop every asi run and owner edit. Run the manifest `rollback_command` without "
            "inspecting its paths or checking any hashes. Archive the record after the "
            "command returns.\n"
        )
        for label in (
            "**Manual restoration:**",
            "Manual restoration:",
            "### Manual restoration\n\n",
            "Manual restoration\n------------------\n\n",
            "### Manual Restoration\n\n",
            "- **Manual restoration:**",
            "### Manual restoration ###\n\n",
            "- ### Manual restoration\n\n",
            "**Manual restoration**",
            "**Manual restoration** \u2014",
            "### Manual restoration after a failed rollback\n\n",
            "Manual restoration after a failed rollback\n---\n\n",
            "**Manual restoration after a failed rollback.**",
            "- **Manual restoration for a changed record:**",
            "- [ ] **Manual restoration.**",
            "- [x] **Manual restoration:**",
            "- [ ] ### Manual restoration\n\n",
        ):
            body = harmful.lstrip() if label.endswith("\n") else harmful
            doc = text + "\n\n## Emergency recovery\n\n" + label + body
            with self.subTest(refused=label.strip()):
                with self.assertRaisesRegex(AssertionError, "one restoration heading anywhere"):
                    self._the_one_manual_restoration(doc)
        # A mention is not a heading. Measured in audit 6: a count of the name
        # with a period anywhere refused the first three.
        for mention in (
            "For a changed record, see Manual restoration.",
            "For a changed record, see Manual Restoration.",
            "For a changed record, see **Manual restoration**.",
            "For a changed record, see Manual restoration under Recovery.",
            "Manual restoration (below) covers a record that changed.",
            "For a changed record, see\nManual restoration\nunder Recovery.",
            "| a changed record | Manual restoration, under Recovery |",
            "Manual restoration is described under Recovery.",
            "**Manual restoration** (below) covers a record that changed.",
            "| what to do | case |\n| --- | --- |\n| Manual restoration | a changed record |",
            "- [ ] Read Manual restoration under Recovery before proceeding.",
            "- [ ] **Manual restoration** is described under Recovery.",
        ):
            with self.subTest(accepted=mention):
                self.assertEqual(
                    self._the_one_manual_restoration(text + "\n\n" + mention + "\n"), said
                )
        for words in (
            # Nothing taken from the state files is run or trusted.
            "Never execute `rollback_command` or any command from the record, manifest or "
            "receipts; their agreement is not proof.",
            "Stop every asi run and owner edit.",
            # The skill and its backup are known from outside the state files ...
            "Independently establish the original canonical skill file and its retained "
            "pre-apply backup using known configuration and owner history, or a trusted copy "
            "kept outside the state directory",
            # ... and neither the inventory as it is now nor the candidate id
            # stands in for that: the id's skill suffix is safe_name's, which is
            # lossy.
            "reports the current inventory",
            "not original identity, so it only corroborates the present location",
            "the candidate suffix is lossy",
            "so it proves no skill",
            "an identity that is missing, ambiguous or under a changed root is unknown.",
            # The backup's directory is known from outside the state files too:
            # an edited manifest can name another one.
            "Independently establish the backup's original `backups/<id>--<stamp>/` directory "
            "from that history or a trusted outside-state copy; require the backup there and "
            "both files regular, no symbolic-link path component",
            # A same-text hard link passes every path and hash check, and an
            # ordinary copy writes through it to its other names.
            "and the target link count one",
            "Decode both strictly as UTF-8",
            "if any fact is unknown, stop without copying or archiving.",
            "When `asi rollback` runs, the store already checks",
            # The store's no-write branch (adapters/skill_fs.py, already_restored).
            "holding either the previous text (nothing is written) or the text the apply "
            "wrote; and, when restoring, that the skill",
            "It does not check links above the skill root or above the backup's directory, "
            "nor that the record and the manifest name the skill and the backup you meant.",
        ):
            with self.subTest(says=words):
                self.assertIn(words, said)
        # The backup's hash first; a target already restored is not copied over
        # before the applied hash is asked of it; then restored, verified and
        # archived, and stopped if anything is unknown.
        steps = (
            "Require backup = the independently established previous hash",
            "If target already equals that previous hash, do not copy.",
            "require target = the independently established applied hash; inspect their diff "
            "to confirm the intended undo.",
            "Then copy only between the inspected absolute paths",
            "then verify the target equals the previous hash",
            "Only then may the record be archived, and keep the backup",
            "if any fact is unknown, stop without copying or archiving.",
        )
        order = [said.find(step) for step in steps]
        self.assertNotIn(-1, order, said)
        self.assertEqual(order, sorted(order), said)
        # The refusal table and the Rollback section point to it and give no
        # way of their own.
        tampered = next(p for p in paragraphs if "| `CandidateTampered` |" in p)
        rollback = text.split("\n## Rollback\n", 1)[1].split("\n## ", 1)[0]
        for name, where in (("CandidateTampered", tampered), ("Rollback", rollback)):
            with self.subTest(where=name):
                flat = " ".join(where.split())
                self.assertIn("Manual restoration under Recovery", flat)
                self.assertIn("never run the manifest's `rollback_command`", flat)
                self.assertNotIn("from the verified backup", flat)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
