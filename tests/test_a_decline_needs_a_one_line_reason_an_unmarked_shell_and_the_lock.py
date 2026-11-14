"""A decline needs a one-line reason, an unmarked shell and the lock.

The reason (and an ``--operator`` label) is checked as typed, before the lock:
not blank, at most 500 code points (200 for the label), one line, no control,
format or unreviewable character, no surrogate; a refusal names the rule and the
code point, never the text. ``--reason -`` reads one line from standard input
and puts it through the same checks. Under the automation marker decline is
refused as an interlock; while another command holds the run lock it waits for
nothing and writes nothing. Only a validated candidate is declined, and never one
an apply started: that record is kept, with the checked way back (run that apply
again, then ``asi rollback``) or a pointer to "Manual restoration".
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import unittest
import unittest.mock
from pathlib import Path
from typing import Any

from agent_skill_improver import cli
from agent_skill_improver.app.lock import RunLock
from agent_skill_improver.domain.approval import AUTOMATION_ENV_VAR


def _absent_before_c247b(name: str) -> Any:
    """What a name c247b adds stands for on a tree without it: a failure, by assertion."""

    def absent(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"{name} does not exist: this tree has no decline")

    return absent


try:
    from agent_skill_improver.domain.candidate import decline_reason_problem
except ImportError:  # a tree without c247b: its tests fail by assertion
    decline_reason_problem = _absent_before_c247b("decline_reason_problem")

from tests.test_a_declined_candidate_cannot_be_applied_or_rolled_back import (
    DeclineTreeCase,
    _Stdin,
)
from tests.test_apply_and_rollback_act_on_the_record_as_it_is_under_the_lock import (
    _restoration_headings,
)

MARK = "zebraword"
AWS = "AK" + "IA" + "A" * 16
SRC = Path(__file__).resolve().parents[1] / "src"
OPERATIONS = Path(__file__).resolve().parents[1] / "docs" / "OPERATIONS.md"
BAD = {
    "empty": "",
    "spaces": "   ",
    "501 code points": MARK + "y" * 492,
    "inner line feed": MARK + "\nmore",
    "inner carriage return": MARK + "\rmore",
    "inner tab": MARK + "\tmore",
    "terminal escape": MARK + "\x1b[31mred",
    "right-to-left override": MARK + "\u202emore",
    "line separator": MARK + "\u2028more",
    "tag character": MARK + "\U000e0041more",
    "byte order mark first": "\ufeff" + MARK,
    "byte order mark inside": MARK + "\ufeffmore",
}


class _CountingBuffer(io.BytesIO):
    reads = 0

    def readline(self, *args: Any) -> bytes:
        self.reads += 1
        return super().readline(*args)

    def read(self, *args: Any) -> bytes:
        self.reads += 1
        return super().read(*args)


class ADeclineNeedsAReasonTests(DeclineTreeCase):
    def assert_refused_writing_nothing(self, outcome: tuple[int, str, str], before: Any) -> str:
        rc, out, err = outcome
        self.assertEqual(rc, cli.RC_REFUSED, out + err)
        self.assertEqual(self.tree(), before)
        return err

    def test_a_decline_without_a_reason_is_a_usage_error(self) -> None:
        before = self.tree()
        try:
            rc = self.cli("decline", self.cid)[0]
        except SystemExit as stop:
            rc = stop.code
        self.assertEqual(rc, 2)
        self.assertEqual(self.tree(), before)

    def test_a_blank_long_multiline_or_control_reason_is_refused_unquoted_and_nothing_written(
        self,
    ) -> None:
        before = self.tree()
        for name, text in BAD.items():
            with self.subTest(reason=name):
                err = self.assert_refused_writing_nothing(self.decline(text), before)
                self.assertNotIn(MARK, err)
                if text.strip():
                    self.assertNotIn(text, err)
            with self.subTest(operator=name):
                if not text.strip():
                    continue  # a blank label is no label: the output says "unspecified"
                bad = text if name != "501 code points" else MARK + "y" * 192
                err = self.assert_refused_writing_nothing(
                    self.decline("fine", "--operator", bad), before
                )
                self.assertNotIn(MARK, err)
        for what, limit in (("reason", 500), ("operator label", 200)):
            with self.subTest(surrogate=what):
                problem = decline_reason_problem("left\udcffright", limit=limit, what=what)
                self.assertIsNotNone(problem, "a lone surrogate was accepted")
                self.assertIn("U+DCFF at character 5", problem)
                self.assertNotIn("left", problem)
                self.assertNotIn("right", problem)

    def test_a_reason_of_exactly_500_code_points_is_accepted(self) -> None:
        rc, out, err = self.decline("é" * 500)
        self.assertEqual(rc, 0, err)
        stored = self.raw_record()["decline"]["reason"]
        self.assertEqual((len(stored), len(stored.encode("utf-8"))), (500, 1000))

    def test_a_reason_on_standard_input_goes_through_the_same_checks_and_masking(self) -> None:
        line = "not now " + AWS
        self.assertEqual(self.decline(line)[0], 0)
        by_argv = self.raw_record()["decline"]["reason"]
        self.setUp()
        rc, out, err = self.decline("-", stdin=(line + "\n").encode())
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.raw_record()["decline"]["reason"], by_argv)
        self.assertNotIn(AWS, by_argv)
        self.setUp()
        before = self.tree()
        refused = {
            "two lines": b"one\ntwo\n",
            "not UTF-8": b"\xff\xfeok\n",
            "501 code points": b"y" * 501 + b"\n",
            "ends at once": b"",
            "a blank line": b"\n",
            "an encoded surrogate": b"\xed\xb3\xbf\n",
            "fills the bound with no line end": b"y" * cli.REASON_LINE_BOUND,
        }
        for name, data in refused.items():
            with self.subTest(name):
                self.assert_refused_writing_nothing(self.decline("-", stdin=data), before)
        with self.subTest("a CRLF line end"):
            rc, out, err = self.decline("-", stdin=b"crlf ended\r\n")
            self.assertEqual(rc, 0, err)
            self.assertEqual(self.raw_record()["decline"]["reason"], "crlf ended")
        with self.subTest("500 four-byte code points"):
            self.setUp()
            data = ("\U00020000" * 500 + "\n").encode()
            self.assertEqual(len(data), 2001)
            rc, out, err = self.decline("-", stdin=data)
            self.assertEqual(rc, 0, err)
            self.assertEqual(self.raw_record()["decline"]["reason"], "\U00020000" * 500)

    def test_decline_under_the_automation_marker_is_refused_and_says_it_is_an_interlock(
        self,
    ) -> None:
        before = self.tree()
        err = self.assert_refused_writing_nothing(
            self.decline("too broad", env={AUTOMATION_ENV_VAR: "1"}), before
        )
        self.assertIn("automation marker", err)
        self.assertIn("an interlock, not caller authentication", err)

    def test_decline_while_a_daily_run_holds_the_lock_is_refused_and_writes_nothing(self) -> None:
        before = self.tree()
        with RunLock(self.state.lock_path(), owner="daily 2026-02-02 pid=1"):
            err = self.assert_refused_writing_nothing(self.decline(), before)
        self.assertIn("LockHeld", err)
        self.assertIn("daily 2026-02-02 pid=1", err)
        self.assertIn("nothing was written", err)

    def test_only_a_validated_candidate_can_be_declined(self) -> None:
        said = {
            "proposed": ("it was never validated", "delete its record"),
            "applied": ("it is applied", f"`asi rollback {self.cid}`"),
            "gate-failed": ("already closed (gate-failed)",),
            "blocked-run": ("already closed (blocked-run)",),
            "rolled-back": ("already closed (rolled-back)",),
            "held": ("its status is 'held'", "only a validated candidate"),
        }
        for status, phrases in said.items():
            with self.subTest(status):
                self.setUp()
                self.edit_record(lambda raw: raw.update(status=status))
                before = self.tree()
                self.assertEqual(self.manifests(), [])
                err = self.assert_refused_writing_nothing(self.decline(), before)
                for phrase in phrases:
                    self.assertIn(phrase, err)

    def test_declining_an_unknown_candidate_is_exit_4(self) -> None:
        before = self.tree()
        rc, out, err = self.cli("decline", "CAND-2026-02-01-09-none", "--reason", "no")
        self.assertEqual(rc, cli.RC_MISSING_FILE, err)
        self.assertEqual(self.tree(), before)

    def test_a_tampered_record_is_refused_and_left_as_it_is(self) -> None:
        self.tamper_patch()
        before = self.tree()
        err = self.assert_refused_writing_nothing(self.decline(), before)
        self.assertIn("CandidateTampered", err)

    def test_a_decline_after_the_live_rubric_changed_or_was_removed_is_recorded(self) -> None:
        for variant in ("comment added", "removed from the configuration"):
            with self.subTest(variant):
                self.setUp()
                if variant == "comment added":
                    rubric = self.base / "rubrics" / "instruction-compliance.toml"
                    rubric.write_text(rubric.read_text() + "\n# a comment added after the run\n")
                else:
                    src = Path(__file__).resolve().parents[1] / "rubrics" / "tool-efficiency.toml"
                    (self.base / "rubrics" / "tool-efficiency.toml").write_bytes(src.read_bytes())
                    text = self.config_path.read_text(encoding="utf-8")
                    old = 'enabled = ["instruction-compliance"]'
                    self.assertIn(old, text)
                    self.config_path.write_text(
                        text.replace(old, 'enabled = ["tool-efficiency"]'), encoding="utf-8"
                    )
                self.assertEqual(self.cli("review", self.cid)[0], cli.RC_REFUSED)
                rc, out, err = self.decline()
                self.assertEqual(rc, 0, err)
                self.assertEqual(self.raw_record()["status"], "declined")
                self.assertTrue(self.receipt_path.is_file())

    def test_a_decline_after_an_apply_stopped_before_or_after_its_skill_write_is_refused(
        self,
    ) -> None:
        for where in ("before", "after"):
            with self.subTest(stopped=where):
                self.setUp()
                getattr(self, f"stop_apply_{where}_its_skill_write")()
                (manifest,) = self.manifests()
                before = self.tree()
                err = self.assert_refused_writing_nothing(self.decline(), before)
                for phrase in (str(manifest), "Keep this record", "Run that apply again"):
                    self.assertIn(phrase, err)
                self.assertIn(f"`asi rollback {self.cid}`", err)
                self.assertNotIn("delete its record", err)
                self.assertEqual(self._main(self.apply_argv)[0], 0)
                self.assertEqual(self._main(self.rollback_argv)[0], 0)
                self.assertEqual(self.skill.read_bytes(), self.original)
                self.assertEqual(self.raw_record()["status"], "rolled-back")
        with self.subTest("stopped after, then tampered with"):
            self.setUp()
            self.stop_apply_after_its_skill_write()
            self.tamper_patch()
            before = self.tree()
            err = self.assert_refused_writing_nothing(self.decline(), before)
            self.assertIn("Keep this record", err)
            self.assertNotIn("CandidateTampered", err)
            self.assertNotIn("delete its record", err)
        for status in ("proposed", "held"):
            with self.subTest("stopped after, status hand-edited", status=status):
                self.setUp()
                self.stop_apply_after_its_skill_write()
                written = self.skill.read_bytes()
                self.edit_record(lambda raw: raw.update(status=status))
                (manifest,) = self.manifests()
                before = self.tree()
                err = self.assert_refused_writing_nothing(self.decline(), before)
                for phrase in (str(manifest), f"'{status}'", "which apply refuses"):
                    self.assertIn(phrase, err)
                self.assertIn("Keep this record", err)
                self.assertNotIn("delete its record", err)
                self.assertNotIn("Run that apply again", err)
                self.assertEqual(self.skill.read_bytes(), written)
        with self.subTest("stopped after, then declined by hand and tampered with"):
            self.setUp()
            self.stop_apply_after_its_skill_write()
            written = self.skill.read_bytes()
            self.hand_decline()
            self.tamper_patch()
            before = self.tree()
            err = self.assert_refused_writing_nothing(self.decline(), before)
            self.assertIn("CandidateTampered", err)
            self.assertIn("patch", err)
            self.assertIn("Keep this record and the backup", err)
            self.assertIn('"Manual restoration"', err)
            for absent in ("delete its record", "An approval token", "rollback_command"):
                self.assertNotIn(absent, err)
            self.assertNotIn("asi rollback", err)
            self.assertEqual(self.skill.read_bytes(), written)

    def test_a_reason_typed_at_a_terminal_is_prompted_on_stderr_and_nothing_is_read_after_it(
        self,
    ) -> None:
        self.assertTrue(hasattr(cli, "_read_one_reason_line"), "no reason reader in cli.py")
        stream = _CountingBuffer(b"typed at a terminal\nmore that is never read\n")
        err = io.StringIO()
        with unittest.mock.patch.object(sys, "stderr", err):
            self.assertEqual(cli._read_one_reason_line(stream, True), "typed at a terminal")
        self.assertEqual(err.getvalue(), cli.REASON_PROMPT)
        self.assertEqual(stream.reads, 1)
        feed = _Stdin(b"", tty=True)
        feed.buffer = _CountingBuffer(b"typed at a terminal\nmore that is never read\n")
        rc, out, said = self.decline("-", stdin=feed)
        self.assertEqual(rc, 0, said)
        self.assertTrue(said.startswith("reason (one line): "), said)
        self.assertEqual(feed.buffer.reads, 1)
        self.assertEqual(self.raw_record()["decline"]["reason"], "typed at a terminal")
        self.setUp()
        before = self.tree()
        self.assert_refused_writing_nothing(
            self.decline("-", stdin=_Stdin(b"\n", tty=True)), before
        )

    def test_a_reason_or_operator_holding_a_lone_surrogate_from_argv_is_refused_by_code_point(
        self,
    ) -> None:
        before = self.tree()
        env = {k: v for k, v in os.environ.items() if k != AUTOMATION_ENV_VAR}
        env.update(PYTHONPATH=f"{SRC}{os.pathsep}{SRC.parent}", PYTHONDONTWRITEBYTECODE="1")
        for flag in ("--reason", "--operator"):
            with self.subTest(flag):
                argv: list[Any] = [sys.executable, "-m", "agent_skill_improver", *self.cfg]
                argv += ["decline", self.cid]
                if flag == "--operator":
                    argv += ["--reason", "fine"]
                argv += [flag.encode(), b"left\xffright"]
                done = subprocess.run(argv, capture_output=True, env=env, timeout=120)
                err = done.stderr.decode("utf-8", "replace")
                self.assertEqual(done.returncode, cli.RC_REFUSED, err)
                self.assertIn("U+DCFF", err)
                self.assertNotIn("left", err)
                self.assertNotIn("right", err)
                self.assertEqual(self.tree(), before)

    def test_the_decline_recovery_points_to_the_one_manual_restoration_and_keeps_the_record(
        self,
    ) -> None:
        # The edited-command counterexample, through the checked path.
        self.stop_apply_after_its_skill_write()
        unrelated = self.base / "unrelated.txt"
        unrelated.write_bytes(b"not a skill\n")
        (manifest,) = self.manifests()
        body = json.loads(manifest.read_text(encoding="utf-8"))
        backup = next(p for p in manifest.parent.iterdir() if p.name != "backup.json")
        body["rollback_command"] = f"cp {backup} {unrelated}"
        manifest.write_text(json.dumps(body), encoding="utf-8")
        self.assertEqual(self.decline()[0], cli.RC_REFUSED)
        rc, out, err = self._main(self.apply_argv)
        self.assertEqual(rc, 0, err)
        self.assertNotIn(str(unrelated), out + err)
        self.assertEqual(self._main(self.rollback_argv)[0], 0)
        self.assertEqual(self.skill.read_bytes(), self.original)
        self.assertEqual(unrelated.read_bytes(), b"not a skill\n")
        # The doc text (docs/OPERATIONS.md, Recovery).
        text = OPERATIONS.read_text(encoding="utf-8")
        self.assertEqual(_restoration_headings(text), 1)
        self.assertEqual(text.count("**Manual restoration.**"), 1)
        for name in (
            "An apply that started cannot be declined",
            "A declined candidate with a backup manifest",
        ):
            self.assertIn(name, text)
        started = _paragraph(text, "An apply that started cannot be declined")
        beside = _paragraph(text, "A declined candidate with a backup manifest")
        for phrase in ("Keep the record", "run that apply again", "asi rollback <id>"):
            self.assertIn(phrase.lower(), started.lower())
        for paragraph in (started, beside):
            self.assertIn('"Manual restoration"', paragraph)
            # The pointer itself, not only a mention: the restore is the one paragraph's.
            self.assertRegex(
                " ".join(paragraph.split()),
                r'\brestore (?:it|the skill) as (?:the paragraph )?"Manual restoration" says\b',
            )
            for step in ("asi skills --json", "shasum", "cp ", "rollback_command"):
                self.assertNotIn(step, paragraph)
        self.assertIn("declined record", " ".join(beside.split()))
        self.assertIn("not archived", " ".join(beside.split()))
        manual = text[text.index("**Manual restoration.**") :]
        manual = manual[: manual.index("\n\n")]
        self.assertNotIn("declined", manual)


def _paragraph(text: str, name: str) -> str:
    at = text.index(f"**{name}.**")
    end = text.find("\n\n", at)
    return text[at : end if end >= 0 else len(text)]


if __name__ == "__main__":
    unittest.main()
