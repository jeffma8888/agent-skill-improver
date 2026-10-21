"""A malformed value typed on the command line is masked and bounded in its refusal.

Measured before (3.11 and 3.13): a 100,000-character ``--day``,
``--mutation-reviewed``, candidate id, ``--limit``, subcommand,
``set-aside --kind``, scheduler flavor, unknown option, ambiguous option or
explicit argument to a flag each came back whole on stderr, about 100 KB; a
credential pasted where ``asi review`` takes a candidate id, typed as a
``--day`` or after ``--commit=``, was printed as typed; and a ``--limit`` of
5,000 nines was refused as not a whole number. Each value now goes through
``quoted_value``, and argparse's own messages through
``_bounded_argparse_message``.
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver import cli
from agent_skill_improver.cli import (
    ARGPARSE_MESSAGE_CHARS,
    ARGPARSE_MESSAGE_TAIL,
    QUOTED_EXTRA_ARGUMENTS,
    RC_REFUSED,
)
from agent_skill_improver.domain.redaction import QUOTED_VALUE_CHARS

from tests.support import build_env, run_cli

LONG = 100_000
#: Usage plus the reason plus one bounded quote; the unbounded echo was ~100 KB.
BOUND = 2_000


def _refused_by_argparse(argv: list[str]) -> str:
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        with unittest.TestCase().assertRaises(SystemExit) as ctx:
            cli.main(argv)
    assert ctx.exception.code == 2, ctx.exception.code
    return err.getvalue()


class AMalformedTypedValueIsMaskedAndBoundedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.cfg = ["--config", str(build_env(Path(cls._tmp.name).resolve()))]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def assertBounded(self, text: str) -> None:
        self.assertLess(len(text), BOUND, text[:300])
        self.assertIn("more chars not quoted]", text)

    def test_a_long_run_day_is_quoted_bounded_and_its_reason_still_follows(self):
        err = _refused_by_argparse([*self.cfg, "daily", "--day", "d" * LONG])
        self.assertBounded(err)
        self.assertIn("argument --day: run day 'ddd", err)
        self.assertIn("cannot name a candidate", err)
        self.assertIn("(maximum 64 characters)", err)
        err = _refused_by_argparse([*self.cfg, "daily", "--mutation-reviewed", "m" * LONG])
        self.assertBounded(err)
        self.assertIn("argument --mutation-reviewed: run day 'mmm", err)

    def test_a_credential_typed_as_a_run_day_is_masked_in_the_hint_too(self):
        token = "g" + "hp_" + "D" * 36
        for typed, hint in ((token + "!", False), (token + "/x", True)):
            with self.subTest(hint=hint):
                err = _refused_by_argparse([*self.cfg, "daily", "--day", typed])
                self.assertNotIn(token, err)
                self.assertIn("run day '[REDACTED:github-token]", err)
                if hint:
                    self.assertIn(
                        "a date is spelled with hyphens, '[REDACTED:github-token]-x'", err
                    )
        err = _refused_by_argparse([*self.cfg, "daily", "--day", "2026/01/02"])
        self.assertIn("run day '2026/01/02' cannot name a candidate", err)
        self.assertIn("a date is spelled with hyphens, '2026-01-02'", err)

    def test_a_long_candidate_id_is_quoted_bounded_by_review_apply_and_rollback(self):
        for command in (
            ["review"],
            ["apply", "--approve", "t", "--i-reviewed-this"],
            ["rollback"],
        ):
            with self.subTest(command=command[0]):
                proc = run_cli([*self.cfg, command[0], "x" * LONG, *command[1:]])
                self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr[-300:])
                self.assertBounded(proc.stderr)
                self.assertIn("StateUnreadable: invalid candidate id 'xxx", proc.stderr)
                self.assertIn("(maximum 204 characters)", proc.stderr)

    def test_a_credential_typed_as_a_candidate_id_is_masked_in_the_refusal(self):
        token = "g" + "hp_" + "C" * 36
        proc = run_cli([*self.cfg, "review", token])
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
        self.assertNotIn(token, proc.stderr)
        self.assertIn("invalid candidate id '[REDACTED:github-token]'", proc.stderr)

    def test_a_page_size_with_more_digits_than_python_reads_is_told_so(self):
        readable = sys.get_int_max_str_digits()
        if not readable:
            self.skipTest("this interpreter reads integers of any length")
        for typed in ("9" * (readable + 700), "-" + "9" * (readable + 1)):
            with self.subTest(digits=len(typed)):
                err = _refused_by_argparse([*self.cfg, "discover", "--limit", typed])
                self.assertBounded(err)
                self.assertIn(
                    f"must be a whole number of at most {readable} digits, the most Python "
                    f"reads from text; got {len(typed.lstrip('-'))} digits: {typed[:4]!r}"[:-1],
                    err,
                )
                self.assertNotIn("must be a whole number, got", err)
        # Underscores between digits are spelled as int() takes them and are
        # not counted, as the limit does not count them.
        typed = "1_" * readable + "1"
        err = _refused_by_argparse([*self.cfg, "discover", "--limit", typed])
        self.assertIn(f"got {readable + 1} digits: '1_1_", err)
        typed = "+" + "9" * (readable + 1)
        err = _refused_by_argparse([*self.cfg, "discover", "--limit", typed])
        self.assertIn(f"got {readable + 1} digits: '+999", err)
        # And the white space int() strips around a number is stripped here too.
        typed = " " + "9" * (readable + 1) + " "
        err = _refused_by_argparse([*self.cfg, "discover", "--limit", typed])
        self.assertIn(f"got {readable + 1} digits: ' 999", err)

    def test_a_value_that_is_not_a_number_at_all_is_not_told_the_digit_limit(self):
        readable = sys.get_int_max_str_digits() or 4300
        for typed in (
            "+-" + "9" * (readable + 1),
            "9" * readable + "__" + "9",
            "_" + "9" * (readable + 1),
            "9" * (readable + 1) + "x",
        ):
            with self.subTest(typed=typed[:6]):
                err = _refused_by_argparse([*self.cfg, "discover", "--limit", typed])
                self.assertBounded(err)
                self.assertIn("argument --limit: must be a whole number, got '", err)
                self.assertNotIn("digits, the most Python", err)

    def test_a_long_page_size_that_is_not_a_page_size_is_quoted_bounded(self):
        err = _refused_by_argparse([*self.cfg, "discover", "--limit", "x" * LONG])
        self.assertBounded(err)
        self.assertIn("argument --limit: must be a whole number, got 'xxx", err)
        # Below the threshold no digit limit can be set under, so int() reads it.
        nines = "9" * (sys.int_info.str_digits_check_threshold - 1)
        err = _refused_by_argparse([*self.cfg, "discover", "--limit", "-" + nines])
        self.assertBounded(err)
        self.assertIn("argument --limit: must be at least 1, got '-999", err)

    def test_an_unknown_choice_is_quoted_bounded_and_the_choices_still_listed(self):
        err = _refused_by_argparse([*self.cfg, "y" * LONG])
        self.assertBounded(err)
        self.assertIn("argument command: invalid choice: 'yyy", err)
        self.assertIn("(choose from 'doctor', 'skills',", err)
        self.assertIn("'selftest')", err)
        err = _refused_by_argparse([*self.cfg, "scheduler-template", "q" * LONG])
        self.assertBounded(err)
        self.assertIn("argument flavor: invalid choice: 'qqq", err)
        self.assertIn("(choose from 'launchd', 'cron')", err)
        err = _refused_by_argparse([*self.cfg, "set-aside", "--kind", "k" * LONG])
        self.assertBounded(err)
        self.assertIn("argument --kind: invalid choice: 'kkk", err)
        self.assertIn("(choose from 'quarantined',", err)

    def test_unrecognized_arguments_are_quoted_bounded_and_the_rest_counted(self):
        err = _refused_by_argparse([*self.cfg, "doctor", "--zz" + "z" * LONG])
        self.assertBounded(err)
        self.assertIn("unrecognized arguments: '--zzz", err)
        many = [f"w{i}" for i in range(QUOTED_EXTRA_ARGUMENTS + 42)]
        err = _refused_by_argparse([*self.cfg, "doctor", *many])
        quoted = " ".join(repr(w) for w in many[:QUOTED_EXTRA_ARGUMENTS])
        self.assertIn(f"unrecognized arguments: {quoted} ... and 42 more not quoted", err)
        self.assertNotIn(repr(many[QUOTED_EXTRA_ARGUMENTS]), err)

    def test_an_ambiguous_option_or_an_explicit_argument_to_a_flag_is_masked_and_bounded(self):
        c, v = "c" * LONG, "v" * LONG
        for typed, full in (
            ("--co=" + c, "ambiguous option: --co=" + c + " could match --config, --commit"),
            ("--commit=" + v, "argument --commit: ignored explicit argument '" + v + "'"),
        ):
            with self.subTest(typed=typed[:6]):
                err = _refused_by_argparse([*self.cfg, "daily", typed])
                self.assertLess(len(err), BOUND, err[:300])
                head = ARGPARSE_MESSAGE_CHARS - ARGPARSE_MESSAGE_TAIL
                left_out = len(full) - ARGPARSE_MESSAGE_CHARS
                self.assertIn(
                    f"error: {full[:head]} ... [{left_out} more chars not shown] ... "
                    f"{full[-ARGPARSE_MESSAGE_TAIL:]}\n",
                    err,
                )
        token = "g" + "hp_" + "E" * 36
        for typed in ("--co=" + token, "--commit=" + token):
            with self.subTest(typed=typed[:6]):
                err = _refused_by_argparse([*self.cfg, "daily", typed])
                self.assertNotIn(token, err)
                self.assertIn("[REDACTED:github-token]", err)

    def test_a_credential_in_a_message_long_enough_to_cut_is_masked_whole(self):
        token = "g" + "hp_" + "F" * 36
        fragment = "hp_" + "F" * 8
        for case, argv in enumerate(
            (
                [*self.cfg, "daily", "--commit=" + token + "-" + "v" * LONG],
                [*self.cfg, "daily", "--commit=" + "v" * LONG + "-" + token],
                ["--=" + "w" * LONG + "-" + token, *self.cfg, "doctor"],
            )
        ):
            with self.subTest(case=case):
                err = _refused_by_argparse(argv)
                self.assertLess(len(err), BOUND, err[:300])
                self.assertIn("more chars not shown] ... ", err)
                self.assertNotIn(fragment, err)
                self.assertIn("[REDACTED:github-token]", err)

    def test_a_message_as_long_as_the_limit_is_printed_whole_and_one_more_is_cut(self):
        whole = "x" * ARGPARSE_MESSAGE_CHARS
        self.assertEqual(cli._bounded_argparse_message(whole), whole)
        cut = cli._bounded_argparse_message(whole + "y")
        self.assertIn(" ... [1 more chars not shown] ... ", cut)
        self.assertTrue(cut.endswith("x" * (ARGPARSE_MESSAGE_TAIL - 1) + "y"), cut[-40:])

    def test_the_longest_message_this_module_builds_is_not_cut(self):
        # Each word is about as long as macOS lets one argv carry (ARG_MAX is
        # 1,048,576 bytes, arguments and environment together), and each
        # quote of it is cut.
        words = ["\x01" * 1_000_000] * (QUOTED_EXTRA_ARGUMENTS + 1)
        message = cli._unrecognized_arguments(words)
        self.assertLess(len(message), ARGPARSE_MESSAGE_CHARS - ARGPARSE_MESSAGE_TAIL)
        self.assertEqual(cli._bounded_argparse_message(message), message)
        self.assertTrue(message.endswith(" ... and 1 more not quoted"), message[-80:])

    def test_a_short_value_is_quoted_whole(self):
        err = _refused_by_argparse([*self.cfg, "daily", "--day", "bad day"])
        self.assertIn("run day 'bad day' cannot name a candidate", err)
        err = _refused_by_argparse([*self.cfg, "discover", "--limit", "abc"])
        self.assertIn("must be a whole number, got 'abc'", err)
        err = _refused_by_argparse([*self.cfg, "discover", "--limit", "0"])
        self.assertIn("must be at least 1, got '0'", err)
        err = _refused_by_argparse([*self.cfg, "frob"])
        self.assertIn("invalid choice: 'frob' (choose from 'doctor',", err)
        err = _refused_by_argparse([*self.cfg, "doctor", "--nope", "x"])
        self.assertIn("unrecognized arguments: '--nope' 'x'\n", err)
        self.assertNotIn("more not quoted", err)
        err = _refused_by_argparse([*self.cfg, "daily", "--co"])
        self.assertIn("error: ambiguous option: --co could match --config, --commit\n", err)
        err = _refused_by_argparse([*self.cfg, "daily", "--commit=x"])
        self.assertIn("error: argument --commit: ignored explicit argument 'x'\n", err)
        typed = "CAND-" + "b" * (QUOTED_VALUE_CHARS - 10) + " id"  # its repr is the limit
        proc = run_cli([*self.cfg, "review", typed])
        self.assertIn(f"invalid candidate id {typed!r}:", proc.stderr)
        self.assertNotIn("more chars not quoted", proc.stderr)


if __name__ == "__main__":
    unittest.main()
