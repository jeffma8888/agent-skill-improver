"""A cron job names a path holding a backslash as itself, whichever cron reads it.

Most crons rewrite a crontab command before the shell sees it (``do_command.c``):
a bare ``%`` ends the command and a backslash before a ``%`` is consumed. They
part ways on a backslash before a backslash: FreeBSD and macOS make the pair
one, cronie and OpenBSD keep both; busybox crond rewrites nothing at all.
``_cron_quote`` in ``cli.py`` doubled every backslash, which only FreeBSD and
macOS undo. Measured before (3.11 and 3.13): of seven config directories holding a
backslash (``back\\slash``, two in a row, three in a row, one before a ``%``,
one ending the name, one before an apostrophe, one between spaces), each was
named as a directory that does not exist under the cronie and OpenBSD rewrite,
and the six without a ``%`` under busybox. Now no backslash pair is printed: each
backslash is followed by a quote ending the quoted text and one resuming it.
"""

from __future__ import annotations

import plistlib
import shlex
import subprocess
import unittest
from collections.abc import Callable

from agent_skill_improver.cli import _cron_quote
from tests.test_a_scheduler_template_fills_each_marker_once import render, shell_words

BACKSLASHED = ("back\\slash", "two\\\\back", "three\\\\\\x", "end\\", "q\\'x", "spaced \\ x")
BEFORE_A_PERCENT = "pct\\%x"


def rewrite(command: str, *, pair_made_one: bool) -> str:
    """``command`` as a rewriting cron hands it to the shell, from its loop."""
    out: list[str] = []
    escaped = False
    for ch in command:
        if escaped:
            kept_alone = ch == "%" or (pair_made_one and ch == "\\")
            out.append(ch if kept_alone else "\\" + ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "%":
            break
        else:
            out.append(ch)
    return "".join(out) + ("\\" if escaped else "")


REWRITES: dict[str, Callable[[str], str]] = {
    "FreeBSD and macOS": lambda command: rewrite(command, pair_made_one=True),
    "cronie and OpenBSD": lambda command: rewrite(command, pair_made_one=False),
    "busybox": lambda command: command,
}


class ABackslashPathIsNamedAsItselfOnEveryCronTests(unittest.TestCase):
    def assert_every_cron_runs_the_job(self, crons: tuple[str, ...], **how: str) -> str:
        code, out, err, paths = render("cron", **how)
        self.assertEqual(code, 0, err)
        (job,) = [
            line
            for line in out.splitlines()
            if line.strip() and not line.startswith("#") and "daily --commit" in line
        ]
        command = job.split(None, 5)[5]
        want = [
            "export",
            "ASI_AUTOMATED_RUN=1",
            ";",
            "cd",
            paths["repo"],
            "&&",
            paths["python"],
            "-m",
            "agent_skill_improver",
        ]
        want += ["--config", paths["config"], "daily", "--commit"]
        want += [
            ">>",
            paths["state"] + "/daily.out.log",
            "2",
            ">>",
            paths["state"] + "/daily.err.log",
        ]
        for cron in crons:
            with self.subTest(cron=cron):
                self.assertEqual(shell_words(REWRITES[cron](command)), want)
        return command

    def test_a_config_path_holding_a_backslash_is_named_as_itself_by_every_cron(self) -> None:
        for name in BACKSLASHED:
            with self.subTest(name=name):
                self.assert_every_cron_runs_the_job(tuple(REWRITES), config_name=name)

    def test_a_state_directory_holding_a_backslash_is_named_as_itself_by_every_cron(self) -> None:
        for name in BACKSLASHED:
            with self.subTest(name=name):
                self.assert_every_cron_runs_the_job(tuple(REWRITES), state_name=name)

    def test_a_backslash_before_a_percent_is_named_as_itself_by_each_rewriting_cron(self) -> None:
        # busybox hands a quoted ``\\%`` to the shell as it is: not portable, as documented.
        self.assert_every_cron_runs_the_job(
            ("FreeBSD and macOS", "cronie and OpenBSD"), config_name=BEFORE_A_PERCENT
        )

    def test_no_backslash_pair_is_printed_for_any_cron_to_read_its_own_way(self) -> None:
        for name in (*BACKSLASHED, BEFORE_A_PERCENT):
            with self.subTest(name=name):
                command = self.assert_every_cron_runs_the_job((), config_name=name)
                self.assertIn("\\", command)
                self.assertNotIn("\\\\", command)

    def test_a_real_shell_reads_each_spelling_back_as_the_path(self) -> None:
        for name in (*BACKSLASHED, BEFORE_A_PERCENT):
            value = f"/base/{name}/config.toml"
            for cron, rewritten in REWRITES.items():
                if cron == "busybox" and "%" in value:
                    continue
                with self.subTest(name=name, cron=cron):
                    shell_sees = rewritten(_cron_quote(value))
                    proc = subprocess.run(
                        ["/bin/sh", "-c", f"printf %s {shell_sees}"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    self.assertEqual(proc.stdout, value, proc.stderr)
                    self.assertEqual(shlex.split(shell_sees), [value])

    def test_the_launchd_plist_carries_a_backslash_path_exactly(self) -> None:
        for name in (*BACKSLASHED, BEFORE_A_PERCENT):
            with self.subTest(name=name):
                code, out, err, paths = render("launchd", config_name=name)
                self.assertEqual(code, 0, err)
                job = plistlib.loads(out.encode("utf-8"))
                self.assertEqual(job["ProgramArguments"][4], paths["config"])
                self.assertEqual(job["StandardOutPath"], paths["state"] + "/daily.out.log")

    def test_the_control_a_path_without_a_backslash_is_spelled_as_before(self) -> None:
        for value, spelled in (
            ("/plain/path", "/plain/path"),
            ("/a/100%", "/a/100\\%"),
            ("/My Projects & co 100%/state", "'/My Projects & co 100\\%/state'"),
            ("/it's/here", "'/it'\"'\"'s/here'"),
        ):
            with self.subTest(value=value):
                self.assertEqual(_cron_quote(value), spelled)


if __name__ == "__main__":
    unittest.main()
