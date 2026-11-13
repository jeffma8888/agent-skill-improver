"""A scheduler template fills each marker once, from the template alone.

``cmd_scheduler_template`` in ``cli.py`` puts four paths into a crontab line or
a property list: the repository, the interpreter, the config file and the state
directory. It replaced one marker at a time, so a path holding the text of a
marker filled later (``__CONFIG__`` or ``__STATE__``) had that value spliced
into it after it was quoted. Measured before (3.11 and 3.13): a config under a
directory named ``x __STATE__ y`` printed a cron line of 17 shell words where
the job has 15 and a plist naming a config that does not exist; one named
``q __STATE__ ;echo INJECTED;`` printed a cron line the shell splits at both
semicolons, two commands the config never named; a repository path holding
``__CONFIG__`` printed the wrong working directory in both, and on 3.11 an
interpreter path holding ``__STATE__`` (a symbolic link to the interpreter,
which that build keeps as ``sys.executable``) the wrong interpreter. Now the
template is read once, each marker replaced by its value as it is met.
"""

from __future__ import annotations

import contextlib
import io
import json
import plistlib
import shlex
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.config import load_config
from tests.support import build_env

MARKERS = "m __REPO__ __PYTHON__ __CONFIG__ __STATE__ m"
INJECTING = "q __STATE__ ;echo INJECTED;"
FLAVORS = ("cron", "launchd")


def cron_command(line: str) -> str:
    """The command of a crontab line as cron hands it to the shell.

    Cron ends the command at a bare ``%`` (the rest is its stdin). The crons
    differ in what they make of a backslash, so these paths hold none, and the
    line is checked to hold none, which keeps this reading exact for it.
    """
    command = line.split(None, 5)[5]
    assert "\\" not in command, command
    return command.split("%", 1)[0]


def shell_words(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def render(
    flavor: str,
    *,
    config_name: str = "plain",
    repo_name: str | None = None,
    python: str | None = None,
    state_name: str | None = None,
) -> tuple[int, str, str, dict[str, str]]:
    """Render one template in this process with the paths named; return what it printed."""
    with TemporaryDirectory() as tmp:
        base = Path(tmp).resolve()
        config_path = Path(build_env(base / config_name, dry_run=False)).resolve()
        if state_name is not None:
            text = config_path.read_text(encoding="utf-8")
            line = 'state_dir = "state"'
            assert text.count(line) == 1
            config_path.write_text(
                text.replace(line, "state_dir = " + json.dumps(str(base / state_name))),
                encoding="utf-8",
            )
        repo = cli.repo_root()
        if repo_name is not None:
            repo = base / repo_name
            shutil.copytree(cli.repo_root() / "packaging", repo / "packaging")
        interpreter = python if python is not None else sys.executable
        paths = {
            "repo": str(repo),
            "python": interpreter,
            "config": str(config_path),
            "state": str(load_config(config_path).state_dir),
        }
        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(cli, "repo_root", return_value=repo),
            mock.patch.object(sys, "executable", interpreter),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            code = cli.main(["--config", str(config_path), "scheduler-template", flavor])
    return code, out.getvalue(), err.getvalue(), paths


class EachMarkerIsFilledOnceTests(unittest.TestCase):
    def assert_runs_exactly(self, flavor: str, **how: str) -> list[str]:
        code, out, err, paths = render(flavor, **how)
        self.assertEqual(code, 0, err)
        argv = [paths["python"], "-m", "agent_skill_improver", "--config", paths["config"]]
        argv += ["daily", "--commit"]
        if flavor == "cron":
            (job,) = [
                line
                for line in out.splitlines()
                if line.strip() and not line.startswith("#") and "daily --commit" in line
            ]
            words = shell_words(cron_command(job))
            self.assertEqual(
                words,
                [
                    "export",
                    "ASI_AUTOMATED_RUN=1",
                    ";",
                    "cd",
                    paths["repo"],
                    "&&",
                    *argv,
                    ">>",
                    paths["state"] + "/daily.out.log",
                ]
                + ["2", ">>", paths["state"] + "/daily.err.log"],
            )
            return words
        job = plistlib.loads(out.encode("utf-8"))
        self.assertEqual(job["ProgramArguments"], argv)
        self.assertEqual(job["WorkingDirectory"], paths["repo"])
        self.assertEqual(job["StandardOutPath"], paths["state"] + "/daily.out.log")
        self.assertEqual(job["StandardErrorPath"], paths["state"] + "/daily.err.log")
        return job["ProgramArguments"]

    def test_the_control_plain_paths_run_exactly_the_configured_job(self) -> None:
        for flavor in FLAVORS:
            with self.subTest(flavor=flavor):
                self.assert_runs_exactly(flavor)

    def test_a_config_path_holding_every_markers_text_is_carried_as_itself(self) -> None:
        for flavor in FLAVORS:
            with self.subTest(flavor=flavor):
                self.assert_runs_exactly(flavor, config_name=MARKERS)

    def test_a_state_directory_holding_every_markers_text_is_carried_as_itself(self) -> None:
        for flavor in FLAVORS:
            with self.subTest(flavor=flavor):
                self.assert_runs_exactly(flavor, state_name=MARKERS)

    def test_a_repository_path_holding_every_markers_text_is_carried_as_itself(self) -> None:
        for flavor in FLAVORS:
            with self.subTest(flavor=flavor):
                self.assert_runs_exactly(flavor, repo_name=MARKERS)

    def test_an_interpreter_path_holding_every_markers_text_is_carried_as_itself(self) -> None:
        for flavor in FLAVORS:
            with self.subTest(flavor=flavor):
                self.assert_runs_exactly(flavor, python=f"/opt/{MARKERS}/bin/python3")

    def test_a_marker_in_the_config_path_cannot_add_a_shell_command(self) -> None:
        words = self.assert_runs_exactly("cron", config_name=INJECTING)
        self.assertEqual(words.count(";"), 1)  # Only the deliberate export prefix.
        self.assertEqual(words.count("&&"), 1)
        argv = self.assert_runs_exactly("launchd", config_name=INJECTING)
        self.assertTrue(argv[4].endswith(f"/{INJECTING}/config/config.toml"), argv[4])


if __name__ == "__main__":
    unittest.main()
