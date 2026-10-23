"""A state directory this process cannot search is refused, not a crash.

Below a directory this process may not search, ``lstat`` raises rather than
answers, and ``Path.is_symlink`` and ``Path.exists`` raise with it; the layout
check asked both. Measured before (3.11 and 3.13): a state directory at mode
0600 or 0000, or with its parent at either mode, fresh or built, ended
``doctor``, ``daily``, ``candidates``, ``set-aside``, ``review``, ``apply`` and
``rollback`` as internal errors (exit 70) with a ``PermissionError``
traceback; with a directory two levels up at either mode, ``doctor`` did, and
the other six refused it as a layout that "must be reachable without following
a symlink". Each now names the first directory on the way that cannot be
searched, with its mode and the fix, in one line, and changes nothing.
"""

from __future__ import annotations

import json
import os
import pwd
import re
import stat
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.app import state as state_module
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.errors import ConfigError, SafetyRefusal
from tests.support import build_env, run_cli

DAY = "2026-03-01"
FORGED = "config error: forged"
STATE_COMMANDS = (
    ["daily", "--day", DAY, "--commit"],
    ["candidates"],
    ["set-aside"],
    ["review", "CAND-abc"],
    ["apply", "CAND-abc"],
    ["rollback", "CAND-abc"],
    ["discover"],
)
PERMS_LINE = (
    "state perms  dirs 0o700 / files 0o600, not measured: "
    "this process cannot reach inside the state directory"
)


def shown_mode(mode: int) -> str:
    return f"0o{mode:03o}"


def root_reason(state: Path, mode: int) -> str:
    return (
        f"the state directory {state} is mode {shown_mode(mode)}, which this process cannot search, "
        "so nothing in it can be checked or used; restore it with chmod 700"
    )


def above_reason(state: Path, blocker: Path, mode: int) -> str:
    return (
        f"the state directory {state} cannot be reached: {blocker} above it is mode "
        f"{shown_mode(mode)}, which this process cannot search; give your user search permission "
        "on it (chmod u+x) or point [run].state_dir elsewhere"
    )


ACL_CAUSE = (
    "which lets its owner search it, but an access control list or another policy denies "
    "this process search"
)
ACL_FIX = (
    "remove that entry (ls -led on macOS or getfacl on Linux shows it) or point "
    "[run].state_dir elsewhere"
)
FOREIGN_FIX = "ask its owner for search permission or point [run].state_dir at a directory you own"


@unittest.skipIf(os.geteuid() == 0, "root searches a directory whatever its mode")
class AnUnsearchableStateDirectoryIsRefusedTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()

    def tree(self, name: str, *, built: bool) -> tuple[Path, Path]:
        """A config whose state directory is ``<name>/g/p/st``."""
        config = Path(build_env(self.base / name / "env", dry_run=False, seed="reach"))
        state = self.base / name / "g" / "p" / "st"
        if built:
            StateStore(state).ensure()
        else:
            state.mkdir(parents=True)
        text = config.read_text(encoding="utf-8")
        new, count = re.subn(
            r"^state_dir = .*$",
            lambda _: f"state_dir = {json.dumps(str(state))}",
            text,
            count=1,
            flags=re.M,
        )
        self.assertEqual(count, 1)
        config.write_text(new, encoding="utf-8")
        return config, state

    def shut(self, path: Path, mode: int) -> None:
        os.chmod(path, mode)
        self.addCleanup(os.chmod, path, 0o700)

    def snapshot(self, top: Path) -> list[tuple[str, int]]:
        return [(".", os.lstat(top).st_mode)] + sorted(
            (str(p.relative_to(top)), os.lstat(p).st_mode) for p in top.rglob("*")
        )

    def run_shut(self, config: Path, shut: Path, mode: int, args: list[str]):
        """Run ``args`` with ``shut`` at ``mode``; assert it was left at that mode."""
        found = stat.S_IMODE(os.lstat(shut).st_mode)
        os.chmod(shut, mode)
        try:
            proc = run_cli(["--config", str(config), *args])
            self.assertEqual(stat.S_IMODE(os.lstat(shut).st_mode), mode, args)
        finally:
            os.chmod(shut, found)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertNotIn("internal error", proc.stderr)
        return proc

    def test_every_state_command_refuses_an_unsearchable_root_in_one_line(self):
        cases = [(m, True, c) for m in (0o600, 0o000) for c in STATE_COMMANDS]
        cases += [(m, False, c) for m in (0o600, 0o000) for c in STATE_COMMANDS[:2]]
        for index, (mode, built, args) in enumerate(cases):
            with self.subTest(mode=oct(mode), built=built, command=args[0]):
                config, state = self.tree(f"c{index}", built=built)
                before = self.snapshot(state)
                proc = self.run_shut(config, state, mode, list(args))
                self.assertEqual(proc.returncode, 3, proc.stderr)
                self.assertEqual(
                    proc.stderr.splitlines(), [f"ConfigError: {root_reason(state, mode)}"]
                )
                self.assertEqual(self.snapshot(state), before)

    def test_doctor_names_it_as_its_one_state_problem_and_measures_nothing(self):
        for index, (mode, built) in enumerate(
            (m, b) for m in (0o600, 0o000) for b in (False, True)
        ):
            with self.subTest(mode=oct(mode), built=built):
                config, state = self.tree(f"d{index}", built=built)
                before = self.snapshot(state)
                reason = root_reason(state, mode)
                proc = self.run_shut(config, state, mode, ["doctor"])
                self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
                lines = proc.stdout.splitlines()
                self.assertEqual(
                    [ln for ln in lines if ln.startswith("PROBLEM")], [f"PROBLEM  {reason}"]
                )
                self.assertIn(PERMS_LINE, lines)
                self.assertNotIn("not confined", proc.stdout)
                self.assertNotIn("not a real directory", proc.stdout)
                payload = json.loads(
                    self.run_shut(config, state, mode, ["--json", "doctor"]).stdout
                )
                self.assertEqual(payload["problems"], [reason])
                perms = payload["state_permissions"]
                self.assertEqual(perms["unreachable"], reason)
                self.assertEqual(perms["unsafe_layout"], [])
                self.assertEqual(perms["repaired"], [])
                self.assertEqual(self.snapshot(state), before)

    def test_a_directory_above_it_that_cannot_be_searched_is_named(self):
        index = 0
        for where in ("parent", "grandparent"):
            for mode in (0o600, 0o000):
                for built in (False, True):
                    index += 1
                    with self.subTest(where=where, mode=oct(mode), built=built):
                        config, state = self.tree(f"a{index}", built=built)
                        blocker = state.parent if where == "parent" else state.parent.parent
                        before = self.snapshot(blocker)
                        reason = above_reason(state, blocker, mode)
                        proc = self.run_shut(
                            config, blocker, mode, ["daily", "--day", DAY, "--commit"]
                        )
                        self.assertEqual(proc.returncode, 3, proc.stderr)
                        self.assertEqual(proc.stderr.splitlines(), [f"ConfigError: {reason}"])
                        proc = self.run_shut(config, blocker, mode, ["doctor"])
                        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
                        self.assertIn(f"PROBLEM  {reason}", proc.stdout.splitlines())
                        self.assertIn(PERMS_LINE, proc.stdout.splitlines())
                        self.assertEqual(self.snapshot(blocker), before)

    def test_the_control_a_root_it_can_search_but_not_list_still_runs(self):
        # 0300 grants search without read: everything below it can be measured,
        # so the run goes ahead and the root gets the mode policy asks for.
        config, state = self.tree("control", built=True)
        os.chmod(state, 0o300)
        self.addCleanup(os.chmod, state, 0o700)
        proc = run_cli(["--config", str(config), "daily", "--day", DAY])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(stat.S_IMODE(os.lstat(state).st_mode), 0o700)
        os.chmod(state, 0o300)
        proc = run_cli(["--config", str(config), "doctor"])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn(PERMS_LINE, proc.stdout)
        payload = json.loads(run_cli(["--config", str(config), "--json", "doctor"]).stdout)
        self.assertIsNone(payload["state_permissions"]["unreachable"])
        self.assertEqual(payload["problems"], [])

    def test_the_reason_names_the_first_directory_on_the_way_it_cannot_search(self):
        top = self.base / "u"
        state = top / "g" / "p" / "st"
        state.mkdir(parents=True)
        reach = state_module._reach_reason
        self.assertIsNone(reach(state))
        self.assertIsNone(reach(state / "absent"))
        self.assertIsNone(reach(top / "absent" / "st"))
        self.shut(state, 0o611)
        # Group and other search do not grant this user, the owner, search: the
        # owner's bit is the cause, not a policy.
        self.assertEqual(reach(state), root_reason(state, 0o611))
        os.chmod(state, 0o600)
        self.assertEqual(reach(state), root_reason(state, 0o600))
        self.shut(state.parent, 0o000)
        self.assertEqual(reach(state), above_reason(state, state.parent, 0o000))
        self.shut(state.parent.parent, 0o600)
        self.assertEqual(reach(state), above_reason(state, state.parent.parent, 0o600))
        # Asked before the ancestor walk, which would name the first entry it
        # cannot measure as a layout fault instead.
        self.assertEqual(
            StateStore(state).layout_problems(),
            [above_reason(state, state.parent.parent, 0o600)],
        )
        # Not created yet: the directory it would be created in is named.
        self.assertEqual(
            reach(state.parent.parent / "new"),
            above_reason(state.parent.parent / "new", state.parent.parent, 0o600),
        )
        # A ``..`` steps back over a directory just measured; the one named is
        # the one reached.
        os.chmod(state.parent.parent, 0o700)
        os.chmod(state.parent, 0o700)
        dotted = top / "g" / "p" / ".." / "p" / "st"
        self.assertEqual(reach(dotted), root_reason(dotted, 0o600))
        os.chmod(state, 0o700)
        os.chmod(state.parent, 0o600)
        self.assertEqual(reach(dotted), above_reason(dotted, top / "g" / "p", 0o600))
        (top / "g" / "x").mkdir()
        sideways = top / "g" / "x" / ".." / "p" / "st"
        self.assertEqual(reach(sideways), above_reason(sideways, top / "g" / "p", 0o600))
        # A directory above named like the state directory is still named as
        # the one above it, not given the state directory's fix.
        same = self.base / "same" / "st" / "st"
        same.mkdir(parents=True)
        self.shut(same.parent, 0o600)
        self.assertEqual(reach(same), above_reason(same, same.parent, 0o600))
        # A component that cannot be measured at all (too long a name) is the
        # layout checks' to name, not a crash here.
        self.assertIsNone(reach(self.base / ("x" * 300) / "st"))

    def test_the_fix_named_matches_why_search_is_refused(self):
        # Another user's directory: this process cannot re-mode it.
        state = self.base / "f" / "p" / "st"
        state.mkdir(parents=True)
        self.shut(state, 0o600)
        uid = os.getuid()
        with mock.patch.object(state_module.os, "getuid", return_value=uid + 1):
            self.assertEqual(
                state_module._reach_reason(state),
                f"the state directory {state} is owned by another user (uid {uid}, "
                "mode 0o600), and this process may not search it, so nothing in it can be "
                f"checked or used; {FOREIGN_FIX}",
            )
            os.chmod(state, 0o700)
            self.shut(state.parent, 0o000)
            self.assertEqual(
                state_module._reach_reason(state),
                f"the state directory {state} cannot be reached: {state.parent} above it is "
                f"owned by another user (uid {uid}, mode 0o000), and this process may "
                f"not search it; {FOREIGN_FIX}",
            )

    @unittest.skipUnless(sys.platform == "darwin", "chmod +a is the macOS ACL interface")
    def test_a_directory_an_acl_shuts_is_named_as_such_not_as_its_mode(self):
        config, state = self.tree("acl", built=True)
        me = pwd.getpwuid(os.getuid()).pw_name
        for shut, mode in ((state, 0o700), (state.parent, 0o755)):
            with self.subTest(shut=shut.name):
                os.chmod(shut, mode)
                before = self.snapshot(state.parent.parent)
                subprocess.run(
                    ["/bin/chmod", "+a", f"user:{me} deny search", str(shut)], check=True
                )
                try:
                    proc = run_cli(["--config", str(config), "daily", "--day", DAY, "--commit"])
                    reason = state_module._reach_reason(state)
                finally:
                    subprocess.run(["/bin/chmod", "-N", str(shut)], check=True)
                if shut == state:
                    want = (
                        f"the state directory {state} is mode 0o700, {ACL_CAUSE}, so nothing in "
                        f"it can be checked or used; {ACL_FIX}"
                    )
                else:
                    want = (
                        f"the state directory {state} cannot be reached: {shut} above it is mode "
                        f"0o755, {ACL_CAUSE}; {ACL_FIX}"
                    )
                self.assertEqual(reason, want)
                self.assertEqual(proc.returncode, 3, proc.stderr)
                self.assertEqual(proc.stderr.splitlines(), [f"ConfigError: {want}"])
                self.assertEqual(self.snapshot(state.parent.parent), before)
                os.chmod(shut, 0o700)

    def test_a_link_on_the_way_is_left_to_the_layout_checks(self):
        shut = self.base / "shut"
        (shut / "st").mkdir(parents=True)
        self.shut(shut, 0o600)
        via_ancestor = self.base / "link" / "st"
        (self.base / "link").symlink_to(shut)
        self.assertIsNone(state_module._reach_reason(via_ancestor))
        store = StateStore(via_ancestor)
        self.assertIsNone(store.reach_problem())
        (problem,) = store.layout_problems()
        self.assertTrue(problem.startswith("ancestor "), problem)
        self.assertIn(" is a symlink to ", problem)
        with self.assertRaises(SafetyRefusal) as ctx:
            store.check_dir("raw")
        self.assertNotIsInstance(ctx.exception, ConfigError)
        root_link = self.base / "rootlink"
        root_link.symlink_to(shut)
        self.assertIsNone(state_module._reach_reason(root_link))
        (problem,) = StateStore(root_link).layout_problems()
        self.assertIn(" is a symlink to ", problem)

    def test_the_store_refuses_before_it_creates_or_repairs_anything(self):
        state = self.base / "lib" / "st"
        StateStore(state).ensure()
        before = self.snapshot(state)
        self.shut(state, 0o600)
        store = StateStore(state)
        reason = root_reason(state, 0o600)
        self.assertEqual(store.reach_problem(), reason)
        self.assertEqual(store.layout_problems(), [reason])
        for call in (store.ensure, store.verify_layout, lambda: store.check_dir("raw")):
            with self.subTest(call=getattr(call, "__name__", "check_dir")):
                with self.assertRaises(ConfigError) as ctx:
                    call()
                self.assertEqual(str(ctx.exception), reason)
        self.assertEqual(store.repair_permissions(), [])
        self.assertEqual(stat.S_IMODE(os.lstat(state).st_mode), 0o600)
        os.chmod(state, 0o700)
        after = self.snapshot(state)
        self.assertEqual([p for p, _ in after], [p for p, _ in before])

    def test_the_reason_names_a_path_with_a_line_break_on_one_line(self):
        state = self.base / f"s\n{FORGED}" / "st"
        state.mkdir(parents=True)
        self.shut(state, 0o600)
        reason = state_module._reach_reason(state)
        self.assertIsNotNone(reason)
        self.assertNotIn("\n", reason)
        self.assertIn(f"s\\n{FORGED}", reason)
        self.shut(state.parent, 0o600)
        reason = state_module._reach_reason(state)
        self.assertNotIn("\n", reason)
        self.assertEqual(reason.count(f"s\\n{FORGED}"), 2)


if __name__ == "__main__":
    unittest.main()
