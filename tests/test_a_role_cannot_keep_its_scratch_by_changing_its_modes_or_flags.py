"""A role's per-call scratch is removed whatever the role did to its modes or flags.

Each model call runs in a fresh private directory under the state tree's
``sandbox`` directory, and the runner removes it when the call returns. The
removal was one ``shutil.rmtree(..., ignore_errors=True)``, which stops at an
entry the role made read-only, unsearchable or immutable and removes no link or
file the role put where its scratch was. Measured before this change, with and
without the OS sandbox and on both Pythons: a subdirectory at mode 500 or 000,
the scratch itself at 500 or 000, ``chflags uchg`` on a file or a directory, and
the scratch replaced by a link or a file, two links inside it to one flagged
file, a link flagged itself, and a tree 1,200 directories deep each left the
role's entries in the state tree, and the call reported success, except that on
3.11 the deep tree raised ``RecursionError`` out of the call.
"""

from __future__ import annotations

import os
import resource
import stat
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_skill_improver.adapters import model_command
from agent_skill_improver.adapters.model_command import CommandModelRunner, RoleCommand
from agent_skill_improver.adapters.sandbox import NoSandbox, OsSandbox, detect, verify
from agent_skill_improver.errors import MalformedModelOutput, TransientFailure

REPLY = 'printf %s "{\\"x\\":1}"'
HAS_FLAGS = hasattr(os, "lchflags")


def _unlock(path: Path) -> None:
    """Give a test's tree back to its owner, so the temporary directory can go."""
    if HAS_FLAGS:
        subprocess.run(["chflags", "-R", "nouchg,nouappnd", str(path)], capture_output=True)
    subprocess.run(["chmod", "-R", "u+rwx", str(path)], capture_output=True)


class _Scratch(unittest.TestCase):
    isolation: object = None

    def setUp(self) -> None:
        held = tempfile.TemporaryDirectory()
        self.base = Path(held.name).resolve()
        self.addCleanup(held.cleanup)
        self.addCleanup(_unlock, self.base)
        self.calls = 0
        self.root = self.base / "sandbox-0"
        self.root.mkdir()
        self.outside = self.base / "outside"
        self.outside.mkdir()
        (self.outside / "keep").write_text("keep", encoding="utf-8")
        os.chmod(self.outside, 0o555)

    def _call(self, script: str, **limits: int) -> None:
        # Each call gets a scratch root of its own, so what one subtest leaves
        # cannot fail the next.
        self.calls += 1
        self.root = self.base / f"sandbox-{self.calls}"
        self.root.mkdir()
        script = script.replace("$OUTSIDE", str(self.outside))
        command = RoleCommand(role="scorer", argv=("/bin/sh", "-c", script), **limits)
        runner = CommandModelRunner(
            {"scorer": command},
            isolation=self.isolation or NoSandbox(),
            workdir_root=self.root,
            parent_env={"PATH": "/usr/bin:/bin"},
        )
        runner.run(role="scorer", system="s", user="u", response_schema="x")

    def _left(self) -> list[str]:
        found = []
        for top, dirs, files in os.walk(self.root):
            found += [str((Path(top) / name).relative_to(self.root)) for name in dirs + files]
        return sorted(found)

    def assertOutsideUntouched(self) -> None:
        self.assertEqual(stat.S_IMODE(os.lstat(self.outside).st_mode), 0o555)
        self.assertEqual(sorted(os.listdir(self.outside)), ["keep"])
        self.assertEqual((self.outside / "keep").read_text(encoding="utf-8"), "keep")


LOCKED = {
    "a subdirectory at 500": 'mkdir "$HOME/sub"; echo x > "$HOME/sub/f"; chmod 500 "$HOME/sub"',
    "a subdirectory at 000": 'mkdir "$HOME/sub"; echo x > "$HOME/sub/f"; chmod 000 "$HOME/sub"',
    "the scratch at 500": 'echo x > "$HOME/f"; chmod 500 "$HOME"',
    "the scratch at 000": 'echo x > "$HOME/f"; chmod 000 "$HOME"',
    "nested directories at 000": (
        'mkdir -p "$HOME/a/b/c"; echo x > "$HOME/a/b/c/f"; '
        'chmod 000 "$HOME/a/b/c" "$HOME/a/b"; chmod 500 "$HOME/a"'
    ),
}
FLAGGED = {
    "a file marked uchg": 'echo x > "$HOME/f"; chflags uchg "$HOME/f"',
    "a directory marked uchg": 'mkdir "$HOME/sub"; echo x > "$HOME/sub/f"; chflags uchg "$HOME/sub"',
    "a file marked uappnd": 'echo x > "$HOME/f"; chflags uappnd "$HOME/f"',
    "the scratch marked uchg": 'echo x > "$HOME/f"; chflags uchg "$HOME"',
    "two links inside to one flagged file": (
        'echo x > "$HOME/f"; ln "$HOME/f" "$HOME/g"; chflags uchg "$HOME/f"'
    ),
    "a link flagged itself": 'ln -s "$HOME/gone" "$HOME/l"; chflags -h uchg "$HOME/l"',
    "a flag and a mode together": (
        'mkdir "$HOME/sub"; echo x > "$HOME/sub/f"; chflags uchg "$HOME/sub/f"; '
        'chmod 000 "$HOME/sub"'
    ),
}


class TheScratchIsRemovedWhateverTheRoleDidToItTests(_Scratch):
    def test_each_entry_the_role_locked_is_removed_when_the_call_returns(self) -> None:
        for name, script in LOCKED.items():
            with self.subTest(name):
                self._call(f"{script}\n{REPLY}")
                self.assertEqual(self._left(), [])

    @unittest.skipUnless(HAS_FLAGS, "only macOS and the BSDs have user flags")
    def test_each_entry_the_role_made_immutable_is_removed_when_the_call_returns(self) -> None:
        for name, script in FLAGGED.items():
            with self.subTest(name):
                self._call(f"{script}\n{REPLY}")
                self.assertEqual(self._left(), [])

    def test_a_link_or_a_file_where_the_scratch_was_is_removed(self) -> None:
        for name, script in {
            "a link to a directory elsewhere": 'rm -rf "$HOME"; ln -s "$OUTSIDE" "$HOME"',
            "a dangling link": 'rm -rf "$HOME"; ln -s "$HOME.gone" "$HOME"',
            "a file": 'rm -rf "$HOME"; echo x > "$HOME"',
        }.items():
            with self.subTest(name):
                self._call(f"{script}\n{REPLY}")
                self.assertEqual(self._left(), [])
                self.assertOutsideUntouched()

    def test_a_link_out_of_the_scratch_is_never_followed(self) -> None:
        # The scratch at 500 makes the first pass leave it, so the second pass
        # meets the link; the directory it points at keeps its mode and content.
        for name, script in {
            "a link to a directory": 'ln -s "$OUTSIDE" "$HOME/out"; chmod 500 "$HOME"',
            "a link to a file": 'ln -s "$OUTSIDE/keep" "$HOME/out"; chmod 000 "$HOME"',
        }.items():
            with self.subTest(name):
                self._call(f"{script}\n{REPLY}")
                self.assertEqual(self._left(), [])
                self.assertOutsideUntouched()

    def test_the_scratch_is_removed_whatever_the_outcome(self) -> None:
        lock = 'mkdir "$HOME/sub"; echo x > "$HOME/sub/f"; chmod 000 "$HOME/sub"; chmod 500 "$HOME"'
        cases = {
            "a non-zero exit": (f"{lock}\nexit 3", {}, TransientFailure),
            "a timeout": (f"{lock}\nsleep 30", {"timeout_seconds": 1}, TransientFailure),
            "output past the cap": (
                f"{lock}\nhead -c 100000 /dev/zero",
                {"max_output_bytes": 1000},
                MalformedModelOutput,
            ),
        }
        for name, (script, limits, error) in cases.items():
            with self.subTest(name):
                with self.assertRaises(error):
                    self._call(script, **limits)
                self.assertEqual(self._left(), [])


class TheSecondPassAndItsLimitTests(_Scratch):
    def test_a_scratch_the_role_left_alone_is_removed_without_forcing_a_mode_or_a_flag(
        self,
    ) -> None:
        forced = AssertionError("a mode or a flag was forced")
        with (
            mock.patch.object(model_command, "_clear_flags", side_effect=forced),
            mock.patch.object(model_command, "_chmod_at", side_effect=forced),
        ):
            self._call(f'echo x > "$HOME/f"; mkdir "$HOME/d"; echo y > "$HOME/d/g"\n{REPLY}')
        self.assertEqual(self._left(), [])

    @unittest.skipUnless(HAS_FLAGS, "only macOS and the BSDs have user flags")
    def test_the_control_a_file_linked_from_elsewhere_keeps_its_flags_and_stays(self) -> None:
        # Unconfined, a role can hard-link a file from elsewhere into its scratch
        # and flag it through the link. Clearing the flag would change that file,
        # so it is left, flag and link both; the file's content is unchanged.
        keep = self.outside / "keep"
        self._call(f'ln "{keep}" "$HOME/hl"; chflags uchg "$HOME/hl"\n{REPLY}')
        self.assertTrue(os.lstat(keep).st_flags & stat.UF_IMMUTABLE)
        self.assertEqual(keep.read_text(encoding="utf-8"), "keep")
        self.assertEqual([name.split("/", 1)[1:] for name in self._left()], [[], ["hl"]])

    @unittest.skipUnless(HAS_FLAGS, "only macOS and the BSDs have user flags")
    def test_a_flagged_file_behind_a_link_out_of_the_scratch_keeps_its_flags(self) -> None:
        # The second pass walks the scratch; a link to a directory elsewhere is
        # not walked into, so a file there that carries a flag keeps it.
        keep = self.outside / "keep"
        os.chflags(keep, stat.UF_IMMUTABLE)
        self._call(f'ln -s "$OUTSIDE" "$HOME/out"; chmod 500 "$HOME"\n{REPLY}')
        self.assertEqual(self._left(), [])
        self.assertTrue(os.lstat(keep).st_flags & stat.UF_IMMUTABLE)
        self.assertOutsideUntouched()

    @unittest.skipUnless(HAS_FLAGS, "only macOS and the BSDs have user flags")
    def test_a_link_flagged_itself_is_cleared_and_removed_and_its_target_keeps_its_flags(
        self,
    ) -> None:
        keep = self.outside / "keep"
        os.chflags(keep, stat.UF_IMMUTABLE)
        self._call(f'ln -s "$OUTSIDE/keep" "$HOME/l"; chflags -h uchg "$HOME/l"\n{REPLY}')
        self.assertEqual(self._left(), [])
        self.assertTrue(os.lstat(keep).st_flags & stat.UF_IMMUTABLE)
        self.assertOutsideUntouched()

    def test_a_tree_deeper_than_the_recursion_limit_is_removed(self) -> None:
        # One descriptor a level is held: make room where the soft limit is low.
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft != resource.RLIM_INFINITY and soft < 4096:
            room = 4096 if hard == resource.RLIM_INFINITY else min(4096, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (room, hard))
            self.addCleanup(resource.setrlimit, resource.RLIMIT_NOFILE, (soft, hard))
        build = (
            "import os, sys\n"
            "fd = os.open(os.environ['HOME'], os.O_RDONLY)\n"
            "for _ in range(1200):\n"
            "    os.mkdir('d', dir_fd=fd)\n"
            "    deeper = os.open('d', os.O_RDONLY, dir_fd=fd)\n"
            "    os.close(fd)\n"
            "    fd = deeper\n"
            "w = os.open('f', os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=fd)\n"
            "os.write(w, b'x')\n"
            "os.close(w)\n"
            "os.fchmod(fd, 0o500)\n"
            "sys.stdout.write('{\"x\": 1}')\n"
        )
        command = RoleCommand(role="scorer", argv=(sys.executable, "-c", build))
        runner = CommandModelRunner(
            {"scorer": command},
            isolation=NoSandbox(),
            workdir_root=self.root,
            parent_env={"PATH": "/usr/bin:/bin"},
        )
        runner.run(role="scorer", system="s", user="u", response_schema="x")
        self.assertEqual(os.listdir(self.root), [])

    def test_a_directory_swapped_for_a_link_after_it_is_opened_does_not_steer_the_walk(
        self,
    ) -> None:
        # A process the role left running renames a directory of the scratch
        # just after the walk opened it and puts a link to a directory elsewhere
        # in its place. The walk goes on through what it holds open, so the
        # directory elsewhere keeps its mode, its subdirectory's mode and a flag.
        os.chmod(self.outside, 0o755)
        (self.outside / "sub").mkdir(mode=0o755)
        os.chmod(self.outside / "sub", 0o755)
        if HAS_FLAGS:
            os.chflags(self.outside / "keep", stat.UF_IMMUTABLE)
        os.chmod(self.outside, 0o555)
        scratch = self.root / "asi-role-raced"
        (scratch / "d" / "inner").mkdir(parents=True)
        (scratch / "d" / "x").write_text("x", encoding="utf-8")
        os.chmod(scratch / "d", 0o500)
        opened = model_command._open_directory

        def racing(directory: int, name: str) -> int | None:
            held = opened(directory, name)
            if name == "d":
                os.rename("d", "moved", src_dir_fd=directory, dst_dir_fd=directory)
                os.symlink(self.outside, "d", dir_fd=directory)
            return held

        with mock.patch.object(model_command, "_open_directory", side_effect=racing):
            model_command.remove_scratch(scratch)
        self.assertEqual(stat.S_IMODE(os.lstat(self.outside).st_mode), 0o555)
        self.assertEqual(stat.S_IMODE(os.lstat(self.outside / "sub").st_mode), 0o755)
        self.assertEqual(sorted(os.listdir(self.outside)), ["keep", "sub"])
        if HAS_FLAGS:
            self.assertTrue(os.lstat(self.outside / "keep").st_flags & stat.UF_IMMUTABLE)
        # What the walk held is emptied; the link and the renamed directory stay.
        self.assertEqual(os.listdir(scratch / "moved"), [])

    @unittest.skipUnless(
        os.chmod in os.supports_follow_symlinks, "chmod here cannot be told not to follow"
    )
    def test_a_directory_swapped_for_a_link_before_it_is_opened_is_unlinked_not_followed(
        self,
    ) -> None:
        # lstat found a directory; by the time it is opened a link to a
        # directory elsewhere stands there. Whether the open is refused at once
        # or first as unreadable (so the mode is set by name), the link is
        # neither followed nor opened, and it is unlinked.
        for refused_first in (False, True):
            with self.subTest(refused_first=refused_first):
                scratch = self.root / f"asi-role-swapped-{refused_first}"
                scratch.mkdir()
                (scratch / "d").symlink_to(self.outside)
                real_stat, real_open = os.stat, os.open
                seen_before_the_swap = real_stat(self.outside)
                refusals = [refused_first]

                def stat_(path, *args, **kwargs):
                    if path == "d" and kwargs.get("dir_fd") is not None:
                        return seen_before_the_swap
                    return real_stat(path, *args, **kwargs)

                def open_(path, *args, **kwargs):
                    if path == "d" and refusals and refusals.pop():
                        raise PermissionError(13, "Permission denied")
                    return real_open(path, *args, **kwargs)

                with (
                    mock.patch.object(model_command.os, "stat", side_effect=stat_),
                    mock.patch.object(model_command.os, "open", side_effect=open_),
                ):
                    model_command.remove_scratch(scratch)
                self.assertOutsideUntouched()
                self.assertFalse(os.path.lexists(scratch))

    def test_a_directory_swapped_for_a_file_before_it_is_opened_is_unlinked_not_opened(
        self,
    ) -> None:
        # lstat found a directory; by the time it is opened a file stands there:
        # one of the scratch's own, or a hard link to a file elsewhere. It is
        # unlinked and never opened, so the file elsewhere keeps its mode.
        elsewhere = self.base / "elsewhere"
        elsewhere.write_text("elsewhere", encoding="utf-8")
        os.chmod(elsewhere, 0o444)
        real_stat = os.stat
        seen_before_the_swap = real_stat(self.outside)
        for kind in ("a file of its own", "a hard link to a file elsewhere"):
            with self.subTest(kind):
                scratch = self.root / f"asi-role-{kind.split()[-1]}"
                scratch.mkdir()
                if kind == "a file of its own":
                    (scratch / "d").write_text("x", encoding="utf-8")
                else:
                    os.link(elsewhere, scratch / "d")

                def stat_(path, *args, **kwargs):
                    if path == "d" and kwargs.get("dir_fd") is not None:
                        return seen_before_the_swap
                    return real_stat(path, *args, **kwargs)

                with mock.patch.object(model_command.os, "stat", side_effect=stat_):
                    model_command.remove_scratch(scratch)
                self.assertFalse(os.path.lexists(scratch))
                self.assertEqual(stat.S_IMODE(os.lstat(elsewhere).st_mode), 0o444)
                self.assertEqual(os.lstat(elsewhere).st_nlink, 1)
                self.assertEqual(elsewhere.read_text(encoding="utf-8"), "elsewhere")
                self.assertOutsideUntouched()

    def test_where_chmod_cannot_refuse_to_follow_nothing_is_followed_and_the_call_returns(
        self,
    ) -> None:
        # CPython raises ValueError (given a directory descriptor) or
        # NotImplementedError when fchmodat cannot honour AT_SYMLINK_NOFOLLOW, as
        # on Linux before glibc 2.32 and for a link on any Linux. The mode is then
        # not set: the unreadable directory stays and no link is followed.
        real_chmod, real_stat, real_open = os.chmod, os.stat, os.open
        for raised in (ValueError, NotImplementedError):

            def chmod(path, mode, *args, **kwargs):
                if kwargs.get("follow_symlinks") is False:
                    raise raised("chmod: cannot use dir_fd and follow_symlinks together")
                return real_chmod(path, mode, *args, **kwargs)

            with self.subTest(raised=raised.__name__, case="a subdirectory at 000"):
                with mock.patch.object(model_command.os, "chmod", side_effect=chmod):
                    self._call(f'{LOCKED["a subdirectory at 000"]}\n{REPLY}')
                self.assertEqual([name.split("/", 1)[1:] for name in self._left()], [[], ["sub"]])
                self.assertOutsideUntouched()
            with self.subTest(raised=raised.__name__, case="swapped for a link"):
                scratch = self.root / f"asi-role-swapped-{raised.__name__}"
                scratch.mkdir()
                (scratch / "d").symlink_to(self.outside)
                seen_before_the_swap = real_stat(self.outside)
                refusals = [True]

                def stat_(path, *args, **kwargs):
                    if path == "d" and kwargs.get("dir_fd") is not None:
                        return seen_before_the_swap
                    return real_stat(path, *args, **kwargs)

                def open_(path, *args, **kwargs):
                    if path == "d" and refusals and refusals.pop():
                        raise PermissionError(13, "Permission denied")
                    return real_open(path, *args, **kwargs)

                with (
                    mock.patch.object(model_command.os, "chmod", side_effect=chmod),
                    mock.patch.object(model_command.os, "stat", side_effect=stat_),
                    mock.patch.object(model_command.os, "open", side_effect=open_),
                ):
                    model_command.remove_scratch(scratch)
                self.assertOutsideUntouched()

    @unittest.skipUnless(HAS_FLAGS, "only macOS and the BSDs have user flags")
    def test_the_working_directory_is_given_back_after_a_flag_is_cleared(self) -> None:
        here = os.getcwd()
        self.addCleanup(os.chdir, here)
        self._call(f'{FLAGGED["a file marked uchg"]}\n{REPLY}')
        self.assertEqual(self._left(), [])
        self.assertEqual(os.getcwd(), here)
        refused = OSError(1, "Operation not permitted")
        with mock.patch.object(model_command.os, "lchflags", side_effect=refused):
            self._call(f'{FLAGGED["a file marked uchg"]}\n{REPLY}')
        self.assertEqual(os.getcwd(), here)
        self.assertEqual([name.split("/", 1)[1:] for name in self._left()], [[], ["f"]])

    def test_remove_scratch_removes_a_locked_tree_without_a_role(self) -> None:
        scratch = self.root / "asi-role-direct"
        (scratch / "a" / "b").mkdir(parents=True)
        (scratch / "a" / "b" / "f").write_text("x", encoding="utf-8")
        os.chmod(scratch / "a" / "b", 0)
        os.chmod(scratch / "a", 0o500)
        model_command.remove_scratch(scratch)
        self.assertEqual(self._left(), [])
        model_command.remove_scratch(scratch)  # nothing there is no error
        self.assertFalse(os.path.lexists(scratch))


def _verified_sandbox() -> OsSandbox | None:
    mechanism = detect()
    if mechanism is None:
        return None
    sandbox = OsSandbox(mechanism)
    with tempfile.TemporaryDirectory() as tmp:
        return sandbox if verify(sandbox, workdir_root=Path(tmp)).get("ok") else None


_SANDBOX = _verified_sandbox()


@unittest.skipIf(_SANDBOX is None, "no verified OS sandbox on this host")
class TheScratchIsRemovedUnderTheOsSandboxTests(_Scratch):
    isolation = _SANDBOX

    def test_a_confined_role_that_locks_its_scratch_leaves_nothing(self) -> None:
        scripts = dict(LOCKED, **(FLAGGED if HAS_FLAGS else {}))
        for name, script in scripts.items():
            with self.subTest(name):
                self._call(f"{script}\n{REPLY}")
                self.assertEqual(self._left(), [])


if __name__ == "__main__":
    unittest.main()
