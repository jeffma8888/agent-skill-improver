"""A listing's workspace line is read in time in proportion to it.

The conversation source's listing names each thread's workspace on a line of
its own, and that line was matched with ``.+?\\s*$``: the lazy path stopped at
each character of a run of space and tried the rest of the run again from
there, so the time grew with the square of the run.
"""

from __future__ import annotations

import random
import re
import unittest

from agent_skill_improver.adapters.conversation_cli import _WORKSPACE_LINE, parse_discovery

from tests.test_skill_identity_and_payloads import deadline

#: The pattern this change replaced, kept to show the matches are the same.
OLD = re.compile(r"^\s*Workspace:\s*(?P<path>.+?)\s*$")
LONG = 262_144  # 256 KiB


def _listing(workspace_line: str) -> str:
    return f'1. "fine" (2026-01-02)\n{workspace_line}\n   Thread: 1700000000102-1\n'


class AListingsWorkspaceLineIsReadInLinearTimeTests(unittest.TestCase):
    def test_a_workspace_with_a_long_run_of_space_inside_is_read_in_linear_time(self):
        path = "/a" + " " * LONG + "b"
        with deadline(5.0):
            threads, passed_over = parse_discovery(_listing(f"   Workspace: {path}   "))
        self.assertEqual([(t.thread_id, t.workspace) for t in threads], [("1700000000102-1", path)])
        self.assertEqual(passed_over, [])

    def test_the_control_a_workspace_reads_as_it_did(self):
        for line, path in (
            ("   Workspace: /w", "/w"),
            ("Workspace:/w", "/w"),
            ("\tWorkspace:   /a b \t ", "/a b"),
            ("   Workspace:  \t ", " "),
            ("   Workspace: \u00a0/w\u00a0", "/w"),
        ):
            with self.subTest(line=line):
                threads, _ = parse_discovery(_listing(line))
                self.assertEqual([t.workspace for t in threads], [path])
                self.assertEqual(OLD.match(line).group("path"), path, "as the old pattern read it")

    def test_the_control_the_pattern_reads_what_its_old_pattern_read(self):
        pieces = [" ", "\t", "a", "/", "Workspace:", " Workspace:", "\r", "\u00a0", "b"]
        pieces += ["\x1c", "\u2028", "\u3000", "\x85", "\n"]
        rng = random.Random(147)
        for _ in range(30_000):
            line = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 16)))
            old, new = OLD.match(line), _WORKSPACE_LINE.match(line)
            self.assertEqual(
                (old.group("path"), old.span()) if old else None,
                (new.group("path"), new.span()) if new else None,
                repr(line),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
