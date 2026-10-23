"""A promotion refusal quotes what it read from the record: status, gate reasons, path.

Measured before (3.11 and 3.13), with a real validated candidate whose stored
record was then edited and ``review`` and ``apply`` run on it: the status both
refuse was shown with ``repr``, so a ``ghp_`` token in it was printed whole and
one of a megabyte made a refusal of a megabyte; ``apply`` read the stored gate
before it checked the record, so a validated record whose gate was edited to
fail printed its stored reasons joined as written (one holding a line break
printed ``config error: forged`` as a line of its own and its token whole,
10,000 of them made 1.02 MB, one that was not a string made ``apply`` fall
with a traceback, and reasons stored as one string were joined a character at
a time); and the canonical path a base-hash refusal names was printed as
written, so a planted folder name holding a token and a line break printed
both, whether the file in it could not be read or was read and hashed
differently, and a folder name of a megabyte made a refusal of 2.0 MB; and a NUL
character in the recorded path made both exit 70 with a traceback.
"""

from __future__ import annotations

import json
import re
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable
from unittest import mock

from agent_skill_improver.app import promote
from agent_skill_improver.app.promote import _why_unreadable
from agent_skill_improver.cli import RC_REFUSED
from agent_skill_improver.domain.candidate import Candidate
from agent_skill_improver.domain.redaction import (
    NAMED_PATH_CHARS,
    QUOTED_VALUES_SHOWN,
    named_path,
    quoted_value,
)
from agent_skill_improver.errors import GateFailed
from tests.support import build_env, run_cli

TOKEN = "gh" + "p_" + "Zq8vLm3pXr7tWk2sQ1aB0123456789abcdef"
FORGED = "config error: forged"
PLANTED = TOKEN + "\n" + FORGED
MASKED = "[REDACTED:github-token]"
#: How a refusal spells ``PLANTED``: quoted, the token masked, the break escaped.
SPELLED = f"{MASKED}\\n{FORGED}"
NO_SUCH_FILE = "could not be read ([Errno 2] No such file or directory), so"
_UNUSED: Any = None

_ENV: dict[str, Any] = {}


def setUpModule() -> None:
    tmp = TemporaryDirectory()
    base = Path(tmp.name).resolve()
    cfg = ["--config", str(build_env(base, dry_run=False, seed="quoted-promotion"))]
    proc = run_cli([*cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
    validated = [
        c["candidate_id"]
        for c in json.loads(proc.stdout)["candidates"]
        if c.get("status") == "validated"
    ]
    cid = validated[0]
    review = json.loads(run_cli([*cfg, "--json", "review", cid]).stdout)
    record = base / "state" / "candidates" / f"{cid}.json"
    skill = base / "skills-canonical" / "tdd" / "SKILL.md"
    _ENV.update(
        tmp=tmp,
        base=base,
        cfg=cfg,
        cid=cid,
        token=review["approval_token"],
        record=record,
        clean_text=record.read_text(encoding="utf-8"),
        skill=skill,
        skill_bytes=skill.read_bytes(),
    )


def tearDownModule() -> None:
    _ENV["tmp"].cleanup()


def _set(*pairs: tuple[tuple[str, ...], Any]) -> Callable[[dict[str, Any]], None]:
    def edit(r: dict[str, Any]) -> None:
        for path, value in pairs:
            node = r
            for step in path[:-1]:
                node = node[step]
            node[path[-1]] = value

    return edit


def _in_folder(folder: str) -> Callable[[dict[str, Any]], None]:
    """The record's skill moved to ``skills-canonical/folder``, its id to match."""
    path = str(Path(_ENV["base"]) / "skills-canonical" / folder / "SKILL.md")
    return _set((("target", "canonical_path"), path), (("target", "skill_id"), folder))


def _gate_failed(reasons: Any) -> Callable[[dict[str, Any]], None]:
    return _set(
        (("validation", "gate", "passed"), False), (("validation", "gate", "reasons"), reasons)
    )


class _Refusals(unittest.TestCase):
    def tearDown(self) -> None:
        _ENV["record"].write_text(_ENV["clean_text"], encoding="utf-8")
        self.assertEqual(_ENV["skill"].read_bytes(), _ENV["skill_bytes"], "the skill was written")

    def refusals(
        self, edit: Callable[[dict[str, Any]], None], *, plant: str | None = None
    ) -> dict[str, str]:
        """What ``review`` and ``apply`` print for the record as ``edit`` leaves it."""
        record = json.loads(_ENV["clean_text"])
        edit(record)
        text = json.dumps(record, indent=2, sort_keys=True) + "\n"
        _ENV["record"].write_text(text, encoding="utf-8")
        if plant is not None:
            folder = Path(_ENV["base"]) / "skills-canonical" / plant
            folder.mkdir()
            self.addCleanup(shutil.rmtree, folder)
            other = _ENV["skill"].read_text(encoding="utf-8") + "\nedited by hand\n"
            (folder / "SKILL.md").write_text(other, encoding="utf-8")
        cid, token = _ENV["cid"], _ENV["token"]
        shown: dict[str, str] = {}
        for action, args in (
            ("review", ("review", cid)),
            ("apply", ("apply", cid, "--approve", token, "--i-reviewed-this", "--operator", "t")),
        ):
            proc = run_cli([*_ENV["cfg"], *args])
            shown[action] = proc.stdout + proc.stderr
            self.assertEqual(proc.returncode, RC_REFUSED, shown[action][-2_000:])
        return shown

    def assert_clean(self, text: str) -> None:
        self.assertNotIn(TOKEN, text)
        self.assertNotIn("Traceback", text)
        self.assertFalse(
            [line for line in text.splitlines() if line.startswith(FORGED[:11])], "a forged line"
        )

    def assert_quoted(self, text: str) -> None:
        self.assert_clean(text)
        self.assertIn(MASKED, text)


class AStatusIsQuotedTests(_Refusals):
    def test_a_status_holding_a_token_and_a_line_break_is_quoted_by_review_and_apply(self):
        shown = self.refusals(_set((("status",), PLANTED)))
        cid = _ENV["cid"]
        self.assertIn(
            f"GateFailed: refusing to review candidate {cid}: status is '{SPELLED}', not "
            "'validated'.",
            shown["review"],
        )
        self.assertIn(f"GateFailed: candidate {cid} has status '{SPELLED}'; only", shown["apply"])
        for action in ("review", "apply"):
            self.assert_quoted(shown[action])

    def test_a_status_of_a_megabyte_is_cut(self):
        shown = self.refusals(_set((("status",), "s" * 1_000_000)))
        for action in ("review", "apply"):
            with self.subTest(action):
                self.assertIn("more chars not quoted", shown[action])
                self.assertLess(len(shown[action]), 1_000)

    def test_a_plain_status_reads_as_before(self):
        shown = self.refusals(_set((("status",), "gate-failed")))
        self.assertIn("status is 'gate-failed', not 'validated'.", shown["review"])
        self.assertIn("has status 'gate-failed'; only a 'validated' candidate", shown["apply"])


class AStoredGateIsCheckedBeforeItIsBelievedTests(_Refusals):
    def test_apply_names_a_validated_record_whose_gate_was_edited_to_fail_as_tampered(self):
        cases = (
            ("a reason holding a token and a line break", _gate_failed([PLANTED])),
            ("ten thousand reasons", _gate_failed(["r" * 100] * 10_000)),
            ("a reason that is not a string", _gate_failed([1])),
            ("reasons stored as one string", _gate_failed(PLANTED)),
        )
        for name, edit in cases:
            with self.subTest(name):
                shown = self.refusals(edit)
                for action in ("review", "apply"):
                    self.assertIn(
                        f"CandidateTampered: refusing to {action} candidate {_ENV['cid']}:",
                        shown[action],
                    )
                    self.assertIn(
                        "status is 'validated' but the recorded gate did not pass", shown[action]
                    )
                    self.assert_clean(shown[action])
                    self.assertLess(len(shown[action]), 2_000)

    def apply_refusal(self, reasons: Any) -> str:
        """What ``apply`` says of a failed gate the record check let through."""
        record = json.loads(_ENV["clean_text"])
        _gate_failed(reasons)(record)
        candidate = Candidate.from_dict(record)
        rubrics = {"configured": object()}
        with mock.patch.object(promote, "verify_full_integrity") as checked:
            with self.assertRaises(GateFailed) as caught:
                promote.apply_candidate(
                    candidate,
                    approval=_UNUSED,
                    skill_store=_UNUSED,
                    state=_UNUSED,
                    clock=_UNUSED,
                    rubrics=rubrics,
                )
        # The record is checked against the configured rubrics before the gate is read.
        checked.assert_called_once_with(candidate, rubrics=rubrics, action="apply")
        return str(caught.exception)

    def test_the_reasons_of_a_failed_gate_are_quoted_masked_and_counted(self):
        message = self.apply_refusal([PLANTED] + ["r" * 100] * 10_000)
        self.assertIn(f"did not pass the gate: '{SPELLED}', 'rrr", message)
        self.assertIn(f"and {10_001 - QUOTED_VALUES_SHOWN} more", message)
        self.assertNotIn("\n", message)
        self.assertLess(len(message), 1_500)
        self.assert_quoted(message)

    def test_a_failed_gate_with_no_reasons_says_so(self):
        message = self.apply_refusal([])
        self.assertTrue(message.endswith("did not pass the gate: no gate decision recorded"))


class APathIsNamedBareOnlyWhenPlainTests(_Refusals):
    def test_a_planted_folder_that_cannot_be_read_is_quoted(self):
        shown = self.refusals(_in_folder(PLANTED))
        for action in ("review", "apply"):
            with self.subTest(action):
                self.assertIn(f"{SPELLED}/SKILL.md' {NO_SUCH_FILE}", shown[action])
                self.assert_quoted(shown[action])

    def test_a_planted_folder_holding_another_skill_file_is_quoted(self):
        shown = self.refusals(_in_folder(PLANTED), plant=PLANTED)
        for action in ("review", "apply"):
            with self.subTest(action):
                self.assertIn(
                    f"BaseHashMismatch: refusing to {action} {_ENV['cid']}: '", shown[action]
                )
                self.assertIn(f"{SPELLED}/SKILL.md' now hashes", shown[action])
                self.assert_quoted(shown[action])

    def test_a_folder_name_of_a_megabyte_is_cut(self):
        shown = self.refusals(_in_folder("d" * 1_000_000))
        path = str(Path(_ENV["base"]) / "skills-canonical" / ("d" * 1_000_000) / "SKILL.md")
        cut = f"[{len(repr(path)) - 512} more chars not quoted] could not be read ("
        for action in ("review", "apply"):
            with self.subTest(action):
                self.assertIn(cut, shown[action])
                self.assertLess(len(shown[action]), 1_500)

    def test_a_path_holding_a_nul_is_refused_not_fallen_over(self):
        hint = "re-run `asi daily --commit` to propose against the configured tree."
        in_folder = self.refusals(_in_folder("tdd\x00" + PLANTED))
        # Past the 512 characters a path is named in, too.
        far_in = self.refusals(_in_folder("d" * 1_000 + "\x00"))
        for action in ("review", "apply"):
            with self.subTest(action):
                refused = (
                    f"BaseHashMismatch: refusing to {action} {_ENV['cid']}: its canonical target '"
                )
                self.assertIn(refused, in_folder[action])
                self.assertIn(f"tdd\\x00{SPELLED}/SKILL.md' holds a NUL", in_folder[action])
                self.assertIn(hint, in_folder[action])
                self.assert_quoted(in_folder[action])
                self.assertIn(refused, far_in[action])
                self.assertIn("more chars not quoted] holds a NUL character", far_in[action])
        # After the name, ``apply`` refuses the name through the integrity check first.
        after_name = self.refusals(
            _set((("target", "canonical_path"), str(_ENV["skill"]) + "\x00" + PLANTED))
        )
        self.assertIn(f"SKILL.md\\x00{SPELLED}' holds a NUL", after_name["review"])
        self.assertIn("but only SKILL.md is ever written", after_name["apply"])
        for action in ("review", "apply"):
            self.assert_quoted(after_name[action])

    def test_a_plain_path_is_named_bare_as_before(self):
        base = Path(_ENV["base"])
        if not re.fullmatch(r"[A-Za-z0-9_./~+-]+", str(base)):
            self.skipTest(f"the temporary folder {base} is not a plain path")
        missing = base / "skills-canonical" / "missing" / "SKILL.md"
        other = base / "skills-canonical" / "other" / "SKILL.md"
        unread = self.refusals(_in_folder("missing"))
        moved = self.refusals(_in_folder("other"), plant="other")
        for action in ("review", "apply"):
            with self.subTest(action):
                self.assertIn(f"its canonical target {missing} {NO_SUCH_FILE}", unread[action])
                self.assertIn(f"{action} {_ENV['cid']}: {other} now hashes", moved[action])


class NamedPathTests(unittest.TestCase):
    def test_a_plain_path_is_itself(self):
        for path in ("/a/b-c/d_e.f/~x+y/SKILL.md", Path("/a/b/SKILL.md"), "a" * 512):
            with self.subTest(path=path):
                self.assertEqual(named_path(path), str(path))

    def test_a_path_one_character_too_long_is_quoted_and_cut(self):
        # The 512 characters the acceptance map states, and one more.
        named = named_path("a" * 513)
        # Cut at the same length, not at the shorter one a quoted value is cut at.
        self.assertTrue(named.startswith("'" + "a" * 511 + "... ["))
        self.assertIn("more chars not quoted", named)
        self.assertLess(len(named), NAMED_PATH_CHARS + 60)

    def test_a_credential_across_the_cut_is_masked_before_the_cut(self):
        # Masked first, then cut: cut first, the token's first characters would
        # be printed as text the masking no longer recognises.
        quoted = "'/" + "a" * 480 + "/" + MASKED + "/SKILL.md'"
        named = named_path("/" + "a" * 480 + "/" + TOKEN + "/SKILL.md")
        self.assertEqual(named, quoted[:512] + f"... [{len(quoted) - 512} more chars not quoted]")
        self.assertNotIn(TOKEN[:8], named)

    def test_a_path_that_is_a_credential_and_nothing_else_is_masked(self):
        # Every character of the token reads as a path; only the masking hides it.
        self.assertEqual(named_path(TOKEN), f"'{MASKED}'")
        self.assertEqual(named_path(f"/a/{TOKEN}/SKILL.md"), f"'/a/{MASKED}/SKILL.md'")

    def test_a_path_holding_anything_a_path_need_not_hold_is_quoted(self):
        # The other characters ``str.splitlines`` breaks a line at, too.
        breaks = ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")
        for odd in ("\n", "\r", "\x1b[2K", "\t", " ", "\u00e9", "'", "\\", ":", "$", *breaks):
            with self.subTest(odd=odd):
                path = f"/a/b{odd}c/SKILL.md"
                self.assertEqual(named_path(path), quoted_value(path, limit=NAMED_PATH_CHARS))
                self.assertTrue(named_path(path).startswith(("'", '"')))
                self.assertEqual(len(named_path(path).splitlines()), 1)


class WhyUnreadableTests(unittest.TestCase):
    def test_an_os_error_is_named_by_its_number_and_words_without_its_file_name(self):
        error = FileNotFoundError(2, "No such file or directory", PLANTED)
        self.assertEqual(_why_unreadable(error), "[Errno 2] No such file or directory")

    def test_an_os_error_with_no_number_is_quoted(self):
        reason = _why_unreadable(OSError(PLANTED))
        self.assertEqual(reason, f"'{SPELLED}'")
        # A number with no words for it is quoted too, rather than left as a bare number.
        self.assertEqual(_why_unreadable(OSError(5, "")), "'[Errno 5] '")


if __name__ == "__main__":
    unittest.main()
