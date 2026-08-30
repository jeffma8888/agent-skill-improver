"""Every ``asi`` command line printed in the docs must really parse.

The apply flag in ``README.md`` was wrong for the whole first draft of this
repository: the docs said ``--i-have-reviewed-this-change`` while the CLI has
only ever accepted ``--i-reviewed-this``.  Nothing failed, because prose is not
executed.  A documented command line is a *derived* value whose source of truth
is the argument parser, so it decays silently every time the parser is edited.
These tests re-derive it.

Only the shape is checked, never the effect: no command is ever run.
"""

from __future__ import annotations

import argparse
import re
import shlex
import unittest
from pathlib import Path

from agent_skill_improver import cli

REPO = Path(__file__).resolve().parents[1]

#: Files whose fenced blocks and inline code are treated as claims about the CLI.
DOC_FILES = (
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/OPERATIONS.md",
    "docs/PRIVACY_AND_THREAT_MODEL.md",
    "docs/RUBRICS.md",
    "docs/SCHEDULING.md",
)

_FENCE = re.compile(r"^\s*```")
_INLINE = re.compile(r"`([^`\n]+)`")

#: Placeholders a reader is expected to substitute.  A command line containing
#: one of these still has its subcommand and flag names checked; only the value
#: is treated as opaque.
_PLACEHOLDER = re.compile(r"^(<[^>]+>|CAND-\.\.\.|\.\.\.|\$[A-Z_]+|[A-Z][A-Z0-9_-]*)$")


def _fenced_command_lines(text: str) -> list[str]:
    """Command lines inside fenced blocks, with shell continuations joined."""
    lines = text.splitlines()
    out: list[str] = []
    inside = False
    i = 0
    while i < len(lines):
        raw = lines[i]
        if _FENCE.match(raw):
            inside = not inside
            i += 1
            continue
        stripped = raw.strip()
        if inside and (stripped == "asi" or stripped.startswith("asi ")):
            joined = stripped
            while joined.endswith("\\") and i + 1 < len(lines):
                i += 1
                joined = joined[:-1].strip() + " " + lines[i].strip()
            out.append(joined)
        i += 1
    return out


def _inline_command_lines(text: str) -> list[str]:
    """Inline code spans that are *entirely* an ``asi`` invocation.

    A span is only a claim about the CLI when the whole span is the command.
    A sentence such as ``asi rubrics`` prints ... puts prose outside the span,
    so taking the span alone is what keeps the collector from reading English
    as arguments.
    """
    out: list[str] = []
    inside = False
    for raw in text.splitlines():
        if _FENCE.match(raw):
            inside = not inside
            continue
        if inside:
            continue
        for span in _INLINE.findall(raw):
            span = span.strip()
            if span == "asi" or span.startswith("asi "):
                out.append(span)
    return out


def _command_lines(text: str) -> list[str]:
    return _fenced_command_lines(text) + _inline_command_lines(text)


def _choices_map(action: argparse.Action) -> dict[str, argparse.ArgumentParser]:
    """Subparsers reachable from an action.

    ``action.choices`` is a mapping for a subparsers action and a plain tuple
    for a positional with a fixed value list (``scheduler-template`` has one),
    so it cannot be walked blindly.
    """
    choices = getattr(action, "choices", None)
    if not isinstance(choices, dict):
        return {}
    return {k: v for k, v in choices.items() if isinstance(v, argparse.ArgumentParser)}


class _Refused(Exception):
    pass


class _StrictParser(argparse.ArgumentParser):
    """argparse exits the process on error; tests need an exception instead."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise _Refused(message)

    def exit(self, status: int = 0, message: str | None = None) -> None:  # type: ignore[override]
        if status:
            raise _Refused(message or f"exit {status}")
        raise SystemExit(0)


def _strict_parser() -> argparse.ArgumentParser:
    parser = cli.build_parser()
    parser.__class__ = _StrictParser
    for action in parser._actions:
        for sub in _choices_map(action).values():
            sub.__class__ = _StrictParser
    return parser


def _sanitise(argv: list[str]) -> list[str]:
    """Replace documentation placeholders with syntactically valid stand-ins."""
    out: list[str] = []
    for token in argv:
        if token.startswith("-") or not _PLACEHOLDER.match(token):
            out.append(token)
        else:
            out.append("PLACEHOLDER")
    return out


def _argv(line: str) -> list[str]:
    """Argument vector of a documented line, minus the program name.

    ``comments=True`` is what drops the trailing ``# what this does`` that the
    docs put on several example lines.  Without it the comment words arrive as
    positional arguments and every annotated example reads as invalid.
    """
    tokens = shlex.split(line, comments=True)
    assert tokens and tokens[0] == "asi", line
    return _sanitise(tokens[1:])


def _parse(parser: argparse.ArgumentParser, line: str) -> argparse.Namespace | None:
    """Strictly parse one documented line; ``None`` means ``--help``/``--version``."""
    argv = _argv(line)
    if not argv:
        return None
    try:
        return parser.parse_args(argv)
    except SystemExit:
        return None


def _subparsers() -> dict[str, argparse.ArgumentParser]:
    out: dict[str, argparse.ArgumentParser] = {}
    for action in cli.build_parser()._actions:
        out.update(_choices_map(action))
    return out


def _check_reference(span: str, subparsers: dict[str, argparse.ArgumentParser]) -> None:
    """Check an inline reference: the subcommand and any flags must be real.

    An inline span is a *reference* to a command, not a copy-pasteable line:
    ``asi rollback`` in a sentence names the command without naming its
    argument.  Requiring a full parse there would force the prose to spell out
    a candidate id every time it mentions the command, so completeness is
    checked only for fenced blocks.  Names are still checked here, which is the
    half that actually decays.
    """
    argv = _argv(span)
    if not argv:
        return
    globals_only = _global_option_strings()
    name = next((t for t in argv if not t.startswith("-")), None)
    flags = [t.split("=", 1)[0] for t in argv if t.startswith("-")]
    if name is None:
        for flag in flags:
            assert flag in globals_only, f"{span}: unknown global flag {flag}"
        return
    assert name in subparsers, f"{span}: unknown subcommand {name!r}"
    allowed = set(globals_only)
    for action in subparsers[name]._actions:
        allowed.update(action.option_strings)
    for flag in flags:
        assert flag in allowed, f"{span}: {name} has no flag {flag}"


def _global_option_strings() -> set[str]:
    out: set[str] = set()
    for action in cli.build_parser()._actions:
        out.update(action.option_strings)
    return out


class DocumentedCommandLinesParse(unittest.TestCase):
    def test_a_the_collector_finds_real_command_lines(self) -> None:
        # A collector that silently matches nothing would make every other test
        # in this module vacuously green, so the population is asserted first.
        fenced = sum(
            len(_fenced_command_lines((REPO / rel).read_text(encoding="utf-8")))
            for rel in DOC_FILES
        )
        inline = sum(
            len(_inline_command_lines((REPO / rel).read_text(encoding="utf-8")))
            for rel in DOC_FILES
        )
        self.assertGreater(fenced, 8, "no fenced asi command lines found")
        self.assertGreater(inline, 3, "no inline asi command spans found")

    def test_b_every_fenced_command_line_parses_completely(self) -> None:
        # Fenced lines are copy-pasteable, so they are held to a full parse:
        # unknown flag, unknown subcommand or missing required argument all fail.
        parser = _strict_parser()
        checked = 0
        for rel in DOC_FILES:
            for line in _fenced_command_lines((REPO / rel).read_text(encoding="utf-8")):
                with self.subTest(doc=rel, line=line):
                    try:
                        _parse(parser, line)
                    except _Refused as exc:
                        self.fail(f"{rel} documents an invalid command: {line!r} ({exc})")
                checked += 1
        self.assertGreater(checked, 8)

    def test_b2_every_inline_reference_names_real_things(self) -> None:
        subparsers = _subparsers()
        checked = 0
        for rel in DOC_FILES:
            for span in _inline_command_lines((REPO / rel).read_text(encoding="utf-8")):
                with self.subTest(doc=rel, span=span):
                    try:
                        _check_reference(span, subparsers)
                    except AssertionError as exc:
                        self.fail(f"{rel} names something the CLI does not have: {exc}")
                checked += 1
        self.assertGreater(checked, 3)

    def test_c_every_documented_subcommand_exists(self) -> None:
        parser = _strict_parser()
        known: set[str] = set()
        for action in parser._actions:
            known |= set(_choices_map(action))
        self.assertIn("apply", known)
        for rel in DOC_FILES:
            for line in _fenced_command_lines((REPO / rel).read_text(encoding="utf-8")):
                ns = _parse(parser, line)
                if ns is None:
                    continue
                with self.subTest(doc=rel, line=line):
                    # `command` is the parser's own dest, so global flags placed
                    # before the subcommand cannot fool this the way taking the
                    # first positional token did.
                    self.assertIn(getattr(ns, "command", None), known)

    def test_d_the_check_rejects_a_planted_bad_flag(self) -> None:
        # Two-sided: prove the parse check would actually fail on a wrong flag,
        # rather than passing because nothing is ever refused.
        parser = _strict_parser()
        with self.assertRaises(_Refused):
            _parse(parser, "asi apply CAND-1 --i-have-reviewed-" + "this-change")

    def test_e_the_check_rejects_a_planted_bad_subcommand(self) -> None:
        parser = _strict_parser()
        with self.assertRaises(_Refused):
            _parse(parser, "asi promote CAND-1")

    def test_f_the_collector_joins_a_continued_command(self) -> None:
        text = (
            "```bash\nasi apply CAND-... \\\n    --approve TOKEN \\\n    --i-reviewed-this\n```\n"
        )
        lines = _fenced_command_lines(text)
        self.assertEqual(len(lines), 1)
        self.assertIn("--approve", lines[0])
        self.assertIn("--i-reviewed-this", lines[0])

    def test_g_prose_after_an_inline_span_is_not_read_as_arguments(self) -> None:
        # The first version of this collector stripped backticks from the whole
        # line, so "`asi rubrics` prints what is loaded" became the argument
        # vector [rubrics`, prints, what, is, loaded] and failed.
        text = "`asi rubrics` prints what is loaded, with versions and hashes.\n"
        self.assertEqual(_inline_command_lines(text), ["asi rubrics"])

    def test_h_fenced_and_inline_collectors_do_not_double_count(self) -> None:
        text = "`asi doctor` first.\n\n```bash\nasi daily\n```\n"
        self.assertEqual(_inline_command_lines(text), ["asi doctor"])
        self.assertEqual(_fenced_command_lines(text), ["asi daily"])

    def test_i_global_flags_before_the_subcommand_are_understood(self) -> None:
        parser = _strict_parser()
        ns = _parse(parser, "asi --config config/example.config.toml scheduler-template launchd")
        assert ns is not None
        self.assertEqual(ns.command, "scheduler-template")
        self.assertEqual(ns.flavor, "launchd")


class InlineReferenceCheckerIsTwoSided(unittest.TestCase):
    """The lenient inline checker must still refuse wrong names."""

    def setUp(self) -> None:
        self.subparsers = _subparsers()

    def test_a_a_bare_command_reference_is_accepted(self) -> None:
        _check_reference("asi rollback", self.subparsers)  # no candidate id: fine

    def test_b_an_unknown_subcommand_is_refused(self) -> None:
        with self.assertRaises(AssertionError):
            _check_reference("asi promote", self.subparsers)

    def test_c_an_unknown_flag_is_refused(self) -> None:
        with self.assertRaises(AssertionError):
            _check_reference("asi daily --publish", self.subparsers)

    def test_d_a_flag_from_the_wrong_subcommand_is_refused(self) -> None:
        # --commit is real, but it belongs to `daily`, not to `apply`.
        with self.assertRaises(AssertionError):
            _check_reference("asi apply --commit", self.subparsers)

    def test_e_a_global_flag_is_accepted_anywhere(self) -> None:
        _check_reference("asi daily --json", self.subparsers)
        _check_reference("asi --json", self.subparsers)

    def test_f_a_trailing_shell_comment_is_not_an_argument(self) -> None:
        self.assertEqual(_argv("asi candidates   # the queue"), ["candidates"])


class DocumentedFlagsExist(unittest.TestCase):
    """A flag named anywhere in the docs must exist on some ``asi`` parser."""

    FLAG = re.compile(r"`(--[a-z][a-z0-9-]*)`")

    #: Flags belonging to other programs quoted in the docs, not to ``asi``.
    #:
    #: Each entry names its owner, because an unexplained exemption set is how a
    #: real defect gets parked: a bare list of strings gives a later reader no way
    #: to tell a legitimately foreign flag from one that was added to silence this
    #: test.  ``test_d`` asserts no entry here is also a real ``asi`` flag, which
    #: is the half that stops the set being used to hide one.
    FOREIGN = {
        "--check": "examples/build_fixtures.py",
        "--no-history": "tools/leakscan.py",
        "--selftest": "tools/leakscan.py",
        "--repo": "tools/leakscan.py",
        "--force": "git",
        "--fetch-depth": "actions/checkout",
        "--unshare-net": "bwrap (bubblewrap)",
        "--ro-bind": "bwrap (bubblewrap)",
        "--die-with-parent": "bwrap (bubblewrap)",
    }

    def _all_option_strings(self) -> set[str]:
        out: set[str] = set()

        def collect(p: argparse.ArgumentParser) -> None:
            for action in p._actions:
                out.update(action.option_strings)
                for sub in _choices_map(action).values():
                    collect(sub)

        collect(cli.build_parser())
        return out

    def test_a_inline_flags_in_docs_are_real(self) -> None:
        known = self._all_option_strings()
        seen: set[str] = set()
        for rel in DOC_FILES:
            text = (REPO / rel).read_text(encoding="utf-8")
            for flag in sorted(set(self.FLAG.findall(text))):
                if flag in self.FOREIGN:
                    continue
                seen.add(flag)
                with self.subTest(doc=rel, flag=flag):
                    self.assertIn(flag, known, f"{rel} names a flag the CLI does not have")
        # A named floor rather than a count: these two are the flags the safety
        # story depends on, so the docs must keep naming them and a collector
        # that stopped finding anything would fail here rather than pass empty.
        self.assertLessEqual({"--commit", "--i-reviewed-this"}, seen)

    def test_b_the_flag_inventory_is_two_sided(self) -> None:
        known = self._all_option_strings()
        self.assertIn("--i-reviewed-this", known)
        self.assertNotIn("--i-have-reviewed-" + "this-change", known)

    def test_c_the_previously_wrong_flag_is_gone_from_every_doc(self) -> None:
        bad = "--i-have-reviewed-" + "this-change"
        for rel in DOC_FILES:
            self.assertNotIn(bad, (REPO / rel).read_text(encoding="utf-8"), rel)

    def test_d_no_exemption_hides_a_real_cli_flag(self) -> None:
        # The exemption set exists so the docs may quote another program's flags.
        # It must never be able to excuse an `asi` flag: if a flag really is on
        # the parser, the parser is the source of truth and no exemption applies.
        known = self._all_option_strings()
        for flag, owner in self.FOREIGN.items():
            with self.subTest(flag=flag, owner=owner):
                self.assertNotIn(flag, known, f"{flag} is a real asi flag, not {owner}'s")

    def test_e_every_exemption_names_its_owner(self) -> None:
        # A blob of strings decays into an unreadable allowance list; a mapping
        # forces the next person adding one to say whose flag it is.
        for flag, owner in self.FOREIGN.items():
            with self.subTest(flag=flag):
                self.assertTrue(flag.startswith("--"), flag)
                self.assertTrue(owner.strip(), f"{flag} has no named owner")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
