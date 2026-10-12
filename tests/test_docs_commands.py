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


class DocumentedRefusalsExist(unittest.TestCase):
    """Every refusal the CLI can print must be explained, and every explained one
    must exist.

    ``cli.main`` prints ``<ClassName>: <message>`` for any ``AsiError``, so the
    class name is the first thing an operator sees and the thing they search the
    docs for. Measured before this class existed, six names the code actually
    raises (``GateFailed``, ``CandidateTampered``, ``ApprovalRequired``,
    ``SelfMutationRefused``, ``ReadOnlyViolation``, ``LockHeld``) appeared nowhere
    in ``docs/OPERATIONS.md``.
    """

    ERRORS = REPO / "src" / "agent_skill_improver" / "errors.py"
    SRC = REPO / "src" / "agent_skill_improver"
    OPERATIONS = REPO / "docs" / "OPERATIONS.md"
    #: Operator-facing: refusals the human-gated paths raise, and the daily lock.
    #: ``RecordDefect``/``TransientFailure`` subclasses are per-record outcomes
    #: that land in quarantine or deferral files, not on stderr, and
    #: ``ConfigError`` is covered by ``asi doctor``.
    ROOTS = ("SafetyRefusal", "LockHeld")
    #: CamelCase code spans in OPERATIONS that are not error classes, each with
    #: the reason, so this cannot quietly become an allowance list for typos.
    NOT_A_CLASS = {
        "PROBLEM": "the word `asi doctor` prints for a failed check",
        "WARN": "the word `asi doctor` prints for a finding that does not fail the check",
    }

    @staticmethod
    def error_classes(source: str) -> dict[str, str | None]:
        """``{class_name: base_name}`` for every class in ``errors.py``."""
        import ast

        out: dict[str, str | None] = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef):
                base = (
                    node.bases[0].id if node.bases and isinstance(node.bases[0], ast.Name) else None
                )
                out[node.name] = base
        return out

    @classmethod
    def operator_facing(cls, classes: dict[str, str | None]) -> set[str]:
        def descends(name: str) -> bool:
            seen: set[str] = set()
            while name and name not in seen:
                if name in cls.ROOTS:
                    return True
                seen.add(name)
                name = classes.get(name) or ""
            return False

        return {name for name in classes if descends(name)}

    @classmethod
    def raised_somewhere(cls, names: set[str]) -> set[str]:
        pattern = re.compile(r"\braise (" + "|".join(sorted(names)) + r")\b")
        found: set[str] = set()
        for path in cls.SRC.rglob("*.py"):
            found.update(pattern.findall(path.read_text(encoding="utf-8")))
        return found

    @staticmethod
    def documented_names(text: str) -> set[str]:
        """Class names a document cites: the leading identifier of a code span."""
        out: set[str] = set()
        for span in re.findall(r"`([^`\n]+)`", text):
            head = re.match(r"[A-Z][A-Za-z0-9]*", span)
            if head and (head.end() == len(span) or span[head.end()] in ":( "):
                out.add(head.group(0))
        return out

    def test_every_raised_operator_facing_refusal_is_documented(self) -> None:
        classes = self.error_classes(self.ERRORS.read_text(encoding="utf-8"))
        facing = self.operator_facing(classes)
        raised = self.raised_somewhere(facing)
        self.assertGreaterEqual(len(raised), 8, "the inventory must not be vacuous")
        documented = self.documented_names(self.OPERATIONS.read_text(encoding="utf-8"))
        self.assertEqual(
            sorted(raised - documented), [], "raised by the code, unexplained in OPERATIONS"
        )

    def test_every_documented_refusal_name_is_a_real_error_class(self) -> None:
        classes = self.error_classes(self.ERRORS.read_text(encoding="utf-8"))
        documented = self.documented_names(self.OPERATIONS.read_text(encoding="utf-8"))
        # Every CamelCase code span is checked, not only the refusal-looking ones:
        # a suffix filter would let an invented name like ``GateFailure`` through.
        self.assertGreaterEqual(len(documented), 10, "the extractor found almost nothing")
        unknown = sorted(documented - set(classes) - set(self.NOT_A_CLASS))
        self.assertEqual(unknown, [], "named in OPERATIONS, absent from errors.py")
        for name, reason in self.NOT_A_CLASS.items():
            self.assertNotIn(name, classes, f"{name} is a real class, drop the exemption")
            self.assertTrue(reason.strip(), name)

    def test_the_lock_path_in_the_docs_is_the_one_the_code_uses(self) -> None:
        daily = (self.SRC / "app" / "daily.py").read_text(encoding="utf-8")
        self.assertIn('state.dir("locks") / "daily.lock"', daily)
        ops = self.OPERATIONS.read_text(encoding="utf-8")
        self.assertIn("`state/locks/daily.lock`", ops)
        self.assertNotIn("`state/lock`", ops, "the old, wrong lock path")

    def test_controls_for_the_extractors(self) -> None:
        classes = self.error_classes(
            "class AsiError(Exception):\n    pass\n"
            "class SafetyRefusal(AsiError):\n    pass\n"
            "class Deep(SafetyRefusal):\n    pass\n"
            "class Other(AsiError):\n    pass\n"
            "class LockHeld(AsiError):\n    pass\n"
        )
        self.assertEqual(self.operator_facing(classes), {"SafetyRefusal", "Deep", "LockHeld"})
        names = self.documented_names(
            "see `GateFailed` and `StateUnreadable: refusing...` and `LockHeld (daily)`; "
            "not `asi apply`, not `SKILL.md`, not `--commit`, not `max_open_candidates`"
        )
        self.assertEqual(names, {"GateFailed", "StateUnreadable", "LockHeld"})


class DocumentedStateTreeIsComplete(unittest.TestCase):
    """Every ``state/<dir>/`` the docs name must exist, and every directory that
    can hold transcript-derived content must have a row in the privacy asset
    table.

    Measured before this class existed, the asset table in
    ``docs/PRIVACY_AND_THREAT_MODEL.md`` named four of the thirteen state
    directories. Candidates (improver-written prose about a transcript),
    quarantine, refusal and deferral records (reasons that quote a role's stderr
    tail or its unparseable output) and receipts (a ``diff_preview`` per
    candidate) were not in it, so a reader deciding what is safe to paste into a
    bug report had no row to consult.
    """

    DOCS = tuple(sorted((REPO / "docs").glob("*.md"))) + (REPO / "README.md",)
    PRIVACY = REPO / "docs" / "PRIVACY_AND_THREAT_MODEL.md"
    PATH_RE = re.compile(r"`state/([a-z_]+)/")
    #: Directories whose contents are bookkeeping, never transcript-derived, each
    #: with the reason, so this cannot quietly become an allowance list. A day
    #: checkpoint is NOT one: it copies each record's ``reason`` string.
    NO_ASSET_ROW = {
        "locks": "the daily lock file, whose only content is the owner's pid",
    }

    @staticmethod
    def subdirs() -> tuple[str, ...]:
        from agent_skill_improver.app.state import SUBDIRS

        return tuple(SUBDIRS)

    def documented_paths(self, path: Path) -> set[str]:
        return set(self.PATH_RE.findall(path.read_text(encoding="utf-8")))

    def test_every_documented_state_directory_exists(self):
        known = set(self.subdirs())
        for doc in self.DOCS:
            with self.subTest(doc=doc.name):
                self.assertLessEqual(self.documented_paths(doc), known)

    def test_every_content_bearing_directory_has_a_privacy_row(self):
        expected = set(self.subdirs()) - set(self.NO_ASSET_ROW)
        self.assertLessEqual(set(self.NO_ASSET_ROW), set(self.subdirs()), "stale exemption")
        text = self.PRIVACY.read_text(encoding="utf-8")
        asset_table = text.split("## Assets", 1)[1].split("\n## ", 1)[0]
        self.assertIn("| Asset |", asset_table, "the section slice must be the table")
        self.assertEqual(expected - set(self.PATH_RE.findall(asset_table)), set())

    def test_the_extractor_sees_the_known_rows(self):
        """The control: a regex that found nothing would pass the subset test."""
        found = self.documented_paths(self.PRIVACY)
        self.assertTrue({"raw", "candidates", "quarantine", "receipts"} <= found, found)
        self.assertNotIn("records", found, "the directory that never existed")


class ExitCodesTests(unittest.TestCase):
    """Exit codes are the scheduler's only view of a run, so each one must mean
    one thing.

    Measured before this class existed, an exception no command anticipated
    (a ``PermissionError`` on the state tree, a bug) left ``main`` with a Python
    traceback and exit status 1, which the exit-code table defines as "the
    command ran and the answer is negative". A scheduler reading 1 as "doctor
    said not ready" would file a crash under the wrong heading.
    """

    OPERATIONS = REPO / "docs" / "OPERATIONS.md"
    ROW = re.compile(r"^\| (\d+) \| ", re.MULTILINE)

    def declared_codes(self) -> dict[str, int]:
        return {name: getattr(cli, name) for name in dir(cli) if name.startswith("RC_")}

    def test_every_declared_code_is_distinct(self):
        codes = self.declared_codes()
        self.assertEqual(len(set(codes.values())), len(codes), codes)

    def test_the_exit_code_table_documents_every_declared_code_and_argparse(self):
        table = self.OPERATIONS.read_text(encoding="utf-8").split("## Exit codes", 1)[1]
        table = table.split("\n## ", 1)[0]
        documented = {int(m) for m in self.ROW.findall(table)}
        self.assertEqual(documented, set(self.declared_codes().values()) | {2})

    def test_an_unanticipated_exception_is_not_reported_as_a_negative_answer(self):
        import contextlib
        import io
        from unittest import mock

        err = io.StringIO()
        with mock.patch.object(cli, "cmd_doctor", side_effect=RuntimeError("boom")):
            with contextlib.redirect_stderr(err):
                rc = cli.main(["doctor"])
        self.assertEqual(rc, cli.RC_INTERNAL_ERROR)
        self.assertNotEqual(rc, cli.RC_NEGATIVE)
        self.assertIn("internal error:", err.getvalue(), "the line a scheduler log will show")
        self.assertIn("RuntimeError: boom", err.getvalue(), "the cause must still be visible")
        self.assertIn("Traceback", err.getvalue(), "the traceback is the bug report")

    def test_the_control_a_refusal_still_exits_3_without_a_traceback(self):
        import contextlib
        import io
        from unittest import mock

        from agent_skill_improver.errors import SafetyRefusal

        err = io.StringIO()
        with mock.patch.object(cli, "cmd_doctor", side_effect=SafetyRefusal("declined")):
            with contextlib.redirect_stderr(err):
                rc = cli.main(["doctor"])
        self.assertEqual(rc, cli.RC_REFUSED)
        self.assertEqual(err.getvalue(), "SafetyRefusal: declined\n")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
