"""Command line interface.

Command surface, and which of them can change anything on disk:

    doctor              read-only. Validates config and reports what is wired.
    skills              read-only. Skill inventory, canonical vs mirror, discovery warnings.
    rubrics             read-only.
    discover            read-only. Lists thread ids the source would ingest.
    daily               writes only under the gitignored state directory.
                        DRY RUN BY DEFAULT: nothing under state/ is written; --commit persists.
    candidates          read-only.
    review              read-only. Prints the argument, the diff, and the approval token.
    apply               writes a candidate (approval token + acknowledgement). Review-gated.
    rollback            restores a verified backup; no token or acknowledgement.
    scheduler-template  prints a launchd or cron template. Installs nothing.
    selftest            re-checks the structural safety invariants on this checkout.

``apply`` and ``rollback`` refuse while the automation marker is active. The daily
driver sets it only inside its own process; the scheduler templates set it for their
whole job command (cron also marks later lines on cronie, Vixie and OpenBSD cron).
Apply needs a content-derived token and acknowledgement but does not authenticate
its caller: a separate unmarked process can still apply. Do not schedule apply.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shlex
import signal
import sys
import threading
import traceback
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, NoReturn, Sequence

from . import __version__
from .clock import SystemClock, day_key, iso
from .config import SOURCE_FIXTURES, Config, load_config, package_dir
from .domain.approval import HumanApproval, automation_active
from .domain.candidate import (
    BIDI_CONTROLS,
    HIDDEN_CONTROLS,
    STATUS_VALIDATED,
    one_line_label,
    validate_run_day,
)
from .domain.skills import BOM
from .domain.validation import (
    UNSEEN_MEASURED,
    UNSEEN_ONE_CONVERSATION,
    UNSEEN_SCOPE,
    UNSEEN_SHORT_OF_K,
    UNSEEN_TOO_FEW,
)
from .domain.isolation import (
    MODE_NONE,
    SOURCE_CLASS_PRIVATE,
    SOURCE_CLASS_SYNTHETIC,
    credential_bearing_env_values,
    decide as decide_isolation,
    denied_env_names,
    source_class,
)
from .errors import AsiError, BlockedRunUnreviewed, ConfigError
from .jsonio import dumps
from .adapters.bounded_process import children_reaped_by_the_kernel
from .adapters.skill_fs import root_presence
from .domain.discovery import MAX_THREAD_ID_LENGTH, shown_passed_over
from .domain.redaction import (
    LISTED_PATH_CHARS,
    LISTED_SKILL_ID_CHARS,
    listed_value,
    named_path,
    quoted_value,
    redact,
)
from .app.state import DIR_MODE, OPEN_STATUSES, SET_ASIDE_KINDS, STATE_FILE_MODE, StateStore
from .app.wiring import build_runner, build_skill_store, build_source, build_state

ENV_CONFIG = "ASI_CONFIG"

#: Exit codes. 2 is argparse's usage error and is not returned by any command.
RC_OK = 0
RC_NEGATIVE = 1
RC_REFUSED = 3
RC_MISSING_FILE = 4
#: The daily run saw the live skill tree change under it. Not a negative answer
#: and not a refusal: an invariant may be broken, and a scheduler must not treat
#: it as "try again tomorrow".
RC_BLOCKED = 5
#: An exception no command anticipated. Kept apart from 1 so a scheduler cannot
#: read a crash as a negative answer; the traceback stays on stderr. The value
#: is sysexits.h EX_SOFTWARE.
RC_INTERNAL_ERROR = 70
#: Signals that stop a command the way Ctrl-C does: each is raised in the main
#: thread as :class:`Stopped`, so every ``finally`` runs (the model or source
#: command is ended with the run, the lock released, a half-written file
#: removed). A stopped command exits 128 plus the signal number, the shell's
#: spelling of death by that signal (POSIX fixes SIGHUP at 1, SIGTERM at 15).
STOP_SIGNALS = ("SIGTERM", "SIGHUP")
RC_STOPPED_BY_SIGHUP = 128 + 1
RC_STOPPED_BY_SIGTERM = 128 + 15


#: What a stopped ``apply`` or ``rollback`` adds to the stop: each is finished
#: by the same command run again, which records a write the stopped run made
#: (``apply_candidate`` through the manifest written before the skill, rollback
#: by finding the previous text already in place).
FINISH_BY_RUNNING_AGAIN = {
    "apply": "run the same apply again to finish it: a skill write this run made and did "
    "not record is recorded then; if this run had recorded it, the apply is refused as "
    "already `applied`, with nothing left to do\n",
    "rollback": "run the same rollback again to finish it: a restore this run made and did "
    "not record is recorded then\n",
}


class Stopped(BaseException):
    """A stop signal arrived. Not an ``Exception``, so no handler that isolates
    one conversation's or one skill's failure can catch it and carry on."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


@contextmanager
def _stop_signals_unwind(*, exiting: bool = False) -> Iterator[None]:
    """Turn each of ``STOP_SIGNALS`` into :class:`Stopped` while a command runs.

    A model or source command runs in a session of its own
    (``start_new_session``), so a signal sent to the run's process does not
    reach it, and ``run_bounded`` ends it in a ``finally``. Measured before:
    SIGTERM, which a scheduler sends to stop a job, or SIGHUP, which a closed
    terminal sends, during a model call killed the run where it stood (exit
    -15, -1, no message) and left the model command running, re-parented to
    process 1 and past its own timeout, until it wrote its answer to an output
    nobody read. Only a signal still at its default action is taken over, so
    one the caller ignores (``nohup`` ignores SIGHUP) stays ignored. A second
    signal while the first unwinds is let pass, so the cleanup it set off is
    not cut short. Off the main thread nothing is installed: Python runs a
    handler on that thread only, and refuses to install one from any other.

    Ctrl-C stays Python's own: each press raises ``KeyboardInterrupt``. From the
    first press a stop signal is let pass too, so it cannot cut short the
    cleanup Ctrl-C set off. With ``exiting`` (the process's own command,
    :func:`entry`) the stop signals are left ignored after a stop or a Ctrl-C
    rather than put back, so one more arriving while the process exits cannot
    end it by the signal and change the status a scheduler reads.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    stopping = False

    def stop(signum: int, frame: object) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        raise Stopped(signum)

    def interrupt(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True
        raise KeyboardInterrupt

    previous: dict[int, Any] = {}
    for name in STOP_SIGNALS:
        signum = getattr(signal, name, None)
        if signum is not None and signal.getsignal(signum) is signal.SIG_DFL:
            previous[signum] = signal.signal(signum, stop)
    if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        previous[signal.SIGINT] = signal.signal(signal.SIGINT, interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            if exiting and stopping and signum != signal.SIGINT:
                # SIG_IGN rather than a handler of ours: Python puts its own
                # handlers back to the default as it shuts down.
                handler = signal.SIG_IGN
            signal.signal(signum, handler)


BENIGN_RUN_STATUSES = ("ok", "stood-down", "partial")
#: ``doctor`` warns when the newest daily receipt is older than this many
#: whole days: a daily schedule that fires leaves a receipt every day, so
#: a gap this long means the scheduler is not running the job.
STALE_RUN_DAYS = 2


def exit_code_for(run_status: str) -> int:
    """Map a daily receipt status to the process exit code."""
    if run_status in BENIGN_RUN_STATUSES:
        return RC_OK
    if run_status == "blocked":
        return RC_BLOCKED
    return RC_NEGATIVE


def _require_isolation(config: Config, state: StateStore) -> None:
    """Refuse before any command reads a private source.

    Built here rather than passed in, so a command cannot accidentally check a
    different runner from the one it is about to use.
    """
    runner = build_runner(config, state)
    decide_isolation(
        source_kind=config.source.kind,
        runner_isolation=runner.isolation(),
        roles=sorted(config.model_roles),
    ).require()


def repo_root() -> Path:
    return package_dir().parent.parent


def default_config_path() -> Path:
    from_env = os.environ.get(ENV_CONFIG)
    if from_env:
        return Path(from_env).expanduser()
    local = repo_root() / "config" / "local.toml"
    if local.is_file():
        return local
    return repo_root() / "config" / "example.config.toml"


def _load(args: argparse.Namespace) -> Config:
    path = Path(args.config).expanduser() if args.config else default_config_path()
    return load_config(path)


#: The two control characters a terminal is meant to obey in prose.
_SHOWN_AS_IS = frozenset("\n\t")


def one_line(text: str) -> str:
    r"""A listing line made of untrusted text, kept to one line: a line break
    inside it is spelled, not obeyed.

    ``escape_for_terminal`` keeps ``\n`` because the listing's own structure is
    made of them; a skill id, a payload file name or a conversation title is
    not structure, and a newline inside one would otherwise start a forged line
    in the very listing that exists to warn about such files. Applied by
    ``_emit`` to every line it prints. Measured before: an executable named
    ``run\n    warning: ...`` printed as two warnings in ``skills``.
    """
    return one_line_label(text)


def quoted(text: str, *, prefix: str = "  | ") -> list[str]:
    """Model-authored or file-derived prose rendered as a quoted block: every
    line it occupies carries ``prefix``, so none of it can stand at column 0
    where the review's own section headers and its apply instruction live.

    ``asi review`` prints the candidate's hypothesis, intended change, risks
    and rollback note as the model wrote them, line breaks included, because
    that prose is what the reviewer is weighing. Measured before this: a
    hypothesis containing ``\nDIFF\n(empty)\n\nTo promote this change ...
    asi apply ... --approve FORGEDTOKEN`` printed a second, fake DIFF and apply
    section above the real one, indistinguishable by shape. Prose that is not
    the tool's own text is quoted; the headers, the diff and the real apply
    line are the only lines that start at the margin.
    """
    # Split on "\n" alone: a bare "\r" or a Unicode line separator stays in
    # its line and reaches the terminal spelled by ``escape_for_terminal``, so
    # the prose is shown as written and still cannot start a line.
    lines = text.split("\n")
    return [prefix.rstrip() if line == "" else f"{prefix}{line}" for line in lines]


def escape_for_terminal(text: str) -> str:
    """Show every byte a terminal would otherwise obey.

    A terminal executes control characters: ``ESC [2K`` erases the line above,
    a carriage return rewrites the current one, a bidi override reorders what
    follows. The CLI quotes model stderr tails, file names and configured paths
    verbatim in its messages, so any of them can carry such a byte. Here each
    one becomes the Python escape it is (``\\x1b``, ``\\r``, ``\\u202e``), which
    a person reads as a warning rather than a terminal acting on it. Newline
    and tab are kept: they are the structure of the output itself. A lone
    surrogate (how a file name that is not valid UTF-8 reaches Python) is shown
    the same way, because a strict UTF-8 stream refuses to encode it and the
    refusal would turn one odd file name into an internal error.
    """
    out: list[str] = []
    for ch in text:
        if ch in _SHOWN_AS_IS:
            out.append(ch)
        elif (
            ch in BIDI_CONTROLS
            or ch in HIDDEN_CONTROLS
            or ch == BOM
            or unicodedata.category(ch) in ("Cc", "Cs")
        ):
            code = ord(ch)
            if ch == "\r":
                out.append("\\r")
            elif code < 0x100:
                out.append(f"\\x{code:02x}")
            elif code < 0x10000:
                out.append(f"\\u{code:04x}")
            else:
                out.append(f"\\U{code:08x}")
        else:
            out.append(ch)
    return "".join(out)


def reads_as_itself(text: str) -> bool:
    """Whether ``text`` reaches the terminal as written when ``_emit`` prints it:
    no line break that ``one_line`` spells and no character that
    ``escape_for_terminal`` spells.

    A command printed for a person to copy must be that command. When a path
    in it holds such a character, the line shows the spelling, and the copied
    command would name another path. Built from the printer's own two steps,
    so it cannot drift from what the printer spells.
    """
    return escape_for_terminal(one_line(text)) == text


def escape_json_for_terminal(text: str) -> str:
    """JSON text with every character a terminal would obey written as the JSON
    escape it is, so the same bytes are safe to read and unchanged to parse.

    The encoder already escapes the C0 controls, but with ``ensure_ascii`` off it
    leaves a bidi override, an invisible tag character, a C1 control or a lone
    surrogate raw, and an operator does read ``--json`` output in a terminal.
    Such a character can only sit inside a JSON string, where ``\\uXXXX`` is the
    one spelling every parser accepts, so ``json.loads`` of the result is the
    same object. Newlines are the encoder's own indentation and stay.
    """
    out: list[str] = []
    for ch in text:
        if ch != "\n" and (
            ch in BIDI_CONTROLS
            or ch in HIDDEN_CONTROLS
            or ch == BOM
            or unicodedata.category(ch) in ("Cc", "Cs")
        ):
            code = ord(ch)
            if code >= 0x10000:
                code -= 0x10000
                out.append(f"\\u{0xD800 + (code >> 10):04x}\\u{0xDC00 + (code & 0x3FF):04x}")
            else:
                out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    return "".join(out)


def _say(text: str) -> None:
    """The one place text for a person reaches stdout."""
    sys.stdout.write(escape_for_terminal(text))


def _say_json(payload: Any) -> None:
    """The one place JSON reaches stdout: for programs, and safe for the eyes."""
    sys.stdout.write(escape_json_for_terminal(dumps(payload)))


def _err(text: str) -> None:
    """The one place text for a person reaches stderr."""
    sys.stderr.write(escape_for_terminal(text))


def _emit(payload: Any, *, as_json: bool, lines: Sequence[str] = ()) -> None:
    if as_json:
        _say_json(payload)
    else:
        # Each element is one line by construction; text inside it that came
        # from a file name or a document must not be able to add another.
        for line in lines:
            _say(one_line(line) + "\n")


# -- commands ------------------------------------------------------------
def _age_in_days(stamp: str, now: Any) -> int | None:
    """Whole days from an ISO stamp the tool wrote to ``now``; None if it does not parse.

    Negative when the stamp is ahead of the clock: a future finish time is
    impossible for a receipt this build wrote, so it is reported as such
    rather than clamped to "today" and read as a recent run.
    """
    import datetime as _dt

    try:
        then = _dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
    except ValueError:
        return None
    return int((now - then).total_seconds() // 86400)


def _last_run_text(last_run: Any, age_days: int | None) -> str:
    """One line answering the health questions; the tool's words plus fields it wrote."""
    if last_run is None:
        return "none recorded (no daily receipt yet)"
    when = last_run.finished_at or "time unknown"
    if age_days is not None and age_days < 0:
        when += f" ({-age_days} day(s) in the future)"
    elif age_days is not None:
        when += f" ({age_days} day(s) ago)"
    elif last_run.finished_at:
        when += " (time does not parse)"
    mutation = "unknown" if last_run.mutation_detected is None else str(last_run.mutation_detected)
    return one_line(
        f"{last_run.day} {last_run.status}{' (dry run)' if last_run.dry_run else ''} at {when}; "
        f"skill tree mutation={mutation}; isolation={last_run.isolation_mode or 'unknown'}"
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    problems: list[str] = []
    warnings: list[str] = []
    try:
        config = _load(args)
    except ConfigError as exc:
        _err(f"config error: {exc}\n")
        return RC_REFUSED
    # One day for every check below that asks what today's run would do.
    today = day_key(SystemClock().now())
    store = build_skill_store(config)
    missing = [r.id for r in store.missing_roots()]
    if missing:
        warnings.append(f"skill root(s) not present on this machine: {', '.join(missing)}")
    if not store.existing_roots():
        problems.append("no configured skill root is a directory; nothing can be discovered")
    elif not store.known_skill_ids():
        # `daily` stands down before discovery in this state, so say so here
        # rather than let a tree with no document read as ready.
        present = ", ".join(r.id for r in store.existing_roots())
        problems.append(
            f"no skill document was discovered under the present root(s) {present}; `daily` "
            "will stand down with nothing to propose"
        )
    else:
        # The run forms every known skill's longest candidate id before it
        # reads a conversation and refuses the whole run when one would not
        # fit; the same question here, for the day a run without --day uses.
        # Measured before: a 186-character skill directory name, or a 171-digit
        # [caps].max_candidates_per_run, left this report at ready while that
        # run exited 3. A longer --day is refused by the run alone.
        from .app.daily import candidate_naming_problem

        naming = candidate_naming_problem(
            today, store.known_skill_ids(), config.caps.max_candidates_per_run
        )
        if naming:
            problems.append(
                f"today's run would be refused before any conversation is read: {naming}"
            )
    for role in ("scorer", "improver", "judge"):
        if role not in config.model_roles:
            problems.append(f"no [model.{role}] command configured")
    # ``asi`` puts SIGCHLD back to its default as it starts; a caller of
    # ``main`` keeps what it has, and while that is an ignore no role or source
    # command is started (``ExitUnreadable``). Measured before (3.11 and 3.13):
    # ``main(["doctor"])`` with SIGCHLD ignored said ready, and ``daily`` called
    # the same way then scored none of 5 conversations and deferred all 5.
    if children_reaped_by_the_kernel():
        problems.append(
            "SIGCHLD is ignored in this process, so no role or source command would be "
            "started (a child's exit status could not be read) and the run would score "
            "nothing; `asi` puts SIGCHLD back to its default as it starts, and a program "
            "that calls `main` must do the same first"
        )
    # The same rules minimal_env applies at the first model call, applied here
    # so the operator learns about a refused role environment from the report
    # rather than from the first scheduled run. Names and shapes only; a value
    # that looks like a credential is never printed.
    for role, command in sorted(config.model_roles.items()):
        role_env = command.env or {}
        denied = denied_env_names(role_env)
        if denied:
            problems.append(
                f"[model.{role}].env carries a name matching a credential, token or proxy "
                f"pattern ({', '.join(denied)}); the run would refuse it at the first model call"
            )
        # A denied name is already reported above; the shape check is for the
        # rename that slips past the name denylist, as in minimal_env.
        for name, shape in credential_bearing_env_values(role_env):
            if name in denied:
                continue
            problems.append(
                f"[model.{role}].env value {name} is shaped like a credential ({shape}); "
                "the run would refuse it at the first model call"
            )
    independence = config.evaluator_independence()
    if not independence["judge_independent"]:
        warnings.append(
            "judge and improver are not distinct commands: the gate will refuse candidates while "
            "[gate].require_independent_judge is true, because a self-graded preference is not "
            "independent evidence"
        )
    # The same rule ``resolve_groups`` applies: a root marked canonical and
    # editable is the canonical one; failing that, the highest-precedence
    # editable root is. Only with no editable root at all are proposals refused.
    # Measured before: the warning said "proposals will be refused" for a config
    # whose editable root was simply not marked canonical, and `daily` proposed.
    editable_roots = sorted(
        (r for r in config.skill_roots if r.editable), key=lambda r: (r.precedence, r.id)
    )
    if not editable_roots:
        warnings.append("no skill root is editable; proposals will be refused")
    elif not any(r.canonical for r in editable_roots):
        warnings.append(
            f"no editable skill root is marked canonical; the highest-precedence editable "
            f"root {editable_roots[0].id!r} is treated as the canonical one"
        )
    snapshots = store.snapshots()
    broken = [s for s in snapshots if not s.loads]
    # A document discovery saw and stepped over is a finding, not a silence:
    # a tree whose only SKILL.md is a dangling link otherwise reads as a tree
    # with no skills, and this report said "ready" over it.
    passed_over = store.passed_over()
    for document in passed_over:
        warnings.append(
            f"skill document passed over in root {document.root_id}: {document.path} is "
            f"{document.reason}"
        )
    # Two documents claiming one skill in the root that would be edited: the
    # resolver refuses to pick by walk order, so `daily` quarantines the skill
    # at propose until one copy remains. Measured before: the newcomer was
    # proposed against and this report said ready.
    for group in store.groups():
        if group.ambiguous_copies:
            listed = ", ".join(str(c.path) for c in group.ambiguous_copies)
            tied_roots = sorted({c.root_id for c in group.ambiguous_copies})
            if len(tied_roots) > 1:
                # Roots of one tier at one precedence: the resolver refuses to
                # pick by root id, so the setting, not a file, is what to change.
                names = ", ".join(repr(r) for r in tied_roots)
                warnings.append(
                    f"skill {group.skill_id!r} has a copy in each of roots {names}, which tie on "
                    f"precedence ({listed}); `daily` refuses to propose for it until the roots "
                    "have distinct precedences or all but one is marked canonical = false or "
                    "editable = false"
                )
                continue
            warnings.append(
                f"skill {group.skill_id!r} has {len(group.ambiguous_copies)} copies in editable "
                f"root {group.ambiguous_copies[0].root_id!r} ({listed}); `daily` refuses to "
                "propose for it until one copy remains"
            )
    # A directory inside a skill's directory that the inventory could not list,
    # or listed but could not examine an entry of: what it holds is unknown, so
    # "carry executable payloads" below is a floor.
    for snapshot in snapshots:
        if snapshot.unlistable_dirs:
            listed = ", ".join(f"{d} ({e})" for d, e in snapshot.unlistable_dirs[:5])
            warnings.append(
                f"skill {snapshot.skill_id!r} in root {snapshot.root_id!r} has "
                f"{len(snapshot.unlistable_dirs)} director"
                f"{'y' if len(snapshot.unlistable_dirs) == 1 else 'ies'} the payload inventory "
                f"could not list or fully examine ({listed}); its payload count is a floor, not "
                "an inventory"
            )
    state = _open_state(config)

    # The layout is judged before anything touches the state tree, the runner's
    # sandbox scratch directory included: a state root that is a symlink is a
    # finding for this report, not a refusal that pre-empts it. With a bad
    # layout the runner gets no scratch root, so nothing is created through
    # the link on the way to the verdict.
    # A state directory this process cannot reach inside is asked about first:
    # below it ``exists`` raises rather than answers, so the question it would
    # answer is already settled. Its reason is the whole layout verdict, which
    # keeps every check below that waits on a sound layout from asking.
    unreachable = state.reach_problem()
    layout_problems: list[str] = []
    state_present = (
        unreachable is not None or config.state_dir.exists() or config.state_dir.is_symlink()
    )
    if state_present:
        layout_problems = state.layout_problems()
    # A candidate record that does not read stands `daily` down (the queue
    # cannot be counted), so it is a problem for this report, not a warning.
    # Read only from a tree that is sound and already has the directory: this
    # report creates nothing on the way to its verdict.
    unreadable_candidates: list[dict[str, str]] = []
    # The literal name rather than ``state.dir("candidates")``: that call would
    # create the directory, and this report creates nothing.
    if state_present and not layout_problems and (config.state_dir / "candidates").is_dir():
        unreadable_candidates = [u.to_dict() for u in state.unreadable_candidates()]
        for record in unreadable_candidates:
            problems.append(
                "candidate record cannot be read and `daily` will stand down until it is moved "
                f"or repaired: {record['reason']}"
            )
    # A checkpoint `daily` would refuse is a problem when it is today's (the
    # next run refuses by name before anything is scored) and a warning for a
    # past day (only `daily --day` for that day would). Measured before: with
    # today's checkpoint truncated, this report said ready.
    unreadable_checkpoints: list[dict[str, str]] = []
    if state_present and not layout_problems and (config.state_dir / "checkpoints").is_dir():
        todays = state.checkpoint_path(today)
        for checkpoint in state.unreadable_checkpoints():
            unreadable_checkpoints.append(checkpoint.to_dict())
            if checkpoint.path == todays:
                problems.append(
                    "today's checkpoint cannot be read and `daily` will refuse to run until "
                    f"it is moved aside or repaired: {checkpoint.reason}"
                )
            else:
                warnings.append(
                    "another day's checkpoint cannot be read; `daily --day` for that day "
                    f"would refuse until it is moved aside or repaired: {checkpoint.reason}"
                )
    # A foreign entry elsewhere in the tree is a warning (left alone, stepped
    # over); at today's checkpoint path or today's receipt path it sits where a
    # committed run would write, which the run refuses by name before the lock
    # rather than exit 70 from its final write, so here it is a problem. The
    # same question the run asks, so the two cannot name the entry differently.
    # Measured before: a link at the checkpoint path, and a directory at either
    # (a directory is not a foreign entry), left the verdict at ready; the
    # directory also let the run spend every model call before it failed.
    named_as_problems: set[Path] = set()
    if state_present and not layout_problems:
        for what, subdir, path in (
            ("today's checkpoint", "checkpoints", lambda: state.checkpoint_path(today)),
            ("today's receipt", "receipts", lambda: state.receipt_path(today, dry_run=False)),
        ):
            if not (config.state_dir / subdir).is_dir():
                continue
            problem = state.artifact_path_problem(path(), what)
            if problem:
                named_as_problems.add(path())
                problems.append(
                    f"{problem}, where today's committed run would write and so refuses to "
                    "start; remove it if you did not put it there"
                )
    # The run lock is the first thing `daily` touches. A link, a directory or a
    # fifo there is refused by name (a directory is not even a foreign entry,
    # so the left-alone warning would not mention it). Measured before: both
    # left the verdict at ready while the next run exited 3.
    # Order matters: ``is_dir`` follows a symlinked ``locks``, and only the
    # ``layout_problems`` term before it keeps ``lock_entry_problem`` (whose
    # ``dir()`` refuses a symlinked ancestor) from being asked through one.
    if state_present and not layout_problems and (config.state_dir / "locks").is_dir():
        lock_problem = state.lock_entry_problem()
        if lock_problem:
            problems.append(
                f"{lock_problem}, which `daily` refuses to use as its lock; remove the entry "
                "if you did not put it there"
            )

    # The isolation decision comes BEFORE anything touches the source, so
    # `doctor` on a host with no sandbox mechanism reports the refusal rather
    # than a preflight result that implies the run could proceed.
    # A state directory that cannot be created is a finding for this report
    # too, not a refusal that pre-empts it: the runner is built without a
    # scratch root and the rest of the checks still run.
    try:
        runner = build_runner(config, None if layout_problems else state)
    except ConfigError as exc:
        problems.append(str(exc))
        runner = build_runner(config, None)
    isolation = decide_isolation(
        source_kind=config.source.kind,
        runner_isolation=runner.isolation(),
        roles=sorted(config.model_roles),
    )
    if not isolation.satisfied:
        problems.append(
            f"isolation not satisfied for a {isolation.source_class} source: "
            + "; ".join(isolation.reasons)
        )
    # A role whose program cannot start is a problem this report can see
    # without running a model. Measured before: a scorer at mode 644 read as
    # ready here and ended `daily` as an internal error.
    for problem in runner.program_problems():
        problems.append(f"model {problem}; set [model.<role>].argv in your config")

    # Existing state may predate the permission policy, or have been created
    # under a permissive umask. Repair it here rather than only checking, because
    # a doctor that reports a problem it could have fixed is asking a human to
    # run chmod for it.
    #
    # The layout is checked before the modes, and a bad layout stops the repair
    # rather than being repaired: a state directory that is a symlink redirects
    # the tree, so repairing "its" modes would chmod a directory outside the
    # configured state directory, and reporting the measured modes would describe
    # someone else's tree as private.
    permissions_repaired: list[str] = []
    permission_problems: list[str] = []
    left_alone: list[str] = []
    misplaced_directories: list[str] = []
    if state_present:
        if unreachable:
            problems.append(unreachable)
        elif layout_problems:
            problems.append(
                f"the state layout under {config.state_dir} is not confined to it: "
                f"{layout_problems[0]}"
            )
        else:
            permissions_repaired = state.repair_permissions()
            permission_problems = state.permission_problems()
            if permission_problems:
                problems.append(
                    f"{len(permission_problems)} path(s) under {config.state_dir} are not "
                    f"owner-only even after an attempted repair: {permission_problems[0]}"
                )
            # A link below the mandatory layout is stepped over, not repaired:
            # its mode is not ours to set and a write replaces it rather than
            # following it. It is still not this tool's entry, so the tree is
            # not called clean while it is there.
            left_alone = state.foreign_entries()
            if len(left_alone) == 1:
                warnings.append(
                    f"1 entry under {config.state_dir} is not a regular file or directory and "
                    f"was left alone, neither repaired nor removed: {left_alone[0]}. This tool "
                    "refuses to read such an entry and replaces rather than follows one it "
                    "writes to; remove it if you did not put it there."
                )
            elif left_alone:
                warnings.append(
                    f"{len(left_alone)} entries under {config.state_dir} are not regular files "
                    f"or directories and were left alone, neither repaired nor removed: "
                    f"{left_alone[0]}. This tool refuses to read such entries and replaces "
                    "rather than follows one it writes to; remove them if you did not put "
                    "them there."
                )
            # A directory where only records are written is not a foreign entry
            # and no read lists it as a record, but a write cannot replace it:
            # a write that comes to that name refuses by name. Measured before:
            # a directory at one conversation record's path left this report at
            # ready and the next run exited 70 from the middle of its pass. One
            # already raised to a problem above (today's receipt or checkpoint)
            # is not repeated here as a warning.
            misplaced_directories = state.misplaced_directories()
            not_yet_named = [
                rel
                for rel in misplaced_directories
                if config.state_dir / rel not in named_as_problems
            ]
            if not_yet_named:
                count = len(not_yet_named)
                warnings.append(
                    f"{count} director{'y sits' if count == 1 else 'ies sit'} where only records "
                    f"are written, under {config.state_dir}: {not_yet_named[0]}. A write that "
                    "comes to that name refuses by name rather than replacing it; remove "
                    f"{'it' if count == 1 else 'them'} if you did not put "
                    f"{'it' if count == 1 else 'them'} there."
                )

    # The health questions the operations doc asks, answered from the newest
    # daily receipt: is it recent, did it detect a skill tree mutation, does
    # it name a confinement mode. Measured before: `doctor` said `ready` with
    # the schedule silent for a week and a mutation in the last receipt.
    # Read after the mode repair above, so a receipt whose mode drifted is
    # repaired and read in this same visit rather than reported as unreadable.
    last_run = None
    last_run_age: int | None = None
    unreadable_receipts: list[dict[str, str]] = []
    kept_receipts: list[str] = []
    if state_present and not layout_problems:
        receipts = state.scan_receipts()
        last_run = receipts.last_run
        unreadable_receipts = [u.to_dict() for u in receipts.unreadable]
        kept_receipts = [str(path) for path in receipts.kept]
        if kept_receipts:
            # Each is the record of a run blocked by a skill tree mutation, kept
            # when a later committed run of its day took its path. Never a run
            # again, so without this line they would accumulate unseen.
            more = "" if len(kept_receipts) <= 3 else f", +{len(kept_receipts) - 3} more"
            # Counted by name, not read: the name is the one a kept receipt
            # gets, and whether each file still reads as one is the operator's
            # question when they open it.
            warnings.append(
                f"{len(kept_receipts)} kept blocked receipt(s) under {state.root / 'receipts'}, "
                f"by name ({', '.join(one_line(k) for k in kept_receipts[:3])}{more}): the name "
                "a receipt that recorded a skill tree mutation gets when a later committed run of "
                "its day takes its path; read skill_tree.changes in each, then move it out of "
                "the state directory"
            )
        for record in unreadable_receipts:
            # A problem, not a warning: `daily` refuses to run while it is there.
            problems.append(
                f"a receipt cannot be read ({record['path']}): {record['reason']}; `daily` "
                "refuses to run while it is there. Read it, then move it aside"
            )
        if last_run is not None and last_run.mutation_detected:
            problems.append(
                f"the last run ({one_line(last_run.day)}) detected a skill tree mutation; "
                f"`daily` refuses to run again until you have read skill_tree.changes in "
                f"{one_line(str(last_run.path))} and start one committed run with "
                f"--commit --mutation-reviewed {one_line(last_run.day)}"
            )
        if last_run is not None and last_run.mutation_detected is None:
            warnings.append(
                f"the last run's receipt ({last_run.path}) does not say whether the skill tree "
                "was mutated; it was edited or written by an older build"
            )
        # Question 1 is judged, not only printed: a stale, future or unreadable
        # finish time each gets its own warning, so a clock-skewed or hand-edited
        # stamp cannot answer "is it recent" with a silent yes.
        if last_run is not None:
            last_run_age = _age_in_days(last_run.finished_at, SystemClock().now())
            if not last_run.finished_at:
                warnings.append(
                    f"the last run's receipt ({last_run.path}) records no finish time; "
                    "it was edited or written by an older build"
                )
            elif last_run_age is None:
                warnings.append(
                    f"the last run's receipt ({last_run.path}) carries a finish time that does "
                    f"not parse: {quoted_value(last_run.finished_at)}; it was edited"
                )
            elif last_run_age < 0:
                warnings.append(
                    f"the last run's receipt ({last_run.path}) says it finished in the future "
                    f"({last_run.finished_at}); the clock or the receipt was edited"
                )
            elif last_run_age > STALE_RUN_DAYS:
                warnings.append(
                    f"the last run finished {last_run_age} day(s) ago; a daily schedule that is "
                    "firing leaves a receipt every day, so check the scheduler, not the job"
                )

    source = build_source(config, state)
    source_status = "ok"
    try:
        source.preflight()
    except AsiError as exc:
        source_status = str(exc)
        problems.append(f"source unavailable: {exc}")
    # A source that reads a directory can say what in it does not read as a
    # thread; each such entry is a finding here, as a passed-over SKILL.md is,
    # so an export the run will step over never looks like one that was taken.
    source_passed_over: list[dict[str, str]] = []
    if source_status == "ok" and hasattr(source, "passed_over"):
        try:
            skipped = list(source.passed_over())
        except (OSError, ValueError, AsiError) as exc:
            # The report must not fall over on the way to its verdict.
            skipped = []
            warnings.append(f"source could not say what it passed over: {exc}")
        # Masked as the daily receipt masks them. Measured before: a key-shaped
        # export name, or a directory above it, was raw in the warning and in
        # ``--json``'s list, while the receipt of a run over it held none.
        for entry, reason in skipped:
            item = shown_passed_over(entry, reason)
            source_passed_over.append(item)
            warnings.append(f"source entry passed over: {item['entry']} {item['reason']}")

    payload = {
        "version": __version__,
        "config_path": str(config.config_path),
        "state_dir": str(config.state_dir),
        "dry_run_default": config.dry_run,
        "source": {
            "kind": config.source.kind,
            "status": source_status,
            "source_class": isolation.source_class,
            "passed_over": source_passed_over,
        },
        "isolation": isolation.to_dict(),
        "state_permissions": {
            "dir_mode": oct(DIR_MODE),
            "file_mode": oct(STATE_FILE_MODE),
            "repaired": permissions_repaired,
            "still_wrong": permission_problems,
            "unsafe_layout": [] if unreachable else layout_problems,
            "unreachable": unreachable,
            "left_alone": left_alone,
            "misplaced_directories": misplaced_directories,
        },
        "roles_configured": sorted(config.model_roles),
        "skill_roots": [
            # ``exists`` is what is on disk; ``is_directory`` is what discovery
            # can use. A root that is a regular file exists and is not usable.
            {**r.to_dict(), "exists": presence[0], "is_directory": presence[1]}
            for r in config.skill_roots
            for presence in (root_presence(r.path),)
        ],
        "skills_discovered": len(store.known_skill_ids()),
        "skills_that_do_not_load": [{"skill_id": s.skill_id, "path": str(s.path)} for s in broken],
        "skill_documents_passed_over": [d.to_dict() for d in passed_over],
        "candidate_records_unreadable": unreadable_candidates,
        "checkpoint_records_unreadable": unreadable_checkpoints,
        "last_run": (
            {**last_run.to_dict(), "age_days": last_run_age} if last_run is not None else None
        ),
        "receipt_records_unreadable": unreadable_receipts,
        "kept_blocked_receipts": kept_receipts,
        "skills_with_executable_payloads": [
            {"skill_id": s.skill_id, "payload_files": list(s.payload_files)}
            for s in snapshots
            if s.payload_files
        ],
        "skills_with_unlistable_directories": [
            {
                "skill_id": s.skill_id,
                "root_id": s.root_id,
                "unlistable_dirs": [{"path": d, "error": e} for d, e in s.unlistable_dirs],
            }
            for s in snapshots
            if s.unlistable_dirs
        ],
        "selection": config.selection.to_dict(),
        "evaluator_independence": independence,
        "rubrics": [
            {"id": r.id, "version": r.version, "criteria": list(r.criterion_ids)}
            for r in config.rubrics
        ],
        "gate": config.gate.to_dict(),
        "protected_paths": [str(p) for p in config.protected_paths],
        "automation_marker_set": automation_active(),
        "problems": problems,
        "warnings": warnings,
    }
    lines = [
        f"agent-skill-improver {__version__}",
        f"config       {config.config_path}",
        f"state        {config.state_dir}  (gitignored, local only)",
        f"source       {config.source.kind} -> {source_status}  " f"[{isolation.source_class}]",
        f"isolation    mode={isolation.mode} verified={isolation.verified} "
        f"required={isolation.required} satisfied={isolation.satisfied}",
        (
            f"state perms  dirs {oct(DIR_MODE)} / files {oct(STATE_FILE_MODE)}, not measured: "
            "this process cannot reach inside the state directory"
            if unreachable
            else f"state perms  dirs {oct(DIR_MODE)} / files {oct(STATE_FILE_MODE)}, "
            f"{len(permissions_repaired)} repaired, {len(permission_problems)} still wrong, "
            f"{len(layout_problems)} not a real directory, {len(left_alone)} left alone"
        ),
        f"roles        {', '.join(sorted(config.model_roles)) or '<none configured>'}",
        f"skill roots  {len(config.skill_roots)} configured, {len(store.existing_roots())} present",
        f"skills       {len(store.known_skill_ids())} discovered, {len(broken)} do not load, "
        f"{sum(1 for s in snapshots if s.payload_files)} carry executable payloads, "
        f"{len(passed_over)} passed over",
        f"selection    bounded by {config.selection.fairness_dimension!r}, "
        f"max {config.selection.max_conversations_per_run}/run, round-robin (never a sorted prefix)",
        f"judge        {'independent of the improver' if independence['judge_independent'] else 'NOT independent of the improver'}",
        f"rubrics      {', '.join(r.id for r in config.rubrics)}",
        f"dry run      {config.dry_run} (default for `asi daily`)",
        "last run     " + _last_run_text(last_run, last_run_age),
    ]
    for warning in warnings:
        lines.append(f"WARN  {warning}")
    for problem in problems:
        lines.append(f"PROBLEM  {problem}")
    lines.append("verdict: " + ("problems found" if problems else "ready"))
    _emit(payload, as_json=args.json, lines=lines)
    return 1 if problems else 0


def cmd_skills(args: argparse.Namespace) -> int:
    config = _load(args)
    store = build_skill_store(config)
    groups = store.groups()
    passed_over = store.passed_over()
    payload = {
        "groups": [g.to_dict() for g in groups],
        "passed_over": [d.to_dict() for d in passed_over],
    }
    lines = []
    for group in groups:
        canonical = group.canonical
        head = f"{group.skill_id}"
        if canonical is None:
            head += "  [NO CANONICAL EDITABLE COPY -- proposals refused]"
        else:
            head += f"  canonical={canonical.root_id} sha={canonical.sha256[:12]}"
            if not canonical.loads:
                head += "  [DOES NOT LOAD]"
        # A copy in the root that would be edited is not a mirror; when that
        # root holds several, they are listed as what they are. A root is
        # named once with a count, never once per copy.
        if group.ambiguous_copies:
            head += f"  ambiguous={_root_labels(group.ambiguous_copies)}"
        mirrors = [m for m in group.mirrors if m not in group.ambiguous_copies]
        if mirrors:
            head += f"  mirrors={_root_labels(mirrors)}"
        if group.divergent_mirrors:
            head += f"  DIVERGENT={_root_labels(group.divergent_mirrors)}"
        lines.append(head)
        if canonical is None:
            lines.append(f"    why: {one_line(group.canonical_reason)}")
        # A root holding several mirror copies of this skill is one name in
        # the head line, so which copy is which, and which one diverged, is
        # said here by path. Measured before: two copies in one mirror root
        # read as "mirrors=mirror,mirror  DIVERGENT=mirror", naming neither.
        several = {
            r for r in (m.root_id for m in mirrors) if sum(1 for m in mirrors if m.root_id == r) > 1
        }
        for snapshot in mirrors:
            if snapshot.root_id in several:
                tag = "  DIVERGENT" if snapshot in group.divergent_mirrors else ""
                lines.append(f"    copy ({snapshot.root_id}): {snapshot.path}{tag}")
        # What discovery noticed about each copy, one line each (``_emit`` keeps
        # a file name's line break from becoming a line of its own).
        for snapshot in ([canonical] if canonical is not None else []) + list(group.mirrors):
            label = "" if snapshot is canonical else f" ({snapshot.root_id})"
            for warning in snapshot.warnings:
                lines.append(f"    warning{label}: {warning}")
    # A SKILL.md discovery stepped over is listed too, with its reason, so a
    # skipped skill never reads as an absent one.
    for document in passed_over:
        lines.append(f"passed over ({document.root_id}): {document.path} is {document.reason}")
    lines.append(
        f"{len(groups)} logical skill(s); "
        f"{sum(1 for g in groups if g.is_mirrored)} mirrored; "
        f"{sum(1 for g in groups if g.divergent_mirrors)} with divergent mirrors; "
        f"{len(passed_over)} document(s) passed over"
    )
    _emit(payload, as_json=args.json, lines=lines)
    return 0


def _root_labels(copies: Sequence[Any]) -> str:
    """Root ids of ``copies`` in first-seen order, each once, with a count past one.

    Two copies of a skill in one root are two documents under one root id;
    listing the id twice says nothing about which is which, so the id carries a
    count instead and the copies are named by path on their own lines. Root ids
    are operator-written config and may themselves contain a space, an x or a
    comma, so this label is a summary for a reader; the JSON is the record.
    """
    counts: dict[str, int] = {}
    for copy in copies:
        counts[copy.root_id] = counts.get(copy.root_id, 0) + 1
    return ",".join(f"{root_id} x{n}" if n > 1 else root_id for root_id, n in counts.items())


def cmd_rubrics(args: argparse.Namespace) -> int:
    config = _load(args)
    payload = {"rubrics": [r.to_dict() for r in config.rubrics]}
    lines = []
    for rubric in config.rubrics:
        lines.append(
            f"{rubric.id} v{rubric.version}  {rubric.title}  sha={rubric.source_sha256[:12]}"
        )
        for crit in rubric.criteria:
            lines.append(f"  - {crit.id} (weight {crit.weight}) {crit.title}")
    _emit(payload, as_json=args.json, lines=lines)
    return 0


#: How much of a thread's title a ``discover`` row shows: the cut it always
#: had, now said (``listed_value``) rather than silent.
DISCOVER_TITLE_CHARS = 60


def cmd_discover(args: argparse.Namespace) -> int:
    config = _load(args)
    state = _open_state(config)
    # `discover` reads the private source, so it carries the same gate as
    # `daily`. It is enforced before preflight for the same reason: a refusal
    # after the first subprocess call is a report, not a boundary.
    _require_isolation(config, state)
    source = build_source(config, state)
    source.preflight()
    # discover_page, not discover: this command exists to answer "what would be
    # ingested", and that answer is wrong without saying whether the list is the
    # whole population or one bounded page. Dropping the flag here was the
    # original defect -- the honesty existed in the domain and never reached the
    # one command a human runs to ask the question.
    page = source.discover_page(
        since=(args.since or "").strip() or config.source.since,
        limit=args.limit or config.source.discovery_limit,
        workspace=args.workspace,
        query=config.source.query or None,
    )
    found = list(page.threads)
    payload = {
        "threads": [t.to_dict() for t in found],
        "page": page.to_dict(),
        "is_total": page.is_total,
        "truncated_by_limit": page.truncated_by_limit,
        "honesty_note": page.honesty_note,
    }
    # A row's id, workspace and title are the export's (or a listing's),
    # untrusted and unbounded, so each is shown as ``listed_value`` shows a
    # stored record's values: masked, then cut, with no quotes added, so a
    # healthy row lists exactly as it did. Measured before: a 200,000-character
    # workspace or id made a row of that length, and a credential in a title or
    # an id was printed whole. The JSON ``threads`` keep each record whole.
    lines = [
        f"{listed_value(t.thread_id)}  "
        f"{listed_value(t.workspace, limit=LISTED_PATH_CHARS)}  "
        f"{listed_value(t.title, limit=DISCOVER_TITLE_CHARS)}"
        for t in found
    ]
    lines.append(f"{len(found)} thread(s) returned, limit {page.limit}")
    lines.append(
        "this is the WHOLE population the source would return"
        if page.is_total
        else "this is a PAGE, not a total: " + page.honesty_note
    )
    _emit(payload, as_json=args.json, lines=lines)
    return 0


#: Every state store a command opened in this process, so ``main`` can name
#: a mode reset the command itself never reported. Cleared on entry to ``main``.
_OPEN_STORES: list[StateStore] = []


def _open_state(config: Config) -> StateStore:
    state = build_state(config)
    _OPEN_STORES.append(state)
    return state


def _repair_note(state: StateStore, when: str, tail: str) -> str | None:
    repaired = state.repairs_so_far()
    if not repaired:
        return None
    shown = ", ".join(one_line(p) for p in repaired[:5])
    more = "" if len(repaired) <= 5 else f", +{len(repaired) - 5} more"
    return (
        f"note: {len(repaired)} path(s) under {one_line(str(state.root))} had a mode other than "
        f"owner-only and were reset to it {when} ({shown}{more}); {tail}\n"
    )


def _report_repairs_of_refused_run(state: StateStore) -> None:
    """Name on stderr the modes a run reset before it refused; nothing else has them."""
    state.repairs_reported = True
    note = _repair_note(
        state,
        "before this run refused",
        "the refusal below wrote nothing else, so no receipt reports the repair",
    )
    if note:
        _err(note)


def _report_unreported_repairs() -> None:
    """Name on stderr a mode reset no command output has named.

    Every store opens its directories through ``dir``, which resets a drifted
    mode on the way and leaves the reset pending for ``ensure`` to report.
    Only ``daily`` calls ``ensure``, and only ``doctor`` reports its own walk;
    the other commands reset and said nothing. Measured before: with ``state/``
    at 0755, ``candidates``, ``review``, ``apply`` and ``rollback`` each put it
    back to 0700 and none of them said so, on success or on refusal.
    """
    for state in _OPEN_STORES:
        if state.repairs_reported:
            continue
        state.repairs_reported = True
        note = _repair_note(state, "while this command ran", "the repair changed no content")
        if note:
            _err(note)


def cmd_daily(args: argparse.Namespace) -> int:
    from .app.daily import run_daily

    config = _load(args)
    if not args.allow_missing_roles:
        config.require_roles("scorer", "improver", "judge")
    state = _open_state(config)
    try:
        # Wiring the runner opens the sandbox scratch root, and the run repairs
        # the whole tree before anything it can refuse over; a refusal anywhere
        # after this line writes no receipt to report a repair in. Measured
        # before: a receipt left at 0644 was reset by the run that then refused
        # as blocked, nothing said so, and the next `doctor` reported 0
        # repaired. A drifted mode means something else touched the private
        # tree; the refusal must not swallow that signal.
        # A layout `run_daily` will refuse gets no scratch root when `dir` would
        # not have refused it first: opening it would create `sandbox` beside
        # the entry being refused, under a repair note that says the refusal
        # wrote nothing else. Measured before: a fresh state directory whose
        # `raw` was a file was left with a new `sandbox` directory by the run
        # that then refused its layout. A root, an ancestor or a `sandbox` that
        # is not a real directory is refused here in the same words as before.
        scratch: StateStore | None = state
        if state.layout_problems():
            state.check_dir("sandbox")
            scratch = None
        runner = build_runner(config, scratch)
        unstartable = runner.program_problems()
        if unstartable:
            raise ConfigError(
                "model " + "; model ".join(unstartable) + "; set [model.<role>].argv in your config"
            )
        # run_daily enforces this too -- that is the load-bearing check, since it
        # is the one a library caller cannot skip. Enforcing here as well means
        # the CLI refuses before `source.preflight()` runs a subprocess.
        decide_isolation(
            source_kind=config.source.kind,
            runner_isolation=runner.isolation(),
            roles=sorted(config.model_roles),
        ).require()
        source = build_source(config, state)
        source.preflight()
        store = build_skill_store(config)  # read-only, always
        receipt = run_daily(
            config,
            source=source,
            runner=runner,
            skill_store=store,
            state=state,
            clock=SystemClock(),
            day=args.day,
            commit=bool(args.commit),
            seed=args.seed,
            mutation_reviewed=args.mutation_reviewed or None,
        )
    except AsiError:
        _report_repairs_of_refused_run(state)
        raise
    state.repairs_reported = True  # the receipt counts and names them
    # Notes carry skill ids and source errors, so the text goes out through
    # _emit like every other listing: one line each, breaks spelled.
    lines = [
        receipt.summary_line(),
        *_set_aside_lines(receipt),
        *(f"note: {n}" for n in receipt.notes),
    ]
    if receipt.dry_run:
        lines.append("receipt: not written (dry run); --json prints it, --commit persists it")
    else:
        lines.append(f"receipt: {state.receipt_path(receipt.day, dry_run=False)}")
    _emit(receipt.to_dict(), as_json=args.json, lines=lines)
    return exit_code_for(receipt.status)


#: Record statuses that mean the run did not finish with that record: each
#: one is named in the text summary with its stage and reason.
SET_ASIDE_STATUSES = ("quarantined", "deferred", "refused")


def _set_aside_lines(receipt: Any) -> list[str]:
    """One line per skill or conversation the run set aside, with the reason.

    The summary line counts them; a count is not a reason. Measured before: a
    failing improver produced ``status=partial ... skills_deferred=1`` and
    nothing else on screen, so the operator reading the log could not tell
    which skill was deferred or why without opening the receipt.
    """
    out: list[str] = []
    for kind, table in (("skill", receipt.skills), ("conversation", receipt.records)):
        for record_id in sorted(table):
            info = table[record_id]
            status = str(info.get("status", ""))
            if status not in SET_ASIDE_STATUSES:
                continue
            reason = str(info.get("reason", "")) or "no reason recorded"
            out.append(f"{status} {kind} {record_id!r} at {info.get('stage', '?')}: {reason}")
    return out


def cmd_candidates(args: argparse.Namespace) -> int:
    config = _load(args)
    state = _open_state(config)
    scan = state.scan_candidates()
    items = (
        list(scan.readable) if args.all else [c for c in scan.readable if c.status in OPEN_STATUSES]
    )
    # A listing that hides records says it hides them. Measured before: right
    # after an apply, `candidates` printed `0 candidate(s)` and the record the
    # operator had just promoted read as vanished; `--all` was the only clue.
    closed_not_listed = len(scan.readable) - len(items)
    # No integrity check has read these records, so the text fields are shown
    # as ``listed_value`` shows them, in the text and the JSON alike: masked,
    # and cut past what a real one can be. Measured before: a record edited to
    # hold a ``ghp_`` token in its skill id, path, status or creation time
    # printed it whole, and one holding a megabyte there made a listing of a
    # megabyte. A real value is listed as before.
    rows = [
        {
            "id": c.id,
            "status": listed_value(c.status),
            "skill_id": listed_value(c.target.skill_id, limit=LISTED_SKILL_ID_CHARS),
            "canonical_path": listed_value(c.target.canonical_path, limit=LISTED_PATH_CHARS),
            "gate_passed": bool(((c.validation or {}).get("gate") or {}).get("passed")),
            "added_lines": c.patch.added_lines,
            "removed_lines": c.patch.removed_lines,
            "created_at": listed_value(c.created_at),
        }
        for c in items
    ]
    payload = {
        "listing": "all" if args.all else "open",
        "closed_not_listed": closed_not_listed,
        "unreadable": [u.to_dict() for u in scan.unreadable],
        "candidates": rows,
    }
    lines = [
        f"{row['id']}  {row['status']:<12} {row['skill_id']:<28} "
        f"gate={'pass' if row['gate_passed'] else 'fail'}"
        f"{' (closed)' if c.status != STATUS_VALIDATED else ''} "
        f"+{row['added_lines']}/-{row['removed_lines']}"
        for c, row in zip(items, rows)
    ]
    # A record that did not read is listed by name: dropping it made a corrupt
    # queue look like an empty one (see ``UnreadableRecord``).
    for record in scan.unreadable:
        lines.append(f"UNREADABLE  {record.path.name}  {record.reason}")
    footer = f"{len(items)} candidate(s)" if args.all else f"{len(items)} open candidate(s)"
    if closed_not_listed:
        footer += f", {closed_not_listed} closed not listed (--all shows them)"
    lines.append(f"{footer}; {len(scan.unreadable)} unreadable record(s)")
    _emit(payload, as_json=args.json, lines=lines)
    return 0


#: How long a set-aside record's id may be and still be listed whole. The id
#: is a conversation's, held to ``MAX_THREAD_ID_LENGTH`` (256) where it is
#: discovered, or a skill's, a directory name of at most 255 bytes, so a cut
#: past 256 characters never shortens a real one.
LISTED_RECORD_ID_CHARS = MAX_THREAD_ID_LENGTH

#: How long a set-aside record's stage may be and still be listed whole. The
#: tool writes a stage word (``ingest``, ``propose``) or ``score:`` and a rubric
#: id, either followed by ``:defer-exhausted`` once deferral is exhausted. A
#: rubric id is whatever its file says, with no bound of its own, so this one
#: is a choice: the longest stage the test suite writes is 106 characters.
LISTED_STAGE_CHARS = 4096

#: How long a set-aside record's reason may be and still be listed whole. A
#: reason is the tool's own words around the values it quotes (each cut at 120
#: characters, ``QUOTED_VALUE_CHARS``), a child's output tail (cut at 500,
#: ``QUOTED_TAIL_CHARS``) and the paths it names, which have no bound of their
#: own, so this one is a choice too: when it was chosen, the longest reason
#: the test suite wrote was 464 characters; its refusal of a candidate at
#: validate now writes one of 579. The record's file keeps a longer one whole.
LISTED_REASON_CHARS = 4096


def cmd_set_aside(args: argparse.Namespace) -> int:
    """List what runs set aside: quarantined, refused and deferred records.

    The ``daily`` summary names the records of one run; this reads them all
    back from the state tree, newest first. Measured before: the only way to
    learn what yesterday's ``quarantined=3`` referred to was to open the files
    under the quarantine directory, whose names flatten the stage.
    """
    config = _load(args)
    state = _open_state(config)
    scan = state.scan_set_aside()
    kinds = [k for k, _ in SET_ASIDE_KINDS]
    wanted = [args.kind] if args.kind else kinds
    items = [r for r in scan.readable if r.kind in wanted]
    not_listed = len(scan.readable) - len(items)
    counts = {k: sum(1 for r in scan.readable if r.kind == k) for k in kinds}
    # No integrity check has read these records either, so their text fields
    # are shown as ``cmd_candidates`` shows a candidate's, in the text and the
    # JSON alike: masked, then cut (an id or a time past what a real one can
    # be, a stage or a reason past 4,096 characters), and the JSON's path too.
    # Measured before: a record edited to hold a ``ghp_`` token in its id,
    # stage, reason or time printed it whole, and one holding a mebibyte there
    # made a listing of a mebibyte. A record the tool wrote, holding no
    # credential shape and no reason past the cut, is listed as before.
    rows = [
        {
            **r.to_dict(),
            "record_id": listed_value(r.record_id, limit=LISTED_RECORD_ID_CHARS),
            "stage": listed_value(r.stage, limit=LISTED_STAGE_CHARS),
            "reason": listed_value(r.reason, limit=LISTED_REASON_CHARS),
            "at": listed_value(r.at),
            # The file name keeps the first 96 characters of the id it is
            # keyed on (``keyed_name``), so the path is masked with the id.
            "path": listed_value(r.path, limit=LISTED_PATH_CHARS),
        }
        for r in items
    ]
    payload = {
        "listing": args.kind or "all",
        "not_listed": not_listed,
        "counts": counts,
        "unreadable": [u.to_dict() for u in scan.unreadable],
        "records": rows,
    }
    lines = []
    for r, row in zip(items, rows):
        attempts = f" (attempt {r.attempts})" if r.kind == "deferred" and r.attempts else ""
        lines.append(
            f"{row['at'] or '?':<20} {r.kind:<11} {row['record_id']!r} at {row['stage']}"
            f"{attempts}: {row['reason'] or 'no reason recorded'}"
        )
    for record in scan.unreadable:
        lines.append(f"UNREADABLE  {record.path.name}  {record.reason}")
    counted = ", ".join(f"{counts[k]} {k}" for k in kinds)
    footer = f"{len(items)} set-aside record(s)"
    if args.kind:
        footer = f"{len(items)} {args.kind} record(s)"
        if not_listed:
            footer += f", {not_listed} of other kinds not listed"
    lines.append(f"{footer} ({counted}); {len(scan.unreadable)} unreadable record(s)")
    _emit(payload, as_json=args.json, lines=lines)
    return 0


def _unseen_line(report: Mapping[str, Any]) -> str | None:
    """The review line for the unseen count, or ``None`` when there is nothing to say.

    Report-only: the gate line above it never read this number. A v3 record
    cannot say whether withholding was off, did not exist yet, or was dropped
    by a rewrite, so ``review_report`` reports it only when withholding is on
    now; with the default (k = 0) a v3 report has no ``unseen`` key and a v3
    review is line for line what it was before.
    """
    unseen = report.get("unseen") or {}
    status = unseen.get("status")
    head = "  unseen failures fixed: "
    if status == UNSEEN_MEASURED:
        withheld = len(unseen.get("withheld", []))
        replayed = unseen.get("replayed", 0)
        missing = f", {withheld - replayed} not replayed" if replayed < withheld else ""
        return f"{head}{unseen.get('fixed')}/{withheld}{missing} ({UNSEEN_SCOPE})"
    pool = len(unseen.get("eligible", []))
    if status == UNSEEN_TOO_FEW:
        return (
            f"{head}not measured (the skill had {pool} failure(s), fewer than "
            f"[caps].unseen_min_failures = {unseen.get('min_failures')})"
        )
    if status == UNSEEN_ONE_CONVERSATION:
        return (
            f"{head}not measured (all {pool} failure(s) are on one conversation, so none can "
            "be withheld while one stays shown)"
        )
    if status == UNSEEN_SHORT_OF_K:
        return (
            f"{head}not measured (fewer than [caps].unseen_withhold = {unseen.get('withhold')} "
            "failure(s) can be withheld while one whole conversation stays shown)"
        )
    if status == "not-recorded":
        return (
            f"{head}not recorded (an {unseen.get('schema')} record: written with "
            "[caps].unseen_withhold = 0 or before withholding existed, or rewritten as v3; "
            "the record cannot tell which)"
        )
    return None


def _replayed_case_lines(outcomes: Sequence[Mapping[str, Any]]) -> list[str]:
    """The cases behind the gate's metrics, one line each, the tool's words only.

    Every target case and every holdout whose verdict changed is named with
    its verdict pair and change word; the holdouts that did not change carry
    no decision information beyond their number, which is given, and stay in
    the JSON. Ids are file- or model-derived and are kept to one line.
    """
    lines: list[str] = []
    unchanged_holdouts = 0
    for outcome in outcomes:
        if outcome["kind"] == "holdout" and outcome["change"] == "unchanged":
            unchanged_holdouts += 1
            continue
        lines.append(
            one_line(
                f"  - [{outcome['kind']}] {outcome['criterion_id']} (rubric {outcome['rubric_id']}) "
                f"in conversation {outcome['conversation_id']}: {outcome['baseline']} -> "
                f"{outcome['candidate']}  {outcome['change']}"
            )
        )
    if unchanged_holdouts:
        lines.append(f"  - {unchanged_holdouts} holdout case(s) unchanged (listed in --json)")
    if not outcomes:
        lines.append("  (none replayed)")
    return lines


def _judgement_lines(judgements: Sequence[Mapping[str, Any]]) -> list[str]:
    """Each blind judgement: the case and arm on one line, the judge's own
    prose quoted beneath it, deeper than the improver's, so neither can stand
    at column 0."""
    lines: list[str] = []
    for judgement in judgements:
        lines.append(
            one_line(
                f"  - {judgement['criterion_id']} in conversation "
                f"{judgement['conversation_id']}: prefers {judgement['preferred_arm']}"
            )
        )
        if judgement["rationale"]:
            lines += quoted(judgement["rationale"], prefix="      | ")
    if not judgements:
        lines.append("  (no judgement recorded)")
    return lines


def cmd_review(args: argparse.Namespace) -> int:
    from .app.promote import read_live_base_text, refuse_a_closed_review, review_report

    config = _load(args)
    state = _open_state(config)
    candidate = state.read_candidate(args.candidate_id)
    # Its status first, before its file is read, as `apply` asks it: read first,
    # a closed candidate whose skill folder had moved was refused as a base
    # mismatch, whose advice is to delete the record `rollback` restores from.
    refuse_a_closed_review(candidate)
    # A review-only command still resolves the candidate against a real store:
    # read-only, never writable, but reading the LIVE configured canonical
    # snapshot rather than trusting anything the candidate record asserts about
    # its own base. Without this, review can only check the record against
    # itself, and a diff or its line counts edited to agree with each other but
    # not with the file they claim to describe would still print a token.
    store = build_skill_store(config, writable=False)
    base_text = read_live_base_text(store, candidate, action="review")
    rubrics = {r.id: r for r in config.rubrics}
    report = review_report(
        candidate,
        base_text=base_text,
        rubrics=rubrics,
        unseen_withhold=config.caps.unseen_withhold,
    )
    # The questions the write will ask of its target, asked before anything is
    # shown, by the method the write asks them with: a target reached through a
    # symbolic link, one that is not the skill document, one inside a protected
    # path, a mirror copy, a copy of a skill with no canonical, no skill
    # document in any configured root, or a skill listed in
    # [guard].protected_skill_ids gets no token here, because `apply` would
    # refuse it. It comes after the record's own checks, in the order `apply`
    # asks them (status, integrity, base, then the target), so a closed or
    # tampered candidate is named as that first.
    store.assert_write_target(
        Path(candidate.target.canonical_path),
        action="review",
        skill_id=candidate.target.skill_id,
        candidate_id=candidate.id,
    )
    # The mirror copies `apply` does not write, as the live inventory holds
    # them: the record's own list is bound by neither the token nor the
    # integrity check, and was printed as written. Each is a path on disk, so
    # it is named as a refusal names one (`named_path`): bare when it reads
    # as a path, else quoted, masked and cut.
    report["mirror_paths"] = list(store.mirror_paths_of(Path(candidate.target.canonical_path)))
    if args.json:
        _say_json(report)
        return 0
    # Only the tool's own words stand at column 0 in this report: the section
    # headers, the diff and the apply instruction. Every content line of the
    # diff carries difflib's prefix character, its only unprefixed line is the
    # ``\ No newline at end of file`` marker ``make_patch`` writes, and
    # ``make_patch`` keeps its two header labels to one line; ``review_report`` has already checked that the
    # stored diff equals the one recomputed from the live base, so the diff
    # printed here is one ``make_patch`` produced. A skill id, path, gate reason
    # or note is kept to one line; the model's prose is quoted (see ``quoted``)
    # so it cannot imitate a header or the apply line.
    out = [
        one_line(f"candidate     {report['candidate_id']}  ({report['status']})"),
        one_line(f"skill         {report['skill_id']}"),
        one_line(f"file          {report['canonical_path']}"),
        f"base sha256   {report['base_sha256']}",
        f"gate          {'PASS' if report['gate'].get('passed') else 'FAIL'}",
    ]
    for reason in report["gate"].get("reasons", []):
        out.append(one_line(f"  gate reason: {reason}"))
    metrics = report["gate"].get("metrics", {})
    if metrics:
        out.append(
            "  metrics: fixes={fixes} regressions={regressions} holdouts={holdout_cases} "
            "judge_pref={judge_candidate_preference_ratio}".format(
                fixes=metrics.get("fixes"),
                regressions=metrics.get("regressions"),
                holdout_cases=metrics.get("holdout_cases"),
                judge_candidate_preference_ratio=metrics.get("judge_candidate_preference_ratio"),
            )
        )
    unseen_line = _unseen_line(report)
    if unseen_line is not None:
        out.append(one_line(unseen_line))
    # Which cases the gate counted, and what the judge said about each: the
    # evidence behind the metrics line, so the operator can read the evidence
    # rather than the summary. Ids are kept to one line; the judge's prose is
    # quoted like the improver's. The unseen cases are listed with them, tagged
    # [unseen], in the text only: report["outcomes"] stays the gate's list.
    replayed = list(report["outcomes"]) + list(report.get("unseen", {}).get("outcomes", []))
    out += ["", "REPLAYED CASES (baseline -> candidate)", *_replayed_case_lines(replayed)]
    out += ["", "JUDGE (blind, candidate vs baseline)", *_judgement_lines(report["judgements"])]
    out += ["", "HYPOTHESIS", *quoted(report["hypothesis"])]
    out += ["", "INTENDED CHANGE", *quoted(report["intended_change"])]
    out += ["", "RISKS"]
    for risk in report["risks"]:
        first, *rest = risk.split("\n")
        out.append(f"  - {first}".rstrip())
        if rest:
            out += quoted("\n".join(rest))
    out += ["", "ROLLBACK", *quoted(report["rollback"]), ""]
    if report["mirror_paths"]:
        out.append("MIRROR COPIES (not written by apply):")
        out += [one_line(f"  - {named_path(p)}") for p in report["mirror_paths"]]
        out.append("")
    out += ["EVIDENCE (rubric failures this claims to fix)"]
    for failure in report["rubric_failures"]:
        refs = ", ".join(f"{e['element_id']}@{e['position']}" for e in failure["evidence"])
        out.append(
            one_line(
                f"  - {failure['criterion_id']} in conversation {failure['conversation_id']}"
                + (f" [{refs}]" if refs else " [no evidence refs]")
            )
        )
    out += ["", "DIFF", report["diff"] or "(empty)"]
    for note in report["notes"]:
        out.append(one_line(f"note: {note}"))
    out += [
        "",
        "To promote this change, review the diff above, then run:",
        f"  {report['apply_command']}",
        "",
        "Nothing has been written to any skill file yet.",
    ]
    _say("\n".join(out) + "\n")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    from .app.promote import apply_candidate

    config = _load(args)
    state = _open_state(config)
    candidate = state.read_candidate(args.candidate_id)
    approval = HumanApproval.from_cli(
        token=args.approve,
        acknowledged=bool(args.i_reviewed_this),
        operator=args.operator or os.environ.get("USER", ""),
        note=args.note or "",
    )
    store = build_skill_store(config, writable=True)
    rubrics = {r.id: r for r in config.rubrics}
    result = apply_candidate(
        candidate,
        approval=approval,
        skill_store=store,
        state=state,
        clock=SystemClock(),
        operator=args.operator or os.environ.get("USER", ""),
        rubrics=rubrics,
    )
    write = result["write"]
    rollback = shlex.join(["asi", "rollback", result["candidate_id"]])
    # Each line through ``_emit``, as a listing prints, so a path is kept to its
    # line and spelled as ``review`` spells it. Measured before: a skill
    # directory named with a line break made the line after the break, and the
    # rest of the manual command, start lines of their own. The manual command
    # is shown only when it reads as itself on the line: a spelled break or
    # control character would make it name another path, and ``asi rollback``
    # restores the same backup and checks what it restores.
    manual = write["rollback_command"]
    if not reads_as_itself(manual):
        manual = (
            "not shown: a path in it holds a line break or a control character, which "
            f"this line can only spell; `{rollback}` restores the same backup"
        )
    lines = [
        f"applied {result['candidate_id']} to {write['path']}",
        *(
            [
                "  finishes        an earlier run of this apply, which wrote the "
                "skill and did not record it"
            ]
            if result.get("finishes_an_earlier_run")
            else []
        ),
        f"  previous sha256 {write['previous_sha256']}",
        f"  new sha256      {write['new_sha256']}",
        f"  backup          {write['backup_path']}",
        f"  rollback        {rollback}",
        f"  (or manually)   {manual}",
    ]
    _emit(result, as_json=args.json, lines=lines)
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    from .app.promote import rollback_candidate

    config = _load(args)
    state = _open_state(config)
    candidate = state.read_candidate(args.candidate_id)
    store = build_skill_store(config, writable=True)
    result = rollback_candidate(candidate, skill_store=store, state=state, clock=SystemClock())
    restored = result["result"]
    # Each line through ``_emit``, as ``apply``'s are, so a path is kept to its
    # line and spelled as ``review`` spells it. Measured before: the store's
    # result was printed as a Python dict, its keys and ``True``/``False``
    # included, on the line that said what was rolled back.
    lines = [
        f"rolled back {result['candidate_id']}: {restored['rolled_back']}",
        f"  restored sha256 {restored['restored_sha256']}",
        f"  from backup     {restored['from_backup']}",
    ]
    if restored["already_restored"]:
        lines.append(
            "  nothing written the file already held the backup's text (a second rollback, "
            "or the backup copied by hand)"
        )
    _emit(result, as_json=args.json, lines=lines)
    return 0


def _cron_quote(value: str) -> str:
    """Spell ``value`` for a crontab command.

    Shell-quoted first. Then for cron itself, which rewrites the command before
    any shell sees it and knows nothing of quotes: a bare ``%`` ends the command
    (the rest is its input), and a backslash before a ``%`` is consumed, so every
    ``%`` is escaped and cron hands the shell the percent sign. The crons part
    ways on a backslash before a backslash: FreeBSD and macOS make the pair one,
    cronie and OpenBSD keep both, and busybox crond rewrites nothing. So no pair
    is printed: each backslash in the path (always inside the shell quotes, as a
    path holding one is always quoted) is followed by a quote ending the quoted
    text and one resuming it, which every cron keeps as written and the shell
    reads as the backslash alone. Measured before, when each backslash was
    doubled: under cronie's rewrite a config under ``back\\slash`` was named
    ``back\\\\slash``, a directory that does not exist.
    """
    return shlex.quote(value).replace("\\", "\\''").replace("%", "\\%")


def _xml_quote(value: str) -> str:
    """Spell ``value`` for the text of an XML element."""
    from xml.sax.saxutils import escape

    return escape(value)


def _refuse_a_path_no_template_carries(label: str, value: str, *, flavor: str) -> None:
    """Refuse a path the printed template could not carry exactly.

    The template is text a person copies into a crontab or a property list.
    The terminal-safe printer shows a control character, a bidi control or a
    lone surrogate as its escape, so a job printed with one would name a path
    that does not exist; and a crontab line ends at a line break, so the rest
    of the path would be a line of its own, which cron reads as another job.
    Measured before: a config under a directory whose name held a newline
    printed a cron job cut in two, and one holding ``\\x01`` or a carriage
    return printed a property list that parsed and named a config file that
    does not exist. No quoting helps: the character does not survive the copy.
    A property list carries a line break inside a string, so it is kept there.
    """
    if escape_for_terminal(value) != value:
        reason = (
            "a character the terminal output shows as its escape, so a job printed with it "
            "would name a path that does not exist"
        )
    elif flavor == "cron" and "\n" in value:
        reason = "a line break, where a crontab line ends, so the rest would be a job of its own"
    else:
        return
    shown = escape_for_terminal(value).replace("\n", "\\n")
    raise ConfigError(
        f"{label} '{shown}' holds {reason}. Nothing was printed. Move the tree to a path "
        f"without one to print a {flavor} template for it."
    )


def cmd_scheduler_template(args: argparse.Namespace) -> int:
    path = (
        repo_root()
        / "packaging"
        / (
            "launchd/com.local.agent-skill-improver.plist.template"
            if args.flavor == "launchd"
            else "cron/crontab.template"
        )
    )
    if not path.is_file():
        _err(f"template not found: {path}\n")
        return RC_MISSING_FILE
    # The job must run the config this invocation was given, and its logs must
    # land in THAT config's state tree: the output can carry private content, and
    # the repository's own state/ is only the right place when it is the
    # configured one. Measured before this was fixed, `--config X` printed a job
    # for the default config with logs under the repository.
    config_path = (
        Path(args.config).expanduser() if args.config else default_config_path()
    ).resolve()
    config = load_config(config_path)
    text = path.read_text(encoding="utf-8")
    # Each path lands in a foreign syntax and must be spelled for it. The cron
    # line is split by the shell on spaces, and cron itself turns a bare ``%``
    # into a newline before the shell ever sees the command; the plist is XML.
    # A plain path is left byte-for-byte as it was, so quoting only appears
    # where it is needed.
    quote = _cron_quote if args.flavor == "cron" else _xml_quote
    values = (
        ("__REPO__", "the repository path", str(repo_root())),
        ("__PYTHON__", "the interpreter path", sys.executable),
        ("__CONFIG__", "the config file path", str(config_path)),
        ("__STATE__", "the state directory", str(config.state_dir)),
    )
    for _marker, label, value in values:
        _refuse_a_path_no_template_carries(label, value, flavor=args.flavor)
    # One pass over the template, so a value is never read again for a marker:
    # replaced one marker at a time, a path holding a later marker's text had
    # that value spliced into it after it was quoted. Measured before: a config
    # under a directory named ``q __STATE__ ;echo INJECTED;`` printed a cron
    # line whose shell commands were split at both semicolons, and the plist
    # named a config that does not exist.
    spelled = {marker: quote(value) for marker, _label, value in values}
    text = re.sub(
        "|".join(re.escape(marker) for marker in spelled),
        lambda found: spelled[found.group(0)],
        text,
    )
    _say(text)
    _err(
        "\nThis template was printed, not installed. Nothing is scheduled. "
        "Read docs/SCHEDULING.md before enabling it.\n"
    )
    return 0


def _leading_statements_source(source: str, function: str, count: int) -> str:
    """The first ``count`` executable statements of ``function``, as source text.

    Parsed rather than read off the first few lines, because the docstring is a
    statement to Python and is not one to a reader: a textual check would either
    match the docstring or have to guess where it ends. The ordering matters here
    and is not cosmetic -- an isolation check placed after the source is read has
    already lost, so the invariant is "leading", not "present anywhere".
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - source in this repo always parses
        return ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            return "\n".join(ast.unparse(stmt) for stmt in body[:count])
    return ""


def _sorted_prefix_slices(source: str) -> list[int]:
    """Line numbers where a ``sorted(...)`` result is sliced, e.g. ``sorted(x)[:n]``.

    Structural on purpose. The obvious implementation is a substring search for
    the shape, and it produced a false positive against the module docstring that
    DOCUMENTS the anti-pattern -- a checker that cannot tell an occurrence from an
    explanation of an occurrence. Parsing instead means comments, docstrings and
    prose can say the words freely while real code cannot use the construct.

    Scope is stated rather than implied: this catches the direct form only. A
    sorted list bound to a name and sliced later is not detected here, which is
    what the round-robin tests in the suite are for.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - source in this repo always parses
        return []
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Slice):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "sorted"
        ):
            hits.append(node.lineno)
    return sorted(hits)


def cmd_selftest(args: argparse.Namespace) -> int:
    """Re-check the structural safety invariants on this checkout.

    These are read off the source, not from a config file, because the point is
    to catch a *code* change that quietly removed an interlock. The test suite
    asserts the same things; this command exists so an operator can check a
    deployed checkout without the tests.
    """
    checks: list[tuple[str, bool, str]] = []
    daily_src = (package_dir() / "app" / "daily.py").read_text(encoding="utf-8")
    checks.append(
        (
            "daily does not import the promotion path",
            not _module_imported(daily_src, "promote"),
            "app/daily.py must not import app.promote; promotion is a separate review-interlocked path",
        )
    )
    checks.append(
        (
            "daily never calls the skill write method",
            "write_canonical" not in _referenced_names(daily_src),
            "app/daily.py must not call write_canonical (prose may name it; code may not)",
        )
    )
    checks.append(
        (
            "daily refuses a writable skill store",
            "refuses a writable skill store" in daily_src,
            "run_daily must raise when handed a writable store",
        )
    )
    candidate_src = (package_dir() / "domain" / "candidate.py").read_text(encoding="utf-8")
    store_src = (package_dir() / "adapters" / "skill_fs.py").read_text(encoding="utf-8")
    checks.append(
        (
            "skill store defaults to read-only",
            "read_only: bool = True" in store_src,
            "FilesystemSkillStore must default to read_only=True",
        )
    )
    checks.append(
        (
            "only the skill document is writable",
            "_assert_document" in store_src and "NonDocumentWriteRefused" in store_src,
            "the write path must refuse any filename other than SKILL.md",
        )
    )
    promote_src = (package_dir() / "app" / "promote.py").read_text(encoding="utf-8")
    verify_call = "verify_full_integrity(candidate"
    checks.append(
        (
            "apply recomputes the candidate's derived fields before it writes",
            verify_call in promote_src
            and "write_canonical" in promote_src
            and promote_src.index(verify_call) < promote_src.index("skill_store.write_canonical"),
            "app/promote.py must call verify_full_integrity before write_canonical; a "
            "token derived from a stored hash cannot bind the text shown for review",
        )
    )
    integrity_src = (package_dir() / "domain" / "integrity.py").read_text(encoding="utf-8")
    checks.append(
        (
            "the approval token is derived from the recomputed patch hash",
            "recomputed_patch_sha256" in candidate_src
            and "self.recomputed_patch_sha256" in candidate_src
            and "sha256_text(patch.new_text)" in integrity_src,
            "Candidate.required_approval_token must hash new_text rather than trust the "
            "stored new_text_sha256 beside it",
        )
    )
    approval_src = (package_dir() / "domain" / "approval.py").read_text(encoding="utf-8")
    checks.append(
        (
            "approval refuses when the automation marker is set",
            "AutomationRefused" in approval_src,
            "HumanApproval.from_cli must refuse under automation",
        )
    )
    selection_src = (package_dir() / "domain" / "selection.py").read_text(encoding="utf-8")
    sliced = _sorted_prefix_slices(selection_src)
    checks.append(
        (
            "selection never truncates a sorted list",
            not sliced,
            "bounding by a sorted prefix is a permanent filter, not a sample"
            + (
                f" (found at line{'s' if len(sliced) > 1 else ''} {', '.join(map(str, sliced))})"
                if sliced
                else ""
            ),
        )
    )
    scoring_src = (package_dir() / "domain" / "scoring.py").read_text(encoding="utf-8")
    checks.append(
        (
            "an unresolvable citation cannot yield a decided verdict",
            "downgraded_from" in scoring_src and "does not resolve in conversation" in scoring_src,
            "a FAIL whose evidence does not resolve must become INSUFFICIENT",
        )
    )
    prompts_src = (package_dir() / "app" / "prompts.py").read_text(encoding="utf-8")
    checks.append(
        (
            "untrusted content is fenced and forged delimiters neutralized",
            "wrap_untrusted" in prompts_src and "NEUTRALIZED" in prompts_src,
            "captured content must be quoted as data, never spliced in as instruction",
        )
    )
    # Behavioural rather than textual: the classification is called, so a rename
    # or a widened set is caught, which grepping for a constant name would not be.
    checks.append(
        (
            "only the synthetic fixture kind escapes confinement",
            source_class(SOURCE_FIXTURES) == SOURCE_CLASS_SYNTHETIC
            and source_class("conversation-cli") == SOURCE_CLASS_PRIVATE
            and source_class("anything-added-later") == SOURCE_CLASS_PRIVATE,
            "source classification must fail closed: unknown kinds are private",
        )
    )
    checks.append(
        (
            "a private source with unconfined roles refuses",
            not decide_isolation(
                source_kind="conversation-cli",
                runner_isolation={"mode": MODE_NONE, "verified": False},
                roles=["scorer"],
            ).satisfied,
            "IsolationRequired must be raised before a private source is read",
        )
    )
    daily_first = _leading_statements_source(daily_src, "run_daily", 2)
    checks.append(
        (
            "the isolation check leads the daily pass",
            "decide_isolation" in daily_first and ".require()" in daily_first,
            "run_daily must decide and require isolation in its first two "
            "statements, before the lock, the skill hash and any export",
        )
    )
    checks.append(
        (
            "state directories are owner-only and artifacts are owner-read-write",
            DIR_MODE == 0o700 and STATE_FILE_MODE == 0o600,
            "state/ holds verbatim private transcripts; 0700 dirs and 0600 files",
        )
    )
    pyproject = repo_root() / "pyproject.toml"
    # Assembled from fragments so this file does not itself contain the banned
    # classifier. tools/leakscan.py bans the literal in tracked content, and a
    # checker that trips its own privacy gate gets excluded from that gate --
    # which is exactly the fail-open hole worth avoiding here.
    blocking_classifier = "Priv" + "ate :: Do " + "Not Upl" + "oad"
    checks.append(
        (
            "package metadata carries no upload-blocking classifier",
            (not pyproject.is_file())
            or blocking_classifier not in pyproject.read_text(encoding="utf-8"),
            "the repository is public-ready and must not be marked unpublishable",
        )
    )
    validation_src = (package_dir() / "domain" / "validation.py").read_text(encoding="utf-8")
    checks.append(
        (
            "review requires exact status validated",
            "STATUS_VALIDATED" in promote_src
            and "candidate.status != STATUS_VALIDATED" in promote_src,
            "review_report must refuse any status other than exactly 'validated'; applied, "
            "rolled-back, gate-failed and blocked-run candidates must never expose a token again",
        )
    )
    checks.append(
        (
            "review resolves the live base before issuing a token",
            "read_live_base_text" in promote_src
            and "base_text" in promote_src.split("def review_report")[1].split("\n\n")[0],
            "review_report must take the live canonical text as a parameter and check the "
            "candidate against it, not just against itself",
        )
    )
    checks.append(
        (
            "persisted validation is parsed strictly and recomputed",
            "class ValidationRecord" in validation_src
            and "recompute_gate_mismatches" in validation_src
            and "judge_independent = bool(self.judge_command_id) and (" in validation_src,
            "ValidationRecord.parse must refuse unknown/missing/wrong-type/duplicate data and "
            "judge independence must be derived from command ids, never a stored boolean",
        )
    )
    validate_src = (package_dir() / "app" / "validate.py").read_text(encoding="utf-8")
    checks.append(
        (
            "every target outcome is bound to an exact candidate failure ref",
            "class TargetCaseRef" in validation_src
            and "target_refs" in validation_src
            and "def evidence_call_count" in validation_src
            and "validation.target_refs is not exactly this candidate" in integrity_src
            and "target_refs=tuple(claimed_refs)" in validate_src,
            "a target outcome must be bound to a candidate failure ref by the whole "
            "(rubric_id, conversation_id, criterion_id) triple, the claim set must equal the "
            "candidate's own rubric_failures, and the model-call count must be exactly derivable "
            "from the persisted evidence rather than asserted beside it",
        )
    )
    ok = all(passed for _, passed, _ in checks)
    payload = {
        "checks": [
            {"name": name, "passed": passed, "requirement": requirement}
            for name, passed, requirement in checks
        ],
        "passed": ok,
    }
    lines = [f"{'ok  ' if passed else 'FAIL'} {name}" for name, passed, _ in checks]
    lines.append("selftest: " + ("passed" if ok else "FAILED"))
    _emit(payload, as_json=args.json, lines=lines)
    return 0 if ok else 1


def _imported_modules(source: str) -> set[str]:
    """Every module name this source actually imports, from the parse tree.

    The first version of this scanned lines containing ``" import "`` and it was
    wrong for an instructive reason: ``app/daily.py`` documents the interlock in
    its own docstring ("this module does not import ... app.promote"), that
    sentence contains the word ``import``, and the checker read the sentence as
    evidence of the thing the sentence denies. Parsing removes the whole class:
    prose may discuss any module, only real imports count.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - source in this repo always parses
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            names.add(base)
            for alias in node.names:
                names.add(
                    f"{base}.{alias.name}" if base and not base.endswith(".") else base + alias.name
                )
    return names


def _module_imported(source: str, leaf: str) -> bool:
    """True when any import in ``source`` names ``leaf`` as a dotted component."""
    return any(leaf in module.split(".") for module in _imported_modules(source))


def _referenced_names(source: str) -> set[str]:
    """Every identifier and attribute name used in code (not in comments or strings)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            out.add(node.arg)
    return out


# -- parser --------------------------------------------------------------
def _run_day_argument(value: str) -> str:
    """``--day``: refused by argparse, before any config is read, when it could
    not name a candidate; an empty value keeps the default (today, UTC)."""
    if not value:
        return value
    try:
        return validate_run_day(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


#: A whole number as int() reads it from text, the digits in group 1.
_WHOLE_NUMBER_TEXT = re.compile(r"[+-]?(\d+(?:_\d+)*)")


def _positive_int_argument(value: str) -> int:
    """``--limit``: a page size of 0 or less has no meaning, and the page
    description would otherwise say ``N of at most -1``. The value is quoted
    bounded (``quoted_value``); a number with more digits than Python reads
    from text (``sys.get_int_max_str_digits``) is told so, where before it was
    refused as not a whole number."""
    try:
        number = int(value)
    except ValueError:
        # int()'s own base-10 spelling: a sign, then digits with single
        # underscores between them, which the digit limit does not count.
        spelled = _WHOLE_NUMBER_TEXT.fullmatch(value.strip())
        digits = spelled.group(1).replace("_", "") if spelled else ""
        readable = sys.get_int_max_str_digits()
        if digits and readable and len(digits) > readable:
            raise argparse.ArgumentTypeError(
                f"must be a whole number of at most {readable} digits, the most Python "
                f"reads from text; got {len(digits)} digits: {quoted_value(value)}"
            ) from None
        raise argparse.ArgumentTypeError(
            f"must be a whole number, got {quoted_value(value)}"
        ) from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {quoted_value(value)}")
    return number


#: How many unrecognized arguments the refusal quotes before it counts the rest.
QUOTED_EXTRA_ARGUMENTS = 8


def _unrecognized_arguments(extras: Sequence[str]) -> str:
    """argparse's own message joins the leftover words raw; this one quotes
    each through ``quoted_value`` and counts what it does not quote."""
    shown = " ".join(quoted_value(word) for word in extras[:QUOTED_EXTRA_ARGUMENTS])
    rest = len(extras) - QUOTED_EXTRA_ARGUMENTS
    more = f" ... and {rest} more not quoted" if rest > 0 else ""
    return f"unrecognized arguments: {shown}{more}"


#: The most of an argparse refusal that is printed whole. Past it, the start
#: and the last ``ARGPARSE_MESSAGE_TAIL`` characters are kept, the end being
#: where argparse names the options an ambiguous one could match.
ARGPARSE_MESSAGE_CHARS = 1_600
ARGPARSE_MESSAGE_TAIL = 200


def _bounded_argparse_message(message: str) -> str:
    """A refusal as ``_EscapingParser.error`` prints it: masked by the
    redaction shapes, then cut in the middle when it is longer than
    ``ARGPARSE_MESSAGE_CHARS``, with the count of what was left out. The
    messages handed to argparse here (the run day's, the page size's, an
    invalid choice and the unrecognized arguments) quote each typed value
    through ``quoted_value`` and stay under the limit; argparse's own
    "ambiguous option" and "ignored explicit argument" repeat the typed word
    whole."""
    text = redact(message).text
    if len(text) <= ARGPARSE_MESSAGE_CHARS:
        return text
    head = ARGPARSE_MESSAGE_CHARS - ARGPARSE_MESSAGE_TAIL
    left_out = len(text) - ARGPARSE_MESSAGE_CHARS
    return (
        f"{text[:head]} ... [{left_out} more chars not shown] ... "
        f"{text[-ARGPARSE_MESSAGE_TAIL:]}"
    )


class _EscapingParser(argparse.ArgumentParser):
    """argparse writes usage and errors to the terminal itself: here what it
    prints is escaped for the terminal (``_print_message``) and what it
    refuses is masked and bounded (``error``). The "unrecognized arguments"
    message is built by ``_unrecognized_arguments``, which ``main`` calls.
    Subparsers inherit this class, so the overrides reach each subcommand's
    parser too."""

    def _print_message(self, message: str, file: Any = None) -> None:
        if message:
            (file or sys.stderr).write(escape_for_terminal(message))

    def error(self, message: str) -> NoReturn:
        super().error(_bounded_argparse_message(message))

    def _check_value(self, action: argparse.Action, value: Any) -> None:
        """argparse's "invalid choice" repeats the typed value whole (a
        subcommand, a ``set-aside --kind`` or a ``scheduler-template``
        flavor); here it is quoted bounded and the choices are still listed."""
        if action.choices is not None and value not in action.choices:
            choices = ", ".join(map(repr, action.choices))
            raise argparse.ArgumentError(
                action, f"invalid choice: {quoted_value(value)} (choose from {choices})"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = _EscapingParser(
        prog="asi",
        description=(
            "Proposal-only daily self-improvement loop for agent skills. "
            "Automation scores, isolates, proposes and validates; apply is separate "
            "with review interlocks, not caller authentication."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"agent-skill-improver {__version__}"
    )
    config_help = (
        f"path to config TOML (default: ${ENV_CONFIG}, else config/local.toml, "
        "else config/example.config.toml)"
    )
    parser.add_argument("--config", help=config_help)
    parser.add_argument("--json", action="store_true", help="machine-readable output")

    # Accepted either before or after the subcommand. `default=SUPPRESS` is what
    # makes that work: without it the subparser would write its own False over a
    # value the user already supplied at the top level.
    common = _EscapingParser(add_help=False)
    common.add_argument("--config", help=config_help, default=argparse.SUPPRESS)
    common.add_argument(
        "--json", action="store_true", help="machine-readable output", default=argparse.SUPPRESS
    )

    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text, description=help_text, parents=[common])

    add("doctor", "validate configuration and report what is wired").set_defaults(func=cmd_doctor)
    add("skills", "skill inventory: canonical vs mirror, discovery warnings").set_defaults(
        func=cmd_skills
    )
    add("rubrics", "show loaded rubrics").set_defaults(func=cmd_rubrics)

    p_discover = add("discover", "list conversations the source would ingest")
    p_discover.add_argument("--since", help="time window, e.g. 1d, 7d, 2026-08-01")
    p_discover.add_argument("--limit", type=_positive_int_argument, help="page size, at least 1")
    p_discover.add_argument("--workspace")
    p_discover.set_defaults(func=cmd_discover)

    p_daily = add(
        "daily",
        "one daily pass: ingest, score, isolate, propose, validate (DRY RUN by default)",
    )
    p_daily.add_argument(
        "--commit",
        action="store_true",
        help="persist candidates and the checkpoint. Still never writes a skill file.",
    )
    p_daily.add_argument(
        "--day",
        type=_run_day_argument,
        help=(
            "run id / idempotence key, defaults to today (UTC); ASCII letters, digits, "
            "dot, underscore or hyphen, since it names every candidate of the run"
        ),
    )
    p_daily.add_argument("--seed", help="deterministic seed for nonces and blind judging")
    p_daily.add_argument(
        "--mutation-reviewed",
        metavar="DAY",
        type=_run_day_argument,
        default="",
        help=(
            "the day of a blocked run whose recorded skill tree mutation you have read; "
            "without it a run after a blocked one is refused (exit 5). Only a committed run "
            "clears the block: a dry run writes no receipt"
        ),
    )
    p_daily.add_argument(
        "--allow-missing-roles",
        action="store_true",
        help="do not fail fast when a model role is unconfigured (for partial runs)",
    )
    p_daily.set_defaults(func=cmd_daily)

    p_candidates = add("candidates", "list candidate proposals")
    p_candidates.add_argument(
        "--all",
        action="store_true",
        help="include closed candidates (applied, rolled-back, gate-failed, blocked-run)",
    )
    p_candidates.set_defaults(func=cmd_candidates)

    p_set_aside = add("set-aside", "list what runs set aside: quarantined, refused, deferred")
    p_set_aside.add_argument(
        "--kind",
        choices=[k for k, _ in SET_ASIDE_KINDS],
        help="list one kind only; the footer still counts the others",
    )
    p_set_aside.set_defaults(func=cmd_set_aside)

    p_review = add("review", "print a candidate's full argument, diff and approval token")
    p_review.add_argument("candidate_id")
    p_review.set_defaults(func=cmd_review)

    p_apply = add("apply", "write a candidate (approval token + acknowledgement; review-gated)")
    p_apply.add_argument("candidate_id")
    p_apply.add_argument("--approve", help="approval token printed by `asi review`")
    p_apply.add_argument(
        "--i-reviewed-this",
        action="store_true",
        dest="i_reviewed_this",
        help="explicit acknowledgement, recorded in the receipt",
    )
    p_apply.add_argument("--operator", help="who is promoting this")
    p_apply.add_argument("--note", help="free-text note for the receipt")
    p_apply.set_defaults(func=cmd_apply)

    p_rollback = add("rollback", "restore a verified backup (no token or acknowledgement)")
    p_rollback.add_argument("candidate_id")
    p_rollback.set_defaults(func=cmd_rollback)

    p_sched = add(
        "scheduler-template",
        "print a marked launchd or cron job (installs nothing; do not schedule apply)",
    )
    p_sched.add_argument("flavor", choices=("launchd", "cron"))
    p_sched.set_defaults(func=cmd_scheduler_template)

    add("selftest", "re-check the structural safety invariants").set_defaults(func=cmd_selftest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit status. The signal handlers it
    takes over while the command runs are put back before it returns."""
    return _main(argv, exiting=False)


def entry() -> NoReturn:
    """``asi`` and ``python -m agent_skill_improver``: run the process's own
    command line and exit with its status. After a stop the stop signals stay
    ignored until the process has exited (``_stop_signals_unwind``). SIGCHLD
    is put back to its default first (``_put_sigchld_back``)."""
    _put_sigchld_back()
    sys.exit(_main(None, exiting=True))


def _put_sigchld_back() -> None:
    """Put SIGCHLD back to its default action, so each child is left for this
    process to reap and its exit status is read.

    A parent can leave SIGCHLD ignored through ``exec``, and Python does not
    reset it. Measured before, started so on macOS (3.11 and 3.13): the kernel
    did not reap the children then, yet ``getsignal`` reported the ignore, so
    the model runner took a role for one the kernel had reaped and left what it
    started running, 5 calls of 5. Linux keeps the ignore through ``exec`` and
    then does reap each child as it exits (its kernel/signal.c, not run here),
    as macOS does with SIGCHLD ignored in the run, where every child's exit
    then read as 0 (measured: 5 of 5 role answers that exited 3 taken, 5 of 5
    exports that exited 1). The commands the run starts then start with it at
    the default too. Only the process's own command does this: a caller of
    :func:`main` keeps the disposition it has, and while that is an ignore no
    child is started (``ExitUnreadable``) and ``doctor`` names it as a problem.
    """
    sigchld = getattr(signal, "SIGCHLD", None)
    if sigchld is not None and threading.current_thread() is threading.main_thread():
        signal.signal(sigchld, signal.SIG_DFL)


def _main(argv: Sequence[str] | None, *, exiting: bool) -> int:
    parser = build_parser()
    args, extras = parser.parse_known_args(list(argv) if argv is not None else None)
    if extras:
        parser.error(_unrecognized_arguments(extras))
    if not hasattr(args, "json"):
        args.json = False
    if not hasattr(args, "config"):
        args.config = None
    _OPEN_STORES.clear()
    # The handlers stay taken over while a failure is reported, so a stop
    # signal then is a stop too, and one more while the stop is reported is
    # let pass. Measured before: with the handlers put back as the command
    # ended, a burst of 40 SIGTERMs 2 ms apart printed the stop and then
    # ended the run by the signal, exit -15, three times of three.
    with _stop_signals_unwind(exiting=exiting):
        try:
            return _run_and_report(args)
        except Stopped as exc:
            _report_unreported_repairs()
            _err(
                f"stopped by {signal.Signals(exc.signum).name} before the command finished; "
                "any model or source command it had started was ended with it\n"
                + FINISH_BY_RUNNING_AGAIN.get(getattr(args, "command", ""), "")
            )
            return 128 + exc.signum
        finally:
            # Nothing from this command is attributed to the next in-process call.
            _OPEN_STORES.clear()


def _run_and_report(args: argparse.Namespace) -> int:
    """Run the parsed command and turn how it ended into an exit status."""
    # A mode reset the command made and did not report itself is named before
    # whatever ends the command, so the refusal or the traceback never hides it.
    try:
        rc = int(args.func(args))
        _report_unreported_repairs()
        return rc
    except BlockedRunUnreviewed as exc:
        # The condition exit 5 reports is still standing: a scheduler must keep
        # paging on it rather than file the morning after as an ordinary refusal.
        _report_unreported_repairs()
        _err(f"{type(exc).__name__}: {exc}\n")
        return RC_BLOCKED
    except AsiError as exc:
        _report_unreported_repairs()
        _err(f"{type(exc).__name__}: {exc}\n")
        return RC_REFUSED
    except FileNotFoundError as exc:
        _report_unreported_repairs()
        _err(f"not found: {exc}\n")
        return RC_MISSING_FILE
    except BrokenPipeError:  # pragma: no cover - piping into head/less
        # The reader went away; stderr may still be a terminal, so the reset is
        # still named there, unless that pipe is gone too.
        try:
            _report_unreported_repairs()
        except OSError:
            pass
        return RC_OK
    except Exception:
        # Last resort. KeyboardInterrupt and SystemExit are not Exceptions and
        # keep their own exit statuses. Naming the failure and keeping the
        # traceback is what makes this a bug report rather than a silent 1.
        _report_unreported_repairs()
        _err("internal error: this is a bug in asi, not a negative answer\n")
        _err(traceback.format_exc())
        return RC_INTERNAL_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
