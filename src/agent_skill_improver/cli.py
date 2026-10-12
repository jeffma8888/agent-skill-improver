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
    apply               THE ONLY command that writes a skill file. Human-gated.
    rollback            restores a previously applied candidate from its backup.
    scheduler-template  prints a launchd or cron template. Installs nothing.
    selftest            re-checks the structural safety invariants on this checkout.

``apply`` refuses to run when the automation marker is set, which the daily driver
always sets, and it needs both a content-derived token and an explicit
acknowledgement flag.
"""

from __future__ import annotations

import argparse
import ast
import os
import shlex
import sys
import traceback
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .clock import SystemClock, iso
from .config import SOURCE_FIXTURES, Config, load_config, package_dir
from .domain.approval import HumanApproval, automation_active
from .domain.candidate import BIDI_CONTROLS, HIDDEN_CONTROLS, STATUS_VALIDATED, one_line_label
from .domain.isolation import (
    MODE_NONE,
    SOURCE_CLASS_PRIVATE,
    SOURCE_CLASS_SYNTHETIC,
    credential_bearing_env_values,
    decide as decide_isolation,
    denied_env_names,
    source_class,
)
from .errors import AsiError, ConfigError
from .jsonio import dumps
from .app.state import DIR_MODE, OPEN_STATUSES, STATE_FILE_MODE, StateStore
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

BENIGN_RUN_STATUSES = ("ok", "stood-down", "partial")


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
            ch in BIDI_CONTROLS or ch in HIDDEN_CONTROLS or unicodedata.category(ch) in ("Cc", "Cs")
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
            ch in BIDI_CONTROLS or ch in HIDDEN_CONTROLS or unicodedata.category(ch) in ("Cc", "Cs")
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
def cmd_doctor(args: argparse.Namespace) -> int:
    problems: list[str] = []
    warnings: list[str] = []
    try:
        config = _load(args)
    except ConfigError as exc:
        _err(f"config error: {exc}\n")
        return RC_REFUSED
    store = build_skill_store(config)
    missing = [r.id for r in store.missing_roots()]
    if missing:
        warnings.append(f"skill root(s) not present on this machine: {', '.join(missing)}")
    if not store.existing_roots():
        problems.append("no configured skill root exists; nothing can be discovered")
    for role in ("scorer", "improver", "judge"):
        if role not in config.model_roles:
            problems.append(f"no [model.{role}] command configured")
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
    canonical_roots = [r.id for r in config.skill_roots if r.canonical and r.editable]
    if not canonical_roots:
        warnings.append("no skill root is marked canonical+editable; proposals will be refused")
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
    state = build_state(config)

    # The layout is judged before anything touches the state tree, the runner's
    # sandbox scratch directory included: a state root that is a symlink is a
    # finding for this report, not a refusal that pre-empts it. With a bad
    # layout the runner gets no scratch root, so nothing is created through
    # the link on the way to the verdict.
    layout_problems: list[str] = []
    state_present = config.state_dir.exists() or config.state_dir.is_symlink()
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

    # The isolation decision comes BEFORE anything touches the source, so
    # `doctor` on a host with no sandbox mechanism reports the refusal rather
    # than a preflight result that implies the run could proceed.
    runner = build_runner(config, None if layout_problems else state)
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
    if state_present:
        if layout_problems:
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

    source = build_source(config, state)
    source_status = "ok"
    try:
        source.preflight()
    except AsiError as exc:
        source_status = str(exc)
        problems.append(f"source unavailable: {exc}")

    payload = {
        "version": __version__,
        "config_path": str(config.config_path),
        "state_dir": str(config.state_dir),
        "dry_run_default": config.dry_run,
        "source": {
            "kind": config.source.kind,
            "status": source_status,
            "source_class": isolation.source_class,
        },
        "isolation": isolation.to_dict(),
        "state_permissions": {
            "dir_mode": oct(DIR_MODE),
            "file_mode": oct(STATE_FILE_MODE),
            "repaired": permissions_repaired,
            "still_wrong": permission_problems,
            "unsafe_layout": layout_problems,
            "left_alone": left_alone,
        },
        "roles_configured": sorted(config.model_roles),
        "skill_roots": [{**r.to_dict(), "exists": r.path.is_dir()} for r in config.skill_roots],
        "skills_discovered": len(store.known_skill_ids()),
        "skills_that_do_not_load": [{"skill_id": s.skill_id, "path": str(s.path)} for s in broken],
        "skill_documents_passed_over": [d.to_dict() for d in passed_over],
        "candidate_records_unreadable": unreadable_candidates,
        "skills_with_executable_payloads": [
            {"skill_id": s.skill_id, "payload_files": list(s.payload_files)}
            for s in snapshots
            if s.payload_files
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
        f"state perms  dirs {oct(DIR_MODE)} / files {oct(STATE_FILE_MODE)}, "
        f"{len(permissions_repaired)} repaired, {len(permission_problems)} still wrong, "
        f"{len(layout_problems)} not a real directory, {len(left_alone)} left alone",
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
        if group.mirrors:
            head += f"  mirrors={','.join(m.root_id for m in group.mirrors)}"
        if group.divergent_mirrors:
            head += f"  DIVERGENT={','.join(m.root_id for m in group.divergent_mirrors)}"
        lines.append(head)
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


def cmd_discover(args: argparse.Namespace) -> int:
    config = _load(args)
    state = build_state(config)
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
        since=args.since or config.source.since,
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
    lines = [f"{t.thread_id}  {t.workspace}  {t.title[:60]}" for t in found]
    lines.append(f"{len(found)} thread(s) returned, limit {page.limit}")
    lines.append(
        "this is the WHOLE population the source would return"
        if page.is_total
        else "this is a PAGE, not a total: " + page.honesty_note
    )
    _emit(payload, as_json=args.json, lines=lines)
    return 0


def cmd_daily(args: argparse.Namespace) -> int:
    from .app.daily import run_daily

    config = _load(args)
    if not args.allow_missing_roles:
        config.require_roles("scorer", "improver", "judge")
    state = build_state(config)
    runner = build_runner(config, state)
    # run_daily enforces this too -- that is the load-bearing check, since it is
    # the one a library caller cannot skip. Enforcing here as well means the CLI
    # refuses before `source.preflight()` runs a subprocess.
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
    )
    # Notes carry skill ids and source errors, so the text goes out through
    # _emit like every other listing: one line each, breaks spelled.
    lines = [receipt.summary_line(), *(f"note: {note}" for note in receipt.notes)]
    if receipt.dry_run:
        lines.append("receipt: not written (dry run); --json prints it, --commit persists it")
    else:
        lines.append(f"receipt: {state.receipt_path(receipt.day, dry_run=False)}")
    _emit(receipt.to_dict(), as_json=args.json, lines=lines)
    return exit_code_for(receipt.status)


def cmd_candidates(args: argparse.Namespace) -> int:
    config = _load(args)
    state = StateStore(config.state_dir)
    scan = state.scan_candidates()
    items = (
        list(scan.readable) if args.all else [c for c in scan.readable if c.status in OPEN_STATUSES]
    )
    payload = {
        "unreadable": [u.to_dict() for u in scan.unreadable],
        "candidates": [
            {
                "id": c.id,
                "status": c.status,
                "skill_id": c.target.skill_id,
                "canonical_path": c.target.canonical_path,
                "gate_passed": bool((c.validation or {}).get("gate", {}).get("passed")),
                "added_lines": c.patch.added_lines,
                "removed_lines": c.patch.removed_lines,
                "created_at": c.created_at,
            }
            for c in items
        ],
    }
    lines = [
        f"{c.id}  {c.status:<12} {c.target.skill_id:<28} "
        f"gate={'pass' if (c.validation or {}).get('gate', {}).get('passed') else 'fail'}"
        f"{' (closed)' if c.status != STATUS_VALIDATED else ''} "
        f"+{c.patch.added_lines}/-{c.patch.removed_lines}"
        for c in items
    ]
    # A record that did not read is listed by name: dropping it made a corrupt
    # queue look like an empty one (see ``UnreadableRecord``).
    for record in scan.unreadable:
        lines.append(f"UNREADABLE  {record.path.name}  {record.reason}")
    lines.append(f"{len(items)} candidate(s); {len(scan.unreadable)} unreadable record(s)")
    _emit(payload, as_json=args.json, lines=lines)
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    from .app.promote import read_live_base_text, review_report

    config = _load(args)
    state = StateStore(config.state_dir)
    candidate = state.read_candidate(args.candidate_id)
    # A review-only command still resolves the candidate against a real store:
    # read-only, never writable, but reading the LIVE configured canonical
    # snapshot rather than trusting anything the candidate record asserts about
    # its own base. Without this, review can only check the record against
    # itself, and a diff or its line counts edited to agree with each other but
    # not with the file they claim to describe would still print a token.
    store = build_skill_store(config, writable=False)
    base_text = read_live_base_text(store, candidate)
    rubrics = {r.id: r for r in config.rubrics}
    report = review_report(candidate, base_text=base_text, rubrics=rubrics)
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
        out += [one_line(f"  - {p}") for p in report["mirror_paths"]]
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
    state = build_state(config)
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
    if args.json:
        _say_json(result)
    else:
        write = result["write"]
        _say(
            "\n".join(
                [
                    f"applied {result['candidate_id']} to {write['path']}",
                    f"  previous sha256 {write['previous_sha256']}",
                    f"  new sha256      {write['new_sha256']}",
                    f"  backup          {write['backup_path']}",
                    "  rollback        " + shlex.join(["asi", "rollback", result["candidate_id"]]),
                    f"  (or manually)   {write['rollback_command']}",
                ]
            )
            + "\n"
        )
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    from .app.promote import rollback_candidate

    config = _load(args)
    state = build_state(config)
    candidate = state.read_candidate(args.candidate_id)
    store = build_skill_store(config, writable=True)
    result = rollback_candidate(candidate, skill_store=store, state=state, clock=SystemClock())
    if args.json:
        _say_json(result)
    else:
        _say(f"rolled back {result['candidate_id']}: {result['result']}\n")
    return 0


def _cron_quote(value: str) -> str:
    """Spell ``value`` for a crontab command.

    Shell-quoted first. Then for cron itself, which rewrites the command before
    any shell sees it and knows nothing of quotes: a bare ``%`` becomes a
    newline, and a backslash escapes the character after it and is consumed. So
    every backslash is doubled (cron collapses the pair back to one) and every
    ``%`` is escaped (cron hands the shell the percent sign). Vixie cron, cronie
    and macOS share this loop verbatim; busybox crond has no such rewriting.
    """
    return shlex.quote(value).replace("\\", "\\\\").replace("%", "\\%")


def _xml_quote(value: str) -> str:
    """Spell ``value`` for the text of an XML element."""
    from xml.sax.saxutils import escape

    return escape(value)


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
    for marker, value in (
        ("__REPO__", str(repo_root())),
        ("__PYTHON__", sys.executable),
        ("__CONFIG__", str(config_path)),
        ("__STATE__", str(config.state_dir)),
    ):
        text = text.replace(marker, quote(value))
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
            "app/daily.py must not import app.promote; promotion is a human-only path",
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
            "token derived from a stored hash cannot bind the text a human reviewed",
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
class _EscapingParser(argparse.ArgumentParser):
    """argparse writes usage and errors to the terminal itself, and its
    "unrecognized arguments" message repeats argv raw. Subparsers inherit this
    class, so the one override covers every message argparse prints."""

    def _print_message(self, message: str, file: Any = None) -> None:
        if message:
            (file or sys.stderr).write(escape_for_terminal(message))


def build_parser() -> argparse.ArgumentParser:
    parser = _EscapingParser(
        prog="asi",
        description=(
            "Safe, human-gated daily self-improvement loop for agent skills. "
            "Automation scores, isolates, proposes and validates; only a human promotes."
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
        return sub.add_parser(name, help=help_text, parents=[common])

    add("doctor", "validate configuration and report what is wired").set_defaults(func=cmd_doctor)
    add("skills", "skill inventory: canonical vs mirror, discovery warnings").set_defaults(
        func=cmd_skills
    )
    add("rubrics", "show loaded rubrics").set_defaults(func=cmd_rubrics)

    p_discover = add("discover", "list conversations the source would ingest")
    p_discover.add_argument("--since", help="time window, e.g. 1d, 7d, 2026-08-01")
    p_discover.add_argument("--limit", type=int)
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
    p_daily.add_argument("--day", help="run id / idempotence key, defaults to today (UTC)")
    p_daily.add_argument("--seed", help="deterministic seed for nonces and blind judging")
    p_daily.add_argument(
        "--allow-missing-roles",
        action="store_true",
        help="do not fail fast when a model role is unconfigured (for partial runs)",
    )
    p_daily.set_defaults(func=cmd_daily)

    p_candidates = add("candidates", "list candidate proposals")
    p_candidates.add_argument("--all", action="store_true", help="include applied and rejected")
    p_candidates.set_defaults(func=cmd_candidates)

    p_review = add("review", "print a candidate's full argument, diff and approval token")
    p_review.add_argument("candidate_id")
    p_review.set_defaults(func=cmd_review)

    p_apply = add("apply", "THE ONLY command that writes a skill file (human-gated)")
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

    p_rollback = add("rollback", "restore an applied candidate from its backup")
    p_rollback.add_argument("candidate_id")
    p_rollback.set_defaults(func=cmd_rollback)

    p_sched = add("scheduler-template", "print a launchd or cron template (installs nothing)")
    p_sched.add_argument("flavor", choices=("launchd", "cron"))
    p_sched.set_defaults(func=cmd_scheduler_template)

    add("selftest", "re-check the structural safety invariants").set_defaults(func=cmd_selftest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not hasattr(args, "json"):
        args.json = False
    if not hasattr(args, "config"):
        args.config = None
    try:
        return int(args.func(args))
    except AsiError as exc:
        _err(f"{type(exc).__name__}: {exc}\n")
        return RC_REFUSED
    except FileNotFoundError as exc:
        _err(f"not found: {exc}\n")
        return RC_MISSING_FILE
    except BrokenPipeError:  # pragma: no cover - piping into head/less
        return RC_OK
    except Exception:
        # Last resort. KeyboardInterrupt and SystemExit are not Exceptions and
        # keep their own exit statuses. Naming the failure and keeping the
        # traceback is what makes this a bug report rather than a silent 1.
        _err("internal error: this is a bug in asi, not a negative answer\n")
        _err(traceback.format_exc())
        return RC_INTERNAL_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
