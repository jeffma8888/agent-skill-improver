"""TOML configuration: sources, model roles, skill roots, rubrics, caps, gate.

Two decisions here are load-bearing rather than cosmetic.

First, *roles are configured independently*. ``scorer``, ``improver`` and
``judge`` are separate commands, so the judge can be a different model from the
improver. A judge that is the same model as the improver is grading its own
homework, and the gate is worth less than it looks.

Second, *protected paths are derived, not configured*. This package's own
source, its config file and its rubric directory are always protected from the
write path. If that list were configurable, a proposal could widen it, and a
loop that can rewrite its own scoring criteria has no gate at all.
"""

from __future__ import annotations

import difflib
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .adapters.conversation_cli import (
    DEFAULT_DISCOVERY_ARGV,
    DEFAULT_EXPORT_ARGV,
    DEFAULT_QUERY_ARGV,
    DEFAULT_WORKSPACE_ARGV,
)
from .adapters.bounded_process import MAX_TIMEOUT_SECONDS
from .adapters.model_command import RoleCommand, role_command_from_mapping
from .domain.authorship import DEFAULT_AGENT_TITLE_MARKERS
from .domain.redaction import QUOTED_VALUE_CHARS, quoted_value, redact
from .domain.isolation import source_class
from .domain.rubric import Rubric
from .domain.skills import SkillRoot
from .domain.validation import GatePolicy
from .errors import ConfigError
from .hashing import sha256_text
from .jsonio import MAX_STORED_INT

ROLE_SCORER = "scorer"
ROLE_IMPROVER = "improver"
ROLE_JUDGE = "judge"
ROLES = (ROLE_SCORER, ROLE_IMPROVER, ROLE_JUDGE)

#: A command-line conversation tool, driven through configurable argv templates.
#: Reads real conversation history, so it is classified PRIVATE and carries the
#: sandbox requirement (see :mod:`agent_skill_improver.domain.isolation`).
SOURCE_CLI = "conversation-cli"

#: Export-shaped JSON on disk. The synthetic corpus this repository ships uses
#: it, and it is the only kind classified as non-private.
SOURCE_FIXTURES = "fixtures"

SOURCE_KINDS = (SOURCE_CLI, SOURCE_FIXTURES)

# Only dimensions that are known BEFORE a thread is fetched belong here: selection has
# to bound the batch before paying to export anything.
#
# ``class`` is the default and it is the load-bearing one. In a corpus where an
# orchestrator generates most of the traffic, bounding by anything else drowns the
# scarce human sessions -- which are the only ones where a skill actually
# mis-served a person.
FAIRNESS_DIMENSIONS = ("class", "workspace", "date")


#: A ``$NAME`` or ``${...}`` that ``os.path.expandvars`` left in place: an unset
#: variable, or a braced form it does not understand at all (``${NAME:-x}`` and
#: the other shell modifiers), which it also keeps literally.
_UNEXPANDED_VARIABLE = re.compile(r"\$(?:\{([^}]*)\}?|([A-Za-z_][A-Za-z0-9_]*))")


def _expand(raw: str, *, base: Path) -> Path:
    """A configured path with ``~`` and ``$VARIABLE`` expanded, or a refusal.

    ``expandvars`` keeps an unset variable as literal text, so a path such as
    ``$ASI_HOME/state`` under a scheduler whose environment lacks the variable
    would quietly become a directory named ``$ASI_HOME``: a second, private
    state tree the operator never looks at, holding the candidates they are
    waiting to review. Measured before this check existed. The same goes for a
    ``~user`` that names no user. Both are refused with the name, before any
    path is created.
    """
    expanded = os.path.expandvars(os.path.expanduser(str(raw)))
    leftover = _UNEXPANDED_VARIABLE.search(expanded)
    if leftover:
        shown = leftover.group(0)
        raise ConfigError(
            f"path {raw!r} still contains {shown} after expansion: the variable is not set "
            "here, or the form is one the expansion does not understand. Refusing to use the "
            "literal text as a directory. A scheduled run has a smaller environment than an "
            "interactive shell, so set the variable where the run happens, spell the plain "
            "$NAME form, or write the path out."
        )
    if expanded.startswith("~"):
        raise ConfigError(
            f"path {raw!r} starts with a ~ that names no known user; refusing to use the "
            "literal text as a directory."
        )
    path = Path(expanded)
    return path if path.is_absolute() else (base / path)


def _state_dir_as_written(path: Path) -> Path:
    """The configured state directory with its ancestors resolved and its last
    component left exactly as written.

    ``Path.resolve()`` on the whole path followed a link *at* the state directory
    and handed ``StateStore`` the link's target, so the store's own refusal of a
    symlinked root never fired: ``daily`` built the private tree inside whatever
    the link pointed at and chmodded that directory to owner-only. The ancestors
    are the operator's filesystem layout (``/tmp`` is a link on macOS) and are
    resolved as before; the leaf is the thing a precreated link would replace,
    and the store judges it with ``lstat``.
    """
    if path.name in ("", ".", ".."):
        return path.resolve()
    return path.parent.resolve() / path.name


#: Every key each table accepts. ``[source]`` has its own closed set below with
#: its own reason; these close the rest. A key this tool does not read is
#: refused rather than ignored, because an ignored key looks effective: measured
#: before this existed, ``[caps].max_open_candidate = 1`` (a dropped letter)
#: loaded without a word and the run kept the default cap of 12, while the
#: operator believed the queue was held to one.
TABLE_KEYS: dict[str, frozenset[str]] = {
    "run": frozenset({"state_dir", "dry_run", "seed", "base_dir"}),
    "selection": frozenset(
        {"max_conversations_per_run", "fairness_dimension", "agent_title_markers"}
    ),
    "caps": frozenset(
        {
            "max_candidates_per_run",
            "max_open_candidates",
            "max_failures_per_candidate",
            "max_skills_per_run",
            "max_extra_holdout_conversations",
            "max_defer_attempts",
            "max_open_candidates_per_skill",
        }
    ),
    "prompt": frozenset(
        {"max_transcript_chars", "max_message_chars", "max_skill_text_chars", "redact_secrets"}
    ),
    "gate": frozenset(
        {
            "min_fixes",
            "max_regressions",
            "max_holdout_degradations",
            "min_holdout_cases",
            "min_judge_preference_ratio",
            "require_judge",
            "require_independent_judge",
        }
    ),
    "guard": frozenset({"protected_skill_ids"}),
    "rubrics": frozenset({"dir", "enabled"}),
    "skill_root": frozenset({"id", "path", "kind", "precedence", "canonical", "editable", "note"}),
    "model.*": frozenset(
        {"argv", "command", "env", "timeout_seconds", "max_output_bytes", "model_label"}
    ),
}

#: The tables a config file may hold at the top level.
TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "run",
        "source",
        "selection",
        "caps",
        "prompt",
        "gate",
        "guard",
        "rubrics",
        "skill_root",
        "model",
    }
)


def _refuse_unknown_keys(table: Mapping[str, Any], accepted: frozenset[str], *, where: str) -> None:
    """Refuse a key this tool would not read, naming the nearest key it would."""
    unknown = sorted(set(table) - accepted)
    if not unknown:
        return
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(accepted), n=1, cutoff=0.6)
        hints.append(f"{key} (did you mean {close[0]}?)" if close else key)
    raise ConfigError(
        f"{where} has unknown key(s): {', '.join(hints)}. Accepted keys are "
        f"{', '.join(sorted(accepted))}. An unknown key is refused rather than ignored, "
        "because a misspelt setting that loads silently leaves the default in force while "
        "the file says otherwise."
    )


def _require_table(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{key}] must be a table")
    return value


def _int_field(
    table: Mapping[str, Any],
    key: str,
    default: int,
    *,
    where: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    value = table.get(key, default)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bound = "" if maximum is None else f" and <= {maximum}"
        raise ConfigError(f"[{where}].{key} must be an integer >= {minimum}{bound}")
    return value


def _bool_field(table: Mapping[str, Any], key: str, default: bool, *, where: str) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"[{where}].{key} must be true or false")
    return value


def _str_field(table: Mapping[str, Any], key: str, default: str, *, where: str) -> str:
    value = table.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"[{where}].{key} must be a string")
    return value


#: Every key ``[source]`` accepts. The table is validated against this and an
#: unknown key is a hard error, which is the mechanical reason no operator can
#: introduce a trust label: there is nowhere to put one. A schema that ignored
#: unknown keys would let ``trusted = true`` sit in a config file looking
#: effective while the sandbox decision was made somewhere else entirely.
SOURCE_KEYS: frozenset[str] = frozenset(
    {
        "kind",
        "discovery_argv",
        "query_argv",
        "workspace_argv",
        "export_argv",
        "since",
        "discovery_limit",
        "query",
        "workspaces",
        "keep_raw",
        "fixtures_dir",
        "max_messages",
        "output_excerpt_chars",
        "timeout_seconds",
    }
)


@dataclass(frozen=True)
class SourceConfig:
    kind: str = SOURCE_FIXTURES
    discovery_argv: tuple[str, ...] = DEFAULT_DISCOVERY_ARGV
    query_argv: tuple[str, ...] = DEFAULT_QUERY_ARGV
    workspace_argv: tuple[str, ...] = DEFAULT_WORKSPACE_ARGV
    export_argv: tuple[str, ...] = DEFAULT_EXPORT_ARGV
    since: str = "1d"
    discovery_limit: int = 40
    query: str = ""
    workspaces: tuple[str, ...] = ()
    keep_raw: bool = True
    fixtures_dir: Path | None = None
    max_messages: int | None = None
    output_excerpt_chars: int = 2000
    timeout_seconds: int = 180

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "discovery_argv": list(self.discovery_argv),
            "query_argv": list(self.query_argv),
            "workspace_argv": list(self.workspace_argv),
            "export_argv": list(self.export_argv),
            "since": self.since,
            "discovery_limit": self.discovery_limit,
            "query": self.query,
            "workspaces": list(self.workspaces),
            "keep_raw": self.keep_raw,
            "fixtures_dir": str(self.fixtures_dir) if self.fixtures_dir else "",
            "max_messages": self.max_messages,
            "output_excerpt_chars": self.output_excerpt_chars,
            "timeout_seconds": self.timeout_seconds,
            "source_class": source_class(self.kind),
        }


@dataclass(frozen=True)
class SelectionConfig:
    max_conversations_per_run: int = 8
    fairness_dimension: str = "class"
    agent_title_markers: tuple[str, ...] = DEFAULT_AGENT_TITLE_MARKERS

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_conversations_per_run": self.max_conversations_per_run,
            "fairness_dimension": self.fairness_dimension,
            "agent_title_markers": list(self.agent_title_markers),
        }


@dataclass(frozen=True)
class CapsConfig:
    max_candidates_per_run: int = 3
    max_open_candidates: int = 12
    max_failures_per_candidate: int = 6
    max_skills_per_run: int = 3
    max_extra_holdout_conversations: int = 3
    max_defer_attempts: int = 3
    max_open_candidates_per_skill: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_candidates_per_run": self.max_candidates_per_run,
            "max_open_candidates": self.max_open_candidates,
            "max_failures_per_candidate": self.max_failures_per_candidate,
            "max_skills_per_run": self.max_skills_per_run,
            "max_extra_holdout_conversations": self.max_extra_holdout_conversations,
            "max_defer_attempts": self.max_defer_attempts,
            "max_open_candidates_per_skill": self.max_open_candidates_per_skill,
        }


@dataclass(frozen=True)
class PromptConfig:
    max_transcript_chars: int = 24000
    max_message_chars: int = 4000
    max_skill_text_chars: int = 24000
    redact_secrets: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_transcript_chars": self.max_transcript_chars,
            "max_message_chars": self.max_message_chars,
            "max_skill_text_chars": self.max_skill_text_chars,
            "redact_secrets": self.redact_secrets,
        }


@dataclass(frozen=True)
class Config:
    config_path: Path
    base_dir: Path
    state_dir: Path
    dry_run: bool
    seed: str
    source: SourceConfig
    selection: SelectionConfig
    caps: CapsConfig
    prompt: PromptConfig
    gate: GatePolicy
    skill_roots: tuple[SkillRoot, ...]
    rubrics: tuple[Rubric, ...]
    model_roles: dict[str, RoleCommand] = field(default_factory=dict)
    protected_paths: tuple[Path, ...] = ()
    protected_skill_ids: tuple[str, ...] = ()

    def rubric(self, rubric_id: str) -> Rubric | None:
        for r in self.rubrics:
            if r.id == rubric_id:
                return r
        return None

    def evaluator_independence(self) -> dict[str, Any]:
        """Which configured roles are actually distinct commands.

        Reported on every run receipt. The one that decides the gate is
        ``judge_vs_improver``: a judge sharing the improver's command is grading
        its own homework, and calling that a blind preference would be dishonest.
        """

        def fingerprint(role: str) -> tuple[Any, ...] | None:
            """What actually executes for this role.

            ``model_label`` is deliberately EXCLUDED. It is free text an operator
            types, so including it meant two roles running the byte-identical
            command counted as independent purely because they were labelled
            differently -- a cosmetic field able to satisfy a safety gate. The
            comparison must be over the thing that runs.
            """
            cmd = self.model_roles.get(role)
            if cmd is None:
                return None
            return (tuple(cmd.argv),)

        def label(role: str) -> str:
            cmd = self.model_roles.get(role)
            return cmd.model_label if cmd is not None else ""

        scorer, improver, judge = (
            fingerprint(ROLE_SCORER),
            fingerprint(ROLE_IMPROVER),
            fingerprint(ROLE_JUDGE),
        )

        def distinct(a: tuple[Any, ...] | None, b: tuple[Any, ...] | None) -> bool | None:
            if a is None or b is None:
                return None
            return a != b

        judge_vs_improver = distinct(judge, improver)
        return {
            "roles_configured": sorted(self.model_roles),
            "judge_vs_improver_distinct": judge_vs_improver,
            "judge_vs_scorer_distinct": distinct(judge, scorer),
            "scorer_vs_improver_distinct": distinct(scorer, improver),
            # Reported for the reader, and deliberately NOT part of the decision:
            # a label is a claim about a command, not a property of one.
            "judge_label": label(ROLE_JUDGE),
            "improver_label": label(ROLE_IMPROVER),
            "labels_distinct": label(ROLE_JUDGE) != label(ROLE_IMPROVER),
            # Unknown independence is treated as NOT independent: an unconfigured
            # judge cannot supply independent evidence either.
            "judge_independent": bool(judge_vs_improver),
            "note": (
                "correlated evaluators are the quiet failure here: if the judge shares a model "
                "with the improver, a passing gate mostly measures self-consistency. This check "
                "compares the COMMAND each role runs, which is a weak proxy: two different "
                "commands can wrap the same model, and two models from one family can share the "
                "failure. A passing check means the obvious mistake was avoided, not that the "
                "judge is genuinely independent. model_label is reported but never counted, "
                "because a label is a claim about a command rather than a property of one."
            ),
        }

    def judge_is_independent(self) -> bool:
        return bool(self.evaluator_independence()["judge_independent"])

    def require_roles(self, *roles: str) -> None:
        missing = [r for r in roles if r not in self.model_roles]
        if missing:
            raise ConfigError(
                "no model command configured for role(s): "
                + ", ".join(missing)
                + ". Add a [model.<role>] table with argv = [...]; see config/example.config.toml."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "base_dir": str(self.base_dir),
            "state_dir": str(self.state_dir),
            "dry_run": self.dry_run,
            "seed": self.seed,
            "source": self.source.to_dict(),
            "selection": self.selection.to_dict(),
            "caps": self.caps.to_dict(),
            "prompt": self.prompt.to_dict(),
            "gate": self.gate.to_dict(),
            "skill_roots": [r.to_dict() for r in self.skill_roots],
            "rubrics": [
                {"id": r.id, "version": r.version, "criteria": list(r.criterion_ids)}
                for r in self.rubrics
            ],
            "model_roles": {k: v.to_dict() for k, v in sorted(self.model_roles.items())},
            "protected_paths": [str(p) for p in self.protected_paths],
            "protected_skill_ids": list(self.protected_skill_ids),
            "evaluator_independence": self.evaluator_independence(),
        }


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def _base_dir_for(config_path: Path, run_table: Mapping[str, Any]) -> Path:
    declared = run_table.get("base_dir")
    if isinstance(declared, str) and declared.strip():
        return _expand(declared, base=config_path.parent).resolve()
    parent = config_path.parent
    # A config that lives in a `config/` directory belongs to the repo above it.
    if parent.name == "config":
        return parent.parent.resolve()
    return parent.resolve()


def _read_utf8_or_refuse(path: Path, *, what: str) -> str:
    """Read a definition file as UTF-8 text, or say why it cannot be read.

    ``is_file`` has already passed when this runs, so the file exists; what can
    still fail is the read itself (a mode the process cannot open, a directory
    that lost its execute bit on the way) or the decode. Measured before this
    helper: a config file at mode 000 ended ``doctor`` as an internal error with
    a traceback, the one exit code that is meant to signal a bug in this tool.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"{what} {path} is not UTF-8 text ({exc.reason} at byte {exc.start})"
        ) from exc
    except OSError as exc:
        raise ConfigError(f"{what} {path} cannot be read ({exc.strerror or exc})") from exc


def _document_nodes(raw: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    """Every value of a parsed TOML document, tables and arrays included, with
    the dotted name that reaches it (``caps.max_open_candidates``,
    ``criterion[0].weight``), in document order. Walked with a stack, since a
    dotted key can nest tables as deep as the file is long."""
    stack: list[tuple[str, Any]] = [("", raw)]
    while stack:
        where, value = stack.pop()
        yield where, value
        if isinstance(value, Mapping):
            stack.extend(
                (f"{where}.{key}" if where else key, item)
                for key, item in reversed(list(value.items()))
            )
        elif isinstance(value, list):
            stack.extend((f"{where}[{i}]", item) for i, item in reversed(list(enumerate(value))))


#: A dotted name a refusal may print bare: the characters TOML's bare keys
#: and ``_document_nodes``' separators use, nothing a terminal acts on.
_PLAIN_NAME = re.compile(r"[A-Za-z0-9_.\[\]-]+")


def _named(where: str) -> str:
    """A dotted name from ``_document_nodes`` as a refusal prints it.

    Bare when it is plain, short and nothing in it is masked, else
    ``quoted_value``: a quoted TOML key may hold a line break or a terminal
    escape, a dotted key may nest as deep as the file is long, and a key is
    where a pasted token lands when it is used as an env name. Measured before:
    a raw line break and a raw ``ESC[31m`` reached the message as written, a
    5,000-character key came back whole, a key nested 50,000 deep made a 339 KB
    message, and a ``ghp_`` or ``sk-ant-`` token used as an env name was
    printed whole, where the ``=`` refusal masked the same name.
    """
    if (
        len(where) <= QUOTED_VALUE_CHARS
        and _PLAIN_NAME.fullmatch(where)
        and redact(where).text == where
    ):
        return where
    return quoted_value(where)


def _refuse_unstorable_integers(raw: Mapping[str, Any], source: Path) -> None:
    """Refuse an integer, anywhere in a config or rubric, that no stored record holds.

    A run writes the numbers it was configured with into its receipt (the caps,
    the selection and source bounds), its candidates (the gate's counts, each
    rubric's version), and ``loads_untrusted`` refuses an integer past
    ``MAX_STORED_INT`` when it reads them back. Measured before this check, with
    that refusal in place: a 401-digit ``[caps]`` or ``[source]`` bound let
    ``doctor`` say ready and the first ``daily --commit`` exit 0, and every
    later ``daily`` was refused over the receipt it had written, even after
    the config was fixed; a 401-digit rubric ``version`` or ``[gate]`` count
    made the candidates it wrote unreadable. The field is named (``_named``),
    the number only by its length. The walk runs before any key is checked, so
    a key no setting reads is refused here too: nothing could take the number.
    """
    for where, value in _document_nodes(raw):
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > MAX_STORED_INT:
            raise ConfigError(
                f"{source}: {_named(where)} is an integer of {len(str(abs(value)))} digits, "
                "larger than any number a stored record holds (about 1.8e308), so no setting "
                "can take it: a run writes the numbers it is configured with into records, "
                "and no later command could read such a record. Use a smaller number."
            )


def _refuse_nul(raw: Mapping[str, Any], config_path: Path) -> None:
    """Refuse a NUL in any key or string of the config, at any depth.

    Every string a config holds is a path, a word of a command line, an
    environment entry, a value rendered into one of those, or a label printed
    beside them, and the platform takes each as a NUL-terminated string. TOML
    can spell one (``"\\u0000"``), and Python refuses it only at the call that
    uses it, with a ``ValueError`` (``embedded null byte``; for a path on 3.13,
    ``lstat: embedded null character in path``). Measured before: a NUL in
    ``[run].state_dir``, ``[run].base_dir``, ``[rubrics].dir`` or a
    ``[[skill_root]]`` path made ``doctor`` and ``daily`` fall there, exit 70;
    in a word of a ``[model.*]`` or ``[source]`` command, or in a
    ``[model.*].env`` entry, ``doctor`` said ready and ``daily`` fell at its
    first call. The field is named (``_named``) and its value never shown: an
    environment entry may be a credential. Walked in document order
    (``_document_nodes``).
    """
    for where, value in _document_nodes(raw):
        if isinstance(value, str):
            if "\x00" in value:
                raise ConfigError(
                    f"{config_path}: {_named(where)} holds a NUL character (\\x00), which no "
                    "path, command word or environment entry can carry; it would be refused at "
                    "the first call that used it. Remove it."
                )
        elif isinstance(value, Mapping):
            for key in value:
                if "\x00" in key:
                    raise ConfigError(
                        f"{config_path}: a key in {_named(where) if where else 'the top level'} holds a "
                        "NUL character (\\x00), which no path, command word or environment entry "
                        "can carry. Remove it."
                    )


def load_config(path: str | Path, *, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load and validate a config file. Raises :class:`ConfigError`."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = tomllib.loads(_read_utf8_or_refuse(config_path, what="config file"))
    except ValueError as exc:
        # TOMLDecodeError is one; so is the plain ValueError ``int`` raises on
        # more digits than Python reads from text, which tomllib lets through.
        raise ConfigError(f"{config_path}: invalid TOML: {exc}") from exc
    _refuse_unstorable_integers(raw, config_path)
    _refuse_nul(raw, config_path)

    over = dict(overrides or {})
    _refuse_unknown_keys(raw, TOP_LEVEL_KEYS, where="the config file")
    for name in ("run", "selection", "caps", "prompt", "gate", "guard", "rubrics"):
        _refuse_unknown_keys(_require_table(raw, name), TABLE_KEYS[name], where=f"[{name}]")
    run_table = _require_table(raw, "run")
    base_dir = _base_dir_for(config_path, run_table)
    state_dir = _state_dir_as_written(
        _expand(_str_field(run_table, "state_dir", "state", where="run"), base=base_dir)
    )
    dry_run = _bool_field(run_table, "dry_run", True, where="run")
    if "dry_run" in over and over["dry_run"] is not None:
        dry_run = bool(over["dry_run"])
    seed = _str_field(run_table, "seed", "", where="run")

    source = _load_source(_require_table(raw, "source"), base_dir=base_dir)
    selection = _load_selection(_require_table(raw, "selection"))
    caps = _load_caps(_require_table(raw, "caps"))
    prompt = _load_prompt(_require_table(raw, "prompt"))
    gate = _load_gate(_require_table(raw, "gate"))
    skill_roots = _load_skill_roots(raw.get("skill_root", []), base_dir=base_dir)
    rubrics = _load_rubrics(
        _require_table(raw, "rubrics"), config_dir=config_path.parent, base_dir=base_dir
    )
    model_roles = _load_model_roles(_require_table(raw, "model"))

    # One labelled set, so the load-time check below and the write path's
    # refusal cannot drift apart by a directory added to one and not the other.
    protected_dirs = {
        "this tool's own package": package_dir(),
        "the config directory": config_path.parent,
        "the state directory": state_dir,
    }
    rub_dir = _require_table(raw, "rubrics").get("dir", "rubrics")
    if isinstance(rub_dir, str):
        protected_dirs["the rubrics directory"] = _expand(rub_dir, base=config_path.parent)
    protected = [config_path, *protected_dirs.values()]
    _assert_editable_roots_outside_protected(skill_roots, protected_dirs)
    guard_table = _require_table(raw, "guard")
    protected_skill_ids = guard_table.get("protected_skill_ids", [])
    if not isinstance(protected_skill_ids, list) or any(
        not isinstance(x, str) for x in protected_skill_ids
    ):
        raise ConfigError("[guard].protected_skill_ids must be a list of strings")

    return Config(
        config_path=config_path,
        base_dir=base_dir,
        state_dir=state_dir,
        dry_run=dry_run,
        seed=seed,
        source=source,
        selection=selection,
        caps=caps,
        prompt=prompt,
        gate=gate,
        skill_roots=skill_roots,
        rubrics=rubrics,
        model_roles=model_roles,
        protected_paths=tuple(dict.fromkeys(p.resolve() for p in protected)),
        protected_skill_ids=tuple(protected_skill_ids),
    )


def _argv_field(
    table: Mapping[str, Any],
    key: str,
    default: tuple[str, ...],
    *,
    required_placeholders: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Read an argv template and check it renders the values it must carry."""
    raw = table.get(key, list(default))
    if not isinstance(raw, list) or not raw or any(not isinstance(x, str) for x in raw):
        raise ConfigError(f"[source].{key} must be a non-empty list of strings")
    argv = tuple(raw)
    for index, part in enumerate(argv):
        if part.strip():
            continue
        # The renderer keeps a literal argument as written, so a blank one is
        # sent to the conversation command verbatim: a request nobody configured.
        raise ConfigError(
            f"[source].{key} element #{index} (counting from 0, value {part!r}) is empty or "
            "whitespace: it would be sent to the conversation command as an argument in that "
            "position. Remove it, or give it the value it was meant to carry."
        )
    text = " ".join(argv)
    missing = [ph for ph in required_placeholders if ph not in text]
    if missing:
        raise ConfigError(
            f"[source].{key} must contain {', '.join(missing)}. "
            "A template that omits a required placeholder silently sends a different "
            "request than the one configured -- an omitted limit, for instance, means "
            "the tool's own default page size decides what gets sampled."
        )
    return argv


def _load_source(table: Mapping[str, Any], *, base_dir: Path) -> SourceConfig:
    unknown = sorted(set(table) - SOURCE_KEYS)
    if unknown:
        raise ConfigError(
            f"[source] has unknown key(s): {', '.join(unknown)}. Accepted keys are "
            f"{', '.join(sorted(SOURCE_KEYS))}. This table is a closed schema on purpose: "
            "whether the model roles must run inside an OS no-egress sandbox is derived "
            "from [source].kind, and an unknown key is refused so a trust label cannot be "
            "introduced here and mistaken for one that has an effect."
        )
    kind = _str_field(table, "kind", SOURCE_FIXTURES, where="source")
    if kind not in SOURCE_KINDS:
        raise ConfigError(f"[source].kind must be one of {SOURCE_KINDS}, got {kind!r}")
    fixtures_raw = _str_field(table, "fixtures_dir", "", where="source")
    fixtures_dir = _expand(fixtures_raw, base=base_dir) if fixtures_raw else None
    if kind == SOURCE_FIXTURES and fixtures_dir is None:
        raise ConfigError('[source].fixtures_dir is required when kind = "fixtures"')
    workspaces = table.get("workspaces", [])
    if not isinstance(workspaces, list) or any(not isinstance(x, str) for x in workspaces):
        raise ConfigError("[source].workspaces must be a list of strings")
    for index, entry in enumerate(workspaces):
        if entry.strip():
            continue
        # An empty entry is skipped by the renderer, so a configured workspace
        # would silently never be searched; a blank one is sent as a name.
        raise ConfigError(
            f"[source].workspaces entry #{index} (counting from 0, value {entry!r}) is empty or "
            "whitespace: it would either be skipped, so that workspace is never searched, or "
            "sent to the discovery command as a workspace name. Remove it or name the workspace."
        )
    since = _str_field(table, "since", "1d", where="source").strip()
    if not since:
        # The renderer drops an argument that renders empty, and with it the
        # flag that introduced it, so the request sent would carry no window at
        # all: a different request from the one configured, and an unbounded one.
        raise ConfigError(
            "[source].since is empty or whitespace: the discovery command would be sent with no "
            "window on it, a different request from the one configured (the limit still bounds "
            "the page, the window no longer does). Set a window the discovery tool understands, "
            'such as "1d".'
        )
    max_messages = _int_field(table, "max_messages", 0, where="source")
    return SourceConfig(
        kind=kind,
        discovery_argv=_argv_field(
            table,
            "discovery_argv",
            DEFAULT_DISCOVERY_ARGV,
            required_placeholders=("{since}", "{limit}"),
        ),
        query_argv=_argv_field(
            table, "query_argv", DEFAULT_QUERY_ARGV, required_placeholders=("{query}",)
        ),
        workspace_argv=_argv_field(
            table,
            "workspace_argv",
            DEFAULT_WORKSPACE_ARGV,
            required_placeholders=("{workspace}",),
        ),
        export_argv=_argv_field(
            table, "export_argv", DEFAULT_EXPORT_ARGV, required_placeholders=("{thread_id}",)
        ),
        since=since,
        discovery_limit=_int_field(table, "discovery_limit", 40, where="source", minimum=1),
        # A query of only whitespace is no query; sent as written it would be
        # a positional that matches nothing.
        query=_str_field(table, "query", "", where="source").strip(),
        workspaces=tuple(workspaces),
        keep_raw=_bool_field(table, "keep_raw", True, where="source"),
        fixtures_dir=fixtures_dir,
        max_messages=max_messages or None,
        output_excerpt_chars=_int_field(
            table, "output_excerpt_chars", 2000, where="source", minimum=1
        ),
        timeout_seconds=_int_field(
            table,
            "timeout_seconds",
            180,
            where="source",
            minimum=1,
            maximum=MAX_TIMEOUT_SECONDS,
        ),
    )


def _load_selection(table: Mapping[str, Any]) -> SelectionConfig:
    dimension = _str_field(table, "fairness_dimension", "class", where="selection")
    if dimension not in FAIRNESS_DIMENSIONS:
        raise ConfigError(
            f"[selection].fairness_dimension must be one of {FAIRNESS_DIMENSIONS}, got {dimension!r}"
        )
    markers = table.get("agent_title_markers", list(DEFAULT_AGENT_TITLE_MARKERS))
    if not isinstance(markers, list) or any(not isinstance(x, str) for x in markers):
        raise ConfigError("[selection].agent_title_markers must be a list of strings")
    return SelectionConfig(
        max_conversations_per_run=_int_field(
            table, "max_conversations_per_run", 8, where="selection", minimum=1
        ),
        fairness_dimension=dimension,
        agent_title_markers=tuple(markers),
    )


def _load_caps(table: Mapping[str, Any]) -> CapsConfig:
    return CapsConfig(
        max_candidates_per_run=_int_field(
            table, "max_candidates_per_run", 3, where="caps", minimum=0
        ),
        max_open_candidates=_int_field(table, "max_open_candidates", 12, where="caps", minimum=0),
        max_failures_per_candidate=_int_field(
            table, "max_failures_per_candidate", 6, where="caps", minimum=1
        ),
        max_skills_per_run=_int_field(table, "max_skills_per_run", 3, where="caps", minimum=1),
        max_extra_holdout_conversations=_int_field(
            table, "max_extra_holdout_conversations", 3, where="caps", minimum=0
        ),
        max_defer_attempts=_int_field(table, "max_defer_attempts", 3, where="caps", minimum=1),
        max_open_candidates_per_skill=_int_field(
            table, "max_open_candidates_per_skill", 1, where="caps", minimum=1
        ),
    )


def _load_prompt(table: Mapping[str, Any]) -> PromptConfig:
    return PromptConfig(
        max_transcript_chars=_int_field(
            table, "max_transcript_chars", 24000, where="prompt", minimum=500
        ),
        max_message_chars=_int_field(table, "max_message_chars", 4000, where="prompt", minimum=100),
        max_skill_text_chars=_int_field(
            table, "max_skill_text_chars", 24000, where="prompt", minimum=500
        ),
        redact_secrets=_bool_field(table, "redact_secrets", True, where="prompt"),
    )


def _load_gate(table: Mapping[str, Any]) -> GatePolicy:
    ratio = table.get("min_judge_preference_ratio", 0.5)
    # Compared as written: ``float`` of an integer past the double range
    # raises OverflowError, which made every command fall at load (exit 70).
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0 <= ratio <= 1:
        raise ConfigError("[gate].min_judge_preference_ratio must be a number in [0, 1]")
    return GatePolicy(
        min_fixes=_int_field(table, "min_fixes", 1, where="gate", minimum=0),
        max_regressions=_int_field(table, "max_regressions", 0, where="gate", minimum=0),
        max_holdout_degradations=_int_field(
            table, "max_holdout_degradations", 0, where="gate", minimum=0
        ),
        min_holdout_cases=_int_field(table, "min_holdout_cases", 1, where="gate", minimum=0),
        min_judge_preference_ratio=float(ratio),
        require_judge=_bool_field(table, "require_judge", True, where="gate"),
        require_independent_judge=_bool_field(
            table, "require_independent_judge", True, where="gate"
        ),
    )


def _load_skill_roots(raw: Any, *, base_dir: Path) -> tuple[SkillRoot, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("at least one [[skill_root]] is required")
    roots: list[SkillRoot] = []
    seen: set[str] = set()
    for idx, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"[[skill_root]] #{idx} is not a table")
        _refuse_unknown_keys(entry, TABLE_KEYS["skill_root"], where=f"[[skill_root]] #{idx}")
        rid = entry.get("id")
        path = entry.get("path")
        if not isinstance(rid, str) or not rid.strip():
            raise ConfigError(f"[[skill_root]] #{idx} needs a non-empty id")
        if rid in seen:
            raise ConfigError(f"duplicate skill_root id {rid!r}")
        seen.add(rid)
        if not isinstance(path, str) or not path.strip():
            raise ConfigError(f"skill_root {rid!r} needs a non-empty path")
        kind = str(entry.get("kind", "custom"))
        if kind not in ("builtin", "custom", "mirror"):
            raise ConfigError(f"skill_root {rid!r} kind must be builtin, custom or mirror")
        roots.append(
            SkillRoot(
                id=rid,
                path=_expand(path, base=base_dir),
                precedence=_int_field(
                    entry, "precedence", 100, where=f"skill_root.{rid}", minimum=0
                ),
                canonical=_bool_field(entry, "canonical", False, where=f"skill_root.{rid}"),
                editable=_bool_field(entry, "editable", True, where=f"skill_root.{rid}"),
                kind=kind,
                note=str(entry.get("note", "")),
            )
        )
    _assert_roots_disjoint(roots)
    return tuple(roots)


def _root_forms(path: Path) -> tuple[Path, ...]:
    """The spellings under which one root directory is compared with another:
    the configured path with ``.`` and ``..`` folded lexically, and the same
    path with every link followed. A root need not exist at load time, so both
    forms are computed without touching it beyond what ``resolve`` reads."""
    absolute = path if path.is_absolute() else path.absolute()
    return (Path(os.path.normpath(absolute)), absolute.resolve())


def _is_same_directory(a: Path, b: Path) -> bool:
    """Whether two spellings reach one directory on disk, judged by identity
    rather than text, so a case variant on a filesystem that folds case is
    caught. Either path missing, unreadable, or unstattable (a NUL byte in the
    spelling is a ``ValueError``) means no verdict here, not a match."""
    try:
        return os.path.samefile(a, b)
    except (OSError, ValueError):
        return False


def _lies_inside(inner: Path, outer: Path) -> bool:
    """Whether ``inner`` is strictly below ``outer``: by text on the given
    forms, or by identity between ``outer`` and any existing ancestor of
    ``inner`` (the inner root itself may not exist yet)."""
    if outer in inner.parents:
        return True
    return any(_is_same_directory(ancestor, outer) for ancestor in inner.parents)


def _assert_roots_disjoint(roots: Sequence[SkillRoot]) -> None:
    """Refuse two roots that name the same directory, or one inside another.

    Every root is scanned on its own, so a directory reachable from two roots
    yields the same file twice under two root ids. Measured before this check
    existed, a ``mirror`` root spelled as the canonical directory made the
    inventory report each skill as a mirror of itself, and the root flagged
    read-only was the exact directory the write landed in: the mirror interlock
    judged the file canonical first and the read-only flag protected nothing.
    A root inside another is the same fault for the inner tree, and the write
    path's walk below a root would judge the inner root as a component of the
    outer.

    Spellings are compared lexically folded and link-resolved, which needs no
    directory to exist; roots that do exist are also compared by identity
    (``samefile``), which is what catches ``Skills`` against ``skills`` on a
    filesystem that folds case. A case variant of a root that does not exist
    yet is the one spelling this check cannot see. Refused by name, with both
    spellings quoted.
    """
    for i, outer in enumerate(roots):
        for inner in roots[i + 1 :]:
            for a in _root_forms(outer.path):
                for b in _root_forms(inner.path):
                    if a == b or _is_same_directory(a, b):
                        raise ConfigError(
                            f"skill_root {outer.id!r} ({outer.path}) and {inner.id!r} "
                            f"({inner.path}) name the same directory {a}. Each skill would "
                            f"be inventoried twice, once per root, and a root flagged "
                            f"read-only would protect nothing. Remove one, or point it at "
                            f"a different tree."
                        )
                    if _lies_inside(b, a) or _lies_inside(a, b):
                        contained, container = (
                            (inner, outer) if _lies_inside(b, a) else (outer, inner)
                        )
                        raise ConfigError(
                            f"skill_root {contained.id!r} ({contained.path}) lies inside "
                            f"skill_root {container.id!r} ({container.path}). The outer root's "
                            f"scan already reaches every skill in the inner one, so each would "
                            f"be inventoried twice under two root ids. Move the inner tree "
                            f"beside the outer, or drop one of the two roots."
                        )


def _assert_editable_roots_outside_protected(
    roots: Sequence[SkillRoot], protected_dirs: Mapping[str, Path]
) -> None:
    """Refuse an editable root that lies inside a protected path.

    Every write under the package, the config directory, the state directory or
    the rubrics directory is refused as self-mutation, and that set is not
    configurable. A root marked canonical and editable inside one of them is
    therefore a tree the pipeline would propose against, score, hold for review
    and then refuse to write, at apply time, after the model calls were paid for.
    Measured before this check existed: such a config loaded, ``doctor`` was
    clean, and the first write raised ``SelfMutationRefused``. Every editable
    root is held to it, not only one marked canonical, because the resolver
    writes to unmarked editable roots too: a skill present only in such a root
    gets that root as its canonical copy even beside a marked one, and when no
    root is marked canonical the highest-precedence editable root is the
    canonical one for every skill. A read-only mirror inside a protected path
    is fine, since nothing is ever written to it,
    and a root that CONTAINS the config or state directory is fine too: only the
    skills under those directories are unwritable, and the write path says so.
    """
    for root in roots:
        if not root.editable:
            continue
        for label, guard in protected_dirs.items():
            for a in _root_forms(root.path):
                for b in _root_forms(guard):
                    if a == b or _is_same_directory(a, b) or _lies_inside(a, b):
                        raise ConfigError(
                            f"skill_root {root.id!r} ({root.path}) is editable but "
                            f"lies inside {label} ({guard}), which is protected: every write there "
                            f"is refused as self-mutation, so candidates against it would be "
                            f"proposed, scored and reviewed and then refused at apply. Keep the "
                            f"skills and {label} apart (move the skills out, or move the config "
                            f"file into its own subdirectory if it sits at the project root), "
                            f"or mark the root canonical = false and editable = false if it is "
                            f"only to be read."
                        )


def _load_rubrics(
    table: Mapping[str, Any], *, config_dir: Path, base_dir: Path
) -> tuple[Rubric, ...]:
    directory = _expand(_str_field(table, "dir", "rubrics", where="rubrics"), base=config_dir)
    enabled = table.get("enabled", [])
    if not isinstance(enabled, list) or not enabled or any(not isinstance(x, str) for x in enabled):
        raise ConfigError("[rubrics].enabled must be a non-empty list of rubric ids")
    repeated = sorted({r for r in enabled if enabled.count(r) > 1})
    if repeated:
        raise ConfigError(
            f"[rubrics].enabled lists {', '.join(repr(r) for r in repeated)} more than once; "
            "a rubric enabled twice would score every conversation against it twice"
        )
    rubrics: list[Rubric] = []
    for rubric_id in enabled:
        # Resolved so a message about it names one real path, not `config/../rubrics`.
        path = (directory / f"{rubric_id}.toml").resolve()
        if not path.is_file():
            raise ConfigError(f"rubric {rubric_id!r} not found at {path}")
        text = _read_utf8_or_refuse(path, what=f"rubric {rubric_id!r}")
        try:
            parsed = tomllib.loads(text)
        except ValueError as exc:  # TOMLDecodeError, or an integer past the digit limit
            raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
        _refuse_unstorable_integers(parsed, path)
        try:
            rubric = Rubric.from_mapping(parsed, source_sha256=sha256_text(text))
        except ValueError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
        if rubric.id != rubric_id:
            raise ConfigError(f"{path}: declares id {rubric.id!r} but is enabled as {rubric_id!r}")
        rubrics.append(rubric)
    return tuple(rubrics)


def _load_model_roles(table: Mapping[str, Any]) -> dict[str, RoleCommand]:
    defaults = table.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise ConfigError("[model.defaults] must be a table")
    _refuse_unknown_keys(defaults, TABLE_KEYS["model.*"], where="[model.defaults]")
    roles: dict[str, RoleCommand] = {}
    for role in ROLES:
        entry = table.get(role)
        if entry is None:
            continue
        if not isinstance(entry, Mapping):
            raise ConfigError(f"[model.{role}] must be a table")
        _refuse_unknown_keys(entry, TABLE_KEYS["model.*"], where=f"[model.{role}]")
        merged = {**{k: v for k, v in defaults.items() if k != "argv"}, **entry}
        if "argv" not in merged and "command" not in merged and "argv" in defaults:
            merged["argv"] = defaults["argv"]
        try:
            roles[role] = role_command_from_mapping(role, merged)
        except ValueError as exc:
            # The message names the role it was checked as; a value the role
            # took from [model.defaults] is written there, so say so, for the
            # inherited key the message names (``command`` is checked as argv).
            message = str(exc)
            inherited = sorted(
                key
                for key in defaults
                if key not in entry
                and key in merged
                and any(
                    f"[model.{role}].{name}" in message or f"'{name}'" in message
                    for name in ({key, "argv"} if key == "command" else {key})
                )
            )
            note = (
                f" ([model.{role}] takes {', '.join(inherited)} from [model.defaults])"
                if inherited
                else ""
            )
            raise ConfigError(f"{exc}{note}") from exc
    # The parent table is closed too, with the same hint: the role tables are
    # the only keys it may hold.
    _refuse_unknown_keys(table, frozenset(ROLES) | {"defaults"}, where="[model]")
    return roles
