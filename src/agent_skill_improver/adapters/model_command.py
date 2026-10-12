"""Generic command-based model adapter, executed under an isolation boundary.

A "model" here is any command that reads one JSON object on stdin and writes one
JSON object on stdout. That is deliberately the lowest useful contract: it works
with a local binary, a local server wrapped in a shell script, or a test fake.

Roles are configured independently (``scorer``, ``improver``, ``judge``) so the
scorer and the judge can be *different* models. That matters: a judge that is
the same model as the improver is grading its own homework.

**The child does not inherit this process.** Two things used to make this adapter
the external-communication leg of a lethal trifecta, and both are removed here
rather than argued about in a prompt:

* the environment was ``dict(os.environ)``, so every credential, token and proxy
  setting the operator happened to have exported was handed to a subprocess that
  had just been fed private transcript text. It is now built from a short
  allowlist by :func:`~agent_skill_improver.domain.isolation.minimal_env`, and a
  credential-shaped name surviving into the child is a refusal, not a warning.
* the process had whatever network and filesystem authority the parent had. It
  now runs under whatever :class:`~agent_skill_improver.adapters.sandbox.OsSandbox`
  the caller supplies -- for a private source that is an OS no-egress sandbox
  proved on this host, and for a synthetic fixture corpus it is
  :class:`~agent_skill_improver.adapters.sandbox.NoSandbox` so public CI stays
  portable.

Each call gets a **fresh 0700 directory**, created immediately before the child
starts and removed immediately after. It is the only path the child may write,
it is where ``HOME`` and ``TMPDIR`` point, and it does not outlive the call, so
one role cannot leave scratch state for the next one to read.

The isolation mode is part of every ``command_id``. A scorecard produced under
``mode = "none"`` and one produced under a verified sandbox are not the same
provenance, and a receipt that cannot distinguish them is not an audit trail.

Failure classification is the other important part:

* non-JSON or contract-violating output -> ``MalformedModelOutput`` (a record
  defect: quarantine this record, keep the batch going);
* timeout, missing binary, non-zero exit -> ``TransientFailure`` (defer; an
  unreachable model is not a verdict);
* a credential in the child environment -> ``IsolationRequired`` (a refusal;
  never retried into success).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .. import SCHEMA_MODEL_REQUEST
from ..domain.isolation import ENV_ALLOWLIST, dropped_env_count, minimal_env
from ..domain.redaction import redact
from ..errors import MalformedModelOutput, TransientFailure
from ..jsonio import loads_untrusted
from .bounded_process import (
    QUOTED_OUTPUT_CHARS,
    QUOTED_TAIL_CHARS,
    OutputOverflow,
    RunTimedOut,
    program_problem,
    quotable,
    run_bounded,
)

#: Mode 0700 on every per-call working directory, set explicitly rather than left
#: to the umask: a permissive umask is exactly the condition under which private
#: scratch content becomes world-readable, and it is not this process's to fix.
WORKDIR_MODE = 0o700

PLACEHOLDER_ROLE = "{role}"


@dataclass(frozen=True)
class RoleCommand:
    """One configured role."""

    role: str
    argv: tuple[str, ...]
    timeout_seconds: int = 300
    max_output_bytes: int = 4 * 1024 * 1024
    env: Mapping[str, str] = None  # type: ignore[assignment]
    model_label: str = ""

    def command_id(self, isolation_mode: str) -> str:
        """Stable identifier recorded on every artifact this role produces.

        Takes the isolation mode as an argument rather than reading it, because
        it is not a property of the command: the same argv confined by a
        verified sandbox and the same argv running free are different
        provenance, and an id that cannot tell them apart makes a receipt
        unable to answer the only question that matters after the fact.
        """
        label = self.model_label or "unlabelled"
        # The id is persisted in every scorecard, candidate and receipt the
        # role touches, so an argv carrying a key as an option value is shown
        # with the key masked. The placeholder is deterministic, so the id still
        # tells two configurations apart.
        return f"{self.role}:{label}:iso={isolation_mode}:{redact(' '.join(self.argv)).text}"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the command itself.

        Deliberately carries no ``command_id``: this object does not know the
        isolation mode, and emitting an id with a guessed mode in it would put a
        wrong provenance string in the config dump.
        """
        return {
            "role": self.role,
            "argv": [redact(part).text for part in self.argv],
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "model_label": self.model_label,
        }


class Isolation(Protocol):
    """What the runner needs from an isolation mechanism.

    Structural on purpose: the runner must not be able to tell an OS sandbox
    from the no-op, so it cannot grow a shortcut for one of them. Deciding which
    is legal for a given source is
    :func:`~agent_skill_improver.domain.isolation.decide`'s job, upstream.
    """

    @property
    def mode(self) -> str:  # pragma: no cover - protocol
        ...

    def wrap(self, argv: Sequence[str], *, workdir: Path) -> list[str]:  # pragma: no cover
        ...

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


class CommandModelRunner:
    """Runs configured role commands under an isolation boundary.

    ``isolation`` is required, with no default. A default would mean a caller
    that forgot about confinement silently got the unconfined path, which is the
    fail-open shape this whole boundary exists to remove.
    """

    def __init__(
        self,
        commands: Mapping[str, RoleCommand],
        *,
        isolation: Isolation,
        workdir_root: Path | None = None,
        verification: Mapping[str, Any] | None = None,
        parent_env: Mapping[str, str] | None = None,
    ) -> None:
        self._commands = dict(commands)
        self._isolation = isolation
        self._workdir_root = Path(workdir_root) if workdir_root is not None else None
        self._verification = dict(verification or {})
        self._parent_env = dict(os.environ if parent_env is None else parent_env)

    def has_role(self, role: str) -> bool:
        return role in self._commands

    @property
    def isolation_mode(self) -> str:
        return self._isolation.mode

    def isolation(self) -> dict[str, Any]:
        """What is actually in force, read off the mechanism rather than config.

        The upstream decision is made from this dict, so it has to describe the
        object that will run the subprocess. A version of this that reported the
        configured intent would pass whenever the config was written correctly
        and the wiring was wrong -- the same defect as a model label satisfying
        an independence check.
        """
        return {
            "mode": self._isolation.mode,
            "mechanism": self._isolation.to_dict(),
            "verified": bool(self._verification.get("ok")),
            "verification": dict(self._verification),
            "env_allowlist": list(ENV_ALLOWLIST),
            "env_names_dropped_from_parent": dropped_env_count(self._parent_env),
            "per_call_workdir_mode": oct(WORKDIR_MODE),
        }

    def program_problems(self) -> list[str]:
        """Why each configured role's program could not be started, if it cannot.

        Asked of the program itself, not the sandbox wrapper around it: a
        confined command still has to exist and be executable. Empty when every
        role can start.
        """
        out: list[str] = []
        for role in sorted(self._commands):
            program = self._commands[role].argv[0] if self._commands[role].argv else ""
            problem = program_problem(program) if program else "has an empty argv"
            if problem:
                out.append(f"role {role!r} command {program!r} {problem}")
        return out

    def command_id(self, role: str) -> str:
        mode = self._isolation.mode
        cmd = self._commands.get(role)
        return cmd.command_id(mode) if cmd else f"{role}:<unconfigured>:iso={mode}"

    def _new_workdir(self) -> Path:
        """A fresh private directory for exactly one model call."""
        root = self._workdir_root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="asi-role-", dir=str(root) if root else None))
        os.chmod(workdir, WORKDIR_MODE)
        return workdir.resolve()

    def run(self, *, role: str, system: str, user: str, response_schema: str) -> dict[str, Any]:
        cmd = self._commands.get(role)
        if cmd is None:
            raise TransientFailure(f"no command configured for role {role!r}")
        request = {
            "schema": SCHEMA_MODEL_REQUEST,
            "role": role,
            "response_schema": response_schema,
            "system": system,
            "user": user,
        }
        payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
        workdir = self._new_workdir()
        try:
            # Raises IsolationRequired if a credential-shaped name survives.
            env = minimal_env(self._parent_env, workdir=str(workdir), extra=cmd.env)
            argv = self._isolation.wrap(cmd.argv, workdir=workdir)
            # Output is bounded WHILE it is read: a flooding role is killed at
            # the cap rather than buffered whole and measured afterwards.
            result = run_bounded(
                argv,
                stdin=payload,
                env=env,
                timeout_seconds=cmd.timeout_seconds,
                max_output_bytes=cmd.max_output_bytes,
            )
        except OSError as exc:
            # Not found, not executable, a directory, a wrong binary format:
            # the command could not be started. Measured before: only
            # not-found was caught, and a role command at mode 644 ended the
            # run as an internal error with a traceback after ``doctor`` had
            # said ready.
            raise TransientFailure(f"role {role!r} command cannot be started: {exc}") from exc
        except RunTimedOut as exc:
            raise TransientFailure(f"role {role!r} timed out after {cmd.timeout_seconds}s") from exc
        except OutputOverflow as exc:
            raise MalformedModelOutput(
                f"role {role!r} produced more than max_output_bytes ({cmd.max_output_bytes}); "
                "the command was stopped there"
            ) from exc
        finally:
            # The scratch directory does not outlive the call, so one role cannot
            # leave private residue for the next one -- or for anything that
            # later walks the state tree.
            shutil.rmtree(workdir, ignore_errors=True)
        if result.returncode != 0:
            tail = quotable(
                result.stderr_tail.decode("utf-8", errors="replace").strip(),
                limit=QUOTED_TAIL_CHARS,
                from_end=True,
            )
            raise TransientFailure(
                f"role {role!r} exited {result.returncode}: {tail or '<no stderr>'}"
            )
        try:
            out = result.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MalformedModelOutput(
                f"role {role!r} produced output that is not UTF-8 ({exc.reason} at byte "
                f"{exc.start}); the contract is one UTF-8 JSON object on stdout"
            ) from exc
        return extract_json_object(out, role=role)


def extract_json_object(text: str, *, role: str = "model") -> dict[str, Any]:
    """Parse a single JSON object from model output.

    Strict first. If that fails, make exactly *one* bounded attempt to recover
    the last balanced top-level object, because real models wrap JSON in prose
    and refusing that outright would quarantine healthy records. Anything less
    tolerant than this is brittle; anything more tolerant starts inventing
    structure that was never there.
    """
    stripped = text.strip()
    if not stripped:
        raise MalformedModelOutput(f"role {role!r} produced empty output")
    try:
        parsed = loads_untrusted(stripped)
    except ValueError:
        parsed = _last_balanced_object(stripped, role=role)
    if not isinstance(parsed, dict):
        raise MalformedModelOutput(
            f"role {role!r} produced a {type(parsed).__name__}, expected a JSON object"
        )
    return parsed


def _last_balanced_object(text: str, *, role: str) -> Any:
    """Find the last balanced ``{...}`` run, respecting strings and escapes."""
    ends: list[int] = []
    depth = 0
    in_string = False
    escaped = False
    start: int | None = None
    spans: list[tuple[int, int]] = []
    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append((start, idx + 1))
                    ends.append(idx)
                    start = None
    for span_start, span_end in reversed(spans):
        chunk = text[span_start:span_end]
        try:
            return loads_untrusted(chunk)
        except ValueError:
            continue
    raise MalformedModelOutput(
        f"role {role!r} produced no parseable JSON object "
        f"(first {QUOTED_OUTPUT_CHARS} chars: "
        f"{quotable(text, limit=QUOTED_OUTPUT_CHARS, from_end=False)!r})"
    )


def role_command_from_mapping(role: str, raw: Mapping[str, Any]) -> RoleCommand:
    """Build a :class:`RoleCommand` from a config table. Raises ``ValueError``."""
    argv_raw = raw.get("argv") or raw.get("command")
    if isinstance(argv_raw, str):
        argv: Sequence[str] = shlex.split(argv_raw)
    elif isinstance(argv_raw, list) and argv_raw and all(isinstance(x, str) for x in argv_raw):
        argv = argv_raw
    else:
        raise ValueError(
            f"[model.{role}] needs 'argv' as a non-empty list of strings (or a shell-ish string)"
        )
    argv = [part.replace(PLACEHOLDER_ROLE, role) for part in argv]
    timeout = raw.get("timeout_seconds", 300)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError(f"[model.{role}].timeout_seconds must be a positive integer")
    max_bytes = raw.get("max_output_bytes", 4 * 1024 * 1024)
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError(f"[model.{role}].max_output_bytes must be a positive integer")
    env_raw = raw.get("env", {})
    if not isinstance(env_raw, Mapping) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in env_raw.items()
    ):
        raise ValueError(f"[model.{role}].env must be a table of string -> string")
    return RoleCommand(
        role=role,
        argv=tuple(argv),
        timeout_seconds=timeout,
        max_output_bytes=max_bytes,
        env=dict(env_raw),
        model_label=str(raw.get("model_label", "")),
    )
