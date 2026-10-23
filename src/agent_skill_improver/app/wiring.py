"""Constructs adapters from config. The only place that decides which adapter runs.

Kept out of :mod:`agent_skill_improver.app.daily` on purpose: the driver receives
ports, so a test can hand it a fake model and a temporary skill tree without any
of this module's imports being reachable.

This is also where the **isolation boundary** is assembled, because it is the
only place that sees the source and the model runner at the same time. The
sequence is deliberate:

1. classify the configured source. A private source requires an OS no-egress
   sandbox; the synthetic fixture corpus does not;
2. for a private source, *detect* a mechanism and then *verify* it on this host.
   Detection alone is a claim -- ``sandbox-exec`` existing says nothing about
   whether the generated profile actually refuses a socket;
3. hand the runner whatever was verified, and let
   :func:`~agent_skill_improver.domain.isolation.decide` upstream turn an
   unverified or absent mechanism into a refusal.

Nothing here silently downgrades. If the host has no mechanism, the runner is
constructed with :class:`~agent_skill_improver.adapters.sandbox.NoSandbox` and an
empty verification, which the decision reads as unsatisfied -- so the refusal
happens at one well-tested place rather than in a branch of this function.
"""

from __future__ import annotations

from typing import Any

from ..adapters.conversation_cli import ConversationCliSource
from ..adapters.fixture_source import FixtureConversationSource
from ..adapters.model_command import CommandModelRunner
from ..adapters.sandbox import NoSandbox, OsSandbox, detect, verify
from ..adapters.skill_fs import FilesystemSkillStore
from ..config import SOURCE_FIXTURES, Config
from ..domain.isolation import isolation_required
from ..errors import ConfigError
from .state import StateStore


def build_state(config: Config) -> StateStore:
    from ..clock import SystemClock

    return StateStore(config.state_dir, clock=SystemClock())


def build_source(config: Config, state: StateStore) -> Any:
    if config.source.kind == SOURCE_FIXTURES:
        if config.source.fixtures_dir is None:
            raise ConfigError('[source].fixtures_dir is required for kind = "fixtures"')
        return FixtureConversationSource(
            config.source.fixtures_dir,
            output_excerpt_chars=config.source.output_excerpt_chars,
            max_messages=config.source.max_messages,
        )
    return ConversationCliSource(
        discovery_argv=config.source.discovery_argv,
        query_argv=config.source.query_argv,
        workspace_argv=config.source.workspace_argv,
        export_argv=config.source.export_argv,
        timeout_seconds=config.source.timeout_seconds,
        raw_store=(state if config.source.keep_raw else None),
        output_excerpt_chars=config.source.output_excerpt_chars,
        max_messages=config.source.max_messages,
    )


def build_runner(config: Config, state: StateStore | None = None) -> CommandModelRunner:
    """Build the model runner, confined when the source is private.

    ``state`` is optional only so a caller inspecting configuration does not have
    to create a state tree. When it is given, sandbox scratch directories live
    under the gitignored state tree instead of the system temp directory, which
    keeps private scratch inside the one place this tool already treats as
    private and 0700.
    """
    workdir_root = state.dir("sandbox") if state is not None else None
    if not isolation_required(config.source.kind):
        # Synthetic corpus: the deterministic fake runs unconfined so public CI
        # is portable. The exception is derived from the source kind here, not
        # granted by a config value.
        return CommandModelRunner(
            config.model_roles,
            isolation=NoSandbox(),
            workdir_root=workdir_root,
        )
    mechanism = detect()
    if mechanism is None:
        # No supported mechanism. Construct the honest thing -- unconfined, with
        # no verification -- and let the decision refuse. Raising here would put
        # a second refusal path in a module with no tests of its own.
        return CommandModelRunner(
            config.model_roles,
            isolation=NoSandbox(),
            workdir_root=workdir_root,
        )
    sandbox = OsSandbox(mechanism)
    report = verify(sandbox, workdir_root=workdir_root)
    return CommandModelRunner(
        config.model_roles,
        isolation=sandbox if report.get("ok") else NoSandbox(),
        workdir_root=workdir_root,
        verification=report,
    )


def build_skill_store(config: Config, *, writable: bool = False) -> FilesystemSkillStore:
    """Read-only unless a caller explicitly asks for a writable store.

    Only ``asi apply`` and ``asi rollback`` pass ``writable=True``.
    """
    return FilesystemSkillStore(
        config.skill_roots,
        read_only=not writable,
        protected_paths=config.protected_paths,
        protected_skill_ids=config.protected_skill_ids,
    )
