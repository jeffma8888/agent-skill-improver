"""Ports: the interfaces the domain and pipeline depend on.

Adapters implement these. The pipeline never imports an adapter module
directly -- it is handed implementations -- so every stage is testable with a
fake that does no I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

from .domain.conversation import NormalizedConversation
from .domain.discovery import DiscoveryPage
from .domain.skills import SkillGroup, SkillRoot, SkillSnapshot


class ThreadRef(Protocol):
    """Minimal discovery record."""

    thread_id: str
    title: str
    workspace: str
    updated_date: str


class ConversationSource(Protocol):
    """Discovers conversations and fetches them in normalized form.

    ``discover_page`` is the interface the driver actually uses, because a bare
    list cannot say whether it was bounded by the limit -- and a page count read
    as a total is the quiet way a harvester overstates its coverage.
    """

    def discover_page(
        self, *, since: str, limit: int, workspace: str | None = None, query: str | None = None
    ) -> DiscoveryPage:  # pragma: no cover - protocol
        ...

    def discover(
        self, *, since: str, limit: int, workspace: str | None = None, query: str | None = None
    ) -> list[Any]:  # pragma: no cover - protocol
        ...

    def fetch(self, thread_id: str) -> NormalizedConversation:  # pragma: no cover - protocol
        ...


class ModelRunner(Protocol):
    """Runs one model role and returns parsed JSON.

    Implementations raise ``MalformedModelOutput`` for a contract violation
    (quarantine the record) and ``TransientFailure`` for anything that means we
    simply did not get an answer (defer the record).

    ``isolation`` is part of the port rather than of one adapter, because the
    driver refuses to run a private source against a runner that is not
    confined. A fake that could not describe its own confinement would be able
    to slip past that check, so every implementation -- including the in-process
    test double -- has to answer the question.
    """

    def run(
        self, *, role: str, system: str, user: str, response_schema: str
    ) -> dict[str, Any]:  # pragma: no cover - protocol
        ...

    def command_id(self, role: str) -> str:  # pragma: no cover - protocol
        ...

    def isolation(self) -> dict[str, Any]:  # pragma: no cover - protocol
        """What confinement is actually in force. Read, never configured."""
        ...


class SkillStore(Protocol):
    """Reads skill roots; writes only when explicitly not read-only."""

    read_only: bool

    def roots(self) -> Sequence[SkillRoot]:  # pragma: no cover - protocol
        ...

    def snapshots(self) -> list[SkillSnapshot]:  # pragma: no cover - protocol
        ...

    def groups(self) -> list[SkillGroup]:  # pragma: no cover - protocol
        ...

    def read_text(self, path: Path) -> str:  # pragma: no cover - protocol
        ...

    def tree_digest(self) -> str:  # pragma: no cover - protocol
        ...

    def write_canonical(
        self, path: Path, text: str, *, expected_sha256: str, backup_dir: Path
    ) -> Any:  # pragma: no cover - protocol
        ...
