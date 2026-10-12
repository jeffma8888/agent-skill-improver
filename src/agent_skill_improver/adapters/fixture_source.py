"""Offline conversation source over a directory of export-shaped JSON files.

This exists for two reasons, and the second one matters more.

1. The whole test suite runs with no network, no model and no external binary.
2. An operator can point the loop at a directory of files and get the *same code
   path* as production -- same normalizer, same scorer, same gate. A test harness
   that bypasses the real pipeline proves very little about the real pipeline.

Files are read in sorted order for reproducibility, but that order is never used
to *bound* a batch: bounding happens in the selection stage, which is fair by
construction. Truncating a sorted list is a permanent filter, not a sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..jsonio import loads_untrusted
from ..domain.conversation import NormalizedConversation
from ..domain.discovery import DiscoveredThread, DiscoveryPage, page_from_threads
from ..errors import RecordDefect, SourceUnavailable
from .transcript_export import normalize_export


class FixtureConversationSource:
    """Reads export-shaped JSON files from a directory.

    The shape is the element-graph contract documented in
    :mod:`agent_skill_improver.adapters.transcript_export`.
    """

    def __init__(
        self,
        directory: Path,
        *,
        output_excerpt_chars: int = 2000,
        max_messages: int | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.output_excerpt_chars = output_excerpt_chars
        self.max_messages = max_messages

    def preflight(self) -> None:
        if not self.directory.is_dir():
            raise SourceUnavailable(f"fixtures_dir {self.directory} is not a directory")

    def _scan(self) -> tuple[list[tuple[Path, dict[str, Any]]], list[tuple[str, str]]]:
        """Every export in the directory that reads as a thread, and every entry
        that does not, with the reason.

        Measured before the second list existed: an export at mode 000, one
        that was not JSON and one that was a JSON array sat beside five good
        ones, and the run reported five ingested with ``is_total`` true and no
        word about the three. Stepping over them is right (one bad file must
        not end the run); doing it silently was not.
        """
        accepted: list[tuple[Path, dict[str, Any]]] = []
        passed_over: list[tuple[str, str]] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                passed_over.append((str(path), f"cannot be read ({exc.strerror or exc})"))
                continue
            except ValueError as exc:
                passed_over.append((str(path), f"is not UTF-8 text ({exc})"))
                continue
            try:
                payload = loads_untrusted(text)
            except ValueError as exc:
                passed_over.append((str(path), f"is not valid JSON ({exc})"))
                continue
            if not isinstance(payload, dict):
                passed_over.append(
                    (str(path), f"is JSON but not an object ({type(payload).__name__})")
                )
                continue
            thread = payload.get("thread")
            if not isinstance(thread, dict):
                passed_over.append((str(path), "has no thread object"))
                continue
            if not isinstance(thread.get("id"), str) or not thread.get("id"):
                passed_over.append((str(path), "has a thread without an id"))
                continue
            accepted.append((path, payload))
        return accepted, passed_over

    def _payloads(self) -> list[tuple[Path, dict[str, Any]]]:
        return self._scan()[0]

    def passed_over(self) -> list[tuple[str, str]]:
        """Entries in the directory that do not read as a thread, with reasons."""
        return self._scan()[1]

    def discover_page(
        self,
        *,
        since: str,
        limit: int,
        workspace: str | None = None,
        query: str | None = None,
    ) -> DiscoveryPage:
        """A page over the fixture directory, bounded exactly like the real source.

        The bound is applied here and the truncation flag is derived, so a fixture
        run exercises the same limit-honesty path as production instead of
        pretending the corpus is always small enough.
        """
        accepted, passed_over = self._scan()
        return page_from_threads(
            self._threads_of(accepted, workspace=workspace),
            limit=limit,
            since=since,
            workspace=workspace or "",
            query=query or "",
            passed_over=passed_over,
        )

    def discover(
        self,
        *,
        since: str,
        limit: int,
        workspace: str | None = None,
        query: str | None = None,
    ) -> list[DiscoveredThread]:
        return list(
            self.discover_page(since=since, limit=limit, workspace=workspace, query=query).threads
        )

    def _threads_of(
        self, accepted: list[tuple[Path, dict[str, Any]]], *, workspace: str | None = None
    ) -> list[DiscoveredThread]:
        found: list[DiscoveredThread] = []
        for _, payload in accepted:
            thread = payload["thread"]  # _scan accepted only payloads with one
            thread_id = str(thread["id"])
            ws = str(thread.get("workspacePath", ""))
            if workspace and ws != workspace:
                continue
            found.append(
                DiscoveredThread(
                    thread_id=thread_id,
                    title=str(thread.get("title", "")),
                    workspace=ws,
                    updated_date="",
                )
            )
        return found

    def fetch(self, thread_id: str) -> NormalizedConversation:
        for path, payload in self._payloads():
            thread = payload.get("thread")
            if isinstance(thread, dict) and thread.get("id") == thread_id:
                return normalize_export(
                    payload,
                    output_excerpt_chars=self.output_excerpt_chars,
                    max_messages=self.max_messages,
                )
        raise RecordDefect(
            f"no fixture for thread {thread_id!r} in {self.directory}", record_id=thread_id
        )
