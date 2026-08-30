"""Discovery records, and honesty about what a page of results actually is.

The failure this module exists to prevent is quiet and expensive: treating a
bounded page of search results as if it were a total.

Two measured facts about the interface this tool reads from:

* the default page size is small (10 in the environment this was designed
  against), so a harvester written against the documented default sees a few
  percent of a day's threads and reports it as "the day";
* a request for N results returns *at most* N, and when it returns exactly N the
  real population is unknown and almost certainly larger.

So every discovery call returns a :class:`DiscoveryPage` that states its own
limit, whether it was limit-bound, and therefore whether the count may be read
as a total. The receipt carries that flag, and the driver refuses to call a
limit-bound page a total anywhere in its output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class DiscoveredThread:
    """One discovery record: everything known about a thread before fetching it."""

    thread_id: str
    title: str = ""
    workspace: str = ""
    updated_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "title": self.title,
            "workspace": self.workspace,
            "updated_date": self.updated_date,
        }


@dataclass(frozen=True)
class DiscoveryPage:
    """One bounded page of discovery results, honest about its own bounds."""

    threads: tuple[DiscoveredThread, ...]
    limit: int
    since: str = ""
    workspace: str = ""
    query: str = ""
    #: True when the source declined to say how large the population is. Always
    #: true for a page that came back full.
    truncated_by_limit: bool = False
    error: str = ""

    @property
    def returned(self) -> int:
        return len(self.threads)

    @property
    def is_total(self) -> bool:
        """False whenever the count must not be read as a population size."""
        return not self.truncated_by_limit and not self.error

    @property
    def honesty_note(self) -> str:
        if self.error:
            return (
                f"discovery for workspace={self.workspace or '<all>'!r} failed, so this page is "
                "not evidence about the population: " + self.error
            )
        if self.truncated_by_limit:
            return (
                f"discovery returned exactly its limit ({self.limit}) for "
                f"workspace={self.workspace or '<all>'}: this is a PAGE, not a total. The real "
                "population is unknown and larger. Raise [source].discovery_limit or narrow "
                "[source].since before reading this count as coverage."
            )
        return (
            f"discovery returned {self.returned} of at most {self.limit} for "
            f"workspace={self.workspace or '<all>'}: fewer than the limit, so this page may be "
            "read as the whole window."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "since": self.since,
            "workspace": self.workspace or "<all>",
            "query": self.query,
            "returned": self.returned,
            "truncated_by_limit": self.truncated_by_limit,
            "is_total": self.is_total,
            "error": self.error,
            "honesty_note": self.honesty_note,
        }


def page_from_threads(
    threads: Sequence[DiscoveredThread],
    *,
    limit: int,
    since: str = "",
    workspace: str = "",
    query: str = "",
) -> DiscoveryPage:
    """Build a page, deriving ``truncated_by_limit`` rather than trusting a caller.

    Derived, not passed in: a caller that has to remember to set the flag will
    eventually forget, and the failure is silent.
    """
    kept = tuple(threads[:limit]) if limit > 0 else tuple(threads)
    return DiscoveryPage(
        threads=kept,
        limit=limit,
        since=since,
        workspace=workspace,
        query=query,
        truncated_by_limit=limit > 0 and len(threads) >= limit,
    )


def error_page(
    *, limit: int, since: str = "", workspace: str = "", query: str = "", error: str
) -> DiscoveryPage:
    return DiscoveryPage(
        threads=(), limit=limit, since=since, workspace=workspace, query=query, error=error
    )


def merge_pages(pages: Iterable[DiscoveryPage]) -> tuple[list[DiscoveredThread], dict[str, Any]]:
    """De-duplicate across pages and summarise, preserving the honesty flags."""
    materialized = list(pages)
    seen: set[str] = set()
    threads: list[DiscoveredThread] = []
    for page in materialized:
        for thread in page.threads:
            if thread.thread_id in seen:
                continue
            seen.add(thread.thread_id)
            threads.append(thread)
    any_truncated = any(p.truncated_by_limit for p in materialized)
    errors = [p.error for p in materialized if p.error]
    summary = {
        "pages": [p.to_dict() for p in materialized],
        "threads_returned": len(threads),
        "is_total": (not any_truncated) and not errors,
        "any_page_truncated_by_limit": any_truncated,
        "errors": errors,
    }
    return threads, summary
