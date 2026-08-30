"""Bounded, fair selection.

The trap this module exists to avoid: bounding a pass with
``sorted(items)[:limit]``. Whenever the sort key correlates with the property
being selected -- and filenames, ids and insertion order almost always do --
that is not a sample, it is a permanent filter. A budget applied that way can
make whole categories *structurally unreachable* while every downstream report
reads as "under-represented" rather than "never looked at".

So selection is round-robin across a declared fairness dimension, best-first
inside each bucket. A thin bucket contributes everything it has and drops out,
so no budget is reserved and then wasted. Items that cannot be classified get
their own bucket rather than being dropped, because an unclassifiable record is
exactly the one a human needs to see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")

UNCLASSIFIED_BUCKET = ""


@dataclass(frozen=True)
class SelectionReport:
    selected_count: int
    considered_count: int
    limit: int
    buckets: dict[str, int]
    selected_per_bucket: dict[str, int]
    exhausted_buckets: tuple[str, ...]
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_count": self.selected_count,
            "considered_count": self.considered_count,
            "limit": self.limit,
            "buckets": dict(self.buckets),
            "selected_per_bucket": dict(self.selected_per_bucket),
            "exhausted_buckets": list(self.exhausted_buckets),
            "truncated": self.truncated,
        }


def select_fair(
    items: Iterable[T],
    *,
    limit: int,
    bucket_of: Callable[[T], str],
    rank_of: Callable[[T], Any] | None = None,
) -> tuple[list[T], SelectionReport]:
    """Round-robin one item per bucket until ``limit`` is reached.

    ``rank_of`` orders items *within* a bucket (best first). It must be a total
    order over the bucket's items so the result is deterministic.
    """
    materialized: list[T] = list(items)
    buckets: dict[str, list[T]] = {}
    for item in materialized:
        key = bucket_of(item) or UNCLASSIFIED_BUCKET
        buckets.setdefault(key, []).append(item)

    if rank_of is not None:
        for key in buckets:
            buckets[key].sort(key=rank_of)

    counts = {k: len(v) for k, v in buckets.items()}
    order = sorted(buckets)
    selected: list[T] = []
    per_bucket: dict[str, int] = {k: 0 for k in order}
    cursors = {k: 0 for k in order}

    if limit <= 0:
        return [], SelectionReport(
            0, len(materialized), limit, counts, per_bucket, (), bool(materialized)
        )

    active = list(order)
    while active and len(selected) < limit:
        progressed = False
        for key in list(active):
            if len(selected) >= limit:
                break
            idx = cursors[key]
            if idx >= len(buckets[key]):
                active.remove(key)
                continue
            selected.append(buckets[key][idx])
            cursors[key] = idx + 1
            per_bucket[key] += 1
            progressed = True
        if not progressed:
            break

    exhausted = tuple(k for k in order if cursors[k] >= len(buckets[k]))
    return selected, SelectionReport(
        selected_count=len(selected),
        considered_count=len(materialized),
        limit=limit,
        buckets=counts,
        selected_per_bucket=per_bucket,
        exhausted_buckets=exhausted,
        truncated=len(selected) < len(materialized),
    )


def bucket_coverage(report: SelectionReport) -> float:
    """Fraction of non-empty buckets that contributed at least one item."""
    non_empty = [k for k, n in report.buckets.items() if n > 0]
    if not non_empty:
        return 1.0
    touched = [k for k in non_empty if report.selected_per_bucket.get(k, 0) > 0]
    return len(touched) / len(non_empty)


def rank_by(*keys: Callable[[Any], Any]) -> Callable[[Any], tuple[Any, ...]]:
    """Compose ranking functions into a single tuple key."""

    def _key(item: Any) -> tuple[Any, ...]:
        return tuple(k(item) for k in keys)

    return _key


def declared_buckets(values: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
