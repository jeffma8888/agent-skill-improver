"""Injectable clock so every artifact is reproducible under test."""

from __future__ import annotations

import datetime as _dt
from typing import Protocol


class Clock(Protocol):
    """Minimal time port."""

    def now(self) -> _dt.datetime:  # pragma: no cover - protocol
        ...


class SystemClock:
    """Real UTC clock."""

    def now(self) -> _dt.datetime:
        return _dt.datetime.now(_dt.timezone.utc)


class FrozenClock:
    """Deterministic clock for tests and replays."""

    def __init__(self, moment: _dt.datetime) -> None:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=_dt.timezone.utc)
        self._moment = moment

    def now(self) -> _dt.datetime:
        return self._moment

    def advance(self, seconds: float) -> None:
        self._moment = self._moment + _dt.timedelta(seconds=seconds)


def iso(moment: _dt.datetime) -> str:
    """Stable ISO-8601 in UTC with a trailing Z, second precision."""
    return moment.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def day_key(moment: _dt.datetime) -> str:
    """UTC calendar day, used as the daily run id and idempotence key."""
    return moment.astimezone(_dt.timezone.utc).strftime("%Y-%m-%d")
