"""Review interlocks for promotion, not caller authentication.

Be precise about what this is and is not.

It is **not** a security boundary. Any process that can read a candidate file
can compute its token, and nothing here defends against a determined local
attacker; it does not try to. Apply does not authenticate its caller: a separate
process without the marker can still apply with a token and acknowledgement.
Do not schedule apply.

It **is** a set of review interlocks that refuse promotion in a marked context
and make every promotion auditable:

1. ``apply`` requires a token derived from the candidate's *content*, with the
   patch hash recomputed from the patch text rather than read from the field
   stored beside it -- so a token stops matching the moment the patch or the base
   file changes, including an edit to the patch text alone;
2. ``apply`` requires an explicit acknowledgement flag, so the intent is in the
   command line and therefore in the shell history and the receipt;
3. ``apply`` refuses outright when the automation marker is active in the
   environment. The daily driver sets it only inside its own process; the
   scheduler templates set it for their whole job command (and later cron lines
   on cronie, Vixie and OpenBSD cron);
4. the daily driver never imports the apply path at all -- enforced by a test
   that reads the module source.

The order matters: (4) is the real guarantee, and (1)-(3) are what catch a human
running the wrong thing at 2am.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..errors import ApprovalRequired, AutomationRefused

#: Set by the daily driver in-process and by scheduler templates for their jobs.
AUTOMATION_ENV_VAR = "ASI_AUTOMATED_RUN"
ORIGIN_CLI = "cli"


def automation_active() -> bool:
    return os.environ.get(AUTOMATION_ENV_VAR, "") not in ("", "0", "false", "False")


class AutomationMarker:
    """Context manager that marks the process as an automated run."""

    def __init__(self, value: str = "1") -> None:
        self._value = value
        self._previous: str | None = None

    def __enter__(self) -> "AutomationMarker":
        self._previous = os.environ.get(AUTOMATION_ENV_VAR)
        os.environ[AUTOMATION_ENV_VAR] = self._value
        return self

    def __exit__(self, *exc: object) -> None:
        if self._previous is None:
            os.environ.pop(AUTOMATION_ENV_VAR, None)
        else:
            os.environ[AUTOMATION_ENV_VAR] = self._previous


@dataclass(frozen=True)
class HumanApproval:
    """Recorded review interlocks for this promotion, not caller authentication.

    Constructible only through :meth:`from_cli`. There is no default
    constructor path that produces an approval without a token, which is what
    stops a caller from passing ``HumanApproval()`` in a hurry.
    """

    token: str
    acknowledged: bool
    origin: str
    operator: str = ""
    note: str = ""

    @staticmethod
    def from_cli(
        *,
        token: str | None,
        acknowledged: bool,
        operator: str = "",
        note: str = "",
    ) -> "HumanApproval":
        if automation_active():
            raise AutomationRefused(
                "promotion refused: the automation marker "
                f"({AUTOMATION_ENV_VAR}) is set. After review, run apply from an "
                "unmarked interactive shell; do not schedule apply. This marker is "
                "an interlock, not caller authentication."
            )
        if not token or not token.strip():
            raise ApprovalRequired(
                "promotion refused: no approval token supplied. Run "
                "`asi review <candidate-id>` and pass the printed token with --approve."
            )
        if not acknowledged:
            raise ApprovalRequired(
                "promotion refused: --i-reviewed-this is required so the "
                "acknowledgement is recorded in the receipt and in shell history."
            )
        return HumanApproval(
            token=token.strip(), acknowledged=True, origin=ORIGIN_CLI, operator=operator, note=note
        )

    def verify(self, *, expected_token: str) -> None:
        if self.origin != ORIGIN_CLI:
            raise ApprovalRequired(f"approval origin {self.origin!r} is not accepted")
        if self.token != expected_token:
            raise ApprovalRequired(
                "approval token does not match this candidate. A token is derived from "
                "the candidate id, its base hash and the hash recomputed from its patch "
                "text, so it stops matching if any of them changed since review."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "acknowledged": self.acknowledged,
            "operator": self.operator,
            "note": self.note,
            # The token itself is intentionally not recorded: it is reproducible
            # from the candidate, and copying it into a receipt only invites
            # someone to paste it back without re-reviewing.
            "token_recorded": False,
        }
