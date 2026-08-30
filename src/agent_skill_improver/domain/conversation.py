"""Normalized conversation schema.

The normalized form deliberately keeps three things the scorer needs in order
to produce *checkable* evidence:

1. ``element_id`` -- the stable element id from the export.
2. ``position`` -- the element's ordinal in the thread.
3. tool events with ``name``, ``status``, and an output excerpt.

A scorer that cannot point at (1) and (2) has not produced evidence, and its
verdict is downgraded to INSUFFICIENT rather than trusted as a FAIL. Keeping
both id *and* position is the point: a fabricated citation almost always gets
one of the two wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .. import SCHEMA_CONVERSATION

# Element kinds in the export contract (see adapters/transcript_export.py).
KIND_MESSAGE = "message"
KIND_TEXT = "text"
KIND_TOOL = "tool"
KIND_THINKING = "thinking"
KIND_CONTEXT = "context"
KIND_DIAGNOSTIC = "diagnostic"
KIND_COMPACTION = "compaction"
KIND_GROUP_SUMMARY = "groupSummary"
KIND_TASK_COMPLETION = "taskCompletion"
KIND_AGENT_MODE = "agentMode"
KIND_WORKSPACE_CHANGE = "workspaceChange"

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

TOOL_STATUS_DONE = "done"
TOOL_STATUS_ERROR = "error"
TOOL_STATUS_RUNNING = "running"


@dataclass(frozen=True)
class EvidenceRef:
    """A citation into a conversation. Both id and position must match."""

    element_id: str
    position: int
    kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"element_id": self.element_id, "position": self.position, "kind": self.kind}

    @staticmethod
    def from_dict(raw: Any) -> "EvidenceRef":
        if not isinstance(raw, dict):
            raise ValueError("evidence ref must be an object")
        element_id = raw.get("element_id")
        position = raw.get("position")
        if not isinstance(element_id, str) or not element_id:
            raise ValueError("evidence ref needs a non-empty element_id")
        if not isinstance(position, int) or isinstance(position, bool):
            raise ValueError("evidence ref needs an integer position")
        kind = raw.get("kind")
        return EvidenceRef(element_id, position, kind if isinstance(kind, str) else "")


@dataclass(frozen=True)
class ToolEvent:
    """One tool invocation, retained as scoreable evidence."""

    element_id: str
    position: int
    name: str
    status: str
    input_text: str
    output_excerpt: str
    output_bytes: int
    duration_ms: int | None = None
    parent_message_id: str | None = None

    @property
    def failed(self) -> bool:
        return self.status == TOOL_STATUS_ERROR

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "position": self.position,
            "name": self.name,
            "status": self.status,
            "input_text": self.input_text,
            "output_excerpt": self.output_excerpt,
            "output_bytes": self.output_bytes,
            "duration_ms": self.duration_ms,
            "parent_message_id": self.parent_message_id,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ToolEvent":
        return ToolEvent(
            element_id=str(raw["element_id"]),
            position=int(raw["position"]),
            name=str(raw.get("name", "")),
            status=str(raw.get("status", "")),
            input_text=str(raw.get("input_text", "")),
            output_excerpt=str(raw.get("output_excerpt", "")),
            output_bytes=int(raw.get("output_bytes", 0)),
            duration_ms=raw.get("duration_ms"),
            parent_message_id=raw.get("parent_message_id"),
        )


@dataclass(frozen=True)
class Message:
    """A user or assistant turn, with its text and the tools it ran."""

    element_id: str
    position: int
    role: str
    text: str
    author: str | None = None
    origin: str | None = None
    tool_events: tuple[ToolEvent, ...] = ()
    text_element_ids: tuple[str, ...] = ()

    @property
    def is_user(self) -> bool:
        return self.role == ROLE_USER

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "position": self.position,
            "role": self.role,
            "text": self.text,
            "author": self.author,
            "origin": self.origin,
            "tool_events": [t.to_dict() for t in self.tool_events],
            "text_element_ids": list(self.text_element_ids),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Message":
        return Message(
            element_id=str(raw["element_id"]),
            position=int(raw["position"]),
            role=str(raw["role"]),
            text=str(raw.get("text", "")),
            author=raw.get("author"),
            origin=raw.get("origin"),
            tool_events=tuple(ToolEvent.from_dict(t) for t in raw.get("tool_events", [])),
            text_element_ids=tuple(str(x) for x in raw.get("text_element_ids", [])),
        )


@dataclass(frozen=True)
class NormalizedConversation:
    """A conversation reduced to what the loop can score and cite."""

    thread_id: str
    title: str = ""
    workspace: str = ""
    profile: str = ""
    agent_name: str = ""
    created_at: int | None = None
    updated_at: int | None = None
    messages: tuple[Message, ...] = ()
    element_index: dict[str, int] = field(default_factory=dict)
    element_kinds: dict[str, str] = field(default_factory=dict)
    total_elements: int = 0
    truncated: bool = False
    source_kind: str = "conversation-export"

    # -- identity -------------------------------------------------------
    @property
    def record_id(self) -> str:
        return self.thread_id

    # -- evidence -------------------------------------------------------
    def resolves(self, ref: EvidenceRef) -> bool:
        """True only when the element exists *and* the position agrees."""
        known = self.element_index.get(ref.element_id)
        if known is None:
            return False
        if known != ref.position:
            return False
        if ref.kind and self.element_kinds.get(ref.element_id, "") != ref.kind:
            return False
        return True

    def unresolved(self, refs: Iterable[EvidenceRef]) -> list[EvidenceRef]:
        return [r for r in refs if not self.resolves(r)]

    # -- convenience signals used by attribution ------------------------
    @property
    def user_messages(self) -> tuple[Message, ...]:
        return tuple(m for m in self.messages if m.is_user)

    @property
    def tool_events(self) -> tuple[ToolEvent, ...]:
        out: list[ToolEvent] = []
        for m in self.messages:
            out.extend(m.tool_events)
        return tuple(out)

    @property
    def failed_tool_events(self) -> tuple[ToolEvent, ...]:
        return tuple(t for t in self.tool_events if t.failed)

    # -- serialization --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_CONVERSATION,
            "thread_id": self.thread_id,
            "title": self.title,
            "workspace": self.workspace,
            "profile": self.profile,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "element_index": dict(self.element_index),
            "element_kinds": dict(self.element_kinds),
            "total_elements": self.total_elements,
            "truncated": self.truncated,
            "source_kind": self.source_kind,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "NormalizedConversation":
        if raw.get("schema") != SCHEMA_CONVERSATION:
            raise ValueError(f"unexpected conversation schema: {raw.get('schema')!r}")
        return NormalizedConversation(
            thread_id=str(raw["thread_id"]),
            title=str(raw.get("title", "")),
            workspace=str(raw.get("workspace", "")),
            profile=str(raw.get("profile", "")),
            agent_name=str(raw.get("agent_name", "")),
            created_at=raw.get("created_at"),
            updated_at=raw.get("updated_at"),
            messages=tuple(Message.from_dict(m) for m in raw.get("messages", [])),
            element_index={str(k): int(v) for k, v in raw.get("element_index", {}).items()},
            element_kinds={str(k): str(v) for k, v in raw.get("element_kinds", {}).items()},
            total_elements=int(raw.get("total_elements", 0)),
            truncated=bool(raw.get("truncated", False)),
            source_kind=str(raw.get("source_kind", "conversation-export")),
        )
