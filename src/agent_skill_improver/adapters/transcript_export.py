"""Normalizer for a transcript export: the element-graph JSON contract.

This is one of the two documented contracts an operator has to satisfy to point
the loop at their own conversation tool (the other is the discovery line grammar
in :mod:`agent_skill_improver.adapters.conversation_cli`). It is written as a
contract rather than as "whatever tool X emits" so the adapter is portable, and
every rule below exists because a plausible-looking guess gets it wrong:

* top level is ``{"thread": {...}, "elements": [...]}``;
* ``thread`` carries ``id`` (required), and optionally ``title``,
  ``workspacePath``, ``profileName``, ``agentName``, ``createdAt``,
  ``updatedAt``;
* every element has ``id``, ``position``, ``kind``, and optionally ``parentId``;
* the topology is exactly two levels deep -- ``message`` elements are roots, and
  ``text`` / ``tool`` / ``thinking`` and any other per-turn detail hang off a
  message via ``parentId``. Elements describing the session rather than a turn
  are parentless;
* a ``message`` carries ``role`` and ``author`` but **no content** -- the words
  live in its child ``text`` elements, whose key is ``content``; a ``user``
  message also carries ``origin``, ``user-input`` when a person typed it (any
  origin outside ``domain.authorship.HUMAN_ORIGINS`` marks the turn generated);
* ``tool`` carries ``name`` / ``status`` / ``input`` / ``output``, where
  ``status`` is a short word such as ``done`` / ``error`` / ``running`` and
  ``input`` is usually an object but is sometimes a bare string.

An unrecognised ``kind`` is preserved in the element index and otherwise
ignored, so a tool that emits richer elements than this list still parses. The
parser never raises on a merely surprising element: refusing an export because
one element was unfamiliar would quarantine healthy conversations.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from ..domain.conversation import (
    KIND_MESSAGE,
    KIND_TEXT,
    KIND_THINKING,
    KIND_TOOL,
    Message,
    NormalizedConversation,
    ToolEvent,
)
from ..errors import RecordDefect
from ..jsonio import loads_untrusted

#: The largest export either conversation source reads: the bound on what the
#: command prints, and on each file of a fixtures directory.
MAX_EXPORT_BYTES = 32 * 1024 * 1024

DEFAULT_OUTPUT_EXCERPT_CHARS = 2000
DEFAULT_INPUT_EXCERPT_CHARS = 800


def normalize_export(
    payload: Any,
    *,
    output_excerpt_chars: int = DEFAULT_OUTPUT_EXCERPT_CHARS,
    input_excerpt_chars: int = DEFAULT_INPUT_EXCERPT_CHARS,
    max_messages: int | None = None,
) -> NormalizedConversation:
    """Convert an export payload into a :class:`NormalizedConversation`."""
    if not isinstance(payload, Mapping):
        raise RecordDefect(f"export must be a JSON object, got {type(payload).__name__}")
    thread = payload.get("thread")
    if not isinstance(thread, Mapping):
        raise RecordDefect("export has no 'thread' object")
    thread_id = thread.get("id")
    if not isinstance(thread_id, str) or not thread_id:
        raise RecordDefect("export thread has no usable 'id'")
    raw_elements = payload.get("elements")
    if not isinstance(raw_elements, list):
        raise RecordDefect("export has no 'elements' array")

    elements: list[dict[str, Any]] = []
    for item in raw_elements:
        if isinstance(item, Mapping) and isinstance(item.get("id"), str):
            elements.append(dict(item))
    # Positions are sorted in practice; sorting defensively costs nothing and
    # makes the parser independent of export ordering.
    elements.sort(key=lambda e: (_as_int(e.get("position"), 0), str(e.get("id"))))

    element_index: dict[str, int] = {}
    element_kinds: dict[str, str] = {}
    for el in elements:
        element_index[str(el["id"])] = _as_int(el.get("position"), 0)
        element_kinds[str(el["id"])] = str(el.get("kind", ""))

    children: dict[str, list[dict[str, Any]]] = {}
    roots: list[dict[str, Any]] = []
    for el in elements:
        parent = el.get("parentId")
        if isinstance(parent, str) and parent:
            children.setdefault(parent, []).append(el)
        else:
            roots.append(el)

    messages: list[Message] = []
    truncated = False
    for el in roots:
        if str(el.get("kind")) != KIND_MESSAGE:
            continue
        if max_messages is not None and len(messages) >= max_messages:
            truncated = True
            break
        message_id = str(el["id"])
        kids = sorted(
            children.get(message_id, []),
            key=lambda e: (_as_int(e.get("position"), 0), str(e.get("id"))),
        )
        text_parts: list[str] = []
        text_ids: list[str] = []
        tools: list[ToolEvent] = []
        for kid in kids:
            kind = str(kid.get("kind", ""))
            if kind == KIND_TEXT:
                content = kid.get("content")
                if isinstance(content, str) and content:
                    text_parts.append(content)
                text_ids.append(str(kid["id"]))
            elif kind == KIND_TOOL:
                tools.append(
                    _tool_event(
                        kid,
                        parent_message_id=message_id,
                        output_excerpt_chars=output_excerpt_chars,
                        input_excerpt_chars=input_excerpt_chars,
                    )
                )
            elif kind == KIND_THINKING:
                # Thinking is deliberately excluded from message text: it is not
                # what the user saw, and scoring the visible behaviour is the
                # point. It stays in element_index so it can still be cited.
                continue
        messages.append(
            Message(
                element_id=message_id,
                position=_as_int(el.get("position"), 0),
                role=str(el.get("role", "")),
                text="\n\n".join(text_parts),
                author=_opt_str(el.get("author")),
                origin=_opt_str(el.get("origin")),
                tool_events=tuple(tools),
                text_element_ids=tuple(text_ids),
            )
        )

    return NormalizedConversation(
        thread_id=thread_id,
        title=str(thread.get("title", "")),
        workspace=str(thread.get("workspacePath", "")),
        profile=str(thread.get("profileName", "")),
        agent_name=str(thread.get("agentName", "")),
        created_at=_opt_int(thread.get("createdAt")),
        updated_at=_opt_int(thread.get("updatedAt")),
        messages=tuple(messages),
        element_index=element_index,
        element_kinds=element_kinds,
        total_elements=len(elements),
        truncated=truncated,
        source_kind="conversation-export",
    )


def normalize_export_text(text: str, **kwargs: Any) -> NormalizedConversation:
    try:
        payload = loads_untrusted(text)
    except ValueError as exc:
        raise RecordDefect(f"export is not valid JSON: {exc}") from exc
    return normalize_export(payload, **kwargs)


def _tool_event(
    raw: Mapping[str, Any],
    *,
    parent_message_id: str,
    output_excerpt_chars: int,
    input_excerpt_chars: int,
) -> ToolEvent:
    output = raw.get("output")
    output_text = output if isinstance(output, str) else _compact_json(output)
    input_value = raw.get("input")
    # `input` is an object for normal calls and a bare string in some records.
    input_text = input_value if isinstance(input_value, str) else _compact_json(input_value)
    return ToolEvent(
        element_id=str(raw["id"]),
        position=_as_int(raw.get("position"), 0),
        name=str(raw.get("name", "")),
        status=str(raw.get("status", "")),
        input_text=_clip(input_text, input_excerpt_chars),
        output_excerpt=_clip(output_text, output_excerpt_chars),
        output_bytes=len(output_text.encode("utf-8")),
        duration_ms=_opt_int(raw.get("durationMs")),
        parent_message_id=parent_message_id,
    )


def _compact_json(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f" ...[clipped {len(text) - limit} chars]"


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # ``loads_untrusted`` never yields a non-finite float, but this is a
        # public normalizer over any payload, and ``int(inf)`` raises.
        return int(value) if math.isfinite(value) else default
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None
