"""A discovery value is sent as written, though it names a placeholder.

``render_argv`` in ``adapters/conversation_cli.py`` fills the discovery
command's placeholders (``{since}``, ``{limit}``, ``{query}``,
``{workspace}``) from the window, the page size, the configured query and the
workspace. It filled them one after another, so a value holding the name of a
placeholder filled after it had that one's value put in its place. Measured
before (3.11 and 3.13, the same on both): of 52 cases (a window, a query or a
workspace naming another placeholder, alone or in other text, with that other
value set or empty, for the default templates and one with the query inline),
28 sent other text than the value, and 6 dropped an argument: a window naming
an empty query or workspace rendered empty and went with ``--after``, so
discovery ran with no window, and a query naming an empty workspace was not
sent, so the search was not narrowed by it. Now each placeholder is filled
once, in one pass.
"""

from __future__ import annotations

import unittest
from typing import Any

from agent_skill_improver.adapters.conversation_cli import ConversationCliSource, render_argv

#: The templates of ``config/example.config.toml``, and the same with the query
#: written inline, written out so a change in the defaults fails here.
DEFAULT = {
    "discovery_argv": ("conversation-cli", "search", "--after", "{since}", "-n", "{limit}"),
    "query_argv": ("{query}",),
    "workspace_argv": ("-w", "{workspace}"),
}
INLINE = dict(
    DEFAULT,
    discovery_argv=("conversation-cli", "search", "{query}", "--after", "{since}", "-n", "{limit}"),
)
NAMES = ("since", "limit", "query", "workspace")


class _Recorded(ConversationCliSource):
    """Records the discovery argv instead of running it."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.argvs: list[list[str]] = []

    def _run(self, argv: Any, **kwargs: Any) -> str:  # type: ignore[override]
        self.argvs.append(list(argv))
        return ""


def _sent(templates: dict[str, Any], **values: Any) -> list[str]:
    source = _Recorded(**templates)
    source.discover_page(
        since=values["since"],
        limit=values["limit"],
        query=values.get("query") or None,
        workspace=values.get("workspace") or None,
    )
    return source.argvs[0]


def _as_written(templates: dict[str, Any], since: str, query: str, workspace: str) -> list[str]:
    """The argv each template asks for, built by hand from the values as written."""
    window = ["--after", since, "-n", "40"]
    asked = ["-w", workspace] if workspace else []
    if templates is INLINE:
        return ["conversation-cli", "search", *([query] if query else []), *window, *asked]
    return ["conversation-cli", "search", *window, *([query] if query else []), *asked]


class AValueNamingAPlaceholderIsSentAsWrittenTests(unittest.TestCase):
    def test_every_value_naming_another_placeholder_is_sent_as_written(self) -> None:
        base = {"since": "1d", "query": "errors", "workspace": "/w"}
        for name, templates in (("default", DEFAULT), ("inline", INLINE)):
            for field in ("since", "query", "workspace"):
                for other in NAMES:
                    if other == field:
                        continue
                    for form in ("{%s}", "notes on {%s} here"):
                        for empty in (False, True):
                            if empty and other in ("since", "limit"):
                                continue
                            values = dict(base, **{field: form % other})
                            if empty:
                                values[other] = ""
                            with self.subTest(
                                templates=name, field=field, value=values[field], empty=empty
                            ):
                                self.assertEqual(
                                    _sent(templates, limit=40, **values),
                                    _as_written(templates, **values),
                                )

    def test_a_window_naming_an_empty_query_keeps_its_flag(self) -> None:
        # Before, the window rendered empty and went with ``--after``: no window.
        self.assertEqual(
            _sent(DEFAULT, since="{query}", limit=40, workspace="/w"),
            ["conversation-cli", "search", "--after", "{query}", "-n", "40", "-w", "/w"],
        )
        self.assertEqual(
            _sent(DEFAULT, since="{workspace}", limit=40, query="errors"),
            ["conversation-cli", "search", "--after", "{workspace}", "-n", "40", "errors"],
        )

    def test_a_query_naming_an_empty_workspace_is_sent(self) -> None:
        for templates in (DEFAULT, INLINE):
            with self.subTest(inline=templates is INLINE):
                self.assertIn(
                    "{workspace}", _sent(templates, since="1d", limit=40, query="{workspace}")
                )

    def test_a_value_with_backslashes_or_group_references_is_sent_as_written(self) -> None:
        # Filled by a function, never read as a replacement template.
        query = "\\1 \\g<0> \\\\ \\n"
        since = "\\g<since>"
        argv = _sent(DEFAULT, since=since, limit=40, query=query)
        self.assertEqual(argv[argv.index("--after") + 1], since)
        self.assertEqual(argv[-1], query)


class EachPlaceholderIsFilledOnceTests(unittest.TestCase):
    def test_one_argument_holding_several_placeholders_fills_each_once(self) -> None:
        values = {"since": "{limit}", "limit": "40", "query": "", "workspace": ""}
        self.assertEqual(
            render_argv(("--x={since}:{limit}", "{limit}-{limit}"), values),
            ["--x={limit}:40", "40-40"],
        )

    def test_a_name_that_is_no_value_here_is_left_as_written(self) -> None:
        values = {"since": "1d", "limit": "40", "query": "", "workspace": ""}
        self.assertEqual(
            render_argv(("cli", "{thread_id}", "{unknown}", "{since}"), values),
            ["cli", "{thread_id}", "{unknown}", "1d"],
        )

    def test_a_thread_id_naming_its_own_placeholder_is_sent_as_written(self) -> None:
        argv = ("conversation-cli", "export", "--thread", "{thread_id}")
        self.assertEqual(
            render_argv(argv, {"thread_id": "{thread_id}"}),
            ["conversation-cli", "export", "--thread", "{thread_id}"],
        )

    def test_with_no_values_the_template_is_left_as_written(self) -> None:
        self.assertEqual(render_argv(("cli", "{since}", ""), {}), ["cli", "{since}", ""])

    def test_the_control_an_empty_value_still_drops_its_argument_and_its_flag(self) -> None:
        self.assertEqual(
            _sent(DEFAULT, since="1d", limit=40, query="", workspace=""),
            ["conversation-cli", "search", "--after", "1d", "-n", "40"],
        )
        values = {"since": "1d", "limit": "40", "query": "", "workspace": ""}
        self.assertEqual(
            render_argv(("cli", "--query", "{query}", "-n", "{limit}"), values),
            ["cli", "-n", "40"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
