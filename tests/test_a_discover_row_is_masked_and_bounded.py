"""A ``discover`` row shows the export's id, workspace and title masked and bounded.

``discover`` lists each thread the source would ingest as one text row, "ID
WORKSPACE TITLE". All three are the export's (``thread.id``,
``thread.workspacePath``, ``thread.title``) or a listing's, untrusted and
unbounded. Measured before (3.11 and 3.13, the same on both): a
200,000-character workspace or id made a row of that length and a listing of
about 200 KB where the healthy fixtures list in 455 bytes, and a credential in a
title or an id was printed whole (the title was cut at 60 characters, after the
credential). Each is now shown as ``listed_value`` shows a stored record's
values: masked, then cut (the id at 120 characters, the workspace at 4,096, the
title at 60), the cut saying how much it left out, with no quotes added, so a
healthy row lists exactly as it did. The JSON ``threads`` keep each record whole.
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.domain.discovery import MAX_THREAD_ID_LENGTH
from agent_skill_improver.domain.redaction import LISTED_PATH_CHARS, QUOTED_VALUE_CHARS
from tests.support import FIXTURE_EXPORTS, build_env, run_cli

GITHUB_TOKEN = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
MASK = "[REDACTED:github-token]"
LONG = 200_000
CONTROL = (
    "1700000000001-1  /tmp/asi-fixture-workspace-a  Add a docstring to the parser helper\n"
    "1700000000005-1  /tmp/asi-fixture-workspace-d  Tail the build log\n"
    "1700000000002-1  /tmp/asi-fixture-workspace-b  Reset the migration branch\n"
    "1700000000003-1  /tmp/asi-fixture-workspace-c  Summarise the vendor README\n"
    "1700000000004-1  /tmp/asi-fixture-workspace-d  Explain the retry policy\n"
    "5 thread(s) returned, limit 50\n"
    "this is the WHOLE population the source would return\n"
)


def discover(fields: dict[str, str], *, as_json: bool = False) -> str:
    """``discover``'s stdout with the compliant export's thread fields replaced."""
    with TemporaryDirectory() as tmp:
        base = Path(tmp).resolve()
        exports = base / "exports"
        shutil.copytree(FIXTURE_EXPORTS, exports)
        path = exports / "compliant.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["thread"].update(fields)
        path.write_text(json.dumps(payload), encoding="utf-8")
        config = build_env(base / "env", fixtures_dir=exports, seed="c179")
        argv = ["--config", str(config), *(["--json"] if as_json else []), "discover"]
        proc = run_cli(argv)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout


def the_row(stdout: str, start: str) -> str:
    (row,) = [line for line in stdout.splitlines() if line.startswith(start)]
    return row


class ARowIsBoundedTests(unittest.TestCase):
    def test_a_long_workspace_is_cut_in_its_row(self) -> None:
        workspace = "/ws/" + "w" * LONG
        stdout = discover({"workspacePath": workspace})
        row = the_row(stdout, "1700000000001-1  ")
        self.assertIn(
            f"  {workspace[:LISTED_PATH_CHARS]}... [{len(workspace) - LISTED_PATH_CHARS} more "
            "chars not shown]  Add a docstring",
            row,
        )
        self.assertLess(len(stdout), 6_000)

    def test_a_long_id_is_cut_in_its_row(self) -> None:
        # The longest id the source takes: a longer one is passed over before
        # it is listed. It is still past the row's cut.
        thread_id = "t-" + "9" * (MAX_THREAD_ID_LENGTH - 2)
        stdout = discover({"id": thread_id})
        row = the_row(stdout, "t-")
        self.assertTrue(
            row.startswith(
                f"{thread_id[:QUOTED_VALUE_CHARS]}... [{len(thread_id) - QUOTED_VALUE_CHARS} "
                "more chars not shown]  /tmp/asi-fixture-workspace-a  "
            )
        )
        self.assertLess(len(stdout), 1_000)

    def test_a_title_up_to_sixty_characters_is_shown_whole(self) -> None:
        title = "T" * 60
        row = the_row(discover({"title": title}), "1700000000001-1  ")
        self.assertTrue(row.endswith(f"  {title}"), row)
        self.assertNotIn("more chars not shown", row)

    def test_a_longer_title_says_how_much_it_left_out(self) -> None:
        title = "T" * 61
        row = the_row(discover({"title": title}), "1700000000001-1  ")
        self.assertTrue(row.endswith(f"  {'T' * 60}... [1 more chars not shown]"), row)


class ACredentialIsMaskedTests(unittest.TestCase):
    def test_a_credential_in_a_title_is_masked(self) -> None:
        stdout = discover({"title": f"deploy {GITHUB_TOKEN} " + "t" * LONG})
        self.assertNotIn(GITHUB_TOKEN, stdout)
        row = the_row(stdout, "1700000000001-1  ")
        self.assertIn(f"  deploy {MASK} ", row)
        left_out = len(f"deploy {MASK} ") + LONG - 60
        self.assertTrue(row.endswith(f"... [{left_out} more chars not shown]"), row[-80:])
        self.assertLess(len(row), 200)

    def test_a_credential_in_an_id_is_masked(self) -> None:
        stdout = discover({"id": f"t-{GITHUB_TOKEN}"})
        self.assertNotIn(GITHUB_TOKEN, stdout)
        self.assertTrue(the_row(stdout, "t-").startswith(f"t-{MASK}  "))

    def test_a_credential_in_a_workspace_is_masked(self) -> None:
        stdout = discover({"workspacePath": f"/ws/{GITHUB_TOKEN}/repo"})
        self.assertNotIn(GITHUB_TOKEN, stdout)
        self.assertIn(f"  /ws/{MASK}/repo  ", the_row(stdout, "1700000000001-1  "))


class TheControlTests(unittest.TestCase):
    def test_the_control_healthy_rows_list_exactly_as_before(self) -> None:
        self.assertEqual(discover({}), CONTROL)

    def test_the_control_the_json_keeps_each_record_whole(self) -> None:
        workspace = "/ws/" + "w" * LONG
        threads = json.loads(discover({"workspacePath": workspace}, as_json=True))["threads"]
        (thread,) = [t for t in threads if t["thread_id"] == "1700000000001-1"]
        self.assertEqual(thread["workspace"], workspace)


if __name__ == "__main__":
    unittest.main()
