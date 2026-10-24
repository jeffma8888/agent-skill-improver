"""``apply`` prints each line of what it did on its own line, as ``review`` does.

Measured before (3.11 and 3.13), with a real validated candidate whose skill
directory was then renamed to a name holding a line break and the record
pointed at it as ``daily`` would have written it: ``review`` named the path on
one line, spelling the break, but ``apply``'s text printed it as written, so
the text after the break started a line of its own, once in the ``applied``
line and once in the manual rollback command. A state directory named so did
the same through the backup path, in the ``backup`` line and the manual command.
``--json`` was already escaped.
"""

from __future__ import annotations

import json
import shlex
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_skill_improver.domain.candidate import make_patch
from tests.support import build_env, run_cli

FORGED = "config error: forged"
SEED = "apply-prints-each-line"
NOT_SHOWN = (
    "  (or manually)   not shown: a path in it holds a line break or a control character, "
    "which this line can only spell; `asi rollback {cid}` restores the same backup"
)


def _applied(
    base: Path, directory: str | None, *, as_json: bool = False, state_dir: str = "state"
) -> dict[str, Any]:
    """Propose, move the skill to ``directory`` (record pointed at it as ``daily``
    would write it), review and apply; return what the run printed and the paths.
    ``state_dir`` names the state directory, where the backup is written."""
    config = build_env(base, dry_run=False, seed=SEED)
    if state_dir != "state":
        text = config.read_text(encoding="utf-8")
        assert text.count('state_dir = "state"\n') == 1
        config.write_text(
            text.replace('state_dir = "state"\n', f"state_dir = {json.dumps(state_dir)}\n"),
            encoding="utf-8",
        )
    cfg = ["--config", str(config)]
    proc = run_cli([*cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
    cid = next(
        c["candidate_id"]
        for c in json.loads(proc.stdout)["candidates"]
        if c.get("status") == "validated"
    )
    record = base / state_dir / "candidates" / f"{cid}.json"
    raw = json.loads(record.read_text(encoding="utf-8"))
    target = Path(raw["target"]["canonical_path"])
    if directory is not None:
        moved = target.parent.parent / directory
        target.parent.rename(moved)
        target = moved / target.name
        raw["target"]["skill_id"] = directory
        raw["target"]["canonical_path"] = str(target)
        raw["patch"]["unified_diff"] = make_patch(
            target.read_text(encoding="utf-8"), raw["patch"]["new_text"], path_label=str(target)
        ).unified_diff
        record.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review = run_cli([*cfg, "--json", "review", cid])
    assert review.returncode == 0, review.stderr
    token = json.loads(review.stdout)["approval_token"]
    flags = ["--json"] if as_json else []
    apply = run_cli([*cfg, *flags, "apply", cid, "--approve", token, "--i-reviewed-this"])
    assert apply.returncode == 0, apply.stderr
    backup = next((base / state_dir / "backups").iterdir()) / f"{target.name}.bak"
    return {
        "cid": cid,
        "target": target,
        "backup": backup,
        "stdout": apply.stdout,
        "new_text": raw["patch"]["new_text"],
    }


class ApplyPrintsEachLineOnItsLineTests(unittest.TestCase):
    def _shas(self, lines: list[str]) -> tuple[str, str]:
        return lines[1].split()[-1], lines[2].split()[-1]

    def test_a_skill_directory_named_with_a_line_break_is_applied_one_line_each(self):
        with TemporaryDirectory() as d:
            run = _applied(Path(d).resolve(), f"tdd\n{FORGED}")
            written = run["target"].read_text(encoding="utf-8")
        lines = run["stdout"].splitlines()
        self.assertEqual(written, run["new_text"])
        self.assertFalse([line for line in lines if line.startswith(FORGED)])
        spelled = str(run["target"]).replace("\n", "\\n")
        previous, new = self._shas(lines)
        self.assertEqual(
            lines,
            [
                f"applied {run['cid']} to {spelled}",
                f"  previous sha256 {previous}",
                f"  new sha256      {new}",
                f"  backup          {run['backup']}",
                f"  rollback        asi rollback {run['cid']}",
                NOT_SHOWN.format(cid=run["cid"]),
            ],
        )

    def test_a_state_directory_named_with_a_line_break_hides_the_manual_command_too(self):
        with TemporaryDirectory() as d:
            run = _applied(Path(d).resolve(), None, state_dir=f"state\n{FORGED}")
            written = run["target"].read_text(encoding="utf-8")
        lines = run["stdout"].splitlines()
        self.assertEqual(written, run["new_text"])
        self.assertFalse([line for line in lines if line.startswith(FORGED)])
        previous, new = self._shas(lines)
        spelled = str(run["backup"]).replace("\n", "\\n")
        self.assertEqual(
            lines,
            [
                f"applied {run['cid']} to {run['target']}",
                f"  previous sha256 {previous}",
                f"  new sha256      {new}",
                f"  backup          {spelled}",
                f"  rollback        asi rollback {run['cid']}",
                NOT_SHOWN.format(cid=run["cid"]),
            ],
        )

    def test_the_control_a_plain_path_is_printed_and_the_manual_command_shown_as_before(self):
        with TemporaryDirectory() as d:
            run = _applied(Path(d).resolve(), None)
        lines = run["stdout"].splitlines()
        previous, new = self._shas(lines)
        manual = shlex.join(["cp", str(run["backup"]), str(run["target"])])
        self.assertEqual(
            lines,
            [
                f"applied {run['cid']} to {run['target']}",
                f"  previous sha256 {previous}",
                f"  new sha256      {new}",
                f"  backup          {run['backup']}",
                f"  rollback        asi rollback {run['cid']}",
                f"  (or manually)   {manual}",
            ],
        )
        self.assertEqual(
            shlex.split(lines[-1].split("   ", 1)[1]),
            ["cp", str(run["backup"]), str(run["target"])],
        )

    def test_a_path_a_shell_must_quote_keeps_its_manual_command(self):
        with TemporaryDirectory() as d:
            run = _applied(Path(d).resolve(), "my skill's\ttab")
        lines = run["stdout"].splitlines()
        self.assertEqual(lines[0], f"applied {run['cid']} to {run['target']}")
        command = lines[-1].split("   ", 1)[1]
        self.assertEqual(command, shlex.join(["cp", str(run["backup"]), str(run["target"])]))
        self.assertEqual(shlex.split(command), ["cp", str(run["backup"]), str(run["target"])])

    def test_a_backslash_or_a_letter_outside_ascii_keeps_its_manual_command(self):
        # None is spelled on the line, so the command reads as itself: a
        # backslash and an "n" are two characters a path can hold, and a letter
        # outside ASCII or a no-break space is printed as itself.
        for directory in ("tdd\\nx", "my skill \u00fc-\u65e5\u672c", "my\u00a0skill"):
            with self.subTest(directory=directory), TemporaryDirectory() as d:
                run = _applied(Path(d).resolve(), directory)
                lines = run["stdout"].splitlines()
                previous, new = self._shas(lines)
                manual = shlex.join(["cp", str(run["backup"]), str(run["target"])])
                self.assertEqual(
                    lines,
                    [
                        f"applied {run['cid']} to {run['target']}",
                        f"  previous sha256 {previous}",
                        f"  new sha256      {new}",
                        f"  backup          {run['backup']}",
                        f"  rollback        asi rollback {run['cid']}",
                        f"  (or manually)   {manual}",
                    ],
                )
                self.assertEqual(
                    shlex.split(lines[-1].split("   ", 1)[1]),
                    ["cp", str(run["backup"]), str(run["target"])],
                )

    def test_a_carriage_return_in_the_path_hides_the_manual_command(self):
        with TemporaryDirectory() as d:
            run = _applied(Path(d).resolve(), f"tdd\r{FORGED}")
            written = run["target"].read_text(encoding="utf-8")
        self.assertEqual(written, run["new_text"])
        self.assertNotIn("\r", run["stdout"])
        lines = run["stdout"].split("\n")[:-1]
        spelled = str(run["target"]).replace("\r", "\\r")
        previous, new = self._shas(lines)
        self.assertEqual(
            lines,
            [
                f"applied {run['cid']} to {spelled}",
                f"  previous sha256 {previous}",
                f"  new sha256      {new}",
                f"  backup          {run['backup']}",
                f"  rollback        asi rollback {run['cid']}",
                NOT_SHOWN.format(cid=run["cid"]),
            ],
        )

    def test_a_control_character_in_the_path_hides_the_manual_command(self):
        # Each one ``escape_for_terminal`` spells, from the C0 set and outside
        # it: a bidi override, a line separator, DEL and a C1 control.
        for character, spelled in (
            ("\x1b", "\\x1b"),
            ("\u202e", "\\u202e"),
            ("\u2028", "\\u2028"),
            ("\x7f", "\\x7f"),
            ("\x85", "\\x85"),
        ):
            with self.subTest(spelled=spelled), TemporaryDirectory() as d:
                run = _applied(Path(d).resolve(), f"tdd{character}[2K")
                self.assertNotIn(character, run["stdout"])
                lines = run["stdout"].split("\n")[:-1]
                self.assertEqual(
                    lines[0], f"applied {run['cid']} to {run['target']}".replace(character, spelled)
                )
                self.assertEqual(len(lines), 6)
                self.assertEqual(lines[-1], NOT_SHOWN.format(cid=run["cid"]))

    def test_json_still_carries_the_paths_and_the_command_as_they_are(self):
        with TemporaryDirectory() as d:
            run = _applied(Path(d).resolve(), f"tdd\n{FORGED}", as_json=True)
        payload = json.loads(run["stdout"])
        self.assertEqual(payload["write"]["path"], str(run["target"]))
        self.assertEqual(
            payload["write"]["rollback_command"],
            shlex.join(["cp", str(run["backup"]), str(run["target"])]),
        )


if __name__ == "__main__":
    unittest.main()
