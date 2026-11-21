"""``asi history``: a record it cannot trust is named, and the rest are shown.

Each test writes fabricated records into a fresh state tree, copies of one real
validated candidate (built once by ``daily``) moved to the test's own skill
roots, and runs the command in-process through ``cli.main``. A torn, misbound,
newer or non-recomputing record, a malformed nested field, a receipt that names
another candidate or holds another schema, an outside path, a live copy that is
not UTF-8 or that discovery passed over: each is named, claims no live match
it cannot back, and never stops the other rows from being shown (exit 0).

What a row says about the skill file is what the copy read during this
command's discovery held (``HISTORY_LIVE_BASIS``); the spies here assert that
nothing is read a second time and that no path a record names is opened.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import shutil
import signal
import tempfile
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

from agent_skill_improver import cli
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.clock import SystemClock
from agent_skill_improver.domain.candidate import Candidate, challenge_for
from agent_skill_improver.domain.integrity import derived_field_mismatches, validation_mismatches
from agent_skill_improver.hashing import sha256_text
from tests.support import build_env, run_cli

TOKEN = "ghp_" + "Q7" * 18
#: ``app/history.HISTORY_LIVE_BASIS``, spelled here so the module imports on a
#: tree without ``app/history.py`` and fails there by assertion, not ImportError;
#: the import-check test pins that the two agree.
HISTORY_LIVE_BASIS = "the copy read during this command's discovery"
HISTORY_SOURCE = Path(cli.__file__).parent / "app" / "history.py"
TEMPLATE_ID = "CAND-2026-01-03-01-tdd"
ORIGINAL_GROUPS = FilesystemSkillStore.groups
ORIGINAL_ITER = FilesystemSkillStore._iter_skill_files
ORIGINAL_STORE_READ_TEXT = FilesystemSkillStore.read_text
ORIGINAL_OPEN = os.open
ORIGINAL_PATH_READ_BYTES = Path.read_bytes
ORIGINAL_PATH_READ_TEXT = Path.read_text
ORIGINAL_PATH_OPEN = Path.open
PRESENT_TENSE = ("holds now", "at read time", "live file holds", "holds this candidate")
CAPPED = re.compile(r"\.\.\. \[\d+ more chars not shown\]$")

_TEMPLATE: dict[str, Any] = {}


def setUpModule() -> None:
    tmp = tempfile.TemporaryDirectory()
    _TEMPLATE["tmp"] = tmp
    base = Path(tmp.name)
    config = build_env(base, dry_run=False, seed="probe")
    proc = run_cli(
        ["--config", str(config), "daily", "--commit", "--day", "2026-01-03"], timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    _TEMPLATE["base"] = base
    _TEMPLATE["record"] = json.loads(
        (base / "state" / "candidates" / f"{TEMPLATE_ID}.json").read_text(encoding="utf-8")
    )


def tearDownModule() -> None:
    _TEMPLATE["tmp"].cleanup()


def _rebased(value: str, old: Path, new: Path) -> str:
    for before, after in ((str(old.resolve()), str(new.resolve())), (str(old), str(new))):
        if value.startswith(before + os.sep):
            return after + value[len(before) :]
    raise AssertionError(f"{value!r} is not under {old}")


def seal(record: dict) -> dict:
    """Re-derive what a changed id, base or text invalidates, so the record recomputes."""
    new_hash = sha256_text(record["patch"]["new_text"])
    record["patch"]["new_text_sha256"] = new_hash
    if isinstance(record.get("validation"), dict):
        record["validation"]["candidate_id"] = record["id"]
        record["validation"]["patch_sha256"] = new_hash
        record["validation"]["base_sha256"] = record["target"]["base_sha256"]
    record["approval_challenge"] = challenge_for(
        record["id"], record["target"]["base_sha256"], new_hash
    )
    return record


def record_for(base: Path, cid: str, created_at: str, **changes: Any) -> dict:
    """The real template under ``cid``, its paths moved to ``base``'s skill roots."""
    record = json.loads(json.dumps(_TEMPLATE["record"]))
    record["id"] = cid
    record["created_at"] = created_at
    target = record["target"]
    target["canonical_path"] = _rebased(target["canonical_path"], _TEMPLATE["base"], base)
    target["mirror_paths"] = [
        _rebased(p, _TEMPLATE["base"], base) for p in target.get("mirror_paths", [])
    ]
    record.update(changes)
    return seal(record)


def _private(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    os.chmod(directory.parent, 0o700)


def write_record(state: Path, record: dict) -> Path:
    _private(state / "candidates")
    path = state / "candidates" / f"{record['id']}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def write_torn(state: Path, name: str) -> Path:
    _private(state / "candidates")
    path = state / "candidates" / name
    path.write_text('{"schema": "asi.candidate.v1", "id": ', encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def write_receipt(state: Path, action: str, cid: str, payload: Any) -> Path:
    _private(state / "receipts")
    path = state / "receipts" / f"{action}--{cid}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def apply_receipt(cid: str, **changes: Any) -> dict:
    receipt = {
        "schema": "asi.apply_receipt.v1",
        "candidate_id": cid,
        "applied_at": "2026-01-03T12:30:00Z",
        "operator": "alice",
        "approval": {"note": "probe note"},
        "write": {"backup_manifest_path": "/nowhere/backups/x/backup.json"},
        "gate": {},
        "mirror_paths_not_written": [],
    }
    receipt.update(changes)
    return receipt


def rollback_receipt(cid: str, **changes: Any) -> dict:
    receipt = {
        "schema": "asi.rollback_receipt.v1",
        "candidate_id": cid,
        "manifest": "/nowhere/backups/x/backup.json",
        "result": {},
        "rolled_back_at": "2026-01-03T13:00:00Z",
    }
    receipt.update(changes)
    return receipt


def _run(config: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main(["--config", str(config), *argv])
        except SystemExit as exc:  # argparse refusing the command line: exit 2
            rc = exc.code
    return rc, out.getvalue(), err.getvalue()


def _private_keys(obj: Any, where: str = "") -> list[str]:
    """Every key starting ``_`` anywhere in the payload (``_live``'s ``_path``, ``_not_compared``)."""
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).startswith("_"):
                found.append(f"{where}.{key}")
            found.extend(_private_keys(value, f"{where}.{key}"))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(_private_keys(value, f"{where}[{index}]"))
    return found


class _Hang(BaseException):
    pass


def _on_alarm(signum: int, frame: object) -> None:
    raise _Hang()


@contextlib.contextmanager
def _bounded(seconds: int = 20) -> Iterator[None]:
    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        yield
    except _Hang:
        raise AssertionError(f"waited past the {seconds} s alarm") from None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


class _ReadSpy:
    """Records, once armed, every open of a watched inode (``os.open``), every
    ``Path`` read or open below a watched directory, and every
    ``FilesystemSkillStore.read_text`` call."""

    def __init__(self, *watched: Path, armed: bool = False) -> None:
        self.prefixes = sorted({str(w) for w in watched} | {str(w.resolve()) for w in watched})
        self.inodes: set[tuple[int, int]] = set()
        for top in watched:
            for entry in [top, *top.rglob("*")]:
                found = entry.lstat()
                self.inodes.add((found.st_dev, found.st_ino))
        self.armed = armed
        self.hits: list[str] = []
        self._stack = contextlib.ExitStack()

    def _under(self, path: Any) -> bool:
        named = os.fspath(path)
        return any(
            text == p or text.startswith(p + os.sep)
            for text in (named, os.path.realpath(named))
            for p in self.prefixes
        )

    def __enter__(self) -> _ReadSpy:
        spy = self

        def opened(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
            fd = ORIGINAL_OPEN(path, flags, mode, dir_fd=dir_fd)
            found = os.fstat(fd)
            if spy.armed and (found.st_dev, found.st_ino) in spy.inodes:
                spy.hits.append(f"os.open {path!r}")
            return fd

        def read_bytes(path: Path) -> bytes:
            if spy.armed and spy._under(path):
                spy.hits.append(f"Path.read_bytes {path}")
            return ORIGINAL_PATH_READ_BYTES(path)

        def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
            if spy.armed and spy._under(path):
                spy.hits.append(f"Path.read_text {path}")
            return ORIGINAL_PATH_READ_TEXT(path, *args, **kwargs)

        def path_open(path: Path, *args: Any, **kwargs: Any) -> Any:
            if spy.armed and spy._under(path):
                spy.hits.append(f"Path.open {path}")
            return ORIGINAL_PATH_OPEN(path, *args, **kwargs)

        def store_read_text(store: FilesystemSkillStore, *args: Any, **kwargs: Any) -> Any:
            if spy.armed:
                spy.hits.append(f"store.read_text {args!r}")
            return ORIGINAL_STORE_READ_TEXT(store, *args, **kwargs)

        self._stack.enter_context(mock.patch.object(os, "open", opened))
        self._stack.enter_context(mock.patch.object(Path, "read_bytes", read_bytes))
        self._stack.enter_context(mock.patch.object(Path, "read_text", read_text))
        self._stack.enter_context(mock.patch.object(Path, "open", path_open))
        self._stack.enter_context(
            mock.patch.object(FilesystemSkillStore, "read_text", store_read_text)
        )
        return self

    def __exit__(self, *exc: object) -> None:
        self._stack.close()


@contextlib.contextmanager
def _after_groups(action: Callable[[], None]) -> Iterator[dict[str, int]]:
    """Run ``action`` once, right after ``store.groups()`` returns."""
    seen = {"done": 0}

    def wrapped(self: FilesystemSkillStore, *args: Any, **kwargs: Any) -> Any:
        found = ORIGINAL_GROUPS(self, *args, **kwargs)
        if not seen["done"]:
            seen["done"] = 1
            action()
        return found

    with mock.patch.object(FilesystemSkillStore, "groups", wrapped):
        yield seen


class _Fresh(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.config = build_env(self.base, dry_run=False, seed="probe")
        self.state = self.base / "state"
        self.live = self.base / "skills-canonical" / "tdd" / "SKILL.md"
        self.live_hash = sha256_text(self.live.read_text(encoding="utf-8"))

    def put(self, cid: str, created_at: str = "2026-01-03T12:00:00Z", **changes: Any) -> dict:
        record = record_for(self.base, cid, created_at, **changes)
        write_record(self.state, record)
        return record

    def text(self, *args: str, skill: str = "tdd") -> str:
        rc, out, err = _run(self.config, "history", skill, *args)
        self.assertEqual(rc, 0, err)
        return out

    def json(self, *args: str, skill: str = "tdd") -> dict:
        rc, out, err = _run(self.config, "--json", "history", skill, *args)
        self.assertEqual(rc, 0, err)
        payload = json.loads(out)
        self.assertEqual(_private_keys(payload), [], "a private key reached --json")
        return payload

    def row(self, payload: dict, cid: str) -> dict:
        found = [r for r in payload["candidates"] if r["id"] == cid]
        self.assertEqual(len(found), 1, [r["id"] for r in payload["candidates"]])
        return found[0]

    def block(self, out: str, cid: str) -> list[str]:
        """The text lines of one row: its head and the indented lines after it."""
        lines = out.splitlines()
        start = [i for i, line in enumerate(lines) if line.startswith(f"{cid}  ")]
        self.assertEqual(len(start), 1, out)
        rows = [lines[start[0]]]
        for line in lines[start[0] + 1 :]:
            if not line.startswith("  "):
                break
            rows.append(line)
        return rows


class UntrustedRecordTests(_Fresh):
    def test_a_torn_record_named_for_the_skill_is_unreadable_and_the_other_rows_are_shown(self):
        self.put("CAND-2026-01-01-01-tdd", "2026-01-01T12:00:00Z")
        self.put("CAND-2026-01-03-01-tdd", "2026-01-03T12:00:00Z")
        write_torn(self.state, "CAND-2026-01-02-01-tdd.json")
        payload = self.json()
        self.assertEqual(
            [r["id"] for r in payload["candidates"]],
            ["CAND-2026-01-01-01-tdd", "CAND-2026-01-03-01-tdd"],
        )
        self.assertEqual(
            [u["name"] for u in payload["unreadable"]], ["CAND-2026-01-02-01-tdd.json"]
        )
        self.assertTrue(payload["unreadable"][0]["reason"])
        self.assertEqual(payload["counts"]["unreadable"], 1)
        self.assertEqual(payload["unattributed_unreadable"], 0)
        out = self.text()
        unreadable = [line for line in out.splitlines() if line.startswith("UNREADABLE  ")]
        self.assertEqual(len(unreadable), 1, out)
        self.assertTrue(unreadable[0].startswith("UNREADABLE  CAND-2026-01-02-01-tdd.json  "))
        self.assertIn("1 unreadable record(s) named for this skill, 0 elsewhere", out)
        with self.subTest("an unreadable record names a skill not in the roots: exit 0, not 4"):
            write_torn(self.state, "CAND-2026-01-02-01-gone.json")
            rc, out, err = _run(self.config, "history", "gone")
            self.assertEqual((rc, err), (0, ""))
            lines = out.splitlines()
            self.assertIn("live: not in the configured skill roots now", lines)
            self.assertIn("0 candidate(s) recorded for this skill", lines)
            self.assertTrue(
                any(
                    line.startswith("UNREADABLE  CAND-2026-01-02-01-gone.json  ") for line in lines
                ),
                out,
            )
            self.assertTrue(
                lines[-1].startswith(
                    "0 candidate(s): 0 open, 0 closed, 0 of unknown status, 0 untrusted; "
                    "1 unreadable record(s) named for this skill, 1 elsewhere"
                ),
                lines[-1],
            )
            # Nothing was compared, so no live-match summary line is printed.
            self.assertFalse([line for line in lines if line.startswith("live match")], out)
            rc, out, _ = _run(self.config, "--json", "history", "gone")
            self.assertEqual(rc, 0)
            gone = json.loads(out)
            self.assertEqual(gone["live"]["state"], "not-in-roots")
            self.assertEqual(gone["counts"]["unreadable"], 1)
            self.assertEqual(gone["candidates"], [])

    def test_a_torn_record_of_another_skill_is_counted_not_attributed(self):
        self.put("CAND-2026-01-03-01-tdd")
        write_torn(self.state, "CAND-2026-01-02-01-code-review.json")
        payload = self.json()
        self.assertEqual(payload["unreadable"], [])
        self.assertEqual(payload["counts"]["unreadable"], 0)
        self.assertEqual(payload["unattributed_unreadable"], 1)
        out = self.text()
        self.assertNotIn("UNREADABLE", out)
        self.assertIn(
            "0 unreadable record(s) named for this skill, 1 elsewhere in candidates/ not attributed",
            out,
        )

    def test_a_record_that_does_not_recompute_is_marked_and_claims_no_live_match(self):
        cid = "CAND-2026-01-03-01-tdd"
        record = record_for(self.base, cid, "2026-01-03T12:00:00Z")
        record["approval_challenge"] = "0" * 64
        write_record(self.state, record)
        write_receipt(self.state, "apply", cid, apply_receipt(cid))
        self.assertEqual(record["target"]["base_sha256"], self.live_hash)
        payload = self.json()
        row = self.row(payload, cid)
        self.assertIs(row["integrity"]["recomputes"], False)
        problems = row["integrity"]["problems"]
        self.assertTrue(problems)
        self.assertEqual(row["live_match"], "not-compared")
        self.assertEqual(row["live_match_reason"], "the record does not recompute")
        self.assertEqual(payload["counts"]["untrusted"], 1)
        self.assertEqual(payload["counts"]["open"], 0)
        out = self.text()
        block = self.block(out, cid)
        self.assertNotIn("(open: asi review", block[0])
        count = len(problems) + row["integrity"]["problems_not_shown"]
        self.assertEqual(block[1], f"  DOES NOT RECOMPUTE ({count} problem(s)): {problems[0]}")
        self.assertIn("  live match  not compared: the record does not recompute", block)
        self.assertTrue(
            any(line.startswith("  (record untrusted) applied    ") for line in block), block
        )
        self.assertIn(
            f"live match: {HISTORY_LIVE_BASIS} matched no replacement among the 0 candidate(s) "
            "compared",
            out.splitlines(),
        )

    def test_an_unknown_status_is_shown_as_unknown_and_the_view_exits_0(self):
        cid = "CAND-2026-01-03-01-tdd"
        self.put(cid, status="superseded-in-v9")
        payload = self.json()
        row = self.row(payload, cid)
        self.assertEqual(row["status"], "superseded-in-v9")
        self.assertIs(row["status_known"], False)
        self.assertIs(row["open"], False)
        self.assertEqual(payload["counts"]["unknown_status"], 1)
        self.assertEqual(payload["counts"]["closed"], 0)
        out = self.text()
        self.assertIn("  status unknown to this asi: superseded-in-v9  ", self.block(out, cid)[0])
        self.assertIn("0 open, 0 closed, 1 of unknown status", out)

    def test_a_record_of_a_newer_schema_is_unreadable_not_an_error(self):
        self.put("CAND-2026-01-01-01-tdd", "2026-01-01T12:00:00Z")
        newer = record_for(self.base, "CAND-2026-01-03-01-tdd", "2026-01-03T12:00:00Z")
        newer["schema"] = "asi.candidate.v2"
        write_record(self.state, newer)
        payload = self.json()
        self.assertEqual([r["id"] for r in payload["candidates"]], ["CAND-2026-01-01-01-tdd"])
        self.assertEqual(
            [u["name"] for u in payload["unreadable"]], ["CAND-2026-01-03-01-tdd.json"]
        )
        reason = payload["unreadable"][0]["reason"]
        self.assertTrue(reason.startswith("refusing to use the candidate record "), reason)
        self.assertIn("it does not have the shape of a candidate", reason)
        self.assertIn("UNREADABLE  CAND-2026-01-03-01-tdd.json  ", self.text())

    def test_a_receipt_that_names_another_candidate_is_not_shown_as_this_ones(self):
        cid = "CAND-2026-01-01-01-tdd"
        other = "CAND-2026-01-09-01-tdd"
        self.put(cid, "2026-01-01T12:00:00Z", status="rolled-back")
        write_receipt(self.state, "apply", cid, apply_receipt(other, operator="mallory"))
        write_receipt(self.state, "rollback", cid, rollback_receipt(other))
        row = self.row(self.json(), cid)
        self.assertEqual(row["apply"], {"unreadable": "names another candidate"})
        self.assertEqual(row["rollback"], {"unreadable": "names another candidate"})
        out = self.text()
        self.assertNotIn("mallory", out)
        block = self.block(out, cid)
        self.assertIn("  apply      receipt unreadable: names another candidate", block)
        self.assertIn("  rollback   receipt unreadable: names another candidate", block)
        with self.subTest("a receipt of another schema is unreadable, not shown"):
            write_receipt(
                self.state,
                "apply",
                cid,
                apply_receipt(cid, schema="asi.apply_receipt.v9", operator="mallory"),
            )
            write_receipt(
                self.state, "rollback", cid, rollback_receipt(cid, schema="asi.rollback_receipt.v0")
            )
            row = self.row(self.json(), cid)
            self.assertEqual(
                row["apply"],
                {"unreadable": "holds schema asi.apply_receipt.v9, not asi.apply_receipt.v1"},
            )
            self.assertEqual(
                row["rollback"],
                {"unreadable": "holds schema asi.rollback_receipt.v0, not asi.rollback_receipt.v1"},
            )
            out = self.text()
            self.assertNotIn("mallory", out)
            self.assertIn(
                "  apply      receipt unreadable: holds schema asi.apply_receipt.v9, not "
                "asi.apply_receipt.v1",
                self.block(out, cid),
            )

    def test_a_record_naming_a_self_consistent_outside_skill_is_not_compared_and_not_opened(self):
        cid = "CAND-2026-01-03-01-tdd"
        outside = self.base / "outside"
        (outside / "tdd").mkdir(parents=True)
        (outside / "tdd" / "SKILL.md").write_bytes(self.live.read_bytes())
        record = record_for(self.base, cid, "2026-01-03T12:00:00Z")
        record["target"]["canonical_path"] = str(outside / "tdd" / "SKILL.md")
        write_record(self.state, record)
        with _ReadSpy(outside, armed=True) as spy:
            payload = self.json()
            out = self.text()
        self.assertEqual(spy.hits, [])
        row = self.row(payload, cid)
        self.assertIs(row["integrity"]["recomputes"], True, row["integrity"])
        self.assertEqual(row["live_match"], "not-compared")
        self.assertEqual(
            row["live_match_reason"],
            f"the record names {outside / 'tdd' / 'SKILL.md'}, which is not this skill's "
            "canonical copy now",
        )
        self.assertIn(
            f"  live match  not compared: {row['live_match_reason']}", self.block(out, cid)
        )

    def test_prose_is_masked_bounded_and_kept_on_one_line(self):
        cid = "CAND-2026-01-03-01-tdd"
        change = ("x" * 5000) + TOKEN + ("y" * 5000)
        self.put(cid, status="applied", intended_change=TOKEN + "\nFORGED " + change)
        write_receipt(
            self.state,
            "apply",
            cid,
            apply_receipt(
                cid, operator=f"ops {TOKEN}\nFORGED", approval={"note": f"n {TOKEN}\nFORGED"}
            ),
        )
        raw = json.dumps(self.json())
        self.assertNotIn(TOKEN, raw)
        row = self.row(json.loads(raw), cid)
        self.assertLessEqual(len(row["intended_change"]), 240)
        self.assertRegex(row["intended_change"], CAPPED)
        out = self.text()
        self.assertNotIn(TOKEN, out)
        self.assertFalse([line for line in out.splitlines() if line.startswith("FORGED")], out)
        block = self.block(out, cid)
        change_line = [line for line in block if line.startswith("  change     ")]
        self.assertEqual(len(change_line), 1)
        self.assertLess(len(change_line[0]), 300)
        applied = [line for line in block if line.startswith("  applied    ")]
        self.assertEqual(len(applied), 1, block)
        self.assertIn("FORGED", applied[0])
        self.assertIn("(unauthenticated)", applied[0])

    def test_no_patch_text_is_printed_in_text_or_json(self):
        cid = "CAND-2026-01-03-01-tdd"
        marker = "PATCH-TEXT-MARKER-c248b"
        record = record_for(self.base, cid, "2026-01-03T12:00:00Z")
        record["patch"]["new_text"] += f"\n{marker}\n"
        record["patch"]["unified_diff"] += f"\n+{marker}\n"
        write_record(self.state, seal(record))
        self.assertNotIn(marker, json.dumps(self.json()))
        self.assertNotIn(marker, self.text())

    def test_the_history_module_imports_no_model_adapter_and_no_promotion_code(self):
        self.assertTrue(HISTORY_SOURCE.is_file(), f"no {HISTORY_SOURCE}")
        tree = ast.parse(HISTORY_SOURCE.read_text(encoding="utf-8"))
        basis = [
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and [getattr(t, "id", None) for t in node.targets] == ["HISTORY_LIVE_BASIS"]
            and isinstance(node.value, ast.Constant)
        ]
        self.assertEqual(basis, [HISTORY_LIVE_BASIS])
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add("." * node.level + (node.module or ""))
        self.assertEqual(
            imported,
            {
                "__future__",
                "math",
                "re",
                "collections.abc",
                "typing",
                "..domain.candidate",
                "..domain.integrity",
                "..domain.redaction",
                "..domain.skills",
                "..errors",
                ".state",
            },
        )
        for name in imported:
            self.assertNotRegex(name, r"model|promot|adapters|daily|judge")

    def test_an_apply_receipt_beside_a_validated_record_is_named_as_an_apply_that_stopped(self):
        cid = "CAND-2026-01-03-01-tdd"
        self.put(cid)
        write_receipt(self.state, "apply", cid, apply_receipt(cid))
        row = self.row(self.json(), cid)
        self.assertEqual(row["status"], "validated")
        self.assertEqual(row["apply"]["operator"], "alice")
        block = self.block(self.text(), cid)
        self.assertIn(
            "  apply receipt present but the record says validated: an apply stopped between its "
            "two writes, or is running now (asi apply finishes it)",
            block,
        )

    def test_rows_are_ordered_by_created_at_not_by_id_or_scan_order(self):
        self.put("CAND-2026-01-01-01-tdd", "2026-01-05T12:00:03Z")
        self.put("CAND-2026-01-02-01-tdd", "2026-01-05T12:00:01Z")
        self.put("CAND-2026-01-03-01-tdd", "2026-01-05T12:00:02Z")
        self.put("CAND-2026-01-03-02-tdd", "2026-01-05T12:00:02Z")
        expected = [
            "CAND-2026-01-02-01-tdd",
            "CAND-2026-01-03-01-tdd",
            "CAND-2026-01-03-02-tdd",
            "CAND-2026-01-01-01-tdd",
        ]
        self.assertEqual([r["id"] for r in self.json()["candidates"]], expected)
        heads = [
            line.split("  ")[0] for line in self.text().splitlines() if line.startswith("CAND-")
        ]
        self.assertEqual(heads, expected)

    def test_a_record_whose_patch_text_does_not_hash_to_its_recorded_hash_is_untrusted(self):
        cid = "CAND-2026-01-03-01-tdd"
        record = record_for(self.base, cid, "2026-01-03T12:00:00Z")
        record["patch"]["new_text_sha256"] = self.live_hash
        write_record(self.state, record)
        row = self.row(self.json(), cid)
        problems = row["integrity"]["problems"]
        self.assertEqual(len(problems), 1, problems)
        self.assertEqual(row["integrity"]["problems_not_shown"], 0)
        self.assertRegex(
            problems[0],
            rf"^patch\.new_text_sha256 records .*{self.live_hash[:12]}.* but new_text hashes ",
        )
        self.assertEqual(row["live_match"], "not-compared")
        block = self.block(self.text(), cid)
        self.assertEqual(block[1], f"  DOES NOT RECOMPUTE (1 problem(s)): {problems[0]}")
        self.assertIn("  live match  not compared: the record does not recompute", block)

    def test_a_record_whose_patch_text_was_edited_fails_several_checks(self):
        cid = "CAND-2026-01-03-01-tdd"
        record = record_for(self.base, cid, "2026-01-03T12:00:00Z")
        record["patch"]["new_text"] += "\nedited by hand\n"
        write_record(self.state, record)
        row = self.row(self.json(), cid)
        problems = row["integrity"]["problems"]
        self.assertEqual(len(problems), 3, problems)
        self.assertTrue(
            any(p.startswith("patch.new_text_sha256 records") for p in problems), problems
        )
        self.assertTrue(any("approval_challenge" in p for p in problems), problems)
        self.assertTrue(any(p.startswith("validation.patch_sha256") for p in problems), problems)
        self.assertEqual(row["live_match"], "not-compared")
        block = self.block(self.text(), cid)
        self.assertTrue(block[1].startswith("  DOES NOT RECOMPUTE (3 problem(s)): "), block)
        self.assertIn("  live match  not compared: the record does not recompute", block)

    def test_a_live_file_that_is_not_utf8_is_not_compared_in_text_and_json(self):
        raw = self.live.read_bytes() + b"\xff\n"
        self.live.write_bytes(raw)
        replaced = raw.decode("utf-8", errors="replace")
        holds = record_for(self.base, "CAND-2026-01-03-01-tdd", "2026-01-03T12:00:00Z")
        holds["patch"]["new_text"] = replaced
        write_record(self.state, seal(holds))
        based = record_for(self.base, "CAND-2026-01-03-02-tdd", "2026-01-03T12:00:01Z")
        based["target"]["base_sha256"] = sha256_text(replaced)
        write_record(self.state, seal(based))
        watched = [self.base / "skills-canonical", self.base / "skills-mirror"]
        spy = _ReadSpy(*watched)

        def arm() -> None:
            spy.armed = True

        with spy, _after_groups(arm):
            payload = self.json()
        self.assertEqual(spy.hits, [])
        live = payload["live"]
        self.assertEqual(live["state"], "canonical")
        self.assertIsNone(live["sha256"])
        self.assertTrue(live["reason"].startswith("not valid UTF-8 at byte "), live)
        for row in payload["candidates"]:
            self.assertIs(row["integrity"]["recomputes"], True, row["integrity"])
            self.assertEqual(row["live_match"], "not-compared")
            self.assertEqual(
                row["live_match_reason"],
                f"the canonical copy is not valid UTF-8 ({live['reason']})",
            )
        spy = _ReadSpy(*watched)
        with spy, _after_groups(arm):
            out = self.text()
        self.assertEqual(spy.hits, [])
        lines = out.splitlines()
        self.assertEqual(
            lines[1],
            f"live: canonical {live['canonical_path']}, not valid UTF-8 ({live['reason']}), "
            f"{HISTORY_LIVE_BASIS}",
        )
        # Nothing was compared, so there is no summary line.
        self.assertFalse([line for line in lines if line.startswith("live match")], out)
        self.assertEqual(
            out.count(f"  live match  not compared: the canonical copy is not valid UTF-8 ("),
            2,
            out,
        )

    def test_a_base_hash_that_is_not_a_digest_is_shown_masked_and_bounded_and_not_compared(self):
        long = ("ab" * 25000) + TOKEN + ("cd" * 25000) + "ef"
        values = {
            "CAND-2026-01-03-01-tdd": long,
            "CAND-2026-01-03-02-tdd": self.live_hash[:63],
            "CAND-2026-01-03-03-tdd": self.live_hash.upper(),
        }
        self.assertEqual(len(long), 100_002 + len(TOKEN) - 38 + 38)
        for index, (cid, value) in enumerate(values.items()):
            record = record_for(self.base, cid, f"2026-01-03T12:00:0{index}Z")
            record["target"]["base_sha256"] = value
            write_record(self.state, seal(record))
        payload = self.json()
        raw = json.dumps(payload)
        self.assertNotIn(TOKEN, raw)
        out = self.text()
        self.assertNotIn(TOKEN, out)
        for cid, value in values.items():
            row = self.row(payload, cid)
            self.assertIsNone(row["base_sha256"])
            self.assertLessEqual(len(row["base_sha256_invalid"]), 240)
            self.assertEqual(row["live_match"], "not-compared")
            base_line = [line for line in self.block(out, cid) if line.startswith("  base       ")]
            self.assertEqual(
                base_line, [f'  base       (not a sha256 digest: "{row["base_sha256_invalid"]}")']
            )
        self.assertEqual(
            self.row(payload, "CAND-2026-01-03-03-tdd")["base_sha256_invalid"],
            values["CAND-2026-01-03-03-tdd"],
        )
        self.assertRegex(self.row(payload, "CAND-2026-01-03-01-tdd")["base_sha256_invalid"], CAPPED)

    def test_an_unreadable_file_name_is_masked_and_kept_on_one_line(self):
        write_torn(self.state, f"CAND-2026-01-01-01-{TOKEN}-tdd.json")
        write_torn(self.state, "CAND-2026-01-01-02-\x1b[2J\nFORGED-tdd.json")
        rc, raw, err = _run(self.config, "--json", "history", "tdd")
        self.assertEqual(rc, 0, err)
        self.assertNotIn(TOKEN, raw)
        self.assertNotIn("\x1b", raw)
        payload = json.loads(raw)
        self.assertEqual(len(payload["unreadable"]), 2)
        self.assertEqual(
            sorted(u["name"][:19] for u in payload["unreadable"]),
            ["CAND-2026-01-01-01-", "CAND-2026-01-01-02-"],
        )
        out = self.text()
        self.assertNotIn(TOKEN, out)
        self.assertNotIn("\x1b", out)
        self.assertFalse([line for line in out.splitlines() if line.startswith("FORGED")], out)
        self.assertEqual(
            len([line for line in out.splitlines() if line.startswith("UNREADABLE  ")]), 2
        )

    def test_malformed_nested_fields_are_named_invalid_and_the_row_is_kept(self):
        ids = [f"CAND-2026-01-03-0{n}-tdd" for n in range(1, 5)]
        first = record_for(self.base, ids[0], "2026-01-03T12:00:01Z")
        first["validation"]["gate"]["metrics"] = "many"
        write_record(self.state, first)
        write_receipt(self.state, "apply", ids[0], apply_receipt(ids[0], approval=[], write=[]))
        second = record_for(self.base, ids[1], "2026-01-03T12:00:02Z")
        second["validation"]["gate"]["reasons"] = "a string"
        second["validation"]["gate"]["passed"] = "yes"
        write_record(self.state, second)
        write_receipt(self.state, "rollback", ids[1], rollback_receipt(ids[1], manifest=5))
        third = record_for(self.base, ids[2], "2026-01-03T12:00:03Z")
        third["validation"]["gate"]["metrics"]["fixes"] = True
        write_record(self.state, third)
        write_receipt(
            self.state,
            "decline",
            ids[2],
            {
                "schema": "asi.decline_receipt.v1",
                "candidate_id": ids[2],
                "operator": {"x": 1},
                "declined_at": "2026-01-03T14:00:00Z",
                "reason": "r",
                "status_before": "validated",
            },
        )
        self.put(ids[3], "2026-01-03T12:00:04Z")
        os.mkfifo(self.state / "receipts" / f"apply--{ids[3]}.json")
        (self.state / "receipts" / f"rollback--{ids[3]}.json").mkdir()
        with _bounded():
            payload = self.json()
            out = self.text()
        rows = {r["id"]: r for r in payload["candidates"]}
        self.assertEqual(sorted(rows), ids)
        self.assertIn("gate.metrics", rows[ids[0]]["invalid_fields"])
        self.assertIn("apply.approval", rows[ids[0]]["invalid_fields"])
        self.assertIn("apply.write", rows[ids[0]]["invalid_fields"])
        self.assertEqual(rows[ids[0]]["apply"]["operator"], "alice")
        self.assertIsNone(rows[ids[0]]["apply"]["note"])
        self.assertIn("gate.reasons", rows[ids[1]]["invalid_fields"])
        self.assertIn("gate.passed", rows[ids[1]]["invalid_fields"])
        self.assertIn("rollback.manifest", rows[ids[1]]["invalid_fields"])
        self.assertIsNone(rows[ids[1]]["gate"]["passed"])
        self.assertIsNone(rows[ids[1]]["rollback"]["manifest"])
        self.assertIn("gate.metrics.fixes", rows[ids[2]]["invalid_fields"])
        self.assertIn("decline_receipt.operator", rows[ids[2]]["invalid_fields"])
        self.assertIsNone(rows[ids[2]]["gate"]["fixes"])
        self.assertEqual(set(rows[ids[3]]["apply"]), {"unreadable"})
        self.assertEqual(set(rows[ids[3]]["rollback"]), {"unreadable"})
        for cid in ids:
            invalid = [line for line in self.block(out, cid) if line.startswith("  invalid    ")]
            self.assertEqual(len(invalid), 0 if cid == ids[3] else 1, (cid, out))
        self.assertIn("  apply      receipt unreadable: ", "\n".join(self.block(out, ids[3])))
        self.assertIn("  rollback   receipt unreadable: ", "\n".join(self.block(out, ids[3])))

    def test_integrity_problems_are_capped_at_five_with_the_rest_counted(self):
        cid = "CAND-2026-01-03-01-tdd"
        record = record_for(self.base, cid, "2026-01-03T12:00:00Z")
        original = record["rubric_failures"]
        self.assertTrue(original)
        record["rubric_failures"] = [
            dict(ref, rubric_version=ref.get("rubric_version", 1) + 1)
            for _ in range(1000)
            for ref in original
        ]
        write_record(self.state, record)
        candidate = Candidate.from_dict(record)
        total = len(derived_field_mismatches(candidate)) + len(
            validation_mismatches(candidate, rubrics=None)
        )
        self.assertGreaterEqual(total, 1000)
        row = self.row(self.json(), cid)
        self.assertEqual(len(row["integrity"]["problems"]), 5)
        self.assertEqual(row["integrity"]["problems_not_shown"], total - 5)
        block = self.block(self.text(), cid)
        self.assertTrue(
            block[1].startswith(f"  DOES NOT RECOMPUTE ({total} problem(s)): "), block[1]
        )

    def test_gate_reasons_past_five_are_counted_not_shown(self):
        cid = "CAND-2026-01-03-01-tdd"
        record = record_for(self.base, cid, "2026-01-03T12:00:00Z")
        record["validation"]["gate"]["reasons"] = [f"reason {n}" for n in range(7)]
        write_record(self.state, record)
        gate = self.row(self.json(), cid)["gate"]
        self.assertEqual(gate["reasons"], [f"reason {n}" for n in range(5)])
        self.assertEqual(gate["reasons_not_shown"], 2)

    def test_an_apply_receipt_that_finished_an_earlier_run_says_so_in_text_and_json(self):
        cid = "CAND-2026-01-03-01-tdd"
        self.put(cid)
        for finishes in (True, False):
            with self.subTest(finishes_an_earlier_run=finishes):
                write_receipt(
                    self.state,
                    "apply",
                    cid,
                    apply_receipt(cid, finishes_an_earlier_run=finishes),
                )
                self.assertIs(
                    self.row(self.json(), cid)["apply"]["finishes_an_earlier_run"], finishes
                )
                applied = [
                    line
                    for line in self.block(self.text(), cid)
                    if line.startswith("  applied    ")
                ]
                self.assertEqual(len(applied), 1, applied)
                self.assertEqual(applied[0].endswith(" (finished an earlier run)"), finishes)

    def test_unreadable_and_set_aside_rows_are_capped_by_the_limit(self):
        self.put("CAND-2026-01-03-01-tdd")
        for n in range(60):
            write_torn(self.state, f"CAND-2026-02-01-{n:02d}-tdd.json")
        state = StateStore(self.base.resolve() / "state", clock=SystemClock())
        for n in range(30):
            state.quarantine("tdd", stage=f"propose:case{n:02d}", reason=f"q{n}")
            state.refuse("tdd", stage=f"validate:case{n:02d}", reason=f"r{n}")
        payload = self.json("--limit", "50")
        self.assertEqual(len(payload["unreadable"]), 50)
        self.assertEqual(payload["unreadable_not_shown"], 10)
        self.assertEqual(payload["counts"]["unreadable"], 60)
        self.assertEqual(len(payload["set_aside"]), 50)
        self.assertEqual(payload["set_aside_not_shown"], 10)
        self.assertEqual(payload["counts"]["set_aside"], 60)
        lines = self.text("--limit", "50").splitlines()
        self.assertEqual(len([line for line in lines if line.startswith("UNREADABLE  ")]), 50)
        self.assertEqual(len([line for line in lines if line.startswith("set aside   ")]), 50)
        self.assertIn("10 earlier unreadable record(s) named for this skill not shown", lines)
        self.assertIn("10 older set-aside record(s) not shown", lines)


class LiveCopyTests(_Fresh):
    def test_a_canonical_copy_discovery_passes_over_is_not_compared(self):
        self.put("CAND-2026-01-03-01-tdd")
        original = self.live.read_bytes()
        listed = self.live.resolve()

        def race(store: FilesystemSkillStore, *args: Any, **kwargs: Any) -> Any:
            found = ORIGINAL_ITER(store, *args, **kwargs)
            if listed in found and not self.live.is_fifo():
                self.live.unlink()
                os.mkfifo(self.live)
            return found

        for mode in ((), ("--json",)):
            with self.subTest(mode=mode or "text"):
                if self.live.is_fifo():
                    self.live.unlink()
                    self.live.write_bytes(original)
                with mock.patch.object(FilesystemSkillStore, "_iter_skill_files", race), _bounded():
                    rc, out, err = _run(self.config, *mode, "history", "tdd")
                self.assertEqual(rc, 0, err)
                if mode:
                    payload = json.loads(out)
                    self.assertEqual(_private_keys(payload), [])
                    self.assertEqual(payload["live"]["state"], "no-canonical")
                    self.assertIsNone(payload["live"]["sha256"])
                    self.assertTrue(payload["candidates"])
                    for row in payload["candidates"]:
                        self.assertEqual(row["live_match"], "not-compared")
                        self.assertTrue(
                            row["live_match_reason"].startswith(
                                "the skill has no canonical copy now ("
                            ),
                            row["live_match_reason"],
                        )
                else:
                    lines = out.splitlines()
                    self.assertTrue(lines[1].startswith("live: no canonical copy ("), lines[1])
                    self.assertFalse([line for line in lines if line.startswith("live match")], out)
                    self.assertIn(
                        "  live match  not compared: the skill has no canonical copy now (", out
                    )

    def test_an_ancestor_swapped_after_groups_is_never_opened_and_claims_nothing_false(self):
        cid = "CAND-2026-01-03-01-tdd"
        record = self.put(cid)
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text(record["patch"]["new_text"], encoding="utf-8")
        skill_dir = self.live.parent

        def swap() -> None:
            skill_dir.rename(skill_dir.with_name("tdd.real"))
            skill_dir.symlink_to(outside, target_is_directory=True)

        def restore() -> None:
            skill_dir.unlink()
            skill_dir.with_name("tdd.real").rename(skill_dir)

        for mode in ((), ("--json",)):
            with self.subTest(mode=mode or "text"):
                spy = _ReadSpy(outside, armed=True)
                with spy, _after_groups(swap) as seen, _bounded():
                    rc, out, err = _run(self.config, *mode, "history", "tdd")
                if seen["done"]:
                    restore()
                self.assertEqual((rc, seen["done"]), (0, 1), err)
                self.assertEqual(spy.hits, [])
                if mode:
                    payload = json.loads(out)
                    self.assertEqual(self.row(payload, cid)["live_match"], "base")
                    self.assertEqual(payload["live"]["sha256"], self.live_hash)
                else:
                    self.assertIn(
                        f"  live match  {HISTORY_LIVE_BASIS} held this candidate's base",
                        self.block(out, cid),
                    )

    def test_a_many_copy_ambiguous_group_reason_is_masked_and_capped_in_text_and_json(self):
        root = self.base / "skills-canonical"
        body = self.live.read_bytes()
        for name in [f"a-{TOKEN}", *(f"pkg{n:04d}" for n in range(999))]:
            (root / name / "tdd").mkdir(parents=True)
            (root / name / "tdd" / "SKILL.md").write_bytes(body)
        ids = [f"CAND-2026-01-03-0{n}-tdd" for n in range(1, 4)]
        for n, cid in enumerate(ids):
            self.put(cid, f"2026-01-03T12:00:0{n}Z")
        payload = self.json()
        raw = json.dumps(payload)
        self.assertNotIn(TOKEN, raw)
        live = payload["live"]
        self.assertEqual(live["state"], "no-canonical")
        self.assertLessEqual(len(live["reason"]), 240)
        self.assertRegex(live["reason"], CAPPED)
        self.assertEqual(len(payload["candidates"]), 3)
        for row in payload["candidates"]:
            self.assertEqual(row["live_match"], "not-compared")
            self.assertLessEqual(len(row["live_match_reason"]), 240)
            self.assertRegex(row["live_match_reason"], CAPPED)
        out = self.text()
        self.assertNotIn(TOKEN, out)
        lines = out.splitlines()
        self.assertTrue(lines[1].startswith("live: no canonical copy ("), lines[1])
        self.assertLess(len(lines[1]), 300)
        matches = [line for line in lines if line.startswith("  live match  ")]
        self.assertEqual(len(matches), 3)
        for line in matches:
            self.assertLess(len(line), 300)

    def test_a_live_file_changed_after_discovery_is_reported_as_discovery_read_it_with_the_qualifier(
        self,
    ):
        cid = "CAND-2026-01-03-01-tdd"
        record = self.put(cid)
        original = self.live.read_bytes()
        replacement = record["patch"]["new_text"]
        replacement_hash = sha256_text(replacement)
        self.assertNotEqual(replacement_hash, self.live_hash)

        def replace_after_discovery() -> None:
            sibling = self.live.with_name("SKILL.md.next")
            sibling.write_text(replacement, encoding="utf-8")
            os.replace(sibling, self.live)

        outputs = {}
        for mode in ("text", "json"):
            self.live.write_bytes(original)
            argv = ("--json", "history", "tdd") if mode == "json" else ("history", "tdd")
            with _after_groups(replace_after_discovery) as seen:
                rc, out, err = _run(self.config, *argv)
            self.assertEqual((rc, seen["done"]), (0, 1), err)
            self.assertEqual(sha256_text(self.live.read_text(encoding="utf-8")), replacement_hash)
            outputs[mode] = out
        self.live.write_bytes(original)
        payload = json.loads(outputs["json"])
        self.assertEqual(_private_keys(payload), [])
        row = self.row(payload, cid)
        self.assertEqual(row["live_match"], "base")
        self.assertEqual(row.get("live_match_basis"), HISTORY_LIVE_BASIS)
        self.assertEqual(payload["live"].get("basis"), HISTORY_LIVE_BASIS)
        self.assertEqual(payload["live"]["sha256"], self.live_hash)
        self.assertNotEqual(payload["live"]["sha256"], replacement_hash)
        lines = outputs["text"].splitlines()
        self.assertTrue(lines[1].startswith("live: canonical "), lines[1])
        self.assertTrue(lines[1].endswith(f", {HISTORY_LIVE_BASIS}"), lines[1])
        self.assertEqual(
            lines[2],
            f"live match: {HISTORY_LIVE_BASIS} matched no replacement among the 1 candidate(s) "
            "compared",
        )
        self.assertIn(
            f"  live match  {HISTORY_LIVE_BASIS} held this candidate's base",
            self.block(outputs["text"], cid),
        )
        for mode, out in outputs.items():
            for phrase in PRESENT_TENSE:
                self.assertNotIn(phrase, out, (mode, phrase))

    def test_a_summary_under_a_limit_claims_nothing_about_the_candidates_it_cut(self):
        older, newer = "CAND-2026-01-02-01-tdd", "CAND-2026-01-03-01-tdd"
        # The older record's replacement is the text discovery reads; its base is other text.
        record = record_for(self.base, older, "2026-01-02T12:00:00Z")
        record["target"]["base_sha256"] = sha256_text("another base\n")
        record["patch"]["new_text"] = self.live.read_text(encoding="utf-8")
        write_record(self.state, seal(record))
        self.put(newer, "2026-01-03T12:00:00Z")
        whole = self.json()
        self.assertEqual(
            [(r["id"], r["live_match"]) for r in whole["candidates"]],
            [(older, "replacement"), (newer, "base")],
        )
        self.assertEqual(
            self.text().splitlines()[2],
            f"live match: {HISTORY_LIVE_BASIS} held the replacement of {older}",
        )
        payload = self.json("--limit", "1")
        self.assertEqual(payload["earlier_not_shown"], 1)
        self.assertEqual(
            [(r["id"], r["live_match"]) for r in payload["candidates"]], [(newer, "base")]
        )
        lines = self.text("--limit", "1").splitlines()
        self.assertEqual(
            lines[2],
            f"live match: {HISTORY_LIVE_BASIS} matched no replacement among the 1 shown "
            "candidate(s) compared; 1 earlier candidate(s) not shown",
        )
        self.assertNotIn("no recorded replacement", "\n".join(lines))

    def test_a_candidate_whose_base_equals_its_replacement_counts_as_holding_its_replacement(
        self,
    ):
        cid = "CAND-2026-01-03-01-tdd"
        record = record_for(self.base, cid, "2026-01-03T12:00:00Z")
        record["patch"]["new_text"] = self.live.read_text(encoding="utf-8")
        write_record(self.state, seal(record))
        payload = self.json()
        row = self.row(payload, cid)
        self.assertIs(row["integrity"]["recomputes"], True)
        self.assertEqual(row["live_match"], "base")
        self.assertEqual(row["base_sha256"], self.live_hash)
        self.assertEqual(row["replacement_sha256"], self.live_hash)
        self.assertEqual(payload["earlier_not_shown"], 0)
        out = self.text()
        self.assertEqual(
            out.splitlines()[2], f"live match: {HISTORY_LIVE_BASIS} held the replacement of {cid}"
        )
        self.assertIn(
            f"  live match  {HISTORY_LIVE_BASIS} held this candidate's base and replacement "
            "(they are equal)",
            self.block(out, cid),
        )


if __name__ == "__main__":
    unittest.main()
