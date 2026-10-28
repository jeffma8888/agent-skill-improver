"""A refusal from the skill store names each path it holds on its line.

The skill store's refusals (``adapters/skill_fs.py``) name the document, the
backup and the manifest they judged. Measured before (3.11 and 3.13, the same on
both), with a real validated candidate whose skill directory was then renamed
``tdd``, a line break and ``config error: forged``, the record edited to match:
``review`` of the document replaced by a symbolic link, by bytes that are not
UTF-8 or by a fifo, ``apply`` of the linked document, and ``rollback`` of an
applied candidate whose skill was then edited or replaced by a link, or whose
backup was then removed or changed, each printed the path as written, so the
text after the break stood on stderr as a line of its own, ``config error:
forged``; ``apply`` of a changed or missing document already named it quoted.
``review`` of the skill with a second copy in the same root printed four lines,
three of them forged: the reason the inventory gives for withholding the
canonical slot (built in ``domain/skills.py``) listed each copy as written.
Each refusal now names its paths through ``named_path``, and so does each list
of copies in that reason: bare, as before, when a path reads as one, and else
quoted, masked and cut, on its line.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.cli import RC_REFUSED
from agent_skill_improver.domain.candidate import make_patch
from agent_skill_improver.domain.skills import SkillRoot, SkillSnapshot, resolve_groups
from tests.support import REPO_ROOT, build_env, run_cli

FORGED = "config error: forged"
LINE_BREAK_NAME = f"tdd\n{FORGED}"
#: How ``named_path`` spells that name: ``repr``, so the break is ``\\n``.
SPELLED = "tdd\\nconfig error: forged"
SKILL_FS = REPO_ROOT / "src" / "agent_skill_improver" / "adapters" / "skill_fs.py"
SKILLS = REPO_ROOT / "src" / "agent_skill_improver" / "domain" / "skills.py"

#: The calls that name a value a refusal may not print as written.
NAMERS = frozenset({"named_path", "quoted_value"})
#: What a refusal may interpolate as it is, none of it a path: the verb or the
#: action the caller passed (a literal at every call), the document's fixed
#: name, a configured root's id (the config holds it to ``_ROOT_ID``), a decode
#: error's position and reason, an OS error's text, what ``_read_back`` found
#: (a short hash or a fixed phrase), the reason a rollback check raised (an OS
#: error's text or a ``ValueError``'s own), the inventory's canonical reason
#: (root ids, quoted, and each copy it lists named where ``domain/skills.py``
#: builds it, held below), the reason discovery gave for an entry it passed
#: over (each path in it named where it is built, which
#: ``test_a_refusal_of_a_copy_discovery_passed_over_says_why`` holds), the
#: manifest fields found missing (a fixed list), and ``read_json``'s own error
#: (a decode error names no content; an OS error quotes its file name with
#: ``repr``).
NOT_PATHS = frozenset(
    {
        "verb",
        "action",
        "SKILL_FILENAME",
        "root.id",
        "exc.start",
        "exc.reason",
        "exc.strerror",
        "exc.strerror or exc",
        "found",
        "reason",
        "group.canonical_reason",
        "passed.reason",
        "', '.join(missing)",
        "exc",
    }
)
#: The paths the refusals name, each through ``named_path``: bare when it reads
#: as a path, as before, and ``quoted_value`` would quote even a plain one.
PATH_NAMES = (
    "target",
    "path",
    "backup",
    "backup_path",
    "manifest_path",
    "spelled",
    "cursor",
    "resolved",
    "walked",
    "guard",
    "canonical",
    "passed.path",
)
#: A short hash, as each hash refusal shows one.
SHORT_HASH = re.compile(r"[a-z_0-9]+\[:12\]")
#: How many paths the refusals name through ``named_path`` at this change.
NAMED_AT_THIS_CHANGE = 69


def _interpolations() -> list[tuple[int, str, ast.FormattedValue]]:
    """Every value an f-string inside a ``raise`` in the skill store interpolates."""
    found = []
    for node in ast.walk(ast.parse(SKILL_FS.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Raise) and node.exc is not None:
            for sub in ast.walk(node.exc):
                if isinstance(sub, ast.FormattedValue):
                    found.append((sub.lineno, ast.unparse(sub.value), sub))
    return found


def _named_by(sub: ast.FormattedValue) -> str | None:
    value = sub.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        if value.func.id in NAMERS:
            return value.func.id
    return None


def _argument(sub: ast.FormattedValue) -> str:
    call = sub.value
    assert isinstance(call, ast.Call) and len(call.args) == 1, ast.unparse(call)
    return ast.unparse(call.args[0])


class EveryPathARefusalNamesIsNamedTests(unittest.TestCase):
    """The rule held over the whole module, so a new refusal cannot slip past it."""

    def test_every_value_a_skill_store_refusal_interpolates_is_named_or_no_path(self) -> None:
        raw = []
        for line, text, sub in _interpolations():
            namer = _named_by(sub)
            if namer:
                if sub.conversion != -1 or sub.format_spec is not None:
                    raw.append(f"line {line}: {text} is formatted after it is named")
                elif namer == "quoted_value" and _argument(sub) in PATH_NAMES:
                    raw.append(f"line {line}: {text} quotes a path a plain one names bare")
                continue
            if sub.conversion != -1:
                raw.append(f"line {line}: {text} with a conversion")
            elif text not in NOT_PATHS and not SHORT_HASH.fullmatch(text):
                raw.append(f"line {line}: {text}")
        self.assertEqual(raw, [])

    def test_each_path_a_refusal_names_goes_through_named_path(self) -> None:
        named = [text for _line, text, sub in _interpolations() if _named_by(sub) == "named_path"]
        self.assertGreaterEqual(len(named), NAMED_AT_THIS_CHANGE)
        for path in PATH_NAMES:
            with self.subTest(path=path):
                self.assertIn(f"named_path({path})", named)
                self.assertNotIn(path, NOT_PATHS)


class TheInventoryNamesEachCopyItListsTests(unittest.TestCase):
    """The reason a skill has no canonical copy is built in ``domain/skills.py``
    and a refusal prints it whole, so each copy it lists is named there."""

    PLAIN = "tdd"

    @staticmethod
    def snap(root_id: str, path: str) -> SkillSnapshot:
        return SkillSnapshot(
            skill_id="tdd", root_id=root_id, path=Path(path), sha256="0" * 64, size_bytes=1
        )

    @staticmethod
    def root(root_id: str, *, precedence: int = 10) -> SkillRoot:
        return SkillRoot(
            id=root_id, path=Path(f"/{root_id}"), precedence=precedence, canonical=True
        )

    def reasons(self, name: str) -> dict[str, str]:
        """The reason for each place a list of copies is built, ``name`` the skill directory."""
        own, nested = f"/one/{name}/SKILL.md", f"/one/x/{name}/SKILL.md"
        tie = [
            self.snap("one", own),
            self.snap("one", nested),
            self.snap("two", f"/two/{name}/SKILL.md"),
        ]
        cases = {
            "same root": resolve_groups(
                [self.snap("one", own), self.snap("one", nested)], [self.root("one")]
            ),
            "tied root": resolve_groups(tie, [self.root("one"), self.root("two")]),
            "withheld": resolve_groups(
                [self.snap("one", nested)],
                [self.root("one")],
                passed_over=[("one", "tdd", "a document this process cannot read (x)")],
            ),
        }
        out = {}
        for case, groups in cases.items():
            [group] = groups
            self.assertIsNone(group.canonical, case)
            out[case] = group.canonical_reason
        return out

    def test_a_line_break_in_a_listed_copy_is_spelled_on_its_line(self) -> None:
        for case, reason in self.reasons(LINE_BREAK_NAME).items():
            with self.subTest(case=case):
                self.assertNotIn("\n", reason)
                self.assertIn(f"'/one/x/{SPELLED}/SKILL.md'", reason)

    def test_a_credential_in_a_listed_copy_is_masked(self) -> None:
        token = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
        for case, reason in self.reasons("tdd-" + token).items():
            with self.subTest(case=case):
                self.assertNotIn(token, reason)
                self.assertIn("[REDACTED:", reason)

    def test_the_control_plain_copies_are_listed_bare_as_before(self) -> None:
        reasons = self.reasons(self.PLAIN)
        self.assertIn("(/one/tdd/SKILL.md, /one/x/tdd/SKILL.md)", reasons["same root"])
        self.assertIn("(/one/tdd/SKILL.md, /one/x/tdd/SKILL.md)", reasons["tied root"])
        self.assertIn("(/one/x/tdd/SKILL.md)", reasons["withheld"])

    def test_every_copy_path_the_inventory_lists_goes_through_named_path(self) -> None:
        tree = ast.parse(SKILLS.read_text(encoding="utf-8"))
        [function] = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "resolve_groups"
        ]
        parents = {
            child: node for node in ast.walk(function) for child in ast.iter_child_nodes(node)
        }
        paths = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Attribute) and node.attr == "path"
        ]
        self.assertGreaterEqual(len(paths), 3)
        for node in paths:
            parent = parents[node]
            with self.subTest(line=node.lineno):
                self.assertIsInstance(parent, ast.Call)
                self.assertEqual(ast.unparse(parent.func), "named_path")


class _Named(unittest.TestCase):
    """One validated candidate for the class, its skill renamed as each test asks."""

    applied = False

    @classmethod
    def setUpClass(cls) -> None:
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.base = Path(tmp.name).resolve()
        cls.cfg = ["--config", str(build_env(cls.base, dry_run=False, seed="c169-named"))]
        proc = run_cli([*cls.cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
        cls.cid = next(
            c["candidate_id"]
            for c in json.loads(proc.stdout)["candidates"]
            if c.get("status") == "validated"
        )
        cls.record = cls.base / "state" / "candidates" / f"{cls.cid}.json"
        cls.root = cls.base / "skills-canonical"
        cls.skill_dir = cls.root / "tdd"
        if cls.applied:
            target = cls.rename(LINE_BREAK_NAME)
            token = json.loads(run_cli([*cls.cfg, "--json", "review", cls.cid]).stdout)
            proc = run_cli(
                [
                    *cls.cfg,
                    "apply",
                    cls.cid,
                    "--approve",
                    token["approval_token"],
                    "--i-reviewed-this",
                ]
            )
            assert proc.returncode == 0, proc.stderr
            cls.backups = sorted((cls.base / "state").rglob("*.bak"))
            assert len(cls.backups) == 1, cls.backups
            cls.applied_bytes = target.read_bytes()

    @classmethod
    def rename(cls, name: str) -> Path:
        """The skill directory renamed ``name``, the record edited to match."""
        moved = cls.root / name
        if cls.skill_dir != moved:
            cls.skill_dir.rename(moved)
            cls.skill_dir = moved
        target = moved / "SKILL.md"
        record = json.loads(cls.record.read_text(encoding="utf-8"))
        record["target"]["skill_id"] = name
        record["target"]["canonical_path"] = str(target)
        record["patch"]["unified_diff"] = make_patch(
            target.read_text(encoding="utf-8"), record["patch"]["new_text"], path_label=str(target)
        ).unified_diff
        cls.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    def cli(self, *args: str):
        return run_cli([*self.cfg, *args])

    def token(self) -> str:
        proc = self.cli("--json", "review", self.cid)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)["approval_token"]

    def linked(self, target: Path) -> None:
        """The document replaced by a symbolic link to a copy of it, put back after."""
        real = target.parent / "real.md"
        target.rename(real)
        os.symlink(real, target)

        def put_back() -> None:
            target.unlink()
            real.rename(target)

        self.addCleanup(put_back)

    def replaced(self, target: Path, data: bytes | None) -> None:
        """The document's bytes replaced (``None``: by a fifo), put back after."""
        kept = target.read_bytes()
        if data is None:
            target.unlink()
            os.mkfifo(target)
        else:
            target.write_bytes(data)

        def put_back() -> None:
            target.unlink()
            target.write_bytes(kept)

        self.addCleanup(put_back)

    def assert_refused_on_one_line(self, proc, kind: str) -> None:
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
        self.assertTrue(proc.stderr.startswith(f"{kind}: "), proc.stderr)
        self.assertEqual(proc.stderr.count("\n"), 1, proc.stderr)
        self.assertTrue(proc.stderr.endswith("\n"), proc.stderr)
        self.assertNotIn("\n" + FORGED, proc.stderr)
        self.assertIn(SPELLED, proc.stderr)


class AReviewOrApplyRefusalNamesTheDocumentOnItsLineTests(_Named):
    def test_review_of_a_linked_document_names_it_on_its_line(self) -> None:
        target = self.rename(LINE_BREAK_NAME)
        self.linked(target)
        self.assert_refused_on_one_line(self.cli("review", self.cid), "LinkedTargetRefused")

    def test_apply_of_a_linked_document_names_it_on_its_line(self) -> None:
        target = self.rename(LINE_BREAK_NAME)
        token = self.token()
        before = target.read_bytes()
        self.linked(target)
        proc = self.cli("apply", self.cid, "--approve", token, "--i-reviewed-this")
        self.assert_refused_on_one_line(proc, "LinkedTargetRefused")
        self.assertEqual((target.parent / "real.md").read_bytes(), before)

    def test_review_of_an_undecodable_document_names_it_on_its_line(self) -> None:
        target = self.rename(LINE_BREAK_NAME)
        self.replaced(target, b"---\nname: tdd\n---\n\xff\xfe bad\n")
        self.assert_refused_on_one_line(self.cli("review", self.cid), "UndecodableDocument")

    def test_review_of_a_fifo_where_the_document_was_names_it_on_its_line(self) -> None:
        target = self.rename(LINE_BREAK_NAME)
        self.replaced(target, None)
        self.assert_refused_on_one_line(self.cli("review", self.cid), "BaseHashMismatch")

    def test_review_of_a_skill_with_a_second_copy_names_each_on_its_line(self) -> None:
        target = self.rename(LINE_BREAK_NAME)
        other = self.root / "x"
        shutil.copytree(target.parent, other / LINE_BREAK_NAME)
        self.addCleanup(shutil.rmtree, other)
        proc = self.cli("review", self.cid)
        self.assert_refused_on_one_line(proc, "MirrorEditRefused")
        self.assertIn("holds 2 copies of this skill", proc.stderr)
        self.assertEqual(proc.stderr.count(SPELLED), 4, "the document, the skill, each copy")

    def test_a_credential_in_the_skill_path_is_masked_in_the_refusal(self) -> None:
        token = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
        target = self.rename("tdd-" + token)
        self.linked(target)
        proc = self.cli("review", self.cid)
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
        self.assertNotIn(token, proc.stderr)
        self.assertIn("[REDACTED:", proc.stderr)
        self.assertEqual(proc.stderr.count("\n"), 1, proc.stderr)

    def test_the_control_a_plain_path_is_named_bare_as_before(self) -> None:
        target = self.rename("tdd")
        self.linked(target)
        proc = self.cli("review", self.cid)
        self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
        self.assertEqual(
            proc.stderr,
            f"LinkedTargetRefused: refusing to read {target}: the document is a symbolic link, "
            "which the write path would refuse to replace, so nothing may be shown or approved "
            "against it.\n",
        )


class ARollbackRefusalNamesTheSkillOnItsLineTests(_Named):
    """One applied candidate, its skill named with a line break, each case put back."""

    applied = True

    def setUp(self) -> None:
        self.target = self.skill_dir / "SKILL.md"
        self.assertEqual(self.target.read_bytes(), self.applied_bytes)

    def test_rollback_of_a_skill_edited_since_its_apply_names_it_on_its_line(self) -> None:
        self.replaced(self.target, self.applied_bytes + b"\nedited later\n")
        self.assert_refused_on_one_line(self.cli("rollback", self.cid), "BaseHashMismatch")

    def test_rollback_of_a_skill_replaced_by_a_link_names_it_on_its_line(self) -> None:
        self.linked(self.target)
        self.assert_refused_on_one_line(self.cli("rollback", self.cid), "BackupUnusable")

    def test_rollback_with_its_backup_removed_names_the_skill_on_its_line(self) -> None:
        backup = self.backups[0]
        kept = backup.read_bytes()
        backup.unlink()
        self.addCleanup(backup.write_bytes, kept)
        self.assert_refused_on_one_line(self.cli("rollback", self.cid), "BackupUnusable")
        self.assertEqual(self.target.read_bytes(), self.applied_bytes)

    def test_rollback_with_its_backup_changed_names_the_skill_on_its_line(self) -> None:
        backup = self.backups[0]
        kept = backup.read_bytes()
        backup.write_text("other\n", encoding="utf-8")
        self.addCleanup(backup.write_bytes, kept)
        self.assert_refused_on_one_line(self.cli("rollback", self.cid), "BackupUnusable")
        self.assertEqual(self.target.read_bytes(), self.applied_bytes)


if __name__ == "__main__":
    unittest.main()
