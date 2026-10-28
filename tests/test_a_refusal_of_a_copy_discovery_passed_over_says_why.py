"""A refusal of a copy discovery passed over says why discovery passed it over.

``assert_canonical_target`` (``adapters/skill_fs.py``) refuses a target that is
no skill document in any configured root, for ``review``, ``apply`` and
``rollback`` alike. Measured before (3.11 and 3.13, the same on both), with a
real validated candidate whose skill was moved under a directory ``x`` before
``daily`` proposed against it: with ``x/tdd`` then made search-only (mode
0o311, so the document reads but the directory does not list), or with a
``SKILL.md`` then written in ``x`` (so ``tdd`` is the material of skill ``x``),
each of the three commands said only "it is not a skill document in any
configured root", while ``doctor`` and ``skills`` named what discovery passed
over and why. The refusal now names the entry discovery passed over at the
target or above it and gives discovery's reason. The reason discovery gives for
a document below another skill's directory names that skill's document through
``named_path`` and its name through ``quoted_value``, so a line break in either
is spelled on its line and a credential is masked.
"""

from __future__ import annotations

import ast
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.cli import RC_REFUSED
from agent_skill_improver.domain.skills import SKILL_FILENAME, SkillRoot
from agent_skill_improver.errors import MirrorEditRefused
from tests.support import REPO_ROOT, build_env, run_cli

DOCUMENT = "---\nname: tdd\ndescription: test-driven development\n---\n# tdd\n"
UNKNOWN = (
    "so it is not a skill document in any configured root and this tool cannot tell whether "
    "it is canonical. An unrecognised target is refused rather than assumed safe."
)
MATERIAL = "a document under another skill's directory is that skill's material, not a skill"
SKILL_FS = REPO_ROOT / "src" / "agent_skill_improver" / "adapters" / "skill_fs.py"
#: What a reason discovery gives may interpolate as it is, none of it a path: an
#: OS error's text or class name, a length, the configured depth, the
#: document's fixed name, the kind of entry found, and what sits at a root's
#: path (``_root_shape_problem``, whose link text is named where it is read,
#: held below).
NOT_PATHS = frozenset(
    {
        "error",
        "len(os.fsencode(str(candidate)))",
        "exc.strerror",
        "exc.strerror or exc",
        "self._max_depth",
        "SKILL_FILENAME",
        "_entry_kind(candidate)",
        "problem",
    }
)


def skip_as_root(case: unittest.TestCase) -> None:
    if os.geteuid() == 0:
        case.skipTest("root lists a search-only directory")


class TheStoreSaysWhyDiscoveryPassedTheTargetOverTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.root = self.base / "root"

    def plant(self, owner: str = "x") -> Path:
        document = self.root / owner / "tdd" / SKILL_FILENAME
        document.parent.mkdir(parents=True)
        document.write_text(DOCUMENT, encoding="utf-8")
        return document

    def refusal(self, target: Path, root: Path | None = None) -> str:
        store = FilesystemSkillStore(
            [SkillRoot(id="r", path=root or self.root, canonical=True)], read_only=False
        )
        with self.assertRaises(MirrorEditRefused) as ctx:
            store.assert_canonical_target(target, action="write")
        return str(ctx.exception)

    def test_a_copy_in_a_directory_that_cannot_be_listed_is_refused_with_that_reason(
        self,
    ) -> None:
        skip_as_root(self)
        document = self.plant()
        os.chmod(document.parent, 0o311)
        self.addCleanup(os.chmod, document.parent, 0o755)
        self.assertEqual(
            self.refusal(document),
            f"refusing to write {document}: discovery passed over {document.parent}, which is a "
            f"directory that could not be listed (Permission denied), {UNKNOWN}",
        )

    def test_a_copy_below_another_skill_is_refused_as_that_skill_s_material(self) -> None:
        document = self.plant()
        owner = self.root / "x" / SKILL_FILENAME
        owner.write_text(DOCUMENT.replace("tdd", "x"), encoding="utf-8")
        self.assertEqual(
            self.refusal(document),
            f"refusing to write {document}: discovery passed over {document}, which is below "
            f"{owner}, the place of skill 'x'; {MATERIAL}, {UNKNOWN}",
        )

    def test_a_link_at_the_document_s_place_is_matched_as_spelled(self) -> None:
        # Discovery records a link as the link, so the target is matched as
        # spelled as well as resolved.
        outside = self.base / "outside.md"
        outside.write_text(DOCUMENT, encoding="utf-8")
        link = self.root / "tdd" / SKILL_FILENAME
        link.parent.mkdir(parents=True)
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):  # pragma: no cover - permissions
            self.skipTest("cannot create a symlink here")
        self.assertIn(
            f": discovery passed over {link}, which is a symlink that resolves outside its root, ",
            self.refusal(link),
        )

    def test_a_target_spelled_through_a_linked_root_is_matched_as_resolved(self) -> None:
        # Discovery records what it passes over below the root's resolved
        # path; a target may come spelled through the root's link, as the
        # root is configured.
        document = self.plant()
        owner = self.root / "x" / SKILL_FILENAME
        owner.write_text(DOCUMENT.replace("tdd", "x"), encoding="utf-8")
        linked = self.base / "linked-root"
        os.symlink(self.root, linked)
        store = FilesystemSkillStore(
            [SkillRoot(id="r", path=linked, canonical=True)], read_only=False
        )
        spelled = linked / "x" / "tdd" / SKILL_FILENAME
        with self.assertRaises(MirrorEditRefused) as ctx:
            store.assert_canonical_target(spelled, action="write")
        self.assertEqual(
            str(ctx.exception),
            f"refusing to write {spelled}: discovery passed over {document}, which is below "
            f"{owner}, the place of skill 'x'; {MATERIAL}, {UNKNOWN}",
        )

    def test_a_line_break_in_the_owning_skill_s_directory_is_spelled_on_its_line(self) -> None:
        name = "x\nconfig error: forged"
        document = self.plant(name)
        (self.root / name / SKILL_FILENAME).write_text(DOCUMENT, encoding="utf-8")
        message = self.refusal(document)
        self.assertNotIn("\n", message)
        self.assertIn("the place of skill 'x\\nconfig error: forged'; ", message)
        store = FilesystemSkillStore([SkillRoot(id="r", path=self.root, canonical=True)])
        store.snapshots(refresh=True)
        [entry] = store.passed_over()
        self.assertNotIn("\n", entry.reason, "as doctor and skills list it, and their JSON")

    def test_a_credential_in_the_owning_skill_s_name_is_masked(self) -> None:
        token = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
        name = "x-" + token
        document = self.plant(name)
        (self.root / name / SKILL_FILENAME).write_text(DOCUMENT, encoding="utf-8")
        message = self.refusal(document)
        reason = message.split(", which is ", 1)[1]
        self.assertNotIn(token, reason)
        self.assertIn("[REDACTED:", reason)

    def test_a_line_break_in_a_dangling_root_link_is_spelled_on_its_line(self) -> None:
        self.root.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(self.base / "nowhere\nconfig error: forged", self.root)
        message = self.refusal(self.root / "tdd" / SKILL_FILENAME)
        self.assertNotIn("\n", message)
        self.assertIn(
            f"discovery passed over {self.root}, which is the configured root itself and is a "
            f"dangling symbolic link to '{self.base}/nowhere\\nconfig error: forged', not a "
            "directory, ",
            message,
        )

    def test_the_control_a_target_nothing_was_passed_over_for_is_refused_as_before(
        self,
    ) -> None:
        self.plant()
        elsewhere = self.base / "elsewhere" / "tdd" / SKILL_FILENAME
        self.assertEqual(
            self.refusal(elsewhere),
            f"refusing to write {elsewhere}: it is not a skill document in any configured root, "
            "so this tool cannot tell whether it is canonical. An unrecognised target is refused "
            "rather than assumed safe.",
        )

    def test_the_control_an_entry_beside_the_target_is_not_named(self) -> None:
        # An entry applies at the target or above it, judged by path
        # components: a sibling whose name the target's begins with is not
        # on the way to it.
        def old(target: Path) -> str:
            return (
                f"refusing to write {target}: it is not a skill document in any configured "
                "root, so this tool cannot tell whether it is canonical. An unrecognised "
                "target is refused rather than assumed safe."
            )

        with self.subTest(entry="a directory that could not be listed"):
            skip_as_root(self)
            root = self.base / "unlistable"
            (root / "tdd").mkdir(parents=True)
            os.chmod(root / "tdd", 0o311)
            self.addCleanup(os.chmod, root / "tdd", 0o755)
            for target in (root / "tdd2" / SKILL_FILENAME, root / "tdd-notes.md"):
                self.assertEqual(self.refusal(target, root), old(target))
        with self.subTest(entry="a dangling link at a document's place"):
            root = self.base / "dangling"
            (root / "tdd").mkdir(parents=True)
            os.symlink(self.base / "nowhere", root / "tdd" / SKILL_FILENAME)
            target = root / "tdd" / (SKILL_FILENAME + ".orig")
            self.assertEqual(self.refusal(target, root), old(target))

    def test_the_control_a_plain_owner_s_reason_reads_as_before(self) -> None:
        self.plant()
        owner = self.root / "x" / SKILL_FILENAME
        owner.write_text(DOCUMENT, encoding="utf-8")
        store = FilesystemSkillStore([SkillRoot(id="r", path=self.root, canonical=True)])
        store.snapshots(refresh=True)
        [entry] = store.passed_over()
        self.assertEqual(entry.reason, f"below {owner}, the place of skill 'x'; {MATERIAL}")

    def test_every_reason_discovery_gives_names_its_paths(self) -> None:
        """Each f-string reason a passed-over entry is built with, read from the source."""
        tree = ast.parse(SKILL_FS.read_text(encoding="utf-8"))
        reasons: list[ast.JoinedStr] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            if name == "skip" and len(node.args) >= 2:
                reasons.append(node.args[1])
            elif name == "PassedOverDocument":
                given = [k.value for k in node.keywords if k.arg == "reason"] + node.args[2:3]
                reasons.extend(given)
        found = [r for r in reasons if isinstance(r, ast.JoinedStr)]
        self.assertGreaterEqual(len(found), 6)
        raw = []
        for reason in found:
            for value in reason.values:
                if not isinstance(value, ast.FormattedValue):
                    continue
                text = ast.unparse(value.value)
                call = value.value
                named = isinstance(call, ast.Call) and ast.unparse(call.func) in {
                    "named_path",
                    "quoted_value",
                }
                if not named and (text not in NOT_PATHS or value.conversion != -1):
                    raw.append(f"line {value.lineno}: {text}")
        self.assertEqual(raw, [])

    def test_a_root_link_s_text_is_named_where_it_is_read(self) -> None:
        tree = ast.parse(SKILL_FS.read_text(encoding="utf-8"))
        [function] = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_root_shape_problem"
        ]
        reads = [
            n
            for n in ast.walk(function)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "os.readlink"
        ]
        parents = {c: n for n in ast.walk(function) for c in ast.iter_child_nodes(n)}
        self.assertEqual(len(reads), 1)
        self.assertEqual(ast.unparse(parents[reads[0]].func), "named_path")


class EachCommandSaysWhyTests(unittest.TestCase):
    """A real candidate against a copy at ``x/tdd``, refused by each command."""

    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.cfg = ["--config", str(build_env(base, dry_run=False, seed="c172-why"))]
        canon = base / "skills-canonical"
        (canon / "x").mkdir()
        (canon / "tdd").rename(canon / "x" / "tdd")
        self.owner = canon / "x" / SKILL_FILENAME
        self.nested = canon / "x" / "tdd"
        proc = run_cli([*self.cfg, "--json", "daily", "--commit", "--day", "2026-02-01"])
        self.cid = next(
            c["candidate_id"]
            for c in json.loads(proc.stdout)["candidates"]
            if c.get("status") == "validated" and c["candidate_id"].endswith("-tdd")
        )
        review = run_cli([*self.cfg, "--json", "review", self.cid])
        self.assertEqual(review.returncode, 0, review.stderr)
        self.token = json.loads(review.stdout)["approval_token"]

    def commands(self) -> dict[str, list[str]]:
        apply = ["apply", self.cid, "--approve", self.token, "--i-reviewed-this"]
        return {"review": ["review", self.cid], "write": apply, "roll back": ["rollback", self.cid]}

    def each(self, change, undo, expected: str) -> None:
        for action, args in self.commands().items():
            with self.subTest(action=action):
                if action == "roll back":
                    applied = run_cli([*self.cfg, *self.commands()["write"]])
                    self.assertEqual(applied.returncode, 0, applied.stderr)
                change()
                try:
                    proc = run_cli([*self.cfg, *args])
                finally:
                    undo()
                self.assertEqual(proc.returncode, RC_REFUSED, proc.stderr)
                self.assertEqual(proc.stderr.count("\n"), 1, proc.stderr)
                self.assertTrue(
                    proc.stderr.startswith(
                        f"MirrorEditRefused: refusing to {action} {self.nested / SKILL_FILENAME}: "
                        f"discovery passed over {expected}"
                    ),
                    proc.stderr,
                )

    def test_each_command_says_the_directory_could_not_be_listed(self) -> None:
        skip_as_root(self)
        self.each(
            lambda: os.chmod(self.nested, 0o311),
            lambda: os.chmod(self.nested, 0o755),
            f"{self.nested}, which is a directory that could not be listed (Permission denied), ",
        )

    def test_each_command_says_whose_material_the_copy_is(self) -> None:
        self.each(
            lambda: self.owner.write_text(DOCUMENT, encoding="utf-8"),
            self.owner.unlink,
            f"{self.nested / SKILL_FILENAME}, which is below {self.owner}, the place of skill 'x'; ",
        )


if __name__ == "__main__":
    unittest.main()
