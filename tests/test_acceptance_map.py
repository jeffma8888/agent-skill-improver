"""The acceptance map must stay true, or it is worse than having no map at all.

`docs/ACCEPTANCE.md` is a table of claims: *this finding is answered here, and
this named test proves it*. A reader cannot check 293 tests by hand, so the
document is what they read instead -- which is exactly why a renamed or deleted
test would leave a confident document pointing at nothing. That failure is
silent: nothing errors, and the map keeps reading as evidence.

So the map is verified mechanically here, and the checks are two-sided. A
checker that reports "all references resolve" is indistinguishable, from its
output alone, from a checker whose extractor found no references at all --
which is the whole failure the surrounding project exists to catch. Every
extractor below therefore has a positive control (it finds what is there), a
negative control (it does not collect prose or module paths), and a mutation
control (a name that does not exist is reported missing).

The document is treated as data, never as authority: where the document and the
suite disagree, this test fails and a human decides which side was wrong.
"""

from __future__ import annotations

import re
import unittest
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "ACCEPTANCE.md"
SRC = REPO / "src" / "agent_skill_improver"
TESTS = REPO / "tests"

# The floor exists so this module cannot pass vacuously. If the map is ever
# genuinely trimmed below it, that is a deliberate decision and this number is
# the thing a human has to change on purpose.
MINIMUM_REFERENCED_TESTS = 100

_BACKTICKED = re.compile(r"`([^`\n]+)`")
_TEST_DEF = re.compile(r"^\s*def (test_\w+)\s*\(", re.MULTILINE)


def backticked(text: str) -> list[str]:
    """Every inline-code span in a Markdown document, in document order."""
    return _BACKTICKED.findall(text)


def referenced_test_names(text: str) -> set[str]:
    """Test *method* names cited by a document.

    A citation is an inline-code span that names a Python identifier beginning
    with ``test_``. Module paths such as ``tests/test_acceptance_map.py`` are
    deliberately excluded: they name a file, not a test, and they are checked by
    the path resolver instead.
    """
    names: set[str] = set()
    for span in backticked(text):
        if not span.startswith("test_"):
            continue
        if "/" in span or span.endswith(".py"):
            continue
        if not span.isidentifier():
            continue
        names.add(span)
    return names


def declared_test_names() -> dict[str, list[str]]:
    """Map every test method declared in the suite to the modules declaring it."""
    declared: dict[str, list[str]] = defaultdict(list)
    for module in sorted(TESTS.glob("test_*.py")):
        for name in _TEST_DEF.findall(module.read_text(encoding="utf-8")):
            declared[name].append(module.name)
    return dict(declared)


def missing_references(referenced: set[str], declared: dict[str, list[str]]) -> list[str]:
    """References with no matching test method, sorted for a stable message."""
    return sorted(referenced - set(declared))


def resolve_reference(ref: str) -> list[Path]:
    """Resolve a backticked artifact reference to the real files it names.

    Three shapes occur in the map and each is resolved on its own terms rather
    than by one join that would silently fail on two of them:

    * a glob (``rubrics/*.toml``) must match at least one file;
    * a directory (``packaging/``) must exist as a directory;
    * a file is tried against the repository root first, then the package
      directory, because the map cites source modules package-relative.

    A bare filename that exists nowhere at those two roots is searched for in
    the tree, because ``SKILL.md`` names a shape of file rather than one path.
    """
    if "*" in ref:
        return sorted(REPO.glob(ref))
    if ref.endswith("/"):
        candidate = REPO / ref
        return [candidate] if candidate.is_dir() else []
    for root in (REPO, SRC):
        candidate = root / ref
        if candidate.exists():
            return [candidate]
    if "/" not in ref:
        return sorted(REPO.glob(f"examples/**/{ref}"))
    return []


def artifact_references(text: str) -> list[str]:
    """Backticked spans that name a repository artifact rather than a symbol.

    A path on the host (one that begins with ``/``, such as ``/var``) names no
    file in the repository, and whether it exists is a fact about the machine
    running the suite: measured before, the map's ``/private/var/root`` resolved
    on macOS and was reported missing on Linux. So it is not taken for one.
    """
    refs: list[str] = []
    for span in backticked(text):
        if span in refs or span.startswith("/"):
            continue
        looks_like_path = "/" in span or span.endswith((".py", ".md", ".toml", ".sh"))
        if looks_like_path:
            refs.append(span)
    return refs


class AcceptanceMapNamesRealTests(unittest.TestCase):
    """Every test the map claims as evidence must exist under that exact name."""

    def setUp(self) -> None:
        self.text = DOC.read_text(encoding="utf-8")
        self.referenced = referenced_test_names(self.text)
        self.declared = declared_test_names()

    def test_the_acceptance_map_exists_and_is_not_empty(self) -> None:
        self.assertTrue(DOC.is_file(), f"{DOC} is missing")
        self.assertGreater(len(self.text), 2000, "the acceptance map is suspiciously short")

    def test_the_map_references_enough_tests_that_this_check_is_not_vacuous(self) -> None:
        self.assertGreaterEqual(
            len(self.referenced),
            MINIMUM_REFERENCED_TESTS,
            "the map cites fewer tests than the floor: either the map was gutted "
            "or the extractor stopped matching",
        )

    def test_the_suite_declares_enough_tests_that_this_check_is_not_vacuous(self) -> None:
        self.assertGreaterEqual(len(self.declared), MINIMUM_REFERENCED_TESTS)

    def test_every_test_named_in_the_acceptance_map_exists_in_the_suite(self) -> None:
        missing = missing_references(self.referenced, self.declared)
        self.assertEqual(
            missing,
            [],
            "the acceptance map cites tests that do not exist: " + ", ".join(missing),
        )

    def test_no_test_name_is_declared_by_two_modules(self) -> None:
        """A cited name must identify exactly one test, or the map is ambiguous."""
        duplicated = {name: modules for name, modules in self.declared.items() if len(modules) > 1}
        self.assertEqual(duplicated, {}, f"test names declared more than once: {duplicated}")

    def test_every_test_module_in_the_suite_is_cited_by_the_map(self) -> None:
        """The map must cover the suite, not just agree with a corner of it."""
        cited_modules = {
            module for name in self.referenced for module in self.declared.get(name, [])
        }
        present = {path.name for path in TESTS.glob("test_*.py")}
        uncited = sorted(present - cited_modules)
        self.assertEqual(uncited, [], f"test modules the map never cites: {uncited}")


class AcceptanceMapNamesRealArtifacts(unittest.TestCase):
    """Every file, package and directory the map points at must resolve."""

    def setUp(self) -> None:
        self.text = DOC.read_text(encoding="utf-8")

    def test_the_map_references_repository_artifacts(self) -> None:
        self.assertGreaterEqual(len(artifact_references(self.text)), 20)

    def test_every_artifact_named_in_the_acceptance_map_resolves(self) -> None:
        unresolved = [ref for ref in artifact_references(self.text) if not resolve_reference(ref)]
        self.assertEqual(unresolved, [], f"the map names artifacts that do not exist: {unresolved}")

    def test_the_map_names_its_own_verifier(self) -> None:
        """The document claims this module exists; that claim is load-bearing."""
        self.assertIn("tests/test_acceptance_map.py", self.text)
        self.assertTrue((TESTS / "test_acceptance_map.py").is_file())


class TheExtractorsAreTwoSided(unittest.TestCase):
    """Controls. A green result above means nothing unless these hold."""

    SAMPLE = (
        "| A finding | An answer | `app/daily.py`, `domain/scoring.py` | "
        "`test_a_writable_skill_store_is_refused_outright`, "
        "`test_the_daily_module_does_not_import_the_promotion_module` |\n"
        "Prose mentioning tests/test_acceptance_map.py and test-driven work, "
        "plus a glob `rubrics/*.toml` and a directory `packaging/`.\n"
    )

    def test_the_test_name_extractor_finds_names_that_are_there(self) -> None:
        found = referenced_test_names(self.SAMPLE)
        self.assertEqual(
            found,
            {
                "test_a_writable_skill_store_is_refused_outright",
                "test_the_daily_module_does_not_import_the_promotion_module",
            },
        )

    def test_the_test_name_extractor_collects_nothing_from_prose(self) -> None:
        prose = "This paragraph talks about test_ names and test-driven development."
        self.assertEqual(referenced_test_names(prose), set())

    def test_the_test_name_extractor_does_not_mistake_a_module_path_for_a_test(self) -> None:
        self.assertEqual(referenced_test_names("`tests/test_acceptance_map.py`"), set())
        self.assertEqual(referenced_test_names("`test_public_safety.py`"), set())

    def test_a_cited_test_that_does_not_exist_is_reported_missing(self) -> None:
        """The mutation control: the comparison must be able to fail."""
        fabricated = "`test_this_name_is_not_declared_anywhere_in_the_suite`"
        missing = missing_references(referenced_test_names(fabricated), declared_test_names())
        self.assertEqual(missing, ["test_this_name_is_not_declared_anywhere_in_the_suite"])

    def test_a_cited_test_that_does_exist_is_not_reported_missing(self) -> None:
        """The silent half: the comparison must also be able to pass."""
        declared = declared_test_names()
        one = sorted(declared)[0]
        self.assertEqual(missing_references({one}, declared), [])

    def test_the_artifact_extractor_finds_paths_globs_and_directories(self) -> None:
        refs = artifact_references(self.SAMPLE)
        self.assertIn("app/daily.py", refs)
        self.assertIn("rubrics/*.toml", refs)
        self.assertIn("packaging/", refs)

    def test_the_artifact_extractor_leaves_out_a_path_on_the_host(self) -> None:
        sample = "`/private/var/root` (root's), below `/var`, and `app/daily.py`"
        self.assertEqual(artifact_references(sample), ["app/daily.py"])
        # The control: a repository path is still taken, wherever it is cited.
        self.assertEqual(artifact_references("`docs/ACCEPTANCE.md`"), ["docs/ACCEPTANCE.md"])

    def test_the_artifact_resolver_refuses_a_path_that_does_not_exist(self) -> None:
        self.assertEqual(resolve_reference("domain/there_is_no_such_module.py"), [])
        self.assertEqual(resolve_reference("no-such-directory/"), [])
        self.assertEqual(resolve_reference("no-such-glob/*.toml"), [])

    def test_the_artifact_resolver_accepts_each_shape_it_claims_to_handle(self) -> None:
        self.assertTrue(resolve_reference("app/daily.py"))  # package-relative
        self.assertTrue(resolve_reference("tools/leakscan.py"))  # repo-relative
        self.assertTrue(resolve_reference("rubrics/*.toml"))  # glob
        self.assertTrue(resolve_reference("packaging/"))  # directory
        self.assertTrue(resolve_reference("SKILL.md"))  # bare filename


if __name__ == "__main__":  # pragma: no cover - convenience only
    unittest.main()
