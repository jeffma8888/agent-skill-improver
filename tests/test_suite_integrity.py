"""Every test that is written must be a test that runs.

The defect this module exists for was measured on this tree. A `load_tests`
hook listed its module's test classes by hand; a class added later was not
added to the list, so its two tests were defined, read, cited as evidence and
never executed. Nothing errored, `Ran N tests / OK` printed as usual, and the
only way to notice was to count definitions against collections.

So the count is taken here, mechanically, in both directions:

* every test method the loader would find on a ``TestCase`` defined under
  ``tests/`` must appear in the suite that ``discover`` actually builds, with
  the uncollected names reported by name; and
* every class under ``tests/`` whose body defines a ``def test_*`` must be
  collected under its own name, so a test written on a class that is not a
  ``TestCase`` (or that a hook left out) is reported rather than ignored.

The checks carry controls, because a checker that finds nothing wrong and a
checker that finds nothing at all print the same word.
"""

from __future__ import annotations

import ast
import importlib
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"

# The suite is normally run as ``discover -s tests -t .``, which imports this
# module as ``tests.test_suite_integrity``; a bare ``discover -s tests`` imports
# it as ``test_suite_integrity``. The inner discovery below must use the same
# top-level directory as the outer one, or every test module is imported a second
# time under a second name and the comparison is between two different worlds.
PACKAGE = __name__.rpartition(".")[0]
TOP_LEVEL = REPO if PACKAGE else TESTS


def module_name(stem: str) -> str:
    return f"{PACKAGE}.{stem}" if PACKAGE else stem


# The floor exists so this module cannot pass vacuously: a discovery that
# imports nothing would otherwise compare one empty set with another.
MINIMUM_COLLECTED_TESTS = 300

TestId = tuple[str, str, str]  # (module, class, method)


def discovered_test_ids() -> set[TestId]:
    """Every test id the real discovery builds, from a fresh loader."""
    loader = unittest.TestLoader()
    suite = loader.discover(str(TESTS), top_level_dir=str(TOP_LEVEL))
    if loader.errors:
        raise AssertionError("discovery reported errors:\n" + "\n".join(loader.errors))
    found: set[TestId] = set()

    def walk(item: unittest.TestSuite | unittest.TestCase) -> None:
        if isinstance(item, unittest.TestSuite):
            for child in item:
                walk(child)
            return
        found.add((type(item).__module__, type(item).__name__, item._testMethodName))

    walk(suite)
    return found


def test_modules() -> list[str]:
    return sorted(module_name(path.stem) for path in TESTS.glob("test_*.py") if path.is_file())


def defined_test_ids(module_names: list[str]) -> set[TestId]:
    """Every test method the loader's own rules find on a TestCase in these modules."""
    loader = unittest.TestLoader()
    defined: set[TestId] = set()
    for name in module_names:
        module = importlib.import_module(name)
        for obj in vars(module).values():
            if not (isinstance(obj, type) and issubclass(obj, unittest.TestCase)):
                continue
            if obj.__module__ != module.__name__ or obj is unittest.TestCase:
                continue
            for method in loader.getTestCaseNames(obj):
                defined.add((module.__name__, obj.__name__, method))
    return defined


def classes_with_test_defs(source: str) -> set[str]:
    """Names of classes in ``source`` whose own body defines a ``def test_*``."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name.startswith("test"):
                    names.add(node.name)
                    break
    return names


class EveryDefinedTestIsCollected(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.modules = test_modules()
        cls.discovered = discovered_test_ids()
        cls.defined = defined_test_ids(cls.modules)

    def test_discovery_is_not_vacuous(self):
        self.assertGreaterEqual(len(self.discovered), MINIMUM_COLLECTED_TESTS)
        self.assertIn(
            (__name__, type(self).__name__, self._testMethodName),
            self.discovered,
            "discovery did not even find the test that is running",
        )

    def test_no_test_method_is_defined_but_uncollected(self):
        missing = sorted(self.defined - self.discovered)
        self.assertEqual(
            missing,
            [],
            "defined on a TestCase but absent from discovery (a load_tests hook "
            "or a module-level filter dropped them):\n"
            + "\n".join(f"  {m}.{c}.{t}" for m, c, t in missing),
        )

    def test_nothing_collected_was_not_defined(self):
        # The converse: discovery must not invent tests the modules do not own.
        extra = sorted(self.discovered - self.defined)
        self.assertEqual(extra, [], f"collected but not defined on a module TestCase: {extra}")

    def test_every_class_that_writes_a_test_is_collected_under_its_name(self):
        collected_classes = {(module, cls) for module, cls, _ in self.discovered}
        orphans: list[str] = []
        for name in self.modules:
            path = TESTS / f"{name.rpartition('.')[2]}.py"
            for cls in sorted(classes_with_test_defs(path.read_text(encoding="utf-8"))):
                if (name, cls) not in collected_classes:
                    orphans.append(f"{name}.{cls}")
        self.assertEqual(
            orphans,
            [],
            "classes that define test_* methods but were never collected "
            "(not a TestCase, or dropped by a hook):\n" + "\n".join(orphans),
        )


class ControlsTests(unittest.TestCase):
    """The extractors find what is there and report what is not."""

    def test_ast_extractor_finds_a_class_with_a_test_and_skips_one_without(self):
        source = (
            "import unittest\n"
            "class Helper:\n"
            "    def run(self): pass\n"
            "class WithTest(unittest.TestCase):\n"
            "    def test_one(self): pass\n"
            "class NotATestCaseButWritesOne:\n"
            "    def test_two(self): pass\n"
        )
        self.assertEqual(classes_with_test_defs(source), {"WithTest", "NotATestCaseButWritesOne"})

    def test_defined_extractor_uses_the_loaders_rules(self):
        # A method not starting with ``test`` is not a test to the loader, and
        # a TestCase imported from elsewhere belongs to its own module.
        ids = defined_test_ids([__name__])
        own = {(c, m) for mod, c, m in ids if mod == __name__}
        self.assertIn((type(self).__name__, self._testMethodName), own)
        self.assertNotIn((type(self).__name__, "setUpClass"), own)
        self.assertFalse(
            any(c == "TestCase" for c, _ in own), "unittest.TestCase itself was counted"
        )

    def test_mutation_control_an_uncollected_test_is_reported(self):
        # Build a module with a hand-listed load_tests that drops a class, the
        # exact shape of the measured defect, and confirm the comparison names it.
        src = (
            "import unittest\n"
            "class Kept(unittest.TestCase):\n"
            "    def test_kept(self): pass\n"
            "class Dropped(unittest.TestCase):\n"
            "    def test_dropped(self): pass\n"
            "def load_tests(loader, tests, pattern):\n"
            "    return loader.loadTestsFromTestCase(Kept)\n"
        )
        name = module_name("_suite_integrity_mutation_control")
        module = type(sys)(name)
        module.__file__ = str(TESTS / "_suite_integrity_mutation_control.py")
        exec(compile(src, module.__file__, "exec"), module.__dict__)
        sys.modules[name] = module
        try:
            defined = defined_test_ids([name])
            loader = unittest.TestLoader()
            collected: set[TestId] = set()

            def walk(item: unittest.TestSuite | unittest.TestCase) -> None:
                if isinstance(item, unittest.TestSuite):
                    for child in item:
                        walk(child)
                    return
                collected.add((name, type(item).__name__, item._testMethodName))

            walk(loader.loadTestsFromModule(module))
        finally:
            del sys.modules[name]
        self.assertEqual(sorted(defined - collected), [(name, "Dropped", "test_dropped")])
        self.assertIn((name, "Kept", "test_kept"), collected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
