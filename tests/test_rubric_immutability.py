"""The examiner is not writable by the examinee.

If the improver could edit the rubric, the cheapest way to pass would be to loosen
the examiner, and nothing in the output would look wrong. So the protection is
structural: derived in code, hashed on every artifact, and not configurable.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.config import load_config, package_dir
from agent_skill_improver.domain.rubric import Rubric
from agent_skill_improver.domain.skills import SKILL_FILENAME, SkillRoot
from agent_skill_improver.errors import ConfigError, SelfMutationRefused
from agent_skill_improver.hashing import sha256_file, sha256_text

from tests.support import REPO_ROOT

RUBRIC_TOML = """
[rubric]
id = "sample"
version = 3
title = "A sample rubric"
min_evidence_refs = 1

[[criterion]]
id = "only-criterion"
title = "The only criterion"
weight = 2
requires_evidence = true
question = "Was the thing done?"
insufficient_when = "the transcript does not show whether the thing was attempted"
"""


class RubricVersioningTests(unittest.TestCase):
    def _load(self, text: str) -> Rubric:
        import tomllib

        return Rubric.from_mapping(tomllib.loads(text), source_sha256=sha256_text(text))

    def test_a_rubric_carries_its_declared_version(self):
        self.assertEqual(self._load(RUBRIC_TOML).version, 3)

    def test_a_rubric_carries_the_hash_of_the_file_it_came_from(self):
        rubric = self._load(RUBRIC_TOML)
        self.assertEqual(rubric.source_sha256, sha256_text(RUBRIC_TOML))

    def test_an_edit_changes_the_hash_even_when_the_version_does_not(self):
        original = self._load(RUBRIC_TOML)
        loosened = self._load(RUBRIC_TOML.replace("Was the thing done?", "Was anything done?"))
        self.assertEqual(original.version, loosened.version)
        self.assertNotEqual(
            original.source_sha256,
            loosened.source_sha256,
            "the hash is what catches an undeclared edit: two scorecards with the "
            "same version and different hashes mean the examination changed silently",
        )

    def test_criterion_ids_are_exposed_for_the_scorecard_contract(self):
        self.assertEqual(self._load(RUBRIC_TOML).criterion_ids, ("only-criterion",))

    def test_the_hash_is_serialized_so_it_reaches_the_artifacts(self):
        self.assertIn("source_sha256", self._load(RUBRIC_TOML).to_dict())

    def test_a_rubric_with_no_criteria_is_a_configuration_error(self):
        import tomllib

        text = "[rubric]\nid = 'empty'\nversion = 1\ntitle = 't'\n"
        with self.assertRaises(Exception):
            Rubric.from_mapping(tomllib.loads(text))


class ShippedRubricTests(unittest.TestCase):
    """The rubrics in this repository must actually load, and stay honest."""

    def rubric_files(self) -> list[Path]:
        return sorted((REPO_ROOT / "rubrics").glob("*.toml"))

    def test_the_library_is_not_empty(self):
        self.assertGreaterEqual(len(self.rubric_files()), 2)

    def test_every_shipped_rubric_parses(self):
        import tomllib

        for path in self.rubric_files():
            with self.subTest(path=path.name):
                rubric = Rubric.from_mapping(
                    tomllib.loads(path.read_text(encoding="utf-8")),
                    source_sha256=sha256_file(path),
                )
                self.assertTrue(rubric.criterion_ids)

    def test_every_declared_id_equals_its_filename_stem(self):
        import tomllib

        for path in self.rubric_files():
            with self.subTest(path=path.name):
                raw = tomllib.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    raw["rubric"]["id"],
                    path.stem,
                    "a mismatch is a silent misconfiguration: the loader looks the "
                    "file up by stem and would enable a different rubric than named",
                )

    def test_every_criterion_states_when_the_answer_is_insufficient(self):
        import tomllib

        for path in self.rubric_files():
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
            for criterion in raw.get("criterion", []):
                with self.subTest(path=path.name, criterion=criterion.get("id")):
                    self.assertTrue(
                        criterion.get("insufficient_when"),
                        "without this field a model that cannot tell will pick, and a "
                        "guessed FAIL becomes a real edit to a real file",
                    )

    def test_criterion_ids_are_unique_within_a_rubric(self):
        import tomllib

        for path in self.rubric_files():
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
            ids = [c["id"] for c in raw.get("criterion", [])]
            with self.subTest(path=path.name):
                self.assertEqual(len(ids), len(set(ids)))


class ProtectedPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _config(self) -> object:
        from tests.support import build_env

        return load_config(build_env(self.base, rubric_ids=("instruction-compliance",)))

    def test_the_rubric_directory_is_protected(self):
        config = self._config()
        rubric_dir = (self.base / "rubrics").resolve()
        self.assertIn(rubric_dir, config.protected_paths)

    def test_the_package_source_is_protected(self):
        self.assertIn(package_dir().resolve(), self._config().protected_paths)

    def test_the_config_file_and_its_directory_are_protected(self):
        config = self._config()
        self.assertIn(config.config_path.resolve(), config.protected_paths)
        self.assertIn(config.config_path.parent.resolve(), config.protected_paths)

    def test_the_state_directory_is_protected(self):
        config = self._config()
        self.assertIn(config.state_dir.resolve(), config.protected_paths)

    def test_a_write_inside_a_protected_path_is_refused(self):
        config = self._config()
        # Pretend a rubric file is a skill document, which is the shape of the
        # attack: get the loop to treat its own examiner as an editable target.
        rubric_dir = self.base / "rubrics"
        fake = rubric_dir / "sneaky" / SKILL_FILENAME
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text("---\nname: sneaky\ndescription: d\n---\n", encoding="utf-8")
        store = FilesystemSkillStore(
            [
                SkillRoot(
                    id="r",
                    path=rubric_dir,
                    precedence=1,
                    canonical=True,
                    editable=True,
                    kind="custom",
                )
            ],
            read_only=False,
            protected_paths=config.protected_paths,
        )
        with self.assertRaises(SelfMutationRefused):
            store.write_canonical(
                fake, "x", expected_sha256=sha256_file(fake), backup_dir=self.base / "bak"
            )

    def test_the_protected_set_is_not_taken_from_configuration(self):
        source = (package_dir() / "config.py").read_text(encoding="utf-8")
        index = source.index("protected = [")
        block = source[index : index + 400]
        for expected in ("package_dir()", "config_path", "state_dir"):
            self.assertIn(
                expected,
                block,
                "these entries are derived in code. A configurable protection list "
                "is not a protection list: the loop's own config would be able to "
                "remove its own examiner from it",
            )

    def test_a_missing_enabled_rubric_is_a_configuration_error(self):
        from tests.support import build_env

        config_path = build_env(self.base, rubric_ids=("instruction-compliance",))
        text = config_path.read_text(encoding="utf-8").replace(
            '"instruction-compliance"', '"does-not-exist"'
        )
        config_path.write_text(text, encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(config_path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
