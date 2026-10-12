"""The examiner is not writable by the examinee.

If the improver could edit the rubric, the cheapest way to pass would be to loosen
the examiner, and nothing in the output would look wrong. So the protection is
structural: derived in code, hashed on every artifact, and not configurable.
"""

from __future__ import annotations

import os
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

    def test_configuration_canonicalizes_the_state_directory_before_use(self):
        from tests.support import build_env

        real_parent = self.base / "real-parent"
        real_parent.mkdir()
        alias = self.base / "state-alias"
        alias.symlink_to(real_parent)
        config_path = build_env(self.base, rubric_ids=("instruction-compliance",))
        text = config_path.read_text(encoding="utf-8").replace(
            'state_dir = "state"', f'state_dir = "{alias / "state"}"'
        )
        config_path.write_text(text, encoding="utf-8")

        config = load_config(config_path)

        self.assertEqual(config.state_dir, (real_parent / "state").resolve())
        self.assertNotIn(alias, config.state_dir.parents)

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
        index = source.index("protected_dirs = {")
        block = source[index : index + 600]
        for expected in ("package_dir()", "config_path", "state_dir"):
            self.assertIn(
                expected,
                block,
                "these entries are derived in code. A configurable protection list "
                "is not a protection list: the loop's own config would be able to "
                "remove its own examiner from it",
            )
        self.assertIn(
            "protected = [config_path, *protected_dirs.values()]",
            block,
            "the write path's protected set is built from the same labelled directories "
            "the load-time root check uses, so neither can drift from the other",
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


class ADefinitionFileThatCannotBeReadIsAConfigErrorTests(unittest.TestCase):
    """``is_file`` passing does not mean the bytes can be read, or that they are
    UTF-8. Measured before: a config file at mode 000 ended ``doctor`` with a
    PermissionError traceback and exit 70, the code reserved for a bug.
    """

    def setUp(self) -> None:
        from tests.support import build_env

        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base, rubric_ids=("instruction-compliance",)))
        self.rubric_path = self.base / "rubrics" / "instruction-compliance.toml"

    def _unreadable(self, path: Path) -> None:
        os.chmod(path, 0o000)
        self.addCleanup(os.chmod, path, 0o644)

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file")
    def test_a_config_file_this_process_cannot_read_is_named_not_a_traceback(self):
        self._unreadable(self.config_path)
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        self.assertEqual(
            str(ctx.exception),
            f"config file {self.config_path} cannot be read (Permission denied)",
        )

    @unittest.skipIf(os.geteuid() == 0, "root reads a mode-000 file")
    def test_an_enabled_rubric_this_process_cannot_read_is_named_not_a_traceback(self):
        self._unreadable(self.rubric_path)
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        self.assertEqual(
            str(ctx.exception),
            f"rubric 'instruction-compliance' {self.rubric_path} cannot be read (Permission denied)",
        )

    def test_a_config_file_that_is_not_utf8_is_named_with_the_byte_offset(self):
        self.config_path.write_bytes(b"[run]\n" + bytes([0xFF]) + b"\n")
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        message = str(ctx.exception)
        self.assertTrue(message.startswith(f"config file {self.config_path} is not UTF-8 text ("))
        self.assertIn("at byte 6", message)

    def test_a_rubric_that_is_not_utf8_is_named_with_the_byte_offset(self):
        self.rubric_path.write_bytes(bytes([0xFF]) + b"\n")
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.config_path)
        message = str(ctx.exception)
        self.assertTrue(
            message.startswith(
                f"rubric 'instruction-compliance' {self.rubric_path} is not UTF-8 text ("
            ),
            message,
        )
        self.assertIn("at byte 0", message)


class AnUnknownConfigKeyIsRefusedNotIgnoredTests(unittest.TestCase):
    """Measured before: ``[caps].max_open_candidate = 1`` (a dropped letter)
    loaded without a word and the run kept the default cap of 12, while the
    operator believed the queue was held to one. ``[source]`` was already a
    closed schema; every other table now is too.
    """

    def setUp(self) -> None:
        from tests.support import build_env

        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.config_path = Path(build_env(self.base, rubric_ids=("instruction-compliance",)))
        self.text = self.config_path.read_text(encoding="utf-8")

    def _load_with(self, old: str, new: str):
        assert self.text.count(old) == 1, old
        self.config_path.write_text(self.text.replace(old, new), encoding="utf-8")
        return load_config(self.config_path)

    def _refusal_for(self, old: str, new: str) -> str:
        with self.assertRaises(ConfigError) as ctx:
            self._load_with(old, new)
        return str(ctx.exception)

    def test_the_control_the_spelt_key_takes_effect(self):
        config = self._load_with("max_open_candidates = 12", "max_open_candidates = 1")
        self.assertEqual(config.caps.max_open_candidates, 1)

    def test_a_misspelt_cap_is_refused_with_the_nearest_key(self):
        message = self._refusal_for("max_open_candidates = 12", "max_open_candidate = 1")
        self.assertTrue(
            message.startswith(
                "[caps] has unknown key(s): max_open_candidate (did you mean "
                "max_open_candidates?). Accepted keys are "
            ),
            message,
        )
        self.assertIn("refused rather than ignored", message)

    def test_an_unknown_top_level_table_is_refused(self):
        message = self._refusal_for("[guard]\n", "[cap]\nmax_open_candidates = 1\n\n[guard]\n")
        self.assertTrue(
            message.startswith("the config file has unknown key(s): cap (did you mean caps?)."),
            message,
        )

    def test_a_key_with_no_near_match_is_named_bare(self):
        message = self._refusal_for('state_dir = "state"', 'state_dir = "state"\nverbosity = 3')
        self.assertTrue(message.startswith("[run] has unknown key(s): verbosity. "), message)

    def test_every_table_is_closed(self):
        cases = {
            "[selection]": ("fairness_dimension", "fairness_dimensions"),
            "[prompt]": ("redact_secrets", "redact_secret"),
            "[gate]": ("require_judge", "require_judges"),
            "[rubrics]": ("enabled", "enable"),
        }
        for where, (good, bad) in cases.items():
            with self.subTest(where=where):
                line = next(ln for ln in self.text.splitlines() if ln.startswith(f"{good} = "))
                message = self._refusal_for(line, line.replace(good, bad, 1))
                self.assertTrue(message.startswith(f"{where} has unknown key(s): {bad} "), message)
        message = self._refusal_for("[guard]\n", "[guard]\nprotected_skill_id = []\n")
        self.assertTrue(
            message.startswith("[guard] has unknown key(s): protected_skill_id "), message
        )

    def test_a_skill_root_entry_and_a_model_role_are_closed_too(self):
        message = self._refusal_for('kind = "mirror"', 'kind = "mirror"\nprecedance = 5')
        self.assertTrue(
            message.startswith(
                "[[skill_root]] #1 has unknown key(s): precedance (did you mean precedence?)."
            ),
            message,
        )
        message = self._refusal_for(
            "[model.defaults]\ntimeout_seconds = 30", "[model.defaults]\ntimeout_second = 30"
        )
        self.assertTrue(
            message.startswith(
                "[model.defaults] has unknown key(s): timeout_second (did you mean "
                "timeout_seconds?)."
            ),
            message,
        )
        message = self._refusal_for("[model.scorer]\n", '[model.scorer]\nmodel = "x"\n')
        self.assertTrue(
            message.startswith(
                "[model.scorer] has unknown key(s): model (did you mean model_label?)."
            ),
            message,
        )
        message = self._refusal_for(
            "[model.scorer]\n", '[model.scorrer]\nargv = ["x"]\n\n[model.scorer]\n'
        )
        self.assertTrue(
            message.startswith("[model] has unknown key(s): scorrer (did you mean scorer?)."),
            message,
        )

    def test_doctor_reports_it_as_a_config_error(self):
        from tests.support import run_cli

        self.config_path.write_text(
            self.text.replace("max_open_candidates = 12", "max_open_candidate = 1"),
            encoding="utf-8",
        )
        proc = run_cli(["--config", str(self.config_path), "doctor"])
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertIn("config error: [caps] has unknown key(s): max_open_candidate", proc.stderr)

    def test_the_shipped_example_config_uses_only_accepted_keys(self):
        config = load_config(REPO_ROOT / "config" / "example.config.toml")
        self.assertTrue(config.caps.max_open_candidates >= 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
