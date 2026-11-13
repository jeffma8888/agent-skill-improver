"""A skill inside a protected path is never proposed.

The config refuses an editable root that lies inside a protected path (this
tool's package, the config file and its directory, the state directory, the
rubrics directory), so that nothing is "proposed, scored and reviewed and then
refused at apply"; a root that holds one is allowed, and the write path refuses
the skills inside it. Measured before (3.11 and 3.13, the config file, or the
rubrics directory, in the folder of ``tdd``, the skill the fixtures fail):
``doctor`` said ready; ``daily --commit`` proposed ``tdd`` and paid the
improver and the judge (one candidate, its gate passed); ``review`` refused it,
``SelfMutationRefused`` "refusing to review <path>: it is inside a protected
path", naming no candidate; no token could be had, so ``apply`` was refused
too; and the record stayed ``validated``, open, counting against
``[caps].max_open_candidates``, until it was deleted or the path was moved
apart from the skill (then ``review`` and ``apply`` took it with the same token).

``propose_candidate`` now asks of the skill's canonical copy what the write asks
(``protected_dir_holding`` in ``domain/skills.py``, which both call) before the
improver is asked, and ``daily`` records the refusal as a policy refusal of the
skill, as it does a skill listed in ``[guard].protected_skill_ids``.
"""

from __future__ import annotations

import contextlib
import datetime
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import agent_skill_improver
from agent_skill_improver import cli
from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app import daily
from agent_skill_improver.app.daily import run_daily
from agent_skill_improver.app.pipeline import propose_candidate
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.config import load_config
from agent_skill_improver.domain.redaction import named_path
from agent_skill_improver.domain.skills import SkillGroup, SkillSnapshot, protected_dir_holding
from agent_skill_improver.errors import SelfMutationRefused
from tests.support import InProcessRunner, build_env

DAY = "2026-02-01"
BESIDE_THE_CONFIG = "the config file in the skill's folder"
IN_THE_RUBRICS = "the rubrics directory the skill's folder"
APART = "the config file in its own directory"
IN_A_PREFIX = "the config file in a folder named as a prefix of the skill's"
THROUGH_A_LINK = "the config file in the skill's folder, the roots spelled through a link"


def refusal(skill_id: str, path: object, guard: object) -> str:
    return (
        f"skill {skill_id!r} at {named_path(path)} lies inside a protected path "
        f"({named_path(guard)}), so it is never a proposal target: this tool never edits "
        "its own source, config, state or rubrics, and `apply` would refuse it. To have it "
        "proposed, keep the skill's folder and that path apart."
    )


def placed(base: Path, where: str, *, subprocess_model: bool = False) -> tuple[Path, Path]:
    """``build_env``'s tree with its config file or rubrics placed as ``where`` says.

    Returns the config path and the folder of ``tdd``. The state directory is
    named whole, so it stays where it was whichever folder the config is in.
    With ``THROUGH_A_LINK`` the skill roots are spelled through a link to the
    tree, while the protected paths are resolved: the inventory walks each root
    resolved, so the skill is still judged by where it is.
    """
    env = base / "env"
    config = build_env(env, seed="c236", use_subprocess_model=subprocess_model)
    skills = env / "skills-canonical"
    tdd = skills / "tdd"
    text = config.read_text(encoding="utf-8")
    assert text.count('state_dir = "state"\n') == 1 and text.count('dir = "../rubrics"\n') == 1
    text = text.replace('state_dir = "state"\n', f"state_dir = {json.dumps(str(env / 'state'))}\n")
    if where == THROUGH_A_LINK:
        (base / "link").symlink_to(env, target_is_directory=True)
        spelled = f"path = {json.dumps(str(env))[:-1]}/skills-"
        assert text.count(spelled) == 2, text
        text = text.replace(spelled, f'path = {json.dumps(str(base / "link"))[:-1]}/skills-')
    rubrics = env / "rubrics"
    if where == IN_THE_RUBRICS:
        for rubric in rubrics.glob("*.toml"):
            shutil.copy(rubric, tdd / rubric.name)
        rubrics = tdd
    text = text.replace('dir = "../rubrics"\n', f"dir = {json.dumps(str(rubrics))}\n")
    folder = {BESIDE_THE_CONFIG: tdd, THROUGH_A_LINK: tdd, IN_A_PREFIX: skills / "td"}.get(
        where, env / "config"
    )
    folder.mkdir(exist_ok=True)
    moved = folder / "asi.toml"
    moved.write_text(text, encoding="utf-8")
    config.unlink()
    return moved, tdd


def frozen_clock() -> FrozenClock:
    return FrozenClock(datetime.datetime(2026, 2, 1, 9, 0, 0, tzinfo=datetime.timezone.utc))


def run_pass(config_path: Path) -> tuple[dict[str, Any], InProcessRunner, Path]:
    config = load_config(config_path)
    runner = InProcessRunner()
    state = StateStore(config.state_dir, clock=frozen_clock())
    store = FilesystemSkillStore(
        config.skill_roots, read_only=True, protected_paths=config.protected_paths
    )
    receipt = run_daily(
        config,
        source=FixtureConversationSource(config.source.fixtures_dir),
        runner=runner,
        skill_store=store,
        state=state,
        clock=frozen_clock(),
        day=DAY,
        commit=True,
    )
    return receipt.to_dict(), runner, config.state_dir


def asked(runner: InProcessRunner) -> list[str]:
    return [role for role, _ in runner.calls if role in ("improver", "judge")]


class ASkillInsideAProtectedPathIsNotProposedTests(unittest.TestCase):
    def test_a_skill_beside_the_config_file_is_refused_before_the_improver(self) -> None:
        for where in (BESIDE_THE_CONFIG, IN_THE_RUBRICS, THROUGH_A_LINK):
            with self.subTest(where=where), tempfile.TemporaryDirectory() as tmp:
                config, tdd = placed(Path(tmp).resolve(), where)
                payload, runner, state_dir = run_pass(config)
                self.assertGreater(payload["counts"]["records_scored"], 0, "scoring still runs")
                self.assertEqual(payload["counts"]["candidates_proposed"], 0)
                self.assertEqual(payload["counts"]["proposals_refused_by_policy"], 1)
                self.assertEqual(asked(runner), [], "neither the improver nor the judge is asked")
                self.assertEqual(payload["skills"]["tdd"]["status"], "refused")
                self.assertEqual(payload["skills"]["tdd"]["stage"], "propose")
                self.assertEqual(
                    payload["skills"]["tdd"]["reason"], refusal("tdd", tdd / "SKILL.md", tdd)
                )
                self.assertEqual(list(state_dir.glob("candidates/*.json")), [])

    def test_the_control_with_the_config_file_apart_proposes_it(self) -> None:
        for where in (APART, IN_A_PREFIX):
            with self.subTest(where=where), tempfile.TemporaryDirectory() as tmp:
                config, _ = placed(Path(tmp).resolve(), where)
                payload, runner, state_dir = run_pass(config)
                self.assertEqual(payload["counts"]["candidates_proposed"], 1)
                self.assertIn("improver", asked(runner))
                self.assertEqual(len(list(state_dir.glob("candidates/*.json"))), 1)

    def test_daily_asks_with_every_protected_path_the_config_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, _ = placed(Path(tmp).resolve(), APART)
            config = load_config(config_path)
            asked_with: list[tuple[Path, ...]] = []

            def recording(*args: Any, **kwargs: Any) -> Any:
                asked_with.append(tuple(kwargs["protected_paths"]))
                return propose_candidate(*args, **kwargs)

            with mock.patch.object(daily, "propose_candidate", recording):
                payload, _, _ = run_pass(config_path)
            self.assertEqual(payload["counts"]["candidates_proposed"], 1)
            self.assertEqual(asked_with, [config.protected_paths])
            package = Path(agent_skill_improver.__file__).resolve().parent
            for guard in (package, config.config_path.resolve(), config.state_dir.resolve()):
                with self.subTest(guard=str(guard)):
                    self.assertIn(guard, asked_with[0])

    def test_the_command_proposes_nothing_and_exits_0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = placed(Path(tmp).resolve(), BESIDE_THE_CONFIG, subprocess_model=True)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                rc = cli.main(["--config", str(config), "daily", "--commit", "--day", DAY])
            self.assertEqual(rc, 0, out.getvalue())
            self.assertIn(" candidates=0 ", out.getvalue())
            self.assertEqual(list((config.parent.parent.parent / "state").glob("candidates/*")), [])


class TheRefusalNamesItsPathsTests(unittest.TestCase):
    def test_the_paths_are_named_each_on_its_line(self) -> None:
        guard = Path("/skills/a\nconfig error: forged/tdd")
        snapshot = SkillSnapshot(
            skill_id="tdd",
            root_id="canonical",
            path=guard / "SKILL.md",
            sha256="0" * 64,
            size_bytes=1,
        )
        group = SkillGroup(skill_id="tdd", canonical=snapshot, mirrors=())
        with tempfile.TemporaryDirectory() as tmp:
            prompt = load_config(build_env(Path(tmp) / "env", use_subprocess_model=False)).prompt
        runner = InProcessRunner()
        with self.assertRaises(SelfMutationRefused) as caught:
            propose_candidate(
                group,
                [],
                runner,
                conversations={},
                cfg=prompt,
                seed="c236",
                candidate_id=f"CAND-{DAY}-01-tdd",
                created_at="2026-02-01T09:00:00Z",
                protected_paths=(guard,),
                read_text=lambda path: self.fail("the skill was read"),
            )
        said = str(caught.exception)
        self.assertEqual(said, refusal("tdd", guard / "SKILL.md", guard))
        self.assertNotIn("\n", said)
        self.assertEqual(runner.calls, [])


class TheJudgmentTests(unittest.TestCase):
    def test_a_path_is_held_by_the_protected_path_it_is_or_lies_inside(self) -> None:
        config, rubrics = Path("/p/asi.toml"), Path("/p/skills/tdd")
        protected = (config, rubrics)
        cases = (
            (Path("/p/asi.toml"), config),
            (Path("/p/skills/tdd"), rubrics),
            (Path("/p/skills/tdd/SKILL.md"), rubrics),
            (Path("/p/skills/tdd/refs/deep/SKILL.md"), rubrics),
            (Path("/p/skills/tdd-x/SKILL.md"), None),
            (Path("/p/skills/td"), None),
            (Path("/p/skills"), None),
            (Path("/p/asi.toml.d/SKILL.md"), None),
        )
        for path, holder in cases:
            with self.subTest(path=str(path)):
                self.assertEqual(protected_dir_holding(path, protected), holder)
        self.assertIsNone(protected_dir_holding(Path("/p/skills/tdd/SKILL.md"), ()))

    def test_the_judgment_takes_both_paths_as_given(self) -> None:
        # Its callers resolve both sides; the judgment reads no filesystem, so a
        # path and a guard that are one place only through a link are two to it.
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp).resolve() / "real" / "tdd"
            real.mkdir(parents=True)
            link = Path(tmp).resolve() / "link"
            link.symlink_to(real.parent, target_is_directory=True)
            self.assertIsNone(protected_dir_holding(link / "tdd" / "SKILL.md", (real,)))
            self.assertIsNone(protected_dir_holding(real / "SKILL.md", (link / "tdd",)))
            self.assertEqual(protected_dir_holding(real / "SKILL.md", (real,)), real)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
