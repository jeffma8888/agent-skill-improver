"""Shared test harness: a throwaway repo-shaped environment on a temp dir.

The layout mirrors the real repo deliberately, because the config loader derives
`base_dir` from whether the config sits in a directory called ``config`` and
derives its protected-path set from that. A harness with a different layout would
test a different code path than the one that ships.

    <tmp>/config/config.toml     <- protected (never writable by the loop)
    <tmp>/rubrics/*.toml         <- protected
    <tmp>/state/                 <- protected, gitignored in the real repo
    <tmp>/skills-canonical/      <- writable target, canonical
    <tmp>/skills-mirror/         <- mirror, refused at the write path
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from agent_skill_improver import SCHEMA_MODEL_REQUEST
from agent_skill_improver.errors import MalformedModelOutput, TransientFailure

from tests import fake_model

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_EXPORTS = REPO_ROOT / "tests" / "fixtures" / "exports"
FIXTURE_MALFORMED = REPO_ROOT / "tests" / "fixtures" / "malformed"
FAKE_MODEL = REPO_ROOT / "tests" / "fake_model.py"

FIX_MARKER = fake_model.FIX_MARKER
FAILING_THREAD = fake_model.FAILING_THREAD

TDD_SKILL = """---
name: tdd
description: Test-driven development. Use when building features or fixing bugs test-first.
---

# Test-driven development

- Write the failing test before the fix.
- Run the test and watch it fail for the right reason.
- Make it pass with the smallest change.
"""

CODE_REVIEW_SKILL = """---
name: code-review
description: Review a diff for correctness and convention adherence.
---

# Code review

- Read the whole diff before commenting.
- Separate blocking findings from advisory ones.
"""

# A real and nasty case: no opening '---', so the frontmatter never parses and
# the skill silently never loads. Discovery must report it rather than ignore it.
BROKEN_SKILL = """name: broken-frontmatter
description: This file is missing its opening delimiter.
---

# Broken

This skill never loads and nothing anywhere errors.
"""

# Lives only in the mirror root, which is non-editable. There is therefore no
# canonical editable copy and proposals against it must be refused outright.
MIRROR_ONLY_SKILL = """---
name: mirror-only
description: Present only in a non-editable mirror root.
---

# Mirror only
"""


def write_skill(root: Path, skill_id: str, text: str) -> Path:
    path = root / skill_id / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def build_skill_tree(base: Path, *, mirror_divergent: bool = False) -> tuple[Path, Path]:
    canonical = base / "skills-canonical"
    mirror = base / "skills-mirror"
    write_skill(canonical, "tdd", TDD_SKILL)
    write_skill(canonical, "code-review", CODE_REVIEW_SKILL)
    write_skill(canonical, "broken-frontmatter", BROKEN_SKILL)
    write_skill(mirror, "tdd", TDD_SKILL + ("\n- mirror has drifted\n" if mirror_divergent else ""))
    write_skill(mirror, "mirror-only", MIRROR_ONLY_SKILL)
    return canonical, mirror


def copy_rubrics(base: Path, rubric_ids: Sequence[str]) -> Path:
    target = base / "rubrics"
    target.mkdir(parents=True, exist_ok=True)
    for rubric_id in rubric_ids:
        source = REPO_ROOT / "rubrics" / f"{rubric_id}.toml"
        (target / f"{rubric_id}.toml").write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return target


def _toml_list(values: Sequence[str]) -> str:
    return "[" + ", ".join(json.dumps(v) for v in values) + "]"


def build_env(
    base: Path,
    *,
    rubric_ids: Sequence[str] = ("instruction-compliance",),
    source_kind: str = "fixtures",
    fixtures_dir: Path | None = None,
    dry_run: bool = False,
    max_conversations: int = 8,
    fairness_dimension: str = "workspace",
    max_open_candidates: int = 12,
    max_candidates_per_run: int = 3,
    min_holdout_cases: int = 1,
    min_fixes: int = 1,
    require_judge: bool = True,
    require_independent_judge: bool = True,
    mirror_divergent: bool = False,
    use_subprocess_model: bool = True,
    protected_skill_ids: Sequence[str] = (),
    seed: str = "test-seed",
) -> Path:
    """Write a complete config tree under ``base`` and return the config path."""
    canonical, mirror = build_skill_tree(base, mirror_divergent=mirror_divergent)
    copy_rubrics(base, rubric_ids)
    config_dir = base / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    def role_argv(role: str) -> str:
        """A DISTINCT command per role, matching the shipped example's shape.

        Distinctness matters: the independence check compares what actually
        executes, so a harness that wired all three roles to the byte-identical
        command could never exercise a passing gate. ``tests/fake_model.py`` reads
        its role from the request body and ignores argv, so the flag is inert to
        the fake and meaningful to the check.

        Note honestly that this is the same weak proxy the real check has: one
        script behind three flags is not three independent evaluators. It is
        exactly enough to exercise the code path.
        """
        if use_subprocess_model:
            return _toml_list([sys.executable, str(FAKE_MODEL), "--role", role])
        return _toml_list(["/nonexistent/model-command", "--role", role])

    exports = fixtures_dir if fixtures_dir is not None else FIXTURE_EXPORTS
    config = f"""
[run]
state_dir = "state"
dry_run = {str(dry_run).lower()}
seed = {json.dumps(seed)}

[source]
kind = {json.dumps(source_kind)}
fixtures_dir = {json.dumps(str(exports))}
since = "7d"
discovery_limit = 50
keep_raw = true
output_excerpt_chars = 2000
timeout_seconds = 30

[selection]
max_conversations_per_run = {max_conversations}
fairness_dimension = {json.dumps(fairness_dimension)}

[caps]
max_open_candidates = {max_open_candidates}
max_candidates_per_run = {max_candidates_per_run}
max_skills_per_run = 3
max_failures_per_candidate = 6
max_extra_holdout_conversations = 3

[prompt]
max_transcript_chars = 24000
max_message_chars = 4000
max_skill_text_chars = 24000
redact_secrets = true

[gate]
min_fixes = {min_fixes}
max_regressions = 0
max_holdout_degradations = 0
min_holdout_cases = {min_holdout_cases}
min_judge_preference_ratio = 0.5
require_judge = {str(require_judge).lower()}
require_independent_judge = {str(require_independent_judge).lower()}

[[skill_root]]
id = "canonical"
path = {json.dumps(str(canonical))}
kind = "custom"
precedence = 10
canonical = true
editable = true

[[skill_root]]
id = "mirror"
path = {json.dumps(str(mirror))}
kind = "mirror"
precedence = 90
canonical = false
editable = false

[rubrics]
dir = "../rubrics"
enabled = {_toml_list(list(rubric_ids))}

[model.defaults]
timeout_seconds = 30

[model.scorer]
argv = {role_argv("scorer")}
model_label = "fake-scorer"

[model.improver]
argv = {role_argv("improver")}
model_label = "fake-improver"

[model.judge]
argv = {role_argv("judge")}
model_label = "fake-judge"

[guard]
protected_skill_ids = {_toml_list(list(protected_skill_ids))}
""".lstrip()
    path = config_dir / "config.toml"
    path.write_text(config, encoding="utf-8")
    return path


class InProcessRunner:
    """Same answers as :mod:`tests.fake_model`, without paying for a subprocess.

    Deliberately shares the fake's logic rather than reimplementing it, so the
    fast tests and the end-to-end subprocess test cannot silently disagree about
    what the model said.
    """

    #: What this double reports when asked about confinement. Overridable so a
    #: test can construct the *satisfied* branch explicitly and visibly, rather
    #: than by patching a module.
    isolation_mode = "in-process"

    def __init__(
        self, *, mode: str = "", fail_roles: Sequence[str] = (), isolation_mode: str | None = None
    ) -> None:
        self.mode = mode
        self.fail_roles = set(fail_roles)
        self.calls: list[tuple[str, str]] = []
        #: Every prompt body any role was shown, so a test can assert that a
        #: given string never reached a model.
        self.prompts: list[str] = []
        if isolation_mode is not None:
            self.isolation_mode = isolation_mode

    def prompts_seen(self) -> list[str]:
        return list(self.prompts)

    def command_id(self, role: str) -> str:
        return f"{role}:in-process-fake:iso={self.isolation_mode}"

    def isolation(self) -> dict[str, Any]:
        """Deliberately NOT an OS sandbox.

        This double runs in the test process, so there is nothing to confine and
        nothing it could honestly claim. Reporting ``in-process`` means every
        isolation test that uses it exercises the *refusing* branch for a private
        source -- which is the branch that matters. A double that reported a
        verified sandbox would make the whole boundary untested.
        """
        return {
            "mode": self.isolation_mode,
            "mechanism": {"mode": self.isolation_mode, "executable": ""},
            "verified": False,
            "verification": {},
            "env_allowlist": [],
            "env_names_dropped_from_parent": 0,
        }

    def run(self, *, role: str, system: str, user: str, response_schema: str) -> dict[str, Any]:
        self.calls.append((role, response_schema))
        self.prompts.append(system + "\n" + user)
        if role in self.fail_roles:
            raise TransientFailure(f"in-process fake refuses role {role!r}")
        if self.mode == "malformed_json" and role == "scorer":
            raise MalformedModelOutput("fake produced no parseable JSON object")
        if role == "scorer":
            return fake_model.scorer(user, self.mode)
        if role == "improver":
            return fake_model.improver(user)
        if role == "judge":
            return fake_model.judge(user)
        raise TransientFailure(f"unknown role {role!r}")


def run_cli(
    args: Sequence[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Invoke the CLI the way an operator (or a scheduler) would."""
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + str(REPO_ROOT)
    full_env.pop("ASI_CONFIG", None)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "agent_skill_improver", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=full_env,
        timeout=300,
    )


def model_request(role: str, user: str) -> str:
    return json.dumps(
        {
            "schema": SCHEMA_MODEL_REQUEST,
            "role": role,
            "response_schema": "asi.scorecard.v1",
            "system": "",
            "user": user,
        }
    )
