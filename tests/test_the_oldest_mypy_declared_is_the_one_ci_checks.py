"""The oldest mypy the project declares is the one CI checks the source with.

``pyproject.toml`` declares the type checker a contributor installs as a floor
(``mypy>=1.8`` in the ``dev`` extra), and CI installs the newest one. Measured
before (the tree at 16c7a1f, 3.11): mypy 1.8.0, 1.9.0, 1.10.0, 1.11.0 and 1.11.2
each found one error, at ``app/wiring.py``, where a conditional expression's two
arms join to ``object`` before 1.12; 1.12.0 and every later one measured
(1.13.0 to 1.18.1, 2.3.1, 2.4.1) passed, so the declared floor was false and
nothing checked it. The wiring now assigns the isolation in an ``if``, which
1.8.0, 1.11.2 and 2.4.1 pass on Linux and macOS, and CI runs the declared floor
too ("Type gate at the declared floor"), as on Linux and as on macOS. These
tests hold that step's pin to the floor, so neither moves without the other,
both checks after the install, and the step to its name, its ``run`` block and
those three lines, so it has no condition, no failure allowed, no shell of its
own (it runs in the runner's ``bash -e``) and no line that lets a check fail.
Its install replaces the newest mypy for every later step of its job, so they
hold too that every other step of the job that runs mypy comes before it, and
that those steps check the source as on Linux and as on macOS: moved before the
install, the floor would pass, the install would keep 1.8.0 (it satisfies
``mypy>=1.8``), and CI would never check with the newest.

They read ``ci.yml`` as text in the form it is written here, not as YAML or as
bash: a step is a line ``- <key>:`` at the steps' indent, and the text of a
``|`` or ``>`` block is passed over when they find the steps (so a step written
inside another step's script is not taken for one). They hold that form, not its
meaning: the same step in another of YAML's forms (a flow mapping, a merge key)
fails them, and a check spelled other than a line ``mypy`` or ``mypy ...``
(``python -m mypy``, a line of a heredoc, which bash does not run) is not seen.
"""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STEP = "Type gate at the declared floor"


def declared_floor() -> tuple[int, ...]:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    (spec,) = [
        s for s in pyproject["project"]["optional-dependencies"]["dev"] if s.startswith("mypy")
    ]
    found = re.fullmatch(r"mypy>=(\d+(?:\.\d+)*)", spec)
    assert found, spec
    return tuple(int(n) for n in found.group(1).split("."))


def workflow() -> list[str]:
    return (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8").splitlines()


def depth(line: str) -> int:
    return len(line) - len(line.lstrip())


def passed_over(line: str) -> bool:
    """A blank line, or one starting with ``#``, wherever it is indented: between
    keys YAML reads it as a comment, and inside a ``run: |`` block it is a line
    of the script, which bash passes over."""
    return not line.strip() or line.strip().startswith("#")


#: A key whose value is a block (``run: |``, ``- run: >-``): the lines below it
#: that are deeper than the key are its text, not YAML.
_BLOCK_KEY = re.compile(r"^(?:- )?[^\s:#][^:#]*:\s*[|>][-+0-9]*\s*(?:#.*)?$")


def structure(lines: list[str]) -> list[int]:
    """The indices of the lines that are YAML here: neither passed over nor the
    text of a ``|`` or ``>`` block (every line after its key that is deeper than
    the key's column, the column after a ``- ``, and the blank lines among them)."""
    text_of_a_block: set[int] = set()
    i = 0
    while i < len(lines):
        if passed_over(lines[i]) or not _BLOCK_KEY.match(lines[i].strip()):
            i += 1
            continue
        column = depth(lines[i]) + (2 if lines[i].lstrip().startswith("- ") else 0)
        i += 1
        while i < len(lines) and (not lines[i].strip() or depth(lines[i]) > column):
            text_of_a_block.add(i)
            i += 1
    return [i for i, line in enumerate(lines) if i not in text_of_a_block and not passed_over(line)]


def floor_start(lines: list[str]) -> int:
    starts = [i for i in structure(lines) if lines[i].strip() == f"- name: {STEP}"]
    if len(starts) != 1:
        raise AssertionError(
            f"ci.yml has {len(starts)} lines '- name: {STEP}' outside a block, not one: the "
            "step is gone, renamed, or its name line holds more than its name"
        )
    return starts[0]


def body_of(lines: list[str], start: int) -> list[str]:
    """Every line of the step starting at ``start`` after its first, stripped.

    The step ends at the next line no deeper than its ``-`` (the next step) that
    is not passed over, so a line passed over is not taken for its end, and is
    not one of its lines.
    """
    body: list[str] = []
    for line in lines[start + 1 :]:
        if passed_over(line):
            continue
        if depth(line) <= depth(lines[start]):
            break
        body.append(line.strip())
    return body


def floor_step() -> list[str]:
    """Every line of the CI step named ``STEP`` after its name, stripped."""
    lines = workflow()
    return body_of(lines, floor_start(lines))


def the_floors_job() -> tuple[list[list[str]], int]:
    """Each step of the job ``STEP`` is in (its first line, then its body), and
    the index of ``STEP`` among them."""
    lines = workflow()
    start = floor_start(lines)
    level = depth(lines[start])
    yaml = structure(lines)
    parent = next(i for i in reversed(yaml) if i < start and depth(lines[i]) < level)
    if lines[parent].strip() != "steps:":
        raise AssertionError(f"the step {STEP!r} is not under a job's steps: {lines[parent]!r}")
    steps: list[list[str]] = []
    for i in (i for i in yaml if i > parent):
        if depth(lines[i]) <= depth(lines[parent]):
            break
        if depth(lines[i]) == level and lines[i].lstrip().startswith("- "):
            steps.append([lines[i].strip(), *body_of(lines, i)])
    return steps, [s[0] for s in steps].index(f"- name: {STEP}")


def mypy_checks(step: list[str]) -> list[str]:
    """The lines of a step that run mypy as this workflow writes it, without the
    ``run:``: a line ``mypy`` or ``mypy ...`` in its ``run`` block, or a key
    ``run: mypy ...``, the step's first line too (``- run: mypy``)."""
    found = [re.sub(r"^(?:- )?run:\s*", "", line) for line in step]
    return [line for line in found if line == "mypy" or line.startswith("mypy ")]


def padded(version: tuple[int, ...]) -> tuple[int, ...]:
    return version + (0,) * (3 - len(version))


class TheDeclaredFloorIsCheckedTests(unittest.TestCase):
    def test_ci_installs_the_declared_floor_exactly(self) -> None:
        pins = [
            re.fullmatch(r'python -m pip install "mypy==([\d.]+)"', line) for line in floor_step()
        ]
        found = [p.group(1) for p in pins if p]
        self.assertEqual(len(found), 1, f'no one line python -m pip install "mypy==N" in {pins}')
        (pin,) = found
        self.assertEqual(padded(tuple(int(n) for n in pin.split("."))), padded(declared_floor()))

    def test_the_floor_checks_the_source_as_on_linux_and_as_on_macos(self) -> None:
        body = floor_step()
        installs = [
            i for i, line in enumerate(body) if line.startswith('python -m pip install "mypy==')
        ]
        self.assertEqual(len(installs), 1, f"no one install of a pinned mypy in {body}")
        (install,) = installs
        self.assertLess(install, body.index("mypy"))
        self.assertLess(install, body.index("mypy --platform darwin"))

    def test_the_step_is_its_run_block_and_nothing_else(self) -> None:
        """No key but ``run`` (so no ``if``, ``continue-on-error`` or ``shell``,
        however spelled), and no line in it but the install and the two checks
        (so no ``set +e``, no ``|| true``): it runs, in the runner's ``bash -e``,
        and a check that fails fails the job."""
        shape = [re.sub(r"mypy==[\d.]+", "mypy==N", line) for line in floor_step()]
        self.assertEqual(
            shape, ["run: |", 'python -m pip install "mypy==N"', "mypy", "mypy --platform darwin"]
        )

    def test_the_newest_checks_both_platforms_before_the_floor_replaces_it(self) -> None:
        """Every other step of the floor's job that runs mypy comes before it, and
        those steps run both checks, ``mypy`` and ``mypy --platform darwin``."""
        steps, floor = the_floors_job()
        after = [step[0] for step in steps[floor + 1 :] if mypy_checks(step)]
        self.assertEqual(after, [], f"these run mypy after {STEP!r} installs the floor")
        before = [check for step in steps[:floor] for check in mypy_checks(step)]
        self.assertIn("mypy", before)
        self.assertIn("mypy --platform darwin", before)


if __name__ == "__main__":
    unittest.main()
