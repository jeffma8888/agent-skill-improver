"""Application layer: use cases that wire ports together.

Import direction is one-way: ``app`` may import ``domain`` and ``adapters``;
neither may import ``app``. The one intentional asymmetry is that
:mod:`agent_skill_improver.app.daily` must not import
:mod:`agent_skill_improver.app.promote`, and a test enforces it by reading the
module source. That is the structural guarantee behind "daily can never mutate a
live skill" -- every other check is a belt on top of it.
"""
