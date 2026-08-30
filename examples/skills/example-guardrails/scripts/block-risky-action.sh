#!/usr/bin/env bash
# Synthetic executable payload. It does nothing except prove a point: this file
# sits inside a "skill", so an unrestricted write path into a skill directory is a
# code-execution primitive wearing a documentation costume.
#
# agent-skill-improver never writes this file. The write path refuses any target
# whose filename is not SKILL.md, and this skill is additionally listed under
# [guard].protected_skill_ids in the example config.
set -euo pipefail
echo "example-guardrails: this payload is inert and is never edited by the loop" >&2
exit 0
