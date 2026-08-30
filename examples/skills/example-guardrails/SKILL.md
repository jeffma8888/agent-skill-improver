---
name: example-guardrails
description: Synthetic example skill that carries an executable payload and a safety rule. Listed in the example config as a protected skill id, so it is never a proposal target.
---

# Example: guardrails

> Synthetic skill. It exists to make one point concrete: **a skill is not a
> document.** It is a directory, and this one contains an executable script
> alongside its prose.

## Why this skill is protected

`scripts/block-risky-action.sh` is a payload, not documentation. Write access to a
skill tree is therefore arbitrary code execution on the host, and in a real setup
that can include write access to the guardrails themselves.

Two consequences, both enforced in code rather than by convention:

- The write path refuses any target filename other than `SKILL.md`, so a proposal
  can never touch a payload.
- This skill id is listed under `[guard].protected_skill_ids`, so it is never a
  proposal target at all.

## Rule

Confirm before doing anything the user cannot undo.
