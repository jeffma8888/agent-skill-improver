#!/usr/bin/env python3
"""A deterministic offline "model" for the three roles, so the loop runs anywhere.

Contract, identical to a real role command: read one JSON object on stdin, write
one JSON object on stdout. Nothing here reaches the network, imports the package,
or reads anything outside its own stdin.

    {"schema": "asi.model_request.v1", "role": ..., "system": ..., "user": ...,
     "response_schema": ...}

What each role does:

* ``scorer``   -- reads control markers embedded in the synthetic transcript (see
  ``examples/build_fixtures.py``) and emits the corresponding verdicts, citing
  real element ids taken from the prompt's CITABLE ELEMENTS block. For a
  ``[[FIXABLE:...]]`` marker it emits FAIL unless the skill text under evaluation
  contains the required rule, which is what makes a baseline-versus-candidate
  replay measure something.
* ``improver`` -- extracts the current skill text from its untrusted envelope and
  appends one rule containing the required text. It never touches the frontmatter.
* ``judge``    -- prefers whichever blind option contains the required rule, and
  answers ``tie`` when both or neither do.

Honest about what it is: this is a *fixture*, not a model. It proves the pipeline,
the contracts, the trust rules and the gate arithmetic are wired correctly. It
proves nothing about whether a real model would score well. It also demonstrates
the intended response to prompt injection -- it records the attempt as an
observation and scores normally -- but a fake cannot be evidence that injection is
solved, and ``docs/PRIVACY_AND_THREAT_MODEL.md`` says so plainly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

MARKER_RE = re.compile(r"\[\[(PASS|FAIL|FIXABLE|BADCITE|INSUFFICIENT|DEFER)(?::([^\]]*))?\]\]")
CITABLE_RE = re.compile(r"## CITABLE ELEMENTS.*?```json\n(.*?)\n```", re.DOTALL)
CONVERSATION_RE = re.compile(r"Score conversation (\S+) against")
RUBRIC_ID_RE = re.compile(r"^rubric_id:\s*(\S+)\s*$", re.MULTILINE)
CRITERION_ID_RE = re.compile(r"^- id:\s*(\S+)\s*$", re.MULTILINE)
SKILL_UNDER_EVAL_RE = re.compile(r"## SKILL TEXT UNDER EVALUATION.*?\nskill_id:\s*(\S+)", re.DOTALL)
FAILURES_RE = re.compile(r"## RUBRIC FAILURES TO ADDRESS.*?```json\n(.*?)\n```", re.DOTALL)
TARGET_SKILL_RE = re.compile(r"Propose a replacement for the skill file '([^']+)'")
REQUIRED_RULE_RE = re.compile(r"required-rule:\s*(.+?)\s*(?:\||$)", re.MULTILINE)

FABRICATED_ELEMENT_ID = "el-does-not-exist-9999"


def fenced(text: str, label: str) -> str | None:
    """Extract one untrusted envelope's body verbatim, or ``None``.

    Returning ``None`` rather than guessing matters: a fake that silently invents a
    base text would produce an enormous diff and the failure would look like a
    model quality problem instead of a harness bug.
    """
    pattern = re.compile(
        r"<<<UNTRUSTED-DATA "
        + re.escape(label)
        + r" ([0-9a-f]+)>>>\n(.*?)\n<<<END-UNTRUSTED-DATA "
        + re.escape(label)
        + r" \1>>>",
        re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(2) if match else None


def markers(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), (m.group(2) or "").strip()) for m in MARKER_RE.finditer(text)]


def split_marker_arg(raw: str) -> tuple[str, str]:
    """``criterion|rule`` -> ``(criterion, rule)``; a bare id has no rule."""
    if "|" in raw:
        criterion, _, rule = raw.partition("|")
        return criterion.strip(), rule.strip()
    return raw.strip(), ""


def citable(text: str) -> list[dict[str, Any]]:
    match = CITABLE_RE.search(text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def pick_reference(elements: list[dict[str, Any]], *, kind: str = "") -> dict[str, Any] | None:
    for element in elements:
        if kind and element.get("kind") != kind:
            continue
        return {
            "element_id": element.get("element_id"),
            "position": element.get("position"),
            "kind": element.get("kind"),
        }
    return None


def injection_noted(user: str) -> str:
    """Describe hostile content instead of obeying it."""
    lowered = user.lower()
    tells = []
    if "ignore all previous instructions" in lowered:
        tells.append("an instruction telling the grader to ignore its instructions")
    if "end-untrusted-data" in lowered and "[neutralized-delimiter]" in lowered:
        tells.append("a forged envelope delimiter, which the harness neutralized")
    elif "[neutralized-delimiter]" in lowered:
        tells.append("a neutralized forged delimiter")
    if not tells:
        return ""
    return (
        " The conversation contains " + " and ".join(tells) + "; recorded as an observation about "
        "the conversation and scored normally."
    )


# --- roles ---------------------------------------------------------------
def run_scorer(request: dict[str, Any]) -> dict[str, Any]:
    user = str(request.get("user", ""))
    conversation_match = CONVERSATION_RE.search(user)
    conversation_id = conversation_match.group(1) if conversation_match else ""
    elements = citable(user)
    criteria_ids = CRITERION_ID_RE.findall(user)
    skill_match = SKILL_UNDER_EVAL_RE.search(user)
    skill_text = fenced(user, "skill-text") or "" if skill_match else ""
    note = injection_noted(user)

    decided: dict[str, dict[str, Any]] = {}
    for kind, raw in markers(user):
        if kind == "DEFER":
            continue
        criterion_id, rule = split_marker_arg(raw)
        if criterion_id not in criteria_ids or criterion_id in decided:
            continue
        message_ref = pick_reference(elements, kind="message")
        if kind == "PASS":
            decided[criterion_id] = {
                "criterion_id": criterion_id,
                "verdict": "PASS",
                "rationale": "The transcript shows the criterion satisfied." + note,
                "evidence": [message_ref] if message_ref else [],
                "confidence": 0.8,
            }
        elif kind == "INSUFFICIENT":
            decided[criterion_id] = {
                "criterion_id": criterion_id,
                "verdict": "INSUFFICIENT",
                "rationale": (
                    "The transcript does not contain enough information to decide this "
                    "criterion." + note
                ),
                "evidence": [],
                "confidence": 0.2,
            }
        elif kind == "BADCITE":
            # A confident accusation with an unresolvable citation. The tool under
            # test must refuse to treat this as a FAIL.
            decided[criterion_id] = {
                "criterion_id": criterion_id,
                "verdict": "FAIL",
                "rationale": "Asserting a failure while citing an element that does not exist."
                + note,
                "evidence": [
                    {"element_id": FABRICATED_ELEMENT_ID, "position": 9999, "kind": "message"}
                ],
                "confidence": 0.99,
            }
        elif kind in ("FAIL", "FIXABLE"):
            satisfied = bool(kind == "FIXABLE" and rule and rule.lower() in skill_text.lower())
            verdict = "PASS" if satisfied else "FAIL"
            rationale = (
                f"The skill text under evaluation now states the required rule ({rule!r}), and the "
                "transcript behaviour matches it."
                if satisfied
                else "A cited turn violates this criterion."
                + (f" required-rule: {rule}" if rule else "")
            )
            decided[criterion_id] = {
                "criterion_id": criterion_id,
                "verdict": verdict,
                "rationale": rationale + note,
                "evidence": [message_ref] if message_ref else [],
                "confidence": 0.85,
            }
    # Criteria with no marker are deliberately omitted. The tool records an
    # omission as INSUFFICIENT, never as PASS, and this fake does not paper over
    # that by inventing a verdict.
    return {
        "schema": "asi.scorecard.v1",
        "conversation_id": conversation_id,
        "criteria": [decided[cid] for cid in criteria_ids if cid in decided],
    }


def run_improver(request: dict[str, Any]) -> dict[str, Any]:
    user = str(request.get("user", ""))
    target_match = TARGET_SKILL_RE.search(user)
    target = target_match.group(1) if target_match else ""
    base = fenced(user, "skill-text")
    if base is None:
        raise SystemExit("fake improver: could not find the current skill text envelope")

    rule = ""
    failures_match = FAILURES_RE.search(user)
    if failures_match:
        try:
            failures = json.loads(failures_match.group(1))
        except json.JSONDecodeError:
            failures = []
        for failure in failures if isinstance(failures, list) else []:
            found = REQUIRED_RULE_RE.search(str(failure.get("rationale", "")))
            if found:
                rule = found.group(1).strip().rstrip(".")
                break
    if not rule:
        rule = "state the evidence behind each claim"

    addition = "\n## Rule added by a reviewed proposal\n\n" f"- Before asserting a fact, {rule}.\n"
    new_text = base.rstrip("\n") + "\n" + addition
    return {
        "schema": "asi.proposal.v1",
        "target_skill_id": target,
        "hypothesis": (
            "The cited failures all share one shape: the assistant asserted something without "
            f"showing its basis. The skill never tells it to {rule}, so adding exactly that rule "
            f"should prevent a recurrence. required-rule: {rule}"
        ),
        "intended_change": (
            "Append one short section at the end of the file containing a single imperative "
            f"bullet: before asserting a fact, {rule}. Nothing else in the file changes, and the "
            "frontmatter block is untouched."
        ),
        "risks": [
            "The rule may be too broad and could make the agent add ceremony to answers where "
            "nobody asked for a citation.",
            "One added bullet in a long file may simply be ignored, in which case the failure "
            "recurs and the change was pure cost.",
        ],
        "rollback": (
            "Delete the appended section, or restore the backup that `asi apply` wrote before "
            "the change and verify the hash matches the recorded previous_sha256."
        ),
        "addressed_criterion_ids": CRITERION_ID_RE.findall(user)[:3],
        "new_skill_text": new_text,
    }


def run_judge(request: dict[str, Any]) -> dict[str, Any]:
    user = str(request.get("user", ""))
    rule = ""
    found = REQUIRED_RULE_RE.search(user)
    if found:
        rule = found.group(1).strip().rstrip(".")
    option_a = fenced(user, "option-A") or ""
    option_b = fenced(user, "option-B") or ""
    has_a = bool(rule) and rule.lower() in option_a.lower()
    has_b = bool(rule) and rule.lower() in option_b.lower()
    if has_a == has_b:
        return {
            "schema": "asi.judgement.v1",
            "preferred": "tie",
            "rationale": (
                "Neither option states the required rule any more clearly than the other, so an "
                "honest tie beats a forced preference."
            ),
        }
    preferred = "A" if has_a else "B"
    return {
        "schema": "asi.judgement.v1",
        "preferred": preferred,
        "rationale": (
            f"Option {preferred} states the specific rule the observed failure needed "
            f"({rule!r}); the other option leaves it implicit."
        ),
    }


ROLES = {"scorer": run_scorer, "improver": run_improver, "judge": run_judge}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic offline fake model.")
    parser.add_argument("--role", required=True, choices=sorted(ROLES))
    args = parser.parse_args(argv)
    raw = sys.stdin.read()
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"fake model: request was not JSON: {exc}\n")
        return 2
    if not isinstance(request, dict):
        sys.stderr.write("fake model: request must be a JSON object\n")
        return 2

    # A record the harness is told to treat as unreachable. Exiting non-zero is how
    # a real command reports "you got no answer", and the loop must DEFER that one
    # record rather than failing the batch.
    if "[[DEFER]]" in str(request.get("user", "")):
        sys.stderr.write("fake model: simulated transient failure for this record\n")
        return 1

    handler = ROLES[args.role]
    response = handler(request)
    sys.stdout.write(json.dumps(response, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
