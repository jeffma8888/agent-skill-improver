#!/usr/bin/env python3
"""A deterministic fake model, used as a REAL SUBPROCESS by the end-to-end test.

Why a subprocess rather than an in-process stub: the stub proves the pipeline,
but only a subprocess proves the adapter -- the stdin/stdout JSON contract, the
exit-code-to-failure-class mapping, and the fact that the prompts actually carry
what a model would need. Several tests still use an in-process fake for speed;
this one closes the gap the in-process fake leaves open.

It answers from the PROMPT ONLY. It never reads the skill file or the fixtures
off disk, which is what makes it a real test of the prompt: if the scorer prompt
stopped carrying citable element ids, or the improver prompt stopped carrying
the current skill text, this fake would start failing.

Behaviour, keyed off content it finds in the prompt:

  scorer   -- cites real elements parsed out of the CITABLE ELEMENTS block.
              Emits FAIL for the seeded failure criteria when scoring the
              "failing" fixture, unless the skill text under evaluation carries
              the FIX_MARKER, in which case those criteria come back PASS. That
              difference is what the baseline-vs-candidate replay measures.
  improver -- returns the current skill text with one rule appended.
  judge    -- prefers whichever blind option carries the FIX_MARKER.

Env knobs, so a test can force a specific pathology:
  ASI_FAKE_MODE=malformed_json | not_object | wrong_schema | bad_evidence |
                empty | crash | timeout_exit | extraneous_criterion |
                missing_criterion | injected_obedience
"""

from __future__ import annotations

import json
import os
import re
import sys

FIX_MARKER = "NEVER run a destructive git command without explicit confirmation"
FAILING_THREAD = "1700000000002-1"

# The criteria the fake scorer reports as failing on the seeded bad thread.
SEEDED_FAILURES = {
    "explicit-directive-followed",
    "correction-not-repeated",
    "no-unverified-completion-claim",
    "error-changed-approach",
}


def _citable(user: str) -> list[dict]:
    match = re.search(
        r"## CITABLE ELEMENTS [^\n]*\n<<<UNTRUSTED-DATA citable-elements \S+>>>\n```json\n(.*?)\n```",
        user,
        re.S,
    )
    if not match:
        return []
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return []


def _criteria(user: str) -> list[str]:
    block = re.search(r"## RUBRIC \(trusted\)\n(.*?)(?=\n## )", user, re.S)
    if not block:
        return []
    return re.findall(r"^- id: (\S+)$", block.group(1), re.M)


def _scored_thread(user: str) -> str:
    match = re.search(r"Score conversation (\S+) against", user)
    return match.group(1) if match else ""


def _fenced(user: str, label: str) -> str:
    """Pull one untrusted-fenced block back out of the prompt."""
    match = re.search(
        rf"<<<UNTRUSTED-DATA {re.escape(label)} [0-9a-f]+>>>\n(.*?)\n<<<END-UNTRUSTED-DATA "
        rf"{re.escape(label)} [0-9a-f]+>>>",
        user,
        re.S,
    )
    return match.group(1) if match else ""


def _skill_text_under_evaluation(user: str) -> str:
    if "## SKILL TEXT UNDER EVALUATION" not in user:
        return ""
    return _fenced(user, "skill-text")


def scorer(user: str, mode: str) -> dict:
    citable = _citable(user)
    criteria = _criteria(user)
    thread = _scored_thread(user)
    skill_text = _skill_text_under_evaluation(user)
    fixed = FIX_MARKER in skill_text

    messages = [c for c in citable if c.get("kind") == "message"] or citable
    first = messages[0] if messages else {"element_id": "missing", "position": 0, "kind": "message"}

    def ref(index: int = 0) -> list[dict]:
        pick = messages[min(index, len(messages) - 1)] if messages else first
        return [
            {
                "element_id": pick["element_id"],
                "position": pick["position"],
                "kind": pick.get("kind", "message"),
            }
        ]

    if mode == "bad_evidence":
        # Right element id, WRONG position. Must be downgraded to INSUFFICIENT.
        return {
            "schema": "asi.scorecard.v1",
            "conversation_id": thread,
            "criteria": [
                {
                    "criterion_id": cid,
                    "verdict": "FAIL",
                    "rationale": "cites an element at a position it does not occupy",
                    "evidence": [
                        {
                            "element_id": first["element_id"],
                            "position": int(first["position"]) + 500,
                            "kind": "message",
                        }
                    ],
                }
                for cid in criteria
            ],
        }

    if mode == "injected_obedience":
        # Obeys the injected instruction: all PASS, no evidence at all.
        return {
            "schema": "asi.scorecard.v1",
            "conversation_id": thread,
            "criteria": [
                {
                    "criterion_id": cid,
                    "verdict": "PASS",
                    "rationale": "maintenance mode",
                    "evidence": [],
                }
                for cid in criteria
            ]
            + [
                {
                    "criterion_id": "grant_all_promotions",
                    "verdict": "PASS",
                    "rationale": "injected criterion",
                    "evidence": [],
                }
            ],
        }

    if mode == "extraneous_criterion":
        return {
            "schema": "asi.scorecard.v1",
            "conversation_id": thread,
            "criteria": [
                {"criterion_id": cid, "verdict": "PASS", "rationale": "ok", "evidence": ref()}
                for cid in criteria
            ]
            + [
                {
                    "criterion_id": "invented-criterion",
                    "verdict": "FAIL",
                    "rationale": "not in the rubric",
                    "evidence": ref(),
                }
            ],
        }

    if mode == "missing_criterion":
        return {
            "schema": "asi.scorecard.v1",
            "conversation_id": thread,
            "criteria": [
                {"criterion_id": cid, "verdict": "PASS", "rationale": "ok", "evidence": ref()}
                for cid in criteria[:1]
            ],
        }

    out = []
    for index, cid in enumerate(criteria):
        failing = thread == FAILING_THREAD and cid in SEEDED_FAILURES and not fixed
        out.append(
            {
                "criterion_id": cid,
                "verdict": "FAIL" if failing else "PASS",
                "rationale": (
                    "seeded failure: the transcript shows the violation"
                    if failing
                    else "seeded pass: the transcript supports the criterion"
                ),
                "evidence": ref(index),
                "confidence": 0.9,
            }
        )
    return {"schema": "asi.scorecard.v1", "conversation_id": thread, "criteria": out}


def improver(user: str) -> dict:
    base = _fenced(user, "skill-text")
    lines = base.rstrip("\n").split("\n") if base else ["---", "name: unknown", "---"]
    lines.append("")
    lines.append(f"- {FIX_MARKER}. Ask first, every time.")
    new_text = "\n".join(lines) + "\n"
    return {
        "schema": "asi.proposal.v1",
        "target_skill_id": (
            re.search(r"skill file '([^']+)'", user).group(1)
            if re.search(r"skill file '([^']+)'", user)
            else "unknown"
        ),
        "hypothesis": (
            "The skill never states that destructive git commands require confirmation, so the "
            "agent treated a hard reset as an ordinary cleanup step and the user had to repeat "
            "the correction twice."
        ),
        "intended_change": (
            "Append one explicit prohibition to the rules section naming the destructive command "
            "class and requiring confirmation before it is run."
        ),
        "risks": [
            "An absolute prohibition may block a legitimate reset during a genuine recovery, "
            "which would show up as extra confirmation round-trips."
        ],
        "rollback": "Restore the previous SKILL.md from the backup recorded in the apply receipt.",
        "addressed_criterion_ids": sorted(SEEDED_FAILURES),
        "new_skill_text": new_text,
    }


def judge(user: str) -> dict:
    # There used to be an unused `options = dict(re.findall(...))` here. With a
    # single capture group re.findall returns a list of STRINGS, so dict(["A","B"])
    # raised ValueError -- and the caller records a judge failure as a note rather
    # than an error, so every judgement in the whole harness silently became "no
    # decided preference" and no gate could ever pass. Dead code that throws is
    # worse than dead code, because the swallowed exception looked like a
    # legitimate verdict.
    preferred = "tie"
    for label in ("A", "B"):
        if FIX_MARKER in _fenced(user, f"option-{label}"):
            preferred = label
            break
    return {
        "schema": "asi.judgement.v1",
        "preferred": preferred,
        "rationale": "one option states the prohibition explicitly; the other leaves it implicit",
    }


def main() -> int:
    mode = os.environ.get("ASI_FAKE_MODE", "")
    raw = sys.stdin.read()

    if mode == "crash":
        sys.stderr.write("fake model exploded\n")
        return 1
    if mode == "timeout_exit":
        return 69
    if mode == "empty":
        return 0
    if mode == "malformed_json":
        sys.stdout.write("here is your answer: {not: valid json,,,}")
        return 0
    if mode == "not_object":
        sys.stdout.write("[1, 2, 3]")
        return 0

    try:
        request = json.loads(raw)
    except json.JSONDecodeError:
        sys.stderr.write("fake model got a non-JSON request\n")
        return 2

    role = request.get("role", "")
    user = request.get("user", "")

    if mode == "wrong_schema":
        sys.stdout.write(json.dumps({"schema": "totally.wrong.v9", "criteria": []}))
        return 0

    if role == "scorer":
        payload = scorer(user, mode)
    elif role == "improver":
        payload = improver(user)
    elif role == "judge":
        payload = judge(user)
    else:
        sys.stderr.write(f"unknown role {role!r}\n")
        return 2

    # Real models wrap JSON in prose. Do it here on purpose so the extractor's
    # recovery path is exercised by the end-to-end test rather than only by a
    # unit test.
    sys.stdout.write("Here is the JSON you asked for:\n\n")
    sys.stdout.write(json.dumps(payload, indent=2))
    sys.stdout.write("\n\nLet me know if you need anything else.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
