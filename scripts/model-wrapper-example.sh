#!/usr/bin/env bash
# Example model wrapper. THIS IS A TEMPLATE -- it does not call a real model.
#
# The contract is deliberately the smallest useful one:
#   stdin  : one JSON object {schema, role, response_schema, system, user}
#   stdout : one JSON object matching the requested response_schema
#   exit 0 : we got an answer.  non-zero: we did not (the record is DEFERRED,
#            not failed -- "unreachable" is not a verdict).
#
# Keep credentials in here (or in the environment this is launched with), never
# in the repo's config. Copy this file, wire it to your provider, and point
# [model.<role>].argv at your copy.
#
# A worked sketch against a hypothetical CLI that takes a prompt on stdin:
#
#   req=$(cat)
#   system=$(printf '%s' "$req" | python3 -c 'import json,sys;print(json.load(sys.stdin)["system"])')
#   user=$(printf '%s' "$req"   | python3 -c 'import json,sys;print(json.load(sys.stdin)["user"])')
#   your-model-cli --system "$system" --json <<<"$user"
#
# Two things worth doing in a real wrapper:
#   1. ask the provider for JSON mode / a response schema if it has one;
#   2. do NOT retry internally on a content error. Let a malformed answer come
#      back so the loop can quarantine that one record with a reason, instead of
#      hiding it behind a retry.

set -euo pipefail
role="${1:-unknown}"
cat >/dev/null   # consume the request so a caller does not see EPIPE

cat >&2 <<MSG
scripts/model-wrapper-example.sh is a TEMPLATE and is not wired to a model.
Role requested: ${role}
Copy it, point it at your provider, and set [model.${role}].argv in your config.
MSG
exit 69   # EX_UNAVAILABLE -> TransientFailure -> the record is deferred, not failed
