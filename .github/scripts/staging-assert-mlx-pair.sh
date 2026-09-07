#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# DISPOSABLE staging-only helper. Does not ship upstream.
#
# Judge one venv the way Studio judges it: the pinned transformers must accept the
# installed tokenizers, the MLX imports must load, and the gate that decides
# chat-only must find nothing to complain about.
#
# Usage: staging-assert-mlx-pair.sh /path/to/python
set -uo pipefail

PY="${1:?usage: staging-assert-mlx-pair.sh <python>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FAILED=0

echo "== installed versions =="
"$PY" - <<'PROBE'
from importlib.metadata import PackageNotFoundError, version
for name in ("transformers", "tokenizers", "mlx", "mlx-lm", "mlx-vlm", "torch"):
    try:
        print(f"  {name}=={version(name)}")
    except PackageNotFoundError:
        print(f"  {name} MISSING")
PROBE

echo "== the pinned transformers must accept the installed tokenizers =="
# Read the requirement out of transformers' own metadata rather than hardcoding a
# window here, so this keeps judging correctly after either pin moves.
if ! "$PY" - <<'PROBE'
import sys
from importlib.metadata import PackageNotFoundError, requires, version

from packaging.requirements import Requirement

try:
    tf, tok = version("transformers"), version("tokenizers")
except PackageNotFoundError as exc:
    print(f"  FAIL: {exc}")
    sys.exit(1)
specs = [
    Requirement(raw)
    for raw in (requires("transformers") or [])
    if Requirement(raw).name == "tokenizers" and Requirement(raw).marker is None
]
if not specs:
    print("  FAIL: transformers declares no unconditional tokenizers requirement")
    sys.exit(1)
window = specs[0].specifier
if tok in window:
    print(f"  OK: transformers=={tf} wants tokenizers{window}, found {tok}")
    sys.exit(0)
print(f"  FAIL: transformers=={tf} wants tokenizers{window}, found {tok}")
sys.exit(1)
PROBE
then
    FAILED=1
fi

echo "== the MLX imports must load =="
# Each in its own interpreter: the first failure would otherwise hide the rest,
# and a half-built native extension can abort rather than raise.
for module in transformers mlx.core mlx_lm mlx_vlm; do
    if out=$("$PY" -c "import ${module}" 2>&1); then
        echo "  OK: import ${module}"
    else
        echo "  FAIL: import ${module}"
        printf '%s\n' "$out" | tail -20 | sed 's/^/      /'
        FAILED=1
    fi
done

echo "== the gate that decides chat-only must find nothing =="
if ! PYTHONPATH="$REPO_ROOT/studio/backend" "$PY" - <<'PROBE'
import json
import sys

from utils.mlx_repair import mlx_stack_blockers

blockers = mlx_stack_blockers()
print("  blockers:", json.dumps(blockers))
sys.exit(1 if blockers else 0)
PROBE
then
    FAILED=1
fi

if [ "$FAILED" -ne 0 ]; then
    echo "::error::this venv would come up chat-only: Train and Export disabled"
    exit 1
fi
echo "this venv can train: transformers/tokenizers agree and the MLX stack imports"
