"""Prove the pinned stack works on real Apple Silicon.

The PR's claim is that a fresh install ends with a usable MLX stack rather than a
chat-only Studio. Two halves: the packages import and compute on Metal, and
Studio's own health check calls the result usable. The second half is the one
that flips chat_only, so it is read from the shipped module, not restated here.
"""

from __future__ import annotations

import importlib.metadata as md
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

failures = []

print("versions:")
for name in ("mlx", "mlx-metal", "mlx-lm", "mlx-vlm"):
    try:
        print(f"  {name}: {md.version(name)}")
    except Exception as exc:
        failures.append(f"{name} is not installed: {exc}")
        print(f"  {name}: MISSING ({exc})")

print("\nimports:")
for mod in ("mlx.core", "mlx_lm", "mlx_lm.sample_utils", "mlx_vlm"):
    try:
        __import__(mod)
        print(f"  {mod}: ok")
    except Exception as exc:
        failures.append(f"import {mod} failed: {type(exc).__name__}: {exc}")
        print(f"  {mod}: FAILED {type(exc).__name__}: {exc}")

# An import that succeeds still proves nothing about Metal. Run something.
try:
    import mlx.core as mx

    a = mx.random.normal((256, 256))
    b = mx.random.normal((256, 256))
    c = (a @ b).sum()
    mx.eval(c)
    print(f"\nmetal matmul ok: {float(c):.4f}, default device = {mx.default_device()}")
except Exception as exc:
    failures.append(f"MLX compute failed: {type(exc).__name__}: {exc}")
    print(f"\nmetal matmul FAILED: {type(exc).__name__}: {exc}")

# Studio's own verdict. This is what gates Train/Export.
sys.path.insert(0, str(ROOT / "studio"))
try:
    from backend.utils import mlx_repair

    blockers = mlx_repair.mlx_stack_blockers()
    available = mlx_repair.mlx_stack_available()
    print(f"\nmlx_stack_blockers(): {blockers}")
    print(f"mlx_stack_available(): {available}")
    if blockers or not available:
        failures.append(f"Studio would still be chat-only: {blockers}")

    print(f"\n_MLX_INSTALL_SPECS: {mlx_repair._MLX_INSTALL_SPECS}")
    print(f"_MLX_MIN_VERSIONS:  {mlx_repair._MLX_MIN_VERSIONS}")
except Exception as exc:
    failures.append(f"could not read Studio's MLX verdict: {type(exc).__name__}: {exc}")
    print(f"\nmlx_repair FAILED: {type(exc).__name__}: {exc}")

if failures:
    print("\nFAILURES:")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("\nOK")
