"""Run the installer's MLX decision on a REAL host and report what it decided.

Nothing here tells the installer what platform it is on. IS_MAC_ARM, NO_TORCH and
the rest are whatever the running machine produces, so the arm64 / Intel / Linux /
Windows legs of the matrix are the assertion. Only the calls that would touch the
machine are replaced.

EXPECT_MLX=1 means the pinned MLX stack must be installed on BOTH the fresh
(SKIP_STUDIO_BASE=1) and update paths; 0 means on neither.
"""

from __future__ import annotations

import contextlib
import io
import os
import platform as _platform
import sys
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "studio"))

import install_python_stack as stack  # noqa: E402

EXPECTED_SPECS = {
    "mlx==0.32.1",
    "mlx-metal==0.32.1",
    "mlx-lm==0.31.3",
    "mlx-vlm>=0.4.4,<0.7.0",
}


def probe(skip_base: bool) -> dict:
    installs: list[tuple[str, tuple]] = []
    steps: list[str] = []

    def fake_progress(label):
        steps.append(label)
        stack._STEP += 1

    stack._progress = fake_progress
    stack.pip_install = lambda label, *a, **k: installs.append((label, a))
    # pip_install_try is a second entry point that shells out to uv; a bare runner has
    # no uv, and leaving it live also means the probe really installs things.
    stack.pip_install_try = lambda label, *a, **k: True
    stack.run = lambda label, *a, **k: True
    for name, value in {
        "_bootstrap_uv": True,
        "_shared_base_requirements": None,
        "_repair_duplicate_core_metadata": True,
        "_repair_damaged_core_payload": True,
        "_bitsandbytes_installed": False,
        "_has_usable_nvidia_gpu": False,
        "_ensure_cuda_torch": None,
        "_ensure_rocm_torch": None,
        "_ensure_xpu_torch": None,
        "_ensure_cpu_torch": None,
        "_ensure_xpu_triton": None,
        "_ensure_flash_attn": None,
        "_repair_bad_anyio": None,
        "_has_working_git": True,
        "_probe_installed_torch_version": "2.10.0",
        "_installed_distribution_version": "2.10.0",
        "_exact_distribution_spec_is_installed": False,
        "_expected_torch_flavor_tag": "",
        "_ensure_expected_torch_flavor": True,
        "_torchcodec_spec_is_installable": True,
        "_torchcodec_index_url": None,
        "_installed_torch_is_windows_rocm": False,
        "_note": None,
    }.items():
        if hasattr(stack, name):
            setattr(stack, name, Mock(return_value=value))
    stack.install_manifest.remove_manifest = Mock(return_value=True)
    stack.install_manifest.set_no_torch_marker = Mock()

    os.environ["SKIP_STUDIO_BASE"] = "1" if skip_base else "0"
    for key in ("STUDIO_LOCAL_REPO", "STUDIO_PACKAGE_NAME", "UNSLOTH_CI_SOURCE_OVERLAY"):
        os.environ.pop(key, None)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = stack.install_python_stack()

    mlx = [(label, args) for label, args in installs if label.startswith("Installing MLX")]
    return {"rc": rc, "steps": steps, "total": stack._TOTAL, "mlx": mlx}


def main() -> int:
    if os.environ.get("FORCE_NO_TORCH") == "1":
        stack.NO_TORCH = True

    expect = os.environ.get("EXPECT_MLX", "0") == "1"
    print(
        f"host: sys.platform={sys.platform} machine={_platform.machine()} "
        f"mac_ver={_platform.mac_ver()[0]!r} python={sys.version_info.major}.{sys.version_info.minor}"
    )
    print(
        f"installer: IS_MAC_ARM={stack.IS_MAC_ARM} IS_MACOS={stack.IS_MACOS} "
        f"IS_WINDOWS={stack.IS_WINDOWS} NO_TORCH={stack.NO_TORCH}"
    )
    if hasattr(stack, "_mlx_pins_are_installable"):
        print(f"installer: _mlx_pins_are_installable()={stack._mlx_pins_are_installable()}")

    failures = []
    for skip_base in (True, False):
        mode = "fresh (SKIP_STUDIO_BASE=1)" if skip_base else "update"
        result = probe(skip_base)
        got = len(result["mlx"])
        print(f"\n[{mode}] rc={result['rc']} steps={len(result['steps'])} _TOTAL={result['total']}")
        print(f"[{mode}] MLX install calls: {got}")
        for label, args in result["mlx"]:
            print(f"    {label}: {list(args)}")
        for step in result["steps"]:
            if step.startswith("MLX"):
                print(f"    step: {step}")

        if result["rc"] != 0:
            failures.append(f"{mode}: install returned {result['rc']}")
        if len(result["steps"]) != result["total"]:
            failures.append(
                f"{mode}: progress budget {result['total']} but {len(result['steps'])} steps emitted"
            )
        if got != int(expect):
            failures.append(f"{mode}: expected {int(expect)} MLX installs, got {got}")
        if expect and got:
            specs = {a for _, args in result["mlx"] for a in args if not a.startswith("-")}
            if specs != EXPECTED_SPECS:
                failures.append(f"{mode}: pins {sorted(specs)} != {sorted(EXPECTED_SPECS)}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
