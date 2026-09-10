#!/usr/bin/env python3
"""What a free-space reading costs, per platform.

PR #10528 replaced a 60 s poll of /api/system with an on-demand read of /api/system/disk, taken
when a download is about to start. That trade only holds if the read is genuinely cheap on every
platform Studio runs on, so this measures it rather than asserting it: shutil.disk_usage is
statvfs on Linux and macOS and GetDiskFreeSpaceExW on Windows, and those are not the same call.

Prints the numbers on every platform and fails only on a genuinely slow one, three orders of
magnitude above what it costs on a healthy host.
"""

from __future__ import annotations

import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

BUDGET_MS = 5.0
SAMPLES = 500


def main() -> int:
    probe = Path(tempfile.gettempdir())
    shutil.disk_usage(probe)  # warm: the first call on Windows loads the DLL entry point

    timings = []
    for _ in range(SAMPLES):
        started = time.perf_counter()
        shutil.disk_usage(probe)
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()

    usage = shutil.disk_usage(probe)
    print(f"platform : {platform.platform()}")
    print(f"probe    : {probe}")
    print(f"free     : {usage.free / 1e9:.1f} GB of {usage.total / 1e9:.1f} GB")
    print(f"median   : {statistics.median(timings) * 1000:.1f} us")
    print(f"p99      : {timings[int(SAMPLES * 0.99)] * 1000:.1f} us")
    print(f"max      : {timings[-1] * 1000:.1f} us")

    median_ms = statistics.median(timings)
    if median_ms > BUDGET_MS:
        print(f"\nFAIL: {median_ms:.2f} ms per reading is too slow to sit in front of a download")
        return 1
    print(f"\nPASS: a reading costs {median_ms * 1000:.1f} us, well inside the {BUDGET_MS} ms budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
