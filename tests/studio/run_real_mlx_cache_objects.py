# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Snapshot copying against the real mlx-vlm cache classes, on real MLX arrays.

The unit suite copies fakes. A fake array cannot show whether ``value + 0`` gives the
snapshot its own storage, and a fake rotating cache cannot show what happens when the
ring wraps under a resumed copy. Both are what the store depends on, so both are checked
here, on the actual classes mlx-vlm builds, in seconds and with no model.

Usage:  python tests/studio/run_real_mlx_cache_objects.py [--json PATH]
"""

import argparse
import json
import platform
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "studio" / "backend"))

HEAD = 1
DIM = 64
CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def preflight():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit(f"needs Apple Silicon, got {platform.system()}/{platform.machine()}")
    import mlx.core as mx
    if getattr(mx, "__file__", None) is None:
        raise SystemExit("mlx resolved to a stub, not the real package")
    import mlx_vlm
    return {
        "mlx": getattr(mx, "__version__", "?"),
        "mlx_vlm": getattr(mlx_vlm, "__version__", "?"),
    }


def freeze(value):
    """A structural fingerprint: arrays by dtype/shape/bytes, everything else by value.

    Deliberately independent of the module under test -- comparing two caches with the
    production traversal would hide a traversal that misses a field.
    """
    import mlx.core as mx
    import numpy as np
    if isinstance(value, mx.array):
        mx.eval(value)
        array = np.asarray(value)
        return ("array", str(array.dtype), array.shape, array.tobytes())
    if isinstance(value, dict):
        return ("dict", tuple((str(k), freeze(v)) for k, v in sorted(value.items(), key = lambda i: str(i[0]))))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(freeze(item) for item in value))
    if hasattr(value, "__dict__"):
        return (type(value).__name__, freeze(vars(value)))
    return ("plain", repr(value))


def advance(cache, start, rows):
    """Feed ``rows`` deterministic rows and return what the cache hands back."""
    import mlx.core as mx
    keys = mx.arange(start * DIM, (start + rows) * DIM, dtype = mx.float32)
    keys = keys.reshape(1, HEAD, rows, DIM)
    result = cache.update_and_fetch(keys, keys + 1000.0)
    mx.eval(result)
    return freeze(result)


def arrays_alive(entry):
    """Whether any array is still reachable, for the release check."""
    import mlx.core as mx
    stack, found = [entry], False
    while stack and not found:
        item = stack.pop()
        if isinstance(item, mx.array):
            found = True
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif hasattr(item, "__dict__"):
            stack.extend(vars(item).values())
    return found


@check
def plain_kv_snapshots_own_their_storage(mod):
    from mlx_vlm.models.cache import KVCache
    live = KVCache()
    advance(live, 0, 256)
    saved = mod.copy_cache_entries([live])
    before = freeze(saved[0])
    advance(live, 256, 256)
    assert freeze(saved[0]) == before, "advancing the live cache rewrote the snapshot"
    assert saved[0] is not live and saved[0].keys is not live.keys


@check
def rotating_caches_resume_identically_across_the_wrap(mod):
    """The hardest layout: a ring whose index and retained rows both move. A resumed
    copy must answer exactly as a cache that was never snapshotted."""
    from mlx_vlm.models.cache import RotatingKVCache
    for window in (255, 256, 257, 1024):
        for keep in (0, 4):
            live = RotatingKVCache(max_size = window, keep = keep)
            reference = RotatingKVCache(max_size = window, keep = keep)
            position = 0
            for rows in (256, 256):
                assert advance(live, position, rows) == advance(reference, position, rows)
                position += rows

            saved = mod.copy_cache_entries([live])[0]
            stored = freeze(saved)
            advance(live, position, 1)                       # the live one moves on
            assert freeze(saved) == stored, (
                f"window={window} keep={keep}: the stored snapshot followed the live cache"
            )

            resumed = mod.copy_cache_entries([saved])[0]
            for rows in [1] * (window + 5) + [256]:
                assert advance(resumed, position, rows) == advance(reference, position, rows), (
                    f"window={window} keep={keep} at row {position}: a resumed rotating "
                    f"cache diverged from one that was never snapshotted"
                )
                assert freeze(saved) == stored
                position += rows


@check
def quantized_caches_survive_the_round_trip(mod):
    from mlx_vlm.models.cache import QuantizedKVCache
    live = QuantizedKVCache(group_size = 64, bits = 8)
    advance(live, 0, 256)
    saved = mod.copy_cache_entries([live])
    before = freeze(saved[0])
    advance(live, 256, 256)
    assert freeze(saved[0]) == before, "the quantized snapshot followed the live cache"
    resumed = mod.copy_cache_entries(saved)[0]
    assert freeze(resumed) == before


@check
def nested_cache_lists_are_copied_through(mod):
    from mlx_vlm.models.cache import CacheList, KVCache
    inner = [KVCache(), KVCache()]
    for entry in inner:
        advance(entry, 0, 256)
    live = CacheList(*inner)
    saved = mod.copy_cache_entries([live])[0]
    before = freeze(saved)
    for entry in inner:
        advance(entry, 256, 256)
    assert freeze(saved) == before, "a nested entry was shared, not copied"


@check
def arrays_caches_are_copied_through(mod):
    import mlx.core as mx
    from mlx_vlm.models.cache import ArraysCache
    live = ArraysCache(2)
    live[0] = mx.arange(64, dtype = mx.float32)
    live[1] = mx.arange(64, dtype = mx.float32) + 1
    saved = mod.copy_cache_entries([live])[0]
    before = freeze(saved)
    live[0] = mx.zeros((64,), dtype = mx.float32)
    assert freeze(saved) == before, "the ArraysCache slot was shared"


@check
def release_frees_every_array(mod):
    from mlx_vlm.models.cache import CacheList, KVCache
    inner = KVCache()
    advance(inner, 0, 256)
    entries = [CacheList(inner)]
    mod.release_cache_entries(entries)
    assert not arrays_alive(entries[0]), "arrays survived the release"


@check
def nbytes_is_measured_not_guessed(mod):
    from mlx_vlm.models.cache import KVCache
    live = KVCache()
    advance(live, 0, 256)
    counted = mod.cache_entries_nbytes([live])
    expected = live.keys.nbytes + live.values.nbytes
    assert counted == expected, f"counted {counted}, arrays hold {expected}"


@check
def offset_is_read_off_the_real_classes(mod):
    from mlx_vlm.models.cache import CacheList, KVCache
    live = KVCache()
    advance(live, 0, 256)
    assert mod.cache_entries_offset([live]) == 256
    assert mod.cache_entries_offset([CacheList(live)]) == 256


@check
def mlx_vlms_own_prefix_trim_does_not_corrupt_a_served_snapshot(mod):
    """Before v0.6.9, ``stream_generate`` trims the cache it is handed by raw slicing:

        cached_len = c.keys.shape[2]
        if cached_len > prefix_len:
            c.keys = c.keys[:, :, :prefix_len, :]; c.values = ...; c.offset = prefix_len

    ``keys.shape[2]`` is the ALLOCATED width, not the row count, so a snapshot whose
    offset is not a multiple of the allocation step (the media block's 647-row origin is
    the case that reaches this) goes through that branch. The store's default install is
    inside that window: unsloth_zoo resolves mlx-vlm 0.6.4 under its transformers cap.
    This replays the trim on a served snapshot and requires the trimmed cache to answer
    exactly as one that reached the same rows without ever being snapshotted.
    """
    from mlx_vlm.models.cache import KVCache, RotatingKVCache

    for factory in (
        lambda: KVCache(),
        lambda: RotatingKVCache(max_size = 1024, keep = 0),
        lambda: RotatingKVCache(max_size = 1024, keep = 4),
        lambda: RotatingKVCache(max_size = 512, keep = 0),
    ):
        for rows in (647, 903, 1159):
            live, reference = factory(), factory()
            assert advance(live, 0, rows) == advance(reference, 0, rows)
            saved = mod.copy_cache_entries([live])[0]

            width = saved.keys.shape[2]
            if width > rows:                       # the branch mlx-vlm would take
                saved.keys = saved.keys[:, :, :rows, :]
                saved.values = saved.values[:, :, :rows, :]
                saved.offset = rows

            for step in (1, 1, 256):
                assert advance(saved, rows, step) == advance(reference, rows, step), (
                    f"{type(live).__name__} at {rows} rows (allocated {width}): the "
                    f"trim mlx-vlm applies to a served snapshot changed its answers"
                )
                rows += step


@check
def every_cache_class_mlx_vlm_exports_is_copyable_or_refused(mod):
    """Forward compatibility: a layout the copier does not recognise must raise, not be
    silently shared. This enumerates what mlx-vlm actually ships today."""
    from mlx_vlm.models import cache as vlm_cache
    shared = []
    for name in dir(vlm_cache):
        cls = getattr(vlm_cache, name)
        if not isinstance(cls, type) or not hasattr(cls, "state"):
            continue
        if not hasattr(cls, "__dict__") or "__slots__" in getattr(cls, "__dict__", {}):
            shared.append(name)
    assert not shared, (
        f"these cache classes use __slots__, so the copier shares them instead of "
        f"copying them when they appear nested: {shared}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default = None)
    args = parser.parse_args()

    versions = preflight()
    print(f"[preflight] {json.dumps(versions)}", flush = True)

    from core.inference import mlx_inference as mod

    results, failed = [], 0
    for fn in CHECKS:
        try:
            fn(mod)
        except AssertionError as exc:
            results.append({"check": fn.__name__, "ok": False, "detail": str(exc)})
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}", flush = True)
        except Exception as exc:                       # a missing class is information too
            results.append({"check": fn.__name__, "ok": None, "detail": f"{type(exc).__name__}: {exc}"})
            print(f"SKIP {fn.__name__}: {type(exc).__name__}: {exc}", flush = True)
        else:
            results.append({"check": fn.__name__, "ok": True})
            print(f"ok   {fn.__name__}", flush = True)

    if args.json:
        Path(args.json).write_text(
            json.dumps({"versions": versions, "results": results}, indent = 2), encoding = "utf-8",
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
