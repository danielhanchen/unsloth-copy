#!/usr/bin/env python3
"""Run the shipped path resolver from three revisions against identical fixtures, and diff.

Answers one question: does an existing install's data move when it upgrades? Not by reading a
diff, which is how the last several regressions in this branch got in, but by executing the real
`storage_roots` from each revision in its own subprocess and comparing what it says.

A USER-DATA path that moves is a failure. A regenerable cache that moves is the point of the
change, so it is printed and allowed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
OUT = REPO / "routing-out"

TREES = {
    "OLD_v0.1.701": REPO.parent / "oldrel",
    "BASE_main": REPO.parent / "basemain",
    "HEAD_pr": REPO,
}

# Losing or moving any of these is a broken upgrade: they hold what the user made.
USER_DATA = {
    "studio_root", "studio_db_path", "auth_root", "auth_db_path",
    "assets_root", "datasets_root", "dataset_uploads_root", "recipe_datasets_root",
    "outputs_root", "exports_root", "rag_root", "rag_db_path", "rag_uploads_root",
    "project_workspaces_root", "documents_root", "tensorboard_root", "studio_bin_root",
    "seed_uploads_root", "unstructured_uploads_root",
}

CHILD = r'''
import json, os, sys
sys.path.insert(0, os.environ["_PROBE_BACKEND"])
from utils.paths import storage_roots as sr
before = dict(os.environ)
out = {"paths": {}, "errors": {}}
SKIP = {"ensure_dir", "ensure_studio_directories", "setup_cache_env"}
for name in sorted(dir(sr)):
    if name.startswith("_") or name in SKIP:
        continue
    fn = getattr(sr, name)
    if not callable(fn) or not hasattr(fn, "__code__"):
        continue
    if fn.__code__.co_argcount or getattr(fn, "__module__", "") != sr.__name__:
        continue
    try:
        val = fn()
    except Exception as exc:
        out["errors"][name] = f"{type(exc).__name__}: {exc}"
        continue
    if isinstance(val, (list, tuple)):
        out["paths"][name] = "|".join(str(v) for v in val)
    elif val is None or isinstance(val, bool):
        continue
    else:
        out["paths"][name] = str(val)
sr.setup_cache_env()
out["env_delta"] = {k: v for k, v in sorted(os.environ.items()) if before.get(k) != v}
print("---PROBE---")
print(json.dumps(out))
'''

# Every install shape that exists in the wild, plus the two the PR adds.
SCENARIOS = {
    "default_fresh": {},
    "legacy_unsloth_dir_present": {
        "mkdir": [".unsloth/studio/cache", ".unsloth/studio/assets", ".cache/huggingface/hub"],
        "write": {".unsloth/studio/studio.db": "sqlite", ".unsloth/studio/auth/auth.db": "sqlite"},
    },
    "desktop_app_shape_v0_1_701": {
        "mkdir": [".unsloth/studio/unsloth_studio/bin", ".unsloth/studio/share",
                  ".unsloth/llama.cpp", ".unsloth/node", ".local/bin"],
        "write": {".unsloth/studio/share/studio.conf": "UNSLOTH_EXE=x\n",
                  ".unsloth/studio/studio.db": "sqlite"},
    },
    "legacy_studio_home_env": {"env": {"UNSLOTH_STUDIO_HOME": "$HOME/oldstudio"},
                               "mkdir": ["oldstudio/cache"]},
    "legacy_studio_home_alias": {"env": {"STUDIO_HOME": "$HOME/oldstudio2"},
                               "mkdir": ["oldstudio2/cache"]},
    "user_set_hf_home": {"env": {"HF_HOME": "$HOME/myhf"}, "mkdir": ["myhf/hub"]},
    "warm_caches_everywhere": {
        "mkdir": [".triton/cache", ".cache/torch_extensions", ".nv/ComputeCache",
                  ".cache/numba", ".data-designer"],
        "write": {".config/matplotlib/matplotlibrc": "backend: Agg\n"},
    },
    "blank_overrides": {"env": {"UNSLOTH_HOME": "", "UNSLOTH_STUDIO_HOME": "", "HF_HOME": ""}},
}


def _fold_home(payload: dict, home: Path) -> dict:
    """Replace the per-revision synthetic HOME with a token, whatever case the host reports."""
    blob = json.dumps(payload)
    for form in (str(home).replace("\\", "\\\\"), str(home)):
        if not form:
            continue
        pattern = re.compile(re.escape(form), re.IGNORECASE)
        blob = pattern.sub(lambda _m: "$HOME", blob)
    # Separators too: the same directory is D:\\a\\x here and D:/a/x elsewhere.
    return json.loads(blob.replace("\\\\", "/"))


def probe(tree: Path, home: Path, extra_env: dict) -> dict:
    backend = tree / "studio" / "backend"
    env = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PATH": os.environ["PATH"],
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "_PROBE_BACKEND": str(backend),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.update(extra_env)
    proc = subprocess.run([sys.executable, "-c", CHILD], env = env, cwd = str(backend),
                          capture_output = True, text = True)
    if "---PROBE---" not in proc.stdout:
        return {"FATAL": (proc.stdout + proc.stderr)[-2500:]}
    return json.loads(proc.stdout.split("---PROBE---", 1)[1])


def main() -> int:
    OUT.mkdir(exist_ok = True)
    for name, tree in TREES.items():
        if not (tree / "studio" / "backend" / "utils" / "paths" / "storage_roots.py").exists():
            print(f"FATAL: {name} is missing storage_roots.py at {tree}")
            return 1

    results: dict = {}
    for scenario, spec in SCENARIOS.items():
        results[scenario] = {}
        for rev, tree in TREES.items():
            home = OUT / scenario / rev.replace(".", "_") / "home"
            home.mkdir(parents = True, exist_ok = True)
            for rel in spec.get("mkdir", []):
                (home / rel).mkdir(parents = True, exist_ok = True)
            for rel, body in spec.get("write", {}).items():
                path = home / rel
                path.parent.mkdir(parents = True, exist_ok = True)
                path.write_text(body)
            env = {k: v.replace("$HOME", str(home)) for k, v in spec.get("env", {}).items()}
            got = probe(tree, home, env)
            # The synthetic HOME differs per revision, so it is folded to a token before any
            # comparison; otherwise every path would "differ".
            #
            # Case-insensitively, and on the JSON-escaped form as well as the plain one. On
            # Windows Path.resolve() returns the directory's REAL on-disk case, which is not
            # necessarily the case this script created it with, and a fold that missed made
            # every path in that scenario read as moved. That is a measurement bug that looks
            # exactly like the regression this job exists to catch, so it is worth the care.
            results[scenario][rev] = _fold_home(got, home)

    (OUT / "routing.json").write_text(json.dumps(results, indent = 1, sort_keys = True))

    failures = []
    print(f"\n{'scenario':<32} {'user data moved OLD->HEAD':<28} caches moved")
    print("-" * 96)
    for scenario, revs in results.items():
        old = revs["OLD_v0.1.701"]
        head = revs["HEAD_pr"]
        if "FATAL" in old or "FATAL" in head:
            failures.append(f"{scenario}: probe failed: {old.get('FATAL') or head.get('FATAL')}")
            print(f"  {scenario:<30} PROBE FAILED")
            continue
        po, ph = old["paths"], head["paths"]
        moved_user = sorted(k for k in po if k in ph and po[k] != ph[k] and k in USER_DATA)
        moved_cache = sorted(k for k in po if k in ph and po[k] != ph[k] and k not in USER_DATA)
        print(f"  {scenario:<30} {str(moved_user or 'NONE'):<28} {moved_cache or 'none'}")
        for key in moved_user:
            failures.append(f"{scenario}: {key}\n      OLD  {po[key]}\n      HEAD {ph[key]}")

    print("\n=== env the PR newly sets, on a default install ===")
    d = results["default_fresh"]
    added = sorted(set(d["HEAD_pr"]["env_delta"]) - set(d["OLD_v0.1.701"]["env_delta"]))
    for key in added:
        print(f"  + {key} = {d['HEAD_pr']['env_delta'][key]}")
    changed = {
        k: (d["OLD_v0.1.701"]["env_delta"][k], d["HEAD_pr"]["env_delta"][k])
        for k in d["OLD_v0.1.701"]["env_delta"]
        if k in d["HEAD_pr"]["env_delta"]
        and d["OLD_v0.1.701"]["env_delta"][k] != d["HEAD_pr"]["env_delta"][k]
    }
    for key, (was, now) in sorted(changed.items()):
        print(f"  ~ {key}\n      OLD  {was}\n      HEAD {now}")
        # A changed value on a default install is only acceptable for a regenerable cache.
        if key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
            failures.append(f"default_fresh: {key} moved a DOWNLOADED cache: {was} -> {now}")

    if failures:
        print("\nFAIL: an upgrade moves data an existing install already has")
        for line in failures:
            print("   " + line)
        return 1
    print("\nPASS: no user-data path moves between v0.1.701-beta and this branch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
