#!/usr/bin/env python3
"""Snapshot an Unsloth Studio install (venv + uv caches) and diff two snapshots.

    python snapshot.py take OUT --venv VENV --studio-home HOME [--cache DIR ...] [--label L]
    python snapshot.py diff BEFORE AFTER   -> prints JSON: package changes, new cache entries, sizes

Cache indexing understands uv's layout: archive-v0/<id>/ holds unpacked wheels (named by the
*.dist-info inside), wheels-v*/ holds metadata, sdists-v*/ + builds-v*/ hold sources/builds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def du(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def site_packages(venv: Path) -> Path | None:
    for pat in ("lib/python*/site-packages", "Lib/site-packages"):
        for m in venv.glob(pat):
            return m
    return None


def freeze(venv: Path) -> dict[str, str]:
    py = venv_python(venv)
    if not py.exists():
        return {}
    code = (
        "import importlib.metadata as m, json;"
        "print(json.dumps({d.metadata['Name'].lower().replace('_','-'): d.version for d in m.distributions()}))"
    )
    try:
        out = subprocess.run([str(py), "-I", "-c", code], capture_output=True, text=True, timeout=120)
        return json.loads(out.stdout or "{}")
    except Exception:  # noqa: BLE001
        return {}


def torch_info(venv: Path) -> dict:
    sp = site_packages(venv)
    info: dict = {"version": None, "wheel_hash": None, "dist_info": None}
    if not sp:
        return info
    for d in sp.glob("torch-*.dist-info"):
        info["dist_info"] = d.name
        ver = d / "METADATA"
        if ver.exists():
            for line in ver.read_text(errors="replace").splitlines():
                if line.startswith("Version:"):
                    info["version"] = line.split(":", 1)[1].strip()
                    break
        rec = d / "RECORD"
        if rec.exists():
            info["wheel_hash"] = hashlib.sha256(rec.read_bytes()).hexdigest()[:16]
        break
    vp = sp / "torch" / "version.py"
    if vp.exists():
        m = re.search(r"__version__ = '([^']+)'", vp.read_text(errors="replace"))
        if m:
            info["version"] = m.group(1)
        m = re.search(r"cuda = (None|'[^']*')", vp.read_text(errors="replace"))
        if m:
            info["cuda"] = m.group(1).strip("'")
    return info


def script_hashes(venv: Path) -> dict:
    sp = site_packages(venv)
    out = {}
    if not sp:
        return out
    for rel in ("studio/setup.sh", "studio/setup.ps1", "studio/install_python_stack.py", "unsloth_cli/commands/studio.py"):
        p = sp / rel
        if p.exists():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def index_cache(cache: Path) -> dict:
    """Return {entry: {"bytes": n, "name": pkg-ver or None}} for package-bearing buckets."""
    entries: dict[str, dict] = {}
    if not cache.exists():
        return entries
    for bucket in cache.iterdir():
        if not bucket.is_dir():
            continue
        b = bucket.name
        if b.startswith("archive-"):
            for ent in bucket.iterdir():
                if not ent.is_dir():
                    continue
                name = None
                for di in ent.glob("*.dist-info"):
                    name = di.name[: -len(".dist-info")]
                    break
                entries[f"{b}/{ent.name}"] = {"bytes": du(ent), "name": name}
        elif b.startswith(("wheels-", "sdists-", "builds-", "built-wheels-")):
            for root, _dirs, files in os.walk(bucket):
                for f in files:
                    p = Path(root) / f
                    try:
                        entries[str(p.relative_to(cache))] = {"bytes": p.stat().st_size, "name": None}
                    except OSError:
                        pass
    return entries


def take(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    venv = Path(args.venv)
    home = Path(args.studio_home)
    snap = {
        "label": args.label,
        "venv": str(venv),
        "venv_exists": venv.exists(),
        "venv_bytes": du(venv),
        "packages": freeze(venv),
        "torch": torch_info(venv),
        "scripts": script_hashes(venv),
        "caches": {},
        "marker": None,
        "prebuilt": {},
    }
    marker = home / "cache" / "uv-cache-dir"
    if marker.exists():
        snap["marker"] = marker.read_text(errors="replace").strip()
    caches = list(args.cache or [])
    caches.append(str(home / "cache" / "uv"))
    for c in caches:
        cp = Path(c).expanduser()
        idx = index_cache(cp)
        snap["caches"][str(cp)] = {"exists": cp.exists(), "bytes": sum(e["bytes"] for e in idx.values()), "entries": idx}
    for name in ("llama.cpp", "whisper.cpp", "node"):
        for cand in (home / name, home.parent / name):
            info = cand / "UNSLOTH_PREBUILT_INFO.json"
            if info.exists():
                try:
                    snap["prebuilt"][name] = json.loads(info.read_text())
                except Exception:  # noqa: BLE001
                    snap["prebuilt"][name] = {"raw": info.read_text(errors="replace")[:500]}
                break
    (out / "snapshot.json").write_text(json.dumps(snap, indent=1, sort_keys=True))
    (out / "freeze.txt").write_text("".join(f"{k}=={v}\n" for k, v in sorted(snap["packages"].items())))
    print(json.dumps({"label": args.label, "venv_bytes": snap["venv_bytes"], "torch": snap["torch"],
                      "packages": len(snap["packages"]), "marker": snap["marker"],
                      "caches": {k: v["bytes"] for k, v in snap["caches"].items()}}))


def diff(args: argparse.Namespace) -> None:
    a = json.loads((Path(args.before) / "snapshot.json").read_text())
    b = json.loads((Path(args.after) / "snapshot.json").read_text())
    pa, pb = a["packages"], b["packages"]
    changed = {k: [pa[k], pb[k]] for k in pa if k in pb and pa[k] != pb[k]}
    added = {k: pb[k] for k in pb if k not in pa}
    removed = {k: pa[k] for k in pa if k not in pb}
    new_cache: list[dict] = []
    cache_growth = {}
    for c, cb in b["caches"].items():
        ca = a["caches"].get(c, {"entries": {}, "bytes": 0})
        for e, meta in cb["entries"].items():
            if e not in ca["entries"]:
                new_cache.append({"cache": c, "entry": e, "bytes": meta["bytes"], "name": meta["name"]})
        cache_growth[c] = cb["bytes"] - ca["bytes"]
    new_cache.sort(key=lambda x: -x["bytes"])
    res = {
        "packages_changed": changed, "packages_added": added, "packages_removed": removed,
        "torch_before": a["torch"], "torch_after": b["torch"],
        "venv_bytes_before": a["venv_bytes"], "venv_bytes_after": b["venv_bytes"],
        "cache_growth_bytes": cache_growth,
        "new_cache_bytes": sum(x["bytes"] for x in new_cache),
        "new_cache_top": new_cache[:25],
        "new_cache_named": sorted({x["name"] for x in new_cache if x["name"]}),
        "marker_before": a["marker"], "marker_after": b["marker"],
        "scripts_before": a["scripts"], "scripts_after": b["scripts"],
        "prebuilt_before": {k: v.get("release_tag") or v.get("tag") for k, v in a["prebuilt"].items()},
        "prebuilt_after": {k: v.get("release_tag") or v.get("tag") for k, v in b["prebuilt"].items()},
    }
    text = json.dumps(res, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("take")
    t.add_argument("out")
    t.add_argument("--venv", required=True)
    t.add_argument("--studio-home", required=True)
    t.add_argument("--cache", action="append")
    t.add_argument("--label", default="")
    d = sub.add_parser("diff")
    d.add_argument("before")
    d.add_argument("after")
    d.add_argument("--out")
    a = ap.parse_args()
    take(a) if a.cmd == "take" else diff(a)


if __name__ == "__main__":
    sys.exit(main())
