#!/usr/bin/env python3
"""Idempotency evidence, fault injection and leftover-state fixtures for the update harness.

One implementation for both shells: run_step.sh and run_step.ps1 both call this, so a fault is
injected the same way on every OS and the same fields land in summary.json.

    python idem.py capture    --venv V --studio-home H --out state.json
    python idem.py compare    BEFORE.json AFTER.json [--out cmp.json]
    python idem.py inject     <kind> --venv V --studio-home H          # fault injection
    python idem.py steps      --log log.txt [--out steps.json]         # which update steps ran
    python idem.py leftovers  seed|check --studio-home H [--out l.json]
    python idem.py prefetch   --studio-home H [--out p.json]           # .update-prefetch marker state

`capture` records exactly what "the second update did nothing" means: the installed distributions,
the manifest (minus its timestamp), the sidecar venv directory mtimes, the prebuilt/cache markers
and the manifest-adjacent marker files. `compare` turns two captures into `idempotent` + reasons.
Stdlib only.
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

SIDECARS = (".venv_t5_530", ".venv_t5_550", ".venv_t5_510")
MANIFEST_NAME = "unsloth_install_manifest.json"
IGNORED_MANIFEST_KEYS = ("completed_at_ms",)
# The relaxed verdict, for the FIRST no-op after an upgrade (settle -> noop). New code writes
# these three on every pass it runs to completion -- `step_results` and `pip_check_ok` are the
# record of the pass itself -- so a second full pass legitimately rewrites them while changing
# nothing else. Everything outside this set still counts, and the strict verdict counts them all.
RELAXED_MANIFEST_KEYS = ("completed_at_ms", "step_results", "pip_check_ok")
# Same idea for the freeze: the upgrade can leave torchao on the index-neutral build, and the
# first pass by the new code repairs it to the +cpu / +cuXXX one. Only the local tag may move,
# and only for torchao: a different release number is a real change.
RELAXED_FREEZE_PACKAGES = ("torchao",)
# Where each marker actually lives: the uv-cache pointer sits in the Studio home, while the
# no-torch and ownership markers sit in sys.prefix (the venv) next to the manifest.
HOME_MARKERS = ("cache/uv-cache-dir",)
VENV_MARKERS = (".unsloth-no-torch", ".unsloth-studio-owned")
PREBUILT = ("llama.cpp", "whisper.cpp", "node")
PREFETCH_DIR = ".update-prefetch"
PREFETCH_MARKER = "PREFETCHED.json"

# The steps an update prints, as REGEXES over one log line. Their presence/absence is the
# fault-injection assertion: after breaking exactly one thing, exactly one of these must come back.
#
# Reinstall and validated-skip are separate keys. `installing prebuilt llama.cpp...` is printed
# before the pre-check runs, so it is only "the step ran" -- it appeared on every update including
# the idempotent offline one, which then prints "prebuilt up to date and validated" and touches
# nothing. Keying the reinstall off the attempt line reported work that never happened.
STEP_PATTERNS = {
    "sidecar_rebuilt": r"\bpre-installed\b",
    "sidecar_current": r"\bsidecar current\b",
    "step_skipped": r"\(satisfied, skipped\)",
    "prebuilt_already_matches": r"already matches",
    "llama_step_ran": r"installing prebuilt llama\.cpp",
    "llama_installed": r"llama\.cpp\s+(?:arm64 CPU )?prebuilt installed",
    "llama_validated": r"llama\.cpp\s+prebuilt up to date",
    "llama_kept": r"llama\.cpp\s+update unavailable|existing prebuilt kept",
    "whisper_installed": r"whisper\.cpp\s+prebuilt install(?:ed|ing)",
    "whisper_validated": r"whisper\.cpp\s+prebuilt up to date",
    "whisper_failed": r"whisper\.cpp\s+prebuilt install failed",
}

# Which route through setup.sh / setup.ps1 the run took, in the order the log settles them. The
# distinction matters because the harness relabels the PR wheel `<version>.postN`: setup compares
# the INSTALLED version string with what PyPI reports as latest, so `2026.9.3.post1` never equals
# `2026.9.3` and a no-op online run can NEVER reach the "is up to date" fast path a real user hits.
# Online no-ops therefore measure a full dependency pass whose steps all skip; only a run whose
# PyPI probe fails (the offline one) can reach a fast path. Recording which is which keeps that
# visible in the report instead of turning up as an unexplained 40 s no-op.
UPDATE_PATH_REASONS = (
    ("fast_path", "installed == PyPI latest", r"is up to date"),
    ("fast_path", "PyPI unreachable + UV_OFFLINE, verified install kept",
     r"PyPI is unreachable and UV_OFFLINE is set"),
    ("deps_pass", "PyPI reports a different latest version", r"->\s*\S+\s+available, updating"),
    ("deps_pass", "PyPI unreachable, updating to be safe", r"could not reach PyPI, updating to be safe"),
    ("deps_pass", "repair forced", r"forcing (?:the )?(?:dependency pass|package repair)"),
)
# The probe answered only if setup could name PyPI's latest version. Under a refuse-all proxy it
# must NOT: a run that still prints it reached PyPI outside the proxy (Windows PowerShell 5.1's
# Invoke-RestMethod ignores HTTP(S)_PROXY -- see run_step.ps1's _UNSLOTH_PS_PROXY_DEFAULTS).
PYPI_PROBE_ANSWERED = (r"is up to date", r"->\s*\S+\s+available, updating")


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def site_packages(venv: Path) -> Path | None:
    for pat in ("lib/python*/site-packages", "Lib/site-packages"):
        for m in sorted(venv.glob(pat)):
            return m
    return None


def sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:32]
    except OSError:
        return None


def freeze(venv: Path) -> dict:
    py = venv_python(venv)
    if not py.exists():
        return {}
    code = ("import importlib.metadata as m, json;"
            "print(json.dumps({d.metadata['Name'].lower().replace('_','-'): d.version for d in m.distributions()}))")
    try:
        out = subprocess.run([str(py), "-I", "-c", code], capture_output=True, text=True, timeout=180)
        return json.loads(out.stdout or "{}")
    except Exception:  # noqa: BLE001
        return {}


def requirement_digests(venv: Path) -> dict:
    """sha of every requirement file the installed wheel ships.

    They live inside the venv, so a `studio update` that replaced the wheel changes them and a
    no-op update must not -- and it is what `fault req-edit` breaks.
    """
    sp = site_packages(venv)
    root = (sp / "studio" / "backend" / "requirements") if sp else None
    out = {}
    if root and root.exists():
        for f in sorted(root.rglob("*.txt")):
            out[str(f.relative_to(root))] = sha(f)
    return out


def dir_stat(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    files = 0
    total = 0
    newest = 0.0
    for root, _dirs, names in os.walk(path):
        for n in names:
            p = os.path.join(root, n)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            files += 1
            total += st.st_size
            newest = max(newest, st.st_mtime)
    return {"exists": True, "mtime": round(path.stat().st_mtime, 3), "newest_mtime": round(newest, 3),
            "files": files, "bytes": total}


def prebuilt_roots(home: Path, name: str) -> tuple[Path, ...]:
    """llama.cpp / whisper.cpp / node live NEXT TO the Studio home (~/.unsloth/llama.cpp), not
    inside it; snapshot.py looks in both places and so must this."""
    return (home / name, home.parent / name)


def find_llama_quantize(home: Path) -> Path | None:
    # Explicit candidates, not a walk: llama.cpp unpacks a full source tree here and walking it
    # costs seconds on every capture, of which there are two per measured step.
    for base in prebuilt_roots(home, "llama.cpp"):
        for rel in ("llama-quantize", "llama-quantize.exe",
                    "build/bin/llama-quantize", "build/bin/llama-quantize.exe",
                    "bin/llama-quantize", "bin/llama-quantize.exe",
                    "build/bin/Release/llama-quantize.exe"):
            p = base / rel
            if p.exists():
                return p
    return None


# ---------------------------------------------------------------- capture / compare
def capture(venv: Path, home: Path) -> dict:
    manifest_path = venv / MANIFEST_NAME
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(errors="replace"))
        except json.JSONDecodeError:
            manifest = {"_unparseable": sha(manifest_path)}
    state = {
        "venv": str(venv), "studio_home": str(home),
        "freeze": freeze(venv),
        "manifest_exists": manifest_path.exists(), "manifest": manifest,
        "requirements": requirement_digests(venv),
        "sidecars": {name: dir_stat(home / name) for name in SIDECARS},
        "markers": {},
        "prebuilt": {},
        "prefetch": prefetch_state(home),
    }
    for root, rels in ((home, HOME_MARKERS), (venv, VENV_MARKERS)):
        for rel in rels:
            p = root / rel
            state["markers"][rel] = {"exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None,
                                     "sha256": sha(p) if p.exists() else None}
    for name in SIDECARS:
        p = home / name / ".unsloth-studio-owned"
        state["markers"][f"{name}/.unsloth-studio-owned"] = {
            "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None,
            "sha256": sha(p) if p.exists() else None}
    for name in PREBUILT:
        found = next((r for r in prebuilt_roots(home, name)
                      if (r / "UNSLOTH_PREBUILT_INFO.json").exists()), None)
        info = (found / "UNSLOTH_PREBUILT_INFO.json") if found else None
        state["prebuilt"][name] = {"exists": bool(info), "root": str(found) if found else None,
                                   "sha256": sha(info) if info else None}
    q = find_llama_quantize(home)
    state["llama_quantize"] = {"path": str(q) if q else None,
                               "bytes": q.stat().st_size if q else None,
                               "mtime": round(q.stat().st_mtime, 3) if q else None}
    return state


def _strip(manifest, ignored = IGNORED_MANIFEST_KEYS):
    if not isinstance(manifest, dict):
        return manifest
    return {k: v for k, v in manifest.items() if k not in ignored}


def _tag_only_change(before: str, after: str) -> bool:
    """True when two versions differ only by their PEP 440 local label (0.16.0 -> 0.16.0+cpu)."""
    return before != after and before.partition("+")[0] == after.partition("+")[0]


def compare(a: dict, b: dict) -> dict:
    """Strict and relaxed idempotency verdicts over two `capture` dumps.

    Both verdicts are computed from the same categories, so a failing job can be read as "which of
    freeze / manifest / requirements / sidecars / markers / binaries moved" rather than as one
    boolean. `relaxed` is for the first no-op after an upgrade; `idempotent` is for every later one.
    """
    reasons: list[str] = []
    relaxed: list[str] = []
    changed: dict[str, list] = {}

    fa, fb = a.get("freeze") or {}, b.get("freeze") or {}
    freeze_diff = {
        "changed": {k: [fa[k], fb[k]] for k in fa if k in fb and fa[k] != fb[k]},
        "added": {k: fb[k] for k in fb if k not in fa},
        "removed": {k: fa[k] for k in fa if k not in fb},
    }
    changed["freeze"] = sorted(set(freeze_diff["changed"]) | set(freeze_diff["added"]) | set(freeze_diff["removed"]))
    if any(freeze_diff.values()):
        reasons.append(f"freeze changed: {json.dumps(freeze_diff)[:400]}")
    freeze_relaxed = {
        "changed": {k: v for k, v in freeze_diff["changed"].items()
                    if not (k in RELAXED_FREEZE_PACKAGES and _tag_only_change(v[0], v[1]))},
        "added": freeze_diff["added"], "removed": freeze_diff["removed"],
    }
    if any(freeze_relaxed.values()):
        relaxed.append(f"freeze changed: {json.dumps(freeze_relaxed)[:400]}")

    ma, mb = _strip(a.get("manifest")), _strip(b.get("manifest"))
    manifest_keys_changed = []
    if a.get("manifest_exists") != b.get("manifest_exists"):
        msg = f"manifest presence changed: {a.get('manifest_exists')} -> {b.get('manifest_exists')}"
        reasons.append(msg)
        relaxed.append(msg)
    elif ma != mb:
        keys = set(ma or {}) | set(mb or {})
        manifest_keys_changed = sorted(k for k in keys if (ma or {}).get(k) != (mb or {}).get(k))
        reasons.append(f"manifest changed (ignoring {','.join(IGNORED_MANIFEST_KEYS)}): {manifest_keys_changed}")
        beyond = [k for k in manifest_keys_changed if k not in RELAXED_MANIFEST_KEYS]
        if beyond:
            relaxed.append(f"manifest changed beyond {','.join(RELAXED_MANIFEST_KEYS)}: {beyond}")
    changed["manifest"] = manifest_keys_changed

    ra, rb = a.get("requirements") or {}, b.get("requirements") or {}
    requirements_changed = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
    changed["requirements"] = requirements_changed
    if requirements_changed:
        msg = f"requirement files changed: {requirements_changed}"
        reasons.append(msg)
        relaxed.append(msg)

    sidecar_changed = []
    for name in SIDECARS:
        sa, sb = a["sidecars"].get(name, {}), b["sidecars"].get(name, {})
        if sa != sb:
            sidecar_changed.append(name)
            msg = f"sidecar {name} changed: {sa} -> {sb}"
            reasons.append(msg)
            relaxed.append(msg)
    changed["sidecars"] = sidecar_changed

    marker_changed = sorted(k for k in set(a["markers"]) | set(b["markers"])
                            if a["markers"].get(k) != b["markers"].get(k))
    changed["markers"] = marker_changed
    for k in marker_changed:
        msg = f"marker {k} changed: {a['markers'].get(k)} -> {b['markers'].get(k)}"
        reasons.append(msg)
        relaxed.append(msg)

    prebuilt_changed = sorted(k for k in set(a["prebuilt"]) | set(b["prebuilt"])
                              if (a["prebuilt"].get(k) or {}).get("sha256") != (b["prebuilt"].get(k) or {}).get("sha256"))
    binaries_changed = list(prebuilt_changed)
    for k in prebuilt_changed:
        msg = f"prebuilt {k} info changed"
        reasons.append(msg)
        relaxed.append(msg)
    if a.get("llama_quantize") != b.get("llama_quantize"):
        binaries_changed.append("llama-quantize")
        msg = f"llama-quantize changed: {a.get('llama_quantize')} -> {b.get('llama_quantize')}"
        reasons.append(msg)
        relaxed.append(msg)
    changed["binaries"] = binaries_changed

    return {"idempotent": not reasons, "reasons": reasons,
            "idempotent_relaxed": not relaxed, "relaxed_reasons": relaxed,
            "relaxed_manifest_keys": list(RELAXED_MANIFEST_KEYS),
            "relaxed_freeze_packages": list(RELAXED_FREEZE_PACKAGES),
            "changed": changed,
            "freeze_diff": freeze_diff,
            "freeze_diff_empty": not any(freeze_diff.values()),
            "freeze_diff_empty_relaxed": not any(freeze_relaxed.values()),
            "manifest_keys_changed": manifest_keys_changed,
            "requirements_changed": requirements_changed,
            "sidecars_changed": sidecar_changed, "markers_changed": marker_changed,
            "prebuilt_changed": prebuilt_changed, "binaries_changed": binaries_changed,
            "sidecar_mtimes_before": {k: v.get("newest_mtime") for k, v in a["sidecars"].items()},
            "sidecar_mtimes_after": {k: v.get("newest_mtime") for k, v in b["sidecars"].items()},
            "marker_bytes_after": {k: v.get("bytes") for k, v in b["markers"].items()}}


# ---------------------------------------------------------------- fault injection
def inject(kind: str, venv: Path, home: Path) -> dict:
    sp = site_packages(venv)
    res: dict = {"kind": kind, "removed": [], "truncated": [], "edited": [], "ok": False}
    if kind == "sidecar-dist-info":
        target = home / ".venv_t5_550"
        tsp = site_packages(target)
        for d in sorted((tsp or target).glob("transformers-*.dist-info")):
            for root, dirs, names in os.walk(d, topdown=False):
                for n in names:
                    os.unlink(os.path.join(root, n))
                for n in dirs:
                    os.rmdir(os.path.join(root, n))
            d.rmdir()
            res["removed"].append(str(d))
    elif kind == "sidecar-truncate":
        target = home / ".venv_t5_550"
        tsp = site_packages(target)
        f = (tsp or target) / "transformers" / "__init__.py"
        if f.exists():
            # Unlink first: uv installs by hard-linking from its cache, so truncating
            # the file in place would also empty the cached copy and every rebuild
            # would link the same zero bytes back in. Real disk damage breaks one
            # tree, not the cache.
            f.unlink()
            with open(f, "wb"):
                pass
            res["truncated"].append(str(f))
    elif kind == "manifest":
        f = venv / MANIFEST_NAME
        if f.exists():
            f.unlink()
            res["removed"].append(str(f))
    elif kind == "req-edit":
        f = (sp or venv) / "studio" / "backend" / "requirements" / "studio.txt"
        if f.exists():
            with open(f, "a", encoding="utf-8") as fh:
                fh.write("\nsix\n")
            res["edited"].append(str(f))
    elif kind == "llama-truncate":
        q = find_llama_quantize(home)
        if q:
            with open(q, "wb"):
                pass
            res["truncated"].append(str(q))
    else:
        raise SystemExit(f"[idem] unknown fault kind {kind!r}")
    res["ok"] = bool(res["removed"] or res["truncated"] or res["edited"])
    if not res["ok"]:
        res["note"] = "nothing to break: the target did not exist (recorded, not fatal)"
    return res


def steps(log: Path) -> dict:
    text = log.read_text(errors="replace") if log.exists() else ""
    lines = text.splitlines()
    found = {}
    for key, pattern in STEP_PATTERNS.items():
        rx = re.compile(pattern)
        hits = [ln.strip()[:200] for ln in lines if rx.search(ln)]
        found[key] = {"count": len(hits), "lines": hits[:12]}
    path, reason, evidence = "unknown", "", ""
    for candidate, why, pattern in UPDATE_PATH_REASONS:
        hit = next((ln.strip()[:200] for ln in lines if re.search(pattern, ln)), None)
        if hit:
            path, reason, evidence = candidate, why, hit
            break
    if path == "unknown" and any(re.search(r"deps\s+\[", ln) for ln in lines):
        path, reason = "deps_pass", "dependency pass, reason not printed"
    # The final word: a forced pass that then finds nothing to do still ends in the dependency
    # pass, and "dependencies up to date" is only printed when the pass was skipped outright.
    if any("dependencies up to date" in ln for ln in lines):
        path = "fast_path"
    return {"log": str(log), "steps": found,
            "ran": sorted(k for k, v in found.items() if v["count"]),
            "update_path": path, "update_path_reason": reason, "update_path_evidence": evidence,
            "pypi_probe_answered": any(re.search(pat, ln) for pat in PYPI_PROBE_ANSWERED for ln in lines)}


# ---------------------------------------------------------------- leftover fixtures
LEFTOVERS = {
    ".update-rollback-1": None,                       # directory
    ".update-stage/READY.json": {"backend_version": "2026.9.9", "shell_version": "0.1.807-beta"},
    ".update-prev/PENDING.json": {"previous_entries": [], "versions": {"backend": "2026.9.9",
                                                                      "shell": "0.1.807-beta"}},
    ".update-failed.json": {"reason": "seeded by the bisect harness", "backend_version": "2026.9.9"},
    ".desktop-update-bundle.json": {"version": "0.1.808-beta", "path": "seeded"},
}


def leftovers_seed(home: Path) -> dict:
    home.mkdir(parents=True, exist_ok=True)
    made = []
    for rel, payload in LEFTOVERS.items():
        p = home / rel
        if payload is None:
            p.mkdir(parents=True, exist_ok=True)
            (p / "placeholder.txt").write_text("seeded by the bisect harness\n")
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, indent=1))
        made.append(rel)
    return {"seeded": made, "studio_home": str(home)}


def leftovers_check(home: Path) -> dict:
    state = {rel: (home / rel).exists() for rel in LEFTOVERS}
    return {"studio_home": str(home), "exists": state,
            "still_present": sorted(k for k, v in state.items() if v),
            "gone": sorted(k for k, v in state.items() if not v)}


def prefetch_state(home: Path) -> dict:
    d = home / PREFETCH_DIR
    marker = d / PREFETCH_MARKER
    data = None
    if marker.exists():
        try:
            data = json.loads(marker.read_text(errors="replace"))
        except json.JSONDecodeError:
            data = {"_unparseable": True}
    return {"dir_exists": d.exists(), "marker_exists": marker.exists(), "marker": data,
            "dir": dir_stat(d) if d.exists() else {"exists": False}}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--venv", required=True)
    c.add_argument("--studio-home", required=True)
    c.add_argument("--out")
    d = sub.add_parser("compare")
    d.add_argument("before")
    d.add_argument("after")
    d.add_argument("--out")
    i = sub.add_parser("inject")
    i.add_argument("kind")
    i.add_argument("--venv", required=True)
    i.add_argument("--studio-home", required=True)
    i.add_argument("--out")
    s = sub.add_parser("steps")
    s.add_argument("--log", required=True)
    s.add_argument("--out")
    lo = sub.add_parser("leftovers")
    lo.add_argument("action", choices=["seed", "check"])
    lo.add_argument("--studio-home", required=True)
    lo.add_argument("--out")
    pf = sub.add_parser("prefetch")
    pf.add_argument("--studio-home", required=True)
    pf.add_argument("--out")
    a = ap.parse_args()

    if a.cmd == "capture":
        res = capture(Path(a.venv), Path(a.studio_home))
    elif a.cmd == "compare":
        res = compare(json.loads(Path(a.before).read_text()), json.loads(Path(a.after).read_text()))
    elif a.cmd == "inject":
        res = inject(a.kind, Path(a.venv), Path(a.studio_home))
    elif a.cmd == "steps":
        res = steps(Path(a.log))
    elif a.cmd == "leftovers":
        res = leftovers_seed(Path(a.studio_home)) if a.action == "seed" else leftovers_check(Path(a.studio_home))
    else:
        res = prefetch_state(Path(a.studio_home))

    text = json.dumps(res, indent=1, sort_keys=True)
    if getattr(a, "out", None):
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        # capture dumps are large; keep stdout to a digest the shell can log.
        if a.cmd == "capture":
            print(json.dumps({"packages": len(res["freeze"]), "manifest": res["manifest_exists"],
                              "sidecars": {k: v.get("files") for k, v in res["sidecars"].items()},
                              "out": a.out}))
            return
    print(text)


if __name__ == "__main__":
    sys.exit(main())
