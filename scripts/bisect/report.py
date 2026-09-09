#!/usr/bin/env python3
"""Aggregate bisect results (local + CI artifacts) into markdown tables, and gate a CI job on them.

    python report.py [--roots outputs/bisect/linux outputs/bisect/ci] > tables.md
    python report.py --roots "$OUT" --step-summary >> "$GITHUB_STEP_SUMMARY"
    python report.py --roots "$OUT" --assert idempotent [--expect-idempotent false]
    python report.py --roots "$OUT" --assert old-shell-stage | prefetch | timing
    python report.py --roots "$OUT" --assert timing --budget '{"macos-15": 65}' --budget-os macos-15 \
        --budget-step update-806-to-pr

Walks every `*/summary.json`, groups by (source, os, experiment), and prints per-step timing,
network attribution, torch before/after and phase breakdowns parsed from the timestamped logs.

`--assert` turns the PR-mode fields the harness records (idempotent, connections_attempted, the
fault step lists, the prefetch marker state) into the job's exit code, so a job fails on the
evidence rather than on a step's exit code alone.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict

PHASE_PATTERNS = [
    (r"\[TAURI:STEP\] clone", "clone"),
    (r"\[TAURI:STEP\] update", "update-start"),
    (r"\[TAURI:STEP\] verify", "verify"),
    (r"installing PyTorch", "torch install"),
    (r"installing unsloth \(", "unsloth core install"),
    (r"\d+/\d+\s+pip bootstrap", "deps: pip bootstrap"),
    (r"\d+/\d+\s+torch check", "deps: torch check"),
    (r"\d+/\d+\s+base packages", "deps: base packages"),
    (r"\d+/\d+\s+base requirements", "deps: base requirements"),
    (r"\d+/\d+\s+studio deps", "deps: studio deps"),
    (r"\d+/\d+\s+data designer", "deps: data designer"),
    (r"\d+/\d+\s+unsloth extras", "deps: unsloth extras"),
    (r"\d+/\d+\s+triton kernels", "deps: triton kernels"),
    (r"\d+/\d+\s+flash-attn", "deps: flash-attn"),
    (r"\d+/\d+\s+torchcodec", "deps: torchcodec"),
    (r"\d+/\d+\s+xformers", "deps: xformers"),
    (r"\d+/\d+\s+mlx", "deps: mlx"),
    (r"\d+/\d+\s+torch flavor", "deps: torch flavor"),
    (r"\d+/\d+\s+verify", "deps: verify"),
    (r"deps\s+installed", "deps: done"),
    (r"installing prebuilt llama\.cpp", "llama.cpp prebuilt"),
    (r"whisper\.cpp\s", "whisper.cpp"),
    (r"node\s+v\d", "frontend/node check"),
    (r"\d+/\d+\s+extra codecs", "deps: extra codecs"),
    (r"\d+/\d+\s+dependency overrides", "deps: overrides"),
    (r"\d+/\d+\s+anyio check", "deps: anyio check"),
    (r"\d+/\d+\s+MLX stack", "deps: MLX stack"),
    (r"Updating core packages|installing unsloth (\d|\S+ ->)", "core packages"),
    (r"existing install detected -- validating", "llama.cpp validate"),
    (r"installing prebuilt whisper|whisper\.cpp\s+installing", "whisper.cpp install"),
    (r"Unsloth Studio Installed", "done"),
    (r"installed files are damaged", "DAMAGE SCAN FAILED"),
    (r"staged update failed", "staged failed"),
]
TS = re.compile(r"^\[\s*([0-9.]+)\]\s?(.*)$")

# What an offline no-op may still DIAL. The fast path asks PyPI for the latest version and the
# prebuilt pre-checks ask GitHub whether the pinned release moved; under `--refuse` every one of
# those is refused, which is the signal we want -- an attempt is expected, a byte is not. Anything
# outside this set means the run went past the fast path and started resolving, so it is reported
# (the byte / exit / tree assertions below are what actually fail the job).
OFFLINE_EXPECTED_HOSTS = ("pypi.org", "github.com", "api.github.com")
# The order the per-step diagnostic prints. `binaries` is the prebuilt llama.cpp / whisper.cpp /
# node markers plus the llama-quantize binary itself.
CHANGE_CATEGORIES = ("freeze", "manifest", "requirements", "sidecars", "markers", "binaries")


def idem_role(s: dict) -> str:
    """settle / noop / offline. Recorded by run_step since this change; derived for older runs."""
    role = s.get("idempotency_role")
    if role in ("settle", "noop", "offline"):
        return role
    label = str(s.get("label") or "")
    if s.get("offline") or label.endswith("-offline"):
        return "offline"
    return "settle" if label.endswith("-settle") else "noop"


def changed_categories(s: dict) -> dict:
    """Which of freeze/manifest/requirements/sidecars/markers/binaries moved, per step."""
    changed = s.get("changed")
    if isinstance(changed, dict):
        return {k: list(changed.get(k) or []) for k in CHANGE_CATEGORIES}
    fd = s.get("freeze_diff") or {}
    return {
        "freeze": sorted(set(fd.get("changed") or {}) | set(fd.get("added") or {}) | set(fd.get("removed") or {})),
        "manifest": list(s.get("manifest_keys_changed") or []),
        "requirements": list(s.get("requirements_changed") or []),
        # Artifacts predating the category lists still carry the sidecar mtimes, which is what
        # "the sidecars were rebuilt" means; markers have no before/after pair to fall back on.
        "sidecars": list(s.get("sidecars_changed") or [
            k for k, v in (s.get("sidecar_mtimes_before") or {}).items()
            if (s.get("sidecar_mtimes_after") or {}).get(k) != v]),
        "markers": list(s.get("markers_changed") or []),
        "binaries": list(s.get("binaries_changed") or s.get("prebuilt_changed") or []),
    }


def conn_counts(s: dict) -> dict:
    """Attempted / refused / bytes, overall and per host, from the proxy log summary."""
    by_host = (s.get("proxy") or {}).get("by_host") or {}
    return {
        "attempted": s.get("connections_attempted", (s.get("proxy") or {}).get("connections", 0)) or 0,
        "refused": s.get("connections_refused", (s.get("proxy") or {}).get("refused", 0)) or 0,
        "bytes_down": (s.get("proxy") or {}).get("total_bytes_down", 0) or 0,
        "hosts": {h: {"attempted": v.get("connections", 0), "refused": v.get("refused", 0),
                      "bytes_down": v.get("bytes_down", 0)} for h, v in by_host.items()},
    }


def idem_diagnostic(s: dict) -> str:
    """One diagnosable line per no-op step: what moved, and what it dialled."""
    role = idem_role(s)
    cats = changed_categories(s)
    moved = ", ".join(f"{k}={cats[k]}"[:120] for k in CHANGE_CATEGORIES if cats[k]) or "nothing"
    c = conn_counts(s)
    hosts = ", ".join(f"{h} {v['attempted']}att/{v['refused']}ref/{v['bytes_down']}B"
                      for h, v in c["hosts"].items()) or "none"
    return (f"[{role}] exit={s.get('exit_code')} idempotent={s.get('idempotent')} "
            f"relaxed={s.get('idempotent_relaxed')} "
            f"path={s.get('update_path', '?')} ({s.get('update_path_reason') or 'n/a'}) "
            f"pypi_probe_answered={s.get('pypi_probe_answered')} changed: {moved} | "
            f"conns {c['attempted']} attempted / {c['refused']} refused / {c['bytes_down']}B down "
            f"[{hosts}]")


def phases(log: pathlib.Path) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    if not log.exists():
        return out
    seen = set()
    for line in log.read_text(errors="replace").splitlines():
        m = TS.match(line)
        if not m:
            continue
        t, text = float(m.group(1)), m.group(2)
        for pat, name in PHASE_PATTERNS:
            if re.search(pat, text):
                key = (name, round(t))
                if name.startswith("deps:") and any(k[0] == name for k in seen):
                    break
                if key in seen:
                    break
                seen.add(key)
                out.append((t, name))
                break
    return out


def phase_durations(ph: list[tuple[float, str]], total: float) -> dict[str, float]:
    d: dict[str, float] = {}
    for i, (t, name) in enumerate(ph):
        end = ph[i + 1][0] if i + 1 < len(ph) else total
        if name in ("done", "deps: done", "update-start"):
            continue
        d[name] = round(max(end - t, 0), 1)
    return d


def mb(x) -> str:
    return f"{(x or 0)/1e6:.0f}"


def load(roots: list[str]) -> list[dict]:
    rows = []
    for root in roots:
        rp = pathlib.Path(root)
        for f in sorted(rp.rglob("summary.json")):
            try:
                s = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            rel = f.relative_to(rp)
            parts = rel.parts
            s["_source"] = root
            s["_exp"] = "/".join(parts[:-2]) if len(parts) > 2 else parts[0]
            s["_dir"] = f.parent
            s.setdefault("label", f.parent.name)
            rows.append(s)
    return rows


def pr_line(s: dict) -> str:
    """One line of the PR-mode fields, empty for a step that has none."""
    bits = []
    if "idempotent" in s:
        bits.append(f"role={idem_role(s)}")
        bits.append(f"idempotent={s['idempotent']}")
        if "idempotent_relaxed" in s:
            bits.append(f"relaxed={s['idempotent_relaxed']}")
        cats = changed_categories(s)
        bits.append("changed=" + (",".join(k for k in CHANGE_CATEGORIES if cats[k]) or "none"))
        if s.get("update_path"):
            bits.append(f"path={s['update_path']}")
        bits.append(f"conns={s.get('connections_attempted')}")
        if s.get("connections_refused"):
            bits.append(f"refused={s['connections_refused']}")
        bits.append(f"down={(s.get('proxy') or {}).get('total_bytes_down', 0)}B")
        if not s["idempotent"]:
            bits.append("reasons=" + "; ".join(s.get("idempotency_reasons") or [])[:300])
    if s.get("fault_kind"):
        bits.append(f"fault={s['fault_kind']}")
        bits.append("ran=" + (",".join(s.get("fault_steps_ran") or []) or "none"))
        counts = {k: v.get("count") for k, v in (s.get("fault_steps") or {}).items()}
        bits.append(f"counts={counts}")
    if "prefetch_supported" in s:
        bits.append(f"prefetch_supported={s['prefetch_supported']}")
        bits.append(f"marker={s.get('prefetch_marker_exists')}")
        bits.append(f"prefetch_bytes={mb(s.get('prefetch_dir_bytes'))}MB")
        bits.append(f"live_venv_unchanged={s.get('live_venv_unchanged')}")
    if "tauri_error_present" in s:
        bits.append(f"tauri_error={s['tauri_error_present']}")
        bits.append(f"failed_json={s.get('update_failed_json_exists')}")
        bits.append(f"stage_dir={s.get('stage_dir_exists')}")
        bits.append(f"pass={s.get('pass')}")
    if "stage_import_exit_code" in s:
        bits.append(f"stage_import_rc={s['stage_import_exit_code']}")
        bits.append(f"live_venv_unchanged={s.get('live_venv_unchanged')}")
        bits.append(f"pass={s.get('pass')}")
    if "leftovers_present" in s:
        bits.append(f"leftovers_present={s['leftovers_present']}")
    return " ".join(bits)


def step_summary(rows: list[dict]) -> None:
    """Compact table for $GITHUB_STEP_SUMMARY, including the PR-mode fields."""
    print("| step | s | exit | assert | proxy MB | conns | notes |")
    print("|---|---:|---:|---|---:|---:|---|")
    for s in sorted(rows, key=lambda r: str(r.get("label"))):
        pr = (s.get("proxy") or {})
        print(f"| {s.get('label')} | {s.get('seconds_total', s.get('seconds_to_health', ''))} "
              f"| {s.get('exit_code', '')} | {s.get('assert_ok', '')} | {mb(pr.get('total_bytes_down'))} "
              f"| {s.get('connections_attempted', '')} | {pr_line(s)[:400]} |")


def check(rows: list[dict], kind: str, budget: dict, budget_os: str, budget_step: str,
          expect: bool = True) -> int:
    """Fail the job on the recorded evidence. Returns a process exit code.

    `expect=False` (the workflow's expect_idempotent input, for a branch that does not carry the
    fix under test) records the same verdict and exits 0: the point of running it there is the
    before/after evidence, not a red job.
    """
    fails: list[str] = []
    seen = 0
    for s in rows:
        label = str(s.get("label") or "")
        if kind == "idempotent" and label.startswith("noop-update"):
            seen += 1
            role = idem_role(s)
            # Printed for every role, including the one that asserts nothing: a step summary that
            # only says "not idempotent" cannot be diagnosed without the artifacts.
            print(f"[report] {label}: {idem_diagnostic(s)}", flush=True)
            c = conn_counts(s)
            if role == "settle":
                # The first update by NEW code after an upgrade rewrites the manifest, may repair
                # torchao to the +cpu build and backfills the uv cache marker. Evidence, not a
                # verdict -- the comparison that means something is the next run against this one.
                if s.get("exit_code") not in (0, "0"):
                    fails.append(f"{label}: settle run exited {s.get('exit_code')}")
                continue
            if role == "noop":
                # settle -> noop, relaxed: only completed_at_ms / step_results / pip_check_ok in
                # the manifest, and only torchao's local tag in the freeze.
                if not s.get("idempotent_relaxed", s.get("idempotent")):
                    fails.append(f"{label}: not idempotent under the relaxed rule "
                                 f"(manifest may move only in {','.join(s.get('relaxed_manifest_keys') or [])}; "
                                 f"freeze only in {','.join(s.get('relaxed_freeze_packages') or [])}): "
                                 f"{'; '.join(s.get('relaxed_reasons') or s.get('idempotency_reasons') or [])[:500]}")
                if s.get("exit_code") not in (0, "0"):
                    fails.append(f"{label}: exit {s.get('exit_code')}")
                continue
            # role == offline: noop -> offline, strict, and it must not move a byte.
            if not s.get("idempotent"):
                fails.append(f"{label}: not idempotent: {'; '.join(s.get('idempotency_reasons') or [])[:500]}")
            if not s.get("freeze_diff_empty", True):
                fails.append(f"{label}: freeze changed: {json.dumps(s.get('freeze_diff'))[:300]}")
            if s.get("exit_code") not in (0, "0"):
                fails.append(f"{label}: exit {s.get('exit_code')} (an offline no-op must succeed "
                             f"from the fast path, not fall through to a dependency pass)")
            # Attempts are expected under `--refuse` -- bytes are not.
            paid = {h: v["bytes_down"] for h, v in c["hosts"].items() if v["bytes_down"]}
            if paid:
                fails.append(f"{label}: offline run downloaded {c['bytes_down']} bytes from {paid}")
            # setup names PyPI's latest version only if the probe got an answer. Under `--refuse`
            # that is impossible through the proxy, so an answer proves the probe went around it
            # (Windows PowerShell 5.1's Invoke-RestMethod ignores HTTP(S)_PROXY) and the whole
            # offline measurement -- counts included -- is fiction.
            if s.get("pypi_probe_answered"):
                fails.append(f"{label}: the PyPI version probe was answered during a refuse-all run, "
                             f"so it bypassed the CONNECT proxy and this run was not offline "
                             f"({s.get('update_path_evidence') or ''})")
            unexpected = sorted(h for h in c["hosts"] if h not in OFFLINE_EXPECTED_HOSTS)
            if unexpected:
                print(f"[report] {label}: dialled beyond the fast-path hosts "
                      f"{list(OFFLINE_EXPECTED_HOSTS)}: {unexpected} (recorded)", flush=True)
        elif kind == "old-shell-stage" and label == "old-shell-stage":
            seen += 1
            if not s.get("pass"):
                fails.append(f"{label}: exit={s.get('exit_code')} tauri_error={s.get('tauri_error_present')} "
                             f"failed_json={s.get('update_failed_json_exists')} stage_dir={s.get('stage_dir_exists')}")
        elif kind == "prefetch" and label.startswith("prefetch-"):
            seen += 1
            if not s.get("prefetch_supported"):
                print(f"[report] {label}: the installed CLI has no `studio prefetch-update` "
                      f"(exit 2); nothing to assert", flush=True)
                continue
            if s.get("prefetch_exit_code") not in (0, "0"):
                fails.append(f"{label}: prefetch exit {s.get('prefetch_exit_code')}")
            if not s.get("prefetch_marker_exists"):
                fails.append(f"{label}: no PREFETCHED.json marker")
            if not s.get("live_venv_unchanged"):
                fails.append(f"{label}: the prefetch changed the live venv: "
                             f"{'; '.join(s.get('live_venv_reasons') or [])[:400]}")
        elif kind == "prefetch" and label.endswith("-offlinepypi"):
            seen += 1
            if s.get("exit_code") not in (0, "0") or not s.get("assert_ok"):
                fails.append(f"{label}: the offline swap failed (exit={s.get('exit_code')}, "
                             f"assert_ok={s.get('assert_ok')})")
            by_host = (s.get("proxy") or {}).get("by_host") or {}
            hosts = [h for h, v in by_host.items() if v.get("bytes_down")]
            # The prefetch warms the uv cache for the wheels, so the swap must move nothing from
            # the package index hosts. Git dependencies (triton-kernels' branch ref) and the torch
            # index are re-resolved by the dependency pass until its per-step skips land; they
            # are recorded and warned about, not failed, so the proof stays about the prefetch.
            denied = [h for h in hosts if h in PREFETCH_ZERO_HOSTS]
            other = [h for h in hosts if h not in PREFETCH_ZERO_HOSTS]
            if denied:
                fails.append(f"{label}: bytes still came down from the index hosts {denied[:5]}")
            if other:
                print(f"::warning::{label}: the swap still fetched from {other[:5]} "
                      f"(dependency-pass steps that a later PR skips)", flush=True)
        elif kind == "timing" and label == budget_step:
            seen += 1
            limit = budget.get(budget_os)
            got = s.get("seconds_total")
            if limit is None:
                print(f"[report] no budget for {budget_os}; {label} took {got}s", flush=True)
            elif got is not None and float(got) > float(limit):
                fails.append(f"{label}: {got}s over the {limit}s budget for {budget_os}")
            else:
                print(f"[report] {label}: {got}s within the {limit}s budget for {budget_os}", flush=True)
    if not seen:
        print(f"[report] --assert {kind}: no matching step found; nothing was verified", flush=True)
        return 1
    for f in fails:
        print(f"::{'error' if expect else 'warning'}::{f}", flush=True)
    print(f"[report] --assert {kind}: {seen} step(s) checked, {len(fails)} failure(s)"
          f"{'' if expect else ' (expect_idempotent=false: recorded, not enforced)'}", flush=True)
    return 1 if (fails and expect) else 0


PREFETCH_ZERO_HOSTS = ("pypi.org", "files.pythonhosted.org")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["outputs/bisect/linux", "outputs/bisect/ci"])
    ap.add_argument("--step-summary", action="store_true", help="compact table for $GITHUB_STEP_SUMMARY")
    ap.add_argument("--assert", dest="assert_kind",
                    choices=["idempotent", "old-shell-stage", "prefetch", "timing"])
    ap.add_argument("--budget", default="{}", help="JSON object of per-OS second budgets")
    ap.add_argument("--budget-os", default="")
    ap.add_argument("--budget-step", default="")
    ap.add_argument("--expect-idempotent", default="true",
                    help="false records the idempotency verdict without failing the job")
    a = ap.parse_args()
    rows = load(a.roots)
    if a.assert_kind:
        try:
            budget = json.loads(a.budget or "{}")
        except json.JSONDecodeError:
            budget = {}
        expect = str(a.expect_idempotent).strip().lower() not in ("0", "false", "no", "off")
        raise SystemExit(check(rows, a.assert_kind, budget, a.budget_os, a.budget_step, expect))
    if a.step_summary:
        step_summary(rows)
        return
    by = defaultdict(list)
    for s in rows:
        by[(s["_source"], s["_exp"])].append(s)
    for (src, exp), steps in sorted(by.items()):
        print(f"\n### {src} / {exp}\n")
        print("| step | s | exit | ok | proxy MB | top hosts (MB) | torch before -> after | pkg chg/add | new cache MB | marker after |")
        print("|---|---:|---:|---|---:|---|---|---|---:|---|")
        for s in steps:
            p = s.get("proxy") or {}
            hosts = ", ".join(f"{h.split('.')[0] if not h.startswith('download') else h.split('.')[0]}={mb(v['bytes_down'])}" for h, v in list((p.get("by_host") or {}).items())[:4])
            d = s.get("diff") or {}
            tb = (d.get("torch_before") or {}).get("version") or "-"
            ta = (d.get("torch_after") or {}).get("version") or "-"
            chg = f"{len(d.get('packages_changed') or {})}/{len(d.get('packages_added') or {})}" if d else "-"
            marker = d.get("marker_after") if d else None
            marker = (marker or "-").replace(str(pathlib.Path.home()), "~")
            marker = re.sub(r".*/(home|runner)/", "~/", marker)
            # A launch step has no exit code and no version assertion: reaching /api/health IS
            # its verdict, and defaulting to `exit_code == 0` printed ok=False for a server that
            # came up fine.
            if "seconds_to_health" in s:
                ok = s["seconds_to_health"] is not None
            else:
                ok = s.get("assert_ok", s.get("exit_code") == 0)
            extra = ""
            if "seconds_to_health" in s:
                extra = (f"health {s['seconds_to_health']}s" if s["seconds_to_health"] is not None
                         else f"NO health in {s.get('health_timeout_seconds', '?')}s")
            print(f"| {s.get('label')} | {s.get('seconds_total', s.get('seconds_to_health', ''))} | {s.get('exit_code','')} | {ok} {extra} | {mb(p.get('total_bytes_down'))} | {hosts} | {tb} -> {ta} | {chg} | {mb(d.get('new_cache_bytes')) if d else '-'} | {marker} |")
        for s in steps:
            line = pr_line(s)
            if line:
                print(f"\n`{s.get('label')}`: {line}")
        # phase breakdown
        for s in steps:
            log = s["_dir"] / "log.txt"
            if not log.exists():
                continue
            ph = phases(log)
            if not ph:
                continue
            durs = phase_durations(ph, float(s.get("seconds_total") or 0))
            if durs:
                print(f"\n`{s.get('label')}` phases (s): " + ", ".join(f"{k} {v}" for k, v in durs.items()))
        # what got downloaded (named cache entries)
        for s in steps:
            d = s.get("diff") or {}
            named = d.get("new_cache_named") or []
            if named and s.get("label", "").startswith("update"):
                big = [x for x in (d.get("new_cache_top") or []) if x.get("bytes", 0) > 20e6][:8]
                print(f"\n`{s.get('label')}` new cache entries: {len(named)} named; >20MB: " + ", ".join(f"{x.get('name') or x['entry']}={mb(x['bytes'])}MB" for x in big))


if __name__ == "__main__":
    main()
