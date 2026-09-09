#!/usr/bin/env python3
"""Build / inspect the PR wheel and derive the pin layer that puts it above the release pins.

The staging matrix updates an OLD installed release to the code under review. The desktop update
is `studio update` run by the OLD release's own CLI, so the only honest way to point it at the PR
is to make the PR wheel resolvable as a normal `unsloth` distribution:

    UV_FIND_LINKS=<dir>  PIP_FIND_LINKS=<dir>        # where the wheel is
    UV_CONSTRAINT=<pins_pr.txt>                      # unsloth==<wheel version>, zoo pinned
    UV_CONFIG_FILE=<uv_pr.toml>                      # release cutoffs, minus unsloth's

Relabelling also folds `unsloth_zoo>=<floor>` back into the wheel's CORE requirements when the
build did not emit it there. `python -m build` reads `[project] dependencies`, where unsloth_zoo
does not appear -- it lives in the `huggingface` extra that `base` points at -- while the wheel
PyPI serves for the same version carries it at the top level. install.sh's fresh-install arm runs
`uv pip install unsloth` with no extra and then skips setup.sh's "base packages" step
(SKIP_STUDIO_BASE=1), so without the core entry a PR install ends with no unsloth_zoo at all and
dies with "unsloth-zoo is not installed, so this environment cannot run". `--no-fold-core` opts
out; the fold is a no-op once the wheel declares it itself.

`uv_<N>.toml` time-travels unsloth and unsloth-zoo with `exclude-newer-package`. The PR wheel is
newer than every cutoff, so the unsloth entry has to go (the zoo entry stays: the source release's
zoo is what a real user of that release has, and it must keep working unless the PR wheel itself
demands a newer one -- see `zoo_floor`).

    python pr_wheel.py build   --source . --wheel-dir DIR [--bump]
    python pr_wheel.py verify  --source . --wheel-dir DIR
    python pr_wheel.py info    --wheel-dir DIR
    python pr_wheel.py relabel --wheel-dir DIR [--post 1]        # in place: <version>.post1
    python pr_wheel.py bump    --wheel-dir DIR --out-dir DIR2 [--post 2]
    python pr_wheel.py pins    --wheel-dir DIR --pins-dir P --out-dir O --name pr [--base 807]

Every subcommand prints one JSON object on stdout. Stdlib only (runners have nothing else).
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

PYPI_JSON = "https://pypi.org/pypi/{}/json"


# ---------------------------------------------------------------- version helpers
def ver_key(v: str) -> tuple:
    """PEP 440-lite ordering key: release segments, then the post number, then any local label."""
    body, _, local = v.partition("+")
    m = re.search(r"\.post(\d+)$", body.strip())
    post = int(m.group(1)) if m else -1
    base = re.sub(r"(\.post\d+|\.dev\d+|[abrc]+\d+)$", "", body.strip())
    parts = []
    for chunk in base.split("."):
        parts.append(int(chunk) if chunk.isdigit() else 0)
    while len(parts) < 4:
        parts.append(0)
    return (tuple(parts), post, local)


def base_version(v: str) -> str:
    """The plain release the wheel was built from, without the harness's .postN suffix."""
    return re.sub(r"\.post\d+$", "", v.partition("+")[0])


def published_versions(dist: str, timeout: float = 30.0) -> set:
    try:
        with urllib.request.urlopen(PYPI_JSON.format(dist), timeout=timeout) as fh:
            return set((json.load(fh).get("releases") or {}))
    except Exception:  # noqa: BLE001 -- offline runner
        return set()


def wheel_version(name: str) -> str:
    # unsloth-2026.9.3-py3-none-any.whl -> 2026.9.3
    return name.split("-")[1]


def find_wheel(wheel_dir: Path, dist: str = "unsloth") -> Path:
    cands = sorted(
        p for p in wheel_dir.glob("*.whl")
        if p.name.split("-")[0].lower().replace("_", "-") == dist
    )
    if not cands:
        raise SystemExit(f"[pr_wheel] no {dist} wheel in {wheel_dir}: {[p.name for p in wheel_dir.glob('*')]}")
    return cands[-1]


def declared_version(source: Path) -> str | None:
    """The version pyproject declares, without building or importing anything."""
    py = source / "pyproject.toml"
    if not py.exists():
        return None
    try:
        import tomllib
    except ModuleNotFoundError:  # python < 3.11 runners
        m = re.search(r'^version\s*=\s*\{\s*attr\s*=\s*"([^"]+)"', py.read_text(), re.M)
        return _attr_version(source, m.group(1)) if m else None
    data = tomllib.loads(py.read_text())
    proj = data.get("project", {})
    if "version" in proj:
        return str(proj["version"])
    attr = ((data.get("tool", {}).get("setuptools", {}).get("dynamic", {}) or {}).get("version", {}) or {}).get("attr")
    return _attr_version(source, attr) if attr else None


def _attr_version(source: Path, attr: str) -> str | None:
    """Static AST read of `pkg.module.__version__` -- never imports the package (it needs torch)."""
    mod, _, name = attr.rpartition(".")
    path = source / (mod.replace(".", "/") + ".py")
    if not path.exists():
        path = source / mod.replace(".", "/") / "__init__.py"
    if not path.exists():
        return None
    tree = ast.parse(path.read_text(errors="replace"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            if isinstance(node.value, ast.Constant):
                return str(node.value.value)
    return None


# ---------------------------------------------------------------- wheel metadata
def wheel_metadata(whl: Path) -> tuple[str, list[str]]:
    with zipfile.ZipFile(whl) as z:
        meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        text = z.read(meta).decode("utf-8", "replace")
    version = ""
    reqs = []
    for line in text.splitlines():
        if line.startswith("Version:") and not version:
            version = line.split(":", 1)[1].strip()
        elif line.startswith("Requires-Dist:"):
            reqs.append(line.split(":", 1)[1].strip())
        elif not line.strip():
            break
    return version, reqs


def core_reqs(reqs: list[str]) -> list[str]:
    """The requirements a bare `pip install unsloth` pulls: no `extra == "..."` marker."""
    return [r for r in reqs if not re.search(r'extra\s*==\s*[\'"]', r)]


def core_zoo(reqs: list[str]) -> str | None:
    """The unsloth_zoo requirement a bare `pip install unsloth` would pull, if any."""
    for r in core_reqs(reqs):
        if re.match(r"^\s*unsloth[-_]zoo(\s|\[|[<>=!~;]|$)", r, re.I):
            return r.strip()
    return None


def fold_core_zoo(meta_text: str, floor: str) -> tuple[str, str | None]:
    """Give the wheel the `unsloth_zoo` core dependency the PUBLISHED wheel of this version has.

    `python -m build` from the checkout produces METADATA straight from `[project] dependencies`,
    where unsloth_zoo does not appear -- it lives in the `huggingface` extra that `base` points at.
    The wheel PyPI serves for the same version carries `Requires-Dist: unsloth_zoo>=<floor>` at the
    top level, and install.sh depends on that: its fresh-install arm runs `uv pip install unsloth`
    with no extra and then hands off to studio/setup.sh with SKIP_STUDIO_BASE=1, which skips the
    only other step that names unsloth-zoo ("base packages"). A PR wheel without the core entry
    therefore installs an environment with no unsloth_zoo at all, and the setup dies with
    "unsloth-zoo is not installed, so this environment cannot run" -- a harness artifact, not a
    defect of the branch (observed on all three OSes; the `update` path passes because
    `studio update` does not set SKIP_STUDIO_BASE).

    No-op when the wheel already declares it, so a build that starts emitting the full dependency
    set needs no change here.
    """
    lines = meta_text.splitlines(keepends = True)
    reqs = [ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("Requires-Dist:")]
    if core_zoo(reqs) or not floor:
        return meta_text, None
    added = f"unsloth_zoo>={floor}"
    for i, ln in enumerate(lines):
        if ln.startswith("Requires-Dist:"):
            lines.insert(i, f"Requires-Dist: {added}\n")
            return "".join(lines), added
    return meta_text, None


def zoo_floor(reqs: list[str]) -> str | None:
    """Highest `unsloth_zoo>=X` floor over every requirement, extras included.

    The Studio venv installs the extras, so an extra that demands a newer zoo demands it for real;
    a floor read from the bare dependencies only would leave the release zoo pin blocking the
    install with a resolution error rather than a clear "the PR needs a newer zoo".
    """
    best = None
    for r in reqs:
        spec, _, _marker = r.partition(";")
        m = re.match(r"^\s*unsloth[-_]zoo\s*(\[[^\]]*\])?\s*(.*)$", spec, re.I)
        if not m:
            continue
        for op, ver in re.findall(r"(>=|==)\s*([0-9][^,\s\)]*)", m.group(2) or ""):
            if best is None or ver_key(ver) > ver_key(best):
                best = ver
    return best


def min_pypi_version_at_least(dist: str, floor: str, timeout: float = 30.0) -> str:
    """Smallest published version >= floor. Falls back to the floor itself when PyPI is unreachable."""
    try:
        with urllib.request.urlopen(PYPI_JSON.format(dist), timeout=timeout) as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 -- offline runner: the floor is the best guess we have
        return floor
    ok = [v for v, files in (data.get("releases") or {}).items()
          if files and not all(f.get("yanked") for f in files) and ver_key(v) >= ver_key(floor)]
    return min(ok, key=ver_key) if ok else floor


# ---------------------------------------------------------------- subcommands
def cmd_build(a: argparse.Namespace) -> dict:
    src, out = Path(a.source).resolve(), Path(a.wheel_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.whl"):
        stale.unlink()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "build"], check=True)
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(out)], cwd=str(src), check=True)
    whl = find_wheel(out)
    version, reqs = wheel_metadata(whl)
    res = {"wheel": str(whl), "wheel_name": whl.name, "version": version,
           "filename_version": wheel_version(whl.name), "declared_version": declared_version(src),
           "zoo_floor": zoo_floor(reqs)}
    if os.environ.get("UNSLOTH_PR_WHEEL_VERSION_BUMP") in ("1", "true", "yes") or a.bump:
        res["bumped"] = cmd_bump(argparse.Namespace(wheel_dir=str(out), out_dir=a.bump_out or str(out) + "2",
                                                    post=a.post, no_pypi_check=False, no_fold_core=False))
    return res


def cmd_verify(a: argparse.Namespace) -> dict:
    src, wd = Path(a.source).resolve(), Path(a.wheel_dir).resolve()
    whl = find_wheel(wd)
    meta_version, reqs = wheel_metadata(whl)
    fname_version = wheel_version(whl.name)
    decl = declared_version(src)
    ok = meta_version == fname_version and (decl is None or decl == base_version(meta_version))
    res = {"wheel_name": whl.name, "version": meta_version, "filename_version": fname_version,
           "declared_version": decl, "zoo_floor": zoo_floor(reqs), "core_zoo": core_zoo(reqs),
           "ok": ok}
    if not ok:
        print(json.dumps(res, indent=1))
        raise SystemExit("[pr_wheel] version mismatch between pyproject, METADATA and the wheel name")
    return res


def cmd_info(a: argparse.Namespace) -> dict:
    whl = find_wheel(Path(a.wheel_dir).resolve())
    version, reqs = wheel_metadata(whl)
    return {"wheel": str(whl), "wheel_name": whl.name, "version": version,
            "zoo_floor": zoo_floor(reqs), "core_zoo": core_zoo(reqs)}


def cmd_bump(a: argparse.Namespace) -> dict:
    """Copy the wheel into out_dir under a post-release version (`2026.9.3.post2`).

    Not a local label (`+pr2`): the installer builds its own floor as `unsloth>=<version>`, and
    PEP 440 forbids `>=` with a local segment -- uv and pip both refuse to parse it, so a local
    label makes the update fail before it starts (observed). `.postN` is greater than the release
    it was built from, legal in `>=`, and unsloth has never published a post-release, so
    `unsloth==<base>.postN` can only be satisfied by the file in the find-links directory. The
    number is chosen to avoid anything actually on PyPI.
    """
    src_dir, out = Path(a.wheel_dir).resolve(), Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.whl"):
        stale.unlink()
    whl = find_wheel(src_dir)
    old, _ = wheel_metadata(whl)
    new = pick_post_version(base_version(old), int(a.post), check_pypi=not a.no_pypi_check)
    parts = whl.name.split("-")
    parts[1] = new
    dest = out / "-".join(parts)
    old_di, new_di = f"unsloth-{old}.dist-info", f"unsloth-{new}.dist-info"
    with zipfile.ZipFile(whl) as zin:
        payload = {}
        for item in zin.infolist():
            name = item.filename.replace(old_di, new_di, 1) if item.filename.startswith(old_di) else item.filename
            payload[name] = zin.read(item.filename)
        order = [n.replace(old_di, new_di, 1) if n.startswith(old_di) else n for n in zin.namelist()]
    meta_text = payload[f"{new_di}/METADATA"].decode("utf-8", "replace")
    meta_text = re.sub(r"^Version: .*$", f"Version: {new}", meta_text, count=1, flags=re.M)
    folded = None
    if not getattr(a, "no_fold_core", False):
        meta_text, folded = fold_core_zoo(meta_text, zoo_floor(wheel_metadata(whl)[1]) or "")
    payload[f"{new_di}/METADATA"] = meta_text.encode("utf-8")
    # Rebuild RECORD with real hashes: an installer that verifies it must still accept the wheel,
    # so blanking the METADATA row is not good enough -- only RECORD itself may have empty columns.
    rows = []
    for line in payload[f"{new_di}/RECORD"].decode("utf-8", "replace").splitlines():
        path = line.split(",")[0].replace(old_di, new_di, 1)
        if not path:
            continue
        if path == f"{new_di}/RECORD":
            rows.append(f"{path},,")
        elif path in payload:
            blob = payload[path]
            digest = base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).rstrip(b"=").decode()
            rows.append(f"{path},sha256={digest},{len(blob)}")
        else:
            rows.append(line.replace(old_di, new_di, 1))
    payload[f"{new_di}/RECORD"] = ("\n".join(rows) + "\n").encode()
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in order:
            zout.writestr(name, payload[name])
    dest_reqs = wheel_metadata(dest)[1]
    return {"wheel": str(dest), "wheel_name": dest.name, "version": new, "previous_version": old,
            "base_version": base_version(old), "zoo_floor": zoo_floor(dest_reqs),
            "folded_core_requirement": folded, "core_zoo": core_zoo(dest_reqs)}


def pick_post_version(base: str, want: int, check_pypi: bool = True) -> str:
    """`<base>.post<n>`, skipping any n that PyPI already publishes."""
    published = published_versions("unsloth") if check_pypi else set()
    n = max(want, 1)
    while f"{base}.post{n}" in published:
        n += 1
    return f"{base}.post{n}"


def cmd_relabel(a: argparse.Namespace) -> dict:
    """Give the freshly built wheel a post-release version, in place.

    A PR branches off main while `unsloth/_version.py` still holds the LAST RELEASE's version, so
    the wheel built from the branch and the wheel on PyPI would share a version string. `unsloth==X`
    is then satisfiable from either, and a job could pass having installed the very release it was
    meant to replace. `X.post1` exists only in the find-links directory, and unlike a local label
    it survives the installer's own `unsloth>=X.post1` floor check.
    """
    wd = Path(a.wheel_dir).resolve()
    tmp = wd / ".relabel"
    res = cmd_bump(argparse.Namespace(wheel_dir=str(wd), out_dir=str(tmp), post=a.post,
                                      no_pypi_check=a.no_pypi_check,
                                      no_fold_core=getattr(a, "no_fold_core", False)))
    old = find_wheel(wd)
    new = Path(res["wheel"])
    dest = wd / new.name
    old.unlink()
    shutil.move(str(new), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    res["wheel"] = str(dest)
    return res


def cmd_pins(a: argparse.Namespace) -> dict:
    """Write pins_<name>.txt + uv_<name>.toml: the release layer with unsloth swapped for the wheel."""
    wd, pins_dir, out = Path(a.wheel_dir).resolve(), Path(a.pins_dir).resolve(), Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    whl = find_wheel(wd)
    version, reqs = wheel_metadata(whl)
    floor = zoo_floor(reqs)

    base_pins = (pins_dir / f"pins_{a.base}.txt").read_text()
    base_toml = (pins_dir / f"uv_{a.base}.toml").read_text()
    m = re.search(r"^unsloth[-_]zoo==(\S+)", base_pins, re.M)
    zoo = m.group(1) if m else None
    zoo_relaxed = False
    if floor and zoo and ver_key(floor) > ver_key(zoo):
        zoo = min_pypi_version_at_least("unsloth_zoo", floor)
        zoo_relaxed = True

    lines = [f"# harness: PR wheel layer over release {a.base} pins ({whl.name})"]
    for line in base_pins.splitlines():
        if re.match(r"^unsloth==", line):
            continue
        if re.match(r"^unsloth[-_]zoo==", line):
            continue
        lines.append(line)
    lines += [f"unsloth=={version}"]
    if zoo:
        lines.append(f"unsloth-zoo=={zoo}")
    pins_path = out / f"pins_{a.name}.txt"
    pins_path.write_text("\n".join(lines) + "\n")

    # Drop unsloth from exclude-newer-package (the PR wheel postdates every release cutoff);
    # drop unsloth-zoo too only when the wheel forced a newer zoo than the release shipped with.
    toml_lines = []
    for line in base_toml.splitlines():
        if "exclude-newer-package" in line:
            keep = re.findall(r'(unsloth-zoo|unsloth)\s*=\s*"([^"]+)"', line)
            entries = [f'{k} = "{v}"' for k, v in keep if k == "unsloth-zoo" and not zoo_relaxed]
            if entries:
                toml_lines.append("exclude-newer-package = { " + ", ".join(entries) + " }")
            continue
        toml_lines.append(line)
    toml_lines.insert(0, f"# harness: PR wheel layer over release {a.base}")
    toml_path = out / f"uv_{a.name}.toml"
    toml_path.write_text("\n".join(toml_lines) + "\n")

    return {"name": a.name, "version": version, "zoo": zoo, "zoo_floor": floor, "zoo_relaxed": zoo_relaxed,
            "core_zoo": core_zoo(reqs),
            "base": str(a.base), "wheel": str(whl), "wheel_name": whl.name,
            "find_links": str(wd), "pins": str(pins_path), "uv_toml": str(toml_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", default=".")
    b.add_argument("--wheel-dir", required=True)
    b.add_argument("--bump", action="store_true")
    b.add_argument("--bump-out", default="")
    b.add_argument("--post", default="2")
    v = sub.add_parser("verify")
    v.add_argument("--source", default=".")
    v.add_argument("--wheel-dir", required=True)
    i = sub.add_parser("info")
    i.add_argument("--wheel-dir", required=True)
    rl = sub.add_parser("relabel")
    rl.add_argument("--wheel-dir", required=True)
    rl.add_argument("--post", default="1")
    rl.add_argument("--no-pypi-check", action="store_true")
    rl.add_argument("--no-fold-core", action="store_true",
                    help="leave METADATA's core dependencies exactly as `python -m build` emitted them")
    u = sub.add_parser("bump")
    u.add_argument("--wheel-dir", required=True)
    u.add_argument("--out-dir", required=True)
    u.add_argument("--post", default="2")
    u.add_argument("--no-pypi-check", action="store_true")
    u.add_argument("--no-fold-core", action="store_true")
    p = sub.add_parser("pins")
    p.add_argument("--wheel-dir", required=True)
    p.add_argument("--pins-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default="pr")
    p.add_argument("--base", default="807")
    a = ap.parse_args()
    fn = {"build": cmd_build, "verify": cmd_verify, "info": cmd_info, "bump": cmd_bump,
          "relabel": cmd_relabel, "pins": cmd_pins}[a.cmd]
    print(json.dumps(fn(a), indent=1))


if __name__ == "__main__":
    main()
