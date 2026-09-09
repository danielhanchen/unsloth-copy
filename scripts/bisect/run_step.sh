#!/usr/bin/env bash
# One measured step of the Unsloth Studio update bisection (Linux / macOS).
#
#   run_step.sh install  <N|pr>  [--no-torch] [--isolated-uv-cache]
#   run_step.sh update   <N_from> <N_to|pr|pr2> [--stage] [--offline-pypi]
#   run_step.sh activate <label>                       # swap a READY stage into the live tree
#   run_step.sh launch   <label> [seconds]             # headless first launch, time to /api/health
#   run_step.sh uninstall <N> <label>
#
#   run_step.sh noop-update <label> [--target pr] [--offline] [--role settle|noop|offline]
#   run_step.sh fault <kind> [--target pr]             # break one thing, update, record what ran
#   run_step.sh old-shell-stage [--shell 0.1.807-beta] # PR A: --stage must be refused
#   run_step.sh nested-old-stage [against_snapshot_dir]# PR A: an old CLI's real --stage still works
#   run_step.sh leftover-seed | leftover-check         # PR A: legacy-state fixtures
#   run_step.sh prefetch [--target pr2]                # PR C: prefetch-update, live venv untouched
#
# Required env: RUN_ROOT (isolated root; the child gets HOME=$RUN_ROOT/home), OUT (results dir).
# Optional: BISECT_DIR (dir of this script), SCRIPTS_DIR (where <N>/install.sh live, fetched
# from the tag if missing), PROXY=0 to skip the logging proxy.
#
# PR-wheel targets (`pr`, `pr2`): the wheel built from the checked-out branch, relabelled
# `<version>.post1` / `.post2`, is served from a local find-links directory layered ABOVE the
# release pins (see pr_wheel.py), so `studio update` run by the OLD installed release resolves
# `unsloth` to the PR and its `unsloth>=<version>` floor check still parses. PR_WHEEL_DIR / PR_WHEEL_DIR2
# point at the wheel dirs (default $RUN_ROOT/prwheel[2]); PR_PIN_BASE (default 807) is the release
# whose pins the PR layer sits on.
set -uo pipefail

BISECT_DIR="${BISECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
PINS="$BISECT_DIR/pins"
: "${RUN_ROOT:?RUN_ROOT required}"; : "${OUT:?OUT required}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$RUN_ROOT/tag_scripts}"
FAKE_HOME="$RUN_ROOT/home"
[ "${BISECT_REAL_HOME:-0}" = 1 ] && FAKE_HOME="$HOME"
STUDIO="$FAKE_HOME/.unsloth/studio"
VENV="$STUDIO/unsloth_studio"
STAGE="$STUDIO/.update-stage"
mkdir -p "$FAKE_HOME" "$OUT" "$SCRIPTS_DIR"
PY3="${PY3:-python3}"
OS_NAME="$(uname -s)"

# PR wheel layer
SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$BISECT_DIR/../.." && pwd)}"
PR_WHEEL_DIR="${PR_WHEEL_DIR:-$RUN_ROOT/prwheel}"
PR_WHEEL_DIR2="${PR_WHEEL_DIR2:-$RUN_ROOT/prwheel2}"
PR_PIN_BASE="${PR_PIN_BASE:-807}"
GEN_PINS="${GEN_PINS:-$RUN_ROOT/pins_gen}"
DENY_HOSTS="${DENY_HOSTS:-pypi.org,files.pythonhosted.org,github.com,objects.githubusercontent.com,release-assets.githubusercontent.com}"

rel() { "$PY3" -c "import json,sys; print(json.load(open('$PINS/releases.json'))['$1']['$2'])"; }

is_pr() { case "$1" in pr|pr2) return 0 ;; *) return 1 ;; esac; }
wheel_dir() { case "$1" in pr2) echo "$PR_WHEEL_DIR2" ;; *) echo "$PR_WHEEL_DIR" ;; esac; }

# Derive (once per run root) the pin + find-links layer for a PR wheel target.
prep_pr() {  # prep_pr <pr|pr2>
    local t=$1
    # Reuse the cached layer only if it was written by THIS pr_wheel.py: a file from before the
    # core-dependency fold has no core_zoo, and every later reader would see an empty answer.
    if [ -s "$GEN_PINS/$t.json" ] && "$PY3" -c "import json,sys; sys.exit(0 if 'core_zoo' in json.load(open(sys.argv[1])) else 1)" "$GEN_PINS/$t.json" 2>/dev/null; then
        return 0
    fi
    mkdir -p "$GEN_PINS"
    if ! "$PY3" "$BISECT_DIR/pr_wheel.py" pins --wheel-dir "$(wheel_dir "$t")" --pins-dir "$PINS" \
            --out-dir "$GEN_PINS" --name "$t" --base "$PR_PIN_BASE" > "$GEN_PINS/$t.json"; then
        cat "$GEN_PINS/$t.json" >&2; rm -f "$GEN_PINS/$t.json"
        echo "[harness] could not build the $t pin layer from $(wheel_dir "$t")" >&2; return 1
    fi
    echo "[harness] $t layer: $("$PY3" -c "import json;d=json.load(open('$GEN_PINS/$t.json'));print(f\"unsloth=={d['version']} zoo=={d['zoo']} (relaxed={d['zoo_relaxed']}) find-links={d['find_links']}\")")"
}
prj() { "$PY3" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$GEN_PINS/$1.json" "$2"; }

target_version() { if is_pr "$1"; then prep_pr "$1" >/dev/null || return 1; prj "$1" version; else rel "$1" pypi_version; fi; }
target_zoo()     { if is_pr "$1"; then prep_pr "$1" >/dev/null || return 1; prj "$1" zoo;     else rel "$1" zoo;          fi; }

installed_version() {  # installed_version <dist> [venv]
    "${2:-$VENV}/bin/python" -I -c "import importlib.metadata as m
try: print(m.version('$1'))
except Exception: print('')" 2>/dev/null
}

fetch_tag_script() {  # fetch_tag_script <N> <install.sh|install.ps1|uninstall.sh>
    local n=$1 name=$2 dest="$SCRIPTS_DIR/$1" path key
    mkdir -p "$dest"
    case "$name" in uninstall.*) path="scripts/$name"; key="" ;; *) path="$name"; key="${name//./_}_blob" ;; esac
    if [ ! -s "$dest/$name" ]; then
        curl -fsSL "https://raw.githubusercontent.com/unslothai/unsloth/v0.1.$n-beta/$path" -o "$dest/$name" || return 1
    fi
    if [ -n "$key" ]; then
        local want got
        want=$(rel "$n" "$key")
        got=$("$PY3" -c "import hashlib,sys; b=open('$dest/$name','rb').read(); print(hashlib.sha1(b'blob %d\0'%len(b)+b).hexdigest())")
        if [ "$want" != "$got" ]; then echo "[harness] BLOB MISMATCH for $n/$name: want $want got $got" >&2; return 1; fi
        echo "[harness] $n/$name blob $got verified against tag v0.1.$n-beta"
    fi
}

rx_bytes() {
    case "$OS_NAME" in
        Linux) awk -F'[: ]+' 'NR>2 && $2!="lo" {s+=$3} END{print s+0}' /proc/net/dev ;;
        Darwin) netstat -ib 2>/dev/null | awk 'NR>1 && $1!="lo0" && $3 ~ /Link/ {s+=$7} END{print s+0}' ;;
        *) echo 0 ;;
    esac
}

# Timestamp every line AND every carriage-return segment: the installers redraw a `deps` progress
# bar with \r, and one timestamp per \n would collapse the whole dependency pass into one line.
now_s() { "$PY3" -c "import time; print(round(time.time(), 3))"; }

ts_filter() { perl -MTime::HiRes=time -e '$|=1;$t=time;$b="";while(sysread(STDIN,$c,65536)){for(split(/(\r|\n)/,$c)){if($_ eq "\r"||$_ eq "\n"){printf "[%7.1f] %s\n",time-$t,$b;$b=""}else{$b.=$_}}}printf "[%7.1f] %s\n",time-$t,$b if length $b'; }
# Same filter without perl (the self-hosted AMD runners have no guarantee of it).
if ! command -v perl >/dev/null 2>&1; then
ts_filter() { "$PY3" -u -c '
import sys, time
t = time.time(); b = ""
out = sys.stdout
while True:
    c = sys.stdin.buffer.read1(65536) if hasattr(sys.stdin.buffer, "read1") else sys.stdin.buffer.read(65536)
    if not c:
        break
    for ch in c.decode("utf-8", "replace"):
        if ch in "\r\n":
            out.write("[%7.1f] %s\n" % (time.time() - t, b)); out.flush(); b = ""
        else:
            b += ch
if b:
    out.write("[%7.1f] %s\n" % (time.time() - t, b)); out.flush()
'; }
fi

PROXY_PID=""
start_proxy() {  # start_proxy <dir> [extra connect_proxy.py serve args...]
    local d=$1; shift
    [ "${PROXY:-1}" = 0 ] && { PROXY_URL=""; return; }
    rm -f "$d/proxy.port"
    "$PY3" "$BISECT_DIR/connect_proxy.py" serve --port 0 --log "$d/proxy.jsonl" --port-file "$d/proxy.port" \
        ${1+"$@"} >"$d/proxy.out" 2>&1 &
    PROXY_PID=$!
    for _ in $(seq 1 50); do [ -s "$d/proxy.port" ] && break; sleep 0.1; done
    PROXY_URL="http://127.0.0.1:$(cat "$d/proxy.port")"
}
stop_proxy() { [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null; PROXY_PID=""; }

# Environment the product sees. HOME is the isolated one; nothing from this shell's UV_* /
# UNSLOTH_STUDIO_HOME / venv leaks in; the pin layers and the proxy are added.
child_env() {  # child_env <target> [extra VAR=val ...]
    local n=$1; shift
    local pin_txt pin_toml find_links=""
    if is_pr "$n"; then
        prep_pr "$n" >/dev/null || return 1
        pin_txt=$(prj "$n" pins); pin_toml=$(prj "$n" uv_toml); find_links=$(prj "$n" find_links)
    else
        pin_txt="$PINS/pins_$n.txt"; pin_toml="$PINS/uv_$n.toml"
    fi
    local -a e=(env -u UV_CACHE_DIR -u UNSLOTH_STUDIO_HOME -u STUDIO_HOME -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME
        -u XDG_CACHE_HOME -u XDG_DATA_HOME -u XDG_CONFIG_HOME -u UV_CONSTRAINT -u UV_CONFIG_FILE -u PIP_CONSTRAINT -u UV_PYTHON
        -u UV_FIND_LINKS -u PIP_FIND_LINKS -u UV_OFFLINE
        -u UNSLOTH_COMPILE_LOCATION -u UNSLOTH_WORKSPACE
        "HOME=$FAKE_HOME" "UNSLOTH_SKIP_AUTOSTART=1" "UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK=1" "PYTHONUNBUFFERED=1"
        "UV_CONSTRAINT=$pin_txt" "UV_CONFIG_FILE=$pin_toml" "PIP_CONSTRAINT=$pin_txt"
        "NO_PROXY=127.0.0.1,localhost" "no_proxy=127.0.0.1,localhost")
    [ -n "$find_links" ] && e+=("UV_FIND_LINKS=$find_links" "PIP_FIND_LINKS=$find_links")
    if [ -n "${PROXY_URL:-}" ]; then e+=("HTTPS_PROXY=$PROXY_URL" "HTTP_PROXY=$PROXY_URL" "ALL_PROXY=$PROXY_URL" "https_proxy=$PROXY_URL" "http_proxy=$PROXY_URL"); fi
    e+=("$@")
    printf '%q ' "${e[@]}"
}

snapshot() {  # snapshot <dir> <label> [venv]
    "$PY3" "$BISECT_DIR/snapshot.py" take "$1" --venv "${3:-$VENV}" --studio-home "$STUDIO" \
        --cache "$FAKE_HOME/.cache/uv" --label "$2" 2>&1 | tail -1
}

idem() { "$PY3" "$BISECT_DIR/idem.py" "$@"; }

assert_versions() {  # assert_versions <venv> <expect_unsloth> <expect_zoo> -> prints json, returns 1 on mismatch
    local py="$1/bin/python"
    if [ "$2" = "-" ]; then echo '{"skipped": true, "ok": true}'; return 0; fi
    "$py" -I -c "
import importlib.metadata as m, json, sys
got = {'unsloth': None, 'unsloth_zoo': None}
for k in got:
    try: got[k] = m.version(k)
    except Exception as e: got[k] = f'ERR:{e}'
ok = got['unsloth'] == '$2' and got['unsloth_zoo'] == '$3'
print(json.dumps({'expected': {'unsloth': '$2', 'unsloth_zoo': '$3'}, 'got': got, 'ok': ok}))
sys.exit(0 if ok else 1)"
}

# finish_step <stepdir> <label> <t0> <rc> <rx0> <venv> <expect_v> <expect_z> [extra_json_file]
# expect_v of "-" skips the version assertion. EXPECT_RC (default 0) is the exit code that counts
# as success; EXPECT_RC=any never fails on the exit code (the caller judges it from summary.json).
finish_step() {
    local d=$1 label=$2 t0=$3 rc=$4 rx0=$5 venv=$6 ev=$7 ez=$8 extra=${9:-}
    local t1 rx1 seconds assertion assert_rc=0 proxy_json="{}" expect_rc="${EXPECT_RC:-0}"
    t1=$(now_s); rx1=$(rx_bytes)
    seconds=$("$PY3" -c "print(round($t1-$t0,1))")
    stop_proxy
    [ -s "$d/proxy.jsonl" ] && proxy_json=$("$PY3" "$BISECT_DIR/connect_proxy.py" summary "$d/proxy.jsonl")
    snapshot "$d/after" "$label:after" "$venv" > "$d/after.line"
    assertion=$(assert_versions "$venv" "$ev" "$ez") || assert_rc=$?
    local diff_json="null"
    if [ -f "$d/before/snapshot.json" ]; then
        "$PY3" "$BISECT_DIR/snapshot.py" diff "$d/before" "$d/after" --out "$d/diff.json" >/dev/null 2>&1 && diff_json=$(cat "$d/diff.json")
    fi
    "$PY3" - "$d" "$label" "$seconds" "$rc" "$assert_rc" "$rx0" "$rx1" "$expect_rc" "$extra" <<PY
import json, os, sys
d, label, seconds, rc, arc, rx0, rx1, expect_rc, extra = sys.argv[1:]
summary = {
  "label": label, "os": "$OS_NAME", "seconds_total": float(seconds), "exit_code": int(rc),
  "expected_exit_code": expect_rc,
  "assert_ok": int(arc) == 0, "assertion": json.loads('''$assertion''' or 'null'),
  "rx_bytes_host_delta": int(rx1) - int(rx0),
  "proxy": json.loads('''$proxy_json'''),
  "diff": json.loads('''$diff_json'''),
}
summary["connections_attempted"] = summary["proxy"].get("connections", 0)
summary["connections_refused"] = summary["proxy"].get("refused", 0)
if extra and os.path.exists(extra):
    summary.update(json.load(open(extra)))
json.dump(summary, open(f"{d}/summary.json", "w"), indent=1)
p = summary["proxy"].get("by_host", {})
top = ", ".join(f"{h}={v['bytes_down']/1e6:.0f}MB" for h, v in list(p.items())[:5])
extra_bits = ""
if "idempotent" in summary:
    extra_bits = f" idempotent={summary['idempotent']} conns={summary['connections_attempted']}"
print(f"[harness] {label}: {seconds}s exit={rc} assert_ok={summary['assert_ok']} proxy_down={summary['proxy'].get('total_bytes_down',0)/1e6:.0f}MB{extra_bits} [{top}]")
PY
    echo "$assertion"
    local rc_bad=0
    [ "$expect_rc" = any ] || { [ "$rc" = "$expect_rc" ] || rc_bad=1; }
    if [ "$rc_bad" != 0 ] || [ "$assert_rc" != 0 ]; then
        echo "[harness] STEP FAILED: $label (exit=$rc expected=$expect_rc assert_rc=$assert_rc)" >&2; return 1
    fi
}

cmd_install() {
    local n=$1; shift
    local flags=("$@") label="install-$n" d="$OUT/install-$n" script
    mkdir -p "$d"
    local ev ez
    if is_pr "$n"; then
        prep_pr "$n" || return 1
        script="$SOURCE_ROOT/install.sh"
        [ -s "$script" ] || { echo "[harness] no branch installer at $script" >&2; return 1; }
        ev=$(target_version "$n"); ez=$(target_zoo "$n")
        # install.sh's fresh-install arm runs `uv pip install unsloth` with no extra and then hands
        # off to studio/setup.sh with SKIP_STUDIO_BASE=1, which skips the only other step that
        # names unsloth-zoo. The wheel's own core metadata is therefore the ONLY thing that puts
        # unsloth_zoo in the venv -- and `python -m build` from the checkout does not emit it (it
        # lives in the `huggingface` extra that `base` points at), which is what made every
        # `install pr` die with "unsloth-zoo is not installed, so this environment cannot run".
        # pr_wheel.py folds it back in at relabel time; refuse here rather than spend a full
        # install proving it again.
        if [ "$(prj "$n" core_zoo)" = "None" ]; then
            echo "[harness] the $n wheel declares no core unsloth_zoo requirement, so a fresh" >&2
            echo "[harness] install would leave the venv without unsloth_zoo. Rebuild the layer" >&2
            echo "[harness] with pr_wheel.py relabel (without --no-fold-core)." >&2
            return 1
        fi
        # --tauri is what the desktop passes: take the frontend the wheel bundles instead of
        # running `npm install` in the checkout. Without it a PR install measures a frontend build
        # that no user performs, and dies on any peer-dependency drift in studio/frontend
        # (observed locally). The release installers are untouched.
        case " ${flags[*]-} " in *" --tauri "*) ;; *) flags+=(--tauri) ;; esac
    else
        fetch_tag_script "$n" install.sh || return 1
        script="$SCRIPTS_DIR/$n/install.sh"
        ev=$(rel "$n" pypi_version); ez=$(rel "$n" zoo)
    fi
    snapshot "$d/before" "$label:before" > "$d/before.line" 2>/dev/null || true
    start_proxy "$d"
    echo "[harness] === $label  (installer $script, pins unsloth==$ev zoo==$ez, flags: ${flags[*]:-none})"
    local t0 rx0 rc; t0=$(now_s); rx0=$(rx_bytes)
    local -a inst_env=(); is_pr "$n" && inst_env+=(SKIP_STUDIO_FRONTEND=1)
    ( cd "$FAKE_HOME" && eval "$(child_env "$n" ${inst_env[@]+"${inst_env[@]}"})" sh "$script" ${flags[@]+"${flags[@]}"} ) 2>&1 | ts_filter | tee "$d/log.txt" >/dev/null
    rc=${PIPESTATUS[0]}
    finish_step "$d" "$label" "$t0" "$rc" "$rx0" "$VENV" "$ev" "$ez"
}

cmd_update() {
    local from=$1 to=$2; shift 2
    local staged=0 offline_pypi=0 arg
    for arg in "$@"; do
        case "$arg" in
            --stage) staged=1 ;;
            --offline-pypi) offline_pypi=1 ;;
            *) echo "[harness] unknown update flag $arg" >&2; return 2 ;;
        esac
    done
    local sfx=""; [ $staged = 1 ] && sfx="-staged"; [ $offline_pypi = 1 ] && sfx="$sfx-offlinepypi"
    local label="update-$from-to-$to$sfx" d="$OUT/update-$from-to-$to$sfx"
    mkdir -p "$d"
    local ev ez; ev=$(target_version "$to") || return 1; ez=$(target_zoo "$to") || return 1
    snapshot "$d/before" "$label:before" > "$d/before.line"
    if [ $offline_pypi = 1 ]; then start_proxy "$d" --deny-hosts "$DENY_HOSTS"; else start_proxy "$d"; fi
    local -a args=(studio update); [ $staged = 1 ] && args+=(--stage)
    echo "[harness] === $label  (venv CLI of the installed release, UNSLOTH_DESKTOP_BACKEND_VERSION=$ev, pins unsloth==$ev zoo==$ez$([ $offline_pypi = 1 ] && echo ", proxy denies $DENY_HOSTS"))"
    local t0 rx0 rc; t0=$(now_s); rx0=$(rx_bytes)
    ( cd "$FAKE_HOME" && eval "$(child_env "$to" UNSLOTH_TAURI_UPDATE=1 SKIP_STUDIO_FRONTEND=1 "UNSLOTH_DESKTOP_BACKEND_VERSION=$ev")" \
        "$VENV/bin/python" -I -X utf8 -m unsloth_cli "${args[@]}" ) 2>&1 | ts_filter | tee "$d/log.txt" >/dev/null
    rc=${PIPESTATUS[0]}
    local target="$VENV"; [ $staged = 1 ] && target="$STAGE/unsloth_studio"
    idem steps --log "$d/log.txt" --out "$d/steps.json" > /dev/null 2>&1 || true
    "$PY3" -c "import json;d=json.load(open('$d/steps.json'));json.dump({'update_steps':d['steps'],'update_steps_ran':d['ran'],'update_path':d['update_path'],'update_path_reason':d['update_path_reason'],'update_path_evidence':d['update_path_evidence'],'pypi_probe_answered':d['pypi_probe_answered']},open('$d/extra.json','w'))" 2>/dev/null || echo '{}' > "$d/extra.json"
    finish_step "$d" "$label" "$t0" "$rc" "$rx0" "$target" "$ev" "$ez" "$d/extra.json"
}

# ---------------------------------------------------------------- PR-mode subcommands

# Re-run `studio update` against the version that is already installed: it must do nothing.
# --offline puts the proxy in refuse-all mode and sets UV_OFFLINE=1, so "no work" is proved by the
# connection count rather than by the absence of visible output.
#
# --role names which comparison this run is, because the runs are not interchangeable:
#
#   settle  the FIRST update run by the NEW code after an upgrade. It legitimately rewrites the
#           manifest (pass_inputs / step_results / uv_version / steps_total are only written by
#           code that has them), repairs torchao to the +cpu build the index actually serves, and
#           backfills the uv cache marker. Recorded, never asserted.
#   noop    the second run. Compared against the settle run under the RELAXED rule: the manifest
#           may move only in completed_at_ms / step_results / pip_check_ok, and the freeze only in
#           torchao's local tag.
#   offline the third run. Compared against the second under the STRICT rule, and additionally
#           required to move zero bytes and exit 0.
cmd_noop_update() {
    local sfx=$1; shift
    local offline=0 target="${NOOP_TARGET:-pr}" role="" arg
    while [ $# -gt 0 ]; do
        arg=$1
        case "$arg" in
            --offline) offline=1 ;;
            --target) shift; target=$1 ;;
            --role) shift; role=$1 ;;
            *) echo "[harness] unknown noop-update flag $arg" >&2; return 2 ;;
        esac
        shift
    done
    if [ -z "$role" ]; then role=$([ $offline = 1 ] && echo offline || echo noop); fi
    case "$role" in settle|noop|offline) ;; *) echo "[harness] unknown --role $role" >&2; return 2 ;; esac
    local label="noop-update-$sfx$([ $offline = 1 ] && echo -offline)" d="$OUT/noop-update-$sfx$([ $offline = 1 ] && echo -offline)"
    mkdir -p "$d"
    local ev ez; ev=$(installed_version unsloth); ez=$(installed_version unsloth_zoo)
    [ -n "$ev" ] || { echo "[harness] nothing installed in $VENV" >&2; return 1; }
    snapshot "$d/before" "$label:before" > "$d/before.line"
    idem capture --venv "$VENV" --studio-home "$STUDIO" --out "$d/state_before.json" > /dev/null
    if [ $offline = 1 ]; then start_proxy "$d" --refuse; else start_proxy "$d"; fi
    local -a extra_env=(UNSLOTH_TAURI_UPDATE=1 SKIP_STUDIO_FRONTEND=1 "UNSLOTH_DESKTOP_BACKEND_VERSION=$ev")
    [ $offline = 1 ] && extra_env+=(UV_OFFLINE=1)
    echo "[harness] === $label  (role=$role, target == installed unsloth==$ev zoo==$ez, offline=$offline, pin layer $target)"
    local t0 rx0 rc; t0=$(now_s); rx0=$(rx_bytes)
    ( cd "$FAKE_HOME" && eval "$(child_env "$target" "${extra_env[@]}")" \
        "$VENV/bin/python" -I -X utf8 -m unsloth_cli studio update ) 2>&1 | ts_filter | tee "$d/log.txt" >/dev/null
    rc=${PIPESTATUS[0]}
    idem capture --venv "$VENV" --studio-home "$STUDIO" --out "$d/state_after.json" > /dev/null
    idem compare "$d/state_before.json" "$d/state_after.json" --out "$d/idempotency.json" > /dev/null
    idem steps --log "$d/log.txt" --out "$d/steps.json" > /dev/null 2>&1 || true
    "$PY3" - "$d" "$offline" "$target" "$role" <<'PY'
import json, os, sys
d, offline, target, role = sys.argv[1:]
cmp_ = json.load(open(f"{d}/idempotency.json"))
steps = json.load(open(f"{d}/steps.json")) if os.path.exists(f"{d}/steps.json") else {"steps": {}, "ran": []}
json.dump({
    "offline": offline == "1", "pin_target": target, "idempotency_role": role,
    "idempotent": cmp_["idempotent"], "idempotency_reasons": cmp_["reasons"],
    "idempotent_relaxed": cmp_["idempotent_relaxed"], "relaxed_reasons": cmp_["relaxed_reasons"],
    "relaxed_manifest_keys": cmp_["relaxed_manifest_keys"],
    "relaxed_freeze_packages": cmp_["relaxed_freeze_packages"],
    "freeze_diff_empty": cmp_["freeze_diff_empty"], "freeze_diff": cmp_["freeze_diff"],
    "changed": cmp_["changed"],
    "manifest_keys_changed": cmp_["manifest_keys_changed"],
    "requirements_changed": cmp_["requirements_changed"],
    "sidecars_changed": cmp_["sidecars_changed"],
    "markers_changed": cmp_["markers_changed"],
    "prebuilt_changed": cmp_["prebuilt_changed"],
    "binaries_changed": cmp_["binaries_changed"],
    "sidecar_mtimes_before": cmp_["sidecar_mtimes_before"],
    "sidecar_mtimes_after": cmp_["sidecar_mtimes_after"],
    "marker_bytes_after": cmp_["marker_bytes_after"],
    "update_steps": steps["steps"], "update_steps_ran": steps["ran"],
    # Which route through setup.sh this run took. The `.postN` relabel means an ONLINE no-op can
    # never reach the "installed == PyPI latest" fast path, so it measures a dependency pass whose
    # steps all skip; only the offline run (probe refused) can reach a fast path. Recorded so the
    # report says which, instead of leaving a 40 s "no-op" unexplained.
    "update_path": steps.get("update_path"), "update_path_reason": steps.get("update_path_reason"),
    "update_path_evidence": steps.get("update_path_evidence"),
    # Under `--refuse` the PyPI probe MUST fail. A run that still names PyPI's latest version
    # reached it outside the CONNECT proxy, so the offline measurement is not offline at all.
    "pypi_probe_answered": steps.get("pypi_probe_answered"),
}, open(f"{d}/extra.json", "w"), indent=1)
PY
    local step_rc=0
    finish_step "$d" "$label" "$t0" "$rc" "$rx0" "$VENV" "$ev" "$ez" "$d/extra.json" || step_rc=$?
    # EXPECT_IDEMPOTENT=0 (a branch that predates the fix under test) keeps the evidence and lets
    # report.py record the verdict instead of failing here. The step is still reported.
    if [ "$step_rc" != 0 ] && [ "${EXPECT_IDEMPOTENT:-1}" = 0 ]; then
        echo "[harness] $label failed but EXPECT_IDEMPOTENT=0: recorded, not fatal" >&2
        return 0
    fi
    return $step_rc
}

# Break exactly one thing, then update: exactly the affected step must run.
cmd_fault() {
    local kind=$1; shift
    local target="${NOOP_TARGET:-pr}" arg
    while [ $# -gt 0 ]; do
        arg=$1
        case "$arg" in
            --target) shift; target=$1 ;;
            *) echo "[harness] unknown fault flag $arg" >&2; return 2 ;;
        esac
        shift
    done
    local label="fault-$kind" d="$OUT/fault-$kind"; mkdir -p "$d"
    local ev ez; ev=$(installed_version unsloth); ez=$(installed_version unsloth_zoo)
    [ -n "$ev" ] || { echo "[harness] nothing installed in $VENV" >&2; return 1; }
    snapshot "$d/before" "$label:before" > "$d/before.line"
    idem capture --venv "$VENV" --studio-home "$STUDIO" --out "$d/state_before.json" > /dev/null
    idem inject "$kind" --venv "$VENV" --studio-home "$STUDIO" --out "$d/injection.json"
    start_proxy "$d"
    echo "[harness] === $label  (fault $kind injected, then studio update; installed unsloth==$ev)"
    local t0 rx0 rc; t0=$(now_s); rx0=$(rx_bytes)
    ( cd "$FAKE_HOME" && eval "$(child_env "$target" UNSLOTH_TAURI_UPDATE=1 SKIP_STUDIO_FRONTEND=1 "UNSLOTH_DESKTOP_BACKEND_VERSION=$ev")" \
        "$VENV/bin/python" -I -X utf8 -m unsloth_cli studio update ) 2>&1 | ts_filter | tee "$d/log.txt" >/dev/null
    rc=${PIPESTATUS[0]}
    idem capture --venv "$VENV" --studio-home "$STUDIO" --out "$d/state_after.json" > /dev/null
    idem compare "$d/state_before.json" "$d/state_after.json" --out "$d/repair.json" > /dev/null
    idem steps --log "$d/log.txt" --out "$d/steps.json" > /dev/null 2>&1 || true
    "$PY3" - "$d" "$kind" <<'PY'
import json, sys
d, kind = sys.argv[1:]
steps = json.load(open(f"{d}/steps.json"))
rep = json.load(open(f"{d}/repair.json"))
inj = json.load(open(f"{d}/injection.json"))
json.dump({"fault_kind": kind, "fault_injection": inj,
           "fault_steps": steps["steps"], "fault_steps_ran": steps["ran"],
           "repair_reasons": rep["reasons"], "freeze_diff": rep["freeze_diff"],
           "freeze_diff_empty": rep["freeze_diff_empty"]},
          open(f"{d}/extra.json", "w"), indent=1)
PY
    finish_step "$d" "$label" "$t0" "$rc" "$rx0" "$VENV" "$ev" "$ez" "$d/extra.json"
}

# PR A: an old Tauri shell asking a new CLI to stage must be refused, loudly and without a stage dir.
cmd_old_shell_stage() {
    local shell_ver="0.1.807-beta" arg
    while [ $# -gt 0 ]; do
        arg=$1
        case "$arg" in
            --shell) shift; shell_ver=$1 ;;
            *) echo "[harness] unknown old-shell-stage flag $arg" >&2; return 2 ;;
        esac
        shift
    done
    local label="old-shell-stage" d="$OUT/old-shell-stage"; mkdir -p "$d"
    local ev ez; ev=$(installed_version unsloth); ez=$(installed_version unsloth_zoo)
    rm -f "$STUDIO/.update-failed.json"
    snapshot "$d/before" "$label:before" > "$d/before.line"
    start_proxy "$d"
    echo "[harness] === $label  (UNSLOTH_TAURI_SHELL_VERSION=$shell_ver against installed unsloth==$ev; exit 1 expected)"
    local t0 rx0 rc; t0=$(now_s); rx0=$(rx_bytes)
    ( cd "$FAKE_HOME" && eval "$(child_env "${NOOP_TARGET:-pr}" UNSLOTH_TAURI_UPDATE=1 SKIP_STUDIO_FRONTEND=1 \
        "UNSLOTH_TAURI_SHELL_VERSION=$shell_ver" "UNSLOTH_DESKTOP_BACKEND_VERSION=$ev")" \
        "$VENV/bin/python" -I -X utf8 -m unsloth_cli studio update --stage ) 2>&1 | ts_filter | tee "$d/log.txt" >/dev/null
    rc=${PIPESTATUS[0]}
    "$PY3" - "$d" "$STUDIO" "$shell_ver" "$rc" <<'PY'
import json, os, sys
d, studio, shell_ver, rc = sys.argv[1:]
log = open(f"{d}/log.txt", errors="replace").read()
needle = "[TAURI:ERROR] background staging is no longer supported"
failed = os.path.join(studio, ".update-failed.json")
payload = None
if os.path.exists(failed):
    try:
        payload = json.loads(open(failed, errors="replace").read())
    except json.JSONDecodeError:
        payload = {"_unparseable": open(failed, errors="replace").read()[:500]}
stage = os.path.join(studio, ".update-stage")
res = {
    "shell_version": shell_ver, "expected_exit_code": "1",
    "tauri_error_present": needle in log,
    "tauri_error_lines": [ln.strip()[:200] for ln in log.splitlines() if "[TAURI:ERROR]" in ln][:5],
    "update_failed_json_exists": os.path.exists(failed), "update_failed_json": payload,
    "update_failed_fields_are_strings": bool(payload) and all(
        isinstance(v, str) for v in payload.values() if not isinstance(v, (dict, list))),
    "stage_dir_exists": os.path.exists(stage),
}
res["pass"] = int(rc) == 1 and res["tauri_error_present"] and res["update_failed_json_exists"] and not res["stage_dir_exists"]
json.dump(res, open(f"{d}/extra.json", "w"), indent=1)
print("[harness] old-shell-stage:", json.dumps({k: res[k] for k in ("tauri_error_present", "update_failed_json_exists", "stage_dir_exists", "pass")}))
PY
    local step_rc=0
    EXPECT_RC=1; finish_step "$d" "$label" "$t0" "$rc" "$rx0" "$VENV" "$ev" "$ez" "$d/extra.json" || step_rc=$?
    EXPECT_RC=0
    return $step_rc
}

# PR A compat: the 806 CLI's own --stage still works against the PR wheel. Run AFTER
# `update 806 pr --stage`; asserts the live venv is untouched and the staged venv can import the
# shim the old Tauri shell calls at activation.
cmd_nested_old_stage() {
    local against="${1:-$OUT/update-806-to-pr-staged/before}"
    local label="nested-old-stage" d="$OUT/nested-old-stage"; mkdir -p "$d"
    local t0; t0=$(now_s)
    if [ ! -f "$against/snapshot.json" ]; then
        echo "[harness] nested-old-stage: no baseline snapshot at $against" >&2; return 1
    fi
    rm -rf "$d/before"; cp -R "$against" "$d/before"
    snapshot "$d/after" "$label:after" "$VENV" > "$d/after.line"
    "$PY3" "$BISECT_DIR/snapshot.py" diff "$d/before" "$d/after" --out "$d/diff.json" > /dev/null 2>&1 || true
    local stage_py="$STAGE/unsloth_studio/bin/python" import_rc=127 import_out=""
    if [ -x "$stage_py" ]; then
        import_out=$("$stage_py" -I -c "from unsloth_cli._studio_stage import finalize_for_activation" 2>&1)
        import_rc=$?
    fi
    printf '%s\n' "$import_out" > "$d/stage_import.txt"
    "$PY3" - "$d" "$STAGE" "$import_rc" "$(now_s)" "$t0" <<'PY'
import json, os, sys
d, stage, import_rc, t1, t0 = sys.argv[1:]
diff = json.load(open(f"{d}/diff.json")) if os.path.exists(f"{d}/diff.json") else {}
live_unchanged = not (diff.get("packages_changed") or diff.get("packages_added") or diff.get("packages_removed"))
res = {"label": "nested-old-stage", "seconds_total": round(float(t1) - float(t0), 1), "exit_code": 0,
       "stage_dir_exists": os.path.isdir(stage),
       "staged_venv_exists": os.path.isdir(os.path.join(stage, "unsloth_studio")),
       "live_venv_unchanged": live_unchanged,
       "live_venv_diff": {k: diff.get(k) for k in ("packages_changed", "packages_added", "packages_removed")},
       "stage_import_exit_code": int(import_rc),
       "stage_import_output": open(f"{d}/stage_import.txt", errors="replace").read().strip()[:400],
       "diff": diff}
res["pass"] = res["staged_venv_exists"] and res["live_venv_unchanged"] and res["stage_import_exit_code"] == 0
json.dump(res, open(f"{d}/summary.json", "w"), indent=1)
print("[harness] nested-old-stage:", json.dumps({k: res[k] for k in
      ("staged_venv_exists", "live_venv_unchanged", "stage_import_exit_code", "pass")}))
sys.exit(0 if res["pass"] else 1)
PY
}

# PR A: prepare the legacy leftover state the Tauri cleanup has to reconcile, and record it after.
cmd_leftover_seed() {
    local d="$OUT/leftover-seed"; mkdir -p "$d"
    idem leftovers seed --studio-home "$STUDIO" --out "$d/seeded.json"
    "$PY3" -c "import json;s=json.load(open('$d/seeded.json'));json.dump({'label':'leftover-seed','exit_code':0,'seeded':s['seeded']},open('$d/summary.json','w'),indent=1)"
}
cmd_leftover_check() {
    local d="$OUT/leftover-check"; mkdir -p "$d"
    idem leftovers check --studio-home "$STUDIO" --out "$d/state.json"
    "$PY3" -c "import json;s=json.load(open('$d/state.json'));json.dump({'label':'leftover-check','exit_code':0,'leftovers_present':s['still_present'],'leftovers_gone':s['gone'],'leftovers':s['exists']},open('$d/summary.json','w'),indent=1)"
}

# PR C: prefetch into the uv cache only. The live venv must not move; exit 2 means the installed
# CLI predates PR C (recorded as unsupported, not as a failure).
cmd_prefetch() {
    local target="pr2" arg
    while [ $# -gt 0 ]; do
        arg=$1
        case "$arg" in
            --target) shift; target=$1 ;;
            *) echo "[harness] unknown prefetch flag $arg" >&2; return 2 ;;
        esac
        shift
    done
    local label="prefetch-$target" d="$OUT/prefetch-$target"; mkdir -p "$d"
    local ev ez tv; ev=$(installed_version unsloth); ez=$(installed_version unsloth_zoo)
    tv=$(target_version "$target") || return 1
    snapshot "$d/before" "$label:before" > "$d/before.line"
    idem capture --venv "$VENV" --studio-home "$STUDIO" --out "$d/state_before.json" > /dev/null
    start_proxy "$d"
    echo "[harness] === $label  (prefetch-update toward unsloth==$tv; live venv stays at $ev)"
    local t0 rx0 rc; t0=$(now_s); rx0=$(rx_bytes)
    ( cd "$FAKE_HOME" && eval "$(child_env "$target" UNSLOTH_TAURI_UPDATE=1 SKIP_STUDIO_FRONTEND=1 "UNSLOTH_DESKTOP_BACKEND_VERSION=$tv")" \
        "$VENV/bin/python" -I -X utf8 -m unsloth_cli studio prefetch-update ) 2>&1 | ts_filter | tee "$d/log.txt" >/dev/null
    rc=${PIPESTATUS[0]}
    idem capture --venv "$VENV" --studio-home "$STUDIO" --out "$d/state_after.json" > /dev/null
    idem compare "$d/state_before.json" "$d/state_after.json" --out "$d/venv_delta.json" > /dev/null
    idem prefetch --studio-home "$STUDIO" --out "$d/prefetch_state.json" > /dev/null
    "$PY3" - "$d" "$target" "$tv" "$rc" <<'PY'
import json, sys
d, target, tv, rc = sys.argv[1:]
delta = json.load(open(f"{d}/venv_delta.json"))
pf = json.load(open(f"{d}/prefetch_state.json"))
res = {"prefetch_target": target, "prefetch_target_version": tv,
       "prefetch_supported": int(rc) != 2, "prefetch_exit_code": int(rc),
       "prefetch_dir_exists": pf["dir_exists"], "prefetch_marker_exists": pf["marker_exists"],
       "prefetch_marker": pf["marker"], "prefetch_dir_bytes": (pf.get("dir") or {}).get("bytes"),
       "live_venv_unchanged": delta["freeze_diff_empty"],
       "live_venv_reasons": delta["reasons"]}
json.dump(res, open(f"{d}/extra.json", "w"), indent=1)
print("[harness] prefetch:", json.dumps({k: res[k] for k in
      ("prefetch_supported", "prefetch_exit_code", "prefetch_marker_exists", "live_venv_unchanged")}))
PY
    # The live venv must still hold the OLD version: that is the assertion, not the new one.
    local step_rc=0
    EXPECT_RC=any; finish_step "$d" "$label" "$t0" "$rc" "$rx0" "$VENV" "$ev" "$ez" "$d/extra.json" || step_rc=$?
    EXPECT_RC=0
    return $step_rc
}

cmd_activate() {  # mirrors staged_update.rs activate_ready: swap runtime entries, drop the previous tree
    local label="activate-$1" d="$OUT/activate-$1"; mkdir -p "$d"
    local t0; t0=$(now_s)
    "$PY3" - "$STUDIO" <<'PY' 2>&1 | tee "$d/log.txt"
import json, os, shutil, sys, time
home, stage, prev = sys.argv[1], os.path.join(sys.argv[1], ".update-stage"), os.path.join(sys.argv[1], ".update-previous")
ready = os.path.join(stage, "READY.json")
if not os.path.exists(ready):
    print("[activate] no READY.json; nothing to activate"); sys.exit(2)
print("[activate] READY:", open(ready).read().strip())
shutil.rmtree(prev, ignore_errors=True); os.makedirs(prev)
t = time.time(); swapped = []
for name in ["unsloth_studio", ".venv_t5_530", ".venv_t5_550", ".venv_t5_510", "node", "llama.cpp", "whisper.cpp"]:
    s, l = os.path.join(stage, name), os.path.join(home, name)
    if not os.path.exists(s): continue
    if os.path.exists(l): os.rename(l, os.path.join(prev, name))
    os.rename(s, l); swapped.append(name)
marker = os.path.join(stage, "uv-cache-dir")
if os.path.exists(marker):
    os.makedirs(os.path.join(home, "cache"), exist_ok=True)
    shutil.copy(marker, os.path.join(home, "cache", "uv-cache-dir"))
print(f"[activate] swapped {swapped} in {time.time()-t:.2f}s")
t = time.time(); shutil.rmtree(stage, ignore_errors=True); shutil.rmtree(prev, ignore_errors=True)
print(f"[activate] removed previous tree in {time.time()-t:.1f}s")
PY
    local rc=${PIPESTATUS[0]}
    "$PY3" -c "import json; json.dump({'label':'$label','seconds_total':round($(now_s)-$t0,1),'exit_code':$rc}, open('$d/summary.json','w'))"
    return $rc
}

cmd_launch() {
    local label="launch-$1" d="$OUT/launch-$1" secs=${2:-150}; mkdir -p "$d"
    local port=$((20000 + RANDOM % 20000))
    # A PR install self-heals against the PR layer, not against a release it never had.
    local pins_target="${PINS_TARGET:-807}"
    [ -s "$GEN_PINS/pr.json" ] && pins_target="${PINS_TARGET:-pr}"
    start_proxy "$d"
    echo "[harness] === $label  (headless studio run on :$port for up to ${secs}s, pin layer $pins_target)"
    local t0; t0=$(now_s)
    # Every word after `eval` is re-parsed, so the trampoline has to be quoted the way child_env
    # quotes its own words. Passing `-c "import os,sys; os.setsid(); ..."` unquoted made the shell
    # split on the `;`, and the launch died with "syntax error near unexpected token `;'" before
    # the server ever started -- which is why every POSIX leg reported time_to_health=none while
    # Windows (a different script) reported a healthy server (observed, run 34331078036).
    local -a launch_cmd=("$PY3" -c 'import os,sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])'
        "$VENV/bin/python" -I -X utf8 -m unsloth_cli studio --api-only -p "$port" -H 127.0.0.1)
    ( cd "$FAKE_HOME" && eval "exec $(child_env "$pins_target" UNSLOTH_TAURI_UPDATE=1) $(printf '%q ' "${launch_cmd[@]}")" ) > "$d/server.log" 2>&1 &
    local spid=$! healthy="" i
    for i in $(seq 1 "$secs"); do
        if curl -fsS -m 2 "http://127.0.0.1:$port/api/health" > "$d/health.json" 2>/dev/null; then healthy=$("$PY3" -c "print(round($(now_s)-$t0,1))"); break; fi
        kill -0 "$spid" 2>/dev/null || break
        sleep 1
    done
    # keep it alive a while longer to catch background repairs, then stop the whole group
    [ -n "$healthy" ] && sleep 45
    [ "$spid" != "$$" ] && kill -TERM -- "-$spid" 2>/dev/null; sleep 3; [ "$spid" != "$$" ] && kill -KILL -- "-$spid" 2>/dev/null
    stop_proxy
    local proxy_json="{}"; [ -s "$d/proxy.jsonl" ] && proxy_json=$("$PY3" "$BISECT_DIR/connect_proxy.py" summary "$d/proxy.jsonl")
    grep -n -i -E 'self-heal|repair|installing|uv pip|pip install|downloading|warm' "$d/server.log" | head -40 > "$d/install_activity.txt"
    "$PY3" - "$d" "$label" "${healthy:-}" "$secs" "$proxy_json" <<'PY'
import json, sys
d, label, healthy, secs, proxy_json = sys.argv[1:]
log = open(f"{d}/server.log", errors="replace").read()
json.dump({"label": label,
           "seconds_to_health": float(healthy) if healthy else None,
           "health_timeout_seconds": int(secs),
           # A server that never answered is either slow or never started, and a summary carrying
           # only `null` cannot tell those apart. The head of its log can.
           "server_log_lines": len(log.splitlines()),
           "server_log_head": [ln[:300] for ln in log.splitlines()[:5]],
           "proxy": json.loads(proxy_json),
           "install_activity_lines": sum(1 for _ in open(f"{d}/install_activity.txt"))},
          open(f"{d}/summary.json", "w"), indent=1)
PY
    [ -n "$healthy" ] || echo "[harness] $label: the server never answered /api/health in ${secs}s; server.log line 1: $(head -1 "$d/server.log" 2>/dev/null)" >&2
    echo "[harness] $label: time_to_health=${healthy:-none}s proxy_down=$("$PY3" -c "import json; print(round(json.loads('''$proxy_json''').get('total_bytes_down',0)/1e6))")MB activity_lines=$(wc -l < "$d/install_activity.txt")"
}

cmd_uninstall() {
    local n=$1 label="uninstall-$2" d="$OUT/uninstall-$2"; mkdir -p "$d/bin"
    fetch_tag_script "$n" uninstall.sh || return 1
    # pkill guard: this box is shared, so only patterns naming our isolated root may kill anything.
    cat > "$d/bin/pkill" <<EOF
#!/bin/sh
case "\$*" in *"$FAKE_HOME"*) exec /usr/bin/pkill "\$@" ;; *) echo "[pkill-guard] refused: \$*" >&2; exit 1 ;; esac
EOF
    chmod +x "$d/bin/pkill"
    snapshot "$d/before" "$label:before" > "$d/before.line"
    local t0; t0=$(now_s)
    ( cd "$FAKE_HOME" && eval "$(child_env "$n" "PATH=$d/bin:$PATH")" sh "$SCRIPTS_DIR/$n/uninstall.sh" ) 2>&1 | ts_filter | tee "$d/log.txt" >/dev/null
    local rc=${PIPESTATUS[0]}
    snapshot "$d/after" "$label:after" > "$d/after.line"
    "$PY3" "$BISECT_DIR/snapshot.py" diff "$d/before" "$d/after" --out "$d/diff.json" >/dev/null 2>&1
    "$PY3" - "$d" "$label" "$rc" "$FAKE_HOME" "$STUDIO" <<'PY'
import json, os, sys
d, label, rc, home, studio = sys.argv[1:]
diff = json.load(open(f"{d}/diff.json"))
gl = f"{home}/.cache/uv"
res = {"label": label, "exit_code": int(rc), "studio_root_exists": os.path.exists(studio),
       "global_uv_cache_exists": os.path.exists(gl), "global_uv_cache_growth_bytes": diff["cache_growth_bytes"].get(gl),
       "studio_cache_growth_bytes": diff["cache_growth_bytes"].get(f"{studio}/cache/uv"),
       "seconds_total": None}
json.dump(res, open(f"{d}/summary.json", "w"), indent=1); print("[harness]", json.dumps(res))
PY
    return $rc
}

case "${1:-}" in
    install) shift; cmd_install "$@" ;;
    update) shift; cmd_update "$@" ;;
    activate) shift; cmd_activate "$@" ;;
    launch) shift; cmd_launch "$@" ;;
    uninstall) shift; cmd_uninstall "$@" ;;
    noop-update) shift; cmd_noop_update "$@" ;;
    fault) shift; cmd_fault "$@" ;;
    old-shell-stage) shift; cmd_old_shell_stage "$@" ;;
    nested-old-stage) shift; cmd_nested_old_stage "$@" ;;
    leftover-seed) shift; cmd_leftover_seed "$@" ;;
    leftover-check) shift; cmd_leftover_check "$@" ;;
    prefetch) shift; cmd_prefetch "$@" ;;
    *) sed -n 2,22p "$0"; exit 2 ;;
esac
