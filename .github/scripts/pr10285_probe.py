"""Staging-only probe driver for PR 10285 (Studio OS isolation).

Run from studio/backend of the checkout under test:

    python .github/scripts/pr10285_probe.py capability [--label L]   # one JSON line: PR10285_CAP {...}
    python .github/scripts/pr10285_probe.py escape                    # terminal-tool escape probe, PR10285_ESC lines
    python .github/scripts/pr10285_probe.py python-escape             # python-tool escape probe (static scanner may block)
    python .github/scripts/pr10285_probe.py limited-survivor          # Limited-mode detached child check

Every line the later analysis needs starts with PR10285_ so it survives log noise.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time


def _bootstrap() -> None:
    backend = os.getcwd()
    if backend not in sys.path:
        sys.path.insert(0, backend)
    os.environ.setdefault("UNSLOTH_STUDIO_HOME", tempfile.mkdtemp(prefix = "pr10285-home-"))
    os.environ["PR10285_HF_CANARY"] = "hf_canary_should_never_leak"


def capability(label: str) -> int:
    _bootstrap()
    from core.inference.os_sandbox import capability_snapshot

    t0 = time.time()
    c = capability_snapshot(force = True)
    out = {
        "label": label,
        "seconds": round(time.time() - t0, 2),
        "environment": c.environment,
        "backend": c.backend,
        "protection_state": c.protection_state,
        "available": c.available,
        "qualified": c.qualified,
        "transient": c.transient,
        "retryable": getattr(c, "retryable", None),
        "profile_id": c.profile_id,
        "limitations": list(c.limitations),
        "reason": c.reason[:400],
        "remediation": (getattr(c, "remediation", None) or "")[:400],
        "probe_generation": c.probe_generation[:16],
    }
    print("PR10285_CAP " + json.dumps(out), flush = True)
    return 0


def escape() -> int:
    _bootstrap()
    from core.inference import tools
    from core.inference.os_sandbox import capability_snapshot

    c = capability_snapshot(force = True)
    print("PR10285_ESC_STATE " + json.dumps({"protection_state": c.protection_state, "profile_id": c.profile_id}), flush = True)
    secret = tempfile.NamedTemporaryFile("w", prefix = "pr10285-secret-", delete = False)
    secret.write("SECRET")
    secret.close()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    if sys.platform == "win32":
        # cmd only: the isolated Terminal tool runs cmd, and powershell is on the
        # static blocklist, so it would be refused before the sandbox is involved.
        term = textwrap.dedent(
            f"""
            echo PR10285_ESC uid=%USERNAME%
            echo PR10285_ESC secret_read=& type "{secret.name}" 2>&1
            echo PR10285_ESC userprofile_list=& dir /b "%USERPROFILE%" 2>&1 | findstr /n . | findstr "^1:"
            echo PR10285_ESC write_windows_temp=& (echo x > C:\\Windows\\Temp\\pr10285.txt && echo WRITTEN) 2>&1
            echo PR10285_ESC write_program_files=& (echo x > "C:\\Program Files\\pr10285.txt" && echo WRITTEN) 2>&1
            echo PR10285_ESC write_workdir=& (echo x > ok.txt && echo WRITTEN) 2>&1
            echo PR10285_ESC tcp_loopback=& python -c "import socket; s=socket.create_connection(('127.0.0.1', {port}), timeout=3); print('CONNECTED')" 2>&1
            echo PR10285_ESC tcp_internet=& python -c "import socket; s=socket.create_connection(('1.1.1.1', 443), timeout=3); print('CONNECTED')" 2>&1
            echo PR10285_ESC devnull=& python -c "open('nul','rb').close(); print('OPENED')" 2>&1
            echo PR10285_ESC canary=& set PR10285 2>&1
            echo PR10285_ESC env_count=& set | find /c "="
            echo PR10285_ESC integrity=& whoami /groups | findstr /i "Mandatory Label" 2>&1
            echo PR10285_ESC appcontainer=& whoami /groups | findstr /i "APPLICATION PACKAGE" 2>&1
            start /b cmd /c "ping -n 1285 127.0.0.1 > nul"
            echo PR10285_ESC detached_spawned=yes
            """
        )
    else:
        term = textwrap.dedent(
            f"""
            echo "PR10285_ESC uid=$(id -u) $(id -un)"
            echo "PR10285_ESC secret_read=$(cat {secret.name} 2>&1 | head -c 60)"
            echo "PR10285_ESC home_list=$(ls /home 2>&1 | head -c 80)"
            echo "PR10285_ESC ssh_list=$(ls -a ~/.ssh /home/runner/.ssh 2>&1 | head -c 80)"
            echo "PR10285_ESC write_etc=$( (echo x > /etc/pr10285) 2>&1 | head -c 80)"
            echo "PR10285_ESC write_root=$( (echo x > /pr10285) 2>&1 | head -c 80)"
            echo "PR10285_ESC write_usr=$( (echo x > /usr/pr10285) 2>&1 | head -c 80)"
            echo "PR10285_ESC write_workdir=$( (echo x > ok.txt && echo written) 2>&1 | head -c 80)"
            echo "PR10285_ESC tcp_loopback=$(python3 -c 'import socket; s=socket.create_connection((\"127.0.0.1\", {port}), timeout=3); print(\"CONNECTED\")' 2>&1 | tail -1 | head -c 80)"
            echo "PR10285_ESC tcp_internet=$(python3 -c 'import socket; s=socket.create_connection((\"1.1.1.1\", 443), timeout=3); print(\"CONNECTED\")' 2>&1 | tail -1 | head -c 80)"
            echo "PR10285_ESC dns=$(python3 -c 'import socket; print(socket.getaddrinfo(\"example.com\", 443)[0][4])' 2>&1 | tail -1 | head -c 80)"
            echo "PR10285_ESC ifaces=$(ls /sys/class/net 2>&1 | tr '\\n' ' ')"
            echo "PR10285_ESC proc1_environ=$(cat /proc/1/environ 2>&1 | head -c 60)"
            echo "PR10285_ESC proc_count=$(ls /proc | grep -c '^[0-9]')"
            echo "PR10285_ESC dev=$(ls /dev | tr '\\n' ' ')"
            echo "PR10285_ESC gpu_nodes=$(ls /dev/dri /dev/kfd /dev/nvidia* 2>&1 | head -c 80)"
            echo "PR10285_ESC canary=$(env | grep -c hf_canary)"
            echo "PR10285_ESC env_count=$(env | wc -l)"
            echo "PR10285_ESC caps=$(grep -E 'CapEff|NoNewPrivs|Seccomp:' /proc/self/status | tr '\\t' '=' | tr '\\n' ' ')"
            echo "PR10285_ESC nested_userns=$(unshare -U true 2>&1 && echo nested-userns-ALLOWED)"
            echo "PR10285_ESC nested_userns_python=$(python3 -c 'import ctypes,os; l=ctypes.CDLL(None, use_errno=True); r=l.unshare(0x10000000); print("ALLOWED" if r==0 else "denied errno=%d" % ctypes.get_errno())' 2>&1 | tail -1)"
            setsid sleep 1285 >/dev/null 2>&1 &
            echo "PR10285_ESC detached_spawned=yes"
            """
        )
    out = tools._bash_exec(term, session_id = "pr10285-escape", timeout = 90)
    print("PR10285_ESC_RAW " + json.dumps(str(out)[:1500]), flush = True)
    for line in str(out).splitlines():
        if "PR10285_ESC" in line or "Execution error" in line or "Blocked" in line:
            print(line.strip()[:400], flush = True)
    time.sleep(2)
    if sys.platform == "win32":
        ps = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process -Filter \"Name='PING.EXE'\" | Select-Object -ExpandProperty CommandLine"], capture_output = True, text = True).stdout
        survivors = "1285" in ps
    else:
        ps = subprocess.run(["pgrep", "-u", str(os.getuid()), "-af", "sleep 1285"], capture_output = True, text = True).stdout
        survivors = bool(ps.strip())
    print("PR10285_ESC_SURVIVORS " + json.dumps({"survivors": survivors, "detail": ps.strip()[:200]}), flush = True)
    return 0


def python_escape() -> int:
    _bootstrap()
    from core.inference import tools

    code = textwrap.dedent(
        """
        import os, json
        r = {}
        try:
            r["dev"] = sorted(os.listdir("/dev" if os.name != "nt" else "C:\\\\"))[:20]
        except Exception as e:
            r["dev"] = "denied:" + type(e).__name__
        r["uid"] = getattr(os, "getuid", lambda: None)()
        r["cwd"] = os.getcwd()
        r["env_count"] = len(os.environ)
        r["canary"] = any("hf_canary" in v for v in os.environ.values())
        r["home_readable"] = os.path.isdir(os.path.expanduser("~"))
        print("PR10285_PYESC " + json.dumps(r))
        """
    )
    out = tools._python_exec(code, session_id = "pr10285-pyescape", timeout = 90)
    for line in str(out).splitlines():
        if "PR10285_PYESC" in line or "Execution error" in line or "unsafe" in line:
            print(line.strip()[:400], flush = True)
    return 0


def limited_survivor() -> int:
    """Limited mode: does a detached grandchild outlive the call?"""
    _bootstrap()
    from core.inference import tools
    from core.inference import tool_isolation

    ui_session = "pr10285-ui"
    snapshot = tool_isolation.capability_snapshot(force = True)
    if snapshot.available:
        print("PR10285_LIMITED " + json.dumps({"skipped": "os isolation available; Limited not offered"}), flush = True)
        return 0
    grant = tool_isolation._LIMITED_GRANTS.issue(
        current_subject = "pr10285", tool_ui_session_id = ui_session, probe_generation = snapshot.probe_generation
    ) if hasattr(tool_isolation, "_LIMITED_GRANTS") else None
    if grant is None:
        print("PR10285_LIMITED " + json.dumps({"skipped": "no grant store"}), flush = True)
        return 0
    marker = "pr10285-limited-marker"
    if sys.platform == "win32":
        cmd = f'start /b cmd /c "ping -n 1286 127.0.0.1 > nul" & echo {marker}'
    else:
        cmd = f"setsid sleep 1286 >/dev/null 2>&1 & echo {marker}"
    try:
        out = tools._bash_exec(
            cmd, session_id = "pr10285-limited", timeout = 30,
            tool_execution_mode = "limited", limited_grant = grant.token,
            current_subject = "pr10285", tool_ui_session_id = ui_session,
        )
    except TypeError as exc:
        print("PR10285_LIMITED " + json.dumps({"skipped": f"signature: {exc}"[:200]}), flush = True)
        return 0
    time.sleep(2)
    if sys.platform == "win32":
        ps = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process -Filter \"Name='PING.EXE'\" | Select-Object -ExpandProperty CommandLine"], capture_output = True, text = True).stdout
        survivors = "1286" in ps
    else:
        ps = subprocess.run(["pgrep", "-u", str(os.getuid()), "-af", "sleep 1286"], capture_output = True, text = True).stdout
        survivors = bool(ps.strip())
    print("PR10285_LIMITED " + json.dumps({"ran": marker in str(out), "survivors": survivors, "tail": str(out)[-200:]}), flush = True)
    return 0


def _record_collector():
    records = []
    return records, records.append


def network_allowlist() -> int:
    """Required mode with the allowlist proxy: approved host reachable, others refused."""
    _bootstrap()
    from core.inference import tools
    from core.inference.os_sandbox import capability_snapshot

    c = capability_snapshot(force = True)
    policies = list(getattr(c, "network_policies", ()) or ())
    print("PR10285_NET_CAP " + json.dumps({
        "protection_state": c.protection_state, "network_policies": policies,
        "allowlist": list(getattr(c, "network_allowlist", ()) or ())[:6],
    }), flush = True)
    if "allowlist" not in policies:
        print("PR10285_NET " + json.dumps({"skipped": "allowlist not offered here"}), flush = True)
        return 0
    if sys.platform == "win32":
        fetch = "python -c \"import urllib.request; print('PR10285_NET pypi=' + str(urllib.request.urlopen('https://pypi.org/simple/pip/', timeout=30).status))\""
        denied = "python -c \"import urllib.request; urllib.request.urlopen('https://example.com/', timeout=15)\" 2>&1 | findstr /i \"403 error\""
        direct = "python -c \"import socket; socket.create_connection(('1.1.1.1', 443), timeout=5); print('PR10285_NET direct=CONNECTED')\" 2>&1 | findstr /i \"PR10285 error\""
        envs = "set | findstr /i PROXY"
    else:
        fetch = "python3 -c \"import urllib.request; print('PR10285_NET pypi=' + str(urllib.request.urlopen('https://pypi.org/simple/pip/', timeout=30).status))\""
        denied = "python3 -c \"import urllib.request; urllib.request.urlopen('https://example.com/', timeout=15)\" 2>&1 | tail -1 | sed 's/^/PR10285_NET denied=/'"
        direct = "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 443), timeout=5); print('CONNECTED')\" 2>&1 | tail -1 | sed 's/^/PR10285_NET direct=/'"
        envs = "env | grep -i proxy | sed -E 's#//[^@]*@#//<cred>@#' | sed 's/^/PR10285_NET env=/'"
    script = "\n".join([fetch, denied, direct, envs])
    records, collect = _record_collector()
    try:
        out = tools._bash_exec(
            script, session_id = "pr10285-net", timeout = 120,
            network_policy = "allowlist", launch_record_callback = collect,
        )
    except TypeError as exc:
        print("PR10285_NET " + json.dumps({"skipped": f"signature: {exc}"[:200]}), flush = True)
        return 0
    for line in str(out).splitlines():
        if "PR10285_NET" in line or "[network]" in line or "Execution error" in line:
            print(line.strip()[:300], flush = True)
    for record in records:
        d = record.as_dict() if hasattr(record, "as_dict") else dict(record)
        print("PR10285_NET_RECORD " + json.dumps({k: d.get(k) for k in ("backend", "profile_id", "network_policy", "network_allowlist")})[:400], flush = True)
    # Deny policy: the same fetch must fail.
    out2 = tools._bash_exec(fetch, session_id = "pr10285-net-deny", timeout = 60, network_policy = "deny")
    print("PR10285_NET_DENY " + json.dumps(str(out2)[-200:]), flush = True)
    return 0


def limited_token() -> int:
    """Windows Limited mode under the restricted token: reads stay, writes are fenced."""
    _bootstrap()
    from core.inference import tools, tool_isolation
    from core.inference.os_sandbox import capability_snapshot

    c = capability_snapshot(force = True)
    print("PR10285_LT_CAP " + json.dumps({
        "limited_backend": getattr(c, "limited_backend", None),
        "limited_profile_id": getattr(c, "limited_profile_id", None),
        "limited_limitations": list(getattr(c, "limited_limitations", ()) or ()),
        "available": c.available,
    }), flush = True)
    if c.available:
        print("PR10285_LT " + json.dumps({"note": "OS isolation available; Limited grants require unavailable capability"}), flush = True)
        return 0
    ui = "pr10285-ui-lt"
    grant = tool_isolation._LIMITED_GRANTS.issue(current_subject = "pr10285", tool_ui_session_id = ui, probe_generation = c.probe_generation)
    common = dict(tool_execution_mode = "limited", limited_grant = grant.token, current_subject = "pr10285", tool_ui_session_id = ui)
    records, collect = _record_collector()
    secret = os.path.join(os.path.expanduser("~"), "pr10285-secret.txt")
    with open(secret, "w") as fh:
        fh.write("SECRET")
    code = f"""
import os, multiprocessing.connection as mc, subprocess, sys
r = {{}}
r['read_profile'] = open({secret!r}).read()
try:
    open({secret!r}, 'a').write('x'); r['write_profile'] = 'WRITTEN'
except PermissionError: r['write_profile'] = 'denied'
try:
    os.remove({secret!r}); r['delete_profile'] = 'DELETED'
except PermissionError: r['delete_profile'] = 'denied'
open('ok.txt', 'w').write('x'); r['write_workdir'] = 'ok'
open(os.devnull, 'rb').close(); r['devnull'] = 'ok'
a, b = mc.Pipe(); a.close(); b.close(); r['pipe'] = 'ok'
try:
    open(os.path.join(os.environ['SystemRoot'], 'Temp', 'pr10285.txt'), 'w').write('x'); r['write_windows_temp'] = 'WRITTEN'
except PermissionError: r['write_windows_temp'] = 'denied'
r['git'] = subprocess.run(['git', '--version'], capture_output=True, text=True).stdout.strip()[:40]
try:
    import torch; r['torch'] = torch.__version__
except Exception as e: r['torch'] = 'no: ' + type(e).__name__
print('PR10285_LT ' + repr(r))
"""
    try:
        out = tools._python_exec(code, session_id = "pr10285-lt", timeout = 180, launch_record_callback = collect, **common)
    except TypeError as exc:
        print("PR10285_LT " + json.dumps({"skipped": f"signature: {exc}"[:200]}), flush = True)
        return 0
    for line in str(out).splitlines():
        if "PR10285_LT" in line or "Execution error" in line:
            print(line.strip()[:600], flush = True)
    out2 = tools._bash_exec("echo PR10285_LT bash=ok && git --version", session_id = "pr10285-lt2", timeout = 60, launch_record_callback = collect, **common)
    for line in str(out2).splitlines():
        if "PR10285_LT" in line or "git version" in line or "Execution error" in line:
            print(("PR10285_LT " + line.strip())[:300], flush = True)
    for record in records:
        d = record.as_dict() if hasattr(record, "as_dict") else dict(record)
        print("PR10285_LT_RECORD " + json.dumps({k: d.get(k) for k in ("backend", "profile_id", "os_isolation", "limitations")})[:400], flush = True)
    try:
        os.remove(secret)
    except OSError:
        pass
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "capability"
    label = "default"
    if "--label" in sys.argv:
        label = sys.argv[sys.argv.index("--label") + 1]
    if cmd == "capability":
        return capability(label)
    if cmd == "escape":
        return escape()
    if cmd == "python-escape":
        return python_escape()
    if cmd == "limited-survivor":
        return limited_survivor()
    if cmd == "network-allowlist":
        return network_allowlist()
    if cmd == "limited-token":
        return limited_token()
    print(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
