"""Does an accented path in a Seatbelt profile actually work on Darwin?

The claim under test is that json.dumps' default \\uXXXX spelling produces a rule
that matches nothing, because SBPL is TinyScheme and has no \\u escape. Asserted
against the kernel, not the manual: each profile is handed to the real
sandbox-exec and a write into the accented workdir is attempted.
"""
import json, os, subprocess, sys, tempfile

sys.path.insert(0, "studio/backend")
os.environ.setdefault("UNSLOTH_STUDIO_HOME", tempfile.mkdtemp(prefix="us-home-"))
from core.inference import sandbox_macos as backend  # noqa: E402

base = tempfile.mkdtemp(prefix="us-sbpl-", dir="/tmp")
workdir = os.path.join(base, "session-café")
private_tmp = os.path.join(base, "tmp-über")
for path in (workdir, private_tmp):
    os.makedirs(path, exist_ok=True)

profile = backend.build_profile(
    workdir=workdir, private_tmp=private_tmp, runtime_paths=backend.runtime_read_paths(workdir)
)
escaped = profile.replace(workdir, json.dumps(workdir)[1:-1]).replace(
    private_tmp, json.dumps(private_tmp)[1:-1]
)
assert "\\u00" in escaped and "\\u00" not in profile, "the two spellings are not different"

target = os.path.join(workdir, "canary.txt")
for label, text in (("json default (\\u00e9)", escaped), ("raw utf-8 (this PR)", profile)):
    if os.path.exists(target):
        os.remove(target)
    proc = subprocess.run(
        ["/usr/bin/sandbox-exec", "-p", text, "/bin/sh", "-c", f"echo written > {target!r}"],
        capture_output=True, text=True, timeout=60,
    )
    wrote = os.path.exists(target)
    err = (proc.stderr or proc.stdout).strip().splitlines()
    print(f"{label:<24} rc={proc.returncode} wrote={wrote} {err[:1]}")

print("\nprofile excerpt:")
for line in profile.splitlines():
    if "caf" in line:
        print("  ", line[:220])
        break
