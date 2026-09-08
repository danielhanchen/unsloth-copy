# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's tool subprocesses cannot reach another account's files.

The owner's sandbox is unchanged: no confinement object, no wrapper, the same
pre-exec. A managed account's child runs under Landlock (Linux) or
sandbox-exec (macOS), or is refused when the host offers neither.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from auth import policy
from core.inference import tool_confinement, tools
from utils.account_context import OWNER, AccountContext, run_as

ALICE = AccountContext("alice-id", "alice")
BOB = AccountContext("bob-id", "bob")

LANDLOCK = sys.platform == "linux" and tool_confinement.landlock_abi() > 0


@pytest.fixture(autouse = True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_ALLOW_UNCONFINED_TOOLS", raising = False)
    # The sandbox caps the child at 10000 processes per user; a busy CI host can
    # already be above that, and then bash cannot fork at all. Not what is tested here.
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_NPROC", "4000000")
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(tools, "_workdirs", {})
    monkeypatch.setattr(tools, "_active_sessions", {})
    monkeypatch.setattr(tools, "_pending_removals", {})
    monkeypatch.setattr(tools, "_removing_sessions", set())
    monkeypatch.setattr(tools, "_legacy_sandbox_migrated", True)
    monkeypatch.setattr(tools, "_start_detached_sweep", lambda: None)
    monkeypatch.setattr(tools, "_legacy_sandbox_root", lambda: str(tmp_path / "legacy"))


def _seed(tmp_path: Path) -> dict[str, Path]:
    """One private file per account, plus the owner's auth database."""
    files = {}
    for account in (OWNER, ALICE, BOB):
        workdir = Path(run_as(account, tools._get_workdir, "chat"))
        secret = workdir / f"{account.username}-secret.txt"
        secret.write_text(f"{account.username.upper()}_PRIVATE")
        files[account.username] = secret
    auth_dir = tmp_path / "studio" / "auth"
    auth_dir.mkdir(parents = True, exist_ok = True)
    files["auth"] = auth_dir / "auth.db"
    files["auth"].write_text("OWNER_AUTH_DB")
    return files


def test_owner_spawns_exactly_as_before():
    assert run_as(OWNER, tools._account_confinement) is None
    kwargs = {"preexec_fn": tools._sandbox_preexec}
    argv = ["bash", "-c", "true"]
    assert tools._apply_confinement(None, kwargs, argv) is argv
    assert kwargs["preexec_fn"] is tools._sandbox_preexec


@pytest.mark.skipif(sys.platform == "linux" and LANDLOCK, reason = "host confines with Landlock")
@pytest.mark.skipif(sys.platform == "darwin", reason = "host confines with sandbox-exec")
def test_managed_account_is_refused_without_a_mechanism(monkeypatch):
    monkeypatch.setattr(tool_confinement, "landlock_abi", lambda: 0)
    with pytest.raises(tool_confinement.ToolConfinementUnavailable):
        run_as(ALICE, tools._account_confinement)
    out = run_as(ALICE, tools._bash_exec, "echo hi", session_id = "chat")
    assert out.startswith("Execution error: Code execution is unavailable for this account")


def test_owner_opt_out_runs_unconfined_when_no_mechanism(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(tool_confinement.ToolConfinementUnavailable):
        run_as(ALICE, tools._account_confinement)
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_UNCONFINED_TOOLS", "1")
    confinement = run_as(ALICE, tools._account_confinement)
    assert confinement.mechanism == "unconfined-by-owner"
    assert confinement.preexec is None and confinement.wrap(["x"]) == ["x"]


def test_macos_profile_hides_install_root_then_allows_own_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tool_confinement.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    run_as(ALICE, tools._get_workdir, "chat")
    confinement = run_as(ALICE, tools._account_confinement)
    assert confinement.mechanism == "sandbox-exec"
    argv = confinement.wrap(["bash", "-c", "true"])
    assert argv[:2] == ["/usr/bin/sandbox-exec", "-p"]
    profile = argv[2]
    studio = str((tmp_path / "studio").resolve())
    alice_root = str((tmp_path / "studio" / "accounts" / "alice-id").resolve())
    assert profile.startswith("(version 1)\n(deny default)")
    deny = profile.index(f'(deny file-read* file-write* (subpath "{studio}"))')
    allow = profile.index(f'(allow file-read* (subpath "{alice_root}"))')
    writable = profile.index(f'(allow file-read* file-write* (subpath "{alice_root}/sandbox"))')
    assert (
        deny < allow < writable
    ), "the account roots must be allowed after the install root is denied"
    assert argv[3:] == ["bash", "-c", "true"]


def test_macos_profile_keeps_the_interpreter_readable_under_a_hidden_root(tmp_path, monkeypatch):
    """Later rules win: an interpreter prefix inside the install root must be allowed again."""
    venv = tmp_path / "studio" / "unsloth_studio"
    (venv / "lib").mkdir(parents = True)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tool_confinement.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(tool_confinement, "_interpreter_roots", lambda: [str(venv)])
    run_as(ALICE, tools._get_workdir, "chat")
    profile = run_as(ALICE, tools._account_confinement).wrap(["bash"])[2]

    studio = str((tmp_path / "studio").resolve())
    deny = profile.rindex(f'(deny file-read* file-write* (subpath "{studio}"))')
    allow = profile.rindex(f'(allow file-read* (subpath "{venv}"))')
    assert deny < allow, "the interpreter must be readable after the install root is denied"
    assert f'(allow file-read* (subpath "{studio}"))' not in profile


def test_macos_profile_hides_other_accounts_temporary_roots(tmp_path, monkeypatch):
    """Every account's tmp_root lives under one per-user temp directory, which the
    profile's runtime grants open wholesale; the account's own subtree is allowed back."""
    import tempfile as _tempfile

    from utils.paths import storage_roots

    shared_tmp = tmp_path / "tmp"
    shared_tmp.mkdir()
    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(shared_tmp))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tool_confinement.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    # Bob has been active, so his temporary root exists before Alice's tool starts.
    run_as(BOB, storage_roots.tmp_root).mkdir(parents = True, exist_ok = True)
    (run_as(BOB, storage_roots.tmp_root) / "dataset.jsonl").write_text("BOB_PRIVATE")

    profile = run_as(ALICE, tools._account_confinement).wrap(["bash"])[2]
    base = str((shared_tmp / "unsloth-studio").resolve())
    alice_tmp = str(run_as(ALICE, storage_roots.tmp_root).resolve())
    bob_tmp = str(run_as(BOB, storage_roots.tmp_root).resolve())

    deny = profile.index(f'(deny file-read* file-write* (subpath "{base}"))')
    allow = profile.index(f'(allow file-read* file-write* (subpath "{alice_tmp}"))')
    assert deny < allow, "the account's own temporary root must be allowed after the deny"
    assert f'(subpath "{bob_tmp}")' not in profile


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
@pytest.mark.parametrize("tool", ["terminal", "python"])
def test_managed_child_cannot_read_or_write_other_accounts(tool, tmp_path):
    files = _seed(tmp_path)
    bob_dir = Path(run_as(BOB, tools._get_workdir, "chat"))
    foreign = {
        "owner": files["unsloth"] if "unsloth" in files else files[OWNER.username],
        "alice": files["alice"],
        "auth": files["auth"],
    }

    def run(command_or_code: str) -> str:
        if tool == "terminal":
            return run_as(BOB, tools._bash_exec, command_or_code, session_id = "chat")
        return run_as(BOB, tools._python_exec, command_or_code, session_id = "chat")

    # Own files: readable and writable, relative and absolute.
    if tool == "terminal":
        assert "BOB_PRIVATE" in run("cat bob-secret.txt")
        assert "written" in run("echo written > note.txt && cat note.txt")
    else:
        assert "BOB_PRIVATE" in run("print(open('bob-secret.txt').read())")
        assert "written" in run(
            "open('note.txt','w').write('written'); print(open('note.txt').read())"
        )
    assert (bob_dir / "note.txt").read_text().strip() == "written"

    for name, path in foreign.items():
        rel = os.path.relpath(path, bob_dir)
        for target in (rel, str(path)):
            if tool == "terminal":
                out = run(f"cat {target}; echo rc=$?")
            else:
                out = run(
                    "try:\n"
                    f"    print(open({target!r}).read())\n"
                    "except OSError as e:\n"
                    "    print('DENIED', e.__class__.__name__)\n"
                )
            assert "PRIVATE" not in out and "AUTH_DB" not in out, (name, target, out)
            assert "rc=0" not in out
            if tool == "terminal":
                out = run(f"echo BOB_OVERWROTE > {target}; echo rc=$?")
            else:
                out = run(
                    "try:\n"
                    f"    open({target!r}, 'w').write('BOB_OVERWROTE')\n"
                    "except OSError as e:\n"
                    "    print('DENIED', e.__class__.__name__)\n"
                )
            assert "rc=0" not in out
        assert path.read_text() != "BOB_OVERWROTE", (name, path)

    # The install root and the other account's tree are not even listable.
    studio = tmp_path / "studio"
    if tool == "terminal":
        out = run(f"ls {studio}; echo rc=$?; ls {studio / 'accounts' / 'alice-id'}; echo rc=$?")
    else:
        out = run(
            "import os\n"
            f"for p in [{str(studio)!r}, {str(studio / 'accounts' / 'alice-id')!r}]:\n"
            "    try:\n"
            "        print(os.listdir(p))\n"
            "    except OSError as e:\n"
            "        print('DENIED', e.__class__.__name__)\n"
        )
    assert "rc=0" not in out
    assert "auth" not in out.replace("DENIED", "") or "PermissionError" in out


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_confinement_survives_nested_processes(tmp_path):
    files = _seed(tmp_path)
    alice = files["alice"]
    out = run_as(
        BOB,
        tools._bash_exec,
        f"python -c \"import subprocess; print(subprocess.run(['cat', {str(alice)!r}], "
        'capture_output=True, text=True).stderr)"',
        session_id = "chat",
    )
    assert "PRIVATE" not in out
    assert "Permission denied" in out or "denied" in out.lower()


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_confined_child_keeps_interpreter_and_system_tools(tmp_path):
    _seed(tmp_path)
    out = run_as(
        BOB,
        tools._python_exec,
        "import json, sqlite3, tempfile, os\n"
        "with tempfile.NamedTemporaryFile('w', delete=False) as f:\n"
        "    f.write('x')\n"
        "print(json.dumps({'tmp': os.path.exists(f.name), 'sqlite': sqlite3.sqlite_version != ''}))\n",
        session_id = "chat",
    )
    assert '"tmp": true' in out and '"sqlite": true' in out
    out = run_as(
        BOB, tools._bash_exec, "ls /usr/bin | head -1; python --version", session_id = "chat"
    )
    assert "Python" in out


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_owner_child_remains_unconfined(tmp_path):
    files = _seed(tmp_path)
    out = run_as(OWNER, tools._bash_exec, f"cat {files['alice']}", session_id = "chat")
    assert "ALICE_PRIVATE" in out


def test_landlock_rules_cover_own_roots_only(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Linux rule builder")
    run_as(ALICE, tools._get_workdir, "chat")
    rules = run_as(ALICE, tool_confinement._landlock_rules, 3, tools._SANDBOX_SITE_DIR)
    handled = tool_confinement._handled_mask(3)
    writable = [p for p, access in rules if access == handled & ~tool_confinement._FS_MAKE_SYM]
    alice_root = str((tmp_path / "studio" / "accounts" / "alice-id").resolve())
    # The workspace is readable, its sandbox and project roots writable.
    assert alice_root not in writable
    assert f"{alice_root}/sandbox" in writable
    assert str((tmp_path / "projects" / "Accounts" / "alice-id" / "Projects").resolve()) in writable
    assert alice_root in [
        p for p, access in rules if access != handled & ~tool_confinement._FS_MAKE_SYM
    ]
    assert str((tmp_path / "studio").resolve()) not in [p for p, _ in rules]
    assert all(
        not p.startswith(str((tmp_path / "studio").resolve()) + os.sep) or p.startswith(alice_root)
        for p, _ in rules
    )
    read_only = [
        p
        for p, access in rules
        if access not in (handled, handled & ~tool_confinement._FS_MAKE_SYM)
    ]
    assert any(
        p.startswith(os.path.realpath(sys.prefix)) or os.path.realpath(sys.prefix).startswith(p)
        for p in read_only
    )


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_managed_child_cannot_plant_a_link_in_its_own_tree(tmp_path):
    """The server follows links for the owner, so a managed child gets no way to make one."""
    files = _seed(tmp_path)
    bob_dir = Path(run_as(BOB, tools._get_workdir, "chat"))
    out = run_as(
        BOB,
        tools._bash_exec,
        f"ln -s {files['alice'].parent} linked; echo rc=$?; mkdir made && echo made > made/f && cat made/f",
        session_id = "chat",
    )
    assert "rc=0" not in out
    assert not (bob_dir / "linked").is_symlink()
    assert "made" in out


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
@pytest.mark.skipif(
    tool_confinement.landlock_abi() < 6, reason = "signal scoping needs Landlock ABI 6"
)
def test_managed_child_cannot_signal_another_accounts_process(tmp_path):
    """Two accounts' tools run as one Unix user; signals stay inside the
    child's own domain, so it can stop its own descendants and nothing else."""
    _seed(tmp_path)
    alice_job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        out = run_as(
            BOB,
            tools._python_exec,
            "import os, signal, subprocess\n"
            f"pid = {alice_job.pid}\n"
            "try:\n"
            "    os.kill(pid, signal.SIGTERM); print('SIGNALLED')\n"
            "except OSError as e:\n"
            "    print('signal denied', e.__class__.__name__)\n"
            "child = subprocess.Popen(['sleep', '30'])\n"
            "child.terminate(); print('own child rc', child.wait())\n",
            session_id = "chat",
        )
        assert "SIGNALLED" not in out and "signal denied" in out, out
        assert "own child rc -15" in out, out
        assert alice_job.poll() is None
    finally:
        alice_job.kill()


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_managed_child_writes_only_its_sandbox_and_projects(tmp_path):
    """The workspace is readable but not writable: the account's database, its
    recorded grants and the server-managed directories live there."""
    files = _seed(tmp_path)
    bob_root = Path(run_as(BOB, storage_roots_workspace))
    (bob_root / "studio.db").write_text("BOB-DB")
    (bob_root / "assets").mkdir(exist_ok = True)
    (bob_root / "assets" / "data.jsonl").write_text('{"text": "BOB-DATA"}')
    out = run_as(
        BOB,
        tools._bash_exec,
        f"cat {bob_root}/assets/data.jsonl; echo rc=$?; "
        f"echo TAMPERED > {bob_root}/studio.db; echo write_rc=$?; "
        f"echo x > {bob_root}/assets/new.txt; echo assets_rc=$?; "
        "echo mine > own.txt; echo sandbox_rc=$?",
        session_id = "chat",
    )
    assert "BOB-DATA" in out and "rc=0" in out, out
    assert "write_rc=0" not in out and "assets_rc=0" not in out, out
    assert "sandbox_rc=0" in out, out
    assert (bob_root / "studio.db").read_text() == "BOB-DB"
    assert files["unsloth"].read_text() == "UNSLOTH_PRIVATE"


def storage_roots_workspace():
    from utils.paths.storage_roots import workspace_root
    return workspace_root()


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_install_under_a_granted_root_stays_hidden(tmp_path, monkeypatch):
    """A Studio home beneath the interpreter prefix (or /opt, /var/lib) must
    not become readable through the grant for that ancestor."""
    home = Path(sys.prefix) / f"mu-confinement-home-{os.getpid()}"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    try:
        files = _seed(home.parent)  # seeds studio under home
        auth = home / "auth" / "auth.db"
        auth.parent.mkdir(parents = True, exist_ok = True)
        auth.write_text("OWNER_AUTH_DB", encoding = "utf-8")
        out = run_as(
            BOB,
            tools._bash_exec,
            f"cat {auth}; echo rc=$?; ls {home}; echo ls_rc=$?; python -c 'import sys; print(sys.prefix)'",
            session_id = "chat",
        )
        assert "OWNER_AUTH_DB" not in out and "rc=0" not in out and "ls_rc=0" not in out, out
        assert sys.prefix in out, out
        rules = run_as(BOB, tool_confinement._landlock_rules, 3, tools._SANDBOX_SITE_DIR)
        assert all(not tool_confinement._contains(p, str(home.resolve())) for p, _ in rules)
    finally:
        import shutil
        shutil.rmtree(home, ignore_errors = True)


def test_sandbox_home_override_is_writable(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "custom-sandboxes"))
    roots = run_as(BOB, tool_confinement._writable_roots)
    assert str((tmp_path / "custom-sandboxes" / "accounts" / "bob-id").resolve()) in roots


def test_landlock_below_abi_3_is_refused(monkeypatch):
    monkeypatch.setattr(tool_confinement, "landlock_abi", lambda: 2)
    assert tool_confinement._linux_confinement(tools._SANDBOX_SITE_DIR) is None
