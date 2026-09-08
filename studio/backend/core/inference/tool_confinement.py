# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Filesystem confinement for a managed account's tool subprocesses.

The sandbox in ``tools.py`` does not stop a child from opening another account's
root or the install root, which holds the owner's data and the auth database, so
for a managed account only every sandboxed child is also confined:

* Linux: a Landlock ruleset (kernel 5.13+, no privileges) applied in the forked
  child, inherited by every descendant and impossible to lift. System paths stay
  readable and executable, the account's own roots are writable, everything else
  does not exist for the child.
* macOS: ``sandbox-exec`` with an equivalent profile wrapping the command.
* Anywhere else: refused rather than run unconfined, unless the owner sets
  ``UNSLOTH_STUDIO_ALLOW_UNCONFINED_TOOLS=1``.

``account_confinement`` returns ``None`` for the owner, so a single-account
install spawns its tools exactly as before.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

from utils.account_context import is_owner_context

_OVERRIDE_ENV = "UNSLOTH_STUDIO_ALLOW_UNCONFINED_TOOLS"

# Landlock syscall numbers are the same on every architecture.
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13  # ABI 2
_FS_TRUNCATE = 1 << 14  # ABI 3
_FS_IOCTL_DEV = 1 << 15  # ABI 5
_SCOPE_SIGNAL = 1 << 1  # ABI 6: signals reach only processes inside the same domain
_FS_ABI1_MASK = (1 << 13) - 1

# Read and execute only; /proc and /sys stay readable as in the owner's sandbox.
_SYSTEM_READ_ROOTS = (
    "/usr",
    "/lib",
    "/lib32",
    "/lib64",
    "/bin",
    "/sbin",
    "/etc",
    "/opt",
    "/run",
    "/snap",
    "/nix",
    "/var/lib",
    "/proc",
    "/sys",
)
_DEVICE_ROOT = "/dev"


class ToolConfinementUnavailable(RuntimeError):
    """This host cannot confine a managed account's tool process."""


@dataclass(frozen = True)
class Confinement:
    """How a managed account's child is confined: ``preexec`` runs in the forked
    child (Linux), ``wrap`` rewrites the argv (macOS), ``mechanism`` names which."""

    mechanism: str
    preexec: Optional[Callable[[], None]] = None
    wrapper: tuple[str, ...] = ()

    def wrap(self, argv: list[str]) -> list[str]:
        return [*self.wrapper, *argv] if self.wrapper else argv


def unconfined_tools_allowed() -> bool:
    return (os.environ.get(_OVERRIDE_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def refusal_message() -> str:
    return (
        "Code execution is unavailable for this account: this host cannot confine tool "
        "processes to your workspace (Landlock ABI 3 on Linux 6.2 or later, sandbox-exec "
        f"on macOS). The installation owner can set {_OVERRIDE_ENV}=1 to allow unconfined "
        "tool processes for managed accounts."
    )


def _existing(paths) -> list[str]:
    seen: list[str] = []
    for raw in paths:
        if not raw:
            continue
        try:
            path = os.path.realpath(raw)
        except (OSError, ValueError):
            continue
        if os.path.exists(path) and path not in seen:
            seen.append(path)
    return seen


def _interpreter_roots() -> list[str]:
    return _existing(
        (
            sys.prefix,
            sys.base_prefix,
            sys.exec_prefix,
            getattr(sys, "base_exec_prefix", ""),
            os.path.dirname(sys.executable),
            os.environ.get("VIRTUAL_ENV", ""),
        )
    )


def _ensure_dirs(paths) -> list[str]:
    """Create the account's own roots: a rule can only name a path that exists."""
    roots = []
    for root in paths:
        try:
            Path(root).mkdir(parents = True, exist_ok = True)
        except OSError:
            continue
        roots.append(str(root))
    return _existing(roots)


def _readable_account_roots() -> list[str]:
    """The account's workspace, readable but never writable: its database and its
    recorded grants live here, and a tool could otherwise grant itself access."""
    from utils.paths.storage_roots import workspace_root
    return _ensure_dirs((workspace_root(),))


def _writable_roots() -> list[str]:
    """Where a tool may write: its account-resolved sandbox, temporary root and
    project workspaces."""
    from core.inference.tools import sandbox_root
    from utils.paths.storage_roots import project_workspaces_root, tmp_root

    return _ensure_dirs((sandbox_root(), tmp_root(), project_workspaces_root()))


def _protected_roots() -> list[str]:
    """The installation itself, which a Studio home under /opt or the interpreter
    prefix would otherwise expose through the read grant for that ancestor."""
    from utils.paths.storage_roots import studio_root
    return _existing((studio_root(),))


def _contains(ancestor: str, path: str) -> bool:
    return path == ancestor or path.startswith(ancestor.rstrip(os.sep) + os.sep)


def _grant_excluding(
    path: str, access: int, protected: list[str], rules: list[tuple[str, int]]
) -> None:
    """Grant ``access`` beneath ``path`` except for the protected roots.

    Landlock has no deny rule, so an ancestor is granted child by child.
    """
    inside = [p for p in protected if _contains(path, p)]
    if not inside:
        # A rule on a plain file may not carry directory rights.
        rules.append((path, access if os.path.isdir(path) else access & ~_FS_READ_DIR))
        return
    if any(p == path for p in inside):
        return
    try:
        children = sorted(os.listdir(path))
    except OSError:
        return
    for name in children:
        child = os.path.join(path, name)
        if os.path.islink(child):
            # A link is opened by its target, which must not be protected.
            if any(_contains(p, os.path.realpath(child)) for p in protected):
                continue
        _grant_excluding(child, access, protected, rules)


# ---------------------------------------------------------------- Linux ----


class _RulesetAttr(ctypes.Structure):
    # The kernel accepts this ABI 1 size from every later ABI, zeroing the rest.
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _ScopedRulesetAttr(ctypes.Structure):
    # ABI 6 layout, used only on a kernel that offers it.
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


_libc = None
if sys.platform == "linux":
    try:
        import ctypes.util
        _name = ctypes.util.find_library("c")
        _libc = ctypes.CDLL(_name, use_errno = True) if _name else None
    except (OSError, AttributeError):
        _libc = None

_landlock_abi: Optional[int] = None


def landlock_abi() -> int:
    """Highest Landlock ABI the running kernel offers, 0 when unavailable."""
    global _landlock_abi
    if _landlock_abi is not None:
        return _landlock_abi
    abi = 0
    if sys.platform == "linux" and _libc is not None:
        try:
            got = _libc.syscall(
                _SYS_LANDLOCK_CREATE_RULESET, None, 0, _LANDLOCK_CREATE_RULESET_VERSION
            )
            abi = int(got) if got > 0 else 0
        except (OSError, AttributeError, ValueError):
            abi = 0
    _landlock_abi = abi
    return abi


def _handled_mask(abi: int) -> int:
    mask = _FS_ABI1_MASK
    if abi >= 2:
        mask |= _FS_REFER
    if abi >= 3:
        mask |= _FS_TRUNCATE
    if abi >= 5:
        mask |= _FS_IOCTL_DEV
    return mask


def _landlock_rules(abi: int, sandbox_site_dir: str) -> list[tuple[str, int]]:
    handled = _handled_mask(abi)
    read = _FS_READ_FILE | _FS_READ_DIR | _FS_EXECUTE
    device = _FS_READ_FILE | _FS_WRITE_FILE | (_FS_IOCTL_DEV if abi >= 5 else 0)
    rules: list[tuple[str, int]] = []
    protected = _protected_roots()
    for path in _existing(_SYSTEM_READ_ROOTS):
        _grant_excluding(path, read, protected, rules)
    for path in _interpreter_roots():
        _grant_excluding(path, read, protected, rules)
    for path in _existing((sandbox_site_dir,)):
        _grant_excluding(path, read, protected, rules)
    for path in _readable_account_roots():
        rules.append((path, read))
    for path in _existing((_DEVICE_ROOT,)):
        rules.append((path, device))
    # Everything but creating links: the server follows links for the owner, so a
    # tool must not be able to plant one pointing outside the account's tree.
    writable = handled & ~_FS_MAKE_SYM
    for path in _writable_roots():
        rules.append((path, writable))
    return rules


def _landlock_preexec(
    handled: int,
    rules: list[tuple[str, int]],
    scoped: int = 0,
) -> None:
    """Runs in the forked child: no imports, no allocation beyond ctypes."""
    libc = _libc
    attr = _ScopedRulesetAttr(handled, 0, scoped) if scoped else _RulesetAttr(handled)
    ruleset_fd = libc.syscall(
        _SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0
    )
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    try:
        for path, access in rules:
            parent_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                beneath = _PathBeneathAttr(access & handled, parent_fd)
                rc = libc.syscall(
                    _SYS_LANDLOCK_ADD_RULE,
                    ruleset_fd,
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(beneath),
                    0,
                )
                if rc < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
            finally:
                os.close(parent_fd)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
        if libc.syscall(_SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset_fd)


# Truncation is only handled from ABI 3 (Linux 6.2); below that a confined child
# could still empty a foreign file it cannot otherwise write.
_MIN_LANDLOCK_ABI = 3


def _linux_confinement(sandbox_site_dir: str) -> Optional[Confinement]:
    abi = landlock_abi()
    if abi < _MIN_LANDLOCK_ABI:
        return None
    handled = _handled_mask(abi)
    rules = _landlock_rules(abi, sandbox_site_dir)
    return Confinement(
        mechanism = f"landlock-abi{abi}",
        # Signals stay inside the child's own domain (ABI 6), so a tool cannot stop
        # another account's. /proc stays readable, but ptrace rules leave only the
        # command line visible.
        preexec = partial(_landlock_preexec, handled, rules, _SCOPE_SIGNAL if abi >= 6 else 0),
    )


# ---------------------------------------------------------------- macOS ----


def _sbpl(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def macos_profile(
    *,
    read_roots: list[str],
    hidden_roots: list[str],
    writable_roots: list[str],
    account_read_roots: list[str] = (),
) -> str:
    """A sandbox-exec profile: later rules win, so the account's own roots are
    allowed after the install root and the home directory are denied."""
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow signal (target same-sandbox))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow ipc-posix-shm)",
        "(allow network*)",
        "(allow file-read-metadata)",
        '(allow file-read* file-write* (subpath "/dev"))',
        '(allow file-read* (subpath "/private/tmp") (subpath "/private/var/db") '
        '(subpath "/private/var/folders") (subpath "/var/folders"))',
    ]
    for path in read_roots:
        lines.append(f"(allow file-read* (subpath {_sbpl(path)}))")
    for path in hidden_roots:
        lines.append(f"(deny file-read* file-write* (subpath {_sbpl(path)}))")
    for path in account_read_roots:
        lines.append(f"(allow file-read* (subpath {_sbpl(path)}))")
    for path in writable_roots:
        lines.append(f"(allow file-read* file-write* (subpath {_sbpl(path)}))")
    return "\n".join(lines) + "\n"


def _macos_confinement(sandbox_site_dir: str) -> Optional[Confinement]:
    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        return None
    from utils.paths.storage_roots import shared_tmp_root, studio_root

    read_roots = _existing(
        (
            "/usr",
            "/bin",
            "/sbin",
            "/etc",
            "/private/etc",
            "/System",
            "/Library",
            "/opt",
            "/Applications",
            *_interpreter_roots(),
            sandbox_site_dir,
        )
    )
    # Creates the account tmp root, and with it the shared base denied below.
    writable_roots = _writable_roots()
    # Every account's tmp_root sits under one per-user temp base: deny it, re-allow only ours.
    hidden_roots = _existing(
        (str(studio_root()), str(shared_tmp_root()), os.path.expanduser("~"))
    )
    profile = macos_profile(
        read_roots = read_roots,
        hidden_roots = hidden_roots,
        account_read_roots = _readable_account_roots(),
        writable_roots = writable_roots,
    )
    return Confinement(mechanism = "sandbox-exec", wrapper = (sandbox_exec, "-p", profile))


# ------------------------------------------------------------- entry point --


def account_confinement(sandbox_site_dir: str) -> Optional[Confinement]:
    """The confinement for the acting account's next tool child.

    ``None`` for the owner, whose sandbox is unchanged; otherwise the platform
    mechanism, or ``ToolConfinementUnavailable`` when the host has none and the
    owner has not opted out.
    """
    if is_owner_context():
        return None
    confinement = None
    if sys.platform == "linux":
        confinement = _linux_confinement(sandbox_site_dir)
    elif sys.platform == "darwin":
        confinement = _macos_confinement(sandbox_site_dir)
    if confinement is not None:
        return confinement
    if unconfined_tools_allowed():
        return Confinement(mechanism = "unconfined-by-owner")
    raise ToolConfinementUnavailable(refusal_message())
