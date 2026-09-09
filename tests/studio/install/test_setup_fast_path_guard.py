# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""setup.sh / setup.ps1 must not skip the dependency pass on a half-built venv.

Both short-circuit all dependency work when the installed unsloth version equals
PyPI's latest, which is true on an interrupted install: unsloth goes in early and
studio.txt never finishes. So update, and the desktop Repair button behind it,
said "up to date" while the server kept dying on `import structlog`.

That branch only runs for a non-local update, which reinstalls from PyPI and
clobbers the tree under test, so assert the guard structurally instead.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_fast_path_consults_the_install_manifest(script: pathlib.Path):
    text = script.read_text(encoding = "utf-8")
    assert "install_manifest" in text, (
        f"{script.name} no longer consults studio/install_manifest.py. Without it "
        "the 'up to date' fast path skips the dependency pass on an interrupted "
        "install, and `unsloth studio update` becomes a silent no-op."
    )
    assert "verify_install" in text, (
        f"{script.name} must call install_manifest.verify_install() so the check "
        "matches what `unsloth studio verify-install` and the desktop preflight use."
    )


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_guard_can_still_force_the_dependency_pass(script: pathlib.Path):
    """The guard has to clear the skip flag, not merely log a warning."""
    text = script.read_text(encoding = "utf-8")
    if script.name.endswith(".ps1"):
        pattern = r"studio install incomplete[\s\S]{0,200}?\$SkipPythonDeps\s*=\s*\$false"
    else:
        pattern = r"studio install incomplete[\s\S]{0,200}?_SKIP_PYTHON_DEPS=false"
    assert re.search(pattern, text), (
        f"{script.name} detects an incomplete install but does not clear the "
        "skip flag, so the dependency pass would still be skipped."
    )


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_duplicate_core_metadata_cannot_take_the_version_fast_path(script: pathlib.Path):
    text = script.read_text(encoding = "utf-8")
    probe = text.find("install_manifest.installed_version_probe")
    zoo_probe = text.find("'unsloth-zoo'", probe)
    repair = text.find("duplicate metadata found", probe)
    if script.name.endswith(".ps1"):
        skip = text.find("$SkipPythonDeps = $true", repair)
    else:
        skip = text.find("_SKIP_PYTHON_DEPS=true", repair)

    assert probe != -1 and zoo_probe != -1 and repair != -1 and skip != -1
    assert probe <= zoo_probe < repair < skip, (
        f"{script.name} must detect duplicate metadata before an arbitrary "
        "version can select the up-to-date fast path"
    )


def test_ps1_drops_the_manifest_before_its_first_install():
    """Nothing may mutate the venv while the marker still says "install finished".

    install_python_stack.py drops it before its own dependency pass, which is
    enough for setup.sh: the stack is the first thing that pass runs. setup.ps1
    replaces pip, torch and triton first, so a run killed there would leave a
    manifest that still verifies and a venv with half a PyTorch.
    """
    text = SETUP_PS1.read_text(encoding = "utf-8")
    pass_start = text.index("if (-not $SkipPythonDeps) {")
    removal = text.find("remove_manifest", pass_start)
    first_install = text.index("Fast-Install", pass_start)
    stack = text.index(r'python "$PSScriptRoot\install_python_stack.py"', pass_start)

    assert removal != -1, (
        "setup.ps1 never drops the install manifest; install_python_stack.py "
        "only does so after setup.ps1 has already replaced pip and torch"
    )
    assert removal < first_install < stack, (
        "setup.ps1 must invalidate the install manifest before its first "
        "Fast-Install, not leave it to install_python_stack.py"
    )


def test_sh_dependency_pass_mutates_nothing_before_the_stack():
    """setup.sh relies on install_python_stack.py dropping the marker, which only
    holds while the stack is the first thing its dependency pass runs."""
    text = SETUP_SH.read_text(encoding = "utf-8")
    pass_start = text.index('if [ "$_SKIP_PYTHON_DEPS" = false ]')
    body = text[pass_start : text.index("install_python_stack", pass_start)]
    assert "fast_install" not in body and "pip install" not in body, (
        "setup.sh installs something before install_python_stack.py drops the "
        "manifest, so an interrupted run would keep a marker that verifies"
    )


def test_sh_guard_runs_before_the_skip_decision():
    text = SETUP_SH.read_text(encoding = "utf-8")
    guard = text.find("studio install incomplete")
    decision = text.find('if [ "$_SKIP_PYTHON_DEPS" = false ]')
    assert guard != -1 and decision != -1
    assert guard < decision, (
        "the incomplete-install guard must run before setup.sh acts on "
        "_SKIP_PYTHON_DEPS, otherwise it can never change the outcome"
    )


INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install.ps1"


@pytest.mark.parametrize("script", [INSTALL_SH, INSTALL_PS1], ids = ["install.sh", "install.ps1"])
def test_the_installer_reports_duplicate_metadata_on_every_platform(script: pathlib.Path):
    """Both installers print the version they just installed.

    importlib.metadata.version() answers from whichever record the finder
    yields first, so on a duplicated install it prints an arbitrary one and the
    run looks clean. Windows and POSIX have to agree here, or the same broken
    venv is reported differently depending on the host.
    """
    text = script.read_text(encoding = "utf-8")
    assert "installed_version_probe" in text, (
        f"{script.name} still reports the installed version through "
        "importlib.metadata.version(), which cannot see a duplicate record"
    )
    assert (
        "duplicate metadata found" in text
    ), f"{script.name} detects the conflict but never says so"


def test_the_sidecar_predicate_asks_the_shim_on_colab_too():
    """No venv interpreter on Colab; the shim is stdlib-only and the installer's own
    `python` asks it. The version grep alone read a sidecar interrupted after
    transformers landed as current on every later run."""
    text = SETUP_SH.read_text(encoding = "utf-8")
    start = text.index("_sidecar_current() {")
    body = text[start : text.index("\n}\n", start)]
    assert 'command -v python' in body
    assert '"$_sc_python" "$SCRIPT_DIR/install_manifest.py" sidecar' in body
    # The grep is the last resort, for a tree with no interpreter to ask at all.
    assert body.index('command -v python') < body.index("_target_has_pkg_version")


def test_the_ps1_sidecar_predicate_reads_the_shim_answer_under_native_error_promotion():
    """The shim answers "stale" with exit 1. With $PSNativeCommandUseErrorActionPreference
    set under ErrorActionPreference Stop that exit is a terminating error, and a catch
    that discards the output lets the fallback grep accept what the shim rejected."""
    text = SETUP_PS1.read_text(encoding = "utf-8")
    start = text.index("function Test-SidecarCurrent {")
    body = text[start : text.index("\nfunction ", start + 1)]
    assert "$PSNativeCommandUseErrorActionPreference = $false" in body
    assert body.index("$PSNativeCommandUseErrorActionPreference = $false") < body.index("& python $shim sidecar")
    assert "finally" in body


# ── the offline rule ──
#
# "could not reach PyPI, updating to be safe" is the right default: an unreachable PyPI
# is usually a blip, and a pass over a warm cache is cheap. It is the wrong answer when
# the caller has SET UV_OFFLINE, because then every install command in that pass can only
# fail -- the update does the slow half of its work and exits non-zero on a venv that was
# already complete. The rule keeps such an install, and only on the evidence the
# incomplete-install guard already demands.


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_an_unreachable_pypi_still_updates_by_default(script: pathlib.Path):
    """Nothing above changes for a plain offline blip."""
    text = script.read_text(encoding = "utf-8")
    assert text.count('substep "could not reach PyPI, updating to be safe..."') == 1


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_the_offline_rule_needs_all_three_conditions(script: pathlib.Path):
    """An installed version, a declared offline mode, and a verified tree. Any two of
    them is a skip that ships a half-built venv or a venv that was never built."""
    text = script.read_text(encoding = "utf-8")
    if script.name.endswith(".ps1"):
        condition = (
            "if ($InstalledVer -and (Test-UvOfflineRequested) -and "
            "(Test-StudioInstallVerified)) {"
        )
        taken = "$SkipPythonDeps = $true"
    else:
        condition = (
            'if [ -n "$INSTALLED_VER" ] && _uv_offline_requested '
            "&& _setup_install_is_verified; then"
        )
        taken = "_SKIP_PYTHON_DEPS=true"
    assert condition in text, f"{script.name} no longer gates the offline skip on all three"
    start = text.index(condition)
    body = text[start : start + 400]
    assert taken in body
    assert "could not reach PyPI" in body, (
        f"{script.name} lost the else branch, so a host that fails any one of the three "
        "conditions now skips silently instead of updating to be safe"
    )


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_the_two_callers_share_one_definition_of_complete(script: pathlib.Path):
    """The guard forces the pass when the tree is not verified and the offline rule keeps
    it when it is. Two copies of that check is how they come to disagree."""
    text = script.read_text(encoding = "utf-8")
    helper = (
        "function Test-StudioInstallVerified"
        if script.name.endswith(".ps1")
        else "_setup_install_is_verified() {"
    )
    assert helper in text
    assert text.count("install_manifest.verify_install(deep = True)") == 1, (
        f"{script.name} has more than one deep verify; the offline rule and the "
        "incomplete-install guard must ask the same question"
    )


def test_the_posix_offline_switch_reads_the_boolish_spellings(tmp_path):
    """Same spelling UV_NO_CACHE accepts, because a user who set one expects the other
    to be read the same way."""
    import subprocess

    text = SETUP_SH.read_text(encoding = "utf-8")
    start = text.index("_uv_offline_requested() {")
    body = text[start : text.index("\n}\n", start) + 3]
    probe = tmp_path / "probe.sh"
    probe.write_text(body + "\nif _uv_offline_requested; then echo yes; else echo no; fi\n")
    for value, expected in (
        ("1", "yes"),
        ("true", "yes"),
        ("TRUE", "yes"),
        ("  yes  ", "yes"),
        ("on", "yes"),
        # uv's boolish parser also takes the single letters; checked against uv 0.10.7,
        # where UV_OFFLINE=t and =y both disable the network.
        ("t", "yes"),
        ("T", "yes"),
        ("y", "yes"),
        ("0", "no"),
        ("false", "no"),
        ("", "no"),
        ("maybe", "no"),
        ("tr", "no"),
    ):
        result = subprocess.run(
            ["sh", str(probe)],
            capture_output = True,
            text = True,
            env = {"PATH": "/usr/bin:/bin", "UV_OFFLINE": value},
        )
        assert result.stdout.strip() == expected, (value, result.stdout)


def test_the_offline_fast_path_never_wipes_a_sidecar():
    """The offline rule keeps the install because nothing can be fetched. A sidecar
    rebuild is a wipe followed by four fetches, so under that rule it would either reach
    for the network or destroy a usable sidecar and then fail. Both shells flag the
    offline keep and clear every rebuild flag behind it."""
    sh = SETUP_SH.read_text(encoding = "utf-8")
    ps1 = SETUP_PS1.read_text(encoding = "utf-8")
    keep_sh = sh.index("keeping the verified install")
    assert "_OFFLINE_FAST_PATH=true" in sh[keep_sh : keep_sh + 400]
    guard_sh = sh.index('if [ "${_OFFLINE_FAST_PATH:-false}" = true ]; then')
    assert (
        sh.index('_sidecar_current "$VENV_T5_510_DIR"')
        < guard_sh
        < sh.index('if [ "$_NEED_T5_530" = true ]; then')
    )
    assert 'eval "_NEED_T5_$1=false"' in sh[guard_sh : guard_sh + 900]
    keep_ps1 = ps1.index("keeping the verified install")
    assert "$script:OfflineFastPath = $true" in ps1[keep_ps1 : keep_ps1 + 400]
    guard_ps1 = ps1.index("if ($script:OfflineFastPath) {")
    assert (
        ps1.index("Test-SidecarCurrent -TargetDir $VenvT5_510Dir")
        < guard_ps1
        < ps1.index("if ($_NeedT5_530 -or $_NeedT5_550 -or $_NeedT5_510) {")
    )
    assert "Set-Variable -Name $flag -Value $false" in ps1[guard_ps1 : guard_ps1 + 900]
