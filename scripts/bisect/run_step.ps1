# One measured step of the Unsloth Studio update bisection (Windows twin of run_step.sh).
#
#   pwsh -File run_step.ps1 -Cmd install   -N 806 [-NoTorch] [-IsolatedUvCache]
#   pwsh -File run_step.ps1 -Cmd update    -N 806 -To 807 [-Stage]
#   pwsh -File run_step.ps1 -Cmd activate  -Label 806-to-807
#   pwsh -File run_step.ps1 -Cmd launch    -Label pair-807 [-Seconds 150]
#   pwsh -File run_step.ps1 -Cmd uninstall -N 807 -Label p1
#
#   pwsh -File run_step.ps1 -Cmd noop-update -Label after-pr [-Target pr] [-Offline] [-Role settle|noop|offline]
#   pwsh -File run_step.ps1 -Cmd fault -Kind sidecar-truncate [-Target pr]
#   pwsh -File run_step.ps1 -Cmd old-shell-stage [-ShellVersion 0.1.807-beta]
#   pwsh -File run_step.ps1 -Cmd leftover-seed | -Cmd leftover-check
#   pwsh -File run_step.ps1 -Cmd prefetch [-Target pr2]
#
# Env: OUT (results dir, required), SCRIPTS_DIR (tag installers, fetched if missing),
#      BISECT_DIR (dir of this script). The install lands in the real %USERPROFILE%\.unsloth\studio
#      (one job per runner, so no fake home is needed on Windows).
#
# PR-wheel targets (-N pr / -To pr / -Target pr2): the wheel built from the checked-out branch is
# served from a local find-links directory layered ABOVE the release pins (pr_wheel.py), so the
# `studio update` run by the OLD installed release resolves `unsloth` to the PR. PR_WHEEL_DIR /
# PR_WHEEL_DIR2 name the wheel dirs; PR_PIN_BASE (default 807) is the release the layer sits on.
param(
    [Parameter(Mandatory = $true)][string]$Cmd,
    [string]$N = "",
    [string]$To = "",
    [string]$Label = "",
    [string]$Kind = "",
    [string]$Target = "pr",
    [string]$ShellVersion = "0.1.807-beta",
    [ValidateSet("", "settle", "noop", "offline")][string]$Role = "",
    [int]$Seconds = 150,
    [switch]$NoTorch,
    [switch]$IsolatedUvCache,
    [switch]$Stage,
    [switch]$Offline,
    [switch]$OfflinePypi
)
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$BisectDir = if ($env:BISECT_DIR) { $env:BISECT_DIR } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Pins = Join-Path $BisectDir "pins"
$Out = $env:OUT; if (-not $Out) { throw "OUT required" }
$ScriptsDir = if ($env:SCRIPTS_DIR) { $env:SCRIPTS_DIR } else { Join-Path $Out "tag_scripts" }
$Studio = Join-Path $env:USERPROFILE ".unsloth\studio"
$Venv = Join-Path $Studio "unsloth_studio"
$StageDir = Join-Path $Studio ".update-stage"
$GlobalUvCache = Join-Path $env:LOCALAPPDATA "uv\cache"
New-Item -ItemType Directory -Force -Path $Out, $ScriptsDir | Out-Null
$Releases = Get-Content (Join-Path $Pins "releases.json") -Raw | ConvertFrom-Json
$SourceRoot = if ($env:SOURCE_ROOT) { $env:SOURCE_ROOT } else { (Resolve-Path (Join-Path $BisectDir "..\..")).Path }
$PrWheelDir = if ($env:PR_WHEEL_DIR) { $env:PR_WHEEL_DIR } else { Join-Path $env:RUNNER_TEMP "prwheel" }
$PrWheelDir2 = if ($env:PR_WHEEL_DIR2) { $env:PR_WHEEL_DIR2 } else { Join-Path $env:RUNNER_TEMP "prwheel2" }
$PrPinBase = if ($env:PR_PIN_BASE) { $env:PR_PIN_BASE } else { "807" }
$GenPins = if ($env:GEN_PINS) { $env:GEN_PINS } else { Join-Path $Out "pins_gen" }
$DenyHosts = if ($env:DENY_HOSTS) { $env:DENY_HOSTS } else { "pypi.org,files.pythonhosted.org,github.com,objects.githubusercontent.com,release-assets.githubusercontent.com" }

function Rel($n, $key) { $Releases.$n.$key }

function Is-PrTarget($t) { return ($t -eq "pr" -or $t -eq "pr2") }
function Wheel-Dir($t) { if ($t -eq "pr2") { return $PrWheelDir2 } else { return $PrWheelDir } }

# Derive (once per results dir) the pin + find-links layer for a PR wheel target.
function Prep-Pr($t) {
    $file = Join-Path $GenPins "$t.json"
    # Reuse the cached layer only if it was written by THIS pr_wheel.py: a file from before the
    # core-dependency fold has no core_zoo, and every later reader would see an empty answer.
    if (Test-Path $file) {
        $cached = Get-Content $file -Raw | ConvertFrom-Json
        if ($cached.PSObject.Properties.Name -contains "core_zoo") { return $cached }
    }
    New-Item -ItemType Directory -Force -Path $GenPins | Out-Null
    # stdout only: pr_wheel.py prints JSON there, and a stderr line folded in would make the
    # cached layer file unparseable for every later step.
    $json = & python (Join-Path $BisectDir "pr_wheel.py") pins --wheel-dir (Wheel-Dir $t) --pins-dir $Pins `
        --out-dir $GenPins --name $t --base $PrPinBase | Out-String
    if ($LASTEXITCODE -ne 0) { throw "[harness] could not build the $t pin layer from $(Wheel-Dir $t)" }
    $json | Out-File $file -Encoding utf8
    $o = $json | ConvertFrom-Json
    Write-Host "[harness] $t layer: unsloth==$($o.version) zoo==$($o.zoo) (relaxed=$($o.zoo_relaxed)) find-links=$($o.find_links)"
    return $o
}
function Target-Version($t) { if (Is-PrTarget $t) { return (Prep-Pr $t).version } else { return (Rel $t pypi_version) } }
function Target-Zoo($t) { if (Is-PrTarget $t) { return (Prep-Pr $t).zoo } else { return (Rel $t zoo) } }

function Installed-Version($dist) {
    $py = Join-Path $Venv "Scripts\python.exe"
    if (-not (Test-Path $py)) { return "" }
    $code = "import importlib.metadata as m`ntry: print(m.version('$dist'))`nexcept Exception: print('')"
    return (& $py -I -c $code 2>$null | Out-String).Trim()
}

function Idem { & python (Join-Path $BisectDir "idem.py") @args }

function Fetch-TagScript($n, $name) {
    $dest = Join-Path $ScriptsDir $n
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $file = Join-Path $dest $name
    $path = if ($name -like "uninstall.*") { "scripts/$name" } else { $name }
    if (-not (Test-Path $file)) {
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/unslothai/unsloth/v0.1.$n-beta/$path" -OutFile $file -UseBasicParsing
    }
    if ($name -notlike "uninstall.*") {
        $key = ($name -replace '\.', '_') + "_blob"
        $want = Rel $n $key
        $bytes = [System.IO.File]::ReadAllBytes($file)
        $header = [System.Text.Encoding]::ASCII.GetBytes("blob $($bytes.Length)") + [byte]0
        $sha1 = [System.Security.Cryptography.SHA1]::Create()
        $got = ([System.BitConverter]::ToString($sha1.ComputeHash($header + $bytes)) -replace '-', '').ToLower()
        if ($want -ne $got) { throw "BLOB MISMATCH for $n/$name want $want got $got" }
        Write-Host "[harness] $n/$name blob $got verified against tag v0.1.$n-beta"
    }
    return $file
}

function Rx-Bytes { try { (Get-NetAdapterStatistics | Measure-Object -Property ReceivedBytes -Sum).Sum } catch { 0 } }

$script:ProxyProc = $null
function Start-Proxy($dir, $extraArgs) {
    Remove-Item (Join-Path $dir "proxy.port") -ErrorAction SilentlyContinue
    $pargs = @((Join-Path $BisectDir "connect_proxy.py"), "serve", "--port", "0", "--log", (Join-Path $dir "proxy.jsonl"), "--port-file", (Join-Path $dir "proxy.port"))
    if ($extraArgs) { $pargs += $extraArgs }
    $script:ProxyProc = Start-Process -FilePath "python" -ArgumentList $pargs -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $dir "proxy.out") -RedirectStandardError (Join-Path $dir "proxy.err")
    for ($i = 0; $i -lt 50; $i++) { if (Test-Path (Join-Path $dir "proxy.port")) { break }; Start-Sleep -Milliseconds 100 }
    return "http://127.0.0.1:$(Get-Content (Join-Path $dir 'proxy.port'))"
}
function Stop-Proxy { if ($script:ProxyProc) { Stop-Process -Id $script:ProxyProc.Id -Force -ErrorAction SilentlyContinue; $script:ProxyProc = $null } }

function Set-ChildEnv($n, $proxy, $extra) {
    foreach ($k in 'UV_CACHE_DIR', 'UNSLOTH_STUDIO_HOME', 'STUDIO_HOME', 'VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME', 'UV_PYTHON',
                   'UV_FIND_LINKS', 'PIP_FIND_LINKS', 'UV_OFFLINE') { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
    $env:UNSLOTH_SKIP_AUTOSTART = "1"
    $env:UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = "1"
    $env:PYTHONUNBUFFERED = "1"; $env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
    if (Is-PrTarget $n) {
        $layer = Prep-Pr $n
        $env:UV_CONSTRAINT = $layer.pins
        $env:UV_CONFIG_FILE = $layer.uv_toml
        $env:PIP_CONSTRAINT = $layer.pins
        $env:UV_FIND_LINKS = $layer.find_links
        $env:PIP_FIND_LINKS = $layer.find_links
    } else {
        $env:UV_CONSTRAINT = Join-Path $Pins "pins_$n.txt"
        $env:UV_CONFIG_FILE = Join-Path $Pins "uv_$n.toml"
        $env:PIP_CONSTRAINT = Join-Path $Pins "pins_$n.txt"
    }
    $env:NO_PROXY = "127.0.0.1,localhost"
    if ($proxy) { $env:HTTPS_PROXY = $proxy; $env:HTTP_PROXY = $proxy; $env:ALL_PROXY = $proxy }
    # HTTP(S)_PROXY reaches uv, curl and Python. It does NOT reach setup.ps1's version probe:
    # `Invoke-RestMethod` on Windows PowerShell 5.1 goes through WebRequest.DefaultWebProxy (the
    # per-user WinINET settings) and ignores the environment entirely. So under the refuse-all
    # proxy the probe still reached PyPI and printed "unsloth 2026.9.3.post1 -> 2026.9.3
    # available, updating..." while our proxy logged 125 refused pypi.org attempts from uv: the
    # offline run was not offline, and the Windows connection counts were fiction (observed, run
    # 34332051981, windows-latest upgrade 806).
    #
    # _UNSLOTH_PS_PROXY_DEFAULTS is the product's OWN corporate-proxy handoff: unsloth_cli passes
    # it through when it is already present (presence, not truthiness) and setup.ps1's prelude
    # replays it into $PSDefaultParameterValues, so every Invoke-RestMethod / Invoke-WebRequest in
    # the child takes our proxy. That is a supported user configuration rather than a machine-wide
    # registry or firewall edit, needs no elevation, and touches nothing outside this step.
    # It does not cover the `install` steps: install.ps1 republishes the variable from the
    # (-NoProfile, therefore empty) caller table, so an install's own Invoke-* calls still go
    # direct. Those steps are online anyway; only the no-op runs assert on the counts.
    $env:_UNSLOTH_PS_PROXY_DEFAULTS = if ($proxy) {
        (@{ "Invoke-RestMethod:Proxy" = $proxy; "Invoke-WebRequest:Proxy" = $proxy } | ConvertTo-Json -Compress)
    } else { "{}" }
    foreach ($kv in $extra.GetEnumerator()) { Set-Item -Path "Env:$($kv.Key)" -Value $kv.Value }
}
function Clear-ChildEnv {
    foreach ($k in 'UV_CONSTRAINT', 'UV_CONFIG_FILE', 'PIP_CONSTRAINT', 'HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'UNSLOTH_TAURI_UPDATE',
                   'SKIP_STUDIO_FRONTEND', 'UNSLOTH_DESKTOP_BACKEND_VERSION', 'UV_FIND_LINKS', 'PIP_FIND_LINKS', 'UV_OFFLINE',
                   'UNSLOTH_TAURI_SHELL_VERSION', '_UNSLOTH_PS_PROXY_DEFAULTS') { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
}

function Snapshot($dir, $label, $venvPath) {
    if (-not $venvPath) { $venvPath = $Venv }
    & python (Join-Path $BisectDir "snapshot.py") take $dir --venv $venvPath --studio-home $Studio --cache $GlobalUvCache --label $label 2>&1 | Select-Object -Last 1
}

function Run-Timed($dir, $exe, $argList) {
    # Stream the child's merged output with an elapsed-seconds prefix into log.txt (like ts_filter).
    # cmd.exe merges stderr into stdout so one synchronous ReadLine loop sees everything in order;
    # PowerShell event handlers run in another scope and cannot see this function's variables.
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $log = Join-Path $dir "log.txt"
    $quoted = ($argList | ForEach-Object { if ($_ -match '\s' -and $_ -notmatch '^"') { '"' + $_ + '"' } else { $_ } }) -join ' '
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "cmd.exe"
    $psi.Arguments = '/d /c ""' + $exe + '" ' + $quoted + ' 2>&1"'
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardInput = $false
    $psi.WorkingDirectory = $env:USERPROFILE
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $p = [System.Diagnostics.Process]::Start($psi)
    $writer = [System.IO.StreamWriter]::new($log, $false, [System.Text.UTF8Encoding]::new($false))
    $reader = $p.StandardOutput
    $buf = New-Object System.Text.StringBuilder
    $chunk = New-Object char[] 4096
    while (($n = $reader.Read($chunk, 0, $chunk.Length)) -gt 0) {
        for ($i = 0; $i -lt $n; $i++) {
            $c = $chunk[$i]
            if ($c -eq "`r" -or $c -eq "`n") {
                if ($buf.Length -gt 0 -or $c -eq "`n") { $writer.WriteLine(("[{0,7:F1}] {1}" -f $sw.Elapsed.TotalSeconds, $buf.ToString())); $writer.Flush() }
                [void]$buf.Clear()
            } else { [void]$buf.Append($c) }
        }
    }
    if ($buf.Length -gt 0) { $writer.WriteLine(("[{0,7:F1}] {1}" -f $sw.Elapsed.TotalSeconds, $buf.ToString())) }
    $p.WaitForExit()
    $writer.Flush(); $writer.Close()
    return @{ rc = $p.ExitCode; seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1) }
}

function Assert-Versions($venv, $ev, $ez) {
    if ($ev -eq "-") { return @{ ok = $true; json = '{"skipped": true, "ok": true}' } }
    $py = Join-Path $venv "Scripts\python.exe"
    $code = "import importlib.metadata as m, json, sys`ngot={}`nfor k in ('unsloth','unsloth_zoo'):`n try: got[k]=m.version(k)`n except Exception as e: got[k]='ERR:'+str(e)`nok = got['unsloth']=='$ev' and got['unsloth_zoo']=='$ez'`nprint(json.dumps({'expected':{'unsloth':'$ev','unsloth_zoo':'$ez'},'got':got,'ok':ok}))`nsys.exit(0 if ok else 1)"
    $out = & $py -I -c $code 2>&1
    return @{ ok = ($LASTEXITCODE -eq 0); json = ($out | Out-String).Trim() }
}

# $extraPath: a JSON file merged into summary.json. $expectRc: the exit code that counts as
# success ("any" never fails on it; the caller judges from summary.json).
function Finish-Step($dir, $label, $res, $rx0, $venv, $ev, $ez, $extraPath, $expectRc) {
    if ($null -eq $expectRc -or $expectRc -eq "") { $expectRc = "0" }
    $rx1 = Rx-Bytes
    Stop-Proxy
    $proxy = "{}"
    if (Test-Path (Join-Path $dir "proxy.jsonl")) { $proxy = (& python (Join-Path $BisectDir "connect_proxy.py") summary (Join-Path $dir "proxy.jsonl") | Out-String) }
    Snapshot (Join-Path $dir "after") "${label}:after" $venv | Out-File (Join-Path $dir "after.line")
    $as = Assert-Versions $venv $ev $ez
    $diff = "null"
    if (Test-Path (Join-Path $dir "before\snapshot.json")) {
        & python (Join-Path $BisectDir "snapshot.py") diff (Join-Path $dir "before") (Join-Path $dir "after") --out (Join-Path $dir "diff.json") | Out-Null
        if (Test-Path (Join-Path $dir "diff.json")) { $diff = Get-Content (Join-Path $dir "diff.json") -Raw }
    }
    $summary = [ordered]@{
        label = $label; os = "Windows"; seconds_total = $res.seconds; exit_code = $res.rc
        assert_ok = $as.ok; assertion = ($as.json | ConvertFrom-Json)
        rx_bytes_host_delta = [long]($rx1 - $rx0)
        proxy = ($proxy | ConvertFrom-Json); diff = ($diff | ConvertFrom-Json)
        expected_exit_code = $expectRc
    }
    $summary["connections_attempted"] = $summary.proxy.connections
    $summary["connections_refused"] = $summary.proxy.refused
    $summary | ConvertTo-Json -Depth 12 | Out-File (Join-Path $dir "summary.json") -Encoding utf8
    if ($extraPath -and (Test-Path $extraPath)) {
        # Merge in Python: ConvertTo-Json -Depth cannot round-trip the nested capture blobs.
        & python -c "import json,sys; p,e=sys.argv[1:]; s=json.load(open(p, encoding='utf-8-sig')); s.update(json.load(open(e, encoding='utf-8-sig'))); json.dump(s, open(p,'w'), indent=1)" (Join-Path $dir "summary.json") $extraPath
    }
    $mb = [math]::Round(($summary.proxy.total_bytes_down / 1e6))
    Write-Host "[harness] ${label}: $($res.seconds)s exit=$($res.rc) assert_ok=$($as.ok) proxy_down=${mb}MB conns=$($summary.proxy.connections)"
    Write-Host $as.json
    $rcBad = ($expectRc -ne "any") -and ("$($res.rc)" -ne "$expectRc")
    if ($rcBad -or -not $as.ok) { Write-Host "[harness] STEP FAILED: $label (exit=$($res.rc) expected=$expectRc)"; return $false }
    return $true
}

switch ($Cmd) {
    "install" {
        $PrInstallArgs = @()
        $label = "install-$N"; $dir = Join-Path $Out $label; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        if (Is-PrTarget $N) {
            # Fresh install of the PR wheel with the BRANCH's own installer.
            $script = Join-Path $SourceRoot "install.ps1"
            if (-not (Test-Path $script)) { throw "[harness] no branch installer at $script" }
            $ev = Target-Version $N; $ez = Target-Zoo $N
            # install.ps1's fresh-install arm installs `unsloth` with no extra and then runs
            # studio/setup.ps1 with SKIP_STUDIO_BASE=1, which skips the only other step that names
            # unsloth-zoo, so the wheel's own core metadata is the only thing that installs it.
            # `python -m build` from the checkout does not emit it (it lives in the `huggingface`
            # extra), which is what made every `install pr` die with "unsloth-zoo is not installed".
            # pr_wheel.py folds it back in at relabel time.
            if (-not (Prep-Pr $N).core_zoo) {
                throw "[harness] the $N wheel declares no core unsloth_zoo requirement, so a fresh install would leave the venv without unsloth_zoo. Rebuild the layer with pr_wheel.py relabel (without --no-fold-core)."
            }
            # -Tauri is what the desktop passes: take the frontend the wheel bundles instead of
            # running `npm install` in the checkout. Without it a PR install measures a frontend
            # build no user performs, and dies on any peer-dependency drift in studio/frontend.
            $PrInstallArgs = @("--tauri")
        } else {
            $script = Fetch-TagScript $N "install.ps1"
            $ev = Rel $N pypi_version; $ez = Rel $N zoo
        }
        Snapshot (Join-Path $dir "before") "${label}:before" | Out-File (Join-Path $dir "before.line")
        $proxy = Start-Proxy $dir
        $extra = @{}; if ($NoTorch) { $extra["UNSLOTH_NO_TORCH"] = "1" }; if ($IsolatedUvCache) { $extra["UNSLOTH_ISOLATE_UV_CACHE"] = "1" }
        if (Is-PrTarget $N) { $extra["SKIP_STUDIO_FRONTEND"] = "1" }
        Set-ChildEnv $N $proxy $extra
        Write-Host "[harness] === $label (tag script $script, pins unsloth==$ev zoo==$ez, notorch=$NoTorch isolated=$IsolatedUvCache)"
        $rx0 = Rx-Bytes
        # Windows PowerShell 5.1, like the desktop and a clean box.
        $installArgs = @("-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", "`"$script`"")
        if ($PrInstallArgs) { $installArgs += $PrInstallArgs }
        $res = Run-Timed $dir "powershell.exe" $installArgs
        $ok = Finish-Step $dir $label $res $rx0 $Venv $ev $ez $null "0"
        Clear-ChildEnv
        if (-not $ok) { exit 1 }
    }
    "update" {
        $suffix = if ($Stage) { "-staged" } else { "" }
        if ($OfflinePypi) { $suffix = "$suffix-offlinepypi" }
        $label = "update-$N-to-$To$suffix"; $dir = Join-Path $Out $label; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $ev = Target-Version $To; $ez = Target-Zoo $To
        Snapshot (Join-Path $dir "before") "${label}:before" | Out-File (Join-Path $dir "before.line")
        $proxy = if ($OfflinePypi) { Start-Proxy $dir @("--deny-hosts", $DenyHosts) } else { Start-Proxy $dir }
        Set-ChildEnv $To $proxy @{ UNSLOTH_TAURI_UPDATE = "1"; SKIP_STUDIO_FRONTEND = "1"; UNSLOTH_DESKTOP_BACKEND_VERSION = $ev }
        Write-Host "[harness] === $label (venv CLI of the installed release, UNSLOTH_DESKTOP_BACKEND_VERSION=$ev)"
        $rx0 = Rx-Bytes
        $argList = @("-I", "-X", "utf8", "-m", "unsloth_cli", "studio", "update"); if ($Stage) { $argList += "--stage" }
        $res = Run-Timed $dir (Join-Path $Venv "Scripts\python.exe") $argList
        $target = if ($Stage) { Join-Path $StageDir "unsloth_studio" } else { $Venv }
        Idem steps --log (Join-Path $dir "log.txt") --out (Join-Path $dir "steps.json") | Out-Null
        & python -c "import json,sys; d=json.load(open(sys.argv[1], encoding='utf-8-sig')); json.dump({'update_steps':d['steps'],'update_steps_ran':d['ran'],'update_path':d['update_path'],'update_path_reason':d['update_path_reason'],'update_path_evidence':d['update_path_evidence'],'pypi_probe_answered':d['pypi_probe_answered']}, open(sys.argv[2],'w'), indent=1)" (Join-Path $dir "steps.json") (Join-Path $dir "extra.json")
        $ok = Finish-Step $dir $label $res $rx0 $target $ev $ez (Join-Path $dir "extra.json") "0"
        Clear-ChildEnv
        if (-not $ok) { exit 1 }
    }
    "activate" {
        $label = "activate-$Label"; $dir = Join-Path $Out $label; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $ready = Join-Path $StageDir "READY.json"
        if (-not (Test-Path $ready)) { Write-Host "[activate] no READY.json"; exit 2 }
        Write-Host "[activate] READY: $(Get-Content $ready -Raw)"
        $prev = Join-Path $Studio ".update-previous"; Remove-Item $prev -Recurse -Force -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force -Path $prev | Out-Null
        $swapped = @()
        foreach ($name in "unsloth_studio", ".venv_t5_530", ".venv_t5_550", ".venv_t5_510", "node", "llama.cpp", "whisper.cpp") {
            $s = Join-Path $StageDir $name; $l = Join-Path $Studio $name
            if (-not (Test-Path $s)) { continue }
            if (Test-Path $l) { Move-Item $l (Join-Path $prev $name) }
            Move-Item $s $l; $swapped += $name
        }
        $marker = Join-Path $StageDir "uv-cache-dir"
        if (Test-Path $marker) { New-Item -ItemType Directory -Force -Path (Join-Path $Studio "cache") | Out-Null; Copy-Item $marker (Join-Path $Studio "cache\uv-cache-dir") -Force }
        $tSwap = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        Remove-Item $StageDir, $prev -Recurse -Force -ErrorAction SilentlyContinue
        $total = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        Write-Host "[activate] swapped $($swapped -join ',') in ${tSwap}s; cleanup done at ${total}s"
        @{ label = $label; seconds_total = $total; swap_seconds = $tSwap; exit_code = 0 } | ConvertTo-Json | Out-File (Join-Path $dir "summary.json") -Encoding utf8
    }
    "launch" {
        $label = "launch-$Label"; $dir = Join-Path $Out $label; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $port = Get-Random -Minimum 20000 -Maximum 40000
        $proxy = Start-Proxy $dir
        $launchPins = if ($env:PINS_TARGET) { $env:PINS_TARGET } elseif (Test-Path (Join-Path $GenPins "pr.json")) { "pr" } else { "807" }
        Set-ChildEnv $launchPins $proxy @{ UNSLOTH_TAURI_UPDATE = "1" }
        Write-Host "[harness] === $label (headless studio run on :$port for up to ${Seconds}s)"
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $p = Start-Process -FilePath (Join-Path $Venv "Scripts\python.exe") -ArgumentList @("-I", "-X", "utf8", "-m", "unsloth_cli", "studio", "--api-only", "-p", "$port", "-H", "127.0.0.1") -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $dir "server.log") -RedirectStandardError (Join-Path $dir "server.err") -WorkingDirectory $env:USERPROFILE
        $healthy = $null
        for ($i = 0; $i -lt $Seconds; $i++) {
            try { $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/health" -UseBasicParsing -TimeoutSec 2 -Proxy $null; if ($r.StatusCode -eq 200) { $healthy = [math]::Round($sw.Elapsed.TotalSeconds, 1); $r.Content | Out-File (Join-Path $dir "health.json"); break } } catch {}
            if ($p.HasExited) { break }
            Start-Sleep 1
        }
        if ($healthy) { Start-Sleep 45 }
        try { & taskkill /PID $p.Id /T /F 2>&1 | Out-Null } catch {}
        Stop-Proxy
        $proxyJson = "{}"; if (Test-Path (Join-Path $dir "proxy.jsonl")) { $proxyJson = (& python (Join-Path $BisectDir "connect_proxy.py") summary (Join-Path $dir "proxy.jsonl") | Out-String) }
        $activity = @(Get-Content (Join-Path $dir "server.log"), (Join-Path $dir "server.err") -ErrorAction SilentlyContinue | Select-String -Pattern 'self-heal|repair|installing|uv pip|pip install|downloading|warm' | Select-Object -First 40)
        $activity | Out-File (Join-Path $dir "install_activity.txt")
        @{ label = $label; seconds_to_health = $healthy; proxy = ($proxyJson | ConvertFrom-Json); install_activity_lines = $activity.Count } | ConvertTo-Json -Depth 8 | Out-File (Join-Path $dir "summary.json") -Encoding utf8
        Write-Host "[harness] ${label}: time_to_health=${healthy}s activity_lines=$($activity.Count)"
        Clear-ChildEnv
    }
    "uninstall" {
        $label = "uninstall-$Label"; $dir = Join-Path $Out $label; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $script = Fetch-TagScript $N "uninstall.ps1"
        Snapshot (Join-Path $dir "before") "${label}:before" | Out-File (Join-Path $dir "before.line")
        Set-ChildEnv $N $null @{}
        $res = Run-Timed $dir "powershell.exe" @("-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", "`"$script`"")
        Snapshot (Join-Path $dir "after") "${label}:after" | Out-File (Join-Path $dir "after.line")
        & python (Join-Path $BisectDir "snapshot.py") diff (Join-Path $dir "before") (Join-Path $dir "after") --out (Join-Path $dir "diff.json") | Out-Null
        $diff = Get-Content (Join-Path $dir "diff.json") -Raw | ConvertFrom-Json
        $summary = [ordered]@{
            label = $label; exit_code = $res.rc; seconds_total = $res.seconds
            studio_root_exists = (Test-Path $Studio); global_uv_cache_exists = (Test-Path $GlobalUvCache)
            global_uv_cache_growth_bytes = $diff.cache_growth_bytes.$GlobalUvCache
            studio_cache_growth_bytes = $diff.cache_growth_bytes.(Join-Path $Studio "cache\uv")
        }
        $summary | ConvertTo-Json | Out-File (Join-Path $dir "summary.json") -Encoding utf8
        Write-Host "[harness] $($summary | ConvertTo-Json -Compress)"
        Clear-ChildEnv
    }
    "noop-update" {
        # Re-run `studio update` against the version that is already installed: it must do nothing.
        # -Offline puts the proxy in refuse-all mode and sets UV_OFFLINE=1, so "no work" is proved
        # by the connection count rather than by the absence of visible output.
        $sfx = if ($Offline) { "$Label-offline" } else { $Label }
        # settle = the first update by the NEW code after an upgrade (records, asserts nothing);
        # noop = compared against settle under the relaxed rule; offline = strict, and zero bytes.
        $role = if ($Role) { $Role } elseif ($Offline) { "offline" } else { "noop" }
        $label = "noop-update-$sfx"; $dir = Join-Path $Out $label; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $ev = Installed-Version "unsloth"; $ez = Installed-Version "unsloth_zoo"
        if (-not $ev) { Write-Host "[harness] nothing installed in $Venv"; exit 1 }
        Snapshot (Join-Path $dir "before") "${label}:before" | Out-File (Join-Path $dir "before.line")
        Idem capture --venv $Venv --studio-home $Studio --out (Join-Path $dir "state_before.json") | Out-Null
        $proxy = if ($Offline) { Start-Proxy $dir @("--refuse") } else { Start-Proxy $dir }
        $extraEnv = @{ UNSLOTH_TAURI_UPDATE = "1"; SKIP_STUDIO_FRONTEND = "1"; UNSLOTH_DESKTOP_BACKEND_VERSION = $ev }
        if ($Offline) { $extraEnv["UV_OFFLINE"] = "1" }
        Set-ChildEnv $Target $proxy $extraEnv
        Write-Host "[harness] === $label (role=$role, target == installed unsloth==$ev zoo==$ez, offline=$Offline, pin layer $Target)"
        $rx0 = Rx-Bytes
        $res = Run-Timed $dir (Join-Path $Venv "Scripts\python.exe") @("-I", "-X", "utf8", "-m", "unsloth_cli", "studio", "update")
        Idem capture --venv $Venv --studio-home $Studio --out (Join-Path $dir "state_after.json") | Out-Null
        Idem compare (Join-Path $dir "state_before.json") (Join-Path $dir "state_after.json") --out (Join-Path $dir "idempotency.json") | Out-Null
        Idem steps --log (Join-Path $dir "log.txt") --out (Join-Path $dir "steps.json") | Out-Null
        & python -c @"
import json, sys
d, offline, target, role = sys.argv[1:]
c = json.load(open(f'{d}/idempotency.json', encoding='utf-8-sig'))
s = json.load(open(f'{d}/steps.json', encoding='utf-8-sig'))
json.dump({'offline': offline == 'True', 'pin_target': target, 'idempotency_role': role,
           'idempotent': c['idempotent'], 'idempotency_reasons': c['reasons'],
           'idempotent_relaxed': c['idempotent_relaxed'], 'relaxed_reasons': c['relaxed_reasons'],
           'relaxed_manifest_keys': c['relaxed_manifest_keys'],
           'relaxed_freeze_packages': c['relaxed_freeze_packages'],
           'freeze_diff_empty': c['freeze_diff_empty'], 'freeze_diff': c['freeze_diff'],
           'changed': c['changed'],
           'manifest_keys_changed': c['manifest_keys_changed'],
           'requirements_changed': c['requirements_changed'],
           'sidecars_changed': c['sidecars_changed'],
           'markers_changed': c['markers_changed'],
           'prebuilt_changed': c['prebuilt_changed'],
           'binaries_changed': c['binaries_changed'],
           'sidecar_mtimes_before': c['sidecar_mtimes_before'],
           'sidecar_mtimes_after': c['sidecar_mtimes_after'],
           'marker_bytes_after': c['marker_bytes_after'],
           'update_steps': s['steps'], 'update_steps_ran': s['ran'],
           'update_path': s.get('update_path'), 'update_path_reason': s.get('update_path_reason'),
           'update_path_evidence': s.get('update_path_evidence'),
           'pypi_probe_answered': s.get('pypi_probe_answered')},
          open(f'{d}/extra.json', 'w'), indent=1)
"@ $dir "$Offline" $Target $role
        $ok = Finish-Step $dir $label $res $rx0 $Venv $ev $ez (Join-Path $dir "extra.json") "0"
        Clear-ChildEnv
        # EXPECT_IDEMPOTENT=0 (a branch that predates the fix under test) keeps the evidence and
        # lets report.py record the verdict instead of failing the job here.
        if (-not $ok) {
            if ($env:EXPECT_IDEMPOTENT -eq "0") {
                Write-Host "[harness] $label failed but EXPECT_IDEMPOTENT=0: recorded, not fatal"
            } else { exit 1 }
        }
    }
    "fault" {
        # Break exactly one thing, then update: exactly the affected step must run.
        $label = "fault-$Kind"; $dir = Join-Path $Out $label; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $ev = Installed-Version "unsloth"; $ez = Installed-Version "unsloth_zoo"
        if (-not $ev) { Write-Host "[harness] nothing installed in $Venv"; exit 1 }
        Snapshot (Join-Path $dir "before") "${label}:before" | Out-File (Join-Path $dir "before.line")
        Idem capture --venv $Venv --studio-home $Studio --out (Join-Path $dir "state_before.json") | Out-Null
        Idem inject $Kind --venv $Venv --studio-home $Studio --out (Join-Path $dir "injection.json")
        $proxy = Start-Proxy $dir
        Set-ChildEnv $Target $proxy @{ UNSLOTH_TAURI_UPDATE = "1"; SKIP_STUDIO_FRONTEND = "1"; UNSLOTH_DESKTOP_BACKEND_VERSION = $ev }
        Write-Host "[harness] === $label (fault $Kind injected, then studio update; installed unsloth==$ev)"
        $rx0 = Rx-Bytes
        $res = Run-Timed $dir (Join-Path $Venv "Scripts\python.exe") @("-I", "-X", "utf8", "-m", "unsloth_cli", "studio", "update")
        Idem capture --venv $Venv --studio-home $Studio --out (Join-Path $dir "state_after.json") | Out-Null
        Idem compare (Join-Path $dir "state_before.json") (Join-Path $dir "state_after.json") --out (Join-Path $dir "repair.json") | Out-Null
        Idem steps --log (Join-Path $dir "log.txt") --out (Join-Path $dir "steps.json") | Out-Null
        & python -c @"
import json, sys
d, kind = sys.argv[1:]
s = json.load(open(f'{d}/steps.json', encoding='utf-8-sig'))
r = json.load(open(f'{d}/repair.json', encoding='utf-8-sig'))
i = json.load(open(f'{d}/injection.json', encoding='utf-8-sig'))
json.dump({'fault_kind': kind, 'fault_injection': i,
           'fault_steps': s['steps'], 'fault_steps_ran': s['ran'],
           'repair_reasons': r['reasons'], 'freeze_diff': r['freeze_diff'],
           'freeze_diff_empty': r['freeze_diff_empty']},
          open(f'{d}/extra.json', 'w'), indent=1)
"@ $dir $Kind
        $ok = Finish-Step $dir $label $res $rx0 $Venv $ev $ez (Join-Path $dir "extra.json") "0"
        Clear-ChildEnv
        if (-not $ok) { exit 1 }
    }
    "old-shell-stage" {
        # PR A: an old Tauri shell asking a new CLI to stage must be refused, loudly, with no stage dir.
        $label = "old-shell-stage"; $dir = Join-Path $Out $label; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $ev = Installed-Version "unsloth"; $ez = Installed-Version "unsloth_zoo"
        Remove-Item (Join-Path $Studio ".update-failed.json") -Force -ErrorAction SilentlyContinue
        Snapshot (Join-Path $dir "before") "${label}:before" | Out-File (Join-Path $dir "before.line")
        $proxy = Start-Proxy $dir
        Set-ChildEnv $Target $proxy @{ UNSLOTH_TAURI_UPDATE = "1"; SKIP_STUDIO_FRONTEND = "1";
            UNSLOTH_TAURI_SHELL_VERSION = $ShellVersion; UNSLOTH_DESKTOP_BACKEND_VERSION = $ev }
        Write-Host "[harness] === $label (UNSLOTH_TAURI_SHELL_VERSION=$ShellVersion against installed unsloth==$ev; exit 1 expected)"
        $rx0 = Rx-Bytes
        $res = Run-Timed $dir (Join-Path $Venv "Scripts\python.exe") @("-I", "-X", "utf8", "-m", "unsloth_cli", "studio", "update", "--stage")
        & python -c @"
import json, os, sys
d, studio, shell_ver, rc = sys.argv[1:]
log = open(f'{d}/log.txt', errors='replace').read()
needle = '[TAURI:ERROR] background staging is no longer supported'
failed = os.path.join(studio, '.update-failed.json')
payload = None
if os.path.exists(failed):
    try:
        payload = json.loads(open(failed, errors='replace').read())
    except json.JSONDecodeError:
        payload = {'_unparseable': open(failed, errors='replace').read()[:500]}
stage = os.path.join(studio, '.update-stage')
res = {'shell_version': shell_ver, 'expected_exit_code': '1',
       'tauri_error_present': needle in log,
       'tauri_error_lines': [ln.strip()[:200] for ln in log.splitlines() if '[TAURI:ERROR]' in ln][:5],
       'update_failed_json_exists': os.path.exists(failed), 'update_failed_json': payload,
       'update_failed_fields_are_strings': bool(payload) and all(
           isinstance(v, str) for v in payload.values() if not isinstance(v, (dict, list))),
       'stage_dir_exists': os.path.exists(stage)}
res['pass'] = int(rc) == 1 and res['tauri_error_present'] and res['update_failed_json_exists'] and not res['stage_dir_exists']
json.dump(res, open(f'{d}/extra.json', 'w'), indent=1)
print('[harness] old-shell-stage:', json.dumps({k: res[k] for k in ('tauri_error_present', 'update_failed_json_exists', 'stage_dir_exists', 'pass')}))
"@ $dir $Studio $ShellVersion "$($res.rc)"
        $ok = Finish-Step $dir $label $res $rx0 $Venv "-" "-" (Join-Path $dir "extra.json") "1"
        Clear-ChildEnv
        if (-not $ok) { exit 1 }
    }
    "leftover-seed" {
        $dir = Join-Path $Out "leftover-seed"; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Idem leftovers seed --studio-home $Studio --out (Join-Path $dir "seeded.json")
        & python -c "import json,sys; s=json.load(open(sys.argv[1], encoding='utf-8-sig')); json.dump({'label':'leftover-seed','exit_code':0,'seeded':s['seeded']}, open(sys.argv[2],'w'), indent=1)" (Join-Path $dir "seeded.json") (Join-Path $dir "summary.json")
    }
    "leftover-check" {
        $dir = Join-Path $Out "leftover-check"; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Idem leftovers check --studio-home $Studio --out (Join-Path $dir "state.json")
        & python -c "import json,sys; s=json.load(open(sys.argv[1], encoding='utf-8-sig')); json.dump({'label':'leftover-check','exit_code':0,'leftovers_present':s['still_present'],'leftovers_gone':s['gone'],'leftovers':s['exists']}, open(sys.argv[2],'w'), indent=1)" (Join-Path $dir "state.json") (Join-Path $dir "summary.json")
    }
    "prefetch" {
        # PR C: prefetch into the uv cache only. Exit 2 means the installed CLI predates PR C
        # (recorded as unsupported, not as a failure); the live venv must not move either way.
        $label = "prefetch-$Target"; $dir = Join-Path $Out $label; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $ev = Installed-Version "unsloth"; $ez = Installed-Version "unsloth_zoo"
        $tv = Target-Version $Target
        Snapshot (Join-Path $dir "before") "${label}:before" | Out-File (Join-Path $dir "before.line")
        Idem capture --venv $Venv --studio-home $Studio --out (Join-Path $dir "state_before.json") | Out-Null
        $proxy = Start-Proxy $dir
        Set-ChildEnv $Target $proxy @{ UNSLOTH_TAURI_UPDATE = "1"; SKIP_STUDIO_FRONTEND = "1"; UNSLOTH_DESKTOP_BACKEND_VERSION = $tv }
        Write-Host "[harness] === $label (prefetch-update toward unsloth==$tv; live venv stays at $ev)"
        $rx0 = Rx-Bytes
        $res = Run-Timed $dir (Join-Path $Venv "Scripts\python.exe") @("-I", "-X", "utf8", "-m", "unsloth_cli", "studio", "prefetch-update")
        Idem capture --venv $Venv --studio-home $Studio --out (Join-Path $dir "state_after.json") | Out-Null
        Idem compare (Join-Path $dir "state_before.json") (Join-Path $dir "state_after.json") --out (Join-Path $dir "venv_delta.json") | Out-Null
        Idem prefetch --studio-home $Studio --out (Join-Path $dir "prefetch_state.json") | Out-Null
        & python -c @"
import json, sys
d, target, tv, rc = sys.argv[1:]
delta = json.load(open(f'{d}/venv_delta.json', encoding='utf-8-sig'))
pf = json.load(open(f'{d}/prefetch_state.json', encoding='utf-8-sig'))
res = {'prefetch_target': target, 'prefetch_target_version': tv,
       'prefetch_supported': int(rc) != 2, 'prefetch_exit_code': int(rc),
       'prefetch_dir_exists': pf['dir_exists'], 'prefetch_marker_exists': pf['marker_exists'],
       'prefetch_marker': pf['marker'], 'prefetch_dir_bytes': (pf.get('dir') or {}).get('bytes'),
       'live_venv_unchanged': delta['freeze_diff_empty'], 'live_venv_reasons': delta['reasons']}
json.dump(res, open(f'{d}/extra.json', 'w'), indent=1)
print('[harness] prefetch:', json.dumps({k: res[k] for k in ('prefetch_supported', 'prefetch_exit_code', 'prefetch_marker_exists', 'live_venv_unchanged')}))
"@ $dir $Target $tv "$($res.rc)"
        $ok = Finish-Step $dir $label $res $rx0 $Venv $ev $ez (Join-Path $dir "extra.json") "any"
        Clear-ChildEnv
        if (-not $ok) { exit 1 }
    }
    default { Write-Host "unknown -Cmd $Cmd"; exit 2 }
}
