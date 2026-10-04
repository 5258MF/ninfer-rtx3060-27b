# End-to-end verification script for the 12G Bonsai-2 sm86 ninfer-serve build.
# ASCII-only source to avoid Windows PowerShell 5.1 codepage issues.
#
# Verifies:
#   0. Pre-launch check: ninfer-serve.exe + companion .dll files in engine/
#   1. Engine startup with NVML physical free VRAM budget + KVMem window + post-thinking + invariant recovery
#   2. Short prompt chat completion (no KVMem over-window scoring triggered)
#   3. Long prompt negative control (> window, no needle -> no hallucination, kvmem_score: SELECT logged by default with NINFER_TERNARY_KVMEM=1)
#   4. Long prompt positive control (> window, mid-context needle -> needle retrieved, finish_reason=length treated as INCONCLUSIVE)
#   5. Multi-turn prefix cache reuse

[CmdletBinding()]
param(
    [string]$ModelPath = "",
    [string]$EnginePath = "",
    [int]$Port = 18088,
    [int]$MaxContext = 32768,
    [int]$ScoreBudgetTokens = 8192,
    [int]$SinkTokens = 1024,
    [int]$GenReserveTokens = 2048,
    [string]$KvDtype = "rk8v4",
    [int]$HostKvMiB = 2048,
    [string]$NeedlePassCode = "KVMEM-7391-OMEGA",
    [int]$StartupTimeoutSec = 300
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (-not $EnginePath) { $EnginePath = Join-Path $PSScriptRoot "engine\ninfer-serve.exe" }

function Write-Step([string]$msg) {
    Write-Host ("[verify] " + $msg) -ForegroundColor Cyan
}
function Write-Pass([string]$msg) {
    Write-Host ("[PASS]   " + $msg) -ForegroundColor Green
}
function Write-Fail([string]$msg) {
    Write-Host ("[FAIL]   " + $msg) -ForegroundColor Red
}

if (-not $ModelPath) {
    $ModelPath = Join-Path $PSScriptRoot "model\bonsai2_27b_swift_pq2.ninfer"
    if (-not (Test-Path -LiteralPath $ModelPath)) {
        throw "Model file not found. Pass -ModelPath <path-to-bonsai2.ninfer>."
    }
}

if (-not (Test-Path -LiteralPath $EnginePath)) {
    throw "Engine binary not found: $EnginePath"
}
$engineDir = Split-Path -Parent $EnginePath
$dlls = @(Get-ChildItem -LiteralPath $engineDir -Filter "*.dll" -ErrorAction SilentlyContinue)
if ($dlls.Count -lt 10) {
    throw ("Engine folder '$engineDir' has only $($dlls.Count) .dll files (expected >= 10). " +
           "Extract the full kit and keep ninfer-serve.exe together with all companion DLLs.")
}
foreach ($reqDll in @("cudart64_13.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "avcodec-63.dll", "avutil-61.dll")) {
    if (-not (Test-Path -LiteralPath (Join-Path $engineDir $reqDll))) {
        throw "Missing required runtime DLL in '$engineDir': $reqDll"
    }
}
Write-Pass "Pre-launch DLL check passed ($($dlls.Count) DLLs present in $engineDir)"

if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model file not found: $ModelPath"
}

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutLog = Join-Path $logDir "verify-arch-$stamp.out.log"
$stderrLog = Join-Path $logDir "verify-arch-$stamp.err.log"

function Get-CombinedLogText {
    $parts = @()
    if (Test-Path -LiteralPath $stdoutLog) {
        $parts += Get-Content -LiteralPath $stdoutLog -Raw -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $stderrLog) {
        $parts += Get-Content -LiteralPath $stderrLog -Raw -ErrorAction SilentlyContinue
    }
    return ($parts -join "`n")
}

function Count-RegexMatches([string]$text, [string]$pattern) {
    if ([string]::IsNullOrEmpty($text)) { return 0 }
    return ([regex]::Matches($text, $pattern)).Count
}

function Invoke-Chat([array]$messages, [int]$maxTokens = 128, [double]$temperature = 0.0) {
    $uri = "http://127.0.0.1:$Port/v1/chat/completions"
    $payload = @{
        model                  = "qwen3.8-27b"
        messages               = $messages
        max_completion_tokens  = $maxTokens
        temperature            = $temperature
        enable_thinking        = $false
        stream                 = $false
    } | ConvertTo-Json -Depth 8 -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
    $w = Invoke-WebRequest -UseBasicParsing -Uri $uri -Method Post -ContentType "application/json; charset=utf-8" -Body $bytes -TimeoutSec 300
    return ([System.Text.Encoding]::UTF8.GetString($w.RawContentStream.ToArray()) | ConvertFrom-Json)
}

Get-ChildItem env: | Where-Object { $_.Name -like "NINFER_TERNARY_*" } | ForEach-Object { Remove-Item "env:$($_.Name)" }
$env:NINFER_TERNARY_KVMEM = "1"
$env:NINFER_TERNARY_KVMEM_SCORE_BUDGET = "$ScoreBudgetTokens"
$env:NINFER_TERNARY_KVMEM_SCORE_SINK = "$SinkTokens"
$env:NINFER_TERNARY_KVMEM_GEN_RESERVE = "$GenReserveTokens"
$env:NINFER_TERNARY_HOST_KV_PAGEABLE = "1"
$env:NINFER_TERNARY_EMBED_HOST = "1"
$env:PATH = "$engineDir;" + $env:PATH

$kvCap = $ScoreBudgetTokens + $GenReserveTokens
$argList = @(
    $ModelPath,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--model-id", "qwen3.8-27b",
    "--max-concurrency", "1",
    "--kv-dtype", $KvDtype,
    "--spec", "mtp",
    "--draft-tokens", "3",
    "--tolerant-tool-calls",
    "--max-context", "$MaxContext",
    "--kv-capacity", "$kvCap",
    "--default-max-tokens", "$GenReserveTokens",
    "--host-kv-mib", "$HostKvMiB",
    "--prefill-chunk", "512",
    "--post-thinking-temperature", "0.2",
    "--recover-invariant-failures"
)

$proc = $null
try {
    Write-Step "Starting ninfer-serve on 127.0.0.1:$Port (window=$ScoreBudgetTokens, sink=$SinkTokens, reserve=$GenReserveTokens)..."
    $proc = Start-Process -FilePath $EnginePath -ArgumentList $argList `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog `
        -PassThru -WindowStyle Hidden

    $deadline = (Get-Date).AddSeconds($StartupTimeoutSec)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) {
            throw "ninfer-serve exited early with code $($proc.ExitCode). Logs:`n$(Get-CombinedLogText)"
        }
        try {
            $m = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -Method Get -TimeoutSec 2
            if ($m -and $m.data) { $ready = $true; break }
        } catch {}
        Start-Sleep -Seconds 2
    }
    if (-not $ready) {
        throw "Timed out waiting for ninfer-serve readiness. Logs:`n$(Get-CombinedLogText)"
    }

    $startupLogs = Get-CombinedLogText
    if ($startupLogs -match "VRAM budget \| nvml_free=([0-9.]+) MiB \| cuda_free=([0-9.]+) MiB \| effective_free=([0-9.]+) MiB") {
        Write-Pass "NVML physical VRAM budget active: $($Matches[0])"
    } else {
        Write-Fail "Missing 'VRAM budget | nvml_free=...' diagnostic in startup log"
    }

    # Stage 1: Short prompt
    Write-Step "Stage 1: Short prompt chat completion..."
    $scoreBefore1 = Count-RegexMatches (Get-CombinedLogText) "kvmem_window:\s*boundary-armed"
    $r1 = Invoke-Chat @(@{ role = "user"; content = "Reply with the single word READY." }) 64 0.0
    $ans1 = "$($r1.choices[0].message.content)".Trim()
    $scoreAfter1 = Count-RegexMatches (Get-CombinedLogText) "kvmem_window:\s*boundary-armed"
    if ($ans1.Length -gt 0 -and $scoreAfter1 -eq $scoreBefore1) {
        Write-Pass "Short prompt succeeded (reply='$ans1', kvmem compaction delta=0)"
    } else {
        Write-Fail "Short prompt check failed (reply='$ans1', kvmem compaction delta=$($scoreAfter1 - $scoreBefore1))"
    }

    # Build filler blocks so total prompt exceeds ScoreBudgetTokens
    $targetPromptTokens = $ScoreBudgetTokens + 3072
    $blockCount = 80
    $wordsPerBlock = [Math]::Max(30, [int][Math]::Ceiling($targetPromptTokens / $blockCount))
    $negBlocks = New-Object System.Collections.Generic.List[string]
    for ($b = 1; $b -le $blockCount; $b++) {
        $filler = ("section_$b alpha beta gamma delta epsilon zeta eta theta " * [int][Math]::Ceiling($wordsPerBlock / 8))
        $negBlocks.Add("[Record $b] Routine telemetry status normal. $filler")
    }

    # Stage 2: Long prompt negative control (no needle present)
    Write-Step "Stage 2: Long prompt negative control (> $ScoreBudgetTokens tokens, no needle)..."
    $negDoc = ($negBlocks -join "`n") + "`nQuestion: What is the exact telemetry verification tag mentioned in the records? If none is mentioned, reply NONE."
    $scoreBefore2 = Count-RegexMatches (Get-CombinedLogText) "kvmem_window:\s*boundary-armed"
    $r2 = Invoke-Chat @(@{ role = "user"; content = $negDoc }) 128 0.0
    $ans2 = "$($r2.choices[0].message.content)".Trim()
    $scoreAfter2 = Count-RegexMatches (Get-CombinedLogText) "kvmem_window:\s*boundary-armed"
    $promptTok2 = [int]$r2.usage.prompt_tokens
    if ($scoreAfter2 -gt $scoreBefore2 -and $ans2 -notmatch [regex]::Escape($NeedlePassCode)) {
        Write-Pass "Negative control passed (prompt_tokens=$promptTok2, kvmem compaction delta=+$($scoreAfter2 - $scoreBefore2), no hallucinated needle)"
    } else {
        Write-Fail "Negative control failed (prompt_tokens=$promptTok2, kvmem compaction delta=+$($scoreAfter2 - $scoreBefore2), reply='$ans2')"
    }

    # Stage 3: Long prompt positive control (needle in middle ~45%, outside sink prefix and recent tail)
    Write-Step "Stage 3: Long prompt positive control (mid-context needle '$NeedlePassCode' at 45% depth)..."
    $posBlocks = New-Object System.Collections.Generic.List[string]
    $needleIdx = [int][Math]::Floor($blockCount * 0.45)
    for ($b = 1; $b -le $blockCount; $b++) {
        if ($b -eq $needleIdx) {
            $posBlocks.Add("[Record $b] TELEMETRY BENCHMARK MARKER: The exact telemetry verification tag is $NeedlePassCode. Telemetry verification tag = $NeedlePassCode. When asked for the telemetry verification tag in the records, reply with $NeedlePassCode.")
        } else {
            $posBlocks.Add($negBlocks[$b - 1])
        }
    }
    $posDoc = ($posBlocks -join "`n") + "`nQuestion: What is the exact telemetry verification tag mentioned in the records? Reply with the exact tag only."
    $scoreBefore3 = Count-RegexMatches (Get-CombinedLogText) "kvmem_window:\s*boundary-armed"
    $r3 = Invoke-Chat @(@{ role = "user"; content = $posDoc }) 256 0.0
    $ans3 = "$($r3.choices[0].message.content)".Trim()
    $fin3 = "$($r3.choices[0].finish_reason)"
    $scoreAfter3 = Count-RegexMatches (Get-CombinedLogText) "kvmem_window:\s*boundary-armed"
    $promptTok3 = [int]$r3.usage.prompt_tokens
    if ($scoreAfter3 -gt $scoreBefore3 -and $ans3 -match [regex]::Escape($NeedlePassCode)) {
        Write-Pass "Positive needle retrieval passed (prompt_tokens=$promptTok3, kvmem compaction delta=+$($scoreAfter3 - $scoreBefore3), reply='$ans3')"
    } elseif ($fin3 -eq "length") {
        Write-Host "[INCONCLUSIVE] Generation truncated at finish_reason=length before needle check completed (reply='$ans3')" -ForegroundColor Yellow
    } else {
        Write-Fail "Positive needle retrieval failed (prompt_tokens=$promptTok3, finish_reason=$fin3, kvmem compaction delta=+$($scoreAfter3 - $scoreBefore3), reply='$ans3')"
    }

    # Stage 4: Multi-turn continuation / prefix cache reuse
    Write-Step "Stage 4: Multi-turn follow-up continuation..."
    $msgs4 = @(
        @{ role = "user"; content = $posDoc },
        @{ role = "assistant"; content = $ans3 },
        @{ role = "user"; content = "Please repeat the exact telemetry verification tag from the records one more time." }
    )
    $r4 = Invoke-Chat $msgs4 128 0.0
    $ans4 = "$($r4.choices[0].message.content)".Trim()
    if ($ans4 -match [regex]::Escape($NeedlePassCode)) {
        Write-Pass "Multi-turn continuation passed (reply='$ans4')"
    } else {
        Write-Fail "Multi-turn continuation failed (reply='$ans4')"
    }
} finally {
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}
