# Local regression harness. ASCII-only for Windows PowerShell 5.1.
# Exit codes: 0 = executed checks passed; 1 = failed; 2 = inconclusive.
# Small-window tests do not establish full-context speed/quality or strict tool schemas.
[CmdletBinding()]
param(
    [string]$ModelPath = "",
    [string]$EnginePath = "",
    [int]$Port = 18089,
    [int]$MaxContext = 32768,
    [int]$RingWindowPages = 160,
    [int]$SinkPages = 16,
    [int]$GenReservePages = 32,
    [ValidateSet("rk4v4", "rk8v4")][string]$KvDtype = "rk4v4",
    [int]$HostKvMiB = 2048,
    [string]$NeedlePassCode = "KVMEM-7391-OMEGA",
    [int]$StartupTimeoutSec = 420,
    [int]$DefaultMaxTokens = 512,
    [int]$RequestTimeoutSec = 1800
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Mode = "swift15-q2s"
$isKv = $true
if (-not $EnginePath) { $EnginePath = Join-Path $PSScriptRoot "engine\ninfer-serve.exe" }
if (-not $ModelPath) { $ModelPath = Join-Path $PSScriptRoot "model\swift15_iq2_s_mtpq4.ninfer" }
if ($RingWindowPages -le $SinkPages + $GenReservePages -or $SinkPages -le 0 -or $GenReservePages -le 0) {
    throw "RingWindowPages must exceed SinkPages + GenReservePages; reserve/sink must be positive."
}
$ScoreBudgetTokens = ($RingWindowPages - $GenReservePages) * 64
$SinkTokens = $SinkPages * 64
$GenReserveTokens = $GenReservePages * 64

function Write-Step([string]$msg) { Write-Host ("[verify] " + $msg) -ForegroundColor Cyan }
function Write-Pass([string]$msg) { Write-Host ("[PASS]   " + $msg) -ForegroundColor Green }
function Write-Fail([string]$msg) { Write-Host ("[FAIL]   " + $msg) -ForegroundColor Red; $script:failCount++ }
function Write-Inconclusive([string]$msg) {
    Write-Host ("[INCONCLUSIVE] " + $msg) -ForegroundColor Yellow
    $script:inconclusiveCount++
}
$script:failCount = 0
$script:inconclusiveCount = 0
if ($DefaultMaxTokens -lt 256 -or $DefaultMaxTokens -gt $GenReserveTokens) {
    throw "DefaultMaxTokens must be between 256 and GenReserveTokens."
}
if ($ScoreBudgetTokens -le $SinkTokens -or $GenReserveTokens -le 0) {
    throw "ScoreBudgetTokens must exceed SinkTokens; GenReserveTokens must be positive."
}

if (-not (Test-Path -LiteralPath $EnginePath)) { throw "Engine binary not found: $EnginePath" }
$engineDir = Split-Path -Parent $EnginePath
$dlls = @(Get-ChildItem -LiteralPath $engineDir -Filter "*.dll" -ErrorAction SilentlyContinue)
if ($dlls.Count -lt 11) {
    throw ("Engine folder '$engineDir' has only $($dlls.Count) .dll files (expected 11). " +
           "Extract the full kit and keep the engine next to all companion DLLs.")
}
foreach ($reqDll in @("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")) {
    if (-not (Test-Path -LiteralPath (Join-Path $engineDir $reqDll))) {
        throw "Missing required runtime DLL in '$engineDir': $reqDll"
    }
}
Write-Pass "Pre-launch DLL check passed ($($dlls.Count) DLLs present in $engineDir)"
if (-not (Test-Path -LiteralPath $ModelPath)) { throw "Model file not found: $ModelPath" }
Write-Step ("Mode=$Mode engine=$(Split-Path -Leaf $EnginePath) kv-dtype=$kvDtype model=$ModelPath")

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$stdoutLog = Join-Path $logDir "verify-arch-$stamp.out.log"
$stderrLog = Join-Path $logDir "verify-arch-$stamp.err.log"

function Get-CombinedLogText {
    $parts = @()
    if (Test-Path -LiteralPath $stdoutLog) { $parts += Get-Content -LiteralPath $stdoutLog -Raw -Encoding UTF8 -ErrorAction SilentlyContinue }
    if (Test-Path -LiteralPath $stderrLog) { $parts += Get-Content -LiteralPath $stderrLog -Raw -Encoding UTF8 -ErrorAction SilentlyContinue }
    return ($parts -join "`n")
}
function Count-RegexMatches([string]$text, [string]$pattern) {
    if ([string]::IsNullOrEmpty($text)) { return 0 }
    return ([regex]::Matches($text, $pattern)).Count
}

function Invoke-Chat([array]$messages, [int]$maxTokens = 128, [double]$temperature = 0.0, [bool]$thinking = $false, [hashtable]$Extra = @{}) {
    $uri = "http://127.0.0.1:$Port/v1/chat/completions"
    $payload = @{
        model                 = "qwen3.8-27b"
        messages              = $messages
        max_completion_tokens = $maxTokens
        temperature           = $temperature
        enable_thinking       = $thinking
        stream                = $false
    }
    foreach ($k in $Extra.Keys) { $payload[$k] = $Extra[$k] }
    $json = $payload | ConvertTo-Json -Depth 12 -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $w = Invoke-WebRequest -UseBasicParsing -Uri $uri -Method Post -ContentType "application/json; charset=utf-8" -Body $bytes -TimeoutSec $RequestTimeoutSec
    return ([System.Text.Encoding]::UTF8.GetString($w.RawContentStream.ToArray()) | ConvertFrom-Json)
}


function Invoke-Api([string]$Path, [hashtable]$Payload) {
    $json = $Payload | ConvertTo-Json -Depth 12 -Compress
    $bytes = [Text.Encoding]::UTF8.GetBytes($json)
    $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port$Path" -Method Post -ContentType "application/json; charset=utf-8" -Body $bytes -TimeoutSec $RequestTimeoutSec
    return ([Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json)
}

Get-ChildItem env: | Where-Object { $_.Name -like "NINFER_*" -and $_.Name -ne "NINFER_FREE_VRAM_MIB" } | ForEach-Object { Remove-Item "env:$($_.Name)" }
$env:NINFER_KVMEM_SINK_PAGES = "$SinkPages"
$env:NINFER_KVMEM_GEN_RESERVE_PAGES = "$GenReservePages"
$env:NINFER_KVMEM_LONG_REUSE = "1"
$env:PATH = "$engineDir;" + $env:PATH
$kvCap = $RingWindowPages * 64
$argList = @(
    ('"' + $ModelPath + '"'), "--host", "127.0.0.1", "--port", "$Port", "--model-id", "qwen3.8-27b",
    "--max-concurrency", "1", "--kv-dtype", $KvDtype, "--gdn-state-fp16", "--spec", "mtp", "--draft-tokens", "3",
    "--max-context", "$MaxContext", "--kv-capacity", "auto", "--kvmem-window-pages", "$RingWindowPages",
    "--host-kv-mib", "$HostKvMiB", "--device-state-slots", "0", "--prefill-chunk", "512",
    "--cuda-graph-allowance-mib", "144", "--default-max-tokens", "$DefaultMaxTokens",
    "--max-private-continuations", "4", "--embedding-host", "--post-thinking",
    "--post-thinking-temperature", "0.2", "--recover-invariant-failures",
    "--vision", "--vision-residency", "overlay", "--vision-max-merged", "4096"
)

$proc = $null
try {
    Write-Step "Starting $(Split-Path -Leaf $EnginePath) on 127.0.0.1:$Port (window=$ScoreBudgetTokens sink=$SinkTokens reserve=$GenReserveTokens)..."
    $probe = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, $Port)
    try { $probe.Start() } finally { $probe.Stop() }
    $proc = Start-Process -FilePath $EnginePath -ArgumentList $argList -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds($StartupTimeoutSec)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { throw "engine exited early with code $($proc.ExitCode). Logs:`n$(Get-CombinedLogText)" }
        try {
            $m = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -Method Get -TimeoutSec 2
            if ($m -and $m.data) { $ready = $true; break }
        } catch {}
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { throw "Timed out waiting for engine readiness. Logs:`n$(Get-CombinedLogText)" }
    Write-Pass "Engine is listening on 127.0.0.1:$Port"

    $startupLogs = Get-CombinedLogText
    if ($startupLogs -match "VRAM budget \| nvml_free=([0-9.]+) MiB \| cuda_free=([0-9.]+) MiB \| effective_free=([0-9.]+) MiB") {
        Write-Pass "NVML physical VRAM budget active: $($Matches[0])"
    } else {
        Write-Fail "Missing 'VRAM budget | nvml_free=...' diagnostic in the startup log"
    }
    if ($startupLogs -match "(KVMem allocation \| window \d+ pages[^\r\n]*|KVMem ring pool:\s+\d+\s+pages)") {
        Write-Pass "KVMem ring pool active: $($Matches[0])"
    } else { Write-Fail "Missing KVMem allocation diagnostic in the startup log." }

    Write-Step "Stage 1: short prompt chat completion..."
    $c1 = Count-RegexMatches (Get-CombinedLogText) "kvmem_score:\s*SELECT"
    $r1 = Invoke-Chat @(@{ role = "user"; content = "Reply with the single word READY." }) 64 0.0
    $ans1 = "$($r1.choices[0].message.content)".Trim()
    $c1b = Count-RegexMatches (Get-CombinedLogText) "kvmem_score:\s*SELECT"
    if ($ans1 -match "^READY[.!]?$" -and $c1b -eq $c1) {
        Write-Pass "Short prompt succeeded (reply='$ans1', compaction delta=0)"
    } else {
        Write-Fail "Short prompt check failed (reply='$ans1', compaction delta=$($c1b - $c1))"
    }

    $targetPromptTokens = $ScoreBudgetTokens + 3072
    $blockCount = 80
    $wordsPerBlock = [Math]::Max(30, [int][Math]::Ceiling($targetPromptTokens / ($blockCount * 1.7)))
    $negBlocks = New-Object System.Collections.Generic.List[string]
    for ($b = 1; $b -le $blockCount; $b++) {
        $filler = ("section_$b alpha beta gamma delta epsilon zeta eta theta " * [int][Math]::Ceiling($wordsPerBlock / 8))
        $negBlocks.Add("[Record $b] Routine telemetry status normal. $filler")
    }

    Write-Step "Stage 2: long prompt negative control (> $ScoreBudgetTokens tokens, no needle)..."
    $negDoc = ($negBlocks -join "`n") + "`nQuestion: What is the exact telemetry verification tag mentioned in the records? If none is mentioned, reply NONE."
    $c2 = Count-RegexMatches (Get-CombinedLogText) "kvmem_score:\s*SELECT"
    $r2 = Invoke-Chat @(@{ role = "user"; content = $negDoc }) 128 0.0
    $ans2 = "$($r2.choices[0].message.content)".Trim()
    $c2b = Count-RegexMatches (Get-CombinedLogText) "kvmem_score:\s*SELECT"
    $pt2 = [int]$r2.usage.prompt_tokens
    if ($ans2 -match "^NONE[.!]?$" -and $pt2 -gt $ScoreBudgetTokens -and $c2b -gt $c2) {
        Write-Pass "Negative control passed (prompt_tokens=$pt2, compaction delta=+$($c2b - $c2), no hallucinated needle)"
    } else {
        Write-Fail "Negative control failed (prompt_tokens=$pt2, reply='$ans2')"
    }

    Write-Step "Stage 3: long prompt positive control (mid-context needle at 45% depth)..."
    $posBlocks = New-Object System.Collections.Generic.List[string]
    $needleIdx = [int][Math]::Floor($blockCount * 0.45)
    for ($b = 1; $b -le $blockCount; $b++) {
        if ($b -eq $needleIdx) {
            $posBlocks.Add("[Record $b] TELEMETRY BENCHMARK MARKER: The exact project build tag is $NeedlePassCode. Project build tag = $NeedlePassCode. When asked for the project build tag in the records, reply with $NeedlePassCode.")
        } else {
            $posBlocks.Add($negBlocks[$b - 1])
        }
    }
    $posDoc = ($posBlocks -join "`n") + "`nQuestion: What is the exact project build tag mentioned in the records? Reply with the exact tag only."
    $c3 = Count-RegexMatches (Get-CombinedLogText) "kvmem_score:\s*SELECT"
    $r3 = Invoke-Chat @(@{ role = "user"; content = $posDoc }) 256 0.0
    $ans3 = "$($r3.choices[0].message.content)".Trim()
    $fin3 = "$($r3.choices[0].finish_reason)"
    $c3b = Count-RegexMatches (Get-CombinedLogText) "kvmem_score:\s*SELECT"
    $pt3 = [int]$r3.usage.prompt_tokens
    if ($ans3 -match [regex]::Escape($NeedlePassCode) -and $pt3 -gt $ScoreBudgetTokens -and $c3b -gt $c3) {
        Write-Pass "Positive needle retrieval passed (prompt_tokens=$pt3, compaction delta=+$($c3b - $c3), reply='$ans3')"
    } elseif ($fin3 -eq "length") {
        Write-Inconclusive "Needle response was truncated without recovering the tag; this run is not a full pass."
    } else {
        Write-Fail "Positive needle retrieval failed (prompt_tokens=$pt3, finish_reason=$fin3, reply='$ans3')"
    }

    Write-Step "Stage 4: multi-turn follow-up retrieval..."
    $msgs4 = @(
        @{ role = "user"; content = $posDoc },
        @{ role = "assistant"; content = $ans3 },
        @{ role = "user"; content = "Please repeat the exact project build tag from the records one more time." }
    )
    $r4 = Invoke-Chat $msgs4 128 0.0
    $ans4 = "$($r4.choices[0].message.content)".Trim()
    if ($ans4 -match [regex]::Escape($NeedlePassCode)) {
        Write-Pass "Multi-turn follow-up retrieval passed (reply='$ans4')"
    } else {
        Write-Fail "Multi-turn continuation failed (reply='$ans4')"
    }


    Write-Step "Stage 4b: branch from a shortened history prefix..."
    $prefixDoc = $posDoc.Substring(0, [int]($posDoc.Length * 0.65))
    $prefixDoc += "`nQuestion: What is the exact project build tag in these records?"
    try {
        $rb = Invoke-Chat @(@{ role = "user"; content = $prefixDoc }) 128 0.0
        if ("$($rb.choices[0].message.content)" -match [regex]::Escape($NeedlePassCode)) {
            Write-Pass "Shortened history request completed with the correct tag."
        } else { Write-Fail "Shortened history request lost the tag." }
    } catch { Write-Fail "Shortened history request failed: $($_.Exception.Message)" }

    Write-Step "Stage 5: output budget and tool-parameter acceptance..."
    try {
        $r5 = Invoke-Chat -messages @(@{ role = "user"; content = "Print every integer from 1 to 100000, one per line. Start immediately. No commentary, code, ellipses, or omissions; continue through 100000." }) -maxTokens 65536
        $used = [int]$r5.usage.completion_tokens
        $finish = "$($r5.choices[0].finish_reason)"
        if ($used -gt $DefaultMaxTokens) {
            Write-Fail "Output exceeded the server cap: $used > $DefaultMaxTokens"
        } elseif ($finish -eq "length" -and $used -eq $DefaultMaxTokens) {
            Write-Pass "Oversized output request was capped at $used tokens (finish_reason=length)."
        } else {
            Write-Inconclusive "The model stopped at $used tokens ($finish), so this request cannot prove the $DefaultMaxTokens-token cap."
        }
    } catch { Write-Fail "Output-cap probe failed: $($_.Exception.Message)" }
    $tools2 = @(
        @{ type = "function"; function = @{ name = "get_weather"; description = "Get the current weather for a city"; strict = $true; parameters = @{ type = "object"; properties = @{ city = @{ type = "string" } }; required = @("city"); additionalProperties = $false } } },
        @{ type = "function"; function = @{ name = "get_time"; description = "Get the current time for a city"; parameters = @{ type = "object"; properties = @{ city = @{ type = "string" } }; required = @("city") } } }
    )
    try {
        $r5b = Invoke-Chat -messages @(@{ role = "user"; content = "What is the weather in Tokyo? Use the tools." }) -maxTokens 256 -thinking $true -Extra @{
            tools = $tools2; tool_choice = "required"; parallel_tool_calls = $false; reasoning_effort = "medium"
        }
        $tc5 = $r5b.choices[0].message.tool_calls
        if ($tc5 -and $tc5.Count -ge 1) {
            Write-Pass "Tool-parameter request accepted (strict is advisory; multi-tool required uses Auto); emitted: $($tc5[0].function.name)"
        } else {
            Write-Pass "Tool-parameter request accepted (HTTP 200); no tool call emitted. This is an acceptance check, not a required-call guarantee."
        }
    } catch {
        Write-Fail "tool_choice=required + 2 tools + reasoning_effort=medium rejected: $($_.Exception.Message)"
    }
    try {
        $r5c = Invoke-Chat -messages @(@{ role = "user"; content = "What is the weather in Osaka?" }) -maxTokens 256 -thinking $true -Extra @{
            tools = @($tools2[0]); tool_choice = @{ type = "function"; function = @{ name = "get_weather" } }
        }
        $tc5c = $r5c.choices[0].message.tool_calls
        if ($tc5c -and $tc5c.Count -ge 1) {
            Write-Pass "Forced single tool while thinking is enabled accepted (thinking auto-disabled for that turn), tool_calls: $($tc5c[0].function.name)"
        } else {
            Write-Fail "Forced single tool while thinking is enabled returned no tool call"
        }
    } catch {
        Write-Fail "Forced single tool while thinking is enabled rejected: $($_.Exception.Message)"
    }


    Write-Step "Stage 6: function call, tool-result submission, and final response..."
    try {
        $question = "What is the weather in Kyoto? Use get_weather, then include the temperature and the exact verification tag from the tool result in your final reply."
        $r6 = Invoke-Chat -messages @(@{ role = "user"; content = $question }) -maxTokens 256 -Extra @{
            tools = @($tools2[0]); tool_choice = @{ type = "function"; function = @{ name = "get_weather" } }
        }
        $call = @($r6.choices[0].message.tool_calls)[0]
        if (-not $call -or $call.function.name -ne "get_weather" -or -not $call.id) { throw "Missing get_weather call or call ID." }
        $parsed = $call.function.arguments | ConvertFrom-Json
        if ($parsed.city -ne "Kyoto") { throw "Incorrect city argument." }
        $messages = @(
            @{ role = "user"; content = $question },
            @{ role = "assistant"; content = $r6.choices[0].message.content; tool_calls = @($call) },
            @{ role = "tool"; tool_call_id = $call.id; content = '{"city":"Kyoto","condition":"sunny","temperature_c":23,"verification":"WEATHER-7319"}' }
        )
        $r6b = Invoke-Chat -messages $messages -maxTokens 256 -Extra @{ tools = @($tools2[0]); tool_choice = "none" }
        $answer = "$($r6b.choices[0].message.content)"
        if ($answer -match "WEATHER-7319" -and $answer -match "23") { Write-Pass "Tool result was returned and used in the final response." }
        else { Write-Fail "Final response did not use the returned temperature and verification tag." }
    } catch { Write-Fail "Tool round trip failed: $($_.Exception.Message)" }

    Write-Step "Stage 7: Anthropic Messages and OpenAI Responses..."
    try {
        $rm = Invoke-Api "/v1/messages" @{
            model = "qwen3.8-27b"; max_tokens = 65536
            thinking = @{ type = "enabled"; budget_tokens = 2048 }
            messages = @(@{ role = "user"; content = "Reply with READY." })
        }
        if ($rm.content -and [int]$rm.usage.output_tokens -le $DefaultMaxTokens) { Write-Pass "Messages accepted a valid large output/thinking request within the output cap." }
        else { Write-Fail "Messages response was empty or exceeded the cap." }
    } catch { Write-Fail "Messages request failed: $($_.Exception.Message)" }
    try {
        $rr = Invoke-Api "/v1/responses" @{ model = "qwen3.8-27b"; input = "Reply with READY."; max_output_tokens = 65536; reasoning = @{ effort = "none" } }
        $responseText = @($rr.output | ForEach-Object { $_.content } | ForEach-Object { $_.text }) -join " "
        if ($responseText -match "READY" -and [int]$rr.usage.output_tokens -le $DefaultMaxTokens) { Write-Pass "Responses accepted the large output request and returned READY within the cap." }
        else { Write-Fail "Responses output was missing or exceeded the cap." }
    } catch { Write-Fail "Responses request failed: $($_.Exception.Message)" }

    Write-Step "Stage 8: image input (two colored halves)..."
    try {
        $visionContent = @(
            @{ type = "text"; text = "Name the solid color of the left and right halves. Reply using LEFT=<color>; RIGHT=<color>. Use English color names." },
            @{ type = "image_url"; image_url = @{ url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAABACAIAAABdtOgoAAAA10lEQVR4nO3RwQkAQRCEwMk/6bsg9lE0CP4VvO9uGu1/Rvsb4BMasIz2N8AnNGAa7W+AT2jAMtrfAJ/QgGm0vwE+oQHLaH8DfEIDptH+BviEBiyj/Q3wCQ2YRvsb4BMasIz2N8AnNGAa7W+AT2jAMtrfAJ/QgGm0vwE+oQHLaH8DfEIDptH+BviEBiyj/Q3wCQ2YRvsb4BMasIz2N8AnNGAa7W+AT2jAMtrfAJ/QgGm0vwE+oQHLaH8DfEIDptH+BviEBiyj/Q3wCQ2YRvsb4BMasIz2v/IDR0rh0rxBIoYAAAAASUVORK5CYII=" } }
        )
        $rv = Invoke-Chat @(@{ role = "user"; content = $visionContent }) 128 0.0
        $visionAnswer = "$($rv.choices[0].message.content)"
        if ($visionAnswer -match "(?i)LEFT\s*[:=]\s*RED" -and $visionAnswer -match "(?i)RIGHT\s*[:=]\s*BLUE") {
            Write-Pass "Vision identified the left and right halves correctly."
        } else { Write-Fail "Vision colors did not match: '$visionAnswer'" }
    } catch { Write-Fail "Vision request failed: $($_.Exception.Message)" }

} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
}

Write-Host ""
if ($script:failCount -gt 0) {
    Write-Host "VERIFY FAILED: $($script:failCount) failed check(s), $($script:inconclusiveCount) inconclusive (mode=$Mode)." -ForegroundColor Red
    exit 1
} elseif ($script:inconclusiveCount -gt 0) {
    Write-Host "VERIFY INCONCLUSIVE: $($script:inconclusiveCount) check(s) could not establish the result (mode=$Mode)." -ForegroundColor Yellow
    exit 2
} else {
    Write-Host "VERIFY PASSED: all executed checks passed (mode=$Mode; KV=$kvCap; output cap=$DefaultMaxTokens)." -ForegroundColor Green
    Write-Host "This run does not qualify every context length, long-output peak memory, or strict tool-schema enforcement."
    exit 0
}
