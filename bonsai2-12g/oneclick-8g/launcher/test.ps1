# 检查引擎是否正常（由 测试.bat 调用）
$Root = Split-Path -Parent $PSScriptRoot
$ProgressPreference = 'SilentlyContinue'
function Say([string]$m, [string]$c = 'Gray') { Write-Host $m -ForegroundColor $c }
$C = @{}
foreach ($line in (Get-Content -LiteralPath (Join-Path $Root '设置.ini') -Encoding UTF8)) {
    $t = $line.Trim()
    if ($t -eq '' -or $t.StartsWith(';') -or $t.StartsWith('#') -or $t.StartsWith('[')) { continue }
    $i = $t.IndexOf('='); if ($i -lt 1) { continue }
    $C[$t.Substring(0, $i).Trim().ToUpper()] = ($t.Substring($i + 1) -replace '\s+[;#].*$', '').Trim()
}
foreach ($a in $args) { if ([string]$a -match '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') { $C[$Matches[1].ToUpper()] = $Matches[2] } }
$port = if ($C['PORT']) { $C['PORT'] } else { '8084' }
$h = if ($C['HOST'] -and $C['HOST'] -ne '0.0.0.0') { $C['HOST'] } else { '127.0.0.1' }
$id = if ($C['MODEL_ID']) { $C['MODEL_ID'] } else { 'qwen3.8-27b' }
$base = "http://${h}:$port/v1"
$hdr = @{}
if ($C['API_KEY']) { $hdr['Authorization'] = 'Bearer ' + $C['API_KEY'] }

Say "检查 $base ……"
try { $m = Invoke-RestMethod -Uri "$base/models" -Headers $hdr -TimeoutSec 5 }
catch { Say "[失败] 连不上引擎：$($_.Exception.Message)" 'Red'; Say '请确认 启动.bat 的窗口开着，并且已经出现 "listening"。'; exit 1 }
Say "[正常] 引擎在线，模型：$(($m.data | ForEach-Object { $_.id }) -join ', ')" 'Green'

$body = @{ model = $id; stream = $false; enable_thinking = $false; max_completion_tokens = 128
           messages = @(@{ role = 'user'; content = '用一句话介绍你自己。' }) } | ConvertTo-Json -Depth 5
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
$t0 = Get-Date
try { $w = Invoke-WebRequest -UseBasicParsing -Uri "$base/chat/completions" -Method Post -Headers $hdr -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec 300
      $r = [System.Text.Encoding]::UTF8.GetString($w.RawContentStream.ToArray()) | ConvertFrom-Json }
catch { Say "[失败] 请求出错：$($_.Exception.Message)" 'Red'; exit 1 }
$sec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
$ans = $r.choices[0].message.content
Say "[正常] 回答：$ans" 'Green'
$tps = $null
if ($r.timings -and $r.timings.predicted_per_second) { $tps = [math]::Round([double]$r.timings.predicted_per_second, 1) }
Say ("  用时 $sec 秒，生成 $($r.usage.completion_tokens) 个 token" + $(if ($tps) { "，速度 $tps token/秒" } else { '' }))
exit 0
