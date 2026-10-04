# ninfer 3060 一键启动器（由 启动.bat 调用）
# 用法：启动.bat [模式] [键=值 ...] [其他 ninfer-serve 参数 ...]
#   不写模式（设置.ini 里 MODE=ask）：先显示当前配置，回车直接启动，按 C 一步步重新选，按 N/K/R/V 直接换模式
#   模式：normal / kvmem / rk8v4 / kvrk ；写了就不显示菜单，直接启动
#   键=值：临时覆盖 设置.ini 里的同名项，例如 PORT=8085 KVMEM_ANSWER=8192（不会写回 设置.ini）
#   其他参数：原样加在最后，例如 --greedy（同一个参数写两次时，以最后一次为准）
# 向导里选的值，在按回车启动时写回 设置.ini（DRYRUN=1 时不写）
# 模型不在包里：第一次用时（配置完、按回车后）自动从魔搭社区下载，支持断点续传，下完校验 SHA256

$ErrorActionPreference = 'Stop'
$CpuRetrieval = ($env:NINFER_TERNARY_KVMEM_CPU_RETRIEVAL -ne '0')
$Root = Split-Path -Parent $PSScriptRoot
try { $Host.UI.RawUI.WindowTitle = 'ninfer 3060 引擎' } catch {}

function Say([string]$m, [string]$c = 'Gray') { Write-Host $m -ForegroundColor $c }
function Fail([string]$m) { Say "[错误] $m" 'Red'; exit 1 }
function Warn([string]$m) { Say "[注意] $m" 'Yellow' }

# ---------- 1. 读 设置.ini ----------
$ini = Join-Path $Root '设置.ini'
if (-not (Test-Path -LiteralPath $ini)) { Fail "找不到设置文件：$ini" }
$C = @{}
foreach ($line in (Get-Content -LiteralPath $ini -Encoding UTF8)) {
    $t = $line.Trim()
    if ($t -eq '' -or $t.StartsWith(';') -or $t.StartsWith('#') -or $t.StartsWith('[')) { continue }
    $i = $t.IndexOf('=')
    if ($i -lt 1) { continue }
    $k = $t.Substring(0, $i).Trim().ToUpper()
    $v = ($t.Substring($i + 1) -replace '\s+[;#].*$', '').Trim()
    $C[$k] = $v
}

# ---------- 2. 命令行 ----------
$Modes = @('normal', 'kvmem', 'rk8v4', 'kvrk')
$Mode = $null
$CliExtra = @()
foreach ($a in $args) {
    $s = [string]$a
    if (-not $Mode -and ($Modes -contains $s.ToLower())) { $Mode = $s.ToLower(); continue }
    if ($s -match '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') { $C[$Matches[1].ToUpper()] = $Matches[2]; continue }
    $CliExtra += $s
}

function Cfg([string]$k, $d) { if ($C.ContainsKey($k) -and $C[$k] -ne '') { return $C[$k] } else { return $d } }
function CfgInt([string]$k, [int]$d) {
    $v = Cfg $k $d
    $n = 0
    if (-not [int]::TryParse([string]$v, [ref]$n)) { Fail "设置项 $k 必须是整数，现在是：$v" }
    return $n
}

# ---------- 模型（魔搭社区 ModelScope；两个文件大小一样，显存占用也一样） ----------
$Models = @{
    swift = @{ Name = 'Swift-Bonsai-2'; Note = '推荐。思考更短、出结果更快'
               File = 'bonsai2_27b_swift_pq2.ninfer'; Size = [long]8306927628
               Sha = 'cc54be3800099ada67165ad352450be83d28e8c4d423be9cae573a7b6e6350a0'
               Page = 'https://www.modelscope.cn/models/fyb423/Swift-Bonsai-2-27B-NInfer'
               Url = 'https://www.modelscope.cn/models/fyb423/Swift-Bonsai-2-27B-NInfer/resolve/master/bonsai2_27b_swift_pq2.ninfer' }
    base  = @{ Name = 'Bonsai-2 原版'; Note = '作者测试难题略好，但思考更长、出结果更慢'
               File = 'ternary-bonsai-2-27b.ninfer'; Size = [long]8306927628
               Sha = '05bbbf01090c6f61113b54556daa22ad0b45036a078af6cd6f02a3fde47ad76c'
               Page = 'https://www.modelscope.cn/models/Lxt1992/ninfer-ternary-bonsai2'
               Url = 'https://www.modelscope.cn/models/Lxt1992/ninfer-ternary-bonsai2/resolve/master/models/ternary-bonsai-2-27b.ninfer' }
}
# 测试用：用一个小文件代替 Swift 模型走一遍下载流程
if ($env:ONECLICK_TEST_URL) { $Models.swift.Url = $env:ONECLICK_TEST_URL; $Models.swift.Size = [long]$env:ONECLICK_TEST_SIZE; $Models.swift.Sha = $env:ONECLICK_TEST_SHA; $Models.swift.File = $env:ONECLICK_TEST_FILE }
function Model-Key { $v = ([string](Cfg 'MODEL' 'swift')).Trim().ToLower(); if ($Models.ContainsKey($v)) { return $v } else { return $null } }   # $null = 设置.ini 里写的是自定义路径
function Model-Path {
    $k = Model-Key
    if ($k) { return (Join-Path $Root ('model\' + $Models[$k].File)) }
    $x = [string](Cfg 'MODEL' ''); if (-not [System.IO.Path]::IsPathRooted($x)) { $x = Join-Path $Root $x }; return $x
}
function Model-State {
    $x = Model-Path; if (-not (Test-Path -LiteralPath $x)) { return 'missing' }
    $k = Model-Key; if ($k -and (Get-Item -LiteralPath $x).Length -ne $Models[$k].Size) { return 'bad' }; return 'ok'
}
function GiB([long]$b) { return ('{0:N1}' -f ($b / 1GB)) }

# ---------- 上限表（RTX 3060 12GB，Swift-Bonsai-2，2026-10-01 停机实测） ----------
#   10-01 新引擎：词嵌入（322 MiB）和视觉权重（282 MiB）都放内存，看图开 / 关上限一样（v = h = n）
#   判定：引擎峰值 ≤ 10758 MiB（12288 − 桌面最多约 1330 − 留 200）；每档跑过 vfy / 藏针 / smoke / 长文藏针
#   普通 / rk8v4 模式是总上下文的上限；KVMem 两种模式是 检索窗口 + 单次回答 的上限
$ModeLimits = @{
    normal = @{ v = 90112;  h = 90112;  n = 90112 }
    rk8v4  = @{ v = 114688; h = 114688; n = 114688 }
    kvmem  = @{ v = 77824;  h = 77824;  n = 77824 }
    kvrk   = @{ v = 102400; h = 102400; n = 102400 }
}
# CPU 索引释放约 400 MiB；普通模式的上限不变。
if ($CpuRetrieval) {
    foreach ($name in @('v','h','n')) {
        $ModeLimits.kvmem[$name] = 88064
        $ModeLimits.kvrk[$name] = 114688
    }
}
$WinMax = 253952; $WinMin = 24576 # 手动窗口合法范围；自动窗口取实际容量减回答
$Pre = @{ normal = 'NORMAL'; rk8v4 = 'RK8V4'; kvmem = 'KVMEM'; kvrk = 'KVRK' }
$ModeName = @{ normal = '普通（全部放显存，int8）'; rk8v4 = 'rk8v4（全部放显存，KV 更省）'
               kvmem = 'KVMem（历史放内存，int8）'; kvrk = 'KVMem + rk8v4（历史放内存，KV 更省）' }
# 2026-10-03：按空闲显存分配 KVMem，查询暂存也计入预算。
$LoadMiB = $(if ($CpuRetrieval) { 7793.75 } else { 8050 }); $KeepMiB = 170
$TokMiB = @{ int8 = 0.0342; rk8v4 = 0.0264 }
$QueryMax = 512; $StashMiB = $(if ($CpuRetrieval) { 0 } else { 144 })
$OutMin = 8192; $WinHard = 128
function Free-MiB {
    if ($env:ONECLICK_FREE_MIB) { return [double]$env:ONECLICK_FREE_MIB }
    try {
        $line = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        return [double]$line.Trim()
    } catch { return -1 }
}
function Physical-Cap($v, [double]$free = (Free-MiB)) {
    $top = [Math]::Min($v.Lim, $v.Ctx - 8192)
    if ($free -lt 0) { return $top }
    $n = [int][Math]::Floor(($free - $LoadMiB - $KeepMiB - $StashMiB) / $TokMiB[$v.Kv] / 1024) * 1024
    return [Math]::Max(0, [Math]::Min($top, $n))
}
function Harness-Out([int]$cap, [int]$ctx, [int]$unit = 64) {
    $target = [int][Math]::Floor($ctx * 9.0 / 64 / $unit) * $unit
    $available = [Math]::Max(0, [int][Math]::Floor(($cap - $unit - 8192) / $unit) * $unit)
    return [Math]::Max(8192, $cap - $unit - [Math]::Min($target, $available))
}
$Dirty = [ordered]@{}                # 向导改过、启动时要写回 设置.ini 的项
foreach ($k in @('KVMEM_ANSWER','KVRK_ANSWER','KVMEM_SINK')) { $C[$k]='auto'; $Dirty[$k]='auto' }
foreach ($k in @('KVMEM_WINDOW','KVRK_WINDOW')) { $C[$k]='0'; $Dirty[$k]='0' }

function Fmt([int]$n) { if ($n -gt 0 -and $n % 1024 -eq 0) { return ('{0}（{1}K）' -f $n, ($n / 1024)) } else { return "$n" } }
function VisKey { if ((CfgInt 'VISION' 1) -eq 0) { return 'n' } elseif ((CfgInt 'VISION_HOST' 0) -eq 1) { return 'h' } else { return 'v' } }
function Set-C([string]$k, $v) { $C[$k] = "$v"; $Dirty[$k] = "$v" }

# 按当前设置算出某个模式的各项数值（不报错、不改东西）
function Get-View([string]$m) {
    $p = $Pre[$m]
    $r = @{ Mode=$m; On=($m -eq 'kvmem' -or $m -eq 'kvrk'); Kv=$(if ($m -eq 'rk8v4' -or $m -eq 'kvrk') { 'rk8v4' } else { 'int8' }); Lim=($ModeLimits[$m][(VisKey)]); Think=(CfgInt 'THINK_BUDGET' 0); Err=@(); Warn=@() }
    if ($r.On) {
        $r.Ctx = CfgInt "${p}_CTX" 262144
        $r.AutoOut = $true
        $r.Cap = Physical-Cap $r
        $r.CapAuto = $r.Cap
        $r.Ans = Harness-Out $r.Cap $r.Ctx
        $r.Win = $r.Cap - $r.Ans
        $r.WinTop = $r.Win
        $r.Mt = $r.Ans
        if ($r.Cap -lt 8384) { $r.Err += '显存容量不足：连完整开头、最低 8K 输出和检索块都放不下。请释放显存后重试。' }
        $min = [int][Math]::Floor($r.Ctx / 8.0 / 64) * 64
        if ($r.Win -lt $min + 64) { $r.Warn += '实际请求将保留完整系统提示及最低 8K 输出，检索可能低于推荐范围。' }
    } else {
        $r.Ctx = CfgInt "${p}_CTX" $(if ($m -eq 'rk8v4') { 114688 } else { 90112 })
        $r.Mt = CfgInt "${p}_MAX_TOKENS" 32768
        $r.Cap = $r.Ctx
    }
    if ($r.Mt -gt $r.Ctx) { $r.Err += '客户端输出上限不能大于总上下文。' }
    return $r
}

function Show-Summary([string]$m) {
    $v = Get-View $m
    Say ''
    Say '  ┌──────────── 当前配置 ────────────'
    Say ('  │ 模式      : ' + $ModeName[$m])
    Say ('  │ 模型文件  : ' + (Model-State))
    Say ('  │ 总上下文  : ' + (Fmt $v.Ctx))
    if ($v.On) {
        Say ('  │ 驻留容量  : ' + (Fmt $v.Cap) + '，按空闲显存估算，放不下时自动缩小重试')
        Say '  │ 固定开头  : 按实际系统、developer 指令及工具定义准确计数'
        $lo = [int][Math]::Floor($v.Ctx / 8.0 / 64) * 64
        $hi = [int][Math]::Floor($v.Ctx * 9.0 / 64 / 64) * 64
        Say ('  │ 检索目标  : ' + (Fmt $lo) + ' 到 ' + (Fmt $hi) + '，不含固定开头；不足时先缩检索')
        Say ('  │ Harness   : contextWindow=' + $v.Ctx + '，maxTokens=' + $v.Mt + '（启动前申请上限）')
        Say '  │ 实际输出  : 收到请求后使用剩余容量，至少预留 8K；包含思考和正文'
    } else { Say ('  │ 单次输出  : ' + (Fmt $v.Mt)) }
    foreach ($message in $v.Warn) { Warn $message }
    foreach ($message in $v.Err) { Say ('  │ [错误] ' + $message) 'Red' }
    Say '  └──────────────────────────────'
}

# 测试用：ONECLICK_INPUT='c|1|2||' 按 | 分成一次次输入
$TestInput = New-Object System.Collections.Queue
if ($env:ONECLICK_INPUT) { $env:ONECLICK_INPUT.Split('|') | ForEach-Object { $TestInput.Enqueue($_) } }
function Read-Answer([string]$prompt) {
    if ($TestInput.Count -gt 0) { $a = [string]$TestInput.Dequeue(); Write-Host "$prompt$a"; return $a.Trim() }
    if ($env:ONECLICK_INPUT) { throw 'test input exhausted' }
    return ([string](Read-Host $prompt)).Trim()
}
function Parse-Tokens([string]$s) {
    $s = $s.Trim().ToLower()
    if ($s -match '^(\d+)k$') { return [int]$Matches[1] * 1024 }
    if ($s -match '^\d{3,7}$') { return [int]$s }
    return $null
}

# 一步：列出选项，回车 = 保持当前；allowNum 时也能直接输入数字（向下取整到 1024，并检查范围）
function Ask-Step([string]$title, [string[]]$intro, $items, $current, [bool]$allowNum, [int]$numMin = 0, [int]$numMax = 0) {
    Say ''
    Say "  ── $title ──" 'Cyan'
    foreach ($l in $intro) { Say "    $l" }
    for ($i = 0; $i -lt $items.Count; $i++) { Say ('   [{0}] {1}' -f ($i + 1), $items[$i].Label) }
    $curLabel = ($items | Where-Object { "$($_.Value)" -eq "$current" } | Select-Object -First 1).Label
    if (-not $curLabel) { $curLabel = "$current" }
    Say "   直接回车 = 保持：$($curLabel.Split('　')[0])"
    if ($allowNum) { Say "   也可以直接输入数字（$numMin 到 $numMax），例如 100000 或 96k" }
    while ($true) {
        $a = Read-Answer '   请选择: '
        if ($a -eq '') { return "$current" }
        if ($a -eq 'auto' -or $a -eq '0') {
            $direct = $items | Where-Object { "$($_.Value)" -eq $a } | Select-Object -First 1
            if ($direct) { return "$($direct.Value)" }
        }
        if ($a -match '^\d{1,2}$' -and [int]$a -ge 1 -and [int]$a -le $items.Count) { return "$($items[[int]$a - 1].Value)" }
        if ($allowNum) {
            $n = Parse-Tokens $a
            if ($n) {
                $n = [int]([Math]::Floor($n / 1024) * 1024)
                if ($n -ge $numMin -and $n -le $numMax) { return "$n" }
                Say "   要在 $numMin 到 $numMax 之间，请重新输入" 'Yellow'; continue
            }
        }
        Say '   看不懂这个输入，请重新输入' 'Yellow'
    }
}

function Run-Wizard([string]$m) {
    $v = Get-View $m; $n = 7
    $mk = Model-Key; $mcur = $(if ($mk) { $mk } else { [string](Cfg 'MODEL' 'swift') })
    $items = @()
    foreach ($k in @('swift', 'base')) {
        $x = $Models[$k]; $got = Test-Path -LiteralPath (Join-Path $Root ('model\' + $x.File))
        $items += @{ Value = $k; Label = ('{0}　{1}（{2}）' -f $x.Name, $x.Note, $(if ($got) { '已下载' } else { '还没下载，' + (GiB $x.Size) + ' GiB，配置完自动下载' })) }
    }
    $sel = Ask-Step "第 1/$n 步：模型" @('两个模型显存占用一样，后面的上限也一样。没下载的会在配置完、按回车后从魔搭社区自动下载') $items $mcur $false
    Set-C 'MODEL' $sel

    $on = Ask-Step "第 2/$n 步：是否开启 KVMem" @(
        '关：全部对话记录放显存，最快、最稳，但上下文最长 88K（int8）/ 112K（rk8v4）',
        '开：显存只放"检索窗口 + 本次回答"，其余历史放内存，上下文可到 256K；',
        '    每轮先从历史里挑相关内容，长对话每轮多约 1–2 秒，极少数情况会漏掉远处细节') @(
        @{ Value = '1'; Label = '开　　长对话 / 大文档（推荐）' },
        @{ Value = '0'; Label = '关　　日常短对话，追求最快' }) $(if ($v.On) { '1' } else { '0' }) $false

    $kv = Ask-Step "第 3/$n 步：KV 量化（显存里对话记录的存储格式）" @() @(
        @{ Value = 'int8';  Label = 'int8　　 精度几乎无损、最快（推荐）' },
        @{ Value = 'rk8v4'; Label = 'rk8v4　　同样显存多放约 30%，质量只差约 0.1%，慢约 5%。KVMem 下单次回答能开到 32K 以上' }) $v.Kv $false
    if ($on -eq '1') { $m = $(if ($kv -eq 'rk8v4') { 'kvrk' } else { 'kvmem' }) } else { $m = $(if ($kv -eq 'rk8v4') { 'rk8v4' } else { 'normal' }) }
    $p = $Pre[$m]; $L0 = $ModeLimits[$m]

    $unit = $(if ($on -eq '1') { '显存上限（检索窗口 + 回答）' } else { '总上下文上限' })
    $cur = $(if ((CfgInt 'VISION' 1) -eq 0) { '0' } else { '1' })
    $vis = Ask-Step "第 4/$n 步：看图" @(
        "视觉权重放在内存里，看图开 / 关上限一样。下面的上限是这个组合的$unit") @(
        @{ Value = '1'; Label = ('开　上限 {0,-6}　推荐，可以发图片（每张图多约 0.1 秒）' -f $L0.v) },
        @{ Value = '0'; Label = ('关　上限 {0,-6}　不能发图（不再省显存）' -f $L0.n) }) $cur $false
    Set-C 'VISION' $(if ($vis -eq '0') { '0' } else { '1' })
    $L = $ModeLimits[$m][(VisKey)]
    $vn = "按第 4 步看图「" + @{ '1' = '开'; '0' = '关' }[$vis] + "」算"
    $v = Get-View $m

    if ($on -eq '1') {
        $c0 = $v.Ctx; if ($c0 -lt 65536 -or $c0 -gt 262144) { $c0 = 262144 }
        $ctx = Ask-Step "第 5/$n 步：总上下文长度（一次对话最多能记住多少 token）" @(
            '上限 262144（256K，模型原生上限）。KVMem 作者说：约 200K 以后如果开始胡说，可以改成 184320') @(
            @{ Value = '262144'; Label = '262144（256K）　推荐，上限' },
            @{ Value = '184320'; Label = '184320（180K）　更稳一点' },
            @{ Value = '131072'; Label = '131072（128K）' }) $c0 $true 65536 262144
        Set-C "${p}_CTX" $ctx

        Set-C "${p}_ANSWER" 'auto'
        Set-C "${p}_WINDOW" 0
        Set-C 'KVMEM_SINK' 'auto'
        $mt = (Get-View $m).Mt
        Say '  开头按请求计数；检索随总上下文缩放；剩余容量给输出，最低 8K。'
    } else {
        $rec = $L
        $c0 = $v.Ctx; if ($c0 -gt $L -or $c0 -lt 16384) { $c0 = $L }
        $items = @(@{ Value = "$L"; Label = ((Fmt $L) + "　推荐，这个组合的上限（实测，按桌面占 1.3 GB 算整卡还剩约 200 MiB）") })
        foreach ($x in (@(16384, 32768, 49152, 65536, 81920, 98304) | Where-Object { $_ -lt $L } | Sort-Object -Descending)) { $items += @{ Value = "$x"; Label = (Fmt $x) } }
        $ctx = [int](Ask-Step "第 5/$n 步：总上下文长度（全部放显存）" @(
            "上限 $L（$vn）。客户端自己的系统提示可能就占几 K，建议不少于 32K") $items $c0 $true 16384 $L)
        Set-C "${p}_CTX" $ctx

        $outMax = $ctx - 8192
        $o0 = $v.Mt; if ($o0 -gt $outMax -or $o0 -lt 1024) { $o0 = [Math]::Min(32768, $outMax) }
        $items = @()
        foreach ($x in (@(8192, 16384, 32768, 49152, $outMax) | Where-Object { $_ -le $outMax } | Sort-Object -Unique)) {
            $lab = Fmt $x; if ($x -eq 32768) { $lab += '　推荐（Qwen 官方评测的上限）' }; if ($x -eq $outMax) { $lab += '　上限' }
            $items += @{ Value = "$x"; Label = $lab }
        }
        $mt = [int](Ask-Step "第 6/$n 步：单次回答上限（一次回答最多多少 token，思考也算在里面）" @(
            "上限 $outMax（总上下文减去 8K 留给输入）。回答越大，能放的对话历史越少") $items $o0 $true 1024 $outMax)
        Set-C "${p}_MAX_TOKENS" $mt
    }

    $t0 = CfgInt 'THINK_BUDGET' 0; if ($t0 -gt $mt - 256) { $t0 = 0 }
    $items = @(@{ Value = '0'; Label = "不限　　推荐。思考 + 正文 ≤ $mt，难题能想完整" })
    foreach ($x in @(4096, 8192, 16384, 24576)) { if ($x -le $mt - 4096) { $items += @{ Value = "$x"; Label = ((Fmt $x) + "　思考到这里就收尾，保证至少留 $($mt - $x) 写正文") } } }
    $th = [int](Ask-Step "第 7/$n 步：思考长度上限" @(
        '到上限时引擎会让模型把思考收尾、转去写正文（不是硬截断）。限得太紧，难题可能答错') $items $t0 $true 1024 ($mt - 256))
    Set-C 'THINK_BUDGET' $th
    return $m
}

# ---------- 模型下载（系统自带 curl.exe，断点续传；没有 curl 时用 .NET） ----------
function Download-Net([string]$url, [string]$part, [long]$total) {
    $out = $null; $resp = $null
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $have = [long]0; if (Test-Path -LiteralPath $part) { $have = (Get-Item -LiteralPath $part).Length }
        $req = [Net.HttpWebRequest]::Create($url); $req.AllowAutoRedirect = $true; $req.UserAgent = 'ninfer-oneclick/1.0'; $req.Timeout = 30000; $req.ReadWriteTimeout = 60000
        if ($have -gt 0) { $req.AddRange([long]$have) }
        $resp = $req.GetResponse()
        if ($have -gt 0 -and [int]$resp.StatusCode -ne 206) { $resp.Close(); Remove-Item -LiteralPath $part -Force; Say '  服务器不支持续传，从头下载' 'Yellow'; return }
        $in = $resp.GetResponseStream(); $out = New-Object IO.FileStream($part, [IO.FileMode]::Append, [IO.FileAccess]::Write)
        $buf = New-Object byte[] (1MB); $sw = [Diagnostics.Stopwatch]::StartNew(); $last = -5000; $got = $have
        while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
            $out.Write($buf, 0, $n); $got += $n
            if ($sw.ElapsedMilliseconds - $last -ge 2000) {
                $last = $sw.ElapsedMilliseconds; $spd = ($got - $have) / [Math]::Max(1, $sw.Elapsed.TotalSeconds)
                Write-Host ("`r  已下载 {0:N2} / {1:N2} GiB（{2:P0}），{3:N1} MB/s    " -f ($got / 1GB), ($total / 1GB), ($got / $total), ($spd / 1MB)) -NoNewline
            }
        }
        Write-Host ''
    } catch { Write-Host ''; Say "  下载出错：$($_.Exception.Message)" 'Yellow' }
    finally { if ($out) { $out.Close() }; if ($resp) { $resp.Close() } }
}

function Download-Model([string]$k, [string]$dst) {
    $m = $Models[$k]; $part = "$dst.part"; $dir = Split-Path -Parent $dst
    New-Item -ItemType Directory -Force $dir | Out-Null
    $have = [long]0; if (Test-Path -LiteralPath $part) { $have = (Get-Item -LiteralPath $part).Length }
    if ($have -gt $m.Size) { Remove-Item -LiteralPath $part -Force; $have = 0 }
    Say ''
    Say "  ── 需要下载模型：$($m.Name) ──" 'Cyan'
    Say ("    大小  ：{0} GiB（{1} 字节）{2}" -f (GiB $m.Size), $m.Size, $(if ($have -gt 0) { '，上次已经下了 ' + (GiB $have) + ' GiB，这次接着下' } else { '' }))
    Say "    来源  ：魔搭社区 $($m.Page)"
    Say "    保存到：$dst"
    Say '    中途关掉窗口也没关系，下次运行 启动.bat 会接着下；下完会自动校验文件（SHA256）。'
    Say '    也可以自己用浏览器从上面的网页下载这个文件，放进 model 文件夹（文件名不要改），再运行 启动.bat。'
    $free = [long](Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='" + $dir.Substring(0, 2) + "'")).FreeSpace
    $need = $m.Size - $have + 512MB
    if ($free -lt $need) { Fail ("硬盘空间不够：{0} 盘只剩 {1} GiB，还需要约 {2} GiB。请清理空间，或把整个文件夹挪到空间大的盘。" -f $dir.Substring(0, 1), (GiB $free), (GiB $need)) }
    $a = (Read-Answer '   [回车] 开始下载    [Q] 退出: ').ToLower()
    if ($a -eq 'q') { exit 0 }
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    $useCurl = (Test-Path -LiteralPath $curl) -and -not $env:ONECLICK_NO_CURL
    $prev = [long]-1; $stall = 0
    for ($try = 1; $try -le 30; $try++) {
        $have = [long]0; if (Test-Path -LiteralPath $part) { $have = (Get-Item -LiteralPath $part).Length }
        if ($have -ge $m.Size) { break }
        if ($have -le $prev) { $stall++ } else { $stall = 0 }; $prev = $have
        if ($stall -ge 3) { break }   # 连续几次一点都没下到（比如网址失效、被拒绝），不再白等
        if ($try -gt 1) { Say "  下载中断了，5 秒后接着下（第 $try 次）……" 'Yellow'; Start-Sleep -Seconds 5 }
        if ($useCurl) {
            $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
            & $curl -L -C - --fail --connect-timeout 30 --speed-limit 10240 --speed-time 60 -o $part $m.Url
            $ErrorActionPreference = $old
        } else { Download-Net $m.Url $part $m.Size }
    }
    $have = [long]0; if (Test-Path -LiteralPath $part) { $have = (Get-Item -LiteralPath $part).Length }
    if ($have -ne $m.Size) { Fail ("下载没完成（{0} / {1} GiB）。请检查网络后重新运行 启动.bat，会接着下。`n       一直下不了的话，用浏览器打开 {2} 下载 {3}，放进 model 文件夹。" -f (GiB $have), (GiB $m.Size), $m.Page, $m.File) }
    Say '  下载完成，正在校验文件（8 GB 大约要 1 分钟）……'
    $h = (Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash.ToLower()
    if ($h -ne $m.Sha) {
        Move-Item -LiteralPath $part "$dst.bad" -Force
        Fail ("文件校验不对（下载时出错了），已改名为 {0}.bad。请删掉它，重新运行 启动.bat 下载。" -f (Split-Path -Leaf $dst))
    }
    Move-Item -LiteralPath $part $dst -Force
    Say '  校验通过，模型已就绪。' 'Green'
}

# 模型在就直接用；不在就下载（DRYRUN 只提示）
function Ensure-Model {
    $x = Model-Path; $k = Model-Key
    if (-not $k) {
        if (-not (Test-Path -LiteralPath $x)) { Fail "找不到模型文件：$x（设置.ini 里 MODEL 写 swift 或 base 就会自动下载）" }
        return $x
    }
    if (Test-Path -LiteralPath $x) {
        if ((Get-Item -LiteralPath $x).Length -eq $Models[$k].Size) { return $x }
        Warn '模型文件大小不对（可能没下完），接着下载。'
        if (Test-Path -LiteralPath "$x.part") { Remove-Item -LiteralPath $x -Force } else { Move-Item -LiteralPath $x "$x.part" -Force }
    }
    if ($dry) { Warn "模型还没下载：$x（DRYRUN=1 不下载）"; return $x }
    Download-Model $k $x
    return $x
}

# 把向导选的值写回 设置.ini（保留说明文字；没有的项加在最后）
function Save-Ini {
    if ($Dirty.Count -eq 0) { return }
    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($l in (Get-Content -LiteralPath $ini -Encoding UTF8)) { $lines.Add($l) }
    foreach ($k in $Dirty.Keys) {
        $hit = $false
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match ('^\s*' + [regex]::Escape($k) + '\s*=')) { $lines[$i] = "$k=$($Dirty[$k])"; $hit = $true; break }
        }
        if (-not $hit) { $lines.Add("$k=$($Dirty[$k])") }
    }
    try { [IO.File]::WriteAllLines($ini, [string[]]$lines, (New-Object Text.UTF8Encoding $true)); Say '  已把这次的选择存进 设置.ini，下次回车直接用。' 'DarkGray' }
    catch { Warn "写 设置.ini 失败：$($_.Exception.Message)" }
}

# ---------- 3. 选模式 / 看配置 ----------
$modeFile = Join-Path $Root 'current-mode.txt'
$dry = (CfgInt 'DRYRUN' 0) -ne 0
if (-not $Mode) {
    $m = ([string](Cfg 'MODE' 'ask')).ToLower()
    if ($Modes -contains $m) { $Mode = $m }
}
if (-not $Mode) {
    $Mode = 'kvmem'
    if (Test-Path -LiteralPath $modeFile) {
        $x = ((Get-Content -LiteralPath $modeFile -EA SilentlyContinue | Select-Object -First 1) + '').Trim().ToLower()
        if ($Modes -contains $x) { $Mode = $x }
    }
    $keys = @{ n = 'normal'; k = 'kvmem'; r = 'rk8v4'; v = 'kvrk' }
    while ($true) {
        $view = Show-Summary $Mode
        Say ''
        if ($view.Err.Count -eq 0) { Say '   [回车] 用这套配置启动    [C] 一步步重新选    [P] 显卡功耗(降温)    [R] 重新读空闲显存    [Q] 退出' }
        else { Say '   配置有错误，请按 C 重新选（或 Q 退出）' 'Red' }
        Say '   直接换模式：[N] 普通  [K] KVMem  [R] rk8v4  [V] KVMem+rk8v4（用 设置.ini 里这个模式的数值）' 'DarkGray'
        $a = (Read-Answer '   请选择: ').ToLower()
        if ($a -eq 'q') { exit 0 }
        if ($a -eq 'c') { $Mode = Run-Wizard $Mode; continue }
        if ($a -eq 'p') { & (Join-Path $PSScriptRoot 'set-power-limit.ps1') -Interactive; continue }
        if ($keys.ContainsKey($a)) { $Mode = $keys[$a]; continue }
        if ($a -eq '' -and $view.Err.Count -eq 0) { break }
    }
    if (-not $dry) { Save-Ini }
}

# ---------- 4. 检查环境 ----------
$gpu = $null
try { $gpu = & nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader,nounits 2>$null | Select-Object -First 1 } catch {}
if (-not $gpu) { Fail '没找到 NVIDIA 显卡或显卡驱动（nvidia-smi 运行失败）。' }
$gp = $gpu -split ',\s*'
$gName = $gp[0]; $gCap = $gp[1]; $gMem = [int]$gp[2]; $gDrv = $gp[3]
$capNum = [double]::Parse($gCap, [Globalization.CultureInfo]::InvariantCulture)
if ($capNum -lt 8.6 -or $capNum -ge 9.0) { Fail "显卡是 $gName（算力 $gCap）。这个包的引擎是按算力 8.6 编译的，只能用在 RTX 30 系（8.6）和 RTX 40 系（8.9）上。" }
if ($capNum -gt 8.6) { Warn "$gName 是 RTX 40 系（算力 $gCap），引擎能跑但没实测过；SM 数会自动识别。" }
if ($gMem -lt 11500) { Warn "显卡显存只有 $gMem MiB，这个包的参数是按 12 GB 调的，可能会启动失败。请把上下文调小。" }
$drvMajor = 0; [void][int]::TryParse(($gDrv -split '\.')[0], [ref]$drvMajor)
if ($drvMajor -gt 0 -and $drvMajor -lt 580) { Warn "显卡驱动版本 $gDrv 可能太旧（需要支持 CUDA 13，建议 580 以上）。启动失败的话请先更新驱动。" }

$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
if (($Mode -eq 'kvmem' -or $Mode -eq 'kvrk') -and $ramGB -lt 24) { Warn "内存只有 $ramGB GB。KVMem 模式在长对话时要占约 10 GB 内存，建议 32 GB。" }

$running = Get-Process ninfer-serve -EA SilentlyContinue
if ($running -and -not $dry) { Fail "已经有一个 ninfer-serve 在运行（进程号 $($running[0].Id)）。请先关掉它的窗口，再启动。" }
$port = CfgInt 'PORT' 8084
$busy = $null
try { $busy = Get-NetTCPConnection -LocalPort $port -State Listen -EA SilentlyContinue } catch {}
if ($busy -and -not $dry) { Fail "端口 $port 已经被别的程序占用。请在 设置.ini 里改 PORT。" }

# 显卡、端口都没问题了，再准备模型（第一次会下载）
$model = Ensure-Model
$modelLabel = $(if (Model-Key) { $Models[(Model-Key)].Name } else { 'custom' })

# ---------- 5. 按模式组参数 ----------
Get-ChildItem env: | Where-Object { $_.Name -like 'NINFER_TERNARY_*' } | ForEach-Object { Remove-Item "env:$($_.Name)" }

function Round64([string]$name, [int]$v) {
    $r = [int]([math]::Floor($v / 64) * 64)
    if ($r -ne $v) { Warn "$name=$v 不是 64 的倍数，已改成 $r。" }
    return $r
}

$kv = 'int8'
$desc = ''
# 10-01 起：开看图一律把视觉权重放内存（看图时逐段搬上显卡），设置.ini 里的 VISION_HOST 不再起作用
$vh = ((CfgInt 'VISION' 1) -ne 0)
$lim = $ModeLimits[$Mode][(VisKey)]
switch ($Mode) {
    'normal' {
        $ctx = CfgInt 'NORMAL_CTX' 90112; $cap = $ctx; $mt = CfgInt 'NORMAL_MAX_TOKENS' 32768; $hostMib = 4096
        if ($ctx -gt $lim) { Warn "NORMAL_CTX=$ctx 超过 $lim，12 GB 显存可能放不下。" }
        $desc = "普通模式：上下文 $ctx（全在显存），单次回答最多 $mt"
    }
    'rk8v4' {
        $kv = 'rk8v4'
        $ctx = CfgInt 'RK8V4_CTX' 114688; $cap = $ctx; $mt = CfgInt 'RK8V4_MAX_TOKENS' 32768; $hostMib = 4096
        if ($ctx -gt $lim) { Warn "RK8V4_CTX=$ctx 超过 $lim，12 GB 显存可能放不下。" }
        $desc = "rk8v4 模式：上下文 $ctx（全在显存），单次回答最多 $mt"
    }
    default {
        $p = 'KVMEM'; $dA = 32768
        if ($Mode -eq 'kvrk') { $p = 'KVRK'; $kv = 'rk8v4'; $dA = 32768 }
        $ctx = CfgInt "${p}_CTX" 262144
        $view = Get-View $Mode
        if ($view.Err.Count -gt 0) { Fail ($view.Err -join '；') }
        $win = [int]$view.Win
        $ans = [int]$view.Ans
        $sink = 64
        $hostMib = CfgInt "${p}_HOST_MIB" 10240
        $cap = $win + $ans; $mt = $ans
        if ($cap -gt $lim) { Warn "${p}_WINDOW + ${p}_ANSWER = $cap，超过 $lim，12 GB 显存可能放不下。" }
        if ($ctx -le $cap) { Warn "${p}_CTX（$ctx）不比显存部分（$cap）大，KVMem 起不到作用，不如用普通模式。" }
        $env:NINFER_TERNARY_KVMEM_CPU_RETRIEVAL = $(if ($CpuRetrieval) { '1' } else { '0' })
        $env:NINFER_TERNARY_KVMEM = '1'
        $env:NINFER_TERNARY_KVMEM_WINDOW = '1'
        $env:NINFER_TERNARY_KVMEM_WINDOW_ASSEMBLY = '1'
        $env:NINFER_TERNARY_KVMEM_SCORE = '1'
        $env:NINFER_TERNARY_KVMEM_SEMANTIC = '1'
        $env:NINFER_TERNARY_KVMEM_SCORE_BUDGET = "$win"
        $env:NINFER_TERNARY_KVMEM_SCORE_SINK = "$sink"
        $env:NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL = '36'
        $env:NINFER_TERNARY_KVMEM_REPLAY_TOKENS = '48'
        $env:NINFER_TERNARY_KVMEM_NEIGHBORS = '0'
        $env:NINFER_TERNARY_HOST_KV_PAGEABLE = '1'
        $env:NINFER_TERNARY_KVMEM_NO_REBAKE = '1'
        $env:NINFER_TERNARY_KVMEM_GEN_RESERVE = '0'
        $env:NINFER_KVMEM_AUTO_ALLOCATION = '1'
        # kv8/kv9：按最后一条用户消息挑块、保护本轮新内容、重算范围 = 挑块范围（这一轮的新输入）
        $env:NINFER_TERNARY_KVMEM_SCORE_QUERY_MODE = 'msg'
        $env:NINFER_TERNARY_KVMEM_SCORE_PROTECT_NEW = '1'
        $env:NINFER_TERNARY_KVMEM_REPLAY_MODE = 'msg'
        $env:NINFER_TERNARY_KVMEM_REPLAY_MAX = "$(CfgInt 'KVMEM_REPLAY_MAX' 6144)"
        $env:NINFER_TERNARY_KVMEM_SCORE_QUERY_MAX = "$QueryMax"
        $name = if ($Mode -eq 'kvrk') { 'KVMem+rk8v4 模式' } else { 'KVMem 模式' }
        $desc = "${name}：总上下文 $ctx，驻留容量 $cap；Harness 输出申请上限 $mt，实际分配在请求时计算"
    }
}
if ($mt -gt $ctx) { Fail "单次回答上限（$mt）不能比上下文（$ctx）还大。" }
$think = CfgInt 'THINK_BUDGET' 0
if ($think -gt 0 -and $think -lt $mt) { $desc += "，思考最多 $think" } else { $think = 0 }

$exeDir = 'engine'   # 2026-10-01 起：一个程序（kv9 + 视觉放内存 + 词嵌入放内存）支持全部四个模式
$exe = Join-Path $Root "$exeDir\ninfer-serve.exe"
if (-not (Test-Path -LiteralPath $exe)) { Fail "找不到引擎程序：$exe" }
$dllCount = @(Get-ChildItem -LiteralPath (Split-Path -Parent $exe) -Filter '*.dll' -ErrorAction SilentlyContinue).Count
if ($dllCount -lt 10) { Fail "engine 文件夹里缺少运行所需的 DLL 文件（当前只有 $dllCount 个 .dll）。请把整个懒人包完整解压，保持所有 .dll 和 ninfer-serve.exe 在同一目录，不要只单独复制 ninfer-serve.exe。" }

$bindHost = [string](Cfg 'HOST' '127.0.0.1')
$modelId = [string](Cfg 'MODEL_ID' 'qwen3.8-27b')
$argv = @('--host', $bindHost, '--port', "$port", '--model-id', $modelId,
          '--kv-dtype', $kv, '--max-concurrency', '1')
$draft = CfgInt 'MTP_DRAFT' 3
if ($draft -gt 0) { $argv += @('--spec', 'mtp', '--draft-tokens', "$draft") }
if ((CfgInt 'VISION' 1) -ne 0) { $argv += @('--vision', '--vision-max-tokens', "$(CfgInt 'VISION_MAX_TOKENS' 4096)") }
if ($vh) { $env:NINFER_TERNARY_VISION_HOST = '1' }
$env:NINFER_TERNARY_EMBED_HOST = '1'   # 10-01：词嵌入放内存（省 322 MiB 显存，速度不变）
$argv += '--tolerant-tool-calls'
$argv += @('--max-context', "$ctx", '--kv-capacity', "$cap", '--default-max-tokens', "$mt", '--host-kv-mib', "$hostMib", '--prefill-chunk', '512')
if ((CfgInt 'RECOVER_INVARIANT' 1) -ne 0) { $argv += '--recover-invariant-failures' }
if ((CfgInt 'POST_THINKING' 1) -ne 0) {
    $ptTemp = [string](Cfg 'POST_THINKING_TEMP' '0.2')
    $argv += @('--post-thinking-temperature', $ptTemp)
    $ptTopP = [string](Cfg 'POST_THINKING_TOP_P' '')
    if ($ptTopP -ne '') { $argv += @('--post-thinking-top-p', $ptTopP) }
    $ptTopK = [string](Cfg 'POST_THINKING_TOP_K' '')
    if ($ptTopK -ne '') { $argv += @('--post-thinking-top-k', $ptTopK) }
    $ptMinP = [string](Cfg 'POST_THINKING_MIN_P' '')
    if ($ptMinP -ne '') { $argv += @('--post-thinking-min-p', $ptMinP) }
} else {
    $argv += @('--post-thinking-temperature', '1.0', '--post-thinking-top-p', '0.95', '--post-thinking-top-k', '20', '--post-thinking-min-p', '0.0')
}
if ($think -gt 0) { $argv += @('--default-thinking-budget', "$think") }
$key = [string](Cfg 'API_KEY' '')
if ($key -ne '') { $argv += @('--api-key', $key) }
elseif ($bindHost -ne '127.0.0.1' -and $bindHost -ne 'localhost') { Warn "HOST=$bindHost 会让局域网里的其他电脑也能访问，建议在 设置.ini 里设置 API_KEY。" }
$iniExtra = [string](Cfg 'EXTRA' '')
if ($iniExtra -ne '') { $argv += ($iniExtra -split '\s+' | Where-Object { $_ -ne '' }) }
$argv += $CliExtra

if (-not $dry) { Set-Content -LiteralPath $modeFile -Value $Mode -Encoding ASCII }

# ---------- 6. 显示并启动 ----------
$shown = ($argv | ForEach-Object { if ($_ -eq $key -and $key -ne '') { '***' } else { $_ } }) -join ' '
Say ''
Say "  $desc" 'Green'
Say "  显卡  ：$gName（$gMem MiB，驱动 $gDrv）  内存：$ramGB GB"
Say "  地址  ：http://${bindHost}:$port/v1   模型名：$modelId"
Say "  参数  ：$shown"
Say '  出现 "listening" 之后就能用了。可以运行 测试.bat 检查。关掉这个窗口就停止引擎。' 'Cyan'
Say ''

function Update-InfoLimits([int]$context, [int]$output) {
    $path = Join-Path $Root '接入信息.txt'
    if (-not (Test-Path -LiteralPath $path)) { return }
    try {
        $text = Get-Content -LiteralPath $path -Raw -Encoding UTF8
        $text = $text -replace '(?m)^(\s*上下文窗口\s*:\s*)\d+', ('${1}' + $context)
        $text = $text -replace '(?m)^(\s*最大输出\s*:\s*)\d+', ('${1}' + $output)
        $text = $text -replace '(?m)^(\s*contextWindow:\s*)\d+', ('${1}' + $context)
        $text = $text -replace '(?m)^(\s*maxTokens:\s*)\d+', ('${1}' + $output)
        [IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $true))
    } catch { Warn "更新接入信息失败：$($_.Exception.Message)" }
}

# ---------- 6b. 客户端（harness）接入信息：显示在窗口里，并写到 接入信息.txt ----------
$vis = (CfgInt 'VISION' 1) -ne 0
$locHost = $(if ($bindHost -eq '0.0.0.0' -or $bindHost -eq '::') { '127.0.0.1' } else { $bindHost })
$baseUrl = "http://${locHost}:$port/v1"
$lanUrls = @()
if ($bindHost -eq '0.0.0.0') {
    try { $lanUrls = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | ForEach-Object { "http://$($_.IPAddress):$port/v1" }) } catch {}
}
$keyTxt = $(if ($key -ne '') { '设置.ini 里 API_KEY 的值' } else { '没设密码，随便填（例如 none），但不能空着' })
$info = @(
  "ninfer 3060 一键包 客户端接入信息（每次启动自动生成，当前模式：$Mode，$(Get-Date -Format 'yyyy-MM-dd HH:mm')）",
  '换了模式或改了 设置.ini，上下文窗口和最大输出会变，请以最新的这份为准。',
  '',
  '==== 在客户端（harness）里这样填 ====',
  '  接口类型     : OpenAI 兼容（Chat Completions）。也支持 Anthropic Messages（/v1/messages）和 OpenAI Responses（/v1/responses）',
  "  Base URL     : $baseUrl"
)
foreach ($u in $lanUrls) { $info += "  局域网地址   : $u （其他电脑用这个）" }
$info += @(
  "  模型名 / ID  : $modelId",
  "  API Key      : $keyTxt",
  "  上下文窗口   : $ctx （有的客户端叫 contextWindow / context length / max context）",
  "  最大输出     : $mt （maxTokens / max_tokens，含思考；不要填得比这个大）",
  "  图片输入     : $(if ($vis) { '支持（input 里加 image）' } else { '不支持（设置.ini 里 VISION=0）' })",
  '  工具调用     : 支持（function calling / tools）',
  '  思考内容     : 放在 reasoning_content 字段（DeepSeek 格式）；请求里加 "enable_thinking": false 可以关思考',
  '  思考档位     : 只有 off / low / medium / xhigh 四档（reasoning_effort），none/minimal/high 等别名会映射到这些档位',
  '  同时请求数   : 1（一次只处理一个请求，多开对话会排队）',
  '',
  '==== DeepSeek Harness（dsh）可以直接用的配置 ====',
  '1) 把下面这段存成 %USERPROFILE%\.dsh\profiles\<你用的 profile>\cordis.patch.yml',
  '   （已经有这个文件的话，把 providers 下面 ninfer-local 这一块合并进去）；',
  '   或者在 dsh 的 设置 → 模型 → 自定义模型 API 里按上面的值填。',
  '2) 在 %USERPROFILE%\.dsh\.env 里加一行：NINFER_LOCAL_API_KEY=none（设了 API_KEY 就填那个值）',
  '',
  '- id: llm-pi-ai',
  '  config:',
  '    providers:',
  '      ninfer-local:',
  '        displayName: ninfer local (RTX 3060)',
  '        apiKeyEnv: NINFER_LOCAL_API_KEY',
  '        api: openai-completions',
  "        baseURL: $baseUrl",
  '        models:',
  "          - id: $modelId",
  "            name: $modelLabel 27B (local, $Mode)",
  "            contextWindow: $ctx",
  "            maxTokens: $mt",
  '            input:',
  '              - text'
)
if ($vis) { $info += '              - image' }
$info += @(
  '            reasoningEfforts:',
  '              "off": none',
  '              low: low',
  '              medium: medium',
  '              xhigh: xhigh',
  '            compat:',
  '              thinkingFormat: deepseek',
  '- id: agent-default-model',
  '  config:',
  '    provider: ninfer-local',
  "    model: $modelId"
)
Say '  ---- 客户端（harness）里这样填（完整说明和 dsh 配置见 接入信息.txt）----' 'Cyan'
Say "  Base URL：$baseUrl    模型名：$modelId    API Key：$keyTxt"
foreach ($u in $lanUrls) { Say "  局域网地址：$u" }
Say "  上下文窗口：$ctx    最大输出：$mt    图片：$(if ($vis) { '支持' } else { '不支持' })    工具调用：支持"
Say ''
if (-not $dry) {
    try { [IO.File]::WriteAllLines((Join-Path $Root '接入信息.txt'), [string[]]$info, (New-Object Text.UTF8Encoding $true)) }
    catch { Warn "写 接入信息.txt 失败：$($_.Exception.Message)" }
}

if ($dry) {
    Get-ChildItem env: | Where-Object { $_.Name -like 'NINFER_TERNARY_*' } | Sort-Object Name | ForEach-Object { Say "  环境变量：$($_.Name)=$($_.Value)" 'DarkGray' }
    Say '  （DRYRUN=1：只显示参数，没有启动引擎）' 'Yellow'; exit 0
}
$ErrorActionPreference = 'Continue'
$env:PATH = (Split-Path -Parent $exe) + ';' + $env:PATH
$retry = [Math]::Max(0, [Math]::Min(3, (CfgInt 'RETRY' 2)))
$fit = 0; $rc = 1
for ($try = 1; $try -le (1 + $retry); $try++) {
    $short = $null
    $attempt = @{ Ready = $false }
    $clock = [Diagnostics.Stopwatch]::StartNew()
    & $exe $model @argv 2>&1 | ForEach-Object {
        $line = "$_"; Write-Host $line
        if ($line -match '\blistening on https?://') { $attempt.Ready = $true }
        if ($line -match 'reservation requires (\d+) bytes, but only (\d+) bytes are available') { $script:short = @([double]$Matches[1], [double]$Matches[2]) }
    }
    $rc = $LASTEXITCODE
    if ($rc -eq 0) { break }
    if (-not $attempt.Ready -and $Mode -in @('kvmem','kvrk') -and $short -and $fit -lt 3) {
        $defMiB = ($short[0] - $short[1]) / 1MB
        $cut = [int][Math]::Ceiling(($defMiB + $KeepMiB + $StashMiB) / $TokMiB[$kv] / 1024) * 1024
        $newCap = $cap - [Math]::Max(1024, $cut)
        $nextAns = Harness-Out $newCap $ctx
        $nextWin = $newCap - $nextAns
        $fit++
        if ($newCap -lt 8384) { Say '显存不足，检索窗口无法继续缩小。请调小回答上限或关闭占显存的程序。' 'Red'; break }
        Say ("显存比预计少，检索窗口 {0} → {1}，回答上限 {2} → {3}，自动重试。" -f $win, $nextWin, $ans, $nextAns) 'Yellow'
        $win = $nextWin; $ans = $nextAns; $mt = $nextAns; $cap = $win + $ans
        $env:NINFER_TERNARY_KVMEM_SCORE_BUDGET = "$win"
        $env:NINFER_TERNARY_KVMEM_GEN_RESERVE = '0'
        $env:NINFER_KVMEM_AUTO_ALLOCATION = '1'
        if ($think -ge $mt) { $think = [Math]::Max(0, $mt - 1) }
        for ($i = 0; $i -lt $argv.Count - 1; $i++) {
            if ($argv[$i] -eq '--kv-capacity') { $argv[$i + 1] = "$cap" }
            if ($argv[$i] -eq '--default-max-tokens') { $argv[$i + 1] = "$mt" }
            if ($argv[$i] -eq '--default-thinking-budget') { $argv[$i + 1] = "$think" }
        }
        Update-InfoLimits $ctx $mt
        $try--; continue
    }
    Say "[引擎退出] 代码 $rc，第 $try 次，模式 $Mode" 'Yellow'
    if (-not $attempt.Ready) { Say '启动失败，请按上面的具体错误调整配置。' 'Yellow'; break }
    if ($try -le $retry) { Say '15 秒后自动重启……'; Start-Sleep -Seconds 15 }
}
exit $rc
