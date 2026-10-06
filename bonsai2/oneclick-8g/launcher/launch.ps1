# ninfer 3060 8GB 一键启动器（由 启动.bat 调用）
# 用法：启动.bat [模式] [键=值 ...] [其他 ninfer-serve 参数 ...]
#   不写模式（设置.ini 里 MODE=ask）：先显示当前配置，回车直接启动，按 C 一步步重新选，按 N/K/R/V 直接换模式
#   模式：normal / kvmem / rk4 / kvrk4 ；写了就不显示菜单，直接启动
#   键=值：临时覆盖 设置.ini 里的同名项，例如 PORT=8085 KVMEM_ANSWER=8192（不会写回 设置.ini）
#   其他参数：原样加在最后，例如 --greedy（同一个参数写两次时，以最后一次为准）
# 向导里选的值，在按回车启动时写回 设置.ini（DRYRUN=1 时不写）
# 模型不在包里：第一次用时（配置完、按回车后）自动从魔搭社区下载 Swift-Bonsai-2 ptq1，支持断点续传，下完校验 SHA256，
#   再用 patch 文件夹里的补丁把 MTP 部分转成 Q4（8 GB 显存要靠这一步省出约 225 MB），转完再校验一次

$ErrorActionPreference = 'Stop'
$launchArguments = @($args)
. (Join-Path $PSScriptRoot 'runtime-options.ps1')
. (Join-Path $PSScriptRoot 'capacity-options.ps1')

$env:NINFER_VISION_CPU = '1' # Host option means CPU encoding; no GPU fallback
$CpuRetrieval = ($env:NINFER_TERNARY_KVMEM_CPU_RETRIEVAL -ne '0')
$Root = Split-Path -Parent $PSScriptRoot
try { $Host.UI.RawUI.WindowTitle = 'ninfer 3060 8G 引擎' } catch {}

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
$Modes = @('normal', 'kvmem', 'rk4', 'kvrk4')
$Mode = $null
$CliExtra = @()
foreach ($a in $launchArguments) {
    $s = [string]$a
    if (-not $Mode -and ($Modes -contains $s.ToLower())) { $Mode = $s.ToLower(); continue }
    if ($s -match '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') { $C[$Matches[1].ToUpper()] = $Matches[2]; continue }
    $CliExtra += $s
}

Initialize-NinferRuntimeOptions $script:C

function Cfg([string]$k, $d) { if ($C.ContainsKey($k) -and $C[$k] -ne '') { return $C[$k] } else { return $d } }
function CfgInt([string]$k, [int]$d) {
    $v = Cfg $k $d
    $n = 0
    if (-not [int]::TryParse([string]$v, [ref]$n)) { Fail "设置项 $k 必须是整数，现在是：$v" }
    return $n
}

# ---------- 模型 ----------
# 引擎用的是 Q4 版（Model）；它由魔搭社区上的 ptq1 原始文件（Src）加 patch 文件夹里的补丁生成
$Q4 = @{ Name = 'Swift-Bonsai-2 ptq1（MTP 转成 Q4，8G 专用）'; File = 'Swift-Bonsai-2-ptq1-mtpq4.ninfer'; Size = [long]6821799948
            Sha = 'f716c651dffd4f38398d23084917e406200eee469ddf4160d0cde70b400f044d' }
$Orig = @{ Name = 'Swift-Bonsai-2 ptq1'; File = 'bonsai2_27b_swift_ptq1.ninfer'; Size = [long]7047407628
          Sha = 'cc9e890728ea7357b1d8a0797a4accdc6ca143031471a63e49c9cf6c8314b6ae'
          Page = 'https://www.modelscope.cn/models/fyb423/Swift-Bonsai-2-27B-NInfer'
          Url = 'https://www.modelscope.cn/models/fyb423/Swift-Bonsai-2-27B-NInfer/resolve/master/bonsai2_27b_swift_ptq1.ninfer' }
$PatchFiles = @{ 'ops.txt' = [long]5449; 'lit.bin' = [long]225797927 }
# 测试用：用一个小文件代替原始文件走一遍下载流程
if ($env:ONECLICK_TEST_URL) { $Orig.Url = $env:ONECLICK_TEST_URL; $Orig.Size = [long]$env:ONECLICK_TEST_SIZE; $Orig.Sha = $env:ONECLICK_TEST_SHA; $Orig.File = $env:ONECLICK_TEST_FILE }
function Model-Custom { $v = ([string](Cfg 'MODEL' 'swift')).Trim(); return ($v.ToLower() -ne 'swift') }   # 设置.ini 里写了自定义路径
function Model-Path {
    if (-not (Model-Custom)) { return (Join-Path $Root ('model\' + $Q4.File)) }
    $x = [string](Cfg 'MODEL' ''); if (-not [System.IO.Path]::IsPathRooted($x)) { $x = Join-Path $Root $x }; return $x
}
function Src-Path { return (Join-Path $Root ('model\' + $Orig.File)) }
function Model-State {
    $x = Model-Path
    if (Model-Custom) { if (Test-Path -LiteralPath $x) { return 'ok' } else { return 'missing' } }
    if ((Test-Path -LiteralPath $x) -and (Get-Item -LiteralPath $x).Length -eq $Q4.Size) { return 'ok' }
    $s = Src-Path
    if ((Test-Path -LiteralPath $s) -and (Get-Item -LiteralPath $s).Length -eq $Orig.Size) { return 'src' }
    if ((Test-Path -LiteralPath "$s.part") -or (Test-Path -LiteralPath $s)) { return 'part' }
    return 'missing'
}
function GiB([long]$b) { return ('{0:N1}' -f ($b / 1GB)) }

# ---------- 上限表（RTX 3060 8GB：默认档以约 7.5 GiB 历史预算设置；核显可选档会超过它，这不是硬性限额；2026-10-01 实测，在 12G 卡上量引擎占用） ----------
#   普通 / rk4 模式是总上下文的上限；KVMem 两种模式是 检索窗口 + 单次回答 的上限
#   KVMem 容量按真实空闲显存计算；recommended使用测试推荐档；auto按当前显存估算；数字是用户手动选择。
$Lim = @{ normal = 57344; kvmem = $(if ($CpuRetrieval) { 57344 } else { 45056 }); rk4 = 86016; kvrk4 = $(if ($CpuRetrieval) { 86016 } else { 65536 }) }
$LimIgpu = $(if ($CpuRetrieval) { 61440 } else { 49152 })
$WinMin = 24576                      # KVMem 检索窗口最少留 24576
$Pre = @{ normal = 'NORMAL'; kvmem = 'KVMEM'; rk4 = 'RK4'; kvrk4 = 'KVRK4' }
$ModeName = @{ normal = '普通（全部放显存，rk8v4）'; kvmem = 'KVMem（历史放内存，rk8v4）'
               rk4 = '普通 rk4v4（全部放显存，KV 压得更狠；不推荐）'; kvrk4 = 'KVMem + rk4v4（推荐：128K自动前缀档，关闭Graph）' }
$Dirty = [ordered]@{}                # 向导改过、启动时要写回 设置.ini 的项

function Fmt([int]$n) { if ($n -gt 0 -and $n % 1024 -eq 0) { return ('{0}（{1}K）' -f $n, ($n / 1024)) } else { return "$n" } }
function Set-C([string]$k, $v) { $C[$k] = "$v"; $Dirty[$k] = "$v" }
function Igpu { return ([string](Cfg 'KVMEM_VRAM' 'std')).ToLower() -eq 'igpu' }
function Mode-Lim([string]$m) {
    $fallback = $(if ($m -eq 'kvmem' -and (Igpu)) { $LimIgpu } else { $Lim[$m] })
    if ($m -eq 'kvmem' -or $m -eq 'kvrk4') {
        return Physical-8G-Cap $(if ($m -eq 'kvrk4') { 'rk4v4' } else { 'rk8v4' }) (CfgInt ($Pre[$m] + '_CTX') 262144) $fallback
    }
    return $fallback
}

function Free-8G-MiB {
    if ($env:ONECLICK_FREE_MIB) { return [double]$env:ONECLICK_FREE_MIB }
    try {
        $a = (& nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1) -split ','
        $total = [double]$a[0].Trim(); $free = [double]$a[1].Trim()
        # 在 12GB 卡上运行 8G 方案也按 8GB 的总预算；真实 8GB 卡就是当前空闲显存。
        return [Math]::Max(0, [Math]::Min($total, 8192) - ($total - $free))
    } catch { return -1 }
}
function Physical-8G-Cap([string]$kv,[int]$ctx,[int]$fallback){$i=Get-CapacityInfo '8g' $kv ([string](Cfg 'CUDA_GRAPH' '0'));return [Math]::Min($ctx,$i.Recommended)}

function Auto-8G-Answer([int]$cap, [int]$outMax) { return $cap } # API envelope, not a physical partition

# 按当前设置算出某个模式的各项数值（不报错、不改东西）
function Get-View([string]$m) {
    $p = $Pre[$m]
    $r = @{ Mode = $m; On = ($m -eq 'kvmem' -or $m -eq 'kvrk4'); Kv = $(if ($m -eq 'rk4' -or $m -eq 'kvrk4') { 'rk4v4' } else { 'rk8v4' })
            Lim = (Mode-Lim $m); Think = (CfgInt 'THINK_BUDGET' 0); Err = @(); Warn = @() }
    if ($r.On) {
        $r.Ctx = CfgInt "${p}_CTX" 262144
        $answer = ([string](Cfg "${p}_ANSWER" 'auto')).ToLower()
        $r.AutoOut = ($answer -eq 'auto')
        $info=Get-CapacityInfo '8g' $r.Kv ([string](Cfg 'CUDA_GRAPH' '1'))
        try{$r.Cap = Resolve-Capacity ([string](Cfg "${p}_RESIDENT" 'recommended')) $info $r.Ctx}catch{$r.Cap=0;$r.Err+=$_.Exception.Message}
        $r.CapacityInfo=$info
        if($info.Estimated -ge 0 -and $r.Cap -gt $info.Estimated){$r.Warn+='高于当前显存估算，可能OOM；用户自选值保留。'}
        $r.Mt = $(if ($r.AutoOut) { $r.Cap } else { CfgInt "${p}_ANSWER" 16384 })
        if ($r.Mt -lt 1 -or $r.Mt -gt $r.Ctx) { $r.Err += 'API output limit must be positive and no larger than logical context' }
        $r.Win = $r.Cap; $r.Ans = 0; $r.WinTop = $r.Cap; $r.CapAuto = $r.Cap
        if ($r.Cap -lt [int]($r.Ctx / 8) + 8192 + 256) { $r.Warn += '低于检索推荐预算；按实际前缀缩减检索，不禁止这个选择' }
    } else {
        $r.Ctx = CfgInt "${p}_CTX" $(if ($m -eq 'rk4') { 86016 } else { 57344 })
        $r.Mt = CfgInt "${p}_MAX_TOKENS" 32768
        $info=Get-CapacityInfo '8g' $r.Kv ([string](Cfg 'CUDA_GRAPH' '1'))
        try{Confirm-ResidentBound $info $r.Ctx}catch{$r.Err+=$_.Exception.Message}
        $r.Cap = $r.Ctx
    }
    if ($r.Mt -gt $r.Ctx) { $r.Err += "单次回答（$($r.Mt)）不能比总上下文（$($r.Ctx)）还大" }
    if ($r.Cap -gt $r.Lim) { $r.Warn += "显存部分 $($r.Cap) 超过这个模式的历史参考值（非硬上限） $($r.Lim)，8 GB 显存可能放不下" }
    if ($r.Think -gt 0 -and $r.Think -ge $r.Mt) { $r.Warn += "思考上限（$($r.Think)）不比单次回答（$($r.Mt)）小，等于不限" }
    return $r
}

function Show-Summary([string]$m) {
    $cv=Get-View $m
    if($cv.On){Show-CapacityInfo $cv.CapacityInfo;Say ('  驻留选择='+[string](Cfg ($Pre[$m]+'_RESIDENT') 'recommended')+'；实际驻留='+$cv.Cap)}
    Show-NinferRuntimeOptions $script:C
    $v = Get-View $m
    $letter = @{ normal = 'N'; kvmem = 'K'; rk4 = 'R'; kvrk4 = 'V' }[$m]
    Say ''
    Say '  ┌──────────── 当前配置（RTX 3060 8GB）────────────'
    $ms = Model-State
    if (Model-Custom) { Say ('  │ 模型      : 自定义 ' + (Model-Path) + $(if ($ms -eq 'missing') { '（文件不存在）' } else { '' })) }
    else {
        $st = @{ ok = '已就绪'; src = '原始文件已下载，按回车后转换（约 1–2 分钟）'
                 part = '没下完，按回车后接着下载'; missing = ('还没下载，按回车后自动下载 ' + (GiB $Orig.Size) + ' GiB 并转换') }[$ms]
        Say ('  │ 模型      : Swift-Bonsai-2 ptq1 + Q4 MTP（' + $st + '）')
    }
    Say ("  │ 模式      : [{0}] {1}" -f $letter, $ModeName[$m])
    if ($m -eq 'kvmem') { Say ('  │ 显存档    : ' + $(if (Igpu) { '核显档（仍按当前空闲显存计算）' } else { '自动（按当前空闲显存计算）' })) }
    Say ('  │ 看图      : ' + $(if ((CfgInt 'VISION' 1) -ne 0) { '开（CPU 视觉编码，线程见运行选项）' } else { '关' }))
    Say ('  │ 总上下文  : ' + (Fmt $v.Ctx))
    Say ('  │ 单次回答  : ' + (Fmt $v.Mt) + '   （思考 + 正文 合计）')
    if ($v.Think -gt 0) { Say ('  │ 思考上限  : ' + (Fmt $v.Think)) } else { Say '  │ 思考上限  : 不限（只受单次回答限制）' }
    $pct = [int](100 * $v.Cap / $v.Lim)
    if ($v.On) { Say ('  │ 驻留预算  : {0}；完整前缀/检索/输出逐请求动态分配' -f $v.Cap) }
    else       { Say ('  │ 显存      : {0} / 上限 {1}（{2}%）' -f $v.Cap, $v.Lim, $pct) }
    Say ('  │ 地址      : http://{0}:{1}/v1   模型名：{2}' -f (Cfg 'HOST' '127.0.0.1'), (Cfg 'PORT' '8084'), (Cfg 'MODEL_ID' 'qwen3.8-27b'))
    Say '  └──────────────────────────────────'
    foreach ($w in $v.Warn) { Say "  [提醒] $w" 'Yellow' }
    foreach ($x in $v.Err)  { Say "  [错误] $x" 'Red' }
    return $v
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

function Run-Wizard([string]$m){
 # 8G固定本包量化模型。
 $m=Ask-CapacityChoice '模式/KV：normal=普通；kvmem=KVMem；kvrk4=rk4v4 KVMem / rk4=普通' $m @('normal','kvmem','rk4','kvrk4')
 $p=$Pre[$m]
 Edit-NinferRuntimeOptions $script:C -AskVision
 foreach($rk in @('CUDA_GRAPH','VISION_DEVICE','CPU_THREADS','VISION')){Set-C $rk $script:C[$rk]}
 $v=Get-View $m
 $info=Get-CapacityInfo '8g' $v.Kv ([string](Cfg 'CUDA_GRAPH' '0'))
 if($v.On){
  $n=Edit-CapacityBudget $script:C "${p}_RESIDENT" $info
  Set-C "${p}_RESIDENT" $script:C["${p}_RESIDENT"]
  $ctx=Ask-CapacityValue '逻辑上下文：64k/128k/200k/256k；小驻留建议128k' ([string](Cfg "${p}_CTX" '131072')) ([Math]::Max(16384,$n)) 262144 @() 64
  Set-C "${p}_CTX" $ctx
  $ans=Ask-CapacityValue 'API回答上限：推荐auto；8k/32k/64k（不改变物理分区）' ([string](Cfg "${p}_ANSWER" 'auto')) 1 ([int]$ctx) @('auto')
  Set-C "${p}_ANSWER" $ans;$mt=$(if($ans -eq 'auto'){$n}else{[int]$ans})
 }else{
  if($info.Max -lt 16384){throw '当前显存不足普通模式最小窗口，请先调整运行选项或释放显存。'}
  Show-CapacityInfo $info
  $ctx=Ask-CapacityValue '普通模式全部驻留：历史推荐非硬上限；超过显存估算可能OOM' ([string](Cfg "${p}_CTX" '65536')) 16384 $info.Max @() 64
  Set-C "${p}_CTX" $ctx
  $mt=Ask-CapacityValue 'API回答上限（建议32k）' ([string](Cfg "${p}_MAX_TOKENS" '32768')) 1 ([int]$ctx)
  Set-C "${p}_MAX_TOKENS" $mt
 }
 $th=Ask-CapacityValue '思考上限：推荐0=不限' ([string](Cfg 'THINK_BUDGET' '0')) 0 ([int]$mt-1)
 Set-C 'THINK_BUDGET' $th;return $m
}

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

function Download-Model($m, [string]$dst) {
    $part = "$dst.part"; $dir = Split-Path -Parent $dst
    New-Item -ItemType Directory -Force $dir | Out-Null
    $have = [long]0; if (Test-Path -LiteralPath $part) { $have = (Get-Item -LiteralPath $part).Length }
    if ($have -gt $m.Size) { Remove-Item -LiteralPath $part -Force; $have = 0 }
    Say ''
    Say "  ── 需要下载模型：$($m.Name) ──" 'Cyan'
    Say ("    大小  ：{0} GiB（{1} 字节）{2}" -f (GiB $m.Size), $m.Size, $(if ($have -gt 0) { '，上次已经下了 ' + (GiB $have) + ' GiB，这次接着下' } else { '' }))
    Say "    来源  ：魔搭社区 $($m.Page)（文件 $($m.File)）"
    Say "    保存到：$dst"
    Say '    中途关掉窗口也没关系，下次运行 启动.bat 会接着下；下完会自动校验文件（SHA256），'
    Say ('    然后自动转换成 8G 用的版本（约 1–2 分钟，转完原始文件会删掉）。转换时要多占约 {0} GiB 硬盘。' -f (GiB $Q4.Size))
    Say '    也可以自己用浏览器从上面的网页下载这个文件，放进 model 文件夹（文件名不要改），再运行 启动.bat。'
    $free = [long](Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='" + $dir.Substring(0, 2) + "'")).FreeSpace
    $need = $m.Size - $have + $Q4.Size + 512MB
    if ($free -lt $need) { Fail ("硬盘空间不够：{0} 盘只剩 {1} GiB，下载加转换还需要约 {2} GiB（转完会删掉原始文件，最后只占约 {3} GiB）。请清理空间，或把整个文件夹挪到空间大的盘。" -f $dir.Substring(0, 1), (GiB $free), (GiB $need), (GiB $Q4.Size)) }
    $a = (Read-Answer '   [回车] 开始下载    [Q] 退出: ').ToLower()
    if ($a -eq 'q') { exit 0 }
        if ($a -eq 'f') { continue }
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
    Say '  下载完成，正在校验文件（7 GB 大约要 1 分钟）……'
    $h = (Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash.ToLower()
    if ($h -ne $m.Sha) {
        Move-Item -LiteralPath $part "$dst.bad" -Force
        Fail ("文件校验不对（下载时出错了），已改名为 {0}.bad。请删掉它，重新运行 启动.bat 下载。" -f (Split-Path -Leaf $dst))
    }
    Move-Item -LiteralPath $part $dst -Force
    Say '  校验通过。' 'Green'
}

# 用补丁把 ptq1 原始文件转成 Q4 MTP 版：ops.txt 每行 "C 偏移 长度"（从原始文件复制）或 "L 偏移 长度"（从 lit.bin 复制）
function Apply-ModelPatch([string]$src, [string]$opsFile, [string]$litFile, [string]$dst) {
    $ops = @(Get-Content -LiteralPath $opsFile | Where-Object { $_ -match '^[CL] \d+ \d+$' })
    $total = [long]0; foreach ($l in $ops) { $total += [long]($l.Split(' ')[2]) }
    $buf = New-Object byte[] (8MB)
    $fs = [IO.File]::OpenRead($src); $fl = [IO.File]::OpenRead($litFile)
    $fo = New-Object IO.FileStream($dst, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None, 1MB)
    try {
        $fo.SetLength($total)
        $done = [long]0; $sw = [Diagnostics.Stopwatch]::StartNew(); $last = -5000
        foreach ($l in $ops) {
            $p = $l.Split(' '); $off = [long]$p[1]; $left = [long]$p[2]
            $in = $(if ($p[0] -eq 'C') { $fs } else { $fl })
            [void]$in.Seek($off, [IO.SeekOrigin]::Begin)
            while ($left -gt 0) {
                $n = $in.Read($buf, 0, [int][Math]::Min([long]$buf.Length, $left))
                if ($n -le 0) { throw "读文件提前结束（$($p[0]) $off）" }
                $fo.Write($buf, 0, $n); $left -= $n; $done += $n
                if ($sw.ElapsedMilliseconds - $last -ge 2000) {
                    $last = $sw.ElapsedMilliseconds
                    Write-Host ("`r  已转换 {0:N2} / {1:N2} GiB（{2:P0}）    " -f ($done / 1GB), ($total / 1GB), ($done / $total)) -NoNewline
                }
            }
        }
        Write-Host ("`r  已转换 {0:N2} / {1:N2} GiB（100%）    " -f ($done / 1GB), ($total / 1GB))
    } finally { $fo.Close(); $fs.Close(); $fl.Close() }
}

function Convert-Model([string]$src, [string]$dst) {
    $pd = Join-Path $Root 'patch'
    foreach ($f in $PatchFiles.Keys) {
        $x = Join-Path $pd $f
        if (-not (Test-Path -LiteralPath $x) -or (Get-Item -LiteralPath $x).Length -ne $PatchFiles[$f]) { Fail "补丁文件缺失或不完整：$x。请重新解压一键包。" }
    }
    $dir = Split-Path -Parent $dst
    $free = [long](Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='" + $dir.Substring(0, 2) + "'")).FreeSpace
    if ($free -lt $Q4.Size + 256MB) { Fail ("硬盘空间不够：转换需要约 {0} GiB，{1} 盘只剩 {2} GiB。" -f (GiB ($Q4.Size + 256MB)), $dir.Substring(0, 1), (GiB $free)) }
    Say ''
    Say '  ── 把模型转换成 8G 用的版本（MTP 部分转成 Q4，省约 225 MB 显存）──' 'Cyan'
    $part = "$dst.part"
    try { Apply-ModelPatch $src (Join-Path $pd 'ops.txt') (Join-Path $pd 'lit.bin') $part }
    catch { Remove-Item -LiteralPath $part -Force -EA SilentlyContinue; Fail ("转换失败：{0}。`n       如果反复出现，请删掉 model 文件夹里的 {1}，重新运行 启动.bat 下载。" -f $_.Exception.Message, (Split-Path -Leaf $src)) }
    Say '  转换完成，正在校验（约 1 分钟）……'
    $h = (Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash.ToLower()
    if ($h -ne $Q4.Sha) {
        Remove-Item -LiteralPath $part -Force
        Move-Item -LiteralPath $src "$src.bad" -Force
        Fail ("转换后的文件校验不对，可能是原始文件坏了，已把它改名为 {0}.bad。请删掉它，重新运行 启动.bat 下载。" -f (Split-Path -Leaf $src))
    }
    Move-Item -LiteralPath $part $dst -Force
    Remove-Item -LiteralPath $src -Force
    Say ('  校验通过，模型已就绪。原始文件 {0} 已删除（省 {1} GiB）。' -f (Split-Path -Leaf $src), (GiB $Orig.Size)) 'Green'
}

# 模型在就直接用；不在就下载 + 转换（DRYRUN 只提示）
function Ensure-Model {
    $x = Model-Path
    if (Model-Custom) {
        if (-not (Test-Path -LiteralPath $x)) { Fail "找不到模型文件：$x（设置.ini 里 MODEL 写 swift 就会自动下载）" }
        return $x
    }
    if (Test-Path -LiteralPath $x) {
        if ((Get-Item -LiteralPath $x).Length -eq $Q4.Size) { return $x }
        Warn '8G 版模型文件大小不对，重新转换。'
        Remove-Item -LiteralPath $x -Force
    }
    $s = Src-Path
    if ($dry) { Warn "模型还没准备好：$x（DRYRUN=1 不下载、不转换）"; return $x }
    if (-not ((Test-Path -LiteralPath $s) -and (Get-Item -LiteralPath $s).Length -eq $Orig.Size)) {
        if (Test-Path -LiteralPath $s) {
            Warn '原始文件大小不对（可能没下完），接着下载。'
            if (Test-Path -LiteralPath "$s.part") { Remove-Item -LiteralPath $s -Force } else { Move-Item -LiteralPath $s "$s.part" -Force }
        }
        Download-Model $Orig $s
    }
    Convert-Model $s $x
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
    $keys = @{ n = 'normal'; k = 'kvmem'; r = 'rk4'; v = 'kvrk4' }
    while ($true) {
        $view = Show-Summary $Mode
        Say ''
        if ($view.Err.Count -eq 0) { Say '   [回车] 用这套配置启动    [C] 一步步重新选    [Q] 退出' }
        else { Say '   配置有错误，请按 C 重新选（或 Q 退出）' 'Red' }
        Say '   直接换模式：[N] 普通  [K] KVMem  [R] 普通 rk4v4  [V] KVMem+rk4v4（用 设置.ini 里这个模式的数值）' 'DarkGray'
        $a = (Read-Answer '   请选择: ').ToLower()
        if ($a -eq 'q') { exit 0 }
        if ($a -eq 'f') { continue }
        if ($a -eq 'c') { try{$Mode=Run-Wizard $Mode}catch{Write-Host $_.Exception.Message -ForegroundColor Yellow};continue }
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
if ($gCap -ne '8.6') { Fail "显卡是 $gName（算力 $gCap）。这个包只能用在 RTX 30 系（算力 8.6）的显卡上。" }
if ($gMem -lt 7900) { Warn "显卡显存只有 $gMem MiB，这个包是按 8 GB 调的，可能会启动失败。" }
if ($gMem -ge 11500) { Say "  提示：你的显卡有 $gMem MiB 显存，用 12G 版一键包能开更大的上下文、生成更快（8G 版能用，只是没必要）。" 'DarkGray' }
$drvMajor = 0; [void][int]::TryParse(($gDrv -split '\.')[0], [ref]$drvMajor)
if ($drvMajor -gt 0 -and $drvMajor -lt 580) { Warn "显卡驱动版本 $gDrv 可能太旧（需要支持 CUDA 13，建议 580 以上）。启动失败的话请先更新驱动。" }

$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
if (($Mode -eq 'kvmem' -or $Mode -eq 'kvrk4') -and $ramGB -lt 24) { Warn "内存只有 $ramGB GB。KVMem 模式在长对话时要占约 10 GB 内存，建议 32 GB。" }

$running = Get-Process ninfer-serve, ninfer-serve-rk4 -EA SilentlyContinue
if ($running -and -not $dry) { Fail "已经有一个 ninfer-serve 引擎在运行（进程号 $($running[0].Id)）。请先关掉它的窗口，再启动。" }
$port = CfgInt 'PORT' 8084
$busy = $null
try { $busy = Get-NetTCPConnection -LocalPort $port -State Listen -EA SilentlyContinue } catch {}
if ($busy -and -not $dry) { Fail "端口 $port 已经被别的程序占用。请在 设置.ini 里改 PORT。" }

# 显卡、端口都没问题了，再准备模型（第一次会下载 + 转换）
$model = Ensure-Model
$modelLabel = $(if (Model-Custom) { 'custom' } else { 'Swift-Bonsai-2 ptq1' })

# ---------- 5a. 组引擎命令行的函数（显存不够自动缩窗时会重新调用）----------
function New-Argv([int]$c, [int]$k, [int]$m, [int]$h) {
    $a = @('--host', $bindHost, '--port', "$port", '--model-id', $modelId, '--kv-dtype', $kv, '--max-concurrency', '1')
    if ($draft -gt 0) { $a += @('--spec', 'mtp', '--draft-tokens', "$draft") }
    if ($vis) { $a += @('--vision', '--vision-max-tokens', "$visionMax") }
    $a += '--tolerant-tool-calls'
    $a += '--recover-invariant-failures'
    if ($postThink -gt 0) { $a += @('--post-thinking-temperature', "$postThink") }
    $a += @('--max-context', "$c", '--kv-capacity', "$k", '--default-max-tokens', "$m", '--host-kv-mib', "$h")
    $a += @('--device-state-slots', '0', '--prefill-chunk', '512')
    if ($think -gt 0) { $a += @('--default-thinking-budget', "$think") }
    if ($key -ne '') { $a += @('--api-key', $key) }
    if ($iniExtra -ne '') { $a += ($iniExtra -split '\s+' | Where-Object { $_ -ne '' }) }
    $a += $CliExtra
    return (Complete-NinferRuntimeArgs -Config $script:C -Arguments $a)
}

# 从引擎日志里读“显存不够”的差额：reservation requires N bytes, but only M bytes are available
function New-LogRedaction([object[]]$Arguments, [string[]]$AdditionalSecrets = @()) {
    $safe = New-Object 'System.Collections.Generic.List[string]'
    $secrets = New-Object 'System.Collections.Generic.List[string]'
    foreach ($secret in $AdditionalSecrets) { if (-not [string]::IsNullOrEmpty($secret)) { $secrets.Add($secret) } }
    $hideNext = $false
    foreach ($item in $Arguments) {
        $value = [string]$item
        if ($hideNext) {
            if ($value.Length -gt 0) { $secrets.Add($value) }
            $safe.Add('***'); $hideNext = $false; continue
        }
        if ($value -match '^(--(?:api[-_]key|access[-_]token|auth[-_]token|authorization|password|secret))(?:=(.*))?$') {
            $flag = $Matches[1]
            if ($value.Contains('=')) {
                if ($Matches[2].Length -gt 0) { $secrets.Add($Matches[2]) }
                $safe.Add($flag + '=***')
            } else { $safe.Add($flag); $hideNext = $true }
        } else { $safe.Add($value) }
    }
    $secretValues = @($secrets | Sort-Object Length -Descending | Select-Object -Unique)
    $text = $safe -join ' '
    foreach ($secret in $secretValues) { $text = $text.Replace($secret, '***') }
    return @{ Text = $text; Secrets = $secretValues }
}
function Protect-LogText([string]$Text, [string[]]$Secrets = @()) {
    foreach ($secret in $Secrets) {
        if (-not [string]::IsNullOrEmpty($secret)) { $Text = $Text.Replace($secret, '***') }
    }
    return $Text
}
function Get-ReservationGap([string]$Text) {
    if ([string]::IsNullOrEmpty($Text)) { return $null }
    $found = [regex]::Matches($Text, 'reservation requires (\d+) bytes, but only (\d+) bytes are available')
    if ($found.Count -eq 0) { return $null }
    $last = $found[$found.Count - 1]
    $need = 0L; $have = 0L
    if (-not [long]::TryParse($last.Groups[1].Value, [ref]$need) -or
        -not [long]::TryParse($last.Groups[2].Value, [ref]$have) -or $need -le $have) { return $null }
    return @{ Need = $need; Have = $have }
}

# 自动缩窗后，把 接入信息.txt 里的上下文 / 最大输出改成这次实际用的值
function Update-InfoFile([int]$c, [int]$m) {
    $p = Join-Path $Root '接入信息.txt'
    if (-not (Test-Path -LiteralPath $p)) { return }
    try {
        $t = Get-Content -LiteralPath $p -Raw -Encoding UTF8
        $t = $t -replace '(?m)^(\s*上下文窗口\s*:\s*)\d+', ('${1}' + $c)
        $t = $t -replace '(?m)^(\s*最大输出\s*:\s*)\d+', ('${1}' + $m)
        $t = $t -replace '(?m)^(\s*contextWindow:\s*)\d+', ('${1}' + $c)
        $t = $t -replace '(?m)^(\s*maxTokens:\s*)\d+', ('${1}' + $m)
        [IO.File]::WriteAllText($p, $t, (New-Object Text.UTF8Encoding $true))
    } catch {}
}

# ---------- 5. 按模式组参数 ----------
Get-ChildItem env: | Where-Object { $_.Name -like 'NINFER_TERNARY_*' } | ForEach-Object { Remove-Item "env:$($_.Name)" }

function Round64([string]$name, [int]$v) {
    $r = [int]([math]::Floor($v / 64) * 64)
    if ($r -ne $v) { Warn "$name=$v 不是 64 的倍数，已改成 $r。" }
    return $r
}

$env:NINFER_KVMEM_AUTO_ALLOCATION = '0'
$env:NINFER_KVMEM_ORPHAN_EVICT_FIX = '0'
$kv = 'rk8v4'; $exeName = 'ninfer-serve.exe'
if ($Mode -eq 'rk4' -or $Mode -eq 'kvrk4') { $kv = 'rk4v4'; $exeName = 'ninfer-serve-rk4.exe' }   # rk4v4 和 rk8v4 的内核同名，只能分成两个程序（放在同一个 engine 文件夹，共用 DLL）
$desc = ''
$limNow = Mode-Lim $Mode
switch ($Mode) {
    { $_ -eq 'normal' -or $_ -eq 'rk4' } {
        $p = $Pre[$Mode]
        $ctx = CfgInt "${p}_CTX" $(if ($Mode -eq 'rk4') { 86016 } else { 57344 }); $cap = $ctx
        $mt = CfgInt "${p}_MAX_TOKENS" 32768; $hostMib = 1024
        if ($ctx -gt $limNow) { Warn "${p}_CTX=$ctx 超过 $limNow，8 GB 显存可能放不下。" }
        $desc = "$($ModeName[$Mode] -replace '（.*$', '') 模式（$kv）：上下文 $ctx（全在显存），单次回答最多 $mt"
    }
    default {
        $p = $Pre[$Mode]
        $ctx = CfgInt "${p}_CTX" 262144
        $view = Get-View $Mode
        if ($view.Err.Count -gt 0) { Fail ($view.Err -join '；') }
        $win = [int]$view.Win; $ans = [int]$view.Ans
        $sink = 64 # bootstrap only; actual full prefix is allocated per request
        $hostMib = CfgInt "${p}_HOST_MIB" 10240
        $cap = [int]$view.Cap; $mt = [int]$view.Mt
        $env:NINFER_TERNARY_KVMEM_CPU_RETRIEVAL = $(if ($CpuRetrieval) { '1' } else { '0' })
        $env:NINFER_TERNARY_KVMEM = '1'
        $env:NINFER_KVMEM_AUTO_ALLOCATION = '1'
        # Auto allocator protects the entire rendered system/developer/tool prefix per request.
        if ($Mode -eq 'kvrk4') {
            $env:NINFER_KVMEM_AUTO_ALLOCATION = '1'
            $env:NINFER_KVMEM_ORPHAN_EVICT_FIX = '1'
        }
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
        $env:NINFER_TERNARY_KVMEM_GEN_RESERVE = "$ans"
        # kv8/kv9：按最后一条用户消息挑块、保护本轮新内容、重算范围 = 挑块范围（这一轮的新输入）
        $env:NINFER_TERNARY_KVMEM_SCORE_QUERY_MODE = 'msg'
        $env:NINFER_TERNARY_KVMEM_SCORE_PROTECT_NEW = '1'
        $env:NINFER_TERNARY_KVMEM_REPLAY_MODE = 'msg'
        $env:NINFER_TERNARY_KVMEM_REPLAY_MAX = "$(CfgInt 'KVMEM_REPLAY_MAX' 6144)"
        $name = if ($Mode -eq 'kvrk4') { 'KVMem+rk4v4 模式' } else { 'KVMem 模式（rk8v4）' }
        $desc = "${name}：上下文 $ctx；驻留预算 $cap；API输出上限 $mt"
        if ($Mode -eq 'kvrk4') { $desc += '；完整系统/工具前缀自动保留，实际sink/检索/输出以[kvmem-alloc]日志为准' }
    }
}
if ($mt -gt $ctx) { Fail "单次回答上限（$mt）不能比上下文（$ctx）还大。" }
$think = CfgInt 'THINK_BUDGET' 0
if ($think -gt 0 -and $think -lt $mt) { $desc += "，思考最多 $think" } else { $think = 0 }

# 8G 固定的省显存设置：词嵌入、视觉权重放内存；不留额外状态槽；读入分段 512
$env:NINFER_TERNARY_EMBED_HOST = '1'
$env:NINFER_TERNARY_VISION_HOST = '1'

$exe = Join-Path $Root "engine\$exeName"
if (-not (Test-Path -LiteralPath $exe)) { Fail "找不到引擎程序：$exe" }
# 引擎需要的 11 个 DLL 必须都在 engine\ 里（解压不全时会报奇怪的错，这里先拦住）
$dlls = @(Get-ChildItem -LiteralPath (Join-Path $Root 'engine') -Filter '*.dll' -ErrorAction SilentlyContinue)
if ($dlls.Count -lt 11) { Fail "engine 文件夹里的 DLL 只有 $($dlls.Count) 个（应该有 11 个）。请重新解压一键包（解压前把整个文件夹删掉）。" }

$bindHost = [string](Cfg 'HOST' '127.0.0.1')
$modelId = [string](Cfg 'MODEL_ID' 'qwen3.8-27b')
$draft = CfgInt 'MTP_DRAFT' 3
$vis = (CfgInt 'VISION' 1) -ne 0
$visionMax = CfgInt 'VISION_MAX_TOKENS' 4096
# 思考写完自动降温（正文和工具调用更稳）；设置.ini 里 POST_THINK_TEMP=0 可以关掉
$postThink = 0.2
$ptv = [string](Cfg 'POST_THINK_TEMP' '0.2')
if ($ptv -ne '') { $postThink = [double]$ptv }
if ($postThink -lt 0) { $postThink = 0 }
$key = [string](Cfg 'API_KEY' '')
$iniExtra = [string](Cfg 'EXTRA' '')
$isKv = ($Mode -eq 'kvmem' -or $Mode -eq 'kvrk4')
$argv = New-Argv $ctx $cap $mt $hostMib
if ($key -eq '' -and $bindHost -ne '127.0.0.1' -and $bindHost -ne 'localhost') { Warn "HOST=$bindHost 会让局域网里的其他电脑也能访问，建议在 设置.ini 里设置 API_KEY。" }

if (-not $dry) { Set-Content -LiteralPath $modeFile -Value $Mode -Encoding ASCII }

# ---------- 6. 显示并启动 ----------
$redaction = New-LogRedaction -Arguments $argv -AdditionalSecrets @($key)
$shown = $redaction.Text
Say ''
Say "  $desc" 'Green'
Say "  显卡  ：$gName（$gMem MiB，驱动 $gDrv）  内存：$ramGB GB"
Say "  地址  ：http://${bindHost}:$port/v1   模型名：$modelId"
Say "  参数  ：$shown"
Say '  出现 "listening" 之后就能用了。可以运行 测试.bat 检查。关掉这个窗口就停止引擎。' 'Cyan'
Say ''

# ---------- 6b. 客户端（自选框架）接入信息：显示在窗口里，并写到 接入信息.txt ----------
$vis = (CfgInt 'VISION' 1) -ne 0
$locHost = $(if ($bindHost -eq '0.0.0.0' -or $bindHost -eq '::') { '127.0.0.1' } else { $bindHost })
$baseUrl = "http://${locHost}:$port/v1"
$lanUrls = @()
if ($bindHost -eq '0.0.0.0') {
    try { $lanUrls = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | ForEach-Object { "http://$($_.IPAddress):$port/v1" }) } catch {}
}
$keyTxt = $(if ($key -ne '') { '设置.ini 里 API_KEY 的值' } else { '没设密码，随便填（例如 none），但不能空着' })
$info = @('NInfer 通用API接入信息', "Base URL: $baseUrl", "模型ID: $modelId", "上下文: $ctx", "最大输出: $mt（API请求上界，实际受完整前缀/检索/驻留动态预算约束）", '支持OpenAI Chat Completions/Responses与Anthropic Messages；客户端框架自选。')
Say '  ---- 客户端（自选框架）里这样填（通用说明见 接入信息.txt）----' 'Cyan'
Say "  Base URL：$baseUrl    模型名：$modelId    API Key：$keyTxt"
foreach ($u in $lanUrls) { Say "  局域网地址：$u" }
Say "  上下文窗口：$ctx    最大输出：$mt    图片：$(if ($vis) { '支持' } else { '不支持' })    工具调用：支持"
Say ''
if (-not $dry) {
    try { [IO.File]::WriteAllLines((Join-Path $Root '接入信息.txt'), [string[]]$info, (New-Object Text.UTF8Encoding $true)) }
    catch { Warn "写 接入信息.txt 失败：$($_.Exception.Message)" }
}

Confirm-ResidentBound (Get-CapacityInfo '8g' $kv ([string](Cfg 'CUDA_GRAPH' '1'))) $cap
if ($dry) {
    Get-ChildItem env: | Where-Object { $_.Name -like 'NINFER_TERNARY_*' -or $_.Name -like 'NINFER_KVMEM_*' } | Sort-Object Name | ForEach-Object { Say "  环境变量：$($_.Name)=$($_.Value)" 'DarkGray' }
    Say '  （DRYRUN=1：只显示参数，没有启动引擎）' 'Yellow'; exit 0
}
$ErrorActionPreference = 'Continue'
$env:PATH = (Split-Path -Parent $exe) + ';' + $env:PATH
# 日志：窗口里照常显示，同时存一份到 logs\（显存不够时要靠它读出差额，出问题也好查）
$logDir = Join-Path $Root 'logs'
try { New-Item -ItemType Directory -Force $logDir | Out-Null } catch {}
try {
    Get-ChildItem $logDir -Filter 'engine-*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -Skip 9 | Remove-Item -Force -ErrorAction SilentlyContinue
} catch {}
$logFile = Join-Path $logDir ('engine-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '.log')
$logW = $null
try {
    $logW = New-Object IO.StreamWriter($logFile, $true, (New-Object Text.UTF8Encoding $false))
    $logW.AutoFlush = $true
    $logW.WriteLine('# ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' mode=' + $Mode + ' ' + $exe + ' ' + $model + ' ' + $shown)
} catch { $logW = $null }
$retry = [Math]::Max(0, [Math]::Min(3, (CfgInt 'RETRY' 2)))
$samples = @()
$finalRc = 1
for ($try = 1; $try -le (1 + $retry); $try++) {
    if ($try -gt 1) {
        if ($isKv) { Say ("  第 {2} 次：显存容量 {0}（检索窗口 {1} + 回答 {3}）..." -f $cap, $win, $try, $ans) 'Cyan' }
        else { Say ("  第 {2} 次：上下文 {0}，输出 {1}..." -f $ctx, $mt, $try) 'Cyan' }
    }
    $attempt = @{ Gap = $null; Ready = $false }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    if(-not (Confirm-CapacityStart $argv)){Return-CapacityMenu $PSCommandPath;exit 1}
    & $exe $model @argv 2>&1 | ForEach-Object {
        $raw = [string]$_; Capture-CapacityFailure $raw
        if ($raw -match '\blistening on https?://') { $attempt.Ready = $true }
        $currentGap = Get-ReservationGap $raw
        if ($currentGap) { $attempt.Gap = $currentGap }
        $line = Protect-LogText -Text $raw -Secrets $redaction.Secrets
        Write-Host $line; if ($logW) { $logW.WriteLine($line) }
    }
    $rc = $LASTEXITCODE; $finalRc = $rc; $secs = [int]$sw.Elapsed.TotalSeconds
    if($script:CapacityFailureLine){break}
    if ($logW) { $logW.WriteLine("# engine exited rc=$rc after $secs s (try $try)"); $logW.Flush() }
    if ($rc -eq 0) { break }
  if($script:CapacityFailureLine){break}
    Say "[引擎退出] 代码 $rc，运行 $secs 秒，第 $try 次，模式 $Mode" 'Yellow'
    if ($attempt.Ready) {
        if ($try -lt (1 + $retry)) { Say '  15 秒后自动重启...'; Start-Sleep -Seconds 15 }
        continue
    }
    if($isKv -and ([string](Cfg ($Pre[$Mode]+'_RESIDENT') 'recommended')) -notin @('auto','recommended')){Say '手动驻留未改动；启动失败，请按C自行调整。' 'Yellow';break}
    $gap = $attempt.Gap
    if (-not $gap) {
        Say '  本轮启动失败，未报告可缩窗的显存容量差额；请查看本轮日志。' 'Yellow'
        break
    }
    if ($try -ge (1 + $retry)) { Say '  已达到重试上限，请释放显存后重新启动。' 'Yellow'; break }
    $samples += @{ Cap = $(if ($isKv) { $cap } else { $ctx }); Need = $gap.Need; Have = $gap.Have }
    $deficit = $gap.Need - $gap.Have
    Say ("  [显存不足] 需要 {0:N0} MiB，可用 {1:N0} MiB，差 {2:N0} MiB" -f ($gap.Need / 1MB), ($gap.Have / 1MB), ($deficit / 1MB)) 'Yellow'
    $step = 4096
    if ($samples.Count -ge 2) {
        $previous = $samples[$samples.Count - 2]; $latest = $samples[$samples.Count - 1]
        $dCap = $previous.Cap - $latest.Cap; $dNeed = $previous.Need - $latest.Need
        if ($dCap -gt 0 -and $dNeed -gt 0) { $step = [int][Math]::Ceiling(($deficit + 128MB) / ($dNeed / $dCap)) }
    }
    $step = [int]([Math]::Ceiling($step / 64.0) * 64); if ($step -lt 1024) { $step = 1024 }
    if ($isKv) {
        $new = [int]([Math]::Floor(($cap - $step) / 64.0) * 64)
        $floorCap = 8448
        if ($new -lt $floorCap) { $new = $floorCap }
        if ($new -ge $cap -or $new -lt 8448) { Say '  检索窗口已达到下限，请释放显存后重试。' 'Yellow'; break }
        $cap = $new; $win = $cap; $ans = 0
        if ($view.AutoOut) { $mt = $cap }
        $env:NINFER_TERNARY_KVMEM_GEN_RESERVE = '0'
        $env:NINFER_TERNARY_KVMEM_SCORE_BUDGET = "$win"
    } else {
        $new = [int]([Math]::Floor(($ctx - $step) / 64.0) * 64)
        if ($new -lt 24576) { $new = 24576 }
        if ($new -ge $ctx) { Say '  上下文已达到下限，请释放显存后重试。' 'Yellow'; break }
        $ctx = $new; $cap = $ctx
        if ($mt -gt $ctx - 8192) { $mt = [int]([Math]::Floor(($ctx - 8192) / 64.0) * 64) }
        if ($think -ge $mt) { $think = [Math]::Max(0, $mt - 1) }
    }
    $argv = New-Argv $ctx $cap $mt $hostMib
    Update-InfoFile $ctx $mt
    if ($logW) { $logW.WriteLine("# auto-shrink: ctx=$ctx cap=$cap win=$(if ($isKv) { $win } else { 0 }) mt=$mt") }
}
if ($logW) { $logW.Close() }
if($script:CapacityFailureLine){Register-CapacityFailure;Return-CapacityMenu $PSCommandPath ([bool]$NoSync)}
exit $finalRc
