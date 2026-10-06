# ninfer Swift 1.5 懒人包 启动器（由 启动.bat 调用）
#   模型：Swift-1.5 Qwen3.8-27B IQ2_S（魔搭 fyb423/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer），MTP 草稿头用补丁转成 Q4
#   引擎：engine\ninfer-serve.exe（Ryan-gsq fork b06908b + KVMem 移植 + 3060 精简 + rk4v4；SM 数运行时自动识别）
#   模式固定 KVMem：完整对话记录放内存（Host KV），显存只放一个驻留窗口；实际完整前缀、检索和输出由引擎逐请求动态分配
#   启动.bat            先显示当前配置：回车=启动，C=一步步重新选，R=重新读空闲显存，Q=退出
#   启动.bat last       直接用上次配置启动，不提问
#   启动.bat dryrun     只显示最终命令，不启动、不下载、不改任何文件
#   测试用环境变量：L8084_INPUT=答案1|答案2|...（代替键盘输入）  L8084_EXTRA_ARGS=额外引擎参数
#   测试下载流程：ONECLICK_TEST_URL / _SIZE / _SHA / _FILE（用小文件代替模型）
$ErrorActionPreference = 'Stop'
$launchArguments = @($args)
. (Join-Path $PSScriptRoot 'runtime-options.ps1')
. (Join-Path $PSScriptRoot 'capacity-options.ps1')

$argv    = @($launchArguments | ForEach-Object { "$_".ToLower() })
$DryRun  = $argv -contains 'dryrun'
$Here    = Split-Path $PSScriptRoot -Parent
$CfgFile = Join-Path $Here '设置.ini'
$Exe     = Join-Path $Here 'engine\ninfer-serve.exe'

# ---------------- 模型 ----------------
# 引擎用 Q4 版；它由魔搭上的原版文件加 patch 文件夹里的补丁生成（只改 MTP 草稿头的 7 个张量，其余原样复制）
$Q4   = @{ Name = 'Swift-1.5 IQ2_S（MTP 草稿头转成 Q4）'; File = 'swift15_iq2_s_mtpq4.ninfer'; Size = [long]10155600640
           Sha = '1941ef5f4bd4be61938521236d0e499c5e6d69ff69b4fbcfe83527f7fd9b3bdd' }
$Orig = @{ Name = 'Swift-1.5 Qwen3.8-27B IQ2_S'; File = 'swift15_iq2s_mtp.ninfer'; Size = [long]10257632000
           Sha = '2df7259e93cbbb972183966e40ff23392e66a8ce4fc84748d08849bc6182045e'
           Page = 'https://www.modelscope.cn/models/fyb423/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer'
           Url = 'https://www.modelscope.cn/models/fyb423/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer/resolve/master/swift15_iq2s_mtp.ninfer' }   # 魔搭说明页写的旧名 swift15_iq2_s.ninfer，实际文件名是这个（同一个文件，SHA256 一样）
$PatchFiles = @{ 'ops.txt' = [long]5427; 'lit.bin' = [long]203691823 }
if ($env:ONECLICK_TEST_URL) { $Orig.Url = $env:ONECLICK_TEST_URL; $Orig.Size = [long]$env:ONECLICK_TEST_SIZE; $Orig.Sha = $env:ONECLICK_TEST_SHA; $Orig.File = $env:ONECLICK_TEST_FILE }
$Model   = Join-Path $Here ('model\' + $Q4.File)
# Existing verified local model; never download a duplicate when this file is present.
if ($env:NINFER_SWIFT15_MODEL) { $Model = $env:NINFER_SWIFT15_MODEL }
$SrcFile = Join-Path $Here ('model\' + $Orig.File)

# ---------------- 固定数据（RTX 3060 12G 上实测标定） ----------------
# 显存窗口页数（1 页 = 64 token）按启动时 nvidia-smi 的空闲显存自动算，目标整卡剩 150–200 MiB：
#   引擎可用 ≈ 空闲 − 权重等（MTP Q4 + 词嵌入放内存时约 9115 MiB）
#   页数 = (引擎可用 − 固定运行时 875 − 留 70) / 每页显存；估多了也没事：放不下会按引擎报的真实可用量自动减页重试
#   每页显存：rk8v4 1.694 MiB；rk4v4 1.17 MiB
$FixedMiB = 875; $LoadMiB = 9115; $KeepMiB = 70
$LoadByMtp = @{ q4 = 9115 }
# MTP 输出头选"精简"（--lm-head-draft）：多放一个 340 MiB 的精简输出头，实测整卡多 346 MiB
$HeadMiB = @{ full = 0; lite = 346 }
$KvInfo = [ordered]@{
  rk4v4 = @{ PageMiB = 1.17;  PageMax = 1152; CtxMax = 262144; Name = 'rk4v4（K、V 都 4 位）' }
  rk8v4 = @{ PageMiB = 1.694; PageMax = 800; CtxMax = 131072; Name = 'rk8v4（K 8 位 + V 4 位）' }
}
$RestMin  = 64      # 开头 + 检索/最近 最少 64 页（4K），否则不启动
$OutCap = 262144 # logical API bound only, not a fixed physical partition
$OutMin   = 8192    # 自动输出至少 8K
$SmallCtx = 131072  # 精简头或 rk8v4 时窗口小，小驻留推荐128K，不限制用户选择，完整前缀/检索/输出按请求分配
$Defaults = [ordered]@{ RESIDENT='recommended'; CUDA_GRAPH='1'; VISION_DEVICE='cpu'; CPU_THREADS='auto';  KV = 'rk4v4'; HEAD = 'full'; VISION = '1'; CTX = '204800'; OUT = 'auto'; SYS = '8192'; THINK = '0'
                        POST_THINKING = '1'; POST_THINKING_TEMP = ''; POST_THINKING_TOP_P = ''; POST_THINKING_TOP_K = ''; POST_THINKING_SAMPLER = ''
                        ADAPTIVE_MTP = '0'; RECOVER_INVARIANT = '1'
                        PORT = '8084'; HOST = '127.0.0.1'; API_KEY = ''; MODEL_ID = 'qwen3.8-27b' }

# ---------------- 小工具 ----------------
$TestInput = New-Object System.Collections.Queue
if ($env:L8084_INPUT) { $env:L8084_INPUT.Split('|') | ForEach-Object { $TestInput.Enqueue($_) } }
function Read-Answer([string]$prompt) {
  if ($TestInput.Count -gt 0) { $a = [string]$TestInput.Dequeue(); Write-Host "$prompt$a"; return $a.Trim() }
  if ($env:L8084_INPUT) { throw 'test input exhausted' }
  return (Read-Host $prompt).Trim()
}
function Parse-Tokens([string]$s) {
  $s = $s.Trim().ToLower()
  if ($s -match '^(\d+(\.\d+)?)k$') { return [int]([double]$matches[1] * 1024) }
  if ($s -match '^\d{3,7}$') { return [int]$s }
  return $null
}
function Fmt([int]$n) { if ($n % 1024 -eq 0) { return ('{0}（{1}K）' -f $n, ($n / 1024)) } else { return "$n" } }
function K([int]$pages) { return ('{0}K' -f ($pages * 64 / 1024)) }
function Fail([string]$m) { Write-Host "  [错误] $m" -ForegroundColor Red; exit 1 }
function GiB([long]$b) { return ('{0:N1}' -f ($b / 1GB)) }
function Free-MiB {
  try {
    $q = & nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    $a = $q -split ','; return ([double]$a[0].Trim() - [double]$a[1].Trim())
  } catch { return -1 }
}
function Pages-From([string]$kv, [double]$availMiB) {
  $i = $KvInfo[$kv]
  $p = [int][Math]::Floor(($availMiB - $FixedMiB - $KeepMiB) / $i.PageMiB)
  $p = $p - ($p % 8)
  return [Math]::Max(0, [Math]::Min(4096, $p))
}
function Est-Pages([string]$kv){
 $c=$script:cfg;if(-not $c){$c=$Defaults}
 $i=Get-CapacityInfo 'swift15' $kv ([string]$c.CUDA_GRAPH) ([string]$c.HEAD)
 try{return [int]((Resolve-Capacity ([string]$c.RESIDENT) $i ([int]$c.CTX))/64)}catch{Write-Host $_.Exception.Message -ForegroundColor Yellow;return 0}
}

function Is-Half($c) { return -not ($c.KV -eq 'rk4v4' -and $c.HEAD -ne 'lite') }   # 兼容旧模式标识；不参与固定分区
function Ctx-Rec($c) { return [Math]::Min(204800, (Ctx-Max $c)) }   # 10-03 晚：rk4v4 + 完整头推荐 200K，最高可设 256K
function Ctx-Max($c) { return 262144 }

function Auto-Out([int]$pages, [int]$sys, [int]$ctx, [bool]$half = $false) { return [Math]::Min($pages * 64,$ctx) }
function Out-Of($c, [int]$pages) { if ("$($c.OUT)" -eq 'auto') { return (Auto-Out $pages ([int]$c.SYS) ([int]$c.CTX) (Is-Half $c)) } else { return [int]$c.OUT } }
function Auto-Desc($c) { return 'API请求上界跟随驻留；实际前缀/检索/输出由引擎逐请求分配' }
function Gen-Pages($c, [int]$pages) { return 0 } # bootstrap only; no fixed generation partition
function Split-Window($c, [int]$pages) { return @(2, [Math]::Max(0,$pages - 2)) } # legal bootstrap; overwritten per request
function Load-For([string]$mtp, [string]$head) { return ($LoadByMtp[$mtp] + $HeadMiB[$head]) }
function Out-Max([int]$pages) { return ($pages * 64) }
function Host-MiB($c) {
  # 内存里的完整对话记录（Host KV）：按"最长对话 × 2"预留（能同时留住 2 个满长对话的接续点），6–10 GB
  # 10-02 实测：引擎本身还要约 14 GB 提交内存（显存在系统里的记账等），Host KV 10 GB 时整机提交 46.9/47.7 GB 太满，所以从 ×2.5 降到 ×2
  $need = [int]$c.CTX / 64 * $KvInfo[$c.KV].PageMiB * 2
  $m = [int]([Math]::Ceiling($need / 1024) * 1024)
  return [Math]::Max(6144, [Math]::Min(10240, $m))
}


function Load-Config {
  $c = [ordered]@{}; foreach ($k in $Defaults.Keys) { $c[$k] = $Defaults[$k] }
  if (Test-Path -LiteralPath $CfgFile) {
    foreach ($line in Get-Content -LiteralPath $CfgFile -Encoding UTF8) {
      if ($line -match '^\s*([A-Za-z_]+)\s*=\s*(.*?)\s*$' -and $c.Contains($matches[1].ToUpper())) { $c[$matches[1].ToUpper()] = $matches[2] }
    }
  }
  if (-not $KvInfo.Contains($c.KV)) { $c.KV = 'rk4v4' }
  if ($c.VISION -ne '0') { $c.VISION = '1' }
  if ($c.POST_THINKING -ne '0') { $c.POST_THINKING = '1' }
  if ($c.ADAPTIVE_MTP -ne '1') { $c.ADAPTIVE_MTP = '0' }
  if ($c.RECOVER_INVARIANT -ne '0') { $c.RECOVER_INVARIANT = '1' }
  if (-not $HeadMiB.Contains($c.HEAD)) { $c.HEAD = 'full' }
  $c.MTP = 'q4'
  if ("$($c.OUT)".ToLower() -eq 'auto') { $c.OUT = 'auto' }
  foreach ($k in 'CTX', 'OUT', 'SYS', 'THINK', 'PORT') { if ($c[$k] -eq 'auto') { continue }; $n = 0; if (-not [int]::TryParse("$($c[$k])", [ref]$n)) { $c[$k] = $Defaults[$k] } }
  if (-not $c.HOST) { $c.HOST = '127.0.0.1' }
  if (-not $c.MODEL_ID) { $c.MODEL_ID = 'qwen3.8-27b' }
  Initialize-NinferRuntimeOptions $c
  return $c
}
function Use-Mtp($c) { $script:LoadMiB = Load-For 'q4' $c.HEAD }
function Save-Config($c) {
  $t = @(
    '; ninfer Swift 1.5 懒人包 设置（启动器每次启动会按你选的写回来；也可以用记事本手改，改完重新运行 启动.bat）',
    '; 推荐直接双击 启动.bat 按 C 一步步选，每一步都有推荐值和上限',
    '',
    '; KV 量化：rk4v4（推荐，总上下文最多 256K，更快）/ rk8v4（小驻留推荐128K；可自选到256K）',
    "KV=$($c.KV)",
    '; MTP 输出头：full（完整，推荐，显存窗口最大）/ lite（精简 --lm-head-draft，解码快约 10%，多占 346 MiB 显存，小驻留推荐128K，不限制用户选择）',
    "HEAD=$($c.HEAD)",
    '; 看图：1 开 / 0 关（CPU 视觉编码、自动线程；语言模型及图片特征仍需显存）',
    "VISION=$($c.VISION)",
    '; CUDA_GRAPH: 1开启 / 0关闭；VISION_DEVICE只允许cpu；CPU_THREADS:auto或正整数',
    "CUDA_GRAPH=$($c.CUDA_GRAPH)",
    "VISION_DEVICE=$($c.VISION_DEVICE)",
    "CPU_THREADS=$($c.CPU_THREADS)",
    '; 总上下文（token）：rk4v4 + 完整头 推荐 204800、最多 262144；小驻留组合推荐131072；可选到262144',
    '; RESIDENT=recommended推荐档 / auto实时估算 / 手动token数；推荐不是上限',
    "RESIDENT=$($c.RESIDENT)",
    "CTX=$($c.CTX)",
    '; OUT=auto：API请求上界跟随驻留，完整前缀/检索/实际输出由引擎动态分配；数值为显式API上限',
    "OUT=$($c.OUT)",
    '; SYS为旧版兼容字段，自动分配忽略它，始终保护实际完整前缀',
    "SYS=$($c.SYS)",
    '; 思考上限：0 = 不限（推荐）',
    "THINK=$($c.THINK)",
    '',
    '; ---- v3 引擎推理控制开关（可在下方手改） ----',
    '; 思考结束后自动切换采样参数（1 开 / 0 关，开启后遇到 </think> 自动把温度降到 0.2，思考发散、正文稳定）',
    "POST_THINKING=$($c.POST_THINKING)",
    '; 思考结束后的自定义采样参数（空 = 使用引擎默认 0.2；可填 POST_THINKING_TEMP=0.2、POST_THINKING_TOP_P=0.95、POST_THINKING_TOP_K=20 或 POST_THINKING_SAMPLER=temp=0.2,top_p=0.95）',
    "POST_THINKING_TEMP=$($c.POST_THINKING_TEMP)",
    "POST_THINKING_TOP_P=$($c.POST_THINKING_TOP_P)",
    "POST_THINKING_TOP_K=$($c.POST_THINKING_TOP_K)",
    "POST_THINKING_SAMPLER=$($c.POST_THINKING_SAMPLER)",
    '; 自适应 MTP 草稿长度（0 固定 3 token / 1 开启 --adaptive-mtp）',
    "ADAPTIVE_MTP=$($c.ADAPTIVE_MTP)",
    '; 内部不变量异常时自动恢复请求而不崩溃整个进程（1 开 / 0 关，推荐 1）',
    "RECOVER_INVARIANT=$($c.RECOVER_INVARIANT)",
    '',
    '; ---- 下面几项向导里不问，需要时手改 ----',
    '; 端口',
    "PORT=$($c.PORT)",
    '; 监听地址：127.0.0.1 只给本机用；0.0.0.0 让局域网其他电脑也能用（请同时设 API_KEY）',
    "HOST=$($c.HOST)",
    '; 访问密码（空 = 不设）',
    "API_KEY=$($c.API_KEY)",
    '; 模型名（客户端里填的模型 ID）',
    "MODEL_ID=$($c.MODEL_ID)"
  )
  [IO.File]::WriteAllLines($CfgFile, [string[]]$t, (New-Object Text.UTF8Encoding $true))
}

# ---------------- 检查配置 ----------------
function Check-Config($c, [int]$pages) {
  $r = @{ Err=@(); Warn=@() }; $ctx=[int]$c.CTX; $out=Out-Of $c $pages
  if ($ctx -lt 16384 -or $ctx -gt (Ctx-Max $c)) { $r.Err += 'Invalid logical context for this KV/head combination' }
  if ($out -lt 1 -or $out -gt $ctx) { $r.Err += 'API output limit must be positive and no larger than context' }
  if ($pages * 64 -lt [int]($ctx / 8) + 8192 + 256) { $r.Warn += '低于检索推荐预算；按完整前缀动态缩减检索' }
  if ([int]$c.THINK -gt $out) { $r.Err += 'Thinking limit exceeds API output limit' }
  return $r
}


function Show-Summary($c, [int]$pages) {
  Show-CapacityInfo (Get-CapacityInfo 'swift15' $c.KV $c.CUDA_GRAPH $c.HEAD)
  Write-Host ('  驻留选择：'+$c.RESIDENT)
  Show-NinferRuntimeOptions $c
  $i = $KvInfo[$c.KV]
  Write-Host ''
  Write-Host '  ┌──────────── ninfer Swift 1.5 懒人包 · 当前配置 ────────────'
  Write-Host ('  │ 显卡      : {0}（{1} MiB，算力 {2}）' -f $script:GpuName, $script:GpuTotal, $script:GpuCap)
  Write-Host '  │ 模型      : Swift-1.5 Qwen3.8-27B IQ2_S（MTP 草稿头 Q4）'
  Write-Host ('  │ 模型文件  : ' + $(if (Test-Path -LiteralPath $Model) { '已就绪' } elseif (Test-Path -LiteralPath $SrcFile) { '原版已下载，启动时转换' } else { '还没下载（回车启动时会先下载约 9.6 GiB）' }))
  Write-Host '  │ 模式      : KVMem（完整对话放内存，显存放窗口；超窗后每轮只算新内容）'
  Write-Host ('  │ KV 量化   : ' + $i.Name)
  if ($c.HEAD -eq 'lite') { Write-Host '  │ MTP 输出头: 精简（解码快约 10%，多占 346 MiB 显存，窗口少约 300 页）' } else { Write-Host '  │ MTP 输出头: 完整（窗口最大）' }
  if ($c.VISION -ne '0') { Write-Host '  │ 看图      : 开（CPU 视觉编码，内存权重，线程见运行选项）' } else { Write-Host '  │ 看图      : 关' }
  Write-Host ('  │ 总上下文  : {0}   上限 {1}' -f (Fmt ([int]$c.CTX)), (Fmt (Ctx-Max $c)))
  if ("$($c.OUT)" -eq 'auto') { Write-Host ('  │ 单次输出  : 自动 = {0}（{1}；每次启动按实际窗口算）' -f (Fmt (Out-Of $c $pages)), (Auto-Desc $c)) }
  else { Write-Host ('  │ 单次输出  : {0}   （思考 + 正文 合计，整段留在显存不滚出去）' -f (Fmt ([int]$c.OUT))) }
  if ([int]$c.THINK -gt 0) { Write-Host ('  │ 思考上限  : ' + (Fmt ([int]$c.THINK))) } else { Write-Host '  │ 思考上限  : 不限（只受单次输出限制）' }
  $v3Opts = @()
  if ($c.POST_THINKING -ne '0') { $v3Opts += $(if ($c.POST_THINKING_TEMP) { "思考后降温($($c.POST_THINKING_TEMP))" } else { '思考后降温(0.2)' }) }
  if ($c.ADAPTIVE_MTP -eq '1') { $v3Opts += '自适应MTP' }
  if ($c.RECOVER_INVARIANT -ne '0') { $v3Opts += '异常自动恢复' }
  Write-Host ('  │ 推理控制  : ' + $(if ($v3Opts.Count -gt 0) { $v3Opts -join ' / ' } else { '默认' }))
  if ($script:Free -gt 0 -and $pages -ge (Gen-Pages $c $pages) + $RestMin) {
    $sp = Split-Window $c $pages
    Write-Host ('  │ 显存窗口  : {0} 页 = {1} token（当前空闲显存 {2} MiB；选择按RESIDENT执行）' -f $pages, (K $pages), [int]$script:Free)
    Write-Host '  │   分配    : 实际完整前缀 + 按逻辑窗口比例检索 + 剩余输出；请求到达后确定'
  } elseif ($script:Free -le 0) { Write-Host '  │ 显存窗口  : 读不到空闲显存，禁止加载；请刷新后重选' }
  Write-Host ('  │ 内存      : 对话记录预留 {0} MiB（Host KV）；最多保留 4 个对话的接续点' -f (Host-MiB $c))
  Write-Host ('  │ 地址      : http://{0}:{1}/v1    模型名 {2}{3}' -f $(if ($c.HOST -eq '0.0.0.0') { '127.0.0.1' } else { $c.HOST }), $c.PORT, $c.MODEL_ID, $(if ($c.API_KEY) { '    （设了 API_KEY）' } else { '' }))
  Write-Host '  └──────────────────────────────────────────────'
  $chk = Check-Config $c $pages
  foreach ($w in $chk.Warn) { Write-Host "  [提醒] $w" -ForegroundColor Yellow }
  foreach ($x in $chk.Err)  { Write-Host "  [错误] $x" -ForegroundColor Red }
  return $chk
}

# ---------------- 一步步选 ----------------
function Ask-Step([string]$title, [string[]]$intro, $items, $current, [string]$numHint) {
  $allowNum = [bool]$numHint
  Write-Host ''
  Write-Host "  ── $title ──" -ForegroundColor Cyan
  foreach ($l in $intro) { Write-Host "    $l" }
  for ($n = 0; $n -lt $items.Count; $n++) { Write-Host ('   [{0}] {1}' -f ($n + 1), $items[$n].Label) }
  $curLabel = ($items | Where-Object { "$($_.Value)" -eq "$current" } | Select-Object -First 1).Label
  if (-not $curLabel) { $curLabel = "$current" }
  Write-Host "   直接回车 = 保持：$($curLabel.Split('　')[0])"
  if ($allowNum) { Write-Host "   也可以直接输入数字，例如 $numHint（按 1024 取整）" }
  while ($true) {
    $a = Read-Answer '   请选择: '
    if ($a -eq '') { return $current }
    if ($a -match '^\d{1,2}$' -and [int]$a -ge 1 -and [int]$a -le $items.Count) { return $items[[int]$a - 1].Value }
    if ($allowNum) { $v = Parse-Tokens $a; if ($v) { return [string]([int]([Math]::Floor($v / 1024) * 1024)) } }
    Write-Host '   看不懂这个输入，请重新输入' -ForegroundColor Yellow
  }
}

function Clamp-Step($c, [string]$key, [int]$lo, [int]$hi, [string]$what) {
  $v = [int]$c[$key]
  if ($v -gt $hi) { Write-Host ('   {0}最多 {1}，已改成 {1}' -f $what, (Fmt $hi)) -ForegroundColor Yellow; $c[$key] = "$hi" }
  elseif ($v -lt $lo) { Write-Host ('   {0}最少 {1}，已改成 {1}' -f $what, (Fmt $lo)) -ForegroundColor Yellow; $c[$key] = "$lo" }
}

function Run-Wizard($c){
 $c.KV=Ask-CapacityChoice '1. KV格式（推荐rk4v4）' $c.KV @('rk4v4','rk8v4')
 $c.HEAD=Ask-CapacityChoice '2. MTP输出头（推荐full；lite多占显存）' $c.HEAD @('full','lite')
 Edit-NinferRuntimeOptions $c -AskVision;Use-Mtp $c
 $i=Get-CapacityInfo 'swift15' $c.KV $c.CUDA_GRAPH $c.HEAD
 $n=Edit-CapacityBudget $c 'RESIDENT' $i
 $c.CTX=Ask-CapacityValue '逻辑上下文：推荐200k；64k/128k/200k/256k' $c.CTX ([Math]::Max(16384,$n)) 262144 @() 64
 $c.OUT=Ask-CapacityValue 'API回答上限：推荐auto；8k/32k/64k' $c.OUT 1 ([int]$c.CTX) @('auto')
 $o=$(if($c.OUT -eq 'auto'){$n}else{[int]$c.OUT})
 $c.THINK=Ask-CapacityValue '思考上限：0不限（推荐）' $c.THINK 0 ($o-1)
 return $c
}

function Build-Args($c, [int]$pages) {
  $a = @($Model, '--host', $c.HOST, '--port', "$($c.PORT)", '--model-id', $c.MODEL_ID,
         '--max-concurrency', '1', '--kv-dtype', $c.KV, '--gdn-state-fp16',
         '--spec', 'mtp', '--draft-tokens', '3',
         '--max-context', "$($c.CTX)", '--kv-capacity', 'auto', '--kvmem-window-pages', "$pages",
         '--host-kv-mib', "$(Host-MiB $c)", '--device-state-slots', '0', '--prefill-chunk', '512',
         '--default-max-tokens', "$($c.OUT)",
         '--max-private-continuations', '4', '--embedding-host')   # --embedding-host：词嵌入（388 MiB）放内存，速度不变，窗口多约 300 页
  if ($c.HEAD -eq 'lite') { $a += @('--lm-head-draft') }
  if ($c.ADAPTIVE_MTP -eq '1') { $a += @('--adaptive-mtp') }
  if ($c.RECOVER_INVARIANT -ne '0') { $a += @('--recover-invariant-failures') }
  if ($c.POST_THINKING -ne '0') {
    $a += @('--post-thinking')
    if ($c.POST_THINKING_TEMP) { $a += @('--post-thinking-temperature', "$($c.POST_THINKING_TEMP)") }
    if ($c.POST_THINKING_TOP_P) { $a += @('--post-thinking-top-p', "$($c.POST_THINKING_TOP_P)") }
    if ($c.POST_THINKING_TOP_K) { $a += @('--post-thinking-top-k', "$($c.POST_THINKING_TOP_K)") }
    if ($c.POST_THINKING_SAMPLER) { $a += @('--post-thinking-sampler', "$($c.POST_THINKING_SAMPLER)") }
  }
  if ($c.VISION -ne '0') { $a += @('--vision', '--vision-residency', 'cpu', '--vision-max-merged', '4096') }
  if ([int]$c.THINK -gt 0) { $a += @('--default-thinking-budget', "$($c.THINK)") }
  if ($c.API_KEY) { $a += @('--api-key', $c.API_KEY) }
  if ($env:L8084_EXTRA_ARGS) { $a += @($env:L8084_EXTRA_ARGS -split '\s+' | Where-Object { $_ }) }
  return (Complete-NinferRuntimeArgs -Config $c -Arguments $a -Swift)
}
function Set-KvmEnv($c, [int]$pages) {
  $sp = Split-Window $c $pages
  $env:NINFER_KVMEM_SINK_PAGES = "$($sp[0])"
  $env:NINFER_KVMEM_GEN_RESERVE_PAGES = "$(Gen-Pages $c $pages)"
  $env:NINFER_KVMEM_LONG_REUSE = '1'
  $env:NINFER_KVMEM_AUTO_ALLOCATION = '1'
}

# ---------------- 模型下载和转换（和 8G 懒人包同一套，已实测） ----------------
function Say([string]$m, [string]$c = 'Gray') { Write-Host $m -ForegroundColor $c }
function Warn([string]$m) { Say "  [注意] $m" 'Yellow' }
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
    Say ('    然后自动用补丁把 MTP 草稿头转成 Q4（约 1–3 分钟，转完原始文件会删掉）。转换时要多占约 {0} GiB 硬盘。' -f (GiB $Q4.Size))
    Say '    也可以自己用浏览器从上面的网页下载这个文件，放进 model 文件夹（文件名不要改），再运行 启动.bat。'
    $free = [long](Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='" + $dir.Substring(0, 2) + "'")).FreeSpace
    $need = $m.Size - $have + $Q4.Size + 512MB
    if ($free -lt $need) { Fail ("硬盘空间不够：{0} 盘只剩 {1} GiB，下载加转换还需要约 {2} GiB（转完会删掉原始文件，最后只占约 {3} GiB）。请清理空间，或把整个文件夹挪到空间大的盘。" -f $dir.Substring(0, 1), (GiB $free), (GiB $need), (GiB $Q4.Size)) }
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
    Say '  下载完成，正在校验文件（10 GB 大约要 1–2 分钟）……'
    $h = (Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash.ToLower()
    if ($h -ne $m.Sha) {
        Move-Item -LiteralPath $part "$dst.bad" -Force
        Fail ("文件校验不对（下载时出错了），已改名为 {0}.bad。请删掉它，重新运行 启动.bat 下载。" -f (Split-Path -Leaf $dst))
    }
    Move-Item -LiteralPath $part $dst -Force
    Say '  校验通过。' 'Green'
}

# 用补丁把原版文件转成 MTP Q4 版：ops.txt 每行 "C 偏移 长度"（从原始文件复制）或 "L 偏移 长度"（从 lit.bin 复制）
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
    $pd = Join-Path $Here 'patch'
    foreach ($f in $PatchFiles.Keys) {
        $x = Join-Path $pd $f
        if (-not (Test-Path -LiteralPath $x) -or (Get-Item -LiteralPath $x).Length -ne $PatchFiles[$f]) { Fail "补丁文件缺失或不完整：$x。请重新解压一键包。" }
    }
    $dir = Split-Path -Parent $dst
    $free = [long](Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='" + $dir.Substring(0, 2) + "'")).FreeSpace
    if ($free -lt $Q4.Size + 256MB) { Fail ("硬盘空间不够：转换需要约 {0} GiB，{1} 盘只剩 {2} GiB。" -f (GiB ($Q4.Size + 256MB)), $dir.Substring(0, 1), (GiB $free)) }
    Say ''
    Say '  ── 用补丁把 MTP 草稿头转成 Q4（省约 95 MiB 显存，速度、猜中率不变）──' 'Cyan'
    $part = "$dst.part"
    try { Apply-ModelPatch $src (Join-Path $pd 'ops.txt') (Join-Path $pd 'lit.bin') $part }
    catch { Remove-Item -LiteralPath $part -Force -EA SilentlyContinue; Fail ("转换失败：{0}。`n       如果反复出现，请删掉 model 文件夹里的 {1}，重新运行 启动.bat 下载。" -f $_.Exception.Message, (Split-Path -Leaf $src)) }
    Say '  转换完成，正在校验（约 1–2 分钟）……'
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

# 模型在就直接用；不在就下载 + 转换
function Ensure-Model {
    if (Test-Path -LiteralPath $Model) {
        if ((Get-Item -LiteralPath $Model).Length -eq $Q4.Size) { return }
        Warn '模型文件大小不对，重新转换。'
        Remove-Item -LiteralPath $Model -Force
    }
    $s = $SrcFile
    $old = Join-Path $Here 'model\swift15_iq2_s.ninfer'   # 旧文件名（魔搭说明页上的名字），手动下载的可能是这个名字
    if (-not (Test-Path -LiteralPath $s) -and (Test-Path -LiteralPath $old) -and (Get-Item -LiteralPath $old).Length -eq $Orig.Size -and -not $env:ONECLICK_TEST_URL) { Move-Item -LiteralPath $old $s }
    if (-not ((Test-Path -LiteralPath $s) -and (Get-Item -LiteralPath $s).Length -eq $Orig.Size)) {
        if (Test-Path -LiteralPath $s) {
            Warn '原版文件大小不对（可能没下完），接着下载。'
            if (Test-Path -LiteralPath "$s.part") { Remove-Item -LiteralPath $s -Force } else { Move-Item -LiteralPath $s "$s.part" -Force }
        }
        Download-Model $Orig $s
    }
    Convert-Model $s $Model
}

# ---------------- 客户端接入信息：显示在窗口里，并写到 接入信息.txt ----------------
function Write-Info($c) {
  $url = 'http://' + $c.HOST + ':' + $c.PORT + '/v1'
  $t = @('NInfer 通用API接入信息', "Base URL: $url", "模型ID: $($c.MODEL_ID)", "上下文: $($c.CTX)", "最大输出: $($c.OUT)（API请求上界；真实输出由完整前缀和检索预算动态限制）", '支持OpenAI与Anthropic兼容API，框架自选；SYS为旧版兼容字段，不决定实际前缀。')
  [IO.File]::WriteAllLines((Join-Path $Here '接入信息.txt'),[string[]]$t,(New-Object Text.UTF8Encoding $true))
}

# ---------------- 主流程 ----------------
$Host.UI.RawUI.WindowTitle = 'ninfer Swift 1.5 懒人包'
if (-not (Test-Path -LiteralPath $Exe)) { Fail "找不到引擎：$Exe。请重新解压懒人包（解压到不含特殊字符的路径，例如 D:\ninfer-swift15）。" }
$dllCount = @(Get-ChildItem -LiteralPath (Split-Path $Exe -Parent) -Filter '*.dll' -ErrorAction SilentlyContinue).Count
if ($dllCount -lt 10) { Fail "engine 文件夹里缺少运行所需的 DLL 文件（当前只有 $dllCount 个 .dll）。请把整个懒人包完整解压，保持所有 .dll 和 ninfer-serve.exe 在同一目录，不要只单独复制 ninfer-serve.exe。" }

# 显卡检查（下载前就查，不合适的电脑不会白下 10 GB）
$gpu = $null
try { $gpu = & nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader,nounits 2>$null | Select-Object -First 1 } catch {}
if (-not $gpu) { Fail '没找到 NVIDIA 显卡或显卡驱动（nvidia-smi 运行失败）。请先装好 NVIDIA 驱动（580 以上）。' }
$g = $gpu -split ','
$script:GpuName = $g[0].Trim(); $script:GpuCap = $g[1].Trim(); $script:GpuTotal = [int]$g[2].Trim(); $drv = $g[3].Trim()
$cap = [double]::Parse($script:GpuCap, [Globalization.CultureInfo]::InvariantCulture)
if ($cap -lt 8.6 -or $cap -ge 9.0) { Fail "显卡是 $($script:GpuName)（算力 $($script:GpuCap)）。这个包的引擎是按算力 8.6 编译的，只能用在 RTX 30 系（8.6）和 RTX 40 系（8.9）上。" }
if ($script:GpuTotal -lt 11500) { Fail "显卡显存只有 $($script:GpuTotal) MiB。Swift 1.5 模型本身就要约 9 GB 显存，需要 12 GB 以上的卡（8 GB 卡请用 8G 懒人包）。" }
if ($cap -gt 8.6) { Write-Host "  [提醒] $($script:GpuName) 是 RTX 40 系（算力 $($script:GpuCap)），引擎能跑但没实测过；SM 数会自动识别。" -ForegroundColor Yellow }
$drvMajor = 0; [void][int]::TryParse(($drv -split '\.')[0], [ref]$drvMajor)
if ($drvMajor -gt 0 -and $drvMajor -lt 580) { Write-Host "  [提醒] 显卡驱动 $drv 可能太旧（引擎要 CUDA 13，建议 580 以上），启动失败的话请先升级驱动。" -ForegroundColor Yellow }
try { $ramGB = [Math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB) } catch { $ramGB = 0 }
if ($ramGB -gt 0 -and $ramGB -lt 30) { Write-Host "  [提醒] 内存只有约 $ramGB GB。KVMem 要把对话记录放内存（6–10 GB），引擎本身还要约 14 GB，建议 32 GB；内存小的话把总上下文调小（按 C）。" -ForegroundColor Yellow }

# 先处理已经在跑的引擎（否则读到的空闲显存不准）
$cfg = Load-Config
$Port = [int]$cfg.PORT
$busy = @(Get-CimInstance Win32_Process -Filter "Name like 'ninfer-serve%'")
$lis  = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if (($busy.Count -gt 0) -or ($lis.Count -gt 0)) {
  Write-Host ''
  Write-Host "  [!] 已经有一个模型在运行（或者端口 $Port 被占用）：" -ForegroundColor Yellow
  foreach ($b in $busy) { Write-Host ('      {0} pid={1}  启动于 {2}  {3}' -f $b.Name, $b.ProcessId, $b.CreationDate, $b.ExecutablePath) }
  if ($DryRun) { Write-Host '      （dryrun：不处理，下面的空闲显存不准）' }
  else {
    Write-Host '      两个同时开会因为显存/内存不够而启动失败。'
    Write-Host '   [K] 关掉正在运行的那个，再启动这个    [Q] 退出（继续用正在运行的那个）'
    $a = (Read-Answer '   请选择: ').ToLower()
    if ($a -ne 'k') { exit 0 }
    foreach ($b in $busy) {
      $par = Get-CimInstance Win32_Process -Filter "ProcessId=$($b.ParentProcessId)" -ErrorAction SilentlyContinue
      if ($par -and ($par.Name -match '^(cmd|powershell)\.exe$')) { Stop-Process -Id $par.ProcessId -Force -ErrorAction SilentlyContinue }
      Stop-Process -Id $b.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep 5
    if (@(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue).Count -gt 0) { Fail "端口 $Port 还被别的程序占着：请改 设置.ini 里的 PORT。" }
  }
}

$script:Free = Free-MiB
Use-Mtp $cfg
$pages = Est-Pages $cfg.KV
if (-not ($argv -contains 'last') -and (-not $DryRun -or $env:L8084_INPUT)) {
  while ($true) {
    $chk = Show-Summary $cfg $pages
    Write-Host ''
    if ($chk.Err.Count -eq 0) { Write-Host '   [回车] 用这套配置启动    [C] 一步步重新选    [R] 重新读空闲显存    [Q] 退出' }
    else { Write-Host '   配置有错误，请按 C 重新选（或 Q 退出）' -ForegroundColor Red }
    $a = (Read-Answer '   请选择: ').ToLower()
    if ($a -eq 'q') { exit 0 }
    if ($a -eq 'c') { try{$cfg = Run-Wizard $cfg; Use-Mtp $cfg}catch{Write-Host $_.Exception.Message -ForegroundColor Yellow};$pages=Est-Pages $cfg.KV;continue }
    if ($a -eq 'r') { $script:Free = Free-MiB; $pages = Est-Pages $cfg.KV; continue }
    if ($a -eq '' -and $chk.Err.Count -eq 0) { break }
  }
} else {
  $chk = Show-Summary $cfg $pages
  if ($chk.Err.Count -gt 0) { exit 1 }
}

# 这次启动用的配置：输出"自动"在这里按估算窗口定下来；之后显存不够自动缩窗口时，输出预留不再变
$pages=Est-Pages $cfg.KV
$check=Check-Config $cfg $pages
if($check.Err.Count -gt 0){$check.Err|ForEach-Object{Write-Host $_ -ForegroundColor Red};if(-not $DryRun){Return-CapacityMenu $PSCommandPath ([bool]$NoSync)};exit 1}
$run = [ordered]@{}; foreach ($k in $cfg.Keys) { $run[$k] = $cfg[$k] }
$run.OUT = "$(Out-Of $cfg $pages)"
$PageMin = 132 # bootstrap minimum, not recommended retrieval floor
if ($DryRun) {
  Write-Host ''; Write-Host '  [dryrun] 只显示，不启动、不下载：'
  if (-not (Test-Path -LiteralPath $Model)) { Write-Host "  （模型还没准备好：$Model）" -ForegroundColor Yellow }
  Write-Host ('  EXE  ' + $Exe)
  Write-Host ('  MODEL ' + $Model)
  $dp = [Math]::Max($pages, $PageMin); $sp = Split-Window $run $dp
  Write-Host ('  ENV  NINFER_KVMEM_SINK_PAGES={0} NINFER_KVMEM_GEN_RESERVE_PAGES={1} NINFER_KVMEM_LONG_REUSE=1 NINFER_KVMEM_AUTO_ALLOCATION=1' -f $sp[0], (Gen-Pages $run $dp))
  Write-Host ('  ARGS ' + ((Build-Args $run $dp) -join ' '))
  exit 0
}
Save-Config $cfg
Ensure-Model
if ($pages -lt $PageMin) { Fail "显存预算不足，不强行扩大驻留。请降低逻辑窗口或释放显存。" }
Write-Info $run

Get-ChildItem env: | Where-Object { $_.Name -like 'NINFER_*' -and $_.Name -ne 'NINFER_FREE_VRAM_MIB' } | ForEach-Object { Remove-Item "env:$($_.Name)" }
$env:PATH = (Split-Path $Exe -Parent) + ';' + $env:PATH
$ErrorActionPreference = 'Continue'
$logDir = Join-Path $Here 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
Get-ChildItem $logDir -Filter 'swift15-*.log' | Sort-Object LastWriteTime -Descending | Select-Object -Skip 9 | Remove-Item -Force -ErrorAction SilentlyContinue
$logFile = Join-Path $logDir ('swift15-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
$logW = New-Object IO.StreamWriter($logFile, $true, (New-Object Text.UTF8Encoding $false)); $logW.AutoFlush = $true

$fit = 0; $crash = 0; $alignRetry = $false
while ($true) {
  $argsNow = Build-Args $run $pages
  if(-not (Confirm-CapacityStart $argsNow)){Return-CapacityMenu $PSCommandPath ([bool]$NoSync);exit 1}
  Set-KvmEnv $run $pages
  $Host.UI.RawUI.WindowTitle = "ninfer Swift 1.5 - port $($run.PORT) KVMem $($run.KV) head-$($run.HEAD) ctx $($run.CTX) out $($run.OUT) win $($pages * 64)"
  Write-Host ''
  Write-Host "  地址  : http://127.0.0.1:$($run.PORT)/v1    模型名: $($run.MODEL_ID)    （这个窗口不要关，关了模型就停了；出现 listening 就能用了）"
  Write-Host ('  命令  : engine\ninfer-serve.exe ' + (($argsNow | Select-Object -Skip 1) -join ' '))
  Write-Host "  日志  : $logFile"
  Write-Host ''
  $logW.WriteLine('# ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' env sink=' + $env:NINFER_KVMEM_SINK_PAGES + ' reserve=' + $env:NINFER_KVMEM_GEN_RESERVE_PAGES + ' longreuse=' + $env:NINFER_KVMEM_LONG_REUSE + ' ' + $Exe + ' ' + ($argsNow -join ' '))
  $script:short = $null
  $attempt = @{ Ready = $false }
  $script:curveMismatch = $false
  $sw = [Diagnostics.Stopwatch]::StartNew()
  & $Exe @($argsNow) 2>&1 | ForEach-Object {
    $line = "$_"; Capture-CapacityFailure $line; Write-Host $line; $logW.WriteLine($line)
    if ($line -match '\blistening on https?://') { $attempt.Ready = $true }
    if ($line -match 'reservation requires (\d+) bytes, but only (\d+) bytes are available') { $script:short = @([double]$matches[1], [double]$matches[2]) }
    if ($line.Contains('Main KV page count is outside the target capacity curve')) { $script:curveMismatch = $true }
  }
  $rc = $LASTEXITCODE; $secs = [int]$sw.Elapsed.TotalSeconds
  $logW.WriteLine("# engine exited rc=$rc after $secs s")
  Write-Host ''
  Write-Host "  [引擎已退出] 返回码=$rc  运行了 $secs 秒"
  if ($rc -eq 0) { break }
  if($script:CapacityFailureLine){break}
  if(-not $attempt.Ready -and $script:short -and $cfg.RESIDENT -notin @('auto','recommended')){Write-Host '手动驻留未改动；显存不足，请按C自行调整。' -ForegroundColor Yellow;break}
  if (-not $attempt.Ready -and $script:curveMismatch -and -not $alignRetry) {
    # 旧引擎按请求分段分配 KV，但某些显卡会把实际分段调大；关闭对齐后两边一致。
    $alignRetry = $true
    $env:NINFER_PREFILL_ALIGN = '0'
    $logW.WriteLine('# capacity curve mismatch: retry with NINFER_PREFILL_ALIGN=0')
    Write-Host '  检测到读入分段与 KV 页数不一致，按指定分段重新启动一次 ...' -ForegroundColor Yellow
    continue
  }
  if (-not $attempt.Ready -and $script:short -and ($fit -lt 2)) {
    # 显存不够：引擎报了真实可用量，按它重算窗口（如果输出选"自动"且旧输出放不下，也跟着重算输出预留）
    $availMiB = $script:short[1] / 1MB
    $next = Pages-From $cfg.KV $availMiB
    if ($next -ge $pages) { $next = $pages - 16 }
    $fit++
    if ("$($cfg.OUT)" -eq 'auto') {
      $run.OUT = "$(Out-Of $cfg $next)"
      $PageMin = 132 # bootstrap minimum, not recommended retrieval floor
    }
    if ($next -lt $PageMin) { Write-Host "  显存不够（可用 $([int]$availMiB) MiB），连最小窗口 $PageMin 页都放不下。把单次输出调小（按 C 重新选），或关掉占显存的程序（游戏、浏览器视频等）再启动。" -ForegroundColor Red; break }
    Write-Host ("  显存比预计少（引擎可用 {0} MiB），窗口 {1} 页 → {2} 页，自动重试 ..." -f [int]$availMiB, $pages, $next) -ForegroundColor Yellow
    $pages = $next
    if ([int]$run.THINK -gt [int]$run.OUT - 256) { $run.THINK = "$([Math]::Max(0, [int]$run.OUT - 256))" }
    Write-Info $run
    continue
  }
  if (-not $attempt.Ready) {
    Write-Host '  启动阶段就失败了。常见原因：' -ForegroundColor Yellow
    Write-Host '    - out of memory / cudaMalloc：显存不够 → 关掉占显存的程序，或把单次输出调小'
    Write-Host '    - bad allocation / cudaMallocHost：内存不够 → 关掉占内存的程序，或把总上下文调小（内存预留跟着变小）'
    break
  }
  $crash++
  if ($crash -ge 3) { break }
  Write-Host '  运行中意外退出，15 秒后自动重启 ...'; Start-Sleep 15
}
$logW.Close()
if($script:CapacityFailureLine){Register-CapacityFailure;Return-CapacityMenu $PSCommandPath ([bool]$NoSync)}
exit $rc
