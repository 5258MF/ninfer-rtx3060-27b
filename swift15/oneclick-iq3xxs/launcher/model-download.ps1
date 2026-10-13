# First-run download and verified MTP-Q4 conversion; no Python required.
function Say([string]$s,[string]$color='Gray'){Write-Host $s -ForegroundColor $color}
function Warn([string]$s){Write-Host $s -ForegroundColor Yellow}
function Fail([string]$s){throw $s}
function GiB([long]$n){return [Math]::Round($n/1GB,2)}
$Q4=@{Name='Swift 1.5 IQ3XXS MTP-Q4';File='swift15_iq3xxs_mtpq4.ninfer';Size=[long]10990447360;Sha='f9f3abc3486642b235a35ed7f972e606076e93bcc793092024afb116a617c5bd'}
$Orig=@{Name='Swift 1.5 IQ3XXS';File='swift15_iq3xxs_mtp.ninfer';Size=[long]11092478720;Sha='2c42cf6f9e7c52866f72add4b6985ea9acac8c180f9ebaf9f18a7e7ccd09c211';Page='https://huggingface.co/fyb1214/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer';Url='https://huggingface.co/fyb1214/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer/resolve/main/swift15_iq3xxs_mtp.ninfer'}
$PatchFiles=@{ 'ops.txt'=[long]5799; 'lit.bin'=[long]203691823 }
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
    Say "    来源  ：Hugging Face $($m.Page)（文件 $($m.File)）"
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
    Say '  ── 用补丁把 MTP 草稿头转成 Q4（只转换 MTP 草稿头）──' 'Cyan'
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

function Ensure-IQ3Model {
    if(Test-Path -LiteralPath $Model){
        if((Get-Item -LiteralPath $Model).Length -eq $Q4.Size){return}
        throw 'IQ3XXS MTP-Q4 模型大小不匹配，请移走不完整文件后重试。'
    }
    if($env:NINFER_IQ3_MODEL){throw 'NINFER_IQ3_MODEL 指向的模型不存在，请修改路径或取消该环境变量。'}
    $src=Join-Path $Here ('model/'+$Orig.File)
    if(Test-Path -LiteralPath $src){
        if((Get-Item -LiteralPath $src).Length -ne $Orig.Size -or (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Orig.Sha){throw '原始 IQ3XXS 模型校验失败，请移走该文件后重试。'}
    }else{Download-Model $Orig $src}
    Convert-Model $src $Model
}
