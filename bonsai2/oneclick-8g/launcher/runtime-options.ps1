# Shared launcher options. CPU vision is intentional; no GPU encoding fallback.
function Initialize-NinferRuntimeOptions([System.Collections.IDictionary]$Config) {
    foreach ($pair in @(@('CUDA_GRAPH','1'), @('VISION_DEVICE','cpu'), @('CPU_THREADS','auto'), @('VISION','1'))) {
        if (-not $Config.Contains($pair[0]) -or [string]::IsNullOrWhiteSpace([string]$Config[$pair[0]])) { $Config[$pair[0]] = $pair[1] }
    }
    if ([string]$Config['CUDA_GRAPH'] -notin @('0','1')) { throw 'CUDA_GRAPH 必须是 0（关闭）或 1（开启）' }
    if ([string]$Config['VISION_DEVICE'] -ne 'cpu') { throw '此版本仅支持 VISION_DEVICE=cpu；不使用 GPU 视觉编码或 overlay 回退' }
    if ([string]$Config['VISION'] -eq '2') { $Config['VISION'] = '1' }
    if ([string]$Config['VISION'] -notin @('0','1')) { throw 'VISION 必须是 0（关闭看图）或 1（CPU看图）' }
    $t = ([string]$Config['CPU_THREADS']).Trim().ToLowerInvariant()
    $n = 0
    if ($t -ne 'auto' -and (-not [int]::TryParse($t,[ref]$n) -or $n -lt 1 -or $n -gt 1024)) { throw 'CPU_THREADS 必须是 auto 或 1–1024 的整数；实际可用线程由后端约束' }
    $Config['CPU_THREADS'] = $t
}
function Show-NinferRuntimeOptions([System.Collections.IDictionary]$Config) {
    Initialize-NinferRuntimeOptions $Config
    Write-Host ('  │ CUDA Graph: ' + $(if ($Config['CUDA_GRAPH'] -eq '0') { '关闭' } else { '开启' }))
    Write-Host ('  │ 视觉后端  : CPU 编码 / 内存权重；线程=' + $Config['CPU_THREADS'] + '；缺库仅回退CPU')
    Write-Host '  │ 容量提示  : Graph 开关影响显存和速度；实际窗口以启动日志为准'
}
function Edit-NinferRuntimeOptions([System.Collections.IDictionary]$Config, [switch]$AskVision) {
    Initialize-NinferRuntimeOptions $Config
    Write-Host ''
    Write-Host '  ── 运行选项：CUDA Graph 与 CPU 视觉 ──' -ForegroundColor Cyan
    while ($true) {
        $x = Read-Answer ('  CUDA Graph：1 开启 / 0 关闭（回车保持 ' + $Config['CUDA_GRAPH'] + '）: ')
        if ($x -eq '') { break }
        if ($x -in @('0','1')) { $Config['CUDA_GRAPH'] = $x; break }
        Write-Host '  请输入 0 或 1' -ForegroundColor Yellow
    }
    Write-Host '  视觉编码固定使用CPU；语言模型、KV和最终图像特征仍占显存。'
    if ($AskVision) {
        while ($true) {
            $x = Read-Answer ('  看图：1 CPU看图 / 0 关闭（回车保持 ' + $Config['VISION'] + '）: ')
            if ($x -eq '') { break }
            if ($x -in @('0','1')) { $Config['VISION'] = $x; break }
        }
    }
    while ($true) {
        $x = Read-Answer ('  CPU视觉线程：auto 或正整数（回车保持 ' + $Config['CPU_THREADS'] + '）: ')
        if ($x -eq '') { break }
        $old = $Config['CPU_THREADS']; $Config['CPU_THREADS'] = $x
        try { Initialize-NinferRuntimeOptions $Config; break } catch { $Config['CPU_THREADS'] = $old; Write-Host $_.Exception.Message -ForegroundColor Yellow }
    }
}
function Complete-NinferRuntimeArgs([System.Collections.IDictionary]$Config, [object[]]$Arguments, [switch]$Swift) {
    Initialize-NinferRuntimeOptions $Config
    $a = @($Arguments)
    if ($Config['VISION'] -eq '0' -and $a -contains '--vision') { throw 'VISION=0 与 EXTRA 的 --vision 冲突；请在配置中启用CPU看图' }
    if ($Config['CUDA_GRAPH'] -eq '1' -and $a -contains '--no-cuda-graph') { throw 'EXTRA 的 --no-cuda-graph 与 CUDA_GRAPH=1 冲突；请使用 CUDA_GRAPH=0' }
    if ($Config['CUDA_GRAPH'] -eq '0') {
        $clean = @()
        for ($i=0; $i -lt $a.Count; $i++) {
            if ($a[$i] -eq '--cuda-graph-allowance-mib') { $i++; continue }
            if ([string]$a[$i] -like '--cuda-graph-allowance-mib=*') { continue }
            if ($a[$i] -ne '--no-cuda-graph') { $clean += $a[$i] }
        }
        $a = @($clean) + @('--no-cuda-graph')
    } elseif ($Swift -and $a -notcontains '--cuda-graph-allowance-mib') { $a += @('--cuda-graph-allowance-mib','144') }
    for ($i=0; $i -lt $a.Count; $i++) {
        if ($a[$i] -eq '--vision-residency' -and ($i+1 -ge $a.Count -or $a[$i+1] -ne 'cpu')) { throw '视觉驻留参数必须是 cpu；不允许 GPU/overlay' }
        if ([string]$a[$i] -like '--vision-residency=*' -and $a[$i] -ne '--vision-residency=cpu') { throw '视觉驻留参数必须是 cpu' }
    }
    $env:NINFER_VISION_CPU = '1'
    $env:NINFER_CPU_THREADS = [string]$Config['CPU_THREADS']
    return $a
}
