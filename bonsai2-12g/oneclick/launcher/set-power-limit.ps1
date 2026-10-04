param(
    [int]$Watts = 135,
    [switch]$Persist = $true,
    [switch]$Interactive = $false
)
$ErrorActionPreference = "Continue"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"", "-Watts", "$Watts")
    if ($Persist) { $argList += "-Persist" }
    if ($Interactive) { $argList += "-Interactive" }
    Start-Process -FilePath "powershell.exe" -ArgumentList $argList -Verb RunAs -Wait
    exit $LASTEXITCODE
}

if ($Interactive) {
    Write-Host ""
    Write-Host "  === RTX 3060 显卡功耗上限设置 ===" -ForegroundColor Cyan
    $cur = & nvidia-smi --query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit,temperature.gpu --format=csv,noheader 2>$null
    Write-Host "  当前状态（当前上限, 默认上限, 最低, 最高, 当前温度）：$cur"
    Write-Host ""
    Write-Host "  [1] 135W（推荐：长跑温度稳定在 78~82°C，风扇安静，吐字几乎不掉速）" -ForegroundColor Green
    Write-Host "  [2] 145W（折中：温度约 83~85°C）"
    Write-Host "  [3] 170W（恢复出厂默认，关闭开机自动限功耗）"
    Write-Host ""
    $ans = (Read-Host "  请选择 [默认 1]").Trim()
    if ($ans -eq "2") { $Watts = 145; $Persist = $true }
    elseif ($ans -eq "3") { $Watts = 170; $Persist = $false }
    else { $Watts = 135; $Persist = $true }
}

Write-Host "  正在将显卡功耗上限设置为 ${Watts}W ..." -ForegroundColor Cyan
& nvidia-smi -pl $Watts

$taskName = "NinferGpuPowerLimit"
if ($Persist -and $Watts -lt 170) {
    & schtasks.exe /Create /TN $taskName /TR "nvidia-smi.exe -pl $Watts" /SC ONSTART /RU SYSTEM /RL HIGHEST /F | Out-Null
    Write-Host "  [已设开机自动生效] 计划任务 $taskName 已更新为开机自动设置 ${Watts}W。" -ForegroundColor Green
} else {
    & schtasks.exe /Delete /TN $taskName /F 2>$null | Out-Null
    Write-Host "  [已取消开机限功耗] 已恢复默认 ${Watts}W。" -ForegroundColor Yellow
}

$after = & nvidia-smi --query-gpu=power.limit,temperature.gpu --format=csv,noheader 2>$null
Write-Host "  当前功耗上限与温度：$after" -ForegroundColor Green
if ($Interactive) {
    Write-Host ""
    Read-Host "  按回车键关闭窗口"
}
