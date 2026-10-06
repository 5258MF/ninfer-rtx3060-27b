$ErrorActionPreference='Stop';
$RepoRoot=Split-Path $PSScriptRoot -Parent
$env:NINFER_LAUNCHER_TEST='1';Remove-Item env:ONECLICK_FREE_MIB -ErrorAction SilentlyContinue;$env:NINFER_TERNARY_KVMEM_CPU_RETRIEVAL='1'
$script:MockTotal=12288;$script:MockFree=11360;$script:MockDevice='NVIDIA GeForce RTX 3060';$script:Unknown=$false
function global:nvidia-smi {if($script:Unknown){throw 'unavailable'};return "$script:MockTotal,$script:MockFree,$script:MockDevice"}
$checks=0
foreach($relativePath in @('swift15/oneclick-iq2s/launcher/capacity-options.ps1','swift15/oneclick-iq3xxs/launcher/capacity-options.ps1','bonsai2/oneclick-8g/launcher/capacity-options.ps1','bonsai2/oneclick-12g/launcher/capacity-options.ps1')){
 $path=Join-Path $RepoRoot $relativePath
 . $path
 foreach($case in @(@('iq3','rk4v4','0',48320),@('swift15','rk4v4','1',83968),@('12g','int8','1',107584),@('8g','rk4v4','0',69504))){
  $i=Get-CapacityInfo $case[0] $case[1] $case[2]
  if($i.Max -ne $case[3] -or $i.Calibration -eq 'legacy'){throw ('wrong calibrated bound '+$case[0])};$checks++
  if((Resolve-Capacity 'auto' $i 262144) -ne $i.Max){throw 'auto mismatch'};$checks++
  if((Resolve-Capacity 'recommended' $i 262144) -ne $i.CurrentRecommended){throw 'recommended mismatch'};$checks++
  Confirm-ResidentBound $i $i.Max;$checks++
  $bad=$false;try{Confirm-ResidentBound $i ($i.Max+64)}catch{$bad=$true};if(-not $bad){throw 'over-bound permitted'};$checks++
  $script:MockFree=9000;$low=Get-CapacityInfo $case[0] $case[1] $case[2];if($low.Max -ge $i.Max){throw 'free VRAM ignored'};$script:MockFree=11360;$checks++
 }
 $i=Get-CapacityInfo 'iq3' 'rk4v4' '0';Confirm-ResidentBound $i 45056;$checks++
 $i=Get-CapacityInfo '8g' 'rk4v4' '0';if($i.FreeMiB -ne 7264){throw 'whole-card 8G accounting wrong'};$checks++
 foreach($case in @(@('iq3','rk8v4','0','full','q4'),@('swift15','rk4v4','1','lite','q4'),@('12g','rk8v4','1','full','q4'))){$i=Get-CapacityInfo $case[0] $case[1] $case[2] $case[3] $case[4];if($i.Calibration -ne 'legacy'){throw 'untested combination mislabeled'};$checks++}
 $script:MockDevice='Other GPU';if((Get-CapacityInfo 'iq3' 'rk4v4' '0').Calibration -ne 'legacy'){throw 'calibration applied to other GPU'};$script:MockDevice='NVIDIA GeForce RTX 3060';$checks++
 $script:Unknown=$true;$i=Get-CapacityInfo 'iq3' 'rk4v4' '0';if($i.Max -ne 0 -or $i.FreeMiB -ge 0){throw 'unknown VRAM not fail-closed'};$script:Unknown=$false;$checks++
}


function Assert($ok,$msg){if(-not $ok){throw $msg}}
$script:q=New-Object System.Collections.Queue
function Read-Answer($prompt){if($q.Count -eq 0){throw 'input exhausted'};return [string]$q.Dequeue()}
$q.Enqueue('1');Assert ((Ask-CapacityChoice 'KV' 'rk8v4' @('rk4v4','rk8v4')) -eq 'rk4v4') 'screenshot 1 regression'
$q.Enqueue('2');Assert ((Ask-CapacityChoice 'KV' 'rk4v4' @('rk4v4','rk8v4')) -eq 'rk8v4') 'KV2'
$q.Enqueue('0');Assert ((Ask-CapacityChoice 'KV' 'rk8v4' @('rk4v4','rk8v4')) -eq 'rk8v4') '0 keep'
$q.Enqueue('');Assert ((Ask-CapacityChoice 'KV' 'rk8v4' @('rk4v4','rk8v4')) -eq 'rk8v4') 'enter'
$q.Enqueue('RK8V4');Assert ((Ask-CapacityChoice 'KV' 'rk4v4' @('rk4v4','rk8v4')) -eq 'rk8v4') 'name compatibility'
$q.Enqueue('0');Assert ((Ask-CapacityChoice 'KVMem' '1' @('1','0')) -eq '0') '0 off changed'
$q.Enqueue('1');Assert ((Ask-CapacityChoice 'KVMem' '0' @('1','0')) -eq '1') '1 on changed'
$q.Enqueue('9');$q.Enqueue('1');Assert ((Ask-CapacityChoice 'head' 'full' @('full','lite')) -eq 'full') 'invalid retry'
$q.Enqueue('1');Assert ((Ask-CapacityValue 'resident' 'recommended' 8448 20000 @('recommended','auto') 64) -eq 'recommended') 'resident1'
$q.Enqueue('2');Assert ((Ask-CapacityValue 'resident' 'recommended' 8448 20000 @('recommended','auto') 64) -eq 'auto') 'resident2'
$q.Enqueue('0');Assert ((Ask-CapacityValue 'API' 'auto' 1 131072 @('auto')) -eq 'auto') 'API0'
$q.Enqueue('1');Assert ((Ask-CapacityValue 'API' 'auto' 1 131072 @('auto')) -eq '1') 'API literal1 changed'
$q.Enqueue('50k');$q.Enqueue('16k');Assert ((Ask-CapacityValue 'resident' 'recommended' 8448 20000 @('recommended','auto') 64) -eq '16384') 'bounds bypassed'

$i=Get-CapacityInfo 'iq3' 'rk4v4' '0'
$null=Resolve-Capacity 'recommended' $i 131072
$cap=$script:CapacityActiveTokens
$argsOk=@('model.ninfer','--kv-capacity','auto','--kvmem-window-pages',[string]($cap/64),'--kv-dtype','rk4v4','--prefill-chunk','512','--max-concurrency','1','--no-cuda-graph')
Assert (Confirm-CapacityStart $argsOk) 'valid final argv refused'
Assert (-not (Confirm-CapacityStart ($argsOk+@('--kvmem-window-pages','800')))) 'EXTRA bypass'
$script:MockFree=10400
Assert (-not (Confirm-CapacityStart $argsOk)) 'prelaunch memory drop not caught'
$script:MockFree=11360
@{passed=$true;capacity_checks=$checks;menu_checks=13;preflight_checks=3;helpers=4}|ConvertTo-Json
