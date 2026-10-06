param([ValidateSet('menu','last','dryrun','wizard-dryrun')][string]$Mode='menu')

$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'capacity-options.ps1')

. (Join-Path $PSScriptRoot 'runtime-options.ps1')

$Here=Split-Path $PSScriptRoot -Parent

$ModelsRoot=Split-Path $Here -Parent

$CfgFile=Join-Path $Here 'settings.ini'

$PolicyFile=Join-Path $Here 'capacity-policy.json'
if($env:NINFER_IQ3_POLICY){$PolicyFile=$env:NINFER_IQ3_POLICY}

$QueueFile=$env:NINFER_IQ3_QUEUE

$Exe=Join-Path $Here 'engine/ninfer-serve.exe'
if($env:NINFER_IQ3_ENGINE){$Exe=$env:NINFER_IQ3_ENGINE}

$Model=Join-Path $Here 'model/swift15_iq3xxs_mtpq4.ninfer'
if($env:NINFER_IQ3_MODEL){$Model=$env:NINFER_IQ3_MODEL}

$ExpectedEngine='fe444757618e651c826f2edbeb5c7dccafb678983a4e57b63281cc32732e3f3b'
if($env:NINFER_IQ3_ENGINE_SHA256){$ExpectedEngine=$env:NINFER_IQ3_ENGINE_SHA256}

$TestInput=New-Object System.Collections.Queue
if($env:L8084_INPUT){$env:L8084_INPUT.Split('|')|ForEach-Object{$TestInput.Enqueue($_)}}
function Read-Answer([string]$Prompt){if($TestInput.Count -gt 0){$x=[string]$TestInput.Dequeue();Write-Host ($Prompt+': '+$x);return $x.Trim()};if($env:L8084_INPUT){throw 'test input exhausted'};return (Read-Host $Prompt).Trim()}

function Load-Config {

 $c=[ordered]@{KV='rk4v4';CUDA_GRAPH='0';VISION='1';VISION_DEVICE='cpu';CPU_THREADS='auto';MARGIN_MIB='300';PORT='8084';RESIDENT='recommended';CTX='131072';OUT='auto';THINK='0'}

 if(Test-Path $CfgFile){foreach($line in Get-Content $CfgFile -Encoding UTF8){if($line -match '^\s*([A-Z_]+)\s*=\s*(.*?)\s*$' -and $c.Contains($matches[1])){$c[$matches[1]]=$matches[2]}}}

 Assert-Config $c

 return $c

}

function Assert-Config($c) {

 Initialize-NinferRuntimeOptions $c

 if($c.KV -notin @('rk4v4','rk8v4')){throw 'KV 只能选 rk4v4 或 rk8v4'}


 $port=0;if(-not [int]::TryParse([string]$c.PORT,[ref]$port) -or $port -lt 1024 -or $port -gt 65535){throw 'PORT须为1024–65535'}

}

function Save-Config($c) {

 Assert-Config $c

 $lines=@('; IQ3XXS local development launcher; no GPU power settings.')

 foreach($k in $c.Keys){$lines+=($k+'='+$c[$k])}

 [IO.File]::WriteAllLines($CfgFile,$lines,(New-Object Text.UTF8Encoding $true))

}

function New-Plan($c,$policy) {

 Assert-Config $c

 $tag='iq3-'+$(if($c.KV -eq 'rk4v4'){'rk4'}else{'rk8'})+$(if($c.CUDA_GRAPH -eq '1'){'-graph'}else{'-nograph'})

 $info=Get-CapacityInfo 'iq3' $c.KV $c.CUDA_GRAPH
 $ctx=Convert-CapacityToken ([string]$c.CTX)
 if($ctx -lt 16384 -or $ctx -gt 262144 -or $ctx%64 -ne 0){throw 'CTX必须为16K–256K，64对齐'}
 $cap=Resolve-Capacity ([string]$c.RESIDENT) $info $ctx
 $hostMiB=10240
 $api=$(if($c.OUT -eq 'auto'){$cap}else{Convert-CapacityToken ([string]$c.OUT)})
 if($api -lt 1 -or $api -gt $ctx){throw 'OUT必须为auto或1–CTX'}
 $think=Convert-CapacityToken ([string]$c.THINK)
 if($think -lt 0 -or $think -ge $api){throw 'THINK必须为0或小于API回答上限'}
 $approved=($c.KV -eq 'rk4v4' -and $c.CUDA_GRAPH -eq '0' -and $cap -eq 33792 -and $ctx -eq 131072)
 $status=$(if($approved){'推荐配置；历史短测通过'}else{'用户自选配置；未作为此组合重新验收'})
 $a=@($Model,'--host','127.0.0.1','--port',[string]$c.PORT,'--model-id','swift15-iq3xxs','--max-concurrency','1','--kv-dtype',[string]$c.KV,'--spec','mtp','--draft-tokens','3','--max-context',"$ctx",'--default-max-tokens',[string]$api,'--prefill-chunk','512','--device-state-slots','0','--kv-capacity','auto','--kvmem-window-pages',[string]($cap/64),'--host-kv-mib',"$hostMiB",'--embedding-host','--gdn-state-fp16')

 if($think -gt 0){$a+=@('--default-thinking-budget',[string]$think)}
 if($c.VISION -eq '1'){$a+=@('--vision','--vision-residency','cpu','--vision-max-merged','4096')}

 $a=@(Complete-NinferRuntimeArgs -Config $c -Arguments $a -Swift)

 $e=[ordered]@{NINFER_TERNARY_EMBED_HOST='1';NINFER_TERNARY_VISION_HOST='1';NINFER_VISION_CPU='1';NINFER_CPU_THREADS=[string]$c.CPU_THREADS;NINFER_KVMEM_SINK_PAGES='2';NINFER_KVMEM_GEN_RESERVE_PAGES='0';NINFER_KVMEM_LONG_REUSE='1';NINFER_KVMEM_AUTO_ALLOCATION='1'}

 return [pscustomobject]@{Case=$tag;Approved=$approved;ValidationStatus=$status;CapacityChoice=$c.RESIDENT;CapacityInfo=$info;ResidentTokens=$cap;FixedTokens=$null;OutputTokens=$api;RetrievalTokens=([int]([Math]::Floor($ctx*9/64/64)*64));AllocationMode="automatic-complete-prefix";ContextTokens=$ctx;Executable=$Exe;Arguments=$a;Environment=$e}

}

function Read-Policy { return [pscustomobject]@{status='idle'} }

function Show-Plan($c,$p){
 Write-Host '`nSwift1.5 IQ3XXS — 推荐值不是上限；C自定义 / R刷新显存' -ForegroundColor Cyan
 Show-NinferRuntimeOptions $c;Show-CapacityInfo $p.CapacityInfo
 Write-Host ('  KV='+$c.KV+'；选择='+$c.RESIDENT+'；驻留='+$p.ResidentTokens+'；逻辑上下文='+$p.ContextTokens+'；API上限='+$p.OutputTokens+'；端口='+$c.PORT)
 Write-Host ('  '+$p.ValidationStatus+'；实际完整前缀/检索/输出由引擎逐请求动态分配。')
 if($p.ResidentTokens -lt $p.RetrievalTokens+16384){Write-Host '  大系统/工具前缀可能压缩检索预算；这是建议，不是固定分区。' -ForegroundColor Yellow}
 if($p.CapacityInfo.Estimated -ge 0 -and $p.ResidentTokens -gt $p.CapacityInfo.Estimated){Write-Host '  超过当前显存估算，可能OOM；手动选择保持不变。' -ForegroundColor Yellow}
}

function Edit-Config($c){
 $c.KV=Ask-CapacityChoice '1. KV（推荐rk4v4）' $c.KV @('rk4v4','rk8v4')
 Edit-NinferRuntimeOptions $c -AskVision
 $info=Get-CapacityInfo 'iq3' $c.KV $c.CUDA_GRAPH
 $n=Edit-CapacityBudget $c 'RESIDENT' $info
 $c.CTX=Ask-CapacityValue '逻辑上下文：推荐128k；64k/128k/200k/256k，不小于驻留' $c.CTX ([Math]::Max(16384,$n)) 262144 @() 64
 $c.OUT=Ask-CapacityValue 'API回答上限：推荐auto；8k/32k/64k（不是物理分区）' $c.OUT 1 ([int]$c.CTX) @('auto')
 $o=$(if($c.OUT -eq 'auto'){$n}else{[int]$c.OUT})
 $c.THINK=Ask-CapacityValue '思考上限：0不限（推荐）' $c.THINK 0 ($o-1)
 $c.PORT=Ask-CapacityValue '端口' $c.PORT 1024 65535
 Assert-Config $c
}

function Assert-Idle($policy){

 if($policy.status -eq 'running'){throw '容量测试仍在运行；开发版禁止并行加载GPU。请先用预览入口。'}

 if($QueueFile -and (Test-Path $QueueFile)){$q=Get-Content $QueueFile -Raw -Encoding UTF8|ConvertFrom-Json;if($q.status -in @('running','waiting_for_capacity')){throw '自动测速队列尚未结束；不启动第二个GPU任务。'}}

 if(@(Get-Process ninfer* -ErrorAction SilentlyContinue).Count -gt 0){throw '已有推理进程；本启动器不会结束它，也不并行加载模型。'}

}

$c=Load-Config;$policy=Read-Policy;$plan=$null
try{$plan=New-Plan $c $policy}catch{if($Mode -in @('menu','wizard-dryrun')){Write-Host $_.Exception.Message -ForegroundColor Yellow}else{throw}}

if($Mode -eq 'wizard-dryrun'){Edit-Config $c;$plan=New-Plan $c $policy;$plan|ConvertTo-Json -Depth 8;exit 0}
if($Mode -eq 'dryrun'){$plan|ConvertTo-Json -Depth 8;exit 0}

while($Mode -eq 'menu'){

 if($plan){Show-Plan $c $plan}else{Write-Host '当前配置无效，请按C修改或Q退出。'}

 $choice=Read-Answer '回车按当前推荐/已保存配置启动 / C自定义 / R刷新显存 / S保存不启动 / Q退出'

 if($choice -ieq 'q'){exit 0}
 if($choice -ieq 'r'){try{$plan=New-Plan $c $policy}catch{$plan=$null;Write-Host $_.Exception.Message -ForegroundColor Yellow};continue}

 if($choice -ieq 'c'){try{Edit-Config $c;$plan=New-Plan $c $policy}catch{Write-Host $_.Exception.Message -ForegroundColor Yellow;$plan=$null};continue}

 if($choice -ieq 's'){Save-Config $c;Write-Host '已保存配置，没有启动GPU任务。';continue}

 if($choice -eq '' -and $plan){break}

}

$policy=Read-Policy;try{$plan=New-Plan $c $policy}catch{Write-Host $_.Exception.Message -ForegroundColor Yellow;Return-CapacityMenu $PSCommandPath;exit 1}


Assert-Idle $policy

if(-not(Test-Path $Exe) -or -not(Test-Path $Model)){throw '缺少本地Swift引擎或IQ3XXS MTP-Q4模型；开发版不自动下载/转换'}

if((Get-FileHash $Exe -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedEngine){throw 'Swift引擎哈希不匹配；拒绝使用其他引擎'}

if((Get-Item $Model).Length -ne 10990447360){throw 'IQ3XXS MTP-Q4模型大小不匹配；此处未做全文件哈希验证'}

if(-not (Confirm-CapacityStart $plan.Arguments)){Return-CapacityMenu $PSCommandPath;exit 1}
Save-Config $c

Get-ChildItem env:|Where-Object{$_.Name -like 'NINFER_*'}|ForEach-Object{Remove-Item ('env:'+$_.Name)}

foreach($k in $plan.Environment.Keys){Set-Item ('env:'+$k) ([string]$plan.Environment[$k])}

Push-Location (Split-Path $Exe -Parent)

$oldE=$ErrorActionPreference;$ErrorActionPreference='Continue'
try{& $Exe @($plan.Arguments) 2>&1 | ForEach-Object {$line=[string]$_;Write-Host $line;Capture-CapacityFailure $line};$exitCode=$LASTEXITCODE}finally{Pop-Location;$ErrorActionPreference=$oldE}
if($script:CapacityFailureLine){Register-CapacityFailure;Return-CapacityMenu $PSCommandPath}

exit $exitCode
