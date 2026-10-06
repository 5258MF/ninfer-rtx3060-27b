param([ValidateSet('menu','last','dryrun')][string]$Mode='menu')

$ErrorActionPreference='Stop'

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

function Read-Answer([string]$Prompt) { return (Read-Host $Prompt).Trim() }

function Load-Config {

 $c=[ordered]@{KV='rk4v4';CUDA_GRAPH='0';VISION='1';VISION_DEVICE='cpu';CPU_THREADS='auto';MARGIN_MIB='300';PORT='8084'}

 if(Test-Path $CfgFile){foreach($line in Get-Content $CfgFile -Encoding UTF8){if($line -match '^\s*([A-Z_]+)\s*=\s*(.*?)\s*$' -and $c.Contains($matches[1])){$c[$matches[1]]=$matches[2]}}}

 Assert-Config $c

 return $c

}

function Assert-Config($c) {

 Initialize-NinferRuntimeOptions $c

 if($c.KV -notin @('rk4v4','rk8v4')){throw 'KV 只能选 rk4v4 或 rk8v4'}

 if($c.MARGIN_MIB -notin @('400','300')){throw 'MARGIN_MIB 只能选400或300'}

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

 $rows=@($policy.rows|Where-Object{$_.case -eq $tag -and $_.margin_mib -eq [int]$c.MARGIN_MIB})

 if($rows.Count -ne 1){throw ('容量记录缺失或重复：'+$tag+' / '+$c.MARGIN_MIB)}

 $row=$rows[0];$cap=[Math]::Min(([int]$row.capacity_tokens - 2048),33792)

 if($cap -le 10240 -or $cap%64 -ne 0){throw '容量记录无效；不采用估算窗口'}

 $ctx=131072;$hostMiB=10240

 if($c.KV -eq 'rk8v4'){$ctx=131072;$hostMiB=7168}

 if($cap -gt $ctx){throw '驻留窗口不能超过逻辑上下文'}

 $approved=($tag -eq 'iq3-rk4-nograph' -and $c.MARGIN_MIB -eq '300' -and $row.status -eq '验证通过' -and $row.retrieval_correct -eq $true -and $row.remaining_mib -ge [int]$c.MARGIN_MIB -and $row.fixed_tokens -eq 2048 -and $row.output_tokens -eq 8192 -and $row.budget_mib -eq 12288)

 $a=@($Model,'--host','127.0.0.1','--port',[string]$c.PORT,'--model-id','swift15-iq3xxs','--max-concurrency','1','--kv-dtype',[string]$c.KV,'--spec','mtp','--draft-tokens','3','--max-context',"$ctx",'--default-max-tokens','16384','--prefill-chunk','512','--device-state-slots','0','--kv-capacity','auto','--kvmem-window-pages',[string]($cap/64),'--host-kv-mib',"$hostMiB",'--embedding-host','--gdn-state-fp16')

 if($c.VISION -eq '1'){$a+=@('--vision','--vision-residency','cpu','--vision-max-merged','4096')}

 $a=@(Complete-NinferRuntimeArgs -Config $c -Arguments $a -Swift)

 $e=[ordered]@{NINFER_TERNARY_EMBED_HOST='1';NINFER_TERNARY_VISION_HOST='1';NINFER_VISION_CPU='1';NINFER_CPU_THREADS=[string]$c.CPU_THREADS;NINFER_KVMEM_SINK_PAGES='32';NINFER_KVMEM_GEN_RESERVE_PAGES='256';NINFER_KVMEM_LONG_REUSE='1';NINFER_KVMEM_AUTO_ALLOCATION='1'}

 return [pscustomobject]@{Case=$tag;Approved=$approved;ValidationStatus=$row.status;MarginMiB=[int]$c.MARGIN_MIB;ResidentTokens=$cap;FixedTokens=$null;OutputTokens=16384;RetrievalTokens=18432;AllocationMode="automatic-complete-prefix";ContextTokens=$ctx;Executable=$Exe;Arguments=$a;Environment=$e}

}

function Read-Policy {if(-not(Test-Path $PolicyFile)){throw '本机容量验收记录不存在；开发版不猜测容量，也不自动下载模型'};return (Get-Content $PolicyFile -Raw -Encoding UTF8|ConvertFrom-Json)}

function Show-Plan($c,$p){

 Write-Host "`nSwift1.5 IQ3XXS — 128K自动前缀档 / 精简短工具链已检查" -ForegroundColor Cyan

 Show-NinferRuntimeOptions $c

 Write-Host ('  KV='+$c.KV+'；余量档='+$p.MarginMiB+'MiB；端口='+$c.PORT)

 Write-Host ('  驻留='+$p.ResidentTokens+'；完整系统/工具前缀自动保留；API输出上限=16384；检索目标='+$p.RetrievalTokens+'；逻辑上下文='+$p.ContextTokens)

 Write-Host ('  当前容量记录：'+$p.ValidationStatus+'；允许按此记录启动='+$p.Approved)

 Write-Host '  MTP Q4 / draft3 / 完整输出头。实际sink/检索/输出以[kvmem-alloc]日志为准。精简工具链已回归；普通长链需配套DSH的128K压缩设置。16K长输出未验证。'

}

function Edit-Config($c){

 Edit-NinferRuntimeOptions $c -AskVision

 $x=Read-Answer ('KV：rk4v4 / rk8v4，回车保持 '+$c.KV);if($x){$c.KV=$x}

 $x=Read-Answer ('余量档：300当前默认 / 400保守档，回车保持 '+$c.MARGIN_MIB);if($x){$c.MARGIN_MIB=$x}

 $x=Read-Answer ('端口，回车保持 '+$c.PORT);if($x){$c.PORT=$x}

 Assert-Config $c

}

function Assert-Idle($policy){

 if($policy.status -eq 'running'){throw '容量测试仍在运行；开发版禁止并行加载GPU。请先用预览入口。'}

 if($QueueFile -and (Test-Path $QueueFile)){$q=Get-Content $QueueFile -Raw -Encoding UTF8|ConvertFrom-Json;if($q.status -in @('running','waiting_for_capacity')){throw '自动测速队列尚未结束；不启动第二个GPU任务。'}}

 if(@(Get-Process ninfer* -ErrorAction SilentlyContinue).Count -gt 0){throw '已有推理进程；本启动器不会结束它，也不并行加载模型。'}

}

$c=Load-Config;$policy=Read-Policy;$plan=New-Plan $c $policy

if($Mode -eq 'dryrun'){$plan|ConvertTo-Json -Depth 8;exit 0}

while($Mode -eq 'menu'){

 Show-Plan $c $plan

 $choice=Read-Answer '回车启动 / C修改 / S保存不启动 / Q退出'

 if($choice -ieq 'q'){exit 0}

 if($choice -ieq 'c'){Edit-Config $c;$plan=New-Plan $c $policy;continue}

 if($choice -ieq 's'){Save-Config $c;Write-Host '已保存配置，没有启动GPU任务。';continue}

 if($choice -eq ''){break}

}

$policy=Read-Policy;$plan=New-Plan $c $policy

if(-not $plan.Approved){throw ('此档容量尚未通过，不允许启动：'+$plan.ValidationStatus)}

Assert-Idle $policy

if(-not(Test-Path $Exe) -or -not(Test-Path $Model)){throw '缺少本地Swift引擎或IQ3XXS MTP-Q4模型；开发版不自动下载/转换'}

if((Get-FileHash $Exe -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedEngine){throw 'Swift引擎哈希不匹配；拒绝使用其他引擎'}

if((Get-Item $Model).Length -ne 10990447360){throw 'IQ3XXS MTP-Q4模型大小不匹配；此处未做全文件哈希验证'}

Save-Config $c

Get-ChildItem env:|Where-Object{$_.Name -like 'NINFER_*'}|ForEach-Object{Remove-Item ('env:'+$_.Name)}

foreach($k in $plan.Environment.Keys){Set-Item ('env:'+$k) ([string]$plan.Environment[$k])}

Push-Location (Split-Path $Exe -Parent)

try{& $Exe @($plan.Arguments);$exitCode=$LASTEXITCODE}finally{Pop-Location}

exit $exitCode
