# Current free VRAM is an enforced configuration bound, not a recommendation.

$script:CapacityStateFile=Join-Path $PSScriptRoot 'capacity-observations.json'

$script:CapacityFailureLine=$null;$script:CapacityActiveInfo=$null;$script:CapacityActiveTokens=0

function Get-CapacityInfo([string]$Profile,[string]$Kv,[string]$Graph='0',[string]$Head='full',[string]$Mtp='q4') {

 $total=-1.0;$free=-1.0;$device='';$nativeLimit=262144;$calibration='legacy'

 try {

  $a=((& nvidia-smi --query-gpu=memory.total,memory.free,name --format=csv,noheader,nounits 2>$null | Select-Object -First 1) -split ',')

  $total=[double]::Parse($a[0].Trim(),[Globalization.CultureInfo]::InvariantCulture);$free=[double]::Parse($a[1].Trim(),[Globalization.CultureInfo]::InvariantCulture)

  if($a.Count -ge 3){$device=$a[2].Trim()}
 if($Profile -eq '8g'){$free=[Math]::Max(0,[Math]::Min($total,8192)-($total-$free))}

 }catch{$free=-1}

 if($env:NINFER_LAUNCHER_TEST -eq '1' -and $env:ONECLICK_FREE_MIB){$free=[double]$env:ONECLICK_FREE_MIB}

 $per=0.01828125;$base=9990.0;$rec=73728

 switch($Profile){

  'iq3' {$base=10786.17;$rec=33792}

  'swift15' {if($Kv -eq 'rk8v4'){$rec=51200}}

  '12g' {$base=7793.75;$per=0.0342;$rec=88064;if($Kv -eq 'rk8v4'){$rec=114688}}

  '8g' {$base=6208.0;$per=0.0176;$rec=36864}

 }

 if($Kv -eq 'rk8v4'){$per=0.02646875;if($Profile -eq 'iq3'){$rec=24576}}

 $cpu=($env:NINFER_TERNARY_KVMEM_CPU_RETRIEVAL -ne '0')

 if(-not $cpu){if($Profile -eq '12g'){$base=8194};if($Profile -eq '8g'){$base=6612}}

 # Graph actual allocation was 157.5MiB in local IQ2 logs (nominal allowance144); round conservatively.

 if($Graph -eq '1'){$base+=160};if($Head -eq 'lite'){$base+=346};if($Mtp -eq 'q6'){$base+=95}

 $key=$Profile+'|'+$Kv+'|'+$Graph+'|'+$Head+'|'+$Mtp+'|cpu='+$cpu

 # Per-model calibration, RTX3060/12GiB only; unmatched combinations retain
 # their existing conservative bounds, never inherit another model's constant.
 # Fixed costs include native reservation rounding and measured prefill/vision
 # transients. WDDM CUDA and NVML budgets are checked as separate constraints.
 $calibrations=@{
  'iq3|rk4v4|0|full|q4|cpu=True'=@{Base=10272.0;Fixed=715.594658203125;Cuda=1600.0;Id='iq3-rk4-3060-20261006'}
  'swift15|rk4v4|1|full|q4|cpu=True'=@{Base=9624.0;Fixed=864.0;Cuda=2400.0;Id='iq2-rk4-graph-3060-20261006'}
  '12g|int8|1|full|q4|cpu=True'=@{Base=7664.0;Fixed=600.0;Cuda=4280.0;Id='bonsai12-int8-graph-3060-20261006'}
  '8g|rk4v4|0|full|q4|cpu=True'=@{Base=6024.0;Fixed=320.0;Cuda=1544.0;Id='bonsai8-rk4-nograph-3060-20261006'}
 }
 if($total -eq 12288 -and $device -match 'RTX 3060' -and $calibrations.ContainsKey($key)){
  $row=$calibrations[$key];$base=[double]$row.Base
  $nativeLimit=[int]([Math]::Floor(($row.Cuda-$row.Fixed)/$per/64)*64)
  $calibration=[string]$row.Id;$key+='|cal='+$calibration
 }
 if(Test-Path $script:CapacityStateFile){try{$rows=@(Get-Content $script:CapacityStateFile -Raw -Encoding UTF8|ConvertFrom-Json);foreach($r in $rows){if($r.Key -eq $key){$base=[Math]::Max($base,[double]$r.BaseMiB)}}}catch{Write-Host '显存校正记录不可读，使用保守基础模型。' -ForegroundColor Yellow}}

 $estimate=-1;$max=0

 if($free -ge 0){$estimate=[int]([Math]::Floor([Math]::Max(0,$free-$base)/$per/64)*64);$max=[Math]::Min($nativeLimit,[Math]::Min(262144,$estimate))}

 return [pscustomobject]@{Profile=$Profile;Kv=$Kv;Graph=$Graph;Head=$Head;Mtp=$Mtp;Key=$key;FreeMiB=$free;TotalMiB=$total;Min=8448;Max=$max;ParameterMax=262144;Recommended=$rec;CurrentRecommended=[Math]::Min($rec,$max);Estimated=$estimate;PerTokenMiB=$per;BaseMiB=$base;DeviceName=$device;NativeCalibrationMax=$nativeLimit;Calibration=$calibration}

}

function Show-CapacityInfo($i){

 if(-not $i){return}

 Write-Host ('  当前可用显存预算='+$i.FreeMiB+' MiB；历史推荐='+$i.Recommended+'；当前推荐='+$i.CurrentRecommended)

 Write-Host '  驻留总预算包含前缀/检索/生成空间；API输出上限不是独占的物理输出窗口。'
 if($i.Calibration -eq 'legacy'){Write-Host '  当前组合尚未采用本轮实测校准，保留原保守估算；不是已验证的硬件极限。'}
 if($i.Calibration -ne 'legacy'){Write-Host ('  CUDA校准容量边界='+$i.NativeCalibrationMax+'；当前上限同时受空闲显存约束；推荐档不是此上限。')}
 if($i.Max -lt $i.Min){Write-Host '  当前显存不足最小驻留，或无法读取显存；禁止加载。请释放显存、修改前面的选项或刷新。' -ForegroundColor Yellow;return}

 Write-Host ('  当前可选驻留范围：'+$i.Min+'–'+$i.Max+' token（64对齐，且不超过逻辑上下文）。超过范围必须重选，不再仅警告放行。')

 Write-Host '  上限按当前显存和运行开销保守估算，启动前复查；不是裸262144参数上限，也不是硬件精确极限。不设300MiB余量门槛。'

 if($i.Profile -eq '8g'){Write-Host '  整卡8GiB预算包含桌面和其他程序，12GB卡测试也按8GiB扣减。'}

}

function Confirm-ResidentBound($Info,[int]$Tokens){

 if($Info.FreeMiB -lt 0){throw '无法读取当前显存，不能确认可选上限；请刷新后重选，禁止绕过加载。'}

 if($Info.Max -lt $Info.Min){throw '当前显存不足最小驻留预算；请释放显存或调整KV/Graph/输出头后重选。'}

 if($Tokens -lt $Info.Min -or $Tokens -gt $Info.Max -or $Tokens%64 -ne 0){throw ('驻留 '+$Tokens+' 超出当前显存允许范围 '+$Info.Min+'–'+$Info.Max+'；请按C重选。')}

 $script:CapacityActiveInfo=$Info;$script:CapacityActiveTokens=$Tokens

}

function Resolve-Capacity([string]$Value,$Info,[int]$Context=262144){

 if([string]::IsNullOrWhiteSpace($Value)){$Value='recommended'};$Value=$Value.ToLowerInvariant()

 $top=[Math]::Min($Info.Max,$Context)

 if($Value -eq 'recommended'){$n=[Math]::Min($Info.Recommended,$top)}

 elseif($Value -eq 'auto'){$n=$top}else{$n=Convert-CapacityToken $Value}

 Confirm-ResidentBound $Info $n

 if($n -gt $Context){throw '驻留不得超过逻辑上下文，请重新选择。'}

 return [int]$n

}

function Edit-CapacityBudget($Config,[string]$Key,$Info){

 Show-CapacityInfo $Info

 if($Info.Max -lt $Info.Min){throw '没有可用的驻留范围；先调整前面的KV/Graph/输出头选项或释放显存，然后再按C。'}

 if(-not $Config.Contains($Key)){$Config[$Key]='recommended'}

 while($true){

  $x=Ask-CapacityValue '驻留总量：recommended当前推荐 / auto当前上限 / 范围内手动token数' ([string]$Config[$Key]) $Info.Min $Info.Max @('recommended','auto') 64

  try{$n=Resolve-Capacity $x $Info;$Config[$Key]=$x;return $n}catch{Write-Host $_.Exception.Message -ForegroundColor Yellow}

 }

}

function Confirm-CapacityStart([object[]]$Arguments=@()) {

 if(-not $script:CapacityActiveInfo){Write-Host '缺少显存预算检查，拒绝加载。' -ForegroundColor Red;return $false}

 $i=$script:CapacityActiveInfo

 if($Arguments.Count){

  $v=@{}

  for($a=0;$a -lt $Arguments.Count;$a++){

   $word=[string]$Arguments[$a]

   if($word -match '^(--[^=]+)=(.*)$'){$v[$matches[1]]=$matches[2]}

   elseif($word -like '--*'){

    if($a+1 -lt $Arguments.Count -and [string]$Arguments[$a+1] -notlike '--*'){$v[$word]=[string]$Arguments[$a+1];$a++}else{$v[$word]=$true}

   }

  }

  $physical=$script:CapacityActiveTokens

  if($v.ContainsKey('--kvmem-window-pages')){$physical=64*[int]$v['--kvmem-window-pages']}

  elseif($v.ContainsKey('--kv-capacity') -and $v['--kv-capacity'] -ne 'auto'){$physical=[int]$v['--kv-capacity']}

  if($physical -ne $script:CapacityActiveTokens){Write-Host '实际命令的驻留参数与菜单预算不一致；拒绝EXTRA/CLI绕过显存范围。' -ForegroundColor Red;return $false}

  foreach($pair in @(@('--kv-dtype',$i.Kv),@('--max-concurrency','1'),@('--prefill-chunk','512'))){

   if($v.ContainsKey($pair[0]) -and [string]$v[$pair[0]] -ne $pair[1]){Write-Host ('实际命令 '+$pair[0]+' 改变了显存条件，请通过配置菜单选择；不能沿用旧上限。') -ForegroundColor Red;return $false}

  }

  if($v.ContainsKey('--cuda-graph-allowance-mib') -and [double]$v['--cuda-graph-allowance-mib'] -gt 160){Write-Host 'EXTRA的Graph预留超过本预算模型，请勿绕过菜单。' -ForegroundColor Red;return $false}

  if(($i.Head -eq 'full' -and $v.ContainsKey('--lm-head-draft')) -or ($i.Graph -eq '1' -and $v.ContainsKey('--no-cuda-graph'))){Write-Host '实际Graph/MTP头与预算条件不同，拒绝启动。' -ForegroundColor Red;return $false}

 }

 $fresh=Get-CapacityInfo $i.Profile $i.Kv $i.Graph $i.Head $i.Mtp

 try{Confirm-ResidentBound $fresh $script:CapacityActiveTokens;return $true}catch{Write-Host ('启动前显存复查失败：'+$_.Exception.Message) -ForegroundColor Yellow;return $false}

}

function Capture-CapacityFailure([string]$Line){

 if($Line -match 'reservation requires (\d+) bytes, but only (\d+) bytes'){$script:CapacityFailureLine=$Line}

}

function Register-CapacityFailure {

 $i=$script:CapacityActiveInfo

 if(-not $i -or -not $script:CapacityFailureLine){return}

 if($script:CapacityFailureLine -notmatch 'reservation requires (\d+) bytes, but only (\d+) bytes'){return}

 $need=[double]$matches[1];$have=[double]$matches[2];$gap=[Math]::Max(0,($need-$have)/1MB)

 $cut=[int]([Math]::Ceiling($gap/$i.PerTokenMiB/64)*64)+64

 $limit=[Math]::Max(0,$script:CapacityActiveTokens-$cut)

 $base=[Math]::Max($i.BaseMiB,$i.FreeMiB-$limit*$i.PerTokenMiB)

 $rows=@();if(Test-Path $script:CapacityStateFile){try{$rows=@(Get-Content $script:CapacityStateFile -Raw -Encoding UTF8|ConvertFrom-Json)|Where-Object{$_.Key -ne $i.Key}}catch{}}

 $rows=@($rows)+@([pscustomobject]@{Key=$i.Key;BaseMiB=$base;ObservedFreeMiB=$i.FreeMiB;RequestedTokens=$script:CapacityActiveTokens;AdjustedTokens=$limit;NeedBytes=$need;HaveBytes=$have;ObservedAt=(Get-Date).ToString('o')})

 if($env:NINFER_LAUNCHER_TEST -ne '1'){$rows|ConvertTo-Json -Depth 5|Set-Content $script:CapacityStateFile -Encoding UTF8}

 Write-Host ('  实际运行预留不足 '+[Math]::Ceiling($gap)+' MiB；该次可用驻留需降至约 '+$limit+' 以下。已校正此组合的预算，未改你的手动配置。') -ForegroundColor Yellow

}

function Return-CapacityMenu([string]$Path,[bool]$NoSync=$false){

 $x=Read-Answer '显存预算不通过：[C] 返回配置并刷新范围 / [Q或回车] 退出'

 if($x -ieq 'c'){$a=@('-NoProfile','-ExecutionPolicy','Bypass','-File',$Path);if($NoSync){$a+='nosync'};& powershell.exe @a}

}



function Convert-CapacityToken([string]$Text){

 $s=$Text.Trim().ToLowerInvariant();$v=0.0

 if($s -match '^(\d+(?:\.\d+)?)k$'){$v=[double]::Parse($matches[1],[Globalization.CultureInfo]::InvariantCulture)*1024}

 elseif($s -match '^\d+$'){$v=[double]$s}else{return -1}

 if($v -gt 262144 -or $v -lt 0 -or $v -ne [Math]::Floor($v)){return -1};return [int]$v

}



function Ask-CapacityValue([string]$Title,[string]$Current,[int]$Min,[int]$Max,[string[]]$Words=@(),[int]$Alignment=1){
 $aliases=@{};$options=@()
 for($n=0;$n -lt $Words.Count;$n++){
  $word=$Words[$n];$label=$(if($word -eq 'recommended'){'当前推荐'}elseif($word -eq 'auto'){'自动'}else{$word})
  if($Min -gt $Words.Count){$key=[string]($n+1);$aliases[$key]=$word;$options+=('['+$key+'] '+$label)}
  elseif($Words.Count -eq 1 -and $Min -gt 0){$aliases['0']=$word;$options+=('[0] '+$label)}
  else{$options+=$word}
 }
 while($true){
  $x=Read-Answer ($Title+'；范围 '+$Min+'–'+$Max+'；'+($options -join ' / ')+'；回车保持 '+$Current)
  if($x -eq ''){$x=$Current};$x=$x.ToLowerInvariant()
  if($aliases.ContainsKey($x)){return $aliases[$x]}
  if($x -in $Words){return $x}
  $v=Convert-CapacityToken $x
  if($v -ge $Min -and $v -le $Max -and $v%$Alignment -eq 0){return [string]$v}
  Write-Host ('输入无效或与前一步冲突；需按 '+$Alignment+' token 对齐。请重选，不会静默截断。') -ForegroundColor Yellow
 }
}



function Ask-CapacityChoice([string]$Title,[string]$Current,[string[]]$Values){
 $numeric=(@($Values|Where-Object{$_ -notmatch '^\d+$'}).Count -eq 0)
 $labels=@{'0'='关';'1'='开';rk4v4='rk4v4（K/V 4位）';rk8v4='rk8v4（K 8位、V 4位）';int8='int8（8位）';q4='q4（Q4草稿权重）';q6='q6（原版草稿权重）';full='full（完整输出头）';lite='lite（精简输出头）';normal='normal（普通模式）';kvmem='kvmem（KVMem）';kvrk='kvrk（KVMem + rk8v4）';kvrk4='kvrk4（KVMem + rk4v4）'}
 Write-Host ('  '+$Title) -ForegroundColor Cyan
 for($n=0;$n -lt $Values.Count;$n++){
  $v=$Values[$n];$key=$(if($numeric){$v}else{[string]($n+1)});$label=$(if($labels.ContainsKey($v)){$labels[$v]}else{$v})
  Write-Host ('    ['+$key+'] '+$label+$(if($v -eq $Current){'（当前）'}else{''}))
 }
 if(-not $numeric){Write-Host '    [0] 保持当前'}
 while($true){
  $x=Read-Answer ('请输入编号；回车保持 '+$Current+'（也兼容名称）')
  if(-not $x -or (-not $numeric -and $x -eq '0')){if($Current -in $Values){return $Current}}
  $x=$x.ToLowerInvariant()
  if($x -in $Values){return $x}
  $n=0
  if(-not $numeric -and [int]::TryParse($x,[ref]$n) -and $n -ge 1 -and $n -le $Values.Count){return $Values[$n-1]}
  Write-Host '请输入上面显示的编号，或直接回车。' -ForegroundColor Yellow
 }
}
