[CmdletBinding()]
param([string]$KitRoot='')
$ErrorActionPreference='Stop'
if(-not $KitRoot){$KitRoot=$PSScriptRoot}
$manifest=Join-Path $KitRoot 'SHA256SUMS.txt'
if(-not(Test-Path -LiteralPath $manifest)){throw 'Missing SHA256SUMS.txt; extract the complete ZIP.'}
$count=0
foreach($line in Get-Content -LiteralPath $manifest -Encoding UTF8){
    if($line -notmatch '^([0-9a-f]{64})  (.+)$'){throw 'Invalid manifest line'}
    $expected=$matches[1];$relative=$matches[2]
    $path=[IO.Path]::GetFullPath((Join-Path $KitRoot $relative))
    $base=[IO.Path]::GetFullPath($KitRoot).TrimEnd('\')+'\'
    if(-not $path.StartsWith($base,[StringComparison]::OrdinalIgnoreCase)){throw 'Invalid manifest path'}
    if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw ('Missing file: '+$relative)}
    if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected){throw ('SHA256 mismatch: '+$relative)}
    $count++
}
if($count -lt 20){throw 'Incomplete release manifest'}
Write-Host ('KIT VERIFICATION PASSED: '+$count+' release files match SHA256SUMS.txt.') -ForegroundColor Green
