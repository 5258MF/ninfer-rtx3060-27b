# Verifies the extracted ninfer-3060-8g-oneclick directory against the embedded SHA256 manifest
# and checks for stale binaries, missing companion DLLs, or garbled (mojibake) root filenames.
# ASCII-only source to avoid Windows PowerShell 5.1 codepage issues.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File verify-kit-manifest.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File verify-kit-manifest.ps1 -KitRoot D:\ninfer-8g

[CmdletBinding()]
param(
    [string]$KitRoot = ""
)

$ErrorActionPreference = "Stop"
if (-not $KitRoot) { $KitRoot = $PSScriptRoot }

$UtcKit = [datetime]::SpecifyKind([datetime]"2026-10-04T00:00:00", [System.DateTimeKind]::Utc)
$UtcDll = [datetime]::SpecifyKind([datetime]"2025-01-01T00:00:00", [System.DateTimeKind]::Utc)
# The rk8v4 engine is the exact binary shipped in the 12G kit (same source tree), so its file
# timestamp is the 12G release stamp rather than the 8G one.
$UtcEngine = [datetime]::SpecifyKind([datetime]"2026-10-03T00:00:00", [System.DateTimeKind]::Utc)

$EmbeddedExpected = [ordered]@{
    "launcher/runtime-options.ps1" = @{ Sha256 = "eff90bdebf0a2344bb1a5b64820464527106e6b327873d81bb37f4d6da0f1f7c"; MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2020-01-01T00:00:00", [System.DateTimeKind]::Utc); Role = "CPU vision and CUDA Graph controls" }
    "engine/ninfer-serve.exe"       = @{ Sha256 = "43cdbfc15ab2ab0714a9bc2d6832b77e4c5704c15c5d42ac2c68926fab443d60"; MinLastWriteUtc = $UtcEngine; Role = "rk8v4 engine (normal / KVMem modes), identical to the 12G kit binary" }
    "engine/ninfer-serve-rk4.exe"   = @{ Sha256 = "1e3e5a57bc989d0ae24328676139b547fc79618a1ca7184027e1db018f196e7a"; MinLastWriteUtc = $UtcKit; Role = "rk4v4 engine (rk4 / kvrk4 modes)" }
    "engine/cudart64_13.dll"        = @{ Sha256 = "b00ca6f53699120da815bf3e06e2e4285fae2f201235b883dcbb50eec51e2a2a"; MinLastWriteUtc = $UtcDll; Role = "CUDA 13 runtime DLL" }
    "engine/msvcp140.dll"           = @{ Sha256 = "7c26614e1d733892c2deac7e245ce115504b1d80592dd0a01b08e3e5a55f89ca"; MinLastWriteUtc = $UtcDll; Role = "MSVC C++ runtime DLL" }
    "engine/vcruntime140.dll"       = @{ Sha256 = "d1f4225df2cd877dbf130d5668a021dce3f94118455ff5ec952061c30afc9ce7"; MinLastWriteUtc = $UtcDll; Role = "VC runtime DLL" }
    "engine/vcruntime140_1.dll"     = @{ Sha256 = "a7146c08f89fe5b04541ab507cdb59ff7b44534d4ba3c668a426c6450a03434e"; MinLastWriteUtc = $UtcDll; Role = "VC runtime 140_1 exception handling DLL" }
    "engine/avcodec-63.dll"         = @{ Sha256 = "6c1261fce61f9e45ef75336e471d717ca90e651ecd9ae4f275d00009e0abbccf"; MinLastWriteUtc = $UtcDll; Role = "FFmpeg avcodec runtime DLL" }
    "engine/avutil-61.dll"          = @{ Sha256 = "362afc7047f618e10a2bf436844517daf29e3e473ffdebf4a23c30ff2e6ab2ab"; MinLastWriteUtc = $UtcDll; Role = "FFmpeg avutil runtime DLL" }
    "engine/avformat-63.dll"        = @{ Sha256 = "f4ff30eec58ae74a0b24961feeb692a1910ef5c342c45a00d1ac45e2facbca00"; MinLastWriteUtc = $UtcDll; Role = "FFmpeg avformat runtime DLL" }
    "engine/avfilter-12.dll"        = @{ Sha256 = "3cf989233fba97f6b339635c395b3e1ba0da70e5612423f981350a7ec2c791c8"; MinLastWriteUtc = $UtcDll; Role = "FFmpeg avfilter runtime DLL" }
    "engine/avdevice-63.dll"        = @{ Sha256 = "145cb445054e801b6a48aa025d6bf36a56666e845c988c90c936e7868bc882bd"; MinLastWriteUtc = $UtcDll; Role = "FFmpeg avdevice runtime DLL" }
    "engine/swscale-10.dll"         = @{ Sha256 = "dc0cbda7376786e63e4b427f1340cfdee809700ec5f81c3016274cdaa8c3eb10"; MinLastWriteUtc = $UtcDll; Role = "FFmpeg swscale runtime DLL" }
    "engine/swresample-7.dll"       = @{ Sha256 = "a81be671fe3fdebb6bc12abc1268c6e065d68de61ddf1efca721e52752f10f8f"; MinLastWriteUtc = $UtcDll; Role = "FFmpeg swresample runtime DLL" }
    "patch/ops.txt"                 = @{ Sha256 = "25678c4a4f70998087d0c5fb4a47436257b90867bc9db445387eacf3222ba1f9"; MinLastWriteUtc = $UtcDll; Role = "MTP Q4 conversion patch script" }
    "patch/lit.bin"                 = @{ Sha256 = "8b924c284cd262883f76706c16a5b5b58fe6bc1fd7f4c21f362218c317b09701"; MinLastWriteUtc = $UtcDll; Role = "MTP Q4 conversion literal payload" }
    "launcher/launch.ps1"           = @{ Sha256 = "eb24ac3a3496d0422a8024ae921241ebc51ec8d674d272b8a97cf302e070a555"; MinLastWriteUtc = $UtcKit; Role = "One-click interactive launcher" }
    "launcher/test.ps1"             = @{ Sha256 = "fef3ba95e1423ce0df107b63cfd78e027e69fd875b20acd81edd7eba94a05ca2"; MinLastWriteUtc = $UtcDll; Role = "Connectivity and speed test script" }
    "verify-arch-engine.ps1"        = @{ Sha256 = "5bec4815a679c8e6a5036bc79f7f91ac5351e4d87081a2c43dc651e3d7573d25"; MinLastWriteUtc = $UtcKit; Role = "End-to-end hardware, needle and agent-compat verification harness" }
}

function Format-CodePoints([string]$s) {
    $parts = foreach ($ch in $s.ToCharArray()) { "U+{0:X4}" -f ([int][char]$ch) }
    return ($parts -join " ")
}

$ManifestFile = Join-Path $KitRoot "SHA256SUMS.txt"
$FileExpected = @{}
if (Test-Path -LiteralPath $ManifestFile) {
    foreach ($line in Get-Content -LiteralPath $ManifestFile -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
        if ($trimmed -match "^([0-9a-fA-F]{64})\s+\*?(.+)$") {
            $FileExpected[$Matches[2].Trim()] = $Matches[1].ToLowerInvariant()
        }
    }
}

Write-Host "=== Verifying ninfer-3060-8g-oneclick kit at: $KitRoot ===" -ForegroundColor Cyan
$failures = New-Object System.Collections.Generic.List[string]

# 1. Embedded ASCII-named core files (hash + not older than the release stamp)
foreach ($entry in $EmbeddedExpected.GetEnumerator()) {
    $relName = $entry.Key
    $meta = $entry.Value
    $fullPath = Join-Path $KitRoot ($relName -replace "/", "\")
    if (-not (Test-Path -LiteralPath $fullPath)) {
        $failures.Add("MISSING: $relName ($($meta.Role))")
        Write-Host ("  [FAIL] {0,-28} MISSING" -f $relName) -ForegroundColor Red
        continue
    }
    $item = Get-Item -LiteralPath $fullPath
    $actualHash = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $expectedHash = $meta.Sha256.ToLowerInvariant()
    $manifestMismatch = $false
    if ($FileExpected.ContainsKey($relName) -and $FileExpected[$relName] -ne $expectedHash) { $manifestMismatch = $true }
    $mtimeUtc = $item.LastWriteTimeUtc
    if ($actualHash -ne $expectedHash -or $manifestMismatch) {
        $failures.Add("HASH_MISMATCH: $relName expected=$expectedHash actual=$actualHash")
        Write-Host ("  [FAIL] {0,-28} sha256={1} (expected {2})" -f $relName, $actualHash.Substring(0, 12), $expectedHash.Substring(0, 12)) -ForegroundColor Red
    } elseif ($mtimeUtc -lt $meta.MinLastWriteUtc) {
        $failures.Add("STALE_TIMESTAMP: $relName mtimeUtc=$($mtimeUtc.ToString('o'))")
        Write-Host ("  [FAIL] {0,-28} stale mtime={1:u}" -f $relName, $mtimeUtc) -ForegroundColor Red
    } else {
        Write-Host ("  [OK]   {0,-28} sha256={1}... size={2,12:N0}" -f $relName, $actualHash.Substring(0, 12), $item.Length) -ForegroundColor Green
    }
}

# 2. Companion DLL count in engine/
$engineDir = Join-Path $KitRoot "engine"
$dllCount = @(Get-ChildItem -LiteralPath $engineDir -Filter "*.dll" -ErrorAction SilentlyContinue).Count
if ($dllCount -lt 11) {
    $failures.Add("INCOMPLETE_ENGINE_DLLS: found $dllCount .dll files in engine/ (expected 11)")
    Write-Host ("  [FAIL] engine/*.dll count = {0} (expected 11)" -f $dllCount) -ForegroundColor Red
} else {
    Write-Host ("  [OK]   engine/*.dll count      = {0} companion DLLs present" -f $dllCount) -ForegroundColor Green
}
$exeCount = @(Get-ChildItem -LiteralPath $engineDir -Filter "ninfer-serve*.exe" -ErrorAction SilentlyContinue).Count
if ($exeCount -lt 2) {
    $failures.Add("INCOMPLETE_ENGINE_EXES: found $exeCount ninfer-serve*.exe in engine/ (expected 2)")
    Write-Host ("  [FAIL] engine/ninfer-serve*.exe count = {0} (expected 2)" -f $exeCount) -ForegroundColor Red
} else {
    Write-Host ("  [OK]   engine/ninfer-serve*.exe = {0} engines present" -f $exeCount) -ForegroundColor Green
}

# 3. Chinese root entry files via exact Unicode codepoints (pure ASCII source) + ZIP codepage mojibake check
$ExpectedChineseRootFiles = @(
    @{ Name = ([string][char]0x542F + [char]0x52A8 + ".bat"); Desc = "Start batch (U+542F U+52A8 .bat)" },
    @{ Name = ([string][char]0x6D4B + [char]0x8BD5 + ".bat"); Desc = "Test batch (U+6D4B U+8BD5 .bat)" },
    @{ Name = ([string][char]0x8BBE + [char]0x7F6E + ".ini"); Desc = "Config INI (U+8BBE U+7F6E .ini)" },
    @{ Name = ([string][char]0x4F7F + [char]0x7528 + [char]0x8BF4 + [char]0x660E + ".txt"); Desc = "User guide TXT (U+4F7F U+7528 U+8BF4 U+660E .txt)" }
)
$OptionalChineseRootFiles = @(
    ([string][char]0x63A5 + [char]0x5165 + [char]0x4FE1 + [char]0x606F + ".txt")
)

$validChineseNames = @{}
foreach ($f in $ExpectedChineseRootFiles) { $validChineseNames[$f.Name] = $true }
foreach ($n in $OptionalChineseRootFiles) { $validChineseNames[$n] = $true }

foreach ($f in $ExpectedChineseRootFiles) {
    $p = Join-Path $KitRoot $f.Name
    if (-not (Test-Path -LiteralPath $p)) {
        $failures.Add("MISSING_OR_GARBLED_ROOT_FILE: $($f.Desc) not found")
        Write-Host ("  [FAIL] Root file missing: {0}" -f $f.Desc) -ForegroundColor Red
    } else {
        Write-Host ("  [OK]   Root file intact : {0}" -f $f.Desc) -ForegroundColor Green
    }
}

$rootFiles = @(Get-ChildItem -LiteralPath $KitRoot -File -ErrorAction SilentlyContinue)
foreach ($rf in $rootFiles) {
    $hasNonAscii = $false
    foreach ($ch in $rf.Name.ToCharArray()) { if ([int][char]$ch -gt 127) { $hasNonAscii = $true; break } }
    if ($hasNonAscii -and -not $validChineseNames.ContainsKey($rf.Name)) {
        $cp = Format-CodePoints $rf.Name
        $failures.Add("GARBLED_FILENAME: '$($rf.Name)' ($cp) in kit root - the ZIP was decoded with the wrong codepage.")
        Write-Host ("  [FAIL] Garbled filename detected: '{0}' ({1})" -f $rf.Name, $cp) -ForegroundColor Red
    }
}

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Host "KIT VERIFICATION FAILED ($($failures.Count) issue(s)):" -ForegroundColor Red
    foreach ($f in $failures) { Write-Host "  - $f" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Remedy: delete the folder and re-extract ninfer-3060-8g-oneclick.zip into a clean ASCII path (e.g. D:\ninfer-8g) using Windows Explorer, 7-Zip or Bandizip." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "KIT VERIFICATION PASSED: engines, DLLs, patch payload, scripts and root entry files all match the release manifest." -ForegroundColor Green
exit 0
