# Verifies that the ninfer-3060-swift15-oneclick directory contains the expected
# engine binary, companion DLLs, launcher scripts, model patch files, and intact
# root entry scripts (catching partial extraction, stale overwrite, or ZIP codepage mojibake).
# Pure ASCII script source to avoid Windows PowerShell 5.1 encoding issues.

[CmdletBinding()]
param(
    [string]$KitRoot = ""
)

$ErrorActionPreference = "Stop"
if (-not $KitRoot) { $KitRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }

$EmbeddedExpected = [ordered]@{
    "launcher/runtime-options.ps1" = @{ Sha256 = "eff90bdebf0a2344bb1a5b64820464527106e6b327873d81bb37f4d6da0f1f7c"; MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2020-01-01T00:00:00", [System.DateTimeKind]::Utc); Role = "CPU vision and CUDA Graph controls" }
    "engine/ninfer-serve.exe"   = @{
        Sha256          = "e83e4c18b6d92b37d3986c7c4c83b7d87b943776c70393ebcf8fc81429117a87"
        MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2026-10-03T13:00:00", [System.DateTimeKind]::Utc)
        Role            = "Swift 1.5 sm86 engine binary (NVML physical VRAM budget + KVMem Eq.10 content scorer ON + Windows TCP keepalive)"
    }
    "engine/cudart64_13.dll"    = @{
        Sha256          = "b00ca6f53699120da815bf3e06e2e4285fae2f201235b883dcbb50eec51e2a2a"
        MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2025-01-01T00:00:00", [System.DateTimeKind]::Utc)
        Role            = "CUDA 13 runtime DLL"
    }
    "engine/cublas64_13.dll"    = @{
        Sha256          = "f1d500d0cd892f5b8c6b6cdbffd82d0c55d5f5427215668e7ceb55aeeccc1b63"
        MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2025-01-01T00:00:00", [System.DateTimeKind]::Utc)
        Role            = "cuBLAS 13 runtime DLL"
    }
    "engine/cublasLt64_13.dll"  = @{
        Sha256          = "b592cd016d7673e9cb97716a22b27c4010ee635377a3ba28f37070a9bdb76a68"
        MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2025-01-01T00:00:00", [System.DateTimeKind]::Utc)
        Role            = "cuBLASLt 13 runtime DLL"
    }
    "engine/msvcp140.dll"       = @{
        Sha256          = "7c26614e1d733892c2deac7e245ce115504b1d80592dd0a01b08e3e5a55f89ca"
        MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2025-01-01T00:00:00", [System.DateTimeKind]::Utc)
        Role            = "MSVC C++ runtime DLL"
    }
    "engine/vcruntime140.dll"   = @{
        Sha256          = "d1f4225df2cd877dbf130d5668a021dce3f94118455ff5ec952061c30afc9ce7"
        MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2025-01-01T00:00:00", [System.DateTimeKind]::Utc)
        Role            = "VC runtime DLL"
    }
    "engine/vcruntime140_1.dll" = @{
        Sha256          = "a7146c08f89fe5b04541ab507cdb59ff7b44534d4ba3c668a426c6450a03434e"
        MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2025-01-01T00:00:00", [System.DateTimeKind]::Utc)
        Role            = "VC runtime 140_1 exception handling DLL"
    }
    "launcher/launch.ps1"       = @{
        Sha256          = "193853a5c8050be67ad606d69c2901d8fff552a71abdc5f74ca5bf3e8567a044"
        MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2026-10-03T13:00:00", [System.DateTimeKind]::Utc)
        Role            = "One-click interactive launcher script"
    }
    "launcher/test.ps1"         = @{
        Sha256          = "c6991fb22c6c0a0f442c2594dfb91105bb383ede7ac78dfde367f04a0f6ce55a"
        MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2026-09-29T00:00:00", [System.DateTimeKind]::Utc)
        Role            = "Connectivity and speed test script"
    }
    "patch/ops.txt"             = @{
        Sha256          = "43e035c1c5cb0d136b880ca7df95cc7d9b5ef61fd9a4bd8b9aa62e57eec9eaea"
        MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2026-10-02T00:00:00", [System.DateTimeKind]::Utc)
        Role            = "MTP Q4 conversion recipe"
    }
    "patch/lit.bin"             = @{
        Sha256          = "71dd3214d5dd3e9138b6989b121fe2f847447382a3fad4d9b33ffb73af9bac72"
        MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2026-10-02T00:00:00", [System.DateTimeKind]::Utc)
        Role            = "MTP Q4 tensor payload"
    }
    "verify-arch-engine.ps1"    = @{
        Sha256          = "ee0e094b994570fe9b491ad042ecf04d9fe1036ef512bbd04968baae8f6529fd"
        MinLastWriteUtc = [datetime]::SpecifyKind([datetime]"2026-10-03T13:00:00", [System.DateTimeKind]::Utc)
        Role            = "End-to-end hardware & needle verification harness"
    }
}

function Format-CodePoints([string]$s) {
    $parts = foreach ($ch in $s.ToCharArray()) {
        "U+{0:X4}" -f ([int][char]$ch)
    }
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

Write-Host "=== Verifying ninfer-3060-swift15-oneclick kit at: $KitRoot ===" -ForegroundColor Cyan
$failures = New-Object System.Collections.Generic.List[string]

# 1. Check embedded ASCII-named core files
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
    if ($FileExpected.ContainsKey($relName) -and $FileExpected[$relName] -ne $expectedHash) {
        $manifestMismatch = $true
    }
    $mtimeUtc = $item.LastWriteTimeUtc
    $staleTime = $mtimeUtc -lt $meta.MinLastWriteUtc
    if ($actualHash -ne $expectedHash -or $manifestMismatch) {
        $failures.Add("HASH_MISMATCH: $relName expected=$expectedHash actual=$actualHash mtimeUtc=$($mtimeUtc.ToString('o'))")
        Write-Host ("  [FAIL] {0,-28} sha256={1} (expected {2}) mtime={3:u}" -f $relName, $actualHash.Substring(0, 12), $expectedHash.Substring(0, 12), $mtimeUtc) -ForegroundColor Red
    } elseif ($staleTime) {
        $failures.Add("STALE_TIMESTAMP: $relName mtimeUtc=$($mtimeUtc.ToString('o')) < $($meta.MinLastWriteUtc.ToString('o'))")
        Write-Host ("  [FAIL] {0,-28} stale mtime={1:u}" -f $relName, $mtimeUtc) -ForegroundColor Red
    } else {
        Write-Host ("  [OK]   {0,-28} sha256={1}... size={2,12:N0} mtime={3:u}" -f $relName, $actualHash.Substring(0, 12), $item.Length, $mtimeUtc) -ForegroundColor Green
    }
}

# 2. Check companion DLL count in engine/
$engineDir = Join-Path $KitRoot "engine"
$dllCount = @(Get-ChildItem -LiteralPath $engineDir -Filter "*.dll" -ErrorAction SilentlyContinue).Count
if ($dllCount -lt 18) {
    $failures.Add("INCOMPLETE_ENGINE_DLLS: found $dllCount .dll files in engine/ (expected 18)")
    Write-Host ("  [FAIL] engine/*.dll count = {0} (expected 18)" -f $dllCount) -ForegroundColor Red
} else {
    Write-Host ("  [OK]   engine/*.dll count        = {0} companion DLLs present" -f $dllCount) -ForegroundColor Green
}

# 3. Check Chinese root entry files via exact Unicode codepoints (pure ASCII source) and detect ZIP codepage mojibake
$ExpectedChineseRootFiles = @(
    @{ Name = ([string][char]0x542F + [char]0x52A8 + ".bat"); Desc = "Start batch (U+542F U+52A8 .bat)" },
    @{ Name = ([string][char]0x6D4B + [char]0x8BD5 + ".bat"); Desc = "Test batch (U+6D4B U+8BD5 .bat)" },
    @{ Name = ([string][char]0x8BBE + [char]0x7F6E + ".ini"); Desc = "Config INI (U+8BBE U+7F6E .ini)" },
    @{ Name = ([string][char]0x4F7F + [char]0x7528 + [char]0x8BF4 + [char]0x660E + ".txt"); Desc = "User guide TXT (U+4F7F U+7528 U+8BF4 U+660E .txt)" },
    @{ Name = ([string][char]0x7248 + [char]0x672C + [char]0x4FE1 + [char]0x606F + ".txt"); Desc = "Version info TXT (U+7248 U+672C U+4FE1 U+606F .txt)" }
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
    foreach ($ch in $rf.Name.ToCharArray()) {
        if ([int][char]$ch -gt 127) { $hasNonAscii = $true; break }
    }
    if ($hasNonAscii -and -not $validChineseNames.ContainsKey($rf.Name)) {
        $cp = Format-CodePoints $rf.Name
        $failures.Add("GARBLED_FILENAME: '$($rf.Name)' ($cp) detected in kit root. Your ZIP extractor decoded UTF-8 filenames with the wrong codepage.")
        Write-Host ("  [FAIL] Garbled filename detected: '{0}' ({1})" -f $rf.Name, $cp) -ForegroundColor Red
    }
}

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Host "KIT VERIFICATION FAILED ($($failures.Count) issue(s)):" -ForegroundColor Red
    foreach ($f in $failures) {
        Write-Host "  - $f" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Remedy: Re-extract ninfer-3060-swift15-q2s-mtp-oneclick.zip into a clean ASCII path (e.g. D:\ninfer-swift15) using Windows 11 Explorer, 7-Zip, or Bandizip, and overwrite all files." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "KIT VERIFICATION PASSED: All core binaries, DLLs, patches, and root entry files match the release manifest." -ForegroundColor Green
exit 0
