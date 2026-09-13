[CmdletBinding()]
param(
    [string]$Model = "small.en"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WhisperRoot = Join-Path $Root "tools\whisper"
$BinDir = Join-Path $WhisperRoot "bin"
$ModelDir = Join-Path $WhisperRoot "models"
New-Item -ItemType Directory -Force -Path $BinDir,$ModelDir | Out-Null

function Test-ModelFile {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [string]$ModelName
    )
    if (!(Test-Path $Path)) { return $false }

    $item = Get-Item $Path
    # Reject tiny error pages / interrupted downloads. Real whisper ggml models are tens of MB or larger.
    if ($item.Length -lt 10MB) { return $false }

    # Official whisper.cpp models README publishes this SHA-1 for small.en.
    if ($ModelName -eq "small.en") {
        $expectedSha1 = "db8a495a91d927739e50b3fc1cc4c6b8f6c2d022"
        $actualSha1 = (Get-FileHash -Algorithm SHA1 -Path $Path).Hash.ToLowerInvariant()
        if ($actualSha1 -ne $expectedSha1) { return $false }
    }
    return $true
}

# The binary download already succeeded on many machines before a model-download failure.
# Reuse it when present instead of downloading the release archive again.
$existingCli = Get-ChildItem $BinDir -Recurse -Filter "whisper-cli.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($existingCli) {
    Write-Host "whisper.cpp binary already present: $($existingCli.FullName)"
} else {
    Write-Host "Finding current stable whisper.cpp Windows release..."
    $headers = @{"User-Agent"="FacelessRC-Agent2"}
    $releases = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/ggml-org/whisper.cpp/releases?per_page=10"
    $release = $releases | Where-Object { -not $_.draft -and -not $_.prerelease } | Select-Object -First 1
    if (-not $release) { throw "Could not find a stable whisper.cpp release." }

    # Prefer official CUDA x64 builds. Fall back to CPU x64 if naming changes or CUDA asset is unavailable.
    $asset = $release.assets | Where-Object { $_.name -match '(?i)^whisper.*(?:cuda|cublas).*x64.*\.zip$' } | Select-Object -First 1
    if (-not $asset) { $asset = $release.assets | Where-Object { $_.name -match '(?i)^whisper.*bin.*x64.*\.zip$' -and $_.name -notmatch '(?i)arm64' } | Select-Object -First 1 }
    if (-not $asset) { throw "No suitable Windows x64 whisper.cpp binary asset found in $($release.tag_name)." }

    $tmp = Join-Path $env:TEMP ("facelessrc_whisper_" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    try {
        $zip = Join-Path $tmp $asset.name
        Write-Host "Downloading whisper.cpp $($release.tag_name): $($asset.name)"
        Invoke-WebRequest -Headers $headers -Uri $asset.browser_download_url -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath (Join-Path $tmp "unzipped") -Force
        $cli = Get-ChildItem (Join-Path $tmp "unzipped") -Recurse -Filter "whisper-cli.exe" | Select-Object -First 1
        if (-not $cli) { throw "whisper-cli.exe was not found in downloaded archive." }
        Copy-Item (Join-Path $cli.Directory.FullName "*") $BinDir -Force -Recurse
    } finally {
        Remove-Item $tmp -Force -Recurse -ErrorAction SilentlyContinue
    }
}

$modelFile = "ggml-$Model.bin"
$modelPath = Join-Path $ModelDir $modelFile

if ((Test-Path $modelPath) -and !(Test-ModelFile -Path $modelPath -ModelName $Model)) {
    Write-Warning "Existing Whisper model is incomplete or invalid. Removing it before retry."
    Remove-Item $modelPath -Force -ErrorAction SilentlyContinue
}

if (!(Test-ModelFile -Path $modelPath -ModelName $Model)) {
    # Official whisper.cpp documentation points to this Hugging Face repository for pre-converted ggml models.
    # Do not append ?download=true: direct resolve + curl is more robust with large redirected/Xet files on Windows.
    $modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$modelFile"
    $partialPath = "$modelPath.partial"
    Remove-Item $partialPath -Force -ErrorAction SilentlyContinue

    Write-Host "Downloading local Whisper model: $modelFile"
    Write-Host "Source: $modelUrl"

    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -L --fail --retry 5 --retry-delay 2 --connect-timeout 30 -o $partialPath $modelUrl
        if ($LASTEXITCODE -ne 0) {
            Remove-Item $partialPath -Force -ErrorAction SilentlyContinue
            throw "curl failed to download $modelFile (exit code $LASTEXITCODE)."
        }
    } else {
        # Fallback for unusual Windows installations without curl.exe.
        Invoke-WebRequest -Uri $modelUrl -OutFile $partialPath
    }

    if (!(Test-ModelFile -Path $partialPath -ModelName $Model)) {
        Remove-Item $partialPath -Force -ErrorAction SilentlyContinue
        throw "Downloaded Whisper model failed validation. The download may have been interrupted or replaced by an error response."
    }

    Move-Item $partialPath $modelPath -Force
    Write-Host "Whisper model download validated."
} else {
    Write-Host "Whisper model already present and valid: $modelPath"
}

$finalCli = Get-ChildItem $BinDir -Recurse -Filter "whisper-cli.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $finalCli) { throw "whisper-cli.exe is missing after setup." }
if (!(Test-ModelFile -Path $modelPath -ModelName $Model)) { throw "Whisper model is missing or invalid after setup." }

Write-Host "whisper.cpp ready: $BinDir"
Write-Host "model: $modelPath"
