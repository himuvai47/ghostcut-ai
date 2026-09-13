[CmdletBinding()]
param([switch]$SkipWhisperDownload)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Root
try {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python was not found in PATH." }
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "FFmpeg was not found in PATH." }
    if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) { throw "FFprobe was not found in PATH." }
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw "Ollama was not found in PATH." }

    if (!(Test-Path ".venv\Scripts\python.exe")) { python -m venv .venv }
    & ".venv\Scripts\python.exe" -m pip install --upgrade pip

    $models = (ollama list) -join "`n"
    foreach ($m in @("qwen3.5:9b","nomic-embed-text")) {
        if ($models -notmatch [regex]::Escape($m)) { Write-Host "Pulling $m..."; ollama pull $m }
        else { Write-Host "$m already installed." }
    }

    if (-not $SkipWhisperDownload) { & "$Root\SETUP_WHISPER_CPP.ps1" }
    Write-Host "`nRunning Agent 2 precheck..."
    & "$Root\CHECK_AGENT2.ps1"
} finally { Pop-Location }
