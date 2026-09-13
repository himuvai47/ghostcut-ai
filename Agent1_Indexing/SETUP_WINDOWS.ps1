$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== FacelessRC Agent 1 Setup ===" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found in PATH. Python 3.11+ is required; your intended Python 3.14.x is supported."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv..."
    python -m venv .venv
}

$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip
& $Py -m pip install -r requirements.txt

Write-Host ""
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "FFmpeg is not in PATH. Install FFmpeg and reopen PowerShell before indexing."
} else {
    Write-Host "OK: FFmpeg found"
}

if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Write-Warning "FFprobe is not in PATH. It normally ships with FFmpeg."
} else {
    Write-Host "OK: FFprobe found"
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Warning "Ollama command not found. Install/start Ollama before indexing."
} else {
    Write-Host "OK: Ollama command found"
    $Models = (& ollama list 2>$null | Out-String)
    if ($Models -notmatch "qwen2\.5vl:7b") {
        Write-Host "Qwen2.5-VL 7B is not listed. Install it with:" -ForegroundColor Yellow
        Write-Host "  ollama pull qwen2.5vl:7b" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Running preflight..." -ForegroundColor Cyan
& $Py indexer.py --check
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Preflight is not complete yet. Fix the item shown above, then run .\CHECK_AGENT1.ps1"
} else {
    Write-Host "Agent 1 setup complete." -ForegroundColor Green
}
