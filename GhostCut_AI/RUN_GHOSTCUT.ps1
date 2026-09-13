[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Py)) { throw "GhostCut virtual environment not found. Run .\SETUP_WINDOWS.ps1 first." }
Push-Location $Root
try {
    Write-Host "Launching GhostCut AI..." -ForegroundColor Cyan
    Write-Host "Close this window or press Ctrl+C to stop the local app."
    & $Py ghostcut_cli.py
    exit $LASTEXITCODE
} finally { Pop-Location }
