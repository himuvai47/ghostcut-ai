[CmdletBinding()] param()
$ErrorActionPreference="Stop"
$Root=Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Setting up FacelessRC Agent 3 v1.0.0..."
if (!(Get-Command python -ErrorAction SilentlyContinue)) { throw "Python not found in PATH" }
if (!(Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "FFmpeg not found in PATH" }
if (!(Get-Command ffprobe -ErrorAction SilentlyContinue)) { throw "FFprobe not found in PATH" }
$Py=Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Py)) { python -m venv (Join-Path $Root ".venv") }
Write-Host "No Python packages or model downloads are required."
Write-Host "Setup complete."
Write-Host "Run .\CHECK_AGENT3.ps1 to verify."
