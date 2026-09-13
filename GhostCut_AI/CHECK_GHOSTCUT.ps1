[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Py)) { throw "GhostCut virtual environment not found. Run .\SETUP_WINDOWS.ps1 first." }
Push-Location $Root
try { & $Py -m ghostcut.check; exit $LASTEXITCODE } finally { Pop-Location }
