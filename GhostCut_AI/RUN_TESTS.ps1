[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Py)) { throw "Run .\SETUP_WINDOWS.ps1 first." }
Push-Location $Root
try {
    & $Py -m pip install pytest httpx | Out-Null
    & $Py -m pytest -q
    exit $LASTEXITCODE
} finally { Pop-Location }
