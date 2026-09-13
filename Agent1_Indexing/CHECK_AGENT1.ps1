$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Virtual environment missing. Run .\SETUP_WINDOWS.ps1 first."
}
& $Py indexer.py --check
exit $LASTEXITCODE
