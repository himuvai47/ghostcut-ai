[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $MyInvocation.MyCommand.Path
$Py=Join-Path $Root '.venv\Scripts\python.exe'
if (!(Test-Path $Py)) { $Py='python' }
Push-Location $Root
try { & $Py -m unittest discover -s tests -v; exit $LASTEXITCODE } finally { Pop-Location }
