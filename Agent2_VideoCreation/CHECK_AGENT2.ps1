[CmdletBinding()]
param([string]$Config = "")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Py)) { throw "Run .\SETUP_WINDOWS.ps1 first." }
$argsList=@("agent2.py","--check")
if ($Config) {$argsList += @("--config",$Config)}
Push-Location $Root
try { & $Py @argsList; exit $LASTEXITCODE } finally { Pop-Location }
