[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$TimelinePath,
    [Parameter(Mandatory=$true)][string]$Workspace,
    [string]$Config = "",
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Py)) { throw "Agent 3 virtual environment not found. Run .\SETUP_WINDOWS.ps1 first." }
$argsList = @("agent3.py","--timeline",$TimelinePath,"--workspace",$Workspace)
if ($Config) { $argsList += @("--config",$Config) }
if ($Force) { $argsList += "--force" }
if ($PSBoundParameters.ContainsKey("Verbose")) { $argsList += "--verbose" }
Push-Location $Root
try { & $Py @argsList; exit $LASTEXITCODE } finally { Pop-Location }
