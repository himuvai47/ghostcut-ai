[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ScriptPath,
    [Parameter(Mandatory=$true)][string]$AudioPath,
    [Parameter(Mandatory=$true)][string]$Agent1Workspace,
    [Parameter(Mandatory=$true)][string]$Workspace,
    [string]$Config = "",
    [switch]$Force,
    [switch]$SkipGlobalReview
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Py)) { throw "Agent 2 virtual environment not found. Run .\SETUP_WINDOWS.ps1 first." }
$argsList = @("agent2.py","--script",$ScriptPath,"--audio",$AudioPath,"--agent1-workspace",$Agent1Workspace,"--workspace",$Workspace)
if ($Config) { $argsList += @("--config",$Config) }
if ($Force) { $argsList += "--force" }
if ($SkipGlobalReview) { $argsList += "--skip-global-review" }
if ($PSBoundParameters.ContainsKey("Verbose")) { $argsList += "--verbose" }
Push-Location $Root
try { & $Py @argsList; exit $LASTEXITCODE } finally { Pop-Location }
