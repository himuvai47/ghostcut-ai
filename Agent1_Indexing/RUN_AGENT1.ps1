[CmdletBinding()]
param(
    [string]$InputPath = "",
    [string]$Workspace = "",
    [string]$Config = "",
    [switch]$Force,
    [switch]$NoAI,
    [switch]$FaceBackfill,
    [switch]$EditorialBackfill,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Virtual environment missing. Run .\SETUP_WINDOWS.ps1 first."
}

if ([string]::IsNullOrWhiteSpace($Workspace)) {
    $Workspace = Join-Path $PSScriptRoot "index"
}

if (-not $FaceBackfill -and -not $EditorialBackfill -and [string]::IsNullOrWhiteSpace($InputPath)) {
    throw "InputPath is required unless -FaceBackfill or -EditorialBackfill is used."
}

$ArgsList = @("indexer.py", "--workspace", $Workspace)
if (-not [string]::IsNullOrWhiteSpace($InputPath)) { $ArgsList += $InputPath }
if (-not [string]::IsNullOrWhiteSpace($Config)) { $ArgsList += @("--config", $Config) }
if ($Force) { $ArgsList += "--force" }
if ($NoAI) { $ArgsList += "--no-ai" }
if ($FaceBackfill) { $ArgsList += "--face-backfill" }
if ($EditorialBackfill) { $ArgsList += "--editorial-backfill" }
if ($Limit -gt 0) { $ArgsList += @("--limit", "$Limit") }
if ($PSBoundParameters.ContainsKey("Verbose")) { $ArgsList += "--verbose" }

& $Py @ArgsList
exit $LASTEXITCODE
