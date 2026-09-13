[CmdletBinding()] param()
$ErrorActionPreference="Stop"
$Root=Split-Path -Parent $MyInvocation.MyCommand.Path
$Py=Join-Path $Root ".venv\Scripts\python.exe"
$ffmpeg=[bool](Get-Command ffmpeg -ErrorAction SilentlyContinue)
$ffprobe=[bool](Get-Command ffprobe -ErrorAction SilentlyContinue)
$nvenc=$false
if ($ffmpeg) { $enc = (& ffmpeg -hide_banner -encoders 2>&1 | Out-String); $nvenc=$enc.Contains("h264_nvenc") }
$pythonOk=Test-Path $Py
[ordered]@{ agent3_version="1.0.0"; python_venv=$pythonOk; ffmpeg=$ffmpeg; ffprobe=$ffprobe; h264_nvenc=$nvenc; passed=($pythonOk -and $ffmpeg -and $ffprobe) } | ConvertTo-Json
