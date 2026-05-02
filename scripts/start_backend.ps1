param(
    [switch]$Reload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvCandidates = @()

if ($env:VIRTUAL_ENV) {
    $venvCandidates += $env:VIRTUAL_ENV
}

$venvCandidates += Join-Path $root "rag"
$venvCandidates += Join-Path $root ".venv"

$pythonExe = $null
foreach ($venvPath in $venvCandidates) {
    $candidate = Join-Path $venvPath "Scripts\python.exe"
    if (Test-Path $candidate) {
        $pythonExe = $candidate
        break
    }
}

if (-not $pythonExe) {
    $pythonExe = "python"
}

Set-Location (Join-Path $root "backend")
$uvicornArgs = @("-m", "uvicorn", "app.main:app", "--port", "8000")
if ($Reload) {
    $uvicornArgs += "--reload"
}

& $pythonExe @uvicornArgs
