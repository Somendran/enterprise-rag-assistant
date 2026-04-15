Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    . $venvActivate
}

Set-Location (Join-Path $root "backend")
python -m uvicorn app.main:app --reload --port 8000
