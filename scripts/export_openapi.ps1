Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
}
$env:PYTHONPATH = Join-Path $root "backend"

Set-Location $root
python -c "import json; from app.main import app; print(json.dumps(app.openapi(), indent=2))" > openapi.json
if ($LASTEXITCODE -ne 0) {
    if (Test-Path .\openapi.json) {
        Remove-Item -LiteralPath .\openapi.json -Force
    }
    throw "Failed to export OpenAPI schema. Install backend dependencies or run inside the project Python environment."
}
Write-Host "Wrote openapi.json"
