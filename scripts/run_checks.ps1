Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:PYTHONPATH = "backend"
python -m compileall backend\app evals\run_eval.py
python -m unittest discover -s backend\tests
python evals\run_eval.py

Set-Location (Join-Path $root "frontend")
npm run lint
npm run build
