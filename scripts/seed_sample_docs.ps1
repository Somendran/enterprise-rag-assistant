param(
    [string]$ApiUrl = "http://localhost:8000",
    [string]$ApiKey = $env:APP_API_KEY,
    [string]$SampleDir = "sample_docs"
)

$ErrorActionPreference = "Stop"

if (-not $ApiKey) {
    throw "APP_API_KEY or -ApiKey is required to seed sample docs."
}

$resolvedSampleDir = Resolve-Path -LiteralPath $SampleDir
$pdfs = Get-ChildItem -LiteralPath $resolvedSampleDir -Filter "*.pdf" -File
if (-not $pdfs) {
    throw "No PDF files found in $resolvedSampleDir."
}

$form = @{ files = @($pdfs) }

Invoke-RestMethod `
    -Uri "$($ApiUrl.TrimEnd('/'))/upload/jobs" `
    -Method Post `
    -Headers @{ "X-API-Key" = $ApiKey } `
    -Form $form | ConvertTo-Json -Depth 10
