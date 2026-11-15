# Real local validation; all model calls use the configured Ollama.
param()
$ErrorActionPreference = 'Stop'
$appRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Join-Path $appRoot '.venv/Scripts/python.exe'
$reportDir = Join-Path $appRoot 'reports'
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
Push-Location $appRoot
try {
    & (Join-Path $appRoot 'start_local.ps1') -ServeOnly
    $env:RERANKER_ENABLED = '0'
    & $pythonExe -X utf8 -m tools.run_demo --mode hybrid-ollama --output reports/demo_grounded_final.json *> (Join-Path $reportDir 'demo-grounded-final.log')
    if ($LASTEXITCODE -ne 0) { throw 'Demo failed; see reports/demo-grounded-final.log' }
    & $pythonExe -X utf8 -m tools.run_demo --mode hybrid-ollama --questions rag/eval_out.json --output reports/out_grounded_final.json *> (Join-Path $reportDir 'out-grounded-final.log')
    if ($LASTEXITCODE -ne 0) { throw 'Out-of-scope run failed; see reports/out-grounded-final.log' }
    & $pythonExe -X utf8 -m tools.compare_retrieval --modes bm25 ollama hybrid-ollama --output reports/retrieval_grounded_final.json *> (Join-Path $reportDir 'retrieval-grounded-final.log')
    if ($LASTEXITCODE -ne 0) { throw 'Retrieval comparison failed' }
    & $pythonExe -X utf8 -m tools.build_submission *> (Join-Path $reportDir 'submission-build.json')
    if ($LASTEXITCODE -ne 0) { throw 'Submission build failed' }
    Set-Content -LiteralPath (Join-Path $reportDir 'grounded-validation.complete') -Value (Get-Date -Format o)
} finally {
    Pop-Location
}
