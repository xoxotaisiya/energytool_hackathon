# После завершения загрузок выполняет реальный прогон и сравнение, без заглушек.
param([int]$WaitMinutes = 120, [switch]$SkipServerRestart)
$ErrorActionPreference = 'Stop'
$appRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Join-Path $appRoot '.venv/Scripts/python.exe'
$reportDir = Join-Path $appRoot 'reports'
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$deadline = (Get-Date).AddMinutes($WaitMinutes)
while ($true) {
    if ((Get-Date) -gt $deadline) { throw 'Не завершилась загрузка моделей за отведённое время' }
    try {
        $tags = Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 5
        $names = @($tags.models.name)
        if ($names -contains 'qwen3:4b' -and $names -contains 'bge-m3:latest') { break }
    } catch {}
    Start-Sleep -Seconds 10
}
Push-Location $appRoot
try {
    if (-not $SkipServerRestart) {
        & (Join-Path $appRoot 'start_local.ps1') -ServeOnly -Restart
    }
    $env:OLLAMA_MODEL = 'qwen3:4b'
    $env:LLM_PROVIDER = 'ollama'
    $env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
    $env:LLM_TIMEOUT = '240'
    $env:RERANKER_ENABLED = '0'
    & $pythonExe -X utf8 -m tools.run_demo --mode bm25 --output reports/demo_bm25.json *> (Join-Path $reportDir 'demo-bm25.log')
    if ($LASTEXITCODE -ne 0) { throw 'Ошибка прогона BM25, см. reports/demo-bm25.log' }
    & $pythonExe -X utf8 -m tools.compare_retrieval --modes bm25 ollama hybrid-ollama *> (Join-Path $reportDir 'retrieval.log')
    if ($LASTEXITCODE -ne 0) { throw 'Ошибка сравнения поиска' }
    & $pythonExe -X utf8 -m tools.run_demo --mode ollama --output reports/demo_dense.json *> (Join-Path $reportDir 'demo-dense.log')
    if ($LASTEXITCODE -ne 0) { throw 'Ошибка прогона dense' }
    & $pythonExe -X utf8 -m tools.run_demo --mode hybrid-ollama --output reports/demo_hybrid.json *> (Join-Path $reportDir 'demo-hybrid.log')
    if ($LASTEXITCODE -ne 0) { throw 'Ошибка прогона hybrid' }
    $env:RERANKER_BACKEND = 'ollama'
    & $pythonExe -X utf8 -m tools.compare_retrieval --modes hybrid-ollama --rerank --sets demo --output reports/retrieval_llm_rerank.json *> (Join-Path $reportDir 'rerank.log')
    if ($LASTEXITCODE -ne 0) { throw 'Ошибка сравнения реранкера' }
    Set-Content -LiteralPath (Join-Path $reportDir 'validation.complete') -Value (Get-Date -Format o)
} finally {
    Pop-Location
}
