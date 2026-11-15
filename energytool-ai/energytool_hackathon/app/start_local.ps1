param([switch]$ServeOnly, [switch]$Pull, [switch]$Restart)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$runtimeRoot = Join-Path $projectRoot '.runtime'
$ollamaExe = Join-Path $runtimeRoot 'ollama/ollama.exe'
$pythonExe = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if ($env:ENERGYTOOL_OLLAMA_EXE) { $ollamaExe = $env:ENERGYTOOL_OLLAMA_EXE }
if (-not (Test-Path -LiteralPath $ollamaExe)) {
    $installed = Get-Command ollama -ErrorAction SilentlyContinue
    if ($installed) { $ollamaExe = $installed.Source }
    else { $ollamaExe = $null }
}
if ($ollamaExe -and $ollamaExe.StartsWith($runtimeRoot, [StringComparison]::OrdinalIgnoreCase)) {
    $env:OLLAMA_MODELS = Join-Path $runtimeRoot 'models'
}
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MAX_LOADED_MODELS = '2'
$env:OLLAMA_NUM_PARALLEL = '1'
# GPU selection is left to Ollama and the machine's configuration.
New-Item -ItemType Directory -Force -Path (Join-Path $runtimeRoot 'logs') | Out-Null
if ($Restart) {
    if (Get-ChildItem (Join-Path $runtimeRoot 'models/blobs/*-partial') -ErrorAction SilentlyContinue) {
        throw 'Не перезапускайте сервер до завершения загрузок моделей'
    }
    $pidFile = Join-Path $runtimeRoot 'server.pid'
    if (Test-Path -LiteralPath $pidFile) {
        $serverPid = [int](Get-Content -LiteralPath $pidFile)
        $owned = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
        if ($owned) {
            if ([IO.Path]::GetFullPath($owned.Path) -ne [IO.Path]::GetFullPath($ollamaExe)) { throw 'PID принадлежит другому процессу' }
            # Освобождаем загруженные runner-процессы перед перезапуском сервера.
            try {
                $loaded = Invoke-RestMethod 'http://127.0.0.1:11434/api/ps' -TimeoutSec 5
                foreach ($model in $loaded.models) { & $ollamaExe stop $model.name }
            } catch {}
            Stop-Process -Id $serverPid
            Start-Sleep -Seconds 1
        }
    }
}
$ready = $false
try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 | Out-Null; $ready = $true } catch {}
if (-not $ready) {
    if (-not $ollamaExe) { throw 'Install Ollama, start its server, or set ENERGYTOOL_OLLAMA_EXE to ollama.exe.' }
    $startArgs = @{
        FilePath = $ollamaExe
        ArgumentList = 'serve'
        WindowStyle = 'Hidden'
        PassThru = $true
        RedirectStandardOutput = (Join-Path $runtimeRoot 'logs/server.stdout.log')
        RedirectStandardError = (Join-Path $runtimeRoot 'logs/server.stderr.log')
    }
    $service = Start-Process @startArgs
    Set-Content -LiteralPath (Join-Path $runtimeRoot 'server.pid') -Value $service.Id
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 | Out-Null; $ready = $true; break } catch {}
    }
    if (-not $ready) { throw 'Ollama не запустилась. См. project/.runtime/logs/server.stderr.log' }
}
if ($Pull) {
    if (-not $ollamaExe) { throw 'For -Pull, install the Ollama CLI or set ENERGYTOOL_OLLAMA_EXE.' }
    $modelToPull = 'qwen3:4b-instruct-2507-q4_K_M'
    $localEnv = Join-Path $PSScriptRoot '.env'
    if (Test-Path -LiteralPath $localEnv) {
        $modelLine = Get-Content -LiteralPath $localEnv -Encoding UTF8 | Where-Object { $_ -match '^\s*OLLAMA_MODEL\s*=' } | Select-Object -Last 1
        if ($modelLine) { $modelToPull = (($modelLine -split '=', 2)[1]).Trim().Trim('"').Trim("'") }
    }
    if ($env:OLLAMA_MODEL) { $modelToPull = $env:OLLAMA_MODEL }
    & $ollamaExe pull $modelToPull
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось загрузить Qwen' }
    & $ollamaExe pull bge-m3
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось загрузить BGE-M3' }
}
if (-not $ServeOnly) {
    if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Создайте .venv и установите requirements.txt' }
    Push-Location $PSScriptRoot
    try { & $pythonExe -X utf8 main.py } finally { Pop-Location }
}
