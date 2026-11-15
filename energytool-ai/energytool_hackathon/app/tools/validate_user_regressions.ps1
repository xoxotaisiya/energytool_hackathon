# Runs local regressions, including a cold model and independent warm repeats.
param([ValidateRange(1,5)][int]$WarmRepeats = 3)
$ErrorActionPreference = 'Stop'
$appRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Join-Path $appRoot '.venv/Scripts/python.exe'
Push-Location $appRoot
try {
    New-Item -ItemType Directory -Force -Path reports | Out-Null
    & $pythonExe -X utf8 -m unittest discover -s tests -q *> reports/unit-policy-reproduced.log
    if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed' }
    & (Join-Path $appRoot 'start_local.ps1') -ServeOnly
    & $pythonExe -X utf8 -m tools.run_demo --mode hybrid-ollama --questions tests/eval_user_regressions.json --label full-first-request-cold --unload-model --output reports/user_full_reproduced.json *> reports/user-full-reproduced.log
    if ($LASTEXITCODE -ne 0) { throw 'Real model run failed' }
    foreach ($round in 1..$WarmRepeats) {
        & $pythonExe -X utf8 -m tools.run_demo --mode hybrid-ollama --questions tests/eval_user_regressions.json --ids u01 --label ('warm-' + $round) --output ('reports/user_warm_reproduced_' + $round + '.json') *> ('reports/user-warm-reproduced-' + $round + '.log')
        if ($LASTEXITCODE -ne 0) { throw 'Warm model run failed' }
    }
    & $pythonExe -X utf8 -m tools.build_submission *> reports/submission-build.json
    if ($LASTEXITCODE -ne 0) { throw 'Submission build failed' }
} finally {
    Pop-Location
}
