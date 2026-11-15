# Saves actual multi-turn bot checks in project/test-results.
param(
    [ValidateSet('all','basic','paraphrases','boundaries','dialogues','robustness')]
    [string]$Group = 'basic',
    [ValidateRange(1,5)][int]$Repeat = 1,
    [string[]]$Ids,
    [string]$Scenarios = (Join-Path $PSScriptRoot 'bot_checks/scenarios.json'),
    [switch]$List
)
$ErrorActionPreference = 'Stop'
$Group = $Group.ToLowerInvariant()
if ($Ids) { $Ids = @($Ids | ForEach-Object { $_.Split(',') } | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
if ($Ids -and -not $PSBoundParameters.ContainsKey('Group')) { $Group = 'all' }
$apps = @(Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'energytool_hackathon') -Directory | Where-Object {
    Test-Path -LiteralPath (Join-Path $_.FullName 'start_local.ps1')
})
if ($apps.Count -ne 1) { throw 'Expected exactly one EnergyTool application' }
$appRoot = $apps[0].FullName
$pythonExe = Join-Path $appRoot '.venv/Scripts/python.exe'
$runArgs = @('-X','utf8','-m','tools.run_acceptance','--scenarios',(Resolve-Path -LiteralPath $Scenarios).Path,
             '--output-root',(Join-Path $PSScriptRoot 'test-results'),'--group',$Group,'--repeat',"$Repeat")
if ($Ids) { $runArgs += @('--ids') + $Ids }
if ($List) { $runArgs += '--list' }
else { & (Join-Path $appRoot 'start_local.ps1') -ServeOnly }
Push-Location $appRoot
try {
    & $pythonExe @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'Check failed. If a test-results folder was created, it contains partial results.' }
} finally {
    Pop-Location
}
