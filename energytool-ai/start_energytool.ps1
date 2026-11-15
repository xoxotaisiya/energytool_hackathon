param([switch]$ServeOnly, [switch]$Pull, [switch]$Restart)
$ErrorActionPreference = 'Stop'
$apps = @(Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'energytool_hackathon') -Directory | Where-Object {
    Test-Path -LiteralPath (Join-Path $_.FullName 'start_local.ps1')
})
if ($apps.Count -ne 1) { throw 'Нужно выбрать единственную рабочую папку energytool_hackathon со start_local.ps1' }
& (Join-Path $apps[0].FullName 'start_local.ps1') @PSBoundParameters
