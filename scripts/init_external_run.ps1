# Initialize run-scoped external validation directory (Windows PowerShell).
# Usage: .\scripts\init_external_run.ps1 [-Initials gb] [-Scope tierA]

param(
    [string]$Initials = "gb",
    [string]$Scope = "tierA"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$UtcNow = (Get-Date).ToUniversalTime()
$RunId = "extval_{0:yyyyMMdd_HHmm}_{1}_{2}" -f $UtcNow, $Initials, $Scope
$RunDir = Join-Path (Join-Path "results" "external_tool_outputs") $RunId

$Subdirs = @("raw", "processed", "logs", "figures", "manifests")
foreach ($sub in $Subdirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $RunDir $sub) | Out-Null
}

$ManifestPath = Join-Path (Join-Path $RunDir "manifests") "run_manifest.csv"
$CreatedUtc = $UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
@(
    "run_id=$RunId",
    "created_utc=$CreatedUtc",
    "scope=$Scope",
    "repo_root=$RepoRoot"
) | Set-Content -Path $ManifestPath -Encoding UTF8

Write-Host "RUN_ID=$RunId"
Write-Host "RUN_DIR=$RunDir"
Write-Output $RunDir
