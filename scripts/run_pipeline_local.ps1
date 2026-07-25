<#
.SYNOPSIS
    Local environment launcher wrapper for SESTRAV in Antigravity IDE / PowerShell.
.DESCRIPTION
    Wraps execution through Miniconda's 'sestrav' environment using conda.bat.
    Ensures CUDA acceleration (RTX 5070 Ti) and python environment variables are active.
.EXAMPLE
    .\scripts\run_pipeline_local.ps1 python pipeline.py
    .\scripts\run_pipeline_local.ps1 pytest
#>

[CmdletBinding()]
param (
    [Parameter(Position=0, ValueFromRemainingArguments=$true)]
    [string[]]$Command
)

$ErrorActionPreference = "Stop"

if (-not $Command -or $Command.Count -eq 0) {
    Write-Error "No command specified. Usage: .\scripts\run_pipeline_local.ps1 <command>"
    exit 1
}

# Resolve conda.bat dynamically (do not hardcode a user-specific install path).
$CondaBat = (Get-Command conda.bat -ErrorAction SilentlyContinue).Source
if (-not $CondaBat) {
    $Candidates = @(
        Join-Path $env:USERPROFILE "miniconda3\condabin\conda.bat"
        Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"
        Join-Path $env:ProgramData "miniconda3\condabin\conda.bat"
        Join-Path $env:ProgramData "anaconda3\condabin\conda.bat"
    )
    $CondaBat = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $CondaBat -or -not (Test-Path $CondaBat)) {
    Write-Error "Conda launcher (conda.bat) not found. Please verify your Miniconda/Anaconda installation is on PATH."
    exit 1
}

# Environment variables
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"

$cmdDisplay = $Command -join " "
Write-Host "[SESTRAV Local] Launching command in 'sestrav' environment..." -ForegroundColor Cyan
Write-Host "[SESTRAV Local] Executing: $cmdDisplay" -ForegroundColor DarkGray

# Execute directly via conda.bat run -n sestrav --no-capture-output
& $CondaBat run -n sestrav --no-capture-output $Command

$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "[SESTRAV Local] Command failed with exit code $exitCode" -ForegroundColor Red
    exit $exitCode
} else {
    Write-Host "[SESTRAV Local] Command completed successfully." -ForegroundColor Green
}
