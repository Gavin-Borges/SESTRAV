# Orchestrate Tier B external validation: prep, PredIG Docker, PRIME WSL, recovery.
# Usage: .\scripts\run_external_tier_b.ps1 [-Initials gb] [-SkipDocker] [-SkipPrime] [-SmokeTestOnly]

param(
    [string]$Initials = "gb",
    [switch]$SkipDocker,
    [switch]$SkipPrime,
    [switch]$SmokeTestOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Write-Log {
    param([string]$Msg)
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $line = "[$ts] $Msg"
    Write-Host $line
    if ($script:LogFile) {
        Add-Content -Path $script:LogFile -Value $line
    }
}

$RunDir = & (Join-Path $PSScriptRoot "init_external_run.ps1") -Initials $Initials -Scope "tierB"
$RunId = Split-Path $RunDir -Leaf
$script:LogFile = Join-Path (Join-Path $RunDir "logs") "commands.log"

Write-Log "SESTRAV Tier B external validation run_id=$RunId"

$freeze = Get-Content "results/freeze_status.json" | ConvertFrom-Json
if (-not $freeze.valid) {
    throw "freeze_status.json reports valid=false; aborting"
}

Write-Log "Preparing Tier B inputs (top 2000 + gold-standard union)"
python -m src.prepare_external_validation_tier_b_inputs

Write-Log "Building Tier B PredIG recombinant CSV"
python -m src.build_predig_recombinant_input `
    --pairs results/external_tier_b_predig_pairs.csv `
    --meta results/external_tier_b_peptide_meta.csv `
    --output results/external_tier_b_predig_input_recombinant.csv

$HashManifest = Join-Path (Join-Path $RunDir "manifests") "input_hashes.json"
$files = @(
    "results/external_tier_b_peptide_meta.csv",
    "results/external_tier_b_predig_pairs.csv",
    "results/external_tier_b_predig_input_recombinant.csv",
    "results/external_tier_b_all_peptides.txt",
    "results/external_tier_b_prefilter_manifest.json"
)
$hashes = @{}
foreach ($f in $files) {
    if (Test-Path $f) {
        $hashes[$f] = (Get-FileHash -Algorithm SHA256 $f).Hash
    }
}
$hashes | ConvertTo-Json | Set-Content $HashManifest -Encoding UTF8
Copy-Item "results/external_tier_b_prefilter_manifest.json" (Join-Path (Join-Path $RunDir "manifests") "tier_b_prefilter_manifest.json") -Force

$PredigInput = "results/external_tier_b_predig_input_recombinant.csv"
$PredigRaw = Join-Path (Join-Path $RunDir "raw") "predig_path_output.csv"
$PrimeRaw = Join-Path (Join-Path $RunDir "raw") "prime21_output.txt"
$AliasDir = "results/external_tool_outputs"
$PredigImage = if ($env:SESTRAV_PREDIG_IMAGE) { $env:SESTRAV_PREDIG_IMAGE } else { "bsceapm/predig@sha256:4a0c8b6b23a968600c4363290dc778a4b6e51cc24032d16ebfdb3119846b0a79" }
New-Item -ItemType Directory -Force -Path $AliasDir | Out-Null

if (-not $SkipDocker) {
    Write-Log "Attempting PredIG Docker run (batched)"
    $dockerOk = $false
    try {
        docker info 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $dockerOk = $true }
    } catch {
        Write-Log "Docker not available: $_"
    }

    if ($dockerOk) {
        $inputFile = $PredigInput
        if ($SmokeTestOnly) {
            $smokePath = Join-Path (Join-Path $RunDir "processed") "predig_smoke_input.csv"
            Import-Csv $PredigInput | Select-Object -First 50 | Export-Csv $smokePath -NoTypeInformation
            $inputFile = $smokePath
        }
        python scripts/run_predig_batched.py --input $inputFile --output "$AliasDir/tier_b_predig_path_output.csv" --image $PredigImage
        if (Test-Path "$AliasDir/tier_b_predig_path_output.csv") {
            Copy-Item "$AliasDir/tier_b_predig_path_output.csv" $PredigRaw -Force
            Write-Log "PredIG Tier B output saved to $PredigRaw"
        } else {
            Write-Log "WARNING: PredIG Tier B output not found"
        }
    } else {
        Write-Log "SKIP PredIG: Docker unavailable"
    }
} else {
    Write-Log "SKIP PredIG: SkipDocker flag set"
}

if (-not $SkipPrime) {
    Write-Log "Attempting PRIME via WSL2 (Tier B combined peptide list)"
    $wslOk = $false
    try {
        wsl --status 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $wslOk = $true }
    } catch {
        Write-Log "WSL not available: $_"
    }

    if ($wslOk) {
        $winPath = (Resolve-Path $RepoRoot).Path
        $drive = $winPath.Substring(0, 1).ToLower()
        $rest = $winPath.Substring(2) -replace "\\", "/"
        $wslRepo = "/mnt/$drive$rest"
        $alleles = Get-Content "results/external_tier_b_prime_alleles.txt" -Raw
        $alleles = $alleles.Trim()
        Write-Log "WSL PRIME Tier B (chunked script; run from PRIME2.1 root)"
        $wslScript = (Resolve-Path (Join-Path $PSScriptRoot "run_prime_tier_b_wsl.sh")).Path
        $wslScriptUnix = "/mnt/" + $wslScript.Substring(0,1).ToLower() + ($wslScript.Substring(2) -replace '\\','/')
        & wsl bash $wslScriptUnix $wslRepo 500 2>&1 | Tee-Object -FilePath $script:LogFile -Append

        if (Test-Path "$AliasDir/tier_b_prime21_output.txt") {
            Copy-Item "$AliasDir/tier_b_prime21_output.txt" $PrimeRaw -Force
            Write-Log "PRIME Tier B output saved to $PrimeRaw"
        } else {
            Write-Log "WARNING: PRIME Tier B output not found"
        }
    } else {
        Write-Log "SKIP PRIME: WSL unavailable"
    }
} else {
    Write-Log "SKIP PRIME: SkipPrime flag set"
}

if ((Test-Path $PredigRaw) -or (Test-Path $PrimeRaw)) {
    Write-Log "Running Tier B recovery analysis"
    python -m src.external_validation_tier_b_recovery --run-dir $RunDir
    if (-not (Test-Path $PrimeRaw)) {
        Write-Log "NOTE: PRIME Tier B missing; recovery used RF/binding/PredIG only"
    }
} else {
    Write-Log "SKIP recovery: no Tier B raw tool outputs"
}

Write-Log "Run directory: $RunDir"
Write-Output $RunDir
