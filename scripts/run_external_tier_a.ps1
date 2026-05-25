# Orchestrate Tier A external validation: prep, PredIG Docker, PRIME WSL, comparison.
# Usage: .\scripts\run_external_tier_a.ps1 [-Initials gb] [-SkipDocker] [-SkipPrime]

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

$RunDir = & (Join-Path $PSScriptRoot "init_external_run.ps1") -Initials $Initials -Scope "tierA"
$RunId = Split-Path $RunDir -Leaf
$script:LogFile = Join-Path (Join-Path $RunDir "logs") "commands.log"

Write-Log "SESTRAV Tier A external validation run_id=$RunId"

Write-Log "Checking freeze_status.json"
$freeze = Get-Content "results/freeze_status.json" | ConvertFrom-Json
if (-not $freeze.valid) {
    throw "freeze_status.json reports valid=false; aborting"
}

Write-Log "Running prepare_external_validation_inputs"
python -m src.prepare_external_validation_inputs

Write-Log "Building PredIG recombinant CSV"
python -m src.build_predig_recombinant_input

$HashManifest = Join-Path (Join-Path $RunDir "manifests") "input_hashes.json"
$files = @(
    "immunogenicity_dataset.csv",
    "results/external_validation_input.csv",
    "results/external_predig_input_recombinant.csv",
    "results/external_prime_peptides.txt"
)
$hashes = @{}
foreach ($f in $files) {
    if (Test-Path $f) {
        $hashes[$f] = (Get-FileHash -Algorithm SHA256 $f).Hash
    }
}
$hashes | ConvertTo-Json | Set-Content $HashManifest -Encoding UTF8

$PredigInput = "results/external_predig_input_recombinant.csv"
$PredigRaw = Join-Path (Join-Path $RunDir "raw") "predig_path_output.csv"
$PrimeRaw = Join-Path (Join-Path $RunDir "raw") "prime21_output.txt"
$AliasDir = "results/external_tool_outputs"
$PredigImage = if ($env:SESTRAV_PREDIG_IMAGE) { $env:SESTRAV_PREDIG_IMAGE } else { "bsceapm/predig@sha256:4a0c8b6b23a968600c4363290dc778a4b6e51cc24032d16ebfdb3119846b0a79" }
New-Item -ItemType Directory -Force -Path $AliasDir | Out-Null

if (-not $SkipDocker) {
    Write-Log "Attempting PredIG Docker run"
    $dockerOk = $false
    try {
        docker info 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $dockerOk = $true }
    } catch {
        Write-Log "Docker not available: $_"
    }

    if ($dockerOk) {
        Write-Log "docker pull $PredigImage"
        & docker pull $PredigImage 2>&1 | Tee-Object -FilePath $script:LogFile -Append

        $inputFile = $PredigInput
        if ($SmokeTestOnly) {
            $smokePath = Join-Path (Join-Path $RunDir "processed") "predig_smoke_input.csv"
            Import-Csv $PredigInput | Select-Object -First 50 | Export-Csv $smokePath -NoTypeInformation
            $inputFile = $smokePath
        }

        $resultsMount = (Resolve-Path "results").Path
        $inputLeaf = Split-Path $inputFile -Leaf
        $dockerArgs = @(
            "run", "--rm",
            "-v", "${resultsMount}:/predig",
            $PredigImage,
            "/predig/$inputLeaf",
            "--output", "/predig/external_tool_outputs/predig_path_output.csv",
            "--model", "path",
            "--type", "recombinant"
        )
        Write-Log ("docker " + ($dockerArgs -join " "))
        & docker @dockerArgs 2>&1 | Tee-Object -FilePath $script:LogFile -Append

        if (Test-Path "$AliasDir/predig_path_output.csv") {
            Copy-Item "$AliasDir/predig_path_output.csv" $PredigRaw -Force
            Write-Log "PredIG output saved to $PredigRaw"
        } else {
            Write-Log "WARNING: PredIG output not found; check Docker logs"
        }
    } else {
        Write-Log "SKIP PredIG: Docker unavailable"
    }
} else {
    Write-Log "SKIP PredIG: SkipDocker flag set"
}

if (-not $SkipPrime) {
    Write-Log "Attempting PRIME via WSL2"
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
        $wslScript = (Resolve-Path (Join-Path $PSScriptRoot "run_prime_tier_a_wsl.sh")).Path
        $wslScriptUnix = "/mnt/" + $wslScript.Substring(0,1).ToLower() + ($wslScript.Substring(2) -replace '\\','/')
        Write-Log "WSL PRIME invocation via run_prime_tier_a_wsl.sh"
        & wsl bash $wslScriptUnix $wslRepo 2>&1 | Tee-Object -FilePath $script:LogFile -Append

        if (Test-Path "$AliasDir/prime21_output.txt") {
            Copy-Item "$AliasDir/prime21_output.txt" $PrimeRaw -Force
            Write-Log "PRIME output saved to $PrimeRaw"
        } else {
            Write-Log "WARNING: PRIME output not found; install PRIME in WSL"
        }
    } else {
        Write-Log "SKIP PRIME: WSL unavailable"
    }
} else {
    Write-Log "SKIP PRIME: SkipPrime flag set"
}

$ProvPath = Join-Path (Join-Path $RunDir "manifests") "provenance.json"
@{
    run_id = $RunId
    created_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    predig_output = (Test-Path $PredigRaw)
    prime_output = (Test-Path $PrimeRaw)
} | ConvertTo-Json | Set-Content $ProvPath -Encoding UTF8
Copy-Item $ProvPath (Join-Path $AliasDir "provenance.json") -Force

Write-Log "Run directory: $RunDir"
Write-Output $RunDir
