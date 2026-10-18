<#
.SYNOPSIS
    Land one prepared unit: verify it, push it and open its PR if needed, and arm
    auto-merge pinned to the exact commit that was prepared.

.DESCRIPTION
    The owner's one-touch landing command. An assistant prepares the unit with
    scripts/push_unit.py, which writes _local/landing/<UnitId>.json in the
    primary checkout. This script:
      1. loads that record and enters the recorded worktree;
      2. refuses unless HEAD is exactly expected_head_sha, the branch matches the
         record, and nothing is modified or untracked;
      3. fetches the base and refuses if the branch is behind it, because the
         ruleset's up-to-date rule would hold the merge and updating the branch
         changes its head;
      4. pushes HEAD if the remote branch does not hold it yet (never forced),
         and opens the PR if there is none;
      5. prints the PR URL, the commit, the reviewer and the diffstat;
      6. arms auto-merge:
           gh pr merge <n> -R <owner/repo> --auto --merge --delete-branch
             --match-head-commit <expected_head_sha>
         run from outside any checkout, so gh has no local branch to switch to
         or delete; the branch is checked out in the unit's worktree.

    Exit codes: 0 done; 3 refused (nothing armed); 5 a git or gh step failed.

.PARAMETER UnitId
    The unit ID given to push_unit.py --unit.

.PARAMETER FullLocalPytest
    If this script pushes, run pre-push's whole test suite (SESTRAV_FULL_PREPUSH=1)
    instead of its fast mode.

.PARAMETER NoAutoMerge
    Stop after printing the summary, and print the command that would arm
    auto-merge.

.EXAMPLE
    land-unit A9
.EXAMPLE
    land-unit A9 -FullLocalPytest -NoAutoMerge
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')]
    [string]$UnitId,
    [switch]$FullLocalPytest,
    [switch]$NoAutoMerge
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$PrFields = 'number,state,url,headRefName,headRefOid,baseRefName,isDraft,title'

function Stop-Refused([string]$Message) {
    Write-Host "[land-unit] REFUSED: $Message" -ForegroundColor Red
    exit 3
}

function Stop-Failed([string]$Message) {
    Write-Host "[land-unit] FAILED: $Message" -ForegroundColor Red
    exit 5
}

# Runs a native command and returns its stdout as one string. stderr is left on
# the console. Windows PowerShell 5.1 does not stop on a native exit code, so
# every call checks it here.
function Invoke-Native([string]$Exe, [string[]]$Arguments, [switch]$AllowFailure) {
    $output = & $Exe @Arguments
    $code = $LASTEXITCODE
    if ($code -ne 0 -and -not $AllowFailure) {
        Stop-Failed "$Exe $($Arguments -join ' ') exited $code"
    }
    return [pscustomobject]@{ Code = $code; Text = ((@($output) -join "`n").Trim()) }
}

function Get-Field($Object, [string]$Name) {
    if ($null -ne $Object -and $Object.PSObject.Properties.Name -contains $Name) {
        return $Object.$Name
    }
    return $null
}

function Get-PullRequest([string]$Selector) {
    $view = Invoke-Native 'gh' @('pr', 'view', $Selector, '--json', $PrFields) -AllowFailure
    if ($view.Code -ne 0 -or -not $view.Text) { return $null }
    return ($view.Text | ConvertFrom-Json)
}

function Find-OpenPullRequest([string]$Branch) {
    $list = Invoke-Native 'gh' @('pr', 'list', '--head', $Branch, '--state', 'open', '--json', $PrFields)
    $found = @()
    if ($list.Text) { $found = @($list.Text | ConvertFrom-Json) }
    if ($found.Count -eq 0) {
        # An empty answer is not proof of absence: cross-check every open PR.
        $all = Invoke-Native 'gh' @('pr', 'list', '--state', 'open', '--limit', '200', '--json', $PrFields)
        if ($all.Text) { $found = @($all.Text | ConvertFrom-Json | Where-Object { $_.headRefName -eq $Branch }) }
    }
    if ($found.Count -gt 0) { return $found[0] }
    return $null
}

function Save-Record($Record, [string]$Path) {
    $json = $Record | ConvertTo-Json -Depth 4
    [IO.File]::WriteAllText($Path, $json + "`n", (New-Object System.Text.UTF8Encoding($false)))
}

# ---- 1. the record and the worktree -----------------------------------------------
$common = Invoke-Native 'git' @('-C', $PSScriptRoot, 'rev-parse', '--path-format=absolute', '--git-common-dir') -AllowFailure
if ($common.Code -ne 0) { Stop-Refused "this script is not inside a git checkout ($PSScriptRoot)" }
$Primary = Split-Path -Parent ([IO.Path]::GetFullPath($common.Text))
$RecordPath = Join-Path $Primary "_local\landing\$UnitId.json"
if (-not (Test-Path -LiteralPath $RecordPath)) {
    Stop-Refused "no landing record at $RecordPath. Prepare the unit with scripts/push_unit.py first."
}
$Record = Get-Content -LiteralPath $RecordPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($field in @('unit_id', 'branch', 'base', 'remote', 'worktree_path', 'pr_title', 'pr_body_path', 'expected_head_sha')) {
    if (-not (Get-Field $Record $field)) { Stop-Refused "the record lacks '$field'" }
}
if ($Record.unit_id -ne $UnitId) { Stop-Refused "the record is for unit '$($Record.unit_id)', not '$UnitId'" }
$Expected = [string]$Record.expected_head_sha
if ($Expected -notmatch '^[0-9a-f]{40}$') { Stop-Refused "expected_head_sha '$Expected' is not a full commit id" }
$Worktree = [IO.Path]::GetFullPath((Join-Path $Primary $Record.worktree_path))
$BodyFile = [IO.Path]::GetFullPath((Join-Path $Primary $Record.pr_body_path))
if (-not (Test-Path -LiteralPath $Worktree -PathType Container)) { Stop-Refused "worktree $Worktree does not exist" }
$BaseRef = "$($Record.remote)/$($Record.base)"

Push-Location -LiteralPath $Worktree
try {
    # ---- 2. exactly the prepared commit, clean ------------------------------------
    $head = (Invoke-Native 'git' @('rev-parse', 'HEAD')).Text
    if ($head -ne $Expected) {
        Stop-Refused "HEAD is $head but the record says $Expected. The unit changed after it was prepared; run push_unit.py again."
    }
    $branch = Invoke-Native 'git' @('symbolic-ref', '--quiet', '--short', 'HEAD') -AllowFailure
    if ($branch.Code -ne 0 -or $branch.Text -ne $Record.branch) {
        Stop-Refused "the worktree is on '$($branch.Text)', not '$($Record.branch)'"
    }
    $dirty = (Invoke-Native 'git' @('status', '--porcelain=v1', '--untracked-files=normal')).Text
    if ($dirty) { Stop-Refused "the worktree is not clean:`n$dirty" }

    # ---- 3. up to date with the base ----------------------------------------------
    Invoke-Native 'git' @('fetch', '--quiet', $Record.remote, $Record.base) | Out-Null
    $counts = (Invoke-Native 'git' @('rev-list', '--left-right', '--count', "$BaseRef...HEAD")).Text -split '\s+'
    if ([int]$counts[1] -eq 0) { Stop-Refused "HEAD has no commits that $BaseRef lacks" }
    if ([int]$counts[0] -gt 0) {
        Stop-Refused "the branch is $($counts[0]) commit(s) behind $BaseRef. It cannot merge until it is updated, which changes its head; update it and run push_unit.py again."
    }

    # ---- 4. pushed, with an open PR -----------------------------------------------
    $remoteLine = (Invoke-Native 'git' @('ls-remote', $Record.remote, "refs/heads/$($Record.branch)")).Text
    $remoteSha = ''
    if ($remoteLine) { $remoteSha = ($remoteLine -split "\s+")[0] }
    if ($remoteSha -ne $head) {
        if ($remoteSha) {
            $ancestor = Invoke-Native 'git' @('merge-base', '--is-ancestor', $remoteSha, $head) -AllowFailure
            if ($ancestor.Code -ne 0) {
                Stop-Refused "$($Record.remote)/$($Record.branch) is at $remoteSha, which the prepared head does not contain. Someone else pushed; re-prepare the unit."
            }
        }
        $priorFull = $env:SESTRAV_FULL_PREPUSH
        if ($FullLocalPytest) { $env:SESTRAV_FULL_PREPUSH = '1' }
        try {
            Write-Host "[land-unit] Pushing $($head.Substring(0, 10)) to $($Record.remote)/$($Record.branch) (pre-push runs)..."
            # Not through Invoke-Native: the hook's progress goes to stdout and should be seen.
            & git push --set-upstream $Record.remote "HEAD:refs/heads/$($Record.branch)"
            if ($LASTEXITCODE -ne 0) { Stop-Failed "git push exited $LASTEXITCODE; nothing was forced" }
        }
        finally {
            if ($null -eq $priorFull) { Remove-Item Env:SESTRAV_FULL_PREPUSH -ErrorAction SilentlyContinue }
            else { $env:SESTRAV_FULL_PREPUSH = $priorFull }
        }
    }
    elseif ($FullLocalPytest) {
        Write-Host "[land-unit] note: the remote already holds $($head.Substring(0, 10)), so nothing was pushed and -FullLocalPytest had no effect."
    }

    $pr = $null
    $knownNumber = Get-Field $Record 'pr_number'
    if ($knownNumber) { $pr = Get-PullRequest ([string]$knownNumber) }
    if ($null -eq $pr -or $pr.state -ne 'OPEN') { $pr = Find-OpenPullRequest $Record.branch }
    if ($null -eq $pr) {
        if (-not (Test-Path -LiteralPath $BodyFile)) { Stop-Refused "PR body file $BodyFile does not exist" }
        $created = Invoke-Native 'gh' @('pr', 'create', '--base', $Record.base, '--head', $Record.branch, '--title', $Record.pr_title, '--body-file', $BodyFile)
        $url = @($created.Text -split "`n")[-1].Trim()
        $pr = Get-PullRequest $url
        if ($null -eq $pr) { Stop-Failed "created a PR ($url) but could not read it back" }
    }
    for ($attempt = 0; $attempt -lt 10 -and $pr.headRefOid -ne $head; $attempt++) {
        Start-Sleep -Seconds 3   # GitHub can lag a push by a few seconds
        $pr = Get-PullRequest ([string]$pr.number)
    }
    if ($pr.state -ne 'OPEN') { Stop-Refused "PR #$($pr.number) is $($pr.state)" }
    if ($pr.baseRefName -ne $Record.base) { Stop-Refused "PR #$($pr.number) targets $($pr.baseRefName), not $($Record.base)" }
    if ($pr.isDraft) { Stop-Refused "PR #$($pr.number) is a draft; mark it ready yourself if it is" }
    if ($pr.headRefOid -ne $head) { Stop-Refused "PR #$($pr.number) reports head $($pr.headRefOid), not $head" }

    if ((Get-Field $Record 'pr_number') -ne $pr.number -or (Get-Field $Record 'pr_url') -ne $pr.url) {
        $Record | Add-Member -NotePropertyName pr_number -NotePropertyValue $pr.number -Force
        $Record | Add-Member -NotePropertyName pr_url -NotePropertyValue $pr.url -Force
        $Record | Add-Member -NotePropertyName pushed -NotePropertyValue $true -Force
        Save-Record $Record $RecordPath
    }

    # ---- 5. what is about to land ---------------------------------------------------
    $reviewer = Get-Field $Record 'reviewer'
    if (-not $reviewer) { $reviewer = 'NONE RECORDED (unreviewed)' }
    Write-Host ''
    Write-Host "Unit      : $UnitId"
    Write-Host "PR        : #$($pr.number)  $($pr.url)"
    Write-Host "Title     : $($pr.title)"
    Write-Host "Branch    : $($Record.branch) -> $($Record.base)"
    Write-Host "Commit    : $head"
    Write-Host "Reviewer  : $reviewer"
    Write-Host 'Diffstat  :'
    Write-Host (Invoke-Native 'git' @('diff', '--stat', "$BaseRef...HEAD")).Text
    Write-Host ''

    $repo = (Invoke-Native 'gh' @('repo', 'view', '--json', 'nameWithOwner', '-q', '.nameWithOwner')).Text
    $mergeArgs = @('pr', 'merge', [string]$pr.number, '-R', $repo, '--auto', '--merge', '--delete-branch', '--match-head-commit', $head)
    if ($NoAutoMerge) {
        Write-Host "[land-unit] -NoAutoMerge: nothing armed. To arm it later:"
        Write-Host "  gh $($mergeArgs -join ' ')"
        exit 0
    }
}
finally {
    Pop-Location
}

# ---- 6. arm auto-merge, from outside any checkout ------------------------------------
Push-Location -LiteralPath ([IO.Path]::GetTempPath())
try {
    & gh @mergeArgs
    if ($LASTEXITCODE -ne 0) { Stop-Failed "gh $($mergeArgs -join ' ') exited $LASTEXITCODE; auto-merge is NOT armed" }
    $after =Invoke-Native 'gh' @('pr', 'view', [string]$pr.number, '-R', $repo, '--json', 'state,autoMergeRequest,mergeStateStatus') -AllowFailure
}
finally {
    Pop-Location
}
Write-Host "[land-unit] Auto-merge armed for PR #$($pr.number) at $head."
if ($after.Text) { Write-Host "[land-unit] $($after.Text)" }
Write-Host '[land-unit] It merges when the required checks pass, and only while the head is still that commit.'
exit 0
