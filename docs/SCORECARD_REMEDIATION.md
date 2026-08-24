# OSSF Scorecard & Security Remediation Guide

This guide tracks remediation for all alerts from the SESTRAV security audit.
Items marked ✅ are complete. Items marked ⬜ are not resolved; each row states
what remains, which is not always a GitHub UI action - it may be a code change
or an upstream tooling limitation.

Scores in the Alert column are as first filed and are NOT kept current. Rows 8
and 9 are the proof: License reads 9/10 there but measures 10 live, and
CII-Best-Practices reads 0/10 but measures 5. Read the Status cell, not the
parenthetical, for current state.

---

## Alert Status Summary

| # | Alert | Severity | Status |
|---|-------|----------|--------|
| 1 | keras CVEs (6×) - GHSA-36fq, 4f3f, cjgq, hjqc, mq84, 7gcm | **High** | ✅ Pinned & regenerated in `requirements.txt` |
| 2 | protobuf DoS - GHSA-m2f8 | **High** | ✅ Pinned & regenerated in `requirements.txt` |
| 3 | Semgrep false positive (`joblib_load` pattern) | Low | ✅ Fixed in `semgrep-rules/sestrav-custom.yml` |
| 4 | Branch-Protection (score 4/10) | **High** | ⬜ Ruleset `Protect Main Branch` (id 16846770) is active, but it is UI-managed and was not applied by the automation script - Step 2 retracted that attribution on 2026-08-22, and `scripts/apply-branch-ruleset.ps1` is untracked and disabled. The check still scores 4/10 live (verified 2026-08-23 at commit 36e3d8d, Scorecard v5.5.0: reason "branch protection is not maximal on development and all release branches"). Its three Warn details bind to live ruleset fields: admin-role bypass (`bypass_actors` RepositoryRole 5, always), `required_approving_review_count: 0`, and `require_last_push_approval: false` |
| 5 | Code-Review (score 0/10) | **High** | ⬜ Same retracted attribution as row 4, and the ruleset does not remediate this check: it sets `required_approving_review_count: 0`, so no approving review is required to merge. The check still scores 0/10 live (verified 2026-08-23 at commit 36e3d8d, Scorecard v5.5.0: reason "Found 0/4 approved changesets -- score normalized to 0"). Approving reviews do occur in this repo on bot-authored PRs (#278 and #280 each carry an owner APPROVED review), but Scorecard deliberately discards those: `probes/codeApproved/impl.go` skips approved changesets whose author is a bot, commented as skewing single-maintainer projects. The 4 changesets it did count are the owner's own merged PRs (#277, #282, #283, #284), which carry zero reviews |
| 6 | Pinned-Dependencies (score 9/10) | Medium | ⬜ The `scorecard-action` upgrade changed which scanner version runs, not what it measures, so it does not remediate this check. It still scores 8/10 live (verified 2026-08-23 at commit 36e3d8d, Scorecard v5.5.0: reason "dependency not pinned by hash detected -- score normalized to 8"; 26 of 34 pipCommand dependencies pinned). The warned sites are the two `pip install` steps in each of `Dockerfile`, `Dockerfile.api` and `Dockerfile.demo`, plus two in `.github/workflows/release.yml` (the release-check venv install, and the PyPI availability poll). All four files are byte-identical at HEAD to the measured commit. Line numbers are deliberately omitted here; Scorecard reports them per run and they rot |
| 7 | Fuzzing (score 0/10) | Medium | ⬜ `fuzzing.yml` runs genuine Hypothesis fuzz CI, but Scorecard's Fuzzing check still scores 0/10 live (verified 2026-08-23 at commit 36e3d8d, Scorecard v5.5.0: reason "project is not fuzzed"). Scorecard has no Hypothesis or property-based-Python detector; its only Python detector is Atheris, matched on `import atheris` (`languageFuzzSpecs`, `checks/raw/fuzzing.go` at ossf/scorecard c395761). Reachable via an Atheris harness, OSS-Fuzz, or ClusterFuzzLite |
| 8 | License (score 9/10) | Low | ✅ SPDX identifier added to `LICENSE` |
| 9 | CII-Best-Practices (score 0/10) | Low | ✅ OpenSSF Best Practices badge attained (project 13191, Passing) - embedded in `README.md` |
| 10 | PyTorch JIT script memory corruption (CVE-2025-3000) | Low | ✅ Resolved - upgraded to `torch==2.13.0`, the first patched release |

---

## ⬜ Manual Steps Required

### Step 1: Regenerate requirements.txt (dependency CVEs) - ✅ Complete

We have successfully regenerated `requirements.txt` from `requirements.in` using `pip-compile`.
At the time of that remediation the compiled file resolved `keras==3.14.1` (secure against
the then-known Keras CVEs) and `protobuf==7.35.0` (secure against the protobuf DoS CVE).
Both have since moved forward with routine dependency updates; **the authoritative pinned
versions are always `requirements.txt` and `environments/requirements.lock`, not the
figures recorded here.**

No conflicts with `mhcflurry` were encountered.

---

### Step 2: Branch Protection Ruleset (HIGH SEVERITY) - ⬜ Ruleset active, check still 4/10

The branch ruleset is managed in the GitHub web UI (Settings > Rules > Rulesets), which is
the source of truth. Read it with `gh api repos/Gavin-Borges/SESTRAV/rulesets/16846770`.

**Corrected 2026-08-22 against the live ruleset, having previously described a
configuration that is not the one in force.** This section used to credit
`apply-branch-ruleset.ps1` with applying the ruleset. That script is gitignored, so no
reader of this repository can inspect what it claims to have done, and it is now disabled
outright: its payload looked up the ruleset under the wrong name, so running it would have
created a *second* ruleset requiring a status check that nothing reports, blocking every
merge. Four further details below were wrong and are fixed: the ruleset name, the bypass
actor, the status-check context strings, and the omission of the code-scanning rule, which
is the rule that actually blocks a merge.

Rules in force (every field below read from the live API, not from the script's payload;
re-read 2026-08-24, when the fifth required status check was found missing from this list -
it was added to the ruleset 2026-08-22T19:14:42-04:00, about two hours after this section
was first written, so the list was accurate when written and stale thereafter):
- **ID / Name:** `16846770` / `Protect Main Branch`
- **Target:** `refs/heads/main`
- **Enforcement:** Active
- **Bypass Actors:** the **repository-admin role** (`actor_type: RepositoryRole`,
  `actor_id: 5`, Always bypass), not a named user account
- **Branch rules:**
  - ☑ Restrict deletions
  - ☑ Block force pushes
  - ☑ Require a pull request before merging (approvals count: 0 to allow solo-project self-approval bypass)
    - ☑ Dismiss stale pull request approvals when new commits are pushed
    - ☑ Require review from Code Owners
    - ☑ Require an additional approval for unattributed changes
  - ☑ Require status checks to pass before merging (strict policy: branch must be up-to-date)
    - Required status check: `test (3.13)`
    - Required status check: `Require human review`
    - Required status check: `check_dco`
    - Required status check: `Cited commits resolve`
    - Required status check: `Cited lines still hold their content`
  - ☑ **Require code scanning results: `CodeQL`**, `security_alerts_threshold: all`,
    `alerts_threshold: errors_and_warnings`. This is what makes CodeQL a merge blocker
    for any alert a pull request introduces, as distinct from the Bandit, semgrep and
    dependency-review jobs, which fail their own CI job but are not required checks and
    so do not gate the merge button. See `SECURITY.md`'s CI gate map.

Note the context strings: they are the bare check-run names (`test (3.13)`), not
workflow-qualified (`SESTRAV CI / test (3.13)`). A required context that no check reports
under stays pending forever, which blocks merges rather than protecting them.

---

### Step 3: Code Review Enforcement (HIGH SEVERITY) - ⬜ Workflow active, check still 0/10

The PR review workflow (`pr-review-check.yml`) is active and has been added as a required status check (`Require human review`) in the branch ruleset. This ensures:
- If you (the repository owner) open the PR, the workflow automatically passes (bypassing approval requirements) for frictionless self-merging.
- If an external contributor opens a PR, it strictly requires at least 1 approving review from a code owner to pass.

**Scorecard does NOT read this as a review constraint, and the earlier claim that it
"successfully detects" one is retracted.** Measured 2026-08-23 at commit 36e3d8d
(Scorecard v5.5.0): Code-Review scores 0/10, and Branch-Protection reports
`Warn: branch 'main' does not require approvers`. The ruleset sets
`required_approving_review_count: 0`, so no approving review is required to merge -
which is exactly what the summary table's row 5 now records. The owner-bypass behaviour
above is a deliberate solo-maintainer tradeoff, not something Scorecard credits.

---

### Step 4: OpenSSF Best Practices Badge (LOW SEVERITY - Score 0→5) - ✅ Complete

The project is registered at the OpenSSF Best Practices Badge Program as
**project 13191** and has attained the **Passing** level. The questionnaire was
answered with the evidence already present in the repository:
- ✅ Security policy: `SECURITY.md` exists
- ✅ Automated tests: `pytest` in CI
- ✅ Static analysis: Bandit, CodeQL, Semgrep in CI
- ✅ Pinned dependencies: `requirements.txt` with hashes

The live badge is embedded at the top of `README.md`:

```markdown
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/13191/badge)](https://www.bestpractices.dev/projects/13191)
```

---

## ✅ Completed Code-Level Changes

### Dependency CVE Pins (`requirements.in`)
Added explicit minimum version pins:
- `keras>=3.13.2` - patches all 6 keras CVEs
- `protobuf>=5.29.6` - patches protobuf DoS CVE

### Semgrep Rule Fix (`semgrep-rules/sestrav-custom.yml`)
Removed overly-broad `joblib_load(...)` pattern that was false-positively matching
`load_verified_joblib(...)` calls in `functions/stage4_immunogenicity_scoring.py`.

### PR Review Enforcement (`pr-review-check.yml`)
Changed `core.warning()` → `core.setFailed()` so the status check blocks merges for external contributors unless approved. It includes a specific check that automatically bypasses enforcement for the repository owner (sole maintainer) to allow frictionless self-merging without sacrificing external contributor review requirements.

### Scorecard Action Upgrade (`scorecard.yml`)
Upgraded `ossf/scorecard-action` from v2.3.1. Current pin, read from the
`uses: ossf/scorecard-action@...` step in `.github/workflows/scorecard.yml`
on 2026-08-23:
`ossf/scorecard-action@2d1146689b8cda280b9bc96326124645441f03bc # v2.4.4`,
which bundles Scorecard v5.5.0 - the version that produced the scores quoted in
the summary table above. This line previously recorded v2.4.3 / `4eaacf05...` /
Scorecard v5.3.0, which was accurate when written and is now stale in all three
values. Re-read the workflow rather than this paragraph if they disagree again.

### Fuzzing Workflow (`.github/workflows/fuzzing.yml`)
Added Hypothesis property-based fuzzing CI workflow. Runs `tests/test_fuzz.py` with
configurable `HYPOTHESIS_MAX_EXAMPLES` (200 on push, 1000 on weekly schedule).
Persists the Hypothesis failure database as a workflow artifact.

### License (`LICENSE`)
Added `SPDX-License-Identifier: MIT` at the top of the file. The SPDX tag is the
machine-readable identifier that Scorecard's License check uses to confirm OSI/FSF
recognition of the MIT license.

---

## Mitigated Upstream Vulnerabilities

### PyTorch Memory Corruption (CVE-2025-3000) - Resolved by upgrade

- **Vulnerability:** CVE-2025-3000 (`GHSA-rrmf-rvhw-rf47`, `PYSEC-2025-194`) is a memory
  corruption (segmentation fault) in PyTorch's `torch.jit.script` function, the
  underlying TorchScript compiler. GitHub rates it **low** severity. Affected versions
  are `<= 2.12.1`.
- **Remediation Status:** **RESOLVED.** PyTorch published `2.13.0`, the first patched
  release, on 2026-07-08. SESTRAV upgraded to `torch==2.13.0` on 2026-08-05 across
  `requirements.in`, `requirements.txt`, `environments/requirements.lock` and
  `environments/requirements-ci-torch-cpu.txt`, and floored `torch>=2.13.0` in
  `pyproject.toml` so that installs which do not use the compiled lockfiles cannot
  resolve back into the affected range. The advisory is no longer suppressed in
  `.github/workflows/security.yml`.
- **Prior position, retained for the record:** before the patch existed, this entry
  documented the advisory as mitigated-not-fixed on the grounds that SESTRAV never
  invokes or imports `torch.jit.script` (all PyTorch use is standard model definitions,
  dataset loading and feedforward inference), runs offline, and accepts no dynamically
  compiled input. That reachability argument still holds and is why the exposure was
  never urgent, but it is no longer the remediation: the upgrade is.
- **Process note:** this entry asserted "no upstream patched version" for four weeks
  after 2.13.0 shipped, because nothing re-evaluated it on a schedule. See the standing
  lesson in `SECURITY.md` - treat an advisory's `firstPatchedVersion` field as the
  authoritative test of patch availability, not prose in a checked-in document.

---

## Fuzzing - How to Run Locally

```bash
# Standard fuzz run (200 examples)
conda run -n sestrav pytest tests/test_fuzz.py -v

# Extended fuzz run (1000 examples, matches weekly CI)
HYPOTHESIS_MAX_EXAMPLES=1000 conda run -n sestrav pytest tests/test_fuzz.py -v
```
