# Security Policy

## Overview

SESTRAV is a solo-maintained, academic research project developed at the University
of Rhode Island. There is one active maintainer. Please keep this in mind when
setting expectations around response timelines.

## Supported Versions

Only the latest major release of SESTRAV is actively supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

**Please do NOT open a public GitHub Issue for security vulnerabilities.**
Public disclosure before a fix is available puts all users at risk.

### Preferred: Private Email Report

Send a confidential report to the maintainer directly:

**Email:** `gavinmborges1104@gmail.com`

Please include the following in your report:

- **Summary:** A clear description of the vulnerability and its potential impact.
- **Reproduction steps:** Step-by-step instructions to reproduce the issue,
  including any sample inputs, scripts, or configuration needed.
- **Environment:** Python version, OS, relevant package versions.
- **Proposed fix (optional):** Any remediation ideas or patches you have.

### Alternative: GitHub Private Vulnerability Reporting

GitHub's built-in private reporting is also available:

1. Navigate to the [SESTRAV Security tab](https://github.com/Gavin-Borges/SESTRAV/security).
2. Click **"Report a vulnerability"**.
3. Fill in the advisory form - this is end-to-end encrypted between you and the maintainer.

## Response Commitment

As a solo-maintained project, the maintainer commits to:

- **Acknowledge** receipt of your report within **3-5 business days**.
- **Provide an initial assessment** (severity, scope, reproducibility) within
  **10 business days** of acknowledgement.
- **Coordinate a fix and disclosure timeline** with you collaboratively.
- **Credit reporters** in the release notes (unless you prefer anonymity).

## Thank You

Responsible disclosure helps keep SESTRAV and its users safe.
Thank you for taking the time to report vulnerabilities privately.

## Maintainership & Continuity

SESTRAV is led by a single active maintainer; the v1 collaborators credited in
`CONTRIBUTORS.md` are no longer active. **There is no backup maintainer and no
succession arrangement**, and none is planned - `GOVERNANCE.md` and `BUS_FACTOR.md`
record that honest status rather than a continuity plan.

The project therefore **cannot** survive the loss of the maintainer as a *maintained*
project. What it does have is weaker but real: the source, full history and build
procedure are public and MIT-licensed, so a third party may fork and continue the work
independently. If you are assessing whether a security report will be actioned in the
maintainer's absence, assume it will not be.

### Account Security
All accounts with write access to this repository **MUST have Two-Factor
Authentication (2FA)** enabled on GitHub. In practice that is one account, the sole
maintainer's.

## Release Integrity & Verification

Release artifacts are distributed via GitHub Releases over HTTPS with a
**SHA-256 checksum manifest**, `SHA256SUMS.txt`, written by the "Generate SHA-256
checksum manifest" step of `.github/workflows/release.yml`, which lets consumers
verify **integrity**:

```bash
sha256sum -c SHA256SUMS.txt   # run in the directory holding the files it lists
```

What it lists depends on the release. For the two shipped releases, v2.0.2 and
v2.0.3, it lists the wheel and the sdist, which with `SHA256SUMS.txt` itself are
their only uploaded assets (GitHub's auto-generated source archives are covered by
neither the checksums nor the attestation). The current workflow, which no tag has
run yet, writes it over `dist/` (the sdist, the wheel and the `.intoto.jsonl`
attestation bundle) and also uploads a results-bundle ZIP and its
`*.manifest.json`, neither of which `SHA256SUMS.txt` lists. That JSON manifest is
written by `src/release_bundle.py` and records the SHA-256 of each `results/` file
packed into the ZIP; it is not in `sha256sum` format, so `sha256sum -c` cannot
check it. The ZIP is covered by the provenance attestation below; the JSON
manifest is covered only through the copy packed inside the ZIP. **`SHA256SUMS.txt`
itself carries no attestation**, so the attestation below does not cover every
uploaded asset: `gh attestation verify` on it returns HTTP 404 for its digest,
while the wheel and the sdist each resolve one (read 2026-10-02, v2.0.3).

Release **authenticity** (signing) is automated by
[`.github/workflows/release.yml`](.github/workflows/release.yml): pushing a version
tag builds the distribution and produces a **keyless SLSA build-provenance
attestation** (Sigstore, via GitHub OIDC - no maintainer-managed keys). From the
release at which signing is introduced onward:

- **Tags**: v2.0.3 IS signed with an SSH key and carries a signature block;
  v2.0.2 and earlier are annotated but unsigned. `version_tags_signed` is
  nevertheless still Unmet, for a different reason than "no signed tag exists":
  GitHub reports v2.0.3 as unverified with reason `unknown_key`, because the SSH
  key that signed it is not the signing key registered on the account. A signing
  key IS registered, and GitHub reports commits signed with it as verified, so the
  next tag should be signed with that registered key. No tag has been signed with
  it yet, so a verified tag is expected, not measured. See `docs/releasing.md`.
- **Artifacts** carry a Sigstore provenance attestation, verifiable with:

  ```bash
  gh attestation verify sestrav-<version>-py3-none-any.whl --repo Gavin-Borges/SESTRAV
  ```

The full release procedure is documented in [`docs/releasing.md`](docs/releasing.md).

> Maintainer note: artifact attestation is keyless, so there is no artifact-signing
> key to record. When tag signing is introduced, note the first signed version here
> and record the signing identity per `BUS_FACTOR.md`.

---

## Personally Identifiable Information (PII) & Patient Health Data (PHI)

SESTRAV is designed as a standalone, offline bioinformatics pipeline. 

- **Data Privacy by Design**: All scoring and feature extraction processes execute strictly on the local machine or host environment. SESTRAV does not transmit, upload, or collect any sequence data, user parameters, or predictive outputs.
- **No PII/PHI Requirement**: The pipeline accepts standard FASTA, CSV, and YAML configurations. It does not require, accept, or process personally identifiable information (PII) or protected health information (PHI). Users are cautioned against introducing patient metadata or identifying fields into sequence inputs.
- **Credential Safety**: The pipeline does not connect to external patient databases and has no credential store or public telemetry APIs.

---

## Vulnerability Triage & Remediation Policy

How findings from Dependabot, code scanning, and the CI security workflows are
prioritized and acted on. The policy is deliberately **two-tiered**, but "blocking"
here means one specific, verifiable thing: gating the merge button via the
`Protect Main Branch` ruleset (id `16846770`) - its **seven** required status checks
(`test (3.13)`, `Require human review`, `check_dco`, `Cited commits resolve`,
`Cited lines still hold their content`, `Bandit Security Scan` and
`CodeQL Static Analysis`) plus its code-scanning rule, which names `CodeQL`. A tool
that is not on either of those two lists cannot block a merge, even if it fails its
own CI job red.

**One of those seven is satisfied automatically for the maintainer, and this list
alone would not tell you.** `Require human review` is a genuine required context, but
`.github/workflows/pr-review-check.yml` returns success without any review when the
pull request author is the repository owner, and the ruleset's
`required_approving_review_count` is **0**, so nothing else supplies one.
`require_code_owner_review` does not either: `.github/CODEOWNERS` is `* @Gavin-Borges`
and GitHub does not let an author approve their own pull request.

This is a **declared accommodation, not a hole**, and the workflow says so in its own
header: it enforces review for EXTERNAL contributors only, and records that it "does
NOT establish independent review for owner-authored changes, which are the large
majority of merges. SESTRAV is solo-maintained; the required CI checks, not a second
reviewer, are the real gate on those changes." `BUS_FACTOR.md` and
`docs/security_compliance.md` record the same position - OpenSSF Silver and Gold are
formally declined on `bus_factor` / `two_person_review` grounds as of 2026-08-17.
Noted here because a reader of the enumeration above, without this paragraph, would
conclude owner merges are human-reviewed. They are CI-reviewed.

**It is satisfied automatically for two dependency bots as well.** The same workflow
returns success, again before reading any review, when the pull request author's login
is exactly one of the entries in its `TRUSTED_BOTS` list: `dependabot[bot]` and
`renovate[bot]`. Its own comment rests that bypass on the CI suite enforcing
correctness, so a bot pull request, like an owner one, is CI-reviewed rather than
human-reviewed. For every
other author the check counts only each reviewer's latest non-comment review, and
passes only on an `APPROVED` one from an account whose `author_association` is `OWNER`,
`MEMBER` or `COLLABORATOR` and which is not the pull request author. A second required
check has a wider exemption: `.github/workflows/dco.yml` exits 0 before checking any
sign-off when the author's login ends in `[bot]`, so `check_dco` passes for any bot
account, not only these two.

**Corrected 2026-09-13.** The opening paragraph of this section read "its five required
status checks" and enumerated five, omitting `Bandit Security Scan` and
`CodeQL Static Analysis`. That understated enforcement, and it contradicted this
document's own CI gate map below, which was corrected on 2026-09-12 to record Bandit as
blocking. Both statements were true of different dates and neither said so. **Do not
quote this enumeration as current** - the count has already changed at least twice (five
as of 2026-08-24, seven as of 2026-09-13). Re-measure instead:

```bash
gh api repos/Gavin-Borges/SESTRAV/rulesets/16846770 \
  --jq '[.rules[] | select(.type=="required_status_checks")
         | .parameters.required_status_checks[].context] | sort'
```

Note the classic branch-protection API is a confident false negative here:
`gh api .../branches/main/protection` returns 404 "Branch not protected", because
this repository uses a ruleset rather than legacy branch protection.

### Severity -> action (target SLA)

| Severity | Action | Target SLA | Gate |
| -------- | ------ | ---------- | ---- |
| Critical | Patch or documented mitigation before the next merge to `main` | 48 hours | **Blocking** |
| High     | Patch or mitigation within the current release cycle | 7 days | **Blocking** |
| Medium   | Scheduled fix; risk-acceptable with written justification | 30 days | Advisory |
| Low      | Best-effort; batched with routine dependency updates | Next cycle | Advisory |

Timelines follow the solo-maintainer cadence described under **Response Commitment** above.
The Severity column states intent; the CI gate map below states what is
mechanically enforced today, and the two are not yet the same for every row.

### CI gate map

Every job in `security.yml`, `dependency-review.yml`, `pii_scan.yml`,
`dependency-lockfile-check.yml`, `lock_manifest_coverage.yml`, `scorecard.yml` and
`dismissal_justifications_advisory.yml` has a row below, named by its check context
(the job's `name:`, or its job id when it has none), followed by the two repository-level GitHub features that raise security
alerts. A row is **Required** only when its context is one of ruleset 16846770's
required status checks listed above, or when the ruleset's code-scanning rule names
its tool; every other row is advisory. Of the seven required contexts, only
`Bandit Security Scan` and `CodeQL Static Analysis` come from these seven workflows.
`tests/test_security_gate_map_covers_security_jobs.py` fails if a job in those seven
workflows has no row here.

| Check context (workflow, job id) | What it checks | Tier |
| -------------------------------- | -------------- | ---- |
| `Bandit Security Scan` (`security.yml`, `bandit`) | Python SAST, `-ll`: MEDIUM+ severity | **Required** - that exact context is among ruleset 16846770's required status checks, so a MEDIUM+ finding fails the job AND holds the merge button. Corrected 2026-09-12: this row read "Advisory - ... is not a required status check, so it does not gate the merge button", which the ruleset refutes; the error understated enforcement |
| `Custom Secret Pattern Scan` (`security.yml`, `secret-pattern-scan`) | `scripts/check_secrets.py`, whose comments define the patterns exactly: an `=` or `:` assignment to a name carrying a credential-class keyword (`api_key`, `token`, `secret`, `password`, `passwd`, `auth`, `private_key`) after any prefix and before only `_`- or `-`-separated suffixes, so `accessToken`, `dbpassword` and `AWS_SECRET_ACCESS_KEY` all qualify; the value must be quoted, except in `.yml`, `.yaml`, `.sh`, `.md`, `.txt` and `Dockerfile*` files, where an unquoted value counts too. The script's bare-value list also names `.env`, `.cfg` and `.ini`, but it never opens files with those suffixes: scannability is decided by a SEPARATE suffix set that omits all three, so for them the exemption is declared and unreachable. Measured 2026-09-23 by calling the script's own helpers: an unquoted high-entropy assignment in a `.cfg` IS flagged when the file is handed to `scan_file` directly, and is never reached by a tree scan. Tracked `pytest.ini` is therefore not scanned. In a scanned file, a value is flagged when it has no whitespace, is longer than 8 characters and has Shannon entropy above 3.0. A file it cannot read, or a scan over fewer than 10 files, also fails | Advisory - no `continue-on-error`, so a finding fails the job red, but it is not a required status check. `scripts/hooks/pre-push` runs the same script locally. Added 2026-09-23: until then this job had no row in the map |
| `CodeQL Static Analysis` (`security.yml`, `codeql`) | Deep SAST, `security-extended` query suite -> Security > Code scanning | **Required, twice** - the context is a required status check, and the ruleset's code-scanning rule names the `CodeQL` tool (`security_alerts_threshold: all`, `alerts_threshold: errors_and_warnings`) |
| `Semgrep Custom Rules (blocking)` (`security.yml`, `semgrep-custom`, `semgrep-rules/`) | SESTRAV's own three ERROR rules: unsafe `pickle.load`, raw `joblib.load` bypassing `load_verified_joblib`, `subprocess(..., shell=True)` | **Blocking at the job level** - no `continue-on-error`, so any finding fails the run. It does NOT hold the merge button: `Semgrep Custom Rules (blocking)` is not among the required status checks on `Protect Main Branch`, re-measured 2026-09-23 against the live ruleset, and promoting it is a separate repo-settings change. The paragraph below this table lists it among the advisory jobs that turn their own run red |
| `Semgrep Security Scan (advisory)` (`security.yml`, `semgrep`) | Registry `p/python` ruleset plus `semgrep-rules/` -> Security > Code scanning | Advisory - the scan step is `continue-on-error`, so a finding leaves this job green. It still surfaces in two places: as a code-scanning alert, and, on a pull request, in the `Semgrep OSS` check run from the `github-advanced-security` app, which a finding can turn red, as PR #418's did. That check run is not required, but it shares one check suite with the `CodeQL` check run, so a red run makes the suite conclude `failure`, and OpenSSF Scorecard's SAST check, which reads check suites rather than runs, then counts the commit as not SAST-scanned; the `semgrep-custom` job's comments in `security.yml` record this cost (code-scanning alert #87). The job does go red on a TOOLING failure: a missing or unparseable SARIF, or one carrying 0 rules, fails its assertion step, so a scan that never ran cannot upload a zero-result analysis |
| `pip-audit Dependency Scan (advisory)` (`security.yml`, `pip-audit`; push to `main`, pull request, weekly) | CVEs in the installed `environments/requirements.lock` set -> run summary | Advisory - the audit step itself is `continue-on-error`, but the `tools/check_lockfile_advisories.py` step after it is not, so an advisory against a package pinned in `environments/requirements.lock` fails the job red unless `environments/accepted_advisories.toml` accepts that exact (advisory, package) PAIR - `load_acceptances` keys on both, so an ID accepted for one package still fails when the same ID surfaces on a different pinned package, as does a tooling failure. Findings on packages the lock does not pin, such as the runner's own `pip`, are reported but never fail. It is not a required status check |
| `Python SBOM (pip-licenses)` (`security.yml`, `python-sbom`) | Regenerates the dependency SBOM from the pinned lockfile and compares it with the committed `docs/sbom.json` and `docs/DEPENDENCY_LICENSES.md` | Advisory - fails red when either committed file is stale, or when the generated SBOM lacks any of its production sentinels (`numpy`, `pandas`, `scikit-learn`); not a required status check |
| `dependency-review` (`dependency-review.yml`, `dependency-review`; pull requests only) | New deps introduced in a PR (`fail-on-severity: moderate`) | Advisory - fails its own CI job on a finding, but is not a required status check, so it does not gate the merge button. The job declares no `name:`, so its check context is its job id |
| `Check lockfile freshness and hash pinning` (`dependency-lockfile-check.yml`, `lockfile-freshness`) | `tools/check_lockfile_freshness.py --check`: each `requirements*.in` pin is honoured by its compiled lockfile, and every compiled entry carries a `--hash=sha256:` pin; on a pull request authored by `dependabot[bot]`, also refuses any edit to `environments/requirements-ci-render.txt` | Advisory - red on a finding, but it runs only on pull requests that touch a file in its `paths:` filter, and it is not a required status check |
| `Locked packages are Dependabot-visible` (`lock_manifest_coverage.yml`, `check_lock_manifest_coverage`) | `tools/check_lock_manifest_coverage.py`: every distribution pinned in `environments/requirements.lock` is declared in a manifest GitHub's dependency graph parses, so none is invisible to Dependabot alerts by name. It matches distribution names only and does not compare versions | Advisory - red on an uncovered distribution; not a required status check |
| `Check for Paths and Credentials` and `Content Scan of Published Refs` (`pii_scan.yml`, `scan-leaks` and `scan-published-refs`) | Workstation absolute paths and AI-tooling filenames, over the tracked tree and over published refs | Advisory, and **split**: pushes to `main`/`release/**` and pull requests are ENFORCED (hard `exit 1`), while **tags are REPORTED only** - the ref loop passes `report`, which downgrades a finding to a `::warning`. That exemption is deliberate and commented in the workflow: a tag is immutable published provenance, so wiring it to `exit 1` would wedge the job red on history no push can change. Note the workflow has **no tag trigger at all**, so a tag is first examined on the next `main` push, pull request or weekly sweep. Neither of its jobs is a required status check, so even the enforced half cannot hold the merge button |
| `Scorecard analysis` (`scorecard.yml`, `analysis`) | OpenSSF Scorecard supply-chain checks -> published score and Security > Code scanning | Advisory - it runs on push to `main` and weekly, never on a pull request, and the workflow sets no score threshold (its only action inputs are `results_file`, `results_format` and `publish_results`), so it cannot gate a merge |
| `Report undocumented alert dismissals` (`dismissal_justifications_advisory.yml`, `dismissal-justifications`; weekly schedule only) | `tools/check_dismissal_justifications.py --branch main`: reports dismissed code-scanning alerts on `main` whose `dismissed_comment` is absent or blank, and warns on dismissed repo-level alerts, which OpenSSF Scorecard raises once per check | Advisory, and cannot turn red on a finding: its only trigger is a weekly `schedule` (cron `17 13 * * 1`, Mondays), so it never runs on a pull request or a push, and its report step is `continue-on-error: true`, so neither a finding nor a failed alert fetch fails the job |
| Dependabot alerts (repository setting) | Known CVEs in dependencies -> Security > Dependabot | Advisory (triaged). Vulnerability alerts and Dependabot security updates were both enabled when re-measured 2026-09-23 |
| GitHub secret scanning and push protection (repository settings) | Secrets in GitHub's provider patterns -> Security > Secret scanning; push protection rejects a push carrying one | Not a status check, so it never holds the merge button; push protection acts at push time instead. Re-measured 2026-09-23 from the repository's `security_and_analysis`: `secret_scanning` and `secret_scanning_push_protection` are `enabled`, while `secret_scanning_non_provider_patterns` and `secret_scanning_validity_checks` are `disabled`, so this layer does not look for generic credentials that follow no provider format. In CI those are left to the advisory `Custom Secret Pattern Scan` |

Advisory findings never block a merge on their own - none of them is a required status
check or a code-scanning rule on `Protect Main Branch`. **Advisory does not mean never
red.** These advisory jobs fail their own run on a finding:
`Custom Secret Pattern Scan`, `Semgrep Custom Rules (blocking)`,
`pip-audit Dependency Scan (advisory)` (on an unaccepted advisory against a locked
package), `Python SBOM (pip-licenses)` (on a stale committed SBOM), `dependency-review`,
`Check lockfile freshness and hash pinning`, `Locked packages are Dependabot-visible`,
`Check for Paths and Credentials` and `Content Scan of Published Refs` (tag findings in
the latter are reported, not failed). The `Semgrep Security Scan (advisory)` job itself
goes red on a tooling failure only; its findings can instead turn the separate
`Semgrep OSS` check run red on a pull request, as its row describes. That failure is
visible in the run and in branch-status UI, but does not prevent the merge button from
going green. **Corrected 2026-09-23:** this paragraph said "Three of them (Dependency
Review, pip-audit and the Semgrep custom-rules job)", which the PII row above already
contradicted, and the table had no row for `Custom Secret Pattern Scan`, the SBOM job,
the two dependency-integrity jobs, Scorecard or secret scanning. The list above is
enumerated rather than counted so that it can be checked against the table. Findings
surface as **(a)** tracked, dismissable alerts in *Security > Code scanning* (CodeQL,
Semgrep and Scorecard SARIF), **(b)** a red or markdown-annotated job run (any workflow
row above), or a red `Semgrep OSS` code-scanning check run on a pull request, **(c)**
Dependabot alerts/PRs, and **(d)** secret-scanning alerts. They are reviewed on the
weekly cadence and logged in the register below whenever consciously deferred.

### Recording a risk acceptance

When a finding is intentionally left unfixed (no upstream patch, not reachable in
SESTRAV's offline model, etc.):

1. Add an entry to the **Risk-Acceptance Register** below: identifier, component,
   rationale, and a re-review trigger or date.
2. If it is noisy in CI, suppress it *at the source* with the verified advisory ID
   and a pointer back to this register entry:
   - For a `pip-audit` finding on `environments/requirements.lock` (the production
     lockfile), add an entry to `environments/accepted_advisories.toml` -
     `.github/workflows/security.yml`'s `pip-audit` job fails closed on any finding
     not listed there (`tools/check_lockfile_advisories.py`). Do **not** use
     pip-audit's `--ignore-vuln` flag for this: it is applied before pip-audit
     writes its own report, so a suppressed finding becomes invisible to every tool
     that reads that report, including the ones meant to notice when it gets fixed
     upstream. That gap is exactly how CVE-2025-3000 sat patched-but-unnoticed for
     four weeks; see the entry below.
   - For anything else (Semgrep/CodeQL SARIF, Dependabot), dismiss-with-reason in
     the Security tab, citing this register entry in the dismissal comment.
   - Never suppress without a register entry.
3. Note material changes in the `CHANGELOG.md` `Unreleased / Security` section as
   the audit trail.

---

## Risk-Acceptance Register & Upstream Mitigations

As a standalone, offline scientific tool, SESTRAV occasionally relies on complex
third-party libraries (e.g., PyTorch) that may contain upstream vulnerabilities
with no available vendor patch. Each consciously-deferred advisory is logged here.

- **CVE-2025-3000 (PyTorch JIT script memory corruption):** **RESOLVED 2026-08-05 - no
  longer risk-accepted.** PyTorch `<= 2.12.1` contains a memory corruption flaw inside
  `torch.jit.script`.
  - **Identifiers:** `CVE-2025-3000`, `GHSA-rrmf-rvhw-rf47`, `PYSEC-2025-194`.
  - **Resolution:** upgraded to `torch==2.13.0`, the first patched release (published
    2026-07-08), across `requirements.in`, `requirements.txt`,
    `environments/requirements.lock` and `environments/requirements-ci-torch-cpu.txt`,
    plus a `torch>=2.13.0` floor in `pyproject.toml`. The lockfiles only govern
    hash-pinned installs; the `pyproject.toml` floor is what stops `pip install -e .` or
    a downstream consumer from resolving back into the affected range (`<= 2.12.1`).
  - **How this entry went stale, recorded deliberately:** it asserted "No upstream patch
    is available" and set a re-review trigger of "publication of a patched release." That
    trigger fired on 2026-07-08 and went unnoticed for four weeks, because nothing
    re-evaluates this register on a schedule - the suppression was set once and never
    revisited. The advisory's own metadata (`firstPatchedVersion: 2.13.0`) was the
    authoritative signal and was queryable the whole time.
  - **Suppressions removed:** the `--ignore-vuln PYSEC-2025-194` /
    `GHSA-rrmf-rvhw-rf47` flags have been deleted from `.github/workflows/security.yml`,
    so pip-audit now reports this advisory again if it ever reappears. Any previously
    dismissed Dependabot alerts for it are superseded by the upgrade.
  - **Side effect, intentional:** torch 2.12.0 declared a `setuptools<82` build-metadata
    cap that collided with this repo's `setuptools>=83.0.0` floor
    (GHSA-h35f-9h28-mq5c) and forced every lockfile recompile through a `uv` override
    file. torch 2.13.0 declares `setuptools>=77.0.3`, so `overrides.txt` has been retired
    and both application lockfiles now resolve unaided. The setuptools floor itself is
    unchanged - it was the security constraint, not the workaround.

- **Standing lesson from the entry above:** a risk acceptance with no scheduled
  re-review is a claim that decays silently. Every **live** entry in this register carries
  a re-review trigger; none of them fire on their own. (Writing this sentence exposed a
  counter-example: the `-W error` bypass below had no trigger at all. One was added rather
  than weakening the rule.) Re-check this register whenever
  dependencies are audited, and treat an advisory's `firstPatchedVersion` field as the
  authoritative test of "is a patch available yet," not the prose in this file.

- **Semgrep `dangerous-subprocess-use-tainted-env-args` (external-tool wrappers):**
  Risk-accepted false positive. The benchmark wrappers shell out to external binaries
  (PRIME, PredIG/Docker) via `subprocess.run`.
  - **Identifiers:** Semgrep
    `python.lang.security.audit.dangerous-subprocess-use-tainted-env-args`,
    Bandit `B603`.
  - **Scope:** `scripts/run_prime_wrapper.py`, `scripts/run_predig_wrapper.py`,
    `scripts/run_predig_batched.py`.
  - **Severity:** Error (Semgrep default); not exploitable in SESTRAV's model.
  - **Mitigation:** All calls use the list/argv form (`shell=False`), so no shell
    interpretation occurs. Command arguments are sourced from the operator's local
    CLI (`argparse`) and constant literals - never from untrusted or network input.
  - **Suppression:** An inline `# nosemgrep` on all three call sites, and a CI step that
    makes GitHub honour it. **Corrected 2026-08-17:** this entry previously stated that
    an inline `# nosemgrep` does *not* clear the taint-mode finding, and that Security-tab
    dismissal was therefore the mechanism. The marker does clear it - the scanner's own
    verdict is `Ran 154 rules on 166 files: 0 findings.` What did not happen was GitHub
    acting on it: semgrep still wrote each finding into the SARIF tagged
    `"suppressions": [{"state": "accepted"}]`, and code scanning ingested it regardless,
    so the alerts were held closed only by manual dismissals that a single edit to the
    suppressed line could invalidate. `security.yml` now drops suppressed results before
    the SARIF upload, so the uploaded analysis matches the scan and no per-alert dismissal
    is required. Unsuppressed findings are untouched and still raise alerts. See
    `docs/security_compliance.md` for the measurement.
  - **Re-review trigger:** if any wrapper begins accepting subprocess arguments from
    untrusted/remote input, or switches to `shell=True`.

- **MCP SDK transport vulnerabilities (session hijacking / origin validation):**
  Risk-accepted. The `mcp` package is pulled in transitively as a dev/CI dependency of
  `semgrep`; it is not an application dependency and no MCP server transport is ever
  invoked by SESTRAV.
  - **Identifiers:** `GHSA-vj7q-gjh5-988w`, `GHSA-jpw9-pfvf-9f58`, `GHSA-hvrp-rf83-w775`.
  - **Scope:** `mcp` (transitive, via `semgrep`) in the CI/dev dependency closure.
  - **Severity:** High (per GitHub advisory). All three are server-transport
    vulnerabilities (session-ID authentication bypass, WebSocket origin validation);
    SESTRAV never starts an MCP server, so the vulnerable code path is unreachable.
  - **Mitigation:** Not used. SESTRAV imports `mcp` only as a side effect of installing
    `semgrep` for CI static analysis; no `mcp.server.*` transport is constructed or run
    anywhere in the codebase or CI.
  - **Suppression:** Dependabot alerts dismissed as `not_used`. This entry is the
    authoritative record; the dismissal comment on each alert points back here.
  - **Re-review trigger:** if `mcp` ever becomes a direct/runtime dependency, or an MCP
    server transport is added to the codebase.

- **Strict Warning Enforcement (`-W error` bypass) for PyTorch Ecosystem:**
  Risk-accepted. OpenSSF Silver requires strict compiler/linter warnings to be enabled and addressed.
  - **Scope:** `pytest` test collection and runtime.
  - **Rationale:** Deep dependencies within `torch_geometric` raise unpatchable `DeprecationWarning`s during import (e.g., `torch_geometric.distributed` deprecation since 2.7.0). Enforcing `-W error` globally breaks the CI pipeline.
  - **Mitigation:** A selected-rule static-analysis gate. **Corrected 2026-09-02:** this entry
    previously said Ruff was "set to fail on any warning or error", which reads wider than the
    gate that runs. What runs is `ruff check .` in the `lint` job of `.github/workflows/ci.yml`,
    with no `--exit-zero`, so any violation of the ENABLED rule set fails the job. The enabled
    set is `E4`, `E7`, `E9`, `F` and `S` (`[tool.ruff.lint]` in `pyproject.toml`), with nine
    rules ignored repository-wide; Pycodestyle's `W` warning category is **not selected at
    all**. Measured 2026-09-02 against this repository's own config, with a control in each
    direction: trailing whitespace (`W291`) and an over-length line (`E501`) exit 0, while an
    unused import (`F401`) and an insecure hash (`S324`) exit 1 - `S324` fires although `S` is
    outside Ruff's default selection, and `E402` exits 0 although `E4` is selected, which is
    how the measurement is shown to be reading this config and not a default one. The Silver
    static-analysis judgement recorded above predates this correction and is not re-assessed
    here. Runtime PyTorch deprecation warnings continue to pass in test execution.
  - **Re-review trigger:** any `torch` or `torch-geometric` major/minor upgrade (which may
    drop the import-time `DeprecationWarning`s this bypass exists for), or any narrowing of
    the pytest warning filter. Added 2026-08-05: this entry previously carried no trigger,
    which the standing lesson above says is exactly how a risk acceptance decays silently.


- **`environments/requirements.lock` is outside Dependabot's dependency graph (a detection
  gap, not an exposure):** Risk-accepted. GitHub's dependency graph for this repository
  (measured 2026-09-19 at `21bbcac`, re-measured 2026-09-28) parses 15 pip-ecosystem
  manifests: `pyproject.toml`, every tracked `requirements*.txt`, and
  `environments/requirements-lock.in`. `environments/requirements.lock` is not among them,
  so the compiled production closure raises no Dependabot alert of its own.
  - **Identifiers:** none. This is a gap in the alerting surface rather than a vulnerability
    in a package, so no advisory id applies.
  - **Scope:** `environments/requirements.lock` alone. Measured first on 2026-09-19 at
    commit `21bbcac` by two instruments that share no input: GitHub's dependency graph
    reported 31 parsed manifests (34 on 2026-09-28; the 15 pip ones unchanged) and this
    file is absent from all of them, and none of the Dependabot alerts the API returned
    (108 on 2026-09-19, 106 on 2026-09-28) names it. Its compile input
    `environments/requirements-lock.in` IS parsed and HAS raised two alerts, but that
    covers the constraint layer only partly: an exact pin there is checked at its own
    version, but a `>=` floor says nothing about the version the lock resolves to. Its
    `pyjwt>=2.14.0` floor raised no alert for GHSA-42vr-xj54-vc7v (published
    2026-09-30), which covers the 2.14.0 the lock shipped.
  - **Severity:** Medium, and for DETECTION only. This is the file CI installs from in both
    the `pip-audit` and `python-sbom` jobs of `.github/workflows/security.yml` (via
    `pip install --require-hashes --no-deps`), so an advisory affecting only the version
    pinned here, and not a version any parsed manifest declares, would not appear in the
    Security tab.
  - **Mitigation:** the closure is audited, just not alerted on. The `pip-audit` job's
    "Install the pinned lockfile (resolver-free) for auditing" step installs this lock and
    the "Audit installed dependency set" step that follows audits the resulting environment;
    the comment above them records why the `pip-audit -r environments/requirements.lock`
    form is deliberately not used. `tools/check_lockfile_freshness.py`, whose
    `LOCKFILE_PAIRS` names this exact `.in`-to-`.lock` pair, fails when a package the `.in`
    pins with `==` is missing from the lock or at another version, or when one it floors
    with `>=` sits below that floor in the lock; `tools/check_hash_pins.py` separately
    requires hash pins. `tools/check_lock_manifest_coverage.py` (CI job `Locked packages are
    Dependabot-visible`) fails if any distribution pinned here is declared in no parsed
    manifest, judged by a filename rule calibrated against the live graph, so every locked
    package is Dependabot-visible by name; it does not compare versions. **Stated
    precisely, because the direction matters:** that `pip-audit` job is named "(advisory)"
    and is not a required status check, so this mitigation fails that job red on an
    unaccepted advisory against a locked package but does not gate a merge.
  - **Suppression:** none, and none is possible. Nothing is dismissed; the alerts are never
    generated in the first place.
  - **Re-review trigger:** if the `pip-audit` job stops installing this lock or is removed;
    if `tools/check_lockfile_freshness.py` is retired, since the `.in`-to-`.lock`
    propagation is what makes the covered layer meaningful; or if
    `tools/check_lock_manifest_coverage.py` is retired; or if GitHub's dependency graph
    starts parsing this file, at which point this entry should be RETIRED rather than
    re-reviewed. Renaming the file to `environments/requirements-lock.txt` was considered;
    since every tracked `requirements*.txt` is parsed, it would very likely close the gap
    directly. It is not done here because 29 tracked files name this path at `26a36521`,
    and every one that depends on the name would have to change in the same commit: among
    them the `Dockerfile`, `singularity.def` and `.dockerignore`, the `security.yml`,
    `iedb_benchmark.yml` and `lock_manifest_coverage.yml` workflows, and
    `tools/check_hash_pins.py`, `tools/check_lock_manifest_coverage.py`,
    `tools/check_lockfile_advisories.py`, `tools/check_lockfile_freshness.py` and
    `tools/update_dependencies.py`. `scripts/check_doc_commit_refs.py` also names its
    basename.
