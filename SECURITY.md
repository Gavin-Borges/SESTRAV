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
3. Fill in the advisory form — this is end-to-end encrypted between you and the maintainer.

## Response Commitment

As a solo-maintained project, the maintainer commits to:

- **Acknowledge** receipt of your report within **3–5 business days**.
- **Provide an initial assessment** (severity, scope, reproducibility) within
  **10 business days** of acknowledgement.
- **Coordinate a fix and disclosure timeline** with you collaboratively.
- **Credit reporters** in the release notes (unless you prefer anonymity).

## Thank You

Responsible disclosure helps keep SESTRAV and its users safe.
Thank you for taking the time to report vulnerabilities privately.

---

## Personally Identifiable Information (PII) & Patient Health Data (PHI)

SESTRAV is designed as a standalone, offline bioinformatics pipeline. 

- **Data Privacy by Design**: All scoring and feature extraction processes execute strictly on the local machine or host environment. SESTRAV does not transmit, upload, or collect any sequence data, user parameters, or predictive outputs.
- **No PII/PHI Requirement**: The pipeline accepts standard FASTA, CSV, and YAML configurations. It does not require, accept, or process personally identifiable information (PII) or protected health information (PHI). Users are cautioned against introducing patient metadata or identifying fields into sequence inputs.
- **Credential Safety**: The pipeline does not connect to external patient databases and has no credential store or public telemetry APIs.

---

## Vulnerability Triage & Remediation Policy

How findings from Dependabot, code scanning, and the CI security workflows are
prioritized and acted on. The policy is deliberately **two-tiered**: a small set
of findings *block* merges/releases, while everything else is surfaced as a
*non-blocking advisory* that is tracked and triaged on a cadence rather than
gating active development.

### Severity → action (target SLA)

| Severity | Action | Target SLA | Gate |
| -------- | ------ | ---------- | ---- |
| Critical | Patch or documented mitigation before the next merge to `main` | 48 hours | **Blocking** |
| High     | Patch or mitigation within the current release cycle | 7 days | **Blocking** |
| Medium   | Scheduled fix; risk-acceptable with written justification | 30 days | Advisory |
| Low      | Best-effort; batched with routine dependency updates | Next cycle | Advisory |

Timelines follow the solo-maintainer cadence described under **Response Commitment** above.

### CI gate map

| Tool / workflow | What it checks | Tier |
| --------------- | -------------- | ---- |
| Bandit (`security.yml`, `-ll`) | Python SAST, MEDIUM+ severity | **Blocking** |
| Dependency Review (`dependency-review.yml`) | New deps introduced in a PR (`fail-on-severity: high`) | **Blocking** |
| CodeQL (`security.yml`) | Deep SAST → Security ▸ Code scanning | Blocking (default severities) |
| Semgrep (`security.yml`) | SAST ruleset `p/python` → Security ▸ Code scanning | Advisory |
| pip-audit (`security.yml`, weekly + PR) | CVEs in the pinned `requirements.lock` → run summary | Advisory |
| Dependabot alerts | Known CVEs in dependencies → Security ▸ Dependabot | Advisory (triaged) |

Advisory findings never turn CI red. They surface as **(a)** tracked, dismissable
alerts in *Security ▸ Code scanning* (Semgrep/CodeQL SARIF), **(b)** a markdown
digest on each `security.yml` run summary (pip-audit), and **(c)** Dependabot
alerts/PRs. They are reviewed on the weekly cadence and logged in the register
below whenever consciously deferred.

### Recording a risk acceptance

When a finding is intentionally left unfixed (no upstream patch, not reachable in
SESTRAV's offline model, etc.):

1. Add an entry to the **Risk-Acceptance Register** below: identifier, component,
   rationale, and a re-review trigger or date.
2. If it is noisy in CI, suppress it *at the source* with the verified advisory ID
   and a comment pointing here (e.g. pip-audit `--ignore-vuln <GHSA>`, or
   dismiss-with-reason in the Security tab). Never suppress without a register entry.
3. Note material changes in the `CHANGELOG.md` `Unreleased / Security` section as
   the audit trail.

---

## Risk-Acceptance Register & Upstream Mitigations

As a standalone, offline scientific tool, SESTRAV occasionally relies on complex
third-party libraries (e.g., PyTorch) that may contain upstream vulnerabilities
with no available vendor patch. Each consciously-deferred advisory is logged here.

- **CVE-2025-3000 (PyTorch JIT script memory corruption):** Mitigated / risk-accepted.
  PyTorch version `<= 2.12.0` contains a critical memory corruption flaw inside
  `torch.jit.script`.
  - **Mitigation:** SESTRAV does not use, import, or execute the TorchScript compiler
    (`torch.jit.script`) anywhere in its pipeline. Because all execution happens
    offline on locally validated datasets and does not accept dynamic compilation
    payloads, the attack surface is completely absent.
  - **Re-review trigger:** any PyTorch version bump, or publication of a patched release.

