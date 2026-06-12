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

## Upstream Dependency Vulnerability Policy & Mitigations

As a standalone, offline scientific tool, SESTRAV occasionally relies on complex third-party libraries (e.g., PyTorch) that may contain upstream vulnerabilities with no available vendor patches. 

Our policy for handling such vulnerabilities is as follows:
- **CVE-2025-3000 (PyTorch JIT script memory corruption):** Mitigated. PyTorch version `<= 2.12.0` contains a critical memory corruption flaw inside `torch.jit.script`.
  - **Mitigation:** SESTRAV does not use, import, or execute the TorchScript compiler (`torch.jit.script`) anywhere in its pipeline. Because all execution happens offline on locally validated datasets and does not accept dynamic compilation payloads, the attack surface is completely absent.

