# SESTRAV Threat Model & Assurance Case

This document is SESTRAV's **assurance case**: it states the project's security
requirements, the threat model and trust boundaries they are defined against, and
a reasoned argument - backed by concrete evidence in the repository - for why the
requirements are met. It satisfies the OpenSSF Best Practices `assurance_case`
criterion (Silver) and supports `security_review` (Gold).

Companion documents: `SECURITY.md` (reporting & response), `docs/security_compliance.md`
(control inventory).

## 1. What SESTRAV is (scope)

SESTRAV is an **offline, locally executed scientific pipeline** that predicts the
immunogenicity of viral peptide epitopes. It is run by researchers on their own
machines or CI. It is **not** a hosted multi-tenant service, stores no user
accounts or credentials, and transmits no user data off the host. An optional
FastAPI microservice and Streamlit demo are developer tools that bind to
`127.0.0.1` (loopback) only.

## 2. Security requirements

1. **Integrity of inputs/outputs.** Untrusted input files (FASTA/CSV/YAML) must
   not be able to corrupt execution or cause unsafe behavior.
2. **Safe deserialization.** Loading model artifacts must not allow arbitrary code
   execution.
3. **No code/command injection.** No execution path may pass untrusted data to a
   shell or to `eval`/`exec`.
4. **Supply-chain integrity.** Dependencies must be pinned and verified; new
   vulnerable dependencies must be blocked and known ones remediated on an SLA.
5. **No secret/PII leakage.** No credentials, keys, or PII/PHI in the repository
   or outputs.
6. **Release integrity.** Released artifacts must be verifiable by consumers.

## 3. Trust boundaries

| Boundary | Trusted | Untrusted |
| --- | --- | --- |
| Input files | the operator running the tool | the *content* of FASTA/CSV/YAML files |
| Model artifacts | artifacts produced by the project's own training | arbitrary pickles from third parties |
| Dependencies | hash-pinned versions in the lockfiles | any unpinned/unknown package version |
| Network | GitHub/PyPI over TLS | any other inbound/outbound connection (none made by the tool) |
| Optional service | loopback clients on the same host | any remote network client (not reachable) |

## 4. Threats and mitigations (the argument)

| # | Threat (STRIDE) | Mitigation | Evidence |
| --- | --- | --- | --- |
| T1 | **Tampering** - malformed FASTA/CSV/config corrupts a run | Pydantic schema validation (`SestravConfig`); length/alphabet validation of peptides; property-based fuzzing of parsers | `docs/security_compliance.md` §4-5; `tests/test_fuzz.py`; `fuzzing.yml` |
| T2 | **Elevation of privilege** - arbitrary code execution via malicious model file | `torch.load(..., weights_only=True)`; `ModelRegistry` validates expected features before use | `docs/security_compliance.md` §5; Bandit B614 fix in `CHANGELOG.md` |
| T3 | **Elevation of privilege** - command/shell injection | No `shell=True`; no `eval`/`exec`; external-tool wrappers use argv-list form with constant/CLI-sourced args | `SECURITY.md` Risk-Acceptance Register; `security.yml` (Bandit/Semgrep) |
| T4 | **Tampering** - compromised dependency | Hash-pinned lockfiles (`--require-hashes`); `dependency-review.yml` blocks vulnerable deps in PRs; Dependabot + `pip-audit` monitor; SLA in `SECURITY.md` | `.github/workflows/dependency-review.yml`; `.github/dependabot.yml`; `environments/requirements.lock` |
| T5 | **Information disclosure** - secret/PII leak | No credential store; `pii_scan.yml` pre-merge gate; `.gitignore` catch-alls for key/cert/db files | `.github/workflows/pii_scan.yml`; `SECURITY.md` (PII/PHI section) |
| T6 | **Spoofing/Tampering** - tampered release | SHA-256 integrity manifests today; cryptographic signing of releases/tags (in progress, see `ROADMAP.md`) | `src/release_bundle.py`; `SECURITY.md` |
| T7 | **Information disclosure** - accidental network exposure of the optional service | Service and demo bind to `127.0.0.1` only | `docs/security_compliance.md` §5; `docker-compose.yml` |

## 5. Residual risks (accepted)

- Complex upstream dependencies may carry advisories that are unreachable from SESTRAV
  or have no available patch; these are risk-assessed and logged in the
  **Risk-Acceptance Register** in `SECURITY.md` (e.g. the `mcp` SDK transport
  advisories, which reach the dependency closure only as a transitive dev/CI dependency
  of `semgrep` and are unreachable because SESTRAV never starts an MCP server).
  CVE-2025-3000 in PyTorch JIT was previously the example here; it was **resolved on
  2026-08-05** by upgrading to `torch 2.13.0`, the first patched release, and is no
  longer risk-accepted.
- Release **signing** (authenticity) is being added; until then, integrity is
  provided by SHA-256 manifests over TLS-delivered GitHub Releases.

## 6. Verification

The argument above is continuously checked: static analysis (Bandit, CodeQL,
Semgrep) and dynamic analysis (Hypothesis fuzzing) run in CI on every push to
`main`, and dependency monitoring runs continuously. See `security.yml`,
`fuzzing.yml`, and `scorecard.yml`.
