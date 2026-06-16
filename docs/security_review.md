# SESTRAV Security Review

This record documents a security review of SESTRAV, satisfying the OpenSSF Best
Practices `security_review` criterion (Gold). It should be repeated at least once
every five years and whenever the architecture changes materially.

- **Review date:** 2026-06
- **Reviewer(s):** Gavin Borges (lead maintainer). _An independent external review
  is recommended and tracked in `ROADMAP.md`._
- **Version reviewed:** 2.0.x line (see `CHANGELOG.md`)
- **Scope:** the full SESTRAV pipeline, the optional FastAPI/Streamlit tools, the
  build/release process, and the dependency supply chain.

## Inputs to the review

- The assurance case and trust boundaries in `docs/threat_model.md`.
- The control inventory in `docs/security_compliance.md`.
- Static analysis results: Bandit, CodeQL, Semgrep (`security.yml`).
- Dynamic analysis: Hypothesis property-based fuzzing (`fuzzing.yml`).
- Dependency posture: hash-pinned lockfiles, `dependency-review.yml`,
  Dependabot, and `pip-audit`.
- The vulnerability response process and risk register in `SECURITY.md`.

## Findings

1. **Security requirements considered.** The six security requirements in
   `docs/threat_model.md` were reviewed against the implementation; each maps to at
   least one implemented, evidence-backed control (threat table T1–T7).
2. **Attack surface is small and offline.** The tool makes no outbound network
   calls during scoring, stores no credentials/PII, and exposes the optional
   service on loopback only — limiting the realistic external attack surface.
3. **Highest-value controls are in place:** safe deserialization
   (`weights_only=True` + `ModelRegistry`), no shell/`eval`/`exec`, input
   validation via Pydantic, hash-pinned dependencies, and pre-merge secret/PII
   scanning.
4. **Static + dynamic analysis** run continuously in CI; no high-severity findings
   are open in first-party code.
5. **Open improvement:** cryptographic **signing of releases and tags** (currently
   integrity-only via SHA-256 manifests). Tracked in `ROADMAP.md` and the Silver
   `signed_releases` work.

## Conclusion

As of the review date, SESTRAV's implemented controls are commensurate with its
threat model (an offline scientific tool processing untrusted data files). No
high-severity issues are outstanding in first-party code. The primary planned
hardening is release/tag signing. Continuous SAST/DAST and dependency monitoring
provide ongoing assurance between reviews.
