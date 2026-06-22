# SESTRAV Governance

This document defines how SESTRAV is governed: who makes decisions, how decisions
are made, the roles people hold, and how the project continues if a maintainer
becomes unavailable. It satisfies the OpenSSF Best Practices "project oversight"
criteria (`governance`, `roles_responsibilities`, `access_continuity`).

## 1. Governance model

SESTRAV uses a **single-maintainer (BDFL-style) model with documented, transparent
processes**. The lead maintainer is responsible for the project's direction,
quality, and security, and is the final decision-maker. To keep decisions
transparent and reproducible rather than ad hoc:

- All substantive decisions are recorded in public artifacts - GitHub Issues,
  Pull Request discussions, `CHANGELOG.md`, and the design docs under `docs/`.
- All code changes flow through Pull Requests against `main` and must pass the
  required CI checks before merge (see `CONTRIBUTING.md`).
- Disagreements are resolved by discussion on the relevant Issue/PR; if consensus
  is not reached, the lead maintainer decides and records the rationale in the thread.

As the contributor base grows, this model is expected to evolve toward a small
maintainer committee with shared merge rights (see `ROADMAP.md`).

## 2. Roles and responsibilities

| Role | Holder | Responsibilities |
| --- | --- | --- |
| **Lead maintainer** | Gavin Borges (@Gavin-Borges) | Architecture, releases, merge authority, final decisions, security response. |
| **Security contact** | Gavin Borges (`gavinmborges1104@gmail.com`) | Triage and coordinate vulnerability reports per `SECURITY.md`. |
| **Release manager** | Lead maintainer | Versioning, tagging, changelog, and release-bundle signing. |
| **Reviewers** | Lead maintainer (+ designated backup, see `BUS_FACTOR.md`) | Review Pull Requests for correctness, tests, and style. |
| **Backup maintainer** | See `BUS_FACTOR.md` | Continuity if the lead is unavailable. |
| **Contributors** | Anyone who opens a PR/Issue | Propose changes per `CONTRIBUTING.md`. |

Significant past contributors are credited in `CONTRIBUTORS.md`.

## 3. Decision-making process

1. **Routine changes** (bug fixes, tests, docs, dependency bumps): proposed via PR,
   merged after CI passes and review.
2. **Substantive changes** (new features, API/output changes, changes to frozen
   `results/` artifacts or the canonical model track): proposed in an Issue first,
   discussed publicly, then implemented via PR. Promotion of experimental
   (ANN/GNN) tracks to canonical requires meeting the quantitative gates in
   `ROADMAP.md`.
3. **Security changes**: follow `SECURITY.md` (private triage, coordinated disclosure).

## 4. Becoming a maintainer

A contributor may be invited to become a maintainer (with merge rights) after a
sustained track record of high-quality contributions and reviews. The lead
maintainer extends the invitation and records it here and in `CONTRIBUTORS.md`.
This is the primary mechanism for raising the project's bus factor over time.

## 5. Continuity (access continuity)

The project is designed to survive the loss of any single individual:

- **Source & history** are public on GitHub and fully cloneable; the project is
  MIT-licensed, so anyone may fork and continue it.
- **No single hidden dependency**: build, test, and release procedures are
  documented in `README.md` and `CONTRIBUTING.md` and automated in CI.
- **Backup maintainer & access recovery** are documented in `BUS_FACTOR.md`.
- **Releases** are reproducible from tagged source; integrity manifests and the
  release process are described in `CONTRIBUTING.md` and `SECURITY.md`.

## 6. Code of conduct

All participation is governed by `CODE_OF_CONDUCT.md`.
