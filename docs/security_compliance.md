# SESTRAV Security & Compliance Posture

This document tracks SESTRAV's posture against the [OpenSSF Best Practices Badge](https://www.bestpractices.dev/projects/13191) (formerly CII Best Practices). SESTRAV has attained the **Passing** level ([project 13191](https://www.bestpractices.dev/projects/13191)) as of version 2.0 and is working toward the Silver/Gold criteria.

## 1. Basics
- **Project Description:** SESTRAV is a T-cell epitope immunogenicity prediction pipeline.
- **URL / Repository:** Hosted on GitHub (https://github.com/Gavin-Borges/SESTRAV).
- **License:** Open Source (specified in repo).

## 2. Change Control
- **Version Control:** Git/GitHub is strictly utilized.
- **Release Tracking:** Releases are tagged with semantic versions (e.g., `v2.0.2`) and published as GitHub Releases with build-provenance attestations.
- **Review:** All pull requests to `main` require a successful GitHub Actions CI check before merging.

## 3. Reporting
- **Vulnerability Reporting:** Described in `SECURITY.md`. Issues can be reported confidentially; the maintainers pledge to respond to vulnerabilities promptly.
- **Bug Tracking:** GitHub Issues is used as the primary issue tracker.

## 4. Quality
- **Automated Testing:** 
  - 200+ pytest test cases (`tests/`) validate core logic, schema conformity, and feature store deterministic behavior.
  - Test suite executes cleanly with no critical failures.
- **CI/CD Integration:** Tests are executed via `.github/workflows/ci.yml` on every pull request.
- **Statement coverage (OpenSSF Silver `test_statement_coverage80`):** Coverage is
  measured on two scopes, both gated in CI:
  - *Library scope* — the importable library surface (`src`/`functions` modules
    with no `__main__` CLI entry point), measured via `.coveragerc.library` and
    gated at `fail_under=80`. Currently ≈83% statement / ≈81% branch-inclusive.
    The omit list is generated mechanically from the presence of a `__main__`
    guard (`tools/check_library_coverage.py --check` enforces it stays in sync),
    so the scope is objective rather than hand-picked.
  - *Whole-repo floor* — a regression floor across the entire tree
    (`pyproject.toml`), currently ≈30% statement. Executable research/pipeline
    scripts (those with `__main__`) are validated by integration tests and the CI
    data/benchmark gates rather than by unit statement coverage.
- **Strict Data Typing:** Pydantic is utilized via `SestravConfig` to enforce configuration schema, preventing runtime type coercion errors and hardcoded path injection.

## 5. Security
- **Secure Architecture:** 
  - Avoidance of `eval()`/`exec()` and unsafe `shell=True` subprocesses.
  - Transitioned from unsafe file loading (`json.loads(open(path).read())`) to context-managed explicit IO logic.
  - Safe model unpickling via `ModelRegistry` validating expected features.
- **Dependency Management:** Dependencies are rigorously pinned with `--require-hashes` using `pip-compile` to prevent supply chain injection.
- **Static Analysis (SAST):** CodeQL, Bandit, and Semgrep are integrated into GitHub actions.

## 6. Analysis
- **Dynamic Analysis / Fuzzing:** Hypothesis property-based fuzz testing is integrated in CI via `.github/workflows/fuzzing.yml`. Tests in `tests/test_fuzz.py` exercise `compute_features` and `get_tcr_positions` under adversarial and edge-case amino acid inputs. Standard runs use 200 examples per push; weekly scheduled runs use 1000 examples. The Hypothesis database is persisted as an artifact to retain failure examples across runs.
- **Pipeline Data Integrity:** `freeze_mode` constraints enforce data immutability during reproducibility benchmarking — this is a correctness guard, not a security control.
- **Hygiene:** 0 High-severity issues in the core Python codebase (verified by Bandit 2026-06-18, session 5 Day 4 audit — see §7).

## 7. Day 4 Security Audit (2026-06-18)

### Bandit (`bandit -r . -ll`)

| Severity | Count | Disposition |
|----------|-------|-------------|
| High     | 0     | — |
| Medium   | 5     | All B614 (PyTorch load/save); all false positives in context — see table |
| Low      | 1184  | Not actionable at `-ll` threshold; dominated by assert-use and subprocess patterns in test harness |

**Medium findings detail (all B614 — CWE-502):**

| Location | Context | Disposition |
|----------|---------|-------------|
| `resave_checkpoint.py:61` | `torch.save(ckpt, path)` — internal maintenance script saving trusted model state | Acceptable: script not exposed to user input; data is internally generated |
| `resave_checkpoint.py:355` | `torch.save(new_ckpt, output_path)` — same script | Acceptable: same rationale |
| `tests/test_graph_builder.py:48` | `torch.save(dist, tmp_path / "PEP_dist.pt")` — test fixture writing a known tensor | `# nosec B614` added 2026-06-18 |
| `tests/test_sestrav_evaluator_extended.py:165` | `torch.save(state, chk)` — test fixture writing known state dict | `# nosec B614` added 2026-06-18 |
| `tests/test_sestrav_evaluator_extended.py:175` | `torch.save(state, chk)` — second test fixture in same file | `# nosec B614` added 2026-06-18 |

**Action:** The two `resave_checkpoint.py` findings are accepted as tolerable operational risk (`torch.save` is necessary for checkpoint maintenance; the file is not reachable from user-facing CLI paths). No `# nosec` added there to preserve visibility; document will be updated if the threat model changes.

### Semgrep (`semgrep scan --config p/python`)

Semgrep is not installed in the `sestrav` conda environment as of Day 4. It is active in CI via `.github/workflows/security_scan.yml` (runs on every PR and weekly schedule). Manual local semgrep scan is a pending environment setup task.

**Action (manual — Gavin):** Run `pip install semgrep` in the sestrav environment and execute `semgrep scan --config p/python` once. Results to be added here.

### pip-audit (`pip-audit -r environments/requirements.lock`)

`pip-audit` is not installed in the `sestrav` conda environment as of Day 4. The lock file is present at `environments/requirements.lock`.

**Known outstanding CVE:** CVE-2025-3000 (torch, CVSS 5.3) — dismissed tolerable_risk in prior session. `torch.jit.script` not exposed to untrusted input. Will reopen when PyTorch patches.

**Action (manual — Gavin):** Run `pip install pip-audit` then `pip-audit -r environments/requirements.lock`. Update this section with full output.

## Future Upgrades
Automated dependency updates are in place via Dependabot (`.github/dependabot.yml`)
alongside the Dependency-review Action and OSSF Scorecard. Signed releases now
ship via the `release.yml` workflow, which attaches a Sigstore build-provenance
attestation to every tagged release (v2.0.2 onward) — satisfying the OpenSSF
`signed_releases` criterion (verify with `gh attestation verify`). Remaining
planned work: publish the package to PyPI, and optionally cryptographically sign
the git tags themselves (`version_tags_signed`, a SUGGESTED criterion) once a
personal signing key is configured. See `ROADMAP.md` for the open multi-person
Silver/Gold criteria and the coverage ratchet.
