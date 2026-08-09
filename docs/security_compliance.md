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
  measured on two scopes; only the library scope is gated in CI:
  - *Library scope* - the importable library surface (`src`/`functions` modules
    with no `__main__` CLI entry point), measured via `.coveragerc.library` and
    gated at `fail_under=95`. Currently 98.91% combined (~99% statement /
    ~98% branch; measured 2026-06-22), clearing the OpenSSF Gold targets
    (>=90% statement, >=80% branch).
    The omit list is generated mechanically from the presence of a `__main__`
    guard (`tools/check_library_coverage.py --check` enforces it stays in sync),
    so the scope is objective rather than hand-picked.
  - *Whole-repo floor* - a regression floor across the entire tree
    (`pyproject.toml`), gated at `fail_under=35`, currently **47.88%** (branch-inclusive,
    re-measured 2026-08-08) - comfortably above the floor. Supersedes the previously
    published 34.37% / "a hair under the floor" (measured 2026-06-22, seven weeks stale).
    Executable research/pipeline
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
- **Pipeline Data Integrity:** `freeze_mode` constraints enforce data immutability during reproducibility benchmarking - this is a correctness guard, not a security control.
- **Hygiene:** 0 High-severity issues in the core Python codebase (verified by Bandit 2026-06-18, session 5 Day 4 audit - see section 7).

## 7. Day 4 Security Audit (2026-06-18)

### Bandit (`bandit -r . -ll`)

| Severity | Count | Disposition |
|----------|-------|-------------|
| High     | 0     | - |
| Medium   | 5     | All B614 (PyTorch load/save); all false positives in context - see table |
| Low      | 1184  | Not actionable at `-ll` threshold; dominated by assert-use and subprocess patterns in test harness |

**Medium findings detail (all B614 - CWE-502):**

| Location | Context | Disposition |
|----------|---------|-------------|
| `resave_checkpoint.py:61` | `torch.save(ckpt, path)` - internal maintenance script saving trusted model state | Acceptable: script not exposed to user input; data is internally generated |
| `resave_checkpoint.py:355` | `torch.save(new_ckpt, output_path)` - same script | Acceptable: same rationale |
| `tests/test_graph_builder.py:48` | `torch.save(dist, tmp_path / "PEP_dist.pt")` - test fixture writing a known tensor | `# nosec B614` added 2026-06-18 |
| `tests/test_sestrav_evaluator_extended.py:165` | `torch.save(state, chk)` - test fixture writing known state dict | `# nosec B614` added 2026-06-18 |
| `tests/test_sestrav_evaluator_extended.py:175` | `torch.save(state, chk)` - second test fixture in same file | `# nosec B614` added 2026-06-18 |

**Action:** The two `resave_checkpoint.py` findings are accepted as tolerable operational risk (`torch.save` is necessary for checkpoint maintenance; the file is not reachable from user-facing CLI paths). No `# nosec` added there to preserve visibility; document will be updated if the threat model changes.

### Semgrep (`semgrep scan --config p/python`)

Run 2026-06-18 (Day 5) with semgrep 1.167.0. **2 findings, both false positives in research-only scripts.**

| Location | Rule | Finding | Disposition |
|----------|------|---------|-------------|
| `scripts/run_predig_wrapper.py:101` | `dangerous-subprocess-use-tainted-env-args` | `subprocess.run(cmd, check=True)` where `cmd` is a Docker invocation list | **False positive.** `cmd` is a Python list (no `shell=True`); list-form subprocess is not shell-injectable. This is a Docker wrapper script for PredIG, not reachable from user-facing endpoints. |
| `scripts/run_prime_wrapper.py:95` | `dangerous-subprocess-use-tainted-env-args` | `subprocess.run(cmd, check=True)` where `cmd` wraps the PRIME/WSL binary | **False positive.** Same rationale - list-form subprocess, paths are internally constructed, not from untrusted user input. |

**Action:** No code changes required. Both scripts are researcher-only external-tool wrappers outside the installed package surface (`sestrav[pipeline]`). If either script ever accepts direct user command-line input, `shlex.quote()` should be applied at that point.

### pip-audit (`pip-audit -r environments/requirements.lock`)

Run 2026-06-18 (Day 5) with pip-audit 2.10.1. **4 packages flagged; 3 are transitive dependencies of mhcflurry/research tools, 1 is the pre-documented torch CVE.**

| Package | Version | Vulnerability | Fix | Disposition |
|---------|---------|--------------|-----|-------------|
| `aiohttp` | 3.13.5 | CVE-2026-54273 through CVE-2026-54280, CVE-2026-50269 (9 CVEs) | >=3.14.1 | **RESOLVED 2026-08-05** (disposition at the time of this run was "tolerable risk - transitive dependency": `aiohttp` is pulled in by `mhcflurry` as an HTTP client for model download, and SESTRAV's inference pipeline makes no network calls at prediction time, with the stated intent to "upgrade when mhcflurry releases a compatible version"). `environments/requirements.lock` now pins `aiohttp==3.14.3`, which satisfies the `>=3.14.1` fix. |
| `cryptography` | 46.0.7 | GHSA-537c-gmf6-5ccf | >=48.0.1 | **RESOLVED 2026-08-05** (disposition at the time of this run was "tolerable risk - transitive dependency" of `mhcflurry`/`paramiko`). A `cryptography>=48.0.1` floor was subsequently added to `environments/requirements-lock.in`, and raised to `>=50.0.0` on 2026-08-05 to close GHSA-g6cj-pr64-35w5 / CVE-2026-69247 (PKCS#7 `EnvelopedData` Bleichenbacher oracle, affects `>=44.0.0,<50.0.0`). The vulnerable code path is unreachable from SESTRAV regardless: the package is never imported by tracked source, and there are zero PKCS#7 / `EnvelopedData` / S-MIME call sites repo-wide. |
| `pyjwt` | 2.12.0 | PYSEC-2026-175-179 | >=2.13.0 | **RESOLVED 2026-08-05** (disposition at the time of this run was "tolerable risk - transitive dependency", pulled in by semgrep/mcp tooling; not a SESTRAV runtime dependency, and SESTRAV does not issue or verify JWTs). `environments/requirements.lock` now pins `pyjwt==2.13.0`, which satisfies the `>=2.13.0` fix. |
| `torch` | 2.11.0 | CVE-2025-3000 (CVSS 5.3) | >=2.13.0 | **RESOLVED 2026-08-05** (disposition at the time of this run was "pre-documented tolerable risk", on the grounds that no upstream patch existed and `torch.jit.script` is not exposed to untrusted input; local-only AV, EPSS 0.08%). PyTorch published 2.13.0, the first patched release, on 2026-07-08. SESTRAV upgraded to `torch==2.13.0` across `requirements.in`, `requirements.txt`, `environments/requirements.lock` and `environments/requirements-ci-torch-cpu.txt`, and floored `torch>=2.13.0` in `pyproject.toml` so non-lockfile consumers cannot resolve into the affected range (`<= 2.12.1`). The unreachability argument still holds and is why the exposure was never urgent, but it is no longer the remediation. |

**Summary (as of 2026-08-05): all four findings from this run are now RESOLVED** - every one is pinned at or above its fix version in `environments/requirements.lock`. At the time of the run there were 0 actionable CVEs in SESTRAV's own code; three of the four findings were transitive dependencies of third-party ML tooling with no exposure surface in the deployed package, and the fourth (`torch`) is a direct runtime dependency whose vulnerable code path (`torch.jit.script`) SESTRAV never invokes.

## Future Upgrades
Automated dependency updates are in place via Dependabot (`.github/dependabot.yml`)
alongside the Dependency-review Action and OSSF Scorecard. Signed releases now
ship via the `release.yml` workflow, which attaches a Sigstore build-provenance
attestation to every tagged release (v2.0.2 onward) - satisfying the OpenSSF
`signed_releases` criterion (verify with `gh attestation verify`). Remaining
planned work: publish the package to PyPI, and cryptographically sign the git tags
themselves (`version_tags_signed`, a SUGGESTED criterion). A maintainer SSH signing
key is configured locally with `tag.gpgsign` enabled, so this is met on the next
release by tagging with `git tag -s`. See `ROADMAP.md` for the open multi-person
Silver/Gold criteria and the coverage ratchet.
