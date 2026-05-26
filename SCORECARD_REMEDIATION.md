# OSSF Scorecard Remediation Guide

This guide describes how to configure the repository settings on GitHub to address the remaining high-severity alerts (`Code-Review` and `Branch-Protection`) and low-severity alerts (`CII-Best-Practices`) that require repository-level settings.

Since you are the sole contributor to the SESTRAV repository, standard enterprise review rules may be impractical. Below are tailored configurations that satisfy OSSF Scorecard requirements without locking you out of your own repository.

---

## 1. Branch Protection & Code Review Rulesets (High Severity)

GitHub Rulesets are the modern, recommended way to protect branches. They allow you to define bypass actors, which is perfect for solo developers who still want Scorecard-compliant protection.

### Recommended Ruleset Configuration for Solo Developers

1. Go to your repository on GitHub.
2. Navigate to **Settings** > **Rules** > **Rulesets** (in the sidebar under "Code and automation").
3. Click **New ruleset** > **Import ruleset** or **New branch ruleset**.
4. Name the ruleset: `Protect Main Branch`.
5. Under **Enforcement status**, select **Active**.
6. Under **Bypass list**, add:
   - **Bypass Actor**: Yourself (your GitHub username)
   - **Bypass Mode**: **Always** (allows you to bypass rules when needed, but scorecard still registers the presence of rulesets).
7. Under **Target branches**, select **Add default branch** (targets `main`).
8. Under **Branch rules**:
   - Check **Restrict deletions** (prevents deleting the main branch).
   - Check **Require linear history** (optional, enforces clean git history).
   - Check **Block force pushes** (prevents overwriting commits).
   - Check **Require a pull request before merging** (Scorecard requirement):
     - Check **Require approvals**.
     - Set **Required number of approvals before merging** to `1`.
     - Check **Dismiss stale pull request approvals when new commits are pushed**.
     - Check **Require review from Code Owners** (if a `CODEOWNERS` file is present).
   - Check **Require status checks to pass before merging** (Scorecard requirement):
     - Add the name of your CI check (e.g. `SESTRAV CI / test (3.11)`).
9. Click **Create** or **Save changes**.

> [!TIP]
> Setting up a Ruleset with yourself as a bypass actor ensures that the repository has strong default guardrails, satisfies OpenSSF Scorecard requirements, and still allows you to merge your own pull requests without needing another human reviewer.

---

## 2. OpenSSF Best Practices Badge (Low Severity)

The OpenSSF (formerly CII) Best Practices Badge program allows you to self-certify that your project follows secure development practices.

### Steps to Earn the Badge

1. Visit the [OpenSSF Best Practices Badge App](https://bestpractices.coreinfrastructure.org/).
2. Log in using your GitHub account.
3. Click **Add New Project**.
4. Paste the URL of your SESTRAV GitHub repository.
5. Answer the questions regarding your project's development practices:
   - Since you have already set up a security policy (`SECURITY.md`), automated test suite (`pytest`), static analysis (`Bandit`, `CodeQL`, `Semgrep`), and pinned dependencies, you will easily qualify for the **Passing** level badge.
6. Once the badge is generated, grab the markdown snippet and paste it at the top of your `README.md`.
   - *Example markdown badge:*
     ```markdown
     [![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/YOUR_ID/badge)](https://bestpractices.coreinfrastructure.org/projects/YOUR_ID)
     ```
7. Scorecard will automatically detect the badge and award a `10/10` score for the `CII-Best-Practices` check during the next scan.

---

## 3. Fuzzing & Property-Based Testing (Medium Severity)

We have integrated property-based testing using the `hypothesis` library in `tests/test_fuzz.py` to test feature extraction robustness.

### How to Run Fuzz Tests Locally

Run the following command from the repository root:
```bash
conda run -n sestrav pytest tests/test_fuzz.py -v
```

This will run property-based test cases to verify that `compute_features` and `get_tcr_positions` do not crash under edge cases (e.g., empty strings, non-canonical characters, extreme binding scores).
