# SESTRAV root Makefile - one-word entry points to checks CI already runs.
#
# Every recipe below is transcribed VERBATIM from a workflow's `run:` step,
# or (where explicitly noted in a comment) from this repo's already-documented
# local-safe variant of a CI command. A target that ran something CI does not
# run would manufacture false local confidence, so if you change a recipe
# here, re-check it against the workflow it cites.
#
# `make` is NOT installed on the Windows workstation this file was drafted
# on (`which make` -> command not found). Install it (e.g. via WSL, or a
# Windows make port) before relying on this file there; every command below
# was verified by running its literal argv directly, not through `make`.
#
# Assumes: `conda activate sestrav` and `pip install -e ".[dev]"` already run
# (see CONTRIBUTING.md). This file does not manage environment setup beyond
# the two convenience targets below.

.DEFAULT_GOAL := help
.PHONY: help setup hooks \
        lint lint-fix fmt typecheck \
        hash-pins lockfile-freshness coverage-scope \
        doc-commit-refs doc-line-citations affiliation secrets \
        bandit semgrep semgrep-custom dco \
        test test-full fuzz \
        snakemake-dryrun verify-benchmark \
        ci

help:
	@echo "SESTRAV developer targets (see .github/workflows/ for each check's CI source):"
	@echo "  setup               pip install -e \".[dev]\"          (CONTRIBUTING.md)"
	@echo "  hooks               install git hooks                  (CONTRIBUTING.md)"
	@echo ""
	@echo "  lint                ruff check .                       (ci.yml:lint)"
	@echo "  lint-fix            ruff check . --fix   [NOT a CI check - mutates tree]"
	@echo "  fmt                 ruff format .         [NOT a CI check - mutates tree]"
	@echo "  typecheck           mypy src/ ...                      (ci.yml:lint)"
	@echo ""
	@echo "  hash-pins           tools/check_hash_pins.py           (ci.yml:lint)"
	@echo "  lockfile-freshness  tools/check_lockfile_freshness.py  (dependency-lockfile-check.yml)"
	@echo "  coverage-scope      tools/check_library_coverage.py    (ci.yml:test)"
	@echo "  doc-commit-refs     scripts/check_doc_commit_refs.py   (doc_commit_refs.yml)"
	@echo "  doc-line-citations  scripts/check_doc_line_citations.py(doc_line_citations.yml)"
	@echo "  affiliation         scripts/check_affiliation_claims.py(affiliation_claims.yml)"
	@echo "  secrets             scripts/check_secrets.py           (security.yml:secret-pattern-scan)"
	@echo "  bandit              bandit -r . -ll -x ...             (security.yml:bandit, LOCAL-SAFE variant)"
	@echo "  semgrep-custom      semgrep scan --config semgrep-rules/ --error  (security.yml:semgrep-custom, BLOCKING)"
	@echo "  semgrep             semgrep scan --config p/python --config semgrep-rules/  (security.yml:semgrep, advisory)"
	@echo "  dco                 verify Signed-off-by on this branch's commits (dco.yml)"
	@echo ""
	@echo "  test                pre-push's fast pytest argv, Windows-safe    (scripts/hooks/pre-push:148-154)"
	@echo "  test-full           CI's exact coverage-gated pytest invocation  (ci.yml:test) [may crash on Windows]"
	@echo "  fuzz                Hypothesis property tests                   (fuzzing.yml)"
	@echo ""
	@echo "  snakemake-dryrun    both CI Snakemake DAG dry-run legs          (ci.yml:test)"
	@echo "  verify-benchmark    mock-mode ingestion + evaluator + report    (sestrav_verify_benchmarking.yml)"
	@echo ""
	@echo "  ci                  the fast, Windows-safe subset above, in one shot"

# --- setup (CONTRIBUTING.md, not a CI check) --------------------------------

setup:
	pip install -e ".[dev]"

hooks:
	bash scripts/hooks/install.sh

# --- lint / format / types ---------------------------------------------------

lint:
	ruff check .

# NOT a CI step - CI only checks (`ruff check .`), it never mutates the tree.
# Convenience wrapper for CONTRIBUTING.md's documented fix command.
lint-fix:
	ruff check . --fix

# NOT a CI step, for the same reason.
fmt:
	ruff format .

typecheck:
	mypy src/ --ignore-missing-imports --no-error-summary

# --- dependency / coverage-scope / doc-citation gates -----------------------

hash-pins:
	python tools/check_hash_pins.py

lockfile-freshness:
	python tools/check_lockfile_freshness.py --check

coverage-scope:
	python tools/check_library_coverage.py --check

doc-commit-refs:
	python scripts/check_doc_commit_refs.py

doc-line-citations:
	python scripts/check_doc_line_citations.py

affiliation:
	python scripts/check_affiliation_claims.py

secrets:
	python scripts/check_secrets.py

# --- security scanners --------------------------------------------------------

# CI's literal command (security.yml:bandit:"Run Bandit") is:
#   bandit -r . -ll -x ./tests/,./.ci_test_venv/,./.venv/,./build/
# That command is unusable on a dev checkout: `_local/` (gitignored, tens of
# thousands of vendored .py files) and `.claude/worktrees/` (gitignored, full
# duplicate checkouts) do not exist in a CI checkout, so CI never needs to
# exclude them - but they DO exist here, and scanning them produces false
# HIGH/MEDIUM findings and a 20+ minute runtime (see CLAUDE.md). The two
# extra excludes below are CLAUDE.md's own documented local-safe form; they
# are no-ops in CI's own checkout, so this does not diverge from what CI
# enforces on any path CI can see.
bandit:
	bandit -r . -ll -x ./tests/,./.ci_test_venv/,./.venv/,./build/,./_local/,./.claude/

# Literal transcription of security.yml:semgrep-custom:"Run SESTRAV custom
# rules" - the BLOCKING job (no continue-on-error). No network dependency
# (--metrics=off, config is the tracked semgrep-rules/ dir only).
semgrep-custom:
	semgrep scan --config semgrep-rules/ --error --metrics=off

# Mirrors the detection config of security.yml:semgrep:"Run Semgrep" (the
# advisory job), with `--sarif --output semgrep.sarif` dropped: those two
# flags only produce a file for a later GitHub code-scanning upload step and
# change no detection behavior. CI's own job is continue-on-error, so treat
# a nonzero exit here as something to triage, not a hard failure.
# NOTE (workstation-specific): semgrep can import-crash inside a conda env
# that also carries a newer `mcp` package, since semgrep unconditionally
# imports the v1 mcp.server.fastmcp. Run from an isolated venv if it dies on
# import rather than concluding semgrep itself is broken.
semgrep:
	semgrep scan --config p/python --config semgrep-rules/

# Approximates dco.yml:check_dco:"Check Developer Certificate of Origin" for
# a local branch. BASE_REF is hardcoded to `main` because that workflow only
# ever triggers with `branches: [main]`; the bot-author exemption is dropped
# because it has no local meaning.
dco:
	git fetch origin main
	@for c in $$(git rev-list --no-merges origin/main..HEAD); do \
		git show -s --format=%B $$c | grep -qi "Signed-off-by:" || { echo "Missing DCO sign-off: $$c"; exit 1; }; \
	done
	@echo "All commits have DCO sign-offs."

# --- tests --------------------------------------------------------------------

# Transcribed from scripts/hooks/pre-push (220 lines; argv at :148-154), NOT
# from a CI workflow directly. This is deliberate: a bare
# `python -m pytest tests/ -q` exits 127 on Windows workstations from a
# native crash during collection (shap/scipy and a Bio.Align DLL blocked by
# an OS Application Control policy - both confirmed environment-specific and
# unrelated to any branch's content). CI (ubuntu-latest) is authoritative for
# the two ignored files and four deselected cases; this target is the
# Windows-safe stand-in the project's own pre-push hook already uses.
# Re-derive the argv from the hook rather than trusting these line numbers -
# they have moved before.
test:
	python -m pytest tests/ -q \
		--ignore=tests/test_run_analysis_results_guard.py \
		--ignore=tests/test_pipeline_stages.py \
		--deselect 'tests/test_entry_point_help_smoke.py::test_help_parses_and_exits_clean[src.shap_analysis]' \
		--deselect 'tests/test_entry_point_help_smoke.py::test_output_dir_flag_is_advertised_as_required[src.shap_analysis---output-dir]' \
		--deselect 'tests/test_entry_point_help_smoke.py::test_missing_output_dir_flag_is_rejected[src.shap_analysis---output-dir]' \
		--deselect 'tests/test_entry_point_help_smoke.py::test_allow_overwrite_escape_hatch_is_advertised[src.shap_analysis]'

# Literal transcription of ci.yml:test:"Run tests with coverage". Requires
# the `dev` extra (pytest-xdist, pytest-cov) - `pip install -e ".[dev]"`.
# FLAGGED: this runs the full tests/ directory with none of pre-push's
# --ignore/--deselect guards, so on a Windows workstation it is expected to
# hit the same native-crash files `test` above works around. CI on
# ubuntu-latest does not have this problem; this target exists to run the
# exact CI invocation on a machine that can (e.g. WSL2 Ubuntu, or Linux CI
# itself), not as a routine local command on Windows.
test-full:
	PYTHONPATH=$(CURDIR)/tools/coverage_subprocess COVERAGE_PROCESS_START=$(CURDIR)/.coveragerc.library \
		python -m pytest tests/ -n auto \
		--cov=src --cov=functions \
		--cov-config=.coveragerc.library \
		--cov-report=term-missing --cov-report=xml

# Literal transcription of fuzzing.yml:fuzz:"Run Hypothesis fuzz tests" for
# the push/PR path (fixed seed 0 for reproducibility, 200 examples).
fuzz:
	HYPOTHESIS_MAX_EXAMPLES=200 python -m pytest tests/test_fuzz.py -v \
		--tb=short -p hypothesis --hypothesis-seed=0

# --- pipeline / benchmark wiring ----------------------------------------------

# Both legs are literal, from ci.yml:test. Leg 1 also appears verbatim in
# .claude/rules/deletion-safety-battery.md's pre-deletion battery; leg 2
# matches CONTRIBUTING.md's own Snakemake-validation section.
snakemake-dryrun:
	snakemake --snakefile pipeline.smk \
		--configfile tests/fixtures/dag_smoke/config.smoke.yaml \
		--dry-run --cores 1
	snakemake --snakefile pipeline.smk --dry-run --cores 1

# Literal transcription of sestrav_verify_benchmarking.yml:verify-benchmark.
# Mock mode only - no network calls, unlike iedb_benchmark.yml (deliberately
# not wrapped here; it hits the live IEDB API).
verify-benchmark:
	python src/verify/iedb_multi_virus_extractor.py src/verify/targets.json --mock
	python src/verify/sestrav_evaluator.py src/verify/targets.json --mock
	@test -f results/verify/validation_report.json || { echo "Error: validation_report.json was not generated."; exit 1; }
	@echo "Validation Report verified successfully:"
	@cat results/verify/validation_report.json

# --- aggregate ------------------------------------------------------------

# The fast, Windows-safe subset of the above. Deliberately excludes
# test-full, fuzz, verify-benchmark and snakemake-dryrun (slower/heavier, or
# dependent on extras) and semgrep (advisory only) - run those individually.
# This grouping is a convenience composition, not itself transcribed from
# any single CI job.
ci: lint typecheck hash-pins lockfile-freshness coverage-scope \
    doc-commit-refs doc-line-citations affiliation secrets \
    bandit semgrep-custom test
	@echo "Fast local CI subset passed."
