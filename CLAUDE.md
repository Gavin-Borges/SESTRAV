# SESTRAV Agent Execution Protocol

## 1. Project Scope & Source of Truth
- **Project:** SESTRAV (Structural Epitope Scoring via TCR Recognition and Vaccinology).
- **Goal:** Processing IEDB datasets via Snakemake for HPV/EBV therapeutic epitope discovery using ANN/GNN models (PyTorch/PyG).
- Use repository files and committed documentation as the primary source of truth. Prefer `environment.yml` and workflow files over inferred assumptions.

## 2. Agent Behavioral Directives
- **Think Before Acting:** Before modifying code, briefly detail the files you will touch and the logical steps you will take.
- **Minimal Diff Edits:** Make the smallest correct change that satisfies the task.
- **No Hallucinations:** If a biological mechanism, split logic, or tensor shape is not explicitly defined in the code, document it as `[PENDING REVIEW]`. Do not guess.
- **Silent Execution:** Minimize conversational filler. Default to silent execution for file-write operations.

## 3. Strict Constraints & Prohibitions
- NEVER modify, delete, or interact with the `.snakemake/` directory, `data/raw/`, or benchmark outputs.
- NEVER execute heavy training loops or long preprocessing jobs in the terminal. Use `snakemake -n` (dry-run) for validation.
- NEVER introduce new dependencies without explicit approval.
- Do not make speculative biological claims without identifying uncertainty.

## 4. Code Quality Standards
- PEP 8 compliance, Black/Ruff formatting.
- Python type hints and clear Google-style docstrings for new or materially changed functions.
