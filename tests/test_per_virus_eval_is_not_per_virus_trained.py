"""Keep the per-virus within-CV table's DESCRIPTION in agreement with how it is BUILT.

Why this file exists: four tracked files stated that the certified per-virus table
(`results/per_virus_eval_v5_mode31.csv`, the source of the headline mean within-CV
AUC-ROC 0.658) was produced by training a separate model on each virus. It is not.
It is one pooled model's out-of-fold predictions, partitioned by virus.

The generator is `scripts/evaluate_per_virus.py`, the sole writer of that artifact.
It takes `--predictions <OOF csv>`, reads it with `pd.read_csv`, and slices it. There
is no estimator, no splitter and no `.fit()` call anywhere in the module; its only
sklearn import is `sklearn.metrics`. The corroboration is threefold: the nine
published values reproduce exactly by partitioning the tracked pooled mode-31 OOF;
`docs/paper.md` itself states in Discussion that a dedicated per-virus versus pooled
comparison "is required" and has not been run, which contradicts the caption that
claimed per-virus-only training; and the paper's pooled 0.814 is the same OOF file
the slices were cut from.

These tests do not rebuild, retrain or read anything under `data/` or `results/`.
They compare tracked CODE against tracked DOCS, so they run in a clone and in CI.
That is the same ratchet shape as `tests/test_corpus_code_divergence.py`.

Scope limit, stated honestly: the prose half is a ratchet over the four EXACT
phrasings that were retired on 2026-09-17. A reintroduction worded differently would
not be caught by it. The code half is the load-bearing test, because it asserts the
invariant the prose is about. If the generator ever legitimately gains a training
step, `test_generator_performs_no_training` fails first and the failure message says
to update the prose in the same commit.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "scripts" / "evaluate_per_virus.py"

# sklearn submodules that supply model fitting or fold splitting. `sklearn.metrics`
# is deliberately absent: the generator legitimately imports roc_auc_score and friends.
TRAINING_MODULES = (
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.model_selection",
    "sklearn.naive_bayes",
    "sklearn.neighbors",
    "sklearn.neural_network",
    "sklearn.svm",
    "sklearn.tree",
    "src.train_classifier",
    "src.ml_utils",
    "xgboost",
    "lightgbm",
)

FITTING_METHODS = ("fit", "fit_transform", "fit_predict", "partial_fit")

# The four phrasings retired on 2026-09-17, each a claim that the per-virus table was
# produced by per-virus TRAINING. Kept verbatim so a revert is caught.
RETIRED_PHRASES = (
    "trained and evaluated per virus",
    "training and evaluating exclusively on",
    "per-virus-only training",
    "per-virus training and metrics",
)


def _tracked_markdown() -> list[Path]:
    """Every TRACKED markdown file, which is what a reader can actually open.

    Scoped with `git ls-files` rather than a filesystem glob, matching
    `scripts/check_doc_line_citations.py`. The distinction is load-bearing: a
    filesystem walk also picks up `STATE.md`, which is gitignored
    (it is listed in `.gitignore`) and carries the retired phrasing as a dated
    session-70 log entry. Rewriting a historical log to satisfy a ratchet would be
    falsifying the record, and no reader of the repository ever sees that file.

    Note the pathspec: `*.md` is deliberate and reaches nested paths, because git's
    default pathspec `*` crosses `/` (51 of the 63 matches are under a directory).
    A `**/*.md` form would be the false-negative trap in `git-instruments-diffs.md`
    rule 13.
    """
    proc = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip("git ls-files unavailable; cannot scope the scan to tracked docs")
    return [REPO_ROOT / name for name in proc.stdout.splitlines() if name.strip()]


def test_generator_exists() -> None:
    assert GENERATOR_PATH.is_file(), (
        f"{GENERATOR_PATH.relative_to(REPO_ROOT)} is the sole writer of "
        "results/per_virus_eval_v5_mode31.csv. If it moved, re-point this test rather "
        "than deleting it."
    )


def test_generator_performs_no_training() -> None:
    """The per-virus evaluator consumes predictions; it must never produce them.

    This is the invariant the prose in README.md, docs/paper.md and
    docs/zenodo_deposition.md describes. If this test fails, the design changed and
    every one of those descriptions has to change with it, in the same commit.
    """
    tree = ast.parse(GENERATOR_PATH.read_text(encoding="utf-8"))

    fitting_calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in FITTING_METHODS
    ]
    assert not fitting_calls, (
        f"{GENERATOR_PATH.name} now calls {sorted(set(fitting_calls))}. It is "
        "documented everywhere as evaluating pooled out-of-fold predictions with no "
        "training step. Update the per-virus table descriptions in README.md, "
        "docs/paper.md (the Methods sentence and the Table 2 caption) and "
        "docs/zenodo_deposition.md in the same commit."
    )

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    offenders = sorted(
        {name for name in imported if name.startswith(TRAINING_MODULES)}
    )
    assert not offenders, (
        f"{GENERATOR_PATH.name} imports model-training modules {offenders}. See the "
        "message on the fitting-call assertion above; the same doc update applies."
    )


@pytest.mark.parametrize("phrase", RETIRED_PHRASES)
def test_no_doc_claims_the_table_is_per_virus_trained(phrase: str) -> None:
    """Ratchet over the exact wordings retired on 2026-09-17.

    Each of these asserted, in a tracked and reader-visible file, that the per-virus
    table came from per-virus training. They are false against the generator.
    """
    carriers = [
        f"{path.relative_to(REPO_ROOT).as_posix()}:{lineno}"
        for path in _tracked_markdown()
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if phrase in line
    ]
    assert not carriers, (
        f"Retired phrasing {phrase!r} is back in {carriers}. The per-virus within-CV "
        "table is one pooled model's out-of-fold predictions partitioned by virus; "
        "scripts/evaluate_per_virus.py has no training step. Describe it as a "
        "partition of the pooled OOF predictions instead."
    )
