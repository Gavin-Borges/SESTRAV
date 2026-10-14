"""The --feature-mode help no longer marks mode 51 alone as needing --binding-matrix.

`src/train_classifier.py`'s --feature-mode help listed the eleven modes and, against mode 51
only, added "requires --binding-matrix". That is true of 51 and of every other mode except
21: in `train_models`' feature-mode chain each branch reads `binding_matrix_path`, and 21 is
the chain's final `else`, which trains without one. A note on one mode invites the reader to
infer the other nine do not need the matrix, so the note is removed rather than copied nine
times. The rule is stated once, by the --binding-matrix flag's own help, which
tests/test_binding_matrix_requirement_wording.py pins on the branch that rewrites it to
"required for every feature mode except 21". Adding that phrase here as well would fail that
test's one-carrier-per-file assertion, by its own design.

The structural half below derives the set from the dispatch chain itself, so it states the
reason the note was misleading without relying on a literal list of modes.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CLASSIFIER = REPO_ROOT / "src" / "train_classifier.py"
TREE = ast.parse(TRAIN_CLASSIFIER.read_text(encoding="utf-8"))


def _mode_of(test: ast.expr) -> object | None:
    """The constant a `feature_mode == <constant>` test compares against, else None."""
    if (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "feature_mode"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and isinstance(test.comparators[0], ast.Constant)
    ):
        return test.comparators[0].value
    return None


def _dispatch_branches() -> list[tuple[object, list[ast.stmt]]]:
    """The `if feature_mode == ...: / elif ...` chain in `train_models`, as (mode, body)."""
    train_models = next(
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef) and node.name == "train_models"
    )
    for node in ast.walk(train_models):
        if not (isinstance(node, ast.If) and _mode_of(node.test) is not None):
            continue
        branches = []
        current: ast.stmt | None = node
        while isinstance(current, ast.If) and _mode_of(current.test) is not None:
            branches.append((_mode_of(current.test), current.body))
            orelse = current.orelse
            current = orelse[0] if len(orelse) == 1 and isinstance(orelse[0], ast.If) else None
        if len(branches) >= 5:
            return branches
    raise AssertionError("no feature_mode dispatch chain found in train_models")


def _help_of(flag: str) -> str:
    for node in ast.walk(TREE):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == flag
        ):
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            return ast.literal_eval(keywords["help"])
    raise AssertionError(f"no add_argument call for {flag}")


def test_every_branch_of_the_chain_reads_the_binding_matrix():
    branches = _dispatch_branches()
    reading = {
        mode
        for mode, body in branches
        if any(
            isinstance(node, ast.Name) and node.id == "binding_matrix_path"
            for statement in body
            for node in ast.walk(statement)
        )
    }
    # Every mode with its own branch reads the matrix, so 51 is one of many, not the exception.
    assert reading == {mode for mode, _ in branches}
    assert 51 in reading and len(reading) > 1


def test_the_feature_mode_help_leaves_the_matrix_rule_to_its_flag():
    help_text = _help_of("--feature-mode")
    assert "--binding-matrix" not in help_text
    # The mode list itself is intact: mode 51 is still described.
    assert "51 (50+per-allele TCR contact weights, 55 cols)" in help_text
