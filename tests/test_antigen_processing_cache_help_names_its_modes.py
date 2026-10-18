"""The --antigen-processing-cache help names exactly the feature modes that need the cache.

`train_models`' feature-selection chain gives each recognised feature mode its own branch.
Only the branches for modes 33 and 35 read `antigen_processing_cache_path`, and each raises
`--antigen-processing-cache is required for feature-mode <mode>` when it is None; those two
raises are exercised by `tests/test_characterize_train_classifier.py`. The CLI help read
"required for --feature-mode 33+", which is wrong for modes 50, 51 and 166: each is
numerically above 33, and none of their branches reads the cache.

This file ties the help to the code rather than to a literal. It derives, from the dispatch
chain itself, the set of modes whose branch reads the cache, and asserts that the help's
"required for --feature-mode ..." parenthetical names that set and no other. A mode that
starts reading the cache therefore fails here until the help says so.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CLASSIFIER = REPO_ROOT / "src" / "train_classifier.py"
TREE = ast.parse(TRAIN_CLASSIFIER.read_text(encoding="utf-8"))

CACHE_ARGUMENT = "antigen_processing_cache_path"
CACHE_FLAG = "--antigen-processing-cache"


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


def _modes_reading_the_cache() -> set[object]:
    return {
        mode
        for mode, body in _dispatch_branches()
        if any(
            isinstance(node, ast.Name) and node.id == CACHE_ARGUMENT
            for statement in body
            for node in ast.walk(statement)
        )
    }


def _help_of(flag: str) -> str:
    """The help string of the `parser.add_argument(flag, ...)` call in the module."""
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


def test_only_modes_33_and_35_read_the_antigen_processing_cache():
    modes = _modes_reading_the_cache()
    assert modes == {33, 35}
    # The modes the old "33+" wording wrongly took in are in the chain and do not read it.
    chain = {mode for mode, _ in _dispatch_branches()}
    assert {50, 51, 166} <= chain
    assert not ({50, 51, 166} & modes)


def test_the_help_names_exactly_the_modes_that_read_the_cache():
    help_text = _help_of(CACHE_FLAG)
    match = re.search(r"required for --feature-mode ([^)]*)\)", help_text)
    assert match, f"{CACHE_FLAG} help no longer states which modes require it: {help_text!r}"
    named = {int(number) for number in re.findall(r"\d+", match.group(1))}
    assert named == _modes_reading_the_cache()
    assert "33+" not in help_text
