"""src/train_classifier.py's module docstring names every feature mode it builds.

Why this test exists
--------------------
The module docstring - what `pydoc` and an editor show for the module - listed
only modes 21 and 30 while the script builds eleven (RECOGNIZED_FEATURE_MODES,
which a separate test ties to `--feature-mode`'s choices). It also said
evaluation "uses stratified 5-fold cross-validation", while the CLI groups the
folds by peptide unless `--cv-group-by none` is passed (to-do LQ-9 item 4).

Pinned here, reading the source through the AST so nothing heavy is imported:
- the docstring's "Feature modes" list is exactly RECOGNIZED_FEATURE_MODES;
- it states no per-mode input requirement, which each input flag's own help
  states once (the binding-matrix phrase is single-carrier by design, see
  tests/test_binding_matrix_requirement_wording.py);
- it does not describe the CLI's cross-validation as plain stratified.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "src" / "train_classifier.py"


def _module() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _recognized_modes() -> set[str]:
    for node in _module().body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "RECOGNIZED_FEATURE_MODES":
            return {str(mode) for mode in ast.literal_eval(node.value)}
    raise AssertionError("RECOGNIZED_FEATURE_MODES not found")


def _feature_mode_section(doc: str) -> list[str]:
    lines = doc.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Feature modes"))
    section = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        section.append(line)
    return section


def test_the_docstring_lists_exactly_the_recognized_modes() -> None:
    doc = ast.get_docstring(_module()) or ""
    listed = {m.group(1) for line in _feature_mode_section(doc) if (m := re.match(r"^  (\S+)\s", line))}
    assert listed == _recognized_modes()


def test_the_docstring_leaves_input_requirements_to_the_flags() -> None:
    doc = ast.get_docstring(_module()) or ""
    assert "required for every feature mode" not in doc
    assert not re.search(r"\brequires? --(binding-matrix|antigen-processing-cache|self-similarity-cache)\b", doc,
                         re.IGNORECASE)


def test_the_docstring_does_not_call_the_cli_cross_validation_plain_stratified() -> None:
    doc = ast.get_docstring(_module()) or ""
    assert "stratified 5-fold cross-validation" not in doc
    assert "--cv-group-by" in doc


def test_the_lopo_help_does_not_name_stratified_k_fold_as_the_alternative() -> None:
    """--lopo replaces whatever k-fold --cv-group-by selected, which is peptide-grouped by default."""
    for node in ast.walk(_module()):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--lopo"):
            help_kw = next(kw.value for kw in node.keywords if kw.arg == "help")
            text = help_kw.value if isinstance(help_kw, ast.Constant) else ast.unparse(help_kw)
            assert "StratifiedKFold" not in text
            assert "--cv-group-by" in text
            return
    raise AssertionError("--lopo add_argument call not found")
