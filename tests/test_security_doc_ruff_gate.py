"""SECURITY.md must describe the Ruff gate that actually runs.

The "Strict Warning Enforcement" risk acceptance names Ruff as the compensating
control for not running `-W error`, so what that gate covers is load-bearing for
the claim. It previously read "set to fail on any warning or error", which is
wider than the configured gate: Pycodestyle's W category is not selected at all.

These tests pin the corrected claim against pyproject.toml in both directions,
so the doc cannot drift back and the config cannot silently widen underneath it.
"""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SECURITY_MD = REPO_ROOT / "SECURITY.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _ruff_lint_config() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["ruff"]["lint"]


# The retracted sentence, verbatim. Matching the FULL sentence rather than the
# short phrase is deliberate: the correction quotes the phrase to say what was
# wrong, and a substring check would fire on that retraction itself.
RETRACTED_SENTENCE = (
    "We enforce strict warnings exclusively through our linting pipeline (Ruff) "
    "which is set to fail on any warning or error"
)


def test_security_md_does_not_restate_the_retracted_claim() -> None:
    """The retracted wording must not come back as an assertion."""
    text = SECURITY_MD.read_text(encoding="utf-8")
    assert RETRACTED_SENTENCE not in text


def test_security_md_select_list_matches_pyproject() -> None:
    """The doc names the enabled rule set; pyproject is the source of truth."""
    text = SECURITY_MD.read_text(encoding="utf-8")
    select = _ruff_lint_config()["select"]
    for rule in select:
        assert f"`{rule}`" in text, f"SECURITY.md does not name enabled ruff rule {rule}"


def test_w_category_is_really_unselected() -> None:
    """The doc says W is not selected. Fail if the config widens to include it."""
    select = _ruff_lint_config()["select"]
    assert not any(rule.startswith("W") for rule in select), (
        "a W rule is now selected, so SECURITY.md's 'not selected at all' is stale"
    )


def test_ignore_count_matches_the_documented_nine() -> None:
    """The doc says nine rules are ignored repository-wide."""
    assert len(_ruff_lint_config()["ignore"]) == 9
