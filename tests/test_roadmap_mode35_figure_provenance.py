"""Section 4.6 of the feature-upgrade roadmap must keep its mode-35 figure distinct
from the certified pooled mode-31 figure.

Both render to the literal 0.6055 at four decimals and differ by about 5e-06, so a
reader tidying provenance, or an automated reconciliation keyed on the literal, can
repoint that line at results/pooled_cv_metrics_mode31.csv and produce a wrong source
that every presence-based check still passes. The integrity manifest's pooled claim
deliberately excludes the roadmap from its carrier list for that reason, which means
no harness check covers this sentence; this test is the half CI can see.

The detector is exercised against a planted counterexample as well as the live file.
A guard that is never shown to fail is not evidence that it works.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_ROADMAP = REPO_ROOT / "docs" / "proposals" / "2026_feature_upgrade_roadmap.md"
_AUDIT_ARTIFACT = "results/cv_leakage_audit.csv"
_POOLED_ARTIFACT = "results/pooled_cv_metrics_mode31.csv"
_MODE35_KEY = "mode35_grouped_auc_pr"
_MODE31_KEY = "mode31_grouped_auc_pr"


def _section_4_6(text: str) -> str:
    start = text.index("### 4.6")
    rest = text[start:]
    nxt = rest.find("\n### ", 1)
    return rest if nxt == -1 else rest[:nxt]


def _grouped_measurement_sentence(text: str) -> str:
    """The clause that introduces the two peptide-grouped figures and their source."""
    region = _section_4_6(text)
    start = region.index("AUC-PR 0.6055 for mode 35")
    close = region.index(")", region.index(_MODE31_KEY))
    return region[start : close + 1]


def _sources_the_grouped_figures_correctly(sentence: str) -> bool:
    """True when the clause names the audit artifact and NOT the pooled artifact."""
    return _AUDIT_ARTIFACT in sentence and _POOLED_ARTIFACT not in sentence


def _read_metric(relative_path: str, metric: str) -> float:
    rows = (REPO_ROOT / relative_path).read_text(encoding="utf-8").splitlines()
    for row in rows[1:]:
        cells = row.split(",")
        if metric in cells:
            for cell in cells:
                try:
                    return float(cell)
                except ValueError:
                    continue
    raise AssertionError(f"metric {metric} not found in {relative_path}")


def test_the_collision_the_section_warns_about_is_still_real() -> None:
    mode35 = _read_metric(_AUDIT_ARTIFACT, _MODE35_KEY)
    pooled = _read_metric(_POOLED_ARTIFACT, "mode31_pooled_auc_pr")
    assert mode35 != pooled, "the two figures are now identical; the warning needs rewording"
    assert f"{mode35:.4f}" == f"{pooled:.4f}" == "0.6055", (
        f"the 4dp collision has gone: mode35={mode35!r} pooled={pooled!r}. "
        "Section 4.6's warning paragraph is now wrong and must be updated."
    )
    assert abs(mode35 - pooled) < 1e-05, (
        f"the two figures have diverged to {abs(mode35 - pooled)!r}; "
        "update the 5e-06 quoted in section 4.6"
    )


def test_the_roadmap_identifies_its_mode35_figure_by_metric_key() -> None:
    region = _section_4_6(_ROADMAP.read_text(encoding="utf-8"))
    for token in (_MODE35_KEY, _MODE31_KEY, "peptide_grouped_splitter"):
        assert token in region, f"section 4.6 no longer names {token}"


def test_the_grouped_figures_are_sourced_to_the_audit_artifact() -> None:
    sentence = _grouped_measurement_sentence(_ROADMAP.read_text(encoding="utf-8"))
    assert _sources_the_grouped_figures_correctly(sentence), (
        "section 4.6's grouped figures must be sourced to "
        f"{_AUDIT_ARTIFACT}, never to {_POOLED_ARTIFACT}: {sentence!r}"
    )


def test_the_detector_fires_on_a_planted_misattribution() -> None:
    """Anti-vacuity: the same check must REJECT a repointed clause."""
    sentence = _grouped_measurement_sentence(_ROADMAP.read_text(encoding="utf-8"))
    planted = sentence.replace(_AUDIT_ARTIFACT, _POOLED_ARTIFACT)
    assert planted != sentence, "the planted carrier did not change anything"
    assert not _sources_the_grouped_figures_correctly(planted), (
        "the detector passed a clause sourcing the mode-35 figure to the pooled "
        "artifact, so it cannot catch the defect this test exists for"
    )
