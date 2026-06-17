"""Unit tests for src/gold_standard.py.

All functions are pure pandas/CSV transforms, so the tests construct small
stage-output CSVs in a temp results directory and assert on the returned
report frames. Covers the per-stage validators, stem resolution, the combined
report, and negative-discrimination (primary + expanded).
"""

import pandas as pd
import pytest

from src import gold_standard as gs


EBV_PREFIX = "EBV_B95_8_panel8"
HPV_PREFIX = "HPV16_18_panel8"


def _csv(path, df):
    df.to_csv(path, index=False)
    return str(path)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_filter_gs_all_and_by_virus():
    assert len(gs._filter_gs(None)) == len(gs.GOLD_STANDARD) == 15
    ebv = gs._filter_gs("EBV")
    hpv = gs._filter_gs("HPV")
    assert all(g["virus"] == "EBV" for g in ebv)
    assert all(g["virus"] == "HPV" for g in hpv)
    assert len(ebv) + len(hpv) == 15


# --------------------------------------------------------------------------- #
# Stage validators
# --------------------------------------------------------------------------- #
def test_validate_stage1_found_flag(tmp_path):
    path = _csv(tmp_path / "p.csv", pd.DataFrame({"peptide": ["CLGGLLTMV", "ZZZZZZZZ"]}))
    out = gs.validate_stage1(path, virus="EBV")
    found = out.set_index("peptide")["stage1_found"]
    assert found["CLGGLLTMV"]  # present
    assert not found["GLCTLVAML"]  # absent from the generated set


def test_validate_stage2_strong_binder(tmp_path):
    df = pd.DataFrame(
        {"peptide": ["CLGGLLTMV", "GLCTLVAML"], "affinity": [50.0, 9000.0]}
    )
    path = _csv(tmp_path / "b.csv", df)
    out = gs.validate_stage2(path, virus="EBV", ic50_threshold=500).set_index("peptide")
    assert out.loc["CLGGLLTMV", "stage2_found"]
    assert out.loc["CLGGLLTMV", "stage2_strong_binder"]
    assert out.loc["GLCTLVAML", "stage2_found"]
    assert not out.loc["GLCTLVAML", "stage2_strong_binder"]  # affinity above threshold


def test_validate_stage2_without_affinity_column(tmp_path):
    path = _csv(tmp_path / "b.csv", pd.DataFrame({"peptide": ["CLGGLLTMV"]}))
    out = gs.validate_stage2(path, virus="EBV").set_index("peptide")
    assert out.loc["CLGGLLTMV", "stage2_found"]
    assert not out.loc["CLGGLLTMV", "stage2_strong_binder"]


def test_validate_stage4_in_top(tmp_path):
    # 100 ranked rows; CLGGLLTMV at rank 5 is within top 25%.
    rows = [{"peptide": f"DECOY{i}", "immunogenicity_score": 0.0, "rank": i} for i in range(1, 101)]
    rows[4] = {"peptide": "CLGGLLTMV", "immunogenicity_score": 0.9, "rank": 5}
    path = _csv(tmp_path / "r.csv", pd.DataFrame(rows))
    out, top_pct = gs.validate_stage4(path, virus="EBV", top_pct=25)
    assert top_pct == 25
    row = out[out["peptide"] == "CLGGLLTMV"].iloc[0]
    assert row["stage4_found"]
    assert row["in_top_25pct"]


def test_validate_stage4_missing_score_optional(tmp_path):
    path = _csv(tmp_path / "r.csv", pd.DataFrame({"peptide": ["X"], "rank": [1]}))
    out, top_pct = gs.validate_stage4(path, virus="EBV")
    assert out.empty
    assert top_pct == 25


def test_validate_stage4_missing_score_required_raises(tmp_path):
    path = _csv(tmp_path / "r.csv", pd.DataFrame({"peptide": ["X"], "rank": [1]}))
    with pytest.raises(RuntimeError, match="immunogenicity_score"):
        gs.validate_stage4(path, virus="EBV", require_score_column=True)


# --------------------------------------------------------------------------- #
# Stem resolution
# --------------------------------------------------------------------------- #
def test_resolve_stage_paths_picks_present_stem(tmp_path):
    _csv(tmp_path / f"{EBV_PREFIX}_peptides.csv", pd.DataFrame({"peptide": ["A"]}))
    paths = gs._resolve_stage_paths(str(tmp_path), EBV_PREFIX)
    assert paths["peptides"].endswith(f"{EBV_PREFIX}_peptides.csv")
    assert set(paths) == {"peptides", "binding", "ranked"}


def test_resolve_stage_paths_strict_mixed_stems_raises(tmp_path):
    # Canonical + legacy stems both present -> ambiguous under strict mode.
    _csv(tmp_path / f"{EBV_PREFIX}_peptides.csv", pd.DataFrame({"peptide": ["A"]}))
    _csv(tmp_path / "EBV_8_FASTAs_binding.csv", pd.DataFrame({"peptide": ["A"]}))
    with pytest.raises(RuntimeError, match="Mixed output stems"):
        gs._resolve_stage_paths(str(tmp_path), EBV_PREFIX, strict_stems=True)


def test_resolve_stage_paths_defaults_when_absent(tmp_path):
    paths = gs._resolve_stage_paths(str(tmp_path), EBV_PREFIX)
    # Falls back to canonical candidate even with no files present.
    assert paths["peptides"].endswith(f"{EBV_PREFIX}_peptides.csv")


# --------------------------------------------------------------------------- #
# Combined report
# --------------------------------------------------------------------------- #
def _write_full_stage_set(tmp_path, prefix, virus):
    peps = [g["peptide"] for g in gs._filter_gs(virus)]
    _csv(tmp_path / f"{prefix}_peptides.csv", pd.DataFrame({"peptide": peps}))
    _csv(
        tmp_path / f"{prefix}_binding.csv",
        pd.DataFrame({"peptide": peps, "affinity": [10.0] * len(peps)}),
    )
    _csv(
        tmp_path / f"{prefix}_ranked.csv",
        pd.DataFrame(
            {
                "peptide": peps,
                "immunogenicity_score": [0.9] * len(peps),
                "rank": list(range(1, len(peps) + 1)),
            }
        ),
    )


def test_full_validation_report(tmp_path):
    _write_full_stage_set(tmp_path, EBV_PREFIX, "EBV")
    _write_full_stage_set(tmp_path, HPV_PREFIX, "HPV")
    report = gs.full_validation_report(str(tmp_path), top_pct=100)
    assert len(report) == len(gs.GOLD_STANDARD)
    # Every gold-standard peptide should be found at stage 1 and stage 2.
    assert report["stage1_found"].all()
    assert report["stage2_found"].all()
    assert report["stage2_strong_binder"].all()
    assert "in_top_100pct" in report.columns


def test_full_validation_report_missing_files_stable(tmp_path):
    # No stage files at all: report still lists all epitopes, all False.
    report = gs.full_validation_report(str(tmp_path))
    assert len(report) == len(gs.GOLD_STANDARD)
    assert not report["stage1_found"].any()


# --------------------------------------------------------------------------- #
# Negative discrimination
# --------------------------------------------------------------------------- #
def _write_neg_set(tmp_path, prefix, virus, neg_list):
    negs = [n["peptide"] for n in neg_list if n["virus"] == virus]
    # Integrated ranking: negatives placed late (high rank = worse).
    ranked = pd.DataFrame(
        {"peptide": negs + ["FILLER"], "rank": list(range(10, 10 + len(negs))) + [1]}
    )
    # Binding ranking: negatives placed early (strong binders).
    binding = pd.DataFrame(
        {
            "peptide": negs + ["FILLER"],
            "presentation_score": [0.99 - 0.01 * i for i in range(len(negs))] + [0.1],
        }
    )
    _csv(tmp_path / f"{prefix}_ranked.csv", ranked)
    _csv(tmp_path / f"{prefix}_binding.csv", binding)


def test_validate_negative_discrimination(tmp_path):
    _write_neg_set(tmp_path, EBV_PREFIX, "EBV", gs.GOLD_STANDARD_NEGATIVES)
    _write_neg_set(tmp_path, HPV_PREFIX, "HPV", gs.GOLD_STANDARD_NEGATIVES)
    report = gs.validate_negative_discrimination(str(tmp_path))
    assert not report.empty
    assert {"integrated_rank", "binding_rank", "model_pushes_down"} <= set(report.columns)
    # At least some negatives were located in the pipeline output.
    assert report["integrated_rank"].notna().any()


def test_validate_negative_discrimination_missing_files_raises(tmp_path):
    # _resolve_stage_paths always returns the canonical path, so with no stage
    # files present the read fails rather than returning an empty frame.
    with pytest.raises(FileNotFoundError):
        gs.validate_negative_discrimination(str(tmp_path))


def test_validate_expanded_negative_discrimination(tmp_path):
    _write_neg_set(tmp_path, EBV_PREFIX, "EBV", gs.GOLD_STANDARD_NEGATIVES_EXPANDED)
    _write_neg_set(tmp_path, HPV_PREFIX, "HPV", gs.GOLD_STANDARD_NEGATIVES_EXPANDED)
    report = gs.validate_expanded_negative_discrimination(str(tmp_path))
    assert not report.empty
    assert (report["set"] == "expanded").all()


def test_validate_expanded_negative_discrimination_missing_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        gs.validate_expanded_negative_discrimination(str(tmp_path))


# --------------------------------------------------------------------------- #
# Edge cases: empty s4_df, n_v==0, ranked_path None, pushed_down None
# --------------------------------------------------------------------------- #

def test_full_report_s4df_empty_skips_append(tmp_path):
    """Covers the s4_df.empty branch: ranked file exists but has no
    immunogenicity_score column → validate_stage4 returns an empty df → the
    `if not s4_df.empty:` check is False and all_s4 stays empty."""
    prefix = EBV_PREFIX
    # Write a ranked CSV without immunogenicity_score (just peptide + rank).
    peps = [g["peptide"] for g in gs._filter_gs("EBV")]
    _csv(
        tmp_path / f"{prefix}_ranked.csv",
        pd.DataFrame({"peptide": peps, "rank": list(range(1, len(peps) + 1))}),
    )
    report = gs.full_validation_report(str(tmp_path))
    assert len(report) == len(gs.GOLD_STANDARD)
    # No stage4 data should appear in the report since s4_df was empty.
    assert "stage4_found" not in report.columns


def test_full_report_n_v_zero_skips_virus(monkeypatch, tmp_path):
    """Covers line 288: when a virus has zero gold-standard entries, the inner
    `if n_v == 0: continue` guard fires and we skip printing that virus block."""
    # Patch GOLD_STANDARD to hold only HPV entries so EBV iteration hits n_v==0.
    hpv_only = [g for g in gs.GOLD_STANDARD if g["virus"] == "HPV"]
    monkeypatch.setattr(gs, "GOLD_STANDARD", hpv_only)
    report = gs.full_validation_report(str(tmp_path))
    # Report is still built from the (now HPV-only) GOLD_STANDARD.
    assert len(report) == len(hpv_only)
    assert (report["virus"] == "HPV").all()


def test_negative_discrimination_ranked_path_none_returns_empty(monkeypatch, tmp_path):
    """Covers lines 335-336 + 380-381: _resolve_stage_paths returns ranked=None
    for all viruses → all iterations hit `continue` → results is empty → the
    `if report.empty:` guard fires."""
    orig = gs._resolve_stage_paths

    def _patch(results_dir, prefix, strict_stems=False):
        paths = orig(results_dir, prefix, strict_stems=strict_stems)
        return {**paths, "ranked": None, "binding": None}

    monkeypatch.setattr(gs, "_resolve_stage_paths", _patch)
    report = gs.validate_negative_discrimination(str(tmp_path))
    assert report.empty


def test_negative_discrimination_pushed_down_none(tmp_path):
    """Covers line 362->365: a negative peptide not present in ranked_df
    gives integ_rank=None → pushed_down stays None."""
    for virus, prefix in gs.VIRUS_FILE_MAP.items():
        # Ranked and binding files exist but contain only FILLER, not the
        # actual negative peptides, so integ_rank / bind_rank are both None.
        _csv(tmp_path / f"{prefix}_ranked.csv",
             pd.DataFrame({"peptide": ["FILLER"], "rank": [1]}))
        _csv(tmp_path / f"{prefix}_binding.csv",
             pd.DataFrame({"peptide": ["FILLER"], "presentation_score": [0.1]}))

    report = gs.validate_negative_discrimination(str(tmp_path))
    assert not report.empty
    # pushed_down is None/NaN for all rows since the peptides weren't found.
    assert report["model_pushes_down"].isna().all() or (report["model_pushes_down"] == False).all()  # noqa: E712


def test_expanded_negative_binding_path_none_returns_empty(monkeypatch, tmp_path):
    """Covers lines 413-414 + 459-460 (expanded variant): binding=None makes
    the loop continue for all viruses → empty report → empty-guard fires."""
    orig = gs._resolve_stage_paths

    def _patch(results_dir, prefix, strict_stems=False):
        paths = orig(results_dir, prefix, strict_stems=strict_stems)
        return {**paths, "ranked": None, "binding": None}

    monkeypatch.setattr(gs, "_resolve_stage_paths", _patch)
    report = gs.validate_expanded_negative_discrimination(str(tmp_path))
    assert report.empty


def test_expanded_negative_discrimination_pushed_down_none(tmp_path):
    """Covers lines 440->443 (expanded): negative peptide absent from ranked
    gives pushed_down=None."""
    for virus, prefix in gs.VIRUS_FILE_MAP.items():
        _csv(tmp_path / f"{prefix}_ranked.csv",
             pd.DataFrame({"peptide": ["FILLER"], "rank": [1]}))
        _csv(tmp_path / f"{prefix}_binding.csv",
             pd.DataFrame({"peptide": ["FILLER"], "presentation_score": [0.1]}))

    report = gs.validate_expanded_negative_discrimination(str(tmp_path))
    assert not report.empty
    assert report["model_pushes_down"].isna().all() or (report["model_pushes_down"] == False).all()  # noqa: E712
