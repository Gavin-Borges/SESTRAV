"""Unit and integration tests for scripts/benchmark_syfpeithi.py.

Coverage targets:
  _hamming1          - True/False for all distance cases
  _lookup_in_oof     - exact, hamming1, not_in_oof paths
  run_benchmark      - recall/enrichment arithmetic, output file I/O,
                       all-not-in-oof edge case, hamming1 matching in full flow
  main()             - --dry-run, missing file (returns 1), valid run (returns 0)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.benchmark_syfpeithi import (
    RECALL_CUTOFFS,
    SYFPEITHI_CANONICAL,
    SYFPEITHI_TRAINING_VARIANTS,
    _hamming1,
    _lookup_in_oof,
    main,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_oof_df(
    peptides: list[str], scores: list[float], labels: list[int] | None = None
) -> pd.DataFrame:
    n = len(peptides)
    return pd.DataFrame(
        {
            "peptide": peptides,
            "score": scores,
            "label": labels if labels is not None else [1] * n,
        }
    )


def _background(n: int, base: float = 0.0, step: float = 0.04) -> list[tuple[str, float]]:
    """Generate n distinct background peptides (9-mer) with monotonically increasing scores."""
    return [(f"BG{i:03d}AAAA", round(base + i * step, 6)) for i in range(n)]


def _build_oof_csv(
    tmp_path: Path,
    canonical_entries: list[tuple[str, float, int]],
    background: list[tuple[str, float]],
    feature_mode: int = 31,
) -> Path:
    """Write a minimal OOF predictions CSV and return its Path."""
    rows = [
        {"peptide": p, "score": s, "label": lab, "feature_mode": feature_mode}
        for p, s, lab in canonical_entries
    ]
    rows += [
        {"peptide": p, "score": s, "label": 0, "feature_mode": feature_mode} for p, s in background
    ]
    path = tmp_path / "oof.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Structural / metadata
# ---------------------------------------------------------------------------


def test_canonical_set_size():
    assert len(SYFPEITHI_CANONICAL) == 10


def test_canonical_required_keys():
    required = {"peptide", "protein", "virus", "hla", "syfpeithi_score", "source"}
    for entry in SYFPEITHI_CANONICAL:
        assert required.issubset(entry.keys()), f"Missing keys in {entry}"


def test_canonical_peptide_lengths():
    for entry in SYFPEITHI_CANONICAL:
        assert len(entry["peptide"]) == 9, f"Expected 9-mer: {entry['peptide']!r}"


def test_canonical_hla_a0201():
    assert all(e["hla"] == "A*02:01" for e in SYFPEITHI_CANONICAL)


def test_canonical_virus_set():
    assert {e["virus"] for e in SYFPEITHI_CANONICAL} == {"EBV", "HPV16"}


def test_recall_cutoffs_values():
    assert RECALL_CUTOFFS == [5, 10, 25]


def test_training_variants_string_keys_and_values():
    for k, v in SYFPEITHI_TRAINING_VARIANTS.items():
        assert isinstance(k, str) and isinstance(v, str)


def test_training_variants_values_in_canonical():
    canonical_peps = {e["peptide"] for e in SYFPEITHI_CANONICAL}
    for v in SYFPEITHI_TRAINING_VARIANTS.values():
        assert v in canonical_peps, f"Variant target {v!r} not in canonical set"


# ---------------------------------------------------------------------------
# _hamming1
# ---------------------------------------------------------------------------


def test_hamming1_identical_is_false():
    assert _hamming1("FLYALALLL", "FLYALALLL") is False


def test_hamming1_one_sub_is_true():
    assert _hamming1("FLYALALLL", "FLYALAXLL") is True


def test_hamming1_two_subs_is_false():
    assert _hamming1("FLYALALLL", "FLYALAXLX") is False


def test_hamming1_different_lengths_is_false():
    assert _hamming1("AAAAAA", "AAAAAAAA") is False


def test_hamming1_all_different_is_false():
    assert _hamming1("AAAAAAAAA", "CCCCCCCCC") is False


def test_hamming1_known_variant_clgglltmv():
    # CLGGLLYMV is a documented 1-substitution variant of CLGGLLTMV
    assert _hamming1("CLGGLLTMV", "CLGGLLYMV") is True


def test_hamming1_length_one_true():
    assert _hamming1("A", "C") is True


def test_hamming1_length_one_false():
    assert _hamming1("A", "A") is False


# ---------------------------------------------------------------------------
# _lookup_in_oof
# ---------------------------------------------------------------------------


def test_lookup_exact_match():
    df = _make_oof_df(["FLYALALLL", "AAAAAAAAA"], [0.9, 0.1])
    ref = next(r for r in SYFPEITHI_CANONICAL if r["peptide"] == "FLYALALLL")
    result = _lookup_in_oof(ref, df, "peptide", "score")
    assert result["match_type"] == "exact"
    assert abs(result["sestrav_score"] - 0.9) < 1e-9


def test_lookup_exact_match_case_insensitive():
    df = _make_oof_df(["flyalalll", "AAAAAAAAA"], [0.85, 0.1])
    ref = next(r for r in SYFPEITHI_CANONICAL if r["peptide"] == "FLYALALLL")
    result = _lookup_in_oof(ref, df, "peptide", "score")
    assert result["match_type"] == "exact"
    assert abs(result["sestrav_score"] - 0.85) < 1e-9


def test_lookup_hamming1_match():
    # CLGGLLYMV is Hamming1 of CLGGLLTMV (T→Y at position 7)
    df = _make_oof_df(["CLGGLLYMV", "AAAAAAAAA"], [0.75, 0.1])
    ref = next(r for r in SYFPEITHI_CANONICAL if r["peptide"] == "CLGGLLTMV")
    result = _lookup_in_oof(ref, df, "peptide", "score")
    assert result["match_type"].startswith("hamming1:")
    assert "CLGGLLYMV" in result["match_type"]
    assert abs(result["sestrav_score"] - 0.75) < 1e-9


def test_lookup_not_in_oof():
    df = _make_oof_df(["AAAAAAAAA", "CCCCCCCCC"], [0.5, 0.3])
    ref = next(r for r in SYFPEITHI_CANONICAL if r["peptide"] == "GLCTLVAML")
    result = _lookup_in_oof(ref, df, "peptide", "score")
    assert result["match_type"] == "not_in_oof"
    assert result["sestrav_score"] is None
    assert result["sestrav_label"] is None
    assert result["rank_percentile"] is None


def test_lookup_rank_percentile_always_none():
    df = _make_oof_df(["FLYALALLL"], [0.9])
    ref = next(r for r in SYFPEITHI_CANONICAL if r["peptide"] == "FLYALALLL")
    result = _lookup_in_oof(ref, df, "peptide", "score")
    # run_benchmark fills rank_percentile; _lookup_in_oof always returns None
    assert result["rank_percentile"] is None


def test_lookup_uses_label_column():
    df = _make_oof_df(["FLYALALLL"], [0.9], labels=[1])
    ref = next(r for r in SYFPEITHI_CANONICAL if r["peptide"] == "FLYALALLL")
    result = _lookup_in_oof(ref, df, "peptide", "score")
    assert result["sestrav_label"] == 1


# ---------------------------------------------------------------------------
# run_benchmark - recall arithmetic
# ---------------------------------------------------------------------------
#
# Test fixture: 20 peptides total (3 canonical + 17 background).
# With pandas rank(ascending=False, pct=True)*100:
#   rank 1 → pctile  5.0  (≤ 5%)
#   rank 2 → pctile 10.0  (≤ 10%)
#   rank 3 → pctile 15.0  (≤ 25%, not ≤ 10%)
#   ranks 4-20 → background, all below 0.8
#
# Expected: recall@5%=1/3, recall@10%=2/3, recall@25%=3/3.


def _three_canonical_oof(tmp_path: Path) -> Path:
    """OOF with FLYALALLL/FAFRDLCIV/RAHYNIVTF at ranks 1/2/3 out of 20."""
    canonical_entries = [
        ("FLYALALLL", 1.0, 1),
        ("FAFRDLCIV", 0.9, 1),
        ("RAHYNIVTF", 0.8, 0),
    ]
    return _build_oof_csv(tmp_path, canonical_entries, _background(17))


def test_run_benchmark_returns_dict(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    assert isinstance(result, dict)
    assert result["benchmark"] == "syfpeithi"


def test_run_benchmark_n_evaluable(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    assert result["n_evaluable"] == 3
    assert result["n_not_in_oof"] == 7


def test_run_benchmark_n_oof_peptides(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    assert result["n_oof_peptides"] == 20


def test_run_benchmark_recall_at_5pct(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    assert abs(result["recall_cutoffs"][5]["recall"] - 1 / 3) < 1e-6


def test_run_benchmark_recall_at_10pct(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    assert abs(result["recall_cutoffs"][10]["recall"] - 2 / 3) < 1e-6


def test_run_benchmark_recall_at_25pct(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    assert abs(result["recall_cutoffs"][25]["recall"] - 1.0) < 1e-6


def test_run_benchmark_enrichment_at_5pct(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    expected = (1 / 3) / 0.05
    assert abs(result["recall_cutoffs"][5]["enrichment"] - expected) < 1e-4


def test_run_benchmark_baseline_values(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    assert abs(result["recall_cutoffs"][5]["baseline"] - 0.05) < 1e-9
    assert abs(result["recall_cutoffs"][10]["baseline"] - 0.10) < 1e-9
    assert abs(result["recall_cutoffs"][25]["baseline"] - 0.25) < 1e-9


def test_run_benchmark_n_in_top_k(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    assert result["recall_cutoffs"][5]["n_in_top_k"] == 1
    assert result["recall_cutoffs"][10]["n_in_top_k"] == 2
    assert result["recall_cutoffs"][25]["n_in_top_k"] == 3


# ---------------------------------------------------------------------------
# run_benchmark - all-not-in-oof edge case
# ---------------------------------------------------------------------------


def test_run_benchmark_all_not_in_oof(tmp_path):
    path = _build_oof_csv(tmp_path, [], _background(20))
    result = run_benchmark(path, None)
    assert result["n_evaluable"] == 0
    assert result["n_not_in_oof"] == 10
    for k in RECALL_CUTOFFS:
        assert result["recall_cutoffs"][k]["recall"] == 0.0


# ---------------------------------------------------------------------------
# run_benchmark - hamming1 matching in full flow
# ---------------------------------------------------------------------------


def test_run_benchmark_hamming1_scored(tmp_path):
    # CLGGLLYMV is Hamming1 of CLGGLLTMV; should appear as hamming1 match
    canonical_entries = [("CLGGLLYMV", 0.95, 1)]
    path = _build_oof_csv(tmp_path, canonical_entries, _background(9))
    result = run_benchmark(path, None)
    entry = next(e for e in result["per_epitope"] if e["peptide"] == "CLGGLLTMV")
    assert entry["match_type"].startswith("hamming1:")
    assert entry["rank_percentile"] is not None


# ---------------------------------------------------------------------------
# run_benchmark - output file I/O
# ---------------------------------------------------------------------------


def test_run_benchmark_writes_valid_json(tmp_path):
    path = _three_canonical_oof(tmp_path)
    output_path = tmp_path / "out.json"
    run_benchmark(path, output_path)
    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
    assert data["benchmark"] == "syfpeithi"
    assert "per_epitope" in data
    assert "recall_cutoffs" in data


def test_run_benchmark_no_file_when_output_none(tmp_path):
    path = _build_oof_csv(tmp_path, [], _background(10))
    run_benchmark(path, None)
    assert not list(tmp_path.glob("*.json"))


def test_run_benchmark_creates_output_dir(tmp_path):
    path = _three_canonical_oof(tmp_path)
    nested_output = tmp_path / "subdir" / "nested" / "out.json"
    run_benchmark(path, nested_output)
    assert nested_output.exists()


# ---------------------------------------------------------------------------
# run_benchmark - column detection
# ---------------------------------------------------------------------------


def test_run_benchmark_immunogenicity_score_column(tmp_path):
    rows = [{"peptide": "FLYALALLL", "immunogenicity_score": 0.9, "label": 1}]
    rows += [
        {"peptide": f"BG{i:03d}AAAA", "immunogenicity_score": 0.1, "label": 0} for i in range(9)
    ]
    path = tmp_path / "oof_alt.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = run_benchmark(path, None)
    flyalalll = next(e for e in result["per_epitope"] if e["peptide"] == "FLYALALLL")
    assert flyalalll["sestrav_score"] is not None


def test_run_benchmark_feature_mode_from_csv(tmp_path):
    path = _three_canonical_oof(tmp_path)
    result = run_benchmark(path, None)
    assert result["feature_mode"] == 31


def test_run_benchmark_feature_mode_unknown_when_absent(tmp_path):
    rows = [{"peptide": f"BG{i:03d}AAAA", "score": float(i) * 0.05} for i in range(10)]
    path = tmp_path / "no_fmode.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = run_benchmark(path, None)
    assert result["feature_mode"] == "unknown"


# ---------------------------------------------------------------------------
# run_benchmark - per_epitope structure
# ---------------------------------------------------------------------------


def test_run_benchmark_per_epitope_count(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    assert len(result["per_epitope"]) == 10


def test_run_benchmark_per_epitope_required_keys(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    required = {
        "peptide",
        "protein",
        "virus",
        "hla",
        "syfpeithi_score",
        "source",
        "match_type",
        "sestrav_score",
        "sestrav_label",
        "rank_percentile",
    }
    for entry in result["per_epitope"]:
        assert required.issubset(entry.keys()), f"Missing keys in {entry}"


def test_run_benchmark_rank_percentile_none_for_not_in_oof(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    not_in_oof = [e for e in result["per_epitope"] if e["match_type"] == "not_in_oof"]
    assert all(e["rank_percentile"] is None for e in not_in_oof)


def test_run_benchmark_rank_percentile_set_for_matches(tmp_path):
    result = run_benchmark(_three_canonical_oof(tmp_path), None)
    matched = [e for e in result["per_epitope"] if e["match_type"] != "not_in_oof"]
    assert all(e["rank_percentile"] is not None for e in matched)
    assert all(0 < e["rank_percentile"] <= 100 for e in matched)


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


def test_main_dry_run(capsys):
    ret = main(["--dry-run"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "SYFPEITHI reference set" in out
    assert "FLYALALLL" in out
    assert "FAFRDLCIV" in out


def test_main_dry_run_prints_all_ten(capsys):
    main(["--dry-run"])
    out = capsys.readouterr().out
    for entry in SYFPEITHI_CANONICAL:
        assert entry["peptide"] in out


def test_main_missing_predictions_returns_1(tmp_path, capsys):
    missing = tmp_path / "nonexistent.csv"
    ret = main(["--predictions", str(missing)])
    assert ret == 1


def test_main_valid_run_returns_0(tmp_path):
    path = _three_canonical_oof(tmp_path)
    output = tmp_path / "result.json"
    ret = main(["--predictions", str(path), "--output", str(output)])
    assert ret == 0
    assert output.exists()


def test_main_output_json_is_valid(tmp_path):
    path = _three_canonical_oof(tmp_path)
    output = tmp_path / "result.json"
    main(["--predictions", str(path), "--output", str(output)])
    with open(output) as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert data["n_evaluable"] == 3
