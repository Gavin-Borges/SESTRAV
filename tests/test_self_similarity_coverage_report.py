"""Coverage reporting for the self-similarity cache join.

load_self_similarity_cache fills 0.0 for any peptide absent from the cache.
0.0 is a real value in this feature's scale (compute_self_similarity returns it
for a genuine non-match), so a filled row is indistinguishable from a measured
one. Before this report existed the fill was silent, and the guard that was
supposed to announce it could never fire: it subtracted two quantities that are
equal by construction, so its count was identically zero.

These tests pin the report and, critically, pin that a peptide which IS in the
cache with a genuine 0.0 is not counted as missing. That distinction is the
whole defect.
"""

import pandas as pd
import pytest

from src.features import load_self_similarity_cache


def _write_cache(path, rows):
    pd.DataFrame(
        rows, columns=["peptide", "self_similarity_max_identity", "self_similarity_exact_match"]
    ).to_csv(path, index=False)
    return str(path)


def test_partial_coverage_warns_and_counts_only_absent_peptides(tmp_path, capsys):
    # AAAAAAAAA is IN the cache with a genuine 0.0. DDDDDDDDD and EEEEEEEEE are absent.
    cache = _write_cache(
        tmp_path / "sim.csv",
        [("AAAAAAAAA", 0.0, 0.0), ("CCCCCCCCC", 0.85, 1.0)],
    )
    df = pd.DataFrame({"peptide": ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE"]})

    load_self_similarity_cache(cache, df)

    out = capsys.readouterr().out
    assert "WARNING: self-similarity coverage" in out
    assert "2/4 rows (50.0%)" in out
    assert "2 filled with 0.0" in out
    assert "sim.csv" in out


def test_genuine_zero_in_cache_is_not_reported_as_missing(tmp_path, capsys):
    # Every peptide is present; three of them legitimately score 0.0. The old
    # guard could not tell these apart from absent peptides.
    cache = _write_cache(
        tmp_path / "sim.csv",
        [("AAAAAAAAA", 0.0, 0.0), ("CCCCCCCCC", 0.0, 0.0), ("DDDDDDDDD", 0.0, 0.0)],
    )
    df = pd.DataFrame({"peptide": ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD"]})

    load_self_similarity_cache(cache, df)

    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "3/3 rows (100.0%)" in out
    assert "0 filled with 0.0" in out


def test_empty_cache_reports_total_miss(tmp_path, capsys):
    cache = _write_cache(tmp_path / "sim.csv", [])
    df = pd.DataFrame({"peptide": ["AAAAAAAAA", "CCCCCCCCC"]})

    load_self_similarity_cache(cache, df)

    out = capsys.readouterr().out
    assert "WARNING: self-similarity coverage" in out
    assert "0/2 rows (0.0%)" in out


@pytest.mark.parametrize("n_absent", [0, 1, 2])
def test_returned_frame_is_unchanged_by_the_report(tmp_path, n_absent):
    # The report must be observation only. Values, length and null-freeness are
    # exactly what they were before it existed.
    cache = _write_cache(
        tmp_path / "sim.csv",
        [("AAAAAAAAA", 0.0, 0.0), ("CCCCCCCCC", 0.85, 1.0)],
    )
    peptides = ["AAAAAAAAA", "CCCCCCCCC"] + ["D" * 9, "E" * 9][:n_absent]
    df = pd.DataFrame({"peptide": peptides})

    result = load_self_similarity_cache(cache, df)

    assert len(result) == len(peptides)
    assert not result.isnull().any().any()
    assert result.loc[0, "self_similarity_max_identity"] == 0.0
    assert result.loc[1, "self_similarity_max_identity"] == 0.85
    assert result.loc[1, "self_similarity_exact_match"] == 1.0
    for i in range(2, 2 + n_absent):
        assert result.loc[i, "self_similarity_max_identity"] == 0.0
        assert result.loc[i, "self_similarity_exact_match"] == 0.0
