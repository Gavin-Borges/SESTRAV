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

The report also names the filled peptides, bounded and deduplicated. The name
sample is derived from the SAME NaN mask as the count, not from cache-index
membership, so a peptide present in the cache with a NaN value is both counted
and named. A sibling branch detected misses with `~peptide.isin(cache.index)`;
measured in isolation on a cache holding one NaN value plus one absent peptide,
that reports 1 of 2 filled rows and names only the absent one. The tests below
pin the mask so that undercount cannot come back in with the names.
"""

import pandas as pd
import pytest

from src.features import SELF_SIMILARITY_MISS_NAME_LIMIT, load_self_similarity_cache


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


def test_nan_value_in_cache_counts_as_filled(tmp_path, capsys):
    # CCCCCCCCC is IN the cache but its identity value is NaN, so it maps to NaN
    # and is filled with 0.0 exactly like an absent peptide. Testing index
    # membership alone would call this row covered, which is the whole defect.
    cache = _write_cache(
        tmp_path / "sim.csv",
        [("AAAAAAAAA", 0.4, 0.0), ("CCCCCCCCC", float("nan"), 0.0)],
    )
    df = pd.DataFrame({"peptide": ["AAAAAAAAA", "CCCCCCCCC"]})

    result = load_self_similarity_cache(cache, df)

    out = capsys.readouterr().out
    assert "WARNING: self-similarity coverage" in out
    assert "1/2 rows (50.0%)" in out
    assert result.loc[1, "self_similarity_max_identity"] == 0.0


def test_empty_frame_reports_nothing_and_does_not_divide_by_zero(tmp_path, capsys):
    cache = _write_cache(tmp_path / "sim.csv", [("AAAAAAAAA", 0.4, 0.0)])

    result = load_self_similarity_cache(cache, pd.DataFrame({"peptide": []}))

    assert len(result) == 0
    assert capsys.readouterr().out == ""


def test_nan_valued_and_absent_peptides_are_both_counted_and_named(tmp_path, capsys):
    """The decisive case for the name sample's detection basis.

    NANVALUEP is IN the cache index but carries NaN; ABSENTPEP is not in the
    index at all. Both are filled with a substituted 0.0 and are therefore
    indistinguishable from a measured non-match, so both must be counted AND
    both must be nameable. Deriving the names from `~peptide.isin(cache.index)`
    reports 1 of 2 and names only ABSENTPEP, which is the original silent-fill
    defect surviving inside its own fix.
    """
    cache = _write_cache(
        tmp_path / "sim.csv",
        [("PRESENTAA", 0.4, 0.0), ("NANVALUEP", float("nan"), 0.0)],
    )
    df = pd.DataFrame({"peptide": ["PRESENTAA", "NANVALUEP", "ABSENTPEP"]})

    result = load_self_similarity_cache(cache, df)

    out = capsys.readouterr().out
    assert "WARNING: self-similarity coverage" in out
    assert "1/3 rows (33.3%)" in out
    assert "2 filled with 0.0" in out
    # Both names, in corpus order, on the distinct basis.
    assert "2 distinct filled peptides: NANVALUEP, ABSENTPEP" in out
    # The NaN-valued row really was substituted, exactly like the absent one.
    assert result.loc[1, "self_similarity_max_identity"] == 0.0
    assert result.loc[2, "self_similarity_max_identity"] == 0.0


def test_duplicate_heavy_miss_set_names_distinct_peptides(tmp_path, capsys):
    # 30 rows of one filled peptide plus 3 others: 33 filled ROWS, 4 distinct
    # names. Without an order-preserving dedupe the whole window renders as
    # copies of DUPEPEPTI, which defeats the point of showing a sample.
    cache = _write_cache(tmp_path / "sim.csv", [("COVEREDAA", 0.7, 1.0)])
    peptides = (
        ["COVEREDAA"] + ["DUPEPEPTI"] * 30 + ["OTHERAAAA", "OTHERBBBB", "OTHERCCCC"]
    )
    df = pd.DataFrame({"peptide": peptides})

    load_self_similarity_cache(cache, df)

    out = capsys.readouterr().out
    # Row-basis counts stay row-basis: 34 rows, 1 covered, 33 filled.
    assert "1/34 rows (2.9%)" in out
    assert "33 filled with 0.0" in out
    # Sample states its own DISTINCT basis and lists each name once.
    assert (
        "4 distinct filled peptides: DUPEPEPTI, OTHERAAAA, OTHERBBBB, OTHERCCCC" in out
    )
    assert out.count("DUPEPEPTI") == 1
    assert "more" not in out


def test_name_sample_is_capped_but_the_totals_stay_true(tmp_path, capsys):
    # Far more distinct filled peptides than the cap. The emitted line must stay
    # bounded while still stating the real row and distinct totals.
    missing = [f"PEP{i:02d}AAAA" for i in range(50)]
    cache = _write_cache(tmp_path / "sim.csv", [("COVEREDAA", 0.7, 1.0)])
    df = pd.DataFrame({"peptide": ["COVEREDAA"] + missing})

    load_self_similarity_cache(cache, df)

    out = capsys.readouterr().out
    assert "1/51 rows (2.0%)" in out
    assert "50 filled with 0.0" in out
    assert "50 distinct filled peptides:" in out
    # Exactly the first SELF_SIMILARITY_MISS_NAME_LIMIT names are shown.
    assert missing[SELF_SIMILARITY_MISS_NAME_LIMIT - 1] in out
    assert missing[SELF_SIMILARITY_MISS_NAME_LIMIT] not in out
    assert sum(name in out for name in missing) == SELF_SIMILARITY_MISS_NAME_LIMIT
    # The remainder is on the DISTINCT basis, matching what the sample lists.
    assert f"and {50 - SELF_SIMILARITY_MISS_NAME_LIMIT} more" in out


def test_name_sample_stays_bounded_on_a_large_corpus(tmp_path, capsys):
    # Corpus scale must not reach the output. 2000 distinct filled peptides.
    missing = [f"Q{i:04d}AAAA" for i in range(2000)]
    cache = _write_cache(tmp_path / "sim.csv", [("COVEREDAA", 0.7, 1.0)])
    df = pd.DataFrame({"peptide": ["COVEREDAA"] + missing})

    load_self_similarity_cache(cache, df)

    out = capsys.readouterr().out
    assert "2000 filled with 0.0" in out
    assert "2000 distinct filled peptides:" in out
    assert f"and {2000 - SELF_SIMILARITY_MISS_NAME_LIMIT} more" in out
    assert sum(name in out for name in missing) == SELF_SIMILARITY_MISS_NAME_LIMIT
    # One line, and a short one, whatever the corpus size.
    assert out.count("\n") == 1
    assert len(out) < 500


def test_full_coverage_reports_unconditionally_with_no_name_clause(tmp_path, capsys):
    # Silence is ambiguous between "no fills" and "the diagnostic never ran", so
    # a clean run must still state its coverage. There is nothing to name, and
    # "0 filled with 0.0" already says so, so no distinct clause is emitted.
    cache = _write_cache(
        tmp_path / "sim.csv",
        [("AAAAAAAAA", 0.0, 0.0), ("CCCCCCCCC", 0.85, 1.0)],
    )
    df = pd.DataFrame({"peptide": ["AAAAAAAAA", "CCCCCCCCC"]})

    load_self_similarity_cache(cache, df)

    out = capsys.readouterr().out
    assert out.strip()
    assert out.startswith("Self-similarity coverage:")
    assert "WARNING" not in out
    assert "2/2 rows (100.0%)" in out
    assert "0 filled with 0.0" in out
    assert "distinct filled" not in out


def test_a_single_filled_peptide_is_named_in_the_singular(tmp_path, capsys):
    cache = _write_cache(tmp_path / "sim.csv", [("AAAAAAAAA", 0.4, 0.0)])
    df = pd.DataFrame({"peptide": ["AAAAAAAAA", "MISSINGPE"]})

    load_self_similarity_cache(cache, df)

    out = capsys.readouterr().out
    assert "1 distinct filled peptide: MISSINGPE" in out
