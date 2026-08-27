import pandas as pd

from scripts.benchmark_runner import (
    build_clean_subset,
    compute_contamination,
    filter_contaminated_peptides,
)

# Shared by the compute_contamination tests below. Four distinct 9-mers, so an
# overlap fraction is exact rather than rounded.
EVAL_PEPTIDES = ["CLGGLLTMV", "RAKFKQLLA", "CDEFGHIJK", "AAAAAAAAB"]


def _eval_frame():
    return pd.DataFrame({"peptide": EVAL_PEPTIDES, "label": [1, 0, 1, 0]})


def _training_csv(tmp_path, peptides, column="peptide"):
    path = tmp_path / "training.csv"
    pd.DataFrame({column: peptides}).to_csv(path, index=False)
    return path


def test_filter_contaminated_peptides_exact_match():
    # Exact match should be caught
    train = ["CLGGLLTMV", "RAKFKQLL"]
    eval_seqs = ["CLGGLLTMV", "AAAAAAAAB"]
    clean = filter_contaminated_peptides(train, eval_seqs)
    assert clean == ["AAAAAAAAB"]


def test_filter_contaminated_peptides_substring():
    # Evaluation peptide is substring of training peptide
    train = ["CLGGLLTMVAG", "RAKFKQLL"]
    eval_seqs = ["CLGGLLTMV", "AAAAAAAAB"]
    clean = filter_contaminated_peptides(train, eval_seqs)
    # CLGGLLTMV is a substring of CLGGLLTMVAG, so it should be filtered out
    assert clean == ["AAAAAAAAB"]


def test_filter_contaminated_peptides_no_leakage():
    train = ["CLGGLLTMV", "RAKFKQLL"]
    eval_seqs = ["CDEFGHIJK", "AAAAAAAAB"]
    clean = filter_contaminated_peptides(train, eval_seqs)
    assert clean == ["CDEFGHIJK", "AAAAAAAAB"]


def test_filter_contaminated_peptides_case_insensitivity():
    train = ["clgglltmv", "RAKFKQLL"]
    eval_seqs = ["CLGGLLTMV", "aaaaaaaab"]
    clean = filter_contaminated_peptides(train, eval_seqs)
    assert clean == ["aaaaaaaab"]

    # Evaluation is lowercase and training is uppercase
    train2 = ["CLGGLLTMV"]
    eval_seqs2 = ["clgglltmv"]
    clean2 = filter_contaminated_peptides(train2, eval_seqs2)
    assert clean2 == []


def test_filter_contaminated_peptides_empty():
    assert filter_contaminated_peptides([], ["CLGGLLTMV"]) == ["CLGGLLTMV"]
    assert filter_contaminated_peptides(["CLGGLLTMV"], []) == []


# -- compute_contamination / build_clean_subset ------------------------------
#
# The five tests above cover the pure trie helper only. The two functions the
# CI step actually calls were unexercised, and their missing-corpus branch is
# the one that runs on every CI job, because the comparison corpus is gitignored
# and therefore absent from every checkout.


def test_compute_contamination_missing_file_is_skipped_not_passed(tmp_path):
    """An absent corpus must yield no verdict, not a passing one."""
    result = compute_contamination(_eval_frame(), tmp_path / "does_not_exist.csv")

    assert result["gate_status"] == "skipped"
    # The regression this pins: gate_pass was True here, so a manifest recorded a
    # contamination gate as passed on a comparison that never ran.
    assert result["gate_pass"] is None
    assert "skipped" in result["note"].lower()


def test_compute_contamination_missing_peptide_column_is_skipped(tmp_path):
    """A corpus without a 'peptide' column cannot be compared against either."""
    path = _training_csv(tmp_path, ["CLGGLLTMV"], column="sequence")
    result = compute_contamination(_eval_frame(), path)

    assert result["gate_status"] == "skipped"
    assert result["gate_pass"] is None


def test_compute_contamination_flags_overlap_above_cap(tmp_path):
    """Three of four eval peptides in the corpus is 75%, above the 30% cap."""
    path = _training_csv(tmp_path, EVAL_PEPTIDES[:3])
    result = compute_contamination(_eval_frame(), path)

    assert result["gate_status"] == "fail"
    assert result["gate_pass"] is False
    assert result["overlap_count"] == 3
    assert result["overlap_rate"] == 0.75


def test_compute_contamination_passes_with_no_overlap(tmp_path):
    path = _training_csv(tmp_path, ["WWWWWWWWW", "YYYYYYYYY"])
    result = compute_contamination(_eval_frame(), path)

    assert result["gate_status"] == "pass"
    assert result["gate_pass"] is True
    assert result["overlap_count"] == 0
    assert result["overlap_rate"] == 0.0


def test_compute_contamination_key_set_is_identical_in_every_branch(tmp_path):
    """
    All three shapes must carry the same keys.

    run_manifest.json embeds this dict verbatim, so a consumer written against
    one branch used to break on another: the skip branches carried 'note' and no
    counts, while the computed branch carried counts and no 'note'.
    """
    skipped = compute_contamination(_eval_frame(), tmp_path / "absent.csv")
    failed = compute_contamination(_eval_frame(), _training_csv(tmp_path, EVAL_PEPTIDES[:3]))
    passed = compute_contamination(
        _eval_frame(), _training_csv(tmp_path, ["WWWWWWWWW"])
    )

    assert set(skipped) == set(failed) == set(passed)
    assert {r["gate_status"] for r in (skipped, failed, passed)} == {
        "skipped",
        "fail",
        "pass",
    }


def test_build_clean_subset_returns_none_when_corpus_absent(tmp_path):
    """
    None, never the unfiltered frame.

    Returning df made a frame in which nothing had been excluded look identical
    to one in which nothing needed excluding, and the caller then labelled it
    'clean_holdout' - a contamination-excluded row that excluded nothing.
    """
    df = _eval_frame()

    assert build_clean_subset(df, tmp_path / "absent.csv") is None
    assert build_clean_subset(df, _training_csv(tmp_path, ["X"], column="sequence")) is None


def test_build_clean_subset_removes_overlapping_rows(tmp_path):
    df = _eval_frame()
    clean = build_clean_subset(df, _training_csv(tmp_path, EVAL_PEPTIDES[:3]))

    assert clean is not None
    assert len(clean) < len(df)
    assert list(clean["peptide"]) == ["AAAAAAAAB"]
