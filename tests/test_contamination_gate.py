from scripts.benchmark_runner import filter_contaminated_peptides


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
