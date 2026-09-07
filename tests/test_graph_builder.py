import torch

from src.gnn.graph_builder import (
    GraphBuilder,
    MAX_PEPTIDE_LEN,
    allele_cache_key,
    report_structural_cache_coverage,
    structural_cache_filename,
)


def test_max_peptide_len_constant():
    assert MAX_PEPTIDE_LEN == 11


def test_sequence_to_node_features():
    # Test valid sequence
    seq = "ACDEF"
    features = GraphBuilder.sequence_to_node_features(seq, max_len=11)

    assert features.shape == (11, 20)  # max_len x vocab_size
    # A is index 0, C is index 1, D is index 2, E is index 3, F is index 4
    assert features[0, 0] == 1.0
    assert features[1, 1] == 1.0
    assert features[2, 2] == 1.0
    assert features[3, 3] == 1.0
    assert features[4, 4] == 1.0
    assert features[5].sum() == 0.0  # Padding


def test_build_chain_adj():
    adj = GraphBuilder.build_chain_adj(max_len=5)

    assert adj.shape == (5, 5)
    # Check symmetric normalized adj logic
    # Nodes 0 and 4 have degree 2 (self + 1 neighbor)
    # Nodes 1, 2, 3 have degree 3 (self + 2 neighbors)
    assert adj[0, 0] > 0
    assert adj[0, 1] > 0
    assert adj[0, 2] == 0.0


def test_allele_cache_key_strips_hla_punctuation():
    assert allele_cache_key("HLA-A*02:01") == "A0201"
    assert allele_cache_key("HLA-B*44:02") == "B4402"


def test_structural_cache_filename_pins_the_writer_convention():
    # scripts/run_pandora_structures.py writes exactly this name. Pinned as a
    # literal so that a change on either side has to change this line too.
    assert structural_cache_filename("CLGGLLTMV", "HLA-A*02:01") == "CLGGLLTMV_A0201_dist.pt"


def test_build_spatial_adj_fallback_to_chain(tmp_path):
    # No cached distance matrix -> falls back to the chain adjacency.
    adj = GraphBuilder.build_spatial_adj("ACDEF", "HLA-A*02:01", str(tmp_path), max_len=5)
    expected = GraphBuilder.build_chain_adj(max_len=5)
    assert torch.allclose(adj, expected)


def test_build_spatial_adj_uses_cached_distances(tmp_path):
    # 3-residue distance matrix: 0-1 close, 0-2 far (beyond threshold). The
    # fixture is written under the WRITER's filename; before the reader was
    # taught the allele key it looked for 'PEP_dist.pt' and silently fell back.
    dist = torch.tensor([[0.0, 2.0, 50.0], [2.0, 0.0, 3.0], [50.0, 3.0, 0.0]], dtype=torch.float32)
    name = structural_cache_filename("PEP", "HLA-A*02:01")
    torch.save(dist, tmp_path / name)  # nosec B614 - test fixture, trusted tensor
    adj = GraphBuilder.build_spatial_adj(
        "PEP", "HLA-A*02:01", str(tmp_path), max_len=3, distance_threshold=8.0
    )
    assert adj.shape == (3, 3)
    # Self-loops present; far pair (0,2) has no direct edge -> 0 after norm.
    assert adj[0, 0] > 0
    assert adj[0, 2] == 0.0
    assert not torch.allclose(adj, GraphBuilder.build_chain_adj(max_len=3))


def test_build_spatial_adj_ignores_a_legacy_peptide_only_cache_file(tmp_path):
    # REPLACES a vacuous test. The previous version wrote the fixture under the
    # allele-keyed name and asserted the chain fallback for a DIFFERENT allele.
    # That passes either way: the old peptide-only reader looked for
    # 'PEP_dist.pt', which the fixture was not, so it also missed and also fell
    # back. Measured - the old test still exited 0 with the defect reinstated.
    #
    # This fixture is written under the LEGACY name instead, so it is a file the
    # DEFECT would hit and the fix must not. The chain fallback is now the only
    # correct answer, and a reader that accepted the peptide-only name would
    # return this matrix and fail the assertion.
    dist = torch.tensor([[0.0, 2.0, 50.0], [2.0, 0.0, 3.0], [50.0, 3.0, 0.0]], dtype=torch.float32)
    torch.save(  # nosec B614 - test fixture, trusted tensor
        dist, tmp_path / "PEP_dist.pt"
    )
    adj = GraphBuilder.build_spatial_adj("PEP", "HLA-A*02:01", str(tmp_path), max_len=3)
    assert torch.allclose(adj, GraphBuilder.build_chain_adj(max_len=3))


def test_build_spatial_adj_is_allele_specific(tmp_path):
    # Both alleles cached, with DIFFERENT geometry, so the chain fallback is not
    # reachable on either call and neither branch can pass by falling back. This
    # is what kills the mutation the legacy-name test above cannot see: a reader
    # that builds the right filename but pins the allele to a constant returns
    # one matrix for both calls.
    a_dist = torch.tensor([[0.0, 2.0, 50.0], [2.0, 0.0, 3.0], [50.0, 3.0, 0.0]], dtype=torch.float32)
    b_dist = torch.tensor([[0.0, 50.0, 3.0], [50.0, 0.0, 2.0], [3.0, 2.0, 0.0]], dtype=torch.float32)
    for allele, dist in (("HLA-A*02:01", a_dist), ("HLA-B*07:02", b_dist)):
        torch.save(  # nosec B614 - test fixture, trusted tensor
            dist, tmp_path / structural_cache_filename("PEP", allele)
        )
    a_adj = GraphBuilder.build_spatial_adj("PEP", "HLA-A*02:01", str(tmp_path), max_len=3)
    b_adj = GraphBuilder.build_spatial_adj("PEP", "HLA-B*07:02", str(tmp_path), max_len=3)
    chain = GraphBuilder.build_chain_adj(max_len=3)
    assert not torch.allclose(a_adj, chain)
    assert not torch.allclose(b_adj, chain)
    assert not torch.allclose(a_adj, b_adj)
    # A*02:01 puts the far pair at (0,2); B*07:02 puts it at (0,1).
    assert a_adj[0, 2] == 0.0 and a_adj[0, 1] > 0
    assert b_adj[0, 1] == 0.0 and b_adj[0, 2] > 0


def test_sequence_to_node_features_unknown_aa_leaves_row_zero():
    # 'X' is not in AA_VOCAB - the row should stay all-zero (branch 80->79).
    seq = "GILXFVFTL"
    features = GraphBuilder.sequence_to_node_features(seq, max_len=9)
    assert features.shape == (9, 20)
    assert features[3].sum() == 0.0  # 'X' position stays zero
    assert features[0].sum() == 1.0  # 'G' is valid, encoded normally


def test_default_max_len_uses_constant():
    features = GraphBuilder.sequence_to_node_features("GILGFVFTL")
    assert features.shape == (MAX_PEPTIDE_LEN, 20)


def test_report_structural_cache_coverage_counts_and_warns_on_a_total_miss(tmp_path, capsys):
    missing = report_structural_cache_coverage(
        ["PEP", "TIDE"], ["HLA-A*02:01", "HLA-B*07:02"], str(tmp_path)
    )
    out = capsys.readouterr().out
    assert missing == 2
    assert out.startswith("WARNING: structural cache coverage")
    assert "0/2" in out


def test_report_structural_cache_coverage_counts_partial_hits(tmp_path, capsys):
    dist = torch.zeros((3, 3), dtype=torch.float32)
    torch.save(  # nosec B614 - test fixture, trusted tensor
        dist, tmp_path / structural_cache_filename("PEP", "HLA-A*02:01")
    )
    missing = report_structural_cache_coverage(
        ["PEP", "TIDE"], ["HLA-A*02:01", "HLA-B*07:02"], str(tmp_path)
    )
    out = capsys.readouterr().out
    assert missing == 1
    assert "1/2" in out
    assert out.startswith("WARNING: ")


def test_report_structural_cache_coverage_is_quiet_prefix_on_full_hit(tmp_path, capsys):
    dist = torch.zeros((3, 3), dtype=torch.float32)
    torch.save(  # nosec B614 - test fixture, trusted tensor
        dist, tmp_path / structural_cache_filename("PEP", "HLA-A*02:01")
    )
    missing = report_structural_cache_coverage(["PEP"], ["HLA-A*02:01"], str(tmp_path))
    out = capsys.readouterr().out
    assert missing == 0
    assert out.startswith("Structural cache coverage: 1/1")


def test_report_structural_cache_coverage_output_is_bounded_by_panel_size(tmp_path, capsys):
    # The whole point of the panel-level report: output must not scale with the
    # corpus. 2000 distinct misses must still be exactly one line, and the count
    # must survive intact.
    peptides = [f"PEP{i}" for i in range(2000)]
    alleles = ["HLA-A*02:01"] * 2000
    missing = report_structural_cache_coverage(peptides, alleles, str(tmp_path))
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert missing == 2000
    assert len(lines) == 1
    assert "0/2000" in lines[0]


def test_report_structural_cache_coverage_absent_dir_is_a_total_miss(tmp_path, capsys):
    missing = report_structural_cache_coverage(
        ["PEP"], ["HLA-A*02:01"], str(tmp_path / "does_not_exist")
    )
    assert missing == 1
    assert "0/1" in capsys.readouterr().out


def test_report_structural_cache_coverage_empty_panel_is_silent(tmp_path, capsys):
    assert report_structural_cache_coverage([], [], str(tmp_path)) == 0
    assert capsys.readouterr().out == ""
