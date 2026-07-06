import pandas as pd
from pathlib import Path
from src.verify.iedb_multi_virus_extractor import (
    is_valid_peptide,
    clean_and_pool_epitopes,
    load_proteome_peptides,
    generate_decoys,
    process_target,
)


def test_is_valid_peptide():
    assert is_valid_peptide("GLFYTRTGL") is True
    assert is_valid_peptide("GLFYTRTG") is True
    assert is_valid_peptide("GLFYTRTGLAA") is True
    assert is_valid_peptide("GLFYTRTGLAAA") is False  # length 12
    assert is_valid_peptide("GLFYTRT") is False  # length 7
    assert is_valid_peptide("GLF-TRTGL") is False  # invalid char
    assert is_valid_peptide("") is False
    assert is_valid_peptide(None) is False


def test_clean_and_pool_epitopes():
    records = [
        {
            "linear_sequence": "GLFYTRTGL",
            "mhc_allele_name": "HLA-A*02:01",
            "qualitative_measure": "Positive",
            "source_molecule": "Spike",
        },
        {
            "linear_sequence": "GLFYTRTGL",
            "mhc_allele_name": "HLA-A*02:01",
            "qualitative_measure": "Positive",
            "source_molecule": "Spike",
        },
        {
            "linear_sequence": "GLFYTRTGL",
            "mhc_allele_name": "HLA-A*02:01",
            "qualitative_measure": "Negative",
            "source_molecule": "Spike",
        },  # Conflict resolved to Positive (mean = 2/3 >= 0.5)
        {
            "linear_sequence": "AAYSDQWAL",
            "mhc_allele_name": "HLA-A*24:02",
            "qualitative_measure": "Negative",
            "source_molecule": "Membrane",
        },
        {
            "linear_sequence": "INVALIDPEPTIDE",
            "mhc_allele_name": "HLA-A*02:01",
            "qualitative_measure": "Positive",
        },
    ]
    df = clean_and_pool_epitopes(records)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "peptide" in df.columns
    assert "label" in df.columns
    assert "allele" in df.columns

    # Vote-resolved GLFYTRTGL should be Positive (1)
    row_glf = df[df["peptide"] == "GLFYTRTGL"].iloc[0]
    assert row_glf["label"] == 1

    # AAYSDQWAL should be Negative (0)
    row_aay = df[df["peptide"] == "AAYSDQWAL"].iloc[0]
    assert row_aay["label"] == 0


def test_load_proteome_peptides(tmp_path):
    fasta_file = tmp_path / "test_proteome.fasta"
    fasta_content = ">Seq1\nAAAAAAAALLL\n"
    fasta_file.write_text(fasta_content)

    peptides = load_proteome_peptides(fasta_file, min_len=8, max_len=9)
    # Exclude non-standard AA or check results
    assert len(peptides) > 0
    # Overlapping 8-mers from AAAAAAAALLL:
    # AAAAAAAA, AAAAAAAL, AAAAAALL, AAAAALLL
    # 9-mers:
    # AAAAAAAAL, AAAAAAALL, AAAAAALLL
    assert "AAAAAAAA" in peptides
    assert "AAAAALLL" in peptides
    assert "AAAAAALLL" in peptides


def test_generate_decoys():
    pos_peptides = ["GLFYTRTGL", "AAYSDQWAL"]
    proteome_peptides = ["GLFYTRTGL", "AAYSDQWAL", "LTDEMIAQY", "IPFAMQMAY", "NYNYLYRLF"]
    target_alleles = ["HLA-A*02:01", "HLA-A*24:02"]

    decoys = generate_decoys(pos_peptides, proteome_peptides, target_alleles, decoy_ratio=1.0)
    assert len(decoys) == 2
    for seq, allele, label in decoys:
        assert label == 0
        assert seq not in pos_peptides
        assert allele in target_alleles


def test_process_target_mock(tmp_path):
    config = {
        "taxonomy_id": 2697049,
        "mhc_alleles": ["HLA-A*02:01"],
        "proteome_fasta": str(tmp_path / "mock.fasta"),
        "validation_out": str(tmp_path / "sars2_verify.csv"),
    }
    # Create a small mock fasta
    Path(config["proteome_fasta"]).write_text(">Protein\nYLQPRTFLLNYNYLYRLFLTDEMIAQY\n")

    process_target("SARS-CoV-2", config, tmp_path, mock=True)

    out_path = Path(config["validation_out"])
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert len(df) > 0
    assert "peptide" in df.columns
    assert "label" in df.columns
    # Check 1:1 balance
    pos_count = (df["label"] == 1).sum()
    neg_count = (df["label"] == 0).sum()
    assert pos_count == neg_count
