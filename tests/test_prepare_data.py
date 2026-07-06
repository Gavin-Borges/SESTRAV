import pandas as pd
from src.prepare_data import map_allele_to_supertype_pooling, prepare_allele_features


def test_map_allele_to_supertype_pooling():
    # 1. Core target panel alleles should remain unchanged
    assert map_allele_to_supertype_pooling("HLA-A*02:01") == "HLA-A*02:01"
    assert map_allele_to_supertype_pooling("HLA-B*44:02") == "HLA-B*44:02"

    # 2. Out-of-panel human alleles should map to their supertypes
    assert map_allele_to_supertype_pooling("HLA-A*02:05") == "A02"
    assert map_allele_to_supertype_pooling("HLA-B*35:05") == "B07"
    assert map_allele_to_supertype_pooling("HLA-B*58:09") == "B58"

    # 3. Unrecognized or animal alleles should map to 'Other'
    assert map_allele_to_supertype_pooling("HLA-C*07:02") == "Other"
    assert map_allele_to_supertype_pooling("HLA-pig") == "Other"
    assert map_allele_to_supertype_pooling("HLA-SLA-2*11:04") == "Other"
    assert map_allele_to_supertype_pooling("INVALID-ALLELE") == "Other"
    assert map_allele_to_supertype_pooling(None) == "Other"


def test_prepare_allele_features():
    df = pd.DataFrame(
        {
            "peptide": ["AAAA", "BBBB", "CCCC", "DDDD"],
            "allele": ["HLA-A*02:01", "HLA-A*02:05", "HLA-SLA-2*11:04", "HLA-B*44:02"],
        }
    )

    df_encoded = prepare_allele_features(df, allele_col="allele", encode_mode="one_hot")

    # Check that the allele column has been transformed correctly
    assert df_encoded["allele"].tolist() == ["HLA-A*02:01", "A02", "Other", "HLA-B*44:02"]

    # Check one-hot columns are present
    assert "allele_pooled_HLA-A*02:01" in df_encoded.columns
    assert "allele_pooled_A02" in df_encoded.columns
    assert "allele_pooled_Other" in df_encoded.columns
    assert "allele_pooled_HLA-B*44:02" in df_encoded.columns


def test_prepare_allele_features_no_encoding():
    """Covers line 85->89: encode_mode != 'one_hot' skips one-hot expansion."""
    df = pd.DataFrame(
        {
            "peptide": ["AAAA", "BBBB"],
            "allele": ["HLA-A*02:01", "HLA-A*02:05"],
        }
    )
    df_out = prepare_allele_features(df, allele_col="allele", encode_mode="label")
    # Allele column is still pooled, but no allele_pooled_* columns added.
    assert "allele" in df_out.columns
    assert not any(c.startswith("allele_pooled_") for c in df_out.columns)
