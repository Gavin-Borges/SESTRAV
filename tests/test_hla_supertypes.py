# tests/test_hla_supertypes.py
import pytest
from src.hla_supertypes import get_hla_supertype

def test_exact_matches():
    assert get_hla_supertype("HLA-A*01:01") == "A01"
    assert get_hla_supertype("HLA-A*02:01") == "A02"
    assert get_hla_supertype("HLA-A*03:01") == "A03"
    assert get_hla_supertype("HLA-B*07:02") == "B07"
    assert get_hla_supertype("HLA-B*44:02") == "B44"

def test_family_regex_matches():
    # Test matches covered by regex family maps
    assert get_hla_supertype("HLA-A*02:05") == "A02"
    assert get_hla_supertype("HLA-B*35:05") == "B07"
    assert get_hla_supertype("HLA-B*57:02") == "B58"
    assert get_hla_supertype("HLA-B*58:09") == "B58"

def test_normalization():
    # Test lowercase and spacing handling
    assert get_hla_supertype(" hla-a*02:01 ") == "A02"
    assert get_hla_supertype("HLA-A*02:01") == "A02"
    assert get_hla_supertype("hla-b*44:02") == "B44"

def test_invalid_allele_raises_value_error():
    with pytest.raises(ValueError):
         get_hla_supertype("HLA-C*07:02")
    with pytest.raises(ValueError):
         get_hla_supertype("INVALID-ALLELE")
