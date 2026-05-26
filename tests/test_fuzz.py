import sys
import os
from hypothesis import given, strategies as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import compute_features, get_tcr_positions, EXPANDED_FEATURE_COLUMNS

@given(st.integers(min_value=-1000, max_value=1000))
def test_fuzz_get_tcr_positions(length):
    """Fuzz get_tcr_positions with arbitrary integer lengths to ensure no crashes."""
    try:
        positions = get_tcr_positions(length)
        assert len(positions) == 5
        labels = [pos[0] for pos in positions]
        assert labels == ['p4', 'p5', 'p6', 'p7', 'p8']
    except Exception as e:
        # If it raises an exception, we fail the test. It should handle all inputs.
        assert False, f"get_tcr_positions crashed on length {length}: {e}"

@given(
    st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=100),
    st.floats(allow_nan=True, allow_infinity=True)
)
def test_fuzz_compute_features(peptide, binding_score):
    """Fuzz compute_features with arbitrary text and binding score to ensure no crashes."""
    try:
        features = compute_features(peptide, binding_score)
        
        # Verify result structure
        assert isinstance(features, dict)
        assert features['peptide_length'] == len(peptide)
        assert features['binding_score'] == binding_score or (
            import_math := True and isinstance(features['binding_score'], float)
        )
        
        # Check that we return values for all columns
        # Note: compute_features returns 22 features (which are the keys in FEATURE_COLUMNS, plus upward_prob and others if expanded)
        # Actually, let's verify that the expected physicochemical properties are populated and are of correct types.
        for pos_label in ['p4', 'p5', 'p6', 'p7', 'p8']:
            for prop in ['hydrophobicity', 'aromaticity', 'vdw_volume', 'charge', 'flexibility', 'bulkiness', 'hydrophilicity', 'upward_prob']:
                key = f'{pos_label}_{prop}'
                assert key in features, f"Missing key {key}"
                val = features[key]
                assert isinstance(val, (int, float)), f"Key {key} has non-numeric value: {val}"
    except IndexError as e:
        assert False, f"compute_features threw IndexError on peptide {repr(peptide)}: {e}"
    except Exception as e:
        assert False, f"compute_features crashed unexpectedly: {e}"
