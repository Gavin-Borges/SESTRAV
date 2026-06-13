import pytest
import pandas as pd
from pathlib import Path
from src.core.feature_store import FeatureStore

def test_feature_store_save_load(tmp_path):
    store = FeatureStore(tmp_path)
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    
    # Save
    out_path = store.save_dataset(df, "test.csv")
    assert out_path.exists()
    
    # Load
    loaded_df = store.load_dataset("test.csv")
    assert len(loaded_df) == 2
    assert "A" in loaded_df.columns
    
def test_feature_store_missing_file(tmp_path):
    store = FeatureStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load_dataset("nonexistent.csv")
