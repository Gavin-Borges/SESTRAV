import hashlib

import pytest
import pandas as pd
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


def test_save_features_creates_nested_dirs(tmp_path):
    store = FeatureStore(tmp_path)
    df = pd.DataFrame({"f1": [0.1, 0.2]})
    out_path = store.save_features(df, "sub/dir/features.csv")
    assert out_path.exists()
    assert pd.read_csv(out_path).shape == (2, 1)


def test_verify_integrity_match_and_mismatch(tmp_path):
    store = FeatureStore(tmp_path)
    path = store.save_dataset(pd.DataFrame({"A": [1]}), "d.csv")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    assert store.verify_integrity(path, checksum) is True
    assert store.verify_integrity(path, "deadbeef") is False


def test_verify_integrity_missing_file_returns_false(tmp_path):
    store = FeatureStore(tmp_path)
    assert store.verify_integrity(tmp_path / "ghost.csv", "anything") is False


def test_cached_features_roundtrip_and_miss(tmp_path):
    store = FeatureStore(tmp_path)
    assert store.load_cached_features("c.csv") is None  # cache miss
    df = pd.DataFrame({"x": [1, 2, 3]})
    cache_path = store.save_cached_features(df, "c.csv")
    assert cache_path.exists()
    loaded = store.load_cached_features("c.csv")
    assert loaded is not None and len(loaded) == 3
