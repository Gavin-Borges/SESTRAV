"""The mode-31 feature cache must be honest about what it contains.

Three defects, one invariant: a cached feature matrix must contain exactly the
features its filename claims, for exactly the pool it is loaded against.

1. The key fingerprinted the corpus only. Mode-31 features are built from the
   binding matrix too (prepare_features_31), so a v4-derived and a v5-derived
   frame shared one filename. peptide_binding_matrix_v5 is a strict superset of
   v4 and prepare_features_30 substitutes np.zeros(10) for any peptide the
   matrix omits without raising, so the two frames differ materially.
2. feature_mode=31 with no binding matrix warned and silently computed 21
   physico columns, then saved them under the mode31 key, while
   gnn_config.json still recorded feature_mode=31.
3. On a cache hit nothing checked the row count. The frame is written with
   index=False and keeps no peptide column, so X_feats.iloc[train_idx] has
   nothing to re-align against: a cache at least as long as the pool misaligns
   features against labels in silence.
"""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")

from src.train_gnn import _feature_cache_name, _validate_cached_features


def _matrix(tmp_path: Path, name: str, n_rows: int) -> Path:
    """A binding matrix whose bytes depend on n_rows."""
    path = tmp_path / name
    cols: dict[str, list] = {"peptide": [f"PEP{i:06d}" for i in range(n_rows)]}
    for allele in ("A0101", "A0201", "A0301"):
        cols[f"bind_{allele}"] = [0.5] * n_rows
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.csv"
    path.write_bytes(b"peptide,label\nAAAAAAAAA,1\nCCCCCCCCC,0\n")
    return path


def test_mode31_key_distinguishes_two_binding_matrices(tmp_path):
    """The defect itself: two matrices must not share one cache filename."""
    corpus = _corpus(tmp_path)
    v4 = _matrix(tmp_path, "bm_v4.csv", 3)
    v5 = _matrix(tmp_path, "bm_v5.csv", 7)

    name_v4 = _feature_cache_name(31, str(corpus), str(v4))
    name_v5 = _feature_cache_name(31, str(corpus), str(v5))

    assert name_v4 != name_v5, (
        "mode-31 features are a function of the binding matrix; keying on the "
        "corpus alone lets a v4-derived frame be served to a v5 run"
    )


def test_mode50_key_also_distinguishes_matrices(tmp_path):
    """Mode 50 builds per-allele columns from the matrix as well."""
    corpus = _corpus(tmp_path)
    assert _feature_cache_name(50, str(corpus), str(_matrix(tmp_path, "a.csv", 3))) != (
        _feature_cache_name(50, str(corpus), str(_matrix(tmp_path, "b.csv", 9)))
    )


def test_mode21_key_ignores_the_binding_matrix(tmp_path):
    """Mode 21 is physico-only, so its key must NOT churn on an unused matrix.

    Otherwise the fix would needlessly invalidate every existing mode-21 cache.
    """
    corpus = _corpus(tmp_path)
    bare = _feature_cache_name(21, str(corpus), None)
    with_matrix = _feature_cache_name(21, str(corpus), str(_matrix(tmp_path, "m.csv", 5)))
    assert bare == with_matrix


def test_key_still_tracks_the_corpus(tmp_path):
    """The original dataset fingerprint must survive the change."""
    matrix = _matrix(tmp_path, "m.csv", 4)
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_bytes(b"peptide,label\nAAAAAAAAA,1\n")
    b.write_bytes(b"peptide,label\nCCCCCCCCC,0\n")
    assert _feature_cache_name(31, str(a), str(matrix)) != _feature_cache_name(
        31, str(b), str(matrix)
    )


def test_validate_rejects_a_cache_longer_than_the_pool():
    """The silent direction: .iloc succeeds and pairs the wrong rows."""
    X = pd.DataFrame({"f1": range(10)})
    with pytest.raises(ValueError, match="rows against a training pool"):
        _validate_cached_features(X, "physico_features_mode31_deadbeef.csv", 8)


def test_validate_rejects_a_cache_shorter_than_the_pool():
    X = pd.DataFrame({"f1": range(5)})
    with pytest.raises(ValueError, match="rows against a training pool"):
        _validate_cached_features(X, "physico_features_mode31_deadbeef.csv", 8)


def test_validate_accepts_a_matching_cache():
    X = pd.DataFrame({"f1": range(8)})
    assert _validate_cached_features(X, "physico_features_mode31_deadbeef.csv", 8) is X


def test_mode31_without_a_binding_matrix_raises_rather_than_falling_back():
    """The fallback wrote 21 columns under the mode31 key. It must be gone."""
    import inspect

    import src.train_gnn as train_gnn

    source = inspect.getsource(train_gnn.train_gnn_v2)
    assert "falling back to mode 21" not in source, (
        "the mode-21 fallback mislabels its own artifact and must not be restored"
    )
    assert "binding_matrix_path required for feature mode 31" in source


def test_both_cache_hit_paths_validate_the_row_count():
    """A guard that exists but is never called is the failure mode this catches."""
    import src.train_gnn as train_gnn

    source = Path(train_gnn.__file__).read_text(encoding="utf-8")
    assert source.count("_validate_cached_features(X_feats, cache_name, len(train_pool))") == 2


def test_mode50_guard_is_not_inside_the_cache_miss_branch():
    """A cache hit used to skip the mode-50 matrix guard entirely."""
    import inspect

    import src.train_gnn as train_gnn

    source = inspect.getsource(train_gnn.train_gnn)
    guard = source.index("binding_matrix_path required for feature mode 50")
    lookup = source.index("store.load_cached_features")
    assert guard < lookup, "the mode-50 guard must fire before the cache is consulted"
