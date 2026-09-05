"""The GNN feature-cache tag must fingerprint the WHOLE dataset file.

src/train_gnn.py names its physicochemical feature cache
`physico_features_mode{mode}_{tag}.csv`, where `tag` is a short SHA-256 of the
dataset. The tag used to come from a single 64 KiB read, so two corpora sharing
their first chunk shared one cache entry. FeatureStore.load_cached_features
validates neither row count nor columns, so that collision silently pairs the
previous corpus's feature rows with the new corpus's labels.

The dataset builder's final merge concatenates its parts without shuffling (the
`merged = pd.concat(parts, ...)` call in scripts/build_dataset_v5.py), so a
rebuild that only extends a later part is exactly the shared-prefix case.
"""

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("torch")

import src.train_gnn as train_gnn
from src.train_gnn import _dataset_cache_tag

CHUNK = 65536
HEADER = b"peptide,label\n"
# 96,000 bytes of identical rows: the shared prefix comfortably exceeds one chunk.
FILLER = b"AAAAAAAAA,1\n" * 8000


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two CSVs with byte-identical first chunks and different tails."""
    corpus_a = tmp_path / "corpus_a.csv"
    corpus_b = tmp_path / "corpus_b.csv"
    corpus_a.write_bytes(HEADER + FILLER + b"CCCCCCCCC,0\n")
    corpus_b.write_bytes(HEADER + FILLER + b"DDDDDDDDD,1\n")
    assert corpus_a.read_bytes()[:CHUNK] == corpus_b.read_bytes()[:CHUNK]
    assert corpus_a.read_bytes() != corpus_b.read_bytes()
    return corpus_a, corpus_b


def test_cache_tag_distinguishes_corpora_sharing_first_chunk(tmp_path):
    corpus_a, corpus_b = _write_pair(tmp_path)
    assert _dataset_cache_tag(str(corpus_a)) != _dataset_cache_tag(str(corpus_b))


def test_cache_tag_is_prefix_of_full_file_sha256(tmp_path):
    corpus_a, _ = _write_pair(tmp_path)
    expected = hashlib.sha256(corpus_a.read_bytes()).hexdigest()[:8]
    assert _dataset_cache_tag(str(corpus_a)) == expected


def test_both_cache_sites_use_the_streaming_helper():
    """Guards the wiring: a fixed helper is useless if a call site keeps the old form."""
    source = Path(train_gnn.__file__).read_text(encoding="utf-8")
    assert "read(65536)).hexdigest()" not in source
    # Both call sites now route through _feature_cache_name, which is what folds
    # the binding matrix into the key; a site that rebuilt the name inline would
    # reintroduce the matrix-blind key this module exists to prevent.
    assert source.count("_feature_cache_name(feature_mode, data_path, binding_matrix_path)") == 2
    assert source.count('f"physico_features_mode') == 1
