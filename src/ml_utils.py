"""ml_utils.py - ML utility functions for the SESTRAV v5 training pipeline.

Provides MultiStratifiedKFold: a cross-validation splitter that stratifies
simultaneously on binary label, negative_origin (real tested-negative vs decoy),
HLA supertype, and peptide length group.

Standard StratifiedKFold on label alone produces folds with unbalanced
negative_origin distributions, inflating variance in the real-tested-negative
AUC-ROC metric (Amendment 6, Part 16). Use this splitter instead.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.hla_supertypes import get_hla_supertype

LOGGER = logging.getLogger(__name__)

_ORIGIN_REAL: str = "tested_negative"
_DECOY_ORIGINS: frozenset[str] = frozenset(
    {"self_proteome_decoy", "allele_matched_nonbinder", "published_negative"}
)
_LENGTH_9: str = "9mer"
_LENGTH_OTHER: str = "other"
_SUPERTYPE_UNKNOWN: str = "unk"


def _bin_origin(origin: str | None) -> str:
    if origin == _ORIGIN_REAL:
        return "real"
    if origin in _DECOY_ORIGINS:
        return "decoy"
    return "unk"


def _bin_length(peptide: str | None) -> str:
    if peptide and len(str(peptide)) == 9:
        return _LENGTH_9
    return _LENGTH_OTHER


def _bin_supertype(allele: str | None) -> str:
    if not allele:
        return _SUPERTYPE_UNKNOWN
    try:
        st = get_hla_supertype(str(allele))
        return st if st else _SUPERTYPE_UNKNOWN
    except Exception:  # noqa: BLE001 - defensive; hla_supertypes may raise on bad input
        return _SUPERTYPE_UNKNOWN


def make_stratification_key(
    labels: pd.Series,
    negative_origin: pd.Series | None = None,
    hla_alleles: pd.Series | None = None,
    peptides: pd.Series | None = None,
) -> pd.Series:
    """Build a composite stratification key for use with StratifiedKFold.

    Key format: ``label|origin_bin|supertype|length_bin``

    Each component reduces a high-cardinality column to a low-cardinality
    bin so that StratifiedKFold can keep each fold's distribution balanced.
    """
    n = len(labels)
    idx = labels.index

    label_str = labels.astype(int).astype(str)

    origin_str = (
        negative_origin.fillna("").map(_bin_origin)
        if negative_origin is not None
        else pd.Series(["unk"] * n, index=idx)
    )

    if hla_alleles is not None:
        supertype_str = hla_alleles.fillna("").apply(_bin_supertype)
    else:
        supertype_str = pd.Series([_SUPERTYPE_UNKNOWN] * n, index=idx)

    if peptides is not None:
        length_str = peptides.fillna("").apply(_bin_length)
    else:
        length_str = pd.Series([_LENGTH_OTHER] * n, index=idx)

    return label_str + "|" + origin_str + "|" + supertype_str + "|" + length_str


class MultiStratifiedKFold:
    """Cross-validation splitter stratified on label + negative_origin + HLA supertype + length.

    Falls back to label-only stratification when any composite stratum has
    fewer than min_stratum_size samples (prevents StratifiedKFold from
    failing on rare combination strata with n < n_splits).

    Parameters
    ----------
    n_splits:
        Number of folds.
    shuffle:
        Whether to shuffle before splitting.
    random_state:
        Seed for reproducibility.
    min_stratum_size:
        Minimum sample count per composite stratum before falling back
        to label-only stratification.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        min_stratum_size: int = 5,
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.min_stratum_size = min_stratum_size

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        negative_origin: pd.Series | None = None,
        hla_alleles: pd.Series | None = None,
        peptides: pd.Series | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (train_indices, test_indices) for each fold.

        ``X`` is used only for its length; feature values are ignored here.
        """
        y_series = y if isinstance(y, pd.Series) else pd.Series(y)

        composite = make_stratification_key(
            y_series,
            negative_origin=negative_origin,
            hla_alleles=hla_alleles,
            peptides=peptides,
        )

        min_count = int(composite.value_counts().min())
        if min_count < self.min_stratum_size:
            LOGGER.warning(
                "Composite stratum minimum count %d < min_stratum_size=%d; "
                "falling back to label-only stratification for this split.",
                min_count,
                self.min_stratum_size,
            )
            strat_key = y_series.astype(str)
        else:
            strat_key = composite

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        dummy_X = np.zeros((len(y_series), 1))
        yield from skf.split(dummy_X, strat_key.to_numpy())

    def get_n_splits(
        self,
        X: object = None,
        y: object = None,
        groups: object = None,
    ) -> int:
        return self.n_splits
