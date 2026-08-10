"""ml_utils.py - ML utility functions for the SESTRAV v5 training pipeline.

Provides two composite-stratified cross-validation splitters that stratify
simultaneously on binary label, negative_origin (real tested-negative vs
decoy), HLA supertype, and peptide length group:

- MultiStratifiedKFold: standard StratifiedKFold on the composite key.
  NOT peptide-grouped - see docs/claims_register.md D15.
- PeptideGroupedKFold: the same composite key, but peptide-grouped, so no
  peptide appears in both the train and validation side of a fold. Use this
  one for any certified generalization estimate.

Standard StratifiedKFold on label alone produces folds with unbalanced
negative_origin distributions, inflating variance in the real-tested-negative
AUC-ROC metric (Amendment 6, Part 16). Use one of the splitters above instead.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.hla_supertypes import get_hla_supertype

LOGGER = logging.getLogger(__name__)

# Both the IEDB bulk export ("tested_negative") and the API bridge ("iedb_api")
# are genuine assay-confirmed negatives - the same pairing scripts/build_dataset_v5.py
# and scripts/analyze_hiv1_binding_bias.py use. Omitting "iedb_api" here fragmented
# the composite key badly enough that its rarest stratum on the v5 corpus was 2 rows,
# which is what made the min_stratum_size fallback fire on every v5 split.
_ORIGIN_REAL: frozenset[str] = frozenset({"tested_negative", "iedb_api"})
_DECOY_ORIGINS: frozenset[str] = frozenset(
    {"self_proteome_decoy", "allele_matched_nonbinder", "published_negative"}
)
_LENGTH_9: str = "9mer"
_LENGTH_OTHER: str = "other"
_SUPERTYPE_UNKNOWN: str = "unk"

STRATUM_COMPONENTS: tuple[str, str, str, str] = ("label", "origin", "supertype", "length")

# Coarsening ladder, finest first. Length is dropped before supertype: length
# is the weakest balance requirement of the four, while origin and supertype
# guard the real-negative/decoy split and allele skew this splitter exists for
# (Amendment 6, Part 16). The rung actually selected is recorded on the
# splitter instance as .stratification_components_ so degradation is never
# silent to a caller that checks it.
_COARSENING_LADDER: tuple[tuple[str, ...], ...] = (
    ("label", "origin", "supertype", "length"),
    ("label", "origin", "supertype"),
    ("label", "origin", "length"),
    ("label", "origin"),
    ("label",),
)


def _bin_origin(origin: str | None) -> str:
    if origin in _ORIGIN_REAL:
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


def _stratification_parts(
    labels: pd.Series,
    negative_origin: pd.Series | None,
    hla_alleles: pd.Series | None,
    peptides: pd.Series | None,
) -> dict[str, pd.Series]:
    """Build each stratification component, positionally aligned to ``labels``.

    Returns one bin per entry of STRATUM_COMPONENTS, so any subset can be
    joined into a stratification key of the caller's choosing (see
    make_stratification_key and MultiStratifiedKFold's coarsening ladder).
    """
    n = len(labels)
    idx = labels.index

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

    return {
        "label": labels.astype(int).astype(str),
        "origin": origin_str,
        "supertype": supertype_str,
        "length": length_str,
    }


def _join_components(parts: dict[str, pd.Series], components: tuple[str, ...]) -> pd.Series:
    key = parts[components[0]]
    for name in components[1:]:
        key = key + "|" + parts[name]
    return key


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
    parts = _stratification_parts(labels, negative_origin, hla_alleles, peptides)
    return _join_components(parts, STRATUM_COMPONENTS)


def _coarsened_key(
    y_series: pd.Series,
    negative_origin: pd.Series | None,
    hla_alleles: pd.Series | None,
    peptides: pd.Series | None,
    min_stratum_size: int,
) -> tuple[pd.Series, tuple[str, ...]]:
    """Return the finest composite key whose rarest stratum clears min_stratum_size.

    Walks _COARSENING_LADDER from finest to coarsest and returns the first rung
    that survives, with the component tuple that produced it. Raises if even
    label-only stratification is too sparse: at that point the corpus cannot
    support the requested fold count, and silently proceeding would produce
    folds missing a class entirely.
    """
    parts = _stratification_parts(y_series, negative_origin, hla_alleles, peptides)
    for components in _COARSENING_LADDER:
        candidate = _join_components(parts, components)
        min_count = int(candidate.value_counts().min())
        if min_count >= min_stratum_size:
            return candidate, components
    label_min = int(parts["label"].value_counts().min())
    raise ValueError(
        f"Cannot stratify: even label-only strata have a minimum count of "
        f"{label_min}, below min_stratum_size={min_stratum_size}. "
        "Reduce min_stratum_size or n_splits, or check the input labels."
    )


def _warn_if_coarsened(components_used: tuple[str, ...]) -> None:
    if components_used != _COARSENING_LADDER[0]:
        LOGGER.warning(
            "Composite stratum too sparse at full resolution; coarsened "
            "stratification to %s for this split.",
            "|".join(components_used),
        )


class MultiStratifiedKFold:
    """Cross-validation splitter stratified on label + negative_origin + HLA supertype + length.

    NOT peptide-grouped: rows sharing a peptide can land on opposite sides of
    a fold boundary, and on the v5 corpus 71.0% of held-out rows have their
    exact peptide present in that fold's training set
    (docs/claims_register.md D15). Use PeptideGroupedKFold below for any
    certified generalization estimate; this class remains for reproducing
    pre-Phase-0 numbers and for corpora with no peptide grouping concern.

    Walks a coarsening ladder (see _COARSENING_LADDER) from the full
    composite key down to label-only, and uses the finest rung whose rarest
    stratum still has at least min_stratum_size samples (StratifiedKFold
    fails outright on strata smaller than n_splits). The rung actually used
    is recorded as .stratification_components_ after split() runs, so
    degradation is always visible to the caller rather than silent. Raises
    ValueError if even label-only stratification is too sparse.

    Parameters
    ----------
    n_splits:
        Number of folds.
    shuffle:
        Whether to shuffle before splitting.
    random_state:
        Seed for reproducibility. Forwarded to sklearn only when shuffle is
        True; sklearn raises if random_state is set with shuffle=False.
    min_stratum_size:
        Minimum sample count per stratum a coarsening rung must meet before
        it is accepted.
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
        self.stratification_components_: tuple[str, ...] | None = None

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
        strat_key, components_used = _coarsened_key(
            y_series, negative_origin, hla_alleles, peptides, self.min_stratum_size
        )
        self.stratification_components_ = components_used
        _warn_if_coarsened(components_used)

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            # sklearn raises if random_state is set while shuffle=False.
            random_state=self.random_state if self.shuffle else None,
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


class PeptideGroupedKFold:
    """Peptide-grouped, composite-stratified cross-validation splitter.

    Same stratification composite as MultiStratifiedKFold, but the peptide
    string is the fold GROUP: every row carrying a given peptide lands in
    exactly one fold. This is the splitter that makes a SESTRAV CV number a
    generalization estimate rather than a memorization estimate, because
    every feature_mode=31 feature is a pure function of the peptide string
    while the v5 corpus is deduplicated on (peptide, hla_allele), not on
    peptide alone (docs/claims_register.md D15).

    A separate sibling class rather than a groups= kwarg on
    MultiStratifiedKFold, because the two underlying sklearn splitters have
    incompatible failure semantics on sparse strata: StratifiedKFold raises,
    StratifiedGroupKFold only warns and proceeds. One min_stratum_size knob
    cannot mean the same thing for both, so the coarsening ladder is applied
    here as a genuine safety net (not merely to stay consistent with the
    ungrouped splitter), even though StratifiedGroupKFold alone would not
    require it for correctness.

    The split() keyword set is identical to MultiStratifiedKFold.split's, so
    a caller can swap which class it constructs without changing the call
    site - only peptides= becomes mandatory, since it doubles as the group.
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
        self.stratification_components_: tuple[str, ...] | None = None

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        negative_origin: pd.Series | None = None,
        hla_alleles: pd.Series | None = None,
        peptides: pd.Series | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield peptide-disjoint (train_indices, test_indices) for each fold.

        ``X`` is used only for its length; feature values are ignored here.
        ``peptides`` is mandatory: it is both the length-bin input and the
        fold group.
        """
        if peptides is None:
            raise ValueError(
                "PeptideGroupedKFold requires peptides=. Without it there is no "
                "group to hold together and the split silently degenerates to "
                "MultiStratifiedKFold, which is the peptide leak this class "
                "exists to close (docs/claims_register.md D15)."
            )
        y_series = y if isinstance(y, pd.Series) else pd.Series(y)
        peptide_series = peptides if isinstance(peptides, pd.Series) else pd.Series(peptides)
        if len(peptide_series) != len(y_series):
            raise ValueError(
                f"peptides has {len(peptide_series)} rows but y has {len(y_series)}; "
                "a grouped split requires one peptide per sample."
            )

        strat_key, components_used = _coarsened_key(
            y_series, negative_origin, hla_alleles, peptide_series, self.min_stratum_size
        )
        self.stratification_components_ = components_used
        _warn_if_coarsened(components_used)

        sgkf = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            # sklearn raises if random_state is set with shuffle=False.
            random_state=self.random_state if self.shuffle else None,
        )
        dummy_X = np.zeros((len(y_series), 1))
        groups = peptide_series.fillna("").astype(str).to_numpy()
        yield from sgkf.split(dummy_X, strat_key.to_numpy(), groups=groups)

    def get_n_splits(
        self,
        X: object = None,
        y: object = None,
        groups: object = None,
    ) -> int:
        return self.n_splits
