"""
SESTRAV naming compatibility helpers.

Provides one-release alias support for renamed proteome IDs and model filenames.
"""

from __future__ import annotations

import os
from typing import Dict, List


PROTEOME_ID_ALIASES: Dict[str, str] = {
    "HPV_8_FASTAs": "HPV16_18_panel8",
    "EBV_8_FASTAs": "EBV_B95_8_panel8",
    "EBV_panel8_B958": "EBV_B95_8_panel8",
}


PROTEOME_ID_LEGACY_BY_CANONICAL: Dict[str, List[str]] = {
    "HPV16_18_panel8": ["HPV_8_FASTAs"],
    "EBV_B95_8_panel8": ["EBV_panel8_B958", "EBV_8_FASTAs"],
}


MODEL_NAME_ALIASES: Dict[str, List[str]] = {
    "rf_30feature_integrated.joblib": ["rf_30f_immunogenicity.joblib"],
    "xgb_30feature_integrated.joblib": ["xgb_30f_immunogenicity.joblib"],
    "ann_30feature_integrated.pt": ["ann_30f_immunogenicity.pt"],
    "rf_21feature_legacy.joblib": ["rf_immunogenicity.joblib"],
    "xgb_21feature_legacy.joblib": ["xgb_immunogenicity.joblib"],
    "ann_21feature_legacy.pt": ["ann_immunogenicity.pt"],
}


def canonicalize_proteome_id(proteome_id: str) -> str:
    """Return canonical proteome_id if a legacy alias is supplied."""
    return PROTEOME_ID_ALIASES.get(proteome_id, proteome_id)


def proteome_id_candidates(proteome_id: str) -> List[str]:
    """Return canonical-first candidate list for matching legacy output stems."""
    canonical = canonicalize_proteome_id(proteome_id)
    candidates = [canonical]
    for legacy in PROTEOME_ID_LEGACY_BY_CANONICAL.get(canonical, []):
        if legacy not in candidates:
            candidates.append(legacy)
    return candidates


def resolve_model_path(path: str) -> str:
    """
    Resolve a model path against canonical/legacy aliases.

    Returns the first existing candidate path, or the original path if none exist.
    """
    if not path:
        return path
    if os.path.isfile(path):
        return path

    directory, filename = os.path.split(path)
    aliases = MODEL_NAME_ALIASES.get(filename, [])
    for alias in aliases:
        candidate = os.path.join(directory, alias) if directory else alias
        if os.path.isfile(candidate):
            return candidate
    return path


def canonical_output_filename(base_name: str, dataset_mode: str, dataset_version: str) -> str:
    """
    Build a mode/version tagged filename for interpretation-critical summaries.

    Example:
      h2_tier_a_summary__modeA_baseline__IEDB-20260424-EBV_HPV16_BASELINE-v1.csv
    """
    return f"{base_name}__{dataset_mode}__{dataset_version}.csv"
