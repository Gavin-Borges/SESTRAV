"""
SESTRAV Data Preparation Module
================================

Handles HLA allele pooling via supertype mapping to combat allele drift and 
out-of-distribution generalization issues (such as shifts to animal MHC or non-panel HLA).
"""

import pandas as pd
import numpy as np
import logging
from src.hla_supertypes import get_hla_supertype, HLA_SUPERTYPE_MAP

logger = logging.getLogger(__name__)

# The core 10-allele panel representing the target domain of SESTRAV
CORE_PANEL = {
    "HLA-A*01:01",
    "HLA-A*02:01",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*24:02",
    "HLA-B*07:02",
    "HLA-B*08:01",
    "HLA-B*27:05",
    "HLA-B*35:01",
    "HLA-B*44:02"
}

def map_allele_to_supertype_pooling(allele: str) -> str:
    """
    Map an HLA allele to itself if it resides in the core training panel,
    or pool it into its corresponding HLA supertype family if it is an out-of-panel human allele.
    Maps completely unrecognized alleles (or animal MHC) to 'Unknown'/'Other'.
    
    Biological Rationale:
    HLA alleles are highly polymorphic, but they share structural pockets and binding motifs
    that can be classified into 'supertypes' (e.g., A02, B07, B44). When testing on datasets with
    unseen human alleles, mapping them to their supertype allows the model to leverage shared 
    biophysical properties. Conversely, animal MHC alleles (e.g., swine SLA or chicken BF) do not
    fit human supertype definitions, and mapping them to 'Unknown'/'Other' prevents corrupting 
    human-calibrated binding predictions.
    
    Args:
        allele (str): The HLA allele to resolve.
        
    Returns:
        str: Core allele name, supertype family, or 'Other'.
    """
    if pd.isna(allele) or not isinstance(allele, str):
        return "Other"
        
    clean_allele = allele.strip()
    
    # 1. If it's part of the core training panel, retain the specific allele.
    if clean_allele in CORE_PANEL:
        return clean_allele
        
    # 2. Otherwise, attempt to resolve its supertype family
    try:
        supertype = get_hla_supertype(clean_allele)
        return supertype
    except ValueError:
        # Fallback for completely unrecognized alleles (including swine SLA, chicken BF, etc.)
        return "Other"

def prepare_allele_features(df: pd.DataFrame, allele_col: str = "allele", encode_mode: str = "one_hot") -> pd.DataFrame:
    """
    Apply robust supertype pooling transformation to the allele column, 
    then generate one-hot encodings or embeddings.
    
    Args:
        df (pd.DataFrame): Dataset containing the allele column.
        allele_col (str): Column name for alleles.
        encode_mode (str): Mode of encoding, e.g., 'one_hot'.
        
    Returns:
        pd.DataFrame: Dataframe with transformed and encoded allele features.
    """
    df_processed = df.copy()
    
    # Transform allele column to target alleles, supertypes, or 'Other'
    df_processed[allele_col] = df_processed[allele_col].apply(map_allele_to_supertype_pooling)
    
    # Perform one-hot encoding on the pooled allele categories
    if encode_mode == "one_hot":
        one_hot = pd.get_dummies(df_processed[allele_col], prefix="allele_pooled")
        df_processed = pd.concat([df_processed, one_hot], axis=1)
        
    return df_processed
