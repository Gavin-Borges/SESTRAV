# src/hla_supertypes.py
import re

HLA_SUPERTYPE_MAP = {
    # Locus A Supertypes
    "HLA-A*01:01": "A01",
    "HLA-A*02:01": "A02",
    "HLA-A*03:01": "A03",
    "HLA-A*11:01": "A03",
    "HLA-A*24:02": "A24",
    # Locus B Supertypes
    "HLA-B*07:02": "B07",
    "HLA-B*35:01": "B07",
    "HLA-B*08:01": "B08",
    "HLA-B*27:05": "B27",
    "HLA-B*44:02": "B44",
}

# Extend with common HLA-A/B family patterns for robust fallback matching
HLA_FAMILY_SUPERTYPE_MAP = {
    r"^HLA-A\*02:\d+$": "A02",
    r"^HLA-A\*01:\d+$": "A01",
    r"^HLA-A\*03:\d+$": "A03",
    r"^HLA-A\*11:\d+$": "A03",
    r"^HLA-A\*24:\d+$": "A24",
    r"^HLA-B\*07:\d+$": "B07",
    r"^HLA-B\*35:\d+$": "B07",
    r"^HLA-B\*08:\d+$": "B08",
    r"^HLA-B\*27:\d+$": "B27",
    r"^HLA-B\*44:\d+$": "B44",
    r"^HLA-B\*57:\d+$": "B58",
    r"^HLA-B\*58:\d+$": "B58",
}

def get_hla_supertype(allele: str) -> str:
    """
    Normalize HLA allele string and resolve its corresponding supertype.
    Falls back to family regex matching if the exact allele is not hardcoded.
    """
    clean_allele = str(allele).strip().upper().replace(" ", "")
    # Ensure format HLA-X*XX:XX
    if clean_allele in HLA_SUPERTYPE_MAP:
        return HLA_SUPERTYPE_MAP[clean_allele]
        
    for pattern, supertype in HLA_FAMILY_SUPERTYPE_MAP.items():
        if re.match(pattern, clean_allele):
            return supertype
            
    raise ValueError(f"Unknown or unmapped HLA allele sequence: {allele}")
