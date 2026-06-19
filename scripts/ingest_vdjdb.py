import argparse
import os
import sys
import urllib.request
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ssl_fix  # noqa: F401, E402 — patch SSL before any network calls
from _dataset_utils import normalize_peptides, validate_against_schema, write_provenance

VDJDB_RELEASES_API = "https://api.github.com/repos/antigenomics/vdjdb-db/releases/latest"


def _resolve_vdjdb_asset_url() -> str:
    """Resolve the latest VDJdb release asset URL via GitHub API."""
    import json
    req = urllib.request.Request(VDJDB_RELEASES_API)
    req.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
        data = json.loads(resp.read())
    assets = data.get("assets", [])
    # Find the first .zip asset (e.g. vdjdb-2026-06-03.zip)
    for asset in assets:
        if asset["name"].endswith(".zip"):
            return asset["browser_download_url"]
    raise RuntimeError(
        f"No .zip asset found in VDJdb latest release (tag: {data.get('tag_name')})"
    )


def fetch_vdjdb(output_dir: str = "data") -> str | None:
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "vdjdb.zip")
    if not os.path.exists(zip_path):
        try:
            url = _resolve_vdjdb_asset_url()
            print(f"Downloading VDJdb from {url}...")
            urllib.request.urlretrieve(url, zip_path)  # nosec B310
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print("Downloaded and extracted VDJdb.")
        except Exception as e:
            print(f"Failed to fetch VDJdb: {e}")
            print("Please manually download VDJdb and provide the TSV path.")
            return None
    # Find vdjdb.txt (may be at top level or in a subdirectory)
    for candidate in [
        os.path.join(output_dir, "vdjdb.txt"),
        os.path.join(output_dir, "vdjdb_full.txt"),
    ]:
        if os.path.exists(candidate):
            return candidate
    # Search extracted files
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            if f == "vdjdb.txt" or f == "vdjdb_full.txt":
                return os.path.join(root, f)
    print("Warning: could not find vdjdb.txt in extracted archive.")
    return None



def ingest_vdjdb(input_path, output_path, schema_path):
    fetched = False
    if not input_path or not os.path.exists(input_path):
        input_path = fetch_vdjdb()
        fetched = True
        if not input_path or not os.path.exists(input_path):
            raise FileNotFoundError(f"Could not find or fetch VDJdb input: {input_path}")

    print(f"Ingesting VDJdb from {input_path}")
    df = pd.read_csv(input_path, sep='\t')

    # --- Column name normalization across VDJdb releases ---
    # Old format: 'Epitope', 'MHC A', 'Epitope species', 'Epitope gene', 'CDR3.alpha', 'CDR3.beta'
    # New format (2026+): 'antigen.epitope', 'mhc.a', 'antigen.species', 'antigen.gene', 'cdr3', 'gene'
    _col_map = {
        'antigen.epitope': 'Epitope',
        'mhc.a': 'MHC A',
        'antigen.species': 'Epitope species',
        'antigen.gene': 'Epitope gene',
    }
    for old_name, new_name in _col_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    if 'Epitope' not in df.columns or 'MHC A' not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("VDJdb input missing expected columns 'Epitope'/'antigen.epitope' "
                         "and 'MHC A'/'mhc.a'")

    # --- Species filter: exclude non-viral entries ---
    # SESTRAV is a viral antigen immunogenicity predictor. VDJdb also contains
    # human autoantigens/tumor antigens, synthetic controls, and bacterial/fungal
    # epitopes which are biologically incompatible with viral training and must
    # be excluded. Confirmed non-viral species from VDJdb 2026-06-03 release.
    _EXCLUDE_SPECIES = frozenset([
        # Human autoantigens and tumor/neoantigen epitopes
        'HomoSapiens', 'Homo sapiens',
        # Synthetic / non-natural peptides
        'Synthetic',
        # Bacteria
        'KlebsiellaOxytoca', 'M.tuberculosis', 'E.Coli',
        'StreptomycesKanamyceticus',
        # Fungi
        'AspergillusOryzae', 'FusariumOxysporum', 'CryptococcusNeoformans',
        # Plants
        'Wheat',
    ])
    _species_col = 'Epitope species'
    if _species_col in df.columns:
        n_before = len(df)
        df = df[~df[_species_col].isin(_EXCLUDE_SPECIES)]
        n_excluded = n_before - len(df)
        if n_excluded > 0:
            print(f"Excluded {n_excluded} non-viral rows "
                  f"(human autoantigens, bacteria, fungi, synthetic controls).")

    # --- CDR3 extraction ---
    # New VDJdb (2026+) has one CDR3 per row with 'gene' column (TRA or TRB).
    # Pivot into alpha/beta columns keyed on (Epitope, MHC A).
    _new_cdr3_format = ('cdr3' in df.columns and 'gene' in df.columns
                        and 'CDR3.alpha' not in df.columns)

    if _new_cdr3_format:
        # Pivot: for each (Epitope, MHC A), extract TRA and TRB CDR3 sequences.
        tra = df[df['gene'] == 'TRA'][['Epitope', 'MHC A', 'cdr3']].drop_duplicates(
            subset=['Epitope', 'MHC A']).rename(columns={'cdr3': 'CDR3.alpha'})
        trb = df[df['gene'] == 'TRB'][['Epitope', 'MHC A', 'cdr3']].drop_duplicates(
            subset=['Epitope', 'MHC A']).rename(columns={'cdr3': 'CDR3.beta'})
        # Start from TRB (dominant chain), left-join alpha
        pivot = trb.merge(tra, on=['Epitope', 'MHC A'], how='outer')
        _alpha_col, _beta_col = 'CDR3.alpha', 'CDR3.beta'
    else:
        pivot = None
        # Legacy CDR3 column detection
        _alpha_cols = ['CDR3.alpha', 'cdr3.alpha', 'CDR3_alpha', 'cdr3_alpha']
        _beta_cols  = ['CDR3.beta',  'cdr3.beta',  'CDR3_beta',  'cdr3_beta',  'CDR3', 'cdr3']
        _alpha_col = next((c for c in _alpha_cols if c in df.columns), None)
        _beta_col  = next((c for c in _beta_cols  if c in df.columns), None)

    # Sort deterministically before deduplication so the retained CDR3 is stable across runs.
    sort_cols = ['Epitope', 'MHC A']
    if not _new_cdr3_format and _beta_col:
        sort_cols.append(_beta_col)
    df = df.sort_values(sort_cols, na_position='last')

    # Keep unique epitope/allele pairs (many TCRs bind the same epitope).
    unique_epitopes = df.drop_duplicates(subset=['Epitope', 'MHC A']).copy()
    unique_epitopes = unique_epitopes[unique_epitopes['MHC A'].str.contains('HLA', na=False)].copy()

    # Build the v4 DataFrame
    df_v4 = pd.DataFrame({
        'peptide': unique_epitopes['Epitope'].values,
        'label': 1,  # VDJdb contains known positive binders
        'virus': unique_epitopes.get('Epitope species', pd.Series(dtype='object')).fillna('Unknown').values,
        'protein': unique_epitopes.get('Epitope gene', pd.Series(dtype='object')).fillna('Unknown').values,
        'strain': 'Unknown',
        'hla_allele': unique_epitopes['MHC A'].values,
        'source_type': 'Virus',
        'database_source': 'VDJdb',
    })

    # Attach CDR3 sequences
    if _new_cdr3_format and pivot is not None:
        df_v4 = df_v4.merge(pivot, left_on=['peptide', 'hla_allele'],
                            right_on=['Epitope', 'MHC A'], how='left')
        df_v4 = df_v4.drop(columns=['Epitope', 'MHC A'], errors='ignore')
        df_v4 = df_v4.rename(columns={'CDR3.alpha': 'tcr_alpha_cdr3', 'CDR3.beta': 'tcr_beta_cdr3'})
    else:
        if _alpha_col:
            df_v4['tcr_alpha_cdr3'] = unique_epitopes[_alpha_col].where(
                unique_epitopes[_alpha_col].notna(), other=None).values
        else:
            df_v4['tcr_alpha_cdr3'] = None
        if _beta_col:
            df_v4['tcr_beta_cdr3'] = unique_epitopes[_beta_col].where(
                unique_epitopes[_beta_col].notna(), other=None).values
        else:
            df_v4['tcr_beta_cdr3'] = None

    # Ensure CDR3 columns exist
    for col in ['tcr_alpha_cdr3', 'tcr_beta_cdr3']:
        if col not in df_v4.columns:
            df_v4[col] = None

    cdr3_alpha_count = int(df_v4['tcr_alpha_cdr3'].notna().sum())
    cdr3_beta_count  = int(df_v4['tcr_beta_cdr3'].notna().sum())
    if cdr3_alpha_count or cdr3_beta_count:
        print(f"CDR3 sequences captured — alpha: {cdr3_alpha_count}, beta: {cdr3_beta_count} rows")
    else:
        print("CDR3 columns not found in this VDJdb release; tcr_alpha_cdr3/tcr_beta_cdr3 set to null.")

    df_v4 = df_v4.dropna(subset=['peptide', 'hla_allele'])

    # Keep only high-resolution Class I alleles (e.g. HLA-A*02:01).
    valid_allele_mask = df_v4['hla_allele'].str.match(r'^[A-Z]+-[A-Z]\*\d{2}:\d{2}$').fillna(False)
    print(f"Dropped {int((~valid_allele_mask).sum())} entries due to low-resolution HLA alleles.")
    df_v4 = df_v4[valid_allele_mask]

    df_v4 = normalize_peptides(df_v4)
    df_v4 = df_v4.drop_duplicates(subset=['peptide', 'hla_allele'])
    df_v4 = df_v4.sort_values(['peptide', 'hla_allele']).reset_index(drop=True)

    # Replace NaN with None so JSON schema validation sees null, not NaN.
    for col in ['tcr_alpha_cdr3', 'tcr_beta_cdr3']:
        if col in df_v4.columns:
            df_v4[col] = df_v4[col].where(df_v4[col].notna(), other=None)

    validate_against_schema(df_v4, schema_path)
    print(f"Extracted {len(df_v4)} unique positive epitopes from VDJdb.")
    df_v4.to_csv(output_path, index=False)
    write_provenance(
        output_path, sources=[input_path], row_count=len(df_v4),
        extra={"database": "VDJdb", "fetched_latest_release": fetched,
               "url": VDJDB_RELEASES_API if fetched else None,
               "tcr_alpha_cdr3_populated": cdr3_alpha_count,
               "tcr_beta_cdr3_populated": cdr3_beta_count}
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest VDJdb TSV into SESTRAV v4 Schema")
    parser.add_argument("--input", required=False, help="Path to VDJdb TSV file (will fetch if omitted)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--schema", default="data/immunogenicity_dataset_v4_schema.json",
                        help="v4 schema path")
    args = parser.parse_args()

    ingest_vdjdb(args.input, args.output, args.schema)
