"""
scripts/filter_validation_cohorts.py
====================================
Retrieves SARS-CoV-2 and Influenza A Class I ELISPOT assay datasets from the
IEDB Query API, applies standard pipeline cleaning (length/AA filtering, majority
voting, allele normalization), and filters out any exact or substring training
overlaps using bidirectional Aho-Corasick tries.

Outputs:
  - data/external/sars2_clean.csv
  - data/external/influenza_clean.csv
"""

import os
import sys
import urllib.request
import urllib.parse
import json
import pandas as pd
import ahocorasick

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN = 8
MAX_LEN = 11

# v5 is the required contamination-filter corpus. A missing path must raise.
# The previous exists-skip loop returned an EMPTY LIST, which main() turned into
# sys.exit(1), so the script could not run at all. It did not build an empty trie
# and it did not filter nothing: filter_bidirectional_overlap was never reached.
TRAINING_DATA_PATHS = [
    os.path.join(PROJECT_ROOT, "data", "immunogenicity_dataset_v5.csv"),
]

SARS2_OUTPUT = os.path.join(PROJECT_ROOT, "data", "external", "sars2_clean.csv")
INFLUENZA_OUTPUT = os.path.join(PROJECT_ROOT, "data", "external", "influenza_clean.csv")


def normalise_allele(raw):
    """Normalize allele notation to HLA-A*02:01."""
    if pd.isna(raw) or not str(raw).strip():
        return None
    s = str(raw).strip()
    if not s.upper().startswith("HLA-"):
        s = "HLA-" + s
    if "*" not in s:
        for prefix in ("HLA-A", "HLA-B", "HLA-C"):
            if s.upper().startswith(prefix):
                s = s[: len(prefix)] + "*" + s[len(prefix) :]
                break
    if "*" in s and ":" not in s:
        star_pos = s.index("*")
        suffix = s[star_pos + 1 :]
        if len(suffix) >= 4:
            s = s[: star_pos + 1] + suffix[:2] + ":" + suffix[2:]
    return s


def is_mhc_class_i(allele):
    """Filter out Class II MHCs."""
    if not allele:
        return False
    return not allele.startswith(("HLA-DR", "HLA-DP", "HLA-DQ"))


def fetch_cohort_data(organism_name):
    """Query IEDB IQ-API for class I ELISPOT assays for a given organism."""
    print(f"Fetching IEDB data for organism: {organism_name}")
    # Construct PostgREST query
    base_url = "https://query-api.iedb.org/tcell_search"
    params = {
        "source_organism_name": f"ilike.%{organism_name}%",
        "mhc_class": "eq.I",
        "assay_names": "ilike.%elispot%",
        "select": "linear_sequence,qualitative_measure,mhc_allele_name",
        "limit": 10000,
    }
    query_str = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_str}"

    try:
        # base_url is a hardcoded HTTPS IEDB endpoint (no user-controlled scheme);
        # reject anything that is not HTTPS as defense-in-depth before opening.
        if not url.lower().startswith("https://"):
            raise ValueError(f"Refusing non-HTTPS URL: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:  # nosec B310 - fixed trusted HTTPS endpoint, scheme validated above
            data = json.loads(response.read().decode("utf-8"))
        print(f"  Successfully retrieved {len(data)} raw records.")
        return data
    except Exception as e:
        print(f"  Error fetching data: {e}", file=sys.stderr)
        return []


def clean_and_curate(records, virus_name):
    """Apply SESTRAV-standard validation, mapping, and majority vote deduplication."""
    cleaned_rows = []

    for r in records:
        pep = r.get("linear_sequence")
        val_measure = r.get("qualitative_measure")
        allele_raw = r.get("mhc_allele_name")

        if not pep or not val_measure:
            continue

        pep = str(pep).strip().upper()
        # 1. Length & standard AA filter
        if not (MIN_LEN <= len(pep) <= MAX_LEN) or not all(aa in STANDARD_AA for aa in pep):
            continue

        # 2. Binary label mapping
        val_lower = str(val_measure).strip().lower()
        if val_lower.startswith("positive"):
            label = 1
        elif val_lower == "negative":
            label = 0
        else:
            continue

        # 3. Allele normalization & Class I check
        allele = normalise_allele(allele_raw)
        if not allele or not is_mhc_class_i(allele):
            continue

        cleaned_rows.append({"peptide": pep, "label": label, "allele": allele, "virus": virus_name})

    if not cleaned_rows:
        return pd.DataFrame()

    df = pd.DataFrame(cleaned_rows)

    # 4. Deduplicate by peptide (majority voting)
    # Compute mean label per peptide to handle multiple assays
    agg = (
        df.groupby("peptide")
        .agg(
            mean_label=("label", "mean"),
            allele=("allele", "first"),  # grab first representative allele
            virus=("virus", "first"),
        )
        .reset_index()
    )

    # Resolve exact ties (0.5) by dropping, and map remaining via majority vote
    agg = agg[agg["mean_label"] != 0.5].copy()
    agg["label"] = (agg["mean_label"] > 0.5).astype(int)

    # Return in standardized layout
    return agg[["peptide", "label", "virus", "allele"]].reset_index(drop=True)


def _training_path_rel(path):
    """Repo-relative POSIX path for error messages (no absolute workstation path)."""
    try:
        rel = os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        rel = os.path.basename(path)
    return rel.replace("\\", "/")


def load_training_peptides():
    """Load all unique peptides from required training datasets.

    Every path in TRAINING_DATA_PATHS must exist. A missing reference is a
    hard error. Raising here is a usability fix, not a correctness one: the
    previous exists-skip loop returned an empty peptide set, and main() already
    refused to continue on that (sys.exit(1) with an ERROR line naming a
    filename that no longer existed). The gain is that the failure now names
    the real missing path at the point of the read.
    """
    missing = [p for p in TRAINING_DATA_PATHS if not os.path.exists(p)]
    if missing:
        listed = ", ".join(_training_path_rel(p) for p in missing)
        raise FileNotFoundError(
            "Required training reference corpus missing: " + listed
        )

    peptides = set()
    for path in TRAINING_DATA_PATHS:
        df = pd.read_csv(path)
        if "peptide" not in df.columns:
            raise ValueError(
                "Training reference "
                + _training_path_rel(path)
                + " has no peptide column"
            )
        peptides.update(df["peptide"].dropna().str.strip().str.upper())
        print(
            "Loaded "
            + str(len(df))
            + " rows from training dataset: "
            + os.path.basename(path)
        )
    if not peptides:
        raise ValueError("Training reference corpus contained no peptides")
    return list(peptides)


def filter_bidirectional_overlap(train_peptides, eval_df, name=""):
    """
    Remove any peptide from eval_df that has an exact match or substring match
    with the training set (either direction).

    An empty train_peptides is a HARD ERROR, not an early return. The previous
    combined guard returned eval_df UNFILTERED in that case, which is the one
    outcome this function exists to prevent: the caller receives a cohort that
    was never contamination-checked, and nothing downstream can tell it apart
    from a filtered one. An empty eval_df is different and stays an early
    return, because there is genuinely nothing to filter.

    main() cannot reach this with an empty set now that load_training_peptides
    raises, so this guard exists for any other caller.
    """
    if not train_peptides:
        raise ValueError(
            "Refusing to filter "
            + (name or "cohort")
            + " against an empty training peptide set: the result would be "
            "unfiltered but indistinguishable from a contamination-filtered cohort"
        )
    if eval_df.empty:
        return eval_df

    eval_peptides = eval_df["peptide"].unique()

    # Build Aho-Corasick automaton of evaluation peptides to find E in T
    A_eval = ahocorasick.Automaton()
    for pep in eval_peptides:
        A_eval.add_word(pep, pep)
    A_eval.make_automaton()

    eval_in_train = set()
    for train_pep in train_peptides:
        for end_idx, eval_pep in A_eval.iter(train_pep):
            eval_in_train.add(eval_pep)

    # Build Aho-Corasick automaton of training peptides to find T in E
    A_train = ahocorasick.Automaton()
    for train_pep in train_peptides:
        A_train.add_word(train_pep, train_pep)
    A_train.make_automaton()

    train_in_eval = set()
    for eval_pep in eval_peptides:
        for end_idx, train_pep in A_train.iter(eval_pep):
            train_in_eval.add(eval_pep)

    contaminated = eval_in_train.union(train_in_eval)

    print(f"Contamination analysis for {name}:")
    print(f"  Total validation peptides: {len(eval_df)}")
    print(f"  Eval-in-Train overlaps: {len(eval_in_train)}")
    print(f"  Train-in-Eval overlaps: {len(train_in_eval)}")
    print(f"  Total unique contaminated excluded: {len(contaminated)}")

    clean_df = eval_df[~eval_df["peptide"].isin(contaminated)].copy()
    print(f"  Clean validation peptides: {len(clean_df)}")
    return clean_df


def main():
    print("SESTRAV Zero-Overlap Validation Cohorts Filtering")
    print("=" * 60)

    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(SARS2_OUTPUT), exist_ok=True)

    # Load training dataset peptides
    train_peptides = load_training_peptides()
    print(f"Total unique training peptides loaded: {len(train_peptides)}")

    # 1. Fetch, clean, and filter SARS-CoV-2
    sars2_raw = fetch_cohort_data("Severe acute respiratory syndrome coronavirus 2")
    if sars2_raw:
        sars2_df = clean_and_curate(sars2_raw, "SARS-CoV-2")
        sars2_clean = filter_bidirectional_overlap(train_peptides, sars2_df, "SARS-CoV-2")
        sars2_clean.to_csv(SARS2_OUTPUT, index=False)
        print(f"[SUCCESS] Wrote clean SARS-CoV-2 cohort to {SARS2_OUTPUT}")
    else:
        print("WARNING: No raw SARS-CoV-2 data fetched.")

    print("-" * 60)

    # 2. Fetch, clean, and filter Influenza A
    flu_raw = fetch_cohort_data("Influenza A virus")
    if flu_raw:
        flu_df = clean_and_curate(flu_raw, "InfluenzaA")
        flu_clean = filter_bidirectional_overlap(train_peptides, flu_df, "InfluenzaA")
        flu_clean.to_csv(INFLUENZA_OUTPUT, index=False)
        print(f"[SUCCESS] Wrote clean Influenza A cohort to {INFLUENZA_OUTPUT}")
    else:
        print("WARNING: No raw Influenza A data fetched.")

    print("=" * 60)
    print("Cohort filtering pipeline completed successfully.")


if __name__ == "__main__":
    main()
