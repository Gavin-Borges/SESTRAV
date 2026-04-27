"""
SESTRAV Gold-Standard Validation Module
Checks recovery of 15 well-characterized epitopes at each pipeline stage.

All 15 epitopes are recoverable from the current 8-antigen FASTA files
(confirmed: 15/15 Stage 1 recovery on Job 55598638, April 2026).

Strain notes:
  GLCTLVAML   — From BMLF1 (SM protein). BMLF1 is present in
                the EBV panel proteome FASTA.
  FLRGRAYGI   — B95-8 EBNA3A carries the Ile variant at p9 (FLRGRAYGI).
                The Leu variant (FLRGRAYGL) is from EBV type-2 strains.
                Both variants are held out from training; pipeline
                validation uses the B95-8 Ile variant.
  HPVGEADYFEY — B95-8 EBNA1 (P03211) contains the Glu variant.
                GD1-strain EBNA1 (Q3KSS4) has HPVGDADYFEY (Asp at p5).
                Current FASTA uses B95-8 EBNA1.
"""

import pandas as pd
from src.naming import proteome_id_candidates

GOLD_STANDARD = [
    {'peptide': 'CLGGLLTMV',   'protein': 'LMP2A',    'allele': 'HLA-A*02:01', 'virus': 'EBV'},
    {'peptide': 'GLCTLVAML',   'protein': 'BMLF1',    'allele': 'HLA-A*02:01', 'virus': 'EBV'},
    {'peptide': 'FLRGRAYGI',   'protein': 'EBNA3A',   'allele': 'HLA-B*08:01', 'virus': 'EBV'},
    {'peptide': 'RAKFKQLL',    'protein': 'BZLF1',    'allele': 'HLA-B*08:01', 'virus': 'EBV'},
    {'peptide': 'IVTDFSVIK',   'protein': 'EBNA3B',   'allele': 'HLA-A*11:01', 'virus': 'EBV'},
    {'peptide': 'RPPIFIRRL',   'protein': 'EBNA3A',   'allele': 'HLA-B*27:05', 'virus': 'EBV'},
    {'peptide': 'HPVGEADYFEY', 'protein': 'EBNA1',    'allele': 'HLA-B*35:01', 'virus': 'EBV'},
    {'peptide': 'TYSAGIVQI',   'protein': 'EBNA3B',   'allele': 'HLA-A*24:02', 'virus': 'EBV'},
    {'peptide': 'AVFDRKSDAK',  'protein': 'EBNA3B',   'allele': 'HLA-A*11:01', 'virus': 'EBV'},
    {'peptide': 'YVLDHLIVV',   'protein': 'BRLF1',    'allele': 'HLA-A*02:01', 'virus': 'EBV'},
    {'peptide': 'YMLDLQPET',   'protein': 'HPV16_E7', 'allele': 'HLA-A*02:01', 'virus': 'HPV'},
    {'peptide': 'RAHYNIVTF',   'protein': 'HPV16_E7', 'allele': 'HLA-B*35:01', 'virus': 'HPV'},
    {'peptide': 'LLMGTLGIV',   'protein': 'HPV16_E7', 'allele': 'HLA-A*02:01', 'virus': 'HPV'},
    {'peptide': 'KLPQLCTEL',   'protein': 'HPV16_E6', 'allele': 'HLA-A*02:01', 'virus': 'HPV'},
    {'peptide': 'TIHDIILECV',  'protein': 'HPV16_E6', 'allele': 'HLA-A*02:01', 'virus': 'HPV'},
]


GOLD_STANDARD_NEGATIVES = [
    # Strong-binder experimentally-negative peptides (IEDB T-cell negative,
    # NOT in the 201 cross-label conflict set, affinity < 500 nM for at
    # least one allele in the 10-allele panel).  Purpose: test whether the
    # integrated model correctly rejects non-immunogenic strong binders.
    #
    # --- Original set (10): curated from v1 analysis ---
    # EBV negatives (5) — sorted by predicted binding affinity
    {'peptide': 'LIPETVPYI',  'allele': 'HLA-A*02:01', 'virus': 'EBV',   'affinity_nM': 25.8},
    {'peptide': 'LPQGQLTAY',  'allele': 'HLA-B*35:01', 'virus': 'EBV',   'affinity_nM': 26.1},
    {'peptide': 'MLLLIVAGI',  'allele': 'HLA-A*02:01', 'virus': 'EBV',   'affinity_nM': 26.3},
    {'peptide': 'FTYPVLEEM',  'allele': 'HLA-A*02:01', 'virus': 'EBV',   'affinity_nM': 35.4},
    {'peptide': 'SYVKQPLCL',  'allele': 'HLA-A*24:02', 'virus': 'EBV',   'affinity_nM': 41.1},
    # HPV16 negatives (5) — sorted by presentation score
    {'peptide': 'CLLIRPLLL',  'allele': 'HLA-B*08:01', 'virus': 'HPV',   'affinity_nM': 75.9},
    {'peptide': 'IVYRDGNPY',  'allele': 'HLA-B*35:01', 'virus': 'HPV',   'affinity_nM': 64.1},
    {'peptide': 'RLCVQSTHV',  'allele': 'HLA-A*02:01', 'virus': 'HPV',   'affinity_nM': 38.9},
    {'peptide': 'DKKQRFHNI',  'allele': 'HLA-B*08:01', 'virus': 'HPV',   'affinity_nM': 172.0},
    {'peptide': 'AMFQDPQER',  'allele': 'HLA-A*11:01', 'virus': 'HPV',   'affinity_nM': 101.1},
    #
    # --- Expansion set (15): v2 high-presentation-score negatives ---
    # Selected from v2 training dataset negatives with highest MHC
    # presentation scores.  All are IEDB T-cell negative and absent from
    # the 201 cross-label conflict set.
    # EBV expansion (10) — sorted by presentation score
    {'peptide': 'HYQTLCTNF',   'allele': 'HLA-A*24:02', 'virus': 'EBV', 'affinity_nM': None},
    {'peptide': 'DYMAIHRSL',   'allele': 'HLA-A*24:02', 'virus': 'EBV', 'affinity_nM': None},
    {'peptide': 'AYAEATSSL',   'allele': 'HLA-A*24:02', 'virus': 'EBV', 'affinity_nM': None},
    {'peptide': 'LTEWGSGNRTY', 'allele': 'HLA-A*01:01', 'virus': 'EBV', 'affinity_nM': None},
    {'peptide': 'FYISLIQGL',   'allele': 'HLA-A*24:02', 'virus': 'EBV', 'affinity_nM': None},
    {'peptide': 'FYMTHGLGTL',  'allele': 'HLA-A*24:02', 'virus': 'EBV', 'affinity_nM': None},
    {'peptide': 'FYPLATYPL',   'allele': 'HLA-A*24:02', 'virus': 'EBV', 'affinity_nM': None},
    {'peptide': 'GIDPHLPTL',   'allele': 'HLA-A*02:01', 'virus': 'EBV', 'affinity_nM': None},
    {'peptide': 'NYNPGTLSSL',  'allele': 'HLA-A*24:02', 'virus': 'EBV', 'affinity_nM': None},
    {'peptide': 'IVTDLSIIK',   'allele': 'HLA-A*11:01', 'virus': 'EBV', 'affinity_nM': None},
    # HPV16 expansion (5) — sorted by presentation score
    {'peptide': 'CYSVYGTTL',   'allele': 'HLA-A*24:02', 'virus': 'HPV', 'affinity_nM': None},
    {'peptide': 'VYLTAPTGCI',  'allele': 'HLA-A*24:02', 'virus': 'HPV', 'affinity_nM': None},
    {'peptide': 'QPETTDLYCY',  'allele': 'HLA-B*35:01', 'virus': 'HPV', 'affinity_nM': None},
    {'peptide': 'LRLCVQSTH',   'allele': 'HLA-B*27:05', 'virus': 'HPV', 'affinity_nM': None},
    {'peptide': 'IVYIIFVYI',   'allele': 'HLA-A*02:01', 'virus': 'HPV', 'affinity_nM': None},
]

GOLD_STANDARD_NEGATIVES_EXPANDED = [
    # Kept for backward compatibility; these are now merged into
    # GOLD_STANDARD_NEGATIVES above. The list below mirrors the expansion
    # entries with their original presentation_score metadata.
    # EBV expansion (10) — sorted by presentation score
    {'peptide': 'HYQTLCTNF',  'allele': 'HLA-A*24:02', 'virus': 'EBV', 'presentation_score': 0.964},
    {'peptide': 'DYMAIHRSL',  'allele': 'HLA-A*24:02', 'virus': 'EBV', 'presentation_score': 0.960},
    {'peptide': 'AYAEATSSL',  'allele': 'HLA-A*24:02', 'virus': 'EBV', 'presentation_score': 0.957},
    {'peptide': 'LTEWGSGNRTY', 'allele': 'HLA-A*01:01', 'virus': 'EBV', 'presentation_score': 0.945},
    {'peptide': 'FYISLIQGL',  'allele': 'HLA-A*24:02', 'virus': 'EBV', 'presentation_score': 0.940},
    {'peptide': 'FYMTHGLGTL', 'allele': 'HLA-A*24:02', 'virus': 'EBV', 'presentation_score': 0.937},
    {'peptide': 'FYPLATYPL',  'allele': 'HLA-A*24:02', 'virus': 'EBV', 'presentation_score': 0.923},
    {'peptide': 'GIDPHLPTL',  'allele': 'HLA-A*02:01', 'virus': 'EBV', 'presentation_score': 0.918},
    {'peptide': 'NYNPGTLSSL', 'allele': 'HLA-A*24:02', 'virus': 'EBV', 'presentation_score': 0.905},
    {'peptide': 'IVTDLSIIK',  'allele': 'HLA-A*11:01', 'virus': 'EBV', 'presentation_score': 0.902},
    # HPV16 expansion (5) — sorted by presentation score
    {'peptide': 'CYSVYGTTL',  'allele': 'HLA-A*24:02', 'virus': 'HPV', 'presentation_score': 0.842},
    {'peptide': 'VYLTAPTGCI', 'allele': 'HLA-A*24:02', 'virus': 'HPV', 'presentation_score': 0.580},
    {'peptide': 'QPETTDLYCY', 'allele': 'HLA-B*35:01', 'virus': 'HPV', 'presentation_score': 0.520},
    {'peptide': 'LRLCVQSTH',  'allele': 'HLA-B*27:05', 'virus': 'HPV', 'presentation_score': 0.333},
    {'peptide': 'IVYIIFVYI',  'allele': 'HLA-A*02:01', 'virus': 'HPV', 'presentation_score': 0.330},
]


VIRUS_FILE_MAP = {
    'EBV': 'EBV_B95_8_panel8',
    'HPV': 'HPV16_18_panel8',
}


def _filter_gs(virus):
    """Return gold-standard entries for a specific virus, or all if None."""
    if virus is None:
        return GOLD_STANDARD
    return [gs for gs in GOLD_STANDARD if gs['virus'] == virus]


def validate_stage1(peptides_csv, virus=None):
    """Check which gold-standard peptides appear in sliding-window output."""
    df = pd.read_csv(peptides_csv)
    peptides_generated = set(df['peptide'].unique())
    results = []
    for gs in _filter_gs(virus):
        results.append({
            **gs,
            'stage1_found': gs['peptide'] in peptides_generated
        })
    return pd.DataFrame(results)


def validate_stage2(binding_csv, virus=None, ic50_threshold=500):
    """Check which gold-standard peptides have strong binding predictions."""
    df = pd.read_csv(binding_csv)
    results = []
    for gs in _filter_gs(virus):
        match = df[df['peptide'] == gs['peptide']]
        found = len(match) > 0
        strong_binder = False
        if found and 'affinity' in df.columns:
            strong_binder = bool(match['affinity'].min() < ic50_threshold)
        results.append({
            **gs,
            'stage2_found': found,
            'stage2_strong_binder': strong_binder
        })
    return pd.DataFrame(results)


def validate_stage4(ranked_csv, virus=None, top_pct=25):
    """Check which gold-standard peptides rank in the top N% of predictions."""
    df = pd.read_csv(ranked_csv)
    if 'immunogenicity_score' not in df.columns:
        return pd.DataFrame(), top_pct
    threshold_rank = len(df) * top_pct / 100
    results = []
    for gs in _filter_gs(virus):
        match = df[df['peptide'] == gs['peptide']]
        found = len(match) > 0
        in_top = False
        best_rank = None
        if found and 'rank' in df.columns:
            best_rank = match['rank'].min()
            in_top = best_rank <= threshold_rank
        results.append({
            **gs,
            'stage4_found': found,
            'rank': best_rank,
            f'in_top_{top_pct}pct': in_top
        })
    return pd.DataFrame(results), top_pct


def full_validation_report(results_dir, top_pct=25):
    """Run all validation stages across both viruses and produce a combined report.

    Args:
        results_dir: directory containing per-virus pipeline output CSVs
        top_pct: percentile cutoff for ranking evaluation (default 25)

    Returns:
        DataFrame with one row per gold-standard epitope and all stage results
    """
    import os
    all_s1, all_s2, all_s4 = [], [], []

    for virus, prefix in VIRUS_FILE_MAP.items():
        candidates = proteome_id_candidates(prefix)

        def _first_existing(suffix):
            for cand in candidates:
                path = os.path.join(results_dir, f"{cand}_{suffix}.csv")
                if os.path.isfile(path):
                    return path
            return os.path.join(results_dir, f"{prefix}_{suffix}.csv")

        peptides = _first_existing("peptides")
        binding = _first_existing("binding")
        ranked = _first_existing("ranked")

        if os.path.isfile(peptides):
            all_s1.append(validate_stage1(peptides, virus=virus))
        if os.path.isfile(binding):
            all_s2.append(validate_stage2(binding, virus=virus))
        if os.path.isfile(ranked):
            s4_df, _ = validate_stage4(ranked, virus=virus, top_pct=top_pct)
            if not s4_df.empty:
                all_s4.append(s4_df)

    s1 = pd.concat(all_s1, ignore_index=True) if all_s1 else pd.DataFrame()
    s2 = pd.concat(all_s2, ignore_index=True) if all_s2 else pd.DataFrame()
    s4 = pd.concat(all_s4, ignore_index=True) if all_s4 else pd.DataFrame()

    n_total = len(GOLD_STANDARD)
    # Start from canonical gold-standard table so reporting remains stable
    # even when some stage output files are missing.
    report = pd.DataFrame(GOLD_STANDARD).copy()
    report["stage1_found"] = False
    report["stage2_found"] = False
    report["stage2_strong_binder"] = False

    if not s1.empty:
        report = report.drop(columns=["stage1_found"]).merge(
            s1[["peptide", "stage1_found"]],
            on="peptide",
            how="left",
        )
        report["stage1_found"] = report["stage1_found"].fillna(False)

    if not s2.empty:
        report = report.drop(columns=["stage2_found", "stage2_strong_binder"]).merge(
            s2[["peptide", "stage2_found", "stage2_strong_binder"]],
            on="peptide",
            how="left",
        )
        report["stage2_found"] = report["stage2_found"].fillna(False)
        report["stage2_strong_binder"] = report["stage2_strong_binder"].fillna(False)

    top_col = f'in_top_{top_pct}pct'
    if not s4.empty and top_col in s4.columns:
        report = report.merge(
            s4[['peptide', 'stage4_found', 'rank', top_col]],
            on='peptide', how='left'
        )

    print("=" * 70)
    print("SESTRAV GOLD-STANDARD VALIDATION REPORT")
    print("=" * 70)

    for virus in ['EBV', 'HPV']:
        v_mask = report['virus'] == virus
        n_v = int(v_mask.sum())
        if n_v == 0:
            continue
        s1_n = int(report.loc[v_mask, 'stage1_found'].sum())
        s2_n = int(report.loc[v_mask, 'stage2_found'].sum()) if 'stage2_found' in report else 0
        sb_n = int(report.loc[v_mask, 'stage2_strong_binder'].sum()) if 'stage2_strong_binder' in report else 0
        print(f"\n  {virus} ({n_v} epitopes):")
        print(f"    Stage 1 (Peptide Generation): {s1_n}/{n_v}")
        print(f"    Stage 2 (MHC Binding):        {s2_n}/{n_v} found, {sb_n}/{n_v} strong binders")
        if not s4.empty and top_col in report.columns:
            s4_found = int(report.loc[v_mask, 'stage4_found'].sum())
            s4_top = int(report.loc[v_mask, top_col].sum())
            print(f"    Stage 4 (Ranking):            {s4_found}/{n_v} found, {s4_top}/{n_v} in top {top_pct}%")

    s1_total = int(report['stage1_found'].sum())
    s2_total = int(report['stage2_found'].sum()) if 'stage2_found' in report else 0
    sb_total = int(report['stage2_strong_binder'].sum()) if 'stage2_strong_binder' in report else 0
    print(f"\n  COMBINED ({n_total} epitopes):")
    print(f"    Stage 1: {s1_total}/{n_total}")
    print(f"    Stage 2: {s2_total}/{n_total} found, {sb_total}/{n_total} strong binders")
    if not s4.empty and top_col in report.columns:
        s4_total = int(report['stage4_found'].sum())
        top_total = int(report[top_col].sum())
        print(f"    Stage 4: {s4_total}/{n_total} found, {top_total}/{n_total} in top {top_pct}%")

    print("=" * 70)
    return report


def validate_negative_discrimination(results_dir, top_pct=25):
    """Check whether the integrated model pushes gold-standard negatives
    lower in the ranking than the binding-only baseline does.

    For each gold-standard negative peptide, compare its rank-percentile
    under integrated scoring vs binding-only (presentation_score) scoring.
    A successful model should rank these *higher* (worse percentile) than
    binding alone, indicating it learned to reject non-immunogenic binders.

    Returns:
        DataFrame with one row per gold-standard negative and rank data
        for both scoring methods.
    """
    import os
    gs_neg_peptides = {gs['peptide'] for gs in GOLD_STANDARD_NEGATIVES}
    results = []

    for virus, prefix in VIRUS_FILE_MAP.items():
        ranked_path = None
        binding_path = None
        for cand in proteome_id_candidates(prefix):
            rp = os.path.join(results_dir, f"{cand}_ranked.csv")
            bp = os.path.join(results_dir, f"{cand}_binding.csv")
            if os.path.isfile(rp):
                ranked_path = rp
            if os.path.isfile(bp):
                binding_path = bp
            if ranked_path and binding_path:
                break

        if ranked_path is None or binding_path is None:
            continue

        ranked_df = pd.read_csv(ranked_path)
        binding_df = pd.read_csv(binding_path)
        n_total = len(ranked_df)

        binding_ranked = binding_df.sort_values(
            'presentation_score', ascending=False
        ).reset_index(drop=True)
        binding_ranked['bind_rank'] = range(1, len(binding_ranked) + 1)

        for gs in GOLD_STANDARD_NEGATIVES:
            if gs['virus'] != virus:
                continue
            pep = gs['peptide']

            integ_match = ranked_df[ranked_df['peptide'] == pep]
            bind_match = binding_ranked[binding_ranked['peptide'] == pep]

            integ_rank = int(integ_match['rank'].min()) if len(integ_match) > 0 else None
            bind_rank = int(bind_match['bind_rank'].min()) if len(bind_match) > 0 else None

            integ_pct = (integ_rank / n_total * 100) if integ_rank else None
            bind_pct = (bind_rank / len(binding_ranked) * 100) if bind_rank else None

            pushed_down = None
            if integ_pct is not None and bind_pct is not None:
                pushed_down = integ_pct > bind_pct

            results.append({
                'peptide': pep,
                'virus': gs['virus'],
                'allele': gs['allele'],
                'affinity_nM': gs['affinity_nM'],
                'integrated_rank': integ_rank,
                'integrated_rank_pct': round(integ_pct, 2) if integ_pct else None,
                'binding_rank': bind_rank,
                'binding_rank_pct': round(bind_pct, 2) if bind_pct else None,
                'model_pushes_down': pushed_down,
            })

    report = pd.DataFrame(results)

    if report.empty:
        print("[GS Neg] No gold-standard negatives found in pipeline output")
        return report

    n_found = report['integrated_rank'].notna().sum()
    n_pushed = report['model_pushes_down'].sum() if 'model_pushes_down' in report else 0
    print("=" * 70)
    print("GOLD-STANDARD NEGATIVE DISCRIMINATION REPORT")
    print("=" * 70)
    print(f"  Negatives evaluated: {n_found}/{len(GOLD_STANDARD_NEGATIVES)}")
    print(f"  Model pushes down (vs binding): {n_pushed}/{n_found}")
    for _, row in report.iterrows():
        tag = "PUSHED DOWN" if row.get('model_pushes_down') else "NOT pushed down"
        print(f"    {row['peptide']} ({row['virus']}): "
              f"integrated={row['integrated_rank_pct']}% "
              f"vs binding={row['binding_rank_pct']}% -- {tag}")
    print("=" * 70)
    return report


def validate_expanded_negative_discrimination(results_dir, top_pct=25):
    """Run negative discrimination on the expanded candidate set.

    Same methodology as validate_negative_discrimination but uses
    GOLD_STANDARD_NEGATIVES_EXPANDED (pending expert review).
    Results are kept separate from the primary validation.
    """
    import os
    results = []

    for virus, prefix in VIRUS_FILE_MAP.items():
        ranked_path = None
        binding_path = None
        for cand in proteome_id_candidates(prefix):
            rp = os.path.join(results_dir, f"{cand}_ranked.csv")
            bp = os.path.join(results_dir, f"{cand}_binding.csv")
            if os.path.isfile(rp):
                ranked_path = rp
            if os.path.isfile(bp):
                binding_path = bp
            if ranked_path and binding_path:
                break

        if ranked_path is None or binding_path is None:
            continue

        ranked_df = pd.read_csv(ranked_path)
        binding_df = pd.read_csv(binding_path)
        n_total = len(ranked_df)

        binding_ranked = binding_df.sort_values(
            'presentation_score', ascending=False
        ).reset_index(drop=True)
        binding_ranked['bind_rank'] = range(1, len(binding_ranked) + 1)

        for gs in GOLD_STANDARD_NEGATIVES_EXPANDED:
            if gs['virus'] != virus:
                continue
            pep = gs['peptide']

            integ_match = ranked_df[ranked_df['peptide'] == pep]
            bind_match = binding_ranked[binding_ranked['peptide'] == pep]

            integ_rank = int(integ_match['rank'].min()) if len(integ_match) > 0 else None
            bind_rank = int(bind_match['bind_rank'].min()) if len(bind_match) > 0 else None

            integ_pct = (integ_rank / n_total * 100) if integ_rank else None
            bind_pct = (bind_rank / len(binding_ranked) * 100) if bind_rank else None

            pushed_down = None
            if integ_pct is not None and bind_pct is not None:
                pushed_down = integ_pct > bind_pct

            results.append({
                'peptide': pep,
                'virus': gs['virus'],
                'allele': gs['allele'],
                'presentation_score': gs['presentation_score'],
                'set': 'expanded',
                'integrated_rank': integ_rank,
                'integrated_rank_pct': round(integ_pct, 2) if integ_pct else None,
                'binding_rank': bind_rank,
                'binding_rank_pct': round(bind_pct, 2) if bind_pct else None,
                'model_pushes_down': pushed_down,
            })

    report = pd.DataFrame(results)

    if report.empty:
        print("[GS Neg Expanded] No expanded negatives found in pipeline output")
        return report

    n_found = report['integrated_rank'].notna().sum()
    n_pushed = int(report['model_pushes_down'].sum()) if 'model_pushes_down' in report else 0
    print("=" * 70)
    print("EXPANDED GOLD-STANDARD NEGATIVE DISCRIMINATION REPORT")
    print("=" * 70)
    print(f"  Candidates evaluated: {n_found}/{len(GOLD_STANDARD_NEGATIVES_EXPANDED)}")
    print(f"  Model pushes down (vs binding): {n_pushed}/{n_found}")
    for _, row in report.iterrows():
        tag = "PUSHED DOWN" if row.get('model_pushes_down') else "NOT pushed down"
        print(f"    {row['peptide']} ({row['virus']}): "
              f"integrated={row['integrated_rank_pct']}% "
              f"vs binding={row['binding_rank_pct']}% -- {tag}")
    print("=" * 70)
    return report
