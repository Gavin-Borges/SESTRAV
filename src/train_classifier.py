"""
SESTRAV Offline Model Training Script

Trains RandomForest and XGBoost classifiers on cleaned IEDB immunogenicity data.
Produces serialized .joblib models for use in the production scoring pipeline.

Feature modes:
  --feature-mode 21  (default, legacy)
      21 sequence-only features (p4-p8 physicochemical + peptide_length).
      binding_score excluded because IEDB exports lack allele info.

  --feature-mode 30
      20 physicochemical features + 10 per-allele MHCflurry binding scores.
      Requires --binding-matrix pointing to peptide_binding_matrix.csv
      (produced by CMB 523 Project 2).

Evaluation uses stratified 5-fold cross-validation on the full training pool
for reliable metric estimates, then retrains on the full pool for the final
serialized model.

Usage:
    python -m src.train_classifier --data immunogenicity_dataset.csv
    python -m src.train_classifier --data immunogenicity_dataset.csv \\
        --feature-mode 30 --binding-matrix models/peptide_binding_matrix.csv
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.ml_utils import MultiStratifiedKFold
from xgboost import XGBClassifier
from joblib import dump

from src.artifact_integrity import MODEL_CHECKSUM_MANIFEST, update_checksum_manifest
from src.features import (
    compute_features, FEATURE_COLUMNS, TRAIN_FEATURE_COLUMNS,
    FEATURE_COLUMNS_30, FEATURE_COLUMNS_31, FEATURE_COLUMNS_33, FEATURE_COLUMNS_35,
    BINDING_ALLELE_COLUMNS, PHYSICO_COLUMNS,
    FEATURE_COLUMNS_ALLELE, HLA_PSEUDO_COLS,
    FEATURE_COLUMNS_50, EXPANDED_PHYSICO_COLUMNS,
    compute_sample_weights,
    get_esm_cls_token, get_cb_cb_edges, compute_wl_features,
    FEATURE_COLUMNS_30_ESM, FEATURE_COLUMNS_30_GRAPH,
    load_antigen_processing_cache, load_self_similarity_cache,
)
from src.evaluate_metrics import evaluate
from src.iedb_data_loader import GOLD_STANDARD_EPITOPES
from src.subgroup_eval import evaluate_subgroups, pick_operating_threshold


def prepare_features(df, include_binding=False, binding_col='binding_score'):
    """Compute feature matrix for all peptides in the DataFrame.

    When include_binding is False (training mode), returns 21 features
    (excluding binding_score).  When True (inference mode), returns all 22.
    """
    feature_records = []
    for _, row in df.iterrows():
        peptide = row['peptide']
        binding = 0.0
        if include_binding:
            binding = row.get(binding_col, 0.0)
            if pd.isna(binding):
                binding = 0.0
        feats = compute_features(peptide, binding_score=binding)
        feature_records.append(feats)
    cols = FEATURE_COLUMNS if include_binding else TRAIN_FEATURE_COLUMNS
    return pd.DataFrame(feature_records)[cols]


def prepare_features_30(df, binding_matrix_path):
    """Build the 30-feature matrix by joining physico features with per-allele binding.

    Returns a DataFrame with columns matching FEATURE_COLUMNS_30.
    """
    binding_df = pd.read_csv(binding_matrix_path)
    bind_cols_present = [c for c in BINDING_ALLELE_COLUMNS if c in binding_df.columns]
    if len(bind_cols_present) < 10:
        raise ValueError(
            f"Binding matrix has only {len(bind_cols_present)}/10 expected allele columns"
        )
    binding_lookup = binding_df.set_index('peptide')[bind_cols_present]

    physico_records = []
    for _, row in df.iterrows():
        feats = compute_features(row['peptide'], binding_score=0.0)
        physico_records.append(feats)
    physico_df = pd.DataFrame(physico_records)[PHYSICO_COLUMNS]

    peptides = df['peptide'].values
    bind_rows = []
    for pep in peptides:
        if pep in binding_lookup.index:
            bind_rows.append(binding_lookup.loc[pep].values)
        else:
            bind_rows.append(np.zeros(10))
    bind_df = pd.DataFrame(bind_rows, columns=BINDING_ALLELE_COLUMNS)

    return pd.concat([physico_df.reset_index(drop=True),
                      bind_df.reset_index(drop=True)], axis=1)[FEATURE_COLUMNS_30]


def prepare_features_31(df, binding_matrix_path):
    """Build the 31-feature canonical matrix: 30-feature baseline + peptide_length."""
    base_30 = prepare_features_30(df, binding_matrix_path)
    lengths = df['peptide'].str.len().values
    length_df = pd.DataFrame({'peptide_length': lengths}).reset_index(drop=True)
    return pd.concat([base_30.reset_index(drop=True), length_df], axis=1)[FEATURE_COLUMNS_31]


def prepare_features_33(df, binding_matrix_path, cache_path):
    """Build the 33-feature extended matrix: 31-feature canonical + netchop_score + tap_score."""
    base_31 = prepare_features_31(df, binding_matrix_path)
    df_with_scores = load_antigen_processing_cache(cache_path, df.reset_index(drop=True))
    proc_df = df_with_scores[['netchop_score', 'tap_score']].reset_index(drop=True)
    return pd.concat([base_31.reset_index(drop=True), proc_df], axis=1)[FEATURE_COLUMNS_33]


def prepare_features_35(df, binding_matrix_path, ap_cache_path, sim_cache_path):
    """Build the 35-feature tolerance-aware matrix: 33-feature extended + self-similarity.

    Adds two columns from the human-proteome k-mer lookup cache:
      self_similarity_max_identity  - float [0, 1]; 1.0 = exact match
      self_similarity_exact_match   - float 0/1 (cast from bool for homogeneous dtype)

    Self-peptide matches indicate central-tolerance targets (thymic deletion),
    the mechanistic basis for non-immunogenicity in hard decoys.
    """
    base_33 = prepare_features_33(df, binding_matrix_path, ap_cache_path)
    df_with_sim = load_self_similarity_cache(sim_cache_path, df.reset_index(drop=True))
    sim_df = df_with_sim[['self_similarity_max_identity', 'self_similarity_exact_match']].reset_index(drop=True)
    return pd.concat([base_33.reset_index(drop=True), sim_df], axis=1)[FEATURE_COLUMNS_35]


def prepare_features_30_esm(df, binding_matrix_path):
    """Build the 350-feature matrix by combining 30-feature baseline with ESM CLS tokens."""
    base_30 = prepare_features_30(df, binding_matrix_path)
    esm_rows = []
    for pep in df['peptide'].values:
        esm_rows.append(get_esm_cls_token(pep))
    esm_df = pd.DataFrame(esm_rows, columns=[f"esm_{i}" for i in range(320)])
    return pd.concat([base_30.reset_index(drop=True), esm_df.reset_index(drop=True)], axis=1)[FEATURE_COLUMNS_30_ESM]


def prepare_features_30_graph(df, binding_matrix_path):
    """Build the 62-feature matrix by combining 30-feature baseline with WL graph descriptors."""
    base_30 = prepare_features_30(df, binding_matrix_path)
    graph_rows = []
    for pep in df['peptide'].values:
        edges = get_cb_cb_edges(len(pep))
        graph_rows.append(compute_wl_features(pep, edges))
    graph_df = pd.DataFrame(graph_rows, columns=[f"graph_wl_{i}" for i in range(32)])
    return pd.concat([base_30.reset_index(drop=True), graph_df.reset_index(drop=True)], axis=1)[FEATURE_COLUMNS_30_GRAPH]


def prepare_features_50(df, binding_matrix_path):
    """Build the 50-feature expanded matrix."""
    binding_df = pd.read_csv(binding_matrix_path)
    bind_cols_present = [c for c in BINDING_ALLELE_COLUMNS if c in binding_df.columns]
    if len(bind_cols_present) < 10:
        raise ValueError(
            f"Binding matrix has only {len(bind_cols_present)}/10 expected allele columns"
        )
    binding_lookup = binding_df.set_index('peptide')[bind_cols_present]

    physico_records = []
    for _, row in df.iterrows():
        feats = compute_features(row['peptide'], binding_score=0.0)
        physico_records.append(feats)
    physico_df = pd.DataFrame(physico_records)[EXPANDED_PHYSICO_COLUMNS]

    peptides = df['peptide'].values
    bind_rows = []
    for pep in peptides:
        if pep in binding_lookup.index:
            bind_rows.append(binding_lookup.loc[pep].values)
        else:
            bind_rows.append(np.zeros(10))
    bind_df = pd.DataFrame(bind_rows, columns=BINDING_ALLELE_COLUMNS)

    return pd.concat([physico_df.reset_index(drop=True),
                      bind_df.reset_index(drop=True)], axis=1)[FEATURE_COLUMNS_50]


def prepare_features_166(df, binding_matrix_path):
    """Build the 166-feature allele-aware matrix.

    Combines:
    - 20 physicochemical features (p4-p8 positions)
    - 10 per-allele MHCflurry binding scores
    - 136 HLA pocket pseudo-sequence features (pre-computed in the allele-aware CSV)

    The HLA pseudo-sequence columns (hla_p1_hydrophobicity ... hla_p34_charge)
    are expected to already be present in `df` (written by extract_allele_aware_data.py).
    The binding scores are looked up from the binding_matrix_path by peptide.

    Returns a DataFrame with columns matching FEATURE_COLUMNS_ALLELE (166 cols).
    """
    # Check pseudo-sequence columns are present
    missing_pseudo = [c for c in HLA_PSEUDO_COLS if c not in df.columns]
    if missing_pseudo:
        raise ValueError(
            f"Allele-aware dataset is missing {len(missing_pseudo)} HLA pseudo-sequence columns "
            f"(e.g. {missing_pseudo[0]}). Re-run scripts/extract_allele_aware_data.py."
        )

    # 20 physico features
    physico_records = []
    for _, row in df.iterrows():
        feats = compute_features(row['peptide'], binding_score=0.0)
        physico_records.append(feats)
    physico_df = pd.DataFrame(physico_records)[PHYSICO_COLUMNS]

    # 10 binding scores from matrix
    binding_df = pd.read_csv(binding_matrix_path)
    bind_cols_present = [c for c in BINDING_ALLELE_COLUMNS if c in binding_df.columns]
    if bind_cols_present:
        binding_lookup = binding_df.set_index('peptide')[bind_cols_present]
        bind_rows = []
        for pep in df['peptide'].values:
            if pep in binding_lookup.index:
                bind_rows.append(binding_lookup.loc[pep].values)
            else:
                bind_rows.append(np.zeros(len(bind_cols_present)))
        bind_df_out = pd.DataFrame(bind_rows, columns=BINDING_ALLELE_COLUMNS)
    else:
        bind_df_out = pd.DataFrame(
            np.zeros((len(df), 10)), columns=BINDING_ALLELE_COLUMNS
        )

    # 136 HLA pseudo-sequence features (already in df)
    pseudo_df = df[HLA_PSEUDO_COLS].reset_index(drop=True)

    return pd.concat([
        physico_df.reset_index(drop=True),
        bind_df_out.reset_index(drop=True),
        pseudo_df,
    ], axis=1)[FEATURE_COLUMNS_ALLELE]


def load_all_proteins():
    """Parse panel8 proteome FASTAs and full UniProt FASTAs to get protein name -> sequence mapping."""
    fasta_files = [
        "data/proteomes/EBV_B95_8_panel8.fasta",
        "data/proteomes/HPV16_18_panel8.fasta",
        "data/proteomes/EBV_B95_8_uniprot_reviewed.fasta",
        "data/proteomes/HPV16_uniprot_reviewed.fasta"
    ]
    proteins = {}
    for fpath in fasta_files:
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                header = None
                seq_parts = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if header is not None:
                            name = _get_protein_name_from_header(header)
                            proteins[name] = "".join(seq_parts).upper()
                        header = line[1:]
                        seq_parts = []
                    else:
                        seq_parts.append(line.upper())
                if header is not None:
                    name = _get_protein_name_from_header(header)
                    proteins[name] = "".join(seq_parts).upper()
    return proteins


def _get_protein_name_from_header(header: str) -> str:
    """Extract clean protein identifier from fasta header."""
    parts = header.split()
    if not parts:
        return ""
    first_token = parts[0]
    if first_token.startswith("sp|"):
        return first_token.split("|")[-1]
    return first_token


def _cross_validate(
    X,
    y,
    metadata,
    model_cls,
    model_kwargs,
    n_splits=5,
    random_state=42,
    subgroup_columns=None,
    min_group_size=15,
    sample_weights=None,
    use_lopo=False,
):
    """Run stratified k-fold or Leave-One-Protein-Out (LOPO) CV and return metrics."""
    if use_lopo:
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()
        groups = metadata["protein"].values
        splits = list(logo.split(X, y, groups=groups))
        print(f"  LOPO CV: detected {len(np.unique(groups))} unique groups -> {len(splits)} folds")
    else:
        mskf = MultiStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(mskf.split(
            X, y,
            negative_origin=metadata["negative_origin"] if "negative_origin" in metadata.columns else None,
            hla_alleles=metadata["hla_allele"] if "hla_allele" in metadata.columns else None,
            peptides=metadata["peptide"] if "peptide" in metadata.columns else None,
        ))

    fold_metrics = []
    subgroup_rows = []
    oof_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Sample weights for this fold's training split (if provided)
        sw_tr = None
        if sample_weights is not None:
            sw_tr = sample_weights[train_idx]

        model = model_cls(**model_kwargs)
        model.fit(X_tr, y_tr, sample_weight=sw_tr)
        scores = model.predict_proba(X_val)[:, 1]
        m = evaluate(y_val, scores)
        fold_metrics.append(m)
        fold_meta = metadata.iloc[val_idx].copy().reset_index(drop=True)
        fold_meta["label"] = y_val
        fold_meta["score"] = scores
        fold_meta["fold"] = fold_idx
        oof_rows.append(fold_meta)

        for row in evaluate_subgroups(
            fold_meta,
            score_col="score",
            label_col="label",
            group_columns=subgroup_columns,
            min_group_size=min_group_size,
        ):
            subgroup_rows.append(
                {
                    "fold": fold_idx,
                    **row,
                }
            )
        print(f"    Fold {fold_idx}: AUC-ROC={m['auc_roc']:.4f}  "
              f"AUC-PR={m['auc_pr']:.4f}  "
              f"ISSR@10={m['issr_10']:.4f}  "
              f"ISSR@25={m['issr_25']:.4f}")

    avg = {}
    std = {}
    for key in fold_metrics[0]:
        vals = [fm[key] for fm in fold_metrics]
        avg[key] = float(np.nanmean(vals))
        std[key] = float(np.nanstd(vals))
    subgroup_df = pd.DataFrame(subgroup_rows)
    oof_df = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    return avg, std, subgroup_df, oof_df


def train_models(data_path, model_dir='models', n_cv_folds=5, random_state=42,
                  feature_mode=21, binding_matrix_path=None,
                  antigen_processing_cache_path=None,
                  self_similarity_cache_path=None,
                  use_sample_weights=False, use_lopo=False):
    """
    Full training pipeline:
    1. Load cleaned IEDB data
    2. Remove gold-standard epitopes from training
    3. Map peptides to parent proteins using FASTAs
    4. Compute feature matrix (21 or 30 features depending on mode)
    5. Stratified 5-fold or LOPO cross-validation for reliable metric estimates
       (optionally with EBV/HPV16 and 9-mer/non-9-mer sample weights)
    6. Retrain on full pool with class-imbalance handling
    7. Serialize final models
    """
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")

    # Map peptides to parent proteins
    proteins_db = load_all_proteins()
    mapped_proteins = []
    mapped_count = 0
    synth_count = 0
    for pep in df['peptide'].values:
        parent = None
        for name, seq in proteins_db.items():
            if pep.upper() in seq:
                parent = name
                break
        if parent is not None:
            mapped_proteins.append(parent)
            mapped_count += 1
        else:
            mapped_proteins.append(f"SYNTH_{pep.upper()}")
            synth_count += 1
            
    df['protein'] = mapped_proteins
    print(f"Mapped {mapped_count} peptides to parent proteins. Synthetic fallback used for {synth_count} peptides.")

    gs_mask = df['peptide'].isin(GOLD_STANDARD_EPITOPES)
    gold_standard_df = df[gs_mask].copy()
    train_pool = df[~gs_mask].copy()
    print(f"Held out {len(gold_standard_df)} gold-standard epitope records")
    print(f"Training pool: {len(train_pool)} records")

    try:
        feature_mode = int(feature_mode)
    except (ValueError, TypeError):
        pass

    if feature_mode == 166:
        if binding_matrix_path is None:
            raise ValueError("--binding-matrix is required for feature-mode 166")
        X = prepare_features_166(train_pool, binding_matrix_path)
        feature_cols_used = FEATURE_COLUMNS_ALLELE
        mode_label = "166-feature allele-aware (20 physico + 10 binding + 136 HLA pseudo-seq)"
    elif feature_mode == 50:
        if binding_matrix_path is None:
            raise ValueError("--binding-matrix is required for feature-mode 50")
        X = prepare_features_50(train_pool, binding_matrix_path)
        feature_cols_used = FEATURE_COLUMNS_50
        mode_label = "50-feature expanded multi-allele"
    elif feature_mode == "30_esm":
        if binding_matrix_path is None:
            raise ValueError("--binding-matrix is required for feature-mode 30_esm")
        X = prepare_features_30_esm(train_pool, binding_matrix_path)
        feature_cols_used = FEATURE_COLUMNS_30_ESM
        mode_label = "350-feature (30 baseline + 320 ESM CLS token)"
    elif feature_mode == "30_graph":
        if binding_matrix_path is None:
            raise ValueError("--binding-matrix is required for feature-mode 30_graph")
        X = prepare_features_30_graph(train_pool, binding_matrix_path)
        feature_cols_used = FEATURE_COLUMNS_30_GRAPH
        mode_label = "62-feature (30 baseline + 32 WL graph descriptors)"
    elif feature_mode == 35:
        if binding_matrix_path is None:
            raise ValueError("--binding-matrix is required for feature-mode 35")
        if antigen_processing_cache_path is None:
            raise ValueError("--antigen-processing-cache is required for feature-mode 35")
        if self_similarity_cache_path is None:
            raise ValueError("--self-similarity-cache is required for feature-mode 35")
        X = prepare_features_35(
            train_pool, binding_matrix_path,
            antigen_processing_cache_path, self_similarity_cache_path,
        )
        feature_cols_used = FEATURE_COLUMNS_35
        mode_label = "35-feature tolerance-aware (33 extended + self_similarity_max_identity + self_similarity_exact_match)"
    elif feature_mode == 33:
        if binding_matrix_path is None:
            raise ValueError("--binding-matrix is required for feature-mode 33")
        if antigen_processing_cache_path is None:
            raise ValueError("--antigen-processing-cache is required for feature-mode 33")
        X = prepare_features_33(train_pool, binding_matrix_path, antigen_processing_cache_path)
        feature_cols_used = FEATURE_COLUMNS_33
        mode_label = "33-feature extended (31 canonical + netchop_score + tap_score)"
    elif feature_mode == 31:
        if binding_matrix_path is None:
            raise ValueError("--binding-matrix is required for feature-mode 31")
        X = prepare_features_31(train_pool, binding_matrix_path)
        feature_cols_used = FEATURE_COLUMNS_31
        mode_label = "31-feature canonical (20 physico + 10 binding + peptide_length)"
    elif feature_mode == 30:
        if binding_matrix_path is None:
            raise ValueError("--binding-matrix is required for feature-mode 30")
        X = prepare_features_30(train_pool, binding_matrix_path)
        feature_cols_used = FEATURE_COLUMNS_30
        mode_label = "30-feature multi-allele"
    else:
        X = prepare_features(train_pool, include_binding=False)
        feature_cols_used = TRAIN_FEATURE_COLUMNS
        mode_label = "21-feature sequence-only"

    y = train_pool['label'].values
    metadata_cols = [c for c in ["peptide", "virus", "strain", "protein", "negative_origin", "hla_allele"] if c in train_pool.columns]
    metadata = train_pool[metadata_cols].copy().reset_index(drop=True)
    print(f"Features: {X.shape[1]} ({mode_label})")
    print(f"Class balance: {np.mean(y):.2%} positive, {1 - np.mean(y):.2%} negative")

    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    spw = n_neg / max(n_pos, 1)

    # --- Sample weights (optional bias correction) ---
    sample_weights = None
    if use_sample_weights:
        sample_weights = compute_sample_weights(train_pool)
        # Normalize array to align with X index
        sample_weights = np.array(sample_weights)
        pos_rate_ebv = train_pool[train_pool['virus'] == 'EBV']['label'].mean() if 'virus' in train_pool.columns else None
        pos_rate_hpv = train_pool[train_pool['virus'] == 'HPV16']['label'].mean() if 'virus' in train_pool.columns else None
        nine_mer_pct = (train_pool['peptide'].str.len() == 9).mean()
        print("  Sample weights enabled (virus_weight=0.5, length_weight=0.5)")
        if pos_rate_ebv is not None:
            print(f"    EBV positive rate: {pos_rate_ebv:.2%}  |  HPV16 positive rate: {pos_rate_hpv:.2%}")
        print(f"    9-mer fraction: {nine_mer_pct:.2%}")
        print(f"    Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

    rf_kwargs = dict(
        n_estimators=200,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=1,
    )
    xgb_kwargs = dict(
        n_estimators=200,
        scale_pos_weight=spw,
        random_state=random_state,
        eval_metric='aucpr',
        objective='binary:logistic',
        nthread=1,
    )

    print(f"\n{'=' * 60}")
    print(f"RandomForest {n_cv_folds}-fold cross-validation:")
    print(f"{'=' * 60}")
    subgroup_columns = [c for c in ["virus", "strain", "protein"] if c in train_pool.columns]
    rf_avg, rf_std, rf_subgroups, rf_oof = _cross_validate(
        X, y, metadata, RandomForestClassifier, rf_kwargs,
        n_splits=n_cv_folds, random_state=random_state,
        subgroup_columns=subgroup_columns,
        sample_weights=sample_weights,
        use_lopo=use_lopo,
    )
    print("  Mean:  " + "  ".join(f"{k}={v:.4f}" for k, v in rf_avg.items()))
    print("  Stdev: " + "  ".join(f"{k}={v:.4f}" for k, v in rf_std.items()))

    print(f"\n{'=' * 60}")
    print(f"XGBoost {n_cv_folds}-fold cross-validation:")
    print(f"{'=' * 60}")
    xgb_avg, xgb_std, xgb_subgroups, xgb_oof = _cross_validate(
        X, y, metadata, XGBClassifier, xgb_kwargs,
        n_splits=n_cv_folds, random_state=random_state,
        subgroup_columns=subgroup_columns,
        sample_weights=sample_weights,
        use_lopo=use_lopo,
    )
    print("  Mean:  " + "  ".join(f"{k}={v:.4f}" for k, v in xgb_avg.items()))
    print("  Stdev: " + "  ".join(f"{k}={v:.4f}" for k, v in xgb_std.items()))

    print(f"\n{'=' * 60}")
    print("Retraining final models on full training pool...")
    print(f"{'=' * 60}")

    if feature_mode == 21:
        rf_stem = 'rf_21feature_legacy'
        xgb_stem = 'xgb_21feature_legacy'
    elif feature_mode == 166:
        rf_stem = 'rf_166feature_allele_aware'
        xgb_stem = 'xgb_166feature_allele_aware'
    else:
        rf_stem = f'rf_{feature_mode}feature_integrated'
        xgb_stem = f'xgb_{feature_mode}feature_integrated'

    rf_final = RandomForestClassifier(**rf_kwargs)
    rf_final.fit(X, y)
    rf_path = os.path.join(model_dir, f'{rf_stem}.joblib')
    dump(rf_final, rf_path)
    print(f"  RandomForest saved to {rf_path}")

    xgb_final = XGBClassifier(**xgb_kwargs)
    xgb_final.fit(X, y)
    xgb_path = os.path.join(model_dir, f'{xgb_stem}.joblib')
    dump(xgb_final, xgb_path)
    print(f"  XGBoost saved to {xgb_path}")

    results_rows = []
    for metric_key in rf_avg:
        results_rows.append({
            'metric': metric_key,
            'rf_cv_mean': rf_avg[metric_key],
            'rf_cv_std': rf_std[metric_key],
            'xgb_cv_mean': xgb_avg[metric_key],
            'xgb_cv_std': xgb_std[metric_key],
        })
    results_df = pd.DataFrame(results_rows)
    results_path = os.path.join(model_dir, 'training_results.csv')
    results_df.to_csv(results_path, index=False)
    # Also write a per-mode copy so successive runs of different feature modes do
    # not clobber each other's metrics (the generic file always holds the last run).
    results_mode_path = os.path.join(model_dir, f'training_results_mode{feature_mode}.csv')
    results_df.to_csv(results_mode_path, index=False)
    print(f"\nCV comparison saved to {results_path} and {results_mode_path}")

    subgroup_metrics = pd.concat(
        [
            rf_subgroups.assign(method="RandomForest"),
            xgb_subgroups.assign(method="XGBoost"),
        ],
        ignore_index=True,
    )
    subgroup_path = os.path.join(model_dir, "training_subgroup_metrics.csv")
    subgroup_metrics.to_csv(subgroup_path, index=False)
    print(f"Subgroup CV metrics saved to {subgroup_path}")

    feature_imp = pd.DataFrame({
        'feature': feature_cols_used,
        'rf_importance': rf_final.feature_importances_,
        'xgb_importance': xgb_final.feature_importances_,
    }).sort_values('rf_importance', ascending=False)
    imp_path = os.path.join(model_dir, 'feature_importances.csv')
    feature_imp.to_csv(imp_path, index=False)
    print(f"Feature importances saved to {imp_path}")

    if not rf_oof.empty:
        rf_oof_out = rf_oof.copy()
        rf_oof_out["method"] = "RandomForest"
        rf_oof_out["feature_mode"] = feature_mode
        rf_oof_path = os.path.join(model_dir, "rf_oof_predictions.csv")
        rf_oof_out.to_csv(rf_oof_path, index=False)
        # Per-mode copy (see note above) so the canonical mode-31 OOF is not
        # overwritten by later mode-33/35 runs.
        rf_oof_out.to_csv(os.path.join(model_dir, f"rf_oof_predictions_mode{feature_mode}.csv"), index=False)
        threshold_payload = pick_operating_threshold(
            rf_oof_out,
            score_col="score",
            label_col="label",
            group_col="virus",
            min_group_size=15,
        )
        threshold_payload["method"] = "RandomForest"
        threshold_payload["feature_mode"] = feature_mode
        threshold_payload["selection_objective"] = (
            "maximize minimum subgroup F1, then overall F1, then recall"
        )
        threshold_path = os.path.join(model_dir, "optimal_thresholds.json")
        with open(threshold_path, "w", encoding="utf-8") as f:
            json.dump(threshold_payload, f, indent=2)
        print(f"OOF predictions saved to {rf_oof_path}")
        print(f"Optimal threshold summary saved to {threshold_path}")

    checksum_manifest = os.path.join(model_dir, MODEL_CHECKSUM_MANIFEST)
    update_checksum_manifest(
        checksum_manifest,
        [
            rf_path,
            xgb_path,
            results_path,
            subgroup_path,
            imp_path,
            *( [rf_oof_path, threshold_path] if not rf_oof.empty else [] ),
        ],
    )
    print(f"Artifact checksums updated in {checksum_manifest}")

    return rf_final, xgb_final, rf_avg, xgb_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SESTRAV immunogenicity classifiers')
    parser.add_argument('--data', required=True, help='Path to immunogenicity_dataset.csv')
    parser.add_argument('--model-dir', default='models', help='Output directory for serialized models')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    parser.add_argument('--feature-mode', type=str, default="21",
                        choices=["21", "30", "31", "33", "35", "50", "166", "30_esm", "30_graph"],
                        help='Feature mode: 21 (sequence-only) | 30 (physico+binding) | 31 canonical | '
                             '33 (31+NetChop+TAPreg) | 35 (33+self-similarity) | 50 (expanded) | '
                             '30_esm | 30_graph')
    parser.add_argument('--binding-matrix', default=None,
                        help='Path to peptide_binding_matrix.csv (required for --feature-mode 30+)')
    parser.add_argument('--antigen-processing-cache', default=None,
                        help='Path to antigen processing cache CSV with netchop_score + tap_score '
                             '(required for --feature-mode 33+)')
    parser.add_argument('--self-similarity-cache', default=None,
                        help='Path to self_similarity_cache.csv with self_similarity_max_identity '
                             'and self_similarity_exact_match columns '
                             '(required for --feature-mode 35; generate with '
                             'scripts/precompute_self_similarity.py)')
    parser.add_argument('--sample-weights', action='store_true',
                        help='Apply EBV/HPV16 and 9-mer/non-9-mer bias-correction weights during training')
    parser.add_argument('--lopo', action='store_true',
                        help='Perform Leave-One-Protein-Out (LOPO) cross-validation instead of StratifiedKFold')
    args = parser.parse_args()

    # Input validation
    if not os.path.isfile(args.data):
        parser.error(f"Data file does not exist: '{args.data}'")
    if args.binding_matrix and not os.path.isfile(args.binding_matrix):
        parser.error(f"Binding matrix file does not exist: '{args.binding_matrix}'")
    if args.antigen_processing_cache and not os.path.isfile(args.antigen_processing_cache):
        parser.error(f"Antigen processing cache does not exist: '{args.antigen_processing_cache}'")
    if args.self_similarity_cache and not os.path.isfile(args.self_similarity_cache):
        parser.error(f"Self-similarity cache does not exist: '{args.self_similarity_cache}'")

    train_models(args.data, args.model_dir, n_cv_folds=args.cv_folds,
                 feature_mode=args.feature_mode, binding_matrix_path=args.binding_matrix,
                 antigen_processing_cache_path=args.antigen_processing_cache,
                 self_similarity_cache_path=args.self_similarity_cache,
                 use_sample_weights=args.sample_weights, use_lopo=args.lopo)
