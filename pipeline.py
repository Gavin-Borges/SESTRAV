"""
SESTRAV Standalone Pipeline Entry Point
Runs all 4 stages sequentially without Snakemake for quick local testing.
"""

import os
import re
import logging
from functions.stage1_peptide_generation import generate_peptides
from functions.stage2_mhc_binding_prediction import predict_binding
from functions.stage3_tcr_feature_extraction import extract_tcr_features
from functions.stage4_immunogenicity_scoring import score_immunogenicity, plot_immunogenicity_scores
from src.naming import canonicalize_proteome_id
from src.core.config import SestravConfig
from src.core.feature_store import FeatureStore
from src.core.model_registry import ModelRegistry

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("sestrav.log", encoding="utf-8")],
)
logger = logging.getLogger("sestrav")


def _sanitize_name(name):
    """Allow only alphanumeric, underscores, and hyphens."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def run_pipeline(proteome_id: str, fasta_path: str, config: SestravConfig, registry: ModelRegistry):
    raw_id = _sanitize_name(proteome_id)
    proteome_id = canonicalize_proteome_id(raw_id)
    if proteome_id != raw_id:
        logger.info(
            f"[Naming] Using canonical proteome_id '{proteome_id}' (from legacy '{raw_id}')"
        )

    alleles = config.alleles
    lengths = config.peptide_lengths

    # Use ModelRegistry to resolve the configured model
    model_path = registry.resolve_model(config.model_path.name)
    mc_dropout = config.mc_dropout
    calibration_path = str(config.calibration_path) if config.calibration_path else None
    thresholds_path = str(config.thresholds_path) if config.thresholds_path else None

    peptides_df = generate_peptides(fasta_path, proteome_id, peptide_lengths=lengths)
    binding_df = predict_binding(peptides_df, proteome_id, alleles=alleles)
    features_df = extract_tcr_features(binding_df, proteome_id)

    # Provide the resolved absolute model_path
    ranked_df, model = score_immunogenicity(
        features_df,
        proteome_id,
        model_path=str(model_path),
        calibrate=True,
        mc_dropout=mc_dropout,
        calibration_path=calibration_path,
        thresholds_path=thresholds_path,
        freeze_mode=config.freeze_mode,
    )
    plot_immunogenicity_scores(ranked_df, proteome_id)

    logger.info(f"Pipeline complete for {proteome_id}\n")


if __name__ == "__main__":
    cfg = SestravConfig.load()
    store = FeatureStore(cfg.output_dir)
    registry = ModelRegistry(cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    proteome_files = cfg.proteome_files
    antigens = cfg.antigens

    if antigens:
        for configured_id in antigens:
            canonical_id = canonicalize_proteome_id(configured_id)
            fasta_path = (
                proteome_files.get(canonical_id)
                or proteome_files.get(configured_id)
                or os.path.join("data", "proteomes", f"{canonical_id}.fasta")
            )
            run_pipeline(canonical_id, fasta_path, cfg, registry)
    else:
        for fasta_file in os.listdir("data/proteomes"):
            if fasta_file.endswith(".fasta"):
                inferred_id = _sanitize_name(fasta_file.replace(".fasta", ""))
                run_pipeline(inferred_id, os.path.join("data/proteomes", fasta_file), cfg, registry)
