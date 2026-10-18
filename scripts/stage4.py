import os

import pandas as pd
from functions.stage4_immunogenicity_scoring import score_immunogenicity, plot_immunogenicity_scores

proteome_id = snakemake.wildcards.proteome_id
features = pd.read_csv(snakemake.input[0])
model_path = snakemake.params.get("model_path", None)
if model_path == "":
    model_path = None
# config.yaml's thresholds_path: the threshold picked for the configured model on the
# peptide-grouped OOF. Unset falls back to optimal_thresholds.json beside the model.
thresholds_path = snakemake.params.get("thresholds_path", None) or None
# calibration_path and mc_dropout reach score_immunogenicity only from here and from
# pipeline.py. Until this rule passed them, setting either in config.yaml was a silent
# no-op under Snakemake: calibration fell back to a calibrator beside the model, and
# mc_dropout=true produced no MC Dropout pass at all, though the ANN model card tells a
# reader to enable it exactly that way. Same defect as thresholds_path before it, and the
# mirror of tests/test_pipeline_forwards_freeze_mode.py, where pipeline.py was the side
# that dropped a key.
calibration_path = snakemake.params.get("calibration_path", None) or None
freeze_mode = bool(snakemake.params.get("freeze_mode", False))
mc_dropout = bool(snakemake.params.get("mc_dropout", False))
output_dir = os.path.dirname(str(snakemake.output[0])) or "."

ranked, model = score_immunogenicity(
    features,
    proteome_id,
    model_path=model_path,
    thresholds_path=thresholds_path,
    calibration_path=calibration_path,
    freeze_mode=freeze_mode,
    mc_dropout=mc_dropout,
    output_dir=output_dir,
)
plot_immunogenicity_scores(ranked, proteome_id, output_dir=output_dir)
