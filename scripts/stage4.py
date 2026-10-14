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
freeze_mode = bool(snakemake.params.get("freeze_mode", False))
output_dir = os.path.dirname(str(snakemake.output[0])) or "."

ranked, model = score_immunogenicity(
    features,
    proteome_id,
    model_path=model_path,
    thresholds_path=thresholds_path,
    freeze_mode=freeze_mode,
    output_dir=output_dir,
)
plot_immunogenicity_scores(ranked, proteome_id, output_dir=output_dir)
