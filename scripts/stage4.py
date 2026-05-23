import pandas as pd
from functions.stage4_immunogenicity_scoring import score_immunogenicity, plot_immunogenicity_scores

proteome_id = snakemake.wildcards.proteome_id
features = pd.read_csv(snakemake.input[0])
model_path = snakemake.params.get("model_path", None)
if model_path == "":
    model_path = None
freeze_mode = bool(snakemake.params.get("freeze_mode", False))

ranked, model = score_immunogenicity(
    features,
    proteome_id,
    model_path=model_path,
    freeze_mode=freeze_mode,
)
plot_immunogenicity_scores(ranked, proteome_id)
