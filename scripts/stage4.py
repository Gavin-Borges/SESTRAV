import pandas as pd
from functions.stage4_immunogenicity_scoring import score_immunogenicity, plot_immunogenicity_scores

virus = snakemake.wildcards.virus
features = pd.read_csv(snakemake.input[0])

ranked, model = score_immunogenicity(features, virus)
plot_immunogenicity_scores(ranked, virus)