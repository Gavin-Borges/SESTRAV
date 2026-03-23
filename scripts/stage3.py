import pandas as pd
from functions.stage3_tcr_feature_extraction import extract_tcr_features

virus = snakemake.wildcards.virus
binding = pd.read_csv(snakemake.input[0])

extract_tcr_features(binding, virus)