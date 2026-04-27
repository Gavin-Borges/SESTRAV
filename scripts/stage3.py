import pandas as pd
from functions.stage3_tcr_feature_extraction import extract_tcr_features

proteome_id = snakemake.wildcards.proteome_id
binding = pd.read_csv(snakemake.input[0])

extract_tcr_features(binding, proteome_id)
