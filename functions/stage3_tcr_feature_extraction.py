import numpy as np

def extract_tcr_features(binding_df, virus_name):
    def tcr_core(peptide):
        return peptide[3:-2] if len(peptide) >= 6 else peptide  # Positions 4 to Ω-2

    binding_df["tcr_core"] = binding_df["peptide"].apply(tcr_core)

    # Physicochemical features
    binding_df["hydrophobicity"] = binding_df["tcr_core"].apply(lambda x: sum(aa in "AILMFWYV" for aa in x))
    binding_df["aromaticity"] = binding_df["tcr_core"].apply(lambda x: sum(aa in "FWY" for aa in x))
    binding_df["positive_charge"] = binding_df["tcr_core"].apply(lambda x: sum(aa in "KR" for aa in x))
    binding_df["negative_charge"] = binding_df["tcr_core"].apply(lambda x: sum(aa in "DE" for aa in x))
    binding_df["net_charge"] = binding_df["positive_charge"] - binding_df["negative_charge"]
    binding_df["length_tcr"] = binding_df["tcr_core"].apply(len)
    binding_df["log_affinity"] = -np.log(binding_df["affinity"] + 1e-9)

    # Optionally, you can add molecular weight or VdW volume if you want more features
    # For simplicity, we include ~10 TCR-facing features here

    binding_df.to_csv(f"results/{virus_name}_features.csv", index=False)
    print(f"[Stage 3] Extracted TCR features for {len(binding_df)} peptide-allele pairs")
    return binding_df