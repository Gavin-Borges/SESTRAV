from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def score_immunogenicity(features_df, virus_name):
    feature_cols = ["log_affinity", "hydrophobicity", "aromaticity", "net_charge", "length_tcr"]
    X = features_df[feature_cols]

    # Placeholder: using log_affinity as pseudo-label since no real immunogenicity labels
    y = features_df["log_affinity"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    features_df["immunogenicity_score"] = model.predict(X)
    features_df["rank"] = features_df["immunogenicity_score"].rank(ascending=False)

    features_df = features_df.sort_values("rank")
    features_df.to_csv(f"results/{virus_name}_ranked.csv", index=False)
    print(f"[Stage 4] Scored and ranked {len(features_df)} peptides")
    
    return features_df, model


def plot_immunogenicity_scores(ranked_df, virus_name, top_n=20):
    """
    Plots top N peptides by immunogenicity score and overall distribution.
    """
    # Top N bar plot
    top_df = ranked_df.head(top_n)
    
    plt.figure(figsize=(10,6))
    plt.barh(top_df["peptide"], top_df["immunogenicity_score"], color='skyblue')
    plt.xlabel("Immunogenicity Score")
    plt.ylabel("Peptide")
    plt.title(f"Top {top_n} Immunogenic Peptides - {virus_name}")
    plt.gca().invert_yaxis()  # highest score on top
    plt.tight_layout()
    plt.savefig(f"results/plots/{virus_name}_top{top_n}_immunogenicity.png")
    plt.close()

    # Overall score distribution
    plt.figure(figsize=(8,5))
    plt.hist(ranked_df["immunogenicity_score"], bins=50, color='salmon', alpha=0.7)
    plt.xlabel("Immunogenicity Score")
    plt.ylabel("Number of Peptides")
    plt.title(f"Immunogenicity Score Distribution - {virus_name}")
    plt.tight_layout()
    plt.savefig(f"results/plots/{virus_name}_score_distribution.png")
    plt.close()

    print(f"[Plot] Saved top {top_n} immunogenicity and distribution plots for {virus_name}")