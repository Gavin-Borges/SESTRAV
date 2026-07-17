"""Figure 2: Leave-one-virus-out vs within-virus CV AUC-ROC, per pathogen.

Two-color grouped bar chart, 9 pathogens, dashed chance line at 0.5.
Data:
  Within-virus CV: results/per_virus_eval_v5_mode31.csv (auc_roc per virus)
  LOO:             results/loo_cross_virus_v5_clean.csv (auc_roc per test_virus)
ASCII-only labels per .claude/rules/encoding.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"

within = pd.read_csv(ROOT / "results" / "per_virus_eval_v5_mode31.csv")
loo = pd.read_csv(ROOT / "results" / "loo_cross_virus_v5_clean.csv")

within_map = dict(zip(within["virus"], within["auc_roc"]))
loo_map = dict(zip(loo["test_virus"], loo["auc_roc"]))

# 9 viruses evaluated under LOO, ordered by LOO AUC-ROC descending.
viruses = sorted(loo_map.keys(), key=lambda v: loo_map[v], reverse=True)

within_vals = [within_map[v] for v in viruses]
loo_vals = [loo_map[v] for v in viruses]

# Display labels (ASCII only).
label_map = {"SARS-CoV-2": "SARS-CoV-2"}
labels = [label_map.get(v, v) for v in viruses]

COLOR_WITHIN = "#4C72B0"  # blue
COLOR_LOO = "#C44E52"      # red

x = np.arange(len(viruses))
width = 0.38

fig, ax = plt.subplots(figsize=(11, 5.5))

bars_w = ax.bar(x - width / 2, within_vals, width, label="Within-virus CV",
                color=COLOR_WITHIN, edgecolor="black", linewidth=0.4)
bars_l = ax.bar(x + width / 2, loo_vals, width, label="LOO (held-out virus)",
                color=COLOR_LOO, edgecolor="black", linewidth=0.4)

ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2,
           label="Random chance (0.5)")

# Annotate HIV-1: highest within-CV bar and the anti-predictive LOO outlier.
ax.annotate(f"{within_vals[-1]:.3f}",
            xy=(x[-1] - width / 2, within_vals[-1]),
            xytext=(-6, 4), textcoords="offset points",
            ha="right", va="bottom", fontsize=9, fontweight="bold",
            color=COLOR_WITHIN)
ax.annotate(f"{loo_vals[-1]:.3f}",
            xy=(x[-1] + width / 2, loo_vals[-1]),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=COLOR_LOO)

ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_xlabel("Pathogen (held-out under LOO)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0.0, 1.0)
ax.set_title(
    "Leave-one-virus-out (LOO) cross-virus generalization vs within-virus CV\n"
    "SESTRAV RF mode-31, v5 dataset (IEDB assay-confirmed clean test partitions)",
    fontsize=12,
)

# Legend order: chance line first, then the two bar series.
handles, leg_labels = ax.get_legend_handles_labels()
order = [leg_labels.index("Random chance (0.5)"),
         leg_labels.index("Within-virus CV"),
         leg_labels.index("LOO (held-out virus)")]
ax.legend([handles[i] for i in order], [leg_labels[i] for i in order],
          loc="upper left", fontsize=10, framealpha=0.95)

ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()

fig.savefig(OUT / "figure2_loo_vs_within_cv.tiff", dpi=300,
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(OUT / "figure2_loo_vs_within_cv.png", dpi=150)

print("Figure 2 written. Values plotted (virus, within-CV, LOO):")
for v, w, l in zip(labels, within_vals, loo_vals):
    print(f"  {v:12s}  within={w:.4f}  LOO={l:.4f}")
