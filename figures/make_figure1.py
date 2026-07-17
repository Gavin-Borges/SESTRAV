"""Figure 1: SESTRAV pipeline / dataset-to-evaluation schematic (DAG).

Matplotlib boxes + arrows (no graphviz). ASCII-only labels per
.claude/rules/encoding.md.

Flow: data sources -> v5 dataset assembly -> feature engineering
      -> models -> evaluation paradigms.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"

# Column palette (one hue per pipeline stage).
COL = {
    "source": "#DCE6F1",
    "assembly": "#FCE4D6",
    "feature": "#E2EFDA",
    "model": "#FFF2CC",
    "eval": "#EAD1F0",
}
EDGE = "#555555"

fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 100)
ax.set_ylim(33, 100)
ax.axis("off")


def box(cx, cy, w, h, text, facecolor, fontsize=9, bold=False):
    """Draw a rounded box centered at (cx, cy); return anchor dict."""
    x0, y0 = cx - w / 2, cy - h / 2
    patch = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.6,rounding_size=2.5",
        linewidth=1.1, edgecolor=EDGE, facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", wrap=True)
    return {"cx": cx, "cy": cy, "left": x0, "right": x0 + w,
            "top": y0 + h, "bottom": y0}


def arrow(a, b, from_side="right", to_side="left"):
    """Draw an arrow from box a to box b along the given sides."""
    pa = {
        "right": (a["right"], a["cy"]),
        "left": (a["left"], a["cy"]),
        "top": (a["cx"], a["top"]),
        "bottom": (a["cx"], a["bottom"]),
    }[from_side]
    pb = {
        "right": (b["right"], b["cy"]),
        "left": (b["left"], b["cy"]),
        "top": (b["cx"], b["top"]),
        "bottom": (b["cx"], b["bottom"]),
    }[to_side]
    ax.add_patch(FancyArrowPatch(
        pa, pb, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.2, color=EDGE, shrinkA=1, shrinkB=1,
    ))


# Column x-centers.
X1, X2, X3, X4, X5 = 10, 30, 51, 72, 91

# Stage headers.
for xc, name in [(X1, "Data sources"), (X2, "v5 dataset assembly"),
                 (X3, "Feature engineering"), (X4, "Models"),
                 (X5, "Evaluation")]:
    ax.text(xc, 96, name, ha="center", va="center", fontsize=11,
            fontweight="bold", color="#222222")

# --- Column 1: data sources ---
s1 = box(X1, 78, 15, 9, "IEDB\n(T-cell assays)", COL["source"])
s2 = box(X1, 62, 15, 9, "VDJdb\n(TCR-pMHC)", COL["source"])
s3 = box(X1, 46, 15, 9, "LANL HIV\ndatabase", COL["source"])

# --- Column 2: assembly ---
a1 = box(X2, 62, 17, 24,
         "v5 dataset\n\n"
         "quarantine split\n"
         "HLA normalization\n"
         "hard + viral decoys\n\n"
         "35,597 active\n51,185 total",
         COL["assembly"], bold=False)

# --- Column 3: feature engineering ---
f1 = box(X3, 76, 17, 11,
         "Physicochemical\n+ peptide length", COL["feature"])
f2 = box(X3, 60, 17, 13,
         "10 allele binding\nscores (MHCflurry)", COL["feature"])
f3 = box(X3, 44, 17, 11,
         "ESM-2 t12\nembeddings", COL["feature"])

# --- Column 4: models ---
m1 = box(X4, 66, 15, 12,
         "RF mode-31\n(production)", COL["model"], bold=True)
m2 = box(X4, 50, 15, 10, "GNN", COL["model"])

# --- Column 5: evaluation ---
e1 = box(X5, 74, 13, 11, "Within-virus\nCV", COL["eval"])
e2 = box(X5, 58, 13, 11, "Leave-one-\nvirus-out", COL["eval"])
e3 = box(X5, 42, 13, 11, "External-tool\nbenchmark", COL["eval"])

# --- Arrows: sources -> assembly ---
for s in (s1, s2, s3):
    arrow(s, a1, "right", "left")

# --- assembly -> features ---
for f in (f1, f2, f3):
    arrow(a1, f, "right", "left")

# --- features -> models (features feed both models) ---
for f in (f1, f2, f3):
    for m in (m1, m2):
        arrow(f, m, "right", "left")

# --- models -> evaluation ---
for m in (m1, m2):
    for e in (e1, e2, e3):
        arrow(m, e, "right", "left")

ax.set_title(
    "SESTRAV pipeline: dataset assembly to model evaluation",
    fontsize=14, fontweight="bold", y=0.99,
)

fig.tight_layout()
fig.savefig(OUT / "figure1_pipeline.tiff", dpi=300,
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(OUT / "figure1_pipeline.png", dpi=150)
print("Figure 1 written.")
