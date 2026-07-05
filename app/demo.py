"""
SESTRAV 2.0 - Streamlit Interactive Demo

Run locally:
    streamlit run app/demo.py

This interface targets single-peptide exploration.  Batch scoring is reserved
for the FastAPI /score endpoint and the CLI pipeline.

Headless-server safety:
    matplotlib.use('Agg') is set before any pyplot import so the demo renders
    correctly on WSL2, Linux CI, and Docker environments without a display server.

Research disclaimer:
    SESTRAV is a research-grade immunogenicity predictor.  Predictions are for
    hypothesis generation only and must not drive clinical decisions.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend - headless safe
import matplotlib.pyplot as plt
import numpy as np

import streamlit as st

# Ensure the project root is on sys.path so src.* imports resolve
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SESTRAV 2.0 - Immunogenicity Predictor",
    page_icon="🧬",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Model singleton (cached across user interactions)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading SESTRAV model...")
def _load_model():
    from src.artifact_integrity import load_verified_joblib
    model_path = _ROOT / "models" / "rf_30feature_integrated.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"RF model not found at {model_path}. "
            "Run the SESTRAV training pipeline first."
        )
    return load_verified_joblib(model_path, required_checksum=True)


@st.cache_resource(show_spinner="Loading SHAP explainer...")
def _load_explainer(model):
    import shap
    return shap.TreeExplainer(model)


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _get_binding_score(sequence: str, allele: str) -> float | None:
    try:
        from mhcflurry import Class1PresentationPredictor
        predictor = Class1PresentationPredictor.load()
        result = predictor.predict(
            peptides=[sequence],
            alleles=[allele],
            include_affinity_percentile=False,
        )
        return float(result["presentation_score"].iloc[0])
    except Exception:
        return None


def _build_feature_vector(sequence: str, binding_score: float, allele: str = "") -> tuple[np.ndarray, list[str]]:
    """Build a 30-feature vector matching rf_30feature_integrated.joblib.

    FEATURE_COLUMNS_30 = 20 physicochemical (p4-p8 × 4 properties) +
                          10 per-allele binding scores (bind_A0101 ... bind_B4402).
    When MHCflurry is unavailable, binding columns stay at 0.0 except for the
    queried allele's column which receives the provided binding_score.
    """
    from src.features import compute_features, FEATURE_COLUMNS_30, BINDING_ALLELE_COLUMNS
    feat_dict = compute_features(sequence, binding_score=0.0)

    # Map the allele-specific binding score into the correct column.
    # Allele format: HLA-A*02:01 -> bind_A0201
    allele_col = (
        "bind_" + allele.replace("HLA-", "").replace("*", "").replace(":", "")
        if allele else None
    )
    for col in BINDING_ALLELE_COLUMNS:
        if col == allele_col:
            feat_dict[col] = binding_score
        else:
            feat_dict[col] = 0.0

    cols = FEATURE_COLUMNS_30
    vec = np.array([feat_dict.get(c, 0.0) for c in cols], dtype=np.float64).reshape(1, -1)
    return vec, cols


# ---------------------------------------------------------------------------
# SHAP waterfall (Matplotlib / Agg)
# ---------------------------------------------------------------------------

def _shap_waterfall_figure(
    shap_values,
    feature_names: list[str],
    feature_values: np.ndarray,
    base_value: float,
    peptide: str,
    top_n: int = 12,
) -> plt.Figure:
    """Render a waterfall chart using pure Matplotlib (no shap.plots.waterfall)
    so that the Agg backend is the sole rendering path - no Qt/Tcl dependency."""

    vals = shap_values if shap_values.ndim == 1 else shap_values[0]

    # Keep top_n features by |SHAP| magnitude
    order = np.argsort(np.abs(vals))[::-1][:top_n]
    sel_vals   = vals[order]
    sel_names  = [feature_names[i] for i in order]
    sel_feats  = feature_values[order]

    # Sort ascending for visual waterfall (low → high)
    sort_idx   = np.argsort(sel_vals)
    sel_vals   = sel_vals[sort_idx]
    sel_names  = [sel_names[i] for i in sort_idx]
    sel_feats  = sel_feats[sort_idx]

    colors = ["#e05c5c" if v < 0 else "#4a9e7f" for v in sel_vals]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.42)), dpi=120)
    y_pos = np.arange(len(sel_vals))
    bars = ax.barh(y_pos, sel_vals, color=colors, edgecolor="white", linewidth=0.4)

    for bar, name, fval, sv in zip(bars, sel_names, sel_feats, sel_vals):
        label = f"{name} = {fval:.3g}"
        x_off = 0.003 if sv >= 0 else -0.003
        ha    = "left" if sv >= 0 else "right"
        ax.text(sv + x_off, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha=ha, fontsize=7.5, color="#1a1a1a")

    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * len(y_pos))  # labels embedded in bars
    ax.set_xlabel("SHAP value  (contribution to P(immunogenic))", fontsize=9)
    ax.set_title(f"Feature contributions - {peptide}", fontsize=11, pad=10)
    ax.text(0.98, 0.02, f"Base: {base_value:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#555555")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PDF scorecard
# ---------------------------------------------------------------------------

def _build_pdf_scorecard(
    peptide: str,
    allele: str,
    imm_score: float,
    bind_score: float | None,
    rank: str,
    shap_fig: plt.Figure,
) -> bytes:
    """Assemble a single-page PDF scorecard using Matplotlib as the PDF backend."""
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11),
                                 gridspec_kw={"height_ratios": [1, 3]})

        # --- Header panel ---
        ax_hdr = axes[0]
        ax_hdr.axis("off")
        ax_hdr.text(0.0, 0.95, "SESTRAV 2.0 - Immunogenicity Scorecard",
                    fontsize=15, fontweight="bold", va="top")
        ax_hdr.text(0.0, 0.72, f"Peptide:   {peptide}", fontsize=11, va="top",
                    family="monospace")
        ax_hdr.text(0.0, 0.55, f"Allele:    {allele}", fontsize=11, va="top")
        score_color = "#c0392b" if rank == "HIGH" else ("#f39c12" if rank == "MEDIUM" else "#27ae60")
        ax_hdr.text(0.0, 0.38,
                    f"Immunogenicity Score: {imm_score:.3f}   [{rank}]",
                    fontsize=12, va="top", color=score_color, fontweight="bold")
        bind_txt = f"{bind_score:.3f}" if bind_score is not None else "N/A (MHCflurry unavailable)"
        ax_hdr.text(0.0, 0.20, f"Binding Score:  {bind_txt}", fontsize=10, va="top")
        ax_hdr.text(0.0, 0.04,
                    "⚠ Research use only - not for clinical decision-making.",
                    fontsize=8, va="top", color="#777777")

        # --- SHAP waterfall panel ---
        ax_shap = axes[1]
        ax_shap.axis("off")
        # Draw the SHAP figure onto the PDF axes via a canvas draw call
        shap_fig.canvas.draw()
        img = np.frombuffer(shap_fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = shap_fig.canvas.get_width_height()
        img = img.reshape(h, w, 3)
        ax_shap.imshow(img, aspect="auto", extent=[0, 1, 0, 1])
        ax_shap.set_xlim(0, 1)
        ax_shap.set_ylim(0, 1)

        fig.suptitle("", y=1.0)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# HLA allele catalogue (subset from NetMHCpan 4.1 trained set)
# ---------------------------------------------------------------------------

ALLELE_OPTIONS = [
    "HLA-A*01:01", "HLA-A*02:01", "HLA-A*03:01", "HLA-A*11:01",
    "HLA-A*24:02", "HLA-B*07:02", "HLA-B*08:01", "HLA-B*27:05",
    "HLA-B*35:01", "HLA-B*44:02",
]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main() -> None:
    # Header
    st.markdown(
        """
        <h1 style='font-size:2rem; margin-bottom:0.2rem;'>
            🧬 SESTRAV 2.0
        </h1>
        <p style='color:#666; margin-top:0;'>
            T-cell epitope immunogenicity predictor &nbsp;|&nbsp;
            <em>Research use only - not for clinical decision-making</em>
        </p>
        <hr style='margin:0.5rem 0 1.5rem 0;'>
        """,
        unsafe_allow_html=True,
    )

    # Input form
    with st.form("score_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            sequence = st.text_input(
                "Peptide sequence (8-11 IUPAC amino acids)",
                value="GILGFVFTL",
                max_chars=11,
                placeholder="e.g. GILGFVFTL",
            )
        with col2:
            allele = st.selectbox("HLA allele", ALLELE_OPTIONS, index=1)

        submitted = st.form_submit_button("🔬 Score peptide", use_container_width=True)

    if not submitted:
        st.info("Enter a peptide sequence and click **Score peptide** to begin.")
        return

    # Input validation
    import re
    sequence = sequence.strip().upper()
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{8,11}", sequence):
        st.error(
            "Invalid sequence. Use 8-11 uppercase standard amino acid letters "
            "(no B, J, O, U, X, Z)."
        )
        return

    # Load model
    try:
        model = _load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    with st.spinner("Computing..."):
        # Binding score (optional)
        bind_score = _get_binding_score(sequence, allele)
        if bind_score is None:
            st.warning("MHCflurry is not available - scoring without binding features.")

        # Feature vector (30-feature aligned to rf_30feature_integrated)
        feat_vec, feat_cols = _build_feature_vector(sequence, bind_score or 0.0, allele=allele)

        # Predict
        imm_score = float(model.predict_proba(feat_vec)[0, 1])
        rank = "HIGH" if imm_score >= 0.70 else ("MEDIUM" if imm_score >= 0.40 else "LOW")
        rank_color = {"HIGH": "#c0392b", "MEDIUM": "#e67e22", "LOW": "#27ae60"}[rank]

        # SHAP
        try:
            explainer = _load_explainer(model)
            shap_raw  = explainer.shap_values(feat_vec)
            # RF returns list of 2 arrays [class0, class1]
            shap_vals = shap_raw[1][0] if isinstance(shap_raw, list) else shap_raw[0]
            base_val  = float(explainer.expected_value[1]) if isinstance(
                explainer.expected_value, (list, np.ndarray)
            ) else float(explainer.expected_value)
            shap_ok = True
        except Exception as exc:
            st.warning(f"SHAP computation unavailable: {exc}")
            shap_ok = False

    # --- Results ---
    st.markdown("---")
    st.subheader("Prediction results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Immunogenicity Score", f"{imm_score:.3f}")
    m2.metric("Binding Score", f"{bind_score:.3f}" if bind_score is not None else "N/A")
    m3.metric(
        "Rank",
        rank,
        delta=None,
        help="HIGH ≥ 0.70 | MEDIUM 0.40-0.69 | LOW < 0.40",
    )
    st.markdown(
        f"<p style='font-size:0.85rem; color:{rank_color}; font-weight:600;'>"
        f"Prediction rank: {rank}</p>",
        unsafe_allow_html=True,
    )

    # --- SHAP waterfall ---
    if shap_ok:
        st.markdown("---")
        st.subheader("Feature contributions (SHAP)")
        st.caption(
            "Positive (green) bars push the prediction toward *immunogenic*; "
            "negative (red) bars push toward *non-immunogenic*."
        )
        shap_fig = _shap_waterfall_figure(
            shap_vals, feat_cols, feat_vec[0], base_val, sequence
        )
        st.pyplot(shap_fig, clear_figure=True)

        # PDF download
        st.markdown("---")
        pdf_bytes = _build_pdf_scorecard(
            sequence, allele, imm_score, bind_score, rank, shap_fig
        )
        st.download_button(
            label="⬇ Download PDF scorecard",
            data=pdf_bytes,
            file_name=f"SESTRAV_{sequence}_{allele.replace('*', '').replace(':', '')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # --- Contamination disclosure ---
    with st.expander("ℹ️ Contamination disclosure & caveats"):
        st.markdown(
            """
            **Benchmark overlap & clean-holdout evaluation**

            Because IEDB-derived benchmarks can overlap a model's training data, SESTRAV
            reports results on a contamination-excluded clean holdout, and is evaluated on
            a held-out independent validation cohort (SARS-CoV-2 and Influenza A) with no
            overlap with the training set.

            **Usage limits**
            - Predictions are for research hypothesis generation only.
            - Do not use SESTRAV as a clinical decision engine.
            - Peptides outside the 8-11-mer MHC-I binding length range are not supported.

            See `docs/validation_summary.md` for validation metrics.
            """
        )


if __name__ == "__main__":
    main()
