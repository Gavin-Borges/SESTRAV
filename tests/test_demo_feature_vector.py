"""Coverage for app/demo.py's binding-block assembly (D31) and PDF scorecard.

app/demo.py carried a SECOND, independently wrong implementation of the D31
defect: it wrote the queried allele's single MHCflurry score into that
allele's column and zeroed the other nine - a one-hot pattern appearing in 0
of the 18,535 rows of the tracked binding matrix. The API half of that fix is
covered by tests/test_api_main.py; this file covers the demo half, which the
claims register otherwise had to disclose as source-verified only.

streamlit is stubbed rather than installed. app/demo.py imports it at module
scope and calls st.set_page_config() there, so the module cannot be imported
without it, and streamlit appears in no environments/requirements-ci*.txt -
adding it purely to reach two pure functions would mean a hash-pinned lockfile
regeneration plus SBOM and pip-audit churn. The stub covers exactly the two
streamlit APIs reached at import time (cache_resource, set_page_config); the
functions under test are pure and touch neither.

These tests read the REAL models/peptide_binding_matrix_v5.csv (tracked, so
present in a clean checkout) rather than a fixture, because the property being
asserted is that the demo reads the same panel training does.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest


def _install_streamlit_stub() -> None:
    if "streamlit" in sys.modules and getattr(sys.modules["streamlit"], "_sestrav_stub", False):
        return
    st = types.ModuleType("streamlit")
    st._sestrav_stub = True  # type: ignore[attr-defined]

    def cache_resource(*dargs, **dkwargs):
        # Supports both @st.cache_resource and @st.cache_resource(show_spinner=...)
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        def _wrap(fn):
            return fn

        return _wrap

    st.cache_resource = cache_resource  # type: ignore[attr-defined]
    st.set_page_config = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["streamlit"] = st


@pytest.fixture(scope="module")
def demo() -> Iterator[object]:
    """app/demo.py with streamlit stubbed. Real streamlit wins if installed."""
    if "streamlit" not in sys.modules:
        try:
            import streamlit  # noqa: F401
        except ImportError:
            _install_streamlit_stub()
    import app.demo as demo_mod

    yield demo_mod


PANEL_PEPTIDE = "GILGFVFTL"


def test_panel_peptide_binding_block_matches_the_matrix(demo) -> None:
    """The regression: the ten bind_* columns must carry the peptide's real
    panel row, not a one-hot of the selected allele and not all zeros."""
    import pandas as pd

    from src.features import BINDING_ALLELE_COLUMNS

    built = demo._build_feature_vector(PANEL_PEPTIDE)
    assert built is not None, f"{PANEL_PEPTIDE} should be in the tracked panel"
    vec, cols = built
    assert len(cols) == 31

    got = [vec[0][cols.index(c)] for c in BINDING_ALLELE_COLUMNS]

    matrix = pd.read_csv("models/peptide_binding_matrix_v5.csv")
    row = matrix.loc[matrix["peptide"] == PANEL_PEPTIDE].iloc[0]
    expected = [float(row[c]) for c in BINDING_ALLELE_COLUMNS]

    assert got == pytest.approx(expected), "binding block does not match the panel row"
    assert any(v != 0.0 for v in got), "binding block was all-zero for a panel peptide"

    # The old defect's signature: exactly one nonzero column (one-hot). That
    # pattern occurs in 0 of 18,535 matrix rows, so it must not occur here.
    assert sum(1 for v in got if v != 0.0) > 1, "one-hot binding block - the D31 defect"


def test_out_of_panel_peptide_returns_none(demo) -> None:
    """Signals a miss instead of silently zero-filling, so main() can say so."""
    assert demo._build_feature_vector("KKKKKKKKK") is None


def test_build_feature_vector_takes_no_allele(demo) -> None:
    """D31: the panel is fixed and keyed by peptide, so the allele dropdown
    cannot move the binding block. Pinned at the signature, which is where the
    old implementation went wrong - it accepted and used an allele argument."""
    import inspect

    params = list(inspect.signature(demo._build_feature_vector).parameters)
    assert params == ["sequence"], f"unexpected signature: {params}"


def _scorecard_shap_figure():
    """A minimal stand-in for the real SHAP waterfall figure.

    _build_pdf_scorecard only rasterizes whatever figure it is handed, so the
    panel's contents are irrelevant to what these tests assert.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 4))
    fig.gca().barh(["f1", "f2"], [0.4, -0.2])
    return fig


def test_pdf_scorecard_builds_on_the_pinned_matplotlib(demo) -> None:
    """The download button was unreachable in practice: _build_pdf_scorecard
    called canvas.tostring_rgb(), absent from the pinned Matplotlib 3.11.1, so
    the unguarded call site raised AttributeError. Nothing in this file reached
    the function before, which is why that shipped."""
    import matplotlib.pyplot as plt

    shap_fig = _scorecard_shap_figure()
    try:
        pdf_bytes = demo._build_pdf_scorecard(
            PANEL_PEPTIDE, "HLA-A*02:01", 0.731, 0.512, "HIGH", shap_fig
        )
    finally:
        plt.close(shap_fig)

    assert pdf_bytes.startswith(b"%PDF-"), "did not return a PDF document"
    assert len(pdf_bytes) > 1024, f"implausibly small PDF: {len(pdf_bytes)} bytes"


def test_pdf_scorecard_handles_an_absent_binding_score(demo) -> None:
    """bind_score is None whenever MHCflurry is unavailable, which is the
    configuration the demo actually ships in, so that branch has to render."""
    import matplotlib.pyplot as plt

    shap_fig = _scorecard_shap_figure()
    try:
        pdf_bytes = demo._build_pdf_scorecard(
            PANEL_PEPTIDE, "HLA-A*02:01", 0.128, None, "LOW", shap_fig
        )
    finally:
        plt.close(shap_fig)

    assert pdf_bytes.startswith(b"%PDF-"), "did not return a PDF document"


def test_scorecard_panel_image_is_three_channel(demo) -> None:
    """buffer_rgba() is 4-channel where the removed tostring_rgb() was 3, so a
    fix that swapped the call without adjusting the channel count would hand
    imshow an array of the wrong shape. Pins the drop to RGB."""
    import matplotlib.pyplot as plt
    import numpy as np

    captured = []
    original = plt.Axes.imshow

    def _capture(self, X, *args, **kwargs):
        captured.append(np.asarray(X))
        return original(self, X, *args, **kwargs)

    shap_fig = _scorecard_shap_figure()
    plt.Axes.imshow = _capture
    try:
        demo._build_pdf_scorecard(PANEL_PEPTIDE, "HLA-A*02:01", 0.5, 0.5, "MED", shap_fig)
    finally:
        plt.Axes.imshow = original
        plt.close(shap_fig)

    assert captured, "_build_pdf_scorecard never drew the SHAP panel"
    panel = captured[0]
    assert panel.ndim == 3, f"panel image is not 2D RGB: shape {panel.shape}"
    assert panel.shape[2] == 3, f"panel image has {panel.shape[2]} channels, expected 3"
