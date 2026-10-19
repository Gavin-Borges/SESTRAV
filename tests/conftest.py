"""Fixtures shared across test modules.

The root conftest.py only scrubs the environment at import; fixtures live here.
"""

import pytest


@pytest.fixture
def no_installed_conformal_calibrator(monkeypatch, tmp_path):
    """Repoint Stage 4's default conformal calibrator at a path that does not exist.

    ``DEFAULT_CONFORMAL_CALIBRATOR`` is anchored to the checkout, not the cwd, so
    ``monkeypatch.chdir`` cannot hide it. The file it names,
    models/v5/conformal_calibrator.joblib, is gitignored: absent from every CI
    checkout and fresh clone, present on any workstation that built it. A test whose
    premise is "no calibrator resolves" must remove that candidate itself, or it
    passes in CI and fails wherever the artifact exists.
    """
    absent = tmp_path / "absent_install_root" / "models" / "v5" / "conformal_calibrator.joblib"
    monkeypatch.setattr(
        "functions.stage4_immunogenicity_scoring.DEFAULT_CONFORMAL_CALIBRATOR", str(absent)
    )
    return absent
