"""
SESTRAV 2.0 - FastAPI Microservice

Endpoints
---------
POST /score         Score a single peptide-allele pair.
GET  /health        Liveness probe.
GET  /model-card    Version metadata and training parameters.
GET  /provenance    Dataset checksums and Zenodo DOI.

Security
--------
- Pydantic schemas enforce IUPAC amino acid regex and length 8-11.
- HLA allele validated against the canonical NetMHCpan 4.1 format.
- Models loaded once at startup via ModelManager singleton (no per-request I/O).
- No shell=True, no eval()/exec(), no arbitrary code paths.

Performance
-----------
- Singleton pattern ensures the RF model is deserialized exactly once.
- Feature extraction is vectorised through src.features.compute_features.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger("sestrav-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Paths (relative to project root; mount the repo root as the working dir)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _PROJECT_ROOT / "models" / "rf_31feature_integrated.joblib"
_CHECKSUM_FILE = _PROJECT_ROOT / "models" / "model_artifact_checksums.json"
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

# No archival DOI has been minted yet; the API reports null rather than a fake
# placeholder. Set this to the real DOI (e.g. "10.5281/zenodo.NNNNNNN") once the
# Zenodo record exists.
_ZENODO_DOI: str | None = None

# Single source of truth for the served version: the installed package metadata
# (pyproject [project].version), with a fallback for uninstalled/source runs.
try:
    _APP_VERSION = version("sestrav")
except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled tree
    _APP_VERSION = "2.0.3"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

_IUPAC_PATTERN = r"^[ACDEFGHIKLMNPQRSTVWY]{8,11}$"
_ALLELE_PATTERN = r"^HLA-[AB]\*\d{2}:\d{2}$"


class PeptideInput(BaseModel):
    sequence: str = Field(
        ...,
        pattern=_IUPAC_PATTERN,
        description="Peptide sequence - 8 to 11 uppercase IUPAC amino acids.",
        examples=["GILGFVFTL"],
    )
    allele: str = Field(
        ...,
        pattern=_ALLELE_PATTERN,
        description="HLA allele in NetMHCpan format, e.g. HLA-A*02:01.",
        examples=["HLA-A*02:01"],
    )


class ScoreResponse(BaseModel):
    sequence: str
    allele: str
    immunogenicity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="SESTRAV immunogenicity probability [0, 1].",
    )
    binding_score: float | None = Field(
        None,
        description="MHCflurry presentation_score [0, 1]; None when MHCflurry unavailable.",
    )
    rank: str = Field(..., description="Qualitative rank: HIGH / MEDIUM / LOW.")
    model_version: str
    calibration_note: str = Field(
        (
            "immunogenicity_score is the raw RF predict_proba output. It is NOT passed "
            "through the isotonic calibrator (models/isotonic_calibrator.joblib) or the "
            "per-virus calibration path that the CLI (`sestrav predict`) and pipeline.py "
            "apply. The HIGH/MEDIUM/LOW thresholds (0.70 / 0.40) are applied to this "
            "uncalibrated score, so the same peptide can rank differently here than "
            "through the CLI."
        ),
        description="Discloses that this endpoint returns uncalibrated probabilities (NEW-CAL).",
    )


class ModelCard(BaseModel):
    name: str
    version: str
    feature_mode: str
    training_dataset: str
    cv_folds: int
    contamination_disclosure: str


class ProvenanceInfo(BaseModel):
    dataset_sha256: str | None
    zenodo_doi: str | None
    checksum_manifest: dict[str, Any]


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

from src.core.config import SestravConfig
from src.core.model_registry import ModelRegistry


class ModelManager:
    """Loads and holds the RF model and feature config exactly once."""

    _instance: "ModelManager | None" = None
    _loaded: bool = False

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._loaded = False
            cls._instance = obj
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        import sys

        # Ensure the project root is importable as a package source
        project_root = str(_PROJECT_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Initialize config and registry
        self.config = SestravConfig.load()
        self.registry = ModelRegistry(self.config)

        logger.info("Loading RF model ...")
        # Load verified model from registry - the canonical model_path from config.yaml
        # (rf_31feature_integrated.joblib for the v5 production scorer).
        self.rf_model = self.registry.load(self.config.model_path.name)
        logger.info("RF model loaded successfully.")

        # Load feature config for column ordering
        feat_config_path = self.registry.resolve_model("feature_config.json")
        if feat_config_path.exists():
            with feat_config_path.open("r", encoding="utf-8") as fh:
                self.feature_config: dict[str, Any] = json.load(fh)
        else:
            self.feature_config = {}

        self._loaded = True

    @property
    def is_ready(self) -> bool:
        return self._loaded


_manager = ModelManager()


# ---------------------------------------------------------------------------
# Helper: compute features and score
# ---------------------------------------------------------------------------


def _score_peptide(sequence: str, allele: str) -> tuple[float, float | None]:
    """Returns (immunogenicity_score, binding_score | None)."""
    import sys

    project_root = str(_PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.features import compute_features, FEATURE_COLUMNS_31

    # Compute physicochemical features (binding_score=0 when MHCflurry unavailable)
    binding_score: float | None = None
    try:
        from mhcflurry import Class1PresentationPredictor

        predictor = Class1PresentationPredictor.load()
        result = predictor.predict(
            peptides=[sequence],
            alleles=[allele],
            include_affinity_percentile=False,
        )
        binding_score = float(result["presentation_score"].iloc[0])
    except Exception as exc:
        logger.warning(f"MHCflurry unavailable: {exc}. Scoring without binding features.")

    feat_dict = compute_features(sequence, binding_score=binding_score or 0.0)

    # rf_31feature_integrated.joblib expects FEATURE_COLUMNS_31:
    # 20 physicochemical (p4-p8 x 4 properties) + 10 per-allele binding columns + peptide_length.
    # When MHCflurry is unavailable, binding columns default to 0.0.
    feature_vector = np.array(
        [feat_dict.get(col, 0.0) for col in FEATURE_COLUMNS_31],
        dtype=np.float64,
    ).reshape(1, -1)

    proba = float(_manager.rf_model.predict_proba(feature_vector)[0, 1])
    return proba, binding_score


def _rank_label(score: float) -> str:
    """Buckets a RAW (uncalibrated) predict_proba score. Unlike the CLI/pipeline
    path, this endpoint never applies models/isotonic_calibrator.joblib or a
    per-virus calibrator - see ScoreResponse.calibration_note (NEW-CAL)."""
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# App lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    _manager.load()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SESTRAV 2.0 Immunogenicity Scoring API",
    version=_APP_VERSION,
    description=(
        "T-cell epitope immunogenicity prediction pipeline. "
        "**Research use only - not for clinical decision-making.** "
        "See /model-card for contamination disclosure."
    ),
    lifespan=lifespan,
)


@app.get("/health", tags=["Operations"])
def health_check():
    return {"status": "healthy", "model_loaded": _manager.is_ready}


@app.post(
    "/score",
    response_model=ScoreResponse,
    tags=["Prediction"],
    summary="Score a single peptide-allele pair for immunogenicity.",
)
def score_peptide(body: PeptideInput) -> ScoreResponse:
    if not _manager.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check /health.",
        )
    try:
        imm_score, bind_score = _score_peptide(body.sequence, body.allele)
    except Exception as exc:
        logger.error(f"Scoring error for {body.sequence}/{body.allele}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feature extraction failed. Check server logs.",
        ) from exc

    return ScoreResponse(
        sequence=body.sequence,
        allele=body.allele,
        immunogenicity_score=round(imm_score, 4),
        binding_score=round(bind_score, 4) if bind_score is not None else None,
        rank=_rank_label(imm_score),
        model_version="rf_31feature_integrated",
    )


@app.get(
    "/model-card",
    response_model=ModelCard,
    tags=["Metadata"],
    summary="Model card: version, training details, and contamination disclosure.",
)
def model_card() -> ModelCard:
    return ModelCard(
        name="SESTRAV Random Forest",
        version=_APP_VERSION,
        feature_mode="31-feature integrated (20 physicochemical + 10 per-allele MHCflurry + peptide_length)",
        training_dataset=(
            "immunogenicity_dataset_v5.csv (IEDB + VDJdb; 35,597 active rows / 51,185 total). "
            "The 5,000 self-proteome central-tolerance decoys in the file are quarantined and "
            "absent from this model's training pool (docs/claims_register.md D19)."
        ),
        cv_folds=5,
        contamination_disclosure=(
            "SARS-CoV-2 and Influenza A are among the nine viruses this model is TRAINED on; "
            "they are not a held-out cohort. Only 16 named gold-standard epitopes are excluded "
            "from the training manifold. Cross-validation folds ARE peptide-grouped as of "
            "2026-08-10, so no peptide appears on both sides of a fold boundary: certified "
            "pooled AUC-PR 0.6058, per-virus within-CV mean AUC-ROC 0.658. Cross-validation "
            "metrics published before that date were computed under an ungrouped splitter in "
            "which 71.1% of held-out rows shared their exact peptide with the training fold, "
            "and are retracted as inflated (pooled AUC-PR 0.8347 ungrouped vs 0.6092 grouped). "
            "See docs/claims_register.md D15 for the full remediation record."
        ),
    )


@app.get(
    "/provenance",
    response_model=ProvenanceInfo,
    tags=["Metadata"],
    summary="Dataset checksums and Zenodo DOI for reproducibility.",
)
def provenance() -> ProvenanceInfo:
    manifest: dict[str, Any] = {}
    if _CHECKSUM_FILE.exists():
        with _CHECKSUM_FILE.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

    dataset_sha: str | None = None
    # Point to the actual training dataset (project root), not the expansion v3 file.
    dataset_path = _PROJECT_ROOT / "immunogenicity_dataset.csv"
    if dataset_path.exists():
        from src.artifact_integrity import sha256_file

        dataset_sha = sha256_file(dataset_path)

    return ProvenanceInfo(
        dataset_sha256=dataset_sha,
        zenodo_doi=_ZENODO_DOI,
        checksum_manifest=manifest,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
