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
# Log hygiene (CWE-117 log injection; CodeQL py/log-injection)
# ---------------------------------------------------------------------------
# This module is the project's only HTTP entrypoint, so every request-derived
# value that reaches a log record reaches it from here. The record format is
# "%(asctime)s [%(levelname)s] %(message)s" - one line per record - so a CR or
# LF inside an interpolated field lets a caller append a second, fully-formed
# line to the log. That is how a fabricated ERROR gets into an audit trail, or
# how a real one gets pushed out of view.
#
# On reachability, honestly: this is NOT exploitable today. Both request fields
# carry anchored Pydantic patterns (_IUPAC_PATTERN, _ALLELE_PATTERN below), and
# pydantic 2.13.4 compiles them with the Rust regex engine, which rejected every
# CR/LF payload tested against them. But that protection is incidental rather
# than designed, and it is engine-dependent: the identical anchored pattern
# under Python's own `re.match` ACCEPTS a trailing newline (`re.fullmatch` does
# not). Widening _ALLELE_PATTERN to admit HLA-C or three-field alleles like
# 02:01:01 is a routine change that would not obviously look like re-opening a
# log-forging hole. This sanitizer exists so that day stays a non-event.
#
# Truncation is a second, independent control: it bounds one record so an
# oversized field cannot flood the log or push older lines out of retention.

_LOG_FIELD_MAX_CHARS = 200


def _sanitize_for_log(value: object) -> str:
    """Renders a request-derived value safe to interpolate into a log record.

    Strips CRLF, bare LF and bare CR, then truncates. The bare CR is deliberate:
    CodeQL's published remediation for this rule strips only "\\r\\n" and "\\n",
    yet a lone CR still opens a new line in many log viewers and terminals.
    """
    text = str(value).replace("\r\n", "").replace("\n", "").replace("\r", "")
    if len(text) > _LOG_FIELD_MAX_CHARS:
        text = text[:_LOG_FIELD_MAX_CHARS] + "...[truncated]"
    return text


# ---------------------------------------------------------------------------
# Paths (relative to project root; mount the repo root as the working dir)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def _registry_model_name(configured_path: Path) -> str:
    """Return a models/-relative name while preserving nested directories."""
    models_dir = (_PROJECT_ROOT / "models").resolve()
    if configured_path.is_absolute():
        try:
            return str(configured_path.resolve().relative_to(models_dir))
        except ValueError as exc:
            raise ValueError(
                f"Configured model path escapes models/ directory: {configured_path!s}"
            ) from exc

    try:
        return str(configured_path.relative_to("models"))
    except ValueError:
        return str(configured_path)


class ModelManager:
    """Loads and holds the RF model and feature config exactly once."""

    _instance: "ModelManager | None" = None
    _loaded: bool = False
    _configured_model_path: Path | None = None
    _load_error: str | None = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._loaded = False
            obj._configured_model_path = None
            obj._load_error = None
            cls._instance = obj
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        self._configured_model_path = None
        self._load_error = None
        import sys

        # Ensure the project root is importable as a package source
        project_root = str(_PROJECT_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Initialize config and registry. _CONFIG_PATH, not the bare SestravConfig.load(),
        # whose default "config.yaml" resolves against the current working directory: this
        # call runs BEFORE the model load below, so a wrong cwd failed here first and the
        # registry fix alone would not have made startup cwd-independent.
        self.config = SestravConfig.load(_CONFIG_PATH)
        self._configured_model_path = self.config.model_path
        self.registry = ModelRegistry(self.config)

        logger.info("Loading RF model ...")
        # Load verified model from registry - the canonical model_path from config.yaml
        # (rf_31feature_integrated.joblib for the v5 production scorer).
        self.rf_model = self.registry.load(_registry_model_name(self.config.model_path))
        logger.info("RF model loaded successfully.")

        # Load feature config for column ordering
        feat_config_path = self.registry.resolve_model("feature_config.json")
        if feat_config_path.exists():
            with feat_config_path.open("r", encoding="utf-8") as fh:
                self.feature_config: dict[str, Any] = json.load(fh)
        else:
            self.feature_config = {}

        # Load the ten-allele binding panel once (D31 fix). This is the ONLY
        # mechanism that populates the model's bind_* feature block: the panel
        # is FIXED (docs/paper.md Section 2.2, "Ten MHC binding affinity features"), so this is a lookup keyed by
        # peptide alone, not by the caller's requested allele.
        logger.info("Loading binding panel matrix ...")
        binding_matrix_path = _PROJECT_ROOT / "models" / "peptide_binding_matrix_v5.csv"
        import pandas as pd

        binding_df = pd.read_csv(binding_matrix_path)
        from src.features import BINDING_ALLELE_COLUMNS

        self.binding_matrix: dict[str, dict[str, float]] = (
            binding_df.set_index("peptide")[BINDING_ALLELE_COLUMNS].to_dict(orient="index")
        )
        logger.info(f"Binding panel matrix loaded: {len(self.binding_matrix)} peptides.")

        self._loaded = True

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @property
    def configured_model_path(self) -> str | None:
        if self._configured_model_path is None:
            return None
        return self._configured_model_path.as_posix()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def record_load_failure(self, exc: Exception) -> None:
        self._loaded = False
        path = self.configured_model_path
        if path is None:
            self._load_error = f"{type(exc).__name__} before the model path was loaded"
        else:
            self._load_error = (
                f"{type(exc).__name__} while loading configured model path {path!r}"
            )


_manager = ModelManager()


class PeptideNotInPanelError(Exception):
    """Raised when a peptide has no row in the ten-allele binding panel.

    docs/claims_register.md D31: the panel is a FIXED ten-allele set and the
    bind_* features are defined as a precomputed per-peptide lookup over it
    (docs/paper.md Section 2.2, "Ten MHC binding affinity features"), with
    peptides absent from the matrix receiving zero. Populating the block by
    any other route - broadcasting one live MHCflurry call across the ten
    columns, or one-hotting the requested allele - would fabricate values the
    model was never trained on, so a panel miss cannot be repaired at request
    time and is reported explicitly instead of silently zero-filled, which
    would be indistinguishable from a genuine all-low-affinity panel match.

    Note this does NOT rest on MHCflurry being unavailable: it is declared in
    [project].dependencies, so `pip install ".[api]"` does install it. What
    Dockerfile.api never does is fetch its model weights
    (`mhcflurry-downloads fetch`), which is why the informational call in
    _score_peptide is still expected to fail in the shipped container.
    """


# ---------------------------------------------------------------------------
# Helper: compute features and score
# ---------------------------------------------------------------------------


def _score_peptide(sequence: str, allele: str) -> tuple[float, float | None]:
    """Returns (immunogenicity_score, binding_score | None).

    binding_score is a best-effort, informational MHCflurry call for the
    caller's specific allele - it is NOT what feeds the model. The model's
    bind_* feature block is always populated from the fixed ten-allele panel
    matrix (see PeptideNotInPanelError), independent of whether this call
    succeeds. Raises PeptideNotInPanelError if the peptide has no panel row.
    """
    import sys

    project_root = str(_PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.features import compute_features, FEATURE_COLUMNS_31

    panel_row = _manager.binding_matrix.get(sequence)
    if panel_row is None:
        raise PeptideNotInPanelError(sequence)

    # Best-effort informational value for ScoreResponse.binding_score. Failure
    # here does not affect the feature vector below.
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
        # Same CWE-117 path as the /score handler: `sequence` and `allele` are
        # this call's arguments, so a predictor error message can quote them
        # back into the record. Sanitized for the same reason.
        logger.info(
            "MHCflurry unavailable for the informational binding_score field: %s",
            _sanitize_for_log(exc),
        )

    feat_dict = compute_features(sequence, binding_score=0.0)
    feat_dict.update(panel_row)

    # rf_31feature_integrated.joblib expects FEATURE_COLUMNS_31:
    # 20 physicochemical (p4-p8 x 4 properties) + 10 per-allele binding columns
    # (from the fixed panel lookup above) + peptide_length.
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
    try:
        _manager.load()
    except Exception as exc:
        _manager.record_load_failure(exc)
        logger.error(
            "Model startup load failed; API is serving in degraded mode: %s",
            _sanitize_for_log(exc),
        )
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
    if _manager.is_ready:
        return {"status": "healthy", "model_loaded": True, "reason": None}
    return {
        "status": "degraded",
        "model_loaded": False,
        "reason": _manager.load_error or "Model has not been loaded.",
    }


@app.post(
    "/score",
    response_model=ScoreResponse,
    tags=["Prediction"],
    summary="Score a single peptide-allele pair for immunogenicity.",
)
def score_peptide(body: PeptideInput) -> ScoreResponse:
    if not _manager.is_ready:
        configured_path = _manager.configured_model_path
        path_detail = (
            f" at configured path {configured_path!r}" if configured_path is not None else ""
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model unavailable{path_detail}: "
                f"{_manager.load_error or 'model has not been loaded'}. Check /health."
            ),
        )
    try:
        imm_score, bind_score = _score_peptide(body.sequence, body.allele)
    except PeptideNotInPanelError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Peptide {body.sequence!r} has no row in the ten-allele binding panel "
                "(models/peptide_binding_matrix_v5.csv), so its immunogenicity score "
                "cannot be computed by this deployment. The panel is fixed and keyed by "
                "peptide, not by the requested allele, and the binding features are "
                "defined as a lookup over it, so an out-of-panel peptide has no "
                "substitute the model was trained on. See docs/claims_register.md D31."
            ),
        ) from exc
    except Exception as exc:
        # All three interpolated values are request-derived. body.sequence and
        # body.allele obviously so; `exc` less obviously, because it is raised
        # inside _score_peptide from the caller's own peptide and allele and can
        # therefore echo them back verbatim (PeptideNotInPanelError(sequence) is
        # the in-repo proof that an exception here carries the request payload as
        # its message). Sanitizing only the two named fields would leave the
        # taint path through `exc` open. Lazy %s args are the logging idiom and
        # are used here, but the newline stripping is the security control - %s
        # alone still renders a CR/LF straight into the record.
        logger.error(
            "Scoring error for %s/%s: %s",
            _sanitize_for_log(body.sequence),
            _sanitize_for_log(body.allele),
            _sanitize_for_log(exc),
        )
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
            "pooled AUC-PR 0.6055, per-virus within-CV mean AUC-ROC 0.658. Cross-validation "
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
