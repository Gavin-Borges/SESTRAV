import yaml
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel, field_validator


class ProvenanceConfig(BaseModel):
    timestamp: str
    source_databases: List[str]
    negative_sampling_strategy: str
    checksum: str


class QCThresholdsConfig(BaseModel):
    min_peptide_yield: int
    max_conflict_ratio: float
    max_null_allele_fraction: float
    class_ratio_bounds: List[float]


class DatasetGovernanceConfig(BaseModel):
    current_version: str
    versions: Dict[str, str]
    provenance: ProvenanceConfig
    qc_thresholds: QCThresholdsConfig
    require_checksum_match_in_freeze_mode: bool


class SestravConfig(BaseModel):
    antigens: List[str]
    proteome_files: Dict[str, str]
    alleles: List[str]
    peptide_lengths: List[int]
    binding_backend: str
    feature_mode: int
    model_path: Path
    freeze_mode: bool
    calibration_path: Optional[Path] = None
    thresholds_path: Optional[Path] = None
    binding_matrix_path: Optional[Path] = None
    mc_dropout: bool = False
    dataset_governance: DatasetGovernanceConfig
    dataset_mode: str
    dataset_version: str
    output_dir: Path
    mock_ingestion: bool = False
    mock_evaluation: bool = False
    gnn_checkpoint: Optional[Path] = None
    max_peptide_length: int = 11
    structural_cache_dir: Optional[Path] = Path("data/structural_cache")
    use_spatial_adj: bool = False

    @field_validator(
        "model_path", "binding_matrix_path", "output_dir", "structural_cache_dir", mode="before"
    )
    def validate_paths(cls, v):
        if v is None:
            return v
        return Path(v)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "SestravConfig":
        """Load config from `path`. NOTE: the default is CWD-RELATIVE.

        A bare `SestravConfig.load()` therefore only works when the process starts in the
        project root. This was the FIRST failure in API startup from a foreign cwd - it
        runs before any model lookup, so anchoring the model registry alone did not make
        startup cwd-independent. `api/main.py` fixes it by passing an explicit
        `__file__`-anchored path rather than relying on this default.

        The default is left cwd-relative ON PURPOSE. Several callers still use the bare
        form (src/generate_dataset_v3.py, scripts/batch_experiment_runner.py, pipeline.py)
        and at least the batch runner may legitimately want the config of whatever
        directory it is invoked from. Changing this default would silently alter which
        file those callers read, so it is a separate decision, not a drive-by fix.
        New callers should pass an anchored path, as `src/cli.py` and `api/main.py` do.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
