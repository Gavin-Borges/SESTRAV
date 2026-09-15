import pickle
from pathlib import Path
from typing import Any
from src.core.config import SestravConfig
from src.artifact_integrity import (
    ArtifactIntegrityError,
    load_verified_joblib,
    sha256_file,
)

# Anchored to the installed package, NOT to the current working directory.
# `Path("models")` resolves against os.getcwd(), so every model lookup silently
# depended on where the process happened to start: `uvicorn api.main:app` from any
# directory but the repo root raised FileNotFoundError at startup, and in the API image
# that was masked only by `WORKDIR /app` in Dockerfile.api. It also made the confinement
# check below depend on cwd, which is the wrong property for a security boundary.
# parents[2] walks src/core -> src -> project root.
#
# KNOWN TRADEOFF, recorded deliberately: this exchanges a cwd dependency for an
# INSTALL-LOCATION dependency. pyproject.toml includes `src*` in packages.find, so under
# a NON-EDITABLE wheel install `src/` lands in site-packages and parents[2] points at a
# directory with no models/ - a case where the old cwd-relative code would have worked
# and this does not. No shipped path regresses: the API image copies the project to /app
# with WORKDIR /app, and development uses an editable install, so parents[2] is the
# project root in both. If a non-editable install is ever supported, this must become a
# configurable root (env var or packaged data dir) rather than reverting to cwd, which
# would restore the startup bug AND make the confinement check below cwd-dependent again.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _PROJECT_ROOT / "models"


class ModelRegistry:
    """Registry to handle model artifact resolution, signature validation, and loading."""

    def __init__(self, config: SestravConfig):
        self.config = config

    def resolve_model(self, model_name: str) -> Path:
        """Resolve a model name to its absolute path, confined to the models/ directory."""
        base = MODELS_DIR.resolve()
        p = (base / model_name).resolve()
        if not p.is_relative_to(base):
            raise ValueError(f"Model name escapes models/ directory: {model_name!r}")
        return p

    def validate_signature(self, model_path: Path, expected_features: int) -> bool:
        """Validate a model's checksum and that its feature count matches configuration.

        Fail-closed: the artifact must pass checksum verification (required_checksum=True)
        and, if it exposes n_features_in_, match expected_features. Any verification failure
        - a missing or mismatched checksum, or an unreadable/corrupt artifact - returns False
        rather than being silently accepted.
        """
        if model_path.suffix == ".joblib":
            try:
                model = load_verified_joblib(model_path, required_checksum=True)
            except (
                # load_verified_joblib is verify_artifact_checksum followed by
                # joblib.load, so its raise surface is determinable. Same annotated
                # form as the torch.load tuple below. A programming error is NOT in
                # this set on purpose: the docstring's contract is that a
                # VERIFICATION failure returns False, so a bug should surface as a
                # traceback rather than be reported as an invalid signature.
                ArtifactIntegrityError,  # missing or mismatched checksum manifest
                OSError,  # unreadable path: FileNotFoundError, PermissionError
                EOFError,  # truncated or empty pickle stream (not an OSError)
                pickle.UnpicklingError,  # corrupt or non-pickle payload
                ValueError,  # malformed buffer or unsupported protocol
                ImportError,  # pickled class whose module no longer exists
                AttributeError,  # pickled class gone from a module that still exists
            ):  # unverifiable or unreadable artifact -> invalid
                return False
            n_features = getattr(model, "n_features_in_", None)
            if n_features is not None and n_features != expected_features:
                return False
        return True

    def artifact_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of an artifact."""
        return sha256_file(path)

    def load(self, model_name: str) -> Any:
        """Load a model by name."""
        path = self.resolve_model(model_name)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        if path.suffix == ".joblib":
            return load_verified_joblib(path, required_checksum=True)
        elif path.suffix in [".pt", ".pth"]:
            import torch

            try:
                import numpy as np
                import torch.serialization

                torch.serialization.add_safe_globals(
                    [
                        getattr(getattr(getattr(np, "_core"), "multiarray"), "scalar"),
                        np.dtype,
                    ]
                )
                return torch.load(path, map_location="cpu", weights_only=True)
            except (
                # Measured or read at torch 2.13.0+cu130 / numpy 2.4.6. Every member has a
                # raise site on this exact call path; see the branch commit body for evidence.
                ImportError,  # `import numpy` / `import torch.serialization` above
                AttributeError,  # numpy internal rename breaks the np._core getattr chain
                TypeError,  # malformed safe-globals entry, torch/_weights_only_unpickler.py
                ValueError,  # unpickler operand, or unknown byteorder record in the archive
                OSError,  # unreadable path: FileNotFoundError, PermissionError
                EOFError,  # empty or truncated stream (NOT an OSError subclass)
                RuntimeError,  # PytorchStreamReader failure, unknown map_location
                pickle.UnpicklingError,  # weights_only refusal (NOT an OSError subclass)
            ) as e:
                raise RuntimeError(f"Failed to load torch model: {e}") from e
        else:
            raise ValueError(f"Unsupported model extension: {path.suffix}")
