import hashlib
from pathlib import Path
from typing import Any
from src.core.config import SestravConfig

class ModelRegistry:
    """Registry to handle model artifact resolution, signature validation, and loading."""
    
    def __init__(self, config: SestravConfig):
        self.config = config

    def resolve_model(self, model_name: str) -> Path:
        """Resolve a model name to its absolute path."""
        p = Path("models") / model_name
        return p.resolve()

    def validate_signature(self, model_path: Path, expected_features: int) -> bool:
        """Validate if a model's expected features match our configuration."""
        import joblib
        if model_path.suffix == ".joblib":
            try:
                model = joblib.load(model_path)
                n_features = getattr(model, "n_features_in_", None)
                if n_features is not None and n_features != expected_features:
                    return False
            except Exception:
                pass
        return True

    def artifact_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of an artifact."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def load(self, model_name: str) -> Any:
        """Load a model by name."""
        path = self.resolve_model(model_name)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
            
        if path.suffix == ".joblib":
            import joblib
            return joblib.load(path)
        elif path.suffix in [".pt", ".pth"]:
            import torch
            try:
                import numpy as np
                import torch.serialization
                torch.serialization.add_safe_globals([
                    np._core.multiarray.scalar,  # type: ignore[attr-defined]
                    np.dtype,
                ])
                return torch.load(path, map_location='cpu', weights_only=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load torch model: {e}") from e
        else:
            raise ValueError(f"Unsupported model extension: {path.suffix}")
