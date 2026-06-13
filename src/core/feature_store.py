import pandas as pd
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)

class FeatureStore:
    """Manages the lifecycle of SESTRAV datasets and features."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def save_dataset(self, df: pd.DataFrame, filename: str) -> Path:
        out_path = self.output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved dataset to {out_path} ({len(df)} records)")
        return out_path

    def load_dataset(self, filename: str) -> pd.DataFrame:
        path = self.output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        return pd.read_csv(path)

    def save_features(self, df: pd.DataFrame, filename: str) -> Path:
        out_path = self.output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved features to {out_path}")
        return out_path

    def verify_integrity(self, path: Path, expected_checksum: str) -> bool:
        """Verify the SHA256 checksum of a dataset or feature file."""
        if not path.exists():
            return False
            
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
                
        actual = sha256.hexdigest()
        return actual == expected_checksum

    def load_cached_features(self, cache_name: str) -> pd.DataFrame | None:
        """Load pre-computed features from output cache to speed up development."""
        cache_path = self.output_dir / "cache" / cache_name
        if cache_path.exists():
            logger.info(f"Loading cached features from {cache_path}")
            return pd.read_csv(cache_path)
        return None

    def save_cached_features(self, df: pd.DataFrame, cache_name: str) -> Path:
        """Save computed features to output cache for reuse."""
        cache_path = self.output_dir / "cache" / cache_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached features written to {cache_path}")
        return cache_path

