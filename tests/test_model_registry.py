import pytest
from pathlib import Path
from src.core.config import SestravConfig
from src.core.model_registry import ModelRegistry

def test_model_registry_resolution(tmp_path):
    # Mock a config
    config = SestravConfig.model_construct(output_dir=tmp_path)
    registry = ModelRegistry(config)
    
    # Should resolve to the default models dir
    resolved = registry.resolve_model("some_model.joblib")
    assert resolved.name == "some_model.joblib"
    assert "models" in resolved.parts

def test_model_registry_missing_model(tmp_path):
    config = SestravConfig.model_construct(output_dir=tmp_path)
    registry = ModelRegistry(config)
    
    with pytest.raises(FileNotFoundError):
        registry.load("nonexistent_model.joblib")
