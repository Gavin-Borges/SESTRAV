import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    try:
        from mhcflurry import Class1PresentationPredictor
        
        # Make sure our source tree is in path
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            
        from src.core.config import SestravConfig
        from src.core.model_registry import ModelRegistry

        # Test MHCflurry
        predictor = Class1PresentationPredictor.load()

        # Test Config loading
        config = SestravConfig.load(project_root / "config.yaml")
        registry = ModelRegistry(config)
        
        # Test model loading
        model = registry.load(config.model_path.name)
        assert model.n_features_in_ in (21, 30, 50), f"Model expects unsupported feature count: {model.n_features_in_}"

        print("Pre-flight checks PASSED")
        print(f"  Model name: {config.model_path.name}")
        print(f"  Model features: {model.n_features_in_}")
        print(f"  Antigens: {config.antigens}")
        print(f"  Alleles: {len(config.alleles)} alleles")
        print("  MHCflurry loaded OK")
        
    except Exception as e:
        logging.error(f"FATAL: Dependency verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
