"""
Generate reproducibility_manifest.json containing Git, Conda env, dataset DOIs, and pipeline hashes.
"""
import os
import json
import subprocess
import hashlib
import yaml

def get_git_revision_hash() -> str:
    try:
        # Run git command to get the commit hash
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "unknown"

def get_file_sha256(filepath: str) -> str:
    if not os.path.isfile(filepath):
        return "not_found"
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = "../config.yaml"
        
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
    manifest = {
        "git_commit": get_git_revision_hash(),
        "environment_hashes": {
            "conda_environment_yml": get_file_sha256("environment.yml"),
            "conda_prime_yaml": get_file_sha256("environments/prime.yaml"),
            "conda_predig_yaml": get_file_sha256("environments/predig.yaml"),
            "requirements_txt": get_file_sha256("requirements.txt")
        },
        "dataset_governance": config.get("dataset_governance", {}),
        "pipeline_definition_hashes": {
            "Snakefile": get_file_sha256("Snakefile"),
            "pipeline_smk": get_file_sha256("pipeline.smk"),
            "standardize_outputs_smk": get_file_sha256("standardize_outputs.smk"),
            "config_yaml": get_file_sha256(config_path)
        }
    }
    
    # Save manifest
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "reproducibility_manifest.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Reproducibility manifest generated successfully at {out_path}")

if __name__ == "__main__":
    main()
