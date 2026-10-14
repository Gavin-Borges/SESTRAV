"""Every Stage 4 entry point calls on the threshold picked on the peptide-grouped OOF.

`models/optimal_thresholds.json` (0.325) was picked on the pre-D15 ungrouped OOF and
`models/v5/optimal_thresholds.json` (0.3291) on the grouped one; both describe the same
model bytes. config.yaml named the first, pipeline.py was the only entry point that read
the key at all, and the Snakemake rule and `sestrav predict` resolved the file beside the
model, which is the first one too. These pin the configured file and that each entry
point forwards it.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import yaml

from src import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_config_names_the_threshold_picked_on_the_grouped_oof():
    path = CONFIG["thresholds_path"]
    assert path == "models/v5/optimal_thresholds.json"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert tracked.returncode == 0, f"{path} is not tracked, so a clone would not receive it"
    payload = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    # It was picked for the configured model: same feature mode, same estimator.
    assert payload["feature_mode"] == CONFIG["feature_mode"]
    assert payload["method"] == "RandomForest"


def test_the_snakemake_rule_forwards_the_configured_thresholds():
    smk = (REPO_ROOT / "pipeline.smk").read_text(encoding="utf-8")
    rule = re.search(r"rule score_immunogenicity:\n(.*?)\n\n", smk, re.S)
    assert rule, "rule score_immunogenicity not found"
    assert 'thresholds_path = config.get("thresholds_path", "")' in rule.group(1)

    script = ast.parse((REPO_ROOT / "scripts" / "stage4.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(script)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "score_immunogenicity"
    ]
    assert len(calls) == 1
    assert "thresholds_path" in {keyword.arg for keyword in calls[0].keywords}


def test_predict_forwards_the_configured_thresholds_for_the_configured_model():
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    cmd_predict = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "cmd_predict"
    )
    call = next(
        node
        for node in ast.walk(cmd_predict)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "score_immunogenicity"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert ast.unparse(keywords["thresholds_path"]) == "_configured_thresholds_for(args.model)"


def _files(tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"model")
    other = tmp_path / "other.joblib"
    other.write_bytes(b"other")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"threshold": 0.33}))
    return model, other, thresholds


def test_configured_thresholds_apply_to_the_configured_model(tmp_path):
    model, _, thresholds = _files(tmp_path)
    cfg = {"model_path": str(model), "thresholds_path": str(thresholds)}
    assert Path(cli._configured_thresholds_for(str(model), cfg)) == thresholds


def test_another_model_keeps_the_thresholds_beside_it(tmp_path):
    model, other, thresholds = _files(tmp_path)
    cfg = {"model_path": str(model), "thresholds_path": str(thresholds)}
    assert cli._configured_thresholds_for(str(other), cfg) is None


def test_missing_files_or_keys_fall_back_rather_than_raise(tmp_path):
    model, _, thresholds = _files(tmp_path)
    assert cli._configured_thresholds_for(str(model), {}) is None
    missing = {"model_path": str(model), "thresholds_path": str(tmp_path / "absent.json")}
    assert cli._configured_thresholds_for(str(model), missing) is None
    gone = {"model_path": str(tmp_path / "absent.joblib"), "thresholds_path": str(thresholds)}
    assert cli._configured_thresholds_for(str(model), gone) is None


def test_configured_paths_resolve_against_the_installation_not_the_cwd(tmp_path, monkeypatch):
    """config.yaml's relative paths are anchored to the installation, as _read_config is.

    The tests above pass absolute paths, which joining onto any root leaves unchanged,
    so resolving the configured paths against the working directory passed them all.
    Here the configured paths are relative and the working directory holds neither.
    A tracked file stands in for the model, since only file identity is compared.
    """
    install_root = Path(cli.__file__).resolve().parents[1]
    stand_in = "config.yaml"
    cfg = {"model_path": stand_in, "thresholds_path": CONFIG["thresholds_path"]}
    monkeypatch.chdir(tmp_path)
    resolved = cli._configured_thresholds_for(str(install_root / stand_in), cfg)
    assert resolved is not None, "relative configured paths did not resolve under the install"
    assert Path(resolved).resolve() == (install_root / CONFIG["thresholds_path"]).resolve()
