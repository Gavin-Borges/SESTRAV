import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_SCRIPT = REPO_ROOT / "src" / "data_curation_qc.py"
# Literal on purpose: a module-level import of the constant would turn a
# missing constant into a collection error instead of a failing test.
ALLELE_COLUMN_NAMES = ("hla_allele", "allele", "mhc_allele")
PEP = "ACDEFGHIK"


@pytest.fixture
def temp_dataset(tmp_path):
    # Create a valid dataset
    df = pd.DataFrame({"peptide": ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH"], "label": [1, 0, 1]})
    path = tmp_path / "valid_dataset.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def temp_config(tmp_path):
    # Create a dummy config
    config_content = """
freeze_mode: false
dataset_governance:
  require_checksum_match_in_freeze_mode: true
  provenance:
    checksum: "pending"
"""
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        f.write(config_content)
    return path


def test_qc_script_valid(temp_dataset, temp_config):
    # Run the QC script as a subprocess
    result = subprocess.run(
        [
            sys.executable,
            "src/data_curation_qc.py",
            "--check-dataset",
            str(temp_dataset),
            "--config",
            str(temp_config),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"QC script failed unexpectedly: {result.stderr}"
    assert (
        "All strict dataset QC gates passed successfully" in result.stderr
        or "All strict dataset QC gates passed successfully" in result.stdout
    )


def test_qc_script_invalid_amino_acids(tmp_path, temp_config):
    df = pd.DataFrame(
        {
            "peptide": ["ACDEFGHIK", "XYZPQRSTV"],  # X, Y, Z (X and Z are invalid)
            "label": [1, 0],
        }
    )
    path = tmp_path / "invalid_aa_dataset.csv"
    df.to_csv(path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "src/data_curation_qc.py",
            "--check-dataset",
            str(path),
            "--config",
            str(temp_config),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Found 1 peptides with non-canonical amino acids" in result.stderr


def test_qc_script_duplicates(tmp_path, temp_config):
    df = pd.DataFrame({"peptide": ["ACDEFGHIK", "ACDEFGHIK"], "label": [1, 1]})
    path = tmp_path / "dup_dataset.csv"
    df.to_csv(path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "src/data_curation_qc.py",
            "--check-dataset",
            str(path),
            "--config",
            str(temp_config),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Dataset contains identical peptide-label duplicates" in result.stderr


def test_qc_script_conflicting_labels(tmp_path, temp_config):
    df = pd.DataFrame({"peptide": ["ACDEFGHIK", "ACDEFGHIK"], "label": [1, 0]})
    path = tmp_path / "conflict_dataset.csv"
    df.to_csv(path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "src/data_curation_qc.py",
            "--check-dataset",
            str(path),
            "--config",
            str(temp_config),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Dataset contains conflicting labels" in result.stderr


def test_qc_script_freeze_mode_missing_checksum_fails(tmp_path, temp_dataset):
    config = tmp_path / "missing_checksum.yaml"
    config.write_text(
        """
freeze_mode: true
dataset_governance:
  require_checksum_match_in_freeze_mode: true
  provenance: {}
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "src/data_curation_qc.py",
            "--check-dataset",
            str(temp_dataset),
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "checksum pin is absent" in result.stdout + result.stderr


# --- Allele-aware duplicate and conflict keys --------------------------------
#
# The cases below run the script with cwd=tmp_path and absolute paths, so the
# results/qc/dataset_qc.json it writes on success lands under tmp_path and never
# in the checkout. (The older cases above run with the caller's cwd, so
# test_qc_script_valid writes that file into whatever directory pytest was
# launched from.)


def _run_check(dataset, config, cwd):
    return subprocess.run(
        [sys.executable, str(QC_SCRIPT), "--check-dataset", str(dataset), "--config", str(config)],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def _write_allele_csv(tmp_path, rows, allele_col="hla_allele"):
    path = tmp_path / "allele_dataset.csv"
    pd.DataFrame(rows, columns=["peptide", allele_col, "label"]).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("allele_col", ALLELE_COLUMN_NAMES)
@pytest.mark.parametrize(
    "rows",
    [
        # One peptide, two alleles, same label: a peptide-label duplicate when
        # keyed on the peptide alone.
        [(PEP, "HLA-A*02:01", 1), (PEP, "HLA-B*07:02", 1)],
        # One peptide, two alleles, different labels: allele-specific
        # immunogenicity, a "conflict" when keyed on the peptide alone.
        [(PEP, "HLA-A*02:01", 1), (PEP, "HLA-B*07:02", 0)],
    ],
    ids=["same_label_two_alleles", "different_label_two_alleles"],
)
def test_allele_aware_same_peptide_under_two_alleles_passes(
    tmp_path, temp_config, rows, allele_col
):
    dataset = _write_allele_csv(tmp_path, rows, allele_col)
    result = _run_check(dataset, temp_config, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All strict dataset QC gates passed successfully" in result.stdout + result.stderr
    verdict = json.loads((tmp_path / "results" / "qc" / "dataset_qc.json").read_text())
    assert verdict["status"] == "PASSED"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [(PEP, "HLA-A*02:01", 1), (PEP, "HLA-A*02:01", 1)],
            "identical peptide-allele-label duplicates",
        ),
        (
            # Surrounding whitespace does not make a second allele.
            [(PEP, " HLA-A*02:01 ", 1), (PEP, "HLA-A*02:01", 1)],
            "identical peptide-allele-label duplicates",
        ),
        (
            [(PEP, "HLA-A*02:01", 1), (PEP, "HLA-A*02:01", 0)],
            "conflicting labels for the same peptide and allele",
        ),
        (
            # A blank cell and a sentinel both mean "no allele", so they share
            # one key; groupby on the raw column would drop the NaN row and miss
            # this conflict.
            [(PEP, None, 1), (PEP, "unknown", 0)],
            "conflicting labels for the same peptide and allele",
        ),
    ],
    ids=["duplicate", "duplicate_after_strip", "conflict", "conflict_missing_alleles"],
)
def test_allele_aware_true_duplicate_or_conflict_still_fails(tmp_path, temp_config, rows, message):
    dataset = _write_allele_csv(tmp_path, rows)
    result = _run_check(dataset, temp_config, cwd=tmp_path)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "results" / "qc" / "dataset_qc.json").exists()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_allele_conventions_match_data_qc_gate():
    curation = _load("data_curation_qc_under_test", QC_SCRIPT)
    gate = _load("data_qc_gate_under_test", REPO_ROOT / "scripts" / "data_qc_gate.py")
    assert tuple(curation.ALLELE_COL_PRIORITY) == tuple(gate.ALLELE_COL_PRIORITY)
    assert set(curation.NULL_ALLELE_TOKENS) == set(gate.NULL_ALLELE_TOKENS)


def test_shipped_training_dataset_passes_the_qc_dataset_rule(tmp_path):
    """The pipeline's qc_dataset command, on the dataset and config the repo ships.

    CI runs Snakemake only with --dry-run, so the rule's command never executes
    there; this runs the same script and arguments, with absolute paths and
    cwd=tmp_path so the success file is written under tmp_path.
    """
    config_path = REPO_ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = REPO_ROOT / config["training_dataset"]
    result = _run_check(dataset, config_path, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    verdict = json.loads((tmp_path / "results" / "qc" / "dataset_qc.json").read_text())
    assert verdict["status"] == "PASSED"
    assert verdict["checksum"] == config["dataset_governance"]["provenance"]["checksum"]
    rows = len(pd.read_csv(dataset, usecols=["peptide"]))
    assert verdict["positives"] + verdict["negatives"] == rows
