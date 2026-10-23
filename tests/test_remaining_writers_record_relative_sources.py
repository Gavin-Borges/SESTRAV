"""Three more writers record a caller-supplied path through relative_source.

Why this test exists
--------------------
The CHANGELOG entry that moved most provenance writers onto
`src.provenance_paths.relative_source` named, among the fields it left as
given, `base_input` and `oof_predictions` in src/external_benchmark_comparison.py,
`train_source` in src/external_validation_fairness.py and `peptide_list_path` in
scripts/prepare_netmhcpan_inputs.py. Each was written as the caller gave it, so
an absolute argument put a workstation path into the output:

- run_comparison's Markdown report (results/external_benchmark_comparison.md is
  tracked) lists `base_input`, `oof_predictions`, `predig_file` and `prime_file`
  in its Provenance section. `base_input` defaults to a file under
  `--results-dir`, so an absolute `--results-dir` alone was enough;
- run_fairness writes each training list's `train_source` into
  manifests/training_overlap.json and processed/fairness_analysis.md;
- prepare_inputs writes `peptide_list_path` into inputs/manifest.csv.

Each now records an absolute path outside the checkout as `external:<name>`, an
absolute path inside it repo-relative with "/", and a relative path exactly as
given, so the default arguments record what they always did. relative_source's
default root is pointed at a scratch checkout under tmp_path, as
tests/test_writers_record_no_workstation_path.py does, so every absolute path
here is built from tmp_path.

The NetMHCpan runner used to take each peptide list's file name from the
recorded path, which cannot name a file once it reads `external:<name>`; it now
names it from the group, and the round-trip tests check that every file the
runner and the manifest point at exists.

What is stubbed: run_comparison's merge, metric, bootstrap and figure steps and
its two score parsers, so only the provenance it records is exercised. The
fairness and NetMHCpan writers run unstubbed on tiny synthetic inputs.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.provenance_paths
from src import external_benchmark_comparison as ebc
from src.external_validation_fairness import quantify_overlap_robust, run_fairness
from src.provenance_paths import relative_source

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prepare_netmhcpan_inputs as prep  # noqa: E402

TRACKED_REPORT = REPO_ROOT / "results" / "external_benchmark_comparison.md"
COMPARISON_PATH_KEYS = ("base_input", "oof_predictions", "predig_file", "prime_file")
_PROVENANCE_LINE = re.compile(r"^- \*\*(?P<key>[^*]+):\*\* (?P<value>.*)$")


# --- shared helpers ------------------------------------------------------------


def _scratch_checkout(tmp_path: Path, monkeypatch) -> Path:
    """A directory relative_source treats as the checkout, beside an outside one."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.setattr(src.provenance_paths, "PROJECT_ROOT", checkout)
    return checkout


def _spellings(directory: Path) -> set[str]:
    forms = set()
    for form in (str(directory), os.path.realpath(directory)):
        forms.update({form, form.replace("\\", "/")})
    return forms


def _with_workstation_path(strings, directory: Path) -> list[str]:
    """The strings that are absolute paths or spell ``directory`` in any form."""
    spellings = _spellings(directory)
    return [s for s in strings if os.path.isabs(s) or any(sp in s for sp in spellings)]


def _json_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _json_strings(key)
            yield from _json_strings(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _json_strings(value)


def _write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- src/external_benchmark_comparison.py ------------------------------------


def _provenance(report_text: str) -> dict[str, str]:
    section = report_text.split("## Provenance", 1)[1].split("\n## ", 1)[0]
    found = (_PROVENANCE_LINE.match(line) for line in section.splitlines())
    return {m["key"]: m["value"] for m in found if m}


@pytest.fixture
def stubbed_comparison(monkeypatch):
    """run_comparison with everything but its provenance and report stubbed."""
    merged = pd.DataFrame({"peptide": ["SLYNTVATL"], "label": [1]})
    scores = pd.DataFrame({"peptide": ["SLYNTVATL"], "score": [0.5]})

    def empty(*_args, **_kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(ebc, "build_merged_table", lambda *a, **k: merged.copy())
    for name in (
        "evaluate_all_tools",
        "compute_issr_ratios",
        "spearman_correlation_matrix",
        "negative_discrimination",
        "per_virus_metrics",
        "run_bootstrap_comparisons",
    ):
        monkeypatch.setattr(ebc, name, empty)
    monkeypatch.setattr(ebc, "generate_figures", lambda *a, **k: [])
    monkeypatch.setattr(ebc, "parse_predig", lambda *a, **k: scores.copy())
    monkeypatch.setattr(ebc, "parse_prime", lambda *a, **k: scores.copy())

    def run(**kwargs) -> tuple[dict[str, str], str]:
        Path(kwargs["results_dir"]).mkdir(parents=True, exist_ok=True)
        report = ebc.run_comparison(n_bootstrap=1, **kwargs)
        text = Path(report).read_text(encoding="utf-8")
        return _provenance(text), text

    return run


def test_comparison_records_sources_outside_the_checkout_by_name_only(
    tmp_path, monkeypatch, stubbed_comparison
):
    _scratch_checkout(tmp_path, monkeypatch)
    outside = tmp_path / "elsewhere"
    predig = _write(outside / "predig.csv")
    prime = _write(outside / "prime.txt")

    # No --base-path: it defaults to a file under the absolute --results-dir.
    prov, text = stubbed_comparison(
        results_dir=str(outside / "results"),
        oof_path=str(outside / "oof.csv"),
        predig_path=str(predig),
        prime_path=str(prime),
    )

    assert prov["base_input"] == "external:external_validation_input.csv"
    assert prov["oof_predictions"] == "external:oof.csv"
    assert prov["predig_file"] == "external:predig.csv"
    assert prov["prime_file"] == "external:prime.txt"
    assert _with_workstation_path(text.splitlines(), tmp_path) == []


def test_comparison_records_sources_inside_the_checkout_repo_relative(
    tmp_path, monkeypatch, stubbed_comparison
):
    checkout = _scratch_checkout(tmp_path, monkeypatch)
    predig = _write(checkout / "results" / "external_tool_outputs" / "predig.csv")
    prime = _write(checkout / "results" / "external_tool_outputs" / "prime.txt")

    prov, text = stubbed_comparison(
        results_dir=str(tmp_path / "elsewhere" / "results"),
        base_path=str(checkout / "results" / "external_validation_input.csv"),
        oof_path=str(checkout / "models" / "rf_oof_predictions.csv"),
        predig_path=str(predig),
        prime_path=str(prime),
    )

    assert prov["base_input"] == "results/external_validation_input.csv"
    assert prov["oof_predictions"] == "models/rf_oof_predictions.csv"
    assert prov["predig_file"] == "results/external_tool_outputs/predig.csv"
    assert prov["prime_file"] == "results/external_tool_outputs/prime.txt"
    assert _with_workstation_path(text.splitlines(), tmp_path) == []


def test_comparison_records_relative_sources_exactly_as_given(
    tmp_path, monkeypatch, stubbed_comparison
):
    """The default --oof-path and a relative --results-dir, run from the directory
    they are relative to: each is recorded as the caller spelled it, as before."""
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "in" / "predig.csv")

    prov, _ = stubbed_comparison(results_dir="res", predig_path="in/predig.csv")

    assert prov["base_input"] == os.path.join("res", "external_validation_input.csv")
    assert prov["oof_predictions"] == "models/rf_oof_predictions.csv"
    assert prov["predig_file"] == "in/predig.csv"
    assert "prime_file" not in prov


def test_the_tracked_comparison_report_records_only_fixed_points():
    """Regenerating the tracked report from the arguments it was written with
    records the same path strings: each one is a fixed point of relative_source.
    Only key names are reported on failure."""
    prov = _provenance(TRACKED_REPORT.read_text(encoding="utf-8"))
    present = sorted(key for key in COMPARISON_PATH_KEYS if key in prov)
    assert {"base_input", "oof_predictions"} <= set(present)
    moved = [key for key in present if relative_source(prov[key]) != prov[key]]
    assert moved == []


# --- src/external_validation_fairness.py ---------------------------------------


def _training_list(path: Path, peptides) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Epitope": peptides}).to_csv(path, index=False)
    return str(path)


def test_fairness_records_a_training_list_outside_the_checkout_by_name_only(
    tmp_path, monkeypatch
):
    _scratch_checkout(tmp_path, monkeypatch)
    train = _training_list(tmp_path / "elsewhere" / "predig_train.csv", ["SLYNTVATL"])

    meta = quantify_overlap_robust({"SLYNTVATL", "GILGFVFTL"}, train)

    assert meta["train_source"] == "external:predig_train.csv"
    assert meta["exact_overlap_count"] == 1
    assert _with_workstation_path(_json_strings(meta), tmp_path) == []


def test_fairness_records_a_training_list_inside_the_checkout_repo_relative(
    tmp_path, monkeypatch
):
    checkout = _scratch_checkout(tmp_path, monkeypatch)
    train = _training_list(checkout / "data" / "external" / "predig_train_modf.csv", ["X"])

    meta = quantify_overlap_robust({"SLYNTVATL"}, train)

    assert meta["train_source"] == "data/external/predig_train_modf.csv"


def test_fairness_records_the_default_relative_training_list_as_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _training_list(tmp_path / "data" / "external" / "predig_train_modf.csv", ["X"])

    meta = quantify_overlap_robust({"SLYNTVATL"}, "data/external/predig_train_modf.csv")

    assert meta["train_source"] == "data/external/predig_train_modf.csv"


def test_run_fairness_writes_no_workstation_path_for_its_training_lists(tmp_path, monkeypatch):
    _scratch_checkout(tmp_path, monkeypatch)
    outside = tmp_path / "elsewhere"
    rng = np.random.RandomState(0)
    peptides = [f"A{i:08d}" for i in range(30)]
    label = np.tile([1, 0, 1], 10)
    merged = pd.DataFrame(
        {
            "peptide": peptides,
            "label": label,
            "virus": ["EBV", "HPV16"] * 15,
            "rf_oof_score": label + rng.uniform(0, 0.9, 30),
            "binding_max": rng.uniform(0, 1, 30),
            "predig_max_score": rng.uniform(0, 1, 30),
            "prime_score": rng.uniform(0, 1, 30),
        }
    )
    merged_path = outside / "merged.csv"
    merged.to_csv(merged_path, index=False)
    predig_train = _training_list(outside / "predig_train.csv", peptides[:5])
    prime_train = _training_list(outside / "prime_train.csv", peptides[5:8])
    run_dir = outside / "run"
    (run_dir / "manifests").mkdir(parents=True)  # run_fairness does not create it

    report = run_fairness(
        str(merged_path), str(run_dir), predig_train=predig_train, prime_train=prime_train
    )

    overlap = json.loads((run_dir / "manifests" / "training_overlap.json").read_text("utf-8"))
    assert overlap["predig"]["train_source"] == "external:predig_train.csv"
    assert overlap["prime"]["train_source"] == "external:prime_train.csv"
    assert _with_workstation_path(_json_strings(overlap), tmp_path) == []
    text = Path(report).read_text(encoding="utf-8")
    assert '"train_source": "external:predig_train.csv"' in text
    assert _with_workstation_path(text.splitlines(), tmp_path) == []


# --- scripts/prepare_netmhcpan_inputs.py ---------------------------------------

_HELD_OUT = "\n".join(
    [
        "peptide\thla_allele\tlabel",
        "SLYNTVATL\tHLA-A*02:01\t1",
        "GILGFVFTL\tHLA-B*07:02\t0",
        "AAAWYLWEV\tHLA-A\t1",  # unscorable allele: in the manifest, not the runner
    ]
)
_NAMES = ["TESTV__HLA-A.pep", "TESTV__HLA-A02_01.pep", "TESTV__HLA-B07_02.pep"]


def _held_out_dir(parent: Path) -> Path:
    _write(parent / "loo" / "TESTV_held_out.tsv", _HELD_OUT + "\n")
    return parent / "loo"


def _manifest(out_dir: Path) -> list[dict[str, str]]:
    with (out_dir / "inputs" / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _runner_inputs(out_dir: Path) -> list[str]:
    runner = (out_dir / "run_netmhcpan.sh").read_text(encoding="utf-8")
    return re.findall(r'^netMHCpan -p "([^"]+)"', runner, flags=re.MULTILINE)


def _assert_runner_finds_every_list(out_dir: Path) -> None:
    """The runner cds into out_dir, so each -p argument resolves there."""
    pep_args = _runner_inputs(out_dir)
    assert pep_args == ["inputs/TESTV__HLA-A02_01.pep", "inputs/TESTV__HLA-B07_02.pep"]
    assert all((out_dir / arg).is_file() for arg in pep_args)


def test_netmhcpan_manifest_outside_the_checkout_records_names_only(tmp_path, monkeypatch):
    _scratch_checkout(tmp_path, monkeypatch)
    out_dir = tmp_path / "elsewhere" / "ext_scores"

    records = prep.prepare_inputs(_held_out_dir(tmp_path / "elsewhere"), out_dir)

    recorded = [row["peptide_list_path"] for row in _manifest(out_dir)]
    assert recorded == ["external:" + name for name in _NAMES]
    assert [r["peptide_list_path"] for r in records] == recorded
    # The manifest sits beside the lists, so each recorded name resolves there.
    for value in recorded:
        assert (out_dir / "inputs" / value.removeprefix("external:")).is_file()
    _assert_runner_finds_every_list(out_dir)
    manifest_text = (out_dir / "inputs" / "manifest.csv").read_text(encoding="utf-8")
    assert _with_workstation_path(manifest_text.splitlines(), tmp_path) == []


def test_netmhcpan_manifest_inside_the_checkout_records_repo_relative(tmp_path, monkeypatch):
    checkout = _scratch_checkout(tmp_path, monkeypatch)
    out_dir = checkout / "_local" / "ext_scores"

    prep.prepare_inputs(_held_out_dir(tmp_path / "elsewhere"), out_dir)

    recorded = [row["peptide_list_path"] for row in _manifest(out_dir)]
    assert recorded == ["_local/ext_scores/inputs/" + name for name in _NAMES]
    # Read from the checkout root, each recorded path names the list it wrote.
    assert all((checkout / value).is_file() for value in recorded)
    _assert_runner_finds_every_list(out_dir)


def test_netmhcpan_default_out_dir_is_recorded_as_before(tmp_path, monkeypatch):
    """The default --out-dir, run from the directory it is relative to: the
    manifest records the same "/" paths as before, and they resolve from there."""
    monkeypatch.chdir(tmp_path)
    test_set_dir = _held_out_dir(tmp_path)
    out_dir = Path(prep.DEFAULT_OUT_DIR)

    prep.prepare_inputs(test_set_dir, out_dir)

    recorded = [row["peptide_list_path"] for row in _manifest(out_dir)]
    assert recorded == [(out_dir / "inputs" / name).as_posix() for name in _NAMES]
    assert recorded[0] == "_local/ext_scores/inputs/TESTV__HLA-A.pep"
    assert all((tmp_path / value).is_file() for value in recorded)
    _assert_runner_finds_every_list(tmp_path / out_dir)
