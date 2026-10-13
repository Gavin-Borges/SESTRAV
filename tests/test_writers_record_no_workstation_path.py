"""Writers that record a path through `src.provenance_paths.relative_source`.

Each writer below used to record a caller-supplied path as given, or reduced it
in a way that could keep a username. Each now records the path through
`relative_source` (the shared `scripts/_dataset_utils.write_provenance` is covered
in tests/test_provenance_paths.py). The tests run each writer with paths under
pytest's temporary directory, which is itself a workstation path, run the
detector below over EVERY string, keys included, of the JSON record they read
back, not only the field that changed, and assert CSV and Markdown fields
directly. How each writer is driven, and what is stubbed, is stated in its test.

The detector those checks use is validated first, on known cases in both
directions. The live-tree sweep at the end runs the same detector over every
string of every tracked `*.json`, through a function that is itself validated on
a throwaway git repository, so a sweep that stopped reading some fields would fail
here rather than pass on a clean tree.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
REPO = os.path.realpath(_ROOT)
_USER = "probeuser"

# Each finder names one way a string can carry a workstation path.
_FINDERS = (
    # A drive letter followed by a separator. Not after a letter or digit, so the
    # "s:" of "https:" and the "3:" of "s3:" are not drives.
    ("drive letter", re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")),
    # A drive-relative path: a drive letter glued to a name that reaches a
    # separator ("C:Users\x"), or to a file name with a letter-led extension that
    # ends the token ("C:x.csv"). "HLA-A:02:01" and "A:1.5" are neither.
    (
        "drive-relative path",
        re.compile(
            r"(?<![A-Za-z0-9])[A-Za-z]:"
            r"(?:[^\s\\/:\"'<>|]+[\\/]|[^\s\\/:\"'<>|.]+\.[A-Za-z]\w{0,7}(?![\w:]))"
        ),
    ),
    # An absolute POSIX path: a slash that starts a token, then a component and
    # another separator. Not after a word character, "/", ".", "-" or "~", and not
    # "//", so "n/a", "AUC-PR/ROC", "~2/3" and a URL's "://" are not one.
    ("POSIX absolute path", re.compile(r"(?<![\w/.\-~])/(?![/\\])[^\s/\\]+[/\\]")),
    # A bare top-level directory that holds user or temporary files ("/tmp").
    (
        "bare system directory",
        re.compile(
            r"(?<![\w/\\.\-~])[/\\]"
            r"(?:home|Users|tmp|root|mnt|media|Volumes|private|var)(?![\w.\-])"
        ),
    ),
    ("UNC path", re.compile(r"(?<![\w:/])(?:\\\\|//)[^\s\\/]+[\\/]")),
    ("Users or home component", re.compile(r"(?i)(?:^|[\\/])(?:users|home)[\\/]")),
    # The scheme followed by a slash, as every file URI is. A bare "file:" also
    # ends ordinary prose ("the merged file: existing_rows equals ...").
    ("file: URI", re.compile(r"(?i)(?<!\w)file:/")),
    # Any other "scheme://" off the allowlist relative_source keeps ("smb://").
    (
        "URL scheme off the allowlist",
        re.compile(r"(?i)(?<![\w+.\-])(?!(?:https?|ftps?|s3|gs)://)[a-z][a-z0-9+.\-]+://"),
    ),
    ("URL credentials", re.compile(r"(?i)(?<![\w+.\-])[a-z][a-z0-9+.\-]+://[^/\s?#@]*@")),
    # "~/" or "~user/" anywhere, or a whole string that is "~user". A "~" before a
    # digit is prose ("~2/3 of rows", "~50/50"), never a home directory.
    (
        "home-directory tilde",
        re.compile(r"(?<![\w~])~(?:[A-Za-z_][\w.\-]*)?[\\/]|^~[A-Za-z_][\w.\-]*$"),
    ),
)


def workstation_path_findings(text: str) -> list[str]:
    return [name for name, pattern in _FINDERS if pattern.search(text)]


def _strings(node):
    """Yield every string in a JSON-shaped value, keys included."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(key)
            yield from _strings(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _strings(value)


def _findings(document) -> list[tuple[str, list[str]]]:
    hits = [(s, workstation_path_findings(s)) for s in _strings(document)]
    return [(s, f) for s, f in hits if f]


def _assert_no_workstation_path(document, context):
    hits = _findings(document)
    assert not hits, f"{context}: {hits}"


# --- the detector, validated ---------------------------------------------------

_POSITIVE = [
    "C:\\Users\\" + _USER + "\\x.csv",
    "c:/data/x.csv",
    "C:Users\\" + _USER + "\\x.csv",
    "C:x.csv",
    "C://Users/" + _USER + "/x.csv",
    "/home/" + _USER + "/x.csv",
    "/home/" + _USER,
    "/home",
    "/Users/" + _USER + "/x.csv",
    "/".join(["", "tmp"]),
    "/".join(["", "tmp", "pytest-of-" + _USER, "out.csv"]),
    "\\\\server\\share\\x.csv",
    "//server/share/x.csv",
    "/c/Users/" + _USER + "/x.csv",
    "\\Users\\" + _USER + "\\x.csv",
    "Users/" + _USER + "/x.csv",
    "file:///C:/x.csv",
    "file:/srv/x.csv",
    "smb://fileserver/lab/x.csv",
    "sftp://host/x.csv",
    "https://" + _USER + ":pw@example.org/x.csv",
    "~/x.csv",
    "~" + _USER + "/x.csv",
    "~" + _USER,
    "wrote C:\\Users\\" + _USER + "\\y.csv to disk",
    "external:/home/" + _USER + "/y.csv",
]

_NEGATIVE = [
    "https://example.org/a/b.csv",
    "http://hiv.lanl.gov/x",
    "https://example.org:8443/x",
    "s3://bucket/key/x.csv",
    "gs://bucket/key",
    "ftp://example.org/x",
    "data/raw/x.csv",
    "data\\raw\\x.csv",
    "data/home_ranges.csv",
    "results/x.csv.provenance.json",
    "external:x.csv",
    "external:",
    "HLA-A*02:01",
    "HLA-A:02:01",
    "ratio A:1.5",
    "the C: drive",
    "Def A: positives + real IEDB (iedb_api) negatives, 9 target viruses",
    "2026-10-12T16:13:03.123456+00:00",
    "StratifiedGroupKFold(groups=peptide)",
    "IsotonicRegression(out_of_bounds=clip, y_min=0.0, y_max=1.0)",
    "is_quarantined != True",
    "AUC-PR/ROC, +/- 0.01, n/a",
    "and/or, w/o, mg/dL",
    "profile: default",
    "the merged file: existing_rows equals row_count",
    "~2/3 of rows",
    "a ~50/50 split",
    "~1/4 of the corpus",
    "roughly ~5% of rows",
    "contact: someone@example.org",
]


@pytest.mark.parametrize("text", _POSITIVE)
def test_the_detector_catches_each_workstation_path_shape(text):
    assert workstation_path_findings(text), text


@pytest.mark.parametrize("text", _NEGATIVE)
def test_the_detector_does_not_flag_urls_relative_paths_or_plain_text(text):
    assert workstation_path_findings(text) == [], text


def test_findings_read_every_string_keys_and_nesting_included():
    leak = "C:\\Users\\" + _USER + "\\x.csv"
    document = {
        "sources": ["data/x.csv"],
        "inputs": {leak: "0" * 64},
        "nested": [{"deep": ("ok", leak)}],
    }
    assert [s for s, _ in _findings(document)] == [leak, leak]


# --- each sidecar writer -----------------------------------------------------


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_ingest_published_panels(tmp_path):
    """main(), end to end, on a one-row panel."""
    from scripts.ingest_published_panels import main

    panel = tmp_path / "panel.csv"
    pd.DataFrame(
        [{"peptide": "GILGFVFTL", "label": 1, "hla_allele": "HLA-A*02:01", "protein": "M1"}]
    ).to_csv(panel, index=False)
    out = tmp_path / "out.csv"

    argv = ["--input", str(panel), "--virus", "EBV", "--pmid", "99999", "--output", str(out)]
    assert main(argv) == 0

    prov = _read_json(tmp_path / "out_provenance.json")
    assert (prov["input_file"], prov["output_file"]) == ("external:panel.csv", "external:out.csv")
    _assert_no_workstation_path(prov, "ingest_published_panels")


def test_ingest_iedb_negatives(tmp_path):
    """main(), end to end, on a two-row export."""
    from scripts.ingest_iedb_negatives import main

    row = {
        "Host Organism Name": "Homo sapiens",
        "Qualitative Measure": "Negative",
        "Epitope Name": "GILGFVFTL",
        "Allele Name": "HLA-A*02:01",
        "Assay Group": "IFN-gamma ELISpot",
        "Antigen Name": "Matrix protein 1",
        "Reference PubMed ID": "12345",
        "Description": "Influenza A virus",
        "Response Frequency As Reported": "",
        "Assay ID": "AS-1",
    }
    export = tmp_path / "iedb.csv"
    pd.DataFrame([row, {**row, "Epitope Name": "NLVPMVATV", "Assay ID": "AS-2"}]).to_csv(
        export, index=False
    )
    existing = tmp_path / "existing_v4.csv"
    pd.DataFrame(
        {"peptide": ["AAAAAAAAA"], "hla_allele": ["HLA-A*02:01"], "virus": ["IAV"], "label": [0]}
    ).to_csv(existing, index=False)
    out = tmp_path / "negatives.csv"

    argv = ["--input", str(export), "--output", str(out), "--existing-dataset", str(existing)]
    assert main(argv) == 0

    prov = _read_json(tmp_path / "negatives_provenance.json")
    assert prov["input_file"] == "external:iedb.csv"
    assert prov["existing_dataset"] == "external:existing_v4.csv"
    assert prov["output_file"] == "external:negatives.csv"
    _assert_no_workstation_path(prov, "ingest_iedb_negatives")


def test_ingest_lanl_hiv(tmp_path):
    """main(), end to end, on a one-row export."""
    from scripts.ingest_lanl_hiv import main

    export = tmp_path / "export.csv"
    export.write_text(
        "Optimal Sequence,HLA,Assay Result,PMID\nSLYNTVATL,HLA-A*02:01,Negative,12345\n"
    )
    out = tmp_path / "hiv.csv"

    assert main(["--input", str(export), "--output", str(out)]) == 0

    prov = _read_json(tmp_path / "hiv_provenance.json")
    assert (prov["input_file"], prov["output_file"]) == ("external:export.csv", "external:hiv.csv")
    _assert_no_workstation_path(prov, "ingest_lanl_hiv")


def test_download_lanl_hcv(tmp_path):
    """main(), end to end, on a one-row local export (--input, so nothing is fetched)."""
    from scripts.download_lanl_hcv import main

    export = tmp_path / "export.tsv"
    pd.DataFrame(
        [
            {
                "Sequence": "GILGFVFTL",
                "MHC Restriction": "HLA-A*02:01",
                "Qualitative Measure": "Negative",
                "PMID": "12682213",
                "Protein": "NS3",
                "Assay": "T cell IFN-gamma ELISpot",
            }
        ]
    ).to_csv(export, index=False, sep="\t")
    out = tmp_path / "hcv.csv"

    assert main(["--input", str(export), "--output", str(out)]) == 0

    prov = _read_json(tmp_path / "hcv_provenance.json")
    assert (prov["input_file"], prov["output_file"]) == ("external:export.tsv", "external:hcv.csv")
    _assert_no_workstation_path(prov, "download_lanl_hcv")


def test_build_dataset_v5(tmp_path):
    """main(), end to end, on a four-row base and a two-row negatives file."""
    from scripts.build_dataset_v5 import main

    base = tmp_path / "v4.csv"
    pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "NLVPMVATV", "AAAAAAAAA", "CCCCCCCCC"],
            "label": [1, 1, 0, 0],
            "virus": ["IAV", "CMV", "Self", "Self"],
            "protein": ["M1", "pp65", "", ""],
            "strain": [None, None, None, None],
            "hla_allele": ["HLA-A*02:01"] * 4,
            "source_type": ["Virus", "Virus", "Self", "Self"],
            "database_source": ["IEDB", "IEDB", "UniProt", "UniProt"],
        }
    ).to_csv(base, index=False)
    iedb = tmp_path / "iedb_neg.csv"
    pd.DataFrame(
        {
            "peptide": ["KLGGALQAK", "CINGVCWTV"],
            "label": [0, 0],
            "virus": ["CMV", "HCV"],
            "hla_allele": ["HLA-A*02:01", "HLA-A*02:01"],
            "source_type": ["Virus", "Virus"],
            "database_source": ["IEDB", "IEDB"],
            "negative_origin": ["tested_negative", "tested_negative"],
            "reference_pmid": ["555", "666"],
        }
    ).to_csv(iedb, index=False)
    out = tmp_path / "v5.csv"
    schema = os.path.join(REPO, "data", "immunogenicity_dataset_v5_schema.json")
    argv = ["--base-dataset", str(base), "--iedb-negatives", str(iedb), "--output", str(out)]
    argv += ["--schema", schema, "--conflict-audit-path", str(tmp_path / "conflicts.csv")]

    assert main(argv) == 0

    prov = _read_json(tmp_path / "v5_provenance.json")
    assert prov["output_file"] == "external:v5.csv"
    _assert_no_workstation_path(prov, "build_dataset_v5")


def _write_fixture(work: Path, relative: str, frame: pd.DataFrame) -> None:
    path = work / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_compute_d29_corpus_composition(tmp_path, monkeypatch):
    """main(), end to end. The script reads its input at a fixed repo-relative path,
    so it runs from a scratch directory holding a three-row corpus at that path."""
    from scripts import compute_d29_corpus_composition as module

    work = tmp_path / "work"
    corpus = pd.DataFrame(
        {
            "hla_allele": ["HLA-A*02:01", "HLA-C*06:02", "HLA-B*07:02"],
            "is_quarantined": [False, False, True],
        }
    )
    _write_fixture(work, module.DATASET_SRC, corpus)
    monkeypatch.chdir(work)
    out = tmp_path / "out" / "table.csv"

    module.main(["--output", str(out)])

    prov = _read_json(str(out) + ".provenance.json")
    assert prov["artifact"] == "external:table.csv"
    assert list(prov["inputs"]) == [module.DATASET_SRC]
    _assert_no_workstation_path(prov, "compute_d29_corpus_composition")


def test_compute_loo_binding_confound(tmp_path, monkeypatch):
    """main(), end to end, from a scratch directory holding both nine-virus inputs."""
    from scripts import compute_loo_binding_confound as module

    work = tmp_path / "work"
    viruses = list(module.CANON)
    _write_fixture(
        work,
        module.PV_SRC,
        pd.DataFrame(
            {
                "virus": viruses,
                "auc_roc": [0.8] * 9,
                "auc_roc_real_neg_only": [0.7] * 9,
                "n_neg_decoy": [10] * 9,
                "n_neg_real": [30] * 9,
            }
        ),
    )
    _write_fixture(
        work, module.LOO_SRC, pd.DataFrame({"test_virus": viruses, "auc_roc": [0.6] * 9})
    )
    monkeypatch.chdir(work)
    out = tmp_path / "out" / "table.csv"

    module.main(["--output", str(out)])

    prov = _read_json(str(out) + ".provenance.json")
    assert prov["artifact"] == "external:table.csv"
    assert sorted(prov["inputs"]) == sorted([module.PV_SRC, module.LOO_SRC])
    _assert_no_workstation_path(prov, "compute_loo_binding_confound")


def test_compute_section33_decoy_binding_join(tmp_path, monkeypatch):
    """main(), end to end, from a scratch directory holding a five-row corpus and a
    four-row binding matrix at the script's fixed input paths."""
    from scripts import compute_section33_decoy_binding_join as module

    work = tmp_path / "work"
    _write_fixture(
        work,
        module.DATASET_SRC,
        pd.DataFrame(
            {
                "peptide": ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE", "FFFFFFFFF"],
                "label": [1, 0, 0, 0, 0],
                "virus": ["CMV", "CMV", "CMV", "EBV", "EBV"],
                "is_quarantined": [False] * 5,
                "negative_origin": [
                    "",
                    module.DECOY_ORIGIN,
                    module.DECOY_ORIGIN,
                    "tested_negative",
                    "iedb_api",
                ],
            }
        ),
    )
    _write_fixture(
        work,
        module.MATRIX_SRC,
        pd.DataFrame(
            {
                "peptide": ["AAAAAAAAA", "CCCCCCCCC", "EEEEEEEEE", "FFFFFFFFF"],
                "bind_HLA-A*02:01": [0.9, 0.2, 0.1, 0.3],
            }
        ),
    )
    monkeypatch.chdir(work)
    out = tmp_path / "out" / "table.csv"

    module.main(["--output", str(out)])

    prov = _read_json(str(out) + ".provenance.json")
    assert prov["artifact"] == "external:table.csv"
    assert sorted(prov["inputs"]) == sorted([module.DATASET_SRC, module.MATRIX_SRC])
    _assert_no_workstation_path(prov, "compute_section33_decoy_binding_join")


def test_extract_allele_aware_data_stub_and_full_run(tmp_path, monkeypatch):
    """main() twice: once with no records (the stub) and once with records. The two
    loaders, which read IEDB exports, are stubbed; main() runs to each write."""
    from scripts import extract_allele_aware_data as module

    data_dir = tmp_path / "iedb"
    data_dir.mkdir()
    out = tmp_path / "allele_aware.csv"
    monkeypatch.setattr(sys, "argv", ["x", "--data-dir", str(data_dir), "--output", str(out)])

    # No records loaded: the script writes a provenance stub and exits.
    monkeypatch.setattr(module, "load_tcell_assay_files", lambda *a, **k: pd.DataFrame())
    with pytest.raises(SystemExit):
        module.main()
    stub = _read_json(tmp_path / "allele_aware_stub.json")
    assert (stub["data_dir"], stub["output_path"]) == ("external:iedb", "external:allele_aware.csv")
    _assert_no_workstation_path(stub, "extract_allele_aware_data stub")

    # Records loaded: the full provenance sidecar.
    monkeypatch.setattr(module, "load_tcell_assay_files", lambda *a, **k: pd.DataFrame({"x": [1]}))
    built = pd.DataFrame(
        {"peptide": ["AAAAAAAAA", "CCCCCCCCC"], "allele": ["HLA-A*02:01"] * 2, "label": [1, 0]}
    )
    monkeypatch.setattr(module, "build_allele_aware_dataset", lambda *a, **k: built)
    module.main()
    prov = _read_json(tmp_path / "allele_aware_provenance.json")
    assert prov["output_path"] == "external:allele_aware.csv"
    assert prov["holdout_path"] == "external:allele_aware_holdouts.csv"
    _assert_no_workstation_path(prov, "extract_allele_aware_data")


def test_compute_binding_only_comparison(tmp_path, monkeypatch):
    """main() with compute_table stubbed, which needs real out-of-fold predictions.
    One input absolute, one relative with a subdirectory: the relative one keeps
    the "/" it was recorded with before (`Path.as_posix()`), on Windows too."""
    from scripts import compute_binding_only_comparison as module

    monkeypatch.setattr(module, "compute_table", lambda *a, **k: pd.DataFrame({"x": [1]}))
    mode31, mode10 = tmp_path / "oof31.csv", tmp_path / "sub" / "oof10.csv"
    mode10.parent.mkdir()
    mode31.write_text("a\n1\n")
    mode10.write_text("a\n2\n")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "comparison.csv"
    argv = ["--mode31-oof", str(mode31), "--mode10-oof", "sub/oof10.csv", "--output", str(out)]

    assert module.main(argv) == 0

    prov = _read_json(tmp_path / "comparison.csv.provenance.json")
    assert (prov["mode31_oof"], prov["mode10_oof"]) == ("external:oof31.csv", "sub/oof10.csv")
    _assert_no_workstation_path(prov, "compute_binding_only_comparison")


def test_compute_binding_zerofill_attribution(tmp_path, monkeypatch):
    """main() with compute_table, validate_comparison and both git calls stubbed.

    Its inputs must sit inside the checkout (git_blob raises for any other path),
    so the shape that leaked is an ABSOLUTE spelling of an inside path. Four
    tracked files stand in for the four inputs: this test is about the recorded
    paths, not about what the files hold. The fourth is given relative to the
    checkout, with a subdirectory, and keeps its "/" on Windows too."""
    from scripts import compute_binding_zerofill_attribution as module

    monkeypatch.setattr(module, "compute_table", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(module, "validate_comparison", lambda *a, **k: 0.0)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "0" * 40 + "\n")
    monkeypatch.chdir(REPO)
    names = ["README.md", "LICENSE", "pyproject.toml", "scripts/_dataset_utils.py"]
    inputs = [str(Path(REPO, name)) for name in names[:3]] + [names[3]]
    out = tmp_path / "zerofill.csv"
    argv = ["--mode31-oof", inputs[0], "--mode10-oof", inputs[1]]
    argv += ["--binding-matrix", inputs[2], "--comparison", inputs[3], "--output", str(out)]

    assert module.main(argv) == 0

    prov = _read_json(tmp_path / "zerofill.csv.provenance.json")
    assert [record["path"] for record in prov["inputs"]] == names
    _assert_no_workstation_path(prov, "compute_binding_zerofill_attribution")


def test_compute_mode51_allele_identity_discriminator(tmp_path, monkeypatch):
    """main() with compute_table and the git call stubbed."""
    from scripts import compute_mode51_allele_identity_discriminator as module

    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "0" * 40 + "\n")
    monkeypatch.setattr(
        module, "compute_table", lambda *a, **k: pd.DataFrame(columns=module.OUTPUT_COLUMNS)
    )
    data, matrix = tmp_path / "data.csv", tmp_path / "sub" / "matrix.csv"
    matrix.parent.mkdir()
    data.write_text("peptide\n")
    matrix.write_text("peptide\n")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "table.csv"
    argv = ["--data", str(data), "--binding-matrix", "sub/matrix.csv", "--output", str(out)]

    assert module.main(argv) == 0

    prov = _read_json(tmp_path / "table.csv.provenance.json")
    assert prov["training_data"] == "external:data.csv"
    assert prov["binding_matrix"] == "sub/matrix.csv"
    _assert_no_workstation_path(prov, "compute_mode51_allele_identity_discriminator")


def test_compute_pooled_cv_metrics(tmp_path):
    """main(), end to end, on a four-row out-of-fold file outside the checkout."""
    from scripts.compute_pooled_cv_metrics import _sidecar_key, main

    oof = tmp_path / "oof.csv"
    oof.write_text("label,score,fold\n1,0.9,1\n0,0.2,1\n1,0.7,2\n0,0.4,2\n")
    out = tmp_path / "pooled.csv"

    assert main(["--oof", str(oof), "--output", str(out)]) == 0

    prov = _read_json(tmp_path / "pooled.csv.provenance.json")
    assert list(prov["inputs"]) == ["external:oof.csv"]
    _assert_no_workstation_path(prov, "compute_pooled_cv_metrics")
    assert _sidecar_key(Path(REPO) / "README.md") == "README.md"


# --- the repo-relative-or-bare-name reductions, now relative_source -----------
#
# Each of these recorded `path.resolve().relative_to(root)`, or the bare file name
# outside `root`. A path inside `root` under `Users/<user>/` came out as that
# relative path, username included (run from the checkout's root, a relative
# `Users/<user>/x.csv` argument is such a path). Each test calls the writing
# function directly, with its module's root pointed at a scratch checkout holding a
# file under `Users/<user>/`, and once with a file outside that checkout.


@pytest.fixture(params=["under a Users/<user> directory", "outside the checkout"])
def scratch_input(request, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    if request.param == "outside the checkout":
        path = tmp_path / "elsewhere" / "in.csv"
    else:
        path = root / "Users" / _USER / "in.csv"
    path.parent.mkdir(parents=True)
    path.write_text("peptide,label\nAAAAAAAAA,1\n", encoding="utf-8")
    return root, path


def test_assess_calibration_write_provenance(scratch_input, tmp_path, monkeypatch):
    from scripts import assess_calibration as module

    root, oof = scratch_input
    monkeypatch.setattr(module, "PROJECT_ROOT", root)
    out = tmp_path / "calibration.csv"
    out.write_text("scope,n\npooled_all,1\n", encoding="utf-8")

    module._write_provenance(out, oof, pd.DataFrame({"scope": ["pooled_all"], "n": [1]}))

    prov = _read_json(module._provenance_path(out))
    assert (prov["oof_path"], prov["artifact"]) == ("external:in.csv", "external:calibration.csv")
    _assert_no_workstation_path(prov, "assess_calibration")


def test_audit_cv_leakage_write_provenance(scratch_input, tmp_path, monkeypatch):
    from scripts import audit_cv_leakage as module

    root, dataset = scratch_input
    monkeypatch.setattr(module, "PROJECT_ROOT", root)
    for constant in ("BINDING_MATRIX_PATH", "ANTIGEN_PROCESSING_CACHE_PATH"):
        monkeypatch.setattr(module, constant, root / "models" / (constant.lower() + ".csv"))
    monkeypatch.setattr(module, "SELF_SIMILARITY_CACHE_PATH", root / "data" / "absent.csv")
    out = tmp_path / "audit.csv"
    out.write_text("metric,value\nx,0\n", encoding="utf-8")

    module._write_provenance(out, dataset, n_rows=1)

    prov = _read_json(module._provenance_path(out))
    assert (prov["dataset_path"], prov["artifact"]) == ("external:in.csv", "external:audit.csv")
    _assert_no_workstation_path(prov, "audit_cv_leakage")


def test_build_binding_matrix_v5_write_provenance(scratch_input, tmp_path, monkeypatch):
    from scripts import build_binding_matrix_v5 as module

    root, dataset = scratch_input
    monkeypatch.setattr(module, "PROJECT_ROOT", root)
    out = tmp_path / "matrix_v5.csv"

    module._write_provenance(
        out,
        dataset,
        tmp_path / "matrix_v4.csv",
        1,
        2,
        active_peptide_count=1,
        already_covered_count=0,
    )

    prov = _read_json(out.with_suffix(".provenance.json"))
    assert prov["source_dataset"] == "external:in.csv"
    _assert_no_workstation_path(prov, "build_binding_matrix_v5")


def test_train_classifier_training_provenance_fields(scratch_input, monkeypatch):
    """`_input_provenance_pair`, through the public `training_provenance_fields`;
    relative_source's default root is pointed at the scratch checkout."""
    import src.provenance_paths
    from src.train_classifier import training_provenance_fields

    root, data = scratch_input
    monkeypatch.setattr(src.provenance_paths, "PROJECT_ROOT", root)

    fields = training_provenance_fields(
        data, 35, antigen_processing_cache_path=data, self_similarity_cache_path=data
    )

    for name in ("training_data", "antigen_processing_cache", "self_similarity_cache"):
        assert fields[name + "_path"] == "external:in.csv"
    _assert_no_workstation_path(fields, "train_classifier")


# --- run-metadata writers that are not provenance sidecars -------------------


def _loo_fixture(tmp_path, *, v5):
    """Two viruses, each with two positives and two negatives, plus two decoys."""
    rows = []
    for virus, letter in (("CMV", "C"), ("EBV", "E")):
        for i, label in enumerate([1, 1, 0, 0]):
            rows.append(
                {
                    "peptide": letter * 8 + "ACDE"[i],
                    "label": label,
                    "virus": virus,
                    "source_type": "Virus",
                    "is_quarantined": False,
                    "negative_origin": "" if label else "tested_negative",
                }
            )
    for i in range(2):
        rows.append(
            {
                "peptide": "W" * 8 + "ACDE"[i],
                "label": 0,
                "virus": None,
                "source_type": "Self",
                "is_quarantined": False,
                "negative_origin": "allele_matched_nonbinder",
            }
        )
    frame = pd.DataFrame(rows)
    if not v5:
        frame = frame.drop(columns=["is_quarantined", "negative_origin"])
    dataset = tmp_path / "dataset.csv"
    frame.to_csv(dataset, index=False)
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("peptide,bind_A0201\n", encoding="utf-8")
    return dataset, matrix


@pytest.mark.parametrize("version", ["v4", "v5"])
def test_run_loo_cross_virus(version, tmp_path, monkeypatch):
    """run_loo() end to end on a ten-row corpus, with the mode-31 feature builder
    stubbed (it needs the real binding matrix and caches) and a five-tree forest."""
    import importlib

    module = importlib.import_module("scripts.run_loo_cross_virus_" + version)
    dataset, matrix = _loo_fixture(tmp_path, v5=version == "v5")
    rng = np.random.default_rng(0)
    monkeypatch.setattr(
        module,
        "prepare_features_31",
        lambda frame, _path: pd.DataFrame(rng.random((len(frame), 3))),
    )
    monkeypatch.setattr(module, "RF_PARAMS", dict(n_estimators=5, random_state=0, n_jobs=1))
    monkeypatch.chdir(tmp_path)  # run_loo creates a "results" directory in the cwd
    out_json, out_csv = tmp_path / "out" / "loo.json", tmp_path / "out" / "loo.csv"
    out_json.parent.mkdir()

    module.run_loo(
        str(dataset),
        str(matrix),
        output_json=str(out_json),
        output_csv=str(out_csv),
        min_pos=1,
        min_neg=1,
    )

    prov = _read_json(out_json)
    assert (prov["dataset"], prov["binding_matrix"]) == (
        "external:dataset.csv",
        "external:matrix.csv",
    )
    _assert_no_workstation_path(prov, "run_loo_cross_virus_" + version)


def test_train_gnn_run_config():
    """build_gnn_run_config, whose return value train_gnn_v2 writes to
    gnn_config.json, called directly; the training run itself is not run here."""
    from src.train_gnn import build_gnn_run_config

    arguments = dict(
        esm2_model_name="C:\\Users\\" + _USER + "\\models\\esm2_local",
        node_dim=320,
        feature_mode=31,
        num_continuous_features=31,
        binding_matrix_path="/home/" + _USER + "/bm.csv",
        max_epochs=1,
        mean_best_epoch=1,
        early_stopping_patience=1,
        seed=0,
        pooling="mean",
        edge_mode="full",
    )

    config = build_gnn_run_config(**arguments)

    assert (config["esm2_model_name"], config["binding_matrix_path"]) == (
        "external:esm2_local",
        "external:bm.csv",
    )
    _assert_no_workstation_path(config, "train_gnn")
    arguments.update(esm2_model_name="facebook/esm2_t12_35M_UR50D", binding_matrix_path=None)
    hub = build_gnn_run_config(**arguments)
    assert (hub["esm2_model_name"], hub["binding_matrix_path"]) == (
        "facebook/esm2_t12_35M_UR50D",
        None,
    )


def test_final_validation_and_h2_tier_a(tmp_path, monkeypatch):
    """run_final_validation() end to end into a scratch results directory, running
    the real run_h2_tier_a. Stubbed: the gold-standard and baseline reports (they
    read a populated results tree), and inside H2 the verified model load, the
    feature builder, the binding lookup and the bootstrap CI (they need the real
    model, matrix and corpus). The model is a two-feature decision tree."""
    from sklearn.tree import DecisionTreeClassifier

    from src import final_validation_report as report
    from src import h2_tier_a_evaluation as h2

    rng = np.random.default_rng(0)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    data, matrix, model = inputs / "data.csv", inputs / "matrix.csv", inputs / "model.joblib"
    pd.DataFrame(
        {
            "peptide": ["KLM" + format(i, "06d") for i in range(40)],
            "label": [i % 2 for i in range(40)],
            "virus": ["CMV"] * 40,
        }
    ).to_csv(data, index=False)
    matrix.write_text("peptide\n", encoding="utf-8")
    model.write_bytes(b"not a model")
    template = DecisionTreeClassifier(max_depth=2, random_state=0)
    template.fit(rng.random((4, 2)), [0, 1, 0, 1])

    monkeypatch.setattr(h2, "load_verified_joblib", lambda *a, **k: template)
    monkeypatch.setattr(
        h2, "model_provenance_fields", lambda p: {"model_path": "m", "model_sha256": None}
    )
    monkeypatch.setattr(
        h2, "_build_features", lambda frame, n, _p: pd.DataFrame(rng.random((len(frame), n)))
    )
    monkeypatch.setattr(
        h2, "_attach_binding_max", lambda frame, _p: pd.Series(rng.random(len(frame)))
    )
    monkeypatch.setattr(h2, "_bootstrap_ratio_ci", lambda *a, **k: (float("nan"), float("nan")))
    monkeypatch.setattr(
        report, "full_validation_report", lambda *a, **k: pd.DataFrame({"stage1_found": [True]})
    )
    monkeypatch.setattr(report, "compare_methods", lambda *a, **k: pd.DataFrame({"m": [1]}))
    out = tmp_path / "out"
    out.mkdir()

    report.run_final_validation(
        results_dir=str(out),
        data_path=str(data),
        binding_matrix_path=str(matrix),
        model_path=str(model),
    )

    freeze = _read_json(out / "freeze_status.json")
    assert {k: v["path"] for k, v in freeze["inputs"].items()} == {
        "data_path": "external:data.csv",
        "binding_matrix_path": "external:matrix.csv",
        "model_path": "external:model.joblib",
    }
    _assert_no_workstation_path(freeze, "final_validation_report freeze_status")
    summary = pd.read_csv(out / "h2_tier_a_summary.csv")
    assert set(summary["model_path"]) == {"external:model.joblib"}
    md = (out / "h2_tier_a_summary.md").read_text(encoding="utf-8")
    for line in (
        "- Dataset: `external:data.csv`",
        "- Integrated model template: `external:model.joblib`",
        "- Binding matrix: `external:matrix.csv`",
    ):
        assert line in md
    report_md = (out / "final_validation_report.md").read_text(encoding="utf-8")
    for line in (
        "- Gold-standard stage validation: `external:gold_standard_validation.csv`",
        "- Baseline comparison: `external:baseline_comparison.csv`",
        "- H2 Tier A summary: `external:h2_tier_a_summary.csv`",
        "- H2 Tier A markdown: `external:h2_tier_a_summary.md`",
    ):
        assert line in report_md


def test_evaluate_per_virus(tmp_path):
    """main(), end to end, on a twenty-row prediction file for one virus."""
    from scripts.evaluate_per_virus import main

    predictions = tmp_path / "preds.csv"
    pd.DataFrame(
        {
            "peptide": ["KLM" + format(i, "06d") for i in range(20)],
            "label": [i % 2 for i in range(20)],
            "score": [(i % 2) * 0.5 + i / 100 for i in range(20)],
            "virus": ["CMV"] * 20,
            "negative_origin": ["" if i % 2 else "tested_negative" for i in range(20)],
        }
    ).to_csv(predictions, index=False)
    out = tmp_path / "per_virus.json"
    argv = ["--predictions", str(predictions), "--output-json", str(out)]
    argv += ["--n-bootstrap", "10", "--min-virus-size", "5"]

    assert main(argv) in (0, 2)  # 2: ran, exit criterion not met

    prov = _read_json(out)
    assert prov["predictions_file"] == "external:preds.csv"
    _assert_no_workstation_path(prov, "evaluate_per_virus")


def test_benchmark_syfpeithi(tmp_path):
    """run_benchmark() end to end on a two-row prediction file."""
    from scripts.benchmark_syfpeithi import run_benchmark

    predictions = tmp_path / "oof.csv"
    predictions.write_text("peptide,score,label\nGLCTLVAML,0.9,1\nAAAAAAAAA,0.1,0\n")
    out = tmp_path / "syfpeithi.json"

    run_benchmark(predictions, out)

    prov = _read_json(out)
    assert prov["predictions_file"] == "external:oof.csv"
    _assert_no_workstation_path(prov, "benchmark_syfpeithi")


def test_data_curation_qc(tmp_path):
    """The script end to end, run as tests/test_data_curation_qc.py runs it but from
    a scratch working directory, because it writes results/qc/ under the cwd."""
    dataset = tmp_path / "valid.csv"
    dataset.write_text("peptide,label\nACDEFGHIK,1\nLMNPQRSTV,0\n")
    config = tmp_path / "config.yaml"
    config.write_text("freeze_mode: false\n")
    work = tmp_path / "work"
    work.mkdir()
    script = str(Path(REPO, "src", "data_curation_qc.py"))
    argv = [sys.executable, script, "--check-dataset", str(dataset), "--config", str(config)]

    result = subprocess.run(argv, capture_output=True, text=True, cwd=work)

    assert result.returncode == 0, result.stderr
    written = list(work.rglob("dataset_qc.json"))
    assert len(written) == 1
    prov = _read_json(written[0])
    assert prov["dataset"] == "external:valid.csv"
    _assert_no_workstation_path(prov, "data_curation_qc")


def test_external_validation_tier_b_recovery(tmp_path, monkeypatch):
    """main() with run_recovery stubbed (it reads PredIG and PRIME outputs)."""
    from src import external_validation_tier_b_recovery as module

    tables = (pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [2]}))
    monkeypatch.setattr(module, "run_recovery", lambda *a, **k: tables)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["x", "--run-dir", str(run_dir)])

    module.main()

    prov = _read_json(run_dir / "processed" / "tier_b_recovery_summary.json")
    assert (prov["recovery_csv"], prov["negative_csv"]) == (
        "external:tier_b_gold_standard_recovery.csv",
        "external:tier_b_negative_discrimination.csv",
    )
    _assert_no_workstation_path(prov, "external_validation_tier_b_recovery")


def test_prepare_external_validation_tier_b_inputs(tmp_path, monkeypatch):
    """main() for one virus, with select_tier_b_pool stubbed (it ranks a real
    proteome feature table)."""
    from src import prepare_external_validation_tier_b_inputs as module

    out = tmp_path / "out"
    out.mkdir()
    (out / "features.csv").write_text("peptide\nAAAAAAAAA\n")
    pool = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA"],
            "binding_max": [0.5],
            "forced_gold_standard": [False],
            "virus": ["EBV"],
            "proteome_id": ["P"],
        }
    )
    monkeypatch.setattr(module, "VIRUS_FEATURE_FILES", {"EBV": "features.csv"})
    monkeypatch.setattr(module, "select_tier_b_pool", lambda *a, **k: pool)
    config = str(tmp_path / "absent.yaml")
    monkeypatch.setattr(
        sys, "argv", ["x", "--results-dir", str(out), "--top-n", "1", "--config", config]
    )

    module.main()

    prov = _read_json(out / "external_tier_b_prefilter_manifest.json")
    assert prov["viruses"]["EBV"]["peptide_list"] == "external:external_tier_b_EBV_peptides.txt"
    _assert_no_workstation_path(prov, "prepare_external_validation_tier_b_inputs")


# --- the live tree ------------------------------------------------------------


def _tracked_json_documents(root):
    """`(name, parsed document)` for every tracked `*.json` under `root`, or None
    when `root` is not the top of a git work tree (a `git archive` export, or no git
    at all). A tracked file deleted from the work tree is read from the index."""
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    if os.path.normcase(os.path.realpath(top)) != os.path.normcase(os.path.realpath(root)):
        return None
    documents = []
    for name in sorted(n for n in listed.split("\0") if n.endswith(".json")):
        path = Path(root, name)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        else:
            text = subprocess.run(
                ["git", "-C", str(root), "show", ":" + name],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            ).stdout
        documents.append((name, json.loads(text)))
    return documents


def _tracked_json_findings(root):
    """Findings over every string of every tracked `*.json` under `root`, as
    `(name, string, finders)`, or None outside a git work tree's top."""
    documents = _tracked_json_documents(root)
    if documents is None:
        return None
    return [(name, s, f) for name, document in documents for s, f in _findings(document)]


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)


def test_the_live_sweep_reads_every_string_of_every_tracked_json(tmp_path):
    """The sweep, run on a throwaway repository whose leaks sit outside `sources`:
    in a key, in a nested list, in a non-sidecar JSON file, and in a tracked file
    deleted from the work tree."""
    repo = tmp_path / "repo"
    repo.mkdir()
    try:
        _git(repo, "init", "-q")
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git is not usable here: {exc}")
    key_leak = "C:\\Users\\" + _USER + "\\x.csv"
    nested_leak = "/home/" + _USER + "/y.csv"
    other_leak = "~" + _USER + "/z.csv"
    deleted_leak = "smb://fileserver/lab/w.csv"
    documents = {
        "a_provenance.json": {"sources": ["data/x.csv"], "inputs": {key_leak: "0" * 64}},
        "b.csv.provenance.json": {"notes": ["fine", {"deep": [nested_leak]}]},
        "config.json": {"dataset": other_leak},
        "clean_provenance.json": {"sources": ["data/x.csv"], "note": "~2/3 of rows"},
        "gone_provenance.json": {"artifact": deleted_leak},
        "not_json.txt": key_leak,
    }
    for name, document in documents.items():
        text = document if isinstance(document, str) else json.dumps(document)
        (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "--", *documents)
    (repo / "gone_provenance.json").unlink()

    findings = _tracked_json_findings(repo)

    assert sorted((name, s) for name, s, _ in findings) == [
        ("a_provenance.json", key_leak),
        ("b.csv.provenance.json", nested_leak),
        ("config.json", other_leak),
        ("gone_provenance.json", deleted_leak),
    ]


def test_the_live_sweep_skips_outside_a_git_work_tree(tmp_path):
    assert _tracked_json_findings(tmp_path) is None


def test_no_tracked_json_carries_a_workstation_path_in_any_string():
    findings = _tracked_json_findings(REPO)
    if findings is None:
        pytest.skip("not a git work tree: no tracked JSON to read")
    listed = subprocess.run(
        ["git", "-C", REPO, "ls-files", "--", "*provenance.json"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert listed, "no tracked provenance sidecar found; the sweep would be vacuous"
    assert findings == []


# Keys whose values the writers routed through relative_source record as paths, plus
# `sources` entries and the KEYS of an `inputs` map (compute_d29_corpus_composition and
# its siblings, and compute_pooled_cv_metrics, key that map by path).
_PATH_KEYS = frozenset(
    {
        "antigen_processing_cache_path",
        "artifact",
        "binding_matrix",
        "binding_matrix_path",
        "data_dir",
        "dataset",
        "dataset_path",
        "esm2_model_name",
        "existing_dataset",
        "holdout_path",
        "input_file",
        "mode10_oof",
        "mode31_oof",
        "negative_csv",
        "oof_path",
        "output_file",
        "output_path",
        "path",
        "peptide_list",
        "predictions_file",
        "recovery_csv",
        "self_similarity_cache_path",
        "source_dataset",
        "training_data",
        "training_data_path",
    }
)
# Keys that also hold a path in some tracked JSON although no writer above routes
# them: the rest of what a listing of every path-looking string, by key, turns up.
# Left out of that listing on purpose: `$schema` (a JSON Schema identifier, whose
# "#" relative_source would drop), `uniprot_base_url` (a URL template), `doi` and
# `anchor_pattern` (a regular expression).
_OTHER_PATH_KEYS = frozenset(
    {
        "citing",
        "existing_matrix",
        "model",
        "model_path",
        "output",
        "proteome_fasta",
        "script",
        "target",
        "validation_out",
    }
)


def _recorded_paths(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _PATH_KEYS | _OTHER_PATH_KEYS and isinstance(value, str):
                yield value
            if key == "sources":
                yield from (
                    s for s in (value if isinstance(value, list) else [value]) if isinstance(s, str)
                )
            if key == "inputs" and isinstance(value, dict):
                yield from value
            yield from _recorded_paths(value)
    elif isinstance(node, list):
        for value in node:
            yield from _recorded_paths(value)


def test_every_committed_path_value_is_a_fixed_point_of_relative_source():
    """Regenerating a committed output must not change a path string it records
    merely because the writer now goes through relative_source. That holds only if
    every such value already committed is a fixed point - including the relative
    paths committed with Windows separators, which relative_source keeps as given."""
    from src.provenance_paths import relative_source

    documents = _tracked_json_documents(REPO)
    if documents is None:
        pytest.skip("not a git work tree: no tracked JSON to read")
    values = [(name, v) for name, document in documents for v in _recorded_paths(document)]
    assert any("\\" in v for _, v in values), "no Windows-separator value: check is vacuous"
    assert [(name, v) for name, v in values if relative_source(v) != v] == []


def test_the_fixed_point_check_reads_each_path_key_sources_and_inputs_keys():
    document = {
        "artifact": "a.csv",
        "nested": [{"script": "s.py", "inputs": {"k.csv": "0" * 64}}],
        "sources": ["one.csv", {"url": "u"}, 3],
        "other": {"sources": "two.csv", "note": "not a path key"},
    }
    assert sorted(_recorded_paths(document)) == ["a.csv", "k.csv", "one.csv", "s.py", "two.csv"]
