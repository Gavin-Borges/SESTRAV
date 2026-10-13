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
