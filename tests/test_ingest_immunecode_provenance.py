"""The provenance sidecar written by scripts/ingest_immunecode.py names no absolute path.

`<output>_provenance.json` records in `sources` where the rows came from. These
tests pin how each kind of source is recorded:

* the auto-download, by MIRA_DOWNLOAD_URL, whose final segment is the archive
  name, and never by the path of the archive inside the deleted temporary
  directory it was downloaded to;
* an --input inside the repo, by its repo-relative POSIX path, whether it is
  given absolute or relative to the working directory;
* an --input outside the repo, by its basename alone, prefixed "external:".

In each of those tests every string in the written JSON, keys included, is
also checked for an absolute path, a drive letter, the home directory and the
temporary directory. No test opens a socket: _download_mira is replaced by a
fake that writes the archive into the directory main() hands it. No absolute
path is written literally; each is built at runtime. The tests that run main()
point the repo root at a fake one, under tmp_path except in the other-drive test.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import ingest_immunecode as ingest

ZIP_NAME = "ImmuneCODE-MIRA-Release002.1.zip"
CSV_NAME = "peptide-detail-cov2.csv"
MIRA_CSV = (
    "Peptide,HLA Restrictions,Amino Acids\n"
    "YLQPRTFLL,HLA-A*02:01,\n"
    "LTDEMIAQY,HLA-A*02:01,CASSIRSSYEQYF\n"
    "KIADYNYKL,HLA-B*07:02,\n"
)

# A drive-letter path anywhere in a string: "C:\..." or "C:/...". The lookbehind
# keeps "https://" from matching on its "s:/".
_DRIVE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")


def _write_zip(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(CSV_NAME, MIRA_CSV)


def _write_csv(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MIRA_CSV)


def _strings(node):
    """Yield every string in a parsed JSON document, keys included."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(key)
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _forms(path) -> list[str]:
    """A path as written natively and with forward slashes, lower-cased.

    A filesystem root yields nothing: "/" as a home directory would otherwise
    match the slashes of the URL.
    """
    text = os.path.realpath(str(path)).lower()
    if os.path.dirname(text) == text:
        return []
    return [text, text.replace("\\", "/")]


def _assert_no_workstation_path(data, *forbidden) -> None:
    banned = [os.path.expanduser("~"), tempfile.gettempdir(), *forbidden]
    for value in _strings(data):
        assert not os.path.isabs(value), f"absolute path in sidecar: {value!r}"
        assert not value.startswith(("/", "\\")), f"rooted path in sidecar: {value!r}"
        assert not _DRIVE_PATH.search(value), f"drive letter in sidecar: {value!r}"
        for path in banned:
            for form in _forms(path):
                assert form not in value.lower(), f"{path!r} in sidecar: {value!r}"


def _run(argv, out_dir):
    """Run main() writing into out_dir; return its exit code and the parsed sidecar."""
    output = out_dir / "negatives.csv"
    code = ingest.main([*argv, "--output", str(output)])
    with open(out_dir / "negatives_provenance.json", encoding="utf-8") as handle:
        return code, json.load(handle)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repo root under tmp_path, which the script treats as the repo."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(ingest, "_REPO_ROOT", str(root), raising=False)
    return root


@pytest.fixture
def out_dir(tmp_path):
    path = tmp_path / "out"
    path.mkdir()
    return path


def test_the_repo_root_is_the_checkout_that_holds_the_script():
    here = os.path.dirname(os.path.abspath(__file__))
    assert os.path.realpath(ingest._REPO_ROOT) == os.path.realpath(os.path.dirname(here))


def test_auto_download_records_the_url_not_the_temporary_archive_path(repo, out_dir, monkeypatch):
    handed: list[str] = []

    def fake_download(dest_dir: str) -> str:
        handed.append(dest_dir)
        path = os.path.join(dest_dir, ZIP_NAME)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(CSV_NAME, MIRA_CSV)
        return path

    monkeypatch.setattr(ingest, "_download_mira", fake_download)

    code, data = _run([], out_dir)

    assert code == 0
    assert len(handed) == 1
    assert ingest.MIRA_DOWNLOAD_URL.rsplit("/", 1)[1] == ZIP_NAME
    assert data["sources"] == [ingest.MIRA_DOWNLOAD_URL]
    _assert_no_workstation_path(data, handed[0], out_dir)


@pytest.mark.parametrize(
    ("relative", "write"),
    [(("data", "raw", ZIP_NAME), _write_zip), (("data", "raw", CSV_NAME), _write_csv)],
    ids=["zip", "csv"],
)
def test_an_absolute_input_inside_the_repo_is_recorded_repo_relative_posix(
    repo, out_dir, relative, write
):
    source = repo.joinpath(*relative)
    write(source)

    code, data = _run(["--input", str(source)], out_dir)

    assert code == 0
    assert data["sources"] == ["/".join(relative)]
    _assert_no_workstation_path(data, repo, out_dir)


def test_a_cwd_relative_input_inside_the_repo_is_recorded_from_the_repo_root(
    repo, out_dir, monkeypatch
):
    _write_csv(repo / "data" / "raw" / CSV_NAME)
    monkeypatch.chdir(repo / "data")

    # os.path.join spells it with this platform's separator: a backslash on Windows.
    code, data = _run(["--input", os.path.join("raw", CSV_NAME)], out_dir)

    assert code == 0
    assert data["sources"] == [f"data/raw/{CSV_NAME}"]
    _assert_no_workstation_path(data, repo, out_dir)


def test_a_dotted_spelling_of_the_repo_root_still_contains_its_files(repo, out_dir, monkeypatch):
    source = repo / "data" / CSV_NAME
    _write_csv(source)
    monkeypatch.setattr(ingest, "_REPO_ROOT", os.path.join(str(repo), "data", os.pardir))

    code, data = _run(["--input", str(source)], out_dir)

    assert code == 0
    assert data["sources"] == [f"data/{CSV_NAME}"]
    _assert_no_workstation_path(data, repo, out_dir)


@pytest.mark.parametrize(
    ("name", "write"), [(ZIP_NAME, _write_zip), (CSV_NAME, _write_csv)], ids=["zip", "csv"]
)
@pytest.mark.parametrize("parent", ["elsewhere", "repo-sibling"])
def test_an_input_outside_the_repo_records_only_its_basename_marked_external(
    repo, out_dir, tmp_path, parent, name, write
):
    # "repo-sibling" shares the repo root's name as a string prefix, so a bare
    # startswith() test would wrongly place it inside the repo.
    source = tmp_path / parent / name
    write(source)

    code, data = _run(["--input", str(source)], out_dir)

    assert code == 0
    assert data["sources"] == [f"external:{name}"]
    _assert_no_workstation_path(data, source.parent, out_dir)


@pytest.mark.skipif(os.name != "nt", reason="only Windows paths carry a drive letter")
def test_an_input_on_another_drive_from_the_repo_is_recorded_external(
    tmp_path, out_dir, monkeypatch
):
    # No path on one drive is relative to a root on another: os.path.commonpath
    # and os.path.relpath both raise ValueError. The repo root needs no real
    # drive behind it for that, so pick any letter that is not tmp_path's.
    drive = os.path.splitdrive(str(tmp_path))[0].upper()
    other = "Y:" if drive == "Z:" else "Z:"
    monkeypatch.setattr(ingest, "_REPO_ROOT", other + os.sep + "repo", raising=False)
    source = tmp_path / "elsewhere" / CSV_NAME
    _write_csv(source)

    code, data = _run(["--input", str(source)], out_dir)

    assert code == 0
    assert data["sources"] == [f"external:{CSV_NAME}"]
    _assert_no_workstation_path(data, source.parent, out_dir)


def test_every_other_sidecar_field_is_unchanged(repo, out_dir):
    source = repo / "data" / CSV_NAME
    _write_csv(source)

    code, data = _run(["--input", str(source)], out_dir)

    assert code == 0
    assert set(data) == {
        "output",
        "sources",
        "row_count",
        "generated_utc",
        "git_sha",
        "sha256",
        "database",
        "release",
        "reference_pmid",
        "label",
        "virus",
        "negative_origin",
        "unique_peptides",
        "unique_alleles",
        "allele_distribution",
    }
    assert data["output"] == "negatives.csv"
    assert data["row_count"] == 2
    assert {k: data[k] for k in ("database", "release", "reference_pmid", "label")} == {
        "database": "ImmuneCODE MIRA",
        "release": "002.1",
        "reference_pmid": "32665443",
        "label": 0,
    }
    assert data["virus"] == "SARS-CoV-2"
    assert data["negative_origin"] == "immunecode_mira"
    assert data["unique_peptides"] == 2
    assert data["unique_alleles"] == 2
    assert data["allele_distribution"] == {"HLA-A*02:01": 1, "HLA-B*07:02": 1}
