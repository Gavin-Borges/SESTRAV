"""`src.provenance_paths.relative_source` never yields a workstation path.

A provenance or run-metadata file is written beside the artifact it describes,
and the repo's privacy mandate forbids writing an absolute workstation path into
any repo file. The probe table below builds each input at runtime from the real
username and the real checkout (no workstation path is written literally here),
and every output must name no drive, no `Users`/`home` component, not the
current user and not the checkout, and must be a fixed point.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import sys
from pathlib import Path

import pytest

from src.provenance_paths import relative_source

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import _dataset_utils  # noqa: E402


def _username() -> str:
    try:
        return getpass.getuser() or "someone"
    except Exception:  # no login name available (some containers)
        return "someone"


USER = _username()
REPO = os.path.realpath(_ROOT)
_DRIVE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:")
_ALLOWED_URL = re.compile(r"(?:https?|ftps?|s3|gs)://", re.IGNORECASE)


def _assert_names_no_workstation_location(out: str) -> None:
    url = _ALLOWED_URL.match(out)
    if url:
        # A kept URL carries no credentials, query string or fragment.
        assert "@" not in out and "?" not in out and "#" not in out, out
        body = ""
    else:
        body = out
    components = [c.lower() for c in re.split(r"[\\/:]", body) if c]
    assert not _DRIVE.search(body), out
    assert "users" not in components and "home" not in components, out
    assert USER.lower() not in components, out
    assert "~" + USER.lower() not in components, out
    assert REPO.replace("\\", "/") not in out.replace("\\", "/"), out


# (label, input, expected output). Built at import time from the real username
# and the real checkout, so the table probes this workstation, not a stand-in.
_CASES = [
    ("C:// one-letter scheme", "C://Users/" + USER + "/secret.csv", "external:secret.csv"),
    ("file: URI", "file:///C:/Users/" + USER + "/secret.csv", "external:secret.csv"),
    (
        "external: holding a path",
        "external:C:\\Users\\" + USER + "\\secret.csv",
        "external:secret.csv",
    ),
    ("drive-relative", "C:Users\\" + USER + "\\secret.csv", "external:secret.csv"),
    ("drive-relative bare name", "C:x.csv", "external:x.csv"),
    ("bare ..", "..", "external:"),
    ("abs inside, native separators", os.path.join(REPO, "data", "x.csv"), "data/x.csv"),
    ("abs inside, forward slashes", Path(REPO, "data", "x.csv").as_posix(), "data/x.csv"),
    ("abs outside", "C:\\Users\\" + USER + "\\Downloads\\x.csv", "external:x.csv"),
    ("../x", "../x.csv", "external:x.csv"),
    ("data/../../x", "data/../../x.csv", "external:x.csv"),
    ("POSIX home", "/home/" + USER + "/x.csv", "external:x.csv"),
    ("POSIX home directory itself", "/home/" + USER, "external:"),
    ("POSIX home directory, trailing slash", "/home/" + USER + "/", "external:"),
    ("macOS home directory itself", "/Users/" + USER, "external:"),
    ("Git Bash drive mount", "/c/Users/" + USER + "/x.csv", "external:x.csv"),
    ("root-relative backslash", "\\Users\\" + USER + "\\x.csv", "external:x.csv"),
    ("UNC backslash", "\\\\server\\share\\" + USER + "\\x.csv", "external:x.csv"),
    ("UNC slash", "//server/share/" + USER + "/x.csv", "external:x.csv"),
    ("smb share", "smb://fileserver/lab/data/x.csv", "external:x.csv"),
    ("smb share under a home", "smb://fileserver/Users/" + USER, "external:"),
    ("a scheme off the allowlist alone", "smb://", "external:"),
    ("sftp", "sftp://host/home/" + USER + "/x.csv", "external:x.csv"),
    ("a URL resolved as a path", "https:/" + USER + ":pw@example.org/x.csv", "external:x.csv"),
    ("https URL", "https://example.org/a/x.csv", "https://example.org/a/x.csv"),
    ("s3 URL", "s3://bucket/key/x.csv", "s3://bucket/key/x.csv"),
    ("external: name", "external:x.csv", "external:x.csv"),
    ("empty", "", ""),
    ("tilde home", "~/x.csv", "external:x.csv"),
    ("tilde user", "~" + USER + "/x.csv", "external:x.csv"),
    ("bare tilde user", "~" + USER, "external:"),
    ("bare tilde user, trailing slash", "~" + USER + "/", "external:"),
    ("a home directory itself", "C:\\Users\\" + USER, "external:"),
    ("relative Users component", "Users/" + USER + "/x.csv", "external:x.csv"),
    ("relative backslash, kept as given", "data\\raw\\x.csv", "data\\raw\\x.csv"),
    ("relative", "data/x.csv", "data/x.csv"),
    ("relative non-escaping ..", "data/../src/x.py", "data/../src/x.py"),
]


@pytest.mark.parametrize("label, source, expected", _CASES, ids=[c[0] for c in _CASES])
def test_probe_table(label, source, expected):
    out = relative_source(source, root=REPO)
    assert out == expected
    _assert_names_no_workstation_location(out)
    assert relative_source(out, root=REPO) == out, "not a fixed point"


@pytest.mark.parametrize("source", [c[1] for c in _CASES], ids=[c[0] for c in _CASES])
def test_default_root_is_this_checkout(source):
    assert relative_source(source) == relative_source(source, root=REPO)


def test_bare_dotdot_never_takes_a_name_from_the_working_directory(tmp_path, monkeypatch):
    """Nothing about the cwd may reach the output: from a home directory's child,
    the cwd's parent is the username."""
    cwd = tmp_path / "sentinel_parent_name" / "cwd"
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    for source in ("..", "../", "../..", "./..", "data/../.."):
        out = relative_source(source, root=REPO)
        assert "sentinel_parent_name" not in out, (source, out)
        assert out == "external:", (source, out)


@pytest.mark.parametrize("scheme", ["c", "C", "x"])
def test_a_one_letter_scheme_is_a_drive_not_a_url(scheme):
    out = relative_source(scheme + "://Users/" + USER + "/k.csv", root=REPO)
    assert out == "external:k.csv"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/x",
        "HTTPS://EXAMPLE.ORG/x",
        "ftp://example.org/x",
        "ftps://example.org/x",
        "s3://bucket/x",
        "gs://bucket/x",
        "https://example.org",
        "https://example.org:8443/a/b.csv",
        "http://[::1]:8080/x",
        "s3://my_bucket-1/key",
    ],
)
def test_a_clean_allowlisted_url_is_kept_as_given(url):
    assert relative_source(url, root=REPO) == url


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            "https://" + USER + ":hunter2@example.org/a/x.csv",
            "https://example.org/a/x.csv",
        ),
        ("http://" + USER + "@example.org/x", "http://example.org/x"),
        ("ftp://" + USER + ":pw@ftp.example.org/pub/x.csv", "ftp://ftp.example.org/pub/x.csv"),
        ("ftps://a:b@host:990/x", "ftps://host:990/x"),
        ("gs://" + USER + ":pw@bucket/key/x.csv", "gs://bucket/key/x.csv"),
        ("s3://k:s@bucket/key", "s3://bucket/key"),
        ("https://example.org/x.csv?token=abc", "https://example.org/x.csv"),
        ("https://example.org/x.csv?token=abc#frag", "https://example.org/x.csv"),
        ("https://example.org/x.csv#frag?token=abc", "https://example.org/x.csv"),
        ("https://example.org?token=abc", "https://example.org"),
        ("gs://bucket/key/x.csv?token=abc", "gs://bucket/key/x.csv"),
        # An unencoded "@" in the password: the host is what follows the LAST "@".
        ("https://u:p@ss@example.org/x", "https://example.org/x"),
    ],
)
def test_an_allowlisted_url_loses_its_credentials_query_and_fragment(url, expected):
    out = relative_source(url, root=REPO)
    assert out == expected
    assert USER not in out and "token" not in out and "hunter2" not in out
    _assert_names_no_workstation_location(out)
    assert relative_source(out, root=REPO) == out, "not a fixed point"


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https:///C:/Users/" + USER + "/x.csv", "external:x.csv"),
        ("https://" + USER + ":pw@/x.csv", "external:x.csv"),
        ("https://C:/Users/" + USER + "/x.csv", "external:x.csv"),
        ("https://:8080/x.csv", "external:x.csv"),
        ("https://", "external:"),
        ("https://?token=abc", "external:"),
        ("s3:///home/" + USER + "/x.csv", "external:x.csv"),
        ("gs://" + USER + "@/x.csv", "external:x.csv"),
    ],
)
def test_an_allowlisted_url_without_a_host_is_reduced_as_a_path(url, expected):
    """An empty host, or one that is not a host name or IP literal with a numeric
    port (a drive's "C:"), is what a local path joined onto a scheme looks like."""
    out = relative_source(url, root=REPO)
    assert out == expected
    _assert_names_no_workstation_location(out)
    assert relative_source(out, root=REPO) == out, "not a fixed point"


@pytest.mark.parametrize(
    "source, expected",
    [
        # The authority after "//" is never kept as the last component.
        ("smb://" + USER + ":pw@fileserver", "external:"),
        ("sftp://" + USER + ":pw@fileserver", "external:"),
        ("file://" + USER + ":pw@fileserver", "external:"),
        ("smb://fileserver", "external:"),
        ("\\\\fileserver", "external:"),
        ("smb://" + USER + ":pw@fileserver/x.csv", "external:x.csv"),
        # Nor is a name holding ":" or "@": a user:password@host, or a host:port.
        ("external:" + USER + ":pw@fileserver", "external:"),
        ("external:fileserver:8080", "external:"),
        # A query string or fragment ends the name...
        ("smb://fileserver/share/x.csv?token=abc", "external:x.csv"),
        ("smb://fileserver/share/x.csv#frag", "external:x.csv"),
        ("file:C:/Users/" + USER + "/x.csv?token=abc", "external:x.csv"),
        ("external:x.csv?token=abc", "external:x.csv"),
        # ...and one followed by a separator may hold the name, so none is kept.
        ("smb://fileserver/x.csv?next=/a/token", "external:"),
        ("/home/" + USER + "/x.csv#frag/token", "external:"),
    ],
)
def test_the_reduction_keeps_no_authority_credentials_query_or_fragment(source, expected):
    out = relative_source(source, root=REPO)
    assert out == expected
    for leaked in ("@", "?", "#", "pw", "token", "frag", "fileserver", "8080"):
        assert leaked not in out, (source, out)
    _assert_names_no_workstation_location(out)
    assert relative_source(out, root=REPO) == out, "not a fixed point"


@pytest.mark.parametrize(
    "source, expected",
    [
        (" https://" + USER + ":pw@example.org/x.csv", "https://example.org/x.csv"),
        ("https://example.org/x.csv \n", "https://example.org/x.csv"),
        ("https://example.org/x.csv ?token=abc", "https://example.org/x.csv"),
        (" D:\\data\\x.csv", "external:x.csv"),
        ("C:\\Users\\" + USER + "\\x.csv ", "external:x.csv"),
        ("\t/home/" + USER + "/x.csv", "external:x.csv"),
        (" ~" + USER + "/x.csv", "external:x.csv"),
        (" \\\\server\\share\\x.csv", "external:x.csv"),
        (" /srv/x.csv", "external:x.csv"),
        # A component is judged without its own surrounding whitespace.
        ("data/ D:/x.csv", "external:x.csv"),
        ("data/ .. /../x.csv", "external:x.csv"),
        ("external: x.csv ", "external:x.csv"),
        # A line break can put a second path after the first.
        ("data/x.csv\n/srv/y.csv", "external:y.csv"),
        ("https://example.org/a\n/home/" + USER + "/y.csv", "external:y.csv"),
        # A clean relative path loses only the whitespace around it.
        (" data/x.csv ", "data/x.csv"),
        (" data\\raw\\x.csv\n", "data\\raw\\x.csv"),
    ],
)
def test_whitespace_and_line_breaks_hide_nothing(source, expected):
    out = relative_source(source, root=REPO)
    assert out == expected
    assert "pw" not in out and "srv" not in out and "server" not in out, out
    _assert_names_no_workstation_location(out)
    assert relative_source(out, root=REPO) == out, "not a fixed point"


# (label, input, expected output): a relative path holding an "@" anywhere, or a
# ":" before its first separator. Labelled, so no test id carries the checkout.
_USER_OR_REMOTE_HOST = [
    # A user:password@ with no scheme before it, and a remote host's path.
    ("credentials, host:path", "user:pw@host:/data/x.csv", "external:x.csv"),
    ("key:secret@bucket", "AKIAEXAMPLE:secret@bucket/x.csv", "external:x.csv"),
    ("credentials in a later component", "data/user:pw@host/x.csv", "external:x.csv"),
    # scp and rsync read [user@]host:path, so the username must not survive.
    ("scp user@host:/path", USER + "@fileserver:/srv/x.csv", "external:x.csv"),
    ("scp user@host:name", "user@host:x.csv", "external:"),
    ("scp user@host:/name", "user@host:/x.csv", "external:x.csv"),
    ("scp host:path", "fileserver:data/x.csv", "external:x.csv"),
    ("scp host:name", "host:x.csv", "external:"),
    ("user@host, backslashes", "data\\" + USER + "@fileserver\\x.csv", "external:x.csv"),
    # A plain "@" in a name: the text cannot tell it from a user at a host.
    ("plain @ in a name", "data/a@b.csv", "external:"),
    # Inside the checkout, the repo-relative form is reduced on the second pass.
    (
        "abs inside, user@host component",
        os.path.join(REPO, "data", USER + "@fileserver", "x.csv"),
        "external:x.csv",
    ),
]


@pytest.mark.parametrize(
    "label, source, expected", _USER_OR_REMOTE_HOST, ids=[c[0] for c in _USER_OR_REMOTE_HOST]
)
def test_a_relative_path_naming_a_user_or_remote_host_is_reduced(label, source, expected):
    out = relative_source(source, root=REPO)
    assert out == expected
    for leaked in ("@", "pw", "secret", "host", "fileserver", "srv", "bucket"):
        assert leaked not in out, (source, out)
    assert USER not in out, (source, out)
    _assert_names_no_workstation_location(out)
    assert relative_source(out, root=REPO) == out, "not a fixed point"


@pytest.mark.parametrize(
    "source",
    [
        "data/x.csv",
        "data\\raw\\ctl_all.csv",
        "src/provenance_paths.py",
        "data/raw/run-2026.10.12_v5/x.csv",
        # A ":" after the first separator is not where scp or rsync reads a host.
        "data/host:x.csv",
    ],
)
def test_a_relative_path_naming_no_user_or_remote_host_is_kept_as_given(source):
    assert relative_source(source, root=REPO) == source


def test_an_inside_path_a_second_pass_would_change_is_reduced(tmp_path):
    """A directory whose name starts with a space would give a padded
    repo-relative form, which a second pass would trim: it is reduced instead."""
    repo = tmp_path / "repo"
    padded = repo / " padded"
    padded.mkdir(parents=True)
    (padded / "k.csv").write_text("x\n")
    out = relative_source(str(padded / "k.csv"), root=repo)
    assert out == "external:k.csv"
    assert relative_source(out, root=repo) == out


def test_a_scheme_off_the_allowlist_is_not_kept():
    """`file://` is the reason the allowlist exists: it is a workstation path."""
    assert relative_source("file://localhost/C:/Users/" + USER + "/k.csv", root=REPO) == (
        "external:k.csv"
    )
    assert relative_source("nfs://filer/export/k.csv", root=REPO) == "external:k.csv"


def test_an_absolute_path_inside_a_given_root_is_relative_to_that_root(tmp_path):
    repo = tmp_path / "repo"
    (repo / "data" / "raw").mkdir(parents=True)
    target = repo / "data" / "raw" / "vdjdb.txt"
    target.write_text("x\n")
    assert relative_source(str(target.resolve()), root=repo) == "data/raw/vdjdb.txt"
    assert relative_source(target.resolve(), root=repo) == "data/raw/vdjdb.txt"


def test_an_absolute_path_outside_the_root_keeps_only_its_file_name(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "elsewhere" / "vdjdb.txt"
    outside.parent.mkdir()
    outside.write_text("x\n")
    assert relative_source(str(outside.resolve()), root=repo) == "external:vdjdb.txt"


def _directory_link(link: Path, target: Path) -> None:
    """A symlink, or on Windows without the symlink privilege a junction."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError) as exc:
        if sys.platform != "win32":
            pytest.skip(f"cannot create a directory symlink here: {exc}")
    import _winapi  # Windows only; a junction needs no privilege

    try:
        _winapi.CreateJunction(str(target), str(link))
    except OSError as exc:
        pytest.skip(f"cannot create a directory symlink or junction here: {exc}")


def test_a_link_inside_the_root_that_leads_outside_is_external(tmp_path):
    """Inside-or-outside is judged on resolved paths, so a link cannot smuggle an
    outside file in as a repo-relative path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "k.csv").write_text("x\n")
    _directory_link(repo / "linked", outside)

    assert relative_source(str(repo / "linked" / "k.csv"), root=repo) == "external:k.csv"


def test_a_link_outside_the_root_that_leads_inside_is_repo_relative(tmp_path):
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "data" / "k.csv").write_text("x\n")
    _directory_link(tmp_path / "alias", repo)

    assert relative_source(str(tmp_path / "alias" / "data" / "k.csv"), root=repo) == "data/k.csv"


@pytest.mark.parametrize("value", [None, 7, {"url": "https://example.org/a.zip"}, ["a"]])
def test_a_non_path_value_is_returned_unchanged(value):
    assert relative_source(value, root=REPO) == value


# --- write_provenance, the shared ingest writer -----------------------------


def _sidecar_of(output: Path) -> dict:
    with open(os.path.splitext(str(output))[0] + "_provenance.json", encoding="utf-8") as fh:
        return json.load(fh)


def test_write_provenance_records_sources_through_relative_source(tmp_path):
    output = tmp_path / "artifact.csv"
    output.write_text("peptide\nGILGFVFTL\n")
    outside = tmp_path / "vdjdb.txt"
    inside = os.path.join(REPO, "data", "immunogenicity_dataset_v5.csv")
    sources = [str(outside), Path(inside), "data/iedb", "https://example.org/a", {"k": "v"}, None]

    _dataset_utils.write_provenance(str(output), sources=sources, row_count=1)

    assert _sidecar_of(output)["sources"] == [
        "external:vdjdb.txt",
        "data/immunogenicity_dataset_v5.csv",
        "data/iedb",
        "https://example.org/a",
        {"k": "v"},
        None,
    ]


def test_write_provenance_sanitizes_a_tuple_of_sources(tmp_path):
    output = tmp_path / "artifact.csv"
    output.write_text("peptide\nGILGFVFTL\n")
    sources = (str(tmp_path / "a.tsv"), "https://" + USER + ":pw@example.org/b.zip?x=1")

    _dataset_utils.write_provenance(str(output), sources=sources, row_count=1)

    assert _sidecar_of(output)["sources"] == ["external:a.tsv", "https://example.org/b.zip"]


def test_write_provenance_sanitizes_a_single_source_given_bare(tmp_path):
    output = tmp_path / "artifact.csv"
    output.write_text("peptide\nGILGFVFTL\n")
    _dataset_utils.write_provenance(str(output), sources=str(tmp_path / "in.tsv"), row_count=1)
    assert _sidecar_of(output)["sources"] == "external:in.tsv"


def test_a_real_write_provenance_caller_records_no_workstation_path(tmp_path):
    """End to end through scripts/ingest_tsnadb.py, whose call passes its input path."""
    from ingest_tsnadb import ingest_tsnadb

    tsv = tmp_path / "tsna.tsv"
    tsv.write_text("Peptide\tHLA\nSIINFEKL\tHLA-A*02:01\n")
    output = tmp_path / "out.csv"
    schema = os.path.join(REPO, "data", "immunogenicity_dataset_v4_schema.json")

    ingest_tsnadb(str(tsv.resolve()), str(output), schema)

    prov = _sidecar_of(output)
    assert prov["sources"] == ["external:tsna.tsv"]
    for value in prov["sources"]:
        _assert_names_no_workstation_location(value)
