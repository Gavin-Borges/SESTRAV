"""`sestrav predict` must fail before Stage 1 on preconditions it can check up front.

Two of them used to surface only after Stages 1 to 3, as a raw traceback:

* MHCflurry's presentation-model data is a separate download that a fresh install
  does not include (none of the repository's Dockerfiles fetches it), so Stage 2
  died with mhcflurry's own "Missing MHCflurry downloadable file" RuntimeError.
* With freeze mode and conformal intervals both on, a missing conformal calibrator
  was discovered only inside Stage 4, after a full MHCflurry pass.

Absent model data is simulated by pointing mhcflurry's resolved downloads directory
at an empty temporary directory, so the real mhcflurry resolver is exercised rather
than a stand-in for it. The stage functions are stubbed to record calls, which is how
"before Stage 1" is asserted: none of them may run.
"""

from __future__ import annotations

import os

import sys

from pathlib import Path

import pandas as pd
import pytest

from src import cli

# Mirrors src/cli.py's base message. It changed when the hint stopped recommending a
# bare `mhcflurry-downloads fetch`, which was an arbitrary-write risk: a test that pins
# a message recommending an unsafe command is expected to change when that stops.
FETCH_MESSAGE = (
    "MHCflurry model data is absent, and Stage 2 cannot run without it. "
    "Install it with the hash-verified two-step route, which is the control on every "
    "mhcflurry version and OS: first `python scripts/fetch_verified_mhcflurry.py "
    "--url <mhcflurry_model_archive_url> --sha256 <mhcflurry_model_archive_sha256> "
    "--output-dir DIR`, taking both values from config.yaml, then "
    "`mhcflurry-downloads fetch models_class1_presentation "
    "--already-downloaded-dir DIR`. Do not run a bare "
    "`mhcflurry-downloads fetch`: it unpacks about 135 MB with no integrity check, and a "
    "tampered archive can write outside the target directory, so it is an arbitrary-write "
    "risk rather than just an unverified download."
)


def _py313_note(version, downloads_dir, custom_downloads_dir=False):
    head = (
        f" On Python 3.13 and later the mhcflurry {version} downloader cannot run, "
        "because it imports the 'pipes' module that Python 3.13 removed. Step one "
        "above is unaffected: the verified fetcher imports no mhcflurry and runs on "
        "any interpreter. Only the second command needs a Python 3.11 or 3.12 "
        "environment that has the same mhcflurry "
        "version, "
    )
    if custom_downloads_dir:
        return head + (
            "then copy the models_class1_presentation directory it downloads into "
            f"{downloads_dir}, the directory this one reads, and retry."
        )
    data_dir, release = os.path.split(os.path.normpath(downloads_dir))
    return head + (
        f"with MHCFLURRY_DATA_DIR set to {data_dir} and "
        f"MHCFLURRY_DOWNLOADS_CURRENT_RELEASE set to {release}, which together "
        f"name {downloads_dir}, the directory this one reads, then retry."
    )


# RE-POINTED, not relaxed. This literal used to end "under the working directory",
# which was an accurate description of a vulnerability rather than of a contract:
# the DEFAULT conformal calibrator was whatever models/v5/conformal_calibrator.joblib
# the process working directory happened to supply, and it was verified against the
# checksum manifest sitting beside it, so a planted pickle shipped with a planted
# manifest verified clean. The default is now anchored to the installation root and
# the refusal says so.
CALIBRATOR_MESSAGE = (
    "freeze mode requires a conformal calibrator and none was found: no "
    "--conformal-calibrator was given, and there is no conformal_calibrator.joblib "
    "beside the model or at models/v5/conformal_calibrator.joblib under the SESTRAV "
    "installation root. The default is deliberately NOT searched for under the "
    "working directory. Pass --conformal-calibrator PATH, --no-conformal to run "
    "without intervals, or --no-freeze-mode to let Stage 4 continue without them."
)


class _SysWithVersion:
    """Stand-in for ``src.cli.sys`` that reports a chosen ``version_info``.

    Only the CLI module's reference is replaced, so the interpreter and pytest keep
    the real ``sys``; every other attribute (``stderr`` included) is forwarded.
    """

    def __init__(self, version_info):
        self.version_info = version_info

    def __getattr__(self, name):
        import sys

        return getattr(sys, name)


@pytest.fixture
def mhcflurry_downloads(monkeypatch, tmp_path):
    """Point mhcflurry at an empty downloads directory; return a helper to populate it."""
    import mhcflurry.downloads as downloads

    downloads_dir = tmp_path / "mhcflurry_downloads"
    downloads_dir.mkdir()
    monkeypatch.setattr(downloads, "_DOWNLOADS_DIR", str(downloads_dir))
    monkeypatch.setattr(downloads, "_MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR", None)

    def install():
        (downloads_dir / "models_class1_presentation" / "models").mkdir(parents=True)

    install.downloads_dir = str(downloads_dir)
    return install


@pytest.fixture
def mhcflurry_not_importable(monkeypatch):
    """Make ``import mhcflurry.downloads`` raise ImportError, as with no install."""
    import sys

    monkeypatch.setitem(sys.modules, "mhcflurry.downloads", None)


def _fake_mhcflurry_version(monkeypatch, version):
    """Report ``version`` as the installed mhcflurry; None means not installed."""
    import importlib.metadata

    real_version = importlib.metadata.version

    def _version(name):
        if name != "mhcflurry":
            return real_version(name)
        if version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(importlib.metadata, "version", _version)


@pytest.fixture
def stage_calls(monkeypatch):
    """Stub all four stages; return the list of stage names that were called."""
    import functions.stage1_peptide_generation as s1
    import functions.stage2_mhc_binding_prediction as s2
    import functions.stage3_tcr_feature_extraction as s3
    import functions.stage4_immunogenicity_scoring as s4

    calls: list[str] = []
    frame = pd.DataFrame({"peptide": ["CLGGLLTMV"], "immunogenicity_score": [0.5]})

    def _stub(name, result):
        def _called(*args, **kwargs):
            calls.append(name)
            return result

        return _called

    monkeypatch.setattr(s1, "generate_peptides", _stub("stage1", frame))
    monkeypatch.setattr(s2, "predict_binding", _stub("stage2", frame))
    monkeypatch.setattr(s3, "extract_tcr_features", _stub("stage3", frame))
    monkeypatch.setattr(s4, "score_immunogenicity", _stub("stage4", (frame, None)))
    return calls


def _predict_argv(tmp_path, *extra):
    fasta = tmp_path / "input.fasta"
    model = tmp_path / "model.joblib"
    fasta.write_text(">protein\nACDEFGHIK\n", encoding="utf-8")
    model.write_bytes(b"stub")
    return [
        "predict",
        "--fasta",
        str(fasta),
        "--model",
        str(model),
        "--output",
        str(tmp_path / "out"),
        *extra,
    ]


# ---------------------------------------------------------------------------
# sestrav info
# ---------------------------------------------------------------------------


def test_info_reports_absent_model_data(mhcflurry_downloads, capsys):
    assert cli.main(["info"]) == 0

    assert "  mhcflurry data  : absent\n" in capsys.readouterr().out


def test_info_reports_present_model_data(mhcflurry_downloads, capsys):
    """The other direction, so a hardcoded "absent" cannot pass."""
    mhcflurry_downloads()

    assert cli.main(["info"]) == 0

    assert "  mhcflurry data  : present\n" in capsys.readouterr().out


def test_info_without_mhcflurry_does_not_claim_data_is_absent(mhcflurry_not_importable, capsys):
    """Unknowable is not "absent": info must still exit 0 and say why it cannot tell."""
    assert cli.main(["info"]) == 0

    assert "  mhcflurry data  : unknown (mhcflurry not importable)\n" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# sestrav predict: absent MHCflurry model data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version_info", "mhcflurry_version", "expect_note"),
    [
        # The interpreter gate: the pipes module exists before 3.13.
        ((3, 12, 9), "2.2.1", False),
        ((3, 13, 0), "2.2.1", True),
        ((3, 14, 0), "2.2.1", True),
        # The mhcflurry gate: 2.1.0 is the declared floor, and every release from it
        # to 2.2.1 imports pipes; a later release may not, so it gets no note.
        ((3, 13, 0), "2.1.0", True),
        ((3, 13, 0), "2.2.2", False),
        ((3, 13, 0), "3.0", False),
        ((3, 13, 0), None, False),
    ],
)
def test_predict_without_model_data_fails_before_stage_one(
    monkeypatch,
    mhcflurry_downloads,
    stage_calls,
    tmp_path,
    capsys,
    version_info,
    mhcflurry_version,
    expect_note,
):
    """Exit 1, one stderr line, no traceback, no stage run, no output directory.

    The Python 3.13 note appears only on 3.13+ AND with an mhcflurry no newer than
    2.2.1, and names the installed version and the directory to fetch into.
    """
    monkeypatch.setattr(cli, "sys", _SysWithVersion(version_info))
    monkeypatch.delenv("MHCFLURRY_DOWNLOADS_DIR", raising=False)
    _fake_mhcflurry_version(monkeypatch, mhcflurry_version)

    rc = cli.main(_predict_argv(tmp_path, "--no-freeze-mode"))
    err = capsys.readouterr().err

    assert stage_calls == [], f"stages ran despite absent model data: {stage_calls}"
    assert rc == 1
    note = _py313_note(mhcflurry_version, mhcflurry_downloads.downloads_dir)
    expected = FETCH_MESSAGE + (note if expect_note else "")
    assert err == f"sestrav predict: error: {expected}\n"
    assert "Traceback" not in err
    assert not (tmp_path / "out").exists(), "a refused run created its output directory"


def test_predict_py313_note_with_a_custom_downloads_dir_says_copy(
    monkeypatch, mhcflurry_downloads, stage_calls, tmp_path, capsys
):
    """With MHCFLURRY_DOWNLOADS_DIR set, mhcflurry 2.2.1 refuses to fetch at all.

    Its fetch raises "No release defined" when that variable is set, and --release
    then fails looking up a None release, so re-exporting the variable elsewhere is no
    route. The note must say to copy the downloaded directory in instead.
    """
    monkeypatch.setattr(cli, "sys", _SysWithVersion((3, 13, 0)))
    monkeypatch.setenv("MHCFLURRY_DOWNLOADS_DIR", mhcflurry_downloads.downloads_dir)
    _fake_mhcflurry_version(monkeypatch, "2.2.1")

    rc = cli.main(_predict_argv(tmp_path, "--no-freeze-mode"))
    err = capsys.readouterr().err

    assert rc == 1
    note = _py313_note("2.2.1", mhcflurry_downloads.downloads_dir, custom_downloads_dir=True)
    assert err == f"sestrav predict: error: {FETCH_MESSAGE}{note}\n"
    assert "MHCFLURRY_DOWNLOADS_DIR set to" not in err


def test_py313_note_variables_resolve_to_the_named_directory(monkeypatch, tmp_path):
    """The two variables the note prints must bring mhcflurry to the directory it names.

    Checked against mhcflurry's own configure(), not against this module's reading of
    it: the note is built for a directory laid out as <data dir>/<current release>,
    then those two values are exported and configure() must resolve exactly there.
    """
    import mhcflurry.downloads as downloads

    release = downloads.get_downloads_metadata()["current-release"]
    downloads_dir = str(tmp_path / "mhcflurry-data" / release)
    monkeypatch.setattr(cli, "sys", _SysWithVersion((3, 13, 0)))
    monkeypatch.delenv("MHCFLURRY_DOWNLOADS_DIR", raising=False)
    _fake_mhcflurry_version(monkeypatch, "2.2.1")
    note = cli._mhcflurry_model_data_message(downloads_dir)

    data_dir, printed_release = os.path.split(os.path.normpath(downloads_dir))
    assert f"MHCFLURRY_DATA_DIR set to {data_dir} and " in note
    assert f"MHCFLURRY_DOWNLOADS_CURRENT_RELEASE set to {printed_release}," in note

    monkeypatch.setattr(downloads, "_DOWNLOADS_DIR", downloads._DOWNLOADS_DIR)
    monkeypatch.setattr(downloads, "_CURRENT_RELEASE", downloads._CURRENT_RELEASE)
    monkeypatch.setenv("MHCFLURRY_DATA_DIR", data_dir)
    monkeypatch.setenv("MHCFLURRY_DOWNLOADS_CURRENT_RELEASE", printed_release)
    downloads.configure()
    assert os.path.normpath(downloads.get_downloads_dir()) == os.path.normpath(downloads_dir)


def test_predict_without_mhcflurry_says_so_not_fetch(
    mhcflurry_not_importable, stage_calls, tmp_path, capsys
):
    """No mhcflurry means no fetch command either, so the fetch advice would be wrong."""
    rc = cli.main(_predict_argv(tmp_path, "--no-freeze-mode"))
    err = capsys.readouterr().err

    assert stage_calls == [], f"stages ran without mhcflurry: {stage_calls}"
    assert rc == 1
    assert err.startswith(
        "sestrav predict: error: mhcflurry is not installed or cannot be imported ("
    )
    assert err.endswith(
        "). It is a core dependency of SESTRAV, so reinstall SESTRAV with its "
        "dependencies, then retry.\n"
    )
    assert "mhcflurry-downloads fetch" not in err
    assert "Traceback" not in err


def test_predict_with_model_data_reaches_the_stages(
    mhcflurry_downloads, stage_calls, tmp_path, capsys
):
    """Present data must not be refused, or the precheck would block every real run."""
    mhcflurry_downloads()

    rc = cli.main(_predict_argv(tmp_path, "--no-freeze-mode", "--no-conformal"))
    capsys.readouterr()

    assert rc == 0
    assert stage_calls == ["stage1", "stage2", "stage3", "stage4"]


# ---------------------------------------------------------------------------
# sestrav predict: conformal calibrator (B10)
# ---------------------------------------------------------------------------


@pytest.fixture
def no_canonical_calibrator(monkeypatch, tmp_path):
    """Make the installed default calibrator absent, and plant a decoy in the cwd.

    RE-POINTED, and STRONGER than what it replaced. It used to chdir into an EMPTY
    directory, because that was all it took to make the default unresolvable: the
    default was the relative path models/v5/conformal_calibrator.joblib, so an
    empty cwd meant no calibrator. That is exactly the defect - the artifact handed
    to joblib.load was a function of the process working directory, and the
    checksum manifest beside it was trusted as its anchor, so an attacker supplied
    both. The default is now anchored to the installation root, where the artifact
    is gitignored and so present or absent depending on the checkout; the fixture
    repoints the module constant instead, which is checkout-independent.

    The cwd is still switched, and now carries a DECOY calibrator plus a manifest
    that validates it. Every test using this fixture therefore also asserts that
    the decoy is ignored.
    """
    import functions.stage4_immunogenicity_scoring as s4

    import hashlib
    import json

    workdir = tmp_path / "cwd"
    decoy = workdir / "models" / "v5" / "conformal_calibrator.joblib"
    decoy.parent.mkdir(parents=True)
    payload = b"decoy-stub-not-a-real-pickle"
    decoy.write_bytes(payload)
    (decoy.parent / "model_artifact_checksums.json").write_text(
        json.dumps(
            {
                "generated_utc": "2026-01-01T00:00:00+00:00",
                "artifacts": {
                    decoy.name: {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(
        s4,
        "DEFAULT_CONFORMAL_CALIBRATOR",
        str(tmp_path / "absent_install_root" / "models" / "v5" / "conformal_calibrator.joblib"),
    )
    return decoy


def test_freeze_mode_without_calibrator_fails_before_stage_one(
    mhcflurry_downloads, stage_calls, no_canonical_calibrator, tmp_path, capsys
):
    """Refuse, even though a calibrator sits at models/v5/ in the working directory.

    The decoy the fixture plants is what the old default would have selected and
    deserialized, so the refusal is now also the proof that it is ignored.
    """
    mhcflurry_downloads()
    assert no_canonical_calibrator.is_file(), "the decoy was not planted"

    rc = cli.main(_predict_argv(tmp_path, "--freeze-mode", "--conformal"))
    err = capsys.readouterr().err

    assert stage_calls == [], (
        f"stages ran before the missing calibrator was reported: {stage_calls}"
    )
    assert rc == 1
    assert err == f"sestrav predict: error: {CALIBRATOR_MESSAGE}\n"
    assert "Traceback" not in err
    assert not (tmp_path / "out").exists(), "a refused run created its output directory"


def test_missing_explicit_calibrator_fails_before_stage_one(
    mhcflurry_downloads, stage_calls, tmp_path, capsys
):
    """Stage 4 refuses a missing explicit path even without freeze mode."""
    mhcflurry_downloads()
    missing = str(tmp_path / "absent_calibrator.joblib")

    rc = cli.main(
        _predict_argv(
            tmp_path, "--no-freeze-mode", "--conformal", "--conformal-calibrator", missing
        )
    )
    err = capsys.readouterr().err

    assert stage_calls == [], (
        f"stages ran before the missing calibrator was reported: {stage_calls}"
    )
    assert rc == 1
    assert err == (
        f"sestrav predict: error: conformal calibrator not found: {missing!r}. Pass an "
        "existing --conformal-calibrator path, or --no-conformal to run without intervals.\n"
    )
    assert not (tmp_path / "out").exists(), "a refused run created its output directory"


def test_calibrator_beside_the_model_satisfies_freeze_mode(
    mhcflurry_downloads, stage_calls, no_canonical_calibrator, tmp_path, capsys
):
    """The precheck must find what Stage 4's resolver finds, not refuse it.

    Also pins that an explicitly NAMED path still wins: --model was given, so the
    calibrator beside it is the caller's own choice and outranks both the installed
    default and the decoy the fixture plants in the working directory.
    """
    import functions.stage4_immunogenicity_scoring as s4

    mhcflurry_downloads()
    argv = _predict_argv(tmp_path, "--freeze-mode", "--conformal")
    beside = tmp_path / "conformal_calibrator.joblib"
    beside.write_bytes(b"stub")

    rc = cli.main(argv)
    capsys.readouterr()

    assert rc == 0
    assert stage_calls == ["stage1", "stage2", "stage3", "stage4"]
    assert s4._resolve_conformal_path(str(tmp_path), None) == str(beside)


def test_missing_default_calibrator_without_freeze_mode_still_runs(
    mhcflurry_downloads, stage_calls, no_canonical_calibrator, tmp_path, capsys
):
    """Without freeze mode Stage 4 only warns, so the precheck must not refuse either."""
    import functions.stage4_immunogenicity_scoring as s4

    mhcflurry_downloads()

    rc = cli.main(_predict_argv(tmp_path, "--no-freeze-mode", "--conformal"))
    capsys.readouterr()

    assert rc == 0
    assert stage_calls == ["stage1", "stage2", "stage3", "stage4"]
    # The run proceeding is not the interesting half: it would proceed on the
    # planted decoy too. What must hold is that nothing was resolved at all.
    assert s4._resolve_conformal_path(str(tmp_path), None) is None


# ---------------------------------------------------------------------------
# main(): the handler is narrow
# ---------------------------------------------------------------------------


def test_main_does_not_swallow_other_errors(monkeypatch, mhcflurry_downloads, tmp_path, capsys):
    """Only the precondition type is caught; a genuine bug keeps its traceback.

    This one passes on the unfixed source too, by design: it pins that the new
    handler did not become a blanket ``except Exception``.
    """
    import functions.stage1_peptide_generation as s1

    mhcflurry_downloads()

    def _boom(*args, **kwargs):
        raise ValueError("a genuine bug")

    monkeypatch.setattr(s1, "generate_peptides", _boom)

    with pytest.raises(ValueError, match="a genuine bug"):
        cli.main(_predict_argv(tmp_path, "--no-freeze-mode", "--no-conformal"))
    capsys.readouterr()


# ---------------------------------------------------------------------------
# An import that fails with OSError rather than ImportError
# ---------------------------------------------------------------------------
# Importing mhcflurry.downloads imports torch, and torch raises OSError, not
# ImportError, when a shared library fails to load (WinError 126 on Windows).
# cmd_info's own `import mhcflurry` and `import torch` have caught the pair for
# exactly that reason since before this module existed, so the case is one the
# surrounding code already expects. The helper caught ImportError alone, so the
# OSError escaped as a raw traceback and cut the info report off part way.


@pytest.fixture
def mhcflurry_import_raises_oserror(monkeypatch):
    """Make any ``import mhcflurry...`` raise OSError, as a torch DLL failure does."""
    import builtins
    import sys

    for name in [m for m in sys.modules if m == "mhcflurry" or m.startswith("mhcflurry.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "mhcflurry" or name.startswith("mhcflurry."):
            raise OSError("[WinError 126] The specified module could not be found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


def test_info_survives_an_oserror_from_the_mhcflurry_import(
    mhcflurry_import_raises_oserror, capsys
):
    """info must complete its whole report, not abort at the data line.

    Before the fix this printed "mhcflurry : not installed" and then raised
    OSError, so every line after it - torch, CUDA, the active config - never
    appeared and the command exited on a traceback.
    """
    assert cli.main(["info"]) == 0

    out = capsys.readouterr().out
    assert "  mhcflurry data  : unknown (mhcflurry not importable)\n" in out
    assert "Traceback" not in out
    # Lines that come AFTER the data line: their absence is how the abort showed.
    assert "  torch           : " in out, f"the report stopped at the data line:\n{out}"


def test_predict_survives_an_oserror_from_the_mhcflurry_import(
    stage_calls, mhcflurry_import_raises_oserror, tmp_path, capsys
):
    """predict must refuse with the precondition message, not a traceback."""
    # stage_calls is listed first so the stage modules are imported, and stubbed, before
    # the import patch goes in: stage 2 imports mhcflurry at module level, so the other
    # order fails at setup whenever no earlier test in the process imported stage 2.
    rc = cli.main(_predict_argv(tmp_path, "--no-freeze-mode"))
    err = capsys.readouterr().err

    assert stage_calls == [], f"stages ran despite an unimportable mhcflurry: {stage_calls}"
    assert rc == 1
    assert err.startswith(
        "sestrav predict: error: mhcflurry is not installed or cannot be imported ("
    )
    assert "Traceback" not in err
    assert "mhcflurry-downloads fetch" not in err
# --- SX-U9: the hint must name the hash-verified route, and must stay runnable --------
#
# The old hint said to run a bare `mhcflurry-downloads fetch`. That is an ARBITRARY-WRITE
# risk, not merely an unverified download: mhcflurry's extraction was measured to write
# outside its target for crafted member names on Windows and Linux, and the current
# release still escapes on Windows for a backslash-separated `..`. Upgrading mhcflurry is
# therefore NOT the control; the two-step verified route is, on every version and OS.


def _hint() -> str:
    return cli._MHCFLURRY_VERIFIED_FETCH_HINT


def test_the_hint_names_both_steps_of_the_verified_route() -> None:
    hint = _hint()
    assert "scripts/fetch_verified_mhcflurry.py" in hint, hint
    assert "--already-downloaded-dir" in hint, (
        "step two must point mhcflurry at the already-verified directory, or it "
        f"re-downloads unverified bytes: {hint}"
    )
    # Step two names mhcflurry's own command, so the string is present by design. What
    # must NOT be present is a recommendation to run it BARE.
    assert "Do not run a bare" in hint, hint


def test_the_hint_states_the_risk_as_arbitrary_write() -> None:
    """Wording is the deliverable here, not decoration. 'Unverified download' invites a
    reader to accept the risk on a trusted network; 'writes outside the target
    directory' does not, and it is what was measured."""
    hint = _hint().lower()
    assert "arbitrary-write" in hint, hint
    assert "outside the target directory" in hint, hint


def test_the_hint_does_not_duplicate_the_pinned_digest() -> None:
    """The sha256 must have exactly ONE source. A second copy in a user-facing string is
    a drift carrier: config.yaml could be re-pinned and this message would keep handing
    out the old digest, which a reader would then paste into --sha256."""
    import re

    assert "<mhcflurry_model_archive_sha256>" in _hint()
    assert re.search(r"\b[0-9a-f]{64}\b", _hint()) is None, (
        "a literal 64-hex digest appears in the hint; name the config key instead"
    )


def test_every_fetcher_flag_the_hint_names_really_exists() -> None:
    """Binds the MESSAGE to the TOOL's interface, by running the tool.

    A hint is documentation that rots silently: rename a flag on
    scripts/fetch_verified_mhcflurry.py and this message keeps telling users to pass the
    old one, with nothing failing. So the flags are read back out of the fetcher's own
    --help rather than asserted from memory. Only the STEP ONE flags are checked here;
    --already-downloaded-dir belongs to mhcflurry's CLI, not to this repo, and is
    asserted as a string above."""
    import re
    import subprocess

    hint = _hint()
    step_one = hint.split("then `mhcflurry-downloads")[0]
    flags = sorted(set(re.findall(r"--[a-z0-9][a-z0-9-]*", step_one)))
    assert flags, f"no flags parsed out of step one: {step_one!r}"

    repo_root = Path(__file__).resolve().parents[1]
    fetcher = repo_root / "scripts" / "fetch_verified_mhcflurry.py"
    assert fetcher.is_file(), f"the hint names a script that does not exist: {fetcher}"

    helped = subprocess.run(
        [sys.executable, str(fetcher), "--help"],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert helped.returncode == 0, helped.stdout + helped.stderr
    for flag in flags:
        assert flag in helped.stdout, (
            f"the hint tells users to pass {flag}, which the fetcher's --help does not "
            f"advertise. Either the flag was renamed or the hint is wrong.\n{helped.stdout}"
        )


def test_the_python_313_note_scopes_the_limit_to_step_two(monkeypatch) -> None:
    """Step one imports no mhcflurry, so it runs on any interpreter. Saying "run the
    fetch from 3.11 or 3.12" over-constrained the user into thinking the whole route
    needed an older environment."""
    monkeypatch.setattr(cli.sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(cli, "_mhcflurry_version_tuple", lambda: (2, 2, 1))
    message = cli._mhcflurry_model_data_message(None)
    assert "Step one" in message and "unaffected" in message, message
    assert "Only the second command needs" in message, message
