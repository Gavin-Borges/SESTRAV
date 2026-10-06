"""A DEFAULT calibrator must never be chosen by the process working directory.

Stage 4 resolved two defaults relative to the cwd: the canonical conformal
calibrator (``models/v5/conformal_calibrator.joblib``) and the per-virus
calibration directory (``models/calibration/per_virus``). ``sestrav predict``
runs with ``--conformal`` ON by default, so launching it from any directory that
carried ``models/v5/conformal_calibrator.joblib`` handed that file to
``joblib.load``, which is pickle deserialization of attacker-controlled bytes.

The checksum gate did not stop it. ``src.artifact_integrity.default_manifest_path_for``
returns ``artifact.parent / model_artifact_checksums.json``, so the digest an
artifact is checked against is whatever file sits BESIDE it; a planted pickle
shipped with a planted manifest verifies clean. The artifact and its trust anchor
came from the same attacker-chosen directory, which is no anchor at all.

These tests plant exactly that pair and assert the resolver ignores it. A path the
caller NAMED (``--conformal-calibrator``, an explicit ``per_virus_dir``) is still
honoured exactly as given: naming a file is an act of trust and resolving a
default is not. Nothing harmful is planted - the stub artifacts are a few ASCII
bytes, and every test that could reach a load replaces ``joblib.load`` with a
recorder that refuses to deserialize anything.

Checkout independence, stated because it shapes every assertion below: the
canonical calibrator is gitignored, so it is present in the primary checkout and
absent from a worktree or a fresh clone. The tests therefore assert the
INVARIANT - the answer is whatever the repository root holds, and never what the
cwd holds - rather than a fixed verdict that would only be right in one checkout.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import functions.stage4_immunogenicity_scoring as s4
from src import artifact_integrity


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFORMAL = REPO_ROOT / "models" / "v5" / "conformal_calibrator.joblib"
STUB_BYTES = b"planted-stub-not-a-real-pickle"


def _plant(directory: Path, *parts: str) -> Path:
    """Write a harmless stub artifact plus a manifest that validates it.

    The manifest is the real shape ``verify_artifact_checksum`` consumes, with a
    correct sha256, so the planted pair would pass ``required_checksum=True``.
    That is the point: the test must fail for the resolver's reason, not because
    the decoy happened to be malformed.
    """
    target = directory.joinpath(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(STUB_BYTES)
    manifest = target.parent / artifact_integrity.MODEL_CHECKSUM_MANIFEST
    manifest.write_text(
        json.dumps(
            {
                "generated_utc": "2026-01-01T00:00:00+00:00",
                "artifacts": {
                    target.name: {
                        "sha256": hashlib.sha256(STUB_BYTES).hexdigest(),
                        "size_bytes": len(STUB_BYTES),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return target


def _same_file(a, b) -> bool:
    if a is None or b is None:
        return False
    return os.path.realpath(os.path.abspath(str(a))) == os.path.realpath(os.path.abspath(str(b)))


def _expected_conformal_default():
    """What the repository root holds, which is the only acceptable answer."""
    return str(CANONICAL_CONFORMAL) if CANONICAL_CONFORMAL.is_file() else None


@pytest.fixture
def planted_cwd(monkeypatch, tmp_path):
    """chdir into a directory carrying a decoy calibrator and a matching manifest."""
    workdir = tmp_path / "attacker_cwd"
    workdir.mkdir()
    planted = _plant(workdir, "models", "v5", "conformal_calibrator.joblib")
    monkeypatch.chdir(workdir)
    return planted


@pytest.fixture
def recorded_loads(monkeypatch):
    """Replace ``joblib.load`` with a recorder that never deserializes anything.

    It raises after recording, so a caller that reaches it takes its own
    load-failure path instead of receiving a bogus object. The tests assert on
    the recorded paths, so a load of the planted file is loud rather than silent.
    """
    import joblib

    seen: list[str] = []

    def _recorder(path, *args, **kwargs):
        seen.append(str(path))
        raise RuntimeError("joblib.load was reached; this test refuses to deserialize")

    monkeypatch.setattr(joblib, "load", _recorder)
    return seen


# ---------------------------------------------------------------------------
# The conformal default
# ---------------------------------------------------------------------------


def test_conformal_default_ignores_a_calibrator_planted_in_the_working_directory(
    planted_cwd,
):
    resolved = s4._resolve_conformal_path(None, None)

    assert not _same_file(resolved, planted_cwd), (
        "the default conformal calibrator resolved to a file supplied by the "
        f"working directory: {resolved}"
    )
    assert resolved == _expected_conformal_default()


def test_conformal_default_is_independent_of_the_working_directory(
    monkeypatch, planted_cwd, tmp_path
):
    """The same question, asked from three directories, must give one answer."""
    from_planted = s4._resolve_conformal_path(None, None)

    monkeypatch.chdir(REPO_ROOT)
    from_repo_root = s4._resolve_conformal_path(None, None)

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    from_empty = s4._resolve_conformal_path(None, None)

    assert from_planted == from_repo_root == from_empty == _expected_conformal_default()


def test_conformal_default_trust_anchor_is_a_tracked_manifest_at_the_repository_root(
    planted_cwd,
):
    """The digest a DEFAULT-resolved load is checked against must come from git.

    This is the second half of the defect: the resolver chose the pickle and
    ``default_manifest_path_for`` then chose the digest from the SAME directory,
    so an attacker supplied both. Anchoring the default to the repository root
    makes that manifest the tracked ``models/v5/model_artifact_checksums.json``,
    which an attacker cannot write. Asserted through git rather than by path
    shape, because "tracked" is the property that matters.
    """
    manifest = artifact_integrity.default_manifest_path_for(s4.DEFAULT_CONFORMAL_CALIBRATOR)

    assert not _same_file(
        manifest, planted_cwd.parent / artifact_integrity.MODEL_CHECKSUM_MANIFEST
    ), f"the trust anchor came from the working directory: {manifest}"
    assert Path(manifest) == (
        REPO_ROOT / "models" / "v5" / artifact_integrity.MODEL_CHECKSUM_MANIFEST
    )

    if shutil.which("git") is None:
        pytest.skip("git is not on PATH, so trackedness cannot be measured here")
    relative = Path(manifest).resolve().relative_to(REPO_ROOT).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0, (
        f"the default calibrator's checksum manifest {relative} is not tracked by git, "
        "so it is not a trust anchor"
    )


def test_apply_conformal_never_deserializes_a_calibrator_from_the_working_directory(
    planted_cwd, recorded_loads, capsys
):
    frame = pd.DataFrame({"peptide": ["CLGGLLTMV"], "immunogenicity_score": [0.5]})

    s4._apply_conformal(frame, model_dir=None, conformal_path=None, freeze_mode=False)
    capsys.readouterr()

    assert not any(_same_file(seen, planted_cwd) for seen in recorded_loads), (
        f"joblib.load was handed a working-directory calibrator: {recorded_loads}"
    )


def test_apply_conformal_loads_nothing_when_the_installed_default_is_absent(
    monkeypatch, planted_cwd, recorded_loads, tmp_path, capsys
):
    """With the installed default absent, a planted cwd must produce no load at all.

    The test above is checkout-independent and therefore cannot assert "zero
    loads": the primary checkout does hold the gitignored canonical calibrator,
    which is a legitimate load. Repointing the module constant at an absent path
    removes that legitimate candidate, so the remaining assertion is exact.
    """
    monkeypatch.setattr(
        s4, "DEFAULT_CONFORMAL_CALIBRATOR", str(tmp_path / "absent" / "conformal_calibrator.joblib")
    )
    frame = pd.DataFrame({"peptide": ["CLGGLLTMV"], "immunogenicity_score": [0.5]})

    applied = s4._apply_conformal(frame, model_dir=None, conformal_path=None, freeze_mode=False)
    out = capsys.readouterr().out

    assert recorded_loads == [], f"joblib.load ran with no installed calibrator: {recorded_loads}"
    assert applied is False
    assert "lower_bound" not in frame.columns
    # NOT a bare substring test. `"working directory" in out` is satisfied by a
    # message promising the OPPOSITE, and that inversion was measured to survive as a
    # mutant. Require the NEGATION and the phrase in the same sentence.
    assert re.search(r"not\s+searched[^.]*working directory", out, re.I), (
        "the warning must say the default is NOT searched for under the cwd. A message "
        "that merely MENTIONS the working directory would send a reader to fix a "
        f"refusal by changing directory into one that carries a calibrator. Got: {out!r}"
    )


def test_freeze_mode_refuses_rather_than_using_a_planted_calibrator(
    monkeypatch, planted_cwd, recorded_loads, tmp_path
):
    """Freeze mode is the strictest path, so it is where a silent swap would hurt most."""
    monkeypatch.setattr(
        s4, "DEFAULT_CONFORMAL_CALIBRATOR", str(tmp_path / "absent" / "conformal_calibrator.joblib")
    )
    frame = pd.DataFrame({"peptide": ["CLGGLLTMV"], "immunogenicity_score": [0.5]})

    with pytest.raises(FileNotFoundError) as excinfo:
        s4._apply_conformal(frame, model_dir=None, conformal_path=None, freeze_mode=True)

    assert "working directory" in str(excinfo.value)
    assert recorded_loads == [], f"joblib.load ran under freeze mode: {recorded_loads}"


# ---------------------------------------------------------------------------
# The per-virus default
# ---------------------------------------------------------------------------


def test_per_virus_default_ignores_a_directory_planted_in_the_working_directory(
    monkeypatch, tmp_path
):
    workdir = tmp_path / "attacker_cwd"
    workdir.mkdir()
    planted = _plant(workdir, "models", "calibration", "per_virus", "HIV-1.joblib")
    monkeypatch.chdir(workdir)
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()

    resolved = s4._resolve_calibrator_path(str(model_dir), virus="HIV-1")

    assert not _same_file(resolved, planted), (
        f"the per-virus default resolved into the working directory: {resolved}"
    )
    # models/calibration/per_virus is tracked nowhere and exists in no checkout,
    # so the only correct answer is None.
    assert resolved is None


def test_per_virus_default_dir_is_anchored_to_the_repository_root():
    assert Path(s4.DEFAULT_PER_VIRUS_CALIBRATION_DIR) == (
        REPO_ROOT / "models" / "calibration" / "per_virus"
    )
    assert os.path.isabs(s4.DEFAULT_PER_VIRUS_CALIBRATION_DIR)


def test_an_explicit_per_virus_dir_is_still_used_exactly_as_given(tmp_path):
    """Explicitly passed paths stay user-trusted; only defaults were re-anchored."""
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    pv_dir = tmp_path / "promoted"
    pv_dir.mkdir()
    (pv_dir / "HIV-1.joblib").write_bytes(STUB_BYTES)

    resolved = s4._resolve_calibrator_path(
        str(model_dir), virus="HIV-1", per_virus_dir=str(pv_dir)
    )

    assert resolved == str(pv_dir / "HIV-1.joblib")


# ---------------------------------------------------------------------------
# The supported explicit path, which the fix must not break
# ---------------------------------------------------------------------------


def test_an_explicit_conformal_calibrator_still_resolves_from_any_directory(
    planted_cwd, tmp_path
):
    explicit = tmp_path / "chosen.joblib"
    explicit.write_bytes(STUB_BYTES)

    assert s4._resolve_conformal_path(None, str(explicit)) == str(explicit)


def test_a_calibrator_beside_the_model_still_wins_over_the_installed_default(
    planted_cwd, tmp_path
):
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    beside = model_dir / "conformal_calibrator.joblib"
    beside.write_bytes(STUB_BYTES)

    assert s4._resolve_conformal_path(str(model_dir), None) == str(beside)


def test_the_cli_precheck_agrees_with_the_resolver_from_a_planted_directory(
    planted_cwd, tmp_path
):
    """The freeze-mode precheck must refuse exactly when the resolver finds nothing.

    Asserted as an agreement rather than a fixed verdict, so it holds in a
    checkout that has the gitignored calibrator and in one that does not.
    """
    import argparse

    from src import cli

    model = tmp_path / "model.joblib"
    model.write_bytes(STUB_BYTES)
    args = argparse.Namespace(
        conformal=True, conformal_calibrator=None, model=str(model)
    )

    resolved = s4._resolve_conformal_path(os.path.dirname(str(model)), None)
    refused = False
    try:
        cli._require_conformal_calibrator(args, freeze_mode=True)
    except cli.CliPreconditionError:
        refused = True

    assert refused is (resolved is None)
    assert not _same_file(resolved, planted_cwd)
def test_the_default_holds_in_a_process_that_STARTS_in_the_planted_directory(tmp_path):
    """The blind spot every other test in this file shares, and the only test here
    that can see the real attack.

    `_PROJECT_ROOT` is computed once, when the module is IMPORTED. Every other test
    above chdirs with `monkeypatch` AFTER that import, so the constants were already
    derived from the test runner's cwd and are correct no matter how they were
    derived. Measured: replacing the `__file__` anchor with `os.getcwd()` restores
    the entire vulnerability and leaves all of those tests GREEN.

    The real attack is `sestrav predict` LAUNCHED from the attacker's directory, so
    the hostile cwd is in effect at import. Only a fresh interpreter started there
    reproduces that, which is why this test pays for a subprocess.
    """
    import sys as _sys

    hostile = tmp_path / "hostile"
    planted = _plant(hostile, "models", "v5", "conformal_calibrator.joblib")
    _plant(hostile, "models", "calibration", "per_virus", "EBV.joblib")

    code = (
        "import json;"
        "import functions.stage4_immunogenicity_scoring as s4;"
        "print(json.dumps({"
        "'default': s4.DEFAULT_CONFORMAL_CALIBRATOR,"
        "'per_virus': s4.DEFAULT_PER_VIRUS_CALIBRATION_DIR,"
        "'resolved': s4._resolve_conformal_path(None, None),"
        "}))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    done = subprocess.run(
        [_sys.executable, "-c", code],
        cwd=str(hostile), capture_output=True, text=True, env=env,
    )
    assert done.returncode == 0, f"probe failed: {done.stderr[-800:]}"
    payload = json.loads(done.stdout.strip().splitlines()[-1])

    # Positive control: the fixture really did plant something reachable from that cwd,
    # so a pass cannot come from the attack being impossible in the first place.
    assert planted.is_file()
    assert (hostile / "models" / "v5" / "conformal_calibrator.joblib").is_file()

    for key in ("default", "per_virus"):
        value = Path(payload[key]).resolve()
        assert not value.is_relative_to(hostile.resolve()), (
            f"{key} resolved INSIDE the process's starting directory: {value}. The "
            f"default is cwd-derived, which is the vulnerability this unit closes."
        )
        assert value.is_relative_to(REPO_ROOT), (
            f"{key} resolved outside the installation root: {value}"
        )

    if payload["resolved"] is not None:
        assert not _same_file(payload["resolved"], planted), (
            f"the resolver returned the planted calibrator: {payload['resolved']}"
        )
