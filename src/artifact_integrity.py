"""
Helpers for artifact checksum verification and manifest maintenance.

SESTRAV model files are generated locally and intentionally excluded from git.
That means the practical hardening path is to generate sidecar checksums when
artifacts are created, and verify them again before sensitive loads.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CHECKSUM_MANIFEST = "model_artifact_checksums.json"
PROVENANCE_SIDECAR_SUFFIX = ".provenance.json"
LIBRARY_VERSIONS_FIELD = "library_versions"

# The packages whose version can decide whether a shipped artifact still LOADS.
# Derived by reading what this repo actually pickles, not by assumption: a
# pickle GLOBAL/STACK_GLOBAL opcode walk over every artifact under models/
# (190 .joblib, 62 .pth, 1 .pt) resolves exactly these five distribution roots.
# sklearn, numpy and joblib appear in the ESTIMATOR .joblib dumps; xgboost in
# the BOOSTER dumps; torch in every .pth and .pt. NOT "every .joblib": a
# re-walk resolves xgboost ALONE, and none of the other three, in 30 files
# spanning 9 distinct xgb_*.joblib names, so the two .joblib groups are
# disjoint rather than nested. scipy, pandas and torch_geometric appear in NONE
# of them, which is why they are absent here rather than included as a hedge. A cheaper byte-substring scan of the same corpus additionally reported
# "shap" in 160 files; that is numpy's own "shape" key, and it is recorded here
# as the reason the opcode walk, not a substring search, settled this list.
#
# These are DISTRIBUTION names ("scikit-learn"), not import names ("sklearn"),
# because that is what importlib.metadata resolves.
ARTIFACT_LIBRARY_PACKAGES = (
    "scikit-learn",
    "joblib",
    "numpy",
    "xgboost",
    "torch",
)

# Which of those five can decide whether a GIVEN artifact loads, keyed by the
# artifact's own extension. The tuple above is the UNION over every artifact
# family, so recording it is right and COMPARING all of it is not: an
# `rf_*.joblib` is a graph of sklearn, numpy and joblib objects and holds no
# torch reference at all, so a torch upgrade cannot change how it deserializes.
# Before this mapping existed, `verify_artifact_library_versions(required=True)`
# raised on that upgrade and blocked the RF load anyway - measured, not
# theorised: bump the recorded torch version alone in an RF sidecar and the load
# raises ArtifactIntegrityError with sklearn, numpy and joblib all unchanged.
#
# The split is DELIBERATELY COARSER than the opcode walk above. That walk found
# the two .joblib groups disjoint (estimator dumps resolve sklearn/numpy/joblib,
# the 30 xgb_*.joblib resolve xgboost ALONE), but distinguishing them at load
# time means reading the pickle, and a reader that silently fails returns an
# empty set, records nothing and makes verification VACUOUS - reintroducing by
# the back door the fail-open that `comparable_library_versions` exists to
# close. An extension is total and cannot fail. So .joblib keeps all four
# pickle-based distributions: over-strict by one comparison for an xgboost
# booster, never fail-open, and never wrong about torch.
#
# An extension absent from this mapping has NO entry rather than a default,
# because the two cases are different: a `.csv` under results/ is a real
# artifact that genuinely depends on no library at write time, and treating it
# as unmeasured would warn on every results sidecar this repo writes.
ARTIFACT_FORMAT_LIBRARIES: dict[str, tuple[str, ...]] = {
    ".joblib": ("scikit-learn", "joblib", "numpy", "xgboost"),
    ".pkl": ("scikit-learn", "joblib", "numpy", "xgboost"),
    ".pth": ("torch",),
    ".pt": ("torch",),
}


def artifact_library_dependencies(path: str | Path) -> tuple[str, ...]:
    """Return the distributions whose version can decide whether `path` loads.

    Empty for a format that deserializes through no third-party library at all
    (`.csv`, `.json`, `.md`). Empty is a MEASURED answer here, not an absent
    one, and `verify_artifact_library_versions` treats it as such.
    """
    return ARTIFACT_FORMAT_LIBRARIES.get(Path(path).suffix.lower(), ())


class ArtifactIntegrityError(RuntimeError):
    """Raised when a checksum manifest is missing or an artifact mismatches it."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_manifest_path_for(
    path: str | Path, manifest_name: str = MODEL_CHECKSUM_MANIFEST
) -> Path:
    """Return the default checksum manifest path for an artifact."""
    artifact = Path(path)
    return artifact.parent / manifest_name


def _manifest_key(path: Path, manifest_path: Path) -> str:
    return str(path.resolve().relative_to(manifest_path.parent.resolve())).replace("\\", "/")


def load_checksum_manifest(manifest_path: str | Path) -> dict:
    """Load a checksum manifest, returning an empty structure if it is absent."""
    manifest = Path(manifest_path)
    if not manifest.is_file():
        return {"generated_utc": None, "artifacts": {}}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ArtifactIntegrityError(f"Checksum manifest has invalid format: {manifest}")
    return payload


def update_checksum_manifest(
    manifest_path: str | Path,
    artifact_paths: Iterable[str | Path],
) -> Path:
    """Upsert checksum rows for the provided artifacts into a manifest."""
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = load_checksum_manifest(manifest)
    artifacts = dict(payload.get("artifacts", {}))

    for artifact_path in artifact_paths:
        artifact = Path(artifact_path)
        if not artifact.is_file():
            continue
        key = _manifest_key(artifact, manifest)
        artifacts[key] = {
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_artifact_checksum(
    path: str | Path,
    manifest_path: str | Path | None = None,
    required: bool = False,
) -> bool:
    """Verify an artifact against a checksum manifest.

    Returns True when verification was performed and passed.
    Returns False when no applicable manifest entry exists and verification was optional.
    Raises ArtifactIntegrityError on mismatch or when verification is required but unavailable.
    """
    artifact = Path(path)
    if not artifact.is_file():
        raise ArtifactIntegrityError(f"Artifact not found for checksum verification: {artifact}")

    manifest = Path(manifest_path) if manifest_path else default_manifest_path_for(artifact)
    payload = load_checksum_manifest(manifest)
    artifacts = payload.get("artifacts", {})
    if not artifacts:
        if required:
            raise ArtifactIntegrityError(
                f"Checksum manifest required for '{artifact}', but '{manifest}' was not found or is empty."
            )
        # Optional verification did NOT happen. Say so: no caller in this
        # repository inspects the return value, so without this line a sensitive
        # load proceeds completely unverified and leaves no trace that it did.
        logger.warning(
            "Checksum verification SKIPPED for '%s': manifest '%s' is missing or empty.",
            artifact,
            manifest,
        )
        return False

    # An artifact is only ever matched against its canonical manifest-relative key.
    # Basenames are not unique across the repo (models/, models/v5/ and
    # models/allele_aware/ all carry files with identical names but different
    # contents), so any bare-filename fallback would let one artifact be checked
    # against another artifact's digest. There is no safe fallback: fail closed.
    key: str | None
    try:
        key = _manifest_key(artifact, manifest)
    except ValueError:
        key = None

    entry = artifacts.get(key) if key is not None else None
    if entry is None:
        if required:
            if key is None:
                raise ArtifactIntegrityError(
                    f"Artifact '{artifact}' lies outside the directory of manifest "
                    f"'{manifest}', so no entry in it can describe this artifact."
                )
            raise ArtifactIntegrityError(
                f"No checksum entry found for '{artifact}' in manifest '{manifest}' "
                f"(expected key '{key}')."
            )
        # Same reasoning as above: this is a skip, not a pass.
        logger.warning(
            "Checksum verification SKIPPED for '%s': no entry in manifest '%s' "
            "(expected key '%s').",
            artifact,
            manifest,
            key,
        )
        return False

    expected = entry.get("sha256")
    actual = sha256_file(artifact)
    if expected != actual:
        raise ArtifactIntegrityError(
            f"Checksum verification failed for '{artifact}'. Expected {expected}, got {actual}."
        )
    return True


def load_verified_joblib(
    path: str | Path,
    manifest_path: str | Path | None = None,
    required_checksum: bool = False,
):
    """Verify a joblib artifact when possible, then load it.

    Two independent properties are checked before the load, and they answer
    different questions. The checksum answers "are these the bytes we recorded";
    the library versions answer "can these bytes still be deserialized into the
    object that was pickled". A verified checksum on a pickle written by another
    scikit-learn is a file that is provably intact and may still load wrong, so
    a green checksum is not evidence about the second question.

    Both are governed by `required_checksum`: under it the version check raises
    on measured drift instead of warning. See `verify_artifact_library_versions`
    for why drift is only fatal there, and why an ABSENT record never is.
    """
    from joblib import load as joblib_load

    verify_artifact_checksum(path, manifest_path=manifest_path, required=required_checksum)
    verify_artifact_library_versions(path, required=required_checksum)
    return joblib_load(path)


def _relative_to_project_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def provenance_sidecar_path_for(path: str | Path) -> Path:
    """Return the sidecar path `write_provenance_sidecar` writes for an artifact.

    The suffix is APPENDED rather than replacing the artifact's own, so
    `rf_31feature_integrated.joblib` pairs with
    `rf_31feature_integrated.joblib.provenance.json` and two artifacts whose
    names differ only by extension cannot share one sidecar.
    """
    artifact = Path(path)
    return artifact.with_suffix(artifact.suffix + PROVENANCE_SIDECAR_SUFFIX)


def library_versions(
    packages: Iterable[str] = ARTIFACT_LIBRARY_PACKAGES,
) -> dict[str, str | None]:
    """Return `{distribution: installed version}` for the packages an artifact's
    loadability depends on, for embedding in a provenance sidecar.

    A digest alone does not make an artifact reproducible: a pickled
    RandomForest is a graph of `sklearn` and `numpy` objects, so the version
    that wrote it is part of its provenance and nothing in this repository
    recorded it before this function existed.

    A package that is not installed records `None` rather than raising or being
    omitted, matching `model_provenance_fields`, which records a `None` sha256
    for an absent input instead of failing. The field then has the same shape on
    every artifact, and a reader can distinguish "not installed when this was
    written" from "this writer never asked about it".
    """
    resolved: dict[str, str | None] = {}
    for name in packages:
        try:
            resolved[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            resolved[name] = None
    return resolved


def comparable_library_versions(recorded: object) -> dict[str, str]:
    """Return the entries of a sidecar's `library_versions` that can actually be
    compared: those recording a version STRING.

    This is ONE function because the fail-open it closes came from having two.
    `verify_artifact_library_versions` guarded on the raw mapping being a
    non-empty dict, while `library_version_drift` compared only the
    string-valued entries. A sidecar recording a null for every package is
    non-empty, so it passed the guard, contributed nothing to compare, produced
    an empty drift list, and was reported as "every recorded version matched"
    with no log line at all.

    That is the normal shape of an uninformative record rather than a corrupt
    one: `library_versions` deliberately writes null for a package that was not
    installed. The sidecar is not itself checksummed, so the same shape is also
    what a tampered record would have.

    Anything deciding "is there something to compare here?" must ask this
    function rather than the raw mapping, or the two notions drift apart again.
    """
    if not isinstance(recorded, dict):
        return {}
    return {name: value for name, value in recorded.items() if isinstance(value, str)}


def library_version_drift(
    recorded: object,
    running: dict[str, str | None] | None = None,
) -> list[str]:
    """Return one line per package whose recorded version differs from the
    running environment's, and an empty list when they agree.

    Only packages the sidecar recorded an actual VERSION for are compared. A
    package recorded as `None` was not installed when the artifact was written,
    so the artifact cannot depend on it and its presence now is not drift; a
    recorded version that is missing here IS drift, because the artifact was
    built against something this environment cannot supply.
    """
    claimed = comparable_library_versions(recorded)
    if running is None:
        running = library_versions(sorted(claimed))
    drift: list[str] = []
    for name in sorted(claimed):
        current = running.get(name)
        if current is None:
            drift.append(f"{name}: artifact written under {claimed[name]}, not installed here")
        elif current != claimed[name]:
            drift.append(
                f"{name}: artifact written under {claimed[name]}, running {current}"
            )
    return drift


def verify_artifact_library_versions(
    path: str | Path,
    required: bool = False,
) -> bool:
    """Compare an artifact's provenance sidecar library versions against this
    environment, the companion to `verify_artifact_checksum`.

    Returns True when nothing in the record blocks this load: either the
    comparison ran and every version that could matter matched, or the
    artifact's format deserializes through no third-party library at all.
    Returns False when no comparison was possible, or when drift was found and
    reporting it was optional.
    Raises ArtifactIntegrityError on drift when `required` is True.

    ONLY the distributions `artifact_library_dependencies` names for this
    artifact's format are compared. The sidecar records the whole environment,
    which is the right thing to RECORD and the wrong thing to enforce: the
    recorded set is the union over every artifact family, so enforcing it made
    a torch upgrade raise on an sklearn RandomForest that holds no torch
    reference. Recording stays wide, adjudication is narrow.

    WHY DRIFT WARNS BY DEFAULT AND ONLY RAISES UNDER `required`. A checksum
    mismatch is unconditionally wrong: the bytes are not the bytes. A version
    difference is weaker evidence - the bytes may still deserialize into an
    equivalent object, and scikit-learn itself only WARNS about it
    (`InconsistentVersionWarning`), so raising on every difference would be
    stricter than the library that owns the format and would break a load in an
    environment that merely patched numpy. `required=True` is how a caller
    declares a load sensitive (production scoring, the promotion gate, the API
    scoring path all pass `required_checksum=True`), and a measured mismatch on
    a sensitive load is an error.

    WHY AN ABSENT RECORD NEVER RAISES, even under `required`. No artifact
    written before this field existed carries one, so raising there would make
    `required_checksum=True` unusable against every model on disk today.
    Absence is UNMEASURED, not clean, so it is logged rather than swallowed:
    that silence is exactly the defect the two fallbacks fixed on this branch
    had. An unreadable or malformed sidecar is treated the same way, because a
    corrupt sidecar is not evidence of drift.
    """
    artifact = Path(path)
    sidecar = provenance_sidecar_path_for(artifact)

    relevant = artifact_library_dependencies(artifact)
    if not relevant:
        # A measured empty set, not an unmeasured one. Returning False here
        # would report every results CSV as unverifiable and train the reader
        # to ignore the warning that matters.
        logger.debug(
            "Library version verification not applicable to '%s': its format "
            "deserializes through no third-party library.",
            artifact,
        )
        return True

    recorded: object = None
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict):
            recorded = payload.get(LIBRARY_VERSIONS_FIELD)

    comparable = {
        name: version
        for name, version in comparable_library_versions(recorded).items()
        if name in relevant
    }
    if not comparable:
        logger.warning(
            "Library version verification SKIPPED for '%s': sidecar '%s' is missing, "
            "unreadable, or records no comparable '%s' entry for any of %s. A record "
            "whose values are all null or non-string is UNMEASURED, not clean.",
            artifact,
            sidecar,
            LIBRARY_VERSIONS_FIELD,
            ", ".join(relevant),
        )
        return False

    drift = library_version_drift(comparable)
    if not drift:
        return True

    detail = "; ".join(drift)
    if required:
        raise ArtifactIntegrityError(
            f"Library version drift for '{artifact}' against sidecar '{sidecar}': "
            f"{detail}. The artifact's checksum was required, so this load is "
            f"treated as unsafe rather than allowed to deserialize under a "
            f"different library than wrote it."
        )
    logger.warning(
        "Library version DRIFT for '%s' against sidecar '%s': %s.",
        artifact,
        sidecar,
        detail,
    )
    return False


def write_provenance_sidecar(
    output_path: str | Path,
    *,
    script: str,
    extra: dict[str, object] | None = None,
) -> Path:
    """Write a `<output>.provenance.json` sidecar recording the output's own sha256.

    Mirrors the pattern proven in `scripts/assess_calibration.py` (the one
    sidecar that already PASSes `_local/integrity/integrity_check.py`'s
    provenance check): `artifact` + `sha256` are the two fields that check
    resolves and verifies, and the file is written with `newline=""` so the
    LF that `json.dumps` produces is not rewritten to CRLF on Windows - the
    hash recorded must match the hash git stores under the `results/*.provenance.json`
    `eol=lf` pin in `.gitattributes`, or the check fails on a byte-identical file.

    `extra` is for anything the artifact's reproducibility depends on but that
    isn't captured by the artifact's own bytes - for a script that scores an
    untracked, gitignored model file, that means the model's own path and
    sha256, recorded here specifically because the model can be silently
    overwritten in place after the benchmark ran (see the TSNAdb 0.99
    incident, D-series, 2026-08-12: the model that produced it was overwritten
    with no checksum captured, making the figure permanently unreproducible).
    Pass `{"model_path": ..., "model_sha256": ...}` for that case.

    `library_versions` records the environment the artifact was written in, for
    the same reason: the bytes of a pickled estimator are only loadable by a
    compatible `scikit-learn`, and `pyproject.toml` declares a floor
    (`scikit-learn>=1.3.0`) rather than the exact version any given artifact was
    produced under. It is written here, in the one function every sidecar
    passes through, so that all callers gain the field together instead of each
    remembering to pass it through `extra`.
    """
    output_path = Path(output_path)
    payload: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": script,
        "artifact": _relative_to_project_root(output_path),
        "sha256": sha256_file(output_path),
        LIBRARY_VERSIONS_FIELD: library_versions(),
    }
    if extra:
        payload.update(extra)
    sidecar_path = provenance_sidecar_path_for(output_path)
    with sidecar_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(payload, indent=2) + "\n")
    return sidecar_path


def model_provenance_fields(model_path: str | Path) -> dict[str, object]:
    """Return `{model_path, model_sha256}` for embedding in a benchmark's own
    provenance sidecar via `write_provenance_sidecar`'s `extra` argument.

    `model_sha256` is `None` when the model file is not present locally (it is
    gitignored in this repo) rather than raising, matching how
    `check_provenance` treats a missing referenced artifact as a benign SKIP,
    not a FAIL.
    """
    model_path = Path(model_path)
    return {
        "model_path": _relative_to_project_root(model_path),
        "model_sha256": sha256_file(model_path) if model_path.is_file() else None,
    }


def binding_matrix_provenance_fields(
    binding_matrix_path: str | Path,
) -> dict[str, object]:
    """Return `{binding_matrix_path, binding_matrix_sha256}`, the companion to
    `model_provenance_fields` for the other input a mode-30/31/50 score depends on.

    The model half of this pair exists because a benchmark once recorded no model
    hash, the model was later overwritten, and the figure became permanently
    unreproducible. The binding matrix is the same class of input and was left
    out. It is not a lesser one: `prepare_features_30` substitutes `np.zeros(10)`
    for every peptide the matrix omits, without raising, so a score is a function
    of the matrix's COVERAGE as much as of the model. Measured on the v5 corpus,
    swapping one tracked matrix for another moves pooled AUC-PR by roughly 0.15,
    which is larger than any regression threshold a benchmark would alert on.

    `binding_matrix_sha256` is `None` when the file is not present locally rather
    than raising, matching `model_provenance_fields` and how `check_provenance`
    treats a missing referenced artifact as a benign SKIP.
    """
    binding_matrix_path = Path(binding_matrix_path)
    return {
        "binding_matrix_path": _relative_to_project_root(binding_matrix_path),
        "binding_matrix_sha256": (
            sha256_file(binding_matrix_path) if binding_matrix_path.is_file() else None
        ),
    }
