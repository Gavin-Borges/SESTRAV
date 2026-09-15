import hashlib
import io
import os
import subprocess
import pandas as pd
import pytest


@pytest.fixture
def temp_config(tmp_path):
    config_content = """
dataset_governance:
  qc_thresholds:
    min_peptide_yield: 5
    max_conflict_ratio: 0.15
    max_null_allele_fraction: 0.50
    class_ratio_bounds: [1.5, 4.0]
freeze_mode: false
"""
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        f.write(config_content)
    return path


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "peptide": ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH", "ACDEFGHIKL", "LMNPQRSTVY"],
            "label": [1, 0, 1, 0, 1],
            "allele": ["HLA-A*02:01", "HLA-B*08:01", "HLA-A*02:01", None, "HLA-A*11:01"],
        }
    )


def test_qc_gate_valid(tmp_path, temp_config, valid_df):
    dataset_path = tmp_path / "valid.csv"
    valid_df.to_csv(dataset_path, index=False)

    result = subprocess.run(
        [
            "python",
            "scripts/data_qc_gate.py",
            "--dataset",
            str(dataset_path),
            "--config",
            str(temp_config),
        ],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"QC gate failed on valid dataset: {output}"
    assert "All dataset QC gates passed successfully." in output


def test_qc_gate_length_outlier(tmp_path, temp_config, valid_df):
    # Add a 7-mer length outlier
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "peptide"] = "ACDEFGH"  # 7-mer

    dataset_path = tmp_path / "length_outlier.csv"
    invalid_df.to_csv(dataset_path, index=False)

    quarantine_path = tmp_path / "quarantine.csv"
    result = subprocess.run(
        [
            "python",
            "scripts/data_qc_gate.py",
            "--dataset",
            str(dataset_path),
            "--config",
            str(temp_config),
            "--quarantine",
            str(quarantine_path),
        ],
        capture_output=True,
        text=True,
    )

    # It should pass if we drop row-level outliers but fail if length_and_composition_valid fails,
    # wait: let's verify if the length check causes the gate to fail overall.
    # In scripts/data_qc_gate.py:
    # "length_and_composition_valid": len(indices_to_drop) == 0,
    # So if there are outliers, that check is False, and the gate fails (success = False).
    output = result.stdout + result.stderr
    assert result.returncode == 1, f"QC gate did not fail on length outlier: {output}"
    assert "Dataset QC gate FAILED on one or more admissibility checks." in output

    # Check quarantine output
    assert os.path.exists(quarantine_path)
    q_df = pd.read_csv(quarantine_path)
    assert len(q_df) == 1
    assert "Peptide length 7 outside valid 8-11mer window" in q_df.loc[0, "qc_failure_reason"]


def test_qc_gate_non_canonical_aa(tmp_path, temp_config, valid_df):
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "peptide"] = "ACDEFGHIX"  # 'X' is invalid

    dataset_path = tmp_path / "non_canonical.csv"
    invalid_df.to_csv(dataset_path, index=False)

    result = subprocess.run(
        [
            "python",
            "scripts/data_qc_gate.py",
            "--dataset",
            str(dataset_path),
            "--config",
            str(temp_config),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, "QC gate did not fail on non-canonical AA"


def test_qc_gate_missing_metadata(tmp_path, temp_config, valid_df):
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "peptide"] = None  # missing sequence

    dataset_path = tmp_path / "missing_meta.csv"
    invalid_df.to_csv(dataset_path, index=False)

    result = subprocess.run(
        [
            "python",
            "scripts/data_qc_gate.py",
            "--dataset",
            str(dataset_path),
            "--config",
            str(temp_config),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, "QC gate did not fail on missing peptide"


def test_qc_gate_duplicate_conflict(tmp_path, temp_config, valid_df):
    # Add duplicate with conflicting label
    # valid_df has 5 rows. Let's add 2 identical peptides with conflicting labels.
    conflict_df = pd.concat(
        [
            valid_df,
            pd.DataFrame(
                {
                    "peptide": ["ACDEFGHIK", "ACDEFGHIK"],
                    "label": [1, 0],
                    "allele": ["HLA-A*02:01", "HLA-A*02:01"],
                }
            ),
        ],
        ignore_index=True,
    )

    dataset_path = tmp_path / "duplicate_conflict.csv"
    conflict_df.to_csv(dataset_path, index=False)

    result = subprocess.run(
        [
            "python",
            "scripts/data_qc_gate.py",
            "--dataset",
            str(dataset_path),
            "--config",
            str(temp_config),
        ],
        capture_output=True,
        text=True,
    )
    # Conflict ratio threshold is 0.15. We have 1 conflict out of 5 unique groups (20%), which should fail.
    assert result.returncode == 1, (
        f"QC gate did not fail on duplicate conflicts (status={result.returncode}, err={result.stderr})"
    )


def test_qc_gate_null_allele_fraction(tmp_path, temp_config, valid_df):
    # Null allele threshold is 0.50 (50%).
    # valid_df has 5 rows. Let's make 3 of them have null allele.
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "allele"] = None
    invalid_df.loc[1, "allele"] = None
    invalid_df.loc[3, "allele"] = None  # 3 out of 5 is 60% null, which exceeds 50%

    dataset_path = tmp_path / "high_null_allele.csv"
    invalid_df.to_csv(dataset_path, index=False)

    result = subprocess.run(
        [
            "python",
            "scripts/data_qc_gate.py",
            "--dataset",
            str(dataset_path),
            "--config",
            str(temp_config),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, "QC gate did not fail on high null allele fraction"


def test_qc_gate_class_ratio(tmp_path, temp_config, valid_df):
    # class ratio bounds: [1.5, 4.0].
    # Let's make ratio out of bounds by having too many negatives (e.g. 4 negatives, 1 positive)
    invalid_df = valid_df.copy()
    invalid_df["label"] = [1, 0, 0, 0, 0]  # Ratio: 1/4 = 0.25 (below 1.5)

    dataset_path = tmp_path / "bad_class_ratio.csv"
    invalid_df.to_csv(dataset_path, index=False)

    result = subprocess.run(
        [
            "python",
            "scripts/data_qc_gate.py",
            "--dataset",
            str(dataset_path),
            "--config",
            str(temp_config),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, "QC gate did not fail on out-of-bounds class ratio"


def test_qc_gate_insufficient_yield(tmp_path, temp_config, valid_df):
    # min_peptide_yield is 5. Let's make it have only 4 rows
    invalid_df = valid_df.iloc[:4].copy()

    dataset_path = tmp_path / "low_yield.csv"
    invalid_df.to_csv(dataset_path, index=False)

    result = subprocess.run(
        [
            "python",
            "scripts/data_qc_gate.py",
            "--dataset",
            str(dataset_path),
            "--config",
            str(temp_config),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, "QC gate did not fail on low yield"


@pytest.mark.parametrize("use_crlf", [False, True], ids=["lf", "crlf"])
def test_qc_gate_freeze_mode_crlf_checksum_passes(tmp_path, valid_df, use_crlf):
    """Freeze-mode checksum passes whether the CSV uses LF or CRLF line endings.

    Regression test for the Windows autocrlf bug: git expands LF -> CRLF on
    Windows checkout, causing the SHA-256 to differ from the Linux-CI-computed
    expected digest unless we normalize before hashing (the fix in data_qc_gate.py).

    The expected checksum stored in config is always the LF digest. We use
    lineterminator='\\n' to produce canonical LF bytes on all platforms so the
    CRLF simulation (lf_bytes.replace(b'\\n', b'\\r\\n')) is unambiguous.
    """
    # Force LF line endings regardless of OS (pandas defaults to CRLF on Windows)
    buf = io.StringIO()
    valid_df.to_csv(buf, index=False, lineterminator="\n")
    lf_bytes = buf.getvalue().encode("utf-8")
    assert b"\r\n" not in lf_bytes, "lineterminator='\\n' must produce pure LF bytes"

    # Expected checksum: the LF digest (what the fixed gate computes from either file)
    expected = hashlib.sha256(lf_bytes).hexdigest()

    # Optionally simulate Windows checkout (git autocrlf: LF -> CRLF)
    csv_bytes = lf_bytes.replace(b"\n", b"\r\n") if use_crlf else lf_bytes

    dataset_path = tmp_path / "freeze_dataset.csv"
    dataset_path.write_bytes(csv_bytes)

    config_content = f"""
dataset_governance:
  qc_thresholds:
    min_peptide_yield: 5
    max_conflict_ratio: 0.15
    max_null_allele_fraction: 0.50
    class_ratio_bounds: [1.5, 4.0]
  require_checksum_match_in_freeze_mode: true
  provenance:
    checksum: "{expected}"
freeze_mode: true
"""
    config_path = tmp_path / "freeze_config.yaml"
    config_path.write_text(config_content, encoding="utf-8")

    result = subprocess.run(
        [
            "python",
            "scripts/data_qc_gate.py",
            "--dataset",
            str(dataset_path),
            "--config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"QC gate failed in freeze mode with {'CRLF' if use_crlf else 'LF'} file:\n{output}"
    )
    assert "All dataset QC gates passed successfully." in output


def test_qc_gate_freeze_mode_checksum_mismatch_fails(tmp_path, valid_df):
    buf = io.StringIO()
    valid_df.to_csv(buf, index=False, lineterminator="\n")
    lf_bytes = buf.getvalue().encode("utf-8")
    dataset_path = tmp_path / "freeze_dataset.csv"
    dataset_path.write_bytes(lf_bytes)

    bogus = "0" * 64
    config_content = f"""
dataset_governance:
  qc_thresholds:
    min_peptide_yield: 5
    max_conflict_ratio: 0.15
    max_null_allele_fraction: 0.50
    class_ratio_bounds: [1.5, 4.0]
  require_checksum_match_in_freeze_mode: true
  provenance:
    checksum: "{bogus}"
freeze_mode: true
"""
    config_path = tmp_path / "freeze_config.yaml"
    config_path.write_text(config_content, encoding="utf-8")

    result = subprocess.run(
        [
            "python",
            "scripts/data_qc_gate.py",
            "--dataset",
            str(dataset_path),
            "--config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "does not match expected" in output


def test_qc_gate_malformed_config_does_not_silently_pass(tmp_path, valid_df):
    dataset_path = tmp_path / "valid.csv"
    valid_df.to_csv(dataset_path, index=False)
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("freeze_mode: [unterminated\n", encoding="utf-8")

    result = subprocess.run(
        [
            "python",
            "scripts/data_qc_gate.py",
            "--dataset",
            str(dataset_path),
            "--config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0

    # Assert the REASON, not just the exit code.
    #
    # A bare `returncode != 0` here is vacuous, and measurably so. With the
    # load_config fix reverted, this gate STILL exits 1 - but for an unrelated
    # reason: the small fixture is under the default min_peptide_yield, so the
    # run fails on a QC verdict while the malformed config is swallowed and
    # freeze_mode silently becomes False. Two different failures, one exit code.
    #
    # Measured discriminator. Fix present: no QC report is printed and the YAML
    # ParserError surfaces. Fix reverted: the QC report IS printed and no
    # ParserError appears.
    combined = result.stdout + result.stderr
    assert "ADMISSIBILITY REPORT" not in combined.upper(), (
        "the gate produced a QC report, so it parsed the malformed config as "
        "defaults instead of failing on it:\n" + combined[-2000:]
    )
    assert "ParserError" in combined, (
        "expected the YAML parse failure to surface uncaught:\n" + combined[-2000:]
    )


# ---------------------------------------------------------------------------
# Allele column resolution and the null-allele fraction check.
#
# The gate recognised only a bare "allele" column. Every shipped corpus names it
# "hla_allele", so col_map.get("allele") returned None, the check took its else
# branch and hardcoded a fraction of 1.0 - which the configured threshold of
# 1.00 then accepted as a PASS. The check had never measured a real dataset.
# ---------------------------------------------------------------------------

PEPTIDES_12 = [
    "ACDEFGHIK",
    "LMNPQRSTV",
    "WYACDEFGH",
    "ACDEFGHIKL",
    "LMNPQRSTVY",
    "CDEFGHIKLM",
    "DEFGHIKLMN",
    "EFGHIKLMNP",
    "FGHIKLMNPQ",
    "GHIKLMNPQR",
    "HIKLMNPQRS",
    "IKLMNPQRST",
]
# 8 positive / 4 negative = ratio 2.0, inside class_ratio_bounds [1.5, 4.0],
# so these fixtures isolate the allele checks from every other verdict.
LABELS_12 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]


@pytest.fixture
def permissive_null_config(tmp_path):
    """Config identical to temp_config but with the shipped 1.00 null threshold.

    1.00 cannot fail, since a fraction is bounded on [0, 1] by construction.
    Tests using this fixture prove a property that does not depend on the
    threshold value, which is the point: the threshold itself is a separate
    open ruling and is deliberately not touched here.
    """
    config_content = """
dataset_governance:
  qc_thresholds:
    min_peptide_yield: 5
    max_conflict_ratio: 0.15
    max_null_allele_fraction: 1.00
    class_ratio_bounds: [1.5, 4.0]
freeze_mode: false
"""
    path = tmp_path / "config_permissive.yaml"
    with open(path, "w", encoding="utf-8") as f:
        f.write(config_content)
    return path


def _run_gate(dataset_path, config_path):
    result = subprocess.run(
        [
            "python",
            "scripts/data_qc_gate.py",
            "--dataset",
            str(dataset_path),
            "--config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
    )
    return result, result.stdout + result.stderr


def test_qc_gate_resolves_hla_allele_column(tmp_path, temp_config):
    """The production column name must map, not fall through to the else branch."""
    df = pd.DataFrame(
        {
            "peptide": PEPTIDES_12,
            "label": LABELS_12,
            "hla_allele": ["HLA-A*02:01"] * 11 + [None],
        }
    )
    dataset_path = tmp_path / "hla.csv"
    df.to_csv(dataset_path, index=False)

    result, output = _run_gate(dataset_path, temp_config)

    assert "allele->'hla_allele'" in output, (
        "the gate did not resolve the canonical hla_allele column:\n" + output[-2000:]
    )
    assert "allele->'None'" not in output, (
        "the allele column fell through to the unresolved branch:\n" + output[-2000:]
    )
    # 1 null of 12 = 0.0833, comfortably inside the 0.50 threshold.
    assert "Null Allele Fraction: 0.0833" in output, (
        "expected a measured fraction, not the 1.0 sentinel:\n" + output[-2000:]
    )
    assert result.returncode == 0, output[-2000:]


def test_qc_gate_fails_on_high_null_hla_allele_fraction(tmp_path, temp_config):
    """A genuinely allele-poor corpus must fail, and for the stated reason.

    The exit code alone does not discriminate: before the column-resolution fix
    this same input also exited 1, via the hardcoded 1.0 sentinel. Assert the
    measured fraction.
    """
    df = pd.DataFrame(
        {
            "peptide": PEPTIDES_12,
            "label": LABELS_12,
            "hla_allele": ["HLA-A*02:01"] * 5 + [None] * 7,
        }
    )
    dataset_path = tmp_path / "sparse.csv"
    df.to_csv(dataset_path, index=False)

    result, output = _run_gate(dataset_path, temp_config)

    assert result.returncode != 0, output[-2000:]
    assert "Null Allele Fraction: 0.5833" in output, (
        "expected the measured 7/12 fraction rather than the 1.0 sentinel:\n"
        + output[-2000:]
    )
    assert "null_allele_fraction_passed" in output


def test_qc_gate_cannot_pass_when_allele_column_is_unresolvable(
    tmp_path, permissive_null_config
):
    """An unmeasurable fraction must never report PASS, whatever the threshold.

    Runs at the shipped max_null_allele_fraction of 1.00, which cannot fail on
    its own. The gate must still fail closed, so this property is independent of
    the open threshold ruling.
    """
    df = pd.DataFrame({"peptide": PEPTIDES_12, "label": LABELS_12})
    dataset_path = tmp_path / "no_allele.csv"
    df.to_csv(dataset_path, index=False)

    result, output = _run_gate(dataset_path, permissive_null_config)

    assert result.returncode != 0, (
        "the gate passed a dataset whose allele fraction it could not measure:\n"
        + output[-2000:]
    )
    assert "UNMEASURABLE" in output, output[-2000:]
    assert "allele_column_resolved" in output, output[-2000:]


def test_qc_gate_counts_string_sentinels_as_null_alleles(
    tmp_path, permissive_null_config
):
    """"Unknown" is how the builders write a missing allele; isna() cannot see it."""
    df = pd.DataFrame(
        {
            "peptide": PEPTIDES_12,
            "label": LABELS_12,
            "hla_allele": ["Unknown"] * 12,
        }
    )
    dataset_path = tmp_path / "sentinel.csv"
    df.to_csv(dataset_path, index=False)

    result, output = _run_gate(dataset_path, permissive_null_config)

    assert "Null Allele Fraction: 1.0000" in output, (
        "string sentinels were counted as real alleles:\n" + output[-2000:]
    )
    assert "NaN-only would report 0" in output, (
        "expected the diagnostic contrasting the sentinel count with isna():\n"
        + output[-2000:]
    )


def test_qc_gate_conflict_groups_include_null_allele_rows(tmp_path, temp_config):
    """Conflict grouping must use peptide AND allele, with nulls kept as a key.

    Two failure modes are pinned at once, and the asserted count discriminates
    against both:

    * Unresolved allele column -> grouping collapses to peptide alone, so the
      allele-distinct pair below is falsely flagged as a conflict (2, not 1).
    * Resolved column but a raw groupby -> pandas drops NaN keys by default, so
      the genuine null-allele conflict never reaches label_counts (0, not 1).

    Exactly one of the two appended pairs is a real conflict.
    """
    df = pd.DataFrame(
        {
            # ACDEFGHIK twice with no allele and disagreeing labels: a conflict.
            # LMNPQRSTV twice with DIFFERENT alleles and different labels: not a
            # conflict, because immunogenicity is allele-specific.
            "peptide": PEPTIDES_12 + ["ACDEFGHIK", "LMNPQRSTV"],
            "label": LABELS_12 + [0, 0],
            "hla_allele": (
                [None] + ["HLA-A*02:01"] * 11 + [None, "HLA-B*07:02"]
            ),
        }
    )
    dataset_path = tmp_path / "null_conflict.csv"
    df.to_csv(dataset_path, index=False)

    _result, output = _run_gate(dataset_path, temp_config)

    assert "Conflicting Groups: 1" in output, (
        "expected exactly one conflicting group: the null-allele pair. Two means "
        "the allele column was not resolved and grouping collapsed to peptide "
        "alone; zero means the null-allele rows were dropped by groupby: "
        + output[-2000:]
    )


# ---------------------------------------------------------------------------
# The shipped class_ratio_bounds are DERIVED for the v5 corpus. The derivation
# lives in docs/data_qc_criteria.md under "Derivation of class_ratio_bounds for
# the v5 corpus". These tests read the SHIPPED config rather than a fixture, so
# a future edit that silently re-widens or re-narrows the window has to come
# through here and state why. The previous bound [1.5, 4.0] was fitted to the v3
# corpus (ratio 3.3463) and was never re-derived; it fails v4 (0.8346) and v5
# (0.2051) alike.
# ---------------------------------------------------------------------------

# Shipped v5 composition, reconciled exactly against dedup_dropped in
# data/immunogenicity_dataset_v5_provenance.json. Stated as constants so these
# tests do not read the corpus and stay fast.
V5_POSITIVES = 8712
V5_NEGATIVES = 42473


def _shipped_class_ratio_bounds():
    import pathlib

    import yaml

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))
    bounds = cfg["dataset_governance"]["qc_thresholds"]["class_ratio_bounds"]
    assert len(bounds) == 2, f"class_ratio_bounds must be a [low, high] pair: {bounds}"
    return float(bounds[0]), float(bounds[1])


def test_shipped_bounds_admit_the_v5_corpus_as_built():
    low, high = _shipped_class_ratio_bounds()
    ratio = V5_POSITIVES / V5_NEGATIVES
    assert low <= ratio <= high, (
        f"the shipped bound [{low}, {high}] rejects the corpus it governs "
        f"(ratio {ratio:.4f}). Either the corpus was rebuilt or the bound was "
        "edited without re-deriving it; see docs/data_qc_criteria.md."
    )


def test_shipped_bounds_reject_a_whole_stream_failure_on_either_side():
    """The bound exists to catch the loss or duplication of a whole input stream.

    0.1634 is the ratio when the published panels fail to merge; 0.2949 is the
    ratio when the IEDB export returns only the out-of-panel block. Both are
    modelled in the derivation and both must fall outside the window.
    """
    low, high = _shipped_class_ratio_bounds()
    assert low > 0.1634, (
        f"floor {low} would admit a build whose published panels failed to merge"
    )
    assert high < 0.2949, (
        f"ceiling {high} would admit a build whose IEDB export returned only "
        "the out-of-panel block"
    )


def test_shipped_bounds_keep_a_four_figure_row_margin():
    """A bound whose margin is a few dozen rows is a checksum, not a gate.

    The corpus already carries an exact SHA-256 pin, so a ratio window that
    tight adds nothing and breaks on ordinary input churn.
    """
    import math

    low, _high = _shipped_class_ratio_bounds()
    positives_at_floor = math.ceil(low * V5_NEGATIVES)
    margin_rows = V5_POSITIVES - positives_at_floor
    assert margin_rows >= 1000, (
        f"only {margin_rows} positives would have to vanish to breach the floor; "
        "the derivation requires a four-figure margin."
    )
