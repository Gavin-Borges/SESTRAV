from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest

from scripts import compute_mode51_allele_identity_discriminator as discriminator
from scripts.compute_mode51_allele_identity_discriminator import (
    OTHER,
    allele_identity_features,
    integer_allele_feature,
    supertype_features,
    target_rate_features,
    within_allele_shuffled_weights,
)
from src.features import ALLELE_CONTACT_WEIGHTS, CONTACT_WEIGHT_COLUMNS


def test_synthetic_allele_identity_arms_are_deterministic_and_bounded() -> None:
    panel = sorted(ALLELE_CONTACT_WEIGHTS)
    alleles = pd.Series([panel[0], panel[1], "HLA-B*58:09", "HLA-C*07:02", None])

    one_hot = allele_identity_features(alleles)
    assert one_hot.shape == (5, len(panel) + 1)
    assert one_hot.sum(axis=1).eq(1.0).all()
    assert one_hot.loc[2, f"allele_id_{OTHER}"] == 1.0
    assert one_hot.loc[3, f"allele_id_{OTHER}"] == 1.0
    assert one_hot.loc[4, f"allele_id_{OTHER}"] == 1.0

    integer = integer_allele_feature(alleles)
    assert integer.iloc[:, 0].tolist() == [1.0, 2.0, 0.0, 0.0, 0.0]

    supertype = supertype_features(alleles)
    assert supertype.sum(axis=1).eq(1.0).all()
    assert supertype.loc[2, "hla_supertype_B58"] == 1.0
    assert supertype.loc[3, f"hla_supertype_{OTHER}"] == 1.0


def test_within_allele_shuffle_is_exactly_equal_for_constant_rows() -> None:
    panel = sorted(ALLELE_CONTACT_WEIGHTS)
    alleles = pd.Series([panel[0], panel[0], panel[1], panel[1], "other", "other"])
    first = within_allele_shuffled_weights(alleles, seed=42)
    second = within_allele_shuffled_weights(alleles, seed=999)
    assert list(first.columns) == CONTACT_WEIGHT_COLUMNS
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_target_rate_uses_no_heldout_labels_and_leaves_training_row_out() -> None:
    train_alleles = pd.Series(["a", "a", "b", "c"])
    train_y = np.asarray([1, 0, 1, 0])
    heldout_alleles = pd.Series(["a", "b", "unseen"])

    train, heldout = target_rate_features(train_alleles, train_y, heldout_alleles)

    assert train.iloc[:, 0].tolist() == [0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0]
    assert heldout.iloc[:, 0].tolist() == [0.5, 1.0, 0.5]


def test_main_resolves_git_sha_before_writing_anything(tmp_path, monkeypatch) -> None:
    def no_repository(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    def must_not_run(*args, **kwargs):
        raise AssertionError("compute_table ran before git_sha was resolved")

    monkeypatch.setattr(discriminator.subprocess, "check_output", no_repository)
    monkeypatch.setattr(discriminator, "compute_table", must_not_run)
    output = tmp_path / "table.csv"

    with pytest.raises(subprocess.CalledProcessError):
        discriminator.main(["--output", str(output)])
    assert not output.exists()
