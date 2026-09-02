"""Reproducibility tests for src.train_gnn.set_seed.

set_seed seeds four RNGs and sets two process-global cudnn flags. Drawing from
torch alone exercises exactly one of those lines, so every other line could be
deleted with this file still green. The flag assertions below therefore flip
each flag to the WRONG value first, which makes them independent of whatever
any earlier test in the session happened to leave behind.
"""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from src.train_gnn import set_seed


def test_set_seed_makes_torch_deterministic():
    set_seed(7)
    a = torch.rand(8)
    set_seed(7)
    b = torch.rand(8)
    assert torch.equal(a, b)


def test_set_seed_different_seeds_differ():
    set_seed(1)
    a = torch.rand(8)
    set_seed(2)
    b = torch.rand(8)
    assert not torch.equal(a, b)


def test_set_seed_seeds_python_random():
    set_seed(11)
    a = [random.random() for _ in range(8)]  # noqa: S311 - determinism check, not crypto
    set_seed(11)
    b = [random.random() for _ in range(8)]  # noqa: S311 - determinism check, not crypto
    assert a == b


def test_set_seed_seeds_numpy():
    set_seed(13)
    a = np.random.rand(8)
    set_seed(13)
    b = np.random.rand(8)
    assert np.array_equal(a, b)


def test_set_seed_sets_cudnn_flags():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    set_seed(17)
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
