"""Reproducibility tests for src.train_gnn.set_seed.

set_seed seeds four RNGs and sets three process-global flags: the two cudnn
flags and deterministic-algorithms mode. Drawing from torch alone exercises
exactly one of those lines, so every other line could be deleted with this file
still green. The flag assertions below therefore flip each flag to the WRONG
value first, which makes them independent of whatever any earlier test in the
session happened to leave behind.

Leaving those flags set is set_seed's contract, so nothing here restores them
on its behalf. Confining them to a run is the job of _restore_torch_global_flags
on the two entry points, covered by tests/test_train_gnn_flag_scope.py. The
fixture below only stops THIS file from leaking into the rest of the session.
"""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from src.train_gnn import set_seed  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_global_flags():
    """Restore the process-global flags set_seed writes, so this file leaks nothing."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


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


def test_set_seed_enables_deterministic_algorithms():
    """The cudnn pair only covers cuDNN kernels; this covers the rest of torch."""
    torch.use_deterministic_algorithms(False)
    set_seed(19)
    assert torch.are_deterministic_algorithms_enabled()


def test_set_seed_leaves_deterministic_algorithms_in_warn_only_mode():
    """warn_only is what keeps a kernel with no deterministic form a warning, not an abort.

    Without it, set_seed would turn any such op in a GNN run into a hard
    RuntimeError, which is a behaviour change nobody asked for.
    """
    torch.use_deterministic_algorithms(False)
    set_seed(23)
    assert torch.is_deterministic_algorithms_warn_only_enabled()
