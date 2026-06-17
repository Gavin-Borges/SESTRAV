"""Reproducibility tests for src.train_gnn.set_seed."""
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
