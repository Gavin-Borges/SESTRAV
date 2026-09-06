"""A v2 training pool whose last batch holds one sample must not crash the run.

GraphPredictorV2's physico_block ends in nn.BatchNorm1d, which rejects a (1, C)
input in train mode. A pool with n % batch_size == 1 therefore dies on the final
batch of the first epoch, after every earlier batch has already trained, and
shuffling cannot help because the last batch's SIZE follows from n alone.

The fix is conditional, not a blanket drop_last=True. Blanket dropping would
trade this crash for a ZeroDivisionError, since train_epoch_v2 returns
total_loss / len(dataloader) and a pool smaller than one batch then measures
zero batches. Both halves of that trade are pinned below.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric", reason="v2 GNN path requires torch_geometric")

from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader as PyGDataLoader  # noqa: E402

import src.train_gnn as tg  # noqa: E402
from src.gnn.graph_builder import GraphBuilder  # noqa: E402
from src.gnn.models import GraphEncoderV2, GraphPredictorV2  # noqa: E402
from src.train_gnn import _drops_a_lone_last_sample  # noqa: E402

MAX_LEN = 11
NUM_FEATURES = 12
NODE_DIM = 32


def _graphs(n):
    edge_index, edge_attr = GraphBuilder.build_pyg_chain_graph(MAX_LEN)
    return [
        Data(
            x=torch.rand(MAX_LEN, NODE_DIM),
            edge_index=edge_index,
            edge_attr=edge_attr,
            physico=torch.rand(1, NUM_FEATURES),
            y=torch.tensor([float(i % 2)]),
        )
        for i in range(n)
    ]


def _train_mode_model():
    model = GraphPredictorV2(num_continuous_features=NUM_FEATURES, node_dim=NODE_DIM)
    model.encoder = GraphEncoderV2(node_dim=NODE_DIM, hidden_dim=16, out_dim=128, edge_dim=3)
    model.train()
    return model


# --- the defect itself -------------------------------------------------------


def test_a_lone_final_sample_crashes_the_v2_model_in_train_mode():
    """The crash being guarded. Without the guard this is what a real run hits."""
    loader = PyGDataLoader(_graphs(5), batch_size=4, shuffle=False, drop_last=False)
    model = _train_mode_model()

    with pytest.raises(ValueError, match="Expected more than 1 value per channel"):
        for batch in loader:
            model(batch)


def test_the_guard_prevents_that_crash_and_keeps_every_full_batch():
    n, batch_size = 5, 4
    loader = PyGDataLoader(
        _graphs(n),
        batch_size=batch_size,
        shuffle=False,
        drop_last=_drops_a_lone_last_sample(n, batch_size),
    )
    model = _train_mode_model()

    sizes = []
    for batch in loader:
        model(batch)
        sizes.append(batch.num_graphs)

    assert sizes == [4]


def test_a_single_sample_pool_keeps_its_only_batch():
    """The other half of the trade: an empty loader would divide by zero in train_epoch_v2.

    n == 1 is the only pool size that discriminates here, which is why it is the
    one used. For any other n below batch_size, n % batch_size is n itself and
    the naive drop_last = (n % batch_size == 1) form already returns False, so a
    test written at n == 3 passes with the guard removed and proves nothing.
    """
    n, batch_size = 1, 4
    loader = PyGDataLoader(
        _graphs(n),
        batch_size=batch_size,
        shuffle=False,
        drop_last=_drops_a_lone_last_sample(n, batch_size),
    )

    assert len(loader) == 1


# --- the predicate -----------------------------------------------------------


@pytest.mark.parametrize(
    "n, batch_size, expected",
    [
        (65, 64, True),  # the crash case
        (129, 64, True),
        (64, 64, False),  # exact fit, no partial batch at all
        (66, 64, False),  # partial batch of 2 is fine for BatchNorm
        (63, 64, False),  # whole pool is one short batch, dropping it empties the loader
        (1, 64, False),  # 1 % 64 == 1, but dropping it empties the loader
        (65, 1, False),  # every batch holds one sample; drop_last cannot save this
    ],
)
def test_predicate(n, batch_size, expected):
    assert _drops_a_lone_last_sample(n, batch_size) is expected


# --- wiring ------------------------------------------------------------------


def _pyg_loader_kwargs(func):
    """Each PyGDataLoader(...) call's keywords, keyed by assigned variable.

    Values come back as normalised source via ast.unparse, so an assertion can
    pin WHICH expression a keyword is bound to rather than merely that the
    keyword is present. Asserting presence alone is not enough here: it passes
    for a hardcoded drop_last=True, which is the exact regression this file
    exists to prevent.

    getsource is safe on a decorated target without any unwrapping of our own.
    A sibling branch decorates train_gnn_v2 with functools.wraps, and
    inspect.getsourcelines calls inspect.unwrap on its argument as its first
    step, so the original function's source is what comes back either way.
    """
    source = textwrap.dedent(inspect.getsource(func))
    out = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        callee = node.value.func
        name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
        if name != "PyGDataLoader":
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out[target.id] = {kw.arg: ast.unparse(kw.value) for kw in node.value.keywords}
    return out


def test_both_training_loaders_bind_drop_last_to_the_predicate():
    """Pins the VALUE, not just the keyword.

    An earlier version of this test asserted only that "drop_last" appeared in
    the call, and an unconditional drop_last=True passed it - the precise defect
    the conditional exists to rule out. Binding to the predicate call is what
    makes the mutation bite.
    """
    kwargs = _pyg_loader_kwargs(tg.train_gnn_v2)

    assert kwargs["train_loader"]["drop_last"] == (
        "_drops_a_lone_last_sample(len(train_dataset), batch_size)"
    )
    assert kwargs["full_loader"]["drop_last"] == (
        "_drops_a_lone_last_sample(len(full_dataset), batch_size)"
    )


def test_the_validation_loader_does_not():
    """drop_last on the val loader discards evaluation data and fails nothing loudly.

    It would shift the per-epoch early-stopping AUC-PR and desynchronise
    build_oof_records, which indexes val_preds[i] positionally against val_idx.
    """
    kwargs = _pyg_loader_kwargs(tg.train_gnn_v2)

    assert "drop_last" not in kwargs["val_loader"]
