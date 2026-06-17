# SESTRAV ANN & GNN Models

## Abstract
SESTRAV operates two primary deep learning pipelines: `src/model.py` (ANN) and
`src/train_gnn.py` + `src/gnn/` (GNN). Both handle extreme class imbalances
natively via algebraically derived `pos_weight` inverse-frequency tensors coupled
with `BCEWithLogitsLoss`. SMOTE is explicitly banned due to empirical degradation
of AUC.

## FlexibleMLP (ANN)
Defined in `src/model.py`.
- **Architecture**: Dynamically constructed Multi-Layer Perceptron (MLP).
- **Activations**: Supports ReLU, GELU, and LeakyReLU with Kaiming He initialization.
- **Project 2 Best Configuration**: 256-128-64 ReLU with Dropout 0.2.
- **Optimization**: Adam optimizer, `ReduceLROnPlateau` scheduler (mode="max" tracking AUC-PR).
- **Features**: Consumes continuous tabular representations (30 or 50 features depending on configuration).

## GraphPredictor (GNN)
Defined in `src/gnn/models.py`; trained via `src/train_gnn.py`.

### Graph construction (`src/gnn/graph_builder.py`)
- **Node Features**: Amino acids converted to 20-dimensional one-hot tensors
  (`GraphBuilder.sequence_to_node_features`).
- **Graph Topology**: Normalized chain adjacency matrix connecting sequence
  neighbours with self-loops (`GraphBuilder.build_chain_adj`). An optional
  spatial adjacency (`build_spatial_adj`) uses pre-computed AlphaFold pairwise
  distance matrices when a structural cache is available, falling back to the
  chain graph otherwise.

### Model architecture (`src/gnn/models.py`)
- **`GCNLayer`**: Hand-rolled `nn.Module` applying `adj @ (x @ W) + b` on a
  dense adjacency matrix.
- **`GraphEncoder`**: Two `GCNLayer` blocks (20 → 32 → 64) followed by global
  mean pooling over the node dimension → `(batch, 64)` embedding.
- **`GraphPredictor`**: Combines `GraphEncoder` with a parallel continuous-feature
  dense block (32 units), then a fusion MLP → scalar logit.
  - `num_continuous_features` is inferred from the physicochemical feature matrix
    at training time and must match at inference time.

### Training (`src/train_gnn.py`)
- 5-Fold Stratified Cross-Validation; OOF predictions saved to
  `models/gnn_oof_predictions.csv`.
- Full-dataset retrain; checkpoint saved as `models/gnn/structural_gnn_v2.pth`.

## Tensor Shapes & Boundaries
- **ANN Input**: `(batch, n_features)` — `n_features` inferred dynamically.
- **GNN Node Input** (`node_x`): `(batch, max_len=11, 20)`.
- **GNN Continuous Input** (`feat_x`): `(batch, num_continuous_features)`.
- **GNN Fusion Layer**: `(batch, 64)` pool ‖ `(batch, 32)` physico → `(batch, 96)`.

## Promotion Gate
`src/verify/promote_gnn.py` runs 5 gates (AUC-PR ≥ 0.85, fold-std ≤ 0.02,
latency ≤ 2× RF, ECE < 0.05, escape sensitivity ≥ 80%) before mutating
`config.yaml` and the checksum manifest. Gates are blocked until the v4
dataset is finalized and a benchmark pass is confirmed.

## [PENDING]
- `max_len=11` is hard-coded in `GraphBuilder`; a future shift to MHC-II 15-mers
  will require architectural patches to both `build_chain_adj` and
  `sequence_to_node_features`.
- `torch-geometric` is a declared dependency but is not used in the active GNN
  path (hand-rolled dense-adjacency GCN). Decide: adopt PyG message-passing
  primitives or drop the dependency.
