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
- **Best Benchmark Configuration**: 256-128-64 ReLU with Dropout 0.2.
- **Optimization**: Adam optimizer, `ReduceLROnPlateau` scheduler (mode="max" tracking AUC-PR).
- **Features**: Consumes continuous tabular representations (30 or 50 features depending on configuration).

## GraphPredictor (GNN, v1 dense-adjacency path)
Defined in `src/gnn/models.py`; trained via `src/train_gnn.py`.

> **Scope of this section.** What follows describes the **v1** dense-adjacency GCN.
> The production-candidate research track is v2.3 (`GraphEncoderV2` /
> `GraphPredictorV2`, GINEConv over PyG `Data`/`Batch` objects with ESM-2 node
> embeddings); see `ARCHITECTURE.md` section 6.2 for it. Both paths share the
> cross-validation and promotion machinery documented below.

### Graph construction (`src/gnn/graph_builder.py`)
- **Node Features**: Amino acids converted to 20-dimensional one-hot tensors
  (`GraphBuilder.sequence_to_node_features`).
- **Graph Topology**: Degree-normalized chain adjacency connecting sequence
  neighbours with self-loops (`GraphBuilder.build_chain_adj`). This is a 1D peptide
  chain graph: there are no MHC nodes and no peptide-MHC contact edges anywhere in
  the GNN track, and the v2 PyG path (`GraphBuilder.build_pyg_chain_graph`) builds
  the same chain-plus-self-loop topology with 3-dim one-hot edge features
  (self-loop / forward / backward).
- **Spatial adjacency is present but off.** `GraphBuilder.build_spatial_adj` loads a
  per-peptide pairwise distance matrix (`{peptide}_dist.pt`) from the configured
  structural cache and falls back to the chain graph when none is cached. The code
  does not source those distances from any particular predictor - an earlier revision
  of this line attributed them to AlphaFold, which the implementation does not
  establish. It is disabled in the shipped configuration
  (`use_spatial_adj: false` in `config.yaml`) and exists only on the v1 path.

### Model architecture (`src/gnn/models.py`)
- **`GCNLayer`**: Hand-rolled `nn.Module` applying `adj @ (x @ W) + b` on a
  dense adjacency matrix.
- **`GraphEncoder`**: Two `GCNLayer` blocks (20 -> 32 -> 64) followed by global
  mean pooling over the node dimension -> `(batch, 64)` embedding.
- **`GraphPredictor`**: Combines `GraphEncoder` with a parallel continuous-feature
  dense block (32 units), then a fusion MLP -> scalar logit.
  - `num_continuous_features` is inferred from the physicochemical feature matrix
    at training time and must match at inference time.

### Training (`src/train_gnn.py`)
- **5-fold peptide-grouped, composite-stratified cross-validation**
  (`src.ml_utils.PeptideGroupedKFold`, built by `build_cv_splits`) at both training
  entry points, so every row sharing a peptide lands in exactly one fold. The track
  ran an ungrouped `StratifiedKFold` until 2026-08-12; that is the leakage class D15
  describes, since every mode-31 feature is a pure function of the peptide string.
  `build_cv_splits` raises rather than degenerating to an ungrouped split if the
  training pool has no `peptide` column.
- **OOF schema** (`build_oof_records`), written to
  `models/gnn_oof_predictions.csv` (and a `_<pooling>`-tagged sibling):
  `peptide,hla_allele,label,gnn_oof_score,fold,splitter`. Previously
  `peptide,label,gnn_oof_score` - any three-column frame is a pre-repair artifact.
  `hla_allele` appears when the corpus supplies it, `(peptide, hla_allele)` being the
  v5 dedup key that joins this frame to the RF OOF frame one-to-one; `fold` and
  `splitter` are per-row so provenance travels with the scores rather than in a
  detachable sidecar.
- Full-dataset retrain; checkpoint saved as `models/gnn/structural_gnn_v2.pth`.

## Tensor Shapes & Boundaries
- **ANN Input**: `(batch, n_features)` - `n_features` inferred dynamically.
- **GNN Node Input** (`node_x`): `(batch, max_len=11, 20)`.
- **GNN Continuous Input** (`feat_x`): `(batch, num_continuous_features)`.
- **GNN Fusion Layer**: `(batch, 64)` pool concat `(batch, 32)` physico -> `(batch, 96)`.

## Promotion Gate
`src/verify/promote_gnn.py` runs 5 gates (AUC-PR >= 0.65 under a peptide-grouped
splitter, per-fold AUC-PR std <= 0.02, latency <= 2x RF, ECE < 0.05, escape
sensitivity >= 80%) before mutating `config.yaml` and the checksum manifest. Gate 1
was re-anchored 2026-08-10 from 0.85, which had been set against the pre-remediation
ungrouped RF baseline now retracted as peptide-leakage-inflated
(`docs/claims_register.md` D15). `GATE1_AUC_PR_MIN` is unchanged at 0.65 and the
threshold is absolute - it is not scaled against any other model's score.

Two of the gates now refuse to score an artifact that cannot prove how it was built:

- **Gate 1** calls `grouped_splitter_violation` first. A frame missing the `splitter`
  column, carrying only nulls in it, or naming anything other than
  `PeptideGroupedKFold` fails the gate with no AUC-PR computed and none printed, so a
  leakage-inflated number is never displayed beside a threshold it was not measured
  against.
- **Gate 2** requires the per-row `fold` column and takes the standard deviation of
  the per-fold AUC-PRs, reporting single-class folds instead of dropping them. Its
  former fallback - a leave-one-row-out jackknife, documented as being enabled by a
  `--save-fold-ids` flag on `train_gnn.py` that never existed - is gone; it measured
  the standard error of one pooled AUC-PR, not cross-fold spread.

`python -m src.verify.promote_gnn --dry-run` evaluates the whole scorecard and reports
the mutations that would follow without writing to `config.yaml` or
`models/model_artifact_checksums.json`.

The tracked `models/gnn_oof_predictions.csv` is a v4-era artifact in the old
three-column schema (14,637 rows, 11,779 unique peptides, pooled AUC-PR 0.7160), so it
fails Gate 1 by precondition and Gate 2 for want of fold identity. Promotion stays
blocked until a v5 GNN run under `PeptideGroupedKFold` produces a fresh OOF frame.

## [PENDING]
- `max_len=11` is hard-coded in `GraphBuilder`; a future shift to MHC-II 15-mers
  will require architectural patches to both `build_chain_adj` and
  `sequence_to_node_features`.
- ~~`torch-geometric` is a declared dependency but is not used in the active GNN
  path.~~ **Resolved.** The v2.3 path is the active one and is built on PyG:
  `GraphEncoderV2` uses `torch_geometric.nn.GINEConv` over PyG `Data`/`Batch`
  objects. The hand-rolled dense-adjacency GCN described above survives as the v1
  path, kept for backward compatibility and tests.
