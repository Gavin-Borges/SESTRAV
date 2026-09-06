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
  per-peptide-and-allele pairwise distance matrix
  (`{peptide}_{allele_key}_dist.pt`, e.g. `CLGGLLTMV_A0201_dist.pt`) from the
  configured structural cache and falls back to the chain graph when none is
  cached. Writer and reader share one filename builder
  (`GraphBuilder`'s module-level `structural_cache_filename`), because they
  previously disagreed and every lookup missed silently. The code
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
  `models/gnn_oof_predictions.csv` (and a `_<pooling>`-tagged sibling - note the one
  tagged sibling that is tracked today is a byte-identical duplicate, not a second
  measurement; see the `_mean` note below the promotion gates):
  `peptide,hla_allele,label,gnn_oof_score,fold,splitter`. Previously
  `peptide,label,gnn_oof_score` - any three-column frame is a pre-repair artifact.
  `hla_allele` appears when the corpus supplies it, `(peptide, hla_allele)` being the
  v5 dedup key that joins this frame to the RF OOF frame one-to-one; `fold` and
  `splitter` are per-row so provenance travels with the scores rather than in a
  detachable sidecar.
- **`--edge-mode {full,self-loop-only}`** selects the graph topology on the v2 path.
  `full` is the default chain graph. `self-loop-only` drops the neighbour edges while
  keeping every node feature, which is the N4 ablation. At n=8 seeds removing the edges
  consistently helped (mean paired AUC-PR delta +0.0175, 8 of 8 seeds, sign test
  p = 0.0078), but the 95% CI [0.0107, 0.0244] straddles the pre-registered 0.0160 band,
  so the threshold is not cleared robustly. Read ARCHITECTURE.md 6.3 before quoting it:
  the node features are ESM-2 embeddings that already carry whole-peptide context, so the
  ablation removes redundant local mixing rather than sequence information.
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

`--oof` scores an alternative out-of-fold frame and `--checkpoint` an alternative
checkpoint (Gate 3 latency plus the displayed SHA-256); neither relaxes a gate, and
`--checkpoint` is refused without `--dry-run` so a real promotion cannot certify a
file other than the one just scored.

The tracked `models/gnn_oof_predictions.csv` is a v4-era artifact in the old
three-column schema (14,637 rows, 11,779 unique peptides, pooled AUC-PR 0.7160), so it
fails Gate 1 by precondition and Gate 2 for want of fold identity.

**The `_mean` tracked artifacts are duplicates, not a second measurement, and nothing
should be argued from their existence (recorded 2026-08-24).** `models/gnn_oof_predictions_mean.csv`
is byte-identical to `models/gnn_oof_predictions.csv` - git stores both under the single
blob `3cb0da2` - and `models/gnn/gnn_config_mean.json` is likewise byte-identical to
`models/gnn/gnn_config.json` (blob `a0342d3`). Both `_mean` files were added in the same
commit, `fc9a10d`. The `_mean` suffix distinguishes nothing here: `gnn_config.json`
already declares `"pooling": "mean"`, so the base config was mean-pooled before the
tagged copy existed.

Read `fc9a10d`'s message with that in mind. It records
*"commit canonical mean-pool GNN architecture config (P0 retrain)"* for the JSON and does
**not mention the 14,638-line CSV at all**, so a reader scanning the history sees a
retrain announced beside a large new out-of-fold export and can reasonably infer the one
produced the other. It did not: no rows were produced by that commit that did not already
exist under another name.

Distinct mean-pool **weights** do exist - `models/gnn/structural_gnn_v2_mean.pth`
(sha256 `9421f6f7...`) differs from `models/gnn/structural_gnn_v2.pth` (`039ee362...`) -
but both checkpoints are untracked, and the tracked `_mean` CSV demonstrably does not
correspond to them, since it is the other checkpoint's export. **No out-of-fold frame
from a mean-pool run is tracked in this repository.** Treat the `_mean` CSV as a
duplicate filename, and do not cite it as independent evidence of a mean-pool retrain.
It is left in place rather than deleted because `tests/test_gnn_model_dir_guard.py` and
`tests/test_train_gnn_results_guard.py` both name it; removing it is a test change and a
separate decision.

**A v5 run under `PeptideGroupedKFold` has since been performed, so promotion is no
longer waiting on one (updated 2026-08-15).** It ran 2026-08-13 at feature mode 31 and
returned Gate 1 FAIL at 0.6458 against >= 0.65, Gate 2 FAIL at 0.0234 against <= 0.02,
and Gates 3, 4 and 5 PASS. Promotion stays blocked on that measured null rather than on
the absence of a scoreable frame. Its out-of-fold frame lives under gitignored
`models/scratch/`, which is why the tracked artifact above is unchanged.

## [PENDING]
- `max_len=11` is hard-coded in `GraphBuilder`; a future shift to MHC-II 15-mers
  will require architectural patches to both `build_chain_adj` and
  `sequence_to_node_features`.
- ~~`torch-geometric` is a declared dependency but is not used in the active GNN
  path.~~ **Resolved.** The v2.3 path is the active one and is built on PyG:
  `GraphEncoderV2` uses `torch_geometric.nn.GINEConv` over PyG `Data`/`Batch`
  objects. The hand-rolled dense-adjacency GCN described above survives as the v1
  path, kept for backward compatibility and tests.
