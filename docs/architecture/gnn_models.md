# SESTRAV ANN & GNN Models

## Abstract
SESTRAV operates two primary deep learning pipelines located in `src/model.py` and `src/train_gnn.py` (`src/gnn_module/`). 
Both handle extreme class imbalances natively by leveraging algebraically derived `pos_weight` inverse-frequency tensors coupled with `BCEWithLogitsLoss`. SMOTE is explicitly banned due to empirical degradation of AUC.

## FlexibleMLP (ANN)
Defined in `src/model.py`.
- **Architecture**: Dynamically constructed Multi-Layer Perceptron (MLP).
- **Activations**: Supports ReLU, GELU, and LeakyReLU with Kaiming He initialization.
- **Project 2 Best Configuration**: 256-128-64 ReLU with Dropout 0.2.
- **Optimization**: Adam optimizer, `ReduceLROnPlateau` scheduler (mode="max" tracking AUC-PR).
- **Features**: Consumes continuous tabular representations (30 or 50 features depending on configuration).

## PeptideGNN (GNN)
Defined in `src/gnn_module/gnn_model.py` and orchestrated by `src/train_gnn.py`.
- **Node Features**: Amino acids converted to 20-dimensional one-hot encoded tensors (`seq_to_node_features`).
- **Graph Topology**: Precomputed chain adjacency matrix connecting sequence neighbors with self-loops (`build_chain_adj`).
- **Architecture**:
  - Dual `GCNConv` graph layers (20 -> 32 -> 64) mapping molecular topologies.
  - Global Mean Pooling across sequence nodes.
  - Parallel continuous-feature dense block mapping the physicochemical SESTRAV tabular features (32 units).
  - Fusion block concatenating the GNN output (64) and continuous block (32) into a dense classification head.
- **Evaluation**: 5-Fold Stratified Cross-Validation on combined features.

## Tensor Shapes & Boundaries
- **ANN Input**: `X_tr_t` shape `(batch, n_features)` where `n_features` is dynamically inferred from the extraction layer.
- **GNN Node Input**: `node_x` shape `(batch, max_len=11, 20)`.
- **GNN Continuous Input**: `feat_x` shape `(batch, num_continuous_features)`.
- **GNN Fusion Layer**: Connects a `(batch, 64)` pool tensor and `(batch, 32)` physical tensor into a `(batch, 96)` bridge.

## [PENDING REVIEW]
- The graph sequence matrix explicitly enforces `max_len=11`. If dataset length constraints shift to 15mers (MHC-II bindings in future), `build_chain_adj` and `sequence_to_node_features` will require hard architectural patches.
