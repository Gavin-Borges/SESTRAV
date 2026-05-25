"""
SESTRAV Neural Network Model Infrastructure

Provides the FlexibleMLP architecture, training loop with early stopping,
k-fold stratified cross-validation runner, and final model training.

Backported from CMB 523 Project 2 Colab pipeline into the SESTRAV production
codebase.  All models use BCEWithLogitsLoss with inverse-frequency pos_weight
to handle class imbalance algebraically (v2 dataset: 70.3% positive, 2.36:1
ratio).  SMOTE was explicitly rejected based on Project 1 findings (degraded
AUC from 0.620 to 0.582).

Architecture search (Project 2 best): 256-128-64 ReLU dropout 0.2
  AUC-PR = 0.8252 +/- 0.0248 (5-fold CV, 30 features)

Usage:
    from src.model import FlexibleMLP, set_seeds, get_device
    from src.model import train_one_fold, run_cv, train_final_model
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

from src.evaluate_metrics import evaluate

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Training Hyperparameters (Project 2 defaults)
# ---------------------------------------------------------------------------
N_FOLDS = 5
MAX_EPOCHS = 150
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Architecture Search Space (14 configs from Project 2)
# ---------------------------------------------------------------------------
ARCH_CONFIGS = [
    {"name": "64-32 ReLU d0.3",        "hidden": [64, 32],          "activation": "relu",  "dropout": 0.3},
    {"name": "128-64 ReLU d0.3",       "hidden": [128, 64],         "activation": "relu",  "dropout": 0.3},
    {"name": "128-64-32 ReLU d0.3",    "hidden": [128, 64, 32],     "activation": "relu",  "dropout": 0.3},
    {"name": "256-128-64 ReLU d0.3",   "hidden": [256, 128, 64],    "activation": "relu",  "dropout": 0.3},
    {"name": "128-64-32-16 ReLU d0.3", "hidden": [128, 64, 32, 16], "activation": "relu",  "dropout": 0.3},
    {"name": "64-32 GELU d0.3",        "hidden": [64, 32],          "activation": "gelu",  "dropout": 0.3},
    {"name": "128-64-32 GELU d0.3",    "hidden": [128, 64, 32],     "activation": "gelu",  "dropout": 0.3},
    {"name": "256-128-64 GELU d0.3",   "hidden": [256, 128, 64],    "activation": "gelu",  "dropout": 0.3},
    {"name": "128-64-32 Leaky d0.3",   "hidden": [128, 64, 32],     "activation": "leaky", "dropout": 0.3},
    {"name": "128-64-32 ReLU d0.2",    "hidden": [128, 64, 32],     "activation": "relu",  "dropout": 0.2},
    {"name": "128-64-32 ReLU d0.4",    "hidden": [128, 64, 32],     "activation": "relu",  "dropout": 0.4},
    {"name": "64-32 ReLU d0.2",        "hidden": [64, 32],          "activation": "relu",  "dropout": 0.2},
    {"name": "256-128-64 ReLU d0.2",   "hidden": [256, 128, 64],    "activation": "relu",  "dropout": 0.2},
    {"name": "128-64 GELU d0.2",       "hidden": [128, 64],         "activation": "gelu",  "dropout": 0.2},
]


def get_device():
    """Return the best available torch device (GPU preferred)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seeds(seed=SEED):
    """Lock random seeds for reproducibility across runs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FlexibleMLP(nn.Module):
    """Dynamically constructed multi-layer perceptron.

    Builds a feedforward network from a configuration dict specifying
    hidden layer sizes, activation function, and dropout rate.
    Output is a single logit (for BCEWithLogitsLoss).

    Supported activations: 'relu', 'gelu', 'leaky'.

    Project 2 best: FlexibleMLP(30, [256, 128, 64], dropout=0.2, activation='relu')
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "leaky": nn.LeakyReLU,
    }

    def __init__(self, input_dim, hidden_sizes, dropout, activation="relu"):
        super().__init__()
        if activation not in self.ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Options: {list(self.ACTIVATIONS.keys())}"
            )
        act_fn = self.ACTIVATIONS[activation]

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), act_fn(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Apply Kaiming (He) initialization for stable ReLU/GELU training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_pos_weight(y):
    """Compute inverse-frequency pos_weight for BCEWithLogitsLoss.

    Args:
        y: array-like of binary labels.

    Returns:
        float ratio of negative / positive samples.
    """
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0:
        raise ValueError("No positive samples in training data")
    return n_neg / n_pos


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def train_one_fold(model, X_train, y_train, X_val, y_val, pos_weight,
                   max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LEARNING_RATE,
                   batch_size=BATCH_SIZE, device=None):
    """Train a model on one fold with early stopping on AUC-PR.

    Args:
        model:      nn.Module (will be modified in-place).
        X_train:    np.ndarray of shape (n_train, n_features).
        y_train:    np.ndarray of binary labels for training.
        X_val:      np.ndarray of shape (n_val, n_features).
        y_val:      np.ndarray of binary labels for validation.
        pos_weight: float, inverse-frequency class weight.
        max_epochs: maximum training epochs.
        patience:   early stopping patience (epochs without improvement).
        lr:         learning rate for Adam.
        batch_size: mini-batch size.
        device:     torch.device (default: auto-detect).

    Returns:
        (metrics_dict, scaler) where metrics_dict is from evaluate() and
        scaler is the fitted StandardScaler.
    """
    if device is None:
        device = get_device()

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  # nosemgrep
                              drop_last=False)

    pw = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
    )

    best_auc_pr = -1.0
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).cpu().numpy()
            val_probs = _sigmoid(val_logits)

        if len(np.unique(y_val)) < 2:
            auc_pr = 0.0
        else:
            auc_pr = average_precision_score(y_val, val_probs)

        scheduler.step(auc_pr)

        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t).cpu().numpy()
        val_probs = _sigmoid(val_logits)

    return evaluate(y_val, val_probs), scaler


def run_cv(X, y, strat_key, config, pos_weight, n_folds=N_FOLDS, device=None):
    """Run full k-fold stratified CV for a given architecture configuration.

    Args:
        X:          np.ndarray of shape (n_samples, n_features).
        y:          np.ndarray of binary labels.
        strat_key:  np.ndarray of stratification keys (label_virus).
        config:     dict with keys 'hidden', 'dropout', 'activation'.
        pos_weight: float, inverse-frequency class weight.
        n_folds:    number of CV folds.
        device:     torch.device.

    Returns:
        list of metrics dicts (one per fold) from evaluate().
    """
    if device is None:
        device = get_device()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []

    for train_idx, val_idx in skf.split(X, strat_key):
        set_seeds(SEED)
        model = FlexibleMLP(
            input_dim=X.shape[1],
            hidden_sizes=config["hidden"],
            dropout=config["dropout"],
            activation=config["activation"],
        ).to(device)

        metrics, _ = train_one_fold(
            model, X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            pos_weight, device=device,
        )
        fold_metrics.append(metrics)

    return fold_metrics


def train_final_model(X_train, y_train, X_val, y_val, config, pos_weight,
                      max_epochs=200, patience=15, device=None):
    """Train a final model on a large training set for SHAP/export.

    Returns:
        (model, scaler) tuple. Model is on device in eval mode.
    """
    if device is None:
        device = get_device()

    set_seeds(SEED)
    model = FlexibleMLP(
        input_dim=X_train.shape[1],
        hidden_sizes=config["hidden"],
        dropout=config["dropout"],
        activation=config["activation"],
    ).to(device)

    _, scaler = train_one_fold(
        model, X_train, y_train, X_val, y_val, pos_weight,
        max_epochs=max_epochs, patience=patience, device=device,
    )
    return model, scaler
