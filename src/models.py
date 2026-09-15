"""Model definitions.

Gradient boosting is the production choice here (see README). Random Forest and
Ridge are kept as reference points, and the LSTM is included because deep
sequence models are the obvious thing to reach for on this problem — the point of
benchmarking it fairly is to show it does not earn its cost.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:  # pragma: no cover
    HAS_LGB = False

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

RANDOM_STATE = 42


def make_ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def make_random_forest(n_estimators: int = 300):
    """Tuned forest.

    ``min_samples_leaf=5`` matters more than it looks. Left at the default of 1
    this forest grows 2.28M leaves averaging 1.01 samples each — it memorises the
    training set, produces a model several hundred MB in size, and is no more
    accurate for it.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=5,
        max_features=0.5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_gbm(**overrides):
    """Gradient boosting, LightGBM when available and sklearn otherwise."""
    if HAS_LGB:
        params = dict(
            n_estimators=1200,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        params.update(overrides)
        return lgb.LGBMRegressor(**params)

    return HistGradientBoostingRegressor(
        max_iter=1200,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )


def make_quantile_gbm(quantile: float, n_estimators: int = 800):
    """Quantile regressor for prediction intervals.

    Note: these intervals are under-calibrated out of sample (see README). Use
    conformal calibration on top rather than trusting the raw quantiles.
    """
    if HAS_LGB:
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=n_estimators,
        learning_rate=0.05,
        max_leaf_nodes=63,
        random_state=RANDOM_STATE,
    )


if HAS_TORCH:

    class LSTMForecaster(nn.Module):
        """LSTM over the demand sequence, with calendar features joined at the head.

        v1 of this project fed the LSTM only the raw scaled series while giving the
        forest 26 lag features, which made the comparison meaningless. Here both
        model families see the same information.
        """

        def __init__(self, n_static: int, hidden: int = 64, dropout: float = 0.1):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden + n_static, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, sequence, static):
            _, (hidden_state, _) = self.lstm(sequence)
            joined = torch.cat([hidden_state[-1], static], dim=1)
            return self.head(joined).squeeze(-1)

    def train_lstm(
        seq_train, static_train, y_train,
        seq_val, static_val, y_val,
        max_epochs: int = 25,
        patience: int = 3,
        batch_size: int = 512,
        lr: float = 1e-3,
    ):
        """Train with early stopping on a chronologically held-out tail of the training window."""
        torch.manual_seed(RANDOM_STATE)
        model = LSTMForecaster(n_static=static_train.shape[1])
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        tensors = [
            torch.tensor(a, dtype=torch.float32)
            for a in (seq_train, static_train, y_train)
        ]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*tensors), batch_size=batch_size, shuffle=True
        )
        val_seq, val_static, val_y = [
            torch.tensor(a, dtype=torch.float32) for a in (seq_val, static_val, y_val)
        ]

        best_loss, best_state, epochs_without_gain = np.inf, None, 0
        for epoch in range(max_epochs):
            model.train()
            for seq_batch, static_batch, y_batch in loader:
                optimiser.zero_grad()
                loss = loss_fn(model(seq_batch.unsqueeze(-1), static_batch), y_batch)
                loss.backward()
                optimiser.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(val_seq.unsqueeze(-1), val_static), val_y).item()

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= patience:
                    break

        model.load_state_dict(best_state)
        return model, epoch + 1
