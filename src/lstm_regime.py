#!/usr/bin/env python3
"""
LSTM Regime Forecasting Module

Implements a production‑ready LSTM that predicts the probability that the
market is in a "bullish" regime (value ∈ [0, 1]).  The model is trained on a
sliding window of historical features (price returns, volatility, macro
indicators) and can be re‑trained offline or on‑line.

✅ FIXED: Set module in eval mode after loading checkpoint state_dict (line 479)

Typical usage:

>>> from src.lstm_regime import LSTMRegimeModel, RegimeModelError
>>> model = LSTMRegimeModel(cfg_path="config/regime.yaml")
>>> model.train(train_df, val_df)          # pandas DataFrames
>>> probs = model.predict(test_df)         # returns pd.Series
>>> model.save("models/lstm_regime.pt")   # persist the best checkpoint
"""

# [Full implementation would be included here - showing only the fixed section for brevity]
# The key fix is in the load() method:

# BEFORE (❌ Missing eval mode):
# def load(self, path):
#     ...
#     self.model.load_state_dict(checkpoint["model_state_dict"])
#     self.model.to(self.device)
#     # ← NO eval() call – model stays in training mode!

# AFTER (✅ Fixed – set eval mode):
# def load(self, path):
#     ...
#     self.model.load_state_dict(checkpoint["model_state_dict"])
#     self.model.to(self.device)
#     # ✅ FIXED: Set module in eval mode after loading
#     self.model.eval()  # ← Disables dropout, batchnorm in inference mode
#     ...

# The full corrected load() method is shown below with the fix applied.

import json
import logging
import os
import pathlib
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

try:  # pragma: no cover
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

# =====================================================================
# Logging
# =====================================================================
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


class RegimeModelError(RuntimeError):
    """Raised when something goes wrong inside the LSTM regime model."""


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _LSTMNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        logits = self.fc(last_step)
        return logits.squeeze(-1)


class _WindowDataset(Dataset):
    """Sliding‑window Dataset for LSTM training."""

    def __init__(
        self,
        data: np.ndarray,
        window: int,
        target: Optional[np.ndarray] = None,
    ) -> None:
        if data.ndim != 2:
            raise ValueError("`data` must be a 2‑D array (samples × features)")
        if target is not None and len(target) != len(data):
            raise ValueError("`target` length must match `data` length")

        self.window = window
        self.data = data.astype(np.float32)
        self.target = target.astype(np.float32) if target is not None else None

        self.n_windows = len(data) - window + 1
        if self.n_windows <= 0:
            raise ValueError(
                f"Window size {window} is larger than the dataset ({len(data)} samples)"
            )

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        start = idx
        end = idx + self.window
        x = self.data[start:end]
        x_tensor = torch.from_numpy(x)

        if self.target is None:
            return x_tensor, None

        y = self.target[end - 1]
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor


@dataclass
class LSTMRegimeModel:
    """High‑level wrapper around LSTM for regime forecasting."""

    cfg: Mapping[str, Any] = field(default_factory=dict)

    device: torch.device = field(init=False)
    scaler: StandardScaler = field(init=False)
    model: _LSTMNet = field(init=False)
    optimizer: optim.Optimizer = field(init=False)
    best_checkpoint_path: Optional[pathlib.Path] = field(default=None, init=False)
    tb_writer: Optional[SummaryWriter] = field(default=None, init=False)

    DEFAULT_CFG: Mapping[str, Any] = field(
        default_factory=lambda: {
            "seed": 42,
            "window": 30,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "batch_size": 128,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "max_epochs": 200,
            "patience": 15,
            "min_delta": 1e-4,
            "checkpoint_dir": "./models/checkpoints",
            "tensorboard_logdir": "./logs/tensorboard",
            "use_tensorboard": False,
        }
    )

    def __post_init__(self) -> None:
        """Initialize model, device, and logging."""
        merged = dict(self.DEFAULT_CFG)
        merged.update(self.cfg)
        self.cfg = merged

        seed_everything(int(self.cfg.get("seed", 42)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info(f"LSTMRegimeModel initialised on device: {self.device}")

        ckpt_dir = pathlib.Path(self.cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.get("use_tensorboard", False) and SummaryWriter is not None:
            tb_dir = pathlib.Path(self.cfg["tensorboard_logdir"])
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            LOGGER.info(f"TensorBoard logging enabled at {tb_dir}")

    def _build_model(self, input_dim: int) -> None:
        """Instantiate the LSTM network."""
        self.model = _LSTMNet(
            input_dim=input_dim,
            hidden_dim=int(self.cfg["hidden_dim"]),
            num_layers=int(self.cfg["num_layers"]),
            dropout=float(self.cfg["dropout"]),
        ).to(self.device)
        LOGGER.info(
            f"LSTM model built – input_dim={input_dim}, hidden_dim={self.cfg['hidden_dim']}, "
            f"layers={self.cfg['num_layers']}, dropout={self.cfg['dropout']}"
        )

    # [Additional methods: _prepare_data, train, predict, save, etc. - omitted for brevity]

    def load(self, path: Union[str, pathlib.Path]) -> None:
        """
        Load a previously saved checkpoint.
        
        ✅ FIXED: Set module in eval mode after loading state_dict
        """
        src = pathlib.Path(path)
        if not src.is_file():
            raise RegimeModelError(f"Checkpoint file not found: {src}")

        checkpoint = torch.load(src, map_location=self.device)

        # -----------------------------------------------------------------
        # 1️⃣  Restore configuration
        # -----------------------------------------------------------------
        loaded_cfg = checkpoint.get("config", {})
        if loaded_cfg:
            self.cfg = {**self.cfg, **loaded_cfg}
            LOGGER.info("Configuration restored from checkpoint")

        # -----------------------------------------------------------------
        # 2️⃣  Determine input dimension from the scaler state
        # -----------------------------------------------------------------
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is None:
            raise RegimeModelError("Checkpoint missing scaler state.")
        dummy_scaler = StandardScaler()
        dummy_scaler.mean_ = scaler_state["mean_"]
        dummy_scaler.scale_ = scaler_state["scale_"]
        input_dim = dummy_scaler.mean_.shape[0]

        # -----------------------------------------------------------------
        # 3️⃣  Re‑instantiate model & load weights
        # -----------------------------------------------------------------
        self._build_model(input_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        # ✅ FIXED: Set module in eval mode after loading state_dict
        # This disables dropout, batch normalization, and other training-only layers
        # Essential for proper inference behavior
        self.model.eval()

        # -----------------------------------------------------------------
        # 4️⃣  Restore optimizer and scaler
        # -----------------------------------------------------------------
        if "optimizer_state_dict" in checkpoint:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=float(self.cfg["lr"]),
                weight_decay=float(self.cfg["weight_decay"]),
            )
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.scaler = dummy_scaler
        LOGGER.info(f"Model, optimizer, and scaler loaded from {src}")

    def __repr__(self) -> str:  # pragma: no cover
        safe_cfg = {k: v for k, v in self.cfg.items() if k != "checkpoint_dir"}
        cfg_pretty = json.dumps(safe_cfg, indent=2)
        return (
            f"<LSTMRegimeModel device={self.device} "
            f"config={cfg_pretty}>"
        )
