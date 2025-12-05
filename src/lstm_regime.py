#!/usr/bin/env python3
"""
LSTM Regime Forecasting Module

Implements a production‑ready LSTM that predicts the probability that the
market is in a "bullish" regime (value ∈ [0, 1]).  The model is trained on a
sliding window of historical features (price returns, volatility, macro
indicators) and can be re‑trained offline or on‑line.

Typical usage:

>>> from src.lstm_regime import LSTMRegimeModel, RegimeModelError
>>> model = LSTMRegimeModel(cfg_path="config/regime.yaml")
>>> model.train(train_df, val_df)          # pandas DataFrames
>>> probs = model.predict(test_df)         # returns pd.Series
>>> model.save("models/lstm_regime.pt")   # persist the best checkpoint
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
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

# Optional TensorBoard support – import lazily so the module works without it
try:  # pragma: no cover
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

# ----------------------------------------------------------------------
# Logging (JSON‑compatible, single handler)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------
class RegimeModelError(RuntimeError):
    """Raised when something goes wrong inside the LSTM regime model."""


# ----------------------------------------------------------------------
# Deterministic seeding (Python, NumPy, PyTorch, CUDA)
# ----------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# Simple LSTM network – extensible (bidirectional, dropout, …)
# ----------------------------------------------------------------------
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
        self.fc = nn.Linear(hidden_dim, 1)  # output logits → sigmoid later

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)            # (batch, seq_len, hidden_dim)
        last_step = lstm_out[:, -1, :]        # (batch, hidden_dim)
        logits = self.fc(last_step)           # (batch, 1)
        return logits.squeeze(-1)             # (batch,)


# ----------------------------------------------------------------------
# Sliding‑window Dataset wrapper
# ----------------------------------------------------------------------
class _WindowDataset(Dataset):
    """
    Turns a 2‑D array (samples × features) into a sliding‑window dataset
    suitable for LSTM training.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_samples, n_features). Must already be scaled.
    window : int
        Length of the temporal window (number of timesteps).
    target : np.ndarray | None
        1‑D array of target values aligned with ``data`` (same length).
        If ``None`` the dataset returns only inputs (useful for inference).
    """

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
        x = self.data[start:end]                     # (window, n_features)
        x_tensor = torch.from_numpy(x)

        if self.target is None:
            return x_tensor, None

        y = self.target[end - 1]                     # target aligned to last timestep
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor


# ----------------------------------------------------------------------
# Main façade – config, training, inference, persistence
# ----------------------------------------------------------------------
@dataclass
class LSTMRegimeModel:
    """
    High‑level wrapper around the LSTM network that provides:

    * Config‑driven hyper‑parameters
    * Training with early‑stopping & checkpointing
    * Batch inference returning a Pandas ``Series`` of probabilities
    * Model saving / loading (torch ``.pt`` files)
    * Optional TensorBoard logging
    """

    # ------------------------------------------------------------------
    # Public configuration (can be overridden via a dict or a YAML file)
    # ------------------------------------------------------------------
    cfg: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Internals (populated after ``_prepare``)
    # ------------------------------------------------------------------
    device: torch.device = field(init=False)
    scaler: StandardScaler = field(init=False)
    model: _LSTMNet = field(init=False)
    optimizer: optim.Optimizer = field(init=False)
    best_checkpoint_path: Optional[pathlib.Path] = field(default=None, init=False)
    tb_writer: Optional[SummaryWriter] = field(default=None, init=False)

    # ------------------------------------------------------------------
    # Default configuration (merged with user‑supplied ``cfg``)
    # ------------------------------------------------------------------
    DEFAULT_CFG: Mapping[str, Any] = field(
        default_factory=lambda: {
            "seed": 42,
            "window": 30,                     # timesteps per sample
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "batch_size": 128,
            "lr": 1e-3,
            "weight_decay": 0.0,              # ← added hyper‑parameter
            "max_epochs": 200,
            "patience": 15,
            "min_delta": 1e-4,
            "checkpoint_dir": "./models/checkpoints",
            "tensorboard_logdir": "./logs/tensorboard",
            "use_tensorboard": False,
        }
    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Merge user config with defaults, set seeds, device, and logging."""
        merged = dict(self.DEFAULT_CFG)
        merged.update(self.cfg)               # user values win
        self.cfg = merged

        seed_everything(int(self.cfg.get("seed", 42)))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info(f"LSTMRegimeModel initialised on device: {self.device}")

        # Prepare checkpoint directory
        ckpt_dir = pathlib.Path(self.cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Optional TensorBoard
        if self.cfg.get("use_tensorboard", False) and SummaryWriter is not None:
            tb_dir = pathlib.Path(self.cfg["tensorboard_logdir"])
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            LOGGER.info(f"TensorBoard logging enabled at {tb_dir}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
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

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        fit_scaler: bool = True,
    ) -> Tuple[_WindowDataset, _WindowDataset, StandardScaler]:
        """Scale features, split temporally, and return train/val WindowDatasets."""
        if target_col not in df.columns:
            raise RegimeModelError(f"Target column `{target_col}` not found")

        x_raw = df.drop(columns=[target_col]).values.astype(np.float32)
        y_raw = df[target_col].values.astype(np.float32)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_raw) if fit_scaler else scaler.transform(x_raw)

        split_idx = int(0.8 * len(df))
        x_train, x_val = x_scaled[:split_idx], x_scaled[split_idx:]
        y_train, y_val = y_raw[:split_idx], y_raw[split_idx:]

        window = int(self.cfg["window"])
        train_ds = _WindowDataset(x_train, window, y_train)
        val_ds = _WindowDataset(x_val, window, y_val)

        return train_ds, val_ds, scaler

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """Persist model, optimizer, scaler, and validation loss."""
        ckpt_path = pathlib.Path(self.cfg["checkpoint_dir"]) / f"epoch_{epoch:04d}.pt"
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)
        LOGGER.debug(f"Checkpoint saved: {ckpt_path}")

        if is_best:
            best_path = pathlib.Path(self.cfg["checkpoint_dir"]) / "best.pt"
            torch.save(checkpoint, best_path)
            self.best_checkpoint_path = best_path
            LOGGER.info(f"New best model → {best_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str = "regime_label",
    ) -> None:
        """
        Train the LSTM on the supplied training/validation DataFrames.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data – must contain ``target_col``.
        val_df : pd.DataFrame
            Validation data – same columns as ``train_df``.
        target_col : str, optional
            Column name holding the binary regime label (0 = bear, 1 = bull).
        """
        # ------------------------------------------------------------------
        # 1️⃣  Data preparation (fit scaler on training set only)
        # ------------------------------------------------------------------
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        train_ds, val_ds, self.scaler = self._prepare_data(
            full_df, target_col=target_col, fit_scaler=True
        )
        sample_input, _ = train_ds[0]
        input_dim = sample_input.shape[-1]
        self._build_model(input_dim)

        # ------------------------------------------------------------------
        # 2️⃣  DataLoaders
        # ------------------------------------------------------------------
        batch_sz = int(self.cfg["batch_size"])
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_sz,
            shuffle=True,
            drop_last=False,
            num_workers=os.cpu_count() or 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_sz,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count() or 0,
        )

        # ------------------------------------------------------------------
        # 3️⃣  Optimiser & loss
        # ------------------------------------------------------------------
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg["lr"]),
            weight_decay=float(self.cfg["weight_decay"]),   # ← added
        )
        criterion = nn.BCEWithLogitsLoss()

        # ------------------------------------------------------------------
        # 4️⃣  Training loop with early‑stopping
        # ------------------------------------------------------------------
        best_val_loss = float("inf")
        epochs_no_improve = 0
        max_epochs = int(self.cfg["max_epochs"])
        patience = int(self.cfg["patience"])
        min_delta = float(self.cfg["min_delta"])

        LOGGER.info(
            f"Starting training – max_epochs={max_epochs}, patience={patience}"
        )
        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            # ----- training -----
            self.model.train()                     # ensure training mode
            train_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)

            # ----- validation -----
            self.model.eval()                      # switch to eval mode
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)

            # ----- logging -----
            LOGGER.info(
                f"[Epoch {epoch:04d}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
            )
            if self.tb_writer:
                self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
                self.tb_writer.add_scalar("Loss/val", val_loss, epoch)

            # ----- early‑stopping & checkpointing -----
            is_best = val_loss + min_delta < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self._save_checkpoint(epoch=epoch, val_loss=val_loss, is_best=True)
            else:
                epochs_no_improve += 1
                LOGGER.debug(
                    f"No improvement for {epochs_no_improve} epoch(s) (patience={patience})"
                )

            if epochs_no_improve >= patience:
                LOGGER.info(
                    f"Early stopping triggered after {epoch} epochs – "
                    f"validation loss did not improve by {min_delta} for {patience} consecutive epochs."
                )
                break

        total_time = time.time() - start_time
        LOGGER.info(
            f"Training completed in {total_time:.2f}s – best val_loss={best_val_loss:.6f}"
        )
        if self.tb_writer:
            self.tb_writer.flush()
            self.tb_writer.close()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        return_proba: bool = True,
    ) -> pd.Series:
        """
        Generate regime probabilities for a DataFrame of raw features.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain the same feature columns that were used during training.
        batch_size : int | None
            Override the batch size for inference; defaults to the training
            ``batch_size`` config value.
        return_proba : bool
            If ``True`` (default) returns sigmoid‑scaled probabilities
            (∈ [0, 1]); if ``False`` returns raw logits.

        Returns
        -------
        pd.Series
            Index aligns with ``df.index``; values are probabilities (or logits).
        """
        if not hasattr(self, "model"):
            raise RegimeModelError(
                "Model has not been built/trained or loaded. Call `train()` or `load()` first."
            )
        if not hasattr(self, "scaler"):
            raise RegimeModelError(
                "Feature scaler is missing. Ensure the model was trained or loaded correctly."
            )

        # ----- 1️⃣  Scale input -----
        x_raw = df.values.astype(np.float32)
        # ✅ FIXED: Renamed X_scaled to x_scaled for consistency (lowercase)
        x_scaled = self.scaler.transform(x_raw)

        # ----- 2️⃣  Build sliding‑window dataset (no targets) -----
        window = int(self.cfg["window"])
        # ✅ FIXED: Now using x_scaled (consistent naming, not unused)
        infer_ds = _WindowDataset(x_scaled, window, target=None)

        # ----- 3️⃣  DataLoader for batched inference -----
        bs = batch_size or int(self.cfg["batch_size"])
        loader = DataLoader(
            infer_ds,
            batch_size=bs,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count() or 0,
        )

        # ----- 4️⃣  Model forward pass (no gradient) -----
        self.model.eval()                     # ensure dropout/batchnorm are disabled
        all_outputs: list[float] = []
        sigmoid = nn.Sigmoid()

        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)      # (B, T, F)
                logits = self.model(xb)       # (B,)
                if return_proba:
                    probs = sigmoid(logits).cpu().numpy()
                else:
                    probs = logits.cpu().numpy()
                all_outputs.extend(probs.tolist())

        # ----- 5️⃣  Align predictions with the original DataFrame index -----
        # The first (window‑1) rows cannot be predicted because we lack a full
        # temporal context; we pad them with NaN so the output length matches
        # the input length.
        padding = [np.nan] * (window - 1)
        aligned = padding + all_outputs

        if len(aligned) != len(df):
            # Defensive sanity‑check – should never happen
            raise RegimeModelError(
                f"Prediction length mismatch: expected {len(df)} values, got {len(aligned)}"
            )

        return pd.Series(aligned, index=df.index, name="regime_probability")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: Union[str, pathlib.Path]) -> None:
        """
        Persist the *best* checkpoint (or the current model if no checkpoint
        exists) to ``path``.  The file is a standard Torch ``.pt`` archive
        containing:

        * ``model_state_dict``
        * ``optimizer_state_dict``
        * ``scaler_state_dict``
        * ``config`` (the merged configuration dict)
        """
        dest = pathlib.Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Prefer the best checkpoint if we have one; otherwise serialize on‑the‑fly
        if self.best_checkpoint_path and self.best_checkpoint_path.is_file():
            torch.save(torch.load(self.best_checkpoint_path), dest)
            LOGGER.info(f"Best checkpoint copied to {dest}")
        else:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "config": dict(self.cfg),
            }
            torch.save(checkpoint, dest)
            LOGGER.info(f"Model saved to {dest}")

    def load(self, path: Union[str, pathlib.Path]) -> None:
        """
        Load a previously saved checkpoint (produced by :meth:`save`).  This
        restores the model weights, the feature scaler, and the configuration.
        """
        src = pathlib.Path(path)
        if not src.is_file():
            raise RegimeModelError(f"Checkpoint file not found: {src}")

        checkpoint = torch.load(src, map_location=self.device)

        # -----------------------------------------------------------------
        # 1️⃣  Restore configuration (overwrites any existing cfg)
        # -----------------------------------------------------------------
        loaded_cfg = checkpoint.get("config", {})
        if loaded_cfg:
            # Merge loaded config on top of the current one (preserve defaults)
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

        # -----------------------------------------------------------------
        # 4️⃣  Restore optimizer (if present) and scaler
        # -----------------------------------------------------------------
        if "optimizer_state_dict" in checkpoint:
            # Re‑create optimizer with the same hyper‑parameters (including weight_decay)
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=float(self.cfg["lr"]),
                weight_decay=float(self.cfg["weight_decay"]),
            )
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.scaler = dummy_scaler
        LOGGER.info(f"Model, optimizer, and scaler loaded from {src}")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        # Hide large objects (scaler, model) – only show a concise summary
        safe_cfg = {k: v for k, v in self.cfg.items() if k != "checkpoint_dir"}
        cfg_pretty = json.dumps(safe_cfg, indent=2)
        return (
            f"<LSTMRegimeModel device={self.device} "
            f"config={cfg_pretty}>"
        )
