#!/usr/bin/env python3
"""
LSTM Regime Forecasting Module

Implements a simple (yet production‑ready) LSTM that predicts the
probability that the market is in a “bullish” regime (value ∈ [0, 1]).
The model is trained on a sliding window of historical features
(e.g., price returns, volatility, macro indicators) and can be
re‑trained on‑line or offline.

Typical usage:

    >>> from src.lstm_regime import LSTMRegimeModel, RegimeModelError
    >>> model = LSTMRegimeModel(cfg_path="config/regime.yaml")
    >>> model.train(train_df, val_df)          # pandas DataFrames
    >>> probs = model.predict(test_df)         # returns pd.Series
    >>> model.save("models/lstm_regime.pt")   # persist the best checkpoint
"""

import json
import logging
import os
import pathlib
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ----------------------------------------------------------------------
# Optional TensorBoard support – import lazily so the module works without it
# ----------------------------------------------------------------------
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

# ----------------------------------------------------------------------
# Logging configuration (structured, JSON‑compatible)
# ----------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # avoid duplicate handlers when reloading
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(name)s %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S%z',
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

# ----------------------------------------------------------------------
# Custom exception – makes debugging easier for callers
# ----------------------------------------------------------------------
class RegimeModelError(RuntimeError):
    """Raised when something goes wrong inside the LSTM regime model."""


# ----------------------------------------------------------------------
# Utility: deterministic seeding (covers Python, NumPy, PyTorch, CUDA)
# ----------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic CuDNN behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# Simple LSTM network – you can extend it later (bidirectional, dropout, …)
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
        self.fc = nn.Linear(hidden_dim, 1)  # output = probability (sigmoid later)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_dim)
        # Use the **last** time‑step representation for classification
        last_step = lstm_out[:, -1, :]  # (batch, hidden_dim)
        logits = self.fc(last_step)    # (batch, 1)
        return logits.squeeze(-1)      # (batch,)


# ----------------------------------------------------------------------
# Dataset wrapper – handles scaling & sliding‑window creation
# ----------------------------------------------------------------------
class _WindowDataset(Dataset):
    """
    Turns a 2‑D array (samples × features) into a sliding‑window
    dataset suitable for LSTM training.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_samples, n_features). Must be already scaled.
    window : int
        Length of the temporal window (number of timesteps).
    target : np.ndarray | None
        1‑D array of target values aligned with ``data`` (same length).
        If ``None`` the dataset will only return inputs (useful for inference).
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

        # Number of windows we can extract
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

        # Target is aligned with the *last* element of the window
        y = self.target[end - 1]
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor


# ----------------------------------------------------------------------
# Main façade – handles config, training, inference, persistence
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
    # Configuration (populated from a dict or a YAML/JSON file)
    # ------------------------------------------------------------------
    cfg: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Internals – populated after ``_prepare`` is called
    # ------------------------------------------------------------------
    device: torch.device = field(init=False)
    scaler: StandardScaler = field(init=False)
    model: _LSTMNet = field(init=False)
    best_checkpoint_path: Optional[pathlib.Path] = field(default=None, init=False)

    # ------------------------------------------------------------------
    # Default configuration (can be overridden via a file or dict)
    # ------------------------------------------------------------------
    DEFAULT_CFG: Mapping[str, Any] = field(
        default_factory=lambda: {
            "seed": 42,
            "window": 30,                     # timesteps per sample
            "input_dim": None,                # inferred from data if omitted
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "batch_size": 128,
            "lr": 1e-3,
            "max_epochs": 200,
            "patience": 15,                   # early‑stopping patience
            "min_delta": 1e-4,                # improvement threshold
            "checkpoint_dir": "./models/checkpoints",
            "tensorboard_logdir": "./logs/tensorboard",
            "use_tensorboard": False,
        }
    )

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Merge user config with defaults and initialise the device."""
        # Merge with defaults (user config wins)
        merged = dict(self.DEFAULT_CFG)  # shallow copy
        merged.update(self.cfg)          # overwrite defaults
        self.cfg = merged

        # Set deterministic seeds
        seed_everything(int(self.cfg.get("seed", 42)))

        # Device selection (CUDA if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info(f"LSTMRegimeModel initialised on device: {self.device}")

        # Prepare checkpoint directory
        ckpt_dir = pathlib.Path(self.cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer (optional)
        if self.cfg.get("use_tensorboard", False) and SummaryWriter is not None:
            tb_dir = pathlib.Path(self.cfg["tensorboard_logdir"])
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            LOGGER.info(f"TensorBoard logging enabled at {tb_dir}")
        else:
            self.tb_writer = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_model(self, input_dim: int) -> None:
        """Instantiate the LSTM network with the supplied input dimension."""
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
    ) -> Tuple[TensorDataset, TensorDataset, StandardScaler]:
        """
        Scale features, split into train/val, and wrap into ``TensorDataset`` objects.
        Returns (train_dataset, val_dataset, scaler).
        """
        if target_col not in df.columns:
            raise RegimeModelError(f"Target column `{target_col}` not found in DataFrame")

        # Separate features & target
        X_raw = df.drop(columns=[target_col]).values.astype(np.float32)
        y_raw = df[target_col].values.astype(np.float32)

        # Scale features (fit on training set only)
        scaler = StandardScaler()
        if fit_scaler:
            X_scaled = scaler.fit_transform(X_raw)
        else:
            X_scaled = scaler.transform(X_raw)

        # Split (80 % train / 20 % val) – keep temporal order!
        split_idx = int(0.8 * len(df))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_raw[:split_idx], y_raw[split_idx:]

        # Build sliding‑window datasets
        window = int(self.cfg["window"])
        train_ds = _WindowDataset(X_train, window, y_train)
        val_ds = _WindowDataset(X_val, window, y_val)

        return train_ds, val_ds, scaler

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """Persist model state + optimizer + epoch."""
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
            Column name holding the binary regime label (0 = bear, 1 = bull).
        """
        # ------------------------------------------------------------------
        # 1️⃣  Prepare data & scaler (fit on training set only)
        # ------------------------------------------------------------------
        train_ds, val_ds, self.scaler = self._prepare_data(
            pd.concat([train_df, val_df], ignore_index=True),
            target_col=target_col,
            fit_scaler=True,
        )
        # Infer input dimension from the dataset
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
        )
        criterion = nn.BCEWithLogitsLoss()  # combines sigmoid + BCE

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
            self.model.train()
            epoch_losses: List[float] = []

            for xb, yb in train_loader:
                xb = xb.to(self.device)          # (batch, seq, feat)
                yb = yb.to(self.device)          # (batch,)

                self.optimizer.zero_grad()
                logits = self.model(xb)           # (batch,)
                loss = criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            train_loss = np.mean(epoch_losses)

            # ---------------------------------------------------------
            # Validation pass
            # ---------------------------------------------------------
            self.model.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)

            # ---------------------------------------------------------
            # Logging (console + optional TensorBoard)
            # ---------------------------------------------------------
            LOGGER.info(
                f"[Epoch {epoch:04d}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
            )
            if self.tb_writer:
                self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
                self.tb_writer.add_scalar("Loss/val", val_loss, epoch)

            # ---------------------------------------------------------
            # Early‑stopping logic
            # ---------------------------------------------------------
            is_best = val_loss + min_delta < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self 
                # -------------------------------------------------
                # 5️⃣  Checkpointing
                # -------------------------------------------------
                self._save_checkpoint(epoch=epoch, val_loss=val_loss, is_best=is_best)

            else:
                epochs_no_improve += 1
                LOGGER.debug(
                    f"No improvement for {epochs_no_improve} epoch(s) "
                    f"(patience={patience})"
                )

            # -------------------------------------------------
            # 6️⃣  Early‑stopping termination
            # -------------------------------------------------
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
            Must contain the same feature columns that were used during training
            (the target column is ignored if present).
        batch_size : int | None
            Override the batch size for inference; defaults to the training
            ``batch_size`` config value.
        return_proba : bool
            If ``True`` (default) returns the sigmoid‑scaled probability
            (value ∈ [0, 1]); if ``False`` returns the raw logits.

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

        # -----------------------------------------------------------------
        # 1️⃣  Prepare input – scale using the stored scaler
        # -----------------------------------------------------------------
        X_raw = df.values.astype(np.float32)
        X_scaled = self.scaler.transform(X_raw)

        # -----------------------------------------------------------------
        # 2️⃣  Build sliding‑window dataset (no targets needed)
        # -----------------------------------------------------------------
        window = int(self.cfg["window"])
        infer_ds = _WindowDataset(X_scaled, window, target=None)

        # -----------------------------------------------------------------
        # 3️⃣  DataLoader for batched inference
        # -----------------------------------------------------------------
        bs = batch_size or int(self.cfg["batch_size"])
        loader = DataLoader(
            infer_ds,
            batch_size=bs,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count() or 0,
        )

        # -----------------------------------------------------------------
        # 4️⃣  Model forward pass (no grad)
        # -----------------------------------------------------------------
        self.model.eval()
        all_probs: List[float] = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)                     # (B, T, F)
                logits = self.model(xb)                     # (B,)
                if return_proba:
                    probs = torch.sigmoid(logits).cpu().numpy()
                else:
                    probs = logits.cpu().numpy()
                all_probs.extend(probs.tolist())

        # -----------------------------------------------------------------
        # 5️⃣  Align predictions with the original DataFrame index.
        #    The first (window‑1) rows cannot be predicted because we lack a
        #    full temporal context; we fill them with NaN.
        # -----------------------------------------------------------------
        nan_padding = [np.nan] * (window - 1)
        aligned = nan_padding + all_probs
        if len(aligned) != len(df):
            # Defensive check – should never happen unless something went wrong
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
        * ``scaler_state_dict``
        * ``config`` (the merged configuration dict)
        """
        dest = pathlib.Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Prefer the best checkpoint if we have one; otherwise serialize on‑the‑fly
        if self.best_checkpoint_path and self.best_checkpoint_path.is_file():
            checkpoint_src = self.best_checkpoint_path
            torch.save(torch.load(checkpoint_src), dest)
            LOGGER.info(f"Best checkpoint copied to {dest}")
        else:
            # Serialize current state manually
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
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
            self.cfg = {**self.cfg, **loaded_cfg}
            LOGGER.info("Configuration restored from checkpoint")

        # -----------------------------------------------------------------
        # 2️⃣  Determine input dimension (required to rebuild the network)
        # -----------------------------------------------------------------
        # The scaler holds the feature mean/std; we can infer dim from it.
        if "scaler_state_dict" not in checkpoint:
            raise RegimeModelError("Checkpoint missing scaler state.")
        dummy_scaler = StandardScaler()
        dummy_scaler.mean_ = checkpoint["scaler_state_dict"]["mean_"]
        dummy_scaler.scale_ = checkpoint["scaler_state_dict"]["scale_"]
        input_dim = dummy_scaler.mean_.shape[0]

        # -----------------------------------------------------------------
        # 3️⃣  Re‑instantiate model & load weights
        # -----------------------------------------------------------------
        self._build_model(input_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        # -----------------------------------------------------------------
        # 4️⃣  Restore scaler
        # -----------------------------------------------------------------
        self.scaler = dummy_scaler

        LOGGER.info(f"Model and scaler loaded from {src}")

    # ------------------------------------------------------------------
    # Convenience representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        cfg_summary = json.dumps(
            {k: v for k, v in self.cfg.items() if k != "checkpoint_dir"},
            indent=2,
        )
        return (
            f"<LSTMRegimeModel device={self.device} "
            f"config={cfg_summary}>"
        )
