#!/usr/bin/env python3
"""
BACK‑TEST REPORTING MODULE

Generates a full performance report (JSON, CSV, PDF + plots) from the
CSV that the Citadel Quantum Trader back‑test validator writes.

Author: Lawful Banker
Created: 2024‑11‑26
Version: 2.0 – Production Ready
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# ----------------------------------------------------------------------
# Third‑party
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Optional S3 upload – only imported when used
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore

# ----------------------------------------------------------------------
# Logging configuration (writes to ./logs/backtest_report.log)
# ----------------------------------------------------------------------
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "backtest_report.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper – pretty‑print numbers
# ----------------------------------------------------------------------
def _fmt(v: Any, ndigits: int = 4) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{ndigits}f}"
    return str(v)


# ----------------------------------------------------------------------
# Core class – builds the report
# ----------------------------------------------------------------------
class BacktestReporter:
    """
    Takes a back‑test CSV (as produced by ``BacktestValidator``) and
    creates three artefacts:
        * summary.json
        * summary.csv
        * report.pdf  (contains equity curve & return histogram)
    """

    def __init__(
        self,
        csv_path: Path,
        output_dir: Path,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None,
    ):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip("/") if s3_prefix else ""

        self.df: Optional[pd.DataFrame] = None
        self.metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1️⃣  Load CSV
    # ------------------------------------------------------------------
    def _load_csv(self) -> None:
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Back‑test CSV not found: {self.csv_path}")

        log.info(f"Loading back‑test CSV: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path, parse_dates=["timestamp"])
        # Ensure required columns exist
        required = {
            "gross_pnl",
            "net_pnl",
            "commission",
            "swap",
            "slippage",
            "direction",
            "symbol",
            "price",
            "quantity",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Sort chronologically (important for equity curve)
        self.df.sort_values("timestamp", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 2️⃣  Compute performance metrics
    # ------------------------------------------------------------------
    def _compute_metrics(self) -> None:
        if self.df is None:
            raise RuntimeError("Dataframe not loaded")

        df = self.df.copy()
        df["gross_cum"] = df["gross_pnl"].cumsum()
        df["net_cum"] = df["net_pnl"].cumsum()

        # ------------------------------------------------------------------
        # Basic aggregates
        # ------------------------------------------------------------------
        total_gross = df["gross_pnl"].sum()
        total_net = df["net_pnl"].sum()
        total_commission = df["commission"].sum()
        total_swap = df["swap"].sum()
        total_slippage = df["slippage"].sum()
        n_trades = len(df)
        n_wins = (df["net_pnl"] > 0).sum()
        win_rate = n_wins / n_trades if n_trades else 0.0
        expectancy = total_net / n_trades if n_trades else 0.0

        # ------------------------------------------------------------------
        # Equity curve & draw‑down
        # ------------------------------------------------------------------
        equity = df["net_cum"]
        peak = equity.cummax()
        drawdown = (peak - equity) / peak.replace(to_replace=0, method="ffill")
        max_dd = drawdown.max() * 100  # percent
        max_dd_idx = drawdown.idxmax()
        max_dd_date = df.loc[max_dd_idx, "timestamp"] if not drawdown.empty else None

        # ------------------------------------------------------------------
        # Sharpe (annualised, assuming 252 trading days)
        # ------------------------------------------------------------------
        # Convert per‑trade returns to daily returns approximating 1 trade per day
        daily_returns = df["net_pnl"] / df["net_pnl"].abs().mean()  # normalised
        if len(daily_returns) > 1:
            sharpe = (
                np.sqrt(252) * daily_returns.mean() / daily_returns.std(ddof=1)
            )
        else:
            sharpe = np.nan

        # ------------------------------------------------------------------
        # Profit factor & trade‑ratio
        # ------------------------------------------------------------------
        gross_wins = df.loc[df["gross_pnl"] > 0, "gross_pnl"].sum()
        gross_losses = -df.loc[df["gross_pnl"] < 0, "gross_pnl"].sum()
        profit_factor = gross_wins / gross_losses if gross_losses else np.inf

        # ------------------------------------------------------------------
        # Calmar (annualised return / max draw‑down)
        # ------------------------------------------------------------------
        # Approximate annual return from net cumulative P&L
        total_return = df["net_cum"].iloc[-1] / df["net_cum"].iloc[0] - 1 if df["net_cum"].iloc[0] != 0 else np.nan
        calmar = (total_return * 252) / (max_dd / 100) if max_dd else np.nan

        # ------------------------------------------------------------------
        # Assemble dictionary
        # ------------------------------------------------------------------
        self.metrics = {
            "total_gross_pnl": total_gross,
            "total_net_pnl": total_net,
            "total_commission": total_commission,
            "total_swap": total_swap,
            "total_slippage": total_slippage,
            "n_trades": n_trades,
            "win_rate_pct": win_rate * 100,
            "expectancy": expectancy,
            "max_drawdown_pct": max_dd,
            "max_drawdown_date": max_dd_date.isoformat() if max_dd_date else None,
            "sharpe_ratio": sharpe,
            "profit_factor": profit_factor,
            "calmar_ratio": calmar,
            "average_trade_gross": df["gross_pnl"].mean(),
            "average_trade_net": df["net_pnl"].mean(),
            "average_commission": df["commission"].mean(),
            "average_swap": df["swap"].mean(),
            "average_slippage": df["slippage"].mean(),
        }

        log.info("Performance metrics computed")
        for k, v in self.metrics.items():
            log.debug(f"{k}: {_fmt(v)}")

    # ------------------------------------------------------------------
    # 3️⃣  Plotting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _style_matplotlib() -> None:
        """Apply a clean visual style (Seaborn + Matplotlib defaults)."""
        sns.set_style("whitegrid")
        plt.rcParams.update(
            {
                "figure.figsize": (12, 6),
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "legend.fontsize": 10,
                "lines.linewidth": 2,
                "font.family": "DejaVu Sans",
            }
        )

    def _plot_equity_curve(self, ax: plt.Axes) -> None:
        df = self.df
        equity = df["net_cum"]
        ax.plot(df["timestamp"], equity, label="Net Equity", color="#0066CC")
        ax.set_title("Equity Curve (Net P&L)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity ($)")
        ax.legend()
        ax.grid(True)

    def _plot_return_histogram(self, ax: plt.Axes) -> None:
        returns = self.df["net_pnl"]
        sns.histplot(returns, bins=30, kde=True, ax=ax, color="#009933")
        ax.set_title("Distribution of Net Trade P&L")
        ax.set_xlabel("Net P&L per trade ($)")
        ax.set_ylabel("Frequency")
        ax.grid(True)

    def _create_plots(self) -> Tuple[Path, Path]:
        """Generate PNG files for the PDF (returns paths)."""
        self._style_matplotlib()
        fig, axes = plt.subplots(2, 1, sharex=False)

        self._plot_equity_curve(axes[0])
        self._plot_return_histogram(axes[1])

        png_path = self.output_dir / "backtest_plots.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Plots saved to {png_path}")
        return png_path, png_path  # both pages are in the same PNG

    # ------------------------------------------------------------------
    # 4️⃣  Export artefacts (JSON, CSV, PDF)
    # ------------------------------------------------------------------
    def _export_json(self) -> Path:
        out_path = self.output_dir / "summary.json"
        with open(out_path, "w") as fp:
            json.dump(self.metrics, fp, indent=2, default=str)
        log.info(f"JSON summary written to {out_path}")
        return out_path

    def _export_csv(self) -> Path:
        out_path = self.output_dir / "summary.csv"
        pd.DataFrame([self.metrics]).to_csv(out_path, index=False)
        log.info(f"CSV summary written to {out_path}")
        return out_path

    def _export_pdf(self, plot_png: Path) -> Path:
        pdf_path = self.output_dir / "report.pdf"
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # ------------------------------------------------------------------
        # Cover page
        # ------------------------------------------------------------------
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 10, "Citadel Quantum Trader – Back‑Test Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Helvetica", "", 12)
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        pdf.cell(0, 10, f"Generated on: {now_str}", ln=True, align="C")
        pdf.ln(20)

        # ------------------------------------------------------------------
        # Metrics table (two‑column layout)
        # ------------------------------------------------------------------
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Key Performance Metrics", ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 11)
        col_width = pdf.w / 2 - 20
        for i, (k, v) in enumerate(self.metrics.items()):
            txt = f"{k.replace('_', ' ').title()}: {_fmt(v)}"
            pdf.multi_cell(col_width, 8, txt, border=0)
            if i % 2 == 1:
                pdf.ln(2)

        pdf.ln(10)

        # ------------------------------------------------------------------
        # Plot page
        # ------------------------------------------------------------------
        pdf.add_page()
        pdf.image(str(plot_png), x=10, y=20, w=pdf.w - 20)

        pdf.output(str(pdf_path))
        log.info(f"PDF report written to {pdf_path}")
        return pdf_path

    # ------------------------------------------------------------------
    # 5️⃣  Optional S3 upload
    # ------------------------------------------------------------------
    def _upload_to_s3(self, local_path: Path) -> None:
        if not self.s3_bucket:
            return
        if boto3 is None:
            log.warning("boto3 not installed – skipping S3 upload")
            return

        s3_key = f"{self.s3_prefix}/{local_path.name}" if self.s3_prefix else local_path.name
        s3 = boto3.client("s3")
        try:
            s3.upload_file(str(local_path), self.s3_bucket, s3_key)
            log.info(f"Uploaded {local_path.name} → s3://{self.s3_bucket}/{s3_key}")
        except (BotoCoreError, ClientError) as exc:
            log.error(f"S3 upload failed for {local_path.name}: {exc}")

    # ------------------------------------------------------------------
    # 6️⃣  Public entry point – run the whole pipeline
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Path]:
        """
        Execute the full reporting pipeline.

        Returns a dict with keys:
            * json   → Path to summary.json
            * csv    → Path to summary.csv
            * pdf    → Path to report.pdf
            * plots  → Path to PNG containing the plots
        """
        log.info("=== BACK‑TEST REPORTING STARTED ===")
        self._load_csv()
        self._compute_metrics()
        plot_png, _ = self._create_plots()
        json_path = self._export_json()
        csv_path = self._export_csv()
        pdf_path = self._export_pdf(plot_png)

        # Optional S3 upload
        for p in (json_path, csv_path, pdf_path, plot_png):
            self._upload_to_s3(p)

        log.info("=== BACK‑TEST REPORTING COMPLETED ===")
        return {
            "json": json_path,
            "csv": csv_path,
            "pdf": pdf_path,
            "plots": plot_png,
        }


# ----------------------------------------------------------------------
# CLI entry point (useful for ad‑hoc runs)
# ----------------------------------------------------------------------
def _parse_cli() -> Tuple[Path, Path, Optional[str], Optional[str]]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate back‑test performance report"
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Path to the back‑test CSV produced by BacktestValidator",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("./backtest_report"),
        help="Directory where artefacts will be written",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="Optional S3 bucket name for artefact upload",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="",
        help="Optional prefix inside the S3 bucket",
    )
    args = parser.parse_args()
    return args.csv, args.output_dir, args.s3_bucket, args.s3_prefix


if __name__ == "__main__":
    csv_path, out_dir, bucket, prefix = _parse_cli()
    reporter = BacktestReporter(csv_path, out_dir, bucket, prefix)
    artefacts = reporter.run()
    print("\nGenerated artefacts:")
    for name, p in artefacts.items():
        print(f"  {name}: {p}")

