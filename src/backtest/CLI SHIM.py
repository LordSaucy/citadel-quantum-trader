#!/usr/bin/env python3
"""
BACK‑TEST CLI SHIM

A production‑ready command‑line entry point that drives the
`src.backtest_validator.BacktestValidator` class, writes artefacts
(CSV + JSON) and returns a proper exit status for CI pipelines.

Usage example::

    python -m backtest.main \\
        --symbol EURUSD \\
        --start 2024-06-01 \\
        --end   2024-08-31 \\
        --baseline baseline_expectancy.json \\
        --improvement 5          # require ≥5 % improvement over baseline

The shim can also be imported and called directly:

    from backtest.main import run_backtest_cli
    run_backtest_cli([...])
"""

# ----------------------------------------------------------------------
# Standard library
# ----------------------------------------------------------------------
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# ----------------------------------------------------------------------
# Project imports (relative to repository root)
# ----------------------------------------------------------------------
# The validator lives in the `src` package; we import it lazily so the
# shim can be executed even if the rest of the repo is not on PYTHONPATH.
try:
    from src.backtest_validator import BacktestValidator
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Unable to import BacktestValidator – make sure the repository root "
        "is on PYTHONPATH (e.g. `export PYTHONPATH=$(pwd)`)"
    ) from exc

# ----------------------------------------------------------------------
# Logging configuration (mirrors the rest of the repo)
# ----------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
HANDLER = logging.StreamHandler(sys.stdout)
HANDLER.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
LOGGER.addHandler(HANDLER)


# ----------------------------------------------------------------------
# Helper – pretty‑print a dictionary as a table
# ----------------------------------------------------------------------
def _pretty_print(title: str, data: dict) -> None:
    LOGGER.info("=" * 60)
    LOGGER.info(title)
    LOGGER.info("-" * 60)
    for k, v in data.items():
        LOGGER.info(f"{k:30}: {v}")
    LOGGER.info("=" * 60)


# ----------------------------------------------------------------------
# Core function – can be called from CLI or imported programmatically
# ----------------------------------------------------------------------
def run_backtest_cli(
    *,
    symbol: str,
    start: str,
    end: str,
    baseline: Optional[Path] = None,
    improvement: float = 0.0,
    output_dir: Optional[Path] = None,
) -> int:
    """
    Execute a full back‑test run.

    Parameters
    ----------
    symbol : str
        Instrument ticker (e.g. ``EURUSD``).
    start, end : str
        ISO‑8601 dates (``YYYY‑MM‑DD``) defining the slice of historical data.
    baseline : pathlib.Path, optional
        JSON file containing the baseline ``expectancy`` (used for the
        ``improvement`` check).  If omitted, the improvement check is skipped.
    improvement : float, default 0.0
        Required percentage improvement over the baseline expectancy
        (e.g. ``5`` → require ≥ 5 % higher expectancy).
    output_dir : pathlib.Path, optional
        Directory where artefacts are written.  Defaults to the current working
        directory.

    Returns
    -------
    int
        ``0`` if the back‑test passes **all** checks, ``1`` otherwise.
    """
    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    out_dir = Path(output_dir) if output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "backtest_output.csv"
    json_path = out_dir / "backtest_summary.json"

    # ------------------------------------------------------------------
    # Parse dates
    # ------------------------------------------------------------------
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    except ValueError as exc:
        LOGGER.error("Date parsing error – use YYYY-MM-DD format")
        return 1

    # ------------------------------------------------------------------
    # Load baseline expectancy (if supplied)
    # ------------------------------------------------------------------
    baseline_expectancy: Optional[float] = None
    if baseline:
        try:
            with open(baseline, "r") as f:
                baseline_json = json.load(f)
                baseline_expectancy = float(baseline_json.get("expectancy", 0.0))
                LOGGER.info(
                    f"Baseline expectancy loaded: {baseline_expectancy:.6f}"
                )
        except Exception as exc:
            LOGGER.error(f"Failed to read baseline file '{baseline}': {exc}")
            return 1

    # ------------------------------------------------------------------
    # Run the validator
    # ------------------------------------------------------------------
    try:
        validator = BacktestValidator(
            symbol=symbol,
            start=start_dt,
            end=end_dt,
            baseline_expectancy=baseline_expectancy,
        )
        validator.run()
    except Exception as exc:  # pragma: no cover
        LOGGER.exception(f"Back‑test execution failed: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Export artefacts
    # ------------------------------------------------------------------
    try:
        validator.export_trades_to_csv(csv_path)
        LOGGER.info(f"Trade CSV written to: {csv_path}")

        # Build a concise JSON summary for CI / audit consumption
        summary = {
            "symbol": symbol,
            "period": {"start": start, "end": end},
            "gross_expectancy": validator.gross_expectancy,
            "net_expectancy": validator.net_expectancy,
            "max_drawdown_pct": validator.max_drawdown,
            "risk_per_trade_pct": validator.risk_per_trade * 100,
            "improvement_required_pct": improvement,
            "improvement_met": validator.improvement_met,
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        LOGGER.info(f"Summary JSON written to: {json_path}")
    except Exception as exc:  # pragma: no cover
        LOGGER.error(f"Failed to write artefacts: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Pretty‑print a human‑readable report
    # ------------------------------------------------------------------
    report = {
        "Gross expectancy": f"{validator.gross_expectancy:.6f}",
        "Net expectancy": f"{validator.net_expectancy:.6f}",
        "Max draw‑down (%)": f"{validator.max_drawdown:.2f}",
        "Risk per trade (%)": f"{validator.risk_per_trade * 100:.3f}",
        "Improvement required (%)": f"{improvement:.2f}",
        "Improvement met": str(validator.improvement_met),
    }
    _pretty_print("BACK‑TEST RESULT SUMMARY", report)

    # ------------------------------------------------------------------
    # Decide exit status
    # ------------------------------------------------------------------
    if validator.improvement_met:
        LOGGER.info("✅ Back‑test PASSED all configured tolerances")
        return 0
    else:
        LOGGER.warning("❌ Back‑test FAILED – improvement threshold not met")
        return 1


# ----------------------------------------------------------------------
# CLI entry point (invoked via `python -m backtest.main …`)
# ----------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="Run a full back‑test slice and emit artefacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Instrument ticker (e.g. EURUSD)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY‑MM‑DD)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY‑MM‑DD)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline JSON containing an 'expectancy' field",
    )
    parser.add_argument(
        "--improvement",
        type=float,
        default=0.0,
        help="Required % improvement over baseline expectancy",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write CSV/JSON artefacts (defaults to cwd)",
    )
    args = parser.parse_args()

    exit_code = run_backtest_cli(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        baseline=args.baseline,
        improvement=args.improvement,
        output_dir=args.out_dir,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    _cli()
