# src/utils/heatmap.py
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def plot_depth_heatmap(symbol: str, depth_data: list) -> str:
    """
    depth_data = [{'price': float, 'bid_vol': int, 'ask_vol': int}, ...]
    Returns the absolute path of the saved PNG.
    """
    if not depth_data:
        raise ValueError("Empty depth data – cannot plot heatmap")

    prices = [row["price"] for row in depth_data]
    bids   = [row["bid_vol"] for row in depth_data]
    asks   = [row["ask_vol"] for row in depth_data]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=prices, y=bids, label="Bid Volume", ax=ax, color="green")
    sns.lineplot(x=prices, y=asks, label="Ask Volume", ax=ax, color="red")

    ax.set_title(f"Liquidity heat‑map – {symbol.upper()}")
    ax.set_xlabel("Price")
    ax.set_ylabel("Volume")
    ax.legend(loc="upper right")
    plt.tight_layout()

    # Save under a predictable location (e.g. /tmp/heatmaps)
    out_dir = "/tmp/heatmaps"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{symbol}_{datetime.now():%Y%m%d_%H%M%S}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
