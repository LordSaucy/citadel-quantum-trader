import argparse, os

def parse_cli():
    p = argparse.ArgumentParser(description="Monte‑Carlo equity‑curve simulator")
    p.add_argument("--wr", type=float, default=float(os.getenv("WR", "0.999")),
    p.add_argument("--rr", type=float, default=float(os.getenv("RR", "2.0")))
    p.add_argument("--risk", type=float, default=float(os.getenv("RISK_FRAC", "0.01")))
    p.add_argument("--trades", type=int, default=int(os.getenv("N_TRADES", "1000")))
    p.add_argument("--paths", type=int, default=int(os.getenv("N_PATHS", "10000")))
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    p.add_argument("--plot", type=int, default=int(os.getenv("PLOT_EXAMPLES", "0")))
    return p.parse_args()

if __name__ == "__main__":
    args = parse_cli()
    # assign globals from args
    WR, RR, RISK_FRAC = args.wr, args.rr, args.risk
    N_TRADES, N_PATHS, SEED = args.trades, args.paths, args.seed
    PLOT_EXAMPLES = args.plot
    # then the rest of the script runs as before
