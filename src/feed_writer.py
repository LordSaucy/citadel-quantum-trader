import json, time, sys
from broker_interface import MT5Broker   # reuse the same class the bot uses

def main(pipe_path: str):
    broker = MT5Broker()                     # connects once, stays alive
    with open(pipe_path, 'w') as fifo:
        while True:
            # Pull the newest tick/bar from MT5 (you already have a method)
            tick = broker.get_latest_tick()
            # Serialize to a single‑line JSON (the bots will `json.loads` it)
            fifo.write(json.dumps(tick) + '\n')
            fifo.flush()
            time.sleep(0.5)   # ~2 ticks per second; adjust to your timeframe

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--pipe', required=True, help='Path to FIFO pipe')
    args = p.parse_args()
    main(args.pipe)
