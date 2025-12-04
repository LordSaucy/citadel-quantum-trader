# scripts/json_to_prom.py
import json, sys
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

if len(sys.argv) != 2:
    sys.stderr.write("Usage: json_to_prom.py <mc_summary.json>\n")
    sys.exit(1)

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

registry = CollectorRegistry()
for metric, stats in data.items():
    g = Gauge(f"mc_{metric}", f"Monte‑Carlo {metric}", registry=registry)
    g.set(stats["mean"])               # you could also push the CI bounds
push_to_gateway("localhost:9091", job="monte_carlo", registry=registry)
print("Metrics pushed to Pushgateway")
