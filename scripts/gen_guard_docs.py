# scripts/gen_guard_docs.py
import yaml, json, pathlib
cfg = yaml.safe_load(open("config.yaml"))
guards = cfg.get("guards", {})
print(json.dumps(guards, indent=2))
