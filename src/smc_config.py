
import yaml
from pathlib import Path

CONFIG_PATH = Path("/app/config/config.yaml")   # current live config
SMC_PARAMS = {}

def load_smc_params():
    global SMC_PARAMS
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    SMC_PARAMS = cfg.get("smc_parameters", {})

def get_param(name: str, default=None):
    return SMC_PARAMS.get(name, default)
