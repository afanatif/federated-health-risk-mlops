import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "settings.yaml"

with open(CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

def get_config():
    """Return full config dictionary."""
    return _cfg

def get_path(*keys):
    """Access nested path entries as Path."""
    ref = _cfg
    for k in keys:
        ref = ref[k]
    return Path(ref)
