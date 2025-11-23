import yaml
import os
import torch
from pathlib import Path

def load_config(config_path: str):
    """Load YAML config"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device(preferred: str = "auto"):
    """Return 'cuda' if available, else 'cpu'"""
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif preferred.lower() in ["cuda", "cpu"]:
        if preferred.lower() == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "cpu"

def adjust_batch_for_device(batch_size: int, device: str):
    """Automatically scale batch size based on GPU memory"""
    if device == "cuda":
        # Estimate safe batch size (roughly)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        factor = mem_gb / 8  # assume base 8GB for batch_size
        return max(1, int(batch_size * factor))
    return batch_size
