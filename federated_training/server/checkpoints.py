# federated_training/server/checkpoints.py
import pickle
from pathlib import Path
from typing import Any

CHECKPOINT_DIR = Path("federated_training/server/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

def save_checkpoint(filename: str, obj: Any):
    """
    Save any Python object (e.g., aggregated weights) as a checkpoint.
    """
    path = CHECKPOINT_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(filename: str) -> Any:
    """
    Load a checkpoint.
    """
    path = CHECKPOINT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"Checkpoint loaded: {path}")
    return obj
