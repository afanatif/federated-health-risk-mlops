import yaml
from pathlib import Path
import torch


class ClientConfig:
    """Loads ALL client-level settings from settings.yaml."""

    def __init__(self, node_id: int, config_path: str):
        config_path = Path(config_path).resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"Config file missing: {config_path}")

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # -----------------------------------
        # Read fields
        # -----------------------------------
        self.node_id = node_id

        fed = self.cfg["federated"]["client"]

        self.batch_size = fed.get("batch_size", 8)
        self.epochs = fed.get("epochs", 1)

        runtime = self.cfg.get("runtime", {})
        if runtime.get("device", "auto") == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = runtime["device"]

        self.img_size = fed.get("img_size", 640)

    def summary(self):
        print("\n===== Client Config =====")
        print(f"Node ID: {self.node_id}")
        print(f"Device: {self.device}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print("=========================\n")
