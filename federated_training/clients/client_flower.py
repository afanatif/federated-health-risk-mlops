import os
import numpy as np
import torch
import flwr as fl
from ultralytics import YOLO

from federated_training.clients.client_config import ClientConfig
from federated_training.datasets.dataset import YOLODataset
from federated_training.utils import ensure_dir

from flwr.client import start_numpy_client


class YOLOFLClient(fl.client.NumPyClient):
    """Flower client for YOLOv8 Federated Learning."""

    def __init__(self, node_id: int, config_path: str):
        # Load per-node config
        self.cfg = ClientConfig(node_id=node_id, config_path=config_path)
        self.cfg.summary()

        # Dataset resolver (automatic Yaml + model path)
        self.ds = YOLODataset(
            node_id=node_id,
            batch_size=self.cfg.batch_size,
            img_size=self.cfg.img_size,
            device=self.cfg.device,
        )

        print(f"[Client {node_id}] Using dataset: {self.ds.data_yaml}")
        print(f"[Client {node_id}] Using model: {self.ds.model_path}")

        # Load YOLO model dynamically (n/s/m/l/x/custom)
        self.model = YOLO(self.ds.model_path)

        # Track node_id
        self.node_id = node_id

    # ---------------------------------------
    # Federated Learning API
    # ---------------------------------------
    def get_parameters(self, config=None):
        """Serialize model parameters → numpy arrays."""
        params = [p.detach().cpu().numpy() for p in self.model.model.parameters()]
        return params

    def set_parameters(self, parameters):
        """Load numpy arrays back into the PyTorch model."""
        params = list(self.model.model.parameters())
        for p, new_w in zip(params, parameters):
            p.data = torch.tensor(new_w, device=self.cfg.device)

    def fit(self, parameters, config=None):
        """Run LOCAL TRAINING for one FL round."""
        print(f"\n[Client {self.node_id}] Starting local training...")

        # Load new global weights
        self.set_parameters(parameters)

        # Built-in YOLO training
        self.model.train(
            **self.ds.get_train_args(),
            epochs=self.cfg.epochs,
            device=self.cfg.device,
            verbose=False,
        )

        # Return updated local weights + dataset size
        return self.get_parameters(), self.ds.get_train_size(), {}

    def evaluate(self, parameters, config=None):
        """Local evaluation."""
        print(f"\n[Client {self.node_id}] Evaluating...")

        self.set_parameters(parameters)

        results = self.model.val(
            **self.ds.get_val_args(),
            device=self.cfg.device,
            verbose=False,
        )

        metric = float(results.results_dict.get("metrics/mAP50-95", 0.0))

        return metric, self.ds.get_val_size(), {}


# ---------------------------------------
# Standalone launcher
# ---------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Federated YOLOv8 Client")
    parser.add_argument("--node_id", type=int, required=True)
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    client = YOLOFLClient(node_id=args.node_id, config_path=args.config)
    start_numpy_client(server_address=args.server, client=client)

    # -------------------------------------------------------