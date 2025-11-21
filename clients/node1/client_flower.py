# clients/node1/client_flower.py
import os
import time
import flwr as fl
import torch
import torch.nn as nn
from typing import Tuple

# model helpers (must exist in model.py at repo root)
from model import get_model, model_to_ndarrays, ndarrays_to_model

# Try to import node-specific get_loaders
try:
    from clients.node1.data_loader import get_loaders
except Exception:
    # fallback: try relative import (if executing from node folder)
    try:
        from data_loader import get_loaders
    except Exception as e:
        raise ImportError("Could not import get_loaders for node1") from e

NODE_ID = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    """
    Try a few common get_loaders signatures:
      - get_loaders(node_dir)
      - get_loaders(csv_path=...)
      - get_loaders(images_dir=..., labels_dir=..., batch_size=...)
    Returns: (train_loader, val_loader)
    """
    node_dir = os.path.join("clients", f"node{NODE_ID}", "data")
    images_dir = os.path.join(node_dir, "images")
    labels_dir = os.path.join(node_dir, "labels")
    # Try 1: node_dir
    try:
        return get_loaders(node_dir)
    except TypeError:
        pass
    except Exception:
        # allow other attempts below
        pass

    # Try 2: csv path (common earlier)
    try:
        csv_path = os.path.join(node_dir, "sample.csv")
        return get_loaders(csv_path)
    except Exception:
        pass

    # Try 3: images_dir / labels_dir signature
    try:
        return get_loaders(images_dir=images_dir, labels_dir=labels_dir, batch_size=16)
    except Exception:
        pass

    # Last: try calling with two positional args
    try:
        return get_loaders(images_dir, labels_dir)
    except Exception as e:
        raise RuntimeError(f"get_loaders failed for node{NODE_ID}: {e}")

# -------------------------
# Build model + client
# -------------------------
train_loader, val_loader = load_data()
model = get_model(pretrained=False)  # use pretrained=False in CI to avoid downloads
model.to(DEVICE)

# Use BCE loss because model ends with Sigmoid (if you used BCEWithLogits, adjust)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

class FLClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    # Flower expects get_parameters without config
    def get_parameters(self):
        return model_to_ndarrays(self.model)

    # set_parameters will be invoked by server
    def set_parameters(self, parameters):
        ndarrays_to_model(self.model, parameters)

    def fit(self, parameters, config):
        # load params
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # small local epoch for demo
            for X, y in self.train_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE).float()
                # If X is images, ensure shape [B,3,H,W]; if tabular, shape [B,features]
                preds = self.model(X).squeeze()
                if preds.dim() == 0:
                    preds = preds.unsqueeze(0)
                loss = criterion(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # load params
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE).float()
                preds = self.model(X).squeeze()
                loss = criterion(preds, y).item()
                total_loss += loss * len(y)
                pred_label = (preds > 0.5).float()
                correct += (pred_label == y).sum().item()
                total += len(y)
        avg_loss = float(total_loss / total) if total > 0 else 0.0
        accuracy = float(correct / total) if total > 0 else 0.0
        return avg_loss, total, {"accuracy": accuracy}

if __name__ == "__main__":
    # Start client and connect to server
    fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient(model, train_loader, val_loader))
