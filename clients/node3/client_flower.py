# clients/nodeX/client_flower.py  (replace X with node number)
import os
import flwr as fl
import torch
import torch.nn as nn

# Import model utilities from models/model.py
from models.model import get_model, model_to_ndarrays, ndarrays_to_model

# Import node-specific loader; fallback if path differs
NODE_ID = 3  # <-- set to 1,2,3 in each file
try:
    loader_mod = __import__(f"clients.node{NODE_ID}.data_loader", fromlist=["get_loaders"])
    get_loaders = getattr(loader_mod, "get_loaders")
except Exception:
    # try local file data_loader.py
    try:
        from data_loader import get_loaders  # fallback
    except Exception as e:
        raise RuntimeError(f"Could not import get_loaders for node{NODE_ID}: {e}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    node_dir = os.path.join("clients", f"node{NODE_ID}", "data")
    images_dir = os.path.join(node_dir, "images")
    labels_dir = os.path.join(node_dir, "labels")
    # try common signatures
    attempts = [
        lambda: get_loaders(node_dir),
        lambda: get_loaders(os.path.join(node_dir, "sample.csv")),
        lambda: get_loaders(images_dir=images_dir, labels_dir=labels_dir, batch_size=16),
        lambda: get_loaders(images_dir, labels_dir),
    ]
    for attempt in attempts:
        try:
            loaders = attempt()
            return loaders
        except TypeError:
            continue
        except FileNotFoundError as e:
            # bubble up file-not-found so user can fix missing images/labels
            raise
        except Exception:
            continue
    raise RuntimeError(f"get_loaders failed for node{NODE_ID}")

# load train/val loaders
train_loader, val_loader = load_data()

# instantiate model and optimizer
model = get_model(pretrained=False)  # CI: avoid external weight downloads
model.to(DEVICE)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    # Accept optional config arg (Flower may pass config keyword)
    def get_parameters(self, config=None):
        return model_to_ndarrays(self.model)

    def set_parameters(self, parameters, config=None):
        ndarrays_to_model(self.model, parameters)

    # Fit may be called with config; accept it
    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(1):
            for X, y in self.train_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE).float()
                preds = self.model(X).squeeze()
                if preds.dim() == 0:
                    preds = preds.unsqueeze(0)
                loss = criterion(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    # Evaluate may be called with config; accept it
    def evaluate(self, parameters, config=None):
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
                total_loss += criterion(preds, y).item() * len(y)
                pred_label = (preds > 0.5).float()
                correct += (pred_label == y).sum().item()
                total += len(y)
        avg_loss = float(total_loss / total) if total > 0 else 0.0
        accuracy = float(correct / total) if total > 0 else 0.0
        return avg_loss, total, {"accuracy": accuracy}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient(model, train_loader, val_loader))
