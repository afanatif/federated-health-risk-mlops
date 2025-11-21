# clients/node2/client_flower.py
import os
import flwr as fl
import torch
import torch.nn as nn
from model import get_model, model_to_ndarrays, ndarrays_to_model

try:
    from clients.node2.data_loader import get_loaders
except Exception:
    try:
        from data_loader import get_loaders
    except Exception as e:
        raise ImportError("Could not import get_loaders for node2") from e

NODE_ID = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    node_dir = os.path.join("clients", f"node{NODE_ID}", "data")
    images_dir = os.path.join(node_dir, "images")
    labels_dir = os.path.join(node_dir, "labels")
    try:
        return get_loaders(node_dir)
    except Exception:
        pass
    try:
        csv_path = os.path.join(node_dir, "sample.csv")
        return get_loaders(csv_path)
    except Exception:
        pass
    try:
        return get_loaders(images_dir=images_dir, labels_dir=labels_dir, batch_size=16)
    except Exception as e:
        raise RuntimeError(f"get_loaders failed for node{NODE_ID}: {e}")

train_loader, val_loader = load_data()
model = get_model(pretrained=False)
model.to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self):
        return model_to_ndarrays(self.model)

    def set_parameters(self, parameters):
        ndarrays_to_model(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for X, y in self.train_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE).float()
                preds = self.model(X).squeeze()
                loss = criterion(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
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
