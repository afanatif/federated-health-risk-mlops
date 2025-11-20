# clients/node1/client_flower.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

# ensure data available
try:
    from data.download import ensure_all_nodes as ensure_data_ready
    ensure_data_ready()
except Exception:
    try:
        from data.download import ensure_all_nodes as ensure_data_ready
        ensure_data_ready()
    except Exception:
        # fallback: proceed (assume data present)
        pass

from clients.node1.data_loader import get_loaders
from server.model import SimpleCNN

NODE_ID = 3

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for X, y in self.train_loader:
                # X: (B,C,H,W), y: (B,) float tensor
                X = X.float()
                y = y.float()
                self.optimizer.zero_grad()
                preds = self.model(X).squeeze()
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()
        return [val.cpu().numpy() for val in self.model.state_dict().values()], len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.float()
                y = y.float()
                preds = self.model(X).squeeze()
                batch_loss = self.criterion(preds, y).item()
                loss += batch_loss * len(y)
                pred_label = (preds > 0.5).float()
                correct += (pred_label == y).sum().item()
                total += len(y)
        return float(loss / total) if total>0 else float('nan'), total, {"accuracy": float(correct / total) if total>0 else 0.0}

if __name__ == "__main__":
    train_loader, val_loader = get_loaders(images_dir=f"clients/node{NODE_ID}/data/images",
                                           labels_dir=f"clients/node{NODE_ID}/data/labels",
                                           batch_size=16)
    model = SimpleCNN()
    fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient(model, train_loader, val_loader))
