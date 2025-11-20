# clients/node1/client_flower.py
import flwr as fl
import torch
import numpy as np
from server.model import RiskModel
from clients.node1.data_loader import get_loaders

import torch.nn as nn
import torch.optim as optim

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # one local epoch for demo
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                preds = self.model(X)
                loss = self.criterion(preds.squeeze(), y)
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
                preds = self.model(X).squeeze()
                batch_loss = self.criterion(preds, y).item()
                loss += batch_loss * len(y)
                pred_label = (preds > 0.5).float()
                correct += (pred_label == y).sum().item()
                total += len(y)
        return float(loss / total), total, {"accuracy": float(correct / total)}

if __name__ == "__main__":
    train_loader, val_loader = get_loaders("/data/node1/sample.csv")
    model = RiskModel(input_dim=5)
    fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient(model, train_loader, val_loader))
