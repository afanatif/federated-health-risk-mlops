import flwr as fl
import torch
from clients.node2.data_loader import get_loaders
from model import SimpleModel

train_loader, val_loader = get_loaders("clients/node2/data")
model = SimpleModel()

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.parameters()]

    def fit(self, parameters, config):
        for param, new_val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_val)
        for X, y in self.train_loader:
            pass
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        for param, new_val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_val)
        loss = 0.5
        accuracy = 0.8
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FLClient(model, train_loader, val_loader)
    )
