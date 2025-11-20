# clients/node1/data_loader.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NodeDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        # Minimal preprocessing: fill NA and select features
        df = df.fillna(0)
        features = ["feat1","feat2","feat3","feat4","feat5"]
        self.X = df[features].values.astype(np.float32)
        self.y = df.get("risk", np.zeros(len(df))).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def get_loaders(csv_path="/data/node1/sample.csv", batch_size=32):
    ds = NodeDataset(csv_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader, loader  # train, val (simple)
