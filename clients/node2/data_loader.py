# clients/node2/data_loader.py
import os
from torch.utils.data import DataLoader, random_split
from clients.common.image_dataset import ImagePotholeDataset

def get_loaders(images_dir="clients/node2/data/images", labels_dir="clients/node2/data/labels",
                batch_size=16, val_frac=0.1, seed=42):
    ds = ImagePotholeDataset(images_dir, labels_dir)
    total = len(ds)
    val_count = max(1, int(total * val_frac))
    train_count = total - val_count
    generator = None
    try:
        import torch
        generator = torch.Generator().manual_seed(seed)
    except Exception:
        generator = None
    if train_count <= 0:
        train_ds = ds
        val_ds = ds
    else:
        train_ds, val_ds = random_split(ds, [train_count, val_count], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
