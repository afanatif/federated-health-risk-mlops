# ============================================
# clients/node3/data_loader.py - Node 3
# ============================================

"""
Data loader for Node 3 - YOLO Object Detection
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from clients.common.image_dataset import YOLODataset


def get_loaders(batch_size=8, img_size=640, val_split=0.2):
    """Get train and validation loaders for Node 3."""
    
    data_root = os.path.join("data", "federated", "splits", "iid_5nodes", "node_3")
    images_dir = os.path.join(data_root, "images")
    labels_dir = os.path.join(data_root, "labels")
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    full_dataset = YOLODataset(images_dir=images_dir, labels_dir=labels_dir, img_size=img_size)
    
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    print(f"📊 Node 3: Train={train_size}, Val={val_size}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=full_dataset.collate_fn, num_workers=0, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=full_dataset.collate_fn, num_workers=0, drop_last=False
    )
    
    print(f"✅ Node 3: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_loaders(batch_size=4)
    images, targets = next(iter(train_loader))
    print(f"✅ Node 3 test: Images {images.shape}, Targets {len(targets)}")
