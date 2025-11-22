"""
Data loader for Node 1 - YOLO Object Detection
Loads images and bounding box labels in YOLO format.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from clients.common.image_dataset import YOLODataset


def get_loaders(batch_size=8, img_size=640, val_split=0.2):
    """
    Get train and validation data loaders for Node 1.
    
    Args:
        batch_size: Batch size for training
        img_size: Image size for YOLOv8 (default 640)
        val_split: Validation split ratio (default 0.2)
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    # Data paths for node 1
    data_root = os.path.join("data", "federated", "splits", "iid_5nodes", "node_1")
    images_dir = os.path.join(data_root, "images")
    labels_dir = os.path.join(data_root, "labels")
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Create dataset
    full_dataset = YOLODataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        img_size=img_size
    )
    
    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"📊 Node 1 Data Split:")
    print(f"   Total images: {total_size}")
    print(f"   Train: {train_size} images")
    print(f"   Val: {val_size} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=full_dataset.collate_fn,
        num_workers=0,  # Set to 0 for compatibility
        drop_last=True  # Avoid batch norm issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=full_dataset.collate_fn,
        num_workers=0,
        drop_last=False
    )
    
    print(f"✅ Node 1 DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loader
    print("Testing Node 1 Data Loader...")
    train_loader, val_loader = get_loaders(batch_size=4)
    
    # Get first batch
    images, targets = next(iter(train_loader))
    
    print(f"\n✅ First batch loaded:")
    print(f"   Images shape: {images.shape}")
    print(f"   Batch size: {len(targets)}")
    print(f"   First image boxes: {targets[0]['boxes'].shape}")
    print(f"   First image labels: {targets[0]['labels']}")
