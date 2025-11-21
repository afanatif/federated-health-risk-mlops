"""
Data loader for node3.
Flexibly handles different calling patterns from client_flower.py
Fixed: drop_last=True to prevent BatchNorm errors with single-sample batches
"""
import os
import torch
from torch.utils.data import DataLoader, random_split
from clients.common.image_dataset import ImagePotholeDataset


def get_loaders(images_dir=None, labels_dir=None, batch_size=16, val_frac=0.1, seed=42):
    """
    Create train and validation data loaders.
    
    Args:
        images_dir: Path to images directory, or parent data directory, or None for auto-detect
        labels_dir: Path to labels directory or None
        batch_size: Batch size for loaders
        val_frac: Fraction of data to use for validation
        seed: Random seed for reproducible splits
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Auto-detect paths if not provided
    if images_dir is None:
        # Default: use the standard location relative to this file
        current_file = os.path.abspath(__file__)
        node_dir = os.path.dirname(current_file)  # clients/node3/
        data_dir = os.path.join(node_dir, "data")
        images_dir = os.path.join(data_dir, "images")
        labels_dir = os.path.join(data_dir, "labels")
    else:
        # Check if images_dir is actually a parent directory containing images/ and labels/
        if labels_dir is None:
            # Single argument provided - could be data dir or images dir
            if os.path.basename(images_dir) != "images":
                # Assume it's a parent directory
                potential_images = os.path.join(images_dir, "images")
                potential_labels = os.path.join(images_dir, "labels")
                
                if os.path.isdir(potential_images) and os.path.isdir(potential_labels):
                    # It's a parent directory
                    labels_dir = potential_labels
                    images_dir = potential_images
                else:
                    # Try to find labels next to images
                    parent = os.path.dirname(images_dir)
                    labels_dir = os.path.join(parent, "labels")
            else:
                # images_dir already points to images folder
                parent = os.path.dirname(images_dir)
                labels_dir = os.path.join(parent, "labels")
    
    # Ensure both paths are set
    if labels_dir is None:
        parent = os.path.dirname(images_dir)
        labels_dir = os.path.join(parent, "labels")
    
    print(f"[Node3 DataLoader] Loading data from:")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    
    # Verify directories exist
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")
    
    # Create dataset
    ds = ImagePotholeDataset(images_dir, labels_dir)
    total = len(ds)
    
    if total == 0:
        raise ValueError(f"Dataset is empty! Check {images_dir} and {labels_dir}")
    
    print(f"[Node3 DataLoader] Total samples: {total}")
    
    # Split into train and validation
    val_count = max(1, int(total * val_frac))
    train_count = total - val_count
    
    # Create generator for reproducible splits
    generator = None
    try:
        generator = torch.Generator().manual_seed(seed)
    except Exception:
        pass
    
    if train_count <= 0:
        print("[Node3 DataLoader] Warning: Dataset too small, using all for train and val")
        train_ds = ds
        val_ds = ds
    else:
        train_ds, val_ds = random_split(ds, [train_count, val_count], generator=generator)
    
    print(f"[Node3 DataLoader] Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Create loaders
    # drop_last=True prevents BatchNorm errors when last batch has size 1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    print("[Node3 DataLoader] ✓ Loaders created successfully")
    
    return train_loader, val_loader


# Test if run directly
if __name__ == "__main__":
    print("Testing node3 data_loader...")
    try:
        train_loader, val_loader = get_loaders()
        print(f"✓ Success! Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
