"""
Dataset loader for YOLO format labels.

YOLO Label Format:
Each .txt file contains lines: class_id x_center y_center width height
All values normalized to [0, 1]

Example:
0 0.5 0.5 0.3 0.4  # Pothole at center with 30% width, 40% height
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class YOLODataset(Dataset):
    """
    Dataset that loads images and YOLO format labels for object detection.
    
    Returns:
        image: PIL Image or Tensor
        targets: Dict with 'boxes', 'labels', and image info
    """
    
    def __init__(self, images_dir, labels_dir, img_size=640, transform=None):
        """
        Args:
            images_dir: Path to images folder
            labels_dir: Path to labels folder (YOLO .txt files)
            img_size: Target image size (default 640 for YOLOv8)
            transform: Optional transforms
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.transform = transform
        
        # Get all image files
        self.image_files = [
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        self.image_files.sort()
        
        print(f"📁 Loaded {len(self.image_files)} images from {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get image and YOLO labels.
        
        Returns:
            image: Tensor (3, H, W) normalized
            targets: Dict with boxes, labels, and metadata
        """
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Original image size
        orig_w, orig_h = image.size
        
        # Load YOLO labels
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse YOLO format: class_id x_center y_center width height
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO format (x_center, y_center, w, h) to 
                    # corner format (x1, y1, x2, y2) - still normalized [0,1]
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)
        
        # Resize image to target size
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Convert to tensor and normalize
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # Convert boxes and labels to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # No objects in image
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Create targets dict (YOLOv8 format)
        targets = {
            'boxes': boxes,           # Normalized [0, 1] coordinates
            'labels': labels,         # Class IDs
            'image_id': idx,
            'img_path': img_path,
            'orig_size': (orig_w, orig_h),
            'resized_size': (self.img_size, self.img_size)
        }
        
        return image, targets
    
    def collate_fn(self, batch):
        """
        Custom collate function for batching.
        YOLOv8 expects specific format.
        """
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        # Stack images
        images = torch.stack(images, 0)
        
        return images, targets


# ============================================
# HELPER FUNCTION FOR YOLOV8 FORMAT
# ============================================

def convert_targets_to_yolov8_format(targets_list, batch_idx):
    """
    Convert targets to YOLOv8 training format.
    
    YOLOv8 expects: [batch_idx, class_id, x_center, y_center, width, height]
    All normalized to [0, 1]
    
    Args:
        targets_list: List of target dicts from dataset
        batch_idx: Index of current batch
        
    Returns:
        labels: Tensor (N, 6) where N is total number of boxes
    """
    all_labels = []
    
    for i, targets in enumerate(targets_list):
        boxes = targets['boxes']  # (num_boxes, 4) in x1,y1,x2,y2 format
        class_ids = targets['labels']  # (num_boxes,)
        
        if len(boxes) == 0:
            continue
        
        # Convert from corner format back to center format
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Create labels in YOLOv8 format
        batch_indices = torch.full((len(boxes),), i, dtype=torch.float32)
        
        labels = torch.stack([
            batch_indices,
            class_ids.float(),
            x_center,
            y_center,
            width,
            height
        ], dim=1)
        
        all_labels.append(labels)
    
    if len(all_labels) > 0:
        return torch.cat(all_labels, dim=0)
    else:
        # Return empty tensor if no boxes
        return torch.zeros((0, 6), dtype=torch.float32)


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Test dataset
    images_dir = "data/federated/splits/iid_5nodes/node_1/images"
    labels_dir = "data/federated/splits/iid_5nodes/node_1/labels"
    
    if os.path.exists(images_dir) and os.path.exists(labels_dir):
        dataset = YOLODataset(images_dir, labels_dir, img_size=640)
        
        print(f"\n✅ Dataset created with {len(dataset)} images")
        
        # Test loading one sample
        img, targets = dataset[0]
        print(f"\n📊 Sample 0:")
        print(f"   Image shape: {img.shape}")
        print(f"   Number of boxes: {len(targets['boxes'])}")
        print(f"   Boxes: {targets['boxes']}")
        print(f"   Labels: {targets['labels']}")
        
        # Test collate function
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset, 
            batch_size=2, 
            collate_fn=dataset.collate_fn,
            shuffle=False
        )
        
        images, targets_list = next(iter(loader))
        print(f"\n✅ Batch loaded:")
        print(f"   Batch images shape: {images.shape}")
        print(f"   Number of samples: {len(targets_list)}")
        
        # Convert to YOLOv8 format
        yolo_labels = convert_targets_to_yolov8_format(targets_list, 0)
        print(f"   YOLOv8 labels shape: {yolo_labels.shape}")
        print(f"   YOLOv8 labels format: [batch_idx, class, x_center, y_center, w, h]")
    else:
        print(f"❌ Data directories not found:")
        print(f"   Images: {images_dir}")
        print(f"   Labels: {labels_dir}")
