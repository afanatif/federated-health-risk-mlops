"""
YOLOv8 Model for Pothole Detection (Object Detection)

This uses YOLOv8 nano model for federated learning.
Predicts bounding boxes around potholes.

Features:
- YOLOv8n (lightweight, fast)
- Compatible with YOLO label format
- Utilities for Flower federated learning
"""

from typing import List
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO


class YOLOv8Wrapper(nn.Module):
    """
    Wrapper around YOLOv8 for federated learning compatibility.
    """
    def __init__(self, model_size='n', num_classes=1, pretrained=True):
        """
        Args:
            model_size: 'n', 's', 'm', 'l', 'x' (nano to extra-large)
            num_classes: Number of classes (1 for pothole detection)
            pretrained: Use pretrained COCO weights
        """
        super().__init__()
        
        # Load YOLOv8 model
        if pretrained:
            # Load pretrained model and modify for our classes
            self.model = YOLO(f'yolov8{model_size}.pt')
        else:
            # Load architecture only
            self.model = YOLO(f'yolov8{model_size}.yaml')
        
        # Override number of classes if needed
        self.model.model.nc = num_classes  # Set number of classes
        self.num_classes = num_classes
        
    def forward(self, x):
        """
        Forward pass through YOLOv8.
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            predictions during training, detections during inference
        """
        return self.model.model(x)
    
    def predict(self, x, conf=0.25, iou=0.45):
        """
        Run inference with post-processing.
        
        Args:
            x: Input images
            conf: Confidence threshold
            iou: IoU threshold for NMS
            
        Returns:
            List of detections per image
        """
        return self.model.predict(x, conf=conf, iou=iou, verbose=False)


def get_model(model_size='n', num_classes=1, pretrained=True, img_size=640):
    """
    Create YOLOv8 model for federated learning.
    
    Args:
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        num_classes: Number of object classes (1 for pothole)
        pretrained: Use COCO pretrained weights
        img_size: Input image size (default 640)
        
    Returns:
        model: YOLOv8 model
    """
    model = YOLOv8Wrapper(
        model_size=model_size,
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    # Set image size
    model.model.args['imgsz'] = img_size
    
    return model


def model_to_ndarrays(model):
    """
    Convert YOLOv8 model parameters to numpy arrays for Flower.
    
    Args:
        model: YOLOv8Wrapper model
        
    Returns:
        List of numpy arrays
    """
    # Get state dict from the underlying YOLO model
    state_dict = model.model.model.state_dict()
    return [val.cpu().numpy() for val in state_dict.values()]


def ndarrays_to_model(model, arrays):
    """
    Load numpy arrays into YOLOv8 model (for Flower).
    
    Args:
        model: YOLOv8Wrapper model
        arrays: List of numpy arrays
    """
    state_dict = model.model.model.state_dict()
    keys = list(state_dict.keys())
    
    if len(keys) != len(arrays):
        raise ValueError(f"Length mismatch: {len(keys)} keys vs {len(arrays)} arrays")
    
    # Convert arrays back to tensors
    new_state_dict = {}
    for k, arr in zip(keys, arrays):
        new_state_dict[k] = torch.tensor(arr)
    
    # Load into model
    model.model.model.load_state_dict(new_state_dict, strict=True)


def save_model(model, path):
    """
    Save YOLOv8 model.
    
    Args:
        model: YOLOv8Wrapper model
        path: Save path (.pt file)
    """
    model.model.save(path)


def load_model(model, path):
    """
    Load YOLOv8 model weights.
    
    Args:
        model: YOLOv8Wrapper model
        path: Path to .pt file
    """
    model.model = YOLO(path)


# ============================================
# YOLO LOSS FUNCTION
# ============================================

class YOLOLoss:
    """
    Wrapper for YOLOv8 loss computation.
    Uses YOLOv8's built-in loss calculation.
    """
    def __init__(self, model):
        """
        Args:
            model: YOLOv8Wrapper model
        """
        self.model = model.model.model
        
    def __call__(self, predictions, targets):
        """
        Compute YOLO loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels (YOLO format)
            
        Returns:
            loss: Total loss (box + class + objectness)
        """
        # YOLOv8 computes loss internally during training
        # This is handled by the ultralytics trainer
        return self.model.loss(predictions, targets)


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Create model
    model = get_model(model_size='n', num_classes=1, pretrained=False)
    
    print("✅ YOLOv8 Model Created")
    print(f"   Model type: YOLOv8-nano")
    print(f"   Number of classes: 1 (pothole)")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 640, 640)  # Batch of 2 images
    output = model(dummy_input)
    print(f"\n✅ Forward pass successful")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output type: {type(output)}")
    
    # Test conversion to numpy arrays
    arrays = model_to_ndarrays(model)
    print(f"\n✅ Model to numpy arrays: {len(arrays)} arrays")
    
    # Test loading from numpy arrays
    ndarrays_to_model(model, arrays)
    print(f"✅ Numpy arrays to model: Success")
