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
    def __init__(self, model_size='n', num_classes=1, pretrained=True, img_size=640):
        """
        Args:
            model_size: 'n', 's', 'm', 'l', 'x' (nano to extra-large)
            num_classes: Number of classes (1 for pothole detection)
            pretrained: Use pretrained COCO weights
            img_size: Input image size (default 640)
        """
        super().__init__()
        
        # Store config
        self.model_size = model_size
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Load YOLOv8 model
        if pretrained:
            # Load pretrained model (COCO weights)
            self.model = YOLO(f'yolov8{model_size}.pt')
        else:
            # Load architecture only (no pretrained weights)
            self.model = YOLO(f'yolov8{model_size}.yaml')
        
        # Override number of classes if needed
        if hasattr(self.model.model, 'nc'):
            self.model.model.nc = num_classes  # Set number of classes
        
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
        model: YOLOv8Wrapper model
    """
    model = YOLOv8Wrapper(
        model_size=model_size,
        num_classes=num_classes,
        pretrained=pretrained,
        img_size=img_size
    )
    
    return model


def model_to_ndarrays(model):
    """
    Convert YOLOv8 model parameters to numpy arrays for Flower.
    
    Args:
        model: YOLOv8Wrapper model
        
    Returns:
        List of numpy arrays (model weights)
    """
    # Get state dict from the underlying YOLO model
    state_dict = model.model.model.state_dict()
    
    # Convert all parameters to numpy
    arrays = [val.cpu().numpy() for val in state_dict.values()]
    
    return arrays


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
    
    # Convert arrays back to tensors and load into model
    new_state_dict = {}
    for k, arr in zip(keys, arrays):
        new_state_dict[k] = torch.from_numpy(arr).float()
    
    # Load into model with strict=False to allow flexibility
    model.model.model.load_state_dict(new_state_dict, strict=False)


def save_model(model, path):
    """
    Save YOLOv8 model weights.
    
    Args:
        model: YOLOv8Wrapper model
        path: Save path (.pt file)
    """
    torch.save(model.model.model.state_dict(), path)


def load_model(model, path):
    """
    Load YOLOv8 model weights.
    
    Args:
        model: YOLOv8Wrapper model
        path: Path to .pt file with state dict
    """
    state_dict = torch.load(path)
    model.model.model.load_state_dict(state_dict, strict=False)


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
# EXAMPLE USAGE & TESTING
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 YOLOv8 Model Testing")
    print("=" * 60)
    
    # Test 1: Create model
    print("\n1️⃣  Testing Model Creation...")
    try:
        model = get_model(model_size='n', num_classes=1, pretrained=False)
        print("✅ YOLOv8 Model Created Successfully")
        print(f"   Model type: YOLOv8-nano")
        print(f"   Number of classes: 1 (pothole)")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Model Creation Failed: {e}")
        exit(1)
    
    # Test 2: Forward pass
    print("\n2️⃣  Testing Forward Pass...")
    try:
        dummy_input = torch.randn(2, 3, 640, 640)  # Batch of 2 images
        output = model(dummy_input)
        print(f"✅ Forward Pass Successful")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output type: {type(output)}")
    except Exception as e:
        print(f"❌ Forward Pass Failed: {e}")
        exit(1)
    
    # Test 3: Model to numpy arrays (Flower serialization)
    print("\n3️⃣  Testing Model Serialization (Flower)...")
    try:
        arrays = model_to_ndarrays(model)
        print(f"✅ Model to Numpy Arrays: {len(arrays)} weight matrices")
        total_params = sum(arr.size for arr in arrays)
        print(f"   Total parameters: {total_params:,}")
    except Exception as e:
        print(f"❌ Serialization Failed: {e}")
        exit(1)
    
    # Test 4: Numpy arrays back to model
    print("\n4️⃣  Testing Model Deserialization (Flower)...")
    try:
        ndarrays_to_model(model, arrays)
        print(f"✅ Numpy Arrays to Model: Success")
    except Exception as e:
        print(f"❌ Deserialization Failed: {e}")
        exit(1)
    
    # Test 5: Forward pass after deserialization
    print("\n5️⃣  Testing Forward Pass After Deserialization...")
    try:
        output = model(dummy_input)
        print(f"✅ Forward Pass Still Works After Weight Update")
    except Exception as e:
        print(f"❌ Forward Pass After Deserialization Failed: {e}")
        exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 ALL MODEL TESTS PASSED!")
    print("=" * 60)
    print("\n✅ Model is ready for:")
    print("   - Federated learning (weights can serialize/deserialize)")
    print("   - Training with DataLoaders")
    print("   - Flower client integration")
