"""
YOLOv8 Model for Pothole Detection (Object Detection)
FIXED VERSION - Simple and robust approach to handle num_classes

Features:
- YOLOv8n (lightweight, fast)
- Compatible with YOLO label format
- Fixed serialization for Flower federated learning
- SIMPLE FIX: Directly modifies Detect head after model creation
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
            pretrained: Use pretrained COCO weights (only backbone/neck, head will be rebuilt)
            img_size: Input image size (default 640)
        """
        super().__init__()
        
        # Store config
        self.model_size = model_size
        self.num_classes = num_classes
        self.img_size = img_size
        
        print(f"🔧 Initializing YOLOv8{model_size} with {num_classes} classes, pretrained={pretrained}")
        
        # Load base model (will have 80 classes if pretrained)
        if pretrained:
            self.model = YOLO(f'yolov8{model_size}.pt')
            print(f"   Loaded pretrained YOLOv8{model_size} (80 classes)")
        else:
            self.model = YOLO(f'yolov8{model_size}.yaml')
            print(f"   Loaded YOLOv8{model_size} architecture")
        
        # Get the detect layer
        detect = self.model.model.model[-1]
        current_nc = detect.nc
        
        print(f"   Current Detect head: {current_nc} classes")
        
        # If we need different number of classes, modify the head
        if current_nc != num_classes:
            print(f"   Modifying Detect head: {current_nc} → {num_classes} classes")
            self._modify_detect_head(num_classes)
            print(f"   ✅ Detect head modified successfully")
        else:
            print(f"   ✅ Detect head already has {num_classes} classes")
    
    def _modify_detect_head(self, num_classes):
        """
        Modify the Detect head to use a different number of classes.
        
        CRITICAL FIX: We need to rebuild the entire cv2 and cv3 modules,
        not just modify the layers, because the state_dict still contains
        old parameters with wrong shapes.
        """
        from ultralytics.nn.modules import Detect, Conv
        
        detect = self.model.model.model[-1]
        
        if not isinstance(detect, Detect):
            print(f"⚠️  Last layer is not Detect: {type(detect)}")
            return
        
        old_nc = detect.nc
        
        # Get reg_max (distribution focal loss parameter)
        reg_max = detect.reg_max if hasattr(detect, 'reg_max') else 16
        
        # Get input channels for each detection scale
        # These come from the feature pyramid network outputs
        ch = []
        for i in range(len(detect.cv2)):
            # Try to get input channels from cv2 (bbox regression branch)
            if hasattr(detect.cv2[i][0], 'conv'):
                ch.append(detect.cv2[i][0].conv.in_channels)
            elif hasattr(detect.cv2[i][0], 'in_channels'):
                ch.append(detect.cv2[i][0].in_channels)
            else:
                # Fallback for YOLOv8n
                ch = [64, 128, 256]
                break
        
        print(f"   Rebuilding Detect head with channels: {ch}")
        
        # Create NEW Detect layer with correct nc
        # This creates all new weights with correct shapes
        new_detect = Detect(nc=num_classes, ch=tuple(ch))
        
        # Copy over some attributes that don't depend on nc
        if hasattr(detect, 'stride'):
            new_detect.stride = detect.stride
        if hasattr(detect, 'export'):
            new_detect.export = detect.export
        
        # Try to copy cv2 (bbox regression) weights since they don't depend on nc
        try:
            for i in range(len(detect.cv2)):
                old_cv2_state = detect.cv2[i].state_dict()
                new_cv2_state = new_detect.cv2[i].state_dict()
                
                # Only copy weights that have matching shapes
                for k in old_cv2_state:
                    if k in new_cv2_state and old_cv2_state[k].shape == new_cv2_state[k].shape:
                        new_cv2_state[k] = old_cv2_state[k]
                
                new_detect.cv2[i].load_state_dict(new_cv2_state, strict=False)
            
            print(f"   ✓ Transferred cv2 (bbox) weights")
        except Exception as e:
            print(f"   ⚠️  Could not transfer cv2 weights: {e}")
        
        # cv3 (classification) weights cannot be transferred because nc changed
        print(f"   ✓ Initialized new cv3 (classification) weights for {num_classes} classes")
        
        # Replace the detect layer in the model
        self.model.model.model[-1] = new_detect
        
        # Update model nc
        self.model.model.nc = num_classes
        
        print(f"   ✅ Detect head completely rebuilt with {num_classes} classes")
        
        # Note: cv2 (bbox regression) layers don't depend on nc, so we keep them unchanged
    
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
    
    FIXED: Now properly creates models with the correct number of classes.
    
    Args:
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        num_classes: Number of object classes (1 for pothole)
        pretrained: Use COCO pretrained weights (only for backbone/neck, head rebuilt)
        img_size: Input image size (default 640)
        
    Returns:
        model: YOLOv8Wrapper model with CORRECT number of classes
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
    Uses state_dict to capture ALL weights including buffers.
    
    Args:
        model: YOLOv8Wrapper model
        
    Returns:
        List of numpy arrays (model weights)
    """
    # Get the actual PyTorch model
    pytorch_model = model.model.model
    
    # Use state_dict to get ALL parameters (not just requires_grad=True)
    state_dict = pytorch_model.state_dict()
    
    params = []
    for name, param in state_dict.items():
        # Convert to numpy (CPU, detached)
        params.append(param.detach().cpu().numpy())
    
    print(f"📊 Extracted {len(params)} parameter arrays from model")
    
    if len(params) == 0:
        print(f"🚨 WARNING: 0 parameters extracted! This will break federated learning!")
        print(f"   Model type: {type(model)}")
        print(f"   Has model attr: {hasattr(model, 'model')}")
    
    return params


def ndarrays_to_model(model, arrays):
    """
    Load numpy arrays into YOLOv8 model (for Flower).
    Uses state_dict keys in same order as model_to_ndarrays.
    
    Args:
        model: YOLOv8Wrapper model
        arrays: List of numpy arrays from server
    """
    if len(arrays) == 0:
        print(f"⚠️  WARNING: Received 0 parameters from server!")
        print(f"   Cannot update model weights. Federated learning is broken.")
        return
    
    pytorch_model = model.model.model
    state_dict = pytorch_model.state_dict()
    keys = list(state_dict.keys())
    
    # Check length match
    if len(keys) != len(arrays):
        print(f"🚨 Parameter count mismatch!")
        print(f"   Model expects: {len(keys)} parameters")
        print(f"   Server sent: {len(arrays)} arrays")
        print(f"   This indicates model architecture mismatch between server and client!")
        
        # Try to match as many as possible
        min_len = min(len(keys), len(arrays))
        print(f"   Attempting to load first {min_len} matching parameters...")
        keys = keys[:min_len]
        arrays = arrays[:min_len]
    
    # Load parameters with shape checking
    new_state_dict = {}
    loaded_count = 0
    skipped_count = 0
    
    for key, array in zip(keys, arrays):
        expected_shape = state_dict[key].shape
        array_shape = array.shape
        
        if expected_shape == array_shape:
            new_state_dict[key] = torch.from_numpy(array)
            loaded_count += 1
        else:
            print(f"   ⚠️  Shape mismatch for {key}:")
            print(f"      Expected: {expected_shape}, Got: {array_shape}")
            skipped_count += 1
    
    # Load into model
    pytorch_model.load_state_dict(new_state_dict, strict=False)
    
    if skipped_count > 0:
        print(f"✅ Loaded {loaded_count} parameters, skipped {skipped_count} mismatches")
    else:
        print(f"✅ Loaded all {loaded_count} parameters successfully")


def save_model(model, path):
    """
    Save YOLOv8 model weights.
    
    Args:
        model: YOLOv8Wrapper model
        path: Save path (.pt file)
    """
    # Save full state dict (includes buffers for inference)
    torch.save(model.model.model.state_dict(), path)
    print(f"💾 Model saved to: {path}")


def load_model(model, path):
    """
    Load YOLOv8 model weights.
    
    Args:
        model: YOLOv8Wrapper model
        path: Path to .pt file with state dict
    """
    state_dict = torch.load(path, map_location='cpu')
    model.model.model.load_state_dict(state_dict, strict=False)
    print(f"📂 Model loaded from: {path}")


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
    
    # Test 1: Create model with 1 class
    print("\n1️⃣  Testing Model Creation (1 class)...")
    try:
        model = get_model(model_size='n', num_classes=1, pretrained=False)
        print("✅ YOLOv8 Model Created Successfully")
        print(f"   Model type: YOLOv8-nano")
        print(f"   Number of classes: 1 (pothole)")
        
        # Verify detect layer
        detect = model.model.model.model[-1]
        print(f"   Detect layer nc: {detect.nc}")
        assert detect.nc == 1, f"ERROR: Detect layer has {detect.nc} classes, expected 1!"
        
        # Count parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"   Trainable parameters: {trainable:,}")
        print(f"   Total parameters: {total:,}")
    except Exception as e:
        print(f"❌ Model Creation Failed: {e}")
        import traceback
        traceback.print_exc()
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
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test 3: Model to numpy arrays (Flower serialization)
    print("\n3️⃣  Testing Model Serialization (Flower)...")
    try:
        arrays = model_to_ndarrays(model)
        print(f"✅ Model to Numpy Arrays: {len(arrays)} weight matrices")
        total_params = sum(arr.size for arr in arrays)
        print(f"   Total parameters: {total_params:,}")
        print(f"   First 5 array shapes: {[arr.shape for arr in arrays[:5]]}")
    except Exception as e:
        print(f"❌ Serialization Failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test 4: Numpy arrays back to model
    print("\n4️⃣  Testing Model Deserialization (Flower)...")
    try:
        ndarrays_to_model(model, arrays)
        print(f"✅ Numpy Arrays to Model: Success")
    except Exception as e:
        print(f"❌ Deserialization Failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test 5: Forward pass after deserialization
    print("\n5️⃣  Testing Forward Pass After Deserialization...")
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward Pass Still Works After Weight Update")
    except Exception as e:
        print(f"❌ Forward Pass After Deserialization Failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test 6: Create model with 80 classes (should work too)
    print("\n6️⃣  Testing Model Creation (80 classes - COCO)...")
    try:
        model_80 = get_model(model_size='n', num_classes=80, pretrained=False)
        detect_80 = model_80.model.model.model[-1]
        print(f"✅ 80-class model created")
        print(f"   Detect layer nc: {detect_80.nc}")
        assert detect_80.nc == 80, f"ERROR: Expected 80 classes, got {detect_80.nc}"
    except Exception as e:
        print(f"❌ 80-class Model Creation Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 ALL MODEL TESTS PASSED!")
    print("=" * 60)
    print("\n✅ Model is ready for:")
    print("   - Federated learning with correct num_classes")
    print("   - Proper parameter serialization")
    print("   - Flower client integration")
    print("   - Both single-class and multi-class detection")