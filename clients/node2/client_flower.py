"""
Federated Learning Client for YOLOv8
Node 1/2/3 - FIXED VERSION with consistent model architecture

USAGE:
- Copy this file to clients/node1/client_flower.py (set NODE_ID = 1)
- Copy this file to clients/node2/client_flower.py (set NODE_ID = 2)
- Copy this file to clients/node3/client_flower.py (set NODE_ID = 3)
"""

# ============================================
# CRITICAL: Import PyTorch 2.6 fix FIRST!
# ============================================
import os
import sys

# Add project root to path FIRST
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import the fix BEFORE any other imports
import fix_pytorch26  # This patches torch.load globally

# Now safe to import everything else
import torch
import flwr as fl
from ultralytics import YOLO
from pathlib import Path
import yaml
import json
import traceback

# Optimization
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# Add project root to path (again for safety)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from models.model import get_model, model_to_ndarrays, ndarrays_to_model

# ============================================
# NODE CONFIGURATION - CHANGE THIS FOR EACH NODE
# ============================================
NODE_ID = 2  # ⚠️  CHANGE TO 2 FOR NODE2, 3 FOR NODE3
# ============================================

# ============================================
# MODEL CONFIGURATION - MUST MATCH SERVER!
# ============================================
MODEL_SIZE = 'n'  # MUST match server --model-size
NUM_CLASSES = 1   # MUST match server --num-classes
# ============================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {DEVICE}")

# Create model with EXACT same config as server
print(f"\n🤖 Creating YOLOv8 model...")
print(f"   Model size: {MODEL_SIZE}")
print(f"   Number of classes: {NUM_CLASSES}")
model = get_model(model_size=MODEL_SIZE, num_classes=NUM_CLASSES, pretrained=False)
model.to(DEVICE)

# Count parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
param_arrays = len(model_to_ndarrays(model))

print(f"✅ Model created:")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Total parameters: {total_params:,}")
print(f"   Parameter arrays: {param_arrays}")

# Show first 5 parameter shapes for debugging
print(f"\n   First 5 parameter array shapes:")
arrays = model_to_ndarrays(model)
for i in range(min(5, len(arrays))):
    print(f"     [{i}] {arrays[i].shape}")


def verify_data_exists(node_id):
    """Verify data exists and return detailed info"""
    print(f"\n{'='*80}")
    print(f"🔍 VERIFYING DATA FOR NODE {node_id}")
    print(f"{'='*80}")
    
    data_root = os.path.abspath(f'data/federated/splits/iid_5nodes/node_{node_id}')
    images_dir = os.path.join(data_root, 'images')
    labels_dir = os.path.join(data_root, 'labels')
    
    # Check directories
    for path, name in [(data_root, "Root"), (images_dir, "Images"), (labels_dir, "Labels")]:
        status = "✓" if os.path.exists(path) else "❌"
        print(f"   {name}: {path} {status}")
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"❌ IMAGES DIR NOT FOUND: {images_dir}")
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"❌ LABELS DIR NOT FOUND: {labels_dir}")
    
    # Count files
    jpg_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.JPG"))
    txt_files = list(Path(labels_dir).glob("*.txt"))
    
    print(f"\n📊 Files found:")
    print(f"   Images: {len(jpg_files)}")
    print(f"   Labels: {len(txt_files)}")
    
    if len(jpg_files) == 0:
        raise ValueError(f"❌ NO IMAGES FOUND in {images_dir}")
    if len(txt_files) == 0:
        raise ValueError(f"❌ NO LABELS FOUND in {labels_dir}")
    
    # Show samples
    print(f"\n📁 Sample files:")
    for img in jpg_files[:3]:
        print(f"   📷 {img.name}")
    for lbl in txt_files[:3]:
        print(f"   📝 {lbl.name}")
    
    return {
        'data_root': data_root,
        'images_dir': images_dir,
        'labels_dir': labels_dir,
        'num_images': len(jpg_files),
        'num_labels': len(txt_files)
    }


def fix_labels(labels_dir):
    """Fix labels - ensure all class IDs are 0"""
    print(f"\n🔧 Fixing labels (ensuring class ID = 0)...")
    fixed_count = 0
    total_boxes = 0
    
    for label_file in Path(labels_dir).glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            fixed_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = '0'  # Ensure class ID is 0
                    fixed_lines.append(' '.join(parts) + '\n')
                    total_boxes += 1
            
            with open(label_file, 'w') as f:
                f.writelines(fixed_lines)
            
            fixed_count += 1
        except Exception as e:
            print(f"   ⚠️  Error fixing {label_file.name}: {e}")
    
    print(f"✅ Fixed {fixed_count} label files ({total_boxes} bounding boxes)")


def create_dataset_yaml(node_id, data_root):
    """Create YOLO dataset.yaml"""
    yaml_path = f"dataset_node{node_id}.yaml"
    config = {
        'path': data_root,
        'train': 'images',
        'val': 'images',
        'nc': NUM_CLASSES,  # Use global config
        'names': ['pothole'] if NUM_CLASSES == 1 else [f'class_{i}' for i in range(NUM_CLASSES)]
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\n📝 Created {yaml_path} (nc={NUM_CLASSES})")
    return yaml_path


class YOLOFLClient(fl.client.NumPyClient):
    """Flower client for YOLOv8 federated learning - FIXED VERSION"""
    
    def __init__(self, model, node_id):
        self.model = model
        self.node_id = node_id
        self.device = DEVICE
    
    def get_parameters(self, config=None):
        """
        Get model parameters as numpy arrays.
        CRITICAL FIX: Always return FULL parameter list, not just trainable subset.
        """
        try:
            # Get ALL parameters (not just requires_grad=True)
            params = model_to_ndarrays(self.model)
            
            if len(params) == 0:
                print(f"🚨 CRITICAL ERROR: 0 parameters extracted!")
                print(f"   Federated learning will not work!")
                print(f"   Model state:")
                print(f"   - Type: {type(self.model)}")
                print(f"   - Has model attr: {hasattr(self.model, 'model')}")
                if hasattr(self.model, 'model'):
                    print(f"   - Model.model type: {type(self.model.model)}")
                    if hasattr(self.model.model, 'model'):
                        print(f"   - Model.model.model type: {type(self.model.model.model)}")
                        state_dict = self.model.model.model.state_dict()
                        print(f"   - State dict size: {len(state_dict)}")
                        print(f"   - First 5 keys: {list(state_dict.keys())[:5]}")
            else:
                print(f"📤 Sending {len(params)} parameters to server")
                print(f"   First 5 shapes: {[p.shape for p in params[:5]]}")
            
            return params
        except Exception as e:
            print(f"❌ Error getting parameters: {e}")
            traceback.print_exc()
            return []
    
    def set_parameters(self, parameters, config=None):
        """
        Set model parameters from numpy arrays.
        CRITICAL FIX: Verify parameter count matches before loading.
        """
        try:
            expected_count = len(model_to_ndarrays(self.model))
            received_count = len(parameters)
            
            print(f"📥 Received {received_count} parameters from server")
            
            if received_count != expected_count:
                print(f"🚨 WARNING: Parameter count mismatch!")
                print(f"   Expected: {expected_count}, Got: {received_count}")
                print(f"   This indicates server/client model architecture mismatch!")
                print(f"   Attempting to load anyway (may cause errors)...")
            else:
                print(f"   ✓ Parameter count matches expected: {expected_count}")
            
            ndarrays_to_model(self.model, parameters)
            print(f"   ✅ Parameters loaded successfully")
            
        except Exception as e:
            print(f"❌ Error setting parameters: {e}")
            print(f"   This usually means shape mismatch between server and client models")
            traceback.print_exc()
    
    def fit(self, parameters, config=None):
        """Train on local data"""
        print(f"\n{'='*80}")
        print(f"🏋️  NODE {self.node_id}: STARTING TRAINING ROUND")
        print(f"{'='*80}")
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        yaml_path = None
        train_metrics = {'box_loss': 1.0, 'cls_loss': 1.0, 'dfl_loss': 1.0}
        num_samples = 100
        
        try:
            # STEP 1: Verify data
            print(f"\n{'='*60}")
            print(f"STEP 1: VERIFY DATA")
            print(f"{'='*60}")
            data_info = verify_data_exists(self.node_id)
            num_samples = data_info['num_images']
            
            # STEP 2: Fix labels
            print(f"\n{'='*60}")
            print(f"STEP 2: FIX LABELS")
            print(f"{'='*60}")
            fix_labels(data_info['labels_dir'])
            
            # STEP 3: Create YAML config
            print(f"\n{'='*60}")
            print(f"STEP 3: CREATE YAML CONFIG")
            print(f"{'='*60}")
            yaml_path = create_dataset_yaml(self.node_id, data_info['data_root'])
            
            # STEP 4: Train
            print(f"\n{'='*60}")
            print(f"STEP 4: TRAIN YOLO")
            print(f"{'='*60}")
            print(f"📊 Training config:")
            print(f"   Epochs: 1")
            print(f"   Batch: 4")
            print(f"   Image size: 640")
            print(f"   Device: {self.device}")
            print(f"   Samples: {num_samples}")
            print(f"   Classes: {NUM_CLASSES}")
            
            yolo_model = self.model.model
            
            results = yolo_model.train(
                data=yaml_path,
                epochs=1,
                imgsz=640,
                batch=4,
                device=str(self.device),
                workers=0,
                verbose=False,
                patience=0,
                save=False,
                save_period=-1,
                plots=False,
                val=False,
                exist_ok=True,
                project='runs/train',
                name=f'node{self.node_id}',
                cache=False,
                amp=False,
            )
            
            # Extract metrics from trainer (YOLO stores them there)
            print(f"\n📊 Extracting training metrics...")
            
            # The trainer is now available after training
            if hasattr(yolo_model, 'trainer') and yolo_model.trainer is not None:
                trainer = yolo_model.trainer
                
                # Method 1: Get from loss_items (most reliable for last epoch)
                if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                    try:
                        loss_items = trainer.loss_items
                        if hasattr(loss_items, 'cpu'):  # It's a tensor
                            loss_items = loss_items.cpu().numpy()
                        
                        if len(loss_items) >= 3:
                            train_metrics = {
                                'box_loss': float(loss_items[0]),
                                'cls_loss': float(loss_items[1]),
                                'dfl_loss': float(loss_items[2]),
                            }
                            print(f"✅ Extracted from trainer.loss_items:")
                            for k, v in train_metrics.items():
                                print(f"   {k}: {v:.4f}")
                        else:
                            print(f"⚠️  loss_items has only {len(loss_items)} values")
                    except Exception as e:
                        print(f"⚠️  Error extracting from loss_items: {e}")
                
                # Method 2: Try to get from metrics dict
                elif hasattr(trainer, 'metrics') and trainer.metrics:
                    try:
                        metrics_dict = trainer.metrics
                        train_metrics = {
                            'box_loss': float(metrics_dict.get('train/box_loss', 1.0)),
                            'cls_loss': float(metrics_dict.get('train/cls_loss', 1.0)),
                            'dfl_loss': float(metrics_dict.get('train/dfl_loss', 1.0)),
                        }
                        print(f"✅ Extracted from trainer.metrics")
                    except Exception as e:
                        print(f"⚠️  Error extracting from metrics: {e}")
                else:
                    print(f"⚠️  No metrics found in trainer")
            
            # Method 3: Check results object
            elif hasattr(results, 'results_dict') and results.results_dict:
                try:
                    train_metrics = {
                        'box_loss': float(results.results_dict.get('train/box_loss', 1.0)),
                        'cls_loss': float(results.results_dict.get('train/cls_loss', 1.0)),
                        'dfl_loss': float(results.results_dict.get('train/dfl_loss', 1.0)),
                    }
                    print(f"✅ Extracted from results.results_dict")
                except Exception as e:
                    print(f"⚠️  Error extracting from results: {e}")
            
            else:
                print(f"⚠️  No metrics available - using defaults")
                train_metrics = {'box_loss': 1.0, 'cls_loss': 1.0, 'dfl_loss': 1.0}
            
            print(f"\n✅ TRAINING COMPLETE")
            print(f"📊 Metrics:")
            for k, v in train_metrics.items():
                print(f"   {k}: {v:.4f}")
        
        except Exception as e:
            print(f"\n❌ TRAINING FAILED: {e}")
            traceback.print_exc()
            train_metrics = {'box_loss': 1.0, 'error': str(e)[:100]}
        
        finally:
            # Cleanup
            if yaml_path and os.path.exists(yaml_path):
                os.remove(yaml_path)
        
        # CRITICAL: Return FULL parameter list
        params_to_return = self.get_parameters()
        print(f"\n📤 Sending results:")
        print(f"   Samples: {num_samples}")
        print(f"   Parameters: {len(params_to_return)}")
        print(f"   Metrics: {train_metrics}")
        
        return params_to_return, num_samples, train_metrics
    
    def evaluate(self, parameters, config=None):
        """Evaluate on local data"""
        print(f"\n{'='*80}")
        print(f"🔍 NODE {self.node_id}: EVALUATION")
        print(f"{'='*80}")
        
        self.set_parameters(parameters)
        
        yaml_path = None
        eval_metrics = {'val_loss': 1.0}
        num_samples = 100
        avg_loss = 1.0
        
        try:
            # Verify data
            data_info = verify_data_exists(self.node_id)
            num_samples = data_info['num_images']
            
            # Fix labels
            fix_labels(data_info['labels_dir'])
            
            # Create YAML
            yaml_path = create_dataset_yaml(self.node_id, data_info['data_root'])
            
            # Validate
            print(f"\n🔄 Validating...")
            yolo_model = self.model.model
            
            results = yolo_model.val(
                data=yaml_path,
                batch=4,
                imgsz=640,
                device=str(self.device),
                workers=0,
                verbose=False,
            )
            
            if hasattr(results, 'results_dict') and results.results_dict:
                avg_loss = float(results.results_dict.get('metrics/mAP50(B)', 0.0))
                eval_metrics = {'mAP50': avg_loss}
            
            print(f"✅ Validation complete: mAP50 = {avg_loss:.4f}")
        
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            traceback.print_exc()
            avg_loss = 999.0
            eval_metrics = {'val_loss': 999.0, 'error': str(e)[:100]}
        
        finally:
            if yaml_path and os.path.exists(yaml_path):
                os.remove(yaml_path)
        
        print(f"📤 Sending: Loss={avg_loss:.4f}, Samples={num_samples}")
        return float(avg_loss), num_samples, eval_metrics


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"🚀 FLOWER CLIENT - NODE {NODE_ID}")
    print(f"{'='*80}")
    print(f"📋 Configuration:")
    print(f"   Model: YOLOv8{MODEL_SIZE}")
    print(f"   Classes: {NUM_CLASSES}")
    print(f"   Parameter arrays: {len(model_to_ndarrays(model))}")
    print(f"{'='*80}\n")
    
    client = YOLOFLClient(model=model, node_id=NODE_ID)
    
    try:
        print(f"📡 Connecting to 127.0.0.1:8080...\n")
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=client
        )
    except KeyboardInterrupt:
        print(f"\n⚠️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Connection error: {e}")
        traceback.print_exc()