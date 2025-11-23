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
import logging
from datetime import datetime

# Optimization
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# Add project root to path (again for safety)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from models.model import get_model, model_to_ndarrays, ndarrays_to_model

# ============================================
# LOGGING CONFIGURATION
# ============================================
def setup_client_logging(node_id, log_level=logging.INFO):
    """Setup professional logging for client"""
    log_format = '%(asctime)s | %(levelname)-8s | [Node %(node_id)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create logs directory
    log_dir = os.path.join(project_root, 'clients', f'node{node_id}', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'client_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Custom formatter with node_id
    class NodeFormatter(logging.Formatter):
        def format(self, record):
            record.node_id = node_id
            return super().format(record)
    
    formatter = NodeFormatter(log_format, date_format)
    
    # Setup handlers
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
    
    for handler in handlers:
        handler.setFormatter(formatter)
    
    logger = logging.getLogger(f'client_node{node_id}')
    logger.setLevel(log_level)
    logger.handlers = handlers
    
    # Suppress verbose third-party logs (but allow YOLO progress bars)
    logging.getLogger('flwr').setLevel(logging.WARNING)
    # Keep ultralytics at INFO to allow training progress bars
    logging.getLogger('ultralytics').setLevel(logging.INFO)
    
    return logger

# ============================================
# NODE CONFIGURATION - CHANGE THIS FOR EACH NODE
# ============================================
NODE_ID = 2  # ⚠️  CHANGE TO 2 FOR NODE2, 3 FOR NODE3
# ============================================

# ============================================
# MODEL CONFIGURATION - MUST MATCH SERVER!
# ============================================
MODEL_SIZE = 'n'  # MUST match server --model-size
NUM_CLASSES = 7   # MUST match server --num-classes (7 classes: banner, erosion, hcrack, pothole, stone, trash, vcrack)
# ============================================

# Setup logging
logger = setup_client_logging(NODE_ID)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# Create model with EXACT same config as server
logger.info(f"Initializing YOLOv8 model: size={MODEL_SIZE}, classes={NUM_CLASSES}")
model = get_model(model_size=MODEL_SIZE, num_classes=NUM_CLASSES, pretrained=True)
model.to(DEVICE)

# Count parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
param_arrays = len(model_to_ndarrays(model))

logger.info(
    f"Model initialized: trainable_params={trainable_params:,}, "
    f"total_params={total_params:,}, param_arrays={param_arrays}"
)

# Log first 5 parameter shapes for debugging
arrays = model_to_ndarrays(model)
logger.debug("First 5 parameter array shapes:")
for i in range(min(5, len(arrays))):
    logger.debug(f"  [{i}] {arrays[i].shape}")


def verify_data_exists(node_id):
    """Verify data exists and return detailed info"""
    logger.info("Verifying data directory structure")
    
    # Use cross-platform path handling
    data_root = os.path.abspath(os.path.join('data', 'federated', 'splits', 'iid_5nodes', f'node_{node_id}'))
    images_dir = os.path.join(data_root, 'images')
    labels_dir = os.path.join(data_root, 'labels')
    
    # Check directories
    for path, name in [(data_root, "Root"), (images_dir, "Images"), (labels_dir, "Labels")]:
        exists = os.path.exists(path)
        logger.debug(f"{name} directory: {path} - {'exists' if exists else 'missing'}")
        if not exists and name != "Root":
            raise FileNotFoundError(f"{name} directory not found: {path}")
    
    # Count files
    jpg_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.JPG"))
    txt_files = list(Path(labels_dir).glob("*.txt"))
    
    logger.info(f"Data verification complete: images={len(jpg_files)}, labels={len(txt_files)}")
    
    if len(jpg_files) == 0:
        raise ValueError(f"No images found in {images_dir}")
    if len(txt_files) == 0:
        raise ValueError(f"No labels found in {labels_dir}")
    
    # Log sample files for debugging
    logger.debug("Sample files:")
    for img in jpg_files[:3]:
        logger.debug(f"  Image: {img.name}")
    for lbl in txt_files[:3]:
        logger.debug(f"  Label: {lbl.name}")
    
    return {
        'data_root': data_root,
        'images_dir': images_dir,
        'labels_dir': labels_dir,
        'num_images': len(jpg_files),
        'num_labels': len(txt_files)
    }


def fix_labels(labels_dir):
    """Validate labels format - check for valid YOLO format"""
    logger.info("Validating label file format")
    valid_count = 0
    total_boxes = 0
    invalid_files = []
    
    for label_file in Path(labels_dir).glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Validate class ID is within range [0, NUM_CLASSES-1]
                    try:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < NUM_CLASSES:
                            total_boxes += 1
                        else:
                            invalid_files.append(f"{label_file.name}: invalid class ID {cls_id}")
                    except ValueError:
                        invalid_files.append(f"{label_file.name}: invalid class ID format")
            
            valid_count += 1
        except Exception as e:
            logger.warning(f"Error reading {label_file.name}: {e}")
            invalid_files.append(f"{label_file.name}: {str(e)}")
    
    if invalid_files:
        logger.warning(f"Found {len(invalid_files)} label file(s) with issues")
        for issue in invalid_files[:5]:
            logger.warning(f"  {issue}")
    else:
        logger.info(f"Label validation complete: {valid_count} files, {total_boxes} bounding boxes")


def create_dataset_yaml(node_id, data_root):
    """Create YOLO dataset.yaml by reading from data.yaml (like notebook)"""
    logger.info("Creating dataset YAML configuration")
    
    # Read actual class configuration from data.yaml
    data_yaml_path = os.path.join(data_root, 'data.yaml')
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Create dataset config (matching notebook approach)
    dataset_config = {
        'path': os.path.abspath(data_root),  # Use absolute path for cross-platform compatibility
        'train': 'images',
        'val': 'images',
        'nc': data_config['nc'],
        'names': data_config['names']
    }
    
    # Create temporary dataset.yaml in project root (cross-platform)
    yaml_path = os.path.join(project_root, f"dataset_node{node_id}.yaml")
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    logger.info(
        f"Dataset YAML created: {yaml_path}, "
        f"classes={dataset_config['nc']}, names={dataset_config['names']}"
    )
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
                logger.error("CRITICAL: 0 parameters extracted - federated learning will not work")
                logger.debug(f"Model state: type={type(self.model)}")
                if hasattr(self.model, 'model'):
                    logger.debug(f"Model.model type: {type(self.model.model)}")
                    if hasattr(self.model.model, 'model'):
                        state_dict = self.model.model.model.state_dict()
                        logger.debug(f"State dict size: {len(state_dict)}, keys: {list(state_dict.keys())[:5]}")
            else:
                logger.info(f"Extracting parameters: {len(params)} arrays")
                logger.debug(f"First 5 parameter shapes: {[p.shape for p in params[:5]]}")
            
            return params
        except Exception as e:
            logger.exception(f"Error getting parameters: {e}")
            return []
    
    def set_parameters(self, parameters, config=None):
        """
        Set model parameters from numpy arrays.
        CRITICAL FIX: Verify parameter count matches before loading.
        """
        try:
            expected_count = len(model_to_ndarrays(self.model))
            received_count = len(parameters)
            
            logger.info(f"Receiving parameters from server: {received_count} arrays")
            
            if received_count != expected_count:
                logger.error(
                    f"Parameter count mismatch: expected={expected_count}, "
                    f"received={received_count}. Attempting to load anyway."
                )
            else:
                logger.debug(f"Parameter count verified: {expected_count}")
            
            ndarrays_to_model(self.model, parameters)
            logger.info("Parameters loaded successfully")
            
        except Exception as e:
            logger.exception(f"Error setting parameters: {e}")
    
    def fit(self, parameters, config=None):
        """Train on local data"""
        logger.info("="*80)
        logger.info(f"TRAINING ROUND - Node {self.node_id}")
        logger.info("="*80)
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        yaml_path = None
        train_metrics = {'box_loss': 1.0, 'cls_loss': 1.0, 'dfl_loss': 1.0}
        num_samples = 100
        
        try:
            # STEP 1: Verify data
            logger.info("Step 1: Verifying data")
            data_info = verify_data_exists(self.node_id)
            num_samples = data_info['num_images']
            
            # STEP 2: Validate labels
            logger.info("Step 2: Validating labels")
            fix_labels(data_info['labels_dir'])
            
            # STEP 3: Create YAML config
            logger.info("Step 3: Creating dataset YAML")
            yaml_path = create_dataset_yaml(self.node_id, data_info['data_root'])
            
            # STEP 4: Train
            logger.info("Step 4: Starting local training")
            logger.info(
                f"Training configuration: epochs=1, batch=4, imgsz=640, "
                f"device={self.device}, samples={num_samples}, classes={NUM_CLASSES}"
            )
            
            yolo_model = self.model.model
            
            logger.info("Starting YOLO training (1 epoch)...")
            results = yolo_model.train(
                data=yaml_path,
                epochs=1,
                imgsz=640,
                batch=4,
                device=str(self.device),
                workers=0,
                verbose=True,  # Show training progress bars
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
            logger.info("YOLO training completed")
            
            # Extract metrics from trainer (YOLO stores them there)
            logger.info("Extracting training metrics")
            
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
                            logger.info("Training metrics extracted from trainer.loss_items")
                            logger.info(f"Metrics: {train_metrics}")
                        else:
                            logger.warning(f"loss_items has only {len(loss_items)} values")
                    except Exception as e:
                        logger.warning(f"Error extracting from loss_items: {e}")
                
                # Method 2: Try to get from metrics dict
                elif hasattr(trainer, 'metrics') and trainer.metrics:
                    try:
                        metrics_dict = trainer.metrics
                        train_metrics = {
                            'box_loss': float(metrics_dict.get('train/box_loss', 1.0)),
                            'cls_loss': float(metrics_dict.get('train/cls_loss', 1.0)),
                            'dfl_loss': float(metrics_dict.get('train/dfl_loss', 1.0)),
                        }
                        logger.info("Training metrics extracted from trainer.metrics")
                    except Exception as e:
                        logger.warning(f"Error extracting from metrics: {e}")
                else:
                    logger.warning("No metrics found in trainer")
            
            # Method 3: Check results object
            elif hasattr(results, 'results_dict') and results.results_dict:
                try:
                    train_metrics = {
                        'box_loss': float(results.results_dict.get('train/box_loss', 1.0)),
                        'cls_loss': float(results.results_dict.get('train/cls_loss', 1.0)),
                        'dfl_loss': float(results.results_dict.get('train/dfl_loss', 1.0)),
                    }
                    logger.info("Training metrics extracted from results.results_dict")
                except Exception as e:
                    logger.warning(f"Error extracting from results: {e}")
            
            else:
                logger.warning("No metrics available - using defaults")
                train_metrics = {'box_loss': 1.0, 'cls_loss': 1.0, 'dfl_loss': 1.0}
            
            logger.info("Training round completed successfully")
        
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            train_metrics = {'box_loss': 1.0, 'error': str(e)[:100]}
        
        finally:
            # Cleanup
            if yaml_path and os.path.exists(yaml_path):
                os.remove(yaml_path)
                logger.debug(f"Cleaned up temporary YAML: {yaml_path}")
        
        # CRITICAL: Return FULL parameter list
        params_to_return = self.get_parameters()
        logger.info(
            f"Sending results to server: samples={num_samples}, "
            f"params={len(params_to_return)}, metrics={train_metrics}"
        )
        
        return params_to_return, num_samples, train_metrics
    
    def evaluate(self, parameters, config=None):
        """Evaluate on local data"""
        logger.info("="*80)
        logger.info(f"EVALUATION - Node {self.node_id}")
        logger.info("="*80)
        
        self.set_parameters(parameters)
        
        yaml_path = None
        eval_metrics = {'val_loss': 1.0}
        num_samples = 100
        avg_loss = 1.0
        
        try:
            # Verify data
            data_info = verify_data_exists(self.node_id)
            num_samples = data_info['num_images']
            
            # Validate labels
            fix_labels(data_info['labels_dir'])
            
            # Create YAML
            yaml_path = create_dataset_yaml(self.node_id, data_info['data_root'])
            
            # Validate
            logger.info("Running validation")
            yolo_model = self.model.model
            
            logger.info("Starting YOLO validation...")
            results = yolo_model.val(
                data=yaml_path,
                batch=4,
                imgsz=640,
                device=str(self.device),
                workers=0,
                verbose=True,  # Show validation progress
            )
            logger.info("YOLO validation completed")
            
            if hasattr(results, 'results_dict') and results.results_dict:
                avg_loss = float(results.results_dict.get('metrics/mAP50(B)', 0.0))
                eval_metrics = {'mAP50': avg_loss}
            
            logger.info(f"Validation complete: mAP50={avg_loss:.4f}")
        
        except Exception as e:
            logger.exception(f"Evaluation error: {e}")
            avg_loss = 999.0
            eval_metrics = {'val_loss': 999.0, 'error': str(e)[:100]}
        
        finally:
            if yaml_path and os.path.exists(yaml_path):
                os.remove(yaml_path)
                logger.debug(f"Cleaned up temporary YAML: {yaml_path}")
        
        logger.info(f"Sending evaluation results: loss={avg_loss:.4f}, samples={num_samples}")
        return float(avg_loss), num_samples, eval_metrics


if __name__ == "__main__":
    logger.info("="*80)
    logger.info(f"FLOWER CLIENT - Node {NODE_ID}")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Model: YOLOv8{MODEL_SIZE}")
    logger.info(f"  Classes: {NUM_CLASSES}")
    logger.info(f"  Parameter arrays: {len(model_to_ndarrays(model))}")
    logger.info(f"  Device: {DEVICE}")
    logger.info("="*80)
    
    client = YOLOFLClient(model=model, node_id=NODE_ID)
    
    try:
        logger.info("Connecting to server at 127.0.0.1:8080")
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=client
        )
        logger.info("Client disconnected from server")
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.exception(f"Connection error: {e}")