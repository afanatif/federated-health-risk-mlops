"""
Common Federated Learning Client for YOLOv8
Shared implementation for all nodes
"""

import os
import sys
import torch
import flwr as fl
from ultralytics import YOLO
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

# Import PyTorch 2.6 fix
import fix_pytorch26  # This patches torch.load globally

# Import model utilities for parameter serialization
from models.model import model_to_ndarrays, ndarrays_to_model

# Optimization
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"


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
    logging.getLogger('ultralytics').setLevel(logging.INFO)
    
    return logger


def verify_data_exists(node_id, logger, num_classes):
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


def fix_labels(labels_dir, logger, num_classes):
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
                    # Validate class ID is within range [0, num_classes-1]
                    try:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < num_classes:
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


def create_dataset_yaml(node_id, data_root, logger):
    """
    Create YOLO dataset.yaml by reading from data.yaml (exactly like train.ipynb)
    
    Notebook approach:
    1. Read data.yaml from node directory
    2. Create dataset.yaml in the SAME node directory (not project root)
    3. Use absolute path for cross-platform compatibility
    """
    logger.info("Creating dataset YAML configuration")
    
    # Read actual class configuration from data.yaml (exactly like notebook)
    data_yaml_path = os.path.join(data_root, 'data.yaml')
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Create dataset config (matching notebook approach exactly)
    dataset_config = {
        'path': os.path.abspath(data_root),  # Absolute path (like notebook)
        'train': 'images',
        'val': 'images',
        'nc': data_config['nc'],
        'names': data_config['names']
    }
    
    # Save dataset.yaml in the SAME node directory (like notebook does)
    # Notebook: dataset_yaml = node_path / "dataset.yaml"
    yaml_path = os.path.join(data_root, "dataset.yaml")
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    logger.info(
        f"Dataset YAML created: {yaml_path}, "
        f"classes={dataset_config['nc']}, names={dataset_config['names']}"
    )
    return yaml_path


class YOLOFLClient(fl.client.NumPyClient):
    """Flower client for YOLOv8 federated learning - Common implementation"""
    
    def __init__(self, model, node_id, num_classes, logger, device):
        self.model = model
        self.node_id = node_id
        self.num_classes = num_classes
        self.logger = logger
        self.device = device
    
    def get_parameters(self, config=None):
        """Get model parameters as numpy arrays"""
        try:
            params = model_to_ndarrays(self.model)
            
            if len(params) == 0:
                self.logger.error("CRITICAL: 0 parameters extracted - federated learning will not work")
            else:
                self.logger.info(f"Extracting parameters: {len(params)} arrays")
                self.logger.debug(f"First 5 parameter shapes: {[p.shape for p in params[:5]]}")
            
            return params
        except Exception as e:
            self.logger.exception(f"Error getting parameters: {e}")
            return []
    
    def set_parameters(self, parameters, config=None):
        """Set model parameters from numpy arrays"""
        try:
            received_count = len(parameters)
            self.logger.info(f"Receiving parameters from server: {received_count} arrays")
            
            # Load parameters (model structure should already match - both use pre-trained model)
            ndarrays_to_model(self.model, parameters)
            
            # Ensure model is in train mode after loading parameters
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'train'):
                self.model.model.train()
            
            self.logger.info("Parameters loaded successfully")
            
        except Exception as e:
            self.logger.exception(f"Error setting parameters: {e}")
    
    def fit(self, parameters, config=None):
        """Train on local data"""
        self.logger.info("="*80)
        self.logger.info(f"TRAINING ROUND - Node {self.node_id}")
        self.logger.info("="*80)
        
        # Set parameters from server (if provided)
        if parameters is not None and len(parameters) > 0:
            expected_params = len(model_to_ndarrays(self.model))
            received_params = len(parameters)
            
            if expected_params != received_params:
                self.logger.warning(
                    f"Parameter count mismatch: expected={expected_params}, "
                    f"received={received_params}. Model structure may not match."
                )
                self.logger.warning("Attempting to load parameters anyway (will skip mismatches)...")
            
            self.set_parameters(parameters)
            self.logger.info("Loaded parameters from server")
        else:
            self.logger.info("No initial parameters - using client-initialized model")
            # YOLO will automatically adapt model structure during first training
        
        yaml_path = None
        train_metrics = {'box_loss': 1.0, 'cls_loss': 1.0, 'dfl_loss': 1.0}
        num_samples = 100
        
        try:
            # STEP 1: Verify data
            self.logger.info("Step 1: Verifying data")
            data_info = verify_data_exists(self.node_id, self.logger, self.num_classes)
            num_samples = data_info['num_images']
            
            # STEP 2: Validate labels
            self.logger.info("Step 2: Validating labels")
            fix_labels(data_info['labels_dir'], self.logger, self.num_classes)
            
            # STEP 3: Create YAML config
            self.logger.info("Step 3: Creating dataset YAML")
            yaml_path = create_dataset_yaml(self.node_id, data_info['data_root'], self.logger)
            
            # STEP 4: Train
            self.logger.info("Step 4: Starting local training")
            self.logger.info(
                f"Training configuration: epochs=10, batch=4, imgsz=640, "
                f"device={self.device}, samples={num_samples}, classes={self.num_classes}"
            )
            
            yolo_model = self.model
            
            self.logger.info("Starting YOLO training ...")
            # YOLO will automatically adapt model.nc from dataset.yaml (like notebook)
            # Output: "Overriding model.yaml nc=80 with nc=7"
            results = yolo_model.train(
                data=yaml_path,
                epochs=1,
                imgsz=640,
                batch=4,
                device=str(self.device),
                workers=0,
                verbose=True,
                patience=0,
                save=False,  # Don't save checkpoints (federated learning)
                save_period=-1,
                plots=False,
                val=False,  # No validation during federated rounds
                exist_ok=True,
                project='runs/train',
                name=f'node{self.node_id}',
                cache=False,
                amp=False,
            )
            self.logger.info("YOLO training completed")
            
            # Extract metrics from trainer
            self.logger.info("Extracting training metrics")
            
            if hasattr(yolo_model, 'trainer') and yolo_model.trainer is not None:
                trainer = yolo_model.trainer
                
                # Method 1: Get from metrics dict (most reliable - averaged losses)
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    try:
                        metrics_dict = trainer.metrics
                        
                        # Debug: Log available keys
                        self.logger.debug(f"Available metrics keys: {list(metrics_dict.keys())[:20]}")
                        
                        # Try multiple possible key formats
                        box_loss = (
                            metrics_dict.get('train/box_loss') or
                            metrics_dict.get('box_loss') or
                            metrics_dict.get('train/box') or
                            metrics_dict.get('box')
                        )
                        cls_loss = (
                            metrics_dict.get('train/cls_loss') or
                            metrics_dict.get('cls_loss') or
                            metrics_dict.get('train/cls') or
                            metrics_dict.get('cls')
                        )
                        dfl_loss = (
                            metrics_dict.get('train/dfl_loss') or
                            metrics_dict.get('dfl_loss') or
                            metrics_dict.get('train/dfl') or
                            metrics_dict.get('dfl')
                        )
                        
                        train_metrics = {
                            'box_loss': float(box_loss) if box_loss is not None else 1.0,
                            'cls_loss': float(cls_loss) if cls_loss is not None else 1.0,
                            'dfl_loss': float(dfl_loss) if dfl_loss is not None else 1.0,
                        }
                        self.logger.info("Training metrics extracted from trainer.metrics")
                        self.logger.info(f"Metrics: {train_metrics}")
                    except Exception as e:
                        self.logger.warning(f"Error extracting from metrics: {e}")
                        train_metrics = {'box_loss': 1.0, 'cls_loss': 1.0, 'dfl_loss': 1.0}
                
                # Method 2: Get from loss_items (last batch - less reliable)
                elif hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                    try:
                        loss_items = trainer.loss_items
                        if hasattr(loss_items, 'cpu'):
                            loss_items = loss_items.cpu().numpy()
                        
                        if len(loss_items) >= 3:
                            train_metrics = {
                                'box_loss': float(loss_items[0]),
                                'cls_loss': float(loss_items[1]),
                                'dfl_loss': float(loss_items[2]),
                            }
                            self.logger.warning("Using loss_items (last batch) - may not be averaged!")
                            self.logger.info("Training metrics extracted from trainer.loss_items")
                            self.logger.info(f"Metrics: {train_metrics}")
                        else:
                            train_metrics = {'box_loss': 1.0, 'cls_loss': 1.0, 'dfl_loss': 1.0}
                    except Exception as e:
                        self.logger.warning(f"Error extracting from loss_items: {e}")
                        train_metrics = {'box_loss': 1.0, 'cls_loss': 1.0, 'dfl_loss': 1.0}
                else:
                    self.logger.warning("No metrics found in trainer")
                    train_metrics = {'box_loss': 1.0, 'cls_loss': 1.0, 'dfl_loss': 1.0}
            
            # Method 3: Check results object
            elif hasattr(results, 'results_dict') and results.results_dict:
                try:
                    train_metrics = {
                        'box_loss': float(results.results_dict.get('train/box_loss', 1.0)),
                        'cls_loss': float(results.results_dict.get('train/cls_loss', 1.0)),
                        'dfl_loss': float(results.results_dict.get('train/dfl_loss', 1.0)),
                    }
                    self.logger.info("Training metrics extracted from results.results_dict")
                except Exception as e:
                    self.logger.warning(f"Error extracting from results: {e}")
                    train_metrics = {'box_loss': 1.0, 'cls_loss': 1.0, 'dfl_loss': 1.0}
            
            else:
                self.logger.warning("No metrics available - using defaults")
                train_metrics = {'box_loss': 1.0, 'cls_loss': 1.0, 'dfl_loss': 1.0}
            
            self.logger.info("Training round completed successfully")
        
        except Exception as e:
            self.logger.exception(f"Training failed: {e}")
            train_metrics = {'box_loss': 1.0, 'error': str(e)[:100]}
        
        finally:
            # Cleanup: Remove dataset.yaml (like notebook, it's temporary)
            # Note: Notebook doesn't explicitly delete, but we do for cleanliness
            if yaml_path and os.path.exists(yaml_path):
                # Only delete if it's in the node directory (not if user wants to keep it)
                # For now, we'll keep it (like notebook) - comment out deletion
                # os.remove(yaml_path)
                self.logger.debug(f"Dataset YAML saved at: {yaml_path} (keeping for reference)")
        
        # Return FULL parameter list
        params_to_return = self.get_parameters()
        self.logger.info(
            f"Sending results to server: samples={num_samples}, "
            f"params={len(params_to_return)}, metrics={train_metrics}"
        )
        
        return params_to_return, num_samples, train_metrics

