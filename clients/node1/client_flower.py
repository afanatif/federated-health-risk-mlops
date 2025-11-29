"""
Federated Learning Client for YOLOv8 - Node 1
Uses common client implementation
"""

import os
import sys
import torch
import flwr as fl
from ultralytics import YOLO

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import PyTorch 2.6 fix (must be before ultralytics)
import fix_pytorch26

# Import common client implementation
from clients.common.client_flower import (
    setup_client_logging, 
    YOLOFLClient
)
from clients.common.config import get_pretrained_model_path
from models.model import model_to_ndarrays

# Fix for DFLoss compatibility: Add dummy DFLoss class if missing
# This handles checkpoints saved with older ultralytics versions
try:
    from ultralytics.utils.loss import DFLoss
except (ImportError, AttributeError):
    import ultralytics.utils.loss as loss_module
    import torch.nn as nn
    class DFLoss(nn.Module):
        """Dummy DFLoss class for loading old checkpoints"""
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, *args, **kwargs):
            return None
    loss_module.DFLoss = DFLoss

# ============================================
# NODE CONFIGURATION
# ============================================
NODE_ID = 1

# ============================================
# MODEL CONFIGURATION - MUST MATCH SERVER!
# ============================================
MODEL_SIZE = 'n'  # MUST match server --model-size
NUM_CLASSES = 7   # MUST match server --num-classes

# ============================================
# INITIALIZATION
# ============================================
logger = setup_client_logging(NODE_ID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# Load pre-trained model with 7 classes (same as server)
# This ensures structure matches from the start
try:
    pretrained_model_path = get_pretrained_model_path(project_root=project_root)
    pretrained_model_path = str(pretrained_model_path)
    
    logger.info(f"Loading pre-trained model: {pretrained_model_path}")
    try:
        model = YOLO(pretrained_model_path)
        logger.info(f"✅ Pre-trained model loaded (already has {NUM_CLASSES} classes)")
    except (AttributeError, RuntimeError) as e:
        # Handle version mismatch: load checkpoint manually
        logger.warning(f"⚠️ Direct loading failed (version mismatch): {e}")
        logger.info("Attempting to load checkpoint manually...")
        import torch
        ckpt = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model = YOLO(f'yolov8{MODEL_SIZE}.pt')
            saved_model = ckpt['model']
            if hasattr(saved_model, 'state_dict'):
                model.model.load_state_dict(saved_model.state_dict(), strict=False)
                logger.info("✅ Loaded model weights from checkpoint manually")
            else:
                raise ValueError("Checkpoint format not recognized")
        else:
            raise ValueError("Invalid checkpoint format")
except FileNotFoundError:
    logger.warning(f"⚠️ Pre-trained model not found, using base YOLOv8 (will adapt during training)")
    model = YOLO(f'yolov8{MODEL_SIZE}.pt')

# Count parameters
pytorch_model = model.model.model
trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in pytorch_model.parameters())
param_arrays = len(model_to_ndarrays(model))

logger.info(
    f"Model initialized: trainable_params={trainable_params:,}, "
    f"total_params={total_params:,}, param_arrays={param_arrays}"
)

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    logger.info("="*80)
    logger.info(f"FLOWER CLIENT - Node {NODE_ID}")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Model: YOLOv8{MODEL_SIZE}")
    logger.info(f"  Classes: {NUM_CLASSES}")
    logger.info(f"  Parameter arrays: {param_arrays}")
    logger.info(f"  Device: {device}")
    logger.info("="*80)
    
    client = YOLOFLClient(
        model=model, 
        node_id=NODE_ID,
        num_classes=NUM_CLASSES,
        logger=logger,
        device=device
    )
    
    try:
        # Get server address from environment or use default
        server_address = os.environ.get('SERVER_ADDRESS', '127.0.0.1:8080')
        logger.info(f"Connecting to server at {server_address}")
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
        logger.info("Client disconnected from server")
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.exception(f"Connection error: {e}")

