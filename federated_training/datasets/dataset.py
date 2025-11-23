"""
Generic YOLOv8 dataset resolver for federated training.
Supports:
- Any YOLO model variant (n, s, m, l, x)
- Custom checkpoints (models/my_model.pt)
- Node-specific data.yaml resolution from config/settings.yaml
- YOLO built-in training & validation (no dataloader hacks)
"""

from pathlib import Path
from typing import Optional, Dict
import yaml
import torch

from pathlib import Path
from typing import Optional, Dict
import yaml
import torch
import urllib.request
from ultralytics import YOLO

from pathlib import Path
from typing import Optional, Dict
import yaml
import torch
import os

class YOLODataset:
    def __init__(
        self,
        node_id: Optional[int] = None,
        data_yaml: Optional[str] = None,
        batch_size: int = 16,
        img_size: int = 640,
        device: str = "auto",
    ):
        """
        Load YOLO dataset and model settings for a federated node.
        Handles:
        - Resolving model path relative to monorepo structure
        - Ensuring fallback weights are downloaded to models/incoming/
        """

        # -------------------------------------------------------
        # Load global project settings
        # -------------------------------------------------------
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        SETTINGS_PATH = PROJECT_ROOT / "federated_training/configs/default.yaml"

        if not SETTINGS_PATH.exists():
            raise FileNotFoundError(f"Missing settings.yaml at {SETTINGS_PATH}")

        with open(SETTINGS_PATH, "r") as f:
            cfg = yaml.safe_load(f)

        # Dataset root
        federated_root = PROJECT_ROOT / cfg["paths"]["data_root"]
        models_cfg = cfg.get("models", {})

        # -------------------------------------------------------
        # Resolve model path dynamically (generic!)
        # -------------------------------------------------------
        self.model_path = models_cfg.get("default_model", "yolov8n.pt")
        models_root = PROJECT_ROOT / models_cfg.get("root", "models")
        incoming_root = models_root / "incoming"

        # Ensure incoming folder exists
        incoming_root.mkdir(parents=True, exist_ok=True)

        # Resolve absolute model path
        candidate = Path(self.model_path)
        if not candidate.is_absolute():
            candidate = models_root / self.model_path
        self.model_path = str(candidate.resolve())

        # Set env variable to redirect Ultralytics fallback downloads
        os.environ["YOLO_WEIGHTS_DIR"] = str(incoming_root.resolve())

        print(f"[YOLODataset] Models root: {models_root}")
        print(f"[YOLODataset] Incoming folder (fallback downloads): {incoming_root}")
        print(f"[YOLODataset] Using model path: {self.model_path}")

        # -------------------------------------------------------
        # Resolve dataset YAML for this federated node
        # -------------------------------------------------------
        if data_yaml:
            self.data_yaml = Path(data_yaml).resolve()
        elif node_id is not None:
            candidate = federated_root / f"splits/iid_5nodes/node_{node_id}/data.yaml"
            if not candidate.exists():
                raise FileNotFoundError(f"Node dataset not found: {candidate}")
            self.data_yaml = candidate.resolve()
        else:
            raise ValueError("Either node_id OR data_yaml must be provided.")

        # -------------------------------------------------------
        # Device selection
        # -------------------------------------------------------
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.batch_size = batch_size
        self.img_size = img_size



    # -------------------------------------------------------
    # Train Args for YOLO built-in training
    # -------------------------------------------------------
    def get_train_args(self) -> Dict:
        return {
            "data": str(self.data_yaml),
            "batch": self.batch_size,
            "imgsz": self.img_size,
        }

    # -------------------------------------------------------
    # Val Args for YOLO built-in validation
    # -------------------------------------------------------
    def get_val_args(self) -> Dict:
        return {
            "data": str(self.data_yaml),
            "imgsz": self.img_size,
        }

    def __repr__(self):
        return (
            f"YOLODataset(node_yaml={self.data_yaml}, "
            f"model_path={self.model_path}, "
            f"batch={self.batch_size}, imgsz={self.img_size}, "
            f"device={self.device})"
        )
