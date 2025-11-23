"""
Common configuration for federated learning clients and server.
Centralizes pretrained model path and other shared settings.
"""

import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root directory.
    """
    # Get the directory containing this file
    current_file = Path(__file__).resolve()
    # Go up: common -> clients -> project_root
    project_root = current_file.parent.parent.parent
    return project_root


def get_pretrained_model_path(
    model_path: Optional[str] = None,
    project_root: Optional[Path] = None
) -> Path:
    """
    Get the path to the pretrained model.
    
    Args:
        model_path: Optional custom path to pretrained model.
                   If None, uses default path.
        project_root: Optional project root path.
                     If None, auto-detects from file location.
    
    Returns:
        Path to pretrained model file.
    
    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    if project_root is None:
        project_root = get_project_root()
    else:
        project_root = Path(project_root)
    
    if model_path is None:
        # Default path: runs/train/node1_yolov8/weights/best.pt
        model_path = project_root / 'runs' / 'train' / 'node1_yolov8' / 'weights' / 'best.pt'
    else:
        # Custom path provided
        model_path = Path(model_path)
        if not model_path.is_absolute():
            # Relative to project root
            model_path = project_root / model_path
    
    # Try best.pt first, then last.pt as fallback
    if not model_path.exists():
        # Try last.pt in same directory
        fallback_path = model_path.parent / 'last.pt'
        if fallback_path.exists():
            return fallback_path
        else:
            raise FileNotFoundError(
                f"Pretrained model not found at {model_path} "
                f"or {fallback_path}"
            )
    
    return model_path


# Default pretrained model path (relative to project root)
DEFAULT_PRETRAINED_MODEL = 'runs/train/node1_yolov8/weights/best.pt'

