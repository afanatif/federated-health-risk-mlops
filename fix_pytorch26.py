"""
CRITICAL FIX for PyTorch 2.6 weights_only issue.
This MUST be imported BEFORE ultralytics in ALL client files.

PyTorch 2.6 changed torch.load() default from weights_only=False to True.
This breaks ultralytics checkpoint loading.
"""

import torch
import sys

# Store original torch.load
_original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=None, mmap=None, **kwargs):
    """
    Patched torch.load that always uses weights_only=False.
    This is safe because we trust the YOLOv8 checkpoints.
    """
    # Force weights_only=False to avoid PyTorch 2.6 unpickling errors
    if weights_only is None:
        weights_only = False
    
    return _original_torch_load(
        f, 
        map_location=map_location, 
        pickle_module=pickle_module, 
        weights_only=weights_only,
        mmap=mmap,
        **kwargs
    )

# Replace torch.load globally
torch.load = patched_torch_load

print("✅ PyTorch 2.6 compatibility patch applied (weights_only=False)")