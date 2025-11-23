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
    
    Handles mmap parameter correctly - it's a keyword-only argument for torch.load,
    but must NOT be in kwargs (pickle_load_args) as it causes Unpickler() errors.
    """
    # Force weights_only=False to avoid PyTorch 2.6 unpickling errors
    if weights_only is None:
        weights_only = False
    
    # CRITICAL: Remove mmap from kwargs if present
    # mmap is a keyword-only argument for torch.load, NOT for pickle.Unpickler
    # If it's in kwargs, it gets passed to pickle_load_args and causes TypeError
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mmap'}
    
    # Also check if mmap was passed in kwargs (shouldn't happen, but be safe)
    mmap_value = mmap if mmap is not None else kwargs.get('mmap')
    
    # Call original torch.load
    # mmap must be passed as explicit keyword-only argument, never in kwargs
    if mmap_value is not None:
        return _original_torch_load(
            f, 
            map_location=map_location, 
            pickle_module=pickle_module, 
            weights_only=weights_only,
            mmap=mmap_value,
            **kwargs_clean
        )
    else:
        return _original_torch_load(
            f, 
            map_location=map_location, 
            pickle_module=pickle_module, 
            weights_only=weights_only,
            **kwargs_clean
        )

# Replace torch.load globally
torch.load = patched_torch_load

print("✅ PyTorch 2.6 compatibility patch applied (weights_only=False)")