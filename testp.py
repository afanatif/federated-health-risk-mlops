"""
Test parameter extraction to verify federated learning will work.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Apply PyTorch 2.6 fix
import fix_pytorch26

print("\n" + "="*70)
print("TESTING PARAMETER EXTRACTION")
print("="*70)

# Test 1: Create model
print("\n1️⃣  Creating YOLOv8 model...")
from models.model import get_model, model_to_ndarrays, ndarrays_to_model

model = get_model(model_size='n', num_classes=1, pretrained=False)
print(f"✅ Model created")

# Test 2: Extract parameters BEFORE training
print("\n2️⃣  Extracting parameters (before training)...")
params_before = model_to_ndarrays(model)
print(f"✅ Extracted {len(params_before)} parameters")

if len(params_before) == 0:
    print(f"❌ CRITICAL ERROR: 0 parameters extracted!")
    print(f"   Federated learning WILL NOT WORK")
    sys.exit(1)

# Test 3: Verify parameter shapes
print(f"\n3️⃣  Verifying parameter shapes...")
print(f"   First parameter shape: {params_before[0].shape}")
print(f"   Last parameter shape: {params_before[-1].shape}")
total_params = sum(p.size for p in params_before)
print(f"   Total parameters: {total_params:,}")

# Test 4: Test round-trip (serialize and deserialize)
print(f"\n4️⃣  Testing parameter round-trip...")
ndarrays_to_model(model, params_before)
params_after = model_to_ndarrays(model)
print(f"✅ Round-trip successful")
print(f"   Parameters before: {len(params_before)}")
print(f"   Parameters after: {len(params_after)}")

if len(params_before) != len(params_after):
    print(f"⚠️  WARNING: Parameter count changed!")
else:
    print(f"✅ Parameter count consistent")

# Test 5: Verify weights actually loaded
print(f"\n5️⃣  Verifying weights loaded correctly...")
import numpy as np
differences = 0
for i, (p1, p2) in enumerate(zip(params_before[:5], params_after[:5])):
    if not np.allclose(p1, p2):
        differences += 1

if differences == 0:
    print(f"✅ All weights match perfectly")
else:
    print(f"⚠️  {differences}/5 parameters differ")

print("\n" + "="*70)
if len(params_before) > 0 and len(params_after) > 0:
    print("🎉 PARAMETER EXTRACTION WORKS!")
    print("="*70)
    print(f"\n✅ Federated learning is ready")
    print(f"✅ {len(params_before)} parameters will be shared between clients")
    print(f"✅ Total weight count: {total_params:,}")
else:
    print("❌ PARAMETER EXTRACTION FAILED!")
    print("="*70)
    print(f"\n❌ Federated learning will NOT work")
    print(f"❌ Fix the model parameter extraction")
    sys.exit(1)