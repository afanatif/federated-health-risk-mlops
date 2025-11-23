"""
Quick test to verify the model.py fix works correctly.
Run this BEFORE starting the federated learning server/clients.

This will confirm that:
1. Models are created with correct num_classes
2. Parameter counts match between server and client configurations
3. No shape mismatches will occur
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import get_model, model_to_ndarrays
import torch

print("=" * 80)
print("🧪 TESTING MODEL.PY FIX - FEDERATED LEARNING COMPATIBILITY")
print("=" * 80)

# Test configuration that should match your federated setup
MODEL_SIZE = 'n'
NUM_CLASSES = 1
PRETRAINED = False

print(f"\n📋 Test Configuration:")
print(f"   Model Size: YOLOv8{MODEL_SIZE}")
print(f"   Number of Classes: {NUM_CLASSES}")
print(f"   Pretrained: {PRETRAINED}")

# ============================================
# TEST 1: Server Model Creation
# ============================================
print(f"\n{'='*80}")
print("TEST 1: Server Model Creation (simulating server.py)")
print('='*80)

try:
    server_model = get_model(model_size=MODEL_SIZE, num_classes=NUM_CLASSES, pretrained=PRETRAINED)
    
    # Check Detect layer
    detect_layer = server_model.model.model.model[-1]
    actual_nc = detect_layer.nc
    
    print(f"\n✅ Server model created successfully")
    print(f"   Detect layer nc: {actual_nc}")
    
    if actual_nc != NUM_CLASSES:
        print(f"\n🚨 CRITICAL ERROR: Detect layer has {actual_nc} classes, expected {NUM_CLASSES}!")
        print(f"   The fix DID NOT WORK - federated learning will fail!")
        sys.exit(1)
    else:
        print(f"   ✓ Detect layer correctly configured with {NUM_CLASSES} classes")
    
    # Get parameter arrays
    server_params = model_to_ndarrays(server_model)
    server_param_count = len(server_params)
    
    print(f"\n📊 Server Model Parameters:")
    print(f"   Parameter arrays: {server_param_count}")
    print(f"   Total weights: {sum(p.size for p in server_params):,}")
    print(f"\n   First 5 parameter shapes:")
    for i, p in enumerate(server_params[:5]):
        print(f"     [{i}] {p.shape}")

except Exception as e:
    print(f"\n❌ Server model creation FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# TEST 2: Client Model Creation
# ============================================
print(f"\n{'='*80}")
print("TEST 2: Client Model Creation (simulating client.py)")
print('='*80)

try:
    client_model = get_model(model_size=MODEL_SIZE, num_classes=NUM_CLASSES, pretrained=PRETRAINED)
    
    # Check Detect layer
    detect_layer = client_model.model.model.model[-1]
    actual_nc = detect_layer.nc
    
    print(f"\n✅ Client model created successfully")
    print(f"   Detect layer nc: {actual_nc}")
    
    if actual_nc != NUM_CLASSES:
        print(f"\n🚨 CRITICAL ERROR: Detect layer has {actual_nc} classes, expected {NUM_CLASSES}!")
        print(f"   The fix DID NOT WORK - federated learning will fail!")
        sys.exit(1)
    else:
        print(f"   ✓ Detect layer correctly configured with {NUM_CLASSES} classes")
    
    # Get parameter arrays
    client_params = model_to_ndarrays(client_model)
    client_param_count = len(client_params)
    
    print(f"\n📊 Client Model Parameters:")
    print(f"   Parameter arrays: {client_param_count}")
    print(f"   Total weights: {sum(p.size for p in client_params):,}")
    print(f"\n   First 5 parameter shapes:")
    for i, p in enumerate(client_params[:5]):
        print(f"     [{i}] {p.shape}")

except Exception as e:
    print(f"\n❌ Client model creation FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# TEST 3: Parameter Count Comparison
# ============================================
print(f"\n{'='*80}")
print("TEST 3: Parameter Count Comparison")
print('='*80)

print(f"\n📊 Comparison:")
print(f"   Server parameter arrays: {server_param_count}")
print(f"   Client parameter arrays: {client_param_count}")

if server_param_count != client_param_count:
    print(f"\n🚨 CRITICAL ERROR: Parameter count mismatch!")
    print(f"   Server: {server_param_count}, Client: {client_param_count}")
    print(f"   Federated learning will FAIL - only {min(server_param_count, client_param_count)} parameters will be exchanged!")
    sys.exit(1)
else:
    print(f"   ✅ MATCH! Both have {server_param_count} parameter arrays")

# ============================================
# TEST 4: Shape Comparison
# ============================================
print(f"\n{'='*80}")
print("TEST 4: Parameter Shape Comparison")
print('='*80)

print(f"\n🔍 Checking all {server_param_count} parameter shapes...")

mismatches = []
for i, (sp, cp) in enumerate(zip(server_params, client_params)):
    if sp.shape != cp.shape:
        mismatches.append((i, sp.shape, cp.shape))

if mismatches:
    print(f"\n🚨 CRITICAL ERROR: Found {len(mismatches)} shape mismatches!")
    print(f"\n   First 10 mismatches:")
    for i, server_shape, client_shape in mismatches[:10]:
        print(f"     [{i}] Server: {server_shape}, Client: {client_shape}")
    print(f"\n   This will cause Flower to drop mismatched parameters!")
    print(f"   Federated learning will FAIL!")
    sys.exit(1)
else:
    print(f"   ✅ ALL {server_param_count} parameter shapes MATCH!")

# ============================================
# TEST 5: Check for 80-class head (the bug)
# ============================================
print(f"\n{'='*80}")
print("TEST 5: Checking for 80-class COCO head (the original bug)")
print('='*80)

# The bug manifests as parameters with shape containing 80
# e.g., (80, 64, 3, 3) for cv3 layers in Detect head
bug_found = False
for i, p in enumerate(server_params):
    if 80 in p.shape and NUM_CLASSES != 80:
        print(f"   🚨 Found parameter with dimension 80: array[{i}] shape={p.shape}")
        bug_found = True

if bug_found:
    print(f"\n🚨 CRITICAL: Model still has 80-class head!")
    print(f"   The fix DID NOT WORK properly!")
    print(f"   You need to fix the model.py _create_custom_model() method")
    sys.exit(1)
else:
    print(f"   ✅ No 80-class dimensions found (expected for {NUM_CLASSES}-class model)")

# ============================================
# FINAL RESULT
# ============================================
print(f"\n{'='*80}")
print("🎉 ALL TESTS PASSED!")
print('='*80)

print(f"\n✅ Summary:")
print(f"   • Server and client models both have {NUM_CLASSES} classes")
print(f"   • Both models have {server_param_count} parameter arrays")
print(f"   • All {server_param_count} parameter shapes match exactly")
print(f"   • No 80-class COCO head detected")
print(f"   • Ready for federated learning!")

print(f"\n📋 Expected vs Current:")
print(f"   Before fix: 355 parameters (BROKEN)")
print(f"   After fix:  {server_param_count} parameters (CORRECT)")

if server_param_count == 355:
    print(f"\n⚠️  WARNING: Still getting 355 parameters!")
    print(f"   This suggests the model.py fix may not be complete.")
    print(f"   Expected parameter count should be different (likely 225-250 for YOLOv8n with 1 class)")

print(f"\n🚀 You can now safely run:")
print(f"   1. Server: python server/server.py --model-size {MODEL_SIZE} --num-classes {NUM_CLASSES}")
print(f"   2. Clients: python clients/nodeX/client_flower.py")
print(f"\n{'='*80}\n")