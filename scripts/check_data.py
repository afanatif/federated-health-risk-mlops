"""
Check if your local federated data is ready.
Data location: data/federated/splits/iid_5nodes/node_X/
"""
import os
from pathlib import Path


def check_node(node_name):
    """Check one node's data."""
    print(f"\n{'='*60}")
    print(f"Checking {node_name.upper()}")
    print('='*60)
    
    # NEW PATH STRUCTURE
    base_path = f"data/federated/splits/iid_5nodes/{node_name}"
    images_dir = Path(base_path) / "images"
    labels_dir = Path(base_path) / "labels"
    
    # Check directories
    if not images_dir.exists():
        print(f"❌ Images folder missing: {images_dir}")
        return False
    print(f"✓ Images folder exists")
    
    if not labels_dir.exists():
        print(f"❌ Labels folder missing: {labels_dir}")
        return False
    print(f"✓ Labels folder exists")
    
    # Count files
    images = list(images_dir.glob("*.jpg"))
    labels = list(labels_dir.glob("*.txt"))
    
    print(f"✓ Images found: {len(images)}")
    print(f"✓ Labels found: {len(labels)}")
    
    # Check counts match
    if len(images) != len(labels):
        print(f"⚠️  Warning: {len(images)} images but {len(labels)} labels")
    
    # Check if we have data
    if len(images) == 0:
        print(f"❌ No images found!")
        return False
    
    print(f"✅ {node_name} is ready!")
    return True


def main():
    """Check all three nodes."""
    print("\n" + "="*60)
    print("FEDERATED LEARNING DATA CHECK")
    print("="*60)
    print(f"Looking for data in: data/federated/splits/iid_5nodes/")
    
    all_ready = True
    for node in ["node_1", "node_2", "node_3"]:  # Note: node_1, not node1
        ready = check_node(node)
        if not ready:
            all_ready = False
    
    print("\n" + "="*60)
    if all_ready:
        print("✅ ALL NODES READY FOR TRAINING!")
        print("\nYou can now:")
        print("  1. Test individual node: python clients/node1/data_loader.py")
        print("  2. Start training: python server/server_flower.py")
    else:
        print("❌ SOME NODES NOT READY!")
        print("\nMake sure your data is at:")
        print("  data/federated/splits/iid_5nodes/node_1/images/")
        print("  data/federated/splits/iid_5nodes/node_1/labels/")
        print("  ... (same for node_2 and node_3)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
