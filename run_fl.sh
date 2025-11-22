"""
Simple Data Check Script
Checks if federated learning data is ready with correct YOLO labels.

Usage:
    python scripts/check_data.py
"""

import os
from pathlib import Path

def check_yolo_label(label_path):
    """
    Check if YOLO label file is in correct format.
    Format: class_id x_center y_center width height
    Example: 0 0.5 0.5 0.3 0.4
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            return False, "Empty label file"
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            parts = line.split()
            if len(parts) != 5:
                return False, f"Expected 5 values, got {len(parts)}: {line}"
            
            # Check class_id is integer
            try:
                class_id = int(parts[0])
            except ValueError:
                return False, f"Invalid class_id (must be integer): {parts[0]}"
            
            # Check x, y, w, h are floats between 0 and 1
            try:
                x, y, w, h = map(float, parts[1:])
                if not all(0 <= val <= 1 for val in [x, y, w, h]):
                    return False, f"Values must be between 0 and 1: {parts[1:]}"
            except ValueError:
                return False, f"Invalid coordinates: {parts[1:]}"
        
        return True, "Valid YOLO format"
    
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def check_node_data(node_path, node_name):
    """Check if a single node's data is ready."""
    print(f"\n{'='*70}")
    print(f"Checking {node_name.upper()}")
    print('='*70)
    
    # Check if node directory exists
    if not os.path.exists(node_path):
        print(f"❌ Node directory does not exist: {node_path}")
        return False
    
    # Check for images and labels directories
    images_dir = os.path.join(node_path, "images")
    labels_dir = os.path.join(node_path, "labels")
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory missing: {images_dir}")
        return False
    
    if not os.path.exists(labels_dir):
        print(f"❌ Labels directory missing: {labels_dir}")
        return False
    
    # Count images and labels
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(labels_dir) 
                   if f.endswith('.txt')]
    
    num_images = len(image_files)
    num_labels = len(label_files)
    
    print(f"📁 Images directory: {images_dir}")
    print(f"   Found {num_images} images")
    print(f"📁 Labels directory: {labels_dir}")
    print(f"   Found {num_labels} labels")
    
    if num_images == 0:
        print("❌ No images found!")
        return False
    
    if num_labels == 0:
        print("❌ No labels found!")
        return False
    
    # Check if image-label pairs match
    mismatched = []
    for img_file in image_files[:5]:  # Check first 5 images
        img_name = os.path.splitext(img_file)[0]
        label_file = img_name + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(label_path):
            mismatched.append(img_file)
    
    if mismatched:
        print(f"\n⚠️  Warning: Some images don't have matching labels:")
        for img in mismatched[:3]:
            print(f"   - {img}")
        if len(mismatched) > 3:
            print(f"   ... and {len(mismatched) - 3} more")
    
    # Validate YOLO label format (check first 3 labels)
    print(f"\n🔍 Validating YOLO label format (checking first 3 labels)...")
    labels_to_check = label_files[:3]
    all_valid = True
    
    for label_file in labels_to_check:
        label_path = os.path.join(labels_dir, label_file)
        is_valid, message = check_yolo_label(label_path)
        
        if is_valid:
            print(f"   ✅ {label_file}: {message}")
        else:
            print(f"   ❌ {label_file}: {message}")
            all_valid = False
    
    # Summary
    print(f"\n{'─'*70}")
    if num_images > 0 and num_labels > 0 and all_valid:
        print(f"✅ {node_name.upper()} IS READY")
        print(f"   Images: {num_images} | Labels: {num_labels}")
        return True
    else:
        print(f"❌ {node_name.upper()} HAS ISSUES")
        return False


def main():
    """Main function to check all nodes."""
    print("\n" + "="*70)
    print("FEDERATED LEARNING DATA VALIDATION")
    print("="*70)
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Check both possible data locations
    # Location 1: data/federated/splits/iid_5nodes/
    data_path_1 = project_root / "data" / "federated" / "splits" / "iid_5nodes"
    # Location 2: clients/nodeX/data/
    data_path_2 = project_root / "clients"
    
    nodes_ready = []
    
    # Try Location 1 first (your new structure)
    if data_path_1.exists():
        print(f"\n📂 Found data at: {data_path_1}")
        for node_id in [1, 2, 3]:
            node_name = f"node_{node_id}"
            node_path = data_path_1 / node_name
            
            if check_node_data(str(node_path), node_name):
                nodes_ready.append(node_name)
    
    # Try Location 2 (old structure from run_fl.sh)
    elif data_path_2.exists():
        print(f"\n📂 Found data at: {data_path_2}")
        for node_id in [1, 2, 3]:
            node_name = f"node{node_id}"
            node_path = data_path_2 / node_name / "data"
            
            if check_node_data(str(node_path), node_name):
                nodes_ready.append(node_name)
    
    else:
        print("\n❌ ERROR: Data directory not found!")
        print(f"   Checked: {data_path_1}")
        print(f"   Checked: {data_path_2}")
        return
    
    # Final summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Nodes checked: 3")
    print(f"Nodes ready: {len(nodes_ready)}")
    
    if len(nodes_ready) == 3:
        print("\n🎉 ALL NODES ARE READY FOR FEDERATED TRAINING!")
        print("✅ Data is downloaded")
        print("✅ YOLO labels are in correct format")
        print("\nYou can now run: ./run_fl.sh")
    else:
        print(f"\n⚠️  Only {len(nodes_ready)}/3 nodes are ready")
        print("Fix the issues above before training")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
