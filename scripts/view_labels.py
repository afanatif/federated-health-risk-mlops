"""
View YOLO labels on images from local data.
Data location: data/federated/splits/iid_5nodes/node_X/
"""
import cv2
import os
from pathlib import Path


def visualize_labels(node="node_1", num_images=5):
    """
    Visualize YOLO labels on images.
    
    Args:
        node: Which node to visualize (node_1, node_2, or node_3)
        num_images: How many images to show
    """
    # NEW PATH STRUCTURE
    base_path = f"data/federated/splits/iid_5nodes/{node}"
    images_dir = f"{base_path}/images"
    labels_dir = f"{base_path}/labels"
    output_dir = f"visualized_{node}"
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        print(f"❌ Error: {images_dir} not found!")
        print(f"Make sure your data is at: data/federated/splits/iid_5nodes/{node}/")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_files = sorted(Path(images_dir).glob("*.jpg"))[:num_images]
    
    print(f"\n📊 Visualizing {len(image_files)} images from {node}")
    print(f"Output folder: {output_dir}/\n")
    
    for img_path in image_files:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Read corresponding label
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    # Parse YOLO format: class_id x_center y_center width height
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_c, y_c, box_w, box_h = map(float, parts)
                        
                        # Convert normalized coords to pixels
                        x_c *= w
                        y_c *= h
                        box_w *= w
                        box_h *= h
                        
                        # Get corners
                        x1 = int(x_c - box_w/2)
                        y1 = int(y_c - box_h/2)
                        x2 = int(x_c + box_w/2)
                        y2 = int(y_c + box_h/2)
                        
                        # Draw box (green for pothole, red for no-pothole)
                        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label = "Pothole" if class_id == 0 else "No-Pothole"
                        cv2.putText(img, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save
        output_path = os.path.join(output_dir, img_path.name)
        cv2.imwrite(output_path, img)
        print(f"✓ Saved: {img_path.name}")
    
    print(f"\n✅ Done! Check images in: {output_dir}/")


def check_labels(node="node_1"):
    """Print label file contents to see what's inside."""
    labels_dir = f"data/federated/splits/iid_5nodes/{node}/labels"
    
    if not os.path.exists(labels_dir):
        print(f"❌ {labels_dir} not found!")
        return
    
    label_files = sorted(Path(labels_dir).glob("*.txt"))[:5]
    
    print(f"\n📄 Label contents from {node} (first 5 files):")
    print("="*60)
    
    for label_path in label_files:
        print(f"\n{label_path.name}:")
        with open(label_path, 'r') as f:
            for line in f:
                print(f"  {line.strip()}")
    
    print("="*60)


if __name__ == "__main__":
    # First check what's in the labels
    check_labels("node_1")
    
    # Then visualize
    print("\n" + "="*60)
    visualize_labels("node_1", num_images=10)
    print("="*60)
    
    print("\n💡 TIP: Change 'node_1' to 'node_2' or 'node_3' to check other nodes")
