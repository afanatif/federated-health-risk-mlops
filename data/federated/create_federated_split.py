"""
Create Federated Learning Dataset Splits
=========================================
This script splits the base dataset into node-specific datasets
for federated learning simulation.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import json
from collections import Counter, defaultdict
import random
import numpy as np
import argparse

# Configuration (read from federated/config.yaml if present)
try:
    import yaml
except Exception:
    yaml = None

# default base path (fallback)
_default_base = Path("data/federated_base")

# try to read federated/config.yaml that you added earlier
config_path = Path(__file__).resolve().parent / "config.yaml"
if yaml and config_path.exists():
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        FEDERATED_BASE = Path(cfg.get("dataset", {}).get("base", str(_default_base)))
    except Exception:
        FEDERATED_BASE = _default_base
else:
    FEDERATED_BASE = _default_base

BASE_TRAIN_IMAGES = FEDERATED_BASE / "base_train" / "images"
BASE_TRAIN_LABELS = FEDERATED_BASE / "base_train" / "labels"

# Splits will be written under this directory
SPLIT_OUTPUT_DIR = FEDERATED_BASE / "splits"
# create top-level splits dir (no harm if exists)
SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = ['banner', 'erosion', 'hcrack', 'pothole', 'stone', 'trash', 'vcrack']
NUM_CLASSES = 7


def parse_label_file(label_path):
    """Extract class IDs from a label file"""
    classes = set()
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    classes.add(int(parts[0]))
    except Exception:
        pass
    return classes


def get_all_images_with_classes():
    """Create mapping of images to their classes"""
    image_class_map = {}
    
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(BASE_TRAIN_IMAGES.glob(f"*{ext}")))
    
    print(f"Found {len(image_files)} images")
    
    for img_path in tqdm(image_files, desc="Analyzing images"):
        label_path = BASE_TRAIN_LABELS / f"{img_path.stem}.txt"
        if label_path.exists():
            classes = parse_label_file(label_path)
            image_class_map[img_path.stem] = {
                'path': img_path,
                'classes': classes
            }
    
    return image_class_map


def split_iid(image_class_map, num_nodes, seed=42):
    """
    Strategy 1: IID Split
    Randomly distribute images equally across nodes
    """
    random.seed(seed)
    
    image_stems = list(image_class_map.keys())
    random.shuffle(image_stems)
    
    split_size = len(image_stems) // num_nodes
    node_assignments = {}
    
    for node_id in range(num_nodes):
        start = node_id * split_size
        end = start + split_size if node_id < num_nodes - 1 else len(image_stems)
        node_assignments[node_id] = image_stems[start:end]
    
    return node_assignments


def split_non_iid_class(image_class_map, num_nodes, seed=42):
    """
    Strategy 2: Non-IID by Class
    Each node specializes in certain damage types
    """
    random.seed(seed)
    
    # Define class preferences for each node
    if num_nodes == 3:
        node_class_prefs = {
            0: [3, 5, 0],       # Urban: pothole, trash, banner
            1: [2, 6, 4],       # Highway: hcrack, vcrack, stone
            2: [1, 4, 2, 6]     # Mixed: erosion, stone, cracks
        }
    elif num_nodes == 5:
        node_class_prefs = {
            0: [3, 5],          # Urban dense: pothole, trash
            1: [2, 6],          # Highway: horizontal/vertical cracks
            2: [1, 4],          # Rural: erosion, stone
            3: [0, 5, 3],       # City center: banner, trash, pothole
            4: list(range(7))   # Mixed routes: all classes
        }
    else:  # Default for other node counts
        # Distribute classes round-robin
        node_class_prefs = {i: [] for i in range(num_nodes)}
        for cls_id in range(NUM_CLASSES):
            node_id = cls_id % num_nodes
            node_class_prefs[node_id].append(cls_id)
    
    # Assign images to nodes based on class overlap
    node_assignments = {i: [] for i in range(num_nodes)}
    unassigned = []
    
    for img_stem, img_info in image_class_map.items():
        img_classes = img_info['classes']
        
        # Calculate overlap score for each node
        scores = {}
        for node_id, pref_classes in node_class_prefs.items():
            overlap = len(img_classes & set(pref_classes))
            scores[node_id] = overlap
        
        # Assign to node with highest overlap
        if max(scores.values()) > 0:
            best_node = max(scores.items(), key=lambda x: (x[1], random.random()))[0]
            node_assignments[best_node].append(img_stem)
        else:
            unassigned.append(img_stem)
    
    # Distribute unassigned images
    random.shuffle(unassigned)
    for idx, img_stem in enumerate(unassigned):
        node_id = idx % num_nodes
        node_assignments[node_id].append(img_stem)
    
    return node_assignments


def split_non_iid_quantity(image_class_map, num_nodes, seed=42):
    """
    Strategy 3: Non-IID by Quantity
    Nodes have different amounts of data
    """
    random.seed(seed)
    
    # Define distribution ratios
    if num_nodes == 3:
        ratios = [0.5, 0.3, 0.2]
    elif num_nodes == 5:
        ratios = [0.4, 0.25, 0.15, 0.1, 0.1]
    else:
        # Create decreasing ratios
        ratios = [1.0 / (i + 1) for i in range(num_nodes)]
        total = sum(ratios)
        ratios = [r / total for r in ratios]
    
    image_stems = list(image_class_map.keys())
    random.shuffle(image_stems)
    
    node_assignments = {}
    start_idx = 0
    
    for node_id, ratio in enumerate(ratios):
        count = int(len(image_stems) * ratio)
        if node_id == num_nodes - 1:  # Last node gets remainder
            node_assignments[node_id] = image_stems[start_idx:]
        else:
            node_assignments[node_id] = image_stems[start_idx:start_idx + count]
            start_idx += count
    
    return node_assignments


def split_hybrid(image_class_map, num_nodes, seed=42):
    """
    Strategy 4: Hybrid (Class preference + Quantity imbalance)
    Most realistic scenario
    """
    random.seed(seed)
    
    # First apply class-based split
    node_assignments = split_non_iid_class(image_class_map, num_nodes, seed)
    
    # Then apply quantity constraints
    if num_nodes == 5:
        target_ratios = [0.35, 0.25, 0.2, 0.12, 0.08]
    else:
        target_ratios = [1.0 / (i + 1.5) for i in range(num_nodes)]
        total = sum(target_ratios)
        target_ratios = [r / total for r in target_ratios]
    
    total_images = sum(len(imgs) for imgs in node_assignments.values())
    target_counts = [int(total_images * ratio) for ratio in target_ratios]
    
    # Rebalance
    for node_id in range(num_nodes):
        current = len(node_assignments[node_id])
        target = target_counts[node_id]
        
        if current > target:
            # Move excess to nodes that need more
            excess = current - target
            to_move = random.sample(node_assignments[node_id], excess)
            node_assignments[node_id] = [img for img in node_assignments[node_id] if img not in to_move]
            
            # Distribute to nodes with deficit
            for img in to_move:
                min_node = min(range(num_nodes), key=lambda i: len(node_assignments[i]) - target_counts[i])
                if len(node_assignments[min_node]) < target_counts[min_node]:
                    node_assignments[min_node].append(img)
    
    return node_assignments


def create_node_directories(node_assignments, output_dir, image_class_map):
    """Create physical node directories with images and labels"""
    
    stats = {}
    
    for node_id, image_stems in node_assignments.items():
        node_dir = output_dir / f"node_{node_id}"
        node_images = node_dir / "images"
        node_labels = node_dir / "labels"
        
        node_images.mkdir(parents=True, exist_ok=True)
        node_labels.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreating node_{node_id}...")
        
        class_dist = Counter()
        total_boxes = 0
        
        for img_stem in tqdm(image_stems, desc=f"  Copying files"):
            img_info = image_class_map[img_stem]
            img_path = img_info['path']
            label_path = BASE_TRAIN_LABELS / f"{img_stem}.txt"
            
            # Copy files
            shutil.copy2(img_path, node_images / img_path.name)
            shutil.copy2(label_path, node_labels / f"{img_stem}.txt")
            
            # Count classes
            for cls_id in img_info['classes']:
                class_dist[cls_id] += 1
            
            # Count total boxes
            with open(label_path, 'r') as f:
                total_boxes += sum(1 for line in f if line.strip())
        
        stats[node_id] = {
            'num_images': len(image_stems),
            'num_annotations': total_boxes,
            'class_distribution': {CLASS_NAMES[k]: v for k, v in sorted(class_dist.items())},
            'unique_classes': len(class_dist)
        }
        
        print(f"  ✓ {len(image_stems)} images, {total_boxes} annotations, {len(class_dist)} classes")
    
    return stats


def generate_node_data_yamls(num_nodes, output_dir):
    """Create data.yaml for each node"""
    
    for node_id in range(num_nodes):
        node_yaml = f"""# YOLOv8 Dataset Configuration for Node {node_id}
# Federated Learning - Road Damage Detection

path: {output_dir / f'node_{node_id}'}
train: images
val: ../base_valid/images  # Shared validation set

nc: {NUM_CLASSES}
names: {CLASS_NAMES}

# Node ID: {node_id}
# This is a federated learning node dataset
"""
        
        with open(output_dir / f"node_{node_id}" / "data.yaml", 'w') as f:
            f.write(node_yaml)


def save_split_metadata(strategy, num_nodes, seed, stats, output_dir):
    """Save comprehensive metadata about the split"""
    
    metadata = {
        'strategy': strategy,
        'num_nodes': num_nodes,
        'random_seed': seed,
        'total_images': sum(s['num_images'] for s in stats.values()),
        'total_annotations': sum(s['num_annotations'] for s in stats.values()),
        'nodes': stats,
        'class_names': CLASS_NAMES
    }
    
    with open(output_dir / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate summary report
    report = f"""# Federated Split Summary
## Strategy: {strategy}
## Number of Nodes: {num_nodes}
## Random Seed: {seed}

## Overall Statistics
- Total Images: {metadata['total_images']}
- Total Annotations: {metadata['total_annotations']}

## Per-Node Statistics

"""
    
    for node_id in sorted(stats.keys()):
        node_stats = stats[node_id]
        report += f"""### Node {node_id}
- Images: {node_stats['num_images']} ({node_stats['num_images']/metadata['total_images']*100:.1f}%)
- Annotations: {node_stats['num_annotations']}
- Unique Classes: {node_stats['unique_classes']}/{NUM_CLASSES}

**Class Distribution:**
"""
        for cls_name, count in sorted(node_stats['class_distribution'].items()):
            report += f"- {cls_name}: {count}\n"
        report += "\n"
    
    with open(output_dir / "split_summary.md", 'w') as f:
        f.write(report)
    
    print(f"\n✓ Metadata saved to {output_dir / 'split_metadata.json'}")
    print(f"✓ Summary saved to {output_dir / 'split_summary.md'}")


def main():
    parser = argparse.ArgumentParser(description='Create federated learning dataset splits')
    parser.add_argument('--strategy', type=str, default='iid',
                      choices=['iid', 'non_iid_class', 'non_iid_quantity', 'hybrid'],
                      help='Splitting strategy')
    parser.add_argument('--nodes', type=int, default=5,
                      help='Number of federated nodes')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory (default: federated_base/splits/<strategy>)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = SPLIT_OUTPUT_DIR / f"{args.strategy}_{args.nodes}nodes"

    
    print("="*80)
    print("FEDERATED LEARNING DATASET SPLIT CREATION")
    print("="*80)
    print(f"\nStrategy: {args.strategy}")
    print(f"Number of nodes: {args.nodes}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    
    # Check base dataset
    if not BASE_TRAIN_IMAGES.exists():
        print(f"\n❌ Base dataset not found!")
        print(f"Please run prepare_federated_base.py first")
        return
    
    # Load image-class mapping
    print("\n1. Analyzing base dataset...")
    image_class_map = get_all_images_with_classes()
    print(f"✓ Loaded {len(image_class_map)} images")
    
    # Create split
    print(f"\n2. Creating {args.strategy} split...")
    if args.strategy == 'iid':
        node_assignments = split_iid(image_class_map, args.nodes, args.seed)
    elif args.strategy == 'non_iid_class':
        node_assignments = split_non_iid_class(image_class_map, args.nodes, args.seed)
    elif args.strategy == 'non_iid_quantity':
        node_assignments = split_non_iid_quantity(image_class_map, args.nodes, args.seed)
    elif args.strategy == 'hybrid':
        node_assignments = split_hybrid(image_class_map, args.nodes, args.seed)
    
    # Validate split
    total_assigned = sum(len(imgs) for imgs in node_assignments.values())
    print(f"✓ Split created: {total_assigned} images assigned")
    
    # Create directories
    print("\n3. Creating node directories...")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = create_node_directories(node_assignments, output_dir, image_class_map)
    
    # Generate data.yaml files
    print("\n4. Generating data.yaml files...")
    generate_node_data_yamls(args.nodes, output_dir)
    print(f"✓ Created {args.nodes} data.yaml files")
    
    # Save metadata
    print("\n5. Saving metadata...")
    save_split_metadata(args.strategy, args.nodes, args.seed, stats, output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ FEDERATED SPLIT CREATION COMPLETE!")
    print("="*80)
    print(f"\nOutput location: {output_dir}/")
    print(f"  ├── node_0/  ({stats[0]['num_images']} images)")
    for i in range(1, args.nodes):
        print(f"  ├── node_{i}/  ({stats[i]['num_images']} images)")
    print(f"  ├── split_metadata.json")
    print(f"  └── split_summary.md")
    
    print("\nNext steps:")
    print(f"  1. Review {output_dir / 'split_summary.md'}")
    print("  2. Validate node distributions")
    print("  3. Begin federated training with these node datasets")
    print("="*80)


if __name__ == "__main__":
    main()