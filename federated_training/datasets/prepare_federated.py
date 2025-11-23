"""
Create federated splits (IID/non-IID) from YOLOv8 dataset.
All paths read from config/settings.yaml
"""
import os
import shutil
import random
from pathlib import Path
from collections import Counter
import json
import argparse
import yaml
from tqdm import tqdm

# ---------------------------
# Load global settings
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "config/settings.yaml"
if not SETTINGS_PATH.exists():
    raise FileNotFoundError(f"settings.yaml not found at: {SETTINGS_PATH}")

with open(SETTINGS_PATH, "r") as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg.get("data", {})
YOLO_DATASET = PROJECT_ROOT / DATA_CFG.get("yolo_dataset", "data/Pothole.v1i.yolov8")
FEDERATED_ROOT = PROJECT_ROOT / DATA_CFG.get("federated_root", "data/federated")
CLASS_NAMES = DATA_CFG.get("class_names", ['banner','erosion','hcrack','pothole','stone','trash','vcrack'])
NUM_CLASSES = DATA_CFG.get("nc", len(CLASS_NAMES))

BASE_TRAIN_IMAGES = YOLO_DATASET / "train/images"
BASE_TRAIN_LABELS = YOLO_DATASET / "train/labels"
BASE_VAL_IMAGES = YOLO_DATASET / "valid/images"
BASE_VAL_LABELS = YOLO_DATASET / "valid/labels"

SPLIT_OUTPUT_DIR = FEDERATED_ROOT / "splits"
SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def parse_label_file(label_path: Path):
    classes = set()
    if not label_path.exists():
        return classes
    with open(label_path, "r") as f:
        for line in f:
            try:
                cls = int(float(line.strip().split()[0]))
                classes.add(cls)
            except:
                continue
    return classes

def get_all_images_with_classes():
    image_class_map = {}
    image_files = sorted([p for p in BASE_TRAIN_IMAGES.glob("*.*") if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
    for p in tqdm(image_files, desc="Analyzing images"):
        classes = parse_label_file(BASE_TRAIN_LABELS / f"{p.stem}.txt")
        image_class_map[p.stem] = {"path": p, "classes": classes}
    return image_class_map

def split_iid(image_class_map: dict, num_nodes: int, seed: int = 42):
    random.seed(seed)
    stems = list(image_class_map.keys())
    random.shuffle(stems)
    splits = {}
    per = len(stems) // num_nodes
    for i in range(1, num_nodes+1):
        splits[i] = stems[(i-1)*per:i*per] if i<num_nodes else stems[(i-1)*per:]
    return splits

# ---------------------------
# Create node directories & copy
# ---------------------------
def create_node_directories(node_assignments: dict, output_dir: Path, image_class_map: dict):
    stats = {}
    for node_id, stems in node_assignments.items():
        node_dir = output_dir / f"node_{node_id}"
        node_images = node_dir / "images"
        node_labels = node_dir / "labels"
        node_images.mkdir(parents=True, exist_ok=True)
        node_labels.mkdir(parents=True, exist_ok=True)

        class_dist = Counter()
        total_boxes = 0

        for stem in tqdm(stems, desc=f"Copying for node_{node_id}"):
            info = image_class_map[stem]
            img_src = info["path"]
            lbl_src = BASE_TRAIN_LABELS / f"{stem}.txt"
            shutil.copy2(img_src, node_images / img_src.name)
            if lbl_src.exists():
                shutil.copy2(lbl_src, node_labels / lbl_src.name)
                with open(lbl_src, "r") as f:
                    lines = [l for l in f if l.strip()]
                total_boxes += len(lines)
                for l in lines:
                    cls = int(float(l.split()[0]))
                    class_dist[cls] += 1

        stats[node_id] = {
            "num_images": len(stems),
            "num_annotations": total_boxes,
            "class_distribution": {CLASS_NAMES[k]: v for k,v in sorted(class_dist.items())},
            "unique_classes": len(class_dist)
        }
    return stats

def generate_node_data_yamls(num_nodes: int, output_dir: Path):
    for node_id in range(1, num_nodes+1):
        node_path = output_dir / f"node_{node_id}"
        node_yaml = {
            "path": str(node_path.resolve()),
            "train": "images",
            "val": str(BASE_VAL_IMAGES.resolve()),
            "nc": NUM_CLASSES,
            "names": CLASS_NAMES
        }
        with open(node_path / "data.yaml", "w") as f:
            yaml.safe_dump(node_yaml, f)

def save_split_metadata(strategy: str, num_nodes: int, seed: int, stats: dict, output_dir: Path):
    metadata = {
        "strategy": strategy,
        "num_nodes": num_nodes,
        "seed": seed,
        "total_images": sum(stats[n]["num_images"] for n in stats),
        "total_annotations": sum(stats[n]["num_annotations"] for n in stats),
        "nodes": stats,
        "class_names": CLASS_NAMES
    }
    with open(output_dir / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="iid", choices=["iid"])
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else SPLIT_OUTPUT_DIR / f"{args.strategy}_{args.nodes}nodes"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating federated split at {output_dir}")
    image_map = get_all_images_with_classes()
    if args.strategy == "iid":
        node_assignments = split_iid(image_map, args.nodes, args.seed)

    stats = create_node_directories(node_assignments, output_dir, image_map)
    generate_node_data_yamls(args.nodes, output_dir)
    save_split_metadata(args.strategy, args.nodes, args.seed, stats, output_dir)
    print("Done. Node directories created:")
    for nid in stats:
        print(f" node_{nid}: {stats[nid]['num_images']} images, {stats[nid]['num_annotations']} boxes")

if __name__ == "__main__":
    main()
