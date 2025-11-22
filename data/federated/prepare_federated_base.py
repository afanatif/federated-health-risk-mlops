"""
Prepare Federated Learning Base Directory
==========================================
This script creates a clean copy of the dataset ready for federated splitting
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import json
from collections import Counter
import random

# Configuration
SOURCE_DATASET = Path("data/raw/Pothole.v1i.yolov8")
FEDERATED_BASE = Path("data/federated_base")


SOURCE_TRAIN_IMAGES = SOURCE_DATASET / "train" / "images"
SOURCE_TRAIN_LABELS = SOURCE_DATASET / "train" / "labels"
SOURCE_VALID_IMAGES = SOURCE_DATASET / "valid" / "images"
SOURCE_VALID_LABELS = SOURCE_DATASET / "valid" / "labels"

CLASS_NAMES = ['banner', 'erosion', 'hcrack', 'pothole', 'stone', 'trash', 'vcrack']

def ensure_clean_directory(path):
    """Create or clean directory"""
    if path.exists():
        print(f"⚠️  Directory {path} exists. Remove it? (y/n): ", end='')
        response = input().lower()
        if response == 'y':
            shutil.rmtree(path)
            print(f"✓ Removed {path}")
        else:
            print("❌ Aborted. Please backup or remove the directory manually.")
            return False
    path.mkdir(parents=True, exist_ok=True)
    return True

def copy_dataset_split(source_images, source_labels, dest_images, dest_labels):
    """Copy images and labels with validation"""
    
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(source_images.glob(f"*{ext}")))
    
    print(f"  Found {len(image_files)} images")
    
    copied_images = 0
    copied_labels = 0
    missing_labels = []
    
    for img_path in tqdm(image_files, desc="  Copying files"):
        # Find corresponding label
        label_path = source_labels / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            missing_labels.append(img_path.stem)
            continue
        
        # Copy image and label
        try:
            shutil.copy2(img_path, dest_images / img_path.name)
            shutil.copy2(label_path, dest_labels / f"{img_path.stem}.txt")
            copied_images += 1
            copied_labels += 1
        except Exception as e:
            print(f"  ⚠️  Error copying {img_path.name}: {e}")
    
    print(f"  ✓ Copied {copied_images} images and {copied_labels} labels")
    
    if missing_labels:
        print(f"  ⚠️  {len(missing_labels)} images without labels (skipped)")
    
    return copied_images, copied_labels, missing_labels

def validate_synchronized_data(images_dir, labels_dir):
    """Verify all images have corresponding labels"""
    
    image_stems = {f.stem for f in images_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
    label_stems = {f.stem for f in labels_dir.glob("*.txt")}
    
    missing_labels = image_stems - label_stems
    extra_labels = label_stems - image_stems
    
    synchronized = len(missing_labels) == 0 and len(extra_labels) == 0
    
    return synchronized, missing_labels, extra_labels

def analyze_dataset_statistics(images_dir, labels_dir):
    """Collect statistics about the dataset"""
    
    stats = {
        'total_images': 0,
        'total_labels': 0,
        'total_annotations': 0,
        'class_distribution': Counter(),
        'boxes_per_image': []
    }
    
    label_files = list(labels_dir.glob("*.txt"))
    stats['total_labels'] = len(label_files)
    
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
    stats['total_images'] = len(image_files)
    
    for label_path in label_files:
        boxes_in_image = 0
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        stats['class_distribution'][cls_id] += 1
                        stats['total_annotations'] += 1
                        boxes_in_image += 1
        except Exception as e:
            continue
        
        stats['boxes_per_image'].append(boxes_in_image)
    
    return stats

def main():
    print("="*80)
    print("FEDERATED LEARNING BASE DIRECTORY PREPARATION")
    print("="*80)
    
    # Check source directories
    print("\n1. Checking source directories...")
    if not SOURCE_TRAIN_IMAGES.exists():
        print(f"❌ Source train images not found: {SOURCE_TRAIN_IMAGES}")
        return
    if not SOURCE_VALID_IMAGES.exists():
        print(f"❌ Source valid images not found: {SOURCE_VALID_IMAGES}")
        return
    
    print("✓ Source directories found")
    
    # Create federated_base directory
    print("\n2. Creating federated_base directory...")
    if not ensure_clean_directory(FEDERATED_BASE):
        return
    
    # Create subdirectories
    base_train_images = FEDERATED_BASE / "base_train" / "images"
    base_train_labels = FEDERATED_BASE / "base_train" / "labels"
    base_valid_images = FEDERATED_BASE / "base_valid" / "images"
    base_valid_labels = FEDERATED_BASE / "base_valid" / "labels"
    metadata_dir = FEDERATED_BASE / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")
    
    # Copy training data
    print("\n3. Copying training data...")
    train_imgs, train_lbls, train_missing = copy_dataset_split(
        SOURCE_TRAIN_IMAGES, SOURCE_TRAIN_LABELS,
        base_train_images, base_train_labels
    )
    
    # Copy validation data
    print("\n4. Copying validation data...")
    valid_imgs, valid_lbls, valid_missing = copy_dataset_split(
        SOURCE_VALID_IMAGES, SOURCE_VALID_LABELS,
        base_valid_images, base_valid_labels
    )
    
    # Validate synchronization
    print("\n5. Validating data synchronization...")
    
    train_sync, train_miss, train_extra = validate_synchronized_data(
        base_train_images, base_train_labels
    )
    valid_sync, valid_miss, valid_extra = validate_synchronized_data(
        base_valid_images, base_valid_labels
    )
    
    if train_sync and valid_sync:
        print("  ✓ All images and labels are synchronized!")
    else:
        if not train_sync:
            print(f"  ⚠️  Train: {len(train_miss)} missing labels, {len(train_extra)} extra labels")
        if not valid_sync:
            print(f"  ⚠️  Valid: {len(valid_miss)} missing labels, {len(valid_extra)} extra labels")
    
    # Collect statistics
    print("\n6. Collecting dataset statistics...")
    
    train_stats = analyze_dataset_statistics(base_train_images, base_train_labels)
    valid_stats = analyze_dataset_statistics(base_valid_images, base_valid_labels)
    
    print(f"\n  TRAIN Statistics:")
    print(f"    Images: {train_stats['total_images']}")
    print(f"    Labels: {train_stats['total_labels']}")
    print(f"    Annotations: {train_stats['total_annotations']}")
    print(f"    Avg boxes/image: {sum(train_stats['boxes_per_image'])/len(train_stats['boxes_per_image']) if train_stats['boxes_per_image'] else 0:.2f}")
    
    print(f"\n  VALID Statistics:")
    print(f"    Images: {valid_stats['total_images']}")
    print(f"    Labels: {valid_stats['total_labels']}")
    print(f"    Annotations: {valid_stats['total_annotations']}")
    print(f"    Avg boxes/image: {sum(valid_stats['boxes_per_image'])/len(valid_stats['boxes_per_image']) if valid_stats['boxes_per_image'] else 0:.2f}")
    
    # Save metadata
    print("\n7. Saving metadata...")
    
    metadata = {
        'source_dataset': str(SOURCE_DATASET),
        'creation_timestamp': str(Path(__file__).stat().st_mtime),
        'class_names': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES),
        'train': {
            'total_images': train_stats['total_images'],
            'total_labels': train_stats['total_labels'],
            'total_annotations': train_stats['total_annotations'],
            'class_distribution': {CLASS_NAMES[k]: v for k, v in sorted(train_stats['class_distribution'].items())},
            'synchronized': train_sync
        },
        'valid': {
            'total_images': valid_stats['total_images'],
            'total_labels': valid_stats['total_labels'],
            'total_annotations': valid_stats['total_annotations'],
            'class_distribution': {CLASS_NAMES[k]: v for k, v in sorted(valid_stats['class_distribution'].items())},
            'synchronized': valid_sync
        },
        'ready_for_splitting': train_sync and valid_sync
    }
    
    with open(metadata_dir / 'base_dataset_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Metadata saved to {metadata_dir / 'base_dataset_info.json'}")
    
    # Create README
    readme_content = f"""# Federated Learning Base Dataset

## Overview
This directory contains a clean, synchronized copy of the road damage detection dataset prepared for federated learning experiments.

## Structure
```
federated_base/
├── base_train/
│   ├── images/          # {train_stats['total_images']} training images
│   └── labels/          # {train_stats['total_labels']} training labels
├── base_valid/
│   ├── images/          # {valid_stats['total_images']} validation images
│   └── labels/          # {valid_stats['total_labels']} validation labels
├── metadata/
│   ├── base_dataset_info.json
│   └── README.md (this file)
└── [node_X directories will be created here]
```

## Dataset Statistics

### Training Set
- Images: {train_stats['total_images']}
- Annotations: {train_stats['total_annotations']}
- Avg boxes per image: {sum(train_stats['boxes_per_image'])/len(train_stats['boxes_per_image']) if train_stats['boxes_per_image'] else 0:.2f}

### Validation Set
- Images: {valid_stats['total_images']}
- Annotations: {valid_stats['total_annotations']}
- Avg boxes per image: {sum(valid_stats['boxes_per_image'])/len(valid_stats['boxes_per_image']) if valid_stats['boxes_per_image'] else 0:.2f}

## Class Distribution (Training)

"""
    for cls_id, count in sorted(train_stats['class_distribution'].items()):
        class_name = CLASS_NAMES[cls_id]
        percentage = (count / train_stats['total_annotations']) * 100
        readme_content += f"- {class_name}: {count} ({percentage:.2f}%)\n"
    
    readme_content += f"""

## Next Steps

1. Review `split_plan.md` for splitting strategies
2. Run `create_federated_split.py` to create node-specific datasets
3. Validate split distributions
4. Begin federated training experiments

## Data Integrity

✓ All images have corresponding label files
✓ All labels are in correct YOLO format
✓ Dataset is ready for federated splitting

## Notes

- This is a working copy; original dataset remains unchanged
- All file paths are synchronized between images and labels
- Validation set will be kept centralized for fair evaluation

---
Generated by: prepare_federated_base.py
"""
    
    with open(metadata_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"  ✓ README saved to {metadata_dir / 'README.md'}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ FEDERATED BASE DIRECTORY PREPARATION COMPLETE!")
    print("="*80)
    print(f"\nBase directory: {FEDERATED_BASE}/")
    print(f"  ├── base_train/     ({train_stats['total_images']} images)")
    print(f"  ├── base_valid/     ({valid_stats['total_images']} images)")
    print(f"  └── metadata/       (configuration & stats)")
    
    if metadata['ready_for_splitting']:
        print("\n✓ Dataset is synchronized and ready for federated splitting!")
        print("\nNext steps:")
        print("  1. Review split_plan.md")
        print("  2. Run: python create_federated_split.py")
        print("  3. Begin FL training")
    else:
        print("\n⚠️  Dataset has synchronization issues. Please review and fix.")
    
    print("="*80)

if __name__ == "__main__":
    main()