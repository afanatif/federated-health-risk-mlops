"""
Data Validation and Quality Checks for Federated Learning Dataset
Validates: schema, missing values, outliers, data distribution
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime


class DataValidator:
    """Validates dataset quality and integrity."""
    
    def __init__(self, node_dir: str):
        self.node_dir = Path(node_dir)
        self.images_dir = self.node_dir / "data" / "images"
        self.labels_dir = self.node_dir / "data" / "labels"
        self.validation_report = {}
        
    def validate_all(self) -> Dict:
        """Run all validation checks."""
        print(f"\n{'='*80}")
        print(f"DATA VALIDATION REPORT - {self.node_dir.name}")
        print(f"{'='*80}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. Schema Validation
        schema_results = self.validate_schema()
        
        # 2. Missing Values Check
        missing_results = self.check_missing_values()
        
        # 3. Image Quality Checks
        quality_results = self.validate_image_quality()
        
        # 4. Label Validation
        label_results = self.validate_labels()
        
        # 5. Distribution Analysis
        distribution_results = self.analyze_distribution()
        
        # 6. Outlier Detection
        outlier_results = self.detect_outliers()
        
        # Compile report
        self.validation_report = {
            "node": self.node_dir.name,
            "timestamp": datetime.now().isoformat(),
            "schema": schema_results,
            "missing_values": missing_results,
            "image_quality": quality_results,
            "labels": label_results,
            "distribution": distribution_results,
            "outliers": outlier_results,
            "overall_status": self._determine_status()
        }
        
        self._print_summary()
        return self.validation_report
    
    def validate_schema(self) -> Dict:
        """Validate directory structure and file formats."""
        print("📋 1. SCHEMA VALIDATION")
        print("-" * 80)
        
        results = {
            "images_dir_exists": self.images_dir.exists(),
            "labels_dir_exists": self.labels_dir.exists(),
            "image_files": [],
            "label_files": [],
            "format_compliance": True
        }
        
        # Check images
        if self.images_dir.exists():
            image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
            results["image_files"] = [f.name for f in image_files]
            results["image_count"] = len(image_files)
            print(f"   ✓ Images directory found: {len(image_files)} images")
        else:
            print(f"   ✗ Images directory not found!")
            results["format_compliance"] = False
        
        # Check labels
        if self.labels_dir.exists():
            label_files = list(self.labels_dir.glob("*.txt"))
            results["label_files"] = [f.name for f in label_files]
            results["label_count"] = len(label_files)
            print(f"   ✓ Labels directory found: {len(label_files)} labels")
        else:
            print(f"   ✗ Labels directory not found!")
            results["format_compliance"] = False
        
        # Check pairing
        if results["images_dir_exists"] and results["labels_dir_exists"]:
            unpaired = self._check_image_label_pairing()
            results["unpaired_files"] = unpaired
            if unpaired:
                print(f"   ⚠ Warning: {len(unpaired)} unpaired files found")
                results["format_compliance"] = False
            else:
                print(f"   ✓ All images have corresponding labels")
        
        print()
        return results
    
    def check_missing_values(self) -> Dict:
        """Check for missing or corrupted files."""
        print("🔍 2. MISSING VALUES CHECK")
        print("-" * 80)
        
        results = {
            "corrupted_images": [],
            "empty_labels": [],
            "missing_count": 0
        }
        
        # Check images
        if self.images_dir.exists():
            for img_path in self.images_dir.glob("*.jpg"):
                try:
                    img = Image.open(img_path)
                    img.verify()  # Verify image integrity
                except Exception as e:
                    results["corrupted_images"].append(str(img_path))
                    print(f"   ✗ Corrupted image: {img_path.name}")
        
        # Check labels
        if self.labels_dir.exists():
            for label_path in self.labels_dir.glob("*.txt"):
                if label_path.stat().st_size == 0:
                    results["empty_labels"].append(str(label_path))
                    print(f"   ⚠ Empty label file: {label_path.name}")
        
        results["missing_count"] = len(results["corrupted_images"]) + len(results["empty_labels"])
        
        if results["missing_count"] == 0:
            print(f"   ✓ No missing or corrupted files found")
        else:
            print(f"   ⚠ Found {results['missing_count']} issues")
        
        print()
        return results
    
    def validate_image_quality(self) -> Dict:
        """Validate image quality metrics."""
        print("🖼️  3. IMAGE QUALITY VALIDATION")
        print("-" * 80)
        
        results = {
            "resolutions": [],
            "mean_resolution": None,
            "aspect_ratios": [],
            "file_sizes": [],
            "color_modes": {},
            "quality_issues": []
        }
        
        if not self.images_dir.exists():
            return results
        
        image_files = list(self.images_dir.glob("*.jpg"))[:50]  # Sample first 50
        
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                
                # Resolution
                width, height = img.size
                results["resolutions"].append((width, height))
                
                # Aspect ratio
                aspect_ratio = width / height
                results["aspect_ratios"].append(aspect_ratio)
                
                # File size
                file_size = img_path.stat().st_size
                results["file_sizes"].append(file_size)
                
                # Color mode
                mode = img.mode
                results["color_modes"][mode] = results["color_modes"].get(mode, 0) + 1
                
                # Quality checks
                if width < 100 or height < 100:
                    results["quality_issues"].append(f"{img_path.name}: Too small ({width}x{height})")
                if file_size < 5000:  # Less than 5KB
                    results["quality_issues"].append(f"{img_path.name}: File too small ({file_size} bytes)")
                
            except Exception as e:
                results["quality_issues"].append(f"{img_path.name}: {str(e)}")
        
        # Calculate statistics
        if results["resolutions"]:
            avg_width = np.mean([r[0] for r in results["resolutions"]])
            avg_height = np.mean([r[1] for r in results["resolutions"]])
            results["mean_resolution"] = (avg_width, avg_height)
            print(f"   ✓ Average resolution: {avg_width:.0f}x{avg_height:.0f}")
        
        if results["aspect_ratios"]:
            avg_aspect = np.mean(results["aspect_ratios"])
            print(f"   ✓ Average aspect ratio: {avg_aspect:.2f}")
        
        if results["file_sizes"]:
            avg_size = np.mean(results["file_sizes"]) / 1024  # KB
            print(f"   ✓ Average file size: {avg_size:.1f} KB")
        
        print(f"   ✓ Color modes: {results['color_modes']}")
        
        if results["quality_issues"]:
            print(f"   ⚠ Quality issues found: {len(results['quality_issues'])}")
        else:
            print(f"   ✓ No quality issues detected")
        
        print()
        return results
    
    def validate_labels(self) -> Dict:
        """Validate label format and content."""
        print("🏷️  4. LABEL VALIDATION")
        print("-" * 80)
        
        results = {
            "valid_labels": 0,
            "invalid_labels": [],
            "class_distribution": {},
            "bbox_stats": {
                "x_centers": [],
                "y_centers": [],
                "widths": [],
                "heights": []
            }
        }
        
        if not self.labels_dir.exists():
            return results
        
        label_files = list(self.labels_dir.glob("*.txt"))
        
        for label_path in label_files:
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    
                    if len(parts) != 5:
                        results["invalid_labels"].append(f"{label_path.name}: Invalid format")
                        continue
                    
                    class_id, x, y, w, h = map(float, parts)
                    
                    # Validate ranges
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        results["invalid_labels"].append(f"{label_path.name}: Values out of range")
                        continue
                    
                    # Track statistics
                    results["valid_labels"] += 1
                    results["class_distribution"][int(class_id)] = \
                        results["class_distribution"].get(int(class_id), 0) + 1
                    
                    results["bbox_stats"]["x_centers"].append(x)
                    results["bbox_stats"]["y_centers"].append(y)
                    results["bbox_stats"]["widths"].append(w)
                    results["bbox_stats"]["heights"].append(h)
                    
            except Exception as e:
                results["invalid_labels"].append(f"{label_path.name}: {str(e)}")
        
        print(f"   ✓ Valid labels: {results['valid_labels']}")
        print(f"   ✓ Class distribution: {results['class_distribution']}")
        
        if results["invalid_labels"]:
            print(f"   ⚠ Invalid labels: {len(results['invalid_labels'])}")
        else:
            print(f"   ✓ All labels are valid")
        
        print()
        return results
    
    def analyze_distribution(self) -> Dict:
        """Analyze data distribution."""
        print("📊 5. DISTRIBUTION ANALYSIS")
        print("-" * 80)
        
        results = {
            "class_balance": {},
            "spatial_distribution": {},
            "size_distribution": {}
        }
        
        if not self.labels_dir.exists():
            return results
        
        # Analyze class balance
        label_files = list(self.labels_dir.glob("*.txt"))
        class_counts = {}
        
        for label_path in label_files:
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id = int(line.strip().split()[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
            except:
                continue
        
        results["class_balance"] = class_counts
        
        # Check balance
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            balance_ratio = min_count / max_count if max_count > 0 else 0
            
            print(f"   ✓ Class counts: {class_counts}")
            print(f"   ✓ Balance ratio: {balance_ratio:.2f} (1.0 = perfectly balanced)")
            
            if balance_ratio < 0.5:
                print(f"   ⚠ Classes are imbalanced (ratio < 0.5)")
            else:
                print(f"   ✓ Classes are reasonably balanced")
        
        print()
        return results
    
    def detect_outliers(self) -> Dict:
        """Detect outliers in the dataset."""
        print("🔎 6. OUTLIER DETECTION")
        print("-" * 80)
        
        results = {
            "size_outliers": [],
            "aspect_outliers": [],
            "bbox_outliers": []
        }
        
        if not self.images_dir.exists():
            return results
        
        # Collect image sizes
        image_files = list(self.images_dir.glob("*.jpg"))
        sizes = []
        
        for img_path in image_files[:100]:  # Sample
            try:
                size = img_path.stat().st_size
                sizes.append((img_path.name, size))
            except:
                continue
        
        if sizes:
            sizes_array = np.array([s[1] for s in sizes])
            mean_size = np.mean(sizes_array)
            std_size = np.std(sizes_array)
            
            # Detect outliers (3 sigma rule)
            for name, size in sizes:
                z_score = abs((size - mean_size) / std_size) if std_size > 0 else 0
                if z_score > 3:
                    results["size_outliers"].append(f"{name}: {size/1024:.1f}KB (z={z_score:.2f})")
        
        if results["size_outliers"]:
            print(f"   ⚠ Size outliers detected: {len(results['size_outliers'])}")
        else:
            print(f"   ✓ No significant outliers detected")
        
        print()
        return results
    
    def _check_image_label_pairing(self) -> List[str]:
        """Check if every image has a corresponding label."""
        unpaired = []
        
        image_files = {f.stem for f in self.images_dir.glob("*.jpg")}
        label_files = {f.stem for f in self.labels_dir.glob("*.txt")}
        
        # Images without labels
        unpaired.extend([f"Image: {f}.jpg (no label)" for f in image_files - label_files])
        
        # Labels without images
        unpaired.extend([f"Label: {f}.txt (no image)" for f in label_files - image_files])
        
        return unpaired
    
    def _determine_status(self) -> str:
        """Determine overall validation status."""
        issues = []
        
        if not self.validation_report["schema"]["format_compliance"]:
            issues.append("Schema issues")
        
        if self.validation_report["missing_values"]["missing_count"] > 0:
            issues.append("Missing/corrupted files")
        
        if self.validation_report["image_quality"]["quality_issues"]:
            issues.append("Image quality issues")
        
        if self.validation_report["labels"]["invalid_labels"]:
            issues.append("Invalid labels")
        
        if issues:
            return f"WARNINGS: {', '.join(issues)}"
        else:
            return "PASSED"
    
    def _print_summary(self):
        """Print validation summary."""
        print("="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Overall Status: {self.validation_report['overall_status']}")
        print("="*80)
        print()
    
    def save_report(self, output_path: str):
        """Save validation report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2, default=str)
        print(f"💾 Validation report saved to: {output_path}")


def validate_all_nodes(nodes: List[str] = ["node1", "node2", "node3"]):
    """Validate all federated learning nodes."""
    print("\n" + "="*80)
    print("FEDERATED DATASET VALIDATION")
    print("="*80)
    print(f"Validating {len(nodes)} nodes...\n")
    
    all_reports = {}
    
    for node in nodes:
        node_path = f"clients/{node}"
        if os.path.exists(node_path):
            validator = DataValidator(node_path)
            report = validator.validate_all()
            all_reports[node] = report
            validator.save_report(f"validation_report_{node}.json")
        else:
            print(f"⚠ Warning: Node directory not found: {node_path}\n")
    
    # Save combined report
    with open("validation_report_all_nodes.json", 'w') as f:
        json.dump(all_reports, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("✅ All nodes validated. Reports saved.")
    print("="*80)
    
    return all_reports


if __name__ == "__main__":
    # Run validation on all nodes
    reports = validate_all_nodes()
    
    # Summary
    print("\n📋 QUICK SUMMARY:")
    for node, report in reports.items():
        status = report.get("overall_status", "UNKNOWN")
        print(f"   {node}: {status}")
